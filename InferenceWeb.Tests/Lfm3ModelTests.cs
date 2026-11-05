// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LFM3 (GGUF lfm3) on a tiny synthetic checkpoint: the forward against a float64
// reference written from Liquid's modeling_lfm3.py, the conv state across prefill
// chunks and decode, the KV-state snapshot the server's engine schedules it on,
// greedy decoding through that engine, and tensor parallelism (whose snapshot must be
// byte-identical to the single-device one).
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class Lfm3ModelTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-lfm3-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    // Longer than the sliding window (4) and the conv width (4), so both are exercised.
    private static readonly int[] Prompt = { 72, 101, 108, 108, 111, 44, 32, 76, 70, 77, 51, 33, 10, 50, 48, 50 };

    public Lfm3ModelTests()
    {
        Directory.CreateDirectory(_dir);
        _environment.Set("MAX_CONTEXT", "128");
        // A tiny initial cache, so the forwards below grow it.
        _environment.Set("TS_KV_INITIAL_TOKENS", "2");
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        _environment.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private (Lfm3SyntheticModelBuilder Fixture, string Path) Build(Lfm3SyntheticModelBuilder fixture = null, string name = "lfm3.gguf")
    {
        fixture ??= new Lfm3SyntheticModelBuilder();
        return (fixture, fixture.Write(Path.Combine(_dir, name)));
    }

    private static void Close(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && MathF.Abs(expected[i] - actual[i]) <= tolerance,
                $"logit {i}: expected {expected[i]:R}, got {actual[i]:R} (tolerance {tolerance:R})");
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }

    [Fact]
    public void ResolvesAsLfm3WithItsLayerLayout()
    {
        var (_, path) = Build();
        using var model = ModelBase.Create(path, BackendType.Cpu);
        var lfm3 = Assert.IsType<Lfm3Model>(model);
        Assert.Equal("lfm3", model.Config.Architecture);
        // ParseBaseConfig alone would read layer 0's 0 KV heads.
        Assert.Equal(2, model.Config.NumKVHeads);
        Assert.Equal(4, model.Config.SlidingWindow);
        Assert.StartsWith("lfm3|", model.KVStateFingerprint);
        Assert.Contains("layers=cfcscf", model.KVStateFingerprint);
        Assert.True(model.SupportsKVStateSnapshot);
        Assert.True(model.RequiresPerBlockCapture);
        Assert.False(model.SupportsKVCacheTruncation);
        Assert.IsAssignableFrom<TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixCacheModel>(lfm3);
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void PrefillMatchesTheReferenceForward(bool sandwich, bool tied)
    {
        var (fixture, path) = Build(new Lfm3SyntheticModelBuilder { SandwichNorms = sandwich, TiedOutput = tied });
        using var model = ModelBase.Create(path, BackendType.Cpu);
        Close(fixture.Reference(Prompt), model.Forward(Prompt), 2e-3f);
    }

    [Fact]
    public void ChunkedPrefillAndDecodeCarryTheConvStateAndWindow()
    {
        var (fixture, path) = Build();
        using var model = ModelBase.Create(path, BackendType.Cpu);
        float[][] all = fixture.ReferenceAll(Prompt);

        // Chunks of 3, 5 and 2 tokens, then one token at a time: every chunk boundary
        // lands somewhere else relative to the window and the conv width.
        int position = 0;
        foreach (int chunk in new[] { 3, 5, 2 })
        {
            float[] logits = model.Forward(Prompt[position..(position + chunk)]);
            position += chunk;
            Close(all[position - 1], logits, 2e-3f);
        }
        for (; position < Prompt.Length; position++)
            Close(all[position], model.Forward(new[] { Prompt[position] }), 2e-3f);
    }

    [Fact]
    public void HalfPrecisionKvCacheStaysCloseToTheReference()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        var (fixture, path) = Build();
        using var model = ModelBase.Create(path, BackendType.Cpu);
        float[][] all = fixture.ReferenceAll(Prompt);
        model.Forward(Prompt[..6]);
        for (int position = 6; position < Prompt.Length; position++)
            Close(all[position], model.Forward(new[] { Prompt[position] }), 2e-2f);
    }

    [Fact]
    public void QuantizedProjectionsTrackTheReferenceOfTheStoredWeights()
    {
        // The Q8_0 kernels quantize the activations too (as ggml does), so the float64
        // reference of the stored weights is matched loosely; the next token exactly.
        var (fixture, path) = Build(new Lfm3SyntheticModelBuilder { QuantizeProjections = true });
        using var model = ModelBase.Create(path, BackendType.Cpu);
        float[] expected = fixture.Reference(Prompt);
        float[] actual = model.Forward(Prompt);
        Close(expected, actual, 0.25f);
        Assert.Equal(ArgMax(expected), ArgMax(actual));
    }

    [Fact]
    public void TruncationRefusesEverythingButANoOpOrAReset()
    {
        var (_, path) = Build();
        using var model = ModelBase.Create(path, BackendType.Cpu);
        model.Forward(Prompt[..8]);
        Assert.False(model.TryTruncateKVCache(5));
        Assert.True(model.TryTruncateKVCache(8));
        Assert.True(model.TryTruncateKVCache(0));
        Assert.Equal(0, ((Lfm3Model)model).PrimaryCacheLength);
    }

    [Fact]
    public void SnapshotRoundTripReproducesTheUninterruptedRun()
    {
        var (_, path) = Build();
        using var model = ModelBase.Create(path, BackendType.Cpu);
        const int next = 90;

        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        float[] expected = (float[])model.Forward(new[] { next }).Clone();

        // Two blocks, as the executor writes them: the first K/V-only (not a restore
        // point), the second full (it ends where the state is).
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        int first = 8, second = Prompt.Length - first;
        long kvOnly = model.ComputeKVBlockByteSizeWithoutRecurrentState(first);
        long full = model.ComputeKVBlockByteSize(second);
        Assert.True(kvOnly > 0 && kvOnly < model.ComputeKVBlockByteSize(first));
        var blockA = new byte[kvOnly];
        var blockB = new byte[full];
        Assert.True(model.TryExtractKVBlock(0, first, blockA));
        Assert.True(model.TryExtractKVBlock(first, second, blockB));
        Assert.False(model.TryExtractKVBlock(0, first, new byte[kvOnly + 1]));

        // Dirty the cache and the conv state with another sequence, then restore.
        model.ResetKVCache();
        model.ForwardRefill(Enumerable.Range(0, 20).Select(i => 100 + i).ToArray());
        model.ResetKVCache();
        Assert.False(model.TryInjectKVBlock(4, first, blockA)); // only in order
        Assert.True(model.TryInjectKVBlock(0, first, blockA));
        Assert.True(model.TryInjectKVBlock(first, second, blockB));

        Assert.Equal(expected, model.Forward(new[] { next }));
    }

    [Fact]
    public async Task ConcurrentEngineRequestsMatchEachRequestServedAlone()
    {
        var (_, path) = Build();
        const int maxNewTokens = 8;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 27).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };

        var expected = new List<int[]>();
        using (var reference = ModelBase.Create(path, BackendType.Cpu))
        {
            foreach (int[] prompt in prompts)
            {
                reference.ResetKVCache();
                float[] logits = reference.ForwardRefill(prompt);
                var produced = new List<int>();
                for (int i = 0; i < maxNewTokens; i++)
                {
                    int token = ArgMax(logits);
                    produced.Add(token);
                    if (reference.Tokenizer.EosTokenIds.Contains(token)) break;
                    if (i + 1 < maxNewTokens) logits = reference.Forward(new[] { token });
                }
                expected.Add(produced.ToArray());
            }
        }

        using var model = ModelBase.Create(path, BackendType.Cpu);
        Assert.IsNotAssignableFrom<IBatchedPagedModel>(model); // the snapshot route
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 8,
            NumBlocks = 32,
            BlockSize = 4,
            EnablePrefixCaching = true,
            // Swap ownership of the one cache between the requests many times each.
            DecodeQuantumTokens = 2,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var handles = prompts.Select((p, i) => engine.SubmitRequest(
            new SequenceState($"lfm3-{i}", p, maxNewTokens, cfg.BlockSize, SamplingConfig.Greedy))).ToList();
        var outputs = await Task.WhenAll(handles.Select(async h =>
        {
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync())
                tokens.Add(t);
            await h.Completion;
            return tokens.ToArray();
        }));

        for (int i = 0; i < prompts.Count; i++)
        {
            int n = Math.Min(outputs[i].Length, expected[i].Length);
            Assert.True(n > 0, $"request {i} produced nothing");
            Assert.Equal(expected[i].Take(n), outputs[i].Take(n));
        }
    }

    [Theory]
    [InlineData(2, false)]
    [InlineData(2, true)]
    public void TensorParallelMatchesTheReferenceAndSnapshotsLikeOneDevice(int degree, bool quantized)
    {
        var (fixture, path) = Build(new Lfm3SyntheticModelBuilder { QuantizeProjections = quantized });
        const float tolerance = 2e-3f;
        float[][] all = quantized ? SingleDeviceLogits(path) : fixture.ReferenceAll(Prompt);

        using var group = new MuseSparkTensorParallelTests.CpuGroup(degree);
        using var model = new Lfm3Model(path, BackendType.Cpu, tpDegree: degree, tpGroup: group);
        Close(all[9], model.Forward(Prompt[..10]), tolerance);
        for (int position = 10; position < 13; position++)
            Close(all[position], model.Forward(new[] { Prompt[position] }), tolerance);
        Assert.True(group.Reductions > 0);

        // The TP block layout is the single-device one, byte for byte.
        using var single = new Lfm3Model(path, BackendType.Cpu);
        single.Forward(Prompt[..13]);
        var tpBlock = new byte[model.ComputeKVBlockByteSize(13)];
        var oneBlock = new byte[single.ComputeKVBlockByteSize(13)];
        Assert.Equal(oneBlock.Length, tpBlock.Length);
        Assert.True(model.TryExtractKVBlock(0, 13, tpBlock));
        Assert.True(single.TryExtractKVBlock(0, 13, oneBlock));
        if (!quantized)
            AssertBytesClose(oneBlock, tpBlock);

        // A single-device snapshot restores into the TP model and decoding carries on.
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 13, oneBlock));
        for (int position = 13; position < Prompt.Length; position++)
            Close(all[position], model.Forward(new[] { Prompt[position] }), tolerance);
    }

    /// <summary>
    /// Every position's logits from a single-device model fed one token at a time: the
    /// oracle for a quantized TP run, whose Q8_0 activation rounding the float64 reference
    /// does not share. A rank's row-parallel slice starts on a Q8_0 block boundary, so the
    /// shards quantize exactly the blocks the whole row does.
    /// </summary>
    private static float[][] SingleDeviceLogits(string path)
    {
        using var single = new Lfm3Model(path, BackendType.Cpu);
        return Prompt.Select(token => (float[])single.Forward(new[] { token }).Clone()).ToArray();
    }

    /// <summary>Two F32 snapshots whose floats differ only by reduction order.</summary>
    private static void AssertBytesClose(byte[] expected, byte[] actual)
    {
        var e = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(expected);
        var a = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(actual);
        for (int i = 0; i < e.Length; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) <= 1e-4f + 1e-4f * MathF.Abs(e[i]), $"snapshot float {i}: {e[i]:R} vs {a[i]:R}");
    }

    /// <summary>The GGML route: its native masked softmax (causal + window) and the device
    /// copies a host-side conv, rope, gate or snapshot write has to invalidate.</summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void GgmlCpuMatchesTheReferenceAndRestoresASnapshot()
    {
        var (fixture, path) = Build();
        float[][] all = fixture.ReferenceAll(Prompt);
        using var model = ModelBase.Create(path, BackendType.GgmlCpu);
        Assert.IsType<Lfm3Model>(model);

        int position = 0;
        foreach (int chunk in new[] { 3, 5, 2 })
        {
            float[] logits = model.Forward(Prompt[position..(position + chunk)]);
            position += chunk;
            Close(all[position - 1], logits, 2e-3f);
        }
        for (; position < Prompt.Length - 1; position++)
            Close(all[position], model.Forward(new[] { Prompt[position] }), 2e-3f);

        var block = new byte[model.ComputeKVBlockByteSize(position)];
        Assert.True(model.TryExtractKVBlock(0, position, block));
        model.ResetKVCache();
        model.Forward(Prompt[..7]);
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, position, block));
        Close(all[position], model.Forward(new[] { Prompt[position] }), 2e-3f);
    }

    /// <summary>
    /// The native CUDA forward (ggml_ops_lfm3.cu): every kernel against the float64
    /// reference. Prefill and single tokens interleave over one device-resident cache and
    /// conv state, past the window and the first cache capacity; a snapshot taken from the
    /// device restores into a reset model in both block forms. Head size 64 attends through
    /// flash attention and 16, which has no flash kernel, through the guard's explicit path.
    /// Multi-token F32 matmuls on CUDA round at about 2e-2 here, as the per-op route does.
    /// </summary>
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(KvCacheDtype.F32, 16)]
    [InlineData(KvCacheDtype.F16, 16)]
    [InlineData(KvCacheDtype.F32, 64)]
    [InlineData(KvCacheDtype.F16, 64)]
    public void GgmlCudaKernelsMatchTheReferenceAndRestoreASnapshot(KvCacheDtype dtype, int headDim)
    {
        const float tolerance = 6e-2f;
        // Device-resident caches need a size worth caching, and the first capacity is
        // short of this prompt so the forwards below grow it.
        _environment.Set("TS_KV_INITIAL_TOKENS", "64");
        KvCacheDtypeConfig.Set(dtype);
        int[] prompt = Enumerable.Repeat(Prompt, 6).SelectMany(tokens => tokens).ToArray();
        var (fixture, path) = Build(new Lfm3SyntheticModelBuilder { HeadDim = headDim });
        float[][] all = fixture.ReferenceAll(prompt);
        using var model = Assert.IsType<Lfm3Model>(ModelBase.Create(path, BackendType.GgmlCuda));
        Assert.True(model.UsesNativeForward);

        int position = 0;
        foreach (int chunk in new[] { 3, 1, 5, 1, 1, 2, 7, 1, 50, 1 })
        {
            float[] logits = model.Forward(prompt[position..(position + chunk)]);
            position += chunk;
            Close(all[position - 1], logits, tolerance);
        }
        Assert.True(model.UsesNativeForward);

        int blockEnd = position;
        var full = new byte[model.ComputeKVBlockByteSize(blockEnd)];
        Assert.True(model.TryExtractKVBlock(0, blockEnd, full));
        Close(all[position], model.Forward(new[] { prompt[position] }), tolerance);
        position++;

        // A reset must clear the device's conv state, or this first token would see it.
        model.ResetKVCache();
        Close(all[0], model.Forward(prompt[..1]), tolerance);
        model.ResetKVCache();

        // Prefix (K/V only) then the rest with its state: what an engine swap does.
        int prefix = 40;
        var keysOnly = new byte[model.ComputeKVBlockByteSizeWithoutRecurrentState(prefix)];
        var tail = new byte[model.ComputeKVBlockByteSize(blockEnd - prefix)];
        using (var source = Assert.IsType<Lfm3Model>(ModelBase.Create(path, BackendType.GgmlCuda)))
        {
            Assert.True(source.UsesNativeForward);
            source.Forward(prompt[..blockEnd]);
            Assert.True(source.TryExtractKVBlock(0, prefix, keysOnly));
            Assert.True(source.TryExtractKVBlock(prefix, blockEnd - prefix, tail));
        }
        Assert.True(model.TryInjectKVBlock(0, prefix, keysOnly));
        Assert.True(model.TryInjectKVBlock(prefix, blockEnd - prefix, tail));
        Close(all[blockEnd], model.Forward(new[] { prompt[blockEnd] }), tolerance);
        Close(all[blockEnd + 1], model.Forward(prompt[(blockEnd + 1)..(blockEnd + 2)]), tolerance);

        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, blockEnd, full));
        Close(all[blockEnd], model.Forward(new[] { prompt[blockEnd] }), tolerance);
    }

    /// <summary>One token at a time everything but the matmuls is a TensorSharp kernel and
    /// the matmuls take the exact vector path, so the short conv, the Q/K preparation and the
    /// split-KV attention are held to the reference tightly, past the window.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void GgmlCudaDecodeKernelsAreExactToTheReference()
    {
        _environment.Set("TS_KV_INITIAL_TOKENS", "64");
        int[] prompt = Enumerable.Repeat(Prompt, 3).SelectMany(tokens => tokens).ToArray();
        var (fixture, path) = Build();
        float[][] all = fixture.ReferenceAll(prompt);
        using var model = Assert.IsType<Lfm3Model>(ModelBase.Create(path, BackendType.GgmlCuda));
        Assert.True(model.UsesNativeForward);
        for (int position = 0; position < prompt.Length; position++)
            Close(all[position], model.Forward(new[] { prompt[position] }), 1e-4f);
    }

    /// <summary>The native forward refuses a cache it could not keep on the device, and the
    /// per-op route carries the model instead of failing.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void GgmlCudaTooSmallACacheStaysOnThePerOpPath()
    {
        var (fixture, path) = Build();
        using var model = Assert.IsType<Lfm3Model>(ModelBase.Create(path, BackendType.GgmlCuda));
        Assert.False(model.UsesNativeForward);
        Close(fixture.ReferenceAll(Prompt)[Prompt.Length - 1], model.Forward(Prompt), 6e-2f);
    }

    /// <summary>GGML tensor parallelism over two real devices against the same backend on
    /// one: the per-rank shards resident on each GPU, and the snapshot gathered from them.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData(false)]
    [InlineData(true)]
    public void GgmlCudaTensorParallelMatchesOneDevice(bool quantized)
    {
        var (_, path) = Build(new Lfm3SyntheticModelBuilder { QuantizeProjections = quantized });
        float[][] expected;
        using (var single = ModelBase.Create(path, BackendType.GgmlCuda))
            expected = Prompt.Select(token => (float[])single.Forward(new[] { token }).Clone()).ToArray();

        using var tp = Assert.IsType<Lfm3Model>(ModelBase.Create(path, BackendType.GgmlCuda, tpDegree: 2));
        Close(expected[9], tp.Forward(Prompt[..10]), 2e-3f);
        for (int position = 10; position < 13; position++)
            Close(expected[position], tp.Forward(new[] { Prompt[position] }), 2e-3f);

        var block = new byte[tp.ComputeKVBlockByteSize(13)];
        Assert.True(tp.TryExtractKVBlock(0, 13, block));
        tp.ResetKVCache();
        Assert.True(tp.TryInjectKVBlock(0, 13, block));
        for (int position = 13; position < Prompt.Length; position++)
            Close(expected[position], tp.Forward(new[] { Prompt[position] }), 2e-3f);
    }

    /// <summary>With caches worth keeping on the device, each GPU runs the native graph over
    /// its shard, and the result, a cache that grew and a restored snapshot all match one
    /// device. Reduced sums round differently, so this is the multi-token tolerance.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData(false)]
    [InlineData(true)]
    public void GgmlCudaTensorParallelNativeForwardMatchesOneDevice(bool quantized)
    {
        const float tolerance = 6e-2f;
        _environment.Set("TS_KV_INITIAL_TOKENS", "64");
        int[] prompt = Enumerable.Repeat(Prompt, 6).SelectMany(tokens => tokens).ToArray();
        var (_, path) = Build(new Lfm3SyntheticModelBuilder { HeadDim = 64, QuantizeProjections = quantized });
        var expected = new float[prompt.Length][];
        using (var single = Assert.IsType<Lfm3Model>(ModelBase.Create(path, BackendType.GgmlCuda)))
        {
            Assert.True(single.UsesNativeForward);
            for (int position = 0; position < prompt.Length; position++)
                expected[position] = (float[])single.Forward(new[] { prompt[position] }).Clone();
        }

        using var tp = Assert.IsType<Lfm3Model>(ModelBase.Create(path, BackendType.GgmlCuda, tpDegree: 2));
        Assert.True(tp.UsesNativeForward);
        Close(expected[69], tp.Forward(prompt[..70]), tolerance);
        Close(expected[70], tp.Forward(prompt[70..71]), tolerance);

        var block = new byte[tp.ComputeKVBlockByteSize(71)];
        Assert.True(tp.TryExtractKVBlock(0, 71, block));
        tp.ResetKVCache();
        Close(expected[0], tp.Forward(prompt[..1]), tolerance);
        tp.ResetKVCache();
        Assert.True(tp.TryInjectKVBlock(0, 71, block));
        for (int position = 71; position < prompt.Length; position++)
            Close(expected[position], tp.Forward(new[] { prompt[position] }), tolerance);
        Assert.True(tp.UsesNativeForward);
    }

    [Fact]
    public void TensorParallelRefusesADegreeThatDoesNotDivideTheHeads()
    {
        var (_, path) = Build();
        using var group = new MuseSparkTensorParallelTests.CpuGroup(3);
        var error = Assert.Throws<NotSupportedException>(() => new Lfm3Model(path, BackendType.Cpu, tpDegree: 3, tpGroup: group));
        Assert.Contains("--tp 3", error.Message);
    }

    [Fact]
    public void Lfm2PreTokenizerGroupsDigitsInThrees()
    {
        string pattern = BpeTokenizer.ResolvePreTokenizerPattern("lfm2");
        Assert.NotEqual(BpeTokenizer.ResolvePreTokenizerPattern(null), pattern);
        var matches = System.Text.RegularExpressions.Regex.Matches("year 2024", pattern).Select(m => m.Value);
        Assert.Equal(new[] { "year", " ", "202", "4" }, matches);
    }
}
