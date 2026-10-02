// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma4VerifyPleTests
{
    private const string ModelVariable = "TS_GMTP_TARGET";
    private readonly ITestOutputHelper _output;

    public Gemma4VerifyPleTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(0, 0, 7)]
    [InlineData(0, 1000, 7)]       // Q8 and other formats retain their existing width.
    [InlineData(10, 1000, 7)]     // One small IQ4 PLE matrix must not change it.
    [InlineData(499, 1000, 7)]
    [InlineData(500, 1000, 2)]
    [InlineData(1000, 1000, 2)]   // The IQ4-dominated bandwidth case.
    public void MetalDraftWindow_ChangesOnlyForPredominantlyIq4Matrices(long iq4Bytes, long totalBytes, int expected)
        => Assert.Equal(expected, Gemma4Model.SelectMetalSpecDraftWindow(iq4Bytes, totalBytes));

    [Theory]
    [InlineData("blk.0.attn_q.weight", false, 7)]
    [InlineData("output.weight", false, 7)]
    [InlineData("token_embd.weight", true, 7)]
    [InlineData("token_embd.weight", false, 2)]
    [InlineData("per_layer_token_embd.weight", false, 2)]
    [InlineData("mtp.output.weight", false, 2)]
    public void MetalDraftWindow_CountsF32MatricesButNotEmbeddingOnlyOrDraftTables(
        string name, bool hasTiedOutput, int expected)
    {
        var allocator = new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet);
        using var f32 = new Tensor(allocator, DType.Float32, 16, 16);
        using var iq4 = new QuantizedWeight(new byte[272],
            (int)TensorSharp.Runtime.GgmlTensorType.IQ4_XS, 256, 2);
        var quantWeights = new Dictionary<string, QuantizedWeight> { ["blk.0.ffn_down.weight"] = iq4 };
        var weights = new Dictionary<string, Tensor> { [name] = f32 };
        Assert.Equal(expected, Gemma4Model.SelectMetalSpecDraftWindow(quantWeights, weights, hasTiedOutput));
    }

    [Fact]
    public void MetalDraftWindow_DoesNotCountVectorsOrF32MirrorsOfQuantizedMatrices()
    {
        var allocator = new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet);
        using var mirror = new Tensor(allocator, DType.Float32, 2, 256);
        using var norm = new Tensor(allocator, DType.Float32, 256);
        using var iq4 = new QuantizedWeight(new byte[272],
            (int)TensorSharp.Runtime.GgmlTensorType.IQ4_XS, 256, 2);
        var quantWeights = new Dictionary<string, QuantizedWeight> { ["blk.0.ffn_down.weight"] = iq4 };
        var weights = new Dictionary<string, Tensor>
        {
            ["blk.0.ffn_down.weight"] = mirror,
            ["blk.0.attn_norm.weight"] = norm,
        };
        Assert.Equal(2, Gemma4Model.SelectMetalSpecDraftWindow(quantWeights, weights, false));
    }

    [Fact]
    public void MetalDraftWindow_CountsFusedGateUpInsteadOfItsRetainedOriginals()
    {
        var allocator = new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet);
        using var gate = new Tensor(allocator, DType.Float32, 2, 256);
        using var up = new Tensor(allocator, DType.Float32, 2, 256);
        using var fused = new QuantizedWeight(new byte[544],
            (int)TensorSharp.Runtime.GgmlTensorType.IQ4_XS, 256, 4);
        var quantWeights = new Dictionary<string, QuantizedWeight> { ["blk.0.ffn_gate_up.weight"] = fused };
        var weights = new Dictionary<string, Tensor>
        {
            ["blk.0.ffn_gate.weight"] = gate,
            ["blk.0.ffn_up.weight"] = up,
        };
        Assert.Equal(2, Gemma4Model.SelectMetalSpecDraftWindow(quantWeights, weights, false));
    }

    // Example: TS_GMTP_TARGET=/models/gemma-4-E4B-it-IQ4_XS.gguf
    //          TS_TEST_GGML_BACKEND=metal dotnet test --filter Gemma4VerifyPleTests
    // The model gate reports missing weights as skipped. Once opted in, an
    // incompatible checkpoint or an unavailable native PLE path fails the test.
    [ModelFact(ModelVariable)]
    public void ResidentPle_MatchesUploadedPleAcrossVerifySizesAndSlidingWindowBoundaries()
    {
        string path = Environment.GetEnvironmentVariable(ModelVariable);
        Assert.True(File.Exists(path), $"{ModelVariable} must name a Gemma 4 E-series GGUF file.");
        BackendType backend = ReadBackend();
        using var nativeEnvironment = new NativeEnvScope();
        using ModelBase loaded = ModelBase.Create(path, backend);
        Gemma4Model model = Assert.IsType<Gemma4Model>(loaded);
        var target = (ISpeculativeTarget)model;

        foreach (int promptLength in new[] { 509, 520, 1024 })
        {
            int[] prompt = BuildPrompt(model, promptLength);
            foreach (int rows in new[] { 1, 2, 3, 8 })
            {
                // Hold token IDs and prefill identical. Compare the original
                // graph with resident PLE, then isolate donor-window reuse
                // with PLE unchanged and require bitwise equality.
                int[] tokens = prompt.AsSpan(prompt.Length - rows).ToArray();
                var reference = Run(false, false);
                var resident = Run(true, false);
                var shared = Run(true, true);
                string label = $"prompt={promptLength}, rows={rows}";
                AssertClose(reference.hidden, resident.hidden, label + " hidden");
                AssertClose(reference.logits, resident.logits, label + " logits");
                Assert.True(MemoryMarshal.Cast<float, int>(resident.hidden.AsSpan()).SequenceEqual(
                    MemoryMarshal.Cast<float, int>(shared.hidden.AsSpan())), label + " shared KV hidden differs bitwise");
                Assert.True(MemoryMarshal.Cast<float, int>(resident.logits.AsSpan()).SequenceEqual(
                    MemoryMarshal.Cast<float, int>(shared.logits.AsSpan())), label + " shared KV logits differs bitwise");
                _output.WriteLine(label + ": shared KV hidden/logits match bitwise");
                int vocab = model.Config.VocabSize;
                for (int row = 0; row < rows; row++)
                    Assert.Equal(Argmax(reference.logits.AsSpan(row * vocab, vocab)),
                        Argmax(resident.logits.AsSpan(row * vocab, vocab)));

                (float[] hidden, float[] logits) Run(bool residentPle, bool reuseSharedKv)
                {
                    nativeEnvironment.Set("TS_GMTP_REUSE_KV", "0");
                    model.ResetKVCache();
                    model.ForwardRefill(prompt);
                    Assert.Equal(promptLength, target.CacheSeqLen);
                    long gathersBefore = GatherCount(model);
                    model.GatherPleInVerify = residentPle;
                    nativeEnvironment.Set("TS_GMTP_REUSE_KV", reuseSharedKv ? "1" : "0");
                    var hidden = new float[rows * model.Config.HiddenSize];
                    var logits = new float[rows * model.Config.VocabSize];
                    target.SpecForward(tokens, hidden, logits, allLogitsRows: true);
                    Assert.Equal(promptLength + rows, target.CacheSeqLen);
                    Assert.Equal(gathersBefore + (residentPle && rows > 1 ? 1 : 0), GatherCount(model));
                    return (hidden, logits);
                }
            }
        }
    }

    [ModelFact(ModelVariable)]
    public void ScaledPleProjection_UsesUploadedFallbackAndPreservesOutputs()
    {
        string path = Environment.GetEnvironmentVariable(ModelVariable);
        Assert.True(File.Exists(path), $"{ModelVariable} must name a Gemma 4 E-series GGUF file.");
        using var model = new InspectableGemma4Model(path, ReadBackend());
        QuantizedWeight projection = model.PleProjection;
        Assert.NotNull(projection);
        float savedScale = projection.Scale;
        var target = (ISpeculativeTarget)model;
        int[] prompt = BuildPrompt(model, 64);
        int[] tokens = prompt.AsSpan(prompt.Length - 3).ToArray();
        try
        {
            // Establish that this checkpoint/backend can actually gather PLE;
            // otherwise an unavailable fast path could make the test pass vacuously.
            projection.Scale = 1.0f;
            Run(requestGather: true, expectedGather: true);
            foreach (float scale in new[] { 0.5f, -1.0f })
            {
                projection.Scale = scale;
                var reference = Run(requestGather: false, expectedGather: false);
                var fallback = Run(requestGather: true, expectedGather: false);
                AssertClose(reference.hidden, fallback.hidden, $"PLE scale={scale} hidden");
                AssertClose(reference.logits, fallback.logits, $"PLE scale={scale} logits");
                for (int row = 0; row < tokens.Length; row++)
                {
                    int offset = row * model.Config.VocabSize;
                    Assert.Equal(Argmax(reference.logits.AsSpan(offset, model.Config.VocabSize)),
                        Argmax(fallback.logits.AsSpan(offset, model.Config.VocabSize)));
                }
            }
        }
        finally
        {
            projection.Scale = savedScale;
        }

        (float[] hidden, float[] logits) Run(bool requestGather, bool expectedGather)
        {
            model.ResetKVCache();
            model.ForwardRefill(prompt);
            long gathersBefore = GatherCount(model);
            model.GatherPleInVerify = requestGather;
            var hidden = new float[tokens.Length * model.Config.HiddenSize];
            var logits = new float[tokens.Length * model.Config.VocabSize];
            target.SpecForward(tokens, hidden, logits, allLogitsRows: true);
            Assert.Equal(prompt.Length + tokens.Length, target.CacheSeqLen);
            Assert.Equal(gathersBefore + (expectedGather ? 1 : 0), GatherCount(model));
            return (hidden, logits);
        }
    }

    private sealed class InspectableGemma4Model : Gemma4Model
    {
        public InspectableGemma4Model(string path, BackendType backend) : base(path, backend) { }
        public QuantizedWeight PleProjection => _quantWeights.GetValueOrDefault("per_layer_model_proj.weight");
    }

    // Native diagnostic switches use getenv. Keep libc's table synchronized
    // with .NET's table, as in Glm5NextNativeTensorParallelTests; changing only
    // the managed environment could leave this A/B test on one native path.

    private static BackendType ReadBackend()
        => TestGates.PinnedGgmlBackend;

    private void AssertClose(float[] reference, float[] actual, string label)
    {
        Assert.Equal(reference.Length, actual.Length);
        double squaredError = 0, squaredNorm = 0, maxError = 0, maxValue = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            if (!float.IsFinite(reference[i]) || !float.IsFinite(actual[i]))
                Assert.Fail($"{label}: non-finite value at {i}");
            double difference = actual[i] - reference[i];
            squaredError += difference * difference;
            squaredNorm += (double)reference[i] * reference[i];
            maxError = Math.Max(maxError, Math.Abs(difference));
            maxValue = Math.Max(maxValue, Math.Abs(reference[i]));
        }
        double relativeL2 = Math.Sqrt(squaredError / Math.Max(1e-30, squaredNorm));
        _output.WriteLine($"{label}: relative L2={relativeL2:E3}, max absolute error={maxError:E3}");
        // The same F32 operations may fuse differently in the whole graph. These
        // bounds admit reduction rounding while still detecting wrong PLE rows,
        // omitted projection/norm/scaling, or stale verify inputs.
        Assert.True(relativeL2 <= 1e-4, $"{label}: relative L2={relativeL2:E3}");
        Assert.True(maxError <= 5e-4 * Math.Max(1, maxValue), $"{label}: max absolute error={maxError:E3}");
    }

    private static long GatherCount(Gemma4Model model)
    {
        string report = model.DescribeSpecProfile();
        const string marker = "ple-gather=";
        int start = report.IndexOf(marker, StringComparison.Ordinal);
        Assert.True(start >= 0, "The verify profile must report resident PLE executions.");
        return long.Parse(report.AsSpan(start + marker.Length), CultureInfo.InvariantCulture);
    }

    private static int[] BuildPrompt(ModelBase model, int count)
    {
        var text = new StringBuilder();
        for (int line = 0; ; line++)
        {
            text.AppendLine($"var item{line} = new Widget(); item{line}.Add({line + 3});");
            if (line % 40 != 39) continue;
            var tokens = model.Tokenizer.Encode(text.ToString(), addSpecial: true);
            if (tokens.Count >= count) return tokens.Take(count).ToArray();
        }
    }

    private static int Argmax(ReadOnlySpan<float> values)
    {
        int index = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[index]) index = i;
        return index;
    }
}
