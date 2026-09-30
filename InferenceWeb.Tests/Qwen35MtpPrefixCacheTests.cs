// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Opt-in against the server configuration's actual learned NextN head:
//   TS_SPEC_ADAPTIVE=1 TS_TEST_QWEN35_MTP_MODEL=/path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf TS_TEST_GGML_BACKEND=metal
//   dotnet test InferenceWeb.Tests --filter FullyQualifiedName~Qwen35MtpPrefixCacheTests
// Missing weights or a non-Metal test process are discovery-time skips, not validation passes.
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
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]

public sealed class Qwen35MtpPrefixCacheTests
{
    private const string ModelEnvironment = "TS_TEST_QWEN35_MTP_MODEL";
    private const string ModelName = "qwen3.8-27b-ud-q4_k_xl";
    private const int BlockSize = 64;
    private const int NewTokens = 64;
    private readonly ITestOutputHelper _output;

    private const string Passage =
        "The lighthouse keeper counted the ships each evening, writing their names in a ledger " +
        "bound in green cloth. Seventeen on Monday, twelve on Tuesday, none at all on the night of " +
        "the storm, when the lamp itself seemed to flinch at every gust. By Friday the ledger had " +
        "a new column, for the birds that rested on the gallery rail before crossing the bay. ";

    public Qwen35MtpPrefixCacheTests(ITestOutputHelper output) => _output = output;

    [ModelTheory(ModelEnvironment, ModelName, GgmlBackend = BackendType.GgmlMetal)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task MtpAfterStartupWarmupAndSerializedCheckpoint_MatchesPlainDecoding(bool zeroConfidenceGate)
    {
        using var environment = new TestEnvironment();
        using var model = LoadModel();
        var prefix = model.Tokenizer.Encode("Read the following passage carefully. " + Passage + Passage);
        // Exercise the non-page-aligned checkpoint tail as well as complete blocks.
        while (prefix.Count % BlockSize == 0)
            prefix.AddRange(model.Tokenizer.Encode(" Remember this."));
        Assert.True(prefix.Count > BlockSize);
        var prompt = prefix.Concat(model.Tokenizer.Encode(
            "\n\nRepeat the complete passage above word for word, beginning with The lighthouse keeper " +
            "and preserving all the details. Do not summarize or add a heading:\n")).ToList();
        Assert.True(prompt.Count - prefix.Count > 16);

        Result plain;
        using (var engine = CreateEngine(model, speculate: false))
        {
            plain = await GenerateAsync(engine, prompt, "cold-plain", "cold-plain");
            Assert.Equal(0, plain.Completion.PrefixCacheReusedTokens);
            Assert.Null(plain.Sequence.SpecStats);
        }
        Assert.Equal(NewTokens, plain.Output.Count);

        var store = new SerializedCheckpointStore();
        using (var engine = CreateEngine(model, speculate: true, zeroConfidenceGate))
        {
            engine.PrefixCheckpointStore = store;
            var warmup = await GenerateAsync(engine, prefix, "startup", "startup",
                sharedPrefix: prefix.Count, maxTokens: 1);
            Assert.Single(warmup.Output);
            Assert.Equal(0, warmup.Completion.PrefixCacheReusedTokens);
            Assert.Null(warmup.Sequence.SpecStats);

            var cached = await GenerateAsync(engine, prompt, "cached-mtp", "cached-mtp",
                sharedPrefix: prefix.Count);
            AssertMtpAndReuse(cached, prefix.Count);
            Assert.Equal(plain.Output, cached.Output);
        }
        Assert.Equal(1, store.Saves);
        Assert.True(store.Bytes > 0);

        // A fresh engine has no live cache or in-memory radix checkpoint. Reopen the
        // model's serialized bytes so stale/uninitialized MTP rows cannot be hidden
        // by the state of the request that originally produced the checkpoint.
        using (var engine = CreateEngine(model, speculate: true, zeroConfidenceGate))
        {
            engine.PrefixCheckpointStore = store;
            var restored = await GenerateAsync(engine, prompt, "restored-mtp", "restored-mtp",
                sharedPrefix: prefix.Count);
            Assert.Equal(1, store.Opens);
            AssertMtpAndReuse(restored, prefix.Count);
            Assert.Equal(plain.Output, restored.Output);
        }
        _output.WriteLine($"Serialized checkpoint payload: {store.Bytes / 1048576.0:F1} MiB; " +
            "export/import exercised in memory (filesystem framing is not part of this test).");
    }

    [ModelFact(ModelEnvironment, ModelName, GgmlBackend = BackendType.GgmlMetal)]
    public async Task MtpOnRetainedHolderAfterConcurrentRequests_MatchesPlainDecoding()
    {
        using var environment = new TestEnvironment();
        using var model = LoadModel();
        var prompt = model.Tokenizer.Encode(Passage + Passage +
            "\n\nRepeat the passage above word for word, starting at its first sentence:\n");
        var other = model.Tokenizer.Encode(
            "Write a long numbered list of familiar rivers, mountains, countries, cities, and oceans:\n1.");
        var followup = model.Tokenizer.Encode(
            "\n\nNow repeat the original passage again, starting with The lighthouse keeper " +
            "and continuing without commentary:\n");

        async Task<(Result First, Result Followup)> ConversationAsync(bool speculate, string tag)
        {
            using var engine = CreateEngine(model, speculate);
            var first = new SequenceState(tag + "-first", prompt, 24, BlockSize, SamplingConfig.Greedy,
                cacheScope: tag);
            var neighbor = new SequenceState(tag + "-neighbor", other, 24, BlockSize, SamplingConfig.Greedy,
                cacheScope: tag + "-neighbor");
            InferenceRequestHandle firstHandle;
            InferenceRequestHandle neighborHandle;
            // The worker takes this same lock when admitting each command. Queue
            // both before releasing it so the first scheduled step sees both.
            lock (model.GpuComputeLock)
            {
                firstHandle = engine.SubmitRequest(first);
                neighborHandle = engine.SubmitRequest(neighbor);
            }
            var firstTask = DrainAsync(firstHandle);
            var neighborTask = DrainAsync(neighborHandle);
            await Task.WhenAll(firstTask, neighborTask);
            var initial = await firstTask;
            Assert.Equal(24, initial.Output.Count);
            Assert.Equal(24, (await neighborTask).Output.Count);
            Assert.Equal(0, initial.Completion.PrefixCacheReusedTokens);

            var nextPrompt = prompt.Concat(initial.Output).Concat(followup).ToList();
            var next = await GenerateAsync(engine, nextPrompt, tag + "-followup", tag);
            // This request has no public shared prefix and is a scoped continuation;
            // reusing its generated rows requires the exact retained hybrid state.
            Assert.True(next.Completion.PrefixCacheReusedTokens > prompt.Count);
            Assert.Equal(initial.Sequence.NumComputedTokens, next.Completion.PrefixCacheReusedTokens);
            return (initial, next);
        }

        var plain = await ConversationAsync(speculate: false, "plain");
        var speculative = await ConversationAsync(speculate: true, "mtp");
        Assert.Null(plain.Followup.Sequence.SpecStats);
        Assert.Equal(NewTokens, plain.Followup.Output.Count);
        Assert.Equal(plain.First.Output, speculative.First.Output);
        Assert.Equal(plain.Followup.Output, speculative.Followup.Output);
        AssertMtpAndReuse(speculative.Followup, speculative.First.Sequence.NumComputedTokens);
    }

    private ModelBase LoadModel()
    {
        string configured = Environment.GetEnvironmentVariable(ModelEnvironment);
        string path = File.Exists(configured) ? configured : TestGates.FindGguf(configured, ModelName);
        Assert.False(string.IsNullOrEmpty(path), "The model passed discovery but is no longer available.");
        _output.WriteLine($"Model: {path}; backend: {TestGates.PinnedGgmlBackend}");
        var model = ModelBase.Create(path, TestGates.PinnedGgmlBackend);
        try
        {
            var target = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
            var head = Assert.IsAssignableFrom<IDraftHead>(model);
            Assert.True(head.HasDraftHead);
            Assert.True(head.DraftHeadResumesAfterGap);
            Assert.Equal(DraftHeadKind.PerToken, head.DraftHeadKind);
            Assert.Equal(KvCacheDtype.Q8_0, model.KvCacheDtype);
            Assert.True(target.SpecTrunkFollowsBoundCache);
            Assert.True(Assert.IsAssignableFrom<IBatchedPagedModel>(model).SupportsPerSequenceFusedForward);
            return model;
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    private static InferenceEngine CreateEngine(ModelBase model, bool speculate, bool zeroConfidenceGate = true)
    {
        var engine = new InferenceEngine(model, new SchedulerConfig
        {
            MaxNumBatchedTokens = 1024,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 256,
            SoloPrefillChunkSize = 256,
            NumBlocks = 64,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            StopRepetition = false,
            Speculation = new SpeculationOptions
            {
                Enabled = speculate,
                SpeculatorName = SpeculatorRegistry.Auto,
                MaxDraftTokens = 3,
                // Exercise both a zero gate and the default confidence gate.
                // The adaptive governor stays enabled in every case.
                MinDraftProb = zeroConfidenceGate ? 0 : null,
            },
        }, NullLogger.Instance);
        Assert.True(engine.PrefixCacheActive);
        return engine;
    }

    private void AssertMtpAndReuse(Result result, int expectedReuse)
    {
        Assert.True(expectedReuse > 0);
        Assert.Equal(expectedReuse, result.Completion.PrefixCacheReusedTokens);
        var stats = Assert.IsType<SpeculationStats>(result.Sequence.SpecStats);
        _output.WriteLine($"{result.Sequence.RequestId}: reused={result.Completion.PrefixCacheReusedTokens}, " +
            $"verify={stats.VerifySteps}, drafted={stats.TokensDrafted}, accepted={stats.TokensAccepted}, " +
            $"plain={stats.PlainSteps}, output={result.Output.Count}, parked={stats.ParkedSteps}, " +
            $"governorWins={stats.GovernorWins}, governorLosses={stats.GovernorLosses}");
        Assert.True(stats.VerifySteps > 0, "MTP never verified a window after cache reuse.");
        Assert.True(stats.TokensDrafted > 0, "MTP never proposed a token after cache reuse.");
        Assert.True(stats.TokensAccepted > 0, "MTP never accepted a draft while repeating the passage.");
        Assert.True(stats.PlainMsPerToken > 0, "The adaptive governor never measured its plain baseline.");
    }

    private static Task<Result> GenerateAsync(InferenceEngine engine, List<int> prompt,
        string id, string scope, int sharedPrefix = 0, int maxTokens = NewTokens)
        => DrainAsync(engine.SubmitRequest(new SequenceState(id, prompt, maxTokens, BlockSize,
            SamplingConfig.Greedy, sharedPrefixTokens: sharedPrefix, cacheScope: scope)));

    private static async Task<Result> DrainAsync(InferenceRequestHandle handle)
    {
        var tokens = new List<int>();
        await foreach (int token in handle.Tokens.ReadAllAsync()) tokens.Add(token);
        var completion = await handle.Completion;
        Assert.NotEqual(SequenceStatus.FinishedAborted, completion.Status);
        return new Result(handle.Sequence, completion, tokens);
    }

    private sealed record Result(SequenceState Sequence, InferenceCompletion Completion, List<int> Output);

    private sealed class TestEnvironment : IDisposable
    {
        private readonly Dictionary<string, string> _previous = new(StringComparer.Ordinal);
        private readonly KvCacheDtype _previousDtype = KvCacheDtypeConfig.Current;
        private readonly bool _previousExplicitDtype = KvCacheDtypeConfig.IsExplicitlySet;

        internal TestEnvironment()
        {
            // Match agent-qwen3.8-27b.json, including the quantized cache kernels.
            KvCacheDtypeConfig.Set(KvCacheDtype.Q8_0);
            Set("TS_SPEC_ADAPTIVE", "1");
            // This default is process-static. Detect an earlier test freezing it
            // off rather than silently claiming adaptive-policy coverage.
            Assert.True(new SpeculationCostGovernor().Enabled,
                "Run this fixture in a fresh test process with TS_SPEC_ADAPTIVE=1.");
            Set("TS_PREFIX_CHECKPOINTS_MAX", null);
            Set("TS_RETAINED_FUSED_CACHE_MAX", null);
            Set("TS_PER_SEQ_FUSED", "1");
            Set("TS_SCHED_DISABLE_BATCHED", "0");
            Set("TS_KV_INITIAL_TOKENS", "512");
        }

        private void Set(string name, string value)
        {
            _previous.Add(name, Environment.GetEnvironmentVariable(name));
            Environment.SetEnvironmentVariable(name, value);
        }

        public void Dispose()
        {
            KvCacheDtypeConfig.RestoreForTests(_previousDtype, _previousExplicitDtype);
            foreach (var item in _previous) Environment.SetEnvironmentVariable(item.Key, item.Value);
        }
    }

    private sealed class SerializedCheckpointStore : IPrefixCheckpointStore
    {
        private readonly Dictionary<string, byte[]> _payloads = new(StringComparer.Ordinal);
        internal int Saves { get; private set; }
        internal int Opens { get; private set; }
        internal long Bytes { get; private set; }

        private static string Key(string fingerprint, ReadOnlySpan<int> tokens)
            => fingerprint + "|" + string.Join(",", tokens.ToArray());

        public bool TryOpen(string modelFingerprint, ReadOnlySpan<int> prefixTokens, out Stream payload)
        {
            payload = null;
            if (!_payloads.TryGetValue(Key(modelFingerprint, prefixTokens), out var bytes)) return false;
            Opens++;
            payload = new MemoryStream(bytes, writable: false);
            return true;
        }

        public bool Save(string modelFingerprint, ReadOnlySpan<int> prefixTokens, Action<Stream> writePayload)
        {
            using var stream = new MemoryStream();
            writePayload(stream);
            _payloads[Key(modelFingerprint, prefixTokens)] = stream.ToArray();
            Saves++;
            Bytes = stream.Length;
            return true;
        }
    }
}
