// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Incident (Muse-Glimmer 30B, ggml_metal, multi-agent chat): a parent turn adopted
// 4352 radix pages (= MaxReusablePrefixTokens, the sliding-window ring size),
// prefilled one 256-token chunk past it, lost ownership to a concurrently decoding
// child for one 256-token quantum, and on swap-in EnsureOwnership took its "cannot
// inject" branch (NumComputedTokens > MaxReusablePrefixTokens): ResetForPreemption +
// BlockTable.Clear(). ExecuteStepPerSequence then forwarded the step's planned
// 256-token chunk from position 0 with an empty block table:
//   InvalidOperationException: AdvanceTokens(256) wants 1 blocks but only 0 are allocated.
// A decoding sequence swapped back in past the cap fails the same way one line
// earlier ("has no LastLogits to sample from at position 0").
//
// The fake is a single linear cache (the per-sequence executor's shape) whose
// logits hash every (position, token) row, published as a page-only radix family
// with Muse-Glimmer's cap, and whose block extract refuses a block the "ring" has
// outrun (KvBlockTransfer's reading rule). At HEAD both tests throw; with the fix
// both finish and match an uncontended run token for token.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

public sealed class PerSequenceSwapPastReuseCapTests
{
    private const int VocabSize = 97;

    /// <summary>The incident at production scale: block 256, ring/cap 4352 (17 blocks),
    /// mixed-step prefill chunk 256, decode quantum 256.</summary>
    [Fact]
    public async Task PrefillSwappedBackInPastTheReuseCap_KeepsItsBlocks_AndMatchesColdRun()
    {
        const int blockSize = 256, cap = 17 * blockSize;
        var config = Config(blockSize, maxBatched: 4096, prefillChunk: 256, quantum: 256, numBlocks: 512);

        int[] conversation = Tokens(5000, seed: 7);
        int[] warmPrompt = conversation.Take(4400).Concat(Tokens(20, seed: 3)).ToArray();
        int[] parentPrompt = conversation;                 // shares 4400 tokens with the warm turn
        int[] childPrompt = Tokens(300, seed: 11);
        const int parentMaxNew = 4, childMaxNew = 600;     // child stays below the cap: 900 <= 4352

        int[] coldParent = await RunColdAsync(config, cap, parentPrompt, parentMaxNew);
        int[] coldChild = await RunColdAsync(config, cap, childPrompt, childMaxNew);

        var model = new RingCapHistoryModel(cap);
        var log = new RecordingLogger();
        using var engine = new InferenceEngine(model, config, log);

        // 1. The parent's previous turn, alone: its first 17 blocks become pages of scope "conv".
        await RunAsync(engine, "warm", warmPrompt, maxNew: 2, scope: "conv");

        // 2. A child starts and decodes.
        int decodesBefore = model.DecodeForwards;
        var child = new SequenceState("child", childPrompt.ToList(), childMaxNew, blockSize, SamplingConfig.Greedy, cacheScope: "child");
        var childHandle = engine.SubmitRequest(child);
        await WaitForDecodesAsync(model, decodesBefore, 2);

        // 3. The parent's next turn arrives while the child decodes: it adopts exactly the
        //    page window (4352 tokens) and prefills the rest in 256-token mixed-step chunks.
        var parent = new SequenceState("parent", parentPrompt.ToList(), parentMaxNew, blockSize, SamplingConfig.Greedy, cacheScope: "conv");
        var parentHandle = engine.SubmitRequest(parent);

        // HEAD: faults with "AdvanceTokens(256) wants 1 blocks but only 0 are allocated."
        var parentCompletion = await parentHandle.Completion.WaitAsync(TimeSpan.FromSeconds(60));
        var childCompletion = await childHandle.Completion.WaitAsync(TimeSpan.FromSeconds(60));

        Assert.Equal(SequenceStatus.FinishedLengthCapped, parentCompletion.Status);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, childCompletion.Status);
        Assert.Equal(cap, parentCompletion.PrefixCacheReusedTokens);   // the incident's 4352/…
        Assert.Equal(coldParent, parent.OutputTokens.ToArray());
        Assert.Equal(coldChild, child.OutputTokens.ToArray());
        Assert.Equal(parentPrompt.Length + parent.OutputTokens.Count, parent.NumComputedTokens);

        // Once past the cap the parent keeps the model (a swap-out could not be restored
        // without re-forwarding everything past the cap), so nothing is re-forwarded.
        Assert.Contains(log.Messages, m => m.Contains("past the 4352 this model can swap out"));
        Assert.DoesNotContain(log.Messages, m => m.Contains("Materializing the cached prefix"));
    }

    /// <summary>The same defect on a decoder: a request that decodes past the cap and is
    /// swapped back in lost its logits and blocks (HEAD: "has no LastLogits to sample
    /// from at position 0").</summary>
    [Fact]
    public async Task DecoderSwappedBackInPastTheReuseCap_KeepsItsLogits_AndMatchesColdRun()
    {
        // Timeline at HEAD: long is admitted while child decodes and prefills 46 in one
        // chunk (<= cap); child decodes one quantum; long swaps back in (46 <= cap,
        // restored), decodes one quantum to 54 (> cap); child decodes one quantum; long
        // swaps back in at 54 > cap -> reset branch -> SampleFromLogits throws.
        const int blockSize = 8, cap = 6 * blockSize;
        var config = Config(blockSize, maxBatched: 48, prefillChunk: 48, quantum: 8, numBlocks: 64);

        int[] longPrompt = Tokens(46, seed: 5);      // prompt below the cap, output carries it past
        int[] childPrompt = Tokens(4, seed: 9);
        const int longMaxNew = 16, childMaxNew = 44;  // child: 4 + 44 = 48 <= cap, so only `long` crosses it

        int[] coldLong = await RunColdAsync(config, cap, longPrompt, longMaxNew);
        int[] coldChild = await RunColdAsync(config, cap, childPrompt, childMaxNew);

        var model = new RingCapHistoryModel(cap);
        using var engine = new InferenceEngine(model, config, NullLogger.Instance);

        int decodesBefore = model.DecodeForwards;
        var child = new SequenceState("child", childPrompt.ToList(), childMaxNew, blockSize, SamplingConfig.Greedy, cacheScope: "child");
        var childHandle = engine.SubmitRequest(child);
        await WaitForDecodesAsync(model, decodesBefore, 2);

        var longSeq = new SequenceState("long", longPrompt.ToList(), longMaxNew, blockSize, SamplingConfig.Greedy, cacheScope: "long");
        var longHandle = engine.SubmitRequest(longSeq);

        var longCompletion = await longHandle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
        var childCompletion = await childHandle.Completion.WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(SequenceStatus.FinishedLengthCapped, longCompletion.Status);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, childCompletion.Status);
        Assert.Equal(coldLong, longSeq.OutputTokens.ToArray());
        Assert.Equal(coldChild, child.OutputTokens.ToArray());
    }

    // ------------------------------------------------------------------ helpers

    private sealed class RecordingLogger : Microsoft.Extensions.Logging.ILogger
    {
        private readonly List<string> _messages = new();
        public IReadOnlyList<string> Messages { get { lock (_messages) return _messages.ToArray(); } }
        public IDisposable BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;
        public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel, Microsoft.Extensions.Logging.EventId eventId,
            TState state, Exception exception, Func<TState, Exception, string> formatter)
        {
            lock (_messages) _messages.Add(formatter(state, exception));
        }
    }

    private static SchedulerConfig Config(int blockSize, int maxBatched, int prefillChunk, int quantum, int numBlocks) => new()
    {
        MaxNumBatchedTokens = maxBatched,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = prefillChunk,
        NumBlocks = numBlocks,
        BlockSize = blockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = quantum,
        StopRepetition = false,
    };

    private static int[] Tokens(int count, int seed)
        => Enumerable.Range(0, count).Select(i => 1 + (int)(((uint)(i * 2654435761u) ^ (uint)(seed * 40503)) % (VocabSize - 1))).ToArray();

    private static async Task WaitForDecodesAsync(RingCapHistoryModel model, int before, int count)
    {
        var until = DateTime.UtcNow.AddSeconds(30);
        while (model.DecodeForwards - before < count && DateTime.UtcNow < until)
            await Task.Delay(1);
        Assert.True(model.DecodeForwards - before >= count, "the first request never started decoding");
    }

    private static async Task<int[]> RunColdAsync(SchedulerConfig config, int cap, int[] prompt, int maxNew)
    {
        var model = new RingCapHistoryModel(cap);
        using var engine = new InferenceEngine(model, config, NullLogger.Instance);
        var seq = await RunAsync(engine, "cold", prompt, maxNew, scope: "cold");
        return seq.OutputTokens.ToArray();
    }

    private static async Task<SequenceState> RunAsync(InferenceEngine engine, string id, int[] prompt, int maxNew, string scope)
    {
        var seq = new SequenceState(id, prompt.ToList(), maxNew, engine.PoolStats.blockSize, SamplingConfig.Greedy, cacheScope: scope);
        var completion = await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(60));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        return seq;
    }

    private static ulong Fold(ulong h, int position, int token)
    {
        h = (h ^ (ulong)position) * 1099511628211UL;
        return (h ^ (ulong)token) * 1099511628211UL;
    }

    /// <summary>
    /// One linear cache, like Muse-Glimmer's primary. Logits peak at a hash of every
    /// (position, token) row. <see cref="RingRows"/> is published as
    /// MaxReusablePrefixTokens and the radix page window, and a block whose start the
    /// head has moved more than <see cref="RingRows"/> past cannot be extracted (a ring
    /// layer no longer holds it), exactly the rule KvBlockTransfer.Extract applies.
    /// </summary>
    private sealed class RingCapHistoryModel : IModelArchitecture, IPageOnlyPrefixCacheModel
    {
        private readonly List<int> _rows = new();
        private readonly object _gate = new();
        private int _decodeForwards;

        public RingCapHistoryModel(int ringRows) => RingRows = ringRows;

        public int RingRows { get; }
        public int DecodeForwards => Volatile.Read(ref _decodeForwards);

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new NumberTokenizer();
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public int MaxReusablePrefixTokens => RingRows;
        public string KVStateFingerprint => $"ring-cap-history|rows={RingRows}";

        public float[] Forward(int[] tokens)
        {
            lock (_gate)
            {
                if (tokens.Length == 1)
                {
                    Interlocked.Increment(ref _decodeForwards);
                    Thread.Sleep(1);   // keep the first request decoding while the second arrives
                }
                _rows.AddRange(tokens);
                ulong h = 1469598103934665603UL;
                for (int p = 0; p < _rows.Count; p++) h = Fold(h, p, _rows[p]);
                var logits = new float[VocabSize];
                logits[(int)(h % (ulong)(VocabSize - 1)) + 1] = 10f;
                return logits;
            }
        }

        public void ResetKVCache() { lock (_gate) _rows.Clear(); }

        public void TruncateKVCache(int tokenCount)
        {
            lock (_gate)
            {
                if (tokenCount < _rows.Count)
                    _rows.RemoveRange(tokenCount, _rows.Count - tokenCount);
            }
        }

        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount) => (long)tokenCount * sizeof(int);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            lock (_gate)
            {
                if (startToken < 0 || startToken + tokenCount > _rows.Count) return false;
                if (_rows.Count - startToken > RingRows) return false;   // the ring outran it
                if (destination.Length < tokenCount * sizeof(int)) return false;
                for (int i = 0; i < tokenCount; i++)
                    BitConverter.TryWriteBytes(destination.Slice(i * sizeof(int), sizeof(int)), _rows[startToken + i]);
                return true;
            }
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            lock (_gate)
            {
                if (destToken != _rows.Count || tokenCount > RingRows) return false;
                for (int i = 0; i < tokenCount; i++)
                    _rows.Add(BitConverter.ToInt32(source.Slice(i * sizeof(int), sizeof(int))));
                return true;
            }
        }

        // Muse-Glimmer's record (ModelBase.PageFamilyCapabilities): class S, A1 host pages
        // capped at the ring, resident primary, rewinds within the ring slack.
        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
        {
            Class = FamilyClass.S,
            NamespaceFingerprint = KVStateFingerprint,
            EndState = EndStateSupport.None,
            PrimaryResident = true,
            Truncation = TruncationKind.WithinRingSlack,
            TruncationParameter = 8,
            RewindCapTokens = 16,
            Pages = PageSupport.A1HostSlab,
            PageWindowTokens = RingRows,
        };

        public long QuerySpareBytes(ResourceClass cls) => -1;

        private sealed class NumberTokenizer : ITokenizer
        {
            public string[] Vocab { get; } = Enumerable.Range(0, PerSequenceSwapPastReuseCapTests.VocabSize).Select(i => i.ToString()).ToArray();
            public int BosTokenId => -1;
            public int[] EosTokenIds => Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer) { }
            public bool IsEos(int tokenId) => false;
            public int LookupToken(string tokenStr) => -1;
        }
    }
}
