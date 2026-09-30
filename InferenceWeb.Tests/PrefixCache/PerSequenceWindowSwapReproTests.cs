// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A request swapped back in on the per-sequence executor past the model's pooled reuse
// cap (Muse-Glimmer's 4352-row sliding-window ring) used to kill its own step. A
// Muse-Glimmer 30B multi-agent chat on ggml_metal died with
//   System.InvalidOperationException: AdvanceTokens(256) wants 1 blocks but only 0 are allocated.
//     at BlockTable.AdvanceTokens / BatchExecutor.ExecuteStepPerSequence
// ExecuteStepPerSequence forwards ONE work item per step (rotating on fresh admissions and
// on DecodeQuantumTokens) and calls EnsureOwnership first. For a sequence swapped back in
// with NumComputedTokens > MaxReusablePrefixTokens, EnsureOwnership reset it
// (ResetForPreemption) and freed its blocks, and the step then still forwarded the chunk
// the scheduler had planned at the OLD position into the empty block table. A decoder hit
// the same branch and threw earlier, in SampleFromLogits ("has no LastLogits to sample
// from"), because the reset had dropped its logits. Now an owner past the cap keeps the
// model while it has work, and a swap-in rebuilds in place what the pool cannot restore.
//
// OracleFakes.S2 is the Muse-Glimmer oracle: no holders, A1 host pages, PageWindowTokens =
// MaxReusablePrefixTokens = MuseRingRows (80 = 5 pages of 16), WithinRingSlack truncation.
// It implements IBatchedPagedModel but BatchedForwardAvailable (A2) and
// SupportsPerSequenceFusedForward (holders) are both false, so every step plans
// PerSequence - the same plan the server logged for Muse-Glimmer.
//
// Scaled production knobs: block 256 -> 16, MaxPrefillChunkSize 256 -> 16,
// DecodeQuantumTokens 256 -> 16, window 4352 -> 80.
//
// Before the fix tests 1 and 2 failed with "AdvanceTokens(16) wants 1 blocks but only 0 are
// allocated." and test 3 with "... has no LastLogits to sample from at position 0.".
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;

namespace InferenceWeb.Tests.PrefixCache;

public class PerSequenceWindowSwapReproTests
{
    private const int BlockSize = 16;
    private const int Window = OracleFakes.MuseRingRows;   // 80

    private static SchedulerConfig Config(bool prefixCaching = true) => new()
    {
        BlockSize = BlockSize,
        NumBlocks = 256,              // the engine resizes to MaxContextLength (4096) / 16 = 256 anyway
        MaxNumBatchedTokens = 64,
        MaxPrefillChunkSize = 16,     // the chunk a prefill gets while another request decodes
        SoloPrefillChunkSize = 64,
        DecodeQuantumTokens = 16,     // owner rotation period on the PerSequence path
        MaxNumRunningSequences = 4,
        EnablePrefixCaching = prefixCaching,
        StopRepetition = false,       // oracle logits can look like a loop; do not end early
    };

    private static List<int> Tokens(int count, int start = 1) => Enumerable.Range(start, count).ToList();

    private static SequenceState Request(string id, List<int> prompt, int maxNew, string? scope = null)
        => new(id, prompt, maxNew, BlockSize, SamplingConfig.Greedy, cacheScope: scope);

    private static async Task<List<int>> ColdOutput(List<int> prompt, int maxNew)
    {
        using var model = OracleFakes.S2(BlockSize);
        using var engine = new InferenceEngine(model, Config(prefixCaching: false), NullLogger.Instance);
        var seq = Request("cold", prompt, maxNew);
        var done = await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(20));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, done.Status);
        return seq.OutputTokens.ToList();
    }

    /// <summary>Submit every request while the step loop is parked, so they are all admitted
    /// by ONE Schedule() in submission order (the RadixPagedEngineTests pattern).</summary>
    private static InferenceRequestHandle[] SubmitTogether(InferenceEngine engine, params SequenceState[] requests)
    {
        var gate = new ComputeGate();
        gate.Close();
        engine.ComputeGate = gate;
        var handles = requests.Select(r => engine.SubmitRequest(r)).ToArray();
        gate.Open();
        return handles;
    }

    // 1. The production shape. Parent round N+1 adopts EXACTLY PageWindowTokens of radix pages
    //    in the same Schedule() that admits a child agent (child first, as in the log). The child
    //    prefills (T0), the parent runs one 16-token chunk (T1: 80 -> 96 > window), the child
    //    decodes its 16-token quantum, and the parent's swap-in hits the discard branch.
    [Fact]
    public async Task ParentAdoptingPagesAtTheWindow_BesideADecodingChild_CompletesAndMatchesCold()
    {
        using var model = OracleFakes.S2(BlockSize);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        Assert.True(engine.PrefixCacheActive);
        Assert.Equal(Window, model.MaxReusablePrefixTokens);

        // Round N, solo: its pages [0, 80) are published (CapturableBlocks and CapturePages both
        // clamp at the window).
        var roundN = Request("parent-n", Tokens(120), 4, scope: "root");
        Assert.Equal(SequenceStatus.FinishedLengthCapped,
            (await engine.SubmitRequest(roundN).Completion.WaitAsync(TimeSpan.FromSeconds(20))).Status);

        // Round N+1 re-renders the history: 96 tokens in common with round N, then a new tail, so it
        // diverges from round N's live head (124) by more than the 16-token rewind cap and the
        // primary is not available anyway (the child is admitted first). Only the 80 pages serve it.
        var parentPrompt = Tokens(96).Concat(Tokens(40, 150)).ToList();   // 136 tokens
        var childPrompt = Tokens(20, 200);
        var child = Request("child", childPrompt, 64, scope: "child");
        var parent = Request("parent-n1", parentPrompt, 8, scope: "root");

        var handles = SubmitTogether(engine, child, parent);
        // Today: throws InvalidOperationException "AdvanceTokens(16) wants 1 blocks but only 0 are allocated."
        var parentDone = await handles[1].Completion.WaitAsync(TimeSpan.FromSeconds(20));
        var childDone = await handles[0].Completion.WaitAsync(TimeSpan.FromSeconds(20));

        Assert.Equal(SequenceStatus.FinishedLengthCapped, parentDone.Status);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, childDone.Status);
        Assert.Equal(await ColdOutput(parentPrompt, 8), parent.OutputTokens);
        Assert.Equal(await ColdOutput(childPrompt, 64), child.OutputTokens);
        // Admission reused exactly the window; whether the fix keeps that number depends on the
        // fix (a recompute-based fix may revoke it). Log it rather than pin it:
        //   Assert.Equal(Window, parentDone.PrefixCacheReusedTokens);
    }

    // 2. Radix-independent: prefix caching OFF, no adoption at all. A prompt longer than the window
    //    prefilled in 16-token chunks beside a decoder is swapped out/in every quantum; the swap-in
    //    after it passes 80 computed tokens hits the same branch.
    [Fact]
    public async Task PromptLongerThanTheWindow_InterleavedWithADecoder_CompletesAndMatchesCold()
    {
        using var model = OracleFakes.S2(BlockSize);
        using var engine = new InferenceEngine(model, Config(prefixCaching: false), NullLogger.Instance);

        var decoderPrompt = Tokens(20, 200);
        var longPrompt = Tokens(136);
        var decoder = Request("decoder", decoderPrompt, 120);
        var longRequest = Request("long", longPrompt, 8);

        var handles = SubmitTogether(engine, decoder, longRequest);
        // Today: "long" fails with "AdvanceTokens(16) wants 1 blocks but only 0 are allocated."
        var done = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(30));

        Assert.All(done, d => Assert.Equal(SequenceStatus.FinishedLengthCapped, d.Status));
        Assert.Equal(await ColdOutput(decoderPrompt, 120), decoder.OutputTokens);
        Assert.Equal(await ColdOutput(longPrompt, 8), longRequest.OutputTokens);
    }

    // 3. Decoders: two short prompts whose generations pass the window while they rotate. The
    //    discard nulls LastLogits, so the swap-in decode throws in SampleFromLogits.
    [Fact]
    public async Task TwoDecodersGrowingPastTheWindow_CompleteAndMatchCold()
    {
        using var model = OracleFakes.S2(BlockSize);
        using var engine = new InferenceEngine(model, Config(prefixCaching: false), NullLogger.Instance);

        var promptA = Tokens(20, 1);
        var promptB = Tokens(20, 101);
        var a = Request("a", promptA, 120);
        var b = Request("b", promptB, 120);

        var handles = SubmitTogether(engine, a, b);
        // Today: fails with "Sequence a has no LastLogits to sample from at position 0." (or b)
        var done = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(30));

        Assert.All(done, d => Assert.Equal(SequenceStatus.FinishedLengthCapped, d.Status));
        Assert.Equal(await ColdOutput(promptA, 120), a.OutputTokens);
        Assert.Equal(await ColdOutput(promptB, 120), b.OutputTokens);
    }
}
