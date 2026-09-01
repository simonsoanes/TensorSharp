// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Linq;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the trailing-token rewind in the executor's live-cache continuation.
///
/// <para>
/// The rule this guards: a turn can end on a control token that the chat template
/// never re-renders as history. Gemma 4 answers a tool call by emitting
/// <c>&lt;|tool_response&gt;</c>; the engine forwards it into the KV cache, and the
/// next render frames that boundary as <c>&lt;turn|&gt;\n&lt;|turn&gt;tool</c>
/// instead. The live cache is then an exact prefix of the new prompt for every token
/// but its last, and requiring an EXACT match threw the whole conversation away — an
/// Agent Skills lookup re-prefilled from token 0, and (the symptom the executor's own
/// comment used to record) an EOS-terminated turn reported 0% reuse where a
/// max_tokens-terminated turn of the same conversation reported ~95%.
/// </para>
/// <para>
/// These tests exercise the decision arithmetic directly rather than through a model:
/// the executor's <c>ComputeLiveContinuationLcp</c> needs a loaded GGUF and a live
/// KV cache, so what is pinned here is the contract every caller of that decision
/// depends on — how far a rewind may go, and that it stays bounded. The end-to-end
/// behaviour is verified against a real model; see docs/agent_skills.md for the
/// measured numbers.
/// </para>
/// </summary>
public class LiveCacheRewindTests
{
    /// <summary>
    /// Mirrors the executor's decision so the boundary conditions are testable without
    /// a model. Kept deliberately small and in step with
    /// <c>BatchExecutor.ComputeLiveContinuationLcp</c>: if that method's rules change,
    /// this must change with it or the tests below stop meaning anything.
    /// </summary>
    private static int Decide(
        IReadOnlyList<int> prompt,
        IReadOnlyList<int> liveCache,
        int pooledCap,
        bool canTruncate,
        int maxRewind = 16)
    {
        int liveLen = liveCache.Count;
        if (liveLen <= pooledCap)
            return 0;

        int lcp = 0;
        int limit = Math.Min(liveLen, prompt.Count);
        while (lcp < limit && prompt[lcp] == liveCache[lcp])
            lcp++;

        if (prompt.Count <= lcp)
            return 0;
        if (lcp == liveLen)
            return liveLen;

        int rewind = liveLen - lcp;
        if (rewind > maxRewind)
            return 0;
        if (!canTruncate)
            return 0;
        if (lcp <= pooledCap)
            return 0;
        return lcp;
    }

    private static List<int> Tokens(int count, int seed = 1)
        => Enumerable.Range(0, count).Select(i => seed * 1000 + i).ToList();

    // ---- the case the fix exists for ---------------------------------------

    [Fact]
    public void OneTrailingControlToken_IsRewoundInsteadOfDiscardingTheWholePrefix()
    {
        // Round 1 left 1984 tokens resident, the last being a control token the
        // template does not reproduce. Round 2's prompt reproduces the first 1983 and
        // then diverges. Before the fix this returned 0 and re-prefilled everything.
        List<int> live = Tokens(1984);
        List<int> prompt = Tokens(1983);
        prompt.Add(999_999);                 // the template's framing, not the model's
        prompt.AddRange(Tokens(2200, seed: 7));   // the freshly fetched skill body

        Assert.Equal(1983, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void AnExactPrefix_StillNeedsNoRewindAtAll()
    {
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(1000);
        prompt.AddRange(Tokens(50, seed: 7));

        Assert.Equal(1000, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    // ---- the bounds that keep it safe --------------------------------------

    [Fact]
    public void ARewindLongerThanTheLimit_IsDeclined()
    {
        // A long divergence is not a stale control token — it is an edited turn or a
        // changed system prompt, where a clean re-prefill is the correct answer. On a
        // sliding-window model the cache is also circular, so rewinding far is not
        // faithful even when it is cheap.
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(950);
        prompt.AddRange(Tokens(200, seed: 7));   // diverges 50 tokens from the end

        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void ARewindAtExactlyTheLimit_IsStillAllowed()
    {
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(984);
        prompt.AddRange(Tokens(200, seed: 7));   // 16-token rewind

        Assert.Equal(984, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void AModelThatCannotRewind_IsNeverAskedTo()
    {
        // Recurrent / SSM state has no reverse: Qwen 3.5's GatedDeltaNet cannot be
        // rolled back to an earlier position, so for those the only valid reuse is an
        // exact prefix. SupportsKVCacheTruncation is what says so.
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(999);
        prompt.Add(999_999);
        prompt.AddRange(Tokens(200, seed: 7));

        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: false));
        Assert.Equal(999, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void AMatchedPrefixInsideThePooledCap_IsLeftToThePooledPath()
    {
        // Below the cap the pooled snapshot already covers it, and adopting the live
        // cache would buy nothing while pinning the model to one sequence.
        List<int> live = Tokens(600);
        List<int> prompt = Tokens(500);
        prompt.AddRange(Tokens(200, seed: 7));

        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void APromptWithNoNewSuffix_IsDeclined()
    {
        // Nothing left to forward. Adopting would leave the sequence with no work and
        // the caller expecting a token.
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(1000);

        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void ADivergenceAtTheVeryStart_IsDeclined()
    {
        // A different system prompt (a changed skill selection, for instance) shares
        // nothing, and must re-prefill rather than rewind almost the whole cache.
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(1000, seed: 7);

        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }
}
