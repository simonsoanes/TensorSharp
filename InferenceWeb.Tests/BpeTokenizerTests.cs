// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.Models;

namespace InferenceWeb.Tests;

public class BpeTokenizerTests
{
    [Fact]
    public void Encode_DiscardsStaleCandidateAfterNeighborMerge()
    {
        // Initial candidates for "xabc":
        //   a+b  rank 0 (wins first)
        //   x+a  rank 1 (becomes stale when a changes to ab)
        // The first merge creates ab+c at rank 2 and x+ab at rank 3. The stale
        // rank-1 entry must not merge x+ab ahead of the valid rank-2 candidate.
        string[] vocab = { "x", "a", "b", "c", "ab", "xa", "abc", "xab" };
        string[] merges = { "a b", "x a", "ab c", "x ab" };
        var tokenizer = new BpeTokenizer(
            vocab,
            new int[vocab.Length],
            merges,
            bosTokenId: -1,
            eosTokenIds: Array.Empty<int>(),
            addBos: false,
            addEos: false);

        List<int> ids = tokenizer.Encode("xabc", addSpecial: false);

        Assert.Equal(new[] { 0, 6 }, ids);
        Assert.Equal("xabc", tokenizer.Decode(ids));
    }

    [Fact]
    public void Encode_RequeuesChangedPairAtItsCurrentRank()
    {
        // With no competing ab+c merge, the stale x+a candidate is discarded
        // and the newly queued x+ab candidate is still applied at rank 2.
        string[] vocab = { "x", "a", "b", "ab", "xa", "xab" };
        string[] merges = { "a b", "x a", "x ab" };
        var tokenizer = new BpeTokenizer(
            vocab,
            new int[vocab.Length],
            merges,
            bosTokenId: -1,
            eosTokenIds: Array.Empty<int>(),
            addBos: false,
            addEos: false);

        Assert.Equal(new[] { 5 }, tokenizer.Encode("xab", addSpecial: false));
    }

    [Fact]
    public void Encode_Qwen35KeepsCombiningMarksWithLetters()
    {
        // UTF-8 byte-level normalization maps U+0301 to the two vocab chars
        // U+00CC U+0123. Qwen3.5 pre-tokenizes "a + combining acute" as one
        // piece and the following punctuation as another piece.
        const string letterWithMark = "a\u00CC\u0123";
        const string markWithBang = "\u00CC\u0123!";
        string[] vocab = { "a", letterWithMark, "!", markWithBang };
        var qwen35 = new BpeTokenizer(
            vocab,
            new int[vocab.Length],
            Array.Empty<string>(),
            bosTokenId: -1,
            eosTokenIds: Array.Empty<int>(),
            addBos: false,
            addEos: false,
            preTokenizerType: "qwen35");

        Assert.Equal(new[] { 1, 2 }, qwen35.Encode("a\u0301!", addSpecial: false));
    }

    [Fact]
    public void Encode_Qwen35CombiningMarkRuleDoesNotChangeDefaultTokenizer()
    {
        // The legacy/default Qwen2-style expression ends the letter piece before
        // the mark, so the same mark remains in the punctuation run. Keeping this
        // assertion separate guards the architecture-specific dispatch.
        const string letterWithMark = "a\u00CC\u0123";
        const string markWithBang = "\u00CC\u0123!";
        string[] vocab = { "a", letterWithMark, "!", markWithBang };
        var defaultTokenizer = new BpeTokenizer(
            vocab,
            new int[vocab.Length],
            Array.Empty<string>(),
            bosTokenId: -1,
            eosTokenIds: Array.Empty<int>(),
            addBos: false,
            addEos: false);

        Assert.Equal(new[] { 0, 3 }, defaultTokenizer.Encode("a\u0301!", addSpecial: false));
    }

    [Fact]
    public void ResolveEogTokenIds_IncludesBothQwenControlEndings()
    {
        string[] vocab = { "ordinary", "<|endoftext|>", "<|im_end|>", "extra" };

        int[] eos = ModelBase.ResolveEogTokenIds(vocab, eosId: 2, extraEosIds: new[] { 3 });

        Assert.Equal(new[] { 1, 2, 3 }, eos);
        var tokenizer = new BpeTokenizer(
            vocab,
            new[] { 1, 3, 3, 3 },
            Array.Empty<string>(),
            bosTokenId: -1,
            eosTokenIds: eos,
            addBos: false,
            addEos: false);
        Assert.True(tokenizer.IsEos(1));
        Assert.True(tokenizer.IsEos(2));
        Assert.False(tokenizer.IsEos(0));
    }

    [Fact]
    public void ResolveEogTokenIds_AppliesLlamaHarmonyAndGemmaWorkarounds()
    {
        string[] harmony = { "<|return|>", "<|call|>", "<|end|>", "ordinary" };
        Assert.Equal(new[] { 0, 1 }, ModelBase.ResolveEogTokenIds(harmony, eosId: 2));

        string[] solar = { "<|calls|>", "<|flush|>", "<|end|>", "ordinary" };
        Assert.Equal(new[] { 0, 1 }, ModelBase.ResolveEogTokenIds(solar, eosId: 2));

        string[] gemma = { "<|tool_response>", "</s>", "<eos>", "ordinary" };
        Assert.Equal(new[] { 0, 2 }, ModelBase.ResolveEogTokenIds(gemma, eosId: 1));
    }
}
