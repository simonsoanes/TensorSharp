// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;

namespace InferenceWeb.Tests;

/// <summary>
/// Multi-turn KV reuse for DeepSeek V4.1 in THINKING mode - the shape a user reported as
/// "the KV cache is reset on the second turn".
///
/// <para>Two things are true at once, and these tests pin both. The divergence is
/// REFERENCE-FAITHFUL: V4.1 drops a past assistant turn's reasoning in ordinary chat, so
/// the re-rendered history stops matching the cache exactly one token after the previous
/// turn's <c>&lt;|Assistant|&gt;</c> (<c>&lt;/think&gt;</c> where the cache holds
/// <c>&lt;think&gt;</c>), and nothing downstream can recover the answer region. But
/// everything BEFORE that point is reusable, and it is not a constant: from the third
/// turn on it is the WHOLE of the previous turn's prompt, because that prompt is what
/// rendered the earlier turns with their reasoning already dropped. Resetting there
/// re-prefills the entire conversation every turn.</para>
///
/// <para>No model is loaded. The renderer and <see cref="KVCache"/> decide the plan, which
/// is where the bug lived in 2026-09; the native side's ability to carry it out is covered by
/// TensorSharp.GGML.Native/tests/dsv41_truncate_test.cpp and by the live fixture run in
/// eng/tests/dsv41-inference.py.</para>
///
/// <para>Since 2026-09-17 the CLI no longer plans with <see cref="KVCache"/>: it and the server
/// go through the inference engine's radix prefix cache, which refused this very rewind until
/// 2026-09-29 while these tests kept passing. <see cref="DeepSeek41ThinkingTurnReuseTests"/> pins
/// that path; these still pin the render divergence the reuse depends on.</para>
/// </summary>
public class DeepSeek41MultiTurnKvReuseTests
{
    /// <summary>One token per character, so a token index is a character index and a
    /// divergence can be reported as readable text. Same shape as the one in
    /// <see cref="DeepSeek41ChatTests"/>.</summary>
    private sealed class CharacterTokenizer : ITokenizer
    {
        public string[] Vocab => [];
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) => new(ids.Select(i => (char)i).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer)
            => buffer.AddRange(Encoding.UTF8.GetBytes(((char)tokenId).ToString()));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }

    private const string Eos = "<｜end▁of▁sentence｜>";
    private const string Assistant = "<｜Assistant｜>";

    /// <summary>What the CLI did per turn before it moved onto the engine (2026-09-17): render
    /// the history, forward the suffix the plan keeps, then append the generated tokens to the
    /// mirror as <c>InteractiveSession</c> did (raw ids, no EOS - it broke on EOS before
    /// forwarding it).</summary>
    private sealed class Conversation
    {
        private readonly CharacterTokenizer _tokenizer = new();
        private readonly KVCachePromptRenderer _renderer = new(new GgufPromptRenderer());
        private readonly List<ChatMessage> _history = [];

        public KVCache Cache { get; } = new();
        public List<ReusePlanKind> Plans { get; } = [];
        public List<int> PromptTokens { get; } = [];
        public List<int> ReusedTokens { get; } = [];

        public List<int> Render(string userText)
        {
            _history.Add(new ChatMessage { Role = "user", Content = userText });
            return _renderer.RenderToTokens(_tokenizer, null, _history, "deepseek41",
                addGenerationPrompt: true, enableThinking: true);
        }

        /// <summary>Plan the turn, apply it to the mirror, and record the model's reply.</summary>
        public void Turn(string userText, string reasoning, string answer,
            bool supportsTruncation, int granularity = 1)
        {
            List<int> prompt = Render(userText);
            ReusePlan plan = Cache.PlanReuse(prompt, supportsTruncation, granularity);
            Plans.Add(plan.Kind);
            PromptTokens.Add(prompt.Count);

            if (plan.Kind == ReusePlanKind.PartialReuse)
            {
                ReusedTokens.Add(plan.ReusedPrefixLength);
                Cache.TruncateTo(plan.ReusedPrefixLength);
                Cache.RecordAppend(prompt.Skip(plan.ReusedPrefixLength).ToList(), [1f]);
            }
            else
            {
                ReusedTokens.Add(0);
                Cache.Reset();
                Cache.RecordAppend(prompt, [1f]);
            }

            // Generation: the reasoning block, then the answer. EOS is sampled but never
            // forwarded, so it is in the history text and not in the cache.
            List<int> generated = _tokenizer.Encode(reasoning + "</think>" + answer, addSpecial: false);
            Cache.RecordAppend(generated, [1f]);
            _history.Add(new ChatMessage
            {
                Role = "assistant",
                Content = answer,
                Thinking = reasoning,
                RawOutputTokens = generated,
            });
        }
    }

    /// <summary>
    /// The divergence is one token past the last <c>&lt;|Assistant|&gt;</c> of the
    /// PREVIOUS prompt, so the reusable prefix is that whole prompt minus its trailing
    /// <c>&lt;think&gt;</c>. Stated in the renderer's own output rather than assumed.
    /// </summary>
    [Fact]
    public void TheReusablePrefixIsThePreviousPromptMinusItsThinkToken()
    {
        var tokenizer = new CharacterTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<ChatMessage> history = [new() { Role = "user", Content = "Introduce Final Fantasy 7" }];

        List<int> turn1 = renderer.RenderToTokens(tokenizer, null, history, "deepseek41", true,
            enableThinking: true);
        string turn1Text = tokenizer.Decode(turn1);
        Assert.EndsWith(Assistant + "<think>", turn1Text);

        List<int> generated = tokenizer.Encode("weighing it</think>It is a 1997 RPG.", addSpecial: false);
        var cache = new KVCache();
        cache.RecordAppend(turn1, [1f]);
        cache.RecordAppend(generated, [1f]);

        history.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "It is a 1997 RPG.",
            Thinking = "weighing it",
            RawOutputTokens = generated,
        });
        history.Add(new ChatMessage { Role = "user", Content = "continue" });
        List<int> turn2 = renderer.RenderToTokens(tokenizer, null, history, "deepseek41", true,
            enableThinking: true);

        int common = cache.CommonPrefixLength(turn2);

        // Essentially the whole turn-1 prompt: it ends at the think marker that follows
        // the last <|Assistant|>. With the real V4.1 vocabulary <think> and </think> are
        // single tokens, so the divergence is exactly at turn1.Count - 1; this tokenizer
        // is one token per CHARACTER, so the '<' the two markers share is common as well
        // and the index sits one higher.
        Assert.Equal(turn1.Count - "<think>".Length + 1, common);
        Assert.EndsWith(Assistant + "<", tokenizer.Decode(turn1.Take(common).ToList()));
        // And the reference policy is why: the cache continues with <think>, the render
        // with </think>, and turn 1's reasoning is gone from the prompt.
        Assert.StartsWith("think>", tokenizer.Decode(turn1.Skip(common).ToList()));
        Assert.StartsWith("/think>", tokenizer.Decode(turn2.Skip(common).Take(8).ToList()));
        Assert.DoesNotContain("weighing it", tokenizer.Decode(turn2));
    }

    /// <summary>
    /// The regression this fix exists for. With truncation the plan reuses the prefix; the
    /// reused length GROWS with the conversation, because from turn 3 on it is the whole
    /// previous prompt. Without truncation every turn re-prefills from zero.
    /// </summary>
    [Fact]
    public void TruncationTurnsPerTurnPrefillFromWholeConversationIntoNewestAnswer()
    {
        string[] answers = [new string('a', 400), new string('b', 400), new string('c', 400)];
        string[] reasons = [new string('r', 900), new string('s', 900), new string('t', 900)];
        string[] asks = ["Introduce Final Fantasy 7", "continue", "continue", "continue"];

        var without = new Conversation();
        var with = new Conversation();
        for (int turn = 0; turn < 4; turn++)
        {
            string reasoning = reasons[Math.Min(turn, reasons.Length - 1)];
            string answer = answers[Math.Min(turn, answers.Length - 1)];
            without.Turn(asks[turn], reasoning, answer, supportsTruncation: false);
            with.Turn(asks[turn], reasoning, answer, supportsTruncation: true, granularity: 2);
        }

        // Today's behaviour, and still correct for an executor that cannot rewind.
        Assert.Equal(Enumerable.Repeat(ReusePlanKind.Reset, 4), without.Plans);
        Assert.Equal(Enumerable.Repeat(0, 4), without.ReusedTokens);

        // With truncation: turn 1 is cold, every later turn reuses.
        Assert.Equal(ReusePlanKind.Reset, with.Plans[0]);
        Assert.Equal(
            [ReusePlanKind.PartialReuse, ReusePlanKind.PartialReuse, ReusePlanKind.PartialReuse],
            with.Plans.Skip(1));

        // The reused prefix is the previous prompt (to within the alignment and the
        // <think> token), so it grows, and what each turn forwards does not.
        for (int turn = 1; turn < 4; turn++)
        {
            int previousPrompt = with.PromptTokens[turn - 1];
            Assert.InRange(with.ReusedTokens[turn], previousPrompt - 8, previousPrompt);
            Assert.True(with.ReusedTokens[turn] > with.ReusedTokens[turn - 1],
                $"turn {turn + 1} reused {with.ReusedTokens[turn]}, no more than turn {turn}'s "
                + $"{with.ReusedTokens[turn - 1]}");
        }

        // The point of it: turn 4 forwards about one answer, not four turns of history.
        int forwardedLastTurn = with.PromptTokens[3] - with.ReusedTokens[3];
        Assert.True(forwardedLastTurn < with.PromptTokens[3] / 2,
            $"turn 4 forwarded {forwardedLastTurn} of {with.PromptTokens[3]} prompt tokens");
        // Both conversations rendered the same prompts; only the plan differs.
        Assert.Equal(without.PromptTokens, with.PromptTokens);
    }

    /// <summary>
    /// The alignment a compressed-cache model needs never costs more than granularity-1
    /// tokens, and never turns a reuse into a reset while a prefix remains.
    /// </summary>
    [Fact]
    public void AlignmentCostsAtMostOneTokenOfReuse()
    {
        var unaligned = new Conversation();
        var aligned = new Conversation();
        foreach (var (ask, index) in new[] { "Introduce Final Fantasy 7", "continue", "continue" }.Select((a, i) => (a, i)))
        {
            // An odd-length answer makes the next turn's divergence land on an odd index
            // for at least one of the turns.
            string answer = new string('a', 101 + index);
            unaligned.Turn(ask, "rr", answer, supportsTruncation: true, granularity: 1);
            aligned.Turn(ask, "rr", answer, supportsTruncation: true, granularity: 2);
        }

        Assert.Equal(unaligned.Plans, aligned.Plans);
        for (int turn = 0; turn < aligned.ReusedTokens.Count; turn++)
        {
            Assert.InRange(aligned.ReusedTokens[turn], unaligned.ReusedTokens[turn] - 1,
                unaligned.ReusedTokens[turn]);
            Assert.Equal(0, aligned.ReusedTokens[turn] % 2);
        }
    }

    /// <summary>
    /// Thinking OFF has no divergence at all - the render is a pure extension of the cache
    /// - so it reuses fully even on an executor that cannot rewind. That is the control
    /// for the live benchmark: a reset with --think off would mean something else is wrong.
    /// </summary>
    [Fact]
    public void WithoutThinkingTheRenderExtendsTheCacheExactly()
    {
        var tokenizer = new CharacterTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<ChatMessage> history = [new() { Role = "user", Content = "Introduce Final Fantasy 7" }];

        List<int> turn1 = renderer.RenderToTokens(tokenizer, null, history, "deepseek41", true);
        List<int> generated = tokenizer.Encode("It is a 1997 RPG.", addSpecial: false);
        var cache = new KVCache();
        cache.RecordAppend(turn1, [1f]);
        cache.RecordAppend(generated, [1f]);

        history.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "It is a 1997 RPG.",
            RawOutputTokens = generated,
        });
        history.Add(new ChatMessage { Role = "user", Content = "continue" });
        List<int> turn2 = renderer.RenderToTokens(tokenizer, null, history, "deepseek41", true);

        Assert.True(cache.IsPrefixOf(turn2));
        ReusePlan plan = cache.PlanReuse(turn2, supportsTruncation: false);
        Assert.Equal(ReusePlanKind.PartialReuse, plan.Kind);
        Assert.Equal(cache.Count, plan.ReusedPrefixLength);
    }

    /// <summary>
    /// A tool-enabled thinking turn keeps past reasoning (the reference's own rule), so it
    /// is also a pure extension and never needed truncation. Pinned here so a future
    /// change to the truncation path cannot quietly take it away.
    /// </summary>
    [Fact]
    public void WithToolsThinkingHistoryIsKeptAndTheRenderStillExtendsTheCache()
    {
        var tokenizer = new CharacterTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<ToolFunction> tools = ToolFunction.ParseList(
            "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"w\"}}]");
        List<ChatMessage> history = [new() { Role = "user", Content = "Weather?" }];

        List<int> turn1 = renderer.RenderToTokens(tokenizer, null, history, "deepseek41", true,
            tools: tools, enableThinking: true);
        List<int> generated = tokenizer.Encode("checking</think>It is sunny.", addSpecial: false);
        var cache = new KVCache();
        cache.RecordAppend(turn1, [1f]);
        cache.RecordAppend(generated, [1f]);

        history.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "It is sunny.",
            Thinking = "checking",
            RawOutputTokens = generated,
        });
        history.Add(new ChatMessage { Role = "user", Content = "And tomorrow?" });
        List<int> turn2 = renderer.RenderToTokens(tokenizer, null, history, "deepseek41", true,
            tools: tools, enableThinking: true);

        Assert.Contains("checking</think>", tokenizer.Decode(turn2));
        Assert.True(cache.IsPrefixOf(turn2));
    }

    /// <summary>
    /// The reference history policy is unchanged: reasoning is still dropped from ordinary
    /// chat. The fix reuses the prefix that survives that policy, it does not soften it.
    /// </summary>
    [Fact]
    public void ThePolicyThatCausesTheDivergenceIsStillTheReferenceOne()
    {
        List<ChatMessage> history = [
            new() { Role = "user", Content = "Hi" },
            new() { Role = "assistant", Content = "Hello", Thinking = "A greeting." },
            new() { Role = "user", Content = "And now?" }];

        Assert.Equal(
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, "
            + "the more thorough the reasoning)\n\n<｜User｜>Hi" + Assistant + "</think>Hello" + Eos
            + "<｜User｜>And now?" + Assistant + "<think>",
            ChatTemplate.RenderDeepSeek41(history, enableThinking: true));
    }
}
