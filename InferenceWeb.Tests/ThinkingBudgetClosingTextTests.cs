// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// The thinking budget's closing text (Qwen's hand-over sentence) is forced one token per step ahead of the
/// end token, from the history alone: a peek commits nothing, a rollback into the text resumes it, and a
/// history that left the text closes at once.
/// </summary>
public sealed class ThinkingBudgetClosingTextTests
{
    [Fact]
    public void ClosingText_IsForcedTokenByToken_ThenTheEndToken()
    {
        var sampler = Sampler(limit: 3, text: new[] { 7, 8 });
        var history = new List<int> { 4, 4 };
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));   // under the limit
        history.Add(4);
        Assert.Equal(7, Forced(sampler, history));
        Assert.Equal(7, Forced(sampler, history));                         // a peek commits nothing
        history.Add(7);
        Assert.Equal(8, Forced(sampler, history));
        history.Add(8);
        Assert.Equal(2, Forced(sampler, history));
        history.Add(2);
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));   // closed: the answer samples freely
        history.Add(9);
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
    }

    [Fact]
    public void ARollbackIntoTheText_ResumesIt_AndAHistoryThatLeftIt_ClosesAtOnce()
    {
        var sampler = Sampler(limit: 3, text: new[] { 7, 8 });
        var history = new List<int> { 4, 4, 4, 7, 8, 2, 9 };
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
        history.RemoveRange(4, 3);                                        // [4 4 4 7]
        Assert.Equal(8, Forced(sampler, history));
        history[3] = 5;                                                   // [4 4 4 5]: not the text
        Assert.Equal(2, Forced(sampler, history));
    }

    [Fact]
    public void WithoutText_TheCloseIsBare_AsBefore()
    {
        var sampler = Sampler(limit: 2, text: null);
        Assert.Equal(2, Forced(sampler, new List<int> { 4, 4 }));
    }

    [Fact]
    public void ARepetitionCloseHandsOverToo()
    {
        var sampler = Sampler(limit: 100, text: new[] { 7, 8 });
        var history = new List<int> { 4, 4, 4 };
        Assert.True(sampler.TryRequestThinkingClosure(history));
        Assert.Equal(7, Forced(sampler, history));
        history.Add(7);
        Assert.Equal(8, Forced(sampler, history));
        history.Add(8);
        Assert.Equal(2, Forced(sampler, history));
    }

    [Fact]
    public void TheTextMayNotContainTheEndToken()
        => Assert.Throws<ArgumentException>(() => new ThinkingTokenBudget(3, 2, closingTokenIds: new[] { 7, 2 }));

    private static TokenSampler Sampler(int limit, int[] text)
    {
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(limit, endTokenId: 2, closeOnRepetition: true, closingTokenIds: text);
        return new TokenSampler(config);
    }

    private static int Forced(TokenSampler sampler, List<int> history)
    {
        Assert.True(sampler.TryGetForcedThinkingToken(history, out int token));
        return token;
    }
}
