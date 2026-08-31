// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the transcript EXPANSION in <c>ChatHistoryPreparer.AugmentWithCachedRawTokens</c>.
///
/// <para>
/// The incident this encodes: a turn that ran the skills/code tool loop leaves
/// <c>assistant, tool, assistant, ...</c> in the session's tracked history (each
/// assistant round carrying its raw output tokens), but the client sends that turn back
/// as ONE clean assistant message. The old positional walk broke at the first tool
/// result, the next render diverged thousands of tokens before the live cache's end,
/// and the engine — which only rewinds a few trailing tokens — re-prefilled the whole
/// conversation: 15.7k tokens, 10.7s to first token, 0% KV reuse. Expansion substitutes
/// the tracked transcript for the clean message so the rendered prefix stays
/// byte-identical to the cache.
/// </para>
/// </summary>
public class ToolTranscriptSpliceTests
{
    private static readonly List<int> Raw1 = new() { 11, 12, 13 };
    private static readonly List<int> Raw2 = new() { 21, 22 };
    private static readonly List<int> Raw3 = new() { 31, 32, 33, 34 };

    /// <summary>Tracked history as a skills turn leaves it: two tool rounds, then the answer.</summary>
    private static List<ChatMessage> TrackedTranscript() => new()
    {
        new() { Role = "user", Content = "convert this file" },
        new() { Role = "assistant", Content = "Using the pdf skill.\n\n", RawOutputTokens = Raw1 },
        new() { Role = "tool", Content = "Ran python (exit code 1)\nModuleNotFoundError" },
        new() { Role = "assistant", Content = "Retrying with packages.\n\n", RawOutputTokens = Raw2 },
        new() { Role = "tool", Content = "Ran python (exit code 0)\nFiles produced" },
        // The final round is tracked with the RAW model text, not the parsed content —
        // that is what UpdateTrackedHistory records.
        new() { Role = "assistant", Content = "<thought>done</thought>Here is your PDF.", RawOutputTokens = Raw3 },
    };

    /// <summary>What the Web UI sends back: the rounds' parsed contents, concatenated.</summary>
    private const string CleanAssistantText = "Using the pdf skill.\n\nRetrying with packages.\n\nHere is your PDF.";

    [Fact]
    public void ACleanAssistantTurn_ExpandsIntoTheTrackedToolTranscript()
    {
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = CleanAssistantText },
            new() { Role = "user", Content = "thanks, and now translate it" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, TrackedTranscript());

        // 1 user + 5 transcript messages + 1 new user.
        Assert.Equal(7, result.Count);
        Assert.Equal("user", result[0].Role);
        Assert.Same(Raw1, result[1].RawOutputTokens);
        Assert.Equal("tool", result[2].Role);
        Assert.Same(Raw2, result[3].RawOutputTokens);
        Assert.Equal("tool", result[4].Role);
        Assert.Same(Raw3, result[5].RawOutputTokens);
        Assert.Equal("thanks, and now translate it", result[6].Content);
    }

    [Fact]
    public void TheTurnAfterAnExpandedTurn_SplicesBothTurns()
    {
        // Tracked after turn 2 (a plain turn following the tool turn): the expanded
        // transcript plus turn 2's own record — this is what UpdateTrackedHistory
        // rebuilds from the augmented render history.
        var raw4 = new List<int> { 41, 42 };
        var tracked = TrackedTranscript();
        tracked.Add(new ChatMessage { Role = "user", Content = "thanks, and now translate it" });
        tracked.Add(new ChatMessage { Role = "assistant", Content = "RAW turn-2 text", RawOutputTokens = raw4 });

        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = CleanAssistantText },
            new() { Role = "user", Content = "thanks, and now translate it" },
            new() { Role = "assistant", Content = "parsed turn-2 text" },
            new() { Role = "user", Content = "one more thing" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, tracked);

        Assert.Equal(9, result.Count);
        Assert.Same(Raw3, result[5].RawOutputTokens);       // the transcript's final round
        Assert.Equal("user", result[6].Role);
        Assert.Same(raw4, result[7].RawOutputTokens);       // turn 2, plain positional splice
        Assert.Equal("one more thing", result[8].Content);
    }

    [Fact]
    public void AnEditedAssistantTurn_DoesNotExpand()
    {
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            // The user (or another client) rewrote the assistant text: the intermediate
            // rounds' contents no longer lead it, so expanding would splice a transcript
            // this conversation does not contain.
            new() { Role = "assistant", Content = "Something entirely different." },
            new() { Role = "user", Content = "next" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, TrackedTranscript());

        Assert.Equal(3, result.Count);
        // Falls back to the plain positional splice, which stays content-tolerant.
        Assert.Same(Raw1, result[1].RawOutputTokens);
        Assert.Equal("Something entirely different.", result[1].Content);
    }

    [Fact]
    public void AnEditedUserMessage_StopsAllSplicingAtTheEdit()
    {
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "EDITED PROMPT" },
            new() { Role = "assistant", Content = CleanAssistantText },
            new() { Role = "user", Content = "next" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, TrackedTranscript());

        Assert.Equal(3, result.Count);
        Assert.Null(result[1].RawOutputTokens);
    }

    [Fact]
    public void AClientThatSendsItsOwnToolMessages_IsMatchedPositionally_NotExpanded()
    {
        // An OpenAI-style caller that carries the tool transcript itself: every message
        // lines up one-to-one, so the plain walk handles it and nothing is inserted.
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = "Using the pdf skill.\n\n" },
            new() { Role = "tool", Content = "Ran python (exit code 1)\nModuleNotFoundError" },
            new() { Role = "assistant", Content = "Retrying with packages.\n\n" },
            new() { Role = "tool", Content = "Ran python (exit code 0)\nFiles produced" },
            new() { Role = "assistant", Content = "Here is your PDF." },
            new() { Role = "user", Content = "next" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, TrackedTranscript());

        Assert.Equal(7, result.Count);
        Assert.Same(Raw1, result[1].RawOutputTokens);
        Assert.Same(Raw2, result[3].RawOutputTokens);
        Assert.Same(Raw3, result[5].RawOutputTokens);
    }

    [Fact]
    public void ToolResultsFedBackAsUserTurns_NeverExpand()
    {
        // Mistral 3 renders tool results as user messages; that shape must keep
        // today's behavior — a run of "user" messages is a conversation, not a
        // transcript.
        var tracked = new List<ChatMessage>
        {
            new() { Role = "user", Content = "q" },
            new() { Role = "assistant", Content = "calling", RawOutputTokens = Raw1 },
            new() { Role = "user", Content = "Result of your shell call: ok" },
            new() { Role = "assistant", Content = "answer", RawOutputTokens = Raw2 },
        };
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "q" },
            new() { Role = "assistant", Content = "callinganswer" },
            new() { Role = "user", Content = "next" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, tracked);

        Assert.Equal(3, result.Count);
        Assert.Same(Raw1, result[1].RawOutputTokens);   // plain splice of the aligned assistant
        Assert.Equal("user", result[2].Role);
        Assert.Null(result[2].RawOutputTokens);
    }

    [Fact]
    public void ARoundWithSeveralToolResults_ExpandsAsOneRun()
    {
        var tracked = new List<ChatMessage>
        {
            new() { Role = "user", Content = "q" },
            new() { Role = "assistant", Content = "reading\n", RawOutputTokens = Raw1 },
            new() { Role = "tool", Content = "file one" },
            new() { Role = "tool", Content = "file two" },
            new() { Role = "assistant", Content = "RAW answer", RawOutputTokens = Raw2 },
        };
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "q" },
            new() { Role = "assistant", Content = "reading\nthe answer" },
            new() { Role = "user", Content = "next" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, tracked);

        Assert.Equal(6, result.Count);
        Assert.Same(Raw1, result[1].RawOutputTokens);
        Assert.Equal("tool", result[2].Role);
        Assert.Equal("tool", result[3].Role);
        Assert.Same(Raw2, result[4].RawOutputTokens);
    }
}
