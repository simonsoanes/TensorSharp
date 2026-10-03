// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptFact]
    public void ACompletedTurnShowsEngineTotalsIncludingReasoningAndToolsOutsideTheAnswer()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __sse: [{ __chunk: [
              { thinking: 'I should inspect the file first.' },
              { tool_progress: 'writing', tool: 'shell', text: 'cat notes.txt' },
              { tool_progress: 'running', tool: 'shell', detail: 'cat notes.txt' },
              { tool_progress: 'finished', tool: 'shell', seconds: 0.4 },
              { token: 'The answer is ' }, { token: '42.' },
              { done: true, tokenCount: 187, elapsed: 2.143, tokPerSec: 87.23,
                promptTokens: 512, kvReusedTokens: 420, kvReusePercent: 82 }
            ] }] };
            """, """
            var copied = null;
            navigator.clipboard = { writeText: function (text) { copied = text; return Promise.resolve(); } };
            __page.byId['text'].value = 'Read the file and answer.';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              var stats = __page.byId['chat'].querySelector('.turn-stats');
              var bubble = stats.parentNode.querySelector('.bubble');
              stats.parentNode.querySelector('.copy').dispatch('click');
              return {
                lines: __page.byId['chat'].querySelectorAll('.turn-stats').map(function (node) { return node.textContent; }),
                alongsideAnswer: stats.parentNode === bubble.parentNode,
                insideAnswer: bubble.querySelectorAll('.turn-stats').length,
                copied: copied,
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating()
              };
            });
            """);

        Assert.Equal("187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)",
            Assert.Single(result.GetProperty("lines").EnumerateArray()).GetString());
        Assert.True(result.GetProperty("alongsideAnswer").GetBoolean());
        Assert.Equal(0, result.GetProperty("insideAnswer").GetInt32());
        Assert.Equal("The answer is 42.", result.GetProperty("copied").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        JsonElement assistant = result.GetProperty("history")[1];
        Assert.Equal("The answer is 42.", assistant.GetProperty("content").GetString());
        Assert.Equal("I should inspect the file first.", assistant.GetProperty("thinking").GetString());
        Assert.Equal(187, assistant.GetProperty("stats").GetProperty("tokenCount").GetInt32());
    }

    [WebJavaScriptTheory]
    [InlineData("", "")]
    [InlineData(", promptTokens: 0, kvReusedTokens: 0, kvReusePercent: 0", "")]
    [InlineData(", promptTokens: 3, kvReusedTokens: 1", " · KV 1/3 (33%)")]
    [InlineData(", promptTokens: 512, kvReusedTokens: 420, kvReusePercent: 0", " · KV 420/512 (0%)")]
    [InlineData(", truncated: true", " · truncated (max tokens reached)")]
    [InlineData(", aborted: true", " · stopped by user")]
    public void TurnStatisticsHandleOptionalKvCountersAndCompletionFlags(string optionalFields, string suffix)
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __sse: [
              { token: 'Done.' },
              { done: true, tokenCount: 12, elapsed: 1.234, tokPerSec: 9.99{{optionalFields}} }
            ] };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              return { lines: __page.byId['chat'].querySelectorAll('.turn-stats').map(function (node) { return node.textContent; }) };
            });
            """);

        Assert.Equal("12 tokens · 1.2s · 10.0 tok/s" + suffix,
            Assert.Single(result.GetProperty("lines").EnumerateArray()).GetString());
    }

    [WebJavaScriptTheory]
    [InlineData("{\"done\":true}")]
    [InlineData("{\"done\":true,\"tokenCount\":12,\"tokPerSec\":10}")]
    [InlineData("{\"done\":true,\"tokenCount\":\"12\",\"elapsed\":1,\"tokPerSec\":10}")]
    [InlineData("{\"done\":true,\"tokenCount\":12,\"elapsed\":-1,\"tokPerSec\":10}")]
    [InlineData("{\"done\":true,\"tokenCount\":12,\"elapsed\":1,\"tokPerSec\":1e400}")]
    public void TerminalFramesWithoutUsableCountersFinishWithoutInventedStatistics(string terminalJson)
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __sse: [{ token: 'Done.' }, {{terminalJson}}] };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              return {
                lines: __page.byId['chat'].querySelectorAll('.turn-stats').map(function (node) { return node.textContent; }),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                errors: __page.errorNotices()
              };
            });
            """);

        Assert.Empty(result.GetProperty("lines").EnumerateArray());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
        JsonElement assistant = result.GetProperty("history")[1];
        Assert.Equal("Done.", assistant.GetProperty("content").GetString());
        Assert.False(assistant.TryGetProperty("stats", out _));
    }

    [WebJavaScriptTheory]
    [InlineData("{ imageUrl: '/uploads/image-result.png' }", "imageUrl", "/uploads/image-result.png")]
    [InlineData("{ videoUrl: '/uploads/video-result.mp4' }", "videoUrl", "/uploads/video-result.mp4")]
    public void MediaOnlyTurnsDoNotShowSyntheticZeroTokenStatistics(string mediaFrame, string mediaProperty, string mediaUrl)
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __sse: [
              {{mediaFrame}},
              { done: true, tokenCount: 0, elapsed: 3.5, tokPerSec: 0 }
            ] };
            """, """
            __page.byId['text'].value = 'Make something.';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              return {
                lines: __page.byId['chat'].querySelectorAll('.turn-stats').map(function (node) { return node.textContent; }),
                history: window.TensorAgent.history(), generating: window.TensorAgent.isGenerating()
              };
            });
            """);

        Assert.Empty(result.GetProperty("lines").EnumerateArray());
        Assert.False(result.GetProperty("generating").GetBoolean());
        JsonElement assistant = result.GetProperty("history")[1];
        Assert.Equal(mediaUrl, assistant.GetProperty(mediaProperty).GetString());
        Assert.False(assistant.TryGetProperty("stats", out _));
    }

    [WebJavaScriptFact]
    public void AReopenedConversationDisplaysItsStatisticsAndPreservesThemInTheNextRequest()
    {
        JsonElement result = Run("""
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Performance', updatedAt: '2026-09-30T10:00:00Z', messageCount: 4 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [
                { role: 'user', content: 'An older question' },
                { role: 'assistant', content: 'An older answer' },
                { role: 'user', content: 'A measured question' },
                { role: 'assistant', content: 'A measured answer',
                  stats: { tokenCount: 187, elapsed: 2.143, tokPerSec: 87.23,
                    promptTokens: 512, kvReusedTokens: 420, kvReusePercent: 82 } }
              ]
            };
            R['/api/chat'] = { __sse: [{ token: 'Another answer.' }, { done: true }] };
            """, """
            var shown = __page.byId['chat'].querySelectorAll('.turn-stats').map(function (node) { return node.textContent; });
            __page.byId['text'].value = 'Continue.';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              return { shown: shown, sent: __page.requests('/api/chat').map(function (request) { return request.body; }) };
            });
            """);

        Assert.Equal("187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)",
            Assert.Single(result.GetProperty("shown").EnumerateArray()).GetString());
        JsonElement messages = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("messages");
        Assert.Equal(5, messages.GetArrayLength());
        Assert.False(messages[1].TryGetProperty("stats", out _));
        JsonElement stats = messages[3].GetProperty("stats");
        Assert.Equal(187, stats.GetProperty("tokenCount").GetInt32());
        Assert.Equal(2.143, stats.GetProperty("elapsed").GetDouble());
        Assert.Equal(87.23, stats.GetProperty("tokPerSec").GetDouble());
        Assert.Equal(512, stats.GetProperty("promptTokens").GetInt32());
        Assert.Equal(420, stats.GetProperty("kvReusedTokens").GetInt32());
        Assert.Equal(82, stats.GetProperty("kvReusePercent").GetInt32());
        Assert.Equal("A measured answer", messages[3].GetProperty("content").GetString());
    }

    [WebJavaScriptFact]
    public void AResumedTurnShowsOneStatisticsLineAndOneCompleteHistoryEntry()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'reject' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ __chunk: [
              { token: 'Hel' }, { token: 'lo.' },
              { done: true, tokenCount: 187, elapsed: 2.143, tokPerSec: 87.23,
                promptTokens: 512, kvReusedTokens: 420, kvReusePercent: 82 }
            ] }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              window.TensorAgent.resumeTurn();
              return settle(20);
            }).then(function () {
              return {
                lines: __page.byId['chat'].querySelectorAll('.turn-stats').map(function (node) { return node.textContent; }),
                transcript: __page.transcript(), history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
                errors: __page.errorNotices()
              };
            });
            """);

        Assert.Equal("187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)",
            Assert.Single(result.GetProperty("lines").EnumerateArray()).GetString());
        Assert.Single(Bubbles(result.GetProperty("transcript")), turn => turn.GetProperty("role").GetString() == "assistant");
        Assert.Equal(2, result.GetProperty("history").GetArrayLength());
        Assert.Equal("Hello.", result.GetProperty("history")[1].GetProperty("content").GetString());
        Assert.Equal(187, result.GetProperty("history")[1].GetProperty("stats").GetProperty("tokenCount").GetInt32());
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
    }
}
