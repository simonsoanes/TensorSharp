// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.StreamingWriters;

namespace InferenceWeb.Tests;

/// <summary>
/// The Web UI frames now travel two ways: the Server writes them with
/// <see cref="SseWriter"/> (<c>data: {json}\n\n</c> over Kestrel), an in-process host
/// serialises the same objects itself. The page discriminates frames purely by which
/// key is present and parses them with <c>JSON.parse</c>, so the two paths must
/// produce the same bytes — same property names, same order, nulls emitted.
/// </summary>
public class WebUiSseFrameParityTests
{
    private static object[] SampleFrames() => new[]
    {
        WebUiSseEvents.Token("hel"),
        WebUiSseEvents.Thinking("hmm \"quoted\" <tag> & 中文"),
        WebUiSseEvents.Replace("draft answer", 3, 48, true),
        WebUiSseEvents.ToolCalls(new List<ToolCall>
        {
            new() { Name = "shell", Arguments = new Dictionary<string, object> { ["command"] = "ls -la" } },
        }),
        WebUiSseEvents.SkillStep(new SkillToolInvocation(1, "skills_read", "pdf", "references/api.md", true, 512)),
        WebUiSseEvents.SkillStep(new SkillToolInvocation(2, "shell", null, null, true, 64)
        {
            Files = new[] { new SkillProducedFile("out.pptx", 2048, "/api/code/artifacts/abc/out.pptx") },
        }),
        WebUiSseEvents.ToolProgress("running", "shell", "line\n", 1.5, "python · 2.1 KB"),
        WebUiSseEvents.Done(12, 1.5, 8.0, false, null, "abc123", 100, 40, false),
        WebUiSseEvents.Done(0, 0.25, 0, true, "boom", "abc123", 0, 0, true),
    };

    // What an in-process host (the iOS loopback server) writes for a frame: the same
    // default serializer options SseWriter uses — no naming policy, nulls kept.
    private static string ServicePath(object frame) => "data: " + JsonSerializer.Serialize(frame) + "\n\n";

    private static async Task<string> ServerPath(params object[] frames)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        foreach (object frame in frames)
            await SseWriter.WriteEventAsync(ctx.Response, frame, CancellationToken.None);
        return Encoding.UTF8.GetString(body.ToArray());
    }

    [Fact]
    public async Task EveryFrame_IsByteIdenticalThroughSseWriterAndPlainSerialisation()
    {
        foreach (object frame in SampleFrames())
            Assert.Equal(ServicePath(frame), await ServerPath(frame));
    }

    [Fact]
    public async Task TheWholeSequence_IsTheConcatenationOfItsFrames()
    {
        object[] frames = SampleFrames();
        string expected = string.Concat(frames.Select(ServicePath));

        Assert.Equal(expected, await ServerPath(frames));
    }

    [Fact]
    public void TheDiscriminatorKey_ComesFirstInEveryFrame()
    {
        // index.html switches on key presence (token, thinking, replace,
        // tool_calls, skill_step, tool_progress, done); pinning the first property name
        // pins the builders' shapes against a careless reorder.
        string[] firstKeys = SampleFrames()
            .Select(f => JsonDocument.Parse(JsonSerializer.Serialize(f)).RootElement.EnumerateObject().First().Name)
            .ToArray();

        Assert.Equal(
            new[] { "token", "thinking", "replace", "tool_calls", "skill_step", "skill_step", "tool_progress", "done", "done" },
            firstKeys);
    }

    [Fact]
    public void Done_EmitsNullErrorAndComputedReusePercent()
    {
        Assert.Equal(
            """{"done":true,"tokenCount":12,"elapsed":1.5,"tokPerSec":8,"aborted":false,"truncated":false,"error":null,"sessionId":"abc123","promptTokens":100,"kvReusedTokens":40,"kvReusePercent":40}""",
            JsonSerializer.Serialize(WebUiSseEvents.Done(12, 1.5, 8.0, false, null, "abc123", 100, 40, false)));
    }
}
