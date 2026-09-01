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
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Responses;

namespace InferenceWeb.Tests;

/// <summary>
/// The three stateless chat adapters must give all internally serviced tool rounds one
/// request-private workspace and release it only after the response path has returned.
/// These tests stop at the hosted-model guard, after planning but before inference, so
/// they exercise the real adapter lifetime without loading model weights.
/// </summary>
public sealed class ProtocolAdapterRequestWorkspaceTests : IDisposable
{
    private readonly string _base = Path.Combine(
        Path.GetTempPath(), "ts-adapter-workspace-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    [Theory]
    [InlineData("openai-chat")]
    [InlineData("openai-responses")]
    [InlineData("ollama-chat")]
    public async Task Request_AcquiresPassesAndReleasesAWorkspace(string protocol)
    {
        string workspaceParent = Path.Combine(_base, protocol, "workspaces");
        string uploadRoot = Path.Combine(_base, protocol, "uploads");
        string hostedModel = Path.Combine(_base, protocol, "hosted.gguf");
        var workspaces = new SessionWorkspaceManager(workspaceParent);
        var runner = new RecordingRunner(workspaceParent);
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", hostedModel, "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService();
        using var store = new InMemoryResponsesStore();

        DefaultHttpContext context = ContextFor(RequestBody(protocol));
        await InvokeAdapterAsync(
            protocol, context, service, options, uploadRoot, registry, runner, workspaces, store);

        Assert.Equal(StatusCodes.Status404NotFound, context.Response.StatusCode);
        Assert.Equal(1, runner.DeclareCalls);
        Assert.True(runner.Persists,
            "SkillRequestPlan must receive the lease workspace, which is what makes persists true.");
        Assert.Equal(1, runner.LiveWorkspaceCount);
        Assert.True(runner.WorkspaceWasUsableDuringPlanning);
        string leasedRoot = Assert.IsType<string>(runner.LiveWorkspaceRoot);

        // The adapter's method-scope lease must outlive planning, then disappear on every
        // return path. This request returns a model-not-hosted error immediately after the
        // plan, which also exercises cleanup on failure rather than only the happy path.
        Assert.False(Directory.Exists(leasedRoot));
        Assert.Empty(Directory.EnumerateDirectories(workspaceParent, SessionWorkspace.DirectoryPrefix + "*"));
    }

    private static DefaultHttpContext ContextFor(string body)
    {
        var context = new DefaultHttpContext();
        context.Request.ContentType = "application/json";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(body));
        context.Response.Body = new MemoryStream();
        return context;
    }

    private static string RequestBody(string protocol) => protocol switch
    {
        "openai-responses" =>
            """{"model":"not-hosted.gguf","input":"fix it","stream":false,"store":false}""",
        "openai-chat" or "ollama-chat" =>
            """{"model":"not-hosted.gguf","messages":[{"role":"user","content":"fix it"}],"stream":false}""",
        _ => throw new ArgumentOutOfRangeException(nameof(protocol)),
    };

    private static async Task InvokeAdapterAsync(
        string protocol,
        HttpContext context,
        ModelService service,
        ServerHostingOptions options,
        string uploadRoot,
        SkillRegistry registry,
        ICodeRunner runner,
        SessionWorkspaceManager workspaces,
        IResponsesStore store)
    {
        var queue = new InferenceQueue();
        var uploads = new UploadStoragePolicy(uploadRoot);

        switch (protocol)
        {
            case "openai-chat":
                await new OpenAIChatAdapter(
                    service, queue, options, uploads, registry, runner, workspaces,
                    NullLoggerFactory.Instance).ChatCompletionsAsync(context);
                break;
            case "openai-responses":
                await new OpenAIResponsesAdapter(
                    service, queue, options, uploads, registry, runner, workspaces,
                    NullLoggerFactory.Instance, store).CreateResponseAsync(context);
                break;
            case "ollama-chat":
                await new OllamaAdapter(
                    service, queue, options, uploads, registry, runner, workspaces,
                    NullLoggerFactory.Instance).ChatAsync(context);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(protocol));
        }
    }

    /// <summary>
    /// Only the model-family capability is relevant before HostedModelGuard rejects this
    /// request. No generation API is replaced or reached.
    /// </summary>
    private sealed class ToolCapableUnloadedModelService : ModelService
    {
        public override string Architecture => "qwen2";
    }

    private sealed class RecordingRunner : ICodeRunner
    {
        private readonly string _workspaceParent;

        public RecordingRunner(string workspaceParent) => _workspaceParent = workspaceParent;

        public int DeclareCalls { get; private set; }
        public bool Persists { get; private set; }
        public int LiveWorkspaceCount { get; private set; }
        public string? LiveWorkspaceRoot { get; private set; }
        public bool WorkspaceWasUsableDuringPlanning { get; private set; }

        public bool CanRun => true;
        public string? UnavailableReason => null;

        public ToolFunction Declare() =>
            new() { Name = SkillToolNames.Shell, Description = "runs commands" };

        public IReadOnlyList<ToolFunction> DeclareTools(bool persists)
        {
            DeclareCalls++;
            Persists = persists;
            string[] roots = Directory.Exists(_workspaceParent)
                ? Directory.GetDirectories(
                    _workspaceParent, SessionWorkspace.DirectoryPrefix + "*", SearchOption.TopDirectoryOnly)
                : Array.Empty<string>();
            LiveWorkspaceCount = roots.Length;
            LiveWorkspaceRoot = roots.Length == 1 ? roots[0] : null;
            WorkspaceWasUsableDuringPlanning = LiveWorkspaceRoot != null
                && Directory.Exists(Path.Combine(LiveWorkspaceRoot, "work"))
                && Directory.Exists(Path.Combine(LiveWorkspaceRoot, "env"))
                && Directory.Exists(Path.Combine(LiveWorkspaceRoot, "state"))
                && Directory.Exists(Path.Combine(LiveWorkspaceRoot, "tmp"));
            return new[] { Declare() };
        }

        public SkillToolResult Execute(
            ToolCall call,
            IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null,
            SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null) =>
            SkillToolResult.Failure("inference is not reached by this test");
    }
}
