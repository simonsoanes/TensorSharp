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
using System.Linq;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// The machinery around the tool surface that survives a change OF that surface:
/// partitioning a caller's tools from this host's, the per-language install ledger, the
/// PATH plumbing an environment needs, reading a failure well enough to act on it, and
/// what does not count as a produced file.
///
/// <para>
/// The declare/classify/dispatch sweep this file was named for now lives in
/// <c>ShellToolDeclarationTests</c>, which owns the current two-tool surface. What is
/// left here is everything that was never about WHICH tools are declared — and which
/// therefore has to keep working across exactly the kind of rewrite that just happened.
/// </para>
/// <para>
/// The incident the sweep exists for is worth keeping written down, because it is the
/// reason both files exist: <c>apply_patch</c> and <c>list_files</c> were once declared
/// to the model and implemented by the adapter, while the hand-maintained predicate in
/// <see cref="SkillTools.IsBuiltInTool"/> never learned their names. Every call to
/// either was classified as the CLIENT's, handed to a Web UI that declares no tools and
/// has no handler for a tool-call frame at all, and the turn ended. The model's whole
/// reply had been inside its thinking channel, so the user saw a chat that answered once
/// and then stopped — nothing logged, nothing rendered, every retry hitting the same
/// wall. A name being right everywhere it appeared is not enough; what was missing was a
/// name in a place it did not appear at all, which only a sweep over the DECLARED set
/// can see.
/// </para>
/// </summary>
public class BuiltInToolRegistryTests : IDisposable
{
    private readonly string _scratch;
    private readonly ShellRunner _runner;
    private readonly CodeRunnerAdapter _adapter;
    private readonly SessionWorkspace _workspace;

    public BuiltInToolRegistryTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), "ts-tool-registry-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_scratch);
        _workspace = new SessionWorkspaceManager(_scratch).GetOrCreate("registry");

        var options = new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _scratch,
            AllowInstall = true,
        };
        _runner = new ShellRunner(options);
        _adapter = new CodeRunnerAdapter(_runner, options);
    }

    public void Dispose()
    {
        _runner.Dispose();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    /// <summary>Everything a fully-enabled host puts in front of the model.</summary>
    private List<ToolFunction> Declared()
    {
        var all = new List<ToolFunction>(SkillTools.BuiltIn(allowScripts: true));
        all.AddRange(_adapter.DeclareTools());
        return all;
    }

    private SkillToolContext ContextWithCode() =>
        new(Array.Empty<Skill>()) { CodeRunner = _adapter, Workspace = _workspace };

    // ---- the sweeps below must have something to sweep -----------------------


    // ---- declared => answerable ---------------------------------------------



    // ---- answerable => declared ---------------------------------------------



    [Fact]
    public void TheSkillToolNames_AreClassifiedTooAndAreNotCodeTools()
    {
        Assert.True(SkillTools.IsBuiltInTool(SkillTools.ListToolName));
        Assert.True(SkillTools.IsBuiltInTool(SkillTools.ReadToolName));
        Assert.True(SkillTools.IsBuiltInTool(SkillTools.RunToolName));

        // The two families stay distinct: skills_run is a skill's own script, shell is a
        // command the model wrote, and the operator enables them separately.
        Assert.False(SkillToolNames.IsCodeTool(SkillTools.RunToolName));
        Assert.False(SkillTools.IsSkillTool(SkillToolNames.Shell));
    }

    [Fact]
    public void AnUndeclaredName_IsStillRefused()
    {
        // The guard must not have been satisfied by classifying everything as built-in.
        Assert.False(SkillTools.IsBuiltInTool("get_weather"));
        Assert.False(SkillTools.IsBuiltInTool(null));
        Assert.False(SkillTools.IsBuiltInTool("shell "));   // ordinal, on purpose
        Assert.False(SkillTools.IsBuiltInTool("SHELL"));
    }

    // ---- the third bucket ----------------------------------------------------

    [Fact]
    public void Partition_SplitsOursFromTheClientsFromNobodys()
    {
        // Only the CALLER's tools, never the merged roster the model was shown. Keying
        // on the merged list would put a host tool the classifier had not learned back
        // into the client bucket — which is the bug, exactly, one layer along.
        var theirs = new List<ToolFunction> { new() { Name = "get_weather", Description = "the caller's" } };

        SkillTools.Partition(
            new ToolCall?[]
            {
                new() { Name = SkillToolNames.Shell },
                new() { Name = "get_weather" },
                new() { Name = "browse_the_web" },
                null,
            },
            theirs,
            out List<ToolCall> builtIn, out List<ToolCall> client, out List<ToolCall> unknown);

        Assert.Equal(new[] { SkillToolNames.Shell }, builtIn.Select(c => c.Name));
        Assert.Equal(new[] { "get_weather" }, client.Select(c => c.Name));
        Assert.Equal(new[] { "browse_the_web" }, unknown.Select(c => c.Name));
    }

    [Fact]
    public void Partition_AHostToolTheClassifierDoesNotKnow_IsUnknown_NotTheClients()
    {
        // The regression that matters most. Before, "client" meant "in the roster the
        // model was shown", and this host's own declarations are in that roster — so a
        // declared-but-unclassified tool was forwarded and the turn died. It must land
        // in the third bucket, where the model is told and gets another round.
        SkillTools.Partition(
            new ToolCall?[] { new() { Name = "some_future_host_tool" } },
            Array.Empty<ToolFunction>(),
            out List<ToolCall> builtIn, out List<ToolCall> client, out List<ToolCall> unknown);

        Assert.Empty(builtIn);
        Assert.Empty(client);
        Assert.Equal(new[] { "some_future_host_tool" }, unknown.Select(c => c.Name));
    }

    [Fact]
    public void Partition_AClientToolShadowingABuiltIn_StaysTheClients()
    {
        // Merge keeps the caller's declaration and drops ours on a name collision, so
        // the model was shown THEIR tool. Answering it here would run something else
        // entirely. Matched case-insensitively, the way Merge resolves the collision.
        var theirs = new List<ToolFunction> { new() { Name = "Run_Code", Description = "the caller's" } };

        SkillTools.Partition(
            new ToolCall?[] { new() { Name = "Run_Code" } },
            theirs,
            out List<ToolCall> builtIn, out List<ToolCall> client, out _);

        Assert.Empty(builtIn);
        Assert.Equal(new[] { "Run_Code" }, client.Select(c => c.Name));
    }

    [Fact]
    public void Partition_WithNoDeclaredList_CallsNothingUnknown()
    {
        // Null means the caller does not know what was declared, and then no name can
        // honestly be called invented. The two-way split is the honest answer, and it is
        // what the CLI's --tools contract has always relied on.
        SkillTools.Partition(
            new ToolCall?[] { new() { Name = "get_weather" } },
            null,
            out List<ToolCall> builtIn, out List<ToolCall> client, out List<ToolCall> unknown);

        Assert.Empty(builtIn);
        Assert.Equal(new[] { "get_weather" }, client.Select(c => c.Name));
        Assert.Empty(unknown);
    }

    // ---- the install ledger --------------------------------------------------

    [Fact]
    public void AnNpmInstall_DoesNotSuppressThePipInstallOfTheSameName()
    {
        // One workspace holds a pip environment and an npm one, and plenty of names
        // exist in both registries. A flat ledger made `pptxgenjs, markitdown` on a
        // JavaScript run mark "markitdown" installed, so the later Python install was
        // skipped and the model was told "Already installed this session: markitdown"
        // while `import markitdown` kept failing. A host that states the opposite of the
        // truth is the one tool result a model cannot recover from.
        _workspace.MarkInstalled("javascript", new[] { "markitdown" });

        Assert.True(_workspace.IsInstalled("javascript", "markitdown"));
        Assert.False(_workspace.IsInstalled("python", "markitdown"));
    }

    // ---- the session environment is reachable, not just importable -------------




    // ---- a look must not become "the program" ---------------------------------




    // ---- a missing PROGRAM is not a missing package ---------------------------

    [Theory]
    [InlineData("sh: pdftoppm: command not found", "pdftoppm")]
    [InlineData("bash: line 1: pandoc: command not found", "pandoc")]
    [InlineData("zsh: command not found: ffmpeg", "ffmpeg")]
    [InlineData("sh: 1: soffice: not found", "soffice")]
    public void AMissingCommand_IsReadFromStderr_NotGuessedFromTheExitCode(string stderr, string expected)
    {
        // 127 is the shell's own convention and a program is free to return it for
        // something else; the message names what is actually missing, which is the part
        // the model needs. Undiagnosed, it retries the same line forever — it cannot
        // tell "this host has no pdftoppm" from "I typed it wrong", and neither pip nor
        // npm can supply that one.
        Assert.Equal(expected, CodeDiagnostics.MissingCommand(stderr));
    }

    [Fact]
    public void AnOrdinaryTraceback_IsNotMistakenForAMissingCommand()
    {
        Assert.Null(CodeDiagnostics.MissingCommand("Traceback (most recent call last): ZeroDivisionError"));
        Assert.Null(CodeDiagnostics.MissingCommand(string.Empty));
    }

    [Theory]
    [InlineData(".ses", true)]
    [InlineData("mat-debug-40275.log", true)]
    [InlineData("out/mat-debug-1.log", true)]
    [InlineData("__pycache__/x.pyc", true)]
    [InlineData("deck.pptx", false)]
    [InlineData("notes.log", false)]
    [InlineData("matrix-debug.log", false)]
    public void ALibreOfficeConversion_DoesNotOfferItsOwnScratchAsADownload(string path, bool junk)
    {
        // Every headless soffice call drops a .ses marker and a MAT debug log in the
        // working directory, so one deck came back as four download chips. A tool's
        // scratch is not the user's output — the same call the .pyc filter makes.
        Assert.Equal(junk, CodeArtifactStore.IsRuntimeJunk(path));
    }

    [Fact]
    public void DescribeUnknownTool_NamesTheToolsThatDoExist()
    {
        // A model that guessed a name is one nudge from the right one, and it reads tool
        // results far more reliably than declarations from thousands of tokens ago.
        string message = SkillTools.DescribeUnknownTool("write_slides", Declared());

        Assert.Contains("no tool called 'write_slides'", message, StringComparison.Ordinal);
        Assert.Contains(SkillToolNames.Shell, message, StringComparison.Ordinal);
        Assert.Contains(SkillToolNames.ApplyPatch, message, StringComparison.Ordinal);
        Assert.Contains(SkillTools.ReadToolName, message, StringComparison.Ordinal);
    }

    /// <summary>
    /// A model asking to LOOK at something is not asking which tools exist. Several
    /// skills tell it to check its own output visually and this host cannot show it an
    /// image — which cost one hallucinated call plus four turns totalling 27,810 tokens of
    /// reasoning spent arriving at that conclusion. Answer the question it asked.
    /// </summary>
    [Fact]
    public void DescribeUnknownTool_AnswersAnInventedImageToolWithTheStructuralAlternative()
    {
        foreach (string invented in new[] { "view", "open_image", "display", "screenshot" })
        {
            string message = SkillTools.DescribeUnknownTool(invented, Declared());

            Assert.Contains("cannot show you an image", message, StringComparison.Ordinal);
            Assert.Contains("STRUCTURALLY", message, StringComparison.Ordinal);
            // And it must say to tell the user, so a deck nobody looked at is not
            // presented as one that was checked.
            Assert.Contains("could not look at it", message, StringComparison.Ordinal);
        }
    }

    /// <summary>An ordinary wrong guess still gets the list, which is the useful answer there.</summary>
    [Fact]
    public void DescribeUnknownTool_DoesNotMistakeAnOrdinaryGuessForAnImageRequest()
    {
        string message = SkillTools.DescribeUnknownTool("write_slides", Declared());
        Assert.DoesNotContain("cannot show you an image", message);
    }
}
