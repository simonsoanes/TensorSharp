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
using System.IO;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Host paths must not reach the model.
///
/// <para>
/// Measured on this server's own logs: 13.9% of the characters in every recoverable tool
/// result were absolute workspace paths, and one logged round was lost to a model
/// splicing two of those 32-hex-digit session ids into a directory that never existed.
/// </para>
/// </summary>
public class OutputPathsTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _manager;
    private readonly SessionWorkspace _workspace;

    public OutputPathsTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-paths-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _manager = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
        _workspace = _manager.GetOrCreate("s");
    }

    public void Dispose()
    {
        try { _manager.Release("s"); } catch { /* best effort */ }
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    /// <summary>The shape from the log, verbatim: a traceback frame naming the model's own file.</summary>
    [Fact]
    public void ATracebackFrameLosesItsHostPrefix()
    {
        string text =
            "Traceback (most recent call last):\n"
            + $"  File \"{Path.Combine(_workspace.WorkDirectory, "main.py")}\", line 86, in <module>\n"
            + "    slide.notes_page.shapes.add_textbox()\n";

        string scrubbed = OutputPaths.Scrub(text, _workspace);

        Assert.DoesNotContain(_workspace.WorkDirectory, scrubbed);
        Assert.Contains("File \"main.py\", line 86", scrubbed, StringComparison.Ordinal);
    }

    /// <summary>
    /// The whole point of using <see cref="Path.GetRelativePath(string,string)"/> rather
    /// than a placeholder: what comes back is a path the model can put into its next
    /// command and have it mean the same file.
    /// </summary>
    [Fact]
    public void WhatReplacesTheEnvDirectoryIsStillAUsablePath()
    {
        string text = Path.Combine(_workspace.EnvDirectory, "pptx", "__init__.py") + ": broken\n";

        string scrubbed = OutputPaths.Scrub(text, _workspace, _workspace.WorkDirectory);

        Assert.DoesNotContain(_workspace.EnvDirectory, scrubbed);
        string relative = scrubbed.Split(':')[0];
        Assert.Equal(
            Path.GetFullPath(Path.Combine(_workspace.EnvDirectory, "pptx", "__init__.py")),
            Path.GetFullPath(Path.Combine(_workspace.WorkDirectory, relative)));
    }

    /// <summary>
    /// Relative to where the command RAN, not to the work directory: a model that has
    /// <c>cd</c>'d into a subdirectory reads the path back and uses it from there.
    /// </summary>
    [Fact]
    public void PathsAreRelativeToTheDirectoryTheCommandRanIn()
    {
        string sub = Path.Combine(_workspace.WorkDirectory, "src");
        Directory.CreateDirectory(sub);

        string scrubbed = OutputPaths.Scrub(
            Path.Combine(_workspace.WorkDirectory, "deck.py") + "\n", _workspace, sub);

        Assert.Equal("../deck.py\n", scrubbed.Replace('\\', '/'));
    }

    /// <summary>
    /// The work directory standing alone must become a path, not nothing. This is the
    /// <c>pwd</c> case, and also the <c>cd: …: No such file or directory</c> case that
    /// produced the spliced-path failure in the log.
    /// </summary>
    [Fact]
    public void TheWorkDirectoryOnItsOwnBecomesDot()
    {
        Assert.Equal("cwd is .\n", OutputPaths.Scrub($"cwd is {_workspace.WorkDirectory}\n", _workspace));
    }

    /// <summary>
    /// A directory whose name merely BEGINS with the work directory's is a DIFFERENT
    /// directory, and its name has to survive intact. The path is still made relative —
    /// it is inside the workspace — but rewriting "work" as a prefix of "work-backup"
    /// would silently rename a real file in the output.
    /// </summary>
    [Fact]
    public void ASiblingWhoseNameStartsWithTheWorkDirectorysIsNotTruncated()
    {
        string sibling = _workspace.WorkDirectory + "-backup/deck.py";

        string scrubbed = OutputPaths.Scrub(sibling + "\n", _workspace).Replace('\\', '/');

        Assert.DoesNotContain(_workspace.Root, scrubbed);
        Assert.Contains(
            Path.GetFileName(_workspace.WorkDirectory) + "-backup/deck.py",
            scrubbed,
            StringComparison.Ordinal);
    }

    /// <summary>Text with nothing to rewrite comes back untouched, and by the same reference.</summary>
    [Fact]
    public void UnrelatedOutputIsLeftExactlyAlone()
    {
        const string text = "hello world\n/usr/lib/python3.12/json/__init__.py\n";
        Assert.Same(text, OutputPaths.Scrub(text, _workspace));
    }

    [Fact]
    public void NoWorkspaceMeansNoRewriting()
    {
        const string text = "/some/where\n";
        Assert.Same(text, OutputPaths.Scrub(text, workspace: null));
    }

    /// <summary>
    /// On macOS the same directory is reached as both <c>/var/folders/…</c> and
    /// <c>/private/var/folders/…</c>; which one a process prints depends on how it got
    /// the path, and a rewrite that knows only one spelling misses every mention.
    /// </summary>
    [Fact]
    public void BothMacOsSpellingsOfTheSameDirectoryAreRewritten()
    {
        if (!OperatingSystem.IsMacOS() || !_workspace.WorkDirectory.StartsWith("/var/", StringComparison.Ordinal))
            return;

        string aliased = "/private" + _workspace.WorkDirectory + "/deck.py";
        Assert.Equal("deck.py\n", OutputPaths.Scrub(aliased + "\n", _workspace));
    }

    /// <summary>
    /// End to end through the real shell: the one place this has to hold, because it is
    /// the text the model actually receives.
    /// </summary>
    [Fact]
    public void ARealCommandsOutputCarriesNoHostPaths()
    {
        var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = Path.Combine(_base, "scratch"),
            Timeout = TimeSpan.FromSeconds(30),
        });
        try
        {
            if (!runner.CanRun)
                return;

            // In the dialect this host speaks. `pwd && ls -d "$PWD"` is bash, and Windows
            // PowerShell 5.1 has no `&&` at all - the command was a parse error there, so
            // the assertions below ran against a FAILED command and the test reported a
            // redaction defect that did not exist. `pwd` is a PowerShell alias for
            // Get-Location, so only the second half needs translating.
            CodeExecResult result = runner.Run(
                new ShellRequest(runner.Shell is { Kind: ShellKind.PowerShell }
                    ? "pwd; (Get-Location).Path"
                    : "pwd && ls -d \"$PWD\""),
                _workspace);

            Assert.True(result.Ok);
            Assert.DoesNotContain(_workspace.WorkDirectory, result.Content);
            Assert.DoesNotContain(_workspace.Root, result.Content);
            Assert.DoesNotContain("ts-session-", result.Content);
        }
        finally
        {
            runner.Dispose();
        }
    }
}
