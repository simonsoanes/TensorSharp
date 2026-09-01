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
using System.Linq;
using System.Threading.Tasks;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Reading an install out of a command line the model wrote, when that line also carries
/// the shell plumbing every model puts on it.
///
/// <para>
/// Both cases here are recorded defects with the same shape — the one this codebase calls
/// cardinal. A redirection read as a package name got the install refused, the segment
/// substituted with <c>false</c>, the rest of the pipeline run anyway, and the result
/// reported <b>exit 0</b>. Nothing in that result contradicts "it worked".
/// </para>
/// </summary>
public class ShellInstallParseTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public ShellInstallParseTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-installparse-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- redirections are not words of the command ---------------------------

    [Fact]
    public void ARedirectionIsNotReadAsAWordOfTheCommand()
    {
        Assert.Equal(
            new[] { "pip3", "install", "python-pptx" },
            ShellCommand.WordsOf("pip3 install python-pptx 2>&1").ToArray());

        Assert.Equal(
            new[] { "pip", "install", "numpy" },
            ShellCommand.WordsOf("pip install numpy > log.txt").ToArray());

        Assert.Equal(
            new[] { "pip", "install", "numpy" },
            ShellCommand.WordsOf("pip install numpy >>log.txt 2>&1").ToArray());

        Assert.Equal(
            new[] { "python3", "solve.py" },
            ShellCommand.WordsOf("python3 solve.py < input.txt").ToArray());
    }

    /// <summary>
    /// The command itself must survive intact. A redirection filter that ate real words
    /// would misclassify the command, which is a worse failure than the one it fixes.
    /// </summary>
    [Fact]
    public void OrdinaryWordsAreUntouched()
    {
        Assert.Equal(
            new[] { "echo", "a>b" },
            ShellCommand.WordsOf("echo 'a>b'").ToArray());

        Assert.Equal(
            new[] { "python3", "-c", "print(1)" },
            ShellCommand.WordsOf("python3 -c \"print(1)\"").ToArray());
    }

    /// <summary>
    /// End to end, and the assertion that matters: the install is read correctly, so the
    /// line is never mutilated and the result never says <c>exit 0</c> for a command whose
    /// install was refused.
    /// </summary>
    [Fact]
    public void AnInstallWithARedirectionIsReadWithoutMangingTheLine()
    {
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
            AllowInstall = true,
            // An allow-list of exactly one name, so the install is refused BY NAME rather
            // than by a package that does not exist — deterministic, and offline.
            AllowedPackages = new[] { "an-allowed-package" },
        });
        if (!runner.CanRun || runner.Shell is not { Kind: ShellKind.Posix })
            return;

        CodeExecResult result = runner.Run(
            new ShellRequest("pip3 install python-pptx 2>&1 | tail -5"),
            _workspaces.GetOrCreate("redir"));

        // Refused as a whole. What must never happen is the old behaviour: the install
        // rejected for a package named "2>", the segment replaced with `false`, the
        // residual `false | tail -5` run, and the answer "exit 0".
        Assert.False(result.Ok);
        // The install was refused BY THE PACKAGE THE MODEL NAMED, not by a package name
        // invented out of the redirection. `'2>' is not a valid package name` is the old
        // behaviour, and it sent the model to fix a package it had never asked for.
        Assert.Contains("'python-pptx' is not on this host's allowed-package list", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("'2>'", result.Content);

        // And the deeper half: the residual line still runs (substituting `false` is what
        // keeps `&&` and `||` meaning what the model wrote), so its exit status is
        // reported honestly — but the CALL is not a success, and the result says plainly
        // that what ran is not what was sent. The logged version of this said "exit 0"
        // and nothing else.
        Assert.Contains("did NOT do what you asked", result.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// Two conversations installing at the same time. The substitution used to read a
    /// shared field, so one session could apply its own install spans — offsets into its
    /// own line — to the other session's command text, and when the other line was longer
    /// there was no exception: it just ran a splice of somebody else's command.
    /// </summary>
    [Fact]
    public void TwoSessionsInstallingAtOnce_DoNotSeeEachOthersCommandText()
    {
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(60),
            ScratchDirectory = _base,
            AllowInstall = true,
            AllowedPackages = new[] { "an-allowed-package" },
        });
        if (!runner.CanRun || runner.Shell is not { Kind: ShellKind.Posix })
            return;

        SessionWorkspace a = _workspaces.GetOrCreate("race-a");
        SessionWorkspace b = _workspaces.GetOrCreate("race-b");

        // Deliberately different lengths: the dangerous variant of the race is the one
        // where the other session's line is LONGER, because then the bad offset is still
        // in range and nothing throws.
        const string shortLine = "pip install alpha && echo A";
        const string longLine =
            "pip install beta-with-a-much-longer-name && echo BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB";

        var results = new CodeExecResult[2];
        Parallel.Invoke(
            () => results[0] = runner.Run(new ShellRequest(shortLine), a),
            () => results[1] = runner.Run(new ShellRequest(longLine), b));

        foreach (CodeExecResult result in results)
        {
            // Whatever each said about its own install, neither may carry the other's text.
            Assert.DoesNotContain("BBBBBBBBBB", results[0].Content);
            Assert.DoesNotContain("echo A", results[1].Content);
        }
    }
}
