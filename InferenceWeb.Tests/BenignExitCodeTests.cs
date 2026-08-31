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
/// A command whose exit 1 is its ANSWER must not be reported as a failure.
///
/// <para>
/// <c>grep</c> exits 1 when it finds nothing. <c>diff</c> exits 1 when the files differ.
/// <c>test -f x</c> exits 1 when x is absent. <c>git diff --quiet</c> exits 1 when there
/// are changes. Every one of those is a correct command answering the question it was
/// asked, and every one of them was reported to the model as broken — manufacturing a
/// recovery round out of nothing, in a loop where 39.7% of rounds are already recovery.
/// </para>
/// <para>
/// Claude Code's Bash tool exempts exactly this, as a closed list of command names at exit
/// 1 only: grep, rg, egrep, fgrep, find, diff, test, [, plus git diff and git grep.
/// </para>
/// </summary>
public class BenignExitCodeTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public BenignExitCodeTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-benign-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- the table -----------------------------------------------------------

    [Theory]
    [InlineData("grep needle haystack.txt")]
    [InlineData("egrep needle haystack.txt")]
    [InlineData("rg needle")]
    [InlineData("diff a.txt b.txt")]
    [InlineData("test -f absent.txt")]
    [InlineData("[ -f absent.txt ]")]
    [InlineData("git diff --quiet")]
    [InlineData("git grep needle")]
    [InlineData("/usr/bin/grep needle f")]
    public void ExitOneIsAnAnswerForTheseCommands(string command)
    {
        Assert.True(ShellCommand.ExitCodeIsBenign(command, 1), command);
    }

    /// <summary>Only exit 1, and only those names. `grep` exits 2 on a real error.</summary>
    [Theory]
    [InlineData("grep needle haystack.txt", 2)]
    [InlineData("grep needle haystack.txt", 127)]
    [InlineData("python3 solve.py", 1)]
    [InlineData("git commit -m x", 1)]
    [InlineData("npm install left-pad", 1)]
    [InlineData("make", 1)]
    public void EverythingElseIsStillAFailure(string command, int exitCode)
    {
        Assert.False(ShellCommand.ExitCodeIsBenign(command, exitCode), $"{command} -> {exitCode}");
    }

    /// <summary>
    /// The status belongs to the LAST simple command, because that is whose status the
    /// shell reports. Getting this backwards would exempt a failing build that happened to
    /// start with a grep.
    /// </summary>
    [Fact]
    public void ThePipelinesLastCommandIsTheOneThatCounts()
    {
        // Exits with wc's status, so not exempt.
        Assert.False(ShellCommand.ExitCodeIsBenign("grep x f | wc -l", 1));
        // Exits with grep's status, so exempt.
        Assert.True(ShellCommand.ExitCodeIsBenign("cat f | grep x", 1));
        Assert.True(ShellCommand.ExitCodeIsBenign("python3 gen.py && grep needle out.txt", 1));
        Assert.False(ShellCommand.ExitCodeIsBenign("grep needle out.txt && python3 use.py", 1));
    }

    // ---- end to end ----------------------------------------------------------

    /// <summary>
    /// The assertion that matters, through the real runner: a search that correctly found
    /// nothing comes back as a SUCCESS, with the exit code still shown and explained.
    /// </summary>
    [Fact]
    public void ASearchThatFoundNothingIsNotReportedAsBroken()
    {
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        });
        if (!runner.CanRun || runner.Shell is not { Kind: ShellKind.Posix })
            return;

        SessionWorkspace workspace = _workspaces.GetOrCreate("benign");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "haystack.txt"), "alpha\nbeta\n");

        CodeExecResult result = runner.Run(
            new ShellRequest("grep ts-definitely-not-here haystack.txt"), workspace);

        Assert.True(result.Ok, result.Content);
        // The number is still there — this adds the reading, it does not hide the number.
        Assert.Contains("exit 1", result.Content, StringComparison.Ordinal);
        Assert.Contains("no match", result.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// And a repeated search that keeps correctly finding nothing must not be accused of
    /// looping. That is the false-failure defect reappearing one layer along.
    /// </summary>
    [Fact]
    public void ARepeatedSearchThatKeepsFindingNothingIsNotAccusedOfLooping()
    {
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        });
        if (!runner.CanRun || runner.Shell is not { Kind: ShellKind.Posix })
            return;

        SessionWorkspace workspace = _workspaces.GetOrCreate("repeatsearch");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "haystack.txt"), "alpha\n");

        var request = new ShellRequest("grep ts-absent haystack.txt");
        runner.Run(request, workspace);
        runner.Run(request, workspace);
        CodeExecResult third = runner.Run(request, workspace);

        Assert.True(third.Ok, third.Content);
        Assert.DoesNotContain("already run this command", third.Content);
    }

    /// <summary>A genuinely failing command still fails, and still gets its coaching.</summary>
    [Fact]
    public void AGenuinelyFailingCommandIsUnaffected()
    {
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        });
        if (!runner.CanRun || runner.Shell is not { Kind: ShellKind.Posix })
            return;

        CodeExecResult result = runner.Run(
            new ShellRequest("ts-no-such-program-xyz"), _workspaces.GetOrCreate("realfail"));

        Assert.False(result.Ok);
        Assert.Contains("This is the ENVIRONMENT", result.Content, StringComparison.Ordinal);
    }
}
