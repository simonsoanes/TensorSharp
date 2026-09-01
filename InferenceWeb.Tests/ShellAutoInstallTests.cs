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
using System.Diagnostics;
using System.IO;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Installing the missing dependency and re-running inside the same call — and the bounds
/// that keep it from being worse than the problem.
///
/// <para>
/// This is the largest measured cost in the server's own logs: "the first run dies on a
/// missing dependency" was 17 incidents, 68 rounds, 117,231 output tokens and 116 minutes
/// of wall clock. The host already knew the module and the install command, and spent all
/// of that telling the model to do it — which also meant the model re-typed its whole
/// program, because a shell call carries its program in a heredoc.
/// </para>
/// <para>
/// The bounds matter as much as the loop. A recovery loop that runs a command six times
/// is not a fix; it is the same round budget spent inside one tool call, where the user
/// cannot see it and the caller's <c>timeout_ms</c> has been quietly discarded.
/// </para>
/// </summary>
public class ShellAutoInstallTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public ShellAutoInstallTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-autoinstall-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static bool HavePosixShell =>
        ShellProgram.TryResolve(null, out ShellProgram? shell, out _) && shell is { Kind: ShellKind.Posix };

    private static bool HavePython =>
        CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _);

    private ShellRunner Runner(Action<CodeExecOptions>? tweak = null)
    {
        var options = new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
            AllowInstall = true,
        };
        tweak?.Invoke(options);
        return new ShellRunner(options);
    }

    /// <summary>
    /// The bound that was missing when this shipped: <c>launch.Timeout</c> bounds ONE
    /// process, so a loop that re-runs six times could spend six times the deadline plus
    /// five installs inside a single call — while the tool declaration promised the
    /// caller's <c>timeout_ms</c>. A recovery loop is not a licence to ignore a contract.
    /// </summary>
    [Fact]
    public void TheLoopHonoursTheCallsTimeout_NotJustEachProcesss()
    {
        if (!HavePosixShell || !HavePython) return;

        using ShellRunner runner = Runner(o => o.MaxAutoInstalls = 5);

        // Every run fails on the same import, and no install can fix it (there is no
        // network here), so the loop is at its most expensive: this is the shape that
        // would have multiplied the deadline.
        var clock = Stopwatch.StartNew();
        CodeExecResult result = runner.Run(
            new ShellRequest("python3 -c 'import ts_absent_probe_pkg'")
            {
                Timeout = TimeSpan.FromSeconds(3),
            },
            _workspaces.GetOrCreate("budget"));
        clock.Stop();

        Assert.False(result.Ok);
        // Generously bounded so this is not a flaky timing test: what it forbids is
        // MULTIPLYING the deadline, which would be 15s+ of runs alone at MaxAutoInstalls=5.
        Assert.True(
            clock.Elapsed < TimeSpan.FromSeconds(12),
            $"the call took {clock.Elapsed.TotalSeconds:0.#}s against a 3s timeout");
    }

    /// <summary>
    /// The same package is never installed twice. A package that installs and does not fix
    /// the import is a different problem — the module and the distribution are named
    /// differently — and retrying it is the infinite loop this shape invites.
    /// </summary>
    [Fact]
    public void APackageThatDidNotFixTheImportIsNotInstalledAgain()
    {
        if (!HavePosixShell || !HavePython) return;

        using ShellRunner runner = Runner();
        CodeExecResult result = runner.Run(
            new ShellRequest("python3 -c 'import ts_absent_probe_pkg'"),
            _workspaces.GetOrCreate("once"));

        Assert.False(result.Ok);
        // Named once, not five times. The count is the assertion: a loop that retried the
        // same name would say this repeatedly.
        int mentions = 0;
        int at = 0;
        while ((at = result.Content.IndexOf("ts_absent_probe_pkg' was missing", at, StringComparison.Ordinal)) >= 0)
        {
            mentions++;
            at += 1;
        }
        Assert.Equal(1, mentions);
    }

    /// <summary>
    /// A background job has no output to diagnose yet, so there is nothing to install FOR.
    /// It must not be run twice behind the model's back.
    /// </summary>
    [Fact]
    public void ABackgroundJobIsNeverAutoInstalledFor()
    {
        if (!HavePosixShell || !HavePython) return;

        using ShellRunner runner = Runner();
        CodeExecResult result = runner.Run(
            new ShellRequest("python3 -c 'import ts_absent_probe_pkg'") { Background = true },
            _workspaces.GetOrCreate("bg"));

        Assert.DoesNotContain("was installed and the command was run again", result.Content);
    }

    /// <summary>
    /// With installing off there is nothing to try, and the result says so rather than
    /// naming a command the model cannot run.
    /// </summary>
    [Fact]
    public void WithInstallingOffNothingIsAttempted()
    {
        if (!HavePosixShell || !HavePython) return;

        using ShellRunner runner = Runner(o => o.AllowInstall = false);
        CodeExecResult result = runner.Run(
            new ShellRequest("python3 -c 'import ts_absent_probe_pkg'"),
            _workspaces.GetOrCreate("noinstall"));

        Assert.False(result.Ok);
        Assert.Contains("not enabled on this host", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("was missing", result.Content);
    }
}
