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
using System.Diagnostics;
using System.IO;
using System.Linq;
using TensorSharp.AgentHost.CodeExec;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The guard around every child this host starts.
///
/// <para>
/// The hazard it exists for cannot be reproduced on demand — it needs another thread to
/// be inside malloc at the instant of a <c>fork()</c>, which is a race — so these do not
/// try to stage a wedge. They assert the two things that decide whether the recovery is
/// safe to have at all: that a healthy start is completely unaffected by the guard, and
/// that the evidence the kill rests on actually distinguishes a fork caught before
/// <c>exec</c> from every process that is not one. A watchdog that can kill the wrong
/// child is worse than the hang it replaces.
/// </para>
/// </summary>
public class ForkWatchdogTests
{
    private static bool OnMac => OperatingSystem.IsMacOS();

    // ---- the guard is transparent to a healthy start ---------------------

    [Fact]
    public void TryStart_RunsTheProcessAndCapturesItsOutput()
    {
        if (OperatingSystem.IsWindows())
            return;

        var output = new List<string>();
        var startInfo = new ProcessStartInfo
        {
            FileName = "/bin/echo",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.ArgumentList.Add("forkwatchdog-ok");

        Process Build()
        {
            var built = new Process { StartInfo = startInfo };
            built.OutputDataReceived += (_, e) => { if (e.Data != null) output.Add(e.Data); };
            return built;
        }

        Assert.True(ForkWatchdog.TryStart(Build, out Process? started, out string error), error);
        Assert.NotNull(started);
        Assert.Equal(string.Empty, error);

        using (started)
        {
            started!.BeginOutputReadLine();
            Assert.True(started.WaitForExit(30_000));
            started.WaitForExit();
            Assert.Equal(0, started.ExitCode);
        }

        Assert.Contains("forkwatchdog-ok", output);
    }

    [Fact]
    public void TryStart_ReportsAnExecutableThatDoesNotExist()
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = Path.Combine(Path.GetTempPath(), "no-such-program-" + Guid.NewGuid().ToString("N")),
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        var sw = Stopwatch.StartNew();
        bool ok = ForkWatchdog.TryStart(
            () => new Process { StartInfo = startInfo }, out Process? started, out string error);
        sw.Stop();

        Assert.False(ok);
        Assert.Null(started);
        Assert.NotEqual(string.Empty, error);

        // A start that failed for a reason of its own must not be retried, and must not be
        // waited on. Three attempts of a scan interval each would turn "no such file" into
        // a multi-second stall on a path the model can reach with a typo.
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), $"took {sw.Elapsed}");
    }

    [Fact]
    public void TryStart_DoesNotDisturbAChildThatIsAlreadyRunning()
    {
        if (OperatingSystem.IsWindows())
            return;

        using Process sleeper = StartSleeper(seconds: 10);
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = "/bin/echo",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            startInfo.ArgumentList.Add("hello");

            Assert.True(
                ForkWatchdog.TryStart(
                    () => new Process { StartInfo = startInfo }, out Process? started, out string error),
                error);
            using (started) { started!.WaitForExit(30_000); }

            Assert.False(sleeper.HasExited, "a guarded start killed an unrelated child");
        }
        finally
        {
            TryKill(sleeper);
        }
    }

    // ---- the evidence the kill rests on ----------------------------------

    [Fact]
    public void ChildPids_ListsOurOwnChildrenAndNothingElse()
    {
        if (!OnMac)
            return;

        HashSet<int> before = ForkWatchdog.ChildPids();
        using Process sleeper = StartSleeper(seconds: 10);
        try
        {
            HashSet<int> after = ForkWatchdog.ChildPids();

            Assert.Contains(sleeper.Id, after);
            Assert.DoesNotContain(sleeper.Id, before);

            // Our own pid is not our own child; a scan that returned it would make the
            // watchdog a suicide pact the first time a start ran long.
            Assert.DoesNotContain(Environment.ProcessId, after);
        }
        finally
        {
            TryKill(sleeper);
        }
    }

    [Fact]
    public void PathOf_ReportsWhatAProcessBecame_NotWhatForkedIt()
    {
        if (!OnMac)
            return;

        string self = ForkWatchdog.PathOf(Environment.ProcessId);
        Assert.NotEqual(string.Empty, self);

        using Process sleeper = StartSleeper(seconds: 10);
        try
        {
            string child = ForkWatchdog.PathOf(sleeper.Id);

            // THE discriminator. A child that reached exec reports the program it became,
            // so it can never be mistaken for a fork of this host that did not.
            Assert.EndsWith("sleep", child, StringComparison.Ordinal);
            Assert.NotEqual(self, child);
        }
        finally
        {
            TryKill(sleeper);
        }

        Assert.Equal(string.Empty, ForkWatchdog.PathOf(-1));
    }

    [Fact]
    public void ThreadCountOf_SeparatesALiveRuntimeFromAForkThatCarriesOneThread()
    {
        if (!OnMac)
            return;

        // The backstop for the one case the image path alone cannot rule out: a process
        // that legitimately IS this executable. fork() carries exactly the calling thread,
        // so a wedged child has one; anything with a runtime under it has many, and this
        // test process is that thing.
        Assert.True(
            ForkWatchdog.ThreadCountOf(Environment.ProcessId) > 1,
            "a live .NET process must not look like a single-threaded fork");

        Assert.Equal(-1, ForkWatchdog.ThreadCountOf(-1));
    }

    // ---- helpers ----------------------------------------------------------

    private static Process StartSleeper(int seconds)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "/bin/sleep",
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.ArgumentList.Add(seconds.ToString(System.Globalization.CultureInfo.InvariantCulture));

        var process = Process.Start(startInfo)!;
        // proc_listpids reflects a child as soon as it exists, but the assertions here are
        // about a child that has EXEC'd, which is the state being distinguished.
        for (int i = 0; i < 100 && ForkWatchdog.PathOf(process.Id).EndsWith("sleep", StringComparison.Ordinal) == false; i++)
            System.Threading.Thread.Sleep(20);
        return process;
    }

    private static void TryKill(Process process)
    {
        try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
        catch (Exception) { /* already gone */ }
    }

    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }
}
