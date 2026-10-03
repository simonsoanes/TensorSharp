// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.AgentHost.Skills;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Which lines of the system-wide sandbox log a failed run's stderr may carry. They go to
/// the model, where noise reads as the cause: a Mac app run printed
/// "sharingd(755) deny(1) syscall-unix 494" and "UserEventAgent(665) deny(1)
/// file-read-data ~/Library/HomeKit/homeeventsd" as observations about ITS Python script,
/// because "sharingd" contains "sh" and the HomeKit path is under the user's home.
/// </summary>
public class SandboxViolationFilterTests
{
    private const string Work = "/Users/me/Library/Caches/TensorAgent/scratch/s1/work";
    private const string Home = "/Users/me";
    private const string Prefix = "2026-09-30 11:25:46.123 E  kernel[0:1a2b] (Sandbox) Sandbox: ";

    private static bool Ours(string message, bool stillRunning = false, string interpreter = "bash") =>
        SandboxViolationMonitor.IsPlausiblyOurs(Prefix + message, interpreter, Work, Home, _ => stillRunning);

    [Theory]
    [InlineData("Python(32142) deny(1) system-info vfs.disk-space")]
    [InlineData("python3.13(501) deny(1) file-read-data /etc/master.passwd")]
    [InlineData("node(77) deny(1) network-outbound 93.184.216.34:443")]
    [InlineData("sh(9) deny(1) process-exec /usr/bin/su")]
    [InlineData("bash(12) deny(1) file-write-create /opt/x")]
    public void TheRunsOwnInterpretersAreRecognisedByName(string message)
    {
        Assert.True(Ours(message, stillRunning: true));
    }

    [Theory]
    [InlineData("sharingd(755) deny(1) syscall-unix 494")]
    [InlineData("shortcutsd(88) deny(1) mach-lookup com.apple.x")]
    [InlineData("nodeagentd(90) deny(1) file-read-data /private/var/db/x")]
    public void AnotherProgramWhoseNameStartsTheSameIsNot(string message)
    {
        Assert.False(Ours(message));
    }

    [Fact]
    public void AHomePathCountsOnlyFromAProcessThatHasExited()
    {
        const string daemon = "UserEventAgent(665) deny(1) file-read-data /Users/me/Library/HomeKit/homeeventsd";
        Assert.False(Ours(daemon, stillRunning: true));
        // The run's own processes have exited by the time a failed run is read back, so a
        // helper this list does not name still shows its denial of the user's files.
        Assert.True(Ours("ffmpeg(4001) deny(1) file-read-data /Users/me/.ssh/id_ed25519", stillRunning: false));
    }

    [Fact]
    public void TheWorkspaceIsAlwaysTheRuns()
    {
        Assert.True(Ours("mdworker(3) deny(1) file-read-data " + Work + "/report.xlsx", stillRunning: true));
    }

    [Fact]
    public void ABrowserHelpersPidIsTheLastParenthesis()
    {
        const string helper = "Google Chrome for Testing Helper (Renderer)(4242) deny(1) file-read-data /Users/me/Documents/a.txt";
        Assert.False(Ours(helper, stillRunning: true));
        Assert.True(Ours(helper, stillRunning: false));
    }
}
