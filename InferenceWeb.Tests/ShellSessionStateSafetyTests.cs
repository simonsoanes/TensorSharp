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
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.Win32.SafeHandles;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// The shell state directory is writable by generated code. These tests pin the host
/// reader to files, links and FIFOs an adversarial command can leave there.
/// </summary>
public sealed class ShellSessionStateSafetyTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _manager;
    private readonly SessionWorkspace _workspace;

    public ShellSessionStateSafetyTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-shell-state-" + Guid.NewGuid().ToString("N"));
        _manager = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
        _workspace = _manager.GetOrCreate("s");
    }

    public void Dispose()
    {
        try { _manager.Release("s"); } catch { /* best effort */ }
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void OrdinaryStateStillRestoresTheCurrentDirectory()
    {
        if (!ShellProgram.TryResolve(null, out ShellProgram? shell, out _))
            return;

        string child = Path.Combine(_workspace.WorkDirectory, "child");
        Directory.CreateDirectory(child);
        File.WriteAllText(CwdFile, child);

        var session = new ShellSession(_workspace, shell!);

        Assert.Equal(child, session.CurrentDirectory);
    }

    [Fact]
    public void BomEncodedStateRemainsReadableForWindowsPowerShell()
    {
        string path = Path.Combine(_workspace.ShellStateDirectory, "bom-state");
        File.WriteAllText(path, "C:\\workspace", Encoding.Unicode);

        Assert.True(ShellSession.TryReadStateText(path, out string text));
        Assert.Equal("C:\\workspace", text);
    }

    [Fact]
    public void StateSymlinkIsNotFollowed()
    {
        string target = Path.Combine(_base, "outside-state");
        File.WriteAllText(target, _workspace.WorkDirectory);
        if (!TryCreateFileSymlink(CwdFile, target))
            return; // Windows needs Developer Mode or an elevated token to create one.

        Assert.False(ShellSession.TryReadStateText(CwdFile, out _));
        Assert.False(ShellSession.TryReadBoundedRegularTextUnderRoot(
            _workspace.ShellStateDirectory, CwdFile, 128 * 1024, out _));
    }

    [Fact]
    public void ParentDirectorySymlinkCannotRedirectABoundedWorkspaceRead()
    {
        string outside = Path.Combine(_base, "outside-parent");
        string nested = Path.Combine(outside, "source.py");
        string link = Path.Combine(_workspace.ShellStateDirectory, "linked-parent");
        Directory.CreateDirectory(outside);
        File.WriteAllText(nested, "SECRET_OUTSIDE_SOURCE\n");
        if (!TryCreateDirectorySymlink(link, outside))
            return;

        Assert.False(ShellSession.TryReadBoundedRegularTextUnderRoot(
            _workspace.ShellStateDirectory,
            Path.Combine(link, "source.py"),
            128 * 1024,
            out string text));
        Assert.Empty(text);
    }

    [Fact]
    public void PersistedDirectorySymlinkCannotLeaveTheWorkspace()
    {
        if (!ShellProgram.TryResolve(null, out ShellProgram? shell, out _))
            return;

        string outside = Path.Combine(_base, "outside-directory");
        string link = Path.Combine(_workspace.WorkDirectory, "outside-link");
        Directory.CreateDirectory(outside);
        if (!TryCreateDirectorySymlink(link, outside))
            return; // Windows needs Developer Mode or an elevated token to create one.

        File.WriteAllText(CwdFile, link);
        var session = new ShellSession(_workspace, shell!);

        Assert.Equal(_workspace.WorkDirectory, session.CurrentDirectory);
    }

    [Fact]
    public void InvalidPersistedPathFallsBackWithoutThrowing()
    {
        if (!ShellProgram.TryResolve(null, out ShellProgram? shell, out _))
            return;

        File.WriteAllBytes(CwdFile, new byte[] { (byte)'x', 0, (byte)'y' });
        var session = new ShellSession(_workspace, shell!);

        Assert.Equal(_workspace.WorkDirectory, session.CurrentDirectory);
    }

    [Fact]
    public void OversizedStateIsRejectedUnderTheReadCap()
    {
        string path = Path.Combine(_workspace.ShellStateDirectory, "oversized-state");
        File.WriteAllBytes(path, new byte[1024 * 1024]);

        Assert.False(ShellSession.TryReadStateText(path, out _));
    }

    [Fact]
    public void RootAnchoredReaderAcceptsTheExactCapAndRejectsOneByteMore()
    {
        const int cap = 128 * 1024;
        string path = Path.Combine(_workspace.ShellStateDirectory, "at-cap");
        File.WriteAllText(path, new string('x', cap));

        Assert.True(ShellSession.TryReadBoundedRegularTextUnderRoot(
            _workspace.ShellStateDirectory, path, cap, out string text));
        Assert.Equal(cap, text.Length);

        File.AppendAllText(path, "x");
        Assert.False(ShellSession.TryReadBoundedRegularTextUnderRoot(
            _workspace.ShellStateDirectory, path, cap, out _));
    }

    [Fact]
    public void FifoStateIsRejectedWithoutWaitingForInput()
    {
        if (OperatingSystem.IsWindows())
            return;

        string path = Path.Combine(_workspace.ShellStateDirectory, "fifo-state");
        Assert.Equal(0, MkFifo(path, Convert.ToUInt32("600", 8)));

        // Holding a FIFO open read/write lets even a regressed blocking reader open it.
        // The test thread can then close the only writer after a bounded wait, guaranteeing
        // that a failed assertion cannot strand the process in File.ReadAllText.
        int flags = 2 | (OperatingSystem.IsMacOS() ? 0x00000004 | 0x01000000
                                                   : 0x00000800 | 0x00080000);
        int fd = OpenUnix(path, flags);
        Assert.True(fd >= 0, $"open failed with errno {Marshal.GetLastWin32Error()}");

        using var keeper = new SafeFileHandle((IntPtr)fd, ownsHandle: true);
        AssertFifoRejectedWithoutBlocking(
            () => ShellSession.TryReadStateText(path, out _), keeper, "state reader");
        AssertFifoRejectedWithoutBlocking(
            () => ShellSession.TryReadBoundedRegularTextUnderRoot(
                _workspace.ShellStateDirectory, path, 128 * 1024, out _),
            keeper,
            "root-anchored reader");
    }

    private static void AssertFifoRejectedWithoutBlocking(
        Func<bool> read, SafeFileHandle keeper, string readerName)
    {
        bool accepted = false;
        Exception? failure = null;
        var reader = new Thread(() =>
        {
            try
            {
                accepted = read();
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        }) { IsBackground = true };

        reader.Start();
        bool completedWithoutRescue = reader.Join(TimeSpan.FromSeconds(2));
        if (!completedWithoutRescue)
        {
            // Closing the FIFO's only writer makes a legacy blocking read see EOF. The
            // reader is a background thread as a final guard, so even a broken rescue
            // cannot keep the test process alive.
            keeper.Dispose();
            Assert.True(reader.Join(TimeSpan.FromSeconds(2)),
                "the blocking FIFO reader could not be rescued");
        }

        Assert.Null(failure);
        Assert.True(completedWithoutRescue, $"the {readerName} blocked on a FIFO");
        Assert.False(accepted);
    }

    [Fact]
    public async Task ConcurrentLeafReplacementReturnsOneCompleteVersion()
    {
        string path = Path.Combine(_workspace.ShellStateDirectory, "replaced-source");
        string first = new('a', 64 * 1024);
        string second = new('b', 64 * 1024);
        File.WriteAllText(path, first);

        using var stop = new CancellationTokenSource();
        Task writer = Task.Run(() =>
        {
            int sequence = 0;
            while (!stop.IsCancellationRequested)
            {
                string replacement = path + "."
                    + (sequence++).ToString(System.Globalization.CultureInfo.InvariantCulture);
                File.WriteAllText(replacement, (sequence & 1) == 0 ? first : second);
                try
                {
                    File.Move(replacement, path, overwrite: true);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    try { File.Delete(replacement); } catch { /* best effort */ }
                }
            }
        });

        try
        {
            for (int i = 0; i < 100; i++)
            {
                bool read = ShellSession.TryReadBoundedRegularTextUnderRoot(
                    _workspace.ShellStateDirectory, path, 128 * 1024, out string text);
                Assert.True(!read || text == first || text == second,
                    "a path replacement must never produce mixed or partial content");
            }
        }
        finally
        {
            stop.Cancel();
            await writer.WaitAsync(TimeSpan.FromSeconds(5));
        }
    }

    [Fact]
    public async Task ConcurrentParentSwapNeverReadsThroughTheOutsideSymlink()
    {
        if (OperatingSystem.IsWindows())
            return;

        string root = _workspace.ShellStateDirectory;
        string outside = Path.Combine(_base, "swap-outside");
        string active = Path.Combine(root, "swap-parent");
        string parked = Path.Combine(root, "swap-parent-parked");
        string link = Path.Combine(root, "swap-parent-link");
        Directory.CreateDirectory(outside);
        Directory.CreateDirectory(active);
        File.WriteAllText(Path.Combine(active, "source.py"), "INSIDE\n");
        File.WriteAllText(Path.Combine(outside, "source.py"), "SECRET_OUTSIDE\n");
        if (!TryCreateDirectorySymlink(link, outside))
            return;

        using var stop = new CancellationTokenSource();
        Task toggler = Task.Run(() =>
        {
            while (!stop.IsCancellationRequested)
            {
                Rename(active, parked);
                Rename(link, active);
                Rename(active, link);
                Rename(parked, active);
                Thread.Yield();
            }
        });

        try
        {
            string source = Path.Combine(active, "source.py");
            for (int i = 0; i < 500; i++)
            {
                bool read = ShellSession.TryReadBoundedRegularTextUnderRoot(
                    root, source, 128 * 1024, out string text);
                Assert.True(!read || text == "INSIDE\n",
                    "a parent swap redirected a workspace read outside its root");
            }
        }
        finally
        {
            stop.Cancel();
            await toggler.WaitAsync(TimeSpan.FromSeconds(5));
        }
    }

    private string CwdFile => Path.Combine(_workspace.ShellStateDirectory, "cwd");

    private static bool TryCreateFileSymlink(string path, string target)
    {
        try
        {
            File.CreateSymbolicLink(path, target);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                      or PlatformNotSupportedException)
        {
            return false;
        }
    }

    private static bool TryCreateDirectorySymlink(string path, string target)
    {
        try
        {
            Directory.CreateSymbolicLink(path, target);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                      or PlatformNotSupportedException)
        {
            return false;
        }
    }

    [DllImport("libc", EntryPoint = "mkfifo", SetLastError = true)]
    private static extern int MkFifo(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, uint mode);

    [DllImport("libc", EntryPoint = "open", SetLastError = true)]
    private static extern int OpenUnix(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, int flags);

    private static void Rename(string from, string to)
    {
        if (RenameUnix(from, to) != 0)
            throw new IOException($"rename failed with errno {Marshal.GetLastWin32Error()}");
    }

    [DllImport("libc", EntryPoint = "rename", SetLastError = true)]
    private static extern int RenameUnix(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string from,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string to);
}
