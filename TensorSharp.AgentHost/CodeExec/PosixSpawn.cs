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
using System.Runtime.InteropServices;
using System.Threading;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// <c>posix_spawn</c> and the handful of libc calls around it.
    ///
    /// <para>
    /// This exists so that starting a child process does not go through <c>fork()</c>.
    /// .NET's <see cref="System.Diagnostics.Process"/> forks and then execs, and a
    /// <c>fork()</c> in a multi-threaded process is only safe if the child touches nothing
    /// but async-signal-safe functions — which the child does not get to decide, because
    /// libmalloc's <c>pthread_atfork</c> handler runs first. On macOS 26's xzone allocator
    /// that handler spins forever when another thread held a zone lock at the instant of
    /// the fork, and the child then never execs, never exits and never closes the
    /// close-on-exec status pipe its parent is blocked reading. This host runs 60+ threads
    /// with an inference engine allocating continuously, so it loses that race in practice.
    /// </para>
    /// <para>
    /// <c>posix_spawn</c> has no such window. The child is created already-exec'd — on
    /// macOS by a single kernel call, on glibc by <c>clone(CLONE_VM|CLONE_VFORK)</c> — and
    /// in neither case do the parent's atfork handlers run. Everything the fork path used
    /// to do between the two halves (redirect descriptors, change directory, replace the
    /// environment, reset signals, start a new process group) is expressed instead as file
    /// actions and attributes that the kernel applies itself.
    /// </para>
    /// <para>
    /// The opaque types are allocated as fixed 1KB zeroed blocks rather than as declared
    /// structs. They are pointer-sized on macOS and several hundred bytes on glibc, both
    /// initialised by their own <c>_init</c> call, so over-allocating is the portable way
    /// to hold either without encoding a layout that differs per platform.
    /// </para>
    /// </summary>
    internal static class PosixSpawn
    {
        /// <summary>Generous enough for the largest <c>posix_spawnattr_t</c> in circulation.</summary>
        private const int OpaqueSize = 1024;

        /// <summary>Big enough for a glibc <c>sigset_t</c> (128 bytes); macOS uses 4.</summary>
        private const int SigSetSize = 256;

        internal const short SetPgroup = 0x0002;
        internal const short SetSigDef = 0x0004;
        internal const short SetSigMask = 0x0008;

        /// <summary>
        /// macOS only: close every descriptor the child did not receive through a file
        /// action. The fork path closed them by hand; this asks the kernel to, which also
        /// means a sandboxed child cannot inherit a socket or a model file by accident.
        /// </summary>
        internal const short CloExecDefault = 0x4000;

        internal const int Sigkill = 9;
        private const int F_SETFD = 2;
        private const int FD_CLOEXEC = 1;

        [DllImport("libc", SetLastError = true)]
        private static extern int posix_spawnp(
            out int pid, IntPtr path, IntPtr fileActions, IntPtr attr, IntPtr argv, IntPtr envp);

        [DllImport("libc")] private static extern int posix_spawn_file_actions_init(IntPtr fa);
        [DllImport("libc")] private static extern int posix_spawn_file_actions_destroy(IntPtr fa);
        [DllImport("libc")] private static extern int posix_spawn_file_actions_adddup2(IntPtr fa, int fd, int newFd);

        // macOS 26 deprecated the _np spelling in favour of the plain one; glibc has only
        // the _np spelling. Which exists is decided once, at run time, by calling them.
        [DllImport("libc", EntryPoint = "posix_spawn_file_actions_addchdir")]
        private static extern int AddChdir(IntPtr fa, IntPtr path);
        [DllImport("libc", EntryPoint = "posix_spawn_file_actions_addchdir_np")]
        private static extern int AddChdirNp(IntPtr fa, IntPtr path);

        [DllImport("libc")] private static extern int posix_spawnattr_init(IntPtr attr);
        [DllImport("libc")] private static extern int posix_spawnattr_destroy(IntPtr attr);
        [DllImport("libc")] private static extern int posix_spawnattr_setflags(IntPtr attr, short flags);
        [DllImport("libc")] private static extern int posix_spawnattr_setpgroup(IntPtr attr, int pgroup);
        [DllImport("libc")] private static extern int posix_spawnattr_setsigdefault(IntPtr attr, IntPtr set);
        [DllImport("libc")] private static extern int posix_spawnattr_setsigmask(IntPtr attr, IntPtr set);

        [DllImport("libc")] private static extern int sigfillset(IntPtr set);
        [DllImport("libc")] private static extern int sigemptyset(IntPtr set);

        [DllImport("libc", SetLastError = true)] private static extern int pipe(int[] fds);
        [DllImport("libc", SetLastError = true)] internal static extern int close(int fd);
        [DllImport("libc", SetLastError = true)] private static extern int fcntl(int fd, int cmd, int arg);
        [DllImport("libc", SetLastError = true)] internal static extern int waitpid(int pid, out int status, int options);
        [DllImport("libc", SetLastError = true)]
        private static extern int waitid(int idtype, uint id, byte[] info, int options);
        [DllImport("libc", SetLastError = true)] internal static extern int kill(int pid, int sig);
        [DllImport("libc", SetLastError = true)] internal static extern int getpgid(int pid);

        private enum ChdirSupport { Unknown = 0, Plain, Np, None }

        private static readonly Lazy<ChdirSupport> Chdir = new(ProbeChdir, LazyThreadSafetyMode.ExecutionAndPublication);
        private static readonly Lazy<bool> Supported = new(Probe, LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>
        /// Whether this host can spawn without forking. False on Windows, and on any Unix
        /// whose libc does not offer the calls — in which case the caller falls back to
        /// <see cref="System.Diagnostics.Process"/> behind <see cref="ForkWatchdog"/>.
        /// </summary>
        internal static bool IsSupported => Supported.Value;

        /// <summary>
        /// Wait until <paramref name="pid"/> exits without reaping it. Keeping the leader
        /// waitable also keeps its PID reserved while the caller signals the process group,
        /// so a recycled PID can never make that signal name an unrelated group.
        /// </summary>
        internal static bool WaitForExitWithoutReaping(int pid)
        {
            if (OperatingSystem.IsWindows() || pid <= 0)
                return false;

            // P_PID is 1 and WEXITED is 4 on both supported Unix families. WNOWAIT is
            // unfortunately not ABI-identical: Darwin uses 0x20, Linux 0x01000000.
            const int Pid = 1;
            const int WExited = 0x00000004;
            int noWait = OperatingSystem.IsMacOS() ? 0x00000020 : 0x01000000;
            var info = new byte[256]; // larger than siginfo_t on Darwin and Linux

            try
            {
                int rc;
                do
                {
                    rc = waitid(Pid, unchecked((uint)pid), info, WExited | noWait);
                }
                while (rc < 0 && Marshal.GetLastWin32Error() == 4); // EINTR

                return rc == 0;
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                return false;
            }
        }

        private static bool Probe()
        {
            if (OperatingSystem.IsWindows())
                return false;

            IntPtr fa = Zeroed(OpaqueSize);
            IntPtr attr = Zeroed(OpaqueSize);
            try
            {
                if (posix_spawn_file_actions_init(fa) != 0)
                    return false;
                posix_spawn_file_actions_destroy(fa);
                if (posix_spawnattr_init(attr) != 0)
                    return false;
                posix_spawnattr_destroy(attr);

                // A working directory that cannot be set is not a degraded spawn, it is the
                // wrong directory — every caller runs its child inside a session workspace.
                // Better to fall back to the forked path than to run somewhere else.
                return Chdir.Value != ChdirSupport.None;
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                return false;
            }
            finally
            {
                Marshal.FreeCoTaskMem(fa);
                Marshal.FreeCoTaskMem(attr);
            }
        }

        private static ChdirSupport ProbeChdir()
        {
            IntPtr fa = Zeroed(OpaqueSize);
            IntPtr root = Marshal.StringToCoTaskMemUTF8("/");
            try
            {
                if (posix_spawn_file_actions_init(fa) != 0)
                    return ChdirSupport.None;
                try
                {
                    if (AddChdir(fa, root) == 0)
                        return ChdirSupport.Plain;
                }
                catch (EntryPointNotFoundException) { /* glibc: only the _np spelling */ }

                try
                {
                    if (AddChdirNp(fa, root) == 0)
                        return ChdirSupport.Np;
                }
                catch (EntryPointNotFoundException) { /* neither: fall back to fork */ }

                return ChdirSupport.None;
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                return ChdirSupport.None;
            }
            finally
            {
                posix_spawn_file_actions_destroy(fa);
                Marshal.FreeCoTaskMem(fa);
                Marshal.FreeCoTaskMem(root);
            }
        }

        /// <summary>A pipe whose BOTH ends are close-on-exec until a file action says otherwise.</summary>
        /// <remarks>
        /// The child receives its end through <c>adddup2</c>, and dup2 clears close-on-exec
        /// on the duplicate. So the descriptors the child gets are exactly 0, 1 and 2, and
        /// the originals it would otherwise have inherited are not among them.
        /// </remarks>
        internal static bool TryPipe(out int readFd, out int writeFd)
        {
            var fds = new int[2];
            if (pipe(fds) != 0)
            {
                readFd = writeFd = -1;
                return false;
            }
            readFd = fds[0];
            writeFd = fds[1];
            fcntl(readFd, F_SETFD, FD_CLOEXEC);
            fcntl(writeFd, F_SETFD, FD_CLOEXEC);
            return true;
        }

        /// <summary>
        /// Spawn <paramref name="fileName"/>, already exec'd, with the three standard
        /// descriptors bound to the given pipe ends.
        /// </summary>
        /// <returns>0 on success, otherwise the errno the spawn failed with.</returns>
        internal static int Spawn(
            string fileName, IReadOnlyList<string> arguments, string? workingDirectory,
            IReadOnlyDictionary<string, string> environment,
            int stdinFd, int stdoutFd, int stderrFd, out int pid)
        {
            pid = -1;

            IntPtr fa = Zeroed(OpaqueSize);
            IntPtr attr = Zeroed(OpaqueSize);
            IntPtr sigs = Zeroed(SigSetSize);
            var owned = new List<IntPtr> { fa, attr, sigs };

            bool faReady = false, attrReady = false;
            try
            {
                if (posix_spawn_file_actions_init(fa) != 0)
                    return -1;
                faReady = true;
                if (posix_spawnattr_init(attr) != 0)
                    return -1;
                attrReady = true;

                if (posix_spawn_file_actions_adddup2(fa, stdinFd, 0) != 0
                    || posix_spawn_file_actions_adddup2(fa, stdoutFd, 1) != 0
                    || posix_spawn_file_actions_adddup2(fa, stderrFd, 2) != 0)
                {
                    return -1;
                }

                if (workingDirectory is { Length: > 0 })
                {
                    IntPtr cwd = Marshal.StringToCoTaskMemUTF8(workingDirectory);
                    owned.Add(cwd);
                    int rc = Chdir.Value == ChdirSupport.Plain ? AddChdir(fa, cwd) : AddChdirNp(fa, cwd);
                    if (rc != 0)
                        return rc;
                }

                // Signals back to default and unblocked. The fork path did this by hand,
                // and it matters more than it looks: this host ignores SIGPIPE, and a child
                // that inherited that runs `yes | head -1` forever instead of dying on the
                // closed pipe the way every shell expects.
                sigfillset(sigs);
                posix_spawnattr_setsigdefault(attr, sigs);
                sigemptyset(sigs);
                posix_spawnattr_setsigmask(attr, sigs);

                // Its own process group, which is what makes killing the TREE a single
                // signal to -pgid rather than a walk of a table that races with anything
                // the child is still forking.
                int groupRc = posix_spawnattr_setpgroup(attr, 0);
                if (groupRc != 0)
                    return groupRc;

                short flags = SetPgroup | SetSigDef | SetSigMask;
                if (OperatingSystem.IsMacOS())
                    flags |= CloExecDefault;
                int flagsRc = posix_spawnattr_setflags(attr, flags);
                if (flagsRc != 0)
                    return flagsRc;

                IntPtr path = Marshal.StringToCoTaskMemUTF8(fileName);
                owned.Add(path);

                var argv = new List<string> { fileName };
                argv.AddRange(arguments);
                IntPtr argvBlock = AllocStringArray(argv, owned);

                var envp = new List<string>(environment.Count);
                foreach (KeyValuePair<string, string> pair in environment)
                    envp.Add(pair.Key + "=" + pair.Value);
                IntPtr envpBlock = AllocStringArray(envp, owned);

                // posix_spawnp, not posix_spawn: a bare program name has to resolve against
                // PATH the way ProcessStartInfo.FileName did, and the search happens here
                // in the parent so a name that does not resolve comes back as ENOENT rather
                // than as a child that exits 127 with nothing to say.
                return posix_spawnp(out pid, path, fa, attr, argvBlock, envpBlock);
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                return -1;
            }
            finally
            {
                if (faReady) posix_spawn_file_actions_destroy(fa);
                if (attrReady) posix_spawnattr_destroy(attr);
                foreach (IntPtr p in owned)
                    Marshal.FreeCoTaskMem(p);
            }
        }

        private static IntPtr AllocStringArray(IReadOnlyList<string> values, List<IntPtr> owned)
        {
            IntPtr block = Marshal.AllocCoTaskMem(IntPtr.Size * (values.Count + 1));
            owned.Add(block);
            for (int i = 0; i < values.Count; i++)
            {
                IntPtr item = Marshal.StringToCoTaskMemUTF8(values[i]);
                owned.Add(item);
                Marshal.WriteIntPtr(block, i * IntPtr.Size, item);
            }
            Marshal.WriteIntPtr(block, values.Count * IntPtr.Size, IntPtr.Zero);
            return block;
        }

        private static IntPtr Zeroed(int size)
        {
            IntPtr block = Marshal.AllocCoTaskMem(size);
            for (int i = 0; i < size; i++)
                Marshal.WriteByte(block, i, 0);
            return block;
        }

        /// <summary>
        /// Turn a <c>waitpid</c> status into an exit code, using the convention
        /// <see cref="System.Diagnostics.Process"/> uses on Unix so that a result the model
        /// reads does not change meaning with the mechanism that produced it.
        /// </summary>
        internal static int ExitCodeOf(int status) =>
            (status & 0x7f) == 0 ? (status >> 8) & 0xff : 128 + (status & 0x7f);
    }
}
