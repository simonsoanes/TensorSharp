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
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// Starts a child process, and does not let a fork that never reaches <c>exec</c>
    /// hang the host forever.
    ///
    /// <para>
    /// NOT the primary path any more. <see cref="SpawnedProcess"/> uses
    /// <see cref="PosixSpawn"/> wherever the platform offers it, which removes the hazard
    /// below rather than surviving it — there is no fork, so there is no window to lose.
    /// This remains for the two cases that cannot take that route: Windows, where
    /// <c>CreateProcess</c> is already a single operation and the hazard does not exist,
    /// and any Unix whose libc turns out not to offer the spawn calls. On those, a fork is
    /// still what happens, and this is what makes it survivable.
    /// </para>
    ///
    /// <para>
    /// THE BUG THIS EXISTS FOR. On Unix, .NET starts a child with <c>fork()</c> followed
    /// by <c>execve()</c> — <c>SystemNative_ForkAndExecProcess</c> — and the parent then
    /// blocks in <c>read()</c> on a close-on-exec pipe so it can learn whether the exec
    /// failed. <c>fork()</c> in a multi-threaded process is only safe if the child touches
    /// nothing but async-signal-safe functions, and the child does not get to decide that:
    /// libmalloc registers a <c>pthread_atfork</c> child handler that runs FIRST. On macOS
    /// 26's xzone allocator, if any other thread happened to hold a zone lock at the
    /// instant of the fork, that handler (<c>_malloc_fork_child</c> →
    /// <c>_xzm_foreach_lock.cold.1</c>) spins forever. The child never execs, never exits
    /// and never closes the pipe, so the parent's <c>read()</c> never returns — and it is
    /// inside a P/Invoke, so no CancellationToken, deadline or <c>Process.Kill</c> can
    /// reach it. The calling thread is gone for the life of the process.
    /// </para>
    /// <para>
    /// This host makes that a near-certainty rather than a curiosity. It runs 60+ threads
    /// — eighteen server GC threads, eighteen background GC threads, the inference engine
    /// allocating continuously — and it forks on every shell command, every package
    /// install and every post-edit syntax check. One observed wedge: an
    /// <c>edit_file build_deck.js</c> whose syntax check forked into that handler and
    /// burned a core for twelve minutes with the tool call still showing as running.
    /// </para>
    /// <para>
    /// WHAT THIS DOES. It cannot make <c>fork()</c> safe — that is libmalloc's to fix — so
    /// it makes the wedge recoverable instead. <c>Process.Start()</c> runs on its own
    /// thread; if it has not returned promptly, our children are scanned for one that is
    /// still running THIS executable. A child that has exec'd reports the path of the
    /// program it became, so an image path equal to our own is positive proof of a fork
    /// caught before exec. SIGKILL closes its end of the pipe, the parent's <c>read()</c>
    /// returns, and the start is retried. Recovery is a few seconds and the caller sees
    /// nothing, because the wedge is a race that a retry does not lose twice.
    /// </para>
    /// <para>
    /// Everything here is macOS-only and no-ops elsewhere: that is where the hazard is
    /// proven and where <c>libproc</c> supplies the two facts the scan needs.
    /// </para>
    /// </summary>
    public static class ForkWatchdog
    {
        /// <summary>
        /// How long <c>Process.Start()</c> may run before the children are scanned.
        ///
        /// <para>
        /// Not a deadline: a scan that finds nothing costs two syscalls per child and the
        /// wait simply continues, so this can sit far below the slowest legitimate start
        /// without risking one. What it does bound is how long a real wedge spins a core.
        /// </para>
        /// </summary>
        private static readonly TimeSpan ScanInterval = TimeSpan.FromSeconds(2);

        /// <summary>
        /// How long a suspect must go on running this executable before it is killed.
        ///
        /// <para>
        /// The window between <c>fork()</c> and <c>execve()</c> is where every healthy
        /// child also looks like a wedged one. That window is microseconds — dup2, close,
        /// setsid — so re-reading the image path after this delay separates the two with
        /// about three orders of magnitude to spare, and a descheduled-but-healthy child
        /// is never killed on the strength of one sample.
        /// </para>
        /// </summary>
        private static readonly TimeSpan ExecGrace = TimeSpan.FromMilliseconds(500);

        /// <summary>Total time to wait for one start before reporting failure.</summary>
        private static readonly TimeSpan StartCap = TimeSpan.FromSeconds(60);

        /// <summary>
        /// Starts attempted per call. A wedge needs another thread to be inside malloc at
        /// the instant of the fork; losing that race three times running is not a race.
        /// </summary>
        private const int MaxAttempts = 3;

        private const uint ProcPpidOnly = 6;
        private const int ProcPidPathInfoMaxSize = 4096;
        private const int Sigkill = 9;

        /// <summary><c>PROC_PIDTASKINFO</c>, and the layout of the <c>proc_taskinfo</c> it fills.</summary>
        private const int ProcPidTaskInfo = 4;
        private const int ProcTaskInfoSize = 96;
        private const int ProcTaskInfoThreadNumOffset = 84;

        [DllImport("libproc", SetLastError = true)]
        private static extern int proc_listpids(uint type, uint typeinfo, IntPtr buffer, int buffersize);

        [DllImport("libproc", SetLastError = true)]
        private static extern int proc_pidpath(int pid, IntPtr buffer, uint buffersize);

        [DllImport("libproc", SetLastError = true)]
        private static extern int proc_pidinfo(int pid, int flavor, ulong arg, IntPtr buffer, int buffersize);

        [DllImport("libc", SetLastError = true, EntryPoint = "kill")]
        private static extern int sys_kill(int pid, int sig);

        /// <summary>
        /// Our own image path, taken from libproc rather than from
        /// <see cref="Environment.ProcessPath"/> so that both sides of the comparison come
        /// out of the same kernel field and cannot differ over a symlink or a relative
        /// argv[0]. Empty where it cannot be read, which disables the scan rather than
        /// letting it match everything.
        /// </summary>
        private static readonly Lazy<string> SelfImage = new(
            () => OperatingSystem.IsMacOS() ? PathOf(Environment.ProcessId) : string.Empty,
            LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>
        /// Where reaped forks are reported, wired once by the host.
        ///
        /// <para>
        /// A retry that succeeds leaves no other trace, and "the syntax check silently
        /// took four seconds" is exactly the kind of thing that goes unnoticed until it is
        /// happening constantly. The callers that can hit this — a syntax check, a
        /// violation monitor — hold no logger of their own, so the sink is static and set
        /// by whoever does.
        /// </para>
        /// </summary>
        public static Action<string>? Observer { get; set; }

        /// <summary>
        /// Start a process, retrying if the host's own fork wedges before exec.
        /// </summary>
        /// <param name="factory">
        /// Builds the process, wired with whatever handlers the caller needs. Called again
        /// per attempt: a <see cref="Process"/> that has been started cannot be started a
        /// second time, so a retry needs a fresh one.
        /// </param>
        /// <param name="started">The running process, or null when none could be started.</param>
        /// <param name="error">Why it could not be started; empty on success.</param>
        /// <param name="onWedged">
        /// Called with a description each time a wedged fork is reaped. This is a host
        /// defect working as designed and the operator should be able to see it happening,
        /// which a silently successful retry would hide.
        /// </param>
        public static bool TryStart(
            Func<Process> factory, out Process? started, out string error,
            Action<string>? onWedged = null)
        {
            ArgumentNullException.ThrowIfNull(factory);

            started = null;
            error = string.Empty;

            for (int attempt = 1; attempt <= MaxAttempts; attempt++)
            {
                Process process = factory();
                if (TryStartOnce(process, out bool wedged, out error, onWedged))
                {
                    started = process;
                    return true;
                }

                // Disposal is TryStartOnce's, not ours: it is the only one that knows
                // whether Process.Start() ever returned, and disposing a Process out from
                // under a thread still inside its own start is a race this must not add on
                // top of the one it is recovering from.

                // A start that failed for any reason OTHER than our own fork wedging will
                // fail again identically — a missing interpreter does not become present.
                if (!wedged)
                    return false;
            }

            error = error.Length > 0
                ? error
                : $"the process could not be started after {MaxAttempts} attempts (the host's fork "
                  + "did not reach exec)";
            return false;
        }

        /// <summary>
        /// Start a process that has already been built, without retrying.
        ///
        /// <para>
        /// For the callers whose process handle is captured by something built around it
        /// before it starts, so there is nothing a second attempt could rebuild. They
        /// still get the part that matters: the call returns instead of never returning.
        /// </para>
        /// </summary>
        public static bool TryStart(Process process, out string error, Action<string>? onWedged = null)
        {
            ArgumentNullException.ThrowIfNull(process);
            return TryStartOnce(process, out _, out error, onWedged);
        }

        private static bool TryStartOnce(
            Process process, out bool wedged, out string error, Action<string>? onWedged)
        {
            wedged = false;
            error = string.Empty;

            if (!OperatingSystem.IsMacOS())
                return StartDirect(process, out error);

            HashSet<int> before = ChildPids();

            Exception? failure = null;
            bool ok = false;

            // A TaskCompletionSource rather than an event that has to be disposed. The
            // whole premise here is that the starting thread may never come back, and a
            // disposable signal would then be set by a thread running after this frame
            // released it — an ObjectDisposedException on a background thread, which is a
            // torn-down process. The one thing this guard must never do is turn a hang
            // into a crash.
            var done = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

            // A background thread: if the fork wedges in a way the scan cannot identify,
            // this thread is unrecoverable — it is blocked inside a P/Invoke — and a
            // foreground thread in that state would keep the process alive at shutdown.
            var thread = new Thread(() =>
            {
                try { ok = process.Start(); }
                catch (Exception ex) { failure = ex; }
                finally { done.TrySetResult(true); }
            })
            {
                IsBackground = true,
                Name = "AgentHost process start",
            };
            thread.Start();

            var waited = TimeSpan.Zero;
            var killed = new List<int>();
            bool completed;
            while (!(completed = done.Task.Wait(ScanInterval)))
            {
                waited += ScanInterval;
                killed.AddRange(KillWedgedForks(before, onWedged));
                if (waited >= StartCap)
                    break;
            }

            if (!completed && !done.Task.Wait(ExecGrace))
            {
                // Start() is still inside the kernel and nothing identifiable was there to
                // release it. Report instead of blocking: the caller gets an error it can
                // put in front of the model, which is strictly better than a tool call that
                // never returns.
                //
                // The Process is deliberately NOT disposed. A thread is still inside its
                // Start(), and taking its handles away is how a hang becomes a crash. It
                // is leaked knowingly, and it is one object against a wedge that has
                // already cost a core.
                error = "the process could not be started (the host's fork did not reach exec)";
                return false;
            }

            // Past here the starting thread is done, so the Process is ours to release.
            if (failure != null)
            {
                error = failure.Message;
                Release(process);
                return false;
            }

            if (!ok)
            {
                error = "the process could not be started";
                Release(process);
                return false;
            }

            // Whether the reaped fork was OURS is decided by pid, not by inference. A
            // concurrent caller's wedge is worth killing on sight — it is provably dead
            // weight — but it says nothing about this start, which succeeded and must be
            // handed back rather than retried.
            int pid = TryPid(process);
            if (pid > 0 && killed.Contains(pid))
            {
                wedged = true;
                error = "the host's fork did not reach exec";
                Release(process);
                return false;
            }

            return true;
        }

        private static void Release(Process process)
        {
            try { process.Dispose(); }
            catch (Exception) { /* nothing left to release */ }
        }

        private static bool StartDirect(Process process, out string error)
        {
            error = string.Empty;
            try
            {
                if (process.Start())
                    return true;
                error = "the process could not be started";
            }
            catch (Exception ex) when (
                ex is System.ComponentModel.Win32Exception or InvalidOperationException
                   or System.IO.IOException or PlatformNotSupportedException)
            {
                error = ex.Message;
            }

            Release(process);
            return false;
        }

        /// <summary>
        /// Kill every child that appeared during this start and is still running THIS
        /// executable, and return the pids actually killed.
        /// </summary>
        private static List<int> KillWedgedForks(HashSet<int> before, Action<string>? onWedged)
        {
            var killed = new List<int>();

            string self = SelfImage.Value;
            if (self.Length == 0)
                return killed;

            var suspects = new List<int>();
            foreach (int pid in ChildPids())
            {
                // Predates this start, so whatever it is, it is not what this call forked.
                if (before.Contains(pid))
                    continue;
                if (!string.Equals(PathOf(pid), self, StringComparison.Ordinal))
                    continue;
                suspects.Add(pid);
            }

            if (suspects.Count == 0)
                return killed;

            Thread.Sleep(ExecGrace);

            foreach (int pid in suspects)
            {
                // Re-read: a child that exec'd during the grace is healthy and is now
                // reporting the program it became.
                if (!string.Equals(PathOf(pid), self, StringComparison.Ordinal))
                    continue;

                // The second half of the proof, and the one that makes an image path equal
                // to our own safe to act on. fork() carries exactly the calling thread, so
                // a child that has not exec'd has one; anything that legitimately IS this
                // executable has a CLR under it and is past a dozen within microseconds of
                // starting, let alone after the grace above. Unreadable is not evidence of
                // innocence — a pid whose task info cannot be fetched is judged on the
                // image path alone, which is what this check exists to strengthen rather
                // than replace.
                if (ThreadCountOf(pid) > 1)
                    continue;

                if (sys_kill(pid, Sigkill) != 0)
                    continue;

                killed.Add(pid);
                string detail =
                    $"reaped pid {pid.ToString(System.Globalization.CultureInfo.InvariantCulture)}, "
                    + "a fork of this host that never reached exec (macOS libmalloc atfork hazard)";
                Report(onWedged, detail);
                Report(Observer, detail);
            }

            return killed;
        }

        /// <summary>The pids of our direct children, or empty where they cannot be read.</summary>
        internal static HashSet<int> ChildPids()
        {
            var result = new HashSet<int>();
            if (!OperatingSystem.IsMacOS())
                return result;

            int capacity = 256;
            for (int attempt = 0; attempt < 4; attempt++)
            {
                int bytes = capacity * sizeof(int);
                IntPtr buffer = Marshal.AllocHGlobal(bytes);
                try
                {
                    int written = proc_listpids(ProcPpidOnly, (uint)Environment.ProcessId, buffer, bytes);
                    if (written <= 0)
                        return result;

                    int count = written / sizeof(int);
                    // A full buffer means the answer was truncated, and a truncated answer
                    // could omit the very child being hunted.
                    if (count >= capacity)
                    {
                        capacity *= 4;
                        continue;
                    }

                    for (int i = 0; i < count; i++)
                    {
                        int pid = Marshal.ReadInt32(buffer, i * sizeof(int));
                        if (pid > 0)
                            result.Add(pid);
                    }
                    return result;
                }
                catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
                {
                    return result;
                }
                finally
                {
                    Marshal.FreeHGlobal(buffer);
                }
            }

            return result;
        }

        /// <summary>The image a pid is running, or empty when it cannot be read.</summary>
        internal static string PathOf(int pid)
        {
            IntPtr buffer = Marshal.AllocHGlobal(ProcPidPathInfoMaxSize);
            try
            {
                int length = proc_pidpath(pid, buffer, (uint)ProcPidPathInfoMaxSize);
                return length > 0 ? Marshal.PtrToStringUTF8(buffer, length) ?? string.Empty : string.Empty;
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                return string.Empty;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>
        /// How many threads a pid is running, or -1 when that cannot be read.
        /// </summary>
        internal static int ThreadCountOf(int pid)
        {
            IntPtr buffer = Marshal.AllocHGlobal(ProcTaskInfoSize);
            try
            {
                int written = proc_pidinfo(pid, ProcPidTaskInfo, 0, buffer, ProcTaskInfoSize);
                if (written < ProcTaskInfoSize)
                    return -1;
                return Marshal.ReadInt32(buffer, ProcTaskInfoThreadNumOffset);
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                return -1;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>A diagnostic sink must never be able to fail the recovery it describes.</summary>
        private static void Report(Action<string>? sink, string detail)
        {
            if (sink == null) return;
            try { sink(detail); }
            catch (Exception) { /* best-effort observability */ }
        }

        private static int TryPid(Process process)
        {
            try { return process.Id; }
            catch (Exception) { return -1; }
        }
    }
}
