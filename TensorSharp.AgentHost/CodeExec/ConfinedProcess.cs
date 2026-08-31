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
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>What one confined process did.</summary>
    /// <param name="Started">False when the process could not be launched at all.</param>
    /// <param name="TimedOut">True when it was killed for exceeding its deadline.</param>
    /// <param name="ExitCode">Its exit code, meaningless unless <paramref name="Started"/> and not <paramref name="TimedOut"/>.</param>
    /// <param name="Stdout">Captured standard output, truncated to the caller's cap.</param>
    /// <param name="Stderr">Captured standard error, truncated to the caller's cap.</param>
    /// <param name="Elapsed">Wall time.</param>
    /// <param name="SandboxName">Which sandbox wrapped it, or "none".</param>
    /// <param name="Error">Why it did not start or could not be confined.</param>
    public readonly record struct ConfinedResult(
        bool Started,
        bool TimedOut,
        int ExitCode,
        string Stdout,
        string Stderr,
        TimeSpan Elapsed,
        string SandboxName,
        string? Error)
    {
        /// <summary>True when it ran to completion and reported success.</summary>
        public bool Ok => Started && !TimedOut && ExitCode == 0 && Error == null;
    }

    /// <summary>What to launch, and what it may touch.</summary>
    public sealed class ConfinedLaunch
    {
        /// <summary>The executable.</summary>
        public required string Interpreter { get; init; }

        /// <summary>Its full argument vector. Never a command line — no shell is involved.</summary>
        public required IReadOnlyList<string> Arguments { get; init; }

        /// <summary>The only directory the process may write to, and its working directory unless <see cref="WorkingDirectory"/> says otherwise.</summary>
        public required string WriteDirectory { get; init; }

        /// <summary>
        /// Where the process starts, when that is not the root of what it may write.
        ///
        /// <para>
        /// The shell needs these separated. Its writable region is the whole session
        /// workspace — the work directory it edits in, the environment packages install
        /// into, the state the wrapper saves — while the directory it starts in is
        /// wherever the model last <c>cd</c>'d to. Collapsing the two would mean either a
        /// shell that cannot install anything or one whose <c>ls</c> shows the host's
        /// bookkeeping.
        /// </para>
        /// </summary>
        public string? WorkingDirectory { get; init; }

        /// <summary>A directory it may read but never write. Must exist.</summary>
        public required string ReadOnlyDirectory { get; init; }

        /// <summary>Anything else it may read.</summary>
        public IReadOnlyList<string> ReadablePaths { get; init; } = Array.Empty<string>();

        /// <summary>Whether it may open a socket.</summary>
        public bool AllowNetwork { get; init; }

        /// <summary>
        /// With <see cref="AllowNetwork"/> false: the one loopback TCP port the
        /// process may still reach — the host-side egress proxy an installer is
        /// pointed at. Null for the normal fully-closed run.
        /// </summary>
        public int? AllowLoopbackPort { get; init; }

        /// <summary>How long before it is killed.</summary>
        public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

        /// <summary>Cap on each of stdout and stderr.</summary>
        public int MaxOutputBytes { get; init; } = 32 * 1024;

        /// <summary>Variables to give the child, on top of the minimal set.</summary>
        public IReadOnlyDictionary<string, string> EnvironmentVariables { get; init; } =
            new Dictionary<string, string>(StringComparer.Ordinal);

        /// <summary>
        /// Called once per line the process writes to stdout or stderr, WHILE it runs.
        /// The captured result is unchanged — this is a live tap for a host that wants
        /// to show progress (a pip install's "Collecting ...", a script's own prints)
        /// instead of silence until exit. Invoked on the process's async reader threads;
        /// the callback must be quick and thread-safe. Null taps nothing.
        /// </summary>
        public Action<string>? OnOutputLine { get; init; }
    }

    /// <summary>
    /// Launches one process inside a sandbox and captures what it did.
    ///
    /// <para>
    /// The parts that matter here are the parts that are easy to get subtly wrong
    /// twice: no shell, a scrubbed environment, stdin closed, a deadline that actually
    /// kills, output bounded before it can exhaust memory, and a sandbox that is either
    /// applied or the run is abandoned. Sharing the mechanism is the point; the POLICY —
    /// what may be reached, for how long, whether the network is open — stays with each
    /// caller, because those differ and conflating them is how a switch meant for one
    /// feature silently widens the other.
    /// </para>
    /// <para>
    /// It does NOT yet serve every confined launch in this assembly, and the comment here
    /// used to claim it did. <c>SkillScriptRunner.RunConfined</c> is a second copy of the
    /// same sequence, written before this was extracted; it differs only in which
    /// environment variables it passes through and in being able to report a degraded
    /// sandbox to its caller. Two copies of a security-critical launch is exactly the
    /// shape that drifts until one of them misses a hardening — the shell tool, package
    /// installs and background jobs all come through HERE.
    /// </para>
    /// </summary>
    public static class ConfinedProcess
    {
        /// <summary>Run <paramref name="launch"/> under <paramref name="sandbox"/> and wait for it.</summary>
        /// <param name="sandbox">The sandbox, or null to run unconfined.</param>
        /// <param name="mode">Whether an unusable sandbox is fatal.</param>
        public static ConfinedResult Run(
            ConfinedLaunch launch, ISkillSandbox? sandbox, SkillSandboxMode mode)
        {
            if (!TryStart(launch, sandbox, mode, out ConfinedJob? job, out ConfinedResult failure))
                return failure;
            using (job)
                return job!.WaitForExit(launch.Timeout);
        }

        /// <summary>
        /// Start <paramref name="launch"/> and hand back the running process, WITHOUT
        /// waiting for it.
        ///
        /// <para>
        /// Split out of <see cref="Run"/> for background jobs, which are the one case
        /// where the process has to outlive the call that started it. Doing it here — the
        /// host holding a live handle — rather than by backgrounding inside the shell is
        /// what makes the feature work on every platform: bubblewrap is launched with
        /// <c>--die-with-parent</c>, so a job the shell put in its own background dies the
        /// instant the call returns, while a process this host still owns does not. It is
        /// also the only version where something can still kill the job when the session
        /// ends.
        /// </para>
        /// </summary>
        public static bool TryStart(
            ConfinedLaunch launch, ISkillSandbox? sandbox, SkillSandboxMode mode,
            out ConfinedJob? job, out ConfinedResult failure)
        {
            ArgumentNullException.ThrowIfNull(launch);

            job = null;
            failure = default;

            string fileName = launch.Interpreter;
            IReadOnlyList<string> argv = launch.Arguments;
            IDisposable? cleanup = null;
            string sandboxName = "none";
            string? attachFailure = null;

            if (sandbox != null)
            {
                var request = new SkillSandboxRequest(
                    launch.Interpreter,
                    launch.Arguments,
                    launch.ReadOnlyDirectory,
                    launch.WriteDirectory,
                    launch.AllowNetwork,
                    launch.ReadablePaths)
                {
                    AllowLoopbackPort = launch.AllowLoopbackPort,
                };

                if (sandbox.TryWrap(request, out string wrappedFile, out IReadOnlyList<string> wrappedArgs,
                        out IDisposable wrappedCleanup, out string wrapError))
                {
                    fileName = wrappedFile;
                    argv = wrappedArgs;
                    cleanup = wrappedCleanup;
                    sandboxName = sandbox.Name;
                }
                else if (mode == SkillSandboxMode.Required)
                {
                    failure = Failed($"the sandbox could not be prepared ({wrapError})");
                    return false;
                }
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = fileName,
                WorkingDirectory = launch.WorkingDirectory ?? launch.WriteDirectory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                RedirectStandardInput = true,
                // No shell is interposed by US: the arguments cross as a vector, so a
                // value containing ; | > $ ` is data rather than syntax. The shell tool
                // hands its interpreter a SCRIPT FILE for the same reason — the model's
                // command never becomes part of a command line anyone else has to quote.
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            foreach (string argument in argv)
                startInfo.ArgumentList.Add(argument);

            ApplyEnvironment(startInfo, launch);

            var stdout = new BoundedText(launch.MaxOutputBytes);
            var stderr = new BoundedText(launch.MaxOutputBytes);
            var sw = Stopwatch.StartNew();

            // Only a run that is actually confined has violations to observe, and only
            // macOS surfaces them. The monitor is process-wide and already running, so
            // this costs a property read; what identifies THIS run's denials is the mark.
            SandboxViolationMonitor? violations =
                sandboxName != "none" ? SandboxViolationMonitor.Shared : null;
            DateTime violationsFrom = SandboxViolationMonitor.Mark();

            Process? process = null;
            try
            {
                process = new Process { StartInfo = startInfo };
                process.OutputDataReceived += (_, e) =>
                {
                    if (e.Data == null) return;
                    stdout.AppendLine(e.Data);
                    Tap(launch.OnOutputLine, e.Data);
                };
                process.ErrorDataReceived += (_, e) =>
                {
                    if (e.Data == null) return;
                    stderr.AppendLine(e.Data);
                    Tap(launch.OnOutputLine, e.Data);
                };

                if (!process.Start())
                {
                    failure = Failed($"'{fileName}' could not be started");
                    Cleanup(process, cleanup);
                    return false;
                }

                // A job-object sandbox attaches after the process exists. If it cannot,
                // the child is killed rather than left running outside its confinement.
                if (sandbox != null && !sandbox.TryAttach(process, out string attachError))
                {
                    // Two outcomes, and neither is the one this used to have. It fell
                    // through on anything but Required having already KILLED the child, so
                    // the model got a kill exit code, empty output and no reason at all.
                    //
                    // Required: a failure, plainly. Confinement was demanded and did not
                    // happen, so nothing runs.
                    if (mode == SkillSandboxMode.Required)
                    {
                        TryKill(process);
                        failure = Failed($"the sandbox could not be applied to the process ({attachError})");
                        Cleanup(process, cleanup);
                        return false;
                    }

                    // Preferred: DEGRADE, and say so. This branch is reached only where an
                    // operator has already accepted running unconfined — it is Windows-only
                    // (only WindowsJobObjectSandbox overrides TryAttach; the interface
                    // default returns true) and on Windows CanRun requires
                    // --code-exec-unconfined, because the job object does not confine the
                    // network. AssignProcessToJobObject fails when the parent is itself
                    // already in a job without breakaway — a CI agent, a Windows container
                    // — and refusing every command there would take away a capability the
                    // operator explicitly asked for, over a CPU/memory bound.
                    //
                    // The honest part is the name: "none" is what makes Describe print the
                    // full "Not confined on this host" gap list, so the run continues and
                    // the model is told exactly what did not apply to it.
                    sandboxName = "none";
                    attachFailure = attachError;
                }

                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                // Nothing will be typed at it, and a program that reads stdin would
                // otherwise block until the deadline rather than failing immediately.
                try { process.StandardInput.Close(); } catch (IOException) { /* already gone */ }

                job = new ConfinedJob(
                    process, cleanup, violations, violationsFrom, stdout, stderr, sw, sandboxName, launch)
                {
                    AttachFailure = attachFailure,
                };
                return true;
            }
            catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException or IOException)
            {
                failure = Failed(ex.Message);
                Cleanup(process, cleanup);
                return false;
            }
        }

        private static void Cleanup(Process? process, IDisposable? cleanup)
        {
            process?.Dispose();
            cleanup?.Dispose();
        }

        internal static ConfinedResult Failed(string message) =>
            new(false, false, -1, string.Empty, string.Empty, TimeSpan.Zero, "none", message);

        /// <summary>
        /// Append the sandbox denials observed during a FAILED run to its stderr, so
        /// the model (and the operator reading the same result) sees the kernel's
        /// actual refusal instead of whatever the script made of it. Advisory: the
        /// violation stream is system-wide and filtered heuristically, and the text
        /// says so.
        /// </summary>
        internal static string WithDenials(
            string stderr, SandboxViolationMonitor? violations, DateTime since, ConfinedLaunch launch)
        {
            if (violations == null)
                return stderr;
            // waitForTail: this is only ever called for a FAILED run, which is the one
            // case that reads the denials and so the only one that should pay for them.
            IReadOnlyList<string> denials = violations.DenialsSince(
                since, launch.Interpreter, launch.WriteDirectory, waitForTail: true);
            if (denials.Count == 0)
                return stderr;

            var sb = new StringBuilder(stderr);
            if (sb.Length > 0 && sb[^1] != '\n')
                sb.Append('\n');
            sb.Append("[sandbox denials observed during this run (may include unrelated processes):]\n");
            foreach (string denial in denials)
                sb.Append("  ").Append(denial).Append('\n');
            return sb.ToString();
        }

        /// <summary>
        /// Give the child a minimal environment.
        ///
        /// <para>
        /// The host's environment is where credentials live — <c>AWS_SECRET_ACCESS_KEY</c>,
        /// <c>OPENAI_API_KEY</c>, a database URL. Inheriting it would hand every one of
        /// them to the process, and no sandbox can undo that: the values are already in
        /// the image by then. So the child starts from nothing and gets back only what an
        /// interpreter needs.
        /// </para>
        /// </summary>
        private static void ApplyEnvironment(ProcessStartInfo startInfo, ConfinedLaunch launch)
        {
            startInfo.Environment.Clear();

            startInfo.Environment["PATH"] = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            startInfo.Environment["TMPDIR"] = launch.WriteDirectory;
            startInfo.Environment["TEMP"] = launch.WriteDirectory;
            startInfo.Environment["TMP"] = launch.WriteDirectory;
            startInfo.Environment["HOME"] = launch.WriteDirectory;
            startInfo.Environment["USERPROFILE"] = launch.WriteDirectory;
            startInfo.Environment["LANG"] = "C.UTF-8";

            if (OperatingSystem.IsWindows())
            {
                // Windows will not start a process without these; they name the OS
                // install, not the user, so they carry nothing sensitive.
                foreach (string name in new[] { "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "WINDIR" })
                {
                    if (Environment.GetEnvironmentVariable(name) is { } value)
                        startInfo.Environment[name] = value;
                }
            }

            foreach (KeyValuePair<string, string> pair in launch.EnvironmentVariables)
                startInfo.Environment[pair.Key] = pair.Value;
        }

        /// <summary>A live-output tap must never be able to kill the reader thread.</summary>
        private static void Tap(Action<string>? tap, string line)
        {
            if (tap == null) return;
            try { tap(line); }
            catch (Exception) { /* the tap is best-effort observability */ }
        }

        internal static void TryKill(Process process)
        {
            try { process.Kill(entireProcessTree: true); }
            catch (Exception) { /* already gone, or we cannot reach the tree */ }
        }

        /// <summary>
        /// Accumulates output up to a cap, keeping the HEAD and the TAIL and dropping the
        /// middle.
        ///
        /// <para>
        /// Head-only truncation — which this used to do — reliably throws away the part
        /// that was wanted. The head of a build or a test run is the command echoing and
        /// the first hundred files compiling; the failure is at the end. Keeping both ends
        /// costs nothing and means a truncated result still answers "did it work".
        /// </para>
        /// </summary>
        internal sealed class BoundedText
        {
            private readonly StringBuilder _head = new();
            private readonly Queue<string> _tail = new();
            private readonly object _gate = new();
            private readonly int _limit;
            private int _tailBytes;
            private long _droppedBytes;
            private long _droppedLines;

            public BoundedText(int limit) => _limit = Math.Max(2048, limit);

            private int Half => _limit / 2;

            public void AppendLine(string line)
            {
                lock (_gate)
                {
                    // A single line longer than the whole budget is not a line anyone will
                    // read, and it used to be kept in full because the drain loop only ever
                    // removes WHOLE lines and refuses to empty the queue. `base64 -w0 x.bin`
                    // or a one-line JSON dump therefore returned megabytes through a 32 KB
                    // cap and into the model's context. Cut it here, where the size is known.
                    if (line.Length > Half)
                    {
                        _droppedBytes += line.Length - Half;
                        _droppedLines++;
                        line = line.Substring(0, Half / 2) + " …[line truncated]… "
                             + line.Substring(line.Length - Half / 4);
                    }

                    if (_head.Length + line.Length + 1 <= Half)
                    {
                        _head.Append(line).Append('\n');
                        return;
                    }

                    _tail.Enqueue(line);
                    _tailBytes += line.Length + 1;
                    while (_tailBytes > Half && _tail.Count > 1)
                    {
                        string dropped = _tail.Dequeue();
                        _tailBytes -= dropped.Length + 1;
                        _droppedBytes += dropped.Length + 1;
                        _droppedLines++;
                    }
                }
            }

            public string Text()
            {
                lock (_gate)
                {
                    if (_tail.Count == 0)
                        return _head.ToString();

                    var sb = new StringBuilder(_head.ToString());
                    if (_droppedLines > 0)
                    {
                        sb.Append("\n… ").Append(_droppedLines.ToString(CultureInfo.InvariantCulture))
                          .Append(" lines (").Append(_droppedBytes.ToString(CultureInfo.InvariantCulture))
                          .Append(" bytes) of output were dropped from the middle …\n\n");
                    }
                    foreach (string line in _tail)
                        sb.Append(line).Append('\n');
                    return sb.ToString();
                }
            }

            /// <summary>True when anything was dropped, so the result can say so once.</summary>
            public bool Truncated
            {
                get { lock (_gate) return _droppedLines > 0; }
            }
        }
    }

    /// <summary>
    /// A confined process that has been started and not yet waited for.
    ///
    /// <para>
    /// Disposing it kills the process tree and releases the sandbox's temporary profile.
    /// That is the behaviour a background job needs at session end, and it is why the
    /// session workspace takes ownership of one rather than the call that started it.
    /// </para>
    /// </summary>
    public sealed class ConfinedJob : IDisposable
    {
        /// <summary>
        /// Why the sandbox could not be attached to this process, when it could not and the
        /// run was allowed to continue anyway. Null on every ordinary run.
        ///
        /// <para>
        /// Kept so the reason reaches the model rather than being discarded: a run that
        /// degraded to no confinement should say what failed, not merely that nothing was
        /// confined.
        /// </para>
        /// </summary>
        public string? AttachFailure { get; init; }

        private readonly Process _process;
        private readonly IDisposable? _cleanup;
        private readonly SandboxViolationMonitor? _violations;
        private readonly DateTime _violationsFrom;
        private readonly ConfinedProcess.BoundedText _stdout;
        private readonly ConfinedProcess.BoundedText _stderr;
        private readonly Stopwatch _stopwatch;
        private readonly string _sandboxName;
        private readonly ConfinedLaunch _launch;
        private int _disposed;

        internal ConfinedJob(
            Process process, IDisposable? cleanup, SandboxViolationMonitor? violations,
            DateTime violationsFrom,
            ConfinedProcess.BoundedText stdout, ConfinedProcess.BoundedText stderr,
            Stopwatch stopwatch, string sandboxName, ConfinedLaunch launch)
        {
            _process = process;
            _cleanup = cleanup;
            _violations = violations;
            _violationsFrom = violationsFrom;
            _stdout = stdout;
            _stderr = stderr;
            _stopwatch = stopwatch;
            _sandboxName = sandboxName;
            _launch = launch;
        }

        /// <summary>The operating system's id for the process, for a result the model can act on.</summary>
        public int ProcessId
        {
            get { try { return _process.Id; } catch (InvalidOperationException) { return -1; } }
        }

        /// <summary>Which sandbox wrapped it, or "none".</summary>
        public string SandboxName => _sandboxName;

        /// <summary>Whether it has finished.</summary>
        public bool HasExited
        {
            get { try { return _process.HasExited; } catch (InvalidOperationException) { return true; } }
        }

        /// <summary>
        /// How long to wait for the output pipes to drain AFTER the process itself has
        /// exited. Bounded because a pipe held open by a grandchild never reaches EOF, and
        /// an unbounded wait there is a hung tool call rather than a slow one.
        /// </summary>
        private const int DrainMilliseconds = 2000;

        /// <summary>Wait up to <paramref name="timeout"/>, killing the tree if it runs over.</summary>
        public ConfinedResult WaitForExit(TimeSpan timeout)
        {
            try
            {
                if (!_process.WaitForExit((int)timeout.TotalMilliseconds))
                {
                    ConfinedProcess.TryKill(_process);
                    _stopwatch.Stop();
                    // The output so far comes back WITH the timeout notice rather than
                    // being discarded: a timeout that returns nothing forces the model to
                    // re-run an expensive command blind, and the part it already printed is
                    // usually enough to tell what it was stuck on.
                    return new ConfinedResult(true, true, -1, _stdout.Text(),
                        ConfinedProcess.WithDenials(_stderr.Text(), _violations, _violationsFrom, _launch),
                        _stopwatch.Elapsed, _sandboxName, null);
                }

                // WaitForExit(int) can return before the async readers have drained, so the
                // drain has to be waited for — but BOUNDED, and that is not a detail.
                //
                // The parameterless overload waits for EOF on the redirected pipes, and a
                // pipe stays open while anything still holds the inherited handle. So a
                // command ending in `&` — `nohup python3 server.py > log 2>&1 &`, which a
                // model writes for a server because nothing tells it not to — left this
                // waiting on the GRANDCHILD, forever: the shell exited immediately, the
                // deadline above had already been satisfied, and the call never returned.
                // The loop went on emitting "running…" heartbeats, and the worker thread
                // stayed blocked here even after the client gave up.
                //
                // Verified at the OS level: `( sh -c 'sleep 8 & echo done' ) | cat` prints
                // immediately and the pipe reaches EOF eight seconds later.
                if (!_process.WaitForExit(DrainMilliseconds))
                {
                    // The tree is killed on Dispose regardless, so the only question was
                    // ever whether the call returns. Anything the shell itself printed has
                    // been captured; what is still holding the pipe is a background process
                    // the model was told, in the refusal for `&`, to start with
                    // run_in_background instead.
                    ConfinedProcess.TryKill(_process);
                }
                _stopwatch.Stop();

                string finalStderr = _process.ExitCode == 0
                    ? _stderr.Text()
                    : ConfinedProcess.WithDenials(_stderr.Text(), _violations, _violationsFrom, _launch);
                return new ConfinedResult(true, false, _process.ExitCode, _stdout.Text(), finalStderr,
                    _stopwatch.Elapsed, _sandboxName, null);
            }
            catch (Exception ex) when (ex is InvalidOperationException or IOException)
            {
                return ConfinedProcess.Failed(ex.Message);
            }
        }

        /// <summary>Kill it and release the sandbox's scratch. Safe to call twice.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            ConfinedProcess.TryKill(_process);
            try { _process.Dispose(); } catch (Exception) { }
            _cleanup?.Dispose();
            // The monitor is shared and outlives every run; a job never owns it.
        }
    }
}
