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
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>What to start, and where its output goes.</summary>
    public sealed class SpawnRequest
    {
        /// <summary>The executable. A bare name is resolved against this process's PATH.</summary>
        public required string FileName { get; init; }

        /// <summary>Its argument vector. Never a command line — no shell is involved.</summary>
        public required IReadOnlyList<string> Arguments { get; init; }

        /// <summary>Where the child starts. Applied by the kernel, not by changing ours.</summary>
        public string? WorkingDirectory { get; init; }

        /// <summary>
        /// The child's COMPLETE environment. Nothing is inherited: the host's environment
        /// is where credentials live, and no sandbox can take back a value that is already
        /// in the child's image.
        /// </summary>
        public required IReadOnlyDictionary<string, string> Environment { get; init; }

        /// <summary>Called once per line of standard output, without the line ending.</summary>
        public Action<string>? OnStdoutLine { get; init; }

        /// <summary>Called once per line of standard error, without the line ending.</summary>
        public Action<string>? OnStderrLine { get; init; }
    }

    /// <summary>
    /// A child process, started without forking wherever the platform allows it.
    ///
    /// <para>
    /// The reason this exists rather than <see cref="Process"/> is in
    /// <see cref="PosixSpawn"/>: .NET starts children with <c>fork()</c> + <c>execve()</c>,
    /// and a fork from a process with this many threads can wedge permanently inside
    /// libmalloc's atfork handler, taking the calling thread with it for the life of the
    /// server. <c>posix_spawn</c> has no such window, so on Unix that is what runs.
    /// </para>
    /// <para>
    /// Windows has no <c>posix_spawn</c> and no <c>fork()</c> either — its
    /// <c>CreateProcess</c> is already a single operation, so the hazard does not exist
    /// there — and it keeps using <see cref="Process"/>, as does any Unix whose libc turns
    /// out not to offer the calls. That path still goes through <see cref="ForkWatchdog"/>,
    /// which reaps a wedged fork rather than letting it hang. Prevention where it can be
    /// had, recovery everywhere else.
    /// </para>
    /// <para>
    /// The surface is deliberately the subset the callers in this assembly use, with one
    /// change: draining the output pipes is its OWN bounded wait rather than a property of
    /// waiting for exit. A pipe held open by a grandchild never reaches EOF, and the
    /// distinction between "the process ended" and "everything that inherited its stdout
    /// ended" is the difference between a tool call returning and a tool call hanging.
    /// </para>
    /// </summary>
    public sealed class SpawnedProcess : IDisposable
    {
        private readonly Process? _managed;

        private readonly int _pid;
        private readonly TaskCompletionSource<int> _exited =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly Thread? _stdoutReader;
        private readonly Thread? _stderrReader;
        private int _disposed;

        /// <summary>Which mechanism started it: "posix_spawn" or "fork".</summary>
        public string Mechanism { get; }

        private SpawnedProcess(int pid, Thread stdoutReader, Thread stderrReader)
        {
            _pid = pid;
            _stdoutReader = stdoutReader;
            _stderrReader = stderrReader;
            Mechanism = "posix_spawn";
        }

        private SpawnedProcess(Process managed)
        {
            _managed = managed;
            _pid = -1;
            Mechanism = "fork";
        }

        /// <summary>
        /// The encoding both paths agree on. Without a BOM: it describes a pipe, and a
        /// byte-order mark written into a child's stdin would be data the child did not
        /// ask for.
        /// </summary>
        private static readonly Encoding Utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

        /// <summary>Start <paramref name="request"/>, or explain why it could not start.</summary>
        public static bool TryStart(SpawnRequest request, out SpawnedProcess? process, out string error)
        {
            ArgumentNullException.ThrowIfNull(request);

            return PosixSpawn.IsSupported
                ? TryStartSpawned(request, out process, out error)
                : TryStartManaged(request, out process, out error);
        }

        // ---- the path that does not fork --------------------------------------

        private static bool TryStartSpawned(
            SpawnRequest request, out SpawnedProcess? process, out string error)
        {
            process = null;
            error = string.Empty;

            int inR = -1, inW = -1, outR = -1, outW = -1, errR = -1, errW = -1;
            try
            {
                if (!PosixSpawn.TryPipe(out inR, out inW)
                    || !PosixSpawn.TryPipe(out outR, out outW)
                    || !PosixSpawn.TryPipe(out errR, out errW))
                {
                    error = "the host ran out of file descriptors";
                    CloseAll(ref inR, ref inW, ref outR, ref outW, ref errR, ref errW);
                    return false;
                }

                int rc = PosixSpawn.Spawn(
                    request.FileName, request.Arguments, request.WorkingDirectory, request.Environment,
                    inR, outW, errW, out int pid);

                // The child's ends belong to the child. Holding them here would mean the
                // read ends never see EOF, because this process would still be a writer.
                Close(ref inR);
                Close(ref outW);
                Close(ref errW);

                if (rc != 0 || pid <= 0)
                {
                    error = DescribeErrno(rc, request.FileName);
                    CloseAll(ref inR, ref inW, ref outR, ref outW, ref errR, ref errW);
                    return false;
                }

                // Nothing will be typed at it, and a program that reads stdin would
                // otherwise block until its deadline rather than failing immediately.
                Close(ref inW);

                Thread stdout = Reader(outR, request.OnStdoutLine, "stdout");
                Thread stderr = Reader(errR, request.OnStderrLine, "stderr");
                outR = errR = -1;   // owned by the reader threads from here

                var spawned = new SpawnedProcess(pid, stdout, stderr);
                stdout.Start();
                stderr.Start();
                spawned.StartReaper();
                process = spawned;
                return true;
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException
                                          or OutOfMemoryException)
            {
                error = ex.Message;
                CloseAll(ref inR, ref inW, ref outR, ref outW, ref errR, ref errW);
                return false;
            }
        }

        private static Thread Reader(int fd, Action<string>? onLine, string name)
        {
            var thread = new Thread(() =>
            {
                try
                {
                    using var stream = new FileStream(
                        new SafeFileHandle((IntPtr)fd, ownsHandle: true), FileAccess.Read, bufferSize: 4096);
                    using var reader = new StreamReader(stream, new UTF8Encoding(false));
                    string? line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        if (onLine == null)
                            continue;
                        // A tap must never be able to kill the reader: a thread that dies
                        // here leaves the pipe unread and the child blocks on a full one.
                        try { onLine(line); }
                        catch (Exception) { /* best-effort observability */ }
                    }
                }
                catch (Exception) { /* the pipe went away; there is nothing left to read */ }
            })
            {
                IsBackground = true,
                Name = "AgentHost child " + name,
            };
            return thread;
        }

        private void StartReaper()
        {
            var thread = new Thread(() =>
            {
                int status = 0;
                int rc;
                do
                {
                    rc = PosixSpawn.waitpid(_pid, out status, 0);
                }
                while (rc < 0 && WasInterrupted());

                // A waitpid that failed leaves no status to read; -1 says "we cannot tell"
                // rather than inventing a success.
                _exited.TrySetResult(rc == _pid ? PosixSpawn.ExitCodeOf(status) : -1);
            })
            {
                IsBackground = true,
                Name = "AgentHost child wait",
            };
            thread.Start();
        }

        /// <summary>EINTR: a signal arrived while waiting, which is not the child exiting.</summary>
        private static bool WasInterrupted() =>
            System.Runtime.InteropServices.Marshal.GetLastWin32Error() == 4;

        // ---- the path that still forks, where nothing else is available -------

        private static bool TryStartManaged(
            SpawnRequest request, out SpawnedProcess? process, out string error)
        {
            process = null;

            var startInfo = new ProcessStartInfo
            {
                FileName = request.FileName,
                WorkingDirectory = request.WorkingDirectory ?? string.Empty,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                RedirectStandardInput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                // UTF-8, explicitly, on every one of the three.
                //
                // Left unset, .NET decodes a redirected child's output with the PARENT
                // process's console output encoding, which on Windows is the OEM code
                // page — 437 on an en-US install. The posix path a few lines up has
                // always constructed its readers with `new UTF8Encoding(false)`; this
                // one inherited whatever the host happened to be attached to, and the
                // two therefore disagreed about every byte above 0x7F.
                //
                // What that cost is not theoretical. A Python script printing an arrow
                // writes E2 86 92; decoded as CP437 those are three separate characters
                // (U+0393 U+00E5 U+00C6), and that is the string the model was handed
                // and the string a later edit_file anchor was written against. Every
                // em dash in a generated document, every accented filename in a
                // listing, every CJK title came back mangled — silently, because
                // mojibake is still valid text and nothing downstream could tell.
                //
                // Measured before and after: `python -c "sys.stdout.buffer.write(
                // b'RAW:â')"` returned U+0393 U+00E5 U+00C6 before, and
                // the arrow after. Note the shell was never the culprit — PowerShell
                // passes a native command's bytes straight through to the inherited
                // handle. The decode happened here.
                StandardOutputEncoding = Utf8NoBom,
                StandardErrorEncoding = Utf8NoBom,
                StandardInputEncoding = Utf8NoBom,
            };
            foreach (string argument in request.Arguments)
                startInfo.ArgumentList.Add(argument);

            startInfo.Environment.Clear();
            foreach (KeyValuePair<string, string> pair in request.Environment)
                startInfo.Environment[pair.Key] = pair.Value;

            Process Build()
            {
                var built = new Process { StartInfo = startInfo };
                built.OutputDataReceived += (_, e) =>
                {
                    if (e.Data == null || request.OnStdoutLine == null) return;
                    try { request.OnStdoutLine(e.Data); } catch (Exception) { }
                };
                built.ErrorDataReceived += (_, e) =>
                {
                    if (e.Data == null || request.OnStderrLine == null) return;
                    try { request.OnStderrLine(e.Data); } catch (Exception) { }
                };
                return built;
            }

            if (!ForkWatchdog.TryStart(Build, out Process? started, out error) || started == null)
                return false;

            try
            {
                started.BeginOutputReadLine();
                started.BeginErrorReadLine();
                try { started.StandardInput.Close(); } catch (IOException) { /* already gone */ }
            }
            catch (Exception ex) when (ex is InvalidOperationException or IOException)
            {
                error = ex.Message;
                try { started.Dispose(); } catch (Exception) { }
                return false;
            }

            process = new SpawnedProcess(started);
            error = string.Empty;
            return true;
        }

        // ---- what the callers ask of it ---------------------------------------

        /// <summary>The operating system's id for the process, or -1 once it cannot be read.</summary>
        public int Id
        {
            get
            {
                if (_managed == null)
                    return _pid;
                try { return _managed.Id; }
                catch (InvalidOperationException) { return -1; }
            }
        }

        /// <summary>The Windows process handle, or zero where the process was not started that way.</summary>
        public IntPtr Handle
        {
            get
            {
                if (_managed == null)
                    return IntPtr.Zero;
                try { return _managed.Handle; }
                catch (Exception) { return IntPtr.Zero; }
            }
        }

        /// <summary>Whether it has finished.</summary>
        public bool HasExited
        {
            get
            {
                if (_managed == null)
                    return _exited.Task.IsCompleted;
                try { return _managed.HasExited; }
                catch (InvalidOperationException) { return true; }
            }
        }

        /// <summary>Its exit code, or -1 while it is still running or could not be read.</summary>
        public int ExitCode
        {
            get
            {
                if (_managed == null)
                    return _exited.Task.IsCompleted ? _exited.Task.Result : -1;
                try { return _managed.ExitCode; }
                catch (Exception) { return -1; }
            }
        }

        /// <summary>Wait up to <paramref name="milliseconds"/> for the PROCESS to exit.</summary>
        public bool WaitForExit(int milliseconds)
        {
            if (_managed == null)
                return _exited.Task.Wait(Math.Max(0, milliseconds));
            try { return _managed.WaitForExit(Math.Max(0, milliseconds)); }
            catch (Exception ex) when (ex is InvalidOperationException or SystemException) { return true; }
        }

        /// <summary>
        /// Wait up to <paramref name="milliseconds"/> for the output pipes to reach EOF.
        ///
        /// <para>
        /// Separate from <see cref="WaitForExit"/> and always bounded, because a pipe stays
        /// open while ANYTHING holds the inherited handle. A command ending in <c>&amp;</c>
        /// leaves the grandchild holding it after the shell has exited, so an unbounded
        /// drain there is a tool call that never returns.
        /// </para>
        /// </summary>
        public bool WaitForDrain(int milliseconds)
        {
            if (_managed == null)
            {
                var deadline = Stopwatch.StartNew();
                bool first = _stdoutReader?.Join(Math.Max(0, milliseconds)) ?? true;
                int left = milliseconds - (int)deadline.ElapsedMilliseconds;
                bool second = _stderrReader?.Join(Math.Max(0, left)) ?? true;
                return first && second;
            }

            try
            {
                // The parameterless overload is the one that waits for the readers; giving
                // it a bound is the whole point, so it runs where it can be abandoned.
                var drain = Task.Run(() => { try { _managed.WaitForExit(); } catch (Exception) { } });
                return drain.Wait(Math.Max(0, milliseconds));
            }
            catch (Exception) { return true; }
        }

        /// <summary>Kill it and everything it started. Safe to call on a process that has exited.</summary>
        public void Kill()
        {
            if (_managed != null)
            {
                try { _managed.Kill(entireProcessTree: true); }
                catch (Exception) { /* already gone, or we cannot reach the tree */ }
                return;
            }

            if (_pid <= 0)
                return;

            try
            {
                // The tree first, and ONLY once the kernel confirms this pid really leads
                // its own process group. If setting the group had silently failed the child
                // would still be in OURS, and a signal to -pgid would kill the server.
                if (PosixSpawn.getpgid(_pid) == _pid)
                    PosixSpawn.kill(-_pid, PosixSpawn.Sigkill);

                PosixSpawn.kill(_pid, PosixSpawn.Sigkill);
            }
            catch (Exception) { /* already reaped */ }
        }

        /// <summary>Kill the tree and release what is left. Safe to call twice.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            Kill();

            if (_managed != null)
            {
                try { _managed.Dispose(); } catch (Exception) { }
                return;
            }

            // The reader threads own the read descriptors and close them at EOF, which the
            // kill above has just guaranteed. Joining briefly is what makes that ordering
            // true rather than hoped for; they are background threads, so a straggler
            // cannot hold the host open either way.
            WaitForDrain(1000);
        }

        // ---- helpers -----------------------------------------------------------

        private static string DescribeErrno(int errno, string fileName) => errno switch
        {
            2 => $"'{fileName}' was not found",
            13 => $"'{fileName}' is not executable",
            8 => $"'{fileName}' is not a valid executable for this machine",
            _ => $"'{fileName}' could not be started (errno {errno.ToString(System.Globalization.CultureInfo.InvariantCulture)})",
        };

        private static void Close(ref int fd)
        {
            if (fd < 0) return;
            PosixSpawn.close(fd);
            fd = -1;
        }

        private static void CloseAll(
            ref int a, ref int b, ref int c, ref int d, ref int e, ref int f)
        {
            Close(ref a); Close(ref b); Close(ref c); Close(ref d); Close(ref e); Close(ref f);
        }
    }
}
