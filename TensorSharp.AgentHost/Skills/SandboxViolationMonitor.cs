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
using System.Threading;
using TensorSharp.AgentHost.CodeExec;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// Watches the macOS sandbox violation log while one confined process runs, so a
    /// failed run can say WHAT the sandbox refused.
    ///
    /// <para>
    /// Without this, a denial reaches the model as whatever the script made of it —
    /// a bare <c>PermissionError</c>, a library's "file not found", LibreOffice
    /// exiting 1 with nothing on stderr — and the model (and the operator) retries
    /// blind. The kernel logged the precise refusal the whole time
    /// (<c>deny(1) file-read-data /Users/... </c>); this taps that stream, the way
    /// Anthropic's sandbox-runtime does, and hands the lines observed during the
    /// run's window to the caller. <c>log stream</c> is used rather than a
    /// post-mortem <c>log show</c> because the show path is rate-limited into
    /// uselessness under load — a lesson learned debugging exactly one of these
    /// denials.
    /// </para>
    /// <para>
    /// The stream is system-wide, so lines are FILTERED to those plausibly ours:
    /// the denied path lies in the run's workspace or the user's home, or the
    /// offending process is the interpreter (or a well-known child) launched. The
    /// result is advisory diagnostics on a failed run, labelled as observations —
    /// never a security signal.
    /// </para>
    /// <para>
    /// ONE monitor for the whole process, not one per run, and that is a
    /// performance decision with a measured cause. Starting and stopping
    /// <c>log stream</c> around every command cost ~180 ms of the ~220 ms a trivial
    /// command took — ~26 ms to launch and kill the helper, and ~154 ms in a
    /// <c>Thread.Sleep(150)</c> that existed to let the tail of a just-exited run's
    /// denials arrive. Both were paid on every command including the ones that
    /// succeeded, to produce diagnostics that are only ever read when a run FAILS.
    /// The stream is system-wide anyway, so one of them serves every run: each asks
    /// for the lines that arrived inside its own window, and only a failed run pays
    /// the grace period for the tail.
    /// </para>
    /// </summary>
    public sealed class SandboxViolationMonitor : IDisposable
    {
        /// <summary>
        /// How many recent denial lines are kept. Bounded because the stream is
        /// system-wide and never stops: a machine with an unrelated sandboxed
        /// application logging steadily must not grow this without limit.
        /// </summary>
        private const int MaxLines = 2000;

        /// <summary>How long a FAILED run waits for the tail of its denials. Nothing else waits.</summary>
        private static readonly TimeSpan TailGrace = TimeSpan.FromMilliseconds(150);

        private SpawnedProcess? _stream;
        private readonly List<(DateTime At, string Line)> _lines = new();
        private readonly object _gate = new();

        private SandboxViolationMonitor(SpawnedProcess? stream) => _stream = stream;

        /// <summary>Take ownership of the stream, once it has actually started.</summary>
        /// <remarks>
        /// Two-step because the line sink has to exist before the process does: what
        /// receives a line is this instance, and the request that starts the process needs
        /// that sink handed to it up front rather than attached afterwards.
        /// </remarks>
        private void Attach(SpawnedProcess stream) => _stream = stream;

        private static Dictionary<string, string> HostEnvironment()
        {
            var environment = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (System.Collections.DictionaryEntry entry in Environment.GetEnvironmentVariables())
            {
                if (entry.Key is string key && entry.Value is string value)
                    environment[key] = value;
            }
            return environment;
        }

        /// <summary>Keep one denial line, dropping the oldest once the cap is reached.</summary>
        private void Record(string line)
        {
            if (!line.Contains("deny", StringComparison.Ordinal))
                return;
            lock (_gate)
            {
                _lines.Add((DateTime.UtcNow, line));
                if (_lines.Count > MaxLines)
                    _lines.RemoveRange(0, _lines.Count - MaxLines);
            }
        }

        private static SandboxViolationMonitor? _shared;
        private static readonly object SharedGate = new();

        /// <summary>
        /// The process-wide monitor, started on first use.
        ///
        /// <para>
        /// Returns null where there is nothing to watch — every platform but macOS —
        /// so a caller can skip the whole path rather than holding an inert object.
        /// </para>
        /// </summary>
        public static SandboxViolationMonitor? Shared
        {
            get
            {
                if (!OperatingSystem.IsMacOS())
                    return null;
                if (_shared != null)
                    return _shared;
                lock (SharedGate)
                    return _shared ??= Start();
            }
        }

        /// <summary>
        /// A mark to pass back to <see cref="DenialsSince"/>, taken before the run
        /// starts. Wall-clock rather than a sequence number because the line's arrival
        /// time is what orders it here.
        /// </summary>
        public static DateTime Mark() => DateTime.UtcNow;

        /// <summary>
        /// Start watching, or return an inert monitor where the platform has no
        /// violation log (everywhere but macOS) or the stream cannot start. Never
        /// throws: diagnostics must not be able to fail the run they describe.
        /// </summary>
        public static SandboxViolationMonitor Start()
        {
            if (!OperatingSystem.IsMacOS())
                return new SandboxViolationMonitor(null);

            try
            {
                var monitor = new SandboxViolationMonitor(null);
                var request = new SpawnRequest
                {
                    FileName = "/usr/bin/log",
                    Arguments = new[]
                    {
                        "stream", "--style", "compact",
                        "--predicate", "sender == \"Sandbox\"",
                    },
                    // The one child that INHERITS this process's environment. The rule that
                    // every other launch starts from nothing is about code the model
                    // supplied; this is an Apple diagnostic binary with a fixed argument
                    // vector, and stripping its environment would be a behaviour change to
                    // the only thing that can explain a sandbox denial.
                    Environment = HostEnvironment(),
                    OnStdoutLine = monitor.Record,
                };

                // A monitor is a diagnostic, so a start that fails degrades to an inert one
                // rather than failing the run it was meant to describe.
                if (!SpawnedProcess.TryStart(request, out SpawnedProcess? stream, out _) || stream == null)
                    return new SandboxViolationMonitor(null);

                monitor.Attach(stream);
                return monitor;
            }
            catch (Exception ex) when (ex is IOException or InvalidOperationException
                                          or System.ComponentModel.Win32Exception)
            {
                return new SandboxViolationMonitor(null);
            }
        }

        /// <summary>
        /// The denials observed that plausibly belong to the run: the line names a
        /// path under <paramref name="workDirectory"/> or the user's home, or the
        /// process is <paramref name="interpreter"/> (basename) or a known child.
        /// At most <paramref name="max"/> lines, deduplicated, trimmed to the
        /// interesting part.
        /// </summary>
        /// <param name="since">
        /// Only lines that arrived after this. Taken with <see cref="Mark"/> before the
        /// run started, so one shared stream can serve every run at once.
        /// </param>
        /// <param name="waitForTail">
        /// Whether to pause for the logging pipeline to deliver the tail of a
        /// just-exited run's denials. True only for a FAILED run — it is the one case
        /// that reads them, and paying it on every command cost more than everything
        /// else the command did.
        /// </param>
        public IReadOnlyList<string> DenialsSince(
            DateTime since, string interpreter, string workDirectory, bool waitForTail, int max = 8)
        {
            if (waitForTail)
                Thread.Sleep(TailGrace);

            List<string> snapshot;
            lock (_gate)
                snapshot = _lines.Where(l => l.At >= since).Select(l => l.Line).ToList();
            if (snapshot.Count == 0)
                return Array.Empty<string>();

            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string interpreterName = SafeBaseName(interpreter);
            string[] knownChildren = { "python", "node", "sh", "bash", "soffice", "npm" };

            var seen = new HashSet<string>(StringComparer.Ordinal);
            var relevant = new List<string>();
            foreach (string line in snapshot)
            {
                bool ours =
                    (!string.IsNullOrEmpty(workDirectory) && line.Contains(workDirectory, StringComparison.Ordinal))
                    || (!string.IsNullOrEmpty(home) && line.Contains(home, StringComparison.Ordinal))
                    || (!string.IsNullOrEmpty(interpreterName)
                        && line.Contains(interpreterName, StringComparison.OrdinalIgnoreCase))
                    || knownChildren.Any(c => line.Contains(c, StringComparison.OrdinalIgnoreCase));
                if (!ours || IsStartupNoise(line))
                    continue;

                string compact = CompactDenial(line);
                if (seen.Add(compact))
                {
                    relevant.Add(compact);
                    if (relevant.Count >= max)
                        break;
                }
            }
            return relevant;
        }

        /// <summary>
        /// Denials every confined interpreter produces on startup, which say nothing
        /// about why a script failed.
        ///
        /// <para>
        /// This list earns its place the same way the xcrun_db allowance did: these
        /// lines are appended to a failed run's stderr and go to the MODEL, where
        /// noise reads as the cause. CPython pokes /dev/dtracehelper and
        /// ~/.CFUserTextEncoding on every start under Seatbelt, gets refused, and
        /// carries on completely unaffected — reporting them would point the model
        /// at the wrong thing on every single failure.
        /// </para>
        /// <para>
        /// <c>/dev/tty</c> joined the list when the tool surface became a shell: every
        /// bash reaching for its controlling terminal to set up job control is denied it
        /// and carries on, so the line appeared on EVERY failed command — a constant that
        /// correlates with failure and causes none of it, which is the worst possible
        /// thing to show a model that is looking for the cause. The same applies to the
        /// preference and mDNS lookups every framework build makes on start.
        /// </para>
        /// </summary>
        private static readonly string[] StartupNoise =
        {
            "/dev/dtracehelper",
            ".CFUserTextEncoding",
            "/dev/autofs_nowait",
            "/dev/tty",
            "user-preference-read kcfpreferencesanyapplication",
        };

        private static bool IsStartupNoise(string line) =>
            StartupNoise.Any(n => line.Contains(n, StringComparison.Ordinal));

        /// <summary>Trim a log line to "process deny(...) operation path".</summary>
        private static string CompactDenial(string line)
        {
            // Compact-style lines carry a timestamp/thread prefix before the
            // message; the message is what starts at the process name, which is the
            // last-but-informative portion containing "deny".
            int at = line.IndexOf("Sandbox:", StringComparison.Ordinal);
            if (at >= 0)
                return line.Substring(at + "Sandbox:".Length).Trim();
            at = line.IndexOf("deny", StringComparison.Ordinal);
            return at > 40 ? line.Substring(Math.Max(0, at - 30)).Trim() : line.Trim();
        }

        private static string SafeBaseName(string path)
        {
            try { return Path.GetFileName(path) ?? string.Empty; }
            catch (ArgumentException) { return string.Empty; }
        }

        /// <summary>
        /// Stop the shared stream. For a host shutting down; individual runs never
        /// dispose the monitor, because they no longer own one.
        /// </summary>
        public void Dispose()
        {
            if (_stream == null)
                return;
            try
            {
                if (!_stream.HasExited)
                    _stream.Kill();
                _stream.Dispose();
            }
            catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception)
            {
                // The stream already ended; nothing to clean.
            }
        }
    }
}
