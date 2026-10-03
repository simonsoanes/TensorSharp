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
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// One thing to run, with everything a backend needs to run it: either a shell
    /// COMMAND LINE (the model's text, after the host has read the installs out of it)
    /// or an ARGUMENT VECTOR (a host-built launch — a skill script, a syntax check, an
    /// API probe, a package install — where no shell must ever parse anything).
    ///
    /// <para>
    /// This is the seam between "the host has decided what runs, where, with what
    /// environment and under what policy" and "something actually runs". Everything
    /// above it — reading the model's arguments, extracting installs, building the
    /// environment, the attempt ledger, artifact capture, every sentence of the result
    /// — is shared by every backend. Everything below it is the backend's: on a desktop
    /// that is a confined child process, and on iOS, where nothing may spawn a process,
    /// it is an in-process shell interpreter over embedded interpreters.
    /// </para>
    /// <para>
    /// Exactly one of <see cref="Command"/> and <see cref="Argv"/> is set. A command
    /// launch also carries the <see cref="Session"/> whose working directory and
    /// exported variables it must restore and re-save; an argv launch starts in
    /// <see cref="WorkingDirectory"/> and persists nothing.
    /// </para>
    /// </summary>
    public sealed record ShellLaunch
    {
        /// <summary>A shell command line, post install-substitution. Null for an argv launch.</summary>
        public string? Command { get; init; }

        /// <summary>
        /// A direct launch — executable first, then its arguments — with no shell
        /// involved, so a value containing <c>; | &gt; $ `</c> is data and never syntax.
        /// Null for a command launch.
        /// </summary>
        public IReadOnlyList<string>? Argv { get; init; }

        /// <summary>Where the run starts. For a command launch, the session's current directory unless the call asked for another.</summary>
        public required string WorkingDirectory { get; init; }

        /// <summary>
        /// For a command launch: the directory THIS call asked to run in, or null to
        /// continue where the session left off.
        ///
        /// <para>
        /// The distinction has to survive the seam because it decides what is persisted
        /// afterwards. A per-call directory is for this command only: the session
        /// resumes from where it was BEFORE the move, which is what the wrapper script's
        /// <c>__ts_resume</c> captures and what an in-process backend must reproduce
        /// through <see cref="ShellSession.Save"/>. Without it, one call's
        /// <c>workdir</c> moved the whole conversation.
        /// </para>
        /// </summary>
        public string? CallWorkDirectory { get; init; }

        /// <summary>Variables to give the run, on top of whatever minimal set the backend supplies.</summary>
        public IReadOnlyDictionary<string, string> Environment { get; init; } =
            new Dictionary<string, string>(StringComparer.Ordinal);

        /// <summary>The only directory the run may write, apart from <see cref="WritablePaths"/>.</summary>
        public required string WriteDirectory { get; init; }

        /// <summary>Additional exact paths it may write.</summary>
        public IReadOnlyList<string> WritablePaths { get; init; } = Array.Empty<string>();

        /// <summary>A directory it may read but never write. Must exist.</summary>
        public required string ReadOnlyDirectory { get; init; }

        /// <summary>Anything else it may read — and, just as importantly, execute.</summary>
        public IReadOnlyList<string> ReadablePaths { get; init; } = Array.Empty<string>();

        /// <summary>Whether it may open a socket.</summary>
        public bool AllowNetwork { get; init; }

        /// <summary>
        /// With <see cref="AllowNetwork"/> false: the one loopback TCP port the run may
        /// still reach — the host-side egress proxy an installer is pointed at.
        /// </summary>
        public int? AllowLoopbackPort { get; init; }

        /// <summary>How long before it is stopped. <see cref="System.Threading.Timeout.InfiniteTimeSpan"/> for a background job.</summary>
        public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

        /// <summary>Cap on each of stdout and stderr.</summary>
        public int MaxOutputBytes { get; init; } = 32 * 1024;

        /// <summary>
        /// Called once per line the run writes, WHILE it runs — a live tap for a host
        /// that wants to show progress. Invoked from whatever thread the backend reads
        /// output on; must be quick and thread-safe. Null taps nothing.
        /// </summary>
        public Action<string>? OnOutputLine { get; init; }

        /// <summary>
        /// The session whose state a command launch restores before and saves after.
        /// Required for a command launch; ignored by an argv launch.
        /// </summary>
        public ShellSession? Session { get; init; }

        /// <summary>
        /// What this launch is for, for logs and for a backend that answers some
        /// purposes itself: <c>shell</c>, <c>script</c>, <c>install</c>,
        /// <c>syntax-check</c> or <c>api-probe</c>.
        /// </summary>
        public string Purpose { get; init; } = Purposes.Shell;

        /// <summary>The purposes the host's own callers use. A backend may see others.</summary>
        public static class Purposes
        {
            /// <summary>A model-written shell command line.</summary>
            public const string Shell = "shell";

            /// <summary>A skill's bundled script, launched by argv.</summary>
            public const string Script = "script";

            /// <summary>A host-built package install (pip, npm).</summary>
            public const string Install = "install";

            /// <summary>A parse of files the model just wrote.</summary>
            public const string SyntaxCheck = "syntax-check";

            /// <summary>A read of an installed package's real API after a failed run.</summary>
            public const string ApiProbe = "api-probe";
        }

        /// <summary>Why this launch cannot be started as written, or null when it is well formed.</summary>
        public string? Validate()
        {
            if (Command == null && Argv == null)
                return "a launch must carry either a command line or an argument vector";
            if (Command != null && Argv != null)
                return "a launch carries either a command line or an argument vector, never both";
            if (Argv is { Count: 0 })
                return "an argument vector launch needs at least the executable";
            if (Command != null && Session == null)
                return "a command launch needs the session whose working directory and environment it restores";
            if (string.IsNullOrEmpty(WorkingDirectory))
                return "a launch needs a working directory";
            return null;
        }
    }

    /// <summary>
    /// A launch that has been started and not yet waited for. Mirrors
    /// <see cref="ConfinedJob"/> so a background job can be held by the host and stopped
    /// when the session ends, whatever runs it.
    /// </summary>
    public interface IShellJob : IDisposable
    {
        /// <summary>The operating system's id for the run, or -1 where there is no process.</summary>
        int ProcessId { get; }

        /// <summary>Which sandbox wrapped it, or "none".</summary>
        string SandboxName { get; }

        /// <summary>Whether it has finished.</summary>
        bool HasExited { get; }

        /// <summary>
        /// Why the run continued WITHOUT the confinement that was detected, when it did
        /// — a sandbox that could not wrap or attach in a mode that permits degrading.
        /// Null on every ordinary run. Reported, never swallowed: a run that degraded
        /// should say what failed, not merely that nothing was confined.
        /// </summary>
        string? DegradedReason => null;

        /// <summary>Wait up to <paramref name="timeout"/>, stopping the run if it goes over.</summary>
        ConfinedResult WaitForExit(TimeSpan timeout);

        /// <summary>Stop it now. Safe to call on a run that has already finished.</summary>
        void Kill();
    }

    /// <summary>
    /// Runs one <see cref="ShellLaunch"/>.
    ///
    /// <para>
    /// The host talks to exactly one of these. <see cref="ProcessShellBackend"/> is the
    /// desktop implementation — a confined child process, unchanged from before the
    /// seam existed. A host that cannot start processes supplies its own, and presents
    /// what it actually enforces through <see cref="Sandbox"/>: the host's own
    /// confinement questions (<c>CanRun</c>, the declaration's network promise, the
    /// result's "Not confined on this host" line) are all answered from that object's
    /// capabilities, so a backend that confines writes and the network in its own way
    /// is trusted exactly as far as it says.
    /// </para>
    /// </summary>
    public interface IShellBackend
    {
        /// <summary>Whether interpreters are native host executables rather than embedded runtime names.</summary>
        bool UsesHostProcesses => false;

        /// <summary>Short name for logs: <c>process</c>, <c>in-process</c>.</summary>
        string Name { get; }

        /// <summary>The shell dialect this backend speaks, or null when it has none — in which case <see cref="CanRun"/> is false.</summary>
        ShellProgram? Shell { get; }

        /// <summary>The confinement this backend applies, or null when it applies none.</summary>
        ISkillSandbox? Sandbox { get; }

        /// <summary>Whether this backend can run anything at all. Confinement policy is the host's question, not this one.</summary>
        bool CanRun { get; }

        /// <summary>Why <see cref="CanRun"/> is false, or null.</summary>
        string? UnavailableReason { get; }

        /// <summary>Start <paramref name="launch"/> and hand back the running job WITHOUT waiting for it.</summary>
        /// <param name="job">The running job, or null when it could not be started.</param>
        /// <param name="failure">Why it could not be started; <c>Started</c> is false.</param>
        bool TryStart(ShellLaunch launch, out IShellJob? job, out ConfinedResult failure);

        /// <summary>Run <paramref name="launch"/> and wait for it, up to its own timeout.</summary>
        ConfinedResult Run(ShellLaunch launch)
        {
            ArgumentNullException.ThrowIfNull(launch);
            if (!TryStart(launch, out IShellJob? job, out ConfinedResult failure))
                return failure;
            using (job)
                return job!.WaitForExit(launch.Timeout);
        }
    }

    /// <summary>
    /// The backend every desktop host used before the seam existed, unchanged in what
    /// it does: a command line becomes a wrapper script the session writes, an argv is
    /// launched as it is, and both go through <see cref="ConfinedProcess"/> under the
    /// detected OS sandbox.
    ///
    /// <para>
    /// It also does the one piece of result rewriting that only it can: a shell blames
    /// "<c>/…/state/cmd-7.sh: line 24</c>", a file the model never saw at a line twenty
    /// past anything it wrote, and only the backend that wrote the script knows where
    /// the model's own text began in it. See <see cref="Rewrite"/>.
    /// </para>
    /// </summary>
    public sealed class ProcessShellBackend : IShellBackend
    {
        public bool UsesHostProcesses => true;

        private readonly ISkillSandbox? _sandbox;
        private readonly SkillSandboxMode _mode;
        private readonly ShellProgram? _shell;
        private readonly string? _shellError;
        private readonly ILogger _logger;

        /// <param name="sandbox">The confinement to launch under, or null for none.</param>
        /// <param name="mode">Whether an unusable sandbox is fatal (<see cref="SkillSandboxMode.Required"/>) or degrades.</param>
        /// <param name="shell">The shell for command launches, or null when this host has none — argv launches still work.</param>
        /// <param name="shellUnavailableReason">Why <paramref name="shell"/> is null, when it is.</param>
        /// <param name="logger">Where launches are recorded, as metadata only.</param>
        public ProcessShellBackend(
            ISkillSandbox? sandbox,
            SkillSandboxMode mode,
            ShellProgram? shell = null,
            string? shellUnavailableReason = null,
            ILogger? logger = null)
        {
            _sandbox = sandbox;
            _mode = mode;
            _shell = shell;
            _shellError = shell == null ? shellUnavailableReason : null;
            _logger = logger ?? NullLogger.Instance;
        }

        /// <summary>
        /// Today's detection, exactly as <see cref="ShellRunner"/> always did it: the
        /// strongest OS sandbox this host provides unless the operator turned sandboxing
        /// off, the mode the operator asked for (with <c>--code-exec-unconfined</c>
        /// meaning "preferred"), and the shell found on PATH or named by
        /// <c>--code-exec-shell</c>.
        /// </summary>
        /// <param name="sandbox">A sandbox to use instead of detecting one, or null to detect.</param>
        /// <param name="shell">A shell to use instead of resolving one, or null to resolve.</param>
        public static ProcessShellBackend Detect(
            CodeExecOptions options, ILogger? logger = null,
            ISkillSandbox? sandbox = null, ShellProgram? shell = null)
        {
            ArgumentNullException.ThrowIfNull(options);

            ISkillSandbox? chosen = options.Sandbox == SkillSandboxMode.Off
                ? null
                : sandbox ?? SkillSandboxFactory.Detect();
            SkillSandboxMode mode = options.Unconfined ? SkillSandboxMode.Preferred : options.Sandbox;

            // Resolved once, at construction: the answer cannot change while the process
            // runs, and the prompt has to state the dialect before the first call.
            string? shellError = null;
            if (shell == null)
                ShellProgram.TryResolve(options.Shell, out shell, out shellError);

            return new ProcessShellBackend(chosen, mode, shell, shellError, logger);
        }

        /// <summary>
        /// Detection for a host that only launches argv — skill scripts — under one
        /// sandbox mode, and needs no shell.
        /// </summary>
        public static ProcessShellBackend Detect(SkillSandboxMode mode, ILogger? logger = null) =>
            new(mode == SkillSandboxMode.Off ? null : SkillSandboxFactory.Detect(), mode, logger: logger);

        /// <inheritdoc/>
        public string Name => "process";

        /// <inheritdoc/>
        public ShellProgram? Shell => _shell;

        /// <inheritdoc/>
        public ISkillSandbox? Sandbox => _sandbox;

        /// <summary>The mode launches run under, for a caller that has to describe it.</summary>
        public SkillSandboxMode Mode => _mode;

        /// <inheritdoc/>
        public bool CanRun => _shell != null;

        /// <inheritdoc/>
        public string? UnavailableReason =>
            _shell != null ? null : _shellError ?? "this host has no shell to run commands with";

        /// <inheritdoc/>
        public bool TryStart(ShellLaunch launch, out IShellJob? job, out ConfinedResult failure)
        {
            ArgumentNullException.ThrowIfNull(launch);
            job = null;

            if (launch.Validate() is { } invalid)
            {
                failure = ConfinedProcess.Failed(invalid);
                return false;
            }

            ShellSession.ShellScript? script = null;
            string interpreter;
            IReadOnlyList<string> arguments;
            if (launch.Command != null)
            {
                if (_shell == null)
                {
                    failure = ConfinedProcess.Failed(UnavailableReason!);
                    return false;
                }

                try
                {
                    script = launch.Session!.WriteScript(launch.Command, launch.CallWorkDirectory);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    failure = ConfinedProcess.Failed($"the command could not be prepared: {ex.Message}");
                    return false;
                }

                interpreter = _shell.Path;
                arguments = _shell.ArgumentsFor(script.Value.Path);
            }
            else
            {
                interpreter = launch.Argv![0];
                var rest = new string[launch.Argv.Count - 1];
                for (int i = 1; i < launch.Argv.Count; i++)
                    rest[i - 1] = launch.Argv[i];
                arguments = rest;
            }

            var confined = new ConfinedLaunch
            {
                Interpreter = interpreter,
                Arguments = arguments,
                WriteDirectory = launch.WriteDirectory,
                WritablePaths = launch.WritablePaths,
                WorkingDirectory = launch.WorkingDirectory,
                ReadOnlyDirectory = launch.ReadOnlyDirectory,
                ReadablePaths = launch.ReadablePaths,
                AllowNetwork = launch.AllowNetwork,
                AllowLoopbackPort = launch.AllowLoopbackPort,
                Timeout = launch.Timeout,
                MaxOutputBytes = launch.MaxOutputBytes,
                EnvironmentVariables = launch.Environment,
                OnOutputLine = launch.OnOutputLine,
            };

            if (!ConfinedProcess.TryStart(confined, _sandbox, _mode, out ConfinedJob? started, out failure))
                return false;

            _logger.LogDebug(
                "codeexec.launch backend=process purpose={Purpose} sandbox={Sandbox} pid={Pid}",
                launch.Purpose, started!.SandboxName, started.ProcessId);
            job = new ProcessShellJob(started, script);
            return true;
        }

        /// <summary>
        /// Put the shell's own diagnostics back into the model's frame of reference.
        ///
        /// <para>
        /// The wrapper is the host's business, not the model's, and every mention of it in
        /// the output is a false lead: bash blames "/…/state/cmd-7.sh: line 24", which
        /// names a file the model has never seen at a line twenty past anything it wrote.
        /// The path becomes "command" and the number becomes the line of the command
        /// itself, so a syntax error points at the line the model actually typed.
        /// </para>
        /// </summary>
        internal static ConfinedResult Rewrite(ConfinedResult result, ShellSession.ShellScript script)
        {
            string Fix(string text)
            {
                if (text.Length == 0 || !text.Contains(script.Path, StringComparison.Ordinal))
                    return text;

                // "<path>: line 24: " -> "command line 4: ", then any bare mention of
                // the path (a traceback naming the script, a shell prefixing every line).
                text = Regex.Replace(
                    text,
                    Regex.Escape(script.Path) + @":\s*line\s+(\d+):",
                    match => "command line "
                        + Math.Max(1, int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) - script.CommandOffset)
                            .ToString(CultureInfo.InvariantCulture) + ":");
                return text.Replace(script.Path, "command", StringComparison.Ordinal);
            }

            return result with { Stdout = Fix(result.Stdout), Stderr = Fix(result.Stderr) };
        }

        /// <summary>A <see cref="ConfinedJob"/> behind the seam, rewriting wrapper-script paths on the way out.</summary>
        private sealed class ProcessShellJob : IShellJob
        {
            private readonly ConfinedJob _job;
            private readonly ShellSession.ShellScript? _script;

            public ProcessShellJob(ConfinedJob job, ShellSession.ShellScript? script)
            {
                _job = job;
                _script = script;
            }

            public int ProcessId => _job.ProcessId;

            public string SandboxName => _job.SandboxName;

            public bool HasExited => _job.HasExited;

            public string? DegradedReason => _job.WrapFailure ?? _job.AttachFailure;

            public ConfinedResult WaitForExit(TimeSpan timeout)
            {
                ConfinedResult result = _job.WaitForExit(timeout);
                return _script is { } script ? Rewrite(result, script) : result;
            }

            public void Kill() => _job.Kill();

            public void Dispose() => _job.Dispose();
        }
    }
}
