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
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// Runs a skill's bundled script as a confined child process.
    ///
    /// <para>
    /// A skill is content somebody supplied — uploaded as a ZIP, or pulled off GitHub
    /// into a directory the operator pointed at — and the decision to run one of its
    /// scripts is made by a model reading that same person's Markdown. So the question
    /// is not whether to trust it but what it can reach when it runs.
    /// </para>
    /// <para>
    /// Two layers answer that. <b>In process, always:</b> the path resolves through
    /// <see cref="SkillPathGuard"/> so only files inside the skill can be named; the
    /// interpreter comes from an allow-list rather than from a shebang; no shell is
    /// involved, so arguments are data and never syntax; the environment is scrubbed of
    /// inherited credentials; the working directory is a fresh scratch directory rather
    /// than the skill; stdin is closed; and the run is bounded in time and in captured
    /// output. <b>In the OS, when it can be:</b> <see cref="ISkillSandbox"/> confines
    /// the child so it cannot reach the network, cannot read the user's home directory
    /// (credentials, SSH keys, every other installed skill), and cannot write anywhere
    /// but its scratch directory.
    /// </para>
    /// <para>
    /// With <see cref="SkillSandboxMode.Required"/> — the default — a host that has no
    /// sandbox refuses to run scripts at all and says so, rather than quietly running
    /// them unconfined. That is the whole point of the mode existing: "isolation was
    /// unavailable" must not degrade into "isolation was skipped".
    /// </para>
    /// </summary>
    public sealed class SkillScriptRunner : ISkillScriptRunner
    {
        private readonly SkillScriptRunnerOptions _options;
        private readonly ILogger _logger;
        private readonly ISkillSandbox? _sandbox;

        /// <summary>
        /// Latch for the once-per-process "running unconfined" warning below: the host
        /// state it reports cannot change while the process lives, so repeating it per
        /// script would only bury it.
        /// </summary>
        private static int s_unconfinedHostWarned;

        public SkillScriptRunner(SkillScriptRunnerOptions? options = null, ILogger? logger = null)
        {
            _options = options ?? new SkillScriptRunnerOptions();
            _logger = logger ?? NullLogger.Instance;
            _sandbox = _options.Sandbox == SkillSandboxMode.Off ? null : SkillSandboxFactory.Detect();
        }

        /// <summary>The sandbox in force, or null when running unconfined.</summary>
        public ISkillSandbox? Sandbox => _sandbox;

        /// <summary>
        /// True when this runner will actually run anything. False when the host
        /// demanded a sandbox and none is available — in which case
        /// <see cref="UnavailableReason"/> says so.
        /// </summary>
        /// <summary>
        /// Whether a script may run at all here.
        ///
        /// <para>
        /// Under <see cref="SkillSandboxMode.Required"/> this asks whether the host
        /// actually CONFINES a script, not merely whether an <see cref="ISkillSandbox"/>
        /// object exists. The two came apart on Windows:
        /// <see cref="SkillSandboxWindows"/>'s <c>IsAvailable</c> is unconditionally true
        /// because a job object can always be created, and a job object bounds CPU,
        /// memory and process count but cannot restrict a single file or socket — which
        /// its own <see cref="ISkillSandbox.Capabilities"/> says plainly. Testing only
        /// for existence therefore let <c>required</c> — the DEFAULT — behave exactly
        /// like <c>preferred</c> there, quietly running scripts with full filesystem and
        /// network access on the setting whose entire promise is "sandbox or refuse".
        /// </para>
        /// </summary>
        public bool CanRun =>
            _options.Sandbox != SkillSandboxMode.Required || Confines(_sandbox);

        /// <summary>
        /// The confinement <c>required</c> actually requires: keeping a script out of the
        /// rest of the filesystem and off the network. Resource caps are welcome but are
        /// not isolation, so a sandbox that only bounds them does not qualify.
        /// </summary>
        private static bool Confines(ISkillSandbox? sandbox) =>
            sandbox is not null
            && sandbox.Capabilities.ConfinesWrites
            && sandbox.Capabilities.ConfinesNetwork;

        /// <summary>Why <see cref="CanRun"/> is false, or null.</summary>
        public string? UnavailableReason
        {
            get
            {
                if (CanRun)
                    return null;

                // Name what is missing rather than claiming there is nothing at all: on
                // Windows there IS a sandbox, it just does not confine, and an operator
                // reading "no OS sandbox" would go looking for one that does not exist.
                if (_sandbox is { } present)
                {
                    SkillSandboxCapabilities caps = present.Capabilities;
                    var gaps = new List<string>();
                    if (!caps.ConfinesWrites) gaps.Add("filesystem writes");
                    if (!caps.ConfinesNetwork) gaps.Add("network access");
                    if (!caps.ConfinesHomeReads) gaps.Add("reads of your home directory");

                    return $"this host's sandbox ({present.Name}) cannot confine "
                        + string.Join(", ", gaps)
                        + ", and skill scripts are configured to run only when they can be confined"
                        + " - pass --skills-sandbox preferred to accept that and run them anyway";
                }

                return "this host provides no OS sandbox (checked: "
                    + (OperatingSystem.IsMacOS() ? "sandbox-exec"
                       : OperatingSystem.IsLinux() ? "bubblewrap (install bwrap to enable it)"
                       : OperatingSystem.IsWindows() ? "windows job object"
                       : "none for this platform")
                    + "), and skill scripts are configured to run only when they can be confined";
            }
        }

        /// <inheritdoc />
        public SkillToolResult Run(
            Skill skill, string relativePath, IReadOnlyList<string> arguments,
            Action<string>? onOutput = null, IReadOnlyList<string>? packages = null)
        {
            ArgumentNullException.ThrowIfNull(skill);

            if (!CanRun)
                return SkillToolResult.Failure(UnavailableReason!);

            if (!SkillPathGuard.TryResolveExistingFile(skill.RootDirectory, relativePath, out string? scriptPath, out string? guardError))
                return SkillToolResult.Failure(ExplainUnresolvedScript(skill, relativePath, guardError));

            string normalized = SkillPathGuard.ToSkillRelative(skill.RootDirectory, scriptPath!);
            string extension = Path.GetExtension(normalized);

            if (!TryResolveInterpreter(extension, out string? interpreter, out string? interpreterError))
                return SkillToolResult.Failure($"Cannot run '{normalized}': {interpreterError}");

            // The session workspace, when the host supplies one, is the whole point of
            // running a skill's scripts at all: generate.py writes the pptx there,
            // validate.py finds it there a call later, and both import the packages the
            // session installed. Without one, the script gets the classic per-run
            // scratch that dies with the call.
            string workDirectory;
            bool perCallScratch = _options.Workspace == null;
            try
            {
                workDirectory = _options.Workspace?.WorkDirectory
                    ?? Path.Combine(
                        _options.ScratchDirectory ?? Path.GetTempPath(),
                        "ts-skill-" + Guid.NewGuid().ToString("N"));
                Directory.CreateDirectory(workDirectory);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return SkillToolResult.Failure($"a scratch directory for '{normalized}' could not be created: {ex.Message}");
            }

            var setupNotes = new List<string>();
            string? installLanguage = InstallLanguageFor(extension);

            // Dependencies the model named up front, and any requirements.txt the skill
            // ships (root or the script's own directory, once per session): both go into
            // the session environment BEFORE the first attempt.
            if (installLanguage != null && CanAutoInstall)
            {
                if (packages is { Count: > 0 })
                    InstallInto(installLanguage, packages, setupNotes, onOutput, "requested");

                foreach ((string key, IReadOnlyList<string> required) in RequirementsFor(skill, normalized))
                {
                    if (!_options.Workspace!.TryMarkApplied(key))
                        continue;
                    InstallInto(installLanguage, required, setupNotes, onOutput, key);
                }
            }

            try
            {
                return RunWithAutoInstall(
                    skill, normalized, scriptPath!, interpreter!, arguments,
                    workDirectory, onOutput, installLanguage, setupNotes);
            }
            finally
            {
                if (perCallScratch && _options.DeleteScratchDirectory)
                    TryDeleteDirectory(workDirectory);
            }
        }

        private bool CanAutoInstall =>
            _options.Workspace != null
            && _options.PackageInstaller is { CanInstallPackages: true };

        /// <summary>The install language for a script extension, or null.</summary>
        private static string? InstallLanguageFor(string extension) =>
            extension.ToLowerInvariant() switch
            {
                ".py" => "python",
                ".js" or ".mjs" => "javascript",
                _ => null,
            };

        private void InstallInto(
            string language, IReadOnlyList<string> packages, List<string> notes,
            Action<string>? onOutput, string what)
        {
            string? error = _options.PackageInstaller!.InstallPackages(
                language, packages, _options.Workspace!, onOutput);
            notes.Add(error == null
                ? $"Set up dependencies ({what}): {string.Join(", ", packages)}"
                : $"Could not install {what} dependencies ({string.Join(", ", packages)}): {error}");
        }

        /// <summary>
        /// The dependency lists a skill declares: a <c>requirements.txt</c> at its root
        /// and one beside the script, keyed for once-per-session application. Lines are
        /// taken conservatively — bare names and version pins only; options, includes
        /// and URLs are for pip invocations this host deliberately does not make.
        /// </summary>
        private IEnumerable<(string Key, IReadOnlyList<string> Packages)> RequirementsFor(Skill skill, string normalized)
        {
            string? scriptDir = Path.GetDirectoryName(normalized)?.Replace('\\', '/');
            var candidates = new List<string> { "requirements.txt" };
            if (!string.IsNullOrEmpty(scriptDir))
                candidates.Add(scriptDir + "/requirements.txt");

            foreach (string candidate in candidates.Distinct())
            {
                if (!SkillPathGuard.TryResolveExistingFile(skill.RootDirectory, candidate, out string? path, out _))
                    continue;

                List<string> packages;
                try
                {
                    packages = File.ReadLines(path!)
                        .Select(line => line.Split('#')[0].Trim())
                        .Where(line => line.Length > 0 && !line.StartsWith('-') && !line.Contains("://"))
                        .Take(16)
                        .ToList();
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    continue;
                }

                if (packages.Count > 0)
                    yield return ($"requirements:{skill.Id}:{candidate}", packages);
            }
        }

        /// <summary>
        /// Run the script; when it dies on a missing import and this host can install,
        /// install the module and run it again. This is what "the skill's scripts just
        /// work" means in practice: the pptx validator needs defusedxml and lxml, no
        /// interpreter ships them, and without this loop the model spends whole rounds
        /// discovering that one import at a time.
        /// </summary>
        private SkillToolResult RunWithAutoInstall(
            Skill skill, string normalized, string scriptPath, string interpreter,
            IReadOnlyList<string> arguments, string workDirectory, Action<string>? onOutput,
            string? installLanguage, List<string> setupNotes)
        {
            var attempted = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            SkillToolResult result;

            while (true)
            {
                result = RunConfined(skill, normalized, scriptPath, interpreter, arguments,
                    workDirectory, onOutput, out string stderrText);

                if (result.Ok || installLanguage == null || !CanAutoInstall
                    || attempted.Count >= Math.Max(1, _options.MaxAutoInstallAttempts))
                    break;

                CodeExec.CodeLanguage language = CodeExec.CodeExecOptions.ParseLanguage(installLanguage);
                string? missing = CodeExec.CodeDiagnostics.MissingModule(language, stderrText);
                if (missing == null)
                    break;

                string package = CodeExec.CodeDiagnostics.InstallNameFor(language, missing);
                if (!attempted.Add(package))
                    break;      // installing it did not make the import work; stop

                Tap(onOutput, $"[installing missing dependency: {package}]");
                string? error = _options.PackageInstaller!.InstallPackages(
                    installLanguage, new[] { package }, _options.Workspace!, onOutput);
                if (error != null)
                {
                    setupNotes.Add($"Could not auto-install '{package}': {error}");
                    break;
                }

                setupNotes.Add($"Auto-installed missing dependency: {package}");
                Tap(onOutput, $"[re-running {normalized}]");
            }

            if (setupNotes.Count == 0)
                return result;

            // The model reads what happened in order: setup first, then the run.
            return result with { Content = string.Join("\n", setupNotes) + "\n\n" + result.Content };
        }

        private SkillToolResult RunConfined(
            Skill skill,
            string normalized,
            string scriptPath,
            string interpreter,
            IReadOnlyList<string> arguments,
            string workDirectory,
            Action<string>? onOutput,
            out string stderrText)
        {
            stderrText = string.Empty;
            var scriptArgs = new List<string> { scriptPath };
            if (arguments != null)
                scriptArgs.AddRange(arguments);

            string fileName = interpreter;
            IReadOnlyList<string> argv = scriptArgs;
            IDisposable? cleanup = null;
            string sandboxName = "none";

            // The session's package environment must be readable inside the sandbox, or
            // PYTHONPATH points at a directory the seatbelt profile denies.
            IReadOnlyList<string> readablePaths = _options.ReadablePaths;
            if (_options.Workspace is { } workspace)
            {
                var extended = new List<string>(readablePaths) { workspace.EnvDirectory };
                readablePaths = extended;
            }

            // What was already in the shared directory before this script ran, so the
            // result reports what THIS run produced rather than the session's history.
            Dictionary<string, (long Length, DateTime WriteTime)>? preRun =
                _options.Workspace?.SnapshotWorkFiles();

            if (_sandbox != null)
            {
                var request = new SkillSandboxRequest(
                    interpreter,
                    scriptArgs,
                    skill.RootDirectory,
                    workDirectory,
                    _options.AllowNetwork,
                    readablePaths);

                if (_sandbox.TryWrap(request, out string wrappedFile, out IReadOnlyList<string> wrappedArgs,
                        out IDisposable wrappedCleanup, out string wrapError))
                {
                    fileName = wrappedFile;
                    argv = wrappedArgs;
                    cleanup = wrappedCleanup;
                    sandboxName = _sandbox.Name;
                }
                else if (_options.Sandbox == SkillSandboxMode.Required)
                {
                    return SkillToolResult.Failure(
                        $"'{normalized}' was not run: the sandbox could not be prepared ({wrapError}), and skill "
                        + "scripts are configured to run only when they can be confined.");
                }
                else
                {
                    _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                        "skills.script.unconfined skill={SkillId} script={Script} reason={Reason}",
                        skill.Id, normalized, wrapError);
                }
            }
            else if (_options.Sandbox == SkillSandboxMode.Preferred
                && Interlocked.Exchange(ref s_unconfinedHostWarned, 1) == 0)
            {
                // `preferred` quietly degrades to no confinement when the host has no
                // sandbox at all — the one case the TryWrap-failure warning above never
                // sees, because there is nothing to wrap with.
                _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                    "skills.script.unconfined host={Host}: --skills-sandbox preferred found no OS sandbox, so skill scripts " +
                    "run UNCONFINED — full filesystem and network access with this process's privileges. Use " +
                    "--skills-sandbox required to refuse instead. Reported once.",
                    SkillSandboxFactory.DescribeHost());
            }

            var stdout = new BoundedWriter(_options.MaxOutputBytes);
            var stderr = new BoundedWriter(_options.MaxOutputBytes);
            var sw = Stopwatch.StartNew();

            try
            {
                // No shell: the arguments are passed as a vector, so a path or an
                // argument containing ; | > $ ` is data, not syntax.
                var request = new SpawnRequest
                {
                    FileName = fileName,
                    Arguments = argv,
                    WorkingDirectory = workDirectory,
                    Environment = BuildEnvironment(workDirectory),
                    OnStdoutLine = line => { stdout.AppendLine(line); Tap(onOutput, line); },
                    OnStderrLine = line => { stderr.AppendLine(line); Tap(onOutput, line); },
                };

                if (!SpawnedProcess.TryStart(request, out SpawnedProcess? started, out string startError)
                    || started == null)
                {
                    return SkillToolResult.Failure(startError.Length > 0
                        ? $"'{normalized}' could not be started: {startError}"
                        : $"'{fileName}' could not be started.");
                }

                using var process = started;

                // A job-object style sandbox attaches to a process that is already
                // running. If it cannot, the child is killed rather than left running
                // outside the confinement the caller asked for.
                if (_sandbox != null && !_sandbox.TryAttach(process, out string attachError))
                {
                    TryKill(process);
                    if (_options.Sandbox == SkillSandboxMode.Required)
                    {
                        return SkillToolResult.Failure(
                            $"'{normalized}' was stopped: the sandbox could not be applied to the process "
                            + $"({attachError}), and skill scripts are configured to run only when they can be confined.");
                    }
                    _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                        "skills.script.attach-failed skill={SkillId} script={Script} reason={Reason}",
                        skill.Id, normalized, attachError);
                }

                // Reading the pipes and closing stdin happen as part of starting: a script
                // that reads stdin fails fast instead of blocking until the deadline, and a
                // child whose output nobody drains blocks on a full pipe.
                bool exited = process.WaitForExit((int)_options.Timeout.TotalMilliseconds);
                if (!exited)
                {
                    TryKill(process);
                    _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                        "skills.script.timeout skill={SkillId} script={Script} sandbox={Sandbox} timeoutMs={TimeoutMs}",
                        skill.Id, normalized, sandboxName, (int)_options.Timeout.TotalMilliseconds);
                    return SkillToolResult.Failure(
                        $"'{normalized}' did not finish within "
                        + $"{_options.Timeout.TotalSeconds.ToString("0.#", CultureInfo.InvariantCulture)}s and was stopped."
                        + Describe(stdout, stderr, workDirectory, preRun, _options.Workspace));
                }

                // Waiting for exit can return before the output pipes have drained, so the
                // drain is waited for separately — and BOUNDED. This used to be an
                // unbounded wait, which is the same shape that hung the shell tool: a pipe
                // stays open while anything still holds the inherited handle, so a script
                // that leaves a background process behind never reaches EOF. The tree is
                // killed on dispose regardless; the only question was whether this returns.
                process.WaitForDrain(DrainMilliseconds);
                sw.Stop();
                stderrText = stderr.ToStringWithNotice();

                _logger.LogInformation(LogEventIds.SkillScriptExecuted,
                    "skills.script.ran skill={SkillId} script={Script} sandbox={Sandbox} exit={ExitCode} ms={ElapsedMs} stdout={StdoutBytes}",
                    skill.Id, normalized, sandboxName, process.ExitCode, (long)sw.Elapsed.TotalMilliseconds, stdout.Length);

                var sb = new StringBuilder();
                sb.Append("Ran ").Append(normalized).Append(" (exit code ")
                  .Append(process.ExitCode.ToString(CultureInfo.InvariantCulture)).Append(", sandbox: ")
                  .Append(sandboxName).Append(")\n");

                // Say what the sandbox did NOT confine. The model is deciding what to do
                // with this script's output, and on a platform where the script could
                // have reached the network or the wider filesystem that is a materially
                // different situation from one where it could not.
                IReadOnlyList<string> gaps = _sandbox?.Capabilities.Gaps() ?? AllGaps;
                if (gaps.Count > 0)
                    sb.Append("Not confined on this host: ").Append(string.Join("; ", gaps)).Append(".\n");

                sb.Append(Describe(stdout, stderr, workDirectory, preRun, _options.Workspace));

                // A script that died on a missing import is the single most common way a
                // skill's tooling fails on a fresh host, and the fix is one call away:
                // the session's environment is shared, so the shell can install what the
                // script needs. Without this the model re-runs the script unchanged.
                // A `match` statement on Apple's frozen python3 (3.9) dies as a bare
                // "SyntaxError: invalid syntax" — which reads as a broken script when
                // the actual problem is the host's interpreter. Name it.
                if (process.ExitCode != 0
                    && (stderr.ToStringWithNotice().Contains("SyntaxError", StringComparison.Ordinal)
                        || stderr.ToStringWithNotice().Contains("unsupported operand type(s) for |: 'type'", StringComparison.Ordinal))
                    && interpreter.Contains("python", StringComparison.OrdinalIgnoreCase)
                    && CodeExec.CodeEnvironment.PythonVersionOf(interpreter) is { } version
                    && version < new Version(3, 10))
                {
                    sb.Append("\nNote: this host's Python is ").Append(version)
                      .Append(", and skill scripts commonly need 3.10+ (the 'match' statement). If the ")
                      .Append("script looks correct, the fix is on the host: install a newer Python ")
                      .Append("(e.g. `brew install python@3.12`) and restart the server — it is picked up automatically.\n");
                }

                if (process.ExitCode != 0
                    && CodeExec.CodeDiagnostics.MissingModule(CodeExec.CodeLanguage.Python, stderr.ToStringWithNotice()) is { } missing)
                {
                    string install = CodeExec.CodeDiagnostics.InstallNameFor(CodeExec.CodeLanguage.Python, missing);
                    sb.Append("\nThe module '").Append(missing)
                      .Append("' is not installed in this session's environment. Install it from the shell ")
                      .Append("and run this script again:\n  pip install ").Append(install)
                      .Append("\nThe shell and skill scripts share one environment, so what you install ")
                      .Append("there is visible here.\n");
                }

                // A skill script that fails for any reason OTHER than a missing import
                // used to end the task. The skill directory is read-only by
                // construction — correctly, since a skill is untrusted content that
                // must not rewrite itself and outlive the conversation — so the model
                // had nothing to fix and no way to fix it, and would either retry the
                // identical script or give up.
                //
                // The way out is a session-local COPY. Staged into the workspace on
                // failure, it becomes an ordinary file the shell already handles: read it
                // with sed or cat, change it with apply_patch, run it. The skill on disk
                // is untouched, so the next conversation still gets the original.
                if (process.ExitCode != 0 && _options.Workspace is { } fixWorkspace)
                {
                    string overlay = "skill_" + skill.Id + "_"
                        + Path.GetFileName(normalized).Replace(Path.DirectorySeparatorChar, '_');
                    try
                    {
                        string sourceText = File.ReadAllText(scriptPath);
                        if (fixWorkspace.TryWriteFile(overlay, sourceText, out _))
                        {
                            sb.Append("\nA copy of this script is now in your working directory as '")
                              .Append(overlay)
                              .Append("'. If the script itself is wrong, fix THAT copy: read it from the "
                                      + "shell, change it with apply_patch, and run it from the shell. "
                                      + "The skill's own copy is read-only and unchanged.\n");
                        }
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                    {
                        // Staging is a convenience; a failure to stage must not replace
                        // the script's own error with a filesystem one.
                    }
                }

                // The files this script produced, kept for the user to download. Same
                // contract as the shell tool: the model gets ready-made markdown links.
                IReadOnlyList<SkillProducedFile> files = Array.Empty<SkillProducedFile>();
                if (_options.CaptureProducedFiles != null && _options.Workspace is { } ws)
                {
                    files = _options.CaptureProducedFiles(
                        workDirectory,
                        preRun == null ? null : relative => ws.IsUnchangedSince(preRun, relative));
                    if (files.Count > 0)
                    {
                        sb.Append("\nFiles produced. The user downloads them through these links - copy the ")
                          .Append("markdown links below into your answer verbatim when the user asked for the file:\n");
                        foreach (SkillProducedFile file in files)
                            sb.Append("- [").Append(file.Name).Append("](").Append(file.Url).Append(")\n");
                    }
                }

                return new SkillToolResult(process.ExitCode == 0, sb.ToString(), skill.Id, normalized)
                { Files = files };
            }
            catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException or IOException)
            {
                return SkillToolResult.Failure($"'{normalized}' could not be run: {ex.Message}");
            }
            finally
            {
                cleanup?.Dispose();
            }
        }

        /// <summary>
        /// Give the child a minimal environment.
        ///
        /// <para>
        /// The host process's environment is where credentials live —
        /// <c>AWS_SECRET_ACCESS_KEY</c>, <c>OPENAI_API_KEY</c>, <c>GITHUB_TOKEN</c>, a
        /// database URL. Inheriting it wholesale would hand every one of them to an
        /// uploaded script, and the sandbox cannot help: the values are already in the
        /// process image. So the child starts from nothing and is given back only what
        /// an interpreter needs to run.
        /// </para>
        /// </summary>
        private Dictionary<string, string> BuildEnvironment(string workDirectory)
        {
            var startInfo = new EnvironmentBag();

            foreach (string name in _options.PassThroughEnvironmentVariables)
            {
                string? value = Environment.GetEnvironmentVariable(name);
                if (!string.IsNullOrEmpty(value))
                    startInfo.Environment[name] = value;
            }

            startInfo.Environment["PWD"] = workDirectory;
            startInfo.Environment["TMPDIR"] = workDirectory;
            startInfo.Environment["HOME"] = workDirectory;
            // Keeps CPython from writing .pyc files into the read-only skill directory,
            // which fails under the sandbox and produces a confusing error.
            startInfo.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
            startInfo.Environment["PYTHONUNBUFFERED"] = "1";

            // The session's package environment: what the shell installed, the script can
            // import. This single line is what makes a skill's bundled tooling actually
            // RUNNABLE — validate.py needs defusedxml, thumbnail.py needs Pillow, and
            // none of them ship with the interpreter.
            if (_options.Workspace is { } workspace)
            {
                startInfo.Environment["PYTHONPATH"] = workspace.EnvDirectory;
                startInfo.Environment["NODE_PATH"] = Path.Combine(workspace.EnvDirectory, "node_modules");
            }

            foreach (KeyValuePair<string, string> entry in _options.EnvironmentVariables)
                startInfo.Environment[entry.Key] = entry.Value;

            return startInfo.Environment;
        }

        /// <summary>
        /// A stand-in for the <see cref="ProcessStartInfo"/> this used to fill in, so the
        /// assignments below read exactly as they did when the environment was applied to a
        /// start info rather than handed over as a complete set.
        /// </summary>
        private sealed class EnvironmentBag
        {
            public Dictionary<string, string> Environment { get; } = new(StringComparer.Ordinal);
        }

        /// <summary>
        /// Render the run's output, plus anything it left in the scratch directory.
        /// Listing the produced files is what makes a script that writes a report
        /// useful: the model can name the file in its answer, and the caller can find it.
        /// </summary>
        private static string Describe(
            BoundedWriter stdout, BoundedWriter stderr, string workDirectory,
            Dictionary<string, (long Length, DateTime WriteTime)>? preRun = null,
            SessionWorkspace? workspace = null)
        {
            var sb = new StringBuilder();
            if (stdout.Length > 0)
                sb.Append("\nstdout:\n").Append(stdout.ToStringWithNotice());
            if (stderr.Length > 0)
                sb.Append("\nstderr:\n").Append(stderr.ToStringWithNotice());

            string[] produced = ListProducedFiles(workDirectory, preRun, workspace);
            if (produced.Length > 0)
            {
                sb.Append("\nFiles this run wrote to the working directory:\n");
                foreach (string file in produced.Take(40))
                    sb.Append("- ").Append(file).Append('\n');
                if (produced.Length > 40)
                {
                    sb.Append("- (").Append((produced.Length - 40).ToString(CultureInfo.InvariantCulture))
                      .Append(" more)\n");
                }
            }

            if (sb.Length == 0)
                sb.Append("\n(no output)");
            return sb.ToString();
        }

        private static string[] ListProducedFiles(
            string workDirectory,
            Dictionary<string, (long Length, DateTime WriteTime)>? preRun,
            SessionWorkspace? workspace)
        {
            try
            {
                return Directory
                    .EnumerateFiles(workDirectory, "*", SearchOption.AllDirectories)
                    .Where(f => !Path.GetFileName(f).StartsWith(".tensorsharp-", StringComparison.Ordinal))
                    .Select(f => Path.GetRelativePath(workDirectory, f).Replace(Path.DirectorySeparatorChar, '/'))
                    // Bytecode caches and HOME-redirect fallout are a runtime's mess,
                    // not this script's report.
                    .Where(f => !CodeExec.CodeArtifactStore.IsRuntimeJunk(f))
                    // In a shared session workspace, "produced" means what THIS run
                    // added or changed, not the whole conversation's accumulation.
                    .Where(f => preRun == null || workspace == null || !workspace.IsUnchangedSince(preRun, f))
                    .OrderBy(f => f, StringComparer.Ordinal)
                    .ToArray();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return Array.Empty<string>();
            }
        }

        /// <summary>A live-output tap must never be able to kill the reader thread.</summary>
        private static void Tap(Action<string>? tap, string line)
        {
            if (tap == null) return;
            try { tap(line); }
            catch (Exception) { /* the tap is best-effort observability */ }
        }

        /// <summary>
        /// How long to wait for the output pipes to drain AFTER the script itself has
        /// exited. Bounded because a pipe held open by a process the script left running
        /// never reaches EOF.
        /// </summary>
        private const int DrainMilliseconds = 2000;

        private static void TryKill(SpawnedProcess process)
        {
            try { process.Kill(); }
            catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or System.ComponentModel.Win32Exception)
            {
                // Already exited, or the platform will not walk the tree. Either way
                // there is nothing further to do and the caller is already reporting a
                // timeout.
            }
        }

        private static void TryDeleteDirectory(string directory)
        {
            try
            {
                if (Directory.Exists(directory))
                    Directory.Delete(directory, recursive: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        }

        /// <summary>
        /// Map a file extension to the interpreter that runs it.
        ///
        /// <para>
        /// An allow-list rather than "make it executable and exec it": handing an
        /// arbitrary file to the OS loader would run a shipped binary, and a shebang
        /// line inside an uploaded script would choose the interpreter rather than this
        /// table.
        /// </para>
        /// </summary>
        /// <summary>
        /// Why the script path did not resolve, phrased so the model can fix it itself.
        ///
        /// <para>
        /// The common failure is not a typo. It is a model that puts the whole command
        /// line into <c>path</c> - <c>skills_run(path: "scripts/budget.py 2400")</c> -
        /// because that is how the skill's own SKILL.md writes the invocation
        /// ("RUN <c>scripts/budget.py &lt;payload_kg&gt;</c>"). Answering that with
        /// "'scripts/budget.py 2400' does not exist in this skill" is true and useless:
        /// the model reads it as a missing file, goes and reads the script to check the
        /// name, and retries the identical call. Measured on gemma-4-E4B, that cost three
        /// rounds and the whole round budget, and the request returned nothing.
        /// </para>
        /// <para>
        /// So when the path does not resolve but its leading token does, say which
        /// mistake was made and name the parameter that takes the rest. One round instead
        /// of a dead end.
        /// </para>
        /// </summary>
        private static string ExplainUnresolvedScript(Skill skill, string relativePath, string? guardError)
        {
            string message = $"Cannot run '{relativePath}' from skill '{skill.Id}': {guardError}";

            if (string.IsNullOrEmpty(relativePath))
                return message;

            int split = relativePath.IndexOfAny(ArgumentSeparators);
            if (split <= 0)
                return message;

            string head = relativePath.Substring(0, split);
            if (!SkillPathGuard.TryResolveExistingFile(skill.RootDirectory, head, out _, out _))
                return message;

            string tail = relativePath.Substring(split).Trim();
            return message
                + $". It looks like the arguments were included in 'path': '{head}' does exist."
                + $" Call skills_run again with path=\"{head}\" and args=\"{tail}\".";
        }

        private static readonly char[] ArgumentSeparators = { ' ', '\t' };

        private bool TryResolveInterpreter(string extension, out string? interpreter, out string? error)
        {
            interpreter = null;
            error = null;

            if (_options.Interpreters.TryGetValue(extension, out string? configured))
            {
                interpreter = configured;
                return true;
            }

            error = extension.Length == 0
                ? "it has no file extension, so there is no interpreter for it. Only "
                  + string.Join(", ", _options.Interpreters.Keys) + " files can be run."
                : $"'{extension}' files cannot be run here. Only "
                  + string.Join(", ", _options.Interpreters.Keys) + " files can be.";
            return false;
        }

        /// <summary>
        /// Split a command line into an argument vector the way a POSIX shell would —
        /// honouring single quotes, double quotes and backslash escapes — WITHOUT
        /// invoking a shell.
        ///
        /// <para>
        /// The model writes arguments as one string because a tool parameter cannot be
        /// an array here. Passing that string to a shell would make every metacharacter
        /// in it executable, so it is split in process and handed over as separate
        /// arguments instead: <c>--out "my file.pdf"; rm -rf ~</c> becomes four literal
        /// arguments, one of which is the harmless text <c>rm</c>.
        /// </para>
        /// </summary>
        internal static List<string> SplitArguments(string? commandLine)
        {
            var arguments = new List<string>();
            if (string.IsNullOrWhiteSpace(commandLine))
                return arguments;

            var current = new StringBuilder();
            bool started = false;
            char quote = '\0';

            for (int i = 0; i < commandLine.Length; i++)
            {
                char c = commandLine[i];

                if (quote != '\0')
                {
                    if (c == quote)
                    {
                        quote = '\0';
                        continue;
                    }
                    if (c == '\\' && quote == '"' && i + 1 < commandLine.Length)
                    {
                        char next = commandLine[i + 1];
                        if (next is '"' or '\\')
                        {
                            current.Append(next);
                            i++;
                            continue;
                        }
                    }
                    current.Append(c);
                    continue;
                }

                if (c is '"' or '\'')
                {
                    quote = c;
                    started = true;
                    continue;
                }

                if (char.IsWhiteSpace(c))
                {
                    if (started)
                    {
                        arguments.Add(current.ToString());
                        current.Clear();
                        started = false;
                    }
                    continue;
                }

                if (c == '\\' && i + 1 < commandLine.Length)
                {
                    current.Append(commandLine[++i]);
                    started = true;
                    continue;
                }

                current.Append(c);
                started = true;
            }

            if (started)
                arguments.Add(current.ToString());
            return arguments;
        }

        /// <summary>What is unconfined when there is no sandbox at all.</summary>
        private static readonly IReadOnlyList<string> AllGaps =
            new SkillSandboxCapabilities(false, false, false, false).Gaps();

        /// <summary>Accumulates process output up to a ceiling, then counts what it drops.</summary>
        private sealed class BoundedWriter
        {
            private readonly int _limit;
            private readonly StringBuilder _text = new();
            private long _dropped;

            public BoundedWriter(int limit) => _limit = Math.Max(1024, limit);

            public int Length => _text.Length;

            public void AppendLine(string line)
            {
                lock (_text)
                {
                    if (_text.Length + line.Length + 1 > _limit)
                    {
                        _dropped += line.Length + 1;
                        return;
                    }
                    _text.Append(line).Append('\n');
                }
            }

            public string ToStringWithNotice()
            {
                lock (_text)
                {
                    return _dropped == 0
                        ? _text.ToString()
                        : _text.ToString() + $"\n[{_dropped.ToString(CultureInfo.InvariantCulture)} further bytes of output were dropped.]\n";
                }
            }
        }
    }

    /// <summary>Bounds, isolation policy and interpreter mapping for <see cref="SkillScriptRunner"/>.</summary>
    public sealed class SkillScriptRunnerOptions
    {
        /// <summary>
        /// How hard to insist on OS isolation. <see cref="SkillSandboxMode.Required"/>
        /// by default: a host that cannot confine a script should refuse to run it
        /// rather than run it unconfined.
        /// </summary>
        public SkillSandboxMode Sandbox { get; init; } = SkillSandboxMode.Required;

        /// <summary>Let the script reach the network. Off by default — the sandbox blocks it.</summary>
        public bool AllowNetwork { get; init; }

        /// <summary>How long a script may run before it is killed.</summary>
        public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(60);

        /// <summary>Ceiling on captured stdout and on captured stderr, each.</summary>
        public int MaxOutputBytes { get; init; } = 32 * 1024;

        /// <summary>Where per-run scratch directories are created. Null uses the system temp directory.</summary>
        public string? ScratchDirectory { get; init; }

        /// <summary>
        /// Delete the scratch directory after the run. On by default; turn it off to
        /// keep whatever a script produced.
        /// </summary>
        public bool DeleteScratchDirectory { get; init; } = true;

        /// <summary>Extra paths the script may read, beyond the system and its own skill.</summary>
        public IReadOnlyList<string> ReadablePaths { get; init; } = Array.Empty<string>();

        /// <summary>
        /// The session's persistent workspace. When set, scripts run in its shared
        /// working directory (which survives the call, so one script's output is the
        /// next one's input), can import the packages the session installed via
        /// the shell tool, and per-run scratch handling is bypassed entirely.
        /// </summary>
        public SessionWorkspace? Workspace { get; init; }

        /// <summary>
        /// Keeps the files a script produced for the user to download, returning where
        /// each is fetched from. Supplied by the host (a server points it at its
        /// artifact store); only consulted in workspace mode. Null keeps nothing.
        /// </summary>
        public WorkspaceFileCapture? CaptureProducedFiles { get; init; }

        /// <summary>
        /// Installs packages into the session environment, so a script's missing
        /// dependencies are set up automatically instead of costing the model a
        /// round per import. Wired to the same installer the host uses (wheels
        /// only, allow-list honored). Null disables every install path here.
        /// </summary>
        public ICodeRunner? PackageInstaller { get; init; }

        /// <summary>
        /// Most distinct dependencies one script run may auto-install before it stops
        /// retrying. A bound, not a budget: chains longer than this smell like a
        /// script that is not going to work.
        /// </summary>
        public int MaxAutoInstallAttempts { get; init; } = 5;

        /// <summary>
        /// Extension to interpreter. Replace an entry to point at a virtualenv's
        /// <c>python</c>, or shorten the map to forbid a language outright.
        /// </summary>
        public Dictionary<string, string> Interpreters { get; init; } = new(StringComparer.OrdinalIgnoreCase)
        {
            // The SAME resolution the shell uses (newest Python first), so a package the
            // session installed with one interpreter is importable by the script — a
            // 3.13 pip's wheels under a 3.9 script would not be.
            [".py"] = OperatingSystem.IsWindows()
                ? "python"
                : CodeExec.CodeEnvironment.TryResolveInterpreter(CodeExec.CodeLanguage.Python, out string? python, out _)
                    ? python!
                    : "python3",
            [".js"] = "node",
            [".mjs"] = "node",
            [".sh"] = OperatingSystem.IsWindows() ? "bash" : "/bin/sh",
            [".bash"] = "bash",
        };

        /// <summary>
        /// Environment variables copied from the host into the child. Deliberately
        /// short: everything not listed here is dropped, so a credential the host
        /// happens to carry cannot leak into an uploaded script's stdout.
        /// </summary>
        public IReadOnlyList<string> PassThroughEnvironmentVariables { get; init; } = new[]
        {
            "PATH", "LANG", "LC_ALL", "TZ", "SystemRoot", "COMSPEC", "PATHEXT",
        };

        /// <summary>Extra environment variables for the child process.</summary>
        public Dictionary<string, string> EnvironmentVariables { get; init; } = new(StringComparer.Ordinal);
    }
}
