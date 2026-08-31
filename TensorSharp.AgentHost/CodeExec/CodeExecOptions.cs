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
using System.Linq;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// A language this host can locate an interpreter for.
    ///
    /// <para>
    /// This is no longer a menu offered to the model — under the shell tool the model
    /// simply types <c>python3 solve.py</c> and the shell finds it on PATH. What remains
    /// is the host's own need to name an interpreter: to report at startup what is
    /// available, and to install a skill script's missing dependencies on its behalf.
    /// </para>
    /// </summary>
    public enum CodeLanguage
    {
        /// <summary>Not one we support.</summary>
        Unknown,
        Python,
        JavaScript,
        Shell,
    }

    /// <summary>
    /// Whether a host will run commands the MODEL wrote, and under what terms.
    ///
    /// <para>
    /// Deliberately a separate switch family from <c>--skills-allow-exec</c>, and not a
    /// mode of it. A skill's script is a file an operator put on disk and can read before
    /// enabling; a <c>shell</c> command is written during the request by a model whose
    /// context is influenced by whatever it has been shown — retrieved pages, documents,
    /// a skill someone uploaded. The two deserve separate decisions, so turning on skill
    /// scripts must never turn on this, and vice versa.
    /// </para>
    /// </summary>
    public sealed class CodeExecOptions
    {
        /// <summary>Turn the <c>shell</c> tool on. Off unless asked for.</summary>
        public const string EnabledFlag = "--code-exec";

        /// <summary>Environment override for <see cref="EnabledFlag"/>. Anything but <c>0</c> counts as on.</summary>
        public const string EnabledEnvVar = "TS_CODE_EXEC";

        /// <summary>Let the host install the packages a command asks for.</summary>
        public const string AllowInstallFlag = "--code-exec-allow-install";

        /// <summary>Environment override for <see cref="AllowInstallFlag"/>.</summary>
        public const string AllowInstallEnvVar = "TS_CODE_EXEC_ALLOW_INSTALL";

        /// <summary>Run even where the OS cannot confine the process. CLI only.</summary>
        public const string UnconfinedFlag = "--code-exec-unconfined";

        /// <summary>Seconds a single command may take.</summary>
        public const string TimeoutFlag = "--code-exec-timeout";

        /// <summary>Sampling temperature for a turn that can run code.</summary>
        public const string TemperatureFlag = "--code-exec-temperature";

        /// <summary>
        /// The hosts an install may reach. Promoted from an environment variable to a
        /// real flag, and paired with <see cref="AllowedPackagesFlag"/>: one bounds where
        /// a package may come from, the other which packages may be named.
        /// </summary>
        public const string InstallDomainsFlag = "--code-exec-install-domains";

        /// <summary>Which shell to run commands through, when the host's own choice is wrong.</summary>
        public const string ShellFlag = "--code-exec-shell";

        /// <summary>Restrict installs to a named set of packages.</summary>
        public const string AllowedPackagesFlag = "--code-exec-packages";

        /// <summary>How much of a command's output is kept.</summary>
        public const string MaxOutputFlag = "--code-exec-max-output";

        /// <summary>
        /// Spellings this feature used to accept, each pointing at what replaced it.
        ///
        /// <para>
        /// Both were retired when the tool surface became a shell, for the same reason:
        /// under a shell neither could be ENFORCED, and a switch that only looks like it
        /// confines is the single failure mode this whole area is written to avoid.
        /// </para>
        /// <list type="bullet">
        /// <item><c>--code-exec-languages</c> chose which languages to offer. A shell can
        /// reach every interpreter on PATH, and PATH must include <c>/bin</c> and
        /// <c>/usr/bin</c> for the shell itself to work, so the flag could not have been
        /// honoured. The host now REPORTS which interpreters it found instead of
        /// pretending to gate them.</item>
        /// </list>
        /// </summary>
        /// <remarks>
        /// <c>Advice</c> is a whole sentence rather than a flag name, because one of these
        /// has no replacement and pointing at an unrelated flag would be worse than saying
        /// so. An operator must never have to guess what to write instead — and must never
        /// be sent somewhere that does not do what they wanted.
        /// </remarks>
        public static readonly IReadOnlyList<(string Flag, string Survivor, string Advice)> RemovedFlags = new[]
        {
            ("--code-exec-languages", EnabledFlag,
                "It has no replacement. A shell reaches every interpreter on PATH, and PATH must "
                + $"contain /bin and /usr/bin for the shell to work at all, so {EnabledFlag} now "
                + "reports which interpreters this host has rather than pretending to gate them."),
        };

        /// <summary>Every flag of this family that takes no value.</summary>
        public static readonly IReadOnlyList<string> SwitchFlags = new[]
        {
            EnabledFlag, AllowInstallFlag, UnconfinedFlag,
        };

        /// <summary>Every flag of this family that takes a value.</summary>
        public static readonly IReadOnlyList<string> ValueFlags = new[]
        {
            TimeoutFlag, InstallDomainsFlag, ShellFlag, MaxOutputFlag, AllowedPackagesFlag,
            TemperatureFlag,
        };

        /// <summary>
        /// Packages an install may name. Empty means "any", which is only reachable when
        /// <see cref="AllowInstall"/> was turned on deliberately.
        ///
        /// <para>
        /// This is enforceable again. It was retired when the tool surface became a shell,
        /// because a model typing its own <c>pip install</c> could spell the request in
        /// ways no name list could see. It came back when installs moved BACK to the host:
        /// the model's command is read, not run, and the host builds the argument vector
        /// from names it has checked — so an allow-list applies to every spelling at once.
        /// </para>
        /// </summary>
        public IReadOnlyList<string> AllowedPackages { get; set; } = Array.Empty<string>();

        /// <summary>Whether the model is offered the <c>shell</c> tool at all.</summary>
        public bool Enabled { get; set; }

        /// <summary>
        /// Whether the host will install packages a command asks for.
        ///
        /// <para>
        /// Off even when <see cref="Enabled"/> is on, because reaching the network is the
        /// dangerous half. With it off, an attempted <c>pip install</c> is refused with a
        /// message naming this flag. With it on, the model's install command is READ
        /// rather than run: the host takes the package names out of it and performs the
        /// install itself, pointed at the egress proxy that admits
        /// <see cref="InstallDomains"/>. The command the model wrote still has no socket —
        /// no command ever does.
        /// </para>
        /// </summary>
        public bool AllowInstall { get; set; }

        /// <summary>
        /// Run even when the OS provides no real confinement.
        ///
        /// <para>
        /// The escape hatch for a developer on Windows, where a job object bounds CPU and
        /// memory but cannot restrict one file or one socket. It is refused by the server
        /// — a shared host must not be talked into running model-authored commands with
        /// the filesystem open — and on the CLI it is a deliberate act by the person whose
        /// machine it is.
        /// </para>
        /// </summary>
        public bool Unconfined { get; set; }

        /// <summary>
        /// How long one command may take before it is killed.
        ///
        /// <para>
        /// 120 seconds rather than the 30 the program runner used, because a shell command
        /// is now also how packages are installed and how a build or a test suite is run.
        /// A model may ask for less or more per call, bounded by <see cref="MaxTimeout"/>.
        /// </para>
        /// </summary>
        public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(120);

        /// <summary>The longest a single call may ask for, whatever it passes.</summary>
        public TimeSpan MaxTimeout { get; set; } = TimeSpan.FromMinutes(10);

        /// <summary>
        /// <see cref="MaxTimeout"/>, never below <see cref="Timeout"/> and never above what
        /// a millisecond wait can express.
        ///
        /// <para>
        /// An operator who sets <c>--code-exec-timeout 900</c> is asking for 900 seconds,
        /// so a 600-second cap on what a CALL may request would make the declaration say
        /// "default 900000, maximum 600000" — and a model asking for its stated default
        /// would get less than by saying nothing. The upper bound is the deadline's own
        /// representation: <c>WaitForExit(int)</c> takes milliseconds, and a larger value
        /// threw out of the tool dispatch instead of returning a result, leaving the child
        /// running.
        /// </para>
        /// </summary>
        public TimeSpan EffectiveMaxTimeout
        {
            get
            {
                TimeSpan max = MaxTimeout > Timeout ? MaxTimeout : Timeout;
                TimeSpan ceiling = TimeSpan.FromMilliseconds(int.MaxValue - 1);
                return max > ceiling ? ceiling : max;
            }
        }

        /// <summary>How long the host's own install-on-behalf-of-a-skill-script may take.</summary>
        public TimeSpan InstallTimeout { get; set; } = TimeSpan.FromSeconds(120);

        /// <summary>
        /// Temperature for a turn in which the model can run code, or null to leave the
        /// caller's sampling alone.
        ///
        /// <para>
        /// The built-in defaults this replaces are Ollama's CHAT defaults —
        /// <c>temperature 0.8, top_k 40, top_p 0.9, repeat_penalty 1.1</c> — inherited for
        /// API compatibility and never chosen for code. The reference implementation sets
        /// no sampling at all (<c>ModelSettings.temperature</c> and both penalties default
        /// to <c>None</c>), so applying chat defaults to code is the thing that diverges
        /// from it; this is a correction, not a new opinion.
        /// </para>
        /// <para>
        /// <b>Null by default, which means this changes nothing unless an operator asks.</b>
        /// That is deliberate, and it is the honest reading of the evidence rather than a
        /// hedge. Neither reference implementation has a code-specific sampling profile:
        /// the Agents SDK leaves <c>temperature</c>, <c>top_p</c> and both penalties at
        /// <c>None</c> and omits them from the request entirely, its only model defaults
        /// are keyed on the model NAME rather than the task, and Claude Code's settings
        /// surface has no temperature, top_p or top_k at all — its quality lever is
        /// reasoning effort. There is a real mechanical argument for the change (see
        /// <c>SamplingConfig.ForCodingTurn</c>, where the repetition penalty is the part
        /// that matters), and there is no precedent for making it the default, so the
        /// argument is offered as a switch instead of applied silently.
        /// </para>
        /// <para>
        /// Set a number between 0 and 2 to turn it on; 0.2 is the usual choice. Setting it
        /// also turns the repetition penalty off for coding turns, which is the half with
        /// the mechanical case behind it.
        /// </para>
        /// </summary>
        public float? Temperature { get; set; }

        /// <summary>
        /// Distinct packages the host may install and re-run for, within one command.
        ///
        /// <para>
        /// Deliberately NOT a flag. It bounds a recovery loop rather than granting a
        /// capability — what an operator can actually turn off is installing at all
        /// (<see cref="AllowInstallFlag"/>) and which packages are reachable
        /// (<see cref="AllowedPackagesFlag"/>), and both already gate this. A third
        /// spelling for "how hard should the host try" would be surface with no decision
        /// behind it.
        /// </para>
        /// <para>
        /// Five, because a program with six independent missing imports is a program
        /// whose dependencies the model should be naming up front, and because the loop
        /// re-runs the whole command each time. The same bound the skill-script runner
        /// uses, for the same reason.
        /// </para>
        /// </summary>
        public int MaxAutoInstalls { get; set; } = 5;

        /// <summary>
        /// Bytes of output kept from one command, head and tail with the middle dropped.
        ///
        /// <para>
        /// Middle-out rather than head-only because the tail of a build or a test run is
        /// where the failure is, and head-only truncation reliably discards exactly the
        /// part the model needed.
        /// </para>
        /// </summary>
        public int MaxOutputBytes { get; set; } = 32 * 1024;

        /// <summary>
        /// Hosts a host-performed install may reach, exact names or <c>*.suffix</c>
        /// wildcards. The default is the Python and npm registries and nothing else, so
        /// that "the install needs the network" stops meaning "the install gets the
        /// internet". Enforced where the sandbox can admit exactly the proxy's loopback
        /// port (macOS Seatbelt); elsewhere the installer is pointed at the proxy through
        /// <c>HTTPS_PROXY</c> and follows it, which narrows the install without confining
        /// it. An EMPTY value disables the pinning entirely.
        /// </summary>
        /// <summary>
        /// Whether <see cref="InstallDomains"/> came from the command line.
        ///
        /// <para>
        /// An environment variable is a DEFAULT, and a flag is a decision. Applying the
        /// variable unconditionally meant an operator who added
        /// <c>--code-exec-install-domains</c> to a service that still exported the old
        /// <c>TS_CODE_EXEC_INSTALL_DOMAINS</c> got the variable's list and no warning —
        /// their explicit choice silently discarded, on the one setting that decides where
        /// an installer may connect.
        /// </para>
        /// </summary>
        public bool InstallDomainsSpecified { get; private set; }

        public IReadOnlyList<string> InstallDomains { get; set; } = new[]
        {
            // Exact hosts, no wildcards. "*.pypi.org" also matches upload.pypi.org, and
            // an allowlist whose job is to name the few hosts an installer needs should
            // not quietly admit a sibling nobody considered. An operator who needs a
            // mirror names it; the default admits three hosts and nothing else.
            "pypi.org",
            "files.pythonhosted.org",
            "registry.npmjs.org",
        };

        /// <summary>Env var overriding <see cref="InstallDomains"/>.</summary>
        public const string InstallDomainsEnvVar = "TS_CODE_EXEC_INSTALL_DOMAINS";

        /// <summary>
        /// The shell to run commands through, or null to let the host choose: a POSIX
        /// shell on macOS and Linux, PowerShell on Windows. An operator sets this when
        /// the host's choice is absent or wrong — most usefully on Windows, where the
        /// only <c>bash</c> on a default PATH is the WSL launcher and running through it
        /// would put the workload in a VM that the job object cannot reach.
        /// </summary>
        public string? Shell { get; set; }

        /// <summary>Where session workspaces are created. Null means the system temp directory.</summary>
        public string? ScratchDirectory { get; set; }

        /// <summary>
        /// Where files a command produced are kept so a user can fetch them. Null means
        /// they are not kept at all, and the model is told so.
        /// </summary>
        public string? ArtifactDirectory { get; set; }

        /// <summary>
        /// URL prefix a server serves artifacts under, e.g. <c>/api/code/artifacts</c>.
        /// Null on the CLI, where the pointer handed back is the file's path on disk.
        /// </summary>
        public string? ArtifactUriPrefix { get; set; }

        /// <summary>How hard to insist on OS isolation. Mirrors the skills switch and defaults the same way.</summary>
        public SkillSandboxMode Sandbox { get; set; } = SkillSandboxMode.Required;

        /// <summary>True when nothing has been configured, so the feature is simply absent.</summary>
        public bool IsConfigured => Enabled || AllowInstall || Unconfined;

        // ---- parsing -------------------------------------------------------

        /// <summary>
        /// Refuse a retired spelling, naming what replaced it.
        ///
        /// <para>
        /// Runs BEFORE any flag is applied and before the host does any work, so an
        /// operator who has the old spelling in a script or a config file learns the new
        /// one at startup instead of watching a flag be silently ignored. The CLI's own
        /// argument switch has no unknown-flag trap, so without this a retired
        /// <c>--code-exec-packages</c> would simply do nothing at all.
        /// </para>
        /// </summary>
        /// <returns>Null when the line is clean, otherwise the message to print before exiting.</returns>
        public static string? RejectRemoved(IReadOnlyList<string>? args)
        {
            if (args == null)
                return null;

            foreach (string arg in args)
            {
                if (arg == null)
                    continue;

                // Match the value form too: "--code-exec-packages=numpy" must be refused
                // as loudly as the spaced form, or half the ways to write it slip through.
                int equals = arg.IndexOf('=');
                string name = equals >= 0 ? arg.Substring(0, equals) : arg;

                foreach ((string flag, string _, string advice) in RemovedFlags)
                {
                    if (!name.Equals(flag, StringComparison.OrdinalIgnoreCase))
                        continue;

                    return $"{flag} was removed when code execution became a shell tool. {advice} "
                         + "See the code execution section of --help.";
                }
            }

            return null;
        }

        /// <summary>Read the flags this feature owns out of a command line.</summary>
        /// <param name="args">The full command line.</param>
        /// <param name="remaining">Everything this did not consume, for the caller's own parser.</param>
        public static CodeExecOptions Parse(IReadOnlyList<string> args, out List<string> remaining)
        {
            var options = new CodeExecOptions();
            remaining = new List<string>();
            if (args == null)
                return options;

            for (int i = 0; i < args.Count; i++)
            {
                string arg = args[i];

                if (Matches(arg, EnabledFlag)) { options.Enabled = true; continue; }
                if (Matches(arg, AllowInstallFlag)) { options.AllowInstall = true; continue; }
                if (Matches(arg, UnconfinedFlag)) { options.Unconfined = true; continue; }

                if (TryValue(args, ref i, InstallDomainsFlag, out string? domains))
                {
                    options.InstallDomains = SplitList(domains);
                    options.InstallDomainsSpecified = true;
                    continue;
                }
                if (TryValue(args, ref i, AllowedPackagesFlag, out string? packages))
                {
                    options.AllowedPackages = SplitList(packages);
                    continue;
                }
                if (TryValue(args, ref i, ShellFlag, out string? shell))
                {
                    options.Shell = string.IsNullOrWhiteSpace(shell) ? null : shell!.Trim();
                    continue;
                }
                if (TryValue(args, ref i, TimeoutFlag, out string? timeout))
                {
                    // A value that will not parse is REFUSED, not swallowed. The old code
                    // consumed the value and fell through to `remaining`, so
                    // "--code-exec-timeout abc" left a bare "--code-exec-timeout" for the
                    // host's own parser to trip over and the number was silently the
                    // default — the operator's explicit choice quietly discarded.
                    if (!TryPositiveInt(timeout, out int seconds))
                        throw new ArgumentException($"{TimeoutFlag} needs a positive whole number of seconds, not '{timeout}'.");
                    options.Timeout = TimeSpan.FromSeconds(seconds);
                    continue;
                }
                if (TryValue(args, ref i, TemperatureFlag, out string? temperature))
                {
                    // Refused rather than swallowed, exactly as the timeout flag is: a
                    // value that will not parse must never leave the operator's explicit
                    // choice silently replaced by a default.
                    if (!float.TryParse(
                            temperature, System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out float value)
                        || value > 2.0f)
                    {
                        throw new ArgumentException(
                            $"{TemperatureFlag} needs a number between 0 and 2 (or a negative number to "
                            + $"leave sampling alone), not '{temperature}'.");
                    }
                    options.Temperature = value < 0f ? null : value;
                    continue;
                }
                if (TryValue(args, ref i, MaxOutputFlag, out string? maxOutput))
                {
                    if (!TryPositiveInt(maxOutput, out int bytes))
                        throw new ArgumentException($"{MaxOutputFlag} needs a positive whole number of bytes, not '{maxOutput}'.");
                    options.MaxOutputBytes = bytes;
                    continue;
                }

                remaining.Add(arg);
            }

            return options;
        }

        /// <summary>Apply environment overrides for anything the command line did not set.</summary>
        public void ApplyEnvironment()
        {
            string? installDomains = Environment.GetEnvironmentVariable(InstallDomainsEnvVar);
            if (installDomains != null && !InstallDomainsSpecified)
            {
                InstallDomains = installDomains
                    .Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            }

            if (!Enabled && IsEnvOn(EnabledEnvVar))
                Enabled = true;
            if (!AllowInstall && IsEnvOn(AllowInstallEnvVar))
                AllowInstall = true;
        }

        private static bool IsEnvOn(string name) =>
            Environment.GetEnvironmentVariable(name) is { } value
            && value.Length > 0
            && !value.Equals("0", StringComparison.Ordinal);

        private static bool Matches(string arg, string flag) =>
            arg.Equals(flag, StringComparison.OrdinalIgnoreCase);

        private static bool TryPositiveInt(string? value, out int parsed) =>
            int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsed) && parsed > 0;

        /// <summary>Accepts both <c>--flag value</c> and <c>--flag=value</c>.</summary>
        private static bool TryValue(IReadOnlyList<string> args, ref int i, string flag, out string? value)
        {
            string arg = args[i];
            if (arg.Equals(flag, StringComparison.OrdinalIgnoreCase))
            {
                if (i + 1 < args.Count)
                {
                    value = args[++i];
                    return true;
                }
                value = null;
                return false;
            }

            if (arg.StartsWith(flag + "=", StringComparison.OrdinalIgnoreCase))
            {
                value = arg.Substring(flag.Length + 1);
                return true;
            }

            value = null;
            return false;
        }

        private static List<string> SplitList(string? value) =>
            (value ?? string.Empty)
                .Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .Where(s => s.Length > 0)
                .ToList();

        /// <summary>Map what an operator might write to a language we can locate.</summary>
        public static CodeLanguage ParseLanguage(string? name) => (name ?? string.Empty).Trim().ToLowerInvariant() switch
        {
            "python" or "python3" or "py" => CodeLanguage.Python,
            "javascript" or "js" or "node" or "nodejs" => CodeLanguage.JavaScript,
            "shell" or "sh" or "bash" or "pwsh" or "powershell" => CodeLanguage.Shell,
            _ => CodeLanguage.Unknown,
        };

        /// <summary>The name shown for a language.</summary>
        public static string NameOf(CodeLanguage language) => language switch
        {
            CodeLanguage.Python => "python",
            CodeLanguage.JavaScript => "javascript",
            CodeLanguage.Shell => "shell",
            _ => "unknown",
        };

        /// <summary>Whether a language has a dependency installer at all.</summary>
        public static bool SupportsInstall(CodeLanguage language) =>
            language is CodeLanguage.Python or CodeLanguage.JavaScript;
    }
}
