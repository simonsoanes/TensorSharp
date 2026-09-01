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
using System.IO;
using System.Linq;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>How to build, populate and launch one language's environment.</summary>
    /// <param name="Interpreter">Absolute path to the interpreter that runs the snippet.</param>
    /// <param name="Arguments">Extra arguments before the script path, if any.</param>
    /// <param name="ReadablePaths">Paths the run phase must be able to read, e.g. the environment.</param>
    /// <param name="EnvironmentVariables">Variables the run phase needs, e.g. NODE_PATH.</param>
    public readonly record struct CodeLaunchPlan(
        string Interpreter,
        IReadOnlyList<string> Arguments,
        IReadOnlyList<string> ReadablePaths,
        IReadOnlyDictionary<string, string> EnvironmentVariables);

    /// <summary>An install step to run with the network open and nothing else reachable.</summary>
    /// <param name="Interpreter">The installer executable.</param>
    /// <param name="Arguments">Its full argument vector.</param>
    /// <param name="WriteDirectory">The only directory it may write to.</param>
    public readonly record struct CodeInstallPlan(
        string Interpreter,
        IReadOnlyList<string> Arguments,
        string WriteDirectory);

    /// <summary>
    /// Locates interpreters and works out how to build an environment for each language.
    ///
    /// <para>
    /// Everything here is about the gap between "the interpreter is installed" and "we can
    /// launch it under a sandbox". Two of those gaps are Windows-only and both would
    /// quietly defeat the confinement rather than fail loudly, which is why they are
    /// refused by name rather than worked around.
    /// </para>
    /// </summary>
    public static class CodeEnvironment
    {
        /// <summary>Names to try for each language's interpreter, in order.</summary>
        private static readonly Dictionary<CodeLanguage, string[]> Candidates = new()
        {
            // Newest first: skill scripts routinely use post-3.9 syntax (`match`,
            // PEP 604 unions), and on macOS the bare `python3` is Apple's frozen 3.9.
            // A host with a modern interpreter installed should never fail on syntax
            // the script's own authors consider baseline.
            [CodeLanguage.Python] = OperatingSystem.IsWindows()
                ? new[] { "python.exe", "python3.exe" }
                : new[] { "python3.14", "python3.13", "python3.12", "python3.11", "python3.10", "python3", "python" },
            [CodeLanguage.JavaScript] = OperatingSystem.IsWindows()
                ? new[] { "node.exe" }
                : new[] { "node" },
        };

        /// <summary>
        /// Find the interpreter for <paramref name="language"/>, or explain why it cannot
        /// be used here.
        /// </summary>
        public static bool TryResolveInterpreter(
            CodeLanguage language, out string? path, out string? error)
        {
            path = null;
            error = null;

            if (!Candidates.TryGetValue(language, out string[]? names))
            {
                error = $"'{CodeExecOptions.NameOf(language)}' is not a language this host can run";
                return false;
            }

            foreach (string name in names)
            {
                string? found = Which(name);
                if (found == null)
                    continue;

                path = found;
                return true;
            }

            error = $"no interpreter for {CodeExecOptions.NameOf(language)} was found on PATH "
                  + $"(looked for: {string.Join(", ", names)})";
            return false;
        }

        /// <summary>
        /// The major.minor of a Python interpreter, probed once per path and cached.
        /// Null when the probe fails. Used to turn "SyntaxError: invalid syntax" on a
        /// 3.9 host into "your Python is too old", which is actionable.
        /// </summary>
        public static Version? PythonVersionOf(string interpreter)
        {
            return InterpreterVersions.GetOrAdd(interpreter, path =>
            {
                try
                {
                    // Old Pythons print the version to stderr and new ones to stdout, so
                    // both are collected and searched together.
                    var output = new System.Text.StringBuilder();
                    void Collect(string line) { lock (output) output.Append(line).Append('\n'); }

                    var request = new SpawnRequest
                    {
                        FileName = path,
                        Arguments = new[] { "--version" },
                        Environment = new Dictionary<string, string>(StringComparer.Ordinal)
                        {
                            ["PATH"] = Environment.GetEnvironmentVariable("PATH") ?? string.Empty,
                            ["LANG"] = "C.UTF-8",
                        },
                        OnStdoutLine = Collect,
                        OnStderrLine = Collect,
                    };

                    // The caching is why a hang here would matter more than a one-line probe
                    // suggests: whatever the first caller concluded is memoised for every
                    // caller after it.
                    if (!SpawnedProcess.TryStart(request, out SpawnedProcess? started, out _)
                        || started == null)
                    {
                        return null;
                    }

                    using var process = started;
                    process.WaitForExit(5000);
                    process.WaitForDrain(1000);

                    string text;
                    lock (output) text = output.ToString();
                    var match = System.Text.RegularExpressions.Regex.Match(text, @"Python (\d+)\.(\d+)");
                    return match.Success
                        ? new Version(int.Parse(match.Groups[1].Value), int.Parse(match.Groups[2].Value))
                        : null;
                }
                catch (Exception)
                {
                    return null;
                }
            });
        }

        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, Version?> InterpreterVersions = new();

        /// <summary>
        /// The interpreters and tools this host actually has, named for the model.
        ///
        /// <para>
        /// This is what replaced <c>--code-exec-languages</c>. A shell reaches every
        /// interpreter on PATH and PATH has to contain <c>/bin</c> and <c>/usr/bin</c> for
        /// the shell to work at all, so a flag claiming to gate languages could never have
        /// been honoured. Reporting is the honest version of the same information — and it
        /// is worth more to the model than the flag ever was, because a model that is told
        /// what exists does not spend a round running <c>which python3</c> to find out, and
        /// does not write a program for an interpreter this host has never had.
        /// </para>
        /// <para>
        /// Probed once and cached: the answer cannot change while the process runs, and the
        /// result goes into a tool description that sits in prefix-cache block zero, where
        /// anything that varies between turns costs the whole conversation its prefix reuse.
        /// </para>
        /// </summary>
        public static IReadOnlyList<string> AvailableTools => Available.Value;

        private static readonly Lazy<IReadOnlyList<string>> Available = new(ProbeAvailable);

        private static IReadOnlyList<string> ProbeAvailable()
        {
            var found = new List<string>();

            if (TryResolveInterpreter(CodeLanguage.Python, out string? python, out _) && python != null)
            {
                // Only when the NAME does not already carry it: "python3.12 3.12" is noise,
                // while a bare "python3" that is really 3.9 is exactly what a model needs to
                // know before it writes a match statement.
                string name = System.IO.Path.GetFileName(python);
                Version? version = PythonVersionOf(python);
                found.Add(version != null && !name.Any(char.IsAsciiDigit)
                    ? name + " " + version
                    : version != null && !name.Contains(version.ToString(), StringComparison.Ordinal)
                        ? name + " (" + version + ")"
                        : name);
            }
            if (TryResolveInterpreter(CodeLanguage.JavaScript, out string? node, out _) && node != null)
                found.Add(System.IO.Path.GetFileName(node));

            // Tools a model reaches for constantly and cannot install: naming the ones that
            // are here is cheaper than a failed command, and NOT naming the rest is what
            // stops it planning around ffmpeg on a host that has none.
            // Deliberately no curl or wget: they exist on nearly every host and nothing
            // here can reach the network, so naming them would invite exactly the plan
            // ("I will just fetch it") that the network rule is there to prevent.
            // pdftoppm is here because its ABSENCE cost two logged rounds: the pdf and
            // pptx skills tell the model to render a page to an image, and a program
            // cannot be installed by any package manager here — so discovering it is
            // missing three commands into a plan is a wasted plan, not a fixable error.
            foreach (string name in new[]
            {
                "git", "rg", "jq", "ffmpeg", "pandoc", "soffice", "pdftoppm",
                "dotnet", "go", "cargo", "make",
            })
            {
                if (Which(name) != null)
                    found.Add(name);
            }

            return found;
        }

        /// <summary>Resolve a bare executable name against PATH.</summary>
        public static string? Which(string name) =>
            Path.IsPathRooted(name)
                ? (File.Exists(name) ? name : null)
                : WhichCache.GetOrAdd(name, ProbePath);

        /// <summary>
        /// Answers remembered for the life of the process.
        ///
        /// <para>
        /// A lookup walks every PATH entry plus the two Homebrew directories — a filesystem
        /// probe per candidate — and it is on paths that run per call: the syntax check
        /// resolves two interpreters every time it verifies a file, and the
        /// spelled-differently diagnosis probes on every command-not-found. What is on
        /// PATH does not change while the server runs, and the interpreter versions beside
        /// this are cached for exactly the same reason.
        /// </para>
        /// </summary>
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, string?> WhichCache =
            new(StringComparer.Ordinal);

        private static string? ProbePath(string name)
        {

            string? pathVar = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathVar))
            {
                foreach (string dir in pathVar.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
                {
                    string candidate;
                    try { candidate = Path.Combine(dir.Trim(), name); }
                    catch (ArgumentException) { continue; }      // a malformed PATH entry

                    if (File.Exists(candidate))
                        return candidate;
                }
            }

            // A server launched outside a login shell (launchd, a service manager, a
            // stripped environment) misses Homebrew's directory even though the
            // interpreter is right there. These two are where user-installed tools
            // land on macOS; checking them beats "no interpreter found" on a machine
            // that plainly has one.
            if (!OperatingSystem.IsWindows())
            {
                foreach (string dir in new[] { "/opt/homebrew/bin", "/usr/local/bin" })
                {
                    string candidate = Path.Combine(dir, name);
                    if (File.Exists(candidate))
                        return candidate;
                }
            }

            return null;
        }

        // ---- python --------------------------------------------------------

        /// <summary>Where a venv puts its executables — <c>Scripts</c> on Windows, <c>bin</c> elsewhere.</summary>
        public static string VenvBin(string envDirectory) =>
            Path.Combine(envDirectory, OperatingSystem.IsWindows() ? "Scripts" : "bin");

        /// <summary>
        /// Install packages into <paramref name="envDirectory"/> using the HOST interpreter's
        /// pip with an explicit target, rather than a pip inside the venv.
        ///
        /// <para>
        /// <c>--only-binary=:all:</c> is the load-bearing argument and the reason this is
        /// safe enough to offer at all. Without it pip will happily fetch a source
        /// distribution and execute its <c>setup.py</c> — arbitrary code, running as the
        /// host user, before a single line of the model's own snippet is reached, and
        /// before any sandbox that confines the RUN phase applies. With it, only prebuilt
        /// wheels are accepted: they are unpacked, never executed. A package with no wheel
        /// for this platform fails to install, and that is the correct trade.
        /// </para>
        /// <para>
        /// <c>--no-input</c> and <c>--disable-pip-version-check</c> keep it from ever
        /// blocking on a prompt, which would otherwise consume the whole install timeout.
        /// </para>
        /// </summary>
        public static CodeInstallPlan PythonInstall(
            string interpreter, string envDirectory, IReadOnlyList<string> packages)
        {
            var args = new List<string>
            {
                "-m", "pip", "install",
                "--only-binary=:all:",
                "--no-input",
                "--disable-pip-version-check",
                "--no-warn-script-location",
                "--target", envDirectory,
            };
            args.AddRange(packages);
            return new CodeInstallPlan(interpreter, args, envDirectory);
        }

        // ---- javascript ----------------------------------------------------

        /// <summary>
        /// The npm install step, launched as <c>node npm-cli.js</c> rather than <c>npm</c>.
        ///
        /// <para>
        /// Windows, problem two: <c>npm</c> is <c>npm.cmd</c>, a batch file, and
        /// <see cref="System.Diagnostics.ProcessStartInfo"/> with
        /// <c>UseShellExecute = false</c> — which every confined launch here uses — cannot
        /// start one. The usual workaround is to launch <c>cmd.exe /c npm</c>, which puts a
        /// command interpreter inside the sandbox with the model's package names as its
        /// arguments; that is a shell injection surface introduced purely to run an
        /// installer. Invoking npm's own entry script through <c>node</c> avoids the batch
        /// file, the shell, and the difference between platforms all at once.
        /// </para>
        /// </summary>
        public static bool TryNpmInstall(
            string nodeInterpreter,
            string envDirectory,
            IReadOnlyList<string> packages,
            out CodeInstallPlan plan,
            out string? error)
        {
            plan = default;
            string? cli = FindNpmCli(nodeInterpreter);
            if (cli == null)
            {
                error = "npm's entry script (npm-cli.js) was not found next to node, so packages "
                      + "cannot be installed for javascript on this host";
                return false;
            }

            var args = new List<string>
            {
                cli, "install",
                "--prefix", envDirectory,
                "--no-audit", "--no-fund", "--no-package-lock",
                // Node's equivalent of the wheels-only rule: a package's install scripts
                // are arbitrary code running as the host user, before the sandbox that
                // confines the run phase ever applies.
                "--ignore-scripts",
            };
            args.AddRange(packages);

            plan = new CodeInstallPlan(nodeInterpreter, args, envDirectory);
            error = null;
            return true;
        }

        /// <summary>Locate <c>npm-cli.js</c> relative to the node executable.</summary>
        private static string? FindNpmCli(string nodeInterpreter)
        {
            string? dir = Path.GetDirectoryName(nodeInterpreter);
            if (string.IsNullOrEmpty(dir))
                return null;

            // Windows keeps npm beside node.exe; Unix installs put it one level up in lib/.
            string[] candidates =
            {
                Path.Combine(dir, "node_modules", "npm", "bin", "npm-cli.js"),
                Path.Combine(dir, "..", "lib", "node_modules", "npm", "bin", "npm-cli.js"),
                Path.Combine(dir, "..", "node_modules", "npm", "bin", "npm-cli.js"),
            };

            return candidates.Select(c => Path.GetFullPath(c)).FirstOrDefault(File.Exists);
        }

        // ---- shell ---------------------------------------------------------

    }
}
