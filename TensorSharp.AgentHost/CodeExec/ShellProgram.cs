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

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>Which family of shell we are talking to. Everything platform-specific hangs off this.</summary>
    public enum ShellKind
    {
        /// <summary>bash, sh, zsh — the shell of macOS and Linux.</summary>
        Posix,

        /// <summary>Windows PowerShell or PowerShell 7+.</summary>
        PowerShell,
    }

    /// <summary>
    /// The shell this host runs model-written commands through: where it is, which
    /// dialect it speaks, and how to hand it a script.
    ///
    /// <para>
    /// A single resolved shell per host rather than a per-call choice, because the model
    /// should not have to guess what it is talking to — the prompt states the dialect and
    /// the cheat sheet matches it. Guessing wrong costs a round on every host where the
    /// guess was wrong, which on Windows is all of them.
    /// </para>
    /// </summary>
    public sealed class ShellProgram
    {
        private ShellProgram(string path, ShellKind kind, string name)
        {
            Path = path;
            Kind = kind;
            Name = name;
        }

        /// <summary>Absolute path to the interpreter.</summary>
        public string Path { get; }

        /// <summary>Which dialect it speaks.</summary>
        public ShellKind Kind { get; }

        /// <summary>Its bare name, for the prompt and for logs: <c>bash</c>, <c>pwsh</c>.</summary>
        public string Name { get; }

        /// <summary>The file extension a script for this shell is written with.</summary>
        public string ScriptExtension => Kind == ShellKind.PowerShell ? ".ps1" : ".sh";

        /// <summary>What the model should be told it is typing into.</summary>
        public string DialectName => Kind == ShellKind.PowerShell ? "PowerShell" : Name;

        /// <summary>
        /// Candidates in preference order. POSIX hosts prefer bash over sh because the
        /// prompt's cheat sheet and every heredoc example assume bash; Windows prefers
        /// PowerShell 7 (<c>pwsh</c>) over Windows PowerShell because 5.1's parser
        /// rejects things models write routinely, such as <c>&amp;&amp;</c>.
        /// </summary>
        private static readonly string[] WindowsCandidates = { "pwsh.exe", "powershell.exe" };
        private static readonly string[] PosixCandidates = { "bash", "sh" };

        /// <summary>
        /// Find the shell to use, or explain why this host cannot offer one.
        /// </summary>
        /// <param name="requested">
        /// An operator's <c>--code-exec-shell</c>, or null to let the host choose.
        /// </param>
        public static bool TryResolve(string? requested, out ShellProgram? shell, out string? error)
        {
            shell = null;
            error = null;

            if (!string.IsNullOrWhiteSpace(requested))
            {
                string wanted = requested!.Trim();
                string? resolved = System.IO.Path.IsPathRooted(wanted) && File.Exists(wanted)
                    ? wanted
                    : CodeEnvironment.Which(wanted);
                if (resolved == null)
                {
                    error = $"the shell named by {CodeExecOptions.ShellFlag} was not found: '{wanted}'";
                    return false;
                }
                if (!TryAccept(resolved, out shell, out error))
                    return false;
                return true;
            }

            string[] candidates = OperatingSystem.IsWindows() ? WindowsCandidates : PosixCandidates;
            var refusals = new List<string>();
            foreach (string candidate in candidates)
            {
                string? found = CodeEnvironment.Which(candidate);
                if (found == null)
                    continue;
                if (TryAccept(found, out shell, out string? why))
                    return true;
                if (why != null)
                    refusals.Add(why);
            }

            error = refusals.Count > 0
                ? string.Join("; ", refusals)
                : $"no shell was found on PATH (looked for: {string.Join(", ", candidates)}). "
                  + $"Install one, or name it with {CodeExecOptions.ShellFlag}";
            return false;
        }

        private static bool TryAccept(string path, out ShellProgram? shell, out string? error)
        {
            shell = null;
            error = null;

            string name = System.IO.Path.GetFileNameWithoutExtension(path);
            ShellKind kind = name.Equals("pwsh", StringComparison.OrdinalIgnoreCase)
                          || name.Equals("powershell", StringComparison.OrdinalIgnoreCase)
                ? ShellKind.PowerShell
                : ShellKind.Posix;

            // Windows, the constraint that decides this whole design: the only `bash` on a
            // default PATH is System32's WSL launcher. Running through it would put the
            // workload inside a Linux VM while the job object holds only the launcher on
            // the Windows side — a process "contained" by an object that no longer has
            // anything to contain. Refuse by name rather than work around it, and say what
            // to do instead. An operator who has a real bash (Git Bash, MSYS2) can name it
            // with --code-exec-shell, and it is accepted here because the path test below
            // only catches the System32 shim.
            if (kind == ShellKind.Posix && OperatingSystem.IsWindows() && IsWslLauncher(path))
            {
                error = "the only 'bash' on PATH is the WSL launcher, which would run commands "
                      + "inside a Linux VM where this host's sandbox cannot reach them. This host "
                      + $"uses PowerShell instead; to use a native bash (Git Bash or MSYS2) name it "
                      + $"with {CodeExecOptions.ShellFlag}";
                return false;
            }

            shell = new ShellProgram(path, kind, name);
            return true;
        }

        /// <summary>
        /// True when this <c>bash.exe</c> is Windows' WSL shim rather than a real shell.
        ///
        /// <para>
        /// The shim lives in the protected System32 directory, so a path test is both the
        /// cheapest and the most reliable signal — running it to ask would already have
        /// started the VM.
        /// </para>
        /// </summary>
        internal static bool IsWslLauncher(string path)
        {
            string system32 = Environment.GetFolderPath(Environment.SpecialFolder.System);
            return system32.Length > 0
                && path.StartsWith(system32, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// The argument vector that runs <paramref name="scriptPath"/>.
        ///
        /// <para>
        /// A FILE rather than a <c>-c</c> string, for both dialects. A model's command can
        /// be several kilobytes of heredoc, can contain any byte, and on Windows would
        /// otherwise be re-parsed by PowerShell after Windows' own command-line quoting
        /// had already had a go at it. A file crosses that boundary once, as a path.
        /// </para>
        /// <para>
        /// <c>-c</c> rather than <c>-lc</c> on POSIX: a login shell sources profile scripts
        /// that would change PATH behind the host's back and print their own output into
        /// the model's result. The host sets PATH deliberately, so a shell that adds to it
        /// is a shell whose behaviour depends on whose machine it is.
        /// </para>
        /// </summary>
        public IReadOnlyList<string> ArgumentsFor(string scriptPath) =>
            Kind == ShellKind.PowerShell
                ? new[] { "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", scriptPath }
                : new[] { scriptPath };
    }
}
