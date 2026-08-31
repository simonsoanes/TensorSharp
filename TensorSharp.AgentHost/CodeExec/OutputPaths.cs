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
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// Put every path a command printed back into the model's own frame of reference:
    /// relative to the directory the command ran in.
    ///
    /// <para>
    /// This is the same idea as <c>ShellRunner.Rewrite</c>, which turns the wrapper
    /// script's path into the word "command", carried to its conclusion: the host's
    /// absolute layout is the host's business, and every mention of it in a tool result
    /// is noise the model pays for and can act on wrongly.
    /// </para>
    /// <para>
    /// <b>Measured, not assumed.</b> Across every tool result recoverable from this
    /// server's own logs (377 distinct results, 145,031 characters),
    /// <b>13.9% of the characters were absolute workspace paths</b> — around 5,000
    /// tokens of pure prefix, 113 occurrences across 31 sessions. A single Python
    /// traceback frame spends 120 characters saying
    /// <c>/Users/…/bin/code-scratch/ts-session-7bb2032808b74a2fb145aefafab0aa84/code/main-01086d35.py</c>
    /// to name a file the model calls <c>solve.py</c>, and a traceback has several frames.
    /// </para>
    /// <para>
    /// <b>And it is not only noise.</b> In one logged session the model read two of those
    /// paths out of two different results, spliced them, and ran
    /// <c>cd …/ts-session-7bb203b8702657bd</c> — a session id that never existed. The
    /// round was lost to a string the model should never have been shown. A 32-hex-digit
    /// identifier repeated in front of every filename is an invitation to exactly that
    /// mistake.
    /// </para>
    /// <para>
    /// <b>Nothing is lost by rewriting.</b> Each replacement is a path that still WORKS:
    /// it is computed with <see cref="Path.GetRelativePath(string,string)"/> against the
    /// directory the command actually ran in, so a model that copies
    /// <c>../env/pptx/__init__.py</c> out of a traceback into its next command addresses
    /// the same file the absolute form did. The whole workspace is inside the sandbox, so
    /// a relative path out of the work directory resolves exactly as the absolute one
    /// would have. Absolute paths, by contrast, are refused outright by
    /// <c>apply_patch</c> and by every host-side path resolution — so the form being
    /// removed is the form the model was never allowed to use.
    /// </para>
    /// </summary>
    public static class OutputPaths
    {
        /// <summary>
        /// Rewrite every absolute path inside <paramref name="workspace"/> that appears in
        /// <paramref name="text"/> as a path relative to <paramref name="from"/>.
        /// </summary>
        /// <param name="text">Output as the command produced it.</param>
        /// <param name="workspace">The session workspace, or null to leave the text alone.</param>
        /// <param name="from">
        /// The directory the command ran in, which is what its relative paths are relative
        /// to. Null or outside the workspace falls back to the work directory.
        /// </param>
        public static string Scrub(string text, SessionWorkspace? workspace, string? from = null)
        {
            if (workspace == null || string.IsNullOrEmpty(text))
                return text;

            // The early-out that keeps this free on the common path. Every directory this
            // rewrites is inside Root, so one IndexOf over the output settles whether
            // there is anything to do at all — and for most commands there is not.
            string root = Normalize(workspace.Root);
            if (!Mentions(text, root))
                return text;

            string origin = from is { Length: > 0 } && Contains(root, Normalize(from))
                ? Normalize(from)
                : Normalize(workspace.WorkDirectory);

            // Every directory in every spelling this host might print it in, and then
            // LONGEST SPELLING FIRST — which is two separate reasons to sort, both of
            // which produced a wrong answer when they were missed.
            //
            // Across directories: the work, env, state and temp directories are all
            // inside Root, so replacing Root first leaves "../work/solve.py" behind
            // instead of "solve.py" — the host prefix is gone but the noise is not.
            //
            // Across spellings of ONE directory: on macOS "/var/folders/…/work" is a
            // substring of "/private/var/folders/…/work", so rewriting the short
            // spelling first turns "/private/var/…/work/deck.py" into
            // "/privatedeck.py" — a path that names nothing, produced from one that
            // named the right file.
            var replacements = new List<(string Spelling, string Relative)>();
            foreach (string directory in new[]
            {
                Normalize(workspace.WorkDirectory),
                Normalize(workspace.EnvDirectory),
                Normalize(workspace.StateDirectory),
                Normalize(workspace.TempDirectory),
                root,
            })
            {
                string relative = RelativeTo(origin, directory);
                foreach (string spelling in Spellings(directory))
                    replacements.Add((spelling, relative));
            }
            replacements.Sort(static (a, b) => b.Spelling.Length.CompareTo(a.Spelling.Length));

            foreach ((string spelling, string relative) in replacements)
                text = ReplaceDirectory(text, spelling, relative);

            return text;
        }

        /// <summary>
        /// What <paramref name="directory"/> is called from <paramref name="origin"/>.
        /// "." for the directory itself, so a command that printed its own working
        /// directory still prints a path that means the same thing.
        /// </summary>
        private static string RelativeTo(string origin, string directory)
        {
            string relative = Path.GetRelativePath(origin, directory);
            return relative.Length == 0 ? "." : relative.Replace('\\', '/');
        }

        /// <summary>
        /// Replace <paramref name="directory"/> wherever it heads a path, and on its own.
        ///
        /// <para>
        /// Two cases, and they need different replacements. As a PREFIX
        /// (<c>…/work/solve.py</c>) the separator goes too, so the result is the bare
        /// relative path — <c>solve.py</c>, not <c>./solve.py</c>, which is what the model
        /// wrote and what it will recognise. STANDALONE (<c>cd: …/work: not found</c>) the
        /// directory is the whole path, and the replacement has to be a path in its own
        /// right, which is why <see cref="RelativeTo"/> returns "." rather than "".
        /// </para>
        /// </summary>
        private static string ReplaceDirectory(string text, string directory, string replacement)
        {
            if (!Mentions(text, directory))
                return text;

            StringComparison comparison = Comparison;
            var sb = new System.Text.StringBuilder(text.Length);
            int index = 0;
            while (index < text.Length)
            {
                int hit = text.IndexOf(directory, index, comparison);
                if (hit < 0)
                {
                    sb.Append(text, index, text.Length - index);
                    break;
                }

                sb.Append(text, index, hit - index);
                int after = hit + directory.Length;

                // A directory whose name merely BEGINS with this one is a different
                // directory: rewriting the prefix of ".../work-2/x" would silently
                // rename a real path. Only a separator or a boundary counts.
                if (after < text.Length && (text[after] == '/' || text[after] == '\\'))
                {
                    sb.Append(replacement == "." ? string.Empty : replacement + "/");
                    index = after + 1;
                }
                else if (after >= text.Length || !IsPathChar(text[after]))
                {
                    sb.Append(replacement);
                    index = after;
                }
                else
                {
                    sb.Append(text, hit, directory.Length);
                    index = after;
                }
            }
            return sb.ToString();
        }

        private static bool IsPathChar(char c) =>
            char.IsLetterOrDigit(c) || c == '_' || c == '-' || c == '.';

        /// <summary>
        /// Every way this host might print the same directory.
        ///
        /// <para>
        /// On macOS the per-user temp root is reached as both <c>/var/folders/…</c> and
        /// <c>/private/var/folders/…</c> — one is a symlink to the other, and which one a
        /// process prints depends on how it got the path.
        /// <see cref="Path.GetFullPath(string)"/> resolves neither, so a rewrite that
        /// knows only the spelling the host holds misses every mention in the output. On
        /// Windows a tool is equally free to print forward slashes.
        /// </para>
        /// </summary>
        private static IEnumerable<string> Spellings(string directory)
        {
            yield return directory;

            if (OperatingSystem.IsMacOS())
            {
                if (directory.StartsWith("/private/", StringComparison.Ordinal))
                    yield return directory.Substring("/private".Length);
                else if (directory.StartsWith("/var/", StringComparison.Ordinal)
                      || directory.StartsWith("/tmp/", StringComparison.Ordinal))
                    yield return "/private" + directory;
            }

            if (OperatingSystem.IsWindows() && directory.Contains('\\', StringComparison.Ordinal))
                yield return directory.Replace('\\', '/');
        }

        private static string Normalize(string path) =>
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(path));

        private static bool Contains(string root, string candidate) =>
            candidate.StartsWith(root, Comparison);

        private static bool Mentions(string text, string directory) =>
            text.Contains(directory, Comparison);

        /// <summary>
        /// Case-sensitive only where the filesystem is. A macOS or Windows tool is free to
        /// print a path whose case differs from the one the host stored.
        /// </summary>
        private static StringComparison Comparison =>
            OperatingSystem.IsLinux() ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;
    }
}
