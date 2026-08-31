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
using System.Runtime.InteropServices;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// Confines every skill file access to the skill's own directory.
    ///
    /// <para>
    /// This is the single security boundary of the whole skills feature. A skill
    /// is untrusted content — a user uploads a ZIP, or points the server at a
    /// directory of skills pulled off GitHub — and the model is then invited to
    /// name files inside it. Everything the model asks to read, and every relative
    /// path a <c>SKILL.md</c> body links to, goes through
    /// <see cref="TryResolve"/> before it reaches the filesystem. Without that, a
    /// <c>SKILL.md</c> that says "read <c>../../../../etc/passwd</c>" or a ZIP
    /// entry named <c>../../authorized_keys</c> would turn a skill upload into
    /// arbitrary host file read/write.
    /// </para>
    /// <para>
    /// Three separate escapes are closed, because closing only the obvious one is
    /// what makes this class worth having:
    /// </para>
    /// <list type="number">
    /// <item><b>Lexical</b> — <c>..</c> segments, absolute paths, rooted paths
    ///   (<c>/etc</c>), Windows drive-qualified paths (<c>C:\</c>) and UNC paths
    ///   (<c>\\server\share</c>). Rejected before touching the disk.</item>
    /// <item><b>Canonical</b> — after <see cref="Path.GetFullPath(string)"/>
    ///   collapses the path, the result must still sit under the root. This catches
    ///   the cases where a <c>..</c> survives normalisation on one platform's rules
    ///   but not another's.</item>
    /// <item><b>Symbolic</b> — a symlink inside the skill directory pointing out of
    ///   it. <see cref="Path.GetFullPath(string)"/> does not follow links, so a
    ///   skill containing <c>references/host -> /</c> would otherwise pass the first
    ///   two checks and then read anything. Every existing component is resolved to
    ///   its final target and re-checked.</item>
    /// </list>
    /// </summary>
    public static class SkillPathGuard
    {
        /// <summary>
        /// Path comparison follows the host filesystem's own case rule: Linux
        /// filesystems distinguish <c>Scripts</c> from <c>scripts</c> and Windows and
        /// macOS (on the default case-insensitive volume) do not. Comparing
        /// case-sensitively everywhere would reject legitimate reads on macOS; comparing
        /// case-insensitively everywhere would let a Linux path that merely looks like
        /// the root prefix pass the containment test.
        /// </summary>
        public static readonly StringComparison PathComparison =
            OperatingSystem.IsLinux() ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;

        /// <summary>
        /// Resolve <paramref name="relativePath"/> against <paramref name="skillRoot"/>.
        /// </summary>
        /// <param name="skillRoot">
        /// The skill's own directory. Need not exist yet (a ZIP is validated before
        /// it is written), but must be an absolute path.
        /// </param>
        /// <param name="relativePath">
        /// A path as written in a <c>SKILL.md</c> body, a ZIP entry name, or a
        /// <c>skills_read</c> tool argument. Both separators are accepted, since a
        /// skill authored on Linux is routinely read on Windows and vice versa.
        /// </param>
        /// <param name="fullPath">The resolved absolute path, or null on failure.</param>
        /// <param name="error">A message naming what was wrong, or null on success.</param>
        /// <returns>True when the path is safe to open.</returns>
        public static bool TryResolve(
            string skillRoot,
            string? relativePath,
            out string? fullPath,
            out string? error)
        {
            fullPath = null;
            error = null;

            if (string.IsNullOrWhiteSpace(skillRoot))
            {
                error = "the skill has no root directory";
                return false;
            }
            if (!Path.IsPathRooted(skillRoot))
            {
                error = $"skill root '{skillRoot}' is not an absolute path";
                return false;
            }

            string root = NormalizeDirectory(Path.GetFullPath(skillRoot));

            // An empty path means the skill directory itself, which is always in bounds.
            if (string.IsNullOrWhiteSpace(relativePath))
            {
                fullPath = root;
                return true;
            }

            string candidate = relativePath.Trim();

            if (candidate.IndexOf('\0') >= 0)
            {
                error = "path contains a NUL character";
                return false;
            }

            // Normalise separators first so the lexical checks below see one form.
            candidate = candidate.Replace('\\', '/');

            // A leading "./" is idiomatic in SKILL.md links and carries no meaning.
            while (candidate.StartsWith("./", StringComparison.Ordinal))
                candidate = candidate.Substring(2);

            if (candidate.Length == 0 || candidate == ".")
            {
                fullPath = root;
                return true;
            }

            if (candidate.StartsWith("//", StringComparison.Ordinal))
            {
                error = $"'{relativePath}' is a UNC path; skill paths must be relative to the skill directory";
                return false;
            }
            if (candidate[0] == '/' || candidate[0] == '~')
            {
                error = $"'{relativePath}' is an absolute path; skill paths must be relative to the skill directory";
                return false;
            }
            if (candidate.Length >= 2 && candidate[1] == ':')
            {
                error = $"'{relativePath}' is a drive-qualified path; skill paths must be relative to the skill directory";
                return false;
            }

            foreach (string segment in candidate.Split('/'))
            {
                if (segment == "..")
                {
                    error = $"'{relativePath}' escapes the skill directory";
                    return false;
                }
                // A trailing dot or space is stripped by Windows when the path is
                // opened, so "scripts/run.py." and "scripts/run.py" would name the
                // same file while comparing as different strings. Refuse the shape
                // rather than reason about which one an allowlist matched.
                if (segment.Length > 0 && (segment[^1] == '.' || segment[^1] == ' ') && segment != ".")
                {
                    error = $"'{relativePath}' has a path segment ending in '.' or a space";
                    return false;
                }
            }

            string combined;
            try
            {
                combined = Path.GetFullPath(Path.Combine(root, candidate.Replace('/', Path.DirectorySeparatorChar)));
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                error = $"'{relativePath}' is not a usable path: {ex.Message}";
                return false;
            }

            if (!IsUnder(root, combined))
            {
                error = $"'{relativePath}' escapes the skill directory";
                return false;
            }

            if (!TryResolveSymlinks(root, combined, out string? resolved, out string? linkError))
            {
                error = linkError;
                return false;
            }

            fullPath = resolved;
            return true;
        }

        /// <summary>
        /// Resolve <paramref name="relativePath"/> and require that it names an
        /// existing file. The separate check exists so callers report "no such file"
        /// rather than the security message when a skill simply links to something
        /// it forgot to ship.
        /// </summary>
        public static bool TryResolveExistingFile(
            string skillRoot,
            string? relativePath,
            out string? fullPath,
            out string? error)
        {
            if (!TryResolve(skillRoot, relativePath, out fullPath, out error))
                return false;

            if (File.Exists(fullPath))
                return true;

            error = Directory.Exists(fullPath)
                ? $"'{relativePath}' is a directory, not a file"
                : $"'{relativePath}' does not exist in this skill";
            fullPath = null;
            return false;
        }

        /// <summary>
        /// True when <paramref name="path"/> is <paramref name="root"/> itself or sits
        /// beneath it. Both are expected to be canonical absolute paths.
        /// </summary>
        public static bool IsUnder(string root, string path)
        {
            string normalizedRoot = NormalizeDirectory(root);
            if (string.Equals(NormalizeDirectory(path), normalizedRoot, PathComparison))
                return true;

            string prefix = normalizedRoot.EndsWith(Path.DirectorySeparatorChar)
                ? normalizedRoot
                : normalizedRoot + Path.DirectorySeparatorChar;
            return path.StartsWith(prefix, PathComparison);
        }

        /// <summary>
        /// Walk every component of <paramref name="path"/> that exists on disk and
        /// follow it to its final target, failing if any of them leaves
        /// <paramref name="root"/>.
        ///
        /// <para>
        /// The walk goes component by component rather than resolving only the leaf,
        /// because a link on an intermediate directory redirects everything below it:
        /// with <c>references -&gt; /etc</c>, the leaf <c>references/passwd</c> does
        /// not itself exist as a link and a leaf-only check would wave it through.
        /// </para>
        /// <para>
        /// Components that do not exist yet are left alone — a caller resolving a
        /// destination path for an extraction is asking about a file that is about to
        /// be created, and the lexical and canonical checks already bound it.
        /// </para>
        /// </summary>
        internal static bool TryResolveSymlinks(string root, string path, out string? resolved, out string? error)
        {
            resolved = path;
            error = null;

            string realRoot = ResolveFinal(root) ?? root;

            var pending = new List<string>();
            string? cursor = path;
            while (cursor != null && !string.Equals(NormalizeDirectory(cursor), NormalizeDirectory(root), PathComparison))
            {
                string? parent = Path.GetDirectoryName(cursor);
                if (parent == null || string.Equals(parent, cursor, PathComparison))
                    break;
                pending.Add(Path.GetFileName(cursor));
                cursor = parent;
            }
            pending.Reverse();

            string current = realRoot;
            foreach (string segment in pending)
            {
                current = Path.Combine(current, segment);
                string? final = ResolveFinal(current);
                if (final == null)
                {
                    // Does not exist (yet); nothing to follow.
                    continue;
                }

                current = final;
                if (!IsUnder(realRoot, current))
                {
                    error = "the path resolves through a symbolic link that leaves the directory it is confined to";
                    return false;
                }
            }

            resolved = current;
            return true;
        }

        /// <summary>
        /// The final target of <paramref name="path"/> after following any chain of
        /// links, or null when nothing exists there. Never throws: a broken link, a
        /// cycle, or a permission error all mean "cannot be followed", and the caller
        /// treats that the same as "does not exist" — the containment checks above it
        /// still hold.
        /// </summary>
        private static string? ResolveFinal(string path)
        {
            try
            {
                FileSystemInfo info = Directory.Exists(path)
                    ? new DirectoryInfo(path)
                    : new FileInfo(path);
                if (!info.Exists)
                    return null;

                FileSystemInfo? target = info.ResolveLinkTarget(returnFinalTarget: true);
                return Path.GetFullPath(target?.FullName ?? info.FullName);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException
                                          or PathTooLongException)
            {
                return null;
            }
        }

        /// <summary>
        /// Canonical form for comparison: absolute, with any trailing separator
        /// removed so <c>/a/b</c> and <c>/a/b/</c> compare equal.
        /// </summary>
        public static string NormalizeDirectory(string path)
        {
            string full = Path.GetFullPath(path);
            if (full.Length > 1 && (full[^1] == Path.DirectorySeparatorChar || full[^1] == Path.AltDirectorySeparatorChar))
            {
                string trimmed = full.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                // A root such as "/" or "C:\" must keep its separator.
                if (trimmed.Length > 0 && !(trimmed.Length == 2 && trimmed[1] == ':'))
                    return trimmed;
            }
            return full;
        }

        /// <summary>
        /// The path of <paramref name="fullPath"/> relative to
        /// <paramref name="skillRoot"/>, always spelled with forward slashes so that
        /// what the model sees on Windows matches what a skill author wrote on Linux.
        /// </summary>
        public static string ToSkillRelative(string skillRoot, string fullPath)
        {
            string relative = Path.GetRelativePath(NormalizeDirectory(skillRoot), fullPath);
            return relative.Replace(Path.DirectorySeparatorChar, '/').Replace('\\', '/');
        }
    }
}
