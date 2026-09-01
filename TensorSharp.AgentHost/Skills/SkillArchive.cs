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
using System.IO.Compression;
using System.Linq;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>Ceilings applied while extracting an uploaded skill bundle.</summary>
    public sealed class SkillArchiveLimits
    {
        /// <summary>Largest total decompressed size.</summary>
        public long MaxTotalBytes { get; init; } = 256L * 1024 * 1024;

        /// <summary>Largest single decompressed file.</summary>
        public long MaxEntryBytes { get; init; } = 64L * 1024 * 1024;

        /// <summary>Largest number of files.</summary>
        public int MaxEntries { get; init; } = 4096;

        /// <summary>
        /// Largest ratio of decompressed to compressed bytes tolerated across the whole
        /// archive. A ZIP bomb is a small upload that expands without bound; the byte
        /// ceilings above already stop it, and this catches it earlier and says why.
        /// Zero disables the check.
        /// </summary>
        public int MaxCompressionRatio { get; init; } = 200;
    }

    /// <summary>
    /// Extracts an uploaded skill bundle.
    ///
    /// <para>
    /// <c>ZipFile.ExtractToDirectory</c> is deliberately not used. An entry name in a
    /// ZIP is attacker-controlled text, and the classic "zip slip" attack ships an
    /// entry called <c>../../../.ssh/authorized_keys</c>; the framework helper has
    /// grown guards against the obvious form, but the file that lands still depends
    /// on how the name normalises on the host platform. Every entry here is resolved
    /// through <see cref="SkillPathGuard"/> instead — the same check that confines
    /// the model's own file reads — so the extraction and the reads agree on exactly
    /// one definition of "inside the skill".
    /// </para>
    /// <para>
    /// Size is enforced on the DECOMPRESSED stream rather than trusting the entry's
    /// declared <c>Length</c>, which the archive's own headers supply and which a
    /// crafted ZIP simply lies about.
    /// </para>
    /// </summary>
    public static class SkillArchive
    {
        /// <summary>
        /// Extract <paramref name="archive"/> into <paramref name="destination"/>,
        /// which is created and must not already exist.
        /// </summary>
        /// <exception cref="SkillInstallException">
        /// The archive is malformed, contains an entry that escapes the destination,
        /// or exceeds one of <paramref name="limits"/>.
        /// </exception>
        public static void Extract(Stream archive, string destination, SkillArchiveLimits limits)
        {
            ArgumentNullException.ThrowIfNull(archive);
            ArgumentNullException.ThrowIfNull(limits);
            if (string.IsNullOrWhiteSpace(destination))
                throw new ArgumentException("Destination is required.", nameof(destination));

            Directory.CreateDirectory(destination);
            string root = SkillPathGuard.NormalizeDirectory(destination);

            ZipArchive zip;
            try
            {
                zip = new ZipArchive(archive, ZipArchiveMode.Read, leaveOpen: true);
            }
            catch (InvalidDataException ex)
            {
                throw new SkillInstallException($"the upload is not a valid ZIP archive ({ex.Message}).", ex);
            }

            using (zip)
            {
                long totalWritten = 0;
                long totalCompressed = 0;
                int fileCount = 0;

                foreach (ZipArchiveEntry entry in zip.Entries)
                {
                    string name = entry.FullName;
                    if (name.Length == 0)
                        continue;

                    // A trailing separator marks a directory entry; it carries no bytes,
                    // and the directories that matter are created for the files anyway.
                    bool isDirectory = name[^1] == '/' || name[^1] == '\\';
                    if (isDirectory)
                        continue;

                    if (++fileCount > limits.MaxEntries)
                        throw new SkillInstallException($"the archive holds more than {limits.MaxEntries} files.");

                    if (!SkillPathGuard.TryResolve(root, name, out string? target, out string? guardError))
                        throw new SkillInstallException($"archive entry '{name}' is rejected: {guardError}");

                    totalCompressed += entry.CompressedLength;

                    Directory.CreateDirectory(Path.GetDirectoryName(target!)!);

                    long written;
                    try
                    {
                        using Stream source = entry.Open();
                        using FileStream output = new(target!, FileMode.Create, FileAccess.Write, FileShare.None);
                        written = CopyCapped(source, output, limits.MaxEntryBytes, name);
                    }
                    catch (InvalidDataException ex)
                    {
                        throw new SkillInstallException($"archive entry '{name}' is corrupt ({ex.Message}).", ex);
                    }

                    totalWritten += written;
                    if (totalWritten > limits.MaxTotalBytes)
                    {
                        throw new SkillInstallException(
                            $"the archive expands to more than {SkillTextBudget.FormatBytes(limits.MaxTotalBytes)}.");
                    }

                    if (limits.MaxCompressionRatio > 0
                        && totalCompressed > 0
                        && totalWritten / Math.Max(1, totalCompressed) > limits.MaxCompressionRatio)
                    {
                        throw new SkillInstallException(
                            $"the archive expands more than {limits.MaxCompressionRatio}x and was rejected as a decompression bomb.");
                    }
                }

                if (fileCount == 0)
                    throw new SkillInstallException("the archive is empty.");
            }
        }

        /// <summary>
        /// Find the directory holding <c>SKILL.md</c> inside an extracted bundle.
        ///
        /// <para>
        /// Both shapes people actually upload are accepted: a ZIP of the skill
        /// directory (<c>pdf/SKILL.md</c>, which is what every archive tool produces
        /// when you compress a folder) and a ZIP of its contents
        /// (<c>SKILL.md</c> at the archive root). macOS's Archive Utility also adds a
        /// <c>__MACOSX</c> sibling, which is skipped rather than mistaken for the
        /// skill.
        /// </para>
        /// </summary>
        /// <returns>The skill's directory, or null when the bundle holds no <c>SKILL.md</c>.</returns>
        public static string? LocateSkillRoot(string extractedRoot)
        {
            if (File.Exists(Path.Combine(extractedRoot, SkillManifestParser.SkillFileName)))
                return extractedRoot;

            string[] children;
            try
            {
                children = Directory.GetDirectories(extractedRoot);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return null;
            }

            string[] candidates = children
                .Where(d => !Path.GetFileName(d).Equals("__MACOSX", StringComparison.OrdinalIgnoreCase))
                .Where(d => File.Exists(Path.Combine(d, SkillManifestParser.SkillFileName)))
                .OrderBy(Path.GetFileName, StringComparer.Ordinal)
                .ToArray();

            // Exactly one is unambiguous. Several means the upload is a COLLECTION of
            // skills, which install-one cannot honour; the caller reports that rather
            // than silently picking the alphabetically first.
            return candidates.Length == 1 ? candidates[0] : null;
        }

        /// <summary>
        /// Every skill directory inside an extracted bundle, for the multi-skill case.
        /// </summary>
        public static IReadOnlyList<string> LocateAllSkillRoots(string extractedRoot)
        {
            var found = new List<string>();
            if (File.Exists(Path.Combine(extractedRoot, SkillManifestParser.SkillFileName)))
            {
                found.Add(extractedRoot);
                return found;
            }

            var queue = new Queue<(string Dir, int Depth)>();
            queue.Enqueue((extractedRoot, 0));
            while (queue.Count > 0)
            {
                (string dir, int depth) = queue.Dequeue();
                if (File.Exists(Path.Combine(dir, SkillManifestParser.SkillFileName)))
                {
                    found.Add(dir);
                    continue;
                }
                if (depth >= 3)
                    continue;

                string[] children;
                try
                {
                    children = Directory.GetDirectories(dir);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    continue;
                }

                Array.Sort(children, StringComparer.Ordinal);
                foreach (string child in children)
                {
                    string name = Path.GetFileName(child);
                    if (name.Length == 0 || name[0] == '.'
                        || name.Equals("__MACOSX", StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }
                    queue.Enqueue((child, depth + 1));
                }
            }
            return found;
        }

        /// <summary>
        /// Copy at most <paramref name="maxBytes"/> and fail if the source has more.
        /// The declared entry length is never consulted — it comes from the archive's
        /// own headers, which a crafted upload controls.
        /// </summary>
        private static long CopyCapped(Stream source, Stream destination, long maxBytes, string entryName)
        {
            byte[] buffer = new byte[81920];
            long written = 0;
            while (true)
            {
                int read = source.Read(buffer, 0, buffer.Length);
                if (read <= 0)
                    return written;

                written += read;
                if (written > maxBytes)
                {
                    throw new SkillInstallException(
                        $"archive entry '{entryName}' expands beyond the {SkillTextBudget.FormatBytes(maxBytes)} per-file limit.");
                }
                destination.Write(buffer, 0, read);
            }
        }
    }
}
