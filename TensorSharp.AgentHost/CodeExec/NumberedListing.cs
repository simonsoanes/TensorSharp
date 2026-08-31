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
using System.Text;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// The one way this host ever shows a model the bytes of a file: numbered, bounded,
    /// and with the number visibly not part of the line.
    ///
    /// <para>
    /// Extracted from <see cref="V4ADiff"/>'s patch-failure excerpt, which until now was
    /// the ONLY place a model could see a file's real content — reachable only by
    /// failing. That was the shape of the whole problem: the host had a good renderer and
    /// hid it behind an error. <c>read_file</c>, every refusal on the file surface, and
    /// the ambiguity listing all render through here, so a line looks the same wherever
    /// the model meets it and a string copied out of one result can be pasted into the
    /// next call.
    /// </para>
    /// <para>
    /// <b>The separator is <c>" | "</c>, never a tab and never a bare colon.</b> A tab is
    /// exactly what <c>nl -ba</c> and <c>grep -n</c> emit, and a model that copies that
    /// output back into an edit produces text with a number glued to the front — a
    /// failure <see cref="V4ADiff"/> carries a whole detector for. Claude Code renders
    /// <c>cat -n</c> and then spends a line of its tool description telling the model to
    /// strip it; the intent is the same and this is the form this codebase measured as
    /// survivable. The numbers are a navigation aid and are deliberately not part of the
    /// edit contract.
    /// </para>
    /// <para>
    /// <b>Everything here is bounded, and not as a nicety.</b> A minified bundle or a
    /// one-line JSON document is a single multi-megabyte "line", and an assembled tool
    /// result is NOT passed through the output cap that bounds a command's stdout — so an
    /// unclipped excerpt goes into the result, into the prompt, and then into whatever has
    /// to truncate the prompt to fit.
    /// </para>
    /// </summary>
    public static class NumberedListing
    {
        /// <summary>Longest single line quoted back from a file in an EXCERPT.</summary>
        public const int MaxExcerptLineChars = 200;

        /// <summary>Total size of an excerpt. Enough for a dozen ordinary lines of code.</summary>
        public const int MaxExcerptChars = 2048;

        /// <summary>
        /// Longest single line rendered by <c>read_file</c>, which is allowed more than an
        /// excerpt because the model asked for this file and is about to copy out of it —
        /// a clipped line it then pastes into an edit is an edit that cannot match.
        /// </summary>
        public const int MaxReadLineChars = 2000;

        /// <summary>
        /// Total size of a <c>read_file</c> render. Sized against the context windows this
        /// host actually serves: 64 KB is roughly 16k tokens, which is already most of a
        /// small local model's window, so this is a ceiling that should never be reached
        /// rather than a budget to spend.
        /// </summary>
        public const int MaxReadChars = 64 * 1024;

        /// <summary>
        /// The width the number is padded to. Five digits covers any file worth editing,
        /// and a file with more lines simply renders wider rather than misaligning.
        /// </summary>
        private const int NumberWidth = 5;

        /// <summary>
        /// The prefix for one line, 1-based: two spaces, the number right-aligned, then
        /// <c>" | "</c>.
        /// </summary>
        public static string Prefix(int lineNumber) =>
            "  " + lineNumber.ToString(CultureInfo.InvariantCulture).PadLeft(NumberWidth) + " | ";

        /// <summary>
        /// A NUL byte anywhere in the first stretch means this is not text, whatever the
        /// extension says.
        ///
        /// <para>
        /// Decoding UTF-16-without-a-BOM as Latin-1 yields lines laced with NULs, so both
        /// the line COUNT and the lines themselves would be nonsense — and a listing
        /// states both to the model as facts about its own file.
        /// </para>
        /// </summary>
        public static bool LooksBinary(IReadOnlyList<string> lines)
        {
            if (lines == null)
                return false;

            int scanned = 0;
            for (int i = 0; i < lines.Count; i++)
            {
                if (lines[i].Contains('\0', StringComparison.Ordinal))
                    return true;
                if ((scanned += lines[i].Length + 1) > 8192)
                    break;
            }
            return false;
        }

        /// <summary>
        /// The last index that is a real line.
        ///
        /// <para>
        /// Splitting <c>"a\nb\n"</c> yields a trailing empty element, which is a POSITION
        /// in the array and not a line of the file. Printing it as one puts a blank
        /// numbered line at the end of every listing, and counting it tells the model a
        /// file is one line longer than it is — which then goes into an <c>offset</c> that
        /// reads nothing.
        /// </para>
        /// </summary>
        public static int LastRealLine(IReadOnlyList<string> lines)
        {
            if (lines == null || lines.Count == 0)
                return -1;
            return lines.Count >= 1 && lines[^1].Length == 0 ? lines.Count - 2 : lines.Count - 1;
        }

        /// <summary>How many real lines the file has, for a message that states its length.</summary>
        public static int RealLineCount(IReadOnlyList<string> lines) => LastRealLine(lines) + 1;

        /// <summary>
        /// Append lines <paramref name="from"/>..<paramref name="to"/> (0-based, inclusive)
        /// numbered from 1, stopping when <paramref name="charBudget"/> runs out.
        /// </summary>
        /// <returns>How many characters were spent.</returns>
        public static int Append(
            StringBuilder sb,
            IReadOnlyList<string> lines,
            int from,
            int to,
            int charBudget = MaxExcerptChars,
            int lineClip = MaxExcerptLineChars) =>
            Append(sb, lines, from, to, charBudget, lineClip, out _);

        /// <summary>
        /// The same, reporting the last line it actually emitted.
        /// </summary>
        /// <param name="lastShown">
        /// The 0-based index of the last line written, or <paramref name="from"/> minus one
        /// when nothing was. Callers that RECORD what the model has seen must use this
        /// rather than the requested range: the budget can stop the render early, and a
        /// caller that then remembers the requested range has told itself the model saw
        /// lines that were never rendered.
        /// </param>
        public static int Append(
            StringBuilder sb,
            IReadOnlyList<string> lines,
            int from,
            int to,
            int charBudget,
            int lineClip,
            out int lastShown)
        {
            ArgumentNullException.ThrowIfNull(sb);
            from = Math.Max(0, from);
            lastShown = from - 1;
            if (lines == null || lines.Count == 0)
                return 0;

            to = Math.Min(LastRealLine(lines), to);

            int spent = 0;
            for (int i = from; i <= to && spent < charBudget; i++)
            {
                lastShown = i;
                string line = lines[i];
                string shown = line.Length <= lineClip
                    ? line
                    : line.Substring(0, lineClip) + " … (line continues)";
                sb.Append(Prefix(i + 1)).Append(shown).Append('\n');
                spent += shown.Length + NumberWidth + 5;
            }

            // Never silently short. A listing that stopped early and did not say so is a
            // listing the model reads as the whole region, and it then builds an edit
            // against lines it was never shown.
            if (spent >= charBudget && to > from)
            {
                sb.Append("  … (the rest of that region is not shown here; "
                        + "read a smaller range to see it)\n");
            }

            return spent;
        }

        /// <summary>
        /// The same, as a string, for a caller that is not already building one.
        /// </summary>
        public static string Excerpt(
            IReadOnlyList<string> lines,
            int from,
            int to,
            int charBudget = MaxExcerptChars,
            int lineClip = MaxExcerptLineChars)
        {
            var sb = new StringBuilder();
            Append(sb, lines, from, to, charBudget, lineClip);
            return sb.ToString();
        }

        /// <summary>
        /// A window of <paramref name="radius"/> lines either side of a 0-based line,
        /// clamped to the file — for showing a model where something it named actually is.
        /// </summary>
        public static string Around(
            IReadOnlyList<string> lines, int index, int radius = 3, int charBudget = MaxExcerptChars)
        {
            return Excerpt(lines, index - radius, index + radius, charBudget);
        }

        /// <summary>
        /// Split file text into lines the way every listing here counts them: CRLF folded
        /// to LF first, so a Windows file does not render with a stray CR on each line and
        /// so an offset means the same thing on both platforms.
        /// </summary>
        public static List<string> SplitLines(string text) =>
            (text ?? string.Empty).Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n').ToList();
    }
}
