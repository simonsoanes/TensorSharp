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
using System.Text.RegularExpressions;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// The V4A diff engine: given a file and the body of one <c>*** Update File</c>
    /// section, work out where each hunk goes and produce the new text.
    ///
    /// <para>
    /// A near line-for-line port of the reference implementation
    /// (openai-agents-python, <c>src/agents/apply_diff.py</c>), and deliberately so. This
    /// is the part of a patch tool where "improving" the algorithm makes it worse: every
    /// rung of its matching ladder is enumerable and testable, and the moment a
    /// similarity score or an edit distance is added, a hunk that should have failed
    /// starts landing somewhere plausible and wrong. Silent wrong edits are the exact
    /// failure that deterministic editing exists to eliminate.
    /// </para>
    /// <para>
    /// The ladder, and the whole of the fuzziness budget: match the block of context
    /// exactly (fuzz 0); else with trailing whitespace ignored on both sides (fuzz 1);
    /// else with leading and trailing whitespace ignored (fuzz 100); else FAIL. Fuzz is
    /// carried out to the caller not to change the outcome but to describe it — a patch
    /// that only matched at fuzz 100 landed on indentation the model got wrong, and
    /// saying so is cheaper than the model discovering it later.
    /// </para>
    /// <para>
    /// Three details that look like trivia and are not:
    /// deleted lines join the CONTEXT that is searched for (so a hunk that deletes at its
    /// edges can be found at all); a completely empty line in a hunk is read as a blank
    /// CONTEXT line, because models strip the trailing space from those and without this
    /// coercion every patch touching a blank line fails; and <c>@@</c> anchors are
    /// ADVISORY — an anchor that matches nothing leaves the cursor alone and lets the
    /// context resolve the hunk, rather than failing a patch whose anchor text the model
    /// paraphrased.
    /// </para>
    /// </summary>
    public static class V4ADiff
    {
        private const string EndPatch = "*** End Patch";
        private const string EndFile = "*** End of File";

        private static readonly string[] SectionTerminators =
        {
            EndPatch, "*** Update File:", "*** Delete File:", "*** Add File:",
        };

        private static readonly string[] EndSectionMarkers =
            SectionTerminators.Concat(new[] { EndFile }).ToArray();

        /// <summary>One replacement: drop <see cref="Deleted"/> at <see cref="OriginalIndex"/>, put <see cref="Inserted"/> there.</summary>
        private sealed class Chunk
        {
            public int OriginalIndex;
            public List<string> Deleted = new();
            public List<string> Inserted = new();
        }

        /// <summary>How a patch application went.</summary>
        /// <param name="Ok">Whether <paramref name="Text"/> is a result or the file was left alone.</param>
        /// <param name="Text">The new file content, when <paramref name="Ok"/>.</param>
        /// <param name="Fuzz">
        /// How far down the ladder the match came: 0 exact, 1 trailing whitespace ignored,
        /// 100 all surrounding whitespace ignored, +10000 an end-of-file hunk that had to
        /// fall back to a forward search.
        /// </param>
        /// <param name="LinesAdded">Inserted line count.</param>
        /// <param name="LinesRemoved">Deleted line count.</param>
        /// <param name="Error">Why nothing was produced, phrased for the model.</param>
        public readonly record struct DiffResult(
            bool Ok, string Text, int Fuzz, int LinesAdded, int LinesRemoved, string? Error)
        {
            /// <summary>
            /// Something the model has to be told about a patch that SUCCEEDED, or null.
            ///
            /// <para>
            /// Separate from <see cref="Error"/> because it is not a failure: the patch
            /// applied and the file is written. It exists for the one way a hunk could
            /// land wrongly with no symptom at all — a context that fits in several
            /// places, resolved to the first, at fuzz 0, reported as a clean edit. Every
            /// other way a hunk goes wrong ends in a refusal the model can read; this one
            /// ended in "updated main.py (+1 -1)" with the change made to the wrong
            /// function.
            /// </para>
            /// </summary>
            public string? Note { get; init; }

            internal static DiffResult Failed(string error) =>
                new(false, string.Empty, 0, 0, 0, error);
        }

        /// <summary>
        /// Build a NEW file from a section whose every line is a <c>+</c> line.
        /// </summary>
        /// <param name="diffNewline">The envelope's newline style; a created file has no other source for one.</param>
        public static DiffResult Create(IReadOnlyList<string> diffLines, string? diffNewline = null)
        {
            var output = new List<string>();
            foreach (string line in diffLines)
            {
                if (SectionTerminators.Any(t => line.StartsWith(t, StringComparison.Ordinal)))
                    break;
                if (!line.StartsWith('+'))
                {
                    return DiffResult.Failed(
                        $"every line of an '*** Add File' section must start with '+', and this one does not:\n{line}\n"
                        + "Prefix each line of the new file with '+', including blank lines.");
                }
                output.Add(line.Substring(1));
            }

            // A file that does not exist yet has no newline style of its own to preserve,
            // so it takes the ENVELOPE's — which has to be handed down, because
            // SplitDiffLines strips every CR long before this sees the section and a
            // detection here could only ever have answered "\n".
            string newline = diffNewline ?? "\n";
            string text = string.Join(newline, output);

            // A deliberate departure from the reference, which ends the file exactly where
            // the last '+' line ends. That leaves a new text file with no final newline
            // unless the model thought to send a bare '+' as its last line — which nothing
            // tells it to do, and which it therefore never does. The result is a file that
            // concatenates wrongly, that git reports as "\ No newline at end of file", and
            // that some tools simply misparse. Appending one cannot double up: a model that
            // DID send the bare '+' already produced the trailing newline here.
            if (text.Length > 0 && !text.EndsWith(newline, StringComparison.Ordinal))
                text += newline;

            return new DiffResult(true, text, 0, output.Count, 0, null);
        }

        /// <summary>
        /// Apply an <c>*** Update File</c> section to <paramref name="input"/>.
        /// </summary>
        /// <param name="input">The file as it stands.</param>
        /// <param name="diffLines">The section body, already split into lines.</param>
        /// <param name="diffNewline">
        /// The newline style of the ENVELOPE the section came from, for the one case where
        /// the input cannot supply one: an empty file, or a one-line file with no
        /// terminator. The reference falls back to the diff's style there and has a test
        /// for it, and this could not until the caller started passing it — SplitDiffLines
        /// strips every CR before this ever sees the section, so a fallback computed here
        /// could only ever have said "\n".
        /// </param>
        public static DiffResult Update(
            string input, IReadOnlyList<string> diffLines, string? diffNewline = null, string? path = null)
        {
            // The style of the file wins: a diff written with CRLF must not convert an LF
            // file, and the reverse conversion is the one that shows up as a whole-file
            // change in someone's diff the next morning.
            //
            // When the file has no newline of its own to copy — it is empty, or one line
            // with nothing after it — the reference falls back to the DIFF's style. This
            // port cannot: the section body arrives here already split, by SplitDiffLines,
            // which strips the CRs, so no amount of looking at it can tell CRLF from LF.
            // Said as a constant rather than written as a detection over diffLines that
            // silently always answers "\n" — the same reason Create states its LF plainly.
            // A caller that wants the reference's answer has to carry the envelope's
            // newline style down to here; nothing in this file can recover it.
            string newline = input.Contains('\n', StringComparison.Ordinal)
                ? DetectNewline(input)
                : "\n";

            string normalized = input.Replace("\r\n", "\n", StringComparison.Ordinal);
            var lines = new List<string>(diffLines) { EndPatch };
            List<string> inputLines = normalized.Split('\n').ToList();

            var chunks = new List<Chunk>();
            int cursor = 0;
            int index = 0;
            int fuzz = 0;
            // The FIRST ambiguous hunk only. A patch that repeats the warning per hunk
            // buries the file it applied to under its own advice.
            string? ambiguity = null;

            // Set when a hunk had to be re-sought from its own '@@' line. Reported for
            // the same reason every rung of the ladder is: a hunk that only landed
            // because the host relaxed something is a hunk the model should know about.
            string? anchorRetry = null;

            while (!IsDone(lines, index, EndSectionMarkers))
            {
                (List<string> anchors, bool hasAnchor) = ReadAnchors(lines, ref index);

                if (!hasAnchor && cursor != 0)
                {
                    string current = index < lines.Count ? lines[index] : string.Empty;
                    return DiffResult.Failed(
                        "this line is not part of a hunk — a hunk after the first must start with "
                        + $"'@@' so its position is unambiguous:\n{current}");
                }

                // An anchor that matches nothing stays ADVISORY — the cursor is left where
                // it was and the hunk resolves by its context. That is a recorded finding
                // of this codebase, not an oversight: failing the patch outright taught
                // models to stop writing '@@' headers at all, which made everything worse,
                // and models paraphrase a header line far more often than they get the
                // surrounding context wrong.
                //
                // What the miss DOES change is how much the context has to prove on its
                // own. An anchor exists to disambiguate; when it did not land, nothing
                // disambiguated, so a context that matches in several places is a hunk
                // whose position was never determined — see the ambiguity refusal below.
                bool anchored = false;

                // The line the last anchor matched ON, and its text. Kept for the retry
                // below, which is the only thing that can undo the cursor's step past it.
                int anchorLanding = -1;
                string? anchorText = null;
                for (int a = 0; a < anchors.Count; a++)
                {
                    cursor = AdvanceToAnchor(
                        anchors[a], inputLines, cursor, ref fuzz, forceForward: a > 0,
                        out bool landed, out int landedAt);
                    anchored |= landed;
                    if (landedAt >= 0)
                    {
                        anchorLanding = landedAt;
                        anchorText = anchors[a];
                    }
                }

                if (!TryReadSection(lines, index, out Section section, out string? sectionError))
                    return DiffResult.Failed(sectionError!);

                (int at, int matchFuzz, IReadOnlyList<int> alsoAt) =
                    FindContext(inputLines, section.Context, cursor, section.Eof);

                // The hunk whose '@@' line is ALSO its first line, re-sought from the
                // anchor instead of from one line past it.
                //
                // MEASURED, not hypothetical. A model that reads a file with `head -5`
                // sees five lines and writes a hunk out of them, and the line it picks as
                // the '@@' header is the same line it then quotes as the hunk's first
                // context or '-' line — because those five lines are all it has. The
                // anchor matches at index i, the cursor steps to i+1, and the context
                // search begins one line past the only place it can match. In the logs
                // that is exactly one round: `head -5 create_slides.py`, then a hunk
                // anchored on the import it had just been shown, refused with "Looked for
                // these lines, starting from line 2" while the refusal's own excerpt said
                // "the closest place is line 1". The model never called apply_patch again
                // in that conversation.
                //
                // NOT a fuzz rung, and not a widening of the search. The same three-rung
                // ladder runs, from one line earlier, under two guards that together
                // admit exactly one index: the hunk's first line must BE the anchor, and
                // the retry must land ON the anchor. So this can never move a hunk to
                // somewhere the un-stepped cursor would not already have allowed, and the
                // reference's forward-only contract across hunks is untouched — a later
                // hunk still cannot resolve before an earlier anchor.
                //
                // A departure from the reference, which has the identical `idx + 1` and
                // would refuse this hunk too. It is taken on the reference's OWN stated
                // policy of absorbing unambiguous model malformations in the host rather
                // than spending a round teaching the model to avoid them — the same
                // policy that strips heredoc wrappers and parses non-strictly. The
                // reference's models were post-trained to write '@@' as the enclosing
                // declaration, distinct from the hunk body; ours copy whatever `head`
                // printed.
                //
                // Fires on a MISS and on a match found LATER than the anchor, and the
                // second case is the more dangerous of the two. When the anchor line is
                // also the hunk's first line and the file holds a second identical copy
                // further down, the stepped cursor skips the copy the model pointed at
                // and matches the next one — silently, at fuzz 0, reported as a clean
                // edit. That is the one failure mode this whole file is organised
                // against. Retrying from the anchor can only ever move a hunk EARLIER,
                // to the exact line the model named twice — once as the anchor and once
                // as the hunk's own first line — and only when the context genuinely
                // fits there; when it does not, the later match stands.
                if ((at == -1 || at > anchorLanding)
                    && anchorLanding >= 0
                    && anchorText != null
                    && section.Context.Count > 0
                    && string.Equals(
                        TrimLikeReference(section.Context[0]),
                        TrimLikeReference(anchorText),
                        StringComparison.Ordinal))
                {
                    (int retryAt, int retryFuzz, IReadOnlyList<int> retryAlsoAt) =
                        FindContext(inputLines, section.Context, anchorLanding, section.Eof);

                    // ON the anchor, never before it and never after it. Landing later
                    // would be the match the un-retried search already had, and landing
                    // earlier is impossible from this start — but the equality is
                    // written out rather than assumed, because it is the whole of the
                    // safety argument: exactly one index can ever be chosen here, and it
                    // is the line the model named.
                    if (retryAt == anchorLanding)
                    {
                        at = retryAt;
                        matchFuzz = retryFuzz;
                        alsoAt = retryAlsoAt;
                        anchorRetry ??=
                            "The '@@' line was also this hunk's first line, so the hunk was matched "
                            + "from the anchor itself rather than from the line after it. That is "
                            + "what was meant here, and nothing else could have matched — but the "
                            + "two are normally different: '@@' names the enclosing function or "
                            + "class, and the lines under it are the ones being changed.";
                    }
                }

                if (at == -1)
                    return DiffResult.Failed(DescribeMiss(section, cursor, inputLines, path));

                // The silent-wrong-edit hole, closed — by SAYING SO, not by refusing.
                //
                // A hunk whose context fits in several places is resolved to the first, at
                // fuzz 0, and reported as a clean edit. Every other way a hunk goes wrong
                // ends in a refusal the model can read; this one ends in
                // "updated main.py (+1 -1)" with the change made to the wrong function,
                // and the model then debugs code it never touched.
                //
                // Refusing it was tried and rejected. Taking the first match is the
                // reference's behaviour and is pinned by this codebase's tests, and the
                // first match is usually right — a model reads a file top-down, so the
                // occurrence it saw IS the first one. This codebase has also already paid
                // for one round of making the patcher stricter: failing an unresolvable
                // '@@' anchor taught models to stop writing anchors at all, "which made
                // everything worse". So the fix is the one that costs nothing when the
                // guess was right and everything when it was wrong: apply, and say which
                // place was chosen and what the alternatives were.
                //
                // Only when nothing narrowed the hunk. An anchor that landed positioned it
                // deliberately, and a later hunk positioned by an earlier one's cursor is
                // likewise not a coin toss.
                if (!anchored && alsoAt.Count > 0 && ambiguity == null)
                    ambiguity = DescribeAmbiguity(at, alsoAt, path);

                cursor = at + section.Context.Count;
                fuzz += matchFuzz;
                index = section.EndIndex;

                foreach (Chunk chunk in section.Chunks)
                {
                    chunks.Add(new Chunk
                    {
                        OriginalIndex = chunk.OriginalIndex + at,
                        Deleted = new List<string>(chunk.Deleted),
                        Inserted = new List<string>(chunk.Inserted),
                    });
                }
            }

            // Both notes, when both happened. Joining rather than picking keeps the
            // never-silent rule honest: a hunk can be re-sought from its anchor AND land
            // somewhere the context did not uniquely determine, and dropping either
            // leaves the model told half of why its patch went where it did.
            string? note = ambiguity == null ? anchorRetry
                : anchorRetry == null ? ambiguity
                : ambiguity + "\n" + anchorRetry;

            return ApplyChunks(normalized, chunks, newline, fuzz) with { Note = note };
        }

        private static string DetectNewline(string text) =>
            text.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";

        /// <summary>Split a diff into lines the way the reference does: CR stripped, one trailing blank dropped.</summary>
        public static List<string> SplitDiffLines(string diff)
        {
            var lines = diff.Replace("\r\n", "\n", StringComparison.Ordinal)
                .Split('\n').Select(l => l.TrimEnd('\r')).ToList();
            if (lines.Count > 0 && lines[^1].Length == 0)
                lines.RemoveAt(lines.Count - 1);
            return lines;
        }

        private static bool IsDone(List<string> lines, int index, IReadOnlyList<string> prefixes)
        {
            if (index >= lines.Count)
                return true;
            foreach (string prefix in prefixes)
            {
                if (lines[index].StartsWith(prefix, StringComparison.Ordinal))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Consume the <c>@@</c> headers that introduce one hunk.
        ///
        /// <para>
        /// Several may be stacked — <c>@@ class Base</c> then <c>@@     def method():</c> —
        /// to narrow into nested code that a single header plus three lines of context
        /// cannot locate. A bare <c>@@</c> counts as "a header was here" but contributes
        /// no text to search for.
        /// </para>
        /// </summary>
        private static (List<string> Anchors, bool HasAnchor) ReadAnchors(List<string> lines, ref int index)
        {
            var anchors = new List<string>();
            bool hasAnchor = false;

            while (true)
            {
                bool consumed = false;
                string anchor = string.Empty;

                if (index < lines.Count && lines[index].StartsWith("@@ ", StringComparison.Ordinal))
                {
                    anchor = lines[index].Substring(3);
                    index++;
                    consumed = true;
                }
                else if (index < lines.Count && lines[index] == "@@")
                {
                    index++;
                    consumed = hasAnchor = true;
                }

                if (!consumed)
                    break;
                if (anchor.Length > 0)
                    hasAnchor = true;
                if (TrimLikeReference(anchor).Length > 0)
                    anchors.Add(anchor);
            }

            return (anchors, hasAnchor);
        }

        /// <summary>
        /// Move the cursor past an anchor line.
        ///
        /// <para>
        /// <paramref name="found"/> is false when the anchor is nowhere in the file at or
        /// after the cursor. That used to be treated as advisory — the cursor stayed put
        /// and the hunk resolved by context anyway — and the caller now refuses instead:
        /// an anchor exists to say which of several similar places is meant, so ignoring
        /// an unresolvable one is ignoring the only instruction that disambiguates.
        /// </para>
        /// <para>
        /// An anchor already BEHIND the cursor still counts as found, exactly as before:
        /// the reference treats a hunk whose anchor was passed by an earlier hunk as
        /// already positioned, and changing that would break every multi-hunk patch.
        /// </para>
        /// </summary>
        private static int AdvanceToAnchor(
            string anchor, List<string> inputLines, int cursor, ref int fuzz, bool forceForward,
            out bool found, out int landedAt)
        {
            found = false;

            // Where the FORWARD search actually matched, or -1. Only a forward landing is
            // reported: an anchor found behind the cursor did not position anything, so
            // there is no line for the caller's retry to aim at, and offering one would
            // let a hunk resolve before an anchor it never moved past.
            landedAt = -1;

            // Behind the cursor: already positioned, and not a miss.
            if (!forceForward
                && (inputLines.Take(cursor).Any(l => string.Equals(l, anchor, StringComparison.Ordinal))
                    || inputLines.Take(cursor).Any(
                        l => string.Equals(TrimLikeReference(l), TrimLikeReference(anchor), StringComparison.Ordinal))))
            {
                found = true;
            }

            if (forceForward || !inputLines.Take(cursor).Any(l => string.Equals(l, anchor, StringComparison.Ordinal)))
            {
                for (int i = cursor; i < inputLines.Count; i++)
                {
                    if (string.Equals(inputLines[i], anchor, StringComparison.Ordinal))
                    {
                        cursor = i + 1;
                        landedAt = i;
                        found = true;
                        break;
                    }
                }
            }

            if (!found
                && (forceForward
                    || !inputLines.Take(cursor).Any(
                        l => string.Equals(TrimLikeReference(l), TrimLikeReference(anchor), StringComparison.Ordinal))))
            {
                for (int i = cursor; i < inputLines.Count; i++)
                {
                    if (string.Equals(TrimLikeReference(inputLines[i]), TrimLikeReference(anchor), StringComparison.Ordinal))
                    {
                        cursor = i + 1;
                        landedAt = i;
                        fuzz += 1;
                        found = true;
                        break;
                    }
                }
            }

            return cursor;
        }

        /// <summary>
        /// True when the hunk failed only because its context lines carry line numbers.
        ///
        /// <para>
        /// The defect that numbering CREATES. A model told to read with <c>nl -ba</c> or
        /// <c>grep -n</c> — which is the right advice, because a hunk built from unnumbered
        /// output is a hunk built from a guess about indentation — then copies
        /// <c>   41\ttotal = 0</c> straight into the patch. The number is not in the file,
        /// so nothing matches, and the generic "did not match" message sends it to check
        /// its indentation, which was never the problem.
        /// </para>
        /// <para>
        /// <b>Detected and reported, never stripped and applied.</b> Auto-stripping would
        /// be a fourth rung of the fuzz ladder wearing a different name, and this file's
        /// standing decision is that the ladder does not grow: each rung that forgives more
        /// is a rung on which a hunk that should have failed lands somewhere plausible and
        /// wrong. One sentence naming the exact cause costs one round; a silent repair
        /// costs the debugging session that follows the wrong edit.
        /// </para>
        /// </summary>
        private static bool NumberedContext(Section section, List<string> inputLines, int cursor)
        {
            var stripped = new List<string>(section.Context.Count);
            bool sawNumber = false;
            foreach (string line in section.Context)
            {
                // Line by line, and not all-or-nothing. A model pasting a numbered
                // listing numbers the lines it DELETES as well as the ones it quotes for
                // position, and it may have typed one of them from memory without a
                // number — so requiring every line to carry one missed the common shape
                // entirely. What decides the diagnosis is not how many lines look
                // numbered but whether removing the numbers makes the hunk match, which
                // is checked below and cannot be satisfied by accident.
                Match match = NumberPrefix.Match(line);
                if (match.Success)
                {
                    sawNumber = true;
                    stripped.Add(line.Substring(match.Length));
                }
                else
                {
                    stripped.Add(line);
                }
            }

            // Only when removing them actually WORKS. Otherwise the numbers are a
            // coincidence — a file of numbered data, a changelog — and naming them would
            // send the model to fix something that was already right.
            return sawNumber
                && FindContext(inputLines, stripped, cursor, section.Eof).Index >= 0;
        }

        /// <summary>
        /// A leading line number as the numbering tools emit it: optional indent, digits,
        /// then a tab, a colon or a pipe, then optional space. <c>nl -ba</c> uses a tab,
        /// <c>grep -n</c> a colon, and a rendered listing often a pipe.
        /// </summary>
        private static readonly Regex NumberPrefix = new(
            // Nothing is consumed AFTER a tab or a colon. `nl -ba` writes the number, a
            // tab, then the line verbatim, and `grep -n` the number, a colon, then the
            // line verbatim — so eating one more space there strips a level of the
            // indentation this is trying to restore, and the hunk still would not match.
            // Only the pipe form, which is a rendered listing rather than a tool's output,
            // pads with a space.
            @"^[ \t]*\d+(?:\t|:|[ ]*\|[ ]?)",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        /// <summary>Longest single line quoted back from the file.</summary>
        private const int MaxExcerptLineChars = NumberedListing.MaxExcerptLineChars;

        /// <summary>Total size of the quoted region. Enough for a dozen ordinary lines of code.</summary>
        private const int MaxExcerptChars = NumberedListing.MaxExcerptChars;

        /// <summary>The file's name for a message, or a neutral word when the caller did not pass one.</summary>
        /// <summary>
        /// Show the file's own bytes around the closest thing to the hunk's context,
        /// numbered, so the next hunk can be built from what is there.
        ///
        /// <para>
        /// The anchor is the hunk's FIRST context line, matched on the same ladder the
        /// real search uses. That is the line most likely to be right when the rest of the
        /// block has drifted, and finding it is what turns "this did not match" into "here
        /// is the code you were aiming at". When even that is nowhere, nothing is invented:
        /// a made-up "closest match" would send the model to rebuild a hunk against a
        /// region that has nothing to do with it.
        /// </para>
        /// <para>
        /// Line numbers are rendered separated from the text by " | " rather than by a tab,
        /// on purpose. A tab is what <c>nl</c> and <c>grep -n</c> emit, and a model that
        /// copies numbered output straight back into a hunk produces context lines with a
        /// number glued to the front — a failure this message exists to prevent, not cause.
        /// The separator makes the number visibly not part of the line.
        /// </para>
        /// </summary>
        private static void AppendNearestRegion(
            StringBuilder sb, Section section, List<string> inputLines, int cursor, string? path)
        {
            if (section.Context.Count == 0 || inputLines.Count == 0)
                return;

            // Not text, whatever the extension says: reading it as lines produced NULs and
            // any excerpt would be nonsense presented to the model as its own file.
            if (inputLines.Take(64).Any(line => line.Contains('\0', StringComparison.Ordinal)))
                return;

            var head = new List<string> { section.Context[0] };
            (int at, _, _) = FindContextCore(inputLines, head, 0);

            string lead;
            if (at >= 0)
            {
                lead = "The closest place in " + Named(path) + " is line "
                     + (at + 1).ToString(CultureInfo.InvariantCulture) + ", which actually holds:\n";
            }
            else if (cursor > 0 && cursor <= LastRealLine(inputLines))
            {
                // Nothing in the file resembles the hunk's first line, but an anchor DID
                // position the search — so the file at that position is what the hunk was
                // aimed at, and it is the region the model has to rebuild from. This is
                // more useful than the resemblance search, not less: the hunk's own text
                // has drifted so far that resemblance found nothing.
                at = cursor;
                lead = "At line " + (cursor + 1).ToString(CultureInfo.InvariantCulture)
                     + " of " + Named(path) + ", where this hunk was aimed, the file holds:\n";
            }
            else
            {
                sb.Append("No line of ").Append(Named(path))
                  .Append(" looks like the first line of that block, so it is not there at all — "
                        + "check you are patching the right file.\n");
                return;
            }

            int from = Math.Max(0, at - 2);
            int to = Math.Min(LastRealLine(inputLines), at + Math.Max(section.Context.Count, 3) + 1);

            sb.Append(lead);

            // One renderer for every numbered listing this host produces — the patch
            // excerpt, read_file, and every refusal on the file surface. A line the model
            // meets here looks the same as the line it meets there, so a string copied
            // out of one result pastes into the next call unchanged.
            NumberedListing.Append(sb, inputLines, from, to, MaxExcerptChars, MaxExcerptLineChars);
        }

        /// <summary>
        /// Say that a hunk fits in several places, and where — rather than silently
        /// patching whichever came first.
        ///
        /// <para>
        /// This is the failure with no symptom. Every other way a hunk can go wrong ends in
        /// a refusal the model can read; this one ends in "updated main.py (+1 -1)" with
        /// the change made to the wrong function, at fuzz 0, and nothing anywhere to
        /// contradict it. The model then debugs code it did not change.
        /// </para>
        /// </summary>
        private static string DescribeAmbiguity(int at, IReadOnlyList<int> alsoAt, string? path)
        {
            var sb = new StringBuilder();
            sb.Append("a hunk fit in more than one place in ").Append(Named(path))
              .Append(" and was applied at line ").Append((at + 1).ToString(CultureInfo.InvariantCulture))
              .Append(", the first. It would also have matched at line ");
            for (int i = 0; i < alsoAt.Count; i++)
            {
                if (i > 0)
                    sb.Append(i == alsoAt.Count - 1 ? " and line " : ", line ");
                sb.Append((alsoAt[i] + 1).ToString(CultureInfo.InvariantCulture));
            }
            sb.Append(". If you meant one of those instead, patch again with "
                    + "'@@ <the enclosing function or class line>' above the hunk, copied exactly as "
                    + "that line appears in the file — or with more unchanged lines around the change, "
                    + "until only the place you mean still matches.\n");
            return sb.ToString();
        }

        /// <summary>
        /// The last index that is a real line. Splitting "a\nb\n" yields a trailing empty
        /// element, which is a position in the array and not a line of the file — printing
        /// it as one puts a blank numbered line at the end of every excerpt.
        /// </summary>
        private static int LastRealLine(List<string> inputLines) =>
            NumberedListing.LastRealLine(inputLines);

        private static string Named(string? path) =>
            string.IsNullOrEmpty(path) ? "the file" : path!;

        private sealed class Section
        {
            public List<string> Context = new();
            public List<Chunk> Chunks = new();
            public int EndIndex;
            public bool Eof;
        }

        private static bool TryReadSection(
            List<string> lines, int startIndex, out Section section, out string? error)
        {
            section = new Section();
            error = null;

            var context = new List<string>();
            var deleted = new List<string>();
            var inserted = new List<string>();
            var chunks = new List<Chunk>();
            string mode = "keep";
            int index = startIndex;

            while (index < lines.Count)
            {
                string raw = lines[index];
                if (raw.StartsWith("@@", StringComparison.Ordinal)
                    || raw.StartsWith(EndPatch, StringComparison.Ordinal)
                    || raw.StartsWith("*** Update File:", StringComparison.Ordinal)
                    || raw.StartsWith("*** Delete File:", StringComparison.Ordinal)
                    || raw.StartsWith("*** Add File:", StringComparison.Ordinal)
                    || raw.StartsWith(EndFile, StringComparison.Ordinal))
                {
                    break;
                }
                if (raw == "***")
                    break;
                if (raw.StartsWith("***", StringComparison.Ordinal))
                {
                    error = raw.StartsWith("*** Move to:", StringComparison.Ordinal)
                        ? "'*** Move to:' has to come immediately after the '*** Update File:' line it "
                          + "renames, before any hunk — not after the changes."
                        : $"'{raw}' is not a marker this patch format knows. "
                          + "The markers are '*** Add File:', '*** Update File:', '*** Delete File:', "
                          + "'*** Move to:', '*** End of File' and '*** End Patch'.";
                    return false;
                }

                index++;
                string lastMode = mode;

                // An entirely empty line is a blank CONTEXT line whose leading space the
                // model dropped. Every patch that touches a blank line depends on this.
                string line = raw.Length == 0 ? " " : raw;
                char prefix = line[0];
                mode = prefix switch
                {
                    '+' => "add",
                    '-' => "delete",
                    ' ' => "keep",
                    _ => "invalid",
                };
                if (mode == "invalid")
                {
                    error = $"every line inside a hunk must start with ' ', '+' or '-', and this one does not:\n{line}\n"
                          + "A context line you did not change still needs its leading space.";
                    return false;
                }

                string content = line.Substring(1);
                if (mode == "keep" && lastMode != mode && (deleted.Count > 0 || inserted.Count > 0))
                {
                    chunks.Add(new Chunk
                    {
                        OriginalIndex = context.Count - deleted.Count,
                        Deleted = new List<string>(deleted),
                        Inserted = new List<string>(inserted),
                    });
                    deleted = new List<string>();
                    inserted = new List<string>();
                }

                switch (mode)
                {
                    case "delete":
                        // Deleted lines are part of what is searched for as well as part of
                        // what is removed: the text matched against the file is context and
                        // deletions together, in order.
                        deleted.Add(content);
                        context.Add(content);
                        break;
                    case "add":
                        inserted.Add(content);
                        break;
                    default:
                        context.Add(content);
                        break;
                }
            }

            if (deleted.Count > 0 || inserted.Count > 0)
            {
                chunks.Add(new Chunk
                {
                    OriginalIndex = context.Count - deleted.Count,
                    Deleted = new List<string>(deleted),
                    Inserted = new List<string>(inserted),
                });
            }

            section.Context = context;
            section.Chunks = chunks;

            if (index < lines.Count && lines[index].TrimEnd() == EndFile)
            {
                section.EndIndex = index + 1;
                section.Eof = true;
                return true;
            }

            if (index == startIndex)
            {
                string next = index < lines.Count ? lines[index] : string.Empty;
                error = "this hunk has no lines in it. A '@@' header must be followed by the "
                      + $"context and changes it applies to.{(next.Length > 0 ? "\nFound instead: " + next : string.Empty)}";
                return false;
            }

            section.EndIndex = index;
            section.Eof = false;
            return true;
        }

        private static (int Index, int Fuzz, IReadOnlyList<int> AlsoAt) FindContext(
            List<string> lines, List<string> context, int start, bool eof)
        {
            if (eof)
            {
                // An end-of-file hunk is anchored at the end first, so it cannot land on an
                // identical block earlier in the file. Only if that misses does it fall back
                // to an ordinary forward search — and then the fuzz says so loudly.
                //
                // Never reported as ambiguous: "the end of the file" IS the disambiguation
                // the model asked for, so a second match earlier in the file is precisely
                // the thing the end-anchored search exists to skip past.
                int endStart = Math.Max(0, lines.Count - context.Count);
                (int endIndex, int endFuzz, _) = FindContextCore(lines, context, endStart);
                if (endIndex != -1)
                    return (endIndex, endFuzz, Array.Empty<int>());
                (int fallbackIndex, int fallbackFuzz, _) = FindContextCore(lines, context, start);
                return (fallbackIndex, fallbackFuzz + 10000, Array.Empty<int>());
            }
            return FindContextCore(lines, context, start);
        }

        /// <summary>
        /// Whitespace as the reference counts it: everything <see cref="char.IsWhiteSpace(char)"/>
        /// accepts, plus the four ASCII information separators U+001C–U+001F.
        ///
        /// <para>
        /// Those four are the entire difference between Python's <c>str.strip()</c> and
        /// .NET's <c>Trim()</c> — verified by enumerating both classifications over every
        /// code point, in both directions — and they are not cosmetic here, because these
        /// two functions are what rungs two and three of the ladder compare with. A file
        /// carrying a separator character (a paste out of a tool that uses them as
        /// delimiters, a fixture, a record-oriented export) resolves its hunk in the
        /// reference and was refused here: the same patch applying on one implementation
        /// and failing on the other is the one property this port exists to not have.
        /// </para>
        /// </summary>
        private static bool IsReferenceWhitespace(char value) =>
            char.IsWhiteSpace(value) || (value >= '\u001C' && value <= '\u001F');

        /// <summary>Python's <c>str.rstrip()</c> — rung two of the ladder ignores exactly this.</summary>
        private static string TrimEndLikeReference(string value)
        {
            int end = value.Length;
            while (end > 0 && IsReferenceWhitespace(value[end - 1]))
                end--;
            return end == value.Length ? value : value.Substring(0, end);
        }

        /// <summary>Python's <c>str.strip()</c> — rung three of the ladder ignores exactly this.</summary>
        private static string TrimLikeReference(string value)
        {
            int start = 0;
            int end = value.Length;
            while (start < end && IsReferenceWhitespace(value[start]))
                start++;
            while (end > start && IsReferenceWhitespace(value[end - 1]))
                end--;
            return start == 0 && end == value.Length ? value : value.Substring(start, end - start);
        }

        /// <summary>
        /// The first place the context fits, how far down the ladder it had to go, and
        /// <b>every other place on that same rung</b>.
        ///
        /// <para>
        /// The other places are reported, not used, and the caller decides. Finding them
        /// costs one extra pass over the rung that matched and nothing at all on the rungs
        /// that did not — the search still stops at the first rung that hits — but it is
        /// the only way to know that "the first match" was a choice rather than the answer.
        /// </para>
        /// <para>
        /// Only the WINNING rung is counted. A block that matches exactly in one place and
        /// approximately in three is not ambiguous: the ladder already decided, and the
        /// exact match is the one the model meant.
        /// </para>
        /// </summary>
        private static (int Index, int Fuzz, IReadOnlyList<int> AlsoAt) FindContextCore(
            List<string> lines, List<string> context, int start)
        {
            // Empty context matches where the cursor already is — that is what lets a hunk
            // of pure additions under a '@@' anchor insert at that anchor. It is not
            // ambiguous; it is positional.
            if (context.Count == 0)
                return (start, 0, Array.Empty<int>());

            foreach ((int rungFuzz, Func<string, string> normalize) in Ladder)
            {
                int first = -1;
                List<int>? others = null;
                for (int i = start; i < lines.Count; i++)
                {
                    if (!EqualsSlice(lines, context, i, normalize))
                        continue;
                    if (first < 0)
                    {
                        first = i;
                        continue;
                    }
                    (others ??= new List<int>()).Add(i);
                    if (others.Count >= MaxReportedMatches)
                        break;
                }
                if (first >= 0)
                    return (first, rungFuzz, (IReadOnlyList<int>?)others ?? Array.Empty<int>());
            }
            return (-1, 0, Array.Empty<int>());
        }

        /// <summary>
        /// The whole of the fuzziness budget, in order, as one table rather than three
        /// copied loops — so a rung cannot gain a behaviour the others do not have.
        /// </summary>
        private static readonly (int Fuzz, Func<string, string> Normalize)[] Ladder =
        {
            (0, static value => value),
            (1, TrimEndLikeReference),
            (100, TrimLikeReference),
        };

        /// <summary>Extra matches worth naming. Three places is already enough to say "pick one".</summary>
        private const int MaxReportedMatches = 3;

        private static bool EqualsSlice(
            List<string> source, List<string> target, int start, Func<string, string> map)
        {
            if (start + target.Count > source.Count)
                return false;
            for (int offset = 0; offset < target.Count; offset++)
            {
                if (!string.Equals(map(source[start + offset]), map(target[offset]), StringComparison.Ordinal))
                    return false;
            }
            return true;
        }

        private static DiffResult ApplyChunks(string input, List<Chunk> chunks, string newline, int fuzz)
        {
            List<string> original = input.Split('\n').ToList();
            var destination = new List<string>();
            int cursor = 0;
            int added = 0;
            int removed = 0;

            foreach (Chunk chunk in chunks)
            {
                if (chunk.OriginalIndex > original.Count)
                {
                    return DiffResult.Failed(
                        "a hunk resolved past the end of the file, which means the context it "
                        + "matched is not where the change belongs. Read the file and rebuild the hunk.");
                }
                if (cursor > chunk.OriginalIndex)
                {
                    return DiffResult.Failed(
                        $"two hunks overlap at line {(chunk.OriginalIndex + 1).ToString(CultureInfo.InvariantCulture)}. "
                        + "Hunks must be in file order and must not touch the same lines; "
                        + "merge them into one hunk.");
                }

                destination.AddRange(original.GetRange(cursor, chunk.OriginalIndex - cursor));
                cursor = chunk.OriginalIndex;

                if (chunk.Inserted.Count > 0)
                {
                    destination.AddRange(chunk.Inserted);
                    added += chunk.Inserted.Count;
                }

                cursor += chunk.Deleted.Count;
                removed += chunk.Deleted.Count;
            }

            destination.AddRange(original.GetRange(cursor, original.Count - cursor));
            return new DiffResult(true, string.Join(newline, destination), fuzz, added, removed, null);
        }

        /// <summary>
        /// Why a hunk did not match, said the way that gets it fixed on the next try.
        ///
        /// <para>
        /// The text the model sent is echoed back — including its leading whitespace,
        /// which is very often the actual problem and is invisible in any summary. Claude
        /// Code's editor does the same thing for the same reason: "String to replace not
        /// found in file" followed by the exact string is self-correcting, while "edit
        /// failed" produces a retry loop.
        /// </para>
        /// </summary>
        private static string DescribeMiss(
            Section section, int cursor, List<string> inputLines, string? path)
        {
            var sb = new StringBuilder();
            sb.Append(section.Eof
                ? "the end-of-file hunk did not match. Looked for these lines at the end of the file"
                : "this hunk did not match the file. Looked for these lines");
            sb.Append(cursor > 0
                ? $", starting from line {(cursor + 1).ToString(CultureInfo.InvariantCulture)}:\n"
                : ":\n");

            foreach (string line in section.Context.Take(8))
                sb.Append("  |").Append(line).Append('\n');
            if (section.Context.Count > 8)
                sb.Append("  … and ").Append((section.Context.Count - 8).ToString(CultureInfo.InvariantCulture)).Append(" more\n");

            // The one cause with a one-sentence answer, checked before anything else:
            // the context lines carry line numbers copied out of `nl` or `grep -n`.
            if (NumberedContext(section, inputLines, cursor))
            {
                sb.Append("Those lines start with LINE NUMBERS. They came from `nl`, `grep -n` or a "
                        + "numbered listing, and the numbers are not in the file — with them removed the "
                        + "hunk matches. Send it again with the number and the separator stripped from "
                        + "every context line.\nNothing was written.\n");
                return sb.ToString();
            }

            // What is REALLY there. Echoing only what the model looked for tells it
            // nothing it did not already know, so its next hunk comes out of the same
            // distribution and misses the same way — which is exactly what the logs show:
            // one turn spent rounds 12 and 15 on the same file and failed both times, and
            // between them the host had handed back 20,876 characters of the program with
            // no indication of what differed.
            AppendNearestRegion(sb, section, inputLines, cursor, path);

            sb.Append("Nothing was written — the whole patch was abandoned. ");
            // Splitting "a\nb\n" yields a trailing empty element, which is a position in
            // the array and not a line of the file. Reporting it made every message about
            // a newline-terminated file off by one, which is exactly the kind of detail a
            // model then builds its next (also wrong) hunk around.
            int lineCount = inputLines.Count > 0 && inputLines[^1].Length == 0
                ? inputLines.Count - 1
                : inputLines.Count;
            sb.Append("The file has ").Append(lineCount.ToString(CultureInfo.InvariantCulture))
              .Append(" lines. Read the region you meant to change and build the hunk from what is "
                    + "actually there; indentation must match exactly.\n");
            return sb.ToString();
        }
    }
}
