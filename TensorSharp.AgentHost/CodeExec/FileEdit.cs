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
using System.Text;
using System.Text.RegularExpressions;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// Claude Code's <c>Edit</c>: replace one exact string with another, refusing when the
    /// string is not there or is there more than once.
    ///
    /// <para>
    /// <b>Why a string replacement and not a diff.</b> Anthropic ships the model that is
    /// best at editing code, controls both ends of the loop, and its editing tool emits no
    /// diff at all — no envelope, no <c>@@</c> anchors, no <c>+</c>/<c>-</c>/space line
    /// prefixes, no context lines. The model's entire output obligation is two byte
    /// strings. Anthropic's own published editing tool
    /// (<c>str_replace_based_edit_tool</c>) has the same shape. That matters most for the
    /// models this host serves: a V4A envelope has half a dozen ways to be malformed that
    /// a 4–8B model gets wrong — the leading SPACE on an unchanged line above all — and
    /// every one of them costs a round and pushes the model back toward re-typing the
    /// whole file. There is nothing here to malform.
    /// </para>
    /// <para>
    /// It is worth being precise about which reference this contradicts. Codex's
    /// <c>apply_patch</c> stays, for the job a string replacement structurally cannot do:
    /// several files changed together, all or nothing. Emitting a patch and reading one
    /// are different problems, and this codebase now answers each with the reference that
    /// solved it.
    /// </para>
    /// <para>
    /// <b>"Exact" is not quite exact, in both references and here.</b> Codex's matcher has
    /// a rung that normalises Unicode punctuation, added to "mirror the fuzzy behaviour of
    /// <c>git apply</c>"; Claude Code's has a smart-quote rung, a <c>\uXXXX</c>-literal
    /// rung, and a non-ASCII rung. The rungs below are the union of what those two do,
    /// plus one this host earns for itself: a model that read a file with <c>nl -ba</c> or
    /// <c>grep -n</c> sends line numbers glued to the front of its text, which this
    /// codebase already carries a detector for on the patch side.
    /// </para>
    /// <para>
    /// <b>Every rung above the first is reported on the RESULT, and restyles what it
    /// writes.</b> Tolerance that silently rewrites a file's own punctuation is not
    /// tolerance, it is corruption committed on the model's behalf: matching a curly
    /// <c>‘HELLO’</c> against an ASCII <c>'HELLO'</c> and then writing the replacement in
    /// ASCII changes bytes nobody asked to change. <see cref="Restyle"/> puts the file's
    /// own characters back.
    /// </para>
    /// </summary>
    public static class FileEdit
    {
        /// <summary>How far down the ladder a match came, and what that means.</summary>
        public enum Rung
        {
            /// <summary>Byte-for-byte.</summary>
            Exact = 0,

            /// <summary>Curly quotes, dashes and exotic spaces treated as their ASCII forms.</summary>
            Punctuation = 1,

            /// <summary>The model wrote a literal <c>—</c> where the file has the character.</summary>
            UnicodeEscape = 2,

            /// <summary>The model copied line numbers in with the text.</summary>
            LineNumbers = 3,
        }

        /// <summary>Where a string was found, how many times, and how hard it was to find.</summary>
        /// <param name="Found">Whether it is in the file at all.</param>
        /// <param name="Index">Where the first occurrence starts, in the LF-normalised text.</param>
        /// <param name="Count">How many occurrences that rung found.</param>
        /// <param name="Matched">The file's OWN bytes for the first occurrence.</param>
        /// <param name="Search">The string that actually matched, after any rung transformed it.</param>
        /// <param name="Rung">Which rung hit.</param>
        /// <param name="Offsets">Where each occurrence starts, capped.</param>
        public readonly record struct MatchResult(
            bool Found, int Index, int Count, string Matched, string Search, Rung Rung, IReadOnlyList<int> Offsets);

        /// <summary>Occurrences worth locating for a message. More than this and the count is the point.</summary>
        private const int MaxReportedOffsets = 5;

        /// <summary>
        /// Find <paramref name="oldString"/> in <paramref name="content"/>, walking the
        /// ladder until a rung finds it.
        ///
        /// <para>
        /// Only the WINNING rung is counted. A string that appears once exactly and three
        /// times approximately is not ambiguous — the exact match is the one the model
        /// meant, and refusing it would teach the model that being precise is punished.
        /// </para>
        /// </summary>
        public static MatchResult Find(string content, string oldString)
        {
            content ??= string.Empty;
            oldString ??= string.Empty;

            if (oldString.Length == 0)
                return new MatchResult(false, -1, 0, string.Empty, oldString, Rung.Exact, Array.Empty<int>());

            // Rung one and two, on the string as sent.
            MatchResult direct = Scan(content, oldString, Rung.Exact);
            if (direct.Found)
                return direct;

            // Rung three: the model wrote the escape rather than the character. Gated on
            // the string actually containing that form, so an ordinary edit to a file full
            // of backslashes never takes this path.
            if (UnicodeEscape.IsMatch(oldString))
            {
                string decoded = DecodeUnicodeEscapes(oldString);
                if (!string.Equals(decoded, oldString, StringComparison.Ordinal))
                {
                    MatchResult escaped = Scan(content, decoded, Rung.UnicodeEscape);
                    if (escaped.Found)
                        return escaped;
                }
            }

            // Rung four: line numbers copied in with the text, which is what `nl -ba`,
            // `grep -n` and a rendered listing all invite. Fires only when stripping them
            // makes the string match — otherwise the numbers are a coincidence (a
            // changelog, a table of data) and acting on them would send the model to fix
            // something that was already right.
            string unnumbered = StripLineNumbers(oldString);
            if (!string.Equals(unnumbered, oldString, StringComparison.Ordinal))
            {
                MatchResult numbered = Scan(content, unnumbered, Rung.LineNumbers);
                if (numbered.Found)
                    return numbered;
            }

            return new MatchResult(false, -1, 0, string.Empty, oldString, Rung.Exact, Array.Empty<int>());
        }

        /// <summary>Rungs one and two over one candidate string.</summary>
        private static MatchResult Scan(string content, string needle, Rung rung)
        {
            if (needle.Length == 0 || needle.Length > content.Length)
                return new MatchResult(false, -1, 0, string.Empty, needle, rung, Array.Empty<int>());

            // Rung one: ordinal, and first, always. The fast path and the honest one.
            var exact = new List<int>();
            for (int i = content.IndexOf(needle, StringComparison.Ordinal);
                 i >= 0;
                 i = content.IndexOf(needle, i + needle.Length, StringComparison.Ordinal))
            {
                exact.Add(i);
                if (exact.Count >= MaxReportedOffsets)
                    break;
            }
            if (exact.Count > 0)
            {
                int total = CountOrdinal(content, needle);
                return new MatchResult(
                    true, exact[0], total, content.Substring(exact[0], needle.Length), needle, rung, exact);
            }

            // Rung two: the same search with visually identical characters folded
            // together. Length-preserving by construction — every mapping below is one
            // character to one character — so a hit maps straight back onto the original
            // text and the file's own bytes come out of the slice untouched.
            if (!NeedsPunctuationFold(content) && !NeedsPunctuationFold(needle))
                return new MatchResult(false, -1, 0, string.Empty, needle, rung, Array.Empty<int>());

            // Bounded, because both strings come from the model and this rung is the one
            // that cannot use an index: a long needle against a long file is a
            // multiplication, and there is no deadline anywhere on this path. The cap is
            // far above any real edit — a thousand-line needle in an eight-megabyte file —
            // so reaching it means the call was not an edit, and giving up here degrades
            // to "not found", which is a refusal the model can read.
            const long MaxFoldedComparisons = 64L * 1024 * 1024;
            if ((long)content.Length * needle.Length > MaxFoldedComparisons)
                return new MatchResult(false, -1, 0, string.Empty, needle, rung, Array.Empty<int>());

            var folded = new List<int>();
            int count = 0;
            for (int i = 0; i + needle.Length <= content.Length; i++)
            {
                if (!FoldedEquals(content, i, needle))
                    continue;
                count++;
                if (folded.Count < MaxReportedOffsets)
                    folded.Add(i);
                i += needle.Length - 1;
            }

            if (folded.Count == 0)
                return new MatchResult(false, -1, 0, string.Empty, needle, rung, Array.Empty<int>());

            Rung landed = rung == Rung.Exact ? Rung.Punctuation : rung;
            return new MatchResult(
                true, folded[0], count, content.Substring(folded[0], needle.Length), needle, landed, folded);
        }

        private static int CountOrdinal(string content, string needle)
        {
            int count = 0;
            for (int i = content.IndexOf(needle, StringComparison.Ordinal);
                 i >= 0;
                 i = content.IndexOf(needle, i + needle.Length, StringComparison.Ordinal))
            {
                count++;
            }
            return count;
        }

        private static bool FoldedEquals(string content, int start, string needle)
        {
            for (int offset = 0; offset < needle.Length; offset++)
            {
                if (Fold(content[start + offset]) != Fold(needle[offset]))
                    return false;
            }
            return true;
        }

        private static bool NeedsPunctuationFold(string value)
        {
            for (int i = 0; i < value.Length; i++)
            {
                // "Does anything in here fold", asked directly. It was once "is anything
                // non-ASCII AND does it fold", which is the same question only while Fold
                // touches nothing in ASCII — and the moment it did, the gate started
                // answering differently for a SUBSTRING than for the whole string. That
                // made a replace_all walk stop as soon as it passed the file's last
                // non-ASCII character, leaving occurrences unchanged and reporting them
                // all as edited.
                if (Fold(value[i]) != value[i])
                    return true;
            }
            return false;
        }

        /// <summary>
        /// One character to its ASCII stand-in, or itself.
        ///
        /// <para>
        /// The UNION of what both references fold: Codex's <c>seek_sequence</c> normalises
        /// dashes, quotes and non-breaking spaces to "mirror the fuzzy behaviour of
        /// <c>git apply</c>"; Claude Code folds smart quotes. Strictly one-to-one, so the
        /// search stays length-preserving and a match can always be mapped back onto the
        /// file's own characters.
        /// </para>
        /// <para>
        /// It can only ever equate characters that LOOK the same, which is the whole of
        /// the safety argument: it cannot make one identifier match a different one.
        /// </para>
        /// </summary>
        private static char Fold(char c) => c switch
        {
            // Single quotes and apostrophes.
            //
            // The ASCII backtick is deliberately NOT here, and neither is the acute
            // accent. A backtick is a distinct character with its own meaning in nearly
            // every language this host will be handed — a template literal in JavaScript,
            // command substitution in a shell, a code span in Markdown — so folding it
            // onto an apostrophe would let an edit match text that means something else
            // entirely, which is the one thing this rung must never do. It is also the
            // only ASCII character Fold used to touch, and that made the fold gate answer
            // differently for a substring than for the whole file.
            '‘' or '’' or '‚' or '‛' or '′' => '\'',

            // Double quotes.
            '“' or '”' or '„' or '‟' or '″' or '«' or '»' => '"',

            // Hyphens, dashes and the minus sign.
            '‐' or '‑' or '‒' or '–' or '—' or '―' or '−' => '-',

            // Spaces that are not the space.
            ' ' or ' ' or ' ' or ' ' or ' ' or ' ' or ' '
                or ' ' or ' ' or ' ' or ' ' or ' ' or '　' => ' ',

            // Ellipsis is NOT folded to "..." — that is three characters for one, which
            // would break the length-preserving property the whole rung depends on.
            _ => c,
        };

        private static readonly Regex UnicodeEscape = new(
            @"\\u[0-9a-fA-F]{4}", RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static string DecodeUnicodeEscapes(string value) =>
            UnicodeEscape.Replace(value, m =>
                ((char)int.Parse(m.Value.AsSpan(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture))
                    .ToString());

        /// <summary>
        /// A leading line number as the numbering tools emit it, on every line of a block.
        ///
        /// <para>
        /// The same shapes <see cref="V4ADiff"/> already recognises: <c>nl -ba</c> uses a
        /// tab, <c>grep -n</c> a colon, and this host's own <see cref="NumberedListing"/> a
        /// pipe. Stripped only when EVERY non-blank line carries one — a single numbered
        /// line inside a block is data, not a listing.
        /// </para>
        /// </summary>
        private static readonly Regex NumberPrefix = new(
            @"^[ \t]*\d+(?:\t|:|[ ]*\|[ ]?)", RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static string StripLineNumbers(string value)
        {
            string[] lines = value.Split('\n');
            var stripped = new string[lines.Length];
            bool sawNumber = false;

            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i];
                if (line.Trim().Length == 0)
                {
                    stripped[i] = line;
                    continue;
                }
                Match match = NumberPrefix.Match(line);
                if (!match.Success)
                    return value;   // one bare line means this is not a listing
                sawNumber = true;
                stripped[i] = line.Substring(match.Length);
            }

            return sawNumber ? string.Join("\n", stripped) : value;
        }

        /// <summary>
        /// Rewrite <paramref name="newString"/> into the punctuation the FILE uses, given
        /// what the file actually held where the match landed.
        ///
        /// <para>
        /// Only reachable from a tolerant rung, and only ever substitutes a character for
        /// one that folds to it. Without this, matching leniently and writing literally
        /// silently converts a file's typography: a file holding <c>‘HELLO’</c> edited
        /// with an ASCII <c>'HELLO'</c> would come back ASCII, a change to bytes nobody
        /// asked about, in a diff nobody reviewed.
        /// </para>
        /// <para>
        /// A plain character that the file spells two different ways — an opening and a
        /// closing quote, which is the usual case — is restored by ALTERNATION in the
        /// order the file uses them, with one carve-out taken from the reference: a
        /// quote between two letters is an apostrophe, so <c>don't</c> does not acquire a
        /// closing quote. Anything this cannot resolve confidently is left as the model
        /// wrote it, which is the conservative direction — a literal replacement is
        /// visible in a diff, whereas a wrong guess is not.
        /// </para>
        /// </summary>
        public static string Restyle(string newString, string matched, string search)
        {
            if (string.IsNullOrEmpty(newString) || matched == null || search == null)
                return newString ?? string.Empty;
            if (matched.Length != search.Length || string.Equals(matched, search, StringComparison.Ordinal))
                return newString;

            // What the file spells each of the model's characters as, in the order it
            // uses them.
            var spellings = new Dictionary<char, List<char>>();
            for (int i = 0; i < matched.Length; i++)
            {
                if (matched[i] == search[i])
                    continue;

                // SYMMETRIC, matching the predicate the search itself used. The ladder
                // folds BOTH sides, so it matches in both directions — the file
                // typographic and the model ASCII, and the file ASCII and the model
                // typographic, which is the commoner one because models emit smart quotes
                // on their own. A one-directional guard here bailed on that second case
                // and wrote the model's curly quotes straight into an ASCII source file,
                // turning what would have been a clean "not found" refusal into a written
                // syntax error.
                if (Fold(matched[i]) != Fold(search[i]))
                    return newString;   // not a fold we made; do not invent one

                if (!spellings.TryGetValue(search[i], out List<char>? forms))
                    spellings[search[i]] = forms = new List<char>();
                forms.Add(matched[i]);
            }

            // A character the matched region ALSO spells the model's way is one this
            // cannot resolve, so it is dropped and the model's own bytes stand.
            //
            // Without this, a file that is inconsistent about a character — ASCII quotes
            // in the code and a curly apostrophe in a comment, which is what an editor
            // with smart quotes produces — teaches this the wrong lesson: the plain
            // spelling is never recorded as a form, so the ONE typographic form wins and
            // every occurrence in the replacement is converted. Measured on the real
            // code, `label = 'x'  # don<curly>t touch` edited with plain ASCII came back
            // as `label = <curly>y<curly>  # don't touch` — invalid Python, written
            // silently, under a result line promising the file's own characters were used
            // "so nothing else changed". Conservative is the only safe direction here: a
            // literal replacement is visible in a diff, a wrong guess is not.
            for (int i = 0; i < matched.Length; i++)
            {
                if (matched[i] == search[i])
                    spellings.Remove(search[i]);
            }

            if (spellings.Count == 0)
                return newString;

            var used = new Dictionary<char, int>();
            var sb = new StringBuilder(newString.Length);
            for (int i = 0; i < newString.Length; i++)
            {
                char c = newString[i];
                if (!spellings.TryGetValue(c, out List<char>? forms))
                {
                    sb.Append(c);
                    continue;
                }

                // The reference's carve-out: a quote flanked by letters is an apostrophe
                // and must stay whatever the model typed, or "don't" becomes "don’t" in a
                // file that only ever used curly quotes as quotes.
                if (c == '\'' && i > 0 && i + 1 < newString.Length
                    && char.IsLetter(newString[i - 1]) && char.IsLetter(newString[i + 1]))
                {
                    sb.Append(c);
                    continue;
                }

                int seen = used.TryGetValue(c, out int n) ? n : 0;
                sb.Append(forms.Count == 1 ? forms[0] : forms[seen % forms.Count]);
                used[c] = seen + 1;
            }

            return sb.ToString();
        }

        /// <summary>
        /// Put the REPLACEMENT through whatever transform made the search text match.
        ///
        /// <para>
        /// Without this, a tolerant rung repairs one half of the model's input and writes
        /// the other half through untouched. The line-number rung was the plain disaster:
        /// a model that copies a block out of a <c>read_file</c> result into BOTH
        /// arguments — precisely the mistake the rung exists to absorb — had the prefixes
        /// stripped from what was searched for and written verbatim into the file, so
        /// <c>   42 |     total = 0</c> landed in the source under a result line saying
        /// the edit succeeded and the rest of the file was untouched.
        /// </para>
        /// <para>
        /// Safe in the other direction too, because each rung is reached only after the
        /// exact form has already MISSED. A file that genuinely contains a literal
        /// <c>\uXXXX</c> in its source matches at rung one, so nothing here can rewrite an
        /// escape the model meant literally.
        /// </para>
        /// </summary>
        public static string ApplyRungTo(string newString, Rung rung)
        {
            if (string.IsNullOrEmpty(newString))
                return newString ?? string.Empty;

            return rung switch
            {
                Rung.UnicodeEscape when UnicodeEscape.IsMatch(newString) => DecodeUnicodeEscapes(newString),
                Rung.LineNumbers => StripLineNumbers(newString),
                _ => newString,
            };
        }

        /// <summary>How a rung that fired should be described on a successful result, or null.</summary>
        /// <param name="restyled">
        /// Whether <see cref="Restyle"/> actually changed the replacement. It is a
        /// parameter because the punctuation message used to promise unconditionally that
        /// "the replacement was written using the file's own characters" — a promise that
        /// is simply false whenever the restyle declined to resolve an inconsistent file,
        /// and a false sentence on a successful result is worse than no sentence.
        /// </param>
        public static string? Describe(Rung rung, bool restyled = true) => rung switch
        {
            Rung.Punctuation =>
                "Your text did not match the file byte for byte: the file uses typographic "
                + "quotes, dashes or spaces where you wrote the plain ASCII ones, or the other way "
                + "round. It was matched on those anyway, and "
                + (restyled
                    ? "the replacement was written using the file's own characters so nothing else changed."
                    : "your replacement was written exactly as you sent it, because the file spells "
                      + "that character both ways and guessing which you meant would change bytes you "
                      + "did not ask about."),
            Rung.UnicodeEscape =>
                "Your text contained a literal '\\uXXXX' escape where the file holds the character "
                + "itself. It was matched on the decoded form, and the replacement was decoded the "
                + "same way. Copy text out of a read_file result rather than escaping it.",
            Rung.LineNumbers =>
                "Your text carried line numbers at the start of each line. They were stripped from "
                + "BOTH old_string and new_string before anything was written — line numbers are a "
                + "navigation aid and are never part of the file, so leave the '   42 | ' prefix out "
                + "of both.",
            _ => null,
        };
    }
}
