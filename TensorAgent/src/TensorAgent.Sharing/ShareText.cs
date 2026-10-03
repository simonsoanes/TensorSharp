// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;
using TensorAgent.Sharing.Localization;

namespace TensorAgent.Sharing;

/// <summary>
/// Turning what another app handed over into text a local model can afford to read.
///
/// <para>
/// Everything here is about cost. A phone's context is the scarcest thing in this
/// app — a 22 kB paste was enough to get the process killed by jetsam before the
/// prefill chunk was tuned down (see <c>MauiProgram</c>) — and a shared web page is
/// routinely ten times the size of anything a person would type. Half of what a
/// browser hands over is also not content at all: navigation, cookie banners,
/// "related stories", and in the HTML case the markup itself. Every character removed
/// here is a character the model does not spend a token on.
/// </para>
/// <para>
/// It is deliberately not a readability implementation. The browser is far better
/// placed to decide what the article is, and does — the extension's JavaScript
/// preprocessing file runs inside the page and sends the extracted text. This is what
/// happens for everything else: the apps that share HTML rather than text, and the
/// ones that share text with a page's worth of blank lines in it.
/// </para>
/// </summary>
public static class ShareText
{
    /// <summary>
    /// Plain text out of an HTML fragment or document.
    ///
    /// <para>
    /// A hand-written scanner rather than a parser, and that is the right trade here:
    /// there is no HTML parser in the base class library, this code has to run inside a
    /// share extension where every dependency is memory that is not there, and the job
    /// is not to build a tree but to decide, for each tag, whether it ends a line. What
    /// a malformed document costs is a stray angle bracket in the model's prompt.
    /// </para>
    /// <para>
    /// Three things it does that a naive tag-stripper does not, each of which was
    /// visible in the output before it did: the contents of <c>script</c>,
    /// <c>style</c>, <c>noscript</c>, <c>template</c>, <c>svg</c> and <c>head</c> are
    /// dropped rather than emitted (a page's stylesheet is otherwise the majority of
    /// the "text"), block-level tags become line breaks so paragraphs do not run
    /// together into one wall, and entities are decoded so the model reads an
    /// apostrophe rather than <c>&amp;rsquo;</c>.
    /// </para>
    /// </summary>
    public static string FromHtml(string? html)
    {
        if (string.IsNullOrEmpty(html))
            return string.Empty;

        var text = new StringBuilder(html.Length / 2);
        int i = 0;
        while (i < html.Length)
        {
            char c = html[i];
            if (c != '<')
            {
                text.Append(c);
                i++;
                continue;
            }

            // A comment, which can legally contain anything including a bare '>'.
            if (html.AsSpan(i).StartsWith("<!--"))
            {
                int close = html.IndexOf("-->", i + 4, StringComparison.Ordinal);
                i = close < 0 ? html.Length : close + 3;
                continue;
            }

            int end = html.IndexOf('>', i);
            if (end < 0)
            {
                // An unterminated tag at the very end. Everything after it was markup.
                break;
            }

            ReadOnlySpan<char> tag = html.AsSpan(i + 1, end - i - 1);
            bool closing = tag.Length > 0 && tag[0] == '/';
            ReadOnlySpan<char> name = TagName(closing ? tag[1..] : tag);

            if (!closing && IsDiscarded(name))
            {
                // Skip to the matching close tag, not to the next '>': the whole point
                // is that what is inside is not text.
                int after = SkipElement(html, end + 1, name);
                if (after < 0)
                {
                    i = html.Length;
                    break;
                }
                i = after;
                AppendBreak(text);
                continue;
            }

            if (IsBreaking(name))
                AppendBreak(text);
            if (!closing && name.SequenceEqual("li"))
                text.Append("- ");

            i = end + 1;
        }

        return Normalize(System.Net.WebUtility.HtmlDecode(text.ToString()));
    }

    /// <summary>
    /// The same text, with the things that only cost tokens taken out: carriage
    /// returns, non-breaking and zero-width spaces, trailing blanks on every line, and
    /// runs of empty lines longer than one.
    ///
    /// <para>
    /// The blank-line collapse is the one that matters. Text copied out of a rendered
    /// page arrives with a dozen empty lines between paragraphs, and on a model that
    /// tokenizes a newline as its own token that is a straight percentage off the
    /// budget for nothing at all. Two consecutive newlines still separate paragraphs,
    /// because the model reads structure and removing it would cost more than it saves.
    /// </para>
    /// </summary>
    public static string Normalize(string? text)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        var output = new StringBuilder(text.Length);
        var line = new StringBuilder(120);
        int blankRun = 0;
        bool anyContent = false;

        void FlushLine()
        {
            while (line.Length > 0 && (line[^1] == ' ' || line[^1] == '\t'))
                line.Length--;
            if (line.Length == 0)
            {
                blankRun++;
                // One blank line separates paragraphs; a second and beyond say nothing.
                if (anyContent && blankRun <= 1)
                    output.Append('\n');
                return;
            }
            if (anyContent)
                output.Append('\n');
            output.Append(line);
            line.Clear();
            blankRun = 0;
            anyContent = true;
        }

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            switch (c)
            {
                case '\r':
                    // CRLF and a lone CR both mean one line break.
                    if (i + 1 < text.Length && text[i + 1] == '\n')
                        i++;
                    FlushLine();
                    continue;
                case '\n':
                    FlushLine();
                    continue;
                // Non-breaking space, narrow no-break space, and the assorted Unicode
                // spaces a word processor emits: all a space to a reader and all
                // separate tokens to a tokenizer.
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case ' ':
                case '　':
                    line.Append(' ');
                    continue;
                // Zero-width and bidi marks: invisible, and a token each.
                case '​':
                case '‌':
                case '‍':
                case '⁠':
                case '﻿':
                case '‎':
                case '‏':
                    continue;
                default:
                    // Control characters other than tab have no meaning in shared text
                    // and several of them upset a JSON round trip on the way to the page.
                    if (char.IsControl(c) && c != '\t')
                        continue;
                    line.Append(c);
                    continue;
            }
        }
        FlushLine();

        // A trailing blank line was counted but never written; nothing to undo.
        return output.ToString();
    }

    /// <summary>What <see cref="Shorten"/> did.</summary>
    /// <param name="Text">The text to use, at most the requested length.</param>
    /// <param name="OriginalLength">How long it was before.</param>
    /// <param name="OmittedLength">How many characters are not in <paramref name="Text"/>.</param>
    public readonly record struct ShortenedText(string Text, int OriginalLength, int OmittedLength)
    {
        public bool WasShortened => OmittedLength > 0;
    }

    /// <summary>
    /// Fit text into a budget, keeping the beginning AND the end.
    ///
    /// <para>
    /// Head-only truncation is the obvious implementation and it is wrong for most of
    /// what people share. An article's conclusion is at the end, a mail thread's newest
    /// message is at one end or the other depending on the client, and a document's
    /// summary is often the last thing in it. Cutting the middle keeps both, and says
    /// exactly how much went, so the model can tell the user it only saw part — and so
    /// that a model asked something the missing middle answers can say so rather than
    /// guess.
    /// </para>
    /// <para>
    /// The cut is moved to the nearest line break within a short reach, so the join is
    /// between paragraphs rather than mid-word, and the elision marker is counted
    /// against the budget rather than added on top of it.
    /// </para>
    /// </summary>
    public static ShortenedText Shorten(string? text, int maxChars)
    {
        string value = text ?? string.Empty;
        // No budget means NOTHING fits, which is the opposite of what folding this in
        // with the pass-through case below used to say. Every caller clamps a spent
        // budget to zero before calling, so reading zero as "no limit" handed back the
        // whole of a hundred-kilobyte page at exactly the moment there was no room for
        // any of it.
        if (maxChars <= 0)
            return new ShortenedText(string.Empty, value.Length, value.Length);
        if (value.Length <= maxChars)
            return new ShortenedText(value, value.Length, 0);

        // Below this there is no room for a head, a marker and a tail; a plain head is
        // the honest answer.
        const int MinimumForElision = 400;
        if (maxChars < MinimumForElision)
        {
            const string smallMarker = "\n…";
            if (maxChars <= smallMarker.Length)
            {
                int headOnly = SurrogateSafeCut(value, maxChars);
                return new ShortenedText(value[..headOnly], value.Length, value.Length - headOnly);
            }
            int cut = SurrogateSafeCut(value, BackToBreak(value, maxChars - smallMarker.Length));
            return new ShortenedText(value[..cut] + smallMarker, value.Length, value.Length - cut);
        }

        // Two thirds from the front: the beginning of a shared thing is more often the
        // part being asked about, and this still keeps a real amount of the end.
        // Room for the marker itself, in any language: it is never wider than it would be
        // with the whole length in it.
        int budget = maxChars - Math.Max(64, OmittedMarker(value.Length).Length);
        int headWanted = budget * 2 / 3;
        int tailWanted = budget - headWanted;

        int head = SurrogateSafeCut(value, BackToBreak(value, headWanted));
        int tailStart = SurrogateSafeStart(value, ForwardToBreak(value, value.Length - tailWanted));
        if (tailStart <= head)
        {
            const string fallbackMarker = "\n…";
            int cut = SurrogateSafeCut(value, Math.Min(head, maxChars - fallbackMarker.Length));
            return new ShortenedText(value[..cut] + fallbackMarker, value.Length, value.Length - cut);
        }

        int omitted = tailStart - head;
        string marker = OmittedMarker(omitted);
        return new ShortenedText(
            value[..head] + marker + value[tailStart..],
            value.Length,
            omitted);
    }

    /// <summary>What stands where the middle was cut out, saying how much went.</summary>
    private static string OmittedMarker(int omitted) =>
        "\n\n" + ShareStrings.T("share.text.omitted",
            ("characters", omitted.ToString("N0", ShareStrings.Culture))) + "\n\n";

    private static int SurrogateSafeCut(string value, int cut)
    {
        int safe = Math.Clamp(cut, 0, value.Length);
        if (safe > 0 && safe < value.Length && char.IsHighSurrogate(value[safe - 1]))
            safe--;
        return safe;
    }

    private static int SurrogateSafeStart(string value, int start)
    {
        int safe = Math.Clamp(start, 0, value.Length);
        if (safe > 0 && safe < value.Length
            && char.IsLowSurrogate(value[safe])
            && char.IsHighSurrogate(value[safe - 1]))
        {
            safe++;
        }
        return safe;
    }

    /// <summary>The largest index at or before <paramref name="at"/> that is a line
    /// break, or a space, or <paramref name="at"/> itself when neither is close.</summary>
    private static int BackToBreak(string text, int at)
    {
        int limit = Math.Min(at, text.Length);
        int floor = Math.Max(0, limit - 200);
        for (int i = limit; i > floor; i--)
        {
            if (text[i - 1] == '\n')
                return i;
        }
        for (int i = limit; i > floor; i--)
        {
            if (text[i - 1] == ' ')
                return i;
        }
        return limit;
    }

    /// <summary>The smallest index at or after <paramref name="at"/> that starts a line.</summary>
    private static int ForwardToBreak(string text, int at)
    {
        int start = Math.Max(0, at);
        int ceiling = Math.Min(text.Length, start + 200);
        for (int i = start; i < ceiling; i++)
        {
            if (text[i] == '\n')
                return i + 1;
        }
        return start;
    }

    /// <summary>A one-line description of a shared thing, for a chat title or a log.</summary>
    public static string FirstLine(string? text, int maxChars = 80)
    {
        if (maxChars <= 0)
            return string.Empty;
        string normalized = Normalize(text);
        if (normalized.Length == 0)
            return string.Empty;
        int newline = normalized.IndexOf('\n');
        string line = (newline < 0 ? normalized : normalized[..newline]).Trim();
        if (line.Length <= maxChars)
            return line;
        if (maxChars == 1)
            return "…";
        int cut = SurrogateSafeCut(line, Math.Max(1, maxChars - 1));
        return line[..cut].TrimEnd() + "…";
    }

    private static ReadOnlySpan<char> TagName(ReadOnlySpan<char> tag)
    {
        int i = 0;
        while (i < tag.Length && !char.IsWhiteSpace(tag[i]) && tag[i] != '/')
            i++;
        return tag[..i];
    }

    private static bool IsDiscarded(ReadOnlySpan<char> name) =>
        Equals(name, "script") || Equals(name, "style") || Equals(name, "noscript")
        || Equals(name, "template") || Equals(name, "svg") || Equals(name, "head")
        || Equals(name, "iframe") || Equals(name, "object");

    private static bool IsBreaking(ReadOnlySpan<char> name) =>
        Equals(name, "p") || Equals(name, "br") || Equals(name, "div") || Equals(name, "li")
        || Equals(name, "tr") || Equals(name, "ul") || Equals(name, "ol") || Equals(name, "table")
        || Equals(name, "section") || Equals(name, "article") || Equals(name, "header")
        || Equals(name, "footer") || Equals(name, "nav") || Equals(name, "blockquote")
        || Equals(name, "pre") || Equals(name, "hr") || Equals(name, "figure")
        || (name.Length == 2 && (name[0] is 'h' or 'H') && name[1] is >= '1' and <= '6');

    private static bool Equals(ReadOnlySpan<char> name, string other) =>
        name.Equals(other, StringComparison.OrdinalIgnoreCase);

    private static void AppendBreak(StringBuilder text)
    {
        if (text.Length > 0 && text[^1] != '\n')
            text.Append('\n');
    }

    /// <summary>Index just past <c>&lt;/name&gt;</c>, starting the search at
    /// <paramref name="from"/>, or -1 when the document ends first.</summary>
    private static int SkipElement(string html, int from, ReadOnlySpan<char> name)
    {
        for (int i = from; i < html.Length; i++)
        {
            if (html[i] != '<' || i + 1 >= html.Length || html[i + 1] != '/')
                continue;
            int end = html.IndexOf('>', i);
            if (end < 0)
                return -1;
            if (TagName(html.AsSpan(i + 2, end - i - 2)).Equals(name, StringComparison.OrdinalIgnoreCase))
                return end + 1;
        }
        return -1;
    }
}
