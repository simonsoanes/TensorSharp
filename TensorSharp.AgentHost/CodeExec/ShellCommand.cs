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
using System.Text.Json;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>One simple command, and where it sits in the line it came from.</summary>
    /// <param name="Text">The command, trimmed.</param>
    /// <param name="Start">Its offset in the original line (with CRLF normalised to LF).</param>
    /// <param name="Length">Its length there.</param>
    public readonly record struct ShellSegment(string Text, int Start, int Length);

    /// <summary>
    /// Reading a model's <c>command</c> argument: what it says, what it is, and what it
    /// is therefore allowed to touch.
    ///
    /// <para>
    /// The program runner this replaces got a real security property out of having two
    /// PHASES — installing reached a pinned registry proxy, running reached nothing at
    /// all. A shell has no phases, and the two obvious ways to recover the property are
    /// both bad: dropping it is a regression, and asking the model to declare
    /// <c>network: true</c> is no property at all, because the model will always say yes.
    /// </para>
    /// <para>
    /// So the HOST decides, from the command itself, deterministically. The line is split
    /// into its simple commands and each is checked against a list of package managers.
    /// Nothing here is a guess about intent: a line that never names an installer never
    /// gets a socket, which is the case that matters.
    /// </para>
    /// </summary>
    public static class ShellCommand
    {
        /// <summary>
        /// Read the <c>command</c> argument, whatever shape it arrived in.
        ///
        /// <para>
        /// It is declared as a string because <c>ToolParameter</c> cannot describe an
        /// array — but models trained on Codex emit <c>["bash","-lc","..."]</c> anyway,
        /// and models trained on nothing in particular emit a bare JSON array of words.
        /// Both are accepted: an argv vector whose head is a shell and whose tail is a
        /// <c>-c</c> script yields that script, and any other array is joined. Refusing
        /// either would cost a round to teach a shape the model already knows.
        /// </para>
        /// </summary>
        public static string? ReadCommand(object? raw)
        {
            switch (raw)
            {
                case null:
                    return null;

                case string s:
                    return FromArrayText(s) ?? s;

                case JsonElement { ValueKind: JsonValueKind.String } je:
                    {
                        string text = je.GetString() ?? string.Empty;
                        return FromArrayText(text) ?? text;
                    }

                case JsonElement { ValueKind: JsonValueKind.Array } je:
                    return FromArgv(je.EnumerateArray()
                        .Select(e => e.ValueKind == JsonValueKind.String ? e.GetString() ?? string.Empty : e.ToString())
                        .ToList());

                case IEnumerable<object> list:
                    return FromArgv(list.Select(o => o?.ToString() ?? string.Empty).ToList());

                default:
                    return raw.ToString();
            }
        }

        /// <summary>A string that is really a JSON array, which several families emit.</summary>
        private static string? FromArrayText(string text)
        {
            string trimmed = text.TrimStart();
            if (trimmed.Length == 0 || trimmed[0] != '[')
                return null;
            try
            {
                using JsonDocument doc = JsonDocument.Parse(text);
                if (doc.RootElement.ValueKind != JsonValueKind.Array)
                    return null;
                return FromArgv(doc.RootElement.EnumerateArray()
                    .Select(e => e.ValueKind == JsonValueKind.String ? e.GetString() ?? string.Empty : e.ToString())
                    .ToList());
            }
            catch (JsonException)
            {
                return null;
            }
        }

        /// <summary>Collapse an argv vector to the command line it stands for.</summary>
        internal static string FromArgv(IReadOnlyList<string> argv)
        {
            if (argv.Count == 0)
                return string.Empty;
            if (argv.Count == 1)
                return argv[0];

            // ["bash","-lc","<script>"] and ["powershell","-NoProfile","-Command","<script>"]
            // are the shapes Codex-trained models emit. The script is the whole payload;
            // re-quoting the vector around it would run the shell inside the shell and
            // double every layer of escaping the model already got right.
            string head = System.IO.Path.GetFileNameWithoutExtension(argv[0]);
            bool isShell = head.Equals("bash", StringComparison.OrdinalIgnoreCase)
                        || head.Equals("sh", StringComparison.OrdinalIgnoreCase)
                        || head.Equals("zsh", StringComparison.OrdinalIgnoreCase)
                        || head.Equals("pwsh", StringComparison.OrdinalIgnoreCase)
                        || head.Equals("powershell", StringComparison.OrdinalIgnoreCase)
                        || head.Equals("cmd", StringComparison.OrdinalIgnoreCase);
            if (isShell)
            {
                for (int i = 1; i < argv.Count - 1; i++)
                {
                    string flag = argv[i];
                    if (flag.Equals("-c", StringComparison.Ordinal)
                        || flag.Equals("-lc", StringComparison.Ordinal)
                        || flag.Equals("-ic", StringComparison.Ordinal)
                        || flag.Equals("/c", StringComparison.OrdinalIgnoreCase)
                        || flag.Equals("-Command", StringComparison.OrdinalIgnoreCase)
                        || flag.Equals("-EncodedCommand", StringComparison.OrdinalIgnoreCase))
                    {
                        return argv[i + 1];
                    }
                }
            }

            // Any other vector is a plain command with arguments. Quote each so a space or
            // a metacharacter inside one argument stays inside it.
            return string.Join(" ", argv.Select(QuotePosix));
        }

        /// <summary>Single-quote for a POSIX shell, the only form with no escapes inside it.</summary>
        internal static string QuotePosix(string value)
        {
            if (value.Length > 0 && value.All(c => char.IsLetterOrDigit(c) || "-_./=:,+@".IndexOf(c) >= 0))
                return value;
            return "'" + value.Replace("'", "'\\''", StringComparison.Ordinal) + "'";
        }

        /// <summary>Single-quote for PowerShell, where doubling is the escape.</summary>
        internal static string QuotePowerShell(string value) =>
            "'" + value.Replace("'", "''", StringComparison.Ordinal) + "'";

        // ---- splitting -----------------------------------------------------

        /// <summary>
        /// The simple commands of a line, in order, with quoting and heredoc bodies
        /// respected.
        ///
        /// <para>
        /// Deliberately NOT a shell parser. It answers exactly one question — "which
        /// words start a command here" — and it answers it conservatively: anything it
        /// cannot make sense of yields fewer, larger segments, and a larger segment can
        /// only ever be classified as less privileged, never more. A heredoc body is
        /// skipped whole, because a patch full of <c>|</c> and <c>&amp;&amp;</c> is data.
        /// </para>
        /// </summary>
        public static IReadOnlyList<string> SplitSimpleCommands(string? command) =>
            SplitSegments(command).Select(s => s.Text).ToList();

        /// <summary>
        /// The same split, with each command's SPAN in the original line.
        ///
        /// <para>
        /// The spans are what let one command be replaced by another without rebuilding
        /// the line: an install is answered by the host, and the segment that asked for it
        /// is substituted with <c>true</c> or <c>false</c> so the operators around it —
        /// <c>&amp;&amp;</c>, <c>||</c>, a pipeline, a loop body — keep meaning exactly
        /// what the model wrote.
        /// </para>
        /// </summary>
        public static IReadOnlyList<ShellSegment> SplitSegments(string? command)
        {
            var segments = new List<ShellSegment>();
            if (string.IsNullOrWhiteSpace(command))
                return segments;

            var current = new StringBuilder();
            var pendingHeredocs = new List<(string Tag, bool StripTabs)>();
            char quote = '\0';
            int depth = 0;
            int segmentStart = 0;

            string text = command!.Replace("\r\n", "\n", StringComparison.Ordinal);
            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];

                if (quote != '\0')
                {
                    current.Append(c);
                    if (c == '\\' && quote == '"' && i + 1 < text.Length)
                    {
                        current.Append(text[++i]);
                        continue;
                    }
                    if (c == quote)
                        quote = '\0';
                    continue;
                }

                switch (c)
                {
                    case '\\' when i + 1 < text.Length:
                        current.Append(c).Append(text[++i]);
                        continue;

                    case '\'':
                    case '"':
                        quote = c;
                        current.Append(c);
                        continue;

                    case '#' when current.Length == 0 || char.IsWhiteSpace(text[i - 1]):
                        // A comment runs to the end of the line and cannot contain an operator.
                        while (i < text.Length && text[i] != '\n')
                            i++;
                        i--;
                        continue;

                    case '(':
                    case '{':
                        depth++;
                        current.Append(c);
                        continue;

                    case ')':
                    case '}':
                        if (depth > 0) depth--;
                        current.Append(c);
                        continue;

                    case '<' when i + 1 < text.Length && text[i + 1] == '<' && !IsHereString(text, i):
                        {
                            i += 2;
                            bool strip = i < text.Length && text[i] == '-';
                            if (strip) i++;
                            while (i < text.Length && (text[i] == ' ' || text[i] == '\t')) i++;
                            var tag = new StringBuilder();
                            char tagQuote = '\0';
                            if (i < text.Length && (text[i] == '\'' || text[i] == '"'))
                                tagQuote = text[i++];
                            while (i < text.Length
                                   && (tagQuote != '\0' ? text[i] != tagQuote : (char.IsLetterOrDigit(text[i]) || text[i] == '_')))
                                tag.Append(text[i++]);
                            if (tagQuote != '\0' && i < text.Length) i++;
                            i--;
                            if (tag.Length > 0)
                                pendingHeredocs.Add((tag.ToString(), strip));
                            continue;
                        }

                    case '\n' when pendingHeredocs.Count > 0:
                        {
                            // Skip every queued heredoc body before looking for operators again.
                            i++;
                            foreach ((string tag, bool strip) in pendingHeredocs)
                                i = SkipHeredocBody(text, i, tag, strip);
                            pendingHeredocs.Clear();
                            i--;
                            Flush(segments, current, ref segmentStart, i + 1);
                            continue;
                        }

                    case '\n':
                    case ';':
                        if (depth == 0) { Flush(segments, current, ref segmentStart, i); continue; }
                        current.Append(c);
                        continue;

                    case '&' when depth == 0:
                        {
                            // `2>&1`, `>&2`, `&>log`, `&>>log`: here the '&' belongs to the
                            // REDIRECTION OPERATOR, not to the line, and splitting on it
                            // was a live false success.
                            //
                            // `pip3 install numpy 2>&1 | tail -5` split into
                            // ["pip3 install numpy 2>", "1", "tail -5"]. The install was
                            // read correctly, but its recorded SPAN ended at the '>', so
                            // substituting it produced `true&1 | tail -5` — bash
                            // backgrounds `true`, runs `1` (command not found), and the
                            // pipeline exits with tail's status, which is 0. Reproduced:
                            //   $ bash -c 'true&1 | tail -5'
                            //   bash: 1: command not found
                            //   exit=0
                            // So the model asked for the last five lines of its install
                            // output, never got them, and was told the call succeeded.
                            //
                            // Both spellings, because the character before the '&' differs
                            // between them: in `2>&1` it is '>', and in `x &> log` it is a
                            // space.
                            char previous = current.Length > 0 ? current[^1] : '\0';
                            if (previous == '>' || previous == '<'
                                || (i + 1 < text.Length && text[i + 1] == '>'))
                            {
                                current.Append(c);
                                continue;
                            }

                            int end = i;
                            if (i + 1 < text.Length && text[i + 1] == '&') i++;
                            Flush(segments, current, ref segmentStart, end);
                            continue;
                        }

                    case '|' when depth == 0:
                        {
                            int end = i;
                            if (i + 1 < text.Length && text[i + 1] == '|') i++;
                            Flush(segments, current, ref segmentStart, end);
                            continue;
                        }

                    default:
                        current.Append(c);
                        continue;
                }
            }

            Flush(segments, current, ref segmentStart, text.Length);
            return segments;
        }

        /// <summary><c>&lt;&lt;&lt;</c> is a here-STRING: one word, no body to skip.</summary>
        private static bool IsHereString(string text, int i) =>
            i + 2 < text.Length && text[i + 2] == '<';

        private static int SkipHeredocBody(string text, int from, string tag, bool stripTabs)
        {
            int i = from;
            while (i < text.Length)
            {
                int lineEnd = text.IndexOf('\n', i);
                if (lineEnd < 0) lineEnd = text.Length;
                string line = text.Substring(i, lineEnd - i);
                if ((stripTabs ? line.TrimStart('\t') : line).TrimEnd() == tag)
                    return Math.Min(lineEnd + 1, text.Length);
                i = lineEnd + 1;
            }
            return text.Length;
        }

        /// <summary>
        /// Close the segment that ends at <paramref name="end"/>, recording where its
        /// non-whitespace text actually starts and stops in the original line.
        /// </summary>
        private static void Flush(
            List<ShellSegment> segments, StringBuilder current, ref int start, int end)
        {
            string raw = current.ToString();
            current.Clear();

            string trimmed = raw.Trim();
            if (trimmed.Length > 0)
            {
                int leading = raw.Length - raw.TrimStart().Length;
                segments.Add(new ShellSegment(trimmed, start + leading, trimmed.Length));
            }
            start = end + 1;
        }

        /// <summary>
        /// Drop redirection operators and their targets.
        ///
        /// <para>
        /// Both spellings: joined (<c>&gt;out.txt</c>, <c>2&gt;&amp;1</c>) and separated
        /// (<c>&gt; out.txt</c>). Anything whose target is not a following plain word is
        /// dropped as the operator alone, because a redirection that cannot be read is
        /// still not part of the command.
        /// </para>
        /// </summary>
        private static List<string> WithoutRedirections(List<string> words)
        {
            var kept = new List<string>(words.Count);
            for (int i = 0; i < words.Count; i++)
            {
                string word = words[i];
                if (!IsRedirection(word, out bool targetIsJoined))
                {
                    kept.Add(word);
                    continue;
                }
                // `> out.txt` — the target is the next word, and it is not part of the
                // command either. `2>&1` and `>out.txt` carry their own target.
                if (!targetIsJoined && i + 1 < words.Count)
                    i++;
            }
            return kept;
        }

        /// <summary>
        /// True when <paramref name="word"/> is a redirection: an optional descriptor
        /// number, then <c>&lt;</c> or <c>&gt;</c> (or <c>&gt;&gt;</c>), then optionally
        /// <c>&amp;</c> and a target.
        /// </summary>
        private static bool IsRedirection(string word, out bool targetIsJoined)
        {
            targetIsJoined = false;
            int i = 0;
            while (i < word.Length && char.IsAsciiDigit(word[i]))
                i++;
            if (i >= word.Length || (word[i] != '<' && word[i] != '>'))
                return false;

            char kind = word[i++];
            if (i < word.Length && word[i] == kind)
                i++;                                    // ">>"
            if (i < word.Length && word[i] == '&')
                i++;                                    // ">&"
            targetIsJoined = i < word.Length;           // ">out.txt", "2>&1"
            return true;
        }

        /// <summary>
        /// True when the command ends by putting something in the background with
        /// <c>&amp;</c>.
        ///
        /// <para>
        /// Refused rather than run, because it cannot work here and the way it failed was
        /// the worst kind: every call is a fresh confined process whose whole tree is
        /// killed when the call returns, so a backgrounded job is dead before the model
        /// can look at it — and until the drain was bounded, the call did not return at
        /// all. Nothing in the tool description said <c>&amp;</c> would not work, while
        /// the <c>run_in_background</c> parameter invites exactly this intent ("something
        /// that is meant to keep running — a server, a watcher").
        /// </para>
        /// <para>
        /// Only a TRAILING <c>&amp;</c>. <c>a &amp; b</c> backgrounds <c>a</c> and then
        /// runs <c>b</c> in the foreground, which is a different shape, and <c>&amp;&amp;</c>
        /// is not backgrounding at all.
        /// </para>
        /// </summary>
        public static bool EndsWithBackground(string? command)
        {
            if (string.IsNullOrWhiteSpace(command))
                return false;

            // Comments and trailing blank lines first, so `cmd &   # go` still counts.
            string tail = command!.TrimEnd();
            int lastNewline = tail.LastIndexOf('\n');
            string lastLine = (lastNewline >= 0 ? tail.Substring(lastNewline + 1) : tail).TrimEnd();
            if (!lastLine.EndsWith('&'))
                return false;

            // `&&` is a conjunction, not a background request.
            return !lastLine.EndsWith("&&", StringComparison.Ordinal);
        }

        /// <summary>
        /// True when <paramref name="exitCode"/> is a normal, successful outcome for the
        /// command that produced it.
        ///
        /// <para>
        /// <b>Without this, the host manufactures failures out of correct commands.</b>
        /// <c>grep</c> exits 1 when it finds nothing — that is its ANSWER, not an error.
        /// So does <c>rg</c>. <c>diff</c> exits 1 when the files differ, which is the
        /// question being asked. <c>test -f x</c> exits 1 when x is absent.
        /// <c>git diff --quiet</c> exits 1 when there ARE changes and 0 when the tree is
        /// clean, so a clean tree and a dirty one are both perfectly good answers and one
        /// of them was being reported as a broken command.
        /// </para>
        /// <para>
        /// A false FAILURE is as expensive as a false success and less obvious: the model
        /// is handed a correct result labelled broken, and its next round is spent fixing
        /// a command that had already answered. In a loop where 39.7% of rounds are
        /// already recovery, that is recovery from nothing.
        /// </para>
        /// <para>
        /// This is Claude Code's mechanism, and it is deliberately a CLOSED list of
        /// commands rather than a rule about exit codes: its Bash tool treats exit 1 as a
        /// valid result "only when Claude Code recognizes exit code 1 as a benign outcome
        /// for that command", for <c>grep</c>, <c>rg</c>, <c>egrep</c>, <c>fgrep</c>,
        /// <c>find</c>, <c>diff</c>, <c>test</c> and <c>[</c>, plus <c>git diff</c> and
        /// <c>git grep</c>. Only exit 1 is ever exempted, and only for those names —
        /// <c>grep</c> exits 2 on a real error, and that stays a failure.
        /// </para>
        /// <para>
        /// The word checked is the LAST simple command's, because that is whose status the
        /// shell reports. <c>grep x f | wc -l</c> exits with <c>wc</c>'s status and is not
        /// exempt; <c>cat f | grep x</c> exits with <c>grep</c>'s and is.
        /// </para>
        /// </summary>
        public static bool ExitCodeIsBenign(string? command, int exitCode)
        {
            if (exitCode != 1 || string.IsNullOrWhiteSpace(command))
                return false;

            IReadOnlyList<ShellSegment> segments = SplitSegments(command);
            if (segments.Count == 0)
                return false;

            IReadOnlyList<string> words = WordsOf(segments[^1].Text);
            if (words.Count == 0)
                return false;

            string name = System.IO.Path.GetFileName(words[0]);
            if (BenignAtOne.Contains(name))
                return true;

            // `git diff` and `git grep` only — `git commit` exiting 1 is a real failure.
            return words.Count >= 2
                && string.Equals(name, "git", StringComparison.Ordinal)
                && (words[1] == "diff" || words[1] == "grep");
        }

        private static readonly HashSet<string> BenignAtOne = new(StringComparer.Ordinal)
        {
            "grep", "egrep", "fgrep", "rg", "ripgrep", "ugrep",
            "find", "diff", "cmp", "test", "[",
        };

        /// <summary>The words of one simple command, quotes removed, redirections dropped.</summary>
        internal static IReadOnlyList<string> WordsOf(string segment)
        {
            var words = new List<string>();
            var current = new StringBuilder();
            char quote = '\0';
            bool any = false;

            for (int i = 0; i < segment.Length; i++)
            {
                char c = segment[i];
                if (quote != '\0')
                {
                    if (c == '\\' && quote == '"' && i + 1 < segment.Length) { current.Append(segment[++i]); any = true; continue; }
                    if (c == quote) { quote = '\0'; continue; }
                    current.Append(c);
                    any = true;
                    continue;
                }
                switch (c)
                {
                    case '\\' when i + 1 < segment.Length: current.Append(segment[++i]); any = true; break;
                    case '\'':
                    case '"': quote = c; any = true; break;
                    case ' ':
                    case '\t':
                    case '\n':
                        if (any) { words.Add(current.ToString()); current.Clear(); any = false; }
                        break;
                    default: current.Append(c); any = true; break;
                }
            }
            if (any)
                words.Add(current.ToString());

            // The redirections this method's summary has always promised to drop, and did
            // not. Their absence had a real and expensive shape: `pip3 install python-pptx
            // 2>&1 | tail -5` handed `2>&1` to the install reader as a PACKAGE NAME, which
            // PackageInstaller then rejected — at install time, not parse time — so the
            // install segment was substituted out with `false`, the residual
            // `false | tail -5` ran, and the result said **exit 0**. A one-word redirection
            // produced this codebase's cardinal defect: success reported for something
            // that was never done. Logged twice on the new shell surface.
            //
            // A redirection is the operator word and the target that follows it. Dropped
            // only when the target is a plain word: `> "$OUT"` has already been through
            // quote removal here, and dropping a word this method cannot identify would
            // silently shorten a command line it is being asked to classify.
            words = WithoutRedirections(words);

            // Leading VAR=value assignments are environment, not the command; and a
            // compound construct's keyword is not the command either. Splitting
            //   for p in a b; do pip install $p; done
            // on ';' leaves the middle segment as "do pip install $p", whose first word is
            // `do` — so the install went unrecognised and ran with no network. Stripping
            // both can only ever make MORE commands classifiable, never fewer, so it
            // cannot turn a plain command into one that is granted a socket by accident:
            // the words revealed still have to name a package manager.
            int start = 0;
            while (start < words.Count && (IsAssignment(words[start]) || IsShellKeyword(words[start])))
                start++;
            return start == 0 ? words : words.Skip(start).ToList();
        }

        /// <summary>Words that introduce a construct rather than naming a command.</summary>
        private static readonly string[] ShellKeywords =
        {
            "do", "then", "else", "elif", "if", "while", "until", "for", "case", "in",
            "{", "(", "!", "time", "command", "exec", "nohup", "eval", "sudo", "env",
        };

        private static bool IsShellKeyword(string word) =>
            ShellKeywords.Contains(word, StringComparer.Ordinal);

        private static bool IsAssignment(string word)
        {
            int eq = word.IndexOf('=');
            if (eq <= 0)
                return false;
            for (int i = 0; i < eq; i++)
            {
                if (!char.IsLetterOrDigit(word[i]) && word[i] != '_')
                    return false;
            }
            return true;
        }

        // ---- installer classification --------------------------------------

        /// <summary>
        /// Package managers, and the subcommand that means "reach the registry".
        ///
        /// <para>
        /// An empty subcommand list means every invocation of that tool needs the
        /// network. A tool that is NOT on this list never gets a socket, which is why
        /// <c>curl</c> and <c>wget</c> are deliberately absent: a general fetcher is how
        /// a confined command would reach anything at all, and "the model said it was
        /// downloading a dependency" is not a control.
        /// </para>
        /// </summary>
        private static readonly (string Tool, string[] Subcommands)[] Installers =
        {
            ("pip",    new[] { "install", "download", "wheel" }),
            ("pip3",   new[] { "install", "download", "wheel" }),
            // `uv` and `npx` are RUNNERS as much as installers, and an empty subcommand
            // list made every invocation of them an install — so `uv run main.py` and
            // `npx tsc --noEmit`, which install nothing, were refused with "not an
            // installer this host can run on your behalf". That refusal states a rule that
            // does not apply and offers no alternative. Only the subcommands that really
            // fetch are listed. `uvx` stays all-invocations: it exists to fetch a tool and
            // run it, and there is no form of it that does not.
            ("uv",     new[] { "pip", "add", "sync", "lock", "tool", "install" }),
            ("uvx",    Array.Empty<string>()),
            ("npm",    new[] { "install", "i", "add", "ci", "update", "exec" }),
            ("pnpm",   new[] { "install", "i", "add", "update" }),
            ("yarn",   new[] { "install", "add", "up" }),
            // `npx` stays ALL invocations, and the test asserting that is right: `npx
            // cowsay hi` fetches cowsay from the registry if it is not already in
            // node_modules, which is the whole point of npx. Deciding otherwise would need
            // to know what is in this session's node_modules, which a static parser cannot
            // see. `npx tsc` on a package that IS installed is therefore refused too — so
            // the refusal names the way to run it (ShellInstall.TryReadOne), which is the
            // half that was missing.
            ("npx",    Array.Empty<string>()),
            ("poetry", new[] { "install", "add", "update", "lock" }),
            ("pipenv", new[] { "install", "sync", "update" }),
            ("gem",    new[] { "install", "update" }),
            ("cargo",  new[] { "install", "add", "fetch", "update" }),
            // `go mod tidy` and `go mod download` fetch; `go mod edit`/`go mod why` do
            // not, but the distinction needs a third word and the table only reads two.
            // Kept as-is deliberately: over-recognising `go mod` refuses a command the
            // host could not have performed anyway and says why, which is the better
            // failure of the two.
            ("go",     new[] { "get", "install", "mod" }),
            ("dotnet", new[] { "restore", "add" }),
            // System package managers are listed so they are RECOGNISED, not so they can
            // be performed: the host installs Python and JavaScript libraries and nothing
            // else, and ShellInstall refuses these by name. Without them here, `apt install
            // ffmpeg` on a Linux host runs, fails on a permission or lock error, and tells
            // the model nothing about the actual rule — while on macOS it fails as
            // "command not found", which reads as a different problem entirely.
            ("apt", new[] { "install" }),
            ("apt-get", new[] { "install" }),
            ("yum", new[] { "install" }),
            ("dnf", new[] { "install" }),
            ("apk", new[] { "add" }),
            ("brew", new[] { "install" }),
        };

        /// <summary>Interpreters that reach a registry only through <c>-m pip</c>.</summary>
        private static readonly string[] PythonInterpreters = { "python", "python3", "py" };

        private static bool IsPythonInterpreter(string tool)
        {
            if (PythonInterpreters.Contains(tool, StringComparer.OrdinalIgnoreCase))
                return true;
            // python3.12, python3.13 — a version suffix, not an unrelated tool.
            return tool.StartsWith("python3.", StringComparison.OrdinalIgnoreCase)
                && tool.Length > "python3.".Length
                && tool["python3.".Length..].All(char.IsAsciiDigit);
        }

        /// <summary>What one simple command is.</summary>
        internal static bool IsInstallCommand(string segment)
        {
            IReadOnlyList<string> words = WordsOf(segment);
            if (words.Count == 0)
                return false;

            string tool = System.IO.Path.GetFileNameWithoutExtension(words[0]);

            // `python -m pip install x`, and `python3 -m uv pip install x`. Matched
            // exactly, plus the versioned spellings — a prefix test sent every tool whose
            // name merely begins with "py" (pytest, pylint) down this path.
            if (IsPythonInterpreter(tool))
            {
                int m = words.ToList().IndexOf("-m");
                if (m >= 0 && m + 1 < words.Count)
                    return IsInstallCommand(string.Join(" ", words.Skip(m + 1).Select(QuotePosix)));
                return false;
            }

            foreach ((string name, string[] subcommands) in Installers)
            {
                if (!tool.Equals(name, StringComparison.OrdinalIgnoreCase)
                    && !(name == "pip" && tool.StartsWith("pip3.", StringComparison.OrdinalIgnoreCase)))
                {
                    continue;
                }
                if (subcommands.Length == 0)
                    return true;
                // The subcommand is the first word that is not an option.
                foreach (string word in words.Skip(1))
                {
                    if (word.StartsWith('-'))
                        continue;
                    return subcommands.Contains(word, StringComparer.OrdinalIgnoreCase);
                }
                return false;
            }

            return false;
        }

        /// <summary>
        /// True when any simple command in the line asks to install packages.
        ///
        /// <para>
        /// A single predicate rather than the three-way classification this used to
        /// return. That distinguished "every part is an install" from "some parts are",
        /// which mattered while the LINE was given a socket and the two cases shared it
        /// differently. Installs are performed by the host now and substituted out, so
        /// both cases are handled identically and the extra state was decoration.
        /// </para>
        /// </summary>
        public static bool ContainsInstall(string? command) =>
            SplitSimpleCommands(command).Any(IsInstallCommand);

        // ---- apply_patch interception --------------------------------------

        /// <summary>
        /// The patch body when this command line is nothing but an <c>apply_patch</c>
        /// call, or null when it is not.
        ///
        /// <para>
        /// Codex makes <c>apply_patch</c> reachable from the shell and answers it in the
        /// harness rather than executing anything, and that is the right shape here for a
        /// reason beyond fidelity: this host has no <c>apply_patch</c> binary to install
        /// into the sandbox, and the alternative — a script that calls back out — would
        /// need an interpreter the workspace is not guaranteed to have. Answering it in
        /// the host also makes it behave identically on Windows, where there are no
        /// heredocs at all.
        /// </para>
        /// <para>
        /// Recognised shapes, and only these: the command is a single simple command whose
        /// first word is <c>apply_patch</c> (or one of the spellings models invent), and
        /// the patch arrives either as one heredoc or as one quoted argument. Anything
        /// else — a pipeline, a redirect, a second command — is left to the shell, where
        /// the workspace's own shim explains the two shapes that work.
        /// </para>
        /// </summary>
        public static bool TryReadApplyPatch(string? command, out string patch)
        {
            patch = string.Empty;
            if (string.IsNullOrWhiteSpace(command))
                return false;

            string text = command!.Replace("\r\n", "\n", StringComparison.Ordinal).Trim();

            int firstBreak = text.IndexOf('\n');
            string firstLine = (firstBreak < 0 ? text : text.Substring(0, firstBreak)).Trim();

            IReadOnlyList<string> head = WordsOf(firstLine);
            if (head.Count == 0)
                return false;
            string name = head[0];
            if (!name.Equals(Skills.SkillToolNames.ApplyPatch, StringComparison.Ordinal)
                && !Skills.SkillToolNames.IsApplyPatchAlias(name))
            {
                return false;
            }

            // Shape one: apply_patch <<'EOF' … EOF
            int heredoc = firstLine.IndexOf("<<", StringComparison.Ordinal);
            if (heredoc >= 0 && firstBreak >= 0)
            {
                // A pipeline or a second command on the SAME line means the call asks for
                // more than this can answer. Same rule as a command after the terminator:
                // hand the whole line to the shell rather than doing half of it here.
                if (firstLine.IndexOfAny(new[] { '|', '&', ';', '>' }, heredoc + 2) >= 0)
                    return false;

                int i = heredoc + 2;
                bool strip = i < firstLine.Length && firstLine[i] == '-';
                if (strip) i++;
                while (i < firstLine.Length && (firstLine[i] == ' ' || firstLine[i] == '\t')) i++;
                var tag = new StringBuilder();
                char tagQuote = '\0';
                if (i < firstLine.Length && (firstLine[i] == '\'' || firstLine[i] == '"'))
                    tagQuote = firstLine[i++];
                while (i < firstLine.Length
                       && (tagQuote != '\0' ? firstLine[i] != tagQuote : (char.IsLetterOrDigit(firstLine[i]) || firstLine[i] == '_')))
                {
                    tag.Append(firstLine[i++]);
                }
                if (tag.Length == 0)
                    return false;

                string body = text.Substring(firstBreak + 1);
                int end = FindHeredocEnd(body, tag.ToString(), strip);
                if (end < 0)
                    return false;

                // Anything after the closing TAG LINE is a second command, and this call
                // can only answer one thing. Refusing the interception hands the whole
                // line to the shell, where it fails visibly on the missing apply_patch
                // binary — far better than applying the patch and silently discarding the
                // command the model wrote after it, which is what this used to do: the
                // check looked at the text after the tag's OFFSET, which begins with the
                // tag itself, so the guard always passed.
                int tagLineEnd = body.IndexOf('\n', end);
                string after = tagLineEnd < 0 ? string.Empty : body.Substring(tagLineEnd + 1).Trim();
                if (after.Length > 0)
                    return false;

                patch = body.Substring(0, Math.Max(0, end)).TrimEnd('\n');
                return patch.Length > 0;
            }

            // Shape two: apply_patch '*** Begin Patch … *** End Patch', where the whole
            // envelope is ONE quoted argument. Read from the whole command rather than
            // from its first line: an envelope is multi-line by nature, and parsing only
            // the first line yielded a one-line "patch" that always failed downstream.
            IReadOnlyList<string> whole = WordsOf(text);
            if (whole.Count == 2 && whole[1].Contains("*** Begin Patch", StringComparison.Ordinal))
            {
                patch = whole[1];
                return true;
            }

            return false;
        }

        /// <summary>
        /// True when any simple command in the line invokes <c>apply_patch</c>.
        ///
        /// <para>
        /// Asked AFTER <see cref="TryReadApplyPatch"/> has declined, to tell "the model
        /// wants a patch applied and wrote it in a shape this host cannot answer" apart
        /// from "the model is doing something else". The first has to be refused; letting
        /// the shell run it produces a command-not-found for one word, a success exit code
        /// from the next word, and a file that was never patched.
        /// </para>
        /// </summary>
        public static bool InvokesApplyPatch(string? command)
        {
            foreach (string segment in SplitSimpleCommands(command))
            {
                IReadOnlyList<string> words = WordsOf(segment);
                if (words.Count == 0)
                    continue;
                string first = words[0];
                if (first.Equals(Skills.SkillToolNames.ApplyPatch, StringComparison.Ordinal)
                    || Skills.SkillToolNames.IsApplyPatchAlias(first))
                {
                    return true;
                }
            }
            return false;
        }

        private static int FindHeredocEnd(string body, string tag, bool stripTabs)
        {
            int i = 0;
            while (i <= body.Length)
            {
                int lineEnd = body.IndexOf('\n', i);
                if (lineEnd < 0) lineEnd = body.Length;
                string line = body.Substring(i, lineEnd - i);
                if ((stripTabs ? line.TrimStart('\t') : line).TrimEnd() == tag)
                    return i;
                if (lineEnd == body.Length)
                    break;
                i = lineEnd + 1;
            }
            return -1;
        }

        /// <summary>A one-line label for a command, for logs and the UI's progress line.</summary>
        public static string Summarize(string? command, int max = 60)
        {
            if (string.IsNullOrWhiteSpace(command))
                return "(empty)";
            string first = command!.Replace("\r\n", "\n", StringComparison.Ordinal)
                .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .FirstOrDefault(l => l.Trim().Length > 0)?.Trim() ?? command.Trim();
            return first.Length <= max
                ? first
                : first.Substring(0, max).TrimEnd() + "…";
        }

        /// <summary>Render a byte count the way the tool results do.</summary>
        internal static string FormatBytes(long bytes) => bytes switch
        {
            < 1024 => bytes.ToString(CultureInfo.InvariantCulture) + " B",
            < 1024 * 1024 => (bytes / 1024.0).ToString("0.#", CultureInfo.InvariantCulture) + " KB",
            _ => (bytes / (1024.0 * 1024.0)).ToString("0.#", CultureInfo.InvariantCulture) + " MB",
        };
    }
}
