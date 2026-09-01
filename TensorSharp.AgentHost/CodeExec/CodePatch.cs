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
    /// <summary>
    /// Codex's <c>apply_patch</c> envelope: one call that creates, updates, deletes and
    /// renames FILES, several of them at once.
    ///
    /// <para>
    /// This is the deterministic editor the shell tool leans on, and the reason it exists
    /// alongside a shell that could technically rewrite any file with a heredoc: a
    /// heredoc re-emits the WHOLE file, so a three-line fix costs the tokens of every line
    /// that was already right and re-rolls each of them, which is how a second bug appears
    /// in code the model had got correct. A patch changes what it names and leaves the
    /// rest byte-identical, and the HOST places the bytes — from anchors it either finds
    /// or refuses to guess at.
    /// </para>
    /// <code>
    /// *** Begin Patch
    /// *** Add File: helpers.py
    /// +def clean(s):
    /// +    return s.strip()
    /// *** Update File: main.py
    /// @@ def process(rows):
    /// -    return [r for r in rows]
    /// +    return [clean(r) for r in rows]
    /// *** Delete File: scratch.txt
    /// *** End Patch
    /// </code>
    /// <para>
    /// Two deliberate departures from Codex. It ships the patch as a FREEFORM
    /// grammar-constrained argument, which avoids JSON-escaping every newline but needs
    /// an endpoint that can constrain decoding to a Lark grammar; TensorSharp serves
    /// small local models through a flat JSON tool schema, so the patch arrives as an
    /// ordinary string parameter — or, just as usefully, as a heredoc typed into the
    /// shell, which the host intercepts. And every path is confined to the session's
    /// working directory rather than being repo-relative, because this is a chat
    /// session's workspace, not a checkout.
    /// </para>
    /// <para>
    /// ALL-OR-NOTHING, which is the property that makes a multi-file patch safe to offer:
    /// every hunk is resolved against the current files first, and only if all of them
    /// resolve is anything written. A patch half-applied across three files leaves a
    /// workspace no one — model or user — can reason about, and the model's next move is
    /// to regenerate everything, which is precisely what a patch tool exists to avoid.
    /// The reference implementation applies operations one at a time with no rollback;
    /// that is a known weakness of it, not a behaviour to copy.
    /// </para>
    /// </summary>
    public static class CodePatch
    {
        /// <summary>What a section does to its file.</summary>
        public enum FileOp
        {
            /// <summary>Create it. Every body line is a <c>+</c> line.</summary>
            Add,

            /// <summary>Change it in place, and optionally rename it.</summary>
            Update,

            /// <summary>Remove it. No body.</summary>
            Delete,
        }

        /// <summary>One file's worth of the envelope.</summary>
        /// <param name="Op">What to do.</param>
        /// <param name="Path">The file, relative to the working directory.</param>
        /// <param name="MoveTo">Where to rename it to, for an update that also moves.</param>
        /// <param name="Body">The section's lines, exactly as written.</param>
        public readonly record struct FileSection(
            FileOp Op, string Path, string? MoveTo, IReadOnlyList<string> Body)
        {
            /// <summary>
            /// The newline style of the envelope this came from.
            ///
            /// <para>
            /// Carried because <see cref="V4ADiff.SplitDiffLines"/> strips every CR, so by
            /// the time the matcher sees a section the style is gone — and it is the only
            /// source of one for a file being CREATED, or for an empty file being updated.
            /// </para>
            /// </summary>
            public string Newline { get; init; } = "\n";
        }

        /// <summary>What one section actually did, for the result the model reads.</summary>
        /// <param name="Op">What was done.</param>
        /// <param name="Path">The file it was done to.</param>
        /// <param name="MovedTo">Its new name, when it was renamed.</param>
        /// <param name="LinesAdded">Lines inserted.</param>
        /// <param name="LinesRemoved">Lines deleted.</param>
        /// <param name="Fuzz">How far down the matching ladder this file's hunks landed.</param>
        public readonly record struct FileOutcome(
            FileOp Op, string Path, string? MovedTo, int LinesAdded, int LinesRemoved, int Fuzz)
        {
            /// <summary>
            /// Something the model must be told about a hunk that APPLIED, or null.
            ///
            /// <para>
            /// Carried beside <see cref="Fuzz"/> and for the same reason: a patch can land
            /// exactly where it was asked to and still not be the edit the model meant,
            /// and the only place that can be said is the result of the call that did it.
            /// </para>
            /// </summary>
            public string? Note { get; init; }
        }

        /// <summary>A drive-letter root (<c>C:\</c>) or a UNC root (<c>\\server</c>).</summary>
        private static readonly System.Text.RegularExpressions.Regex WindowsAbsolute =
            new(@"^([A-Za-z]:[\\/]|\\\\)",
                System.Text.RegularExpressions.RegexOptions.Compiled
                | System.Text.RegularExpressions.RegexOptions.CultureInvariant);

        private const string Begin = "*** Begin Patch";
        private const string End = "*** End Patch";
        private const string AddPrefix = "*** Add File:";
        private const string UpdatePrefix = "*** Update File:";
        private const string DeletePrefix = "*** Delete File:";
        private const string MovePrefix = "*** Move to:";

        /// <summary>
        /// Read an envelope into its sections, or say exactly what is wrong with it.
        ///
        /// <para>
        /// A malformed patch is a RECOVERABLE tool error, never an exception: the model
        /// gets the message, fixes the envelope and tries again. Codex makes the same
        /// choice explicitly, with the comment that a malformed patch must not abort the
        /// run during a pre-check.
        /// </para>
        /// </summary>
        public static bool TryParse(string? patch, out IReadOnlyList<FileSection> sections, out string? error)
        {
            var parsed = new List<FileSection>();
            sections = parsed;
            error = null;
            // `parsed` is reassigned by the JSON path's out parameter below, so `sections`
            // is bound to whichever list actually ends up filled at each return — never
            // once at the top, which silently returned an empty list for every envelope.

            if (string.IsNullOrWhiteSpace(patch))
            {
                error = "the patch was empty. Send the whole envelope, from '*** Begin Patch' to '*** End Patch'.";
                return false;
            }

            string text = StripFences(patch!);
            string newline = text.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";

            // Models trained on the Agents SDK send the JSON operation form instead of the
            // envelope. It costs forty lines to accept and a round to refuse.
            if (TryParseJson(text, out List<FileSection> json, out error))
            {
                sections = json;
                return true;
            }
            if (error != null)
                return false;

            List<string> lines = V4ADiff.SplitDiffLines(text);
            while (lines.Count > 0 && lines[0].Trim().Length == 0)
                lines.RemoveAt(0);
            while (lines.Count > 0 && lines[^1].Trim().Length == 0)
                lines.RemoveAt(lines.Count - 1);

            if (lines.Count == 0 || lines[0].Trim() != Begin)
            {
                error = $"a patch must start with '{Begin}' on a line of its own. "
                      + $"This one starts with:\n{(lines.Count > 0 ? lines[0] : "(nothing)")}";
                return false;
            }
            if (lines[^1].Trim() != End)
            {
                error = $"a patch must end with '{End}' on a line of its own. "
                      + $"This one ends with:\n{lines[^1]}\n"
                      + "If the patch was cut short, send it again in full — nothing was written.";
                return false;
            }

            int i = 1;
            int last = lines.Count - 1;
            while (i < last)
            {
                string line = lines[i];

                if (line.Trim().Length == 0) { i++; continue; }

                if (line.StartsWith(DeletePrefix, StringComparison.Ordinal))
                {
                    string path = line.Substring(DeletePrefix.Length).Trim();
                    if (!Validate(path, out error))
                        return false;
                    i++;
                    if (i < last && !IsHeader(lines[i]) && lines[i].Trim().Length > 0)
                    {
                        error = $"'{DeletePrefix} {path}' must not be followed by a diff — a delete removes "
                              + $"the whole file. Found:\n{lines[i]}";
                        return false;
                    }
                    parsed.Add(new FileSection(FileOp.Delete, path, null, Array.Empty<string>()) { Newline = newline });
                    continue;
                }

                if (line.StartsWith(AddPrefix, StringComparison.Ordinal))
                {
                    string path = line.Substring(AddPrefix.Length).Trim();
                    if (!Validate(path, out error))
                        return false;
                    i++;
                    var body = new List<string>();
                    while (i < last && !IsHeader(lines[i]))
                        body.Add(lines[i++]);
                    if (body.Count == 0)
                    {
                        error = $"'{AddPrefix} {path}' has no content. Every line of the new file goes "
                              + "after it prefixed with '+', including blank lines.";
                        return false;
                    }
                    parsed.Add(new FileSection(FileOp.Add, path, null, body) { Newline = newline });
                    continue;
                }

                if (line.StartsWith(UpdatePrefix, StringComparison.Ordinal))
                {
                    string path = line.Substring(UpdatePrefix.Length).Trim();
                    if (!Validate(path, out error))
                        return false;
                    i++;
                    string? moveTo = null;
                    if (i < last && lines[i].StartsWith(MovePrefix, StringComparison.Ordinal))
                    {
                        moveTo = lines[i].Substring(MovePrefix.Length).Trim();
                        if (!Validate(moveTo, out error))
                            return false;
                        i++;
                    }
                    var body = new List<string>();
                    while (i < last && !IsHeader(lines[i]))
                        body.Add(lines[i++]);
                    if (body.Count == 0 && moveTo == null)
                    {
                        error = $"'{UpdatePrefix} {path}' has no hunks. A '@@' line and the lines around "
                              + "your change go after it, or use '*** Move to:' if you only meant to rename it.";
                        return false;
                    }
                    parsed.Add(new FileSection(FileOp.Update, path, moveTo, body) { Newline = newline });
                    continue;
                }

                error = $"this line is not a file header and is not inside one:\n{line}\n"
                      + $"Every change starts with '{AddPrefix} <path>', '{UpdatePrefix} <path>' "
                      + $"or '{DeletePrefix} <path>'.";
                return false;
            }

            if (parsed.Count == 0)
            {
                error = "the patch names no files. It needs at least one "
                      + $"'{AddPrefix}', '{UpdatePrefix}' or '{DeletePrefix}' section.";
                return false;
            }

            sections = parsed;
            return true;
        }

        private static bool IsHeader(string line) =>
            line.StartsWith(AddPrefix, StringComparison.Ordinal)
            || line.StartsWith(UpdatePrefix, StringComparison.Ordinal)
            || line.StartsWith(DeletePrefix, StringComparison.Ordinal)
            || line.Trim() == End;

        /// <summary>
        /// Path rules, checked here so the message names the rule rather than the
        /// filesystem's version of it. Containment against the workspace is checked
        /// separately, by the path guard, when the write is about to happen.
        /// </summary>
        private static bool Validate(string path, out string? error)
        {
            error = null;
            if (string.IsNullOrWhiteSpace(path))
            {
                error = "a file header has no path after it.";
                return false;
            }
            // A bare ':' means a drive or an alternate stream only on Windows; on POSIX it
            // is an ordinary filename character, and refusing "log:1.txt" there told the
            // model its path was absolute when it plainly was not. What IS absolute
            // everywhere is a drive-letter prefix or a UNC root — a model that writes one
            // of those on a POSIX host means an absolute path, and creating a file whose
            // NAME contains backslashes instead would be a stranger outcome than refusing.
            if (System.IO.Path.IsPathRooted(path)
                || WindowsAbsolute.IsMatch(path)
                || (OperatingSystem.IsWindows() && path.Contains(':', StringComparison.Ordinal)))
            {
                error = $"'{path}' is an absolute path. Paths in a patch are relative to this "
                      + "conversation's working directory, never absolute.";
                return false;
            }
            return true;
        }

        /// <summary>
        /// Several families wrap a patch in a Markdown fence however firmly they are told
        /// not to. Unwrapping costs nothing and cannot change what the patch means.
        /// </summary>
        private static string StripFences(string patch)
        {
            string trimmed = patch.Trim();
            if (!trimmed.StartsWith("```", StringComparison.Ordinal))
                return patch;

            int firstBreak = trimmed.IndexOf('\n');
            if (firstBreak < 0)
                return patch;
            string body = trimmed.Substring(firstBreak + 1);
            int fence = body.LastIndexOf("```", StringComparison.Ordinal);
            return fence >= 0 ? body.Substring(0, fence) : body;
        }

        /// <summary>The Agents-SDK JSON form, when that is what arrived.</summary>
        private static bool TryParseJson(string text, out List<FileSection> sections, out string? error)
        {
            sections = new List<FileSection>();
            error = null;

            string trimmed = text.TrimStart();
            if (trimmed.Length == 0 || (trimmed[0] != '{' && trimmed[0] != '['))
                return false;

            JsonDocument document;
            try { document = JsonDocument.Parse(text); }
            catch (JsonException) { return false; }

            using (document)
            {
                JsonElement root = document.RootElement;
                IEnumerable<JsonElement> operations;
                if (root.ValueKind == JsonValueKind.Array)
                    operations = root.EnumerateArray();
                else if (root.TryGetProperty("operations", out JsonElement list) && list.ValueKind == JsonValueKind.Array)
                    operations = list.EnumerateArray();
                else if (root.TryGetProperty("operation", out JsonElement one))
                    operations = new[] { one };
                else if (root.TryGetProperty("type", out _))
                    operations = new[] { root };
                else
                    return false;

                foreach (JsonElement operation in operations)
                {
                    string type = operation.TryGetProperty("type", out JsonElement t) ? t.GetString() ?? string.Empty : string.Empty;
                    string path = operation.TryGetProperty("path", out JsonElement p) ? p.GetString() ?? string.Empty : string.Empty;
                    string diff = operation.TryGetProperty("diff", out JsonElement d) ? d.GetString() ?? string.Empty : string.Empty;
                    string? moveTo = operation.TryGetProperty("move_to", out JsonElement m) ? m.GetString() : null;

                    // `content` is what a model actually sends for a create in this shape,
                    // and reading only `diff` threw the whole file away: an empty body is
                    // Ok to the matcher, so a zero-byte file came back reported as
                    // "added a.py" with no counts to contradict it. The envelope parser
                    // refuses a body-less Add; this path has to as well.
                    bool wholeFile = false;
                    if (diff.Length == 0)
                    {
                        foreach (string alternative in new[] { "content", "text", "body", "contents" })
                        {
                            if (operation.TryGetProperty(alternative, out JsonElement c)
                                && c.ValueKind == JsonValueKind.String
                                && c.GetString() is { Length: > 0 } whole)
                            {
                                diff = whole;
                                wholeFile = true;
                                break;
                            }
                        }
                    }

                    if (!Validate(path, out error))
                        return false;
                    if (moveTo != null && !Validate(moveTo, out error))
                        return false;

                    FileOp op = type switch
                    {
                        "create_file" or "add_file" or "create" or "add" => FileOp.Add,
                        "update_file" or "update" => FileOp.Update,
                        "delete_file" or "delete" => FileOp.Delete,
                        _ => (FileOp)(-1),
                    };
                    if ((int)op < 0)
                    {
                        error = $"'{type}' is not a file operation. Use create_file, update_file or delete_file — "
                              + "or send the '*** Begin Patch' envelope instead, which is the documented form.";
                        return false;
                    }

                    // Whole-file content is not a diff: every line of it is an addition,
                    // and the matcher only ever sees prefixed lines.
                    List<string> body = V4ADiff.SplitDiffLines(diff);
                    if (wholeFile)
                        body = body.Select(line => "+" + line).ToList();

                    if (body.Count == 0 && op != FileOp.Delete && moveTo == null)
                    {
                        error = $"'{type}' for '{path}' carried no content. Send the file's lines under "
                              + "'diff' (or 'content'), or use the '*** Begin Patch' envelope, which is "
                              + "the documented form.";
                        return false;
                    }

                    sections.Add(new FileSection(op, path, moveTo, body)
                    {
                        Newline = diff.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n",
                    });
                }
            }

            if (sections.Count == 0)
            {
                error = "the patch names no files.";
                return false;
            }
            return true;
        }

        /// <summary>Render what a patch did, for the tool result.</summary>
        public static string Describe(IReadOnlyList<FileOutcome> outcomes)
        {
            var sb = new StringBuilder();
            sb.Append("Applied the patch to ")
              .Append(outcomes.Count.ToString(CultureInfo.InvariantCulture))
              .Append(outcomes.Count == 1 ? " file:\n" : " files:\n");
            int unchanged = 0;
            foreach (FileOutcome o in outcomes)
            {
                // A hunk of pure context changes nothing, and "updated deck.py (+0 -0)"
                // reads as an edit. That matters more here than almost anywhere, because
                // the patch tool's own declaration tells the model NOT to read the file
                // back afterwards — correctly, since a hunk that did not match would have
                // refused — so a no-op reported as an edit is a belief the model cannot
                // check. The reference implementation has exactly this defect: a
                // context-only hunk there writes the file back byte-identical and reports
                // "Updated <path>".
                bool noop = o.Op == FileOp.Update
                    && o.LinesAdded == 0 && o.LinesRemoved == 0 && o.MovedTo == null;
                if (noop)
                    unchanged++;

                sb.Append("  ").Append(o.Op switch
                {
                    FileOp.Add => "added   ",
                    FileOp.Delete => "deleted ",
                    _ => noop ? "UNCHANGED " : "updated ",
                }).Append(o.Path);
                if (o.MovedTo != null)
                    sb.Append(" -> ").Append(o.MovedTo);
                if (o.Op == FileOp.Update && !noop)
                {
                    sb.Append("  (+").Append(o.LinesAdded.ToString(CultureInfo.InvariantCulture))
                      .Append(" -").Append(o.LinesRemoved.ToString(CultureInfo.InvariantCulture))
                      .Append(')');
                }
                sb.Append('\n');
            }

            // The clause Claude Code puts on 1,396 of its 1,419 successful edits:
            // "(file state is current in your context — no need to Read it back)". It is
            // the other half of the contract the reference states to its own model —
            // "do not waste tokens by re-reading files after calling apply_patch on them;
            // the tool call will fail if it didn't work" — and it only pays for itself if
            // the success path says so where the model is actually reading. The patch
            // tool's DESCRIPTION already says it; a declaration read once is not the
            // channel a small model acts on.
            if (unchanged < outcomes.Count)
            {
                sb.Append("These files now hold exactly what this patch describes — no need to read them "
                        + "back. A hunk that had not matched would have refused and written nothing.\n");
            }

            if (unchanged > 0)
            {
                sb.Append(unchanged == outcomes.Count
                    ? "This patch changed NOTHING: every hunk was context with no '-' or '+' line in it, "
                      + "so the file is exactly as it was. Send the change you meant, with the lines to "
                      + "remove prefixed '-' and the lines to add prefixed '+'.\n"
                    : "The file(s) marked UNCHANGED were not modified — those hunks were context with no "
                      + "'-' or '+' line in them. Send the change you meant for them.\n");
            }

            // Say when a hunk only matched with whitespace ignored. It applied, so this is
            // not a failure — but the model's copy of the file disagrees with the real one
            // about indentation, and the next hunk it writes from that copy will miss.
            // The two ways a hunk can land imprecisely are different problems with
            // different fixes, and one message for both sent the model to check its
            // indentation when the real issue was that the file had moved on.
            foreach (FileOutcome outcome in outcomes)
            {
                if (outcome.Note is { Length: > 0 } note)
                    sb.Append("Note: ").Append(note);
            }

            if (outcomes.Any(o => o.Fuzz >= 10000))
            {
                sb.Append("Note: a hunk marked '*** End of File' did not match the end of the file and was "
                        + "applied wherever its context matched instead. Read the file and check the change "
                        + "landed where you meant it to.\n");
            }
            else if (outcomes.Any(o => o.Fuzz >= 100))
            {
                sb.Append("Note: at least one hunk matched only after ignoring leading whitespace, so the "
                        + "indentation you sent did not match the file. It applied where it belongs, but "
                        + "read the file before writing another patch against it.\n");
            }

            return sb.ToString();
        }
    }
}
