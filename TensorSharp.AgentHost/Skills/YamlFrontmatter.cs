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

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>What kind of YAML node a <see cref="YamlValue"/> holds.</summary>
    public enum YamlValueKind
    {
        /// <summary>A single string. Every scalar is kept as text: the Agent Skills
        /// frontmatter has no numeric or boolean field, and a version written as
        /// <c>"1.0"</c> must not silently become the number 1.</summary>
        Scalar,
        /// <summary>A block or flow mapping — <c>metadata:</c> is the one the spec defines.</summary>
        Mapping,
        /// <summary>A block or flow sequence.</summary>
        Sequence,
    }

    /// <summary>
    /// One node of a parsed YAML frontmatter document.
    /// </summary>
    public sealed class YamlValue
    {
        private static readonly IReadOnlyDictionary<string, YamlValue> EmptyMap =
            new Dictionary<string, YamlValue>(StringComparer.Ordinal);
        private static readonly IReadOnlyList<YamlValue> EmptyList = Array.Empty<YamlValue>();

        private YamlValue(YamlValueKind kind, string? scalar,
            Dictionary<string, YamlValue>? mapping, List<YamlValue>? sequence)
        {
            Kind = kind;
            _scalar = scalar;
            _mapping = mapping;
            _sequence = sequence;
        }

        private readonly string? _scalar;
        private readonly Dictionary<string, YamlValue>? _mapping;
        private readonly List<YamlValue>? _sequence;

        public YamlValueKind Kind { get; }

        internal static YamlValue FromScalar(string value) =>
            new(YamlValueKind.Scalar, value, null, null);

        internal static YamlValue FromMapping(Dictionary<string, YamlValue> map) =>
            new(YamlValueKind.Mapping, null, map, null);

        internal static YamlValue FromSequence(List<YamlValue> items) =>
            new(YamlValueKind.Sequence, null, null, items);

        /// <summary>The scalar text, or null when this node is a mapping or a sequence.</summary>
        public string? Scalar => Kind == YamlValueKind.Scalar ? _scalar : null;

        /// <summary>The mapping's entries, or an empty map when this node is not a mapping.</summary>
        public IReadOnlyDictionary<string, YamlValue> Mapping => _mapping ?? EmptyMap;

        /// <summary>The sequence's items, or an empty list when this node is not a sequence.</summary>
        public IReadOnlyList<YamlValue> Sequence => _sequence ?? EmptyList;

        /// <summary>
        /// Read a scalar member of a mapping. Returns null when the key is absent
        /// or holds a collection — a caller expecting text never has to type-test.
        /// </summary>
        public string? GetScalar(string key) =>
            _mapping != null && _mapping.TryGetValue(key, out YamlValue? v) ? v.Scalar : null;

        /// <summary>Read a mapping member of a mapping, or null when it is absent or not a mapping.</summary>
        public YamlValue? GetMapping(string key) =>
            _mapping != null && _mapping.TryGetValue(key, out YamlValue? v) && v.Kind == YamlValueKind.Mapping
                ? v
                : null;

        /// <summary>
        /// Every scalar member of a mapping, flattened to strings. Non-scalar members
        /// are skipped: the spec says <c>metadata</c> is a map of string to string,
        /// and a client that writes a nested structure there gets its scalars, not an
        /// exception.
        /// </summary>
        public IReadOnlyDictionary<string, string> ScalarMembers()
        {
            if (_mapping == null || _mapping.Count == 0)
                return new Dictionary<string, string>(StringComparer.Ordinal);

            var result = new Dictionary<string, string>(_mapping.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, YamlValue> entry in _mapping)
            {
                string? scalar = entry.Value.Scalar;
                if (scalar != null)
                    result[entry.Key] = scalar;
            }
            return result;
        }
    }

    /// <summary>
    /// Splits a Markdown document into its YAML frontmatter block and its body, and
    /// parses the frontmatter with a deliberately small YAML reader.
    ///
    /// <para>
    /// The solution ships no YAML library, and pulling one in for six frontmatter
    /// fields would put a third-party parser on the path of every skill a user
    /// uploads. The subset implemented here is exactly what the Agent Skills
    /// specification uses and what the skills in the wild actually contain:
    /// </para>
    /// <list type="bullet">
    /// <item>plain scalars, including ones with embedded colons
    ///   (<c>license: Proprietary. LICENSE.txt has complete terms</c>) and inline
    ///   URLs — a strict YAML parser rejects some of these, which is why the Codex
    ///   loader carries a "repair" pass; reading a plain scalar as the rest of the
    ///   line removes the need for one</item>
    /// <item>single- and double-quoted scalars with escapes, spanning lines</item>
    /// <item>literal and folded block scalars (<c>|</c>, <c>|-</c>, <c>|+</c>,
    ///   <c>&gt;</c>, <c>&gt;-</c>, <c>&gt;+</c>) with optional explicit indentation
    ///   indicators — <c>description: &gt;</c> and <c>description: |-</c> are both
    ///   common in the published Anthropic skills</item>
    /// <item>nested block mappings (<c>metadata:</c>) and block sequences</item>
    /// <item>single-line flow collections (<c>[a, b]</c>, <c>{k: v}</c>)</item>
    /// </list>
    /// <para>
    /// Anything outside that subset — anchors, aliases, tags, multi-document
    /// streams, complex keys — is not silently mis-read: the reader reports it
    /// through <see cref="YamlFrontmatterException"/> or keeps the text verbatim as
    /// a scalar, so a skill never loads with a field that means something other
    /// than what its author wrote.
    /// </para>
    /// </summary>
    public static class YamlFrontmatter
    {
        /// <summary>
        /// Split <paramref name="document"/> into frontmatter and body.
        /// </summary>
        /// <param name="document">The whole <c>SKILL.md</c> text.</param>
        /// <param name="frontmatter">The text between the opening and closing <c>---</c>, exclusive.</param>
        /// <param name="body">Everything after the closing <c>---</c> line.</param>
        /// <returns>
        /// False when the document does not open with <c>---</c> on its first line
        /// or never closes the block. A byte-order mark is tolerated, and so are
        /// CRLF line endings and trailing whitespace on the delimiter lines.
        /// </returns>
        public static bool TrySplit(string? document, out string frontmatter, out string body)
        {
            frontmatter = string.Empty;
            body = document ?? string.Empty;
            if (string.IsNullOrEmpty(document))
                return false;

            int cursor = 0;
            if (document[0] == '\uFEFF')
                cursor = 1;

            if (!TryReadLine(document, ref cursor, out string first))
                return false;
            if (!IsDelimiter(first))
                return false;

            var block = new StringBuilder();
            while (TryReadLine(document, ref cursor, out string line))
            {
                if (IsDelimiter(line))
                {
                    frontmatter = block.ToString();
                    body = document.Substring(cursor);
                    return true;
                }
                block.Append(line).Append('\n');
            }

            // Ran off the end without a closing delimiter: the document has no
            // frontmatter, however much it looked like it did.
            body = document;
            return false;
        }

        /// <summary>
        /// Parse a frontmatter block into a mapping.
        /// </summary>
        /// <exception cref="YamlFrontmatterException">
        /// The block is not a mapping, repeats a key, or uses a construct outside
        /// the supported subset. The message names the 1-based line.
        /// </exception>
        public static YamlValue Parse(string frontmatter)
        {
            string[] lines = SplitLines(frontmatter ?? string.Empty);
            var reader = new BlockReader(lines);
            Dictionary<string, YamlValue> root = reader.ReadMapping(minIndent: 0);
            reader.EnsureConsumed();
            return YamlValue.FromMapping(root);
        }

        /// <summary>
        /// Convenience over <see cref="TrySplit"/> + <see cref="Parse"/>: returns the
        /// parsed frontmatter, or null when the document carries none.
        /// </summary>
        public static YamlValue? ParseDocument(string? document, out string body)
        {
            if (!TrySplit(document, out string frontmatter, out body))
                return null;
            return Parse(frontmatter);
        }

        private static bool IsDelimiter(string line)
        {
            string trimmed = line.TrimEnd();
            return trimmed == "---" || trimmed == "...";
        }

        private static bool TryReadLine(string text, ref int cursor, out string line)
        {
            if (cursor >= text.Length)
            {
                line = string.Empty;
                return false;
            }

            int newline = text.IndexOf('\n', cursor);
            if (newline < 0)
            {
                line = text.Substring(cursor).TrimEnd('\r');
                cursor = text.Length;
                return true;
            }

            int end = newline;
            if (end > cursor && text[end - 1] == '\r')
                end--;
            line = text.Substring(cursor, end - cursor);
            cursor = newline + 1;
            return true;
        }

        private static string[] SplitLines(string text)
        {
            var lines = new List<string>();
            int cursor = 0;
            while (TryReadLine(text, ref cursor, out string line))
                lines.Add(line);
            return lines.ToArray();
        }

        /// <summary>
        /// Line-oriented recursive-descent reader over the frontmatter block.
        /// Indentation alone determines nesting, which is why every method takes the
        /// indent its block starts at rather than tracking a stack.
        /// </summary>
        private sealed class BlockReader
        {
            private readonly string[] _lines;
            private int _index;

            public BlockReader(string[] lines) => _lines = lines;

            private bool AtEnd => _index >= _lines.Length;

            private string Current => _lines[_index];

            private int CurrentLineNumber => _index + 1;

            public void EnsureConsumed()
            {
                SkipBlankAndComments();
                if (!AtEnd)
                    throw Fail(CurrentLineNumber, $"unexpected content '{Current.Trim()}'");
            }

            private void SkipBlankAndComments()
            {
                while (!AtEnd)
                {
                    string trimmed = Current.Trim();
                    if (trimmed.Length != 0 && !trimmed.StartsWith("#", StringComparison.Ordinal))
                        return;
                    _index++;
                }
            }

            private static int IndentOf(string line)
            {
                int i = 0;
                while (i < line.Length && line[i] == ' ')
                    i++;
                return i;
            }

            /// <summary>
            /// Read a block mapping whose keys sit at or beyond <paramref name="minIndent"/>.
            /// The first non-blank line fixes the block's indent; a line indented less
            /// than that ends the block and is left for the caller.
            /// </summary>
            public Dictionary<string, YamlValue> ReadMapping(int minIndent)
            {
                var map = new Dictionary<string, YamlValue>(StringComparer.Ordinal);
                int blockIndent = -1;

                while (true)
                {
                    SkipBlankAndComments();
                    if (AtEnd)
                        return map;

                    int indent = IndentOf(Current);
                    if (indent < minIndent)
                        return map;
                    if (blockIndent < 0)
                        blockIndent = indent;
                    else if (indent < blockIndent)
                        return map;
                    else if (indent > blockIndent)
                        throw Fail(CurrentLineNumber, "unexpected indentation");

                    string line = Current;
                    int lineNumber = CurrentLineNumber;
                    string rest = line.Substring(indent);

                    if (rest.StartsWith("- ", StringComparison.Ordinal) || rest == "-")
                        throw Fail(lineNumber, "a sequence item cannot appear where a mapping key is expected");

                    if (!TrySplitKey(rest, out string key, out string valueText))
                        throw Fail(lineNumber, $"expected 'key: value' but found '{rest.Trim()}'");

                    if (map.ContainsKey(key))
                        throw Fail(lineNumber, $"duplicate key '{key}'");

                    _index++;
                    map[key] = ReadValue(valueText, blockIndent, lineNumber);
                }
            }

            /// <summary>
            /// Split <c>key: value</c>. The key ends at the first colon that is
            /// followed by whitespace or end of line — a plain scalar may then
            /// contain as many further colons as it likes, which is what lets
            /// <c>license: Proprietary. LICENSE.txt has complete terms</c> and
            /// <c>description: Build for AWS: ECS</c> parse without a repair pass.
            /// A quoted key is unwrapped.
            /// </summary>
            private static bool TrySplitKey(string text, out string key, out string value)
            {
                key = string.Empty;
                value = string.Empty;

                int i = 0;
                if (i < text.Length && (text[i] == '"' || text[i] == '\''))
                {
                    char quote = text[i];
                    int end = FindClosingQuote(text, i, quote);
                    if (end < 0)
                        return false;
                    key = quote == '"'
                        ? UnescapeDoubleQuoted(text.Substring(i + 1, end - i - 1))
                        : text.Substring(i + 1, end - i - 1).Replace("''", "'", StringComparison.Ordinal);
                    i = end + 1;
                    if (i >= text.Length || text[i] != ':')
                        return false;
                    value = text.Substring(i + 1).TrimStart();
                    return true;
                }

                for (; i < text.Length; i++)
                {
                    if (text[i] != ':')
                        continue;
                    bool atEnd = i + 1 >= text.Length;
                    if (!atEnd && text[i + 1] != ' ' && text[i + 1] != '\t')
                        continue;

                    key = text.Substring(0, i).Trim();
                    value = atEnd ? string.Empty : text.Substring(i + 1).TrimStart();
                    return key.Length > 0;
                }
                return false;
            }

            private static int FindClosingQuote(string text, int start, char quote)
            {
                for (int i = start + 1; i < text.Length; i++)
                {
                    if (text[i] == '\\' && quote == '"')
                    {
                        i++;
                        continue;
                    }
                    if (text[i] != quote)
                        continue;
                    // '' inside a single-quoted scalar is an escaped quote.
                    if (quote == '\'' && i + 1 < text.Length && text[i + 1] == '\'')
                    {
                        i++;
                        continue;
                    }
                    return i;
                }
                return -1;
            }

            /// <summary>
            /// Read the value that followed a key, given the indent of the key's own
            /// line. An empty value means the value is the indented block underneath.
            /// </summary>
            private YamlValue ReadValue(string valueText, int keyIndent, int keyLineNumber)
            {
                string trimmed = StripTrailingComment(valueText);

                if (trimmed.Length == 0)
                    return ReadIndentedBlock(keyIndent, keyLineNumber);

                char first = trimmed[0];
                if (first == '|' || first == '>')
                    return YamlValue.FromScalar(ReadBlockScalar(trimmed, keyIndent, keyLineNumber));
                if (first == '[' || first == '{')
                    return ParseFlow(trimmed, keyLineNumber);
                if (first == '"' || first == '\'')
                    return YamlValue.FromScalar(ReadQuotedScalar(valueText, first, keyLineNumber));
                if (first == '*' || first == '&' || first == '!')
                    throw Fail(keyLineNumber, "anchors, aliases and tags are not supported in skill frontmatter");

                return YamlValue.FromScalar(trimmed);
            }

            /// <summary>
            /// A key with no inline value owns whatever is indented beneath it: a
            /// nested mapping, a sequence, or — when nothing is indented beneath —
            /// the empty string, which is how YAML spells an explicitly empty value.
            /// </summary>
            private YamlValue ReadIndentedBlock(int keyIndent, int keyLineNumber)
            {
                int probe = _index;
                while (probe < _lines.Length)
                {
                    string candidateTrimmed = _lines[probe].Trim();
                    if (candidateTrimmed.Length == 0 || candidateTrimmed.StartsWith("#", StringComparison.Ordinal))
                    {
                        probe++;
                        continue;
                    }
                    break;
                }

                if (probe >= _lines.Length)
                    return YamlValue.FromScalar(string.Empty);

                int childIndent = IndentOf(_lines[probe]);
                string childRest = _lines[probe].Substring(childIndent);

                // A sequence may sit at the SAME indent as its key, which is legal
                // YAML and is how most hand-written frontmatter spells a list.
                bool isSequence = childRest.StartsWith("- ", StringComparison.Ordinal) || childRest == "-";
                if (isSequence && childIndent >= keyIndent)
                {
                    _index = probe;
                    return YamlValue.FromSequence(ReadSequence(childIndent));
                }

                if (childIndent <= keyIndent)
                    return YamlValue.FromScalar(string.Empty);

                _index = probe;
                return YamlValue.FromMapping(ReadMapping(childIndent));
            }

            private List<YamlValue> ReadSequence(int blockIndent)
            {
                var items = new List<YamlValue>();
                while (true)
                {
                    SkipBlankAndComments();
                    if (AtEnd)
                        return items;

                    int indent = IndentOf(Current);
                    if (indent != blockIndent)
                        return items;

                    string rest = Current.Substring(indent);
                    if (!rest.StartsWith("- ", StringComparison.Ordinal) && rest != "-")
                        return items;

                    int lineNumber = CurrentLineNumber;
                    string itemText = rest == "-" ? string.Empty : rest.Substring(2).TrimStart();
                    _index++;

                    // "- key: value" opens a mapping whose members are indented to
                    // where the item's text began.
                    if (itemText.Length > 0
                        && itemText[0] != '"' && itemText[0] != '\''
                        && itemText[0] != '[' && itemText[0] != '{'
                        && itemText[0] != '|' && itemText[0] != '>'
                        && TrySplitKey(itemText, out string key, out string inlineValue))
                    {
                        int memberIndent = indent + 2;
                        var map = new Dictionary<string, YamlValue>(StringComparer.Ordinal)
                        {
                            [key] = ReadValue(inlineValue, memberIndent, lineNumber),
                        };
                        foreach (KeyValuePair<string, YamlValue> extra in ReadMapping(memberIndent))
                        {
                            if (map.ContainsKey(extra.Key))
                                throw Fail(lineNumber, $"duplicate key '{extra.Key}'");
                            map[extra.Key] = extra.Value;
                        }
                        items.Add(YamlValue.FromMapping(map));
                        continue;
                    }

                    items.Add(itemText.Length == 0
                        ? ReadIndentedBlock(indent, lineNumber)
                        : ReadValue(itemText, indent, lineNumber));
                }
            }

            /// <summary>
            /// Read a <c>|</c> / <c>&gt;</c> block scalar. The header may carry an
            /// explicit indentation indicator and a chomping indicator in either
            /// order (<c>|2-</c> and <c>|-2</c> are both legal).
            /// </summary>
            private string ReadBlockScalar(string header, int keyIndent, int keyLineNumber)
            {
                bool folded = header[0] == '>';
                int explicitIndent = 0;
                char chomping = '\0';

                for (int i = 1; i < header.Length; i++)
                {
                    char c = header[i];
                    if (c == '-' || c == '+')
                    {
                        if (chomping != '\0')
                            throw Fail(keyLineNumber, "block scalar has more than one chomping indicator");
                        chomping = c;
                    }
                    else if (c >= '1' && c <= '9')
                    {
                        if (explicitIndent != 0)
                            throw Fail(keyLineNumber, "block scalar has more than one indentation indicator");
                        explicitIndent = c - '0';
                    }
                    else if (c == ' ' || c == '\t')
                    {
                        break;
                    }
                    else
                    {
                        throw Fail(keyLineNumber, $"unsupported block scalar header '{header}'");
                    }
                }

                var raw = new List<string>();
                int contentIndent = explicitIndent > 0 ? keyIndent + explicitIndent : -1;

                while (!AtEnd)
                {
                    string line = Current;
                    if (line.Trim().Length == 0)
                    {
                        raw.Add(string.Empty);
                        _index++;
                        continue;
                    }

                    int indent = IndentOf(line);
                    if (contentIndent < 0)
                    {
                        // The first non-empty line fixes the block's indentation.
                        if (indent <= keyIndent)
                            break;
                        contentIndent = indent;
                    }
                    else if (indent < contentIndent)
                    {
                        break;
                    }

                    raw.Add(line.Length > contentIndent ? line.Substring(contentIndent) : string.Empty);
                    _index++;
                }

                // Trailing empty lines belong to chomping, not to the content.
                int lastContent = raw.Count - 1;
                while (lastContent >= 0 && raw[lastContent].Length == 0)
                    lastContent--;
                int trailingBlanks = raw.Count - 1 - lastContent;
                raw.RemoveRange(lastContent + 1, trailingBlanks);

                string text = folded ? FoldLines(raw) : string.Join("\n", raw);

                return chomping switch
                {
                    '-' => text,
                    '+' => text.Length == 0 && trailingBlanks == 0
                        ? text
                        : text + new string('\n', trailingBlanks + 1),
                    _ => text.Length == 0 ? text : text + "\n",
                };
            }

            /// <summary>
            /// Fold a <c>&gt;</c> block: adjacent plain lines join with a space, a
            /// blank line becomes a newline, and a MORE-indented line keeps its own
            /// line breaks (YAML calls these "more indented" lines and exempts them
            /// from folding, which is what preserves code samples inside a folded
            /// description).
            /// </summary>
            private static string FoldLines(List<string> lines)
            {
                var sb = new StringBuilder();
                bool previousWasText = false;
                bool previousWasMoreIndented = false;

                foreach (string line in lines)
                {
                    if (line.Length == 0)
                    {
                        sb.Append('\n');
                        previousWasText = false;
                        previousWasMoreIndented = false;
                        continue;
                    }

                    bool moreIndented = line[0] == ' ' || line[0] == '\t';
                    if (previousWasText)
                        sb.Append(moreIndented || previousWasMoreIndented ? '\n' : ' ');

                    sb.Append(line);
                    previousWasText = true;
                    previousWasMoreIndented = moreIndented;
                }
                return sb.ToString();
            }

            /// <summary>
            /// Read a quoted scalar, continuing onto following lines when the closing
            /// quote is not on the key's own line. YAML folds such a continuation to
            /// a single space, which is what a wrapped description means.
            /// </summary>
            private string ReadQuotedScalar(string firstLineValue, char quote, int lineNumber)
            {
                var buffer = new StringBuilder(firstLineValue);
                while (true)
                {
                    string text = buffer.ToString();
                    int end = FindClosingQuote(text, 0, quote);
                    if (end >= 0)
                    {
                        string inner = text.Substring(1, end - 1);
                        return quote == '"'
                            ? UnescapeDoubleQuoted(inner)
                            : inner.Replace("''", "'", StringComparison.Ordinal);
                    }

                    if (AtEnd)
                        throw Fail(lineNumber, "quoted value is never closed");

                    buffer.Append(' ').Append(Current.Trim());
                    _index++;
                }
            }

            private YamlValue ParseFlow(string text, int lineNumber)
            {
                int position = 0;
                YamlValue value = ParseFlowNode(text, ref position, lineNumber);
                SkipFlowWhitespace(text, ref position);
                if (position != text.Length)
                    throw Fail(lineNumber, $"trailing content after flow collection: '{text.Substring(position)}'");
                return value;
            }

            private YamlValue ParseFlowNode(string text, ref int position, int lineNumber)
            {
                SkipFlowWhitespace(text, ref position);
                if (position >= text.Length)
                    throw Fail(lineNumber, "unexpected end of flow collection");

                char c = text[position];
                if (c == '[')
                {
                    position++;
                    var items = new List<YamlValue>();
                    SkipFlowWhitespace(text, ref position);
                    if (position < text.Length && text[position] == ']')
                    {
                        position++;
                        return YamlValue.FromSequence(items);
                    }
                    while (true)
                    {
                        items.Add(ParseFlowNode(text, ref position, lineNumber));
                        SkipFlowWhitespace(text, ref position);
                        if (position >= text.Length)
                            throw Fail(lineNumber, "flow sequence is never closed");
                        if (text[position] == ',') { position++; continue; }
                        if (text[position] == ']') { position++; return YamlValue.FromSequence(items); }
                        throw Fail(lineNumber, $"unexpected '{text[position]}' in flow sequence");
                    }
                }

                if (c == '{')
                {
                    position++;
                    var map = new Dictionary<string, YamlValue>(StringComparer.Ordinal);
                    SkipFlowWhitespace(text, ref position);
                    if (position < text.Length && text[position] == '}')
                    {
                        position++;
                        return YamlValue.FromMapping(map);
                    }
                    while (true)
                    {
                        SkipFlowWhitespace(text, ref position);
                        string key = ReadFlowScalar(text, ref position, lineNumber, stopAtColon: true);
                        SkipFlowWhitespace(text, ref position);
                        if (position >= text.Length || text[position] != ':')
                            throw Fail(lineNumber, $"flow mapping key '{key}' has no value");
                        position++;
                        YamlValue value = ParseFlowNode(text, ref position, lineNumber);
                        if (map.ContainsKey(key))
                            throw Fail(lineNumber, $"duplicate key '{key}'");
                        map[key] = value;
                        SkipFlowWhitespace(text, ref position);
                        if (position >= text.Length)
                            throw Fail(lineNumber, "flow mapping is never closed");
                        if (text[position] == ',') { position++; continue; }
                        if (text[position] == '}') { position++; return YamlValue.FromMapping(map); }
                        throw Fail(lineNumber, $"unexpected '{text[position]}' in flow mapping");
                    }
                }

                return YamlValue.FromScalar(ReadFlowScalar(text, ref position, lineNumber, stopAtColon: false));
            }

            private static string ReadFlowScalar(string text, ref int position, int lineNumber, bool stopAtColon)
            {
                SkipFlowWhitespace(text, ref position);
                if (position >= text.Length)
                    throw Fail(lineNumber, "unexpected end of flow collection");

                char quote = text[position];
                if (quote == '"' || quote == '\'')
                {
                    int end = FindClosingQuote(text, position, quote);
                    if (end < 0)
                        throw Fail(lineNumber, "quoted value is never closed");
                    string inner = text.Substring(position + 1, end - position - 1);
                    position = end + 1;
                    return quote == '"'
                        ? UnescapeDoubleQuoted(inner)
                        : inner.Replace("''", "'", StringComparison.Ordinal);
                }

                int start = position;
                while (position < text.Length)
                {
                    char c = text[position];
                    if (c == ',' || c == ']' || c == '}')
                        break;
                    if (stopAtColon && c == ':')
                        break;
                    position++;
                }
                return text.Substring(start, position - start).Trim();
            }

            private static void SkipFlowWhitespace(string text, ref int position)
            {
                while (position < text.Length && (text[position] == ' ' || text[position] == '\t'))
                    position++;
            }

            /// <summary>
            /// Drop a trailing <c>#</c> comment from a plain scalar. A <c>#</c> only
            /// starts a comment when whitespace precedes it, so a value such as
            /// <c>color: #ff8800</c> or a URL fragment keeps its hash.
            /// </summary>
            private static string StripTrailingComment(string value)
            {
                for (int i = 0; i < value.Length; i++)
                {
                    if (value[i] != '#')
                        continue;
                    if (i == 0 || value[i - 1] == ' ' || value[i - 1] == '\t')
                        return value.Substring(0, i).TrimEnd();
                }
                return value.TrimEnd();
            }

            private static string UnescapeDoubleQuoted(string text)
            {
                if (text.IndexOf('\\') < 0)
                    return text;

                var sb = new StringBuilder(text.Length);
                for (int i = 0; i < text.Length; i++)
                {
                    char c = text[i];
                    if (c != '\\' || i + 1 >= text.Length)
                    {
                        sb.Append(c);
                        continue;
                    }

                    char escape = text[++i];
                    switch (escape)
                    {
                        case 'n': sb.Append('\n'); break;
                        case 't': sb.Append('\t'); break;
                        case 'r': sb.Append('\r'); break;
                        case 'b': sb.Append('\b'); break;
                        case 'f': sb.Append('\f'); break;
                        case 'v': sb.Append('\v'); break;
                        case '0': sb.Append('\0'); break;
                        case 'a': sb.Append('\a'); break;
                        case 'e': sb.Append('\u001b'); break;
                        case '"': sb.Append('"'); break;
                        case '\'': sb.Append('\''); break;
                        case '/': sb.Append('/'); break;
                        case '\\': sb.Append('\\'); break;
                        case ' ': sb.Append(' '); break;
                        case 'N': sb.Append('\u0085'); break;
                        case '_': sb.Append('\u00a0'); break;
                        case 'L': sb.Append('\u2028'); break;
                        case 'P': sb.Append('\u2029'); break;
                        case 'x': AppendCodePoint(sb, text, ref i, 2); break;
                        case 'u': AppendCodePoint(sb, text, ref i, 4); break;
                        case 'U': AppendCodePoint(sb, text, ref i, 8); break;
                        default:
                            // An unknown escape is far more likely to be a Windows
                            // path or a regex the author meant literally than a typo
                            // worth failing the whole skill over.
                            sb.Append('\\').Append(escape);
                            break;
                    }
                }
                return sb.ToString();
            }

            private static void AppendCodePoint(StringBuilder sb, string text, ref int index, int digits)
            {
                if (index + digits >= text.Length)
                {
                    sb.Append(text.Substring(index));
                    index = text.Length - 1;
                    return;
                }

                string hex = text.Substring(index + 1, digits);
                if (!int.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int code)
                    || code < 0 || code > 0x10FFFF
                    || (code >= 0xD800 && code <= 0xDFFF))
                {
                    sb.Append(text[index]).Append(hex);
                    index += digits;
                    return;
                }

                index += digits;
                if (code <= 0xFFFF)
                    sb.Append((char)code);
                else
                    sb.Append(char.ConvertFromUtf32(code));
            }

            private static YamlFrontmatterException Fail(int lineNumber, string message) =>
                new($"line {lineNumber}: {message}");
        }
    }

    /// <summary>Thrown when a skill's YAML frontmatter cannot be read.</summary>
    public sealed class YamlFrontmatterException : Exception
    {
        public YamlFrontmatterException(string message) : base(message) { }

        public YamlFrontmatterException(string message, Exception inner) : base(message, inner) { }
    }
}
