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
    /// <summary>
    /// One skill's <c>SKILL.md</c>, parsed: the frontmatter fields defined by the
    /// Agent Skills specification plus the Markdown body that follows them.
    ///
    /// <para>
    /// Field semantics follow <see href="https://agentskills.io/specification"/>.
    /// The two required fields are <see cref="Name"/> and <see cref="Description"/>;
    /// every other frontmatter key is optional, and keys the specification does not
    /// define are kept in <see cref="ExtraFields"/> rather than dropped, because a
    /// client that writes its own key is entitled to read it back.
    /// </para>
    /// </summary>
    public sealed class SkillManifest
    {
        /// <summary>Longest legal <c>name</c>, per the specification.</summary>
        public const int MaxNameLength = 64;

        /// <summary>Longest legal <c>description</c>, per the specification.</summary>
        public const int MaxDescriptionLength = 1024;

        /// <summary>Longest legal <c>compatibility</c>, per the specification.</summary>
        public const int MaxCompatibilityLength = 500;

        /// <summary>
        /// The skill's identifier: 1-64 characters of lowercase alphanumerics and
        /// single hyphens. This is what a user types to select the skill and what the
        /// model sees in the catalog, so it is always a valid name even when the
        /// frontmatter's was not — see <see cref="SkillManifestParser"/>.
        /// </summary>
        public required string Name { get; init; }

        /// <summary>
        /// What the skill does and when to use it. Collapsed to a single line: the
        /// catalog renders one skill per line, and a description written as a folded
        /// or literal block scalar would otherwise break the list apart.
        /// </summary>
        public required string Description { get; init; }

        /// <summary>
        /// The description exactly as written, with its line structure intact. Shown
        /// in management UIs, where the author's paragraphs are worth keeping.
        /// </summary>
        public required string RawDescription { get; init; }

        /// <summary>The <c>license</c> field, or null.</summary>
        public string? License { get; init; }

        /// <summary>The <c>compatibility</c> field — environment requirements — or null.</summary>
        public string? Compatibility { get; init; }

        /// <summary>
        /// The <c>metadata</c> map. Free-form string-to-string; the specification
        /// reserves nothing inside it, so <c>version</c> and <c>author</c> are
        /// conventions rather than fields.
        /// </summary>
        public IReadOnlyDictionary<string, string> Metadata { get; init; } =
            new Dictionary<string, string>(StringComparer.Ordinal);

        /// <summary>
        /// The <c>allowed-tools</c> field, split on whitespace. Marked experimental by
        /// the specification and advisory here: TensorSharp records it and reports it,
        /// and enforces it only for the skill-owned tools it actually controls.
        /// </summary>
        public IReadOnlyList<string> AllowedTools { get; init; } = Array.Empty<string>();

        /// <summary>Frontmatter keys outside the specification, kept verbatim.</summary>
        public IReadOnlyDictionary<string, string> ExtraFields { get; init; } =
            new Dictionary<string, string>(StringComparer.Ordinal);

        /// <summary>
        /// The Markdown body after the closing <c>---</c>. This is the "instructions"
        /// tier of progressive disclosure: loaded only once a skill is activated.
        /// </summary>
        public required string Body { get; init; }

        /// <summary>
        /// Problems that did not stop the skill from loading — a name that disagrees
        /// with its directory, an over-long description, an unknown frontmatter key.
        /// Surfaced by <c>--list-skills</c> and the management API so an author can
        /// fix them, never fatal, because a skill that is 3 characters over the
        /// description limit is still a skill the user wants to use.
        /// </summary>
        public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();

        /// <summary>
        /// A rough token count for <see cref="Body"/>, used to decide what fits in a
        /// context budget without tokenizing. Four bytes per token is the same
        /// approximation the rest of the ecosystem uses for budgeting; it is never
        /// used for anything the model actually sees.
        /// </summary>
        public int ApproximateBodyTokens => SkillTextBudget.ApproximateTokens(Body);
    }

    /// <summary>
    /// Reads a <c>SKILL.md</c> into a <see cref="SkillManifest"/>.
    ///
    /// <para>
    /// The reader is deliberately forgiving in one direction and strict in the
    /// other. It is <b>strict</b> about the two things that make a skill usable at
    /// all — a parseable frontmatter block and a non-empty description — because a
    /// skill missing either is invisible to the model no matter what else it
    /// contains, and failing loudly at load beats failing silently at inference.
    /// It is <b>forgiving</b> about everything else: a name that disagrees with its
    /// directory, an over-long description, an unrecognised key, a
    /// <c>metadata</c> value that is not a string. Real skills in the wild carry all
    /// of these, and rejecting a skill over one is a worse outcome for the user than
    /// loading it with a warning attached.
    /// </para>
    /// </summary>
    public static class SkillManifestParser
    {
        /// <summary>The file every skill directory must contain.</summary>
        public const string SkillFileName = "SKILL.md";

        private static readonly HashSet<string> KnownFields = new(StringComparer.Ordinal)
        {
            "name", "description", "license", "compatibility", "metadata", "allowed-tools",
        };

        /// <summary>
        /// Parse <paramref name="document"/>.
        /// </summary>
        /// <param name="document">The whole <c>SKILL.md</c> text.</param>
        /// <param name="directoryName">
        /// The name of the directory holding the file. The specification requires
        /// <c>name</c> to match it, so it is both the fallback when the frontmatter
        /// omits a usable name and the thing a mismatch is reported against. Pass
        /// null when parsing a document that has no directory (a validation tool
        /// reading from stdin).
        /// </param>
        /// <param name="manifest">The parsed manifest, or null on failure.</param>
        /// <param name="error">Why parsing failed, or null on success.</param>
        public static bool TryParse(
            string? document,
            string? directoryName,
            out SkillManifest? manifest,
            out string? error)
        {
            manifest = null;
            error = null;

            if (string.IsNullOrWhiteSpace(document))
            {
                error = $"{SkillFileName} is empty";
                return false;
            }

            if (!YamlFrontmatter.TrySplit(document, out string frontmatterText, out string body))
            {
                error = $"{SkillFileName} has no YAML frontmatter block delimited by '---'";
                return false;
            }

            YamlValue frontmatter;
            try
            {
                frontmatter = YamlFrontmatter.Parse(frontmatterText);
            }
            catch (YamlFrontmatterException ex)
            {
                error = $"{SkillFileName} frontmatter is not valid YAML ({ex.Message})";
                return false;
            }

            var warnings = new List<string>();

            string? rawDescription = frontmatter.GetScalar("description");
            string description = CollapseWhitespace(rawDescription);
            if (description.Length == 0)
            {
                error = frontmatter.Mapping.ContainsKey("description")
                    ? $"{SkillFileName} has an empty 'description'"
                    : $"{SkillFileName} is missing the required 'description' field";
                return false;
            }
            if (description.Length > SkillManifest.MaxDescriptionLength)
            {
                warnings.Add(
                    $"description is {description.Length} characters, over the {SkillManifest.MaxDescriptionLength}-character limit");
            }

            string? declaredName = frontmatter.GetScalar("name");
            string normalizedDirectory = NormalizeName(directoryName);
            string name;

            if (TryNormalizeDeclaredName(declaredName, out string normalizedDeclared, out string? nameProblem))
            {
                name = normalizedDeclared;
                if (!string.Equals(normalizedDeclared, declaredName?.Trim(), StringComparison.Ordinal))
                {
                    warnings.Add(
                        $"name '{declaredName?.Trim()}' is not a legal skill name and was normalized to '{normalizedDeclared}'");
                }
                if (normalizedDirectory.Length > 0
                    && !string.Equals(name, normalizedDirectory, StringComparison.Ordinal))
                {
                    warnings.Add(
                        $"name '{name}' does not match its directory '{directoryName}'; the specification requires them to be the same");
                }
            }
            else if (normalizedDirectory.Length > 0)
            {
                name = normalizedDirectory;
                warnings.Add(
                    declaredName == null
                        ? $"{SkillFileName} has no 'name'; using the directory name '{name}'"
                        : $"name '{declaredName.Trim()}' is unusable ({nameProblem}); using the directory name '{name}'");
            }
            else
            {
                error = declaredName == null
                    ? $"{SkillFileName} is missing the required 'name' field"
                    : $"{SkillFileName} has an unusable 'name' ({nameProblem})";
                return false;
            }

            string? compatibility = Trimmed(frontmatter.GetScalar("compatibility"));
            if (compatibility is { Length: > SkillManifest.MaxCompatibilityLength })
            {
                warnings.Add(
                    $"compatibility is {compatibility.Length} characters, over the {SkillManifest.MaxCompatibilityLength}-character limit");
            }

            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
            if (frontmatter.Mapping.TryGetValue("metadata", out YamlValue? metadataNode))
            {
                if (metadataNode.Kind == YamlValueKind.Mapping)
                {
                    foreach (KeyValuePair<string, string> entry in metadataNode.ScalarMembers())
                        metadata[entry.Key] = entry.Value;
                    if (metadata.Count != metadataNode.Mapping.Count)
                        warnings.Add("metadata has non-string values, which the specification does not allow; they were dropped");
                }
                else if (metadataNode.Kind != YamlValueKind.Scalar || metadataNode.Scalar?.Length > 0)
                {
                    warnings.Add("metadata is not a mapping of string keys to string values and was ignored");
                }
            }

            var extras = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, YamlValue> entry in frontmatter.Mapping)
            {
                if (KnownFields.Contains(entry.Key))
                    continue;
                extras[entry.Key] = entry.Value.Scalar ?? DescribeNonScalar(entry.Value);
            }

            manifest = new SkillManifest
            {
                Name = name,
                Description = description,
                RawDescription = (rawDescription ?? string.Empty).Trim(),
                License = Trimmed(frontmatter.GetScalar("license")),
                Compatibility = compatibility,
                Metadata = metadata,
                AllowedTools = ParseAllowedTools(frontmatter),
                ExtraFields = extras,
                Body = body.TrimStart('\n', '\r'),
                Warnings = warnings,
            };
            return true;
        }

        /// <summary>
        /// True when <paramref name="name"/> is exactly what the specification allows:
        /// 1-64 characters, lowercase <c>a-z</c> / <c>0-9</c> / <c>-</c>, no leading or
        /// trailing hyphen, no consecutive hyphens.
        /// </summary>
        public static bool IsValidName(string? name)
        {
            if (string.IsNullOrEmpty(name) || name.Length > SkillManifest.MaxNameLength)
                return false;
            if (name[0] == '-' || name[^1] == '-')
                return false;

            for (int i = 0; i < name.Length; i++)
            {
                char c = name[i];
                bool legal = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-';
                if (!legal)
                    return false;
                if (c == '-' && i + 1 < name.Length && name[i + 1] == '-')
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Coerce arbitrary text into a legal skill name: lowercase, non-alphanumerics
        /// collapsed to single hyphens, trimmed to 64 characters. Returns the empty
        /// string when nothing usable survives (a name of only punctuation, or of
        /// characters from a script with no ASCII fold).
        /// </summary>
        public static string NormalizeName(string? name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return string.Empty;

            var sb = new StringBuilder(name.Length);
            bool pendingHyphen = false;
            foreach (char raw in name.Trim())
            {
                char c = char.ToLowerInvariant(raw);
                bool legal = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
                if (legal)
                {
                    if (pendingHyphen && sb.Length > 0)
                        sb.Append('-');
                    pendingHyphen = false;
                    sb.Append(c);
                    if (sb.Length == SkillManifest.MaxNameLength)
                        break;
                }
                else
                {
                    pendingHyphen = true;
                }
            }
            return sb.ToString();
        }

        private static bool TryNormalizeDeclaredName(
            string? declared,
            out string normalized,
            out string? problem)
        {
            normalized = string.Empty;
            problem = null;

            string trimmed = (declared ?? string.Empty).Trim();
            if (trimmed.Length == 0)
            {
                problem = "it is empty";
                return false;
            }
            if (IsValidName(trimmed))
            {
                normalized = trimmed;
                return true;
            }

            string coerced = NormalizeName(trimmed);
            if (coerced.Length == 0)
            {
                problem = "it contains no letters or digits";
                return false;
            }

            normalized = coerced;
            return true;
        }

        private static IReadOnlyList<string> ParseAllowedTools(YamlValue frontmatter)
        {
            if (!frontmatter.Mapping.TryGetValue("allowed-tools", out YamlValue? node))
                return Array.Empty<string>();

            var tools = new List<string>();
            if (node.Kind == YamlValueKind.Sequence)
            {
                // Not what the specification says (it defines a space-separated
                // string) but a list is the shape most authors reach for, and both
                // mean the same thing.
                foreach (YamlValue item in node.Sequence)
                {
                    string? scalar = item.Scalar?.Trim();
                    if (!string.IsNullOrEmpty(scalar))
                        tools.Add(scalar);
                }
                return tools;
            }

            string text = node.Scalar ?? string.Empty;
            foreach (string part in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
                tools.Add(part);
            return tools;
        }

        private static string DescribeNonScalar(YamlValue value) => value.Kind switch
        {
            YamlValueKind.Sequence => string.Join(", ", CollectScalars(value)),
            YamlValueKind.Mapping => string.Join(", ", DescribeMapping(value)),
            _ => string.Empty,
        };

        private static IEnumerable<string> CollectScalars(YamlValue sequence)
        {
            foreach (YamlValue item in sequence.Sequence)
            {
                if (item.Scalar is { } scalar)
                    yield return scalar;
            }
        }

        private static IEnumerable<string> DescribeMapping(YamlValue mapping)
        {
            foreach (KeyValuePair<string, string> entry in mapping.ScalarMembers())
                yield return $"{entry.Key}={entry.Value}";
        }

        private static string? Trimmed(string? value)
        {
            string? trimmed = value?.Trim();
            return string.IsNullOrEmpty(trimmed) ? null : trimmed;
        }

        /// <summary>
        /// Collapse every run of whitespace — including the newlines a block scalar
        /// carries — to one space. The catalog is a one-skill-per-line list, so a
        /// multi-line description would otherwise fragment it and the model would read
        /// the continuation lines as separate entries.
        /// </summary>
        public static string CollapseWhitespace(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return string.Empty;

            var sb = new StringBuilder(value.Length);
            bool pendingSpace = false;
            foreach (char c in value)
            {
                if (char.IsWhiteSpace(c))
                {
                    pendingSpace = sb.Length > 0;
                    continue;
                }
                if (pendingSpace)
                {
                    sb.Append(' ');
                    pendingSpace = false;
                }
                sb.Append(c);
            }
            return sb.ToString();
        }
    }

    /// <summary>
    /// Token accounting for text that is about to be put in front of a model, without
    /// running a tokenizer over it.
    ///
    /// <para>
    /// The catalog has to fit inside a budget, and the budget is decided before a
    /// tokenizer is necessarily available — the server builds a catalog for the
    /// management API with no model loaded, and the CLI builds one while the model is
    /// still mapping. Tokenizing every skill on every request would also cost more
    /// than the budgeting saves.
    /// </para>
    /// </summary>
    public static class SkillTextBudget
    {
        /// <summary>
        /// Bytes per token. Four is the conventional English-text approximation and is
        /// used here only to decide what to include, never to report a count to the
        /// user as if it were exact.
        /// </summary>
        public const int BytesPerToken = 4;

        /// <summary>Approximate token count of <paramref name="text"/>, rounded up.</summary>
        public static int ApproximateTokens(string? text)
        {
            if (string.IsNullOrEmpty(text))
                return 0;
            long bytes = Encoding.UTF8.GetByteCount(text);
            return (int)Math.Min(int.MaxValue, (bytes + BytesPerToken - 1) / BytesPerToken);
        }

        /// <summary>
        /// Shorten <paramref name="text"/> to at most <paramref name="maxChars"/>
        /// characters, cutting at a word boundary when one is close by and appending an
        /// ellipsis. Returns the input unchanged when it already fits.
        /// </summary>
        public static string Truncate(string text, int maxChars)
        {
            if (maxChars <= 0)
                return string.Empty;
            if (string.IsNullOrEmpty(text) || text.Length <= maxChars)
                return text;

            const string ellipsis = "...";
            int keep = Math.Max(1, maxChars - ellipsis.Length);

            // Do not split a surrogate pair; the result is written straight into the
            // prompt and a lone surrogate does not round-trip through UTF-8.
            if (keep < text.Length && char.IsLowSurrogate(text[keep]))
                keep--;

            int lastSpace = text.LastIndexOf(' ', Math.Min(keep, text.Length - 1));
            if (lastSpace > keep - 24 && lastSpace > 0)
                keep = lastSpace;

            return string.Concat(text.AsSpan(0, keep).TrimEnd(), ellipsis);
        }

        /// <summary>
        /// Render a byte count the way the management surfaces report skill sizes.
        /// </summary>
        public static string FormatBytes(long bytes)
        {
            string[] units = { "B", "KB", "MB", "GB" };
            double value = bytes;
            int unit = 0;
            while (value >= 1024 && unit < units.Length - 1)
            {
                value /= 1024;
                unit++;
            }
            return unit == 0
                ? string.Create(CultureInfo.InvariantCulture, $"{bytes} B")
                : string.Create(CultureInfo.InvariantCulture, $"{value:0.#} {units[unit]}");
        }
    }
}
