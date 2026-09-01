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
using System.Linq;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the <c>SKILL.md</c> reader against the shapes real skills actually use.
///
/// <para>
/// Every fixture below is transcribed from a published skill in
/// <see href="https://github.com/anthropics/skills"/>, because those are the files the
/// feature has to work on and each one exercises a different corner of YAML: a folded
/// block scalar (<c>academy-guide</c>), a literal block scalar with a strip indicator
/// (<c>claude-api</c>), a double-quoted scalar carrying escaped quotes and em dashes
/// (<c>xlsx</c>), and plain scalars containing sentence punctuation that a strict YAML
/// reader would choke on (<c>license: Proprietary. LICENSE.txt has complete terms</c>).
/// A parser that handles only <c>key: value</c> loads two of the nineteen published
/// skills and silently drops the rest, which is exactly the failure this class exists
/// to prevent.
/// </para>
/// </summary>
public class SkillManifestParserTests
{
    // ---- required fields ---------------------------------------------------

    [Fact]
    public void Parse_MinimalSkill_ReadsNameAndDescription()
    {
        const string doc = """
            ---
            name: skill-name
            description: A description of what this skill does and when to use it.
            ---

            # Body
            """;

        Assert.True(SkillManifestParser.TryParse(doc, "skill-name", out var manifest, out string error), error);
        Assert.Equal("skill-name", manifest!.Name);
        Assert.Equal("A description of what this skill does and when to use it.", manifest.Description);
        Assert.Equal("# Body", manifest.Body.Trim());
        Assert.Empty(manifest.Warnings);
    }

    [Fact]
    public void Parse_NoFrontmatter_IsRejected()
    {
        Assert.False(SkillManifestParser.TryParse("# Just markdown\n", "x", out _, out string error));
        Assert.Contains("frontmatter", error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Parse_UnterminatedFrontmatter_IsRejected()
    {
        // A document that opens with --- and never closes it has no frontmatter at
        // all; treating the whole file as YAML would produce nonsense fields.
        Assert.False(SkillManifestParser.TryParse("---\nname: a\ndescription: b\n", "a", out _, out _));
    }

    [Fact]
    public void Parse_MissingDescription_IsRejected()
    {
        // Description is the only field the model sees before a skill is loaded. A
        // skill without one can never be selected, so loading it would be worse than
        // reporting it.
        const string doc = "---\nname: thing\n---\nbody\n";
        Assert.False(SkillManifestParser.TryParse(doc, "thing", out _, out string error));
        Assert.Contains("description", error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Parse_MissingName_FallsBackToTheDirectoryName()
    {
        const string doc = "---\ndescription: does a thing\n---\nbody\n";
        Assert.True(SkillManifestParser.TryParse(doc, "pdf-tools", out var manifest, out _));
        Assert.Equal("pdf-tools", manifest!.Name);
        Assert.Contains(manifest.Warnings, w => w.Contains("directory name", StringComparison.OrdinalIgnoreCase));
    }

    // ---- scalar styles, transcribed from published skills ------------------

    [Fact]
    public void Parse_FoldedBlockScalar_JoinsLinesWithSpaces()
    {
        // anthropics/skills → skills/academy-guide/SKILL.md
        const string doc = """
            ---
            name: academy-guide
            description: >
              Stop and check this skill before finishing any reply to a question about how
              to use Claude or a Claude product — it recommends matching courses,
              tutorials, and use cases from Claude Academy.
            license: Complete terms in LICENSE.txt
            ---
            body
            """;

        Assert.True(SkillManifestParser.TryParse(doc, "academy-guide", out var manifest, out string error), error);
        Assert.Equal(
            "Stop and check this skill before finishing any reply to a question about how to use Claude or a "
            + "Claude product — it recommends matching courses, tutorials, and use cases from Claude Academy.",
            manifest!.Description);
        Assert.Equal("Complete terms in LICENSE.txt", manifest.License);
    }

    [Fact]
    public void Parse_LiteralBlockScalarWithStrip_KeepsItsLineBreaks()
    {
        // anthropics/skills → skills/claude-api/SKILL.md
        const string doc = """
            ---
            name: claude-api
            description: |-
              Reference for the Claude API — model ids, pricing, params.
              TRIGGER — read BEFORE opening the target file.
              SKIP only when another provider is being worked on.
            license: Complete terms in LICENSE.txt
            ---
            body
            """;

        Assert.True(SkillManifestParser.TryParse(doc, "claude-api", out var manifest, out string error), error);

        // RawDescription keeps the author's line structure; Description is collapsed
        // because the catalog renders one skill per line.
        Assert.Equal(3, manifest!.RawDescription.Split('\n').Length);
        Assert.DoesNotContain('\n', manifest.Description);
        Assert.StartsWith("Reference for the Claude API", manifest.Description, StringComparison.Ordinal);
        Assert.EndsWith("another provider is being worked on.", manifest.Description, StringComparison.Ordinal);
    }

    [Fact]
    public void Parse_DoubleQuotedScalar_UnescapesEmbeddedQuotes()
    {
        // anthropics/skills → skills/xlsx/SKILL.md
        const string doc = """"
            ---
            name: xlsx
            description: "Use this skill any time a spreadsheet file is the primary input or output. Trigger especially when the user references a spreadsheet file by name — even casually (like \"the xlsx in my downloads\")."
            license: Proprietary. LICENSE.txt has complete terms
            ---
            body
            """";

        Assert.True(SkillManifestParser.TryParse(doc, "xlsx", out var manifest, out string error), error);
        Assert.Contains("\"the xlsx in my downloads\"", manifest!.Description, StringComparison.Ordinal);

        // The licence line is a plain scalar with a full stop inside it. A strict YAML
        // reader accepts it, but the same shape with a colon (`Build for AWS: ECS`)
        // does not — see below.
        Assert.Equal("Proprietary. LICENSE.txt has complete terms", manifest.License);
    }

    [Fact]
    public void Parse_PlainScalarContainingAColon_IsNotMistakenForAMapping()
    {
        // This is the case the Codex loader carries a "repair" pass for. Reading a
        // plain scalar as the rest of the line makes the repair unnecessary.
        const string doc = """
            ---
            name: aws-deploy
            description: Build for AWS: ECS, Fargate and Lambda. Use when deploying.
            ---
            body
            """;

        Assert.True(SkillManifestParser.TryParse(doc, "aws-deploy", out var manifest, out string error), error);
        Assert.Equal("Build for AWS: ECS, Fargate and Lambda. Use when deploying.", manifest!.Description);
    }

    [Fact]
    public void Parse_MetadataMapping_IsReadAsStrings()
    {
        const string doc = """
            ---
            name: pdf-processing
            description: Extract PDF text, fill forms, merge files.
            license: Apache-2.0
            metadata:
              author: example-org
              version: "1.0"
            ---
            body
            """;

        Assert.True(SkillManifestParser.TryParse(doc, "pdf-processing", out var manifest, out string error), error);
        Assert.Equal("example-org", manifest!.Metadata["author"]);

        // A version written as "1.0" must stay the string "1.0"; a YAML reader that
        // types scalars would turn it into the number 1 and the skill would advertise
        // the wrong version.
        Assert.Equal("1.0", manifest.Metadata["version"]);
    }

    [Fact]
    public void Parse_AllowedTools_SplitsOnWhitespace()
    {
        const string doc = """
            ---
            name: git-helper
            description: Runs git chores.
            allowed-tools: Bash(git:*) Bash(jq:*) Read
            ---
            body
            """;

        Assert.True(SkillManifestParser.TryParse(doc, "git-helper", out var manifest, out _));
        Assert.Equal(new[] { "Bash(git:*)", "Bash(jq:*)", "Read" }, manifest!.AllowedTools);
    }

    [Fact]
    public void Parse_AllowedToolsAsASequence_IsAlsoAccepted()
    {
        // The specification defines a space-separated string, but a YAML list is what
        // most authors reach for and both mean the same thing.
        const string doc = """
            ---
            name: git-helper
            description: Runs git chores.
            allowed-tools:
              - Read
              - Bash(git:*)
            ---
            body
            """;

        Assert.True(SkillManifestParser.TryParse(doc, "git-helper", out var manifest, out _));
        Assert.Equal(new[] { "Read", "Bash(git:*)" }, manifest!.AllowedTools);
    }

    [Fact]
    public void Parse_UnknownFrontmatterKeys_AreKeptNotDropped()
    {
        const string doc = """
            ---
            name: thing
            description: does a thing
            argument-hint: <duration>
            ---
            body
            """;

        Assert.True(SkillManifestParser.TryParse(doc, "thing", out var manifest, out _));
        Assert.Equal("<duration>", manifest!.ExtraFields["argument-hint"]);
    }

    [Fact]
    public void Parse_CrlfAndByteOrderMark_AreTolerated()
    {
        // A skill authored on Windows, or saved by an editor that writes a BOM, is
        // still a skill.
        string doc = "﻿---\r\nname: thing\r\ndescription: does a thing\r\n---\r\n\r\n# Body\r\n";
        Assert.True(SkillManifestParser.TryParse(doc, "thing", out var manifest, out string error), error);
        Assert.Equal("does a thing", manifest!.Description);
        Assert.Contains("# Body", manifest.Body, StringComparison.Ordinal);
    }

    // ---- validation and warnings -------------------------------------------

    [Theory]
    [InlineData("pdf-processing", true)]
    [InlineData("data-analysis", true)]
    [InlineData("a", true)]
    [InlineData("skill1", true)]
    [InlineData("PDF-Processing", false)]   // uppercase
    [InlineData("-pdf", false)]             // leading hyphen
    [InlineData("pdf-", false)]             // trailing hyphen
    [InlineData("pdf--processing", false)]  // consecutive hyphens
    [InlineData("pdf_processing", false)]   // underscore
    [InlineData("pdf processing", false)]   // space
    [InlineData("", false)]
    public void IsValidName_MatchesTheSpecification(string name, bool expected)
        => Assert.Equal(expected, SkillManifestParser.IsValidName(name));

    [Fact]
    public void IsValidName_RejectsAnOverLongName()
        => Assert.False(SkillManifestParser.IsValidName(new string('a', SkillManifest.MaxNameLength + 1)));

    [Theory]
    [InlineData("PDF Processing", "pdf-processing")]
    [InlineData("pdf__processing", "pdf-processing")]
    [InlineData("  Spaced  Out  ", "spaced-out")]
    [InlineData("-leading-and-trailing-", "leading-and-trailing")]
    [InlineData("!!!", "")]
    public void NormalizeName_CoercesToALegalName(string input, string expected)
        => Assert.Equal(expected, SkillManifestParser.NormalizeName(input));

    [Fact]
    public void Parse_NameThatDisagreesWithItsDirectory_LoadsWithAWarning()
    {
        // The specification requires them to match, but a skill copied into a renamed
        // directory is still usable and refusing it helps nobody.
        const string doc = "---\nname: pdf\ndescription: does pdfs\n---\nbody\n";
        Assert.True(SkillManifestParser.TryParse(doc, "pdf-tools", out var manifest, out _));
        Assert.Equal("pdf", manifest!.Name);
        Assert.Contains(manifest.Warnings, w => w.Contains("does not match its directory", StringComparison.Ordinal));
    }

    [Fact]
    public void Parse_IllegalName_IsNormalizedWithAWarning()
    {
        const string doc = "---\nname: PDF Processing\ndescription: does pdfs\n---\nbody\n";
        Assert.True(SkillManifestParser.TryParse(doc, "pdf-processing", out var manifest, out _));
        Assert.Equal("pdf-processing", manifest!.Name);
        Assert.Contains(manifest.Warnings, w => w.Contains("normalized", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void Parse_OverLongDescription_LoadsWithAWarning()
    {
        string description = new('x', SkillManifest.MaxDescriptionLength + 10);
        string doc = $"---\nname: thing\ndescription: {description}\n---\nbody\n";

        Assert.True(SkillManifestParser.TryParse(doc, "thing", out var manifest, out _));
        Assert.Contains(manifest!.Warnings, w => w.Contains("over the", StringComparison.Ordinal));
    }

    [Fact]
    public void Parse_DuplicateKey_IsRejectedRatherThanSilentlyPickingOne()
    {
        // Two descriptions mean the author does not know which one the model sees.
        const string doc = "---\nname: thing\ndescription: first\ndescription: second\n---\nbody\n";
        Assert.False(SkillManifestParser.TryParse(doc, "thing", out _, out string error));
        Assert.Contains("duplicate", error, StringComparison.OrdinalIgnoreCase);
    }

    // ---- budgeting helpers -------------------------------------------------

    [Fact]
    public void Truncate_CutsAtAWordBoundaryAndMarksTheCut()
    {
        string result = SkillTextBudget.Truncate("the quick brown fox jumps over the lazy dog", 20);
        Assert.True(result.Length <= 20, $"'{result}' is {result.Length} characters");
        Assert.EndsWith("...", result, StringComparison.Ordinal);
    }

    [Fact]
    public void Truncate_LeavesShortTextAlone()
        => Assert.Equal("short", SkillTextBudget.Truncate("short", 20));

    [Fact]
    public void Truncate_NeverSplitsASurrogatePair()
    {
        // The result is written straight into a prompt, and a lone surrogate does not
        // round-trip through UTF-8.
        string text = string.Concat(Enumerable.Repeat("\U0001F600", 20));
        string result = SkillTextBudget.Truncate(text, 11);
        Assert.All(result.Where(char.IsSurrogate).Chunk(2), pair => Assert.Equal(2, pair.Length));
        Assert.Equal(-1, result.IndexOf('�'));
    }

    [Fact]
    public void CollapseWhitespace_FlattensEveryRunToOneSpace()
        => Assert.Equal("a b c", SkillManifestParser.CollapseWhitespace("  a\n\n  b\t\tc  "));
}
