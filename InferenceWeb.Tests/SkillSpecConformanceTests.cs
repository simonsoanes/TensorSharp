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
using System.Linq;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins TensorSharp against the Agent Skills specification and against the behaviour
/// of the two reference implementations.
///
/// <para>
/// The numbers asserted here are not TensorSharp's own preferences — they are the
/// specification's limits (<see href="https://agentskills.io/specification"/>) and, where
/// the spec is silent, the values OpenAI's Codex uses in
/// <c>codex-rs/skills/src/interface.rs</c> and <c>codex-rs/ext/skills/src/render.rs</c>.
/// A skill written for Claude Code or Codex has to load here unchanged and mean the
/// same thing, so any drift in these constants is a compatibility break and should
/// fail a test rather than be discovered by a user whose skill stopped working.
/// </para>
/// <para>
/// Where TensorSharp deliberately DIFFERS, the difference is asserted too, with the
/// reason — an undocumented divergence is the thing worth catching, not a documented one.
/// </para>
/// </summary>
public class SkillSpecConformanceTests
{
    // ---- the specification's own limits -------------------------------------

    [Fact]
    public void NameAndDescriptionLimits_MatchTheSpecification()
    {
        // agentskills.io/specification: name max 64 characters, description max 1024.
        // Codex uses the identical pair (codex-rs/skills/src/interface.rs:10-11).
        Assert.Equal(64, SkillManifest.MaxNameLength);
        Assert.Equal(1024, SkillManifest.MaxDescriptionLength);
    }

    [Fact]
    public void CompatibilityLimit_MatchesTheSpecification()
    {
        // agentskills.io/specification: compatibility max 500 characters.
        Assert.Equal(500, SkillManifest.MaxCompatibilityLength);
    }

    [Theory]
    // Every valid/invalid example the specification itself prints.
    [InlineData("pdf-processing", true)]
    [InlineData("data-analysis", true)]
    [InlineData("code-review", true)]
    [InlineData("PDF-Processing", false)]    // uppercase not allowed
    [InlineData("-pdf", false)]              // cannot start with a hyphen
    [InlineData("pdf--processing", false)]   // consecutive hyphens not allowed
    public void NameRules_MatchTheSpecificationsWorkedExamples(string name, bool valid)
        => Assert.Equal(valid, SkillManifestParser.IsValidName(name));

    [Fact]
    public void TheRequiredFileIsSKILLmd()
    {
        // The spec names exactly one required file, and the name is case-sensitive in
        // every published skill.
        Assert.Equal("SKILL.md", SkillManifestParser.SkillFileName);
    }

    [Fact]
    public void EveryFrontmatterFieldTheSpecificationDefines_IsRead()
    {
        // name, description, license, compatibility, metadata, allowed-tools — and an
        // unrecognised key must be KEPT, because the spec says clients may store their
        // own properties and a reader that dropped them would lose a client's data.
        const string doc = """
            ---
            name: conformance
            description: Exercises every field the specification defines.
            license: Apache-2.0
            compatibility: Requires Python 3.14+ and uv
            metadata:
              author: example-org
              version: "1.0"
            allowed-tools: Bash(git:*) Read
            x-vendor-field: kept
            ---

            # Body
            """;

        Assert.True(SkillManifestParser.TryParse(doc, "conformance", out var m, out string error), error);
        Assert.Equal("conformance", m!.Name);
        Assert.Equal("Apache-2.0", m.License);
        Assert.Equal("Requires Python 3.14+ and uv", m.Compatibility);
        Assert.Equal("example-org", m.Metadata["author"]);
        Assert.Equal("1.0", m.Metadata["version"]);
        Assert.Equal(new[] { "Bash(git:*)", "Read" }, m.AllowedTools);
        Assert.Equal("kept", m.ExtraFields["x-vendor-field"]);
    }

    [Fact]
    public void OnlyNameAndDescriptionAreRequired()
    {
        // "Every field other than name and description is optional."
        const string doc = "---\nname: minimal\ndescription: The smallest legal skill.\n---\n";

        Assert.True(SkillManifestParser.TryParse(doc, "minimal", out var m, out _));
        Assert.Null(m!.License);
        Assert.Null(m.Compatibility);
        Assert.Empty(m.Metadata);
        Assert.Empty(m.AllowedTools);
    }

    // ---- the three progressive-disclosure tiers -----------------------------

    [Fact]
    public void TierOne_MetadataOnly_IsWhatAnUnselectedSkillCosts()
    {
        // "Metadata (~100 tokens): the name and description fields are loaded at
        // startup for all skills." A catalogue entry must carry the description and
        // NOT the body.
        using var fixture = new Corpus();
        Skill big = fixture.Write("catalogued", body: new string('x', 20_000));

        SkillPlan plan = SkillPrompt.Plan(Array.Empty<Skill>(), new[] { big }, SkillPromptOptions.Default);

        Assert.Contains("catalogued", plan.Instructions, StringComparison.Ordinal);
        Assert.DoesNotContain(new string('x', 200), plan.Instructions, StringComparison.Ordinal);
    }

    [Fact]
    public void TierTwo_TheBody_ArrivesOnActivation_AndNotBefore()
    {
        // "Instructions: the full SKILL.md body is loaded when the skill is activated."
        //
        // ACTIVATED, not selected. This test used to hand Plan a selected skill, find
        // the body in the system prompt, and call that activation — encoding the very
        // bug it was meant to guard. The user choosing a skill is a hint about what to
        // activate; the decision, and therefore the load, is the model's.
        using var fixture = new Corpus();
        Skill skill = fixture.Write("activated", body: "THE-BODY-MARKER");

        SkillPlan plan = SkillPrompt.Plan(new[] { skill }, Array.Empty<Skill>(), SkillPromptOptions.Default);

        Assert.DoesNotContain("THE-BODY-MARKER", plan.Instructions, StringComparison.Ordinal);
        Assert.Equal(new[] { "activated" }, plan.Deferred.Select(s => s.Id));
        Assert.Contains("skills_read(skill=\"activated\", path=\"SKILL.md\")",
            plan.Instructions, StringComparison.Ordinal);

        // ...and the body is one tool call away, which is what activation costs.
        SkillToolResult read = SkillTools.Execute(
            new ToolCall
            {
                Name = SkillTools.ReadToolName,
                Arguments = new Dictionary<string, object> { ["skill"] = "activated", ["path"] = "SKILL.md" },
            },
            new SkillToolContext(new[] { skill }));

        Assert.True(read.Ok, read.Content);
        Assert.Contains("THE-BODY-MARKER", read.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void TierOne_IsWhatASelectedSkillCostsToo_NotItsWholeBody()
    {
        // The measurement that prompted this: four selected skills inlined 43,777 bytes
        // (~10,944 tokens) of bodies, 52% of it for skills the model never referenced,
        // where the four descriptions come to 1,932 bytes (~483 tokens). The spec
        // budgets ~100 tokens per skill for tier one; bodies are not tier one.
        using var fixture = new Corpus();
        Skill big = fixture.Write("selected", body: new string('x', 20_000));

        SkillPlan plan = SkillPrompt.Plan(new[] { big }, Array.Empty<Skill>(), SkillPromptOptions.Default);

        Assert.DoesNotContain(new string('x', 200), plan.Instructions, StringComparison.Ordinal);
        Assert.True(plan.ApproximateTokens < 1_000,
            $"a one-skill metadata block cost {plan.ApproximateTokens} tokens");
    }

    [Fact]
    public void TierThree_BundledFiles_AreNeverInlined_AndTheirIndexArrivesWithTheSkill()
    {
        // "Resources: files in scripts/, references/ or assets/ are loaded only when
        // required." Their contents must never be in the prompt. Their PATHS are routing
        // information for a skill the model has ALREADY decided to use, so they ride
        // with the activation rather than being paid for on every request.
        using var fixture = new Corpus();
        Skill skill = fixture.Write("resourced", reference: "REFERENCE-FILE-MARKER");

        SkillPlan plan = SkillPrompt.Plan(new[] { skill }, Array.Empty<Skill>(), SkillPromptOptions.Default);
        Assert.DoesNotContain("REFERENCE-FILE-MARKER", plan.Instructions, StringComparison.Ordinal);
        Assert.DoesNotContain("references/detail.md", plan.Instructions, StringComparison.Ordinal);

        SkillToolResult read = SkillTools.Execute(
            new ToolCall
            {
                Name = SkillTools.ReadToolName,
                Arguments = new Dictionary<string, object> { ["skill"] = "resourced", ["path"] = "SKILL.md" },
            },
            new SkillToolContext(new[] { skill }));

        Assert.Contains("references/detail.md", read.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("REFERENCE-FILE-MARKER", read.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void FourRealisticSkills_CostTheMetadataTier_NotAnOrderOfMagnitudeMore()
    {
        // The regression this whole change exists to prevent, pinned as a ratio so it
        // survives wording edits. Measured on the real doc-coauthoring / internal-comms
        // / pdf / pptx set: 12,848 tokens inlined against 1,050 announced. The bodies
        // here are deliberately the same order of size.
        using var fixture = new Corpus();
        var skills = new List<Skill>
        {
            fixture.Write("alpha", body: new string('a', 16_000)),
            fixture.Write("bravo", body: new string('b', 1_500)),
            fixture.Write("charlie", body: new string('c', 8_000)),
            fixture.Write("delta", body: new string('d', 20_000)),
        };
        var options = new SkillPromptOptions { ContextTokens = 262_144, ToolsAvailable = true };

        SkillPlan announced = SkillPrompt.Plan(skills, Array.Empty<Skill>(), options);
        SkillPlan inlined = SkillPrompt.Plan(
            skills, Array.Empty<Skill>(),
            new SkillPromptOptions
            {
                ContextTokens = 262_144, ToolsAvailable = true, InlineSelectedBodies = true,
            });

        Assert.Equal(4, announced.Deferred.Count);
        Assert.Empty(announced.Inlined);
        Assert.Equal(4, inlined.Inlined.Count);

        // An order of magnitude, not a few percent. If this ever narrows, bodies have
        // crept back into the prompt.
        Assert.True(
            announced.ApproximateTokens * 8 < inlined.ApproximateTokens,
            $"announced {announced.ApproximateTokens} tokens vs inlined {inlined.ApproximateTokens} — "
            + "the metadata tier is supposed to be an order of magnitude cheaper");
    }

    // ---- directory conventions ----------------------------------------------

    [Fact]
    public void AnyFileOrDirectory_IsSupported_NotJustTheThreeConventionalOnes()
    {
        // "A skill directory may contain any files and directories beyond the required
        // SKILL.md." The published corpus relies on this: claude-api ships csharp/,
        // go/, python/ and more; pdf ships forms.md and reference.md at its root.
        using var fixture = new Corpus();
        string dir = fixture.Dir("anyshape");
        Directory.CreateDirectory(Path.Combine(dir, "csharp"));
        Directory.CreateDirectory(Path.Combine(dir, "core"));
        File.WriteAllText(Path.Combine(dir, "SKILL.md"),
            "---\nname: anyshape\ndescription: Ships files outside the three conventional directories.\n---\nbody\n");
        File.WriteAllText(Path.Combine(dir, "csharp", "guide.md"), "# C#\n");
        File.WriteAllText(Path.Combine(dir, "core", "helper.py"), "pass\n");
        File.WriteAllText(Path.Combine(dir, "forms.md"), "# Forms\n");

        Skill skill = fixture.Load("anyshape");

        Assert.Contains(skill.BundledFiles, f => f.Path == "csharp/guide.md");
        Assert.Contains(skill.BundledFiles, f => f.Path == "core/helper.py");
        Assert.Contains(skill.BundledFiles, f => f.Path == "forms.md");
    }

    // ---- where TensorSharp matches the reference implementations -------------

    [Fact]
    public void DescriptionsAreCollapsedToOneLine_AsCodexDoes()
    {
        // Codex's parser calls sanitize_single_line on name and description
        // (codex-rs/skills/src/parser.rs) for the same reason: the catalogue is one
        // skill per line, and a folded or literal block scalar would fragment it.
        const string doc = "---\nname: folded\ndescription: >\n  first line\n  second line\n---\nbody\n";

        Assert.True(SkillManifestParser.TryParse(doc, "folded", out var m, out _));
        Assert.DoesNotContain('\n', m!.Description);
        Assert.Equal("first line second line", m.Description);
    }

    [Fact]
    public void TokenEstimation_UsesFourBytesPerToken_AsCodexDoes()
    {
        // codex-rs/ext/skills/src/render.rs: APPROX_BYTES_PER_TOKEN = 4. Budgeting has
        // to happen before a tokenizer is necessarily available, and matching the
        // reference implementation's approximation keeps the budgets comparable.
        Assert.Equal(4, SkillTextBudget.BytesPerToken);
        Assert.Equal(1, SkillTextBudget.ApproximateTokens("abcd"));
        Assert.Equal(2, SkillTextBudget.ApproximateTokens("abcde"));
    }

    [Fact]
    public void ATruncatedCatalogueDescription_EndsWithAnEllipsis_AsCodexDoes()
    {
        // codex-rs/ext/skills/src/render.rs: TRUNCATED_SKILL_DESCRIPTION_SUFFIX = "...".
        string truncated = SkillTextBudget.Truncate(new string('a', 500), 100);

        Assert.True(truncated.Length <= 100);
        Assert.EndsWith("...", truncated, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOverLongSkillIsPaged_NotSilentlyCut()
    {
        // Codex's skills.read returns a next_cursor and its prompt tells the model to
        // "follow next_cursor until EOF". TensorSharp reports the byte offset to
        // continue from, for the same reason: a reference file cut in half without
        // saying so is worse than one the model knows to keep reading.
        using var fixture = new Corpus();
        Skill skill = fixture.Write("paged", reference: new string('r', 5000));

        Assert.True(skill.TryReadResource("references/detail.md", 1024, 0, out var page, out string error), error);
        Assert.True(page.Truncated);
        Assert.Equal(1024, page.NextOffsetBytes);

        Assert.True(skill.TryReadResource("references/detail.md", 8192, page.NextOffsetBytes, out var rest, out _));
        Assert.False(rest.Truncated);
    }

    // ---- where TensorSharp deliberately differs ------------------------------

    [Fact]
    public void ToolNamesUseUnderscores_NotCodexsDottedNamespace()
    {
        // Codex names its tools skills.list / skills.read. TensorSharp cannot: several
        // chat templates splice a tool's name into their markup unescaped — Gemma 4
        // writes `<|tool>declaration:{name}{`, GLM writes `<tool_call>{name}<arg_key>`,
        // Harmony writes `type {name} =` — and a dot in that position is not safe
        // across all eleven protocols. The capability is identical; only the spelling
        // differs, and this test exists so the divergence stays deliberate.
        Assert.Equal("skills_list", SkillTools.ListToolName);
        Assert.Equal("skills_read", SkillTools.ReadToolName);

        foreach (ToolFunction tool in SkillTools.BuiltIn(allowScripts: true))
        {
            Assert.Matches("^[a-z_]+$", tool.Name);
        }
    }

    [Fact]
    public void ToolParametersAreFlatScalars_BecauseToolParameterCannotExpressMore()
    {
        // Codex generates JSON Schema from Rust types and can express nested shapes.
        // TensorSharp's ToolParameter carries only {Type, Description, Enum}; `items`
        // and nested `properties` are dropped when a tool is parsed and cannot be
        // re-emitted, and the Harmony renderer degrades an array to `any[]`. So every
        // parameter here is a scalar, and skills_run accepts EITHER a string or a list
        // for `args` at the reading end (see SkillTools.ReadArgumentList) because models
        // emit an array regardless of what the declaration says.
        foreach (ToolFunction tool in SkillTools.BuiltIn(allowScripts: true))
        {
            foreach (KeyValuePair<string, ToolParameter> parameter in tool.Parameters)
            {
                Assert.True(
                    parameter.Value.Type is "string" or "integer" or "boolean" or "number",
                    $"{tool.Name}.{parameter.Key} is '{parameter.Value.Type}'; a nested schema cannot round-trip through ToolParameter");
            }
        }
    }

    [Fact]
    public void ScriptExecutionIsOffByDefault_WhichIsStricterThanEitherReference()
    {
        // Claude Code and Codex both run a skill's scripts through their general shell
        // tool, gated by the harness's own approval flow. TensorSharp is an inference
        // server with no such flow, so the tool is not even declared unless an operator
        // opts in — and when it is, it runs sandboxed or refuses.
        Assert.DoesNotContain(SkillTools.BuiltIn(), t => t.Name == SkillTools.RunToolName);
        Assert.Contains(SkillTools.BuiltIn(allowScripts: true), t => t.Name == SkillTools.RunToolName);
        Assert.Equal(SkillSandboxMode.Required, new SkillHostOptions().Sandbox);
    }

    /// <summary>Temp-directory fixture for the skills these tests need on disk.</summary>
    private sealed class Corpus : IDisposable
    {
        private readonly string _root = Path.Combine(
            Path.GetTempPath(), "ts-skill-conformance-" + Guid.NewGuid().ToString("N"));

        public Corpus() => Directory.CreateDirectory(_root);

        public string Dir(string name)
        {
            string dir = Path.Combine(_root, name);
            Directory.CreateDirectory(dir);
            return dir;
        }

        public Skill Write(string name, string body = "body", string? reference = null)
        {
            string dir = Dir(name);
            File.WriteAllText(Path.Combine(dir, "SKILL.md"),
                $"---\nname: {name}\ndescription: A skill used by the conformance tests.\n---\n\n{body}\n");
            if (reference != null)
            {
                Directory.CreateDirectory(Path.Combine(dir, "references"));
                File.WriteAllText(Path.Combine(dir, "references", "detail.md"), reference);
            }
            return Load(name);
        }

        public Skill Load(string name)
        {
            var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _root } });
            return registry.Skills.Single(s => s.Id == name);
        }

        public void Dispose()
        {
            try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
        }
    }
}
