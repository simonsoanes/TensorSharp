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
/// Pins the two things the injected skills block must get right: where it goes, and
/// that it is byte-for-byte reproducible.
///
/// <para>
/// <b>Where it goes.</b> The block is merged into a leading <c>system</c> or
/// <c>developer</c> turn rather than appended as a second system message, because that
/// is the only injection point every renderer in this repository handles. Appending is
/// silently dropped by the Mistral 3 renderer, produces a duplicate system turn on
/// Harmony (which lifts <c>messages[0]</c> into its developer block and synthesizes its
/// own system block), and is the shape a GGUF-embedded Jinja template is most likely to
/// reject — which falls back to the hardcoded renderer and changes the entire prompt
/// format. Every one of those failures leaves a request that still returns a plausible
/// answer, so nothing but a test catches them.
/// </para>
/// <para>
/// <b>Reproducibility.</b> This block sits at the very front of the prompt, and the KV
/// prefix cache chains a hash over 256-token blocks from block 0 and stops adopting at
/// the first mismatch. A timestamp, an absolute path, a "3 skills installed" counter, or
/// a selection rendered in whatever order the caller's JSON happened to list it would
/// change block 0 on every turn and drop prefix reuse to zero for the whole
/// conversation — not merely for the part that changed. The clone is guarded for the
/// same reason: <c>StructuredOutputPrompt.CloneMessage</c> drops
/// <c>RawOutputTokens</c> and <c>TextFilePaths</c>, and copying that shape here would
/// make every assistant turn re-tokenize instead of splice, costing a full re-prefill
/// per skill round-trip while still producing a correct answer.
/// </para>
/// </summary>
public class SkillPromptTests : IDisposable
{
    private readonly string _baseDir;

    public SkillPromptTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-prompt-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- helpers -----------------------------------------------------------

    /// <summary>
    /// A <see cref="Skill"/> can only be built by loading one off disk — its
    /// constructor belongs to the registry — so every fixture here is a real skill
    /// directory, which also keeps the file index and the body budget honest.
    /// </summary>
    private void WriteSkill(string name, string description, string body = "Do the thing.")
    {
        string dir = Path.Combine(_baseDir, name);
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, "SKILL.md"),
            $"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n");
    }

    private IReadOnlyList<Skill> Load() =>
        new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } }).Skills;

    private Skill Load(string id) => Load().Single(s => s.Id == id);

    private static List<Skill> Only(params Skill[] skills) => skills.ToList();

    // ---- Apply: where the block lands ---------------------------------------

    [Fact]
    public void Apply_ALeadingSystemMessage_IsMergedIntoRatherThanDuplicated()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are terse." },
            new() { Role = "user", Content = "hello" },
        };

        List<ChatMessage> result = SkillPrompt.Apply(messages, "## Agent skills\nblock body");

        Assert.Equal(2, result.Count);
        Assert.Equal("system", result[0].Role);
        Assert.StartsWith("You are terse.", result[0].Content, StringComparison.Ordinal);
        Assert.Contains("## Agent skills", result[0].Content, StringComparison.Ordinal);
    }

    [Fact]
    public void Apply_ALeadingDeveloperMessage_IsMergedIntoAsWell()
    {
        // Harmony's renderer lifts messages[0] into its developer block whichever of
        // the two roles it carries, so both have to be treated as "the preamble".
        var messages = new List<ChatMessage>
        {
            new() { Role = "developer", Content = "Follow the house style." },
            new() { Role = "user", Content = "hello" },
        };

        List<ChatMessage> result = SkillPrompt.Apply(messages, "## Agent skills\nblock body");

        Assert.Equal(2, result.Count);
        Assert.Equal("developer", result[0].Role);
        Assert.Contains("Follow the house style.", result[0].Content, StringComparison.Ordinal);
        Assert.Contains("## Agent skills", result[0].Content, StringComparison.Ordinal);
    }

    [Fact]
    public void Apply_LeadingMessageMarkerStaysAtOriginalPreambleBoundary()
    {
        const string stablePreamble = "Follow the house style.";
        var original = new ChatMessage
        {
            Role = "developer",
            Content = stablePreamble + "  ",
            CacheControl = new CacheControlMarker(),
            ContentCacheBreakpoints = new List<int> { 6 },
        };

        ChatMessage injected = SkillPrompt.Apply(
            new List<ChatMessage> { original }, "## Agent skills\nblock body")[0];

        Assert.Null(injected.CacheControl);
        Assert.Equal(new[] { 6, stablePreamble.Length }, injected.ContentCacheBreakpoints);
        Assert.StartsWith(stablePreamble + "\n\n", injected.Content, StringComparison.Ordinal);
        Assert.NotNull(original.CacheControl);
    }

    [Fact]
    public void Apply_TrailingWhitespaceMarkersAreClampedBeforeSkillsAreAppended()
    {
        const string stablePreamble = "Stable preamble.";
        string contentWithWhitespace = stablePreamble + " \t  ";
        var original = new ChatMessage
        {
            Role = "system",
            Content = contentWithWhitespace,
            CacheControl = new CacheControlMarker(),
            ContentCacheBreakpoints = new List<int>
            {
                contentWithWhitespace.Length,
                4,
                stablePreamble.Length + 1,
                4,
            },
        };

        ChatMessage injected = SkillPrompt.Apply(
            new List<ChatMessage> { original }, "## Agent skills\nblock body")[0];

        Assert.Equal(new[] { 4, stablePreamble.Length }, injected.ContentCacheBreakpoints);
        Assert.Null(injected.CacheControl);
        Assert.Equal(
            new[] { contentWithWhitespace.Length, 4, stablePreamble.Length + 1, 4 },
            original.ContentCacheBreakpoints);
    }

    [Fact]
    public void Apply_NoLeadingPreamble_PrependsASystemMessage()
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "hello" } };

        List<ChatMessage> result = SkillPrompt.Apply(messages, "## Agent skills\nblock body");

        Assert.Equal(2, result.Count);
        Assert.Equal("system", result[0].Role);
        Assert.Equal("user", result[1].Role);
    }

    [Fact]
    public void Apply_DoesNotMutateTheCallersList()
    {
        // The caller's list is very often the tracked session history; mutating it
        // would leave the skills block permanently glued to the stored conversation and
        // re-injected, growing, on every subsequent turn.
        var original = new ChatMessage { Role = "system", Content = "You are terse." };
        var messages = new List<ChatMessage> { original, new() { Role = "user", Content = "hello" } };

        SkillPrompt.Apply(messages, "## Agent skills\nblock body");

        Assert.Equal(2, messages.Count);
        Assert.Equal("You are terse.", original.Content);
    }

    [Fact]
    public void Apply_ClonePreservesRawOutputTokensAndTextFilePaths()
    {
        // The regression this file exists for. StructuredOutputPrompt's own cloner
        // drops both of these fields; on the server the loss is invisible because
        // ChatHistoryPreparer re-attaches the raw tokens afterwards, but the CLI and
        // the agentic loop have no such repair. Losing RawOutputTokens makes
        // KVCachePromptRenderer re-tokenize each assistant turn instead of splicing it,
        // so the re-rendered prefix stops matching the cache at the first assistant
        // boundary and every skill round-trip pays a full re-prefill — silently, with a
        // still-correct answer.
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are terse." },
            new()
            {
                Role = "assistant",
                Content = "Earlier answer.",
                RawOutputTokens = new List<int> { 11, 22, 33 },
                RawPromptTrailingWhitespace = "\n",
                TextFilePaths = new List<string> { "/tmp/notes.txt" },
                CacheControl = new CacheControlMarker(),
                ContentCacheBreakpoints = new List<int> { 7 },
            },
        };

        List<ChatMessage> result = SkillPrompt.Apply(messages, "## Agent skills\nblock body");

        Assert.NotSame(messages[1], result[1]);                       // really a copy
        Assert.Equal(new[] { 11, 22, 33 }, result[1].RawOutputTokens);
        Assert.Equal("\n", result[1].RawPromptTrailingWhitespace);
        Assert.Equal(new[] { "/tmp/notes.txt" }, result[1].TextFilePaths);
        Assert.Equal("ephemeral", result[1].CacheControl?.Type);
        Assert.Equal(new[] { 7 }, result[1].ContentCacheBreakpoints);
    }

    [Fact]
    public void Apply_AnEmptyPlan_ReturnsTheInputUnchanged()
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "hello" } };

        List<ChatMessage> result = SkillPrompt.Apply(messages, SkillPlan.Empty);

        // The same instance, not a copy: a request with no skills must cost nothing,
        // and a copy would defeat the caller's own identity checks on the history.
        Assert.Same(messages, result);
    }

    // ---- Plan: byte stability ----------------------------------------------

    [Fact]
    public void Plan_TheSameSetInADifferentOrder_RendersIdenticalBytes()
    {
        WriteSkill("alpha", "does alpha things");
        WriteSkill("zebra", "does zebra things");
        Skill alpha = Load("alpha");
        Skill zebra = Load("zebra");

        string forwards = SkillPrompt.Plan(Only(alpha, zebra), null).Instructions;
        string backwards = SkillPrompt.Plan(Only(zebra, alpha), null).Instructions;

        // A caller's JSON lists skills in whatever order the user typed them. If that
        // order reached the prompt, the same conversation would miss the KV prefix
        // cache from block 0 purely because the array was written the other way round.
        Assert.Equal(forwards, backwards);
        Assert.Contains("alpha", forwards, StringComparison.Ordinal);
        Assert.Contains("zebra", forwards, StringComparison.Ordinal);
    }

    [Fact]
    public void Plan_RepeatedCalls_RenderIdenticalBytes()
    {
        WriteSkill("pdf", "does pdfs");
        WriteSkill("xlsx", "does spreadsheets");
        IReadOnlyList<Skill> all = Load();

        string first = SkillPrompt.Plan(Only(all[0]), all.ToList()).Instructions;
        string second = SkillPrompt.Plan(Only(all[0]), all.ToList()).Instructions;

        Assert.Equal(first, second);
    }

    [Fact]
    public void Plan_RendersNoAbsolutePathAndNothingThatVariesBetweenRuns()
    {
        WriteSkill("pdf", "does pdfs");
        Skill pdf = Load("pdf");

        string instructions = SkillPrompt.Plan(Only(pdf), null).Instructions;

        // An absolute path is both a prefix-cache killer (it carries a per-run GUID
        // here, and a per-machine home directory in production) and an information
        // leak: the model is told the host's directory layout for no benefit.
        Assert.DoesNotContain(_baseDir, instructions, StringComparison.Ordinal);
        Assert.DoesNotContain(pdf.RootDirectory, instructions, StringComparison.Ordinal);

        // Nothing dated, and no "N skills installed" style counter, either of which
        // would change block 0 between two otherwise identical requests.
        Assert.DoesNotContain(DateTime.UtcNow.Year.ToString(System.Globalization.CultureInfo.InvariantCulture),
            instructions, StringComparison.Ordinal);
        Assert.DoesNotContain("installed", instructions, StringComparison.Ordinal);
    }

    // ---- Plan: budget ------------------------------------------------------

    [Fact]
    public void Plan_ABodyOverTheInlineBudget_IsDeferredWithAnInstructionToReadIt()
    {
        WriteSkill("big", "ships a very long instruction file", new string('x', 4000));
        Skill big = Load("big");

        SkillPlan plan = SkillPrompt.Plan(
            Only(big), null,
            new SkillPromptOptions { MaxBlockTokens = 4000, MaxInlineBodyTokens = 64 });

        // Announced, not inlined: the model still learns the skill applies, and is told
        // exactly how to fetch it. Silently dropping it would leave the user's explicit
        // "use this skill" doing nothing at all.
        Assert.Empty(plan.Inlined);
        Assert.Equal(new[] { "big" }, plan.Deferred.Select(s => s.Id));
        Assert.Contains("skills_read(skill=\"big\", path=\"SKILL.md\")", plan.Instructions, StringComparison.Ordinal);
        Assert.DoesNotContain("xxxxxxxxxx", plan.Instructions, StringComparison.Ordinal);
    }

    [Fact]
    public void Plan_ASkillInBothTheSelectionAndTheCatalog_IsNotListedTwice()
    {
        // The natural call is Plan(selected, registry.Skills) — the caller hands over
        // the whole registry as the catalog. The selected skill is already fully
        // present, so repeating it in the catalog spends tokens telling the model
        // something it can read two lines above.
        WriteSkill("pdf", "does pdfs");
        WriteSkill("xlsx", "does spreadsheets");
        IReadOnlyList<Skill> all = Load();
        Skill pdf = all.Single(s => s.Id == "pdf");

        SkillPlan plan = SkillPrompt.Plan(Only(pdf), all.ToList());

        // Selected skills are now announced in the same "- id: description" shape as
        // catalogue ones, so the check is that pdf appears ONCE — under the selection —
        // rather than that it never appears with that marker.
        Assert.Equal(new[] { "xlsx" }, plan.Catalog.Select(s => s.Id));
        Assert.Equal(1, CountOccurrences(plan.Instructions, "- pdf:"));
        Assert.Contains("- xlsx:", plan.Instructions, StringComparison.Ordinal);
    }

    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0, at = 0;
        while ((at = haystack.IndexOf(needle, at, StringComparison.Ordinal)) >= 0)
        {
            count++;
            at += needle.Length;
        }
        return count;
    }

    [Fact]
    public void Plan_NothingSelectedAndNothingToAdvertise_IsEmpty()
    {
        SkillPlan plan = SkillPrompt.Plan(new List<Skill>(), new List<Skill>());

        Assert.Same(SkillPlan.Empty, plan);
        Assert.True(plan.IsEmpty);
        Assert.Equal(string.Empty, plan.Instructions);
    }

    // ---- Plan: no tools ------------------------------------------------------

    [Fact]
    public void Plan_WithoutTools_DropsTheDiscoveryCatalog()
    {
        // A catalog is an invitation to call skills_read. Where tool declarations never
        // reach the model, that invitation can only ever be declined, so the tokens are
        // better spent on the bodies the model can actually use.
        WriteSkill("pdf", "does pdfs");
        WriteSkill("xlsx", "does spreadsheets");
        IReadOnlyList<Skill> all = Load();

        SkillPlan plan = SkillPrompt.Plan(
            Only(all.Single(s => s.Id == "pdf")), all.ToList(),
            new SkillPromptOptions { ToolsAvailable = false });

        Assert.Empty(plan.Catalog);
        Assert.False(plan.ToolsAvailable);
        Assert.DoesNotContain("xlsx", plan.Instructions, StringComparison.Ordinal);
    }

    [Fact]
    public void Plan_WithoutTools_TheUsageTextNamesNoToolTheModelCannotCall()
    {
        // Telling a model without tool declarations to "call skills_read" is an instruction
        // it has no way to follow; the best case is that it ignores the line, the worst
        // is that it hallucinates the call as prose and the user sees markup.
        WriteSkill("pdf", "does pdfs");

        SkillPlan plan = SkillPrompt.Plan(
            Only(Load("pdf")), null,
            new SkillPromptOptions { ToolsAvailable = false });

        Assert.DoesNotContain("skills_read", plan.Instructions, StringComparison.Ordinal);
        Assert.DoesNotContain("skills_list", plan.Instructions, StringComparison.Ordinal);
        Assert.Contains("already written above", plan.Instructions, StringComparison.Ordinal);
    }

    [Fact]
    public void Plan_WithoutTools_ASingleOversizedSkillIsInlinedAnyway()
    {
        // With no tools there is no second chance: a body that is not inlined is simply
        // unavailable, and the user's explicit selection would do nothing. Truncated
        // instructions with a notice beat no instructions at all.
        WriteSkill("big", "ships a very long instruction file", new string('x', 4000));

        SkillPlan plan = SkillPrompt.Plan(
            Only(Load("big")), null,
            new SkillPromptOptions { ToolsAvailable = false, MaxBlockTokens = 4000, MaxInlineBodyTokens = 64 });

        Assert.Equal(new[] { "big" }, plan.Inlined.Select(s => s.Id));
        Assert.Empty(plan.Deferred);
        Assert.Contains("truncated", plan.Instructions, StringComparison.Ordinal);
    }

    [Fact]
    public void Plan_WithTools_AnnouncesASelectedSkill_WithoutSpendingItsBody()
    {
        // The normal path. Selection puts the skill in front of the model as a name and
        // a description; the body and the file index are what activation buys. Inlining
        // here was 96% of the skills block on a real four-skill request, and half of it
        // was for skills the model never touched.
        string dir = Path.Combine(_baseDir, "pdf");
        WriteSkill("pdf", "does pdfs", "Read the form, then fill it.");
        Directory.CreateDirectory(Path.Combine(dir, "scripts"));
        // Deliberately NOT the name the usage text uses as its own worked example, or
        // the assertion below would match that instead of a leaked file index.
        File.WriteAllText(Path.Combine(dir, "scripts", "harvest_fields.py"), "print('hi')\n");

        SkillPlan plan = SkillPrompt.Plan(Only(Load("pdf")), null);

        Assert.Empty(plan.Inlined);
        Assert.Equal(new[] { "pdf" }, plan.Deferred.Select(s => s.Id));
        Assert.Contains("- pdf: does pdfs", plan.Instructions, StringComparison.Ordinal);
        Assert.Contains("NOT loaded", plan.Instructions, StringComparison.Ordinal);
        Assert.Contains("skills_read(skill=\"pdf\", path=\"SKILL.md\")", plan.Instructions, StringComparison.Ordinal);
        Assert.DoesNotContain("Read the form, then fill it.", plan.Instructions, StringComparison.Ordinal);
        Assert.DoesNotContain("scripts/harvest_fields.py", plan.Instructions, StringComparison.Ordinal);
    }

    [Fact]
    public void Plan_WhenTheHostAsksForIt_StillInlinesSelectedBodies()
    {
        // The escape hatch stays exercised: a host that knows its model will not make
        // the extra call can still pay tokens for certainty.
        WriteSkill("pdf", "does pdfs", "Read the form, then fill it.");

        SkillPlan plan = SkillPrompt.Plan(
            Only(Load("pdf")), null, new SkillPromptOptions { InlineSelectedBodies = true });

        Assert.Equal(new[] { "pdf" }, plan.Inlined.Select(s => s.Id));
        Assert.Contains("Read the form, then fill it.", plan.Instructions, StringComparison.Ordinal);
    }
}
