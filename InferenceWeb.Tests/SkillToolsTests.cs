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
using System.Text.Json;
using System.Text.RegularExpressions;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the two built-in tools: the shape they are declared in, and the answers they
/// give.
///
/// <para>
/// <b>The shape.</b> These declarations are rendered by every family's template, and
/// <c>ToolParameter</c> can carry only a type name, a description and an enum —
/// <c>items</c>, nested <c>properties</c> and the rest of JSON Schema are dropped when a
/// tool is parsed and cannot be re-emitted, and the Harmony renderer degrades an
/// <c>array</c> parameter to <c>any[]</c>. A parameter declared as an array or an object
/// would therefore reach the model as something it cannot fill in correctly, and the
/// only symptom would be a model that keeps calling the tool wrong. The names are held
/// to <c>[a-z_]</c> for the same reason: several renderers splice a tool's name into
/// their markup unescaped.
/// </para>
/// <para>
/// <b>The answers.</b> Every failure here is a conversational one — the model is
/// mid-task, and an exception would abort a request that is still perfectly answerable —
/// so each error has to be phrased so the model can correct itself in one step. An
/// unknown skill name comes back with the list of names that do exist; a path that
/// escapes the skill says "escapes" rather than "does not exist", because the latter
/// teaches the model the file is missing rather than that it may not look there, and it
/// will retry the same shape until the round budget is gone.
/// </para>
/// </summary>
public class SkillToolsTests : IDisposable
{
    private readonly string _baseDir;

    public SkillToolsTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-tools-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- helpers -----------------------------------------------------------

    private string WriteSkill(string name, string description, string body = "Do the thing.")
    {
        string dir = Path.Combine(_baseDir, name);
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, "SKILL.md"),
            $"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n");
        return dir;
    }

    private SkillToolContext Context(int maxReadBytes = SkillTools.DefaultMaxReadBytes) =>
        new(new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } }).Skills, maxReadBytes);

    private static ToolCall Call(string name, params (string Key, object Value)[] arguments)
    {
        var args = new Dictionary<string, object>();
        foreach ((string key, object value) in arguments)
            args[key] = value;
        return new ToolCall { Name = name, Arguments = args };
    }

    /// <summary>
    /// A detached <see cref="JsonElement"/>, which is how arguments arrive from the
    /// server's request parsers. <c>Deserialize</c> clones, so the element outlives the
    /// document it came from.
    /// </summary>
    private static JsonElement Json(string raw) => JsonSerializer.Deserialize<JsonElement>(raw);

    // ---- declarations ------------------------------------------------------

    [Fact]
    public void BuiltIn_DeclaresExactlyTheTwoReadOnlyTools()
    {
        List<ToolFunction> tools = SkillTools.BuiltIn();

        Assert.Equal(new[] { "skills_list", "skills_read" }, tools.Select(t => t.Name));
    }

    [Fact]
    public void BuiltIn_List_TakesNoParametersAtAll()
    {
        ToolFunction list = SkillTools.BuiltIn().Single(t => t.Name == SkillTools.ListToolName);

        // Parameters and Required must be empty TOGETHER: the Jinja rendering path
        // marks every parameter required when Required is empty, so a tool with
        // optional-only arguments is misdeclared there.
        Assert.Empty(list.Parameters);
        Assert.Empty(list.Required);
    }

    [Fact]
    public void BuiltIn_Read_TakesSkillPathAndOffsetAndRequiresTheFirstTwo()
    {
        ToolFunction read = SkillTools.BuiltIn().Single(t => t.Name == SkillTools.ReadToolName);

        Assert.Equal(new[] { "skill", "path", "offset" }, read.Parameters.Keys);
        Assert.Equal(new[] { "skill", "path" }, read.Required);

        // offset is optional: a model that has not paged yet must not be forced to
        // invent a value for it.
        Assert.DoesNotContain("offset", read.Required);
    }

    [Fact]
    public void BuiltIn_EveryParameterIsAFlatScalar()
    {
        // ToolParameter cannot express `items` or nested `properties`, and the Harmony
        // renderer turns an array parameter into `any[]`. Declaring one would produce a
        // tool the model can see and cannot fill in.
        foreach (ToolFunction tool in SkillTools.BuiltIn(allowScripts: true))
        {
            foreach ((string name, ToolParameter parameter) in tool.Parameters)
            {
                Assert.True(
                    parameter.Type is "string" or "integer" or "number" or "boolean",
                    $"{tool.Name}.{name} is declared as '{parameter.Type}'");
            }
        }
    }

    [Fact]
    public void BuiltIn_Run_AppearsOnlyWhenScriptsAreAllowed()
    {
        // Running a skill's script is arbitrary code execution on the host, decided by
        // a model reading untrusted Markdown. It must not even be offered by default.
        Assert.DoesNotContain(SkillTools.BuiltIn(allowScripts: false), t => t.Name == SkillTools.RunToolName);
        Assert.Contains(SkillTools.BuiltIn(allowScripts: true), t => t.Name == SkillTools.RunToolName);
    }

    [Fact]
    public void BuiltIn_ToolNamesUseOnlyLowercaseLettersAndUnderscores()
    {
        // Several families splice the name into their markup unescaped — Gemma 4 writes
        // `<|tool>declaration:NAME{`, Harmony writes `to=functions.NAME` — so a dot, a
        // space or a quote in a name is a prompt-injection surface, not a style issue.
        foreach (ToolFunction tool in SkillTools.BuiltIn(allowScripts: true))
            Assert.Matches(new Regex("^[a-z_]+$"), tool.Name);
    }

    // ---- merging with the caller's tools ------------------------------------

    [Fact]
    public void Merge_AppendsToAClientList_WithoutTouchingIt()
    {
        var client = new List<ToolFunction> { new() { Name = "get_weather" } };

        List<ToolFunction> merged = SkillTools.Merge(client, allowScripts: false, out var shadowed);

        Assert.Equal(new[] { "get_weather", "skills_list", "skills_read" }, merged.Select(t => t.Name));
        Assert.Empty(shadowed);
        Assert.Single(client);   // the caller's own list is left alone
    }

    [Fact]
    public void Merge_ANullClientList_IsTolerated()
    {
        List<ToolFunction> merged = SkillTools.Merge(null, allowScripts: false, out var shadowed);

        Assert.Equal(2, merged.Count);
        Assert.Empty(shadowed);
    }

    [Fact]
    public void Merge_AClientToolOwningOneOfTheNames_WinsAndIsReported()
    {
        // The client has an implementation and an expectation. Shadowing it would break
        // a working integration to add a feature the client never asked for, so the
        // collision is reported for the host to log instead.
        var client = new List<ToolFunction>
        {
            new() { Name = "skills_read", Description = "the client's own reader" },
        };

        List<ToolFunction> merged = SkillTools.Merge(client, allowScripts: false, out var shadowed);

        Assert.Equal(new[] { "skills_read" }, shadowed);
        Assert.Equal("the client's own reader", merged.Single(t => t.Name == "skills_read").Description);
        Assert.Single(merged, t => t.Name == "skills_read");
    }

    // ---- skills_list --------------------------------------------------------

    [Fact]
    public void Execute_List_NamesEverySkillAndItsBundledFiles()
    {
        string dir = WriteSkill("pdf", "does pdfs");
        Directory.CreateDirectory(Path.Combine(dir, "references"));
        File.WriteAllText(Path.Combine(dir, "references", "api.md"), "# API\n");
        WriteSkill("xlsx", "does spreadsheets");

        SkillToolResult result = SkillTools.Execute(Call(SkillTools.ListToolName), Context());

        Assert.True(result.Ok);
        Assert.Contains("pdf", result.Content, StringComparison.Ordinal);
        Assert.Contains("xlsx", result.Content, StringComparison.Ordinal);
        Assert.Contains("references/api.md", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void Execute_List_WithNoSkills_SaysSoInsteadOfReturningNothing()
    {
        SkillToolResult result = SkillTools.Execute(Call(SkillTools.ListToolName), Context());

        Assert.True(result.Ok);
        Assert.Contains("No skills are available", result.Content, StringComparison.Ordinal);
    }

    // ---- skills_read --------------------------------------------------------

    [Fact]
    public void Execute_Read_ReturnsTheNamedFile()
    {
        string dir = WriteSkill("pdf", "does pdfs");
        Directory.CreateDirectory(Path.Combine(dir, "references"));
        File.WriteAllText(Path.Combine(dir, "references", "api.md"), "# API\nEverything you need.\n");

        SkillToolResult result = SkillTools.Execute(
            Call(SkillTools.ReadToolName, ("skill", "pdf"), ("path", "references/api.md")), Context());

        Assert.True(result.Ok);
        Assert.Equal("pdf", result.SkillId);
        Assert.Equal("references/api.md", result.ResourcePath);
        Assert.Contains("Everything you need.", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void Execute_Read_WithNoPath_DefaultsToTheSkillsOwnInstructions()
    {
        // A model that has just been told a skill exists very often asks for it with no
        // path at all, meaning "give me the instructions". Answering that is better than
        // spending a whole round-trip teaching it the argument.
        WriteSkill("pdf", "does pdfs", "Read the form, then fill it.");

        SkillToolResult result = SkillTools.Execute(
            Call(SkillTools.ReadToolName, ("skill", "pdf")), Context());

        Assert.True(result.Ok);
        Assert.Equal("SKILL.md", result.ResourcePath);
        Assert.Contains("Read the form, then fill it.", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void Execute_Read_ARedundantSkillNamePrefix_IsAccepted()
    {
        // Models routinely answer "read pdf's reference" with path="pdf/references/api.md",
        // because that is how the file is spelled relative to the skills directory
        // rather than to the skill. Refusing it costs a round-trip and the model
        // usually retries the same shape; accepting it cannot widen the sandbox,
        // because the result is still resolved inside the same skill.
        string dir = WriteSkill("pdf", "does pdfs");
        Directory.CreateDirectory(Path.Combine(dir, "references"));
        File.WriteAllText(Path.Combine(dir, "references", "api.md"), "# API\n");

        SkillToolResult result = SkillTools.Execute(
            Call(SkillTools.ReadToolName, ("skill", "pdf"), ("path", "pdf/references/api.md")), Context());

        Assert.True(result.Ok, result.Content);
        Assert.Equal("references/api.md", result.ResourcePath);
    }

    [Fact]
    public void Execute_Read_APathIntoAnotherSkill_SaysItEscapesRatherThanThatItIsMissing()
    {
        // The prefix stripper deliberately leaves "..", so the guard reports what
        // actually happened. "Does not exist" would teach the model the file is missing
        // and it would try another spelling of the same escape until the loop gave up.
        WriteSkill("pdf", "does pdfs");
        WriteSkill("xlsx", "does spreadsheets");

        SkillToolResult result = SkillTools.Execute(
            Call(SkillTools.ReadToolName, ("skill", "pdf"), ("path", "../xlsx/SKILL.md")), Context());

        Assert.False(result.Ok);
        Assert.Contains("escapes the skill directory", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("does not exist", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void Execute_Read_AnUnknownSkill_ListsTheNamesThatDoExist()
    {
        WriteSkill("pdf", "does pdfs");
        WriteSkill("xlsx", "does spreadsheets");

        SkillToolResult result = SkillTools.Execute(
            Call(SkillTools.ReadToolName, ("skill", "pdff"), ("path", "SKILL.md")), Context());

        Assert.False(result.Ok);
        Assert.Contains("pdf", result.Content, StringComparison.Ordinal);
        Assert.Contains("xlsx", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void Execute_Read_WithNoSkillArgument_SaysWhichArgumentIsMissing()
    {
        WriteSkill("pdf", "does pdfs");

        SkillToolResult result = SkillTools.Execute(
            Call(SkillTools.ReadToolName, ("path", "SKILL.md")), Context());

        Assert.False(result.Ok);
        Assert.Contains("'skill' argument", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void Execute_Read_ArgumentsSuppliedAsJsonElements_AreRead()
    {
        // Which shape the arguments arrive in depends on which parser produced them:
        // the server's request parsers hand over JsonElement, the streaming output
        // parsers hand over boxed primitives, and a hand-built call carries strings.
        WriteSkill("pdf", "does pdfs", new string('a', 2000));

        SkillToolResult result = SkillTools.Execute(
            Call(SkillTools.ReadToolName,
                ("skill", Json("\"pdf\"")),
                ("path", Json("\"SKILL.md\"")),
                ("offset", Json("0"))),
            Context());

        Assert.True(result.Ok, result.Content);
        Assert.Equal("SKILL.md", result.ResourcePath);
    }

    [Fact]
    public void Execute_Read_ABoxedIntegerOffset_IsRead()
    {
        WriteSkill("pdf", "does pdfs", new string('a', 2000));

        SkillToolResult result = SkillTools.Execute(
            Call(SkillTools.ReadToolName, ("skill", "pdf"), ("path", "SKILL.md"), ("offset", 64)),
            Context());

        Assert.True(result.Ok, result.Content);
        Assert.Contains("Bytes 64-", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void Execute_Read_ALongFile_ReportsWhereToContinueAndContinuesThere()
    {
        // Progressive disclosure only works if the model is told, in the result itself,
        // exactly how to ask for the rest — it has no other way to know the file was cut.
        string dir = WriteSkill("pdf", "does pdfs");
        File.WriteAllText(Path.Combine(dir, "big.md"), new string('a', 2000));
        SkillToolContext context = Context(maxReadBytes: 512);

        SkillToolResult first = SkillTools.Execute(
            Call(SkillTools.ReadToolName, ("skill", "pdf"), ("path", "big.md")), context);

        Assert.True(first.Ok, first.Content);
        Assert.Contains("Bytes 0-512 of 2000", first.Content, StringComparison.Ordinal);
        Assert.Contains("offset=512", first.Content, StringComparison.Ordinal);

        SkillToolResult second = SkillTools.Execute(
            Call(SkillTools.ReadToolName, ("skill", "pdf"), ("path", "big.md"), ("offset", 512L)), context);

        Assert.True(second.Ok, second.Content);
        Assert.Contains("Bytes 512-1024 of 2000", second.Content, StringComparison.Ordinal);
    }

    // ---- skills_run ---------------------------------------------------------

    [Fact]
    public void Execute_Run_WithNoRunnerConfigured_RefusesUsablyInsteadOfThrowing()
    {
        // The default everywhere. The refusal has to tell the model what to do instead,
        // or it burns its remaining rounds retrying the same call.
        WriteSkill("pdf", "does pdfs");

        SkillToolResult result = SkillTools.Execute(
            Call(SkillTools.RunToolName, ("skill", "pdf"), ("path", "scripts/extract.py")), Context());

        Assert.False(result.Ok);
        Assert.Contains("disabled on this host", result.Content, StringComparison.Ordinal);
        Assert.Contains(SkillTools.ReadToolName, result.Content, StringComparison.Ordinal);
    }

    // ---- dispatch and context -----------------------------------------------

    [Fact]
    public void Execute_ANullCallOrAForeignToolName_IsAnAnswerNotAnException()
    {
        Assert.False(SkillTools.Execute(null, Context()).Ok);
        Assert.Contains("is not a tool this host answers", SkillTools.Execute(Call("get_weather"), Context()).Content,
            StringComparison.Ordinal);
    }

    [Fact]
    public void IsSkillTool_RecognisesTheThreeNamesAndNothingElse()
    {
        Assert.True(SkillTools.IsSkillTool(SkillTools.ListToolName));
        Assert.True(SkillTools.IsSkillTool(SkillTools.ReadToolName));
        Assert.True(SkillTools.IsSkillTool(SkillTools.RunToolName));
        Assert.False(SkillTools.IsSkillTool("get_weather"));
        Assert.False(SkillTools.IsSkillTool(null));
    }

    [Fact]
    public void TryResolve_IsCaseInsensitive()
    {
        // The model echoes the name back in whatever case it rendered it, and a
        // capitalised "PDF" at the start of a sentence must not become "no such skill".
        WriteSkill("pdf", "does pdfs");

        Assert.True(Context().TryResolve("PDF", out Skill? skill, out _));
        Assert.Equal("pdf", skill!.Id);
    }

    [Fact]
    public void TryResolve_AnUnknownName_ExplainsWhatIsAvailable()
    {
        WriteSkill("pdf", "does pdfs");

        Assert.False(Context().TryResolve("nope", out _, out string? error));
        Assert.Contains("Available skills: pdf", error, StringComparison.Ordinal);
    }

    // ---- skills_run argument shapes ----------------------------------------

    [Fact]
    public void ReadArgumentList_AcceptsAJsonArray()
    {
        // Observed on Gemma 4: the declaration says `args` is a string (ToolParameter
        // cannot express an array) and the model emits an array anyway, because that is
        // what an argument list looks like. When the read returned nothing the model
        // abandoned the tool and did the arithmetic in its head, producing a DIFFERENT
        // number from the script — the exact failure running a script is meant to avoid.
        var call = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new()
            {
                ["args"] = System.Text.Json.JsonDocument.Parse("[\"2400\", \"--verbose\"]").RootElement,
            },
        };

        Assert.Equal(new[] { "2400", "--verbose" }, SkillTools.ReadArgumentList(call, "args"));
    }

    [Fact]
    public void ReadArgumentList_AcceptsAShellStyleString()
    {
        var call = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new() { ["args"] = "--out \"my file.pdf\" 2400" },
        };

        Assert.Equal(new[] { "--out", "my file.pdf", "2400" }, SkillTools.ReadArgumentList(call, "args"));
    }

    [Fact]
    public void ReadArgumentList_DoesNotResplitArrayElementsThatContainSpaces()
    {
        // An array's elements are already separated. Re-splitting them on whitespace
        // would turn one filename into two arguments.
        var call = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new()
            {
                ["args"] = System.Text.Json.JsonDocument.Parse("[\"my file.pdf\"]").RootElement,
            },
        };

        Assert.Equal(new[] { "my file.pdf" }, SkillTools.ReadArgumentList(call, "args"));
    }

    [Fact]
    public void ReadArgumentList_AbsentIsNull_NotAnEmptyVector()
    {
        var call = new ToolCall { Name = SkillTools.RunToolName, Arguments = new() };
        Assert.Null(SkillTools.ReadArgumentList(call, "args"));
    }
}
