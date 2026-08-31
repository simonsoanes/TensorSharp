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
using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// The two ends of the model's side of code execution: what the DECLARATION tells it
/// before it writes anything, and what the host can make of the call that comes back.
///
/// <para>
/// This file inherits the job that <c>CodeExecGuidanceTests</c> and
/// <c>BuiltInToolRegistryTests</c> did for the five-tool <c>run_code</c> surface, and it
/// inherits their incidents with it. A declaration that says only "no network access"
/// gets read as "this sandbox cannot reach the internet", and models then abandon a
/// library they were free to install; a declaration that promises download links a
/// keep-nothing host cannot honour makes the model promise the user a file that does not
/// exist. Both halves of each of those has to be stated, so both halves are pinned here.
/// </para>
/// <para>
/// The dialect is the same problem one layer down and it is new to this design. There is
/// one shell per host, chosen at startup, and the cheat sheet in the declaration is the
/// only thing that tells the model which one it is typing into. A heredoc example handed
/// to PowerShell is not a small inaccuracy — it is a syntax error the model will repeat
/// until the round budget runs out, and on Windows the wrong guess is every model's
/// default. So the POSIX text and the PowerShell text are asserted to be right and to be
/// disjoint: neither may quote the other's examples.
/// </para>
/// <para>
/// Reading the call back is deliberately forgiving, because the shapes models actually
/// emit are not the shape declared: <c>ToolParameter</c> is flat and cannot describe an
/// array, yet Codex-trained families send <c>["bash","-lc","…"]</c> anyway, integers
/// arrive as strings, booleans as <c>"yes"</c>, and a bare <c>timeout: 60</c> means a
/// minute rather than a sixteenth of a second. Every one of those, refused, costs a round
/// to teach a shape the model already knows. The flatness itself is asserted too: a
/// parameter this repo's <c>ToolParameter</c> cannot carry is a parameter that reaches the
/// model as a malformed schema.
/// </para>
/// <para>
/// The last section is the drift guard that the <c>list_files</c> incident bought. Naming
/// a tool in a declaration, classifying it as one this host answers, and dispatching it to
/// an implementation are three separate lists, and when they disagree the failure is
/// total and silent: the call is handed to a client that implements nothing and the turn
/// ends having rendered nothing at all. The sweeps over "every declared tool" pass
/// vacuously over an empty declaration list, so the literal names are pinned as well.
/// </para>
/// </summary>
public class ShellToolDeclarationTests : IDisposable
{
    private readonly string _base;
    private readonly ShellRunner _runner;
    private readonly CodeRunnerAdapter _adapter;

    /// <summary>Types this repo's flat <see cref="ToolParameter"/> can actually describe.</summary>
    private static readonly string[] ScalarTypes = { "string", "integer", "boolean" };

    public ShellToolDeclarationTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-shell-decl-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);

        var options = new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = _base,
            AllowInstall = true,
        };
        _runner = new ShellRunner(options);
        _adapter = new CodeRunnerAdapter(_runner, options);
    }

    public void Dispose()
    {
        _runner.Dispose();
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// A <see cref="ShellProgram"/> of a chosen dialect, without needing that shell
    /// installed here.
    ///
    /// <para>
    /// <see cref="ShellProgram.TryResolve"/> accepts an absolute path to a file — that is
    /// the operator's <c>--code-exec-shell</c> path, the one that lets a Windows box name
    /// its Git Bash — and the dialect is decided from the file NAME alone. Nothing in this
    /// file executes a shell; the declarations read only <c>Kind</c> and
    /// <c>DialectName</c>. So both dialects can be asserted on every host instead of the
    /// PowerShell half silently skipping everywhere but Windows.
    /// </para>
    /// </summary>
    private ShellProgram Stub(string name)
    {
        string path = Path.Combine(_base, name);
        File.WriteAllText(path, string.Empty);

        Assert.True(ShellProgram.TryResolve(path, out ShellProgram? shell, out string? error), error);
        Assert.NotNull(shell);
        return shell!;
    }

    /// <summary>Everything the declaration puts in front of the model, prose and parameters together.</summary>
    private static string SurfaceOf(ToolFunction tool) =>
        tool.Description + "\n" + string.Join("\n", tool.Parameters.Values.Select(p => p.Description));

    private static ToolCall Shell(params (string Key, object Value)[] arguments)
    {
        var call = new ToolCall { Name = SkillToolNames.Shell };
        foreach ((string key, object value) in arguments)
            call.Arguments[key] = value;
        return call;
    }

    // ---- what the declaration promises about THIS host -----------------------

    [Fact]
    public void TheDeclaration_WithInstallsOn_SaysInstallingReachesTheRegistry_AndThatNothingElseDoes()
    {
        // Both halves, because either alone is read as the whole truth. "No network" alone
        // had models abandoning installable libraries; "installs reach the registry" alone
        // would have them writing a command to curl a URL.
        ToolFunction shell = ShellTools.DeclareShell(
            new CodeExecOptions { Enabled = true, AllowInstall = true }, Stub("bash"));

        // Installs are available…
        Assert.Contains("Installing", shell.Description, StringComparison.Ordinal);
        Assert.Contains("npm install", shell.Description, StringComparison.Ordinal);
        // …and the host is the one that performs them, which is why an option that would
        // change where a package comes from is refused.
        Assert.Contains("HOST performs the install", shell.Description, StringComparison.Ordinal);
        // …while a command still has none, stated in the same breath so neither half is
        // read as the whole truth.
        Assert.Contains("NO NETWORK", shell.Description, StringComparison.Ordinal);
    }

    [Fact]
    public void TheDeclaration_WithInstallsOff_SaysSoPlainly_AndDescribesNoInstallPhase()
    {
        // A host that cannot install must not describe a phase it does not have, or the
        // model spends a round asking for a package it will never get.
        ToolFunction shell = ShellTools.DeclareShell(
            new CodeExecOptions { Enabled = true, AllowInstall = false }, Stub("bash"));

        Assert.Contains("Installing packages is not enabled here", shell.Description, StringComparison.Ordinal);
        Assert.Contains("no command has network access", shell.Description, StringComparison.Ordinal);
        Assert.DoesNotContain("NOTHING ELSE reaches the network", shell.Description, StringComparison.Ordinal);
        Assert.DoesNotContain("downloads from the package registry", shell.Description, StringComparison.Ordinal);
    }

    [Fact]
    public void TheDeclaration_TellsTheTruthAboutKeptFiles()
    {
        var options = new CodeExecOptions { Enabled = true, AllowInstall = true };
        ShellProgram bash = Stub("bash");

        ToolFunction keeping = ShellTools.DeclareShell(options, bash, keepsArtifacts: true);
        Assert.Contains("download links", keeping.Description, StringComparison.Ordinal);
        Assert.DoesNotContain("are not handed to", keeping.Description, StringComparison.Ordinal);

        // The other way: a model told files are kept on a host that keeps none promises a
        // download nobody can honour, which is the one tool result a user cannot recover
        // from — they go looking for a file that was deleted before the answer was written.
        ToolFunction discarding = ShellTools.DeclareShell(options, bash, keepsArtifacts: false);
        Assert.Contains("are not handed to", discarding.Description, StringComparison.Ordinal);
        Assert.DoesNotContain("download links", discarding.Description, StringComparison.Ordinal);
    }

    // ---- and in the right dialect -------------------------------------------

    [Fact]
    public void APosixHost_IsGivenPosixExamples_AndNoPowerShellOnes()
    {
        ToolFunction shell = ShellTools.DeclareShell(
            new CodeExecOptions { Enabled = true }, Stub("bash"), keepsArtifacts: true);
        string surface = SurfaceOf(shell);

        Assert.Contains("Run a bash command", shell.Description, StringComparison.Ordinal);
        Assert.Contains("use a heredoc", surface, StringComparison.Ordinal);
        Assert.Contains("cat > solve.py <<'EOF'", surface, StringComparison.Ordinal);
        Assert.Contains("grep -rn pattern .", surface, StringComparison.Ordinal);
        Assert.Contains("sed -n", surface, StringComparison.Ordinal);

        // A cheat sheet in the wrong shell is worse than none: the model follows it and
        // spends the round on a syntax error rather than the task.
        Assert.DoesNotContain("Get-ChildItem", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("here-string", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("Set-Content", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("Select-String", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("PowerShell", surface, StringComparison.Ordinal);
    }

    [Fact]
    public void APowerShellHost_IsGivenPowerShellExamples_AndNoPosixOnes()
    {
        ToolFunction shell = ShellTools.DeclareShell(
            new CodeExecOptions { Enabled = true }, Stub("pwsh"), keepsArtifacts: true);
        string surface = SurfaceOf(shell);

        // The name the model is told is the DIALECT, not the binary: "pwsh" means nothing
        // to a model, and "PowerShell" is what its training associates with the syntax.
        Assert.Contains("Run a PowerShell command", shell.Description, StringComparison.Ordinal);
        Assert.Contains("use a here-string", surface, StringComparison.Ordinal);
        Assert.Contains("Set-Content -LiteralPath solve.py", surface, StringComparison.Ordinal);
        Assert.Contains("Get-ChildItem", surface, StringComparison.Ordinal);
        Assert.Contains("Select-String", surface, StringComparison.Ordinal);

        Assert.DoesNotContain("heredoc", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("grep -rn", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("sed -n", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("bash -c", surface, StringComparison.Ordinal);
    }

    [Fact]
    public void APosixHost_IsToldItsShellStateSticks_InItsOwnSpelling()
    {
        // The persistence claim is the one that changes how a model plans a task, and it
        // is spelled per dialect: `cd`/`export` against `Set-Location`/`$env:`. Told the
        // wrong pair, a model either re-does work it already did or writes a no-op.
        Assert.Contains("`cd` moves you for the next call too",
            ShellTools.DeclareShell(new CodeExecOptions(), Stub("bash")).Description, StringComparison.Ordinal);
        Assert.Contains("`Set-Location` moves you for the next call too",
            ShellTools.DeclareShell(new CodeExecOptions(), Stub("pwsh")).Description, StringComparison.Ordinal);
    }

    [Fact]
    public void AShellThatCannotBeFound_IsRefusedWithTheFlagThatFixesIt()
    {
        // Unconditional half of the host-shell question: however this machine is set up,
        // a refusal names the operator flag rather than leaving a bare "not found".
        Assert.False(ShellProgram.TryResolve(
            "no-such-shell-" + Guid.NewGuid().ToString("N"), out ShellProgram? missing, out string? why));
        Assert.Null(missing);
        Assert.Contains(CodeExecOptions.ShellFlag, why, StringComparison.Ordinal);

        // Gated half: on a host that has a shell at all, the declaration names it.
        if (!ShellProgram.TryResolve(null, out ShellProgram? host, out _) || host == null)
            return;

        ToolFunction declared = ShellTools.DeclareShell(new CodeExecOptions { Enabled = true }, host);
        Assert.Contains(host.DialectName, declared.Description, StringComparison.Ordinal);
    }

    // ---- the schema stays inside what ToolParameter can express ---------------

    /// <summary>Both tools, in both dialects and both install modes — the whole declared surface.</summary>
    private List<ToolFunction> EveryDeclaration() => new()
    {
        ShellTools.DeclareShell(
            new CodeExecOptions { Enabled = true, AllowInstall = true }, Stub("bash"), keepsArtifacts: true),
        ShellTools.DeclareShell(
            new CodeExecOptions { Enabled = true, AllowInstall = false }, Stub("pwsh"), keepsArtifacts: false),
        ShellTools.DeclarePatch(),
    };

    [Fact]
    public void EveryDeclaredParameter_IsAScalarTheFlatSchemaCanCarry()
    {
        // ToolParameter is Type/Description/Enum and nothing else — it cannot express an
        // array or a nested object. A parameter that needs one does not fail here, it
        // fails as a schema the model is shown, which is a class of bug nobody debugs from
        // a transcript. ShellCommand.ReadCommand exists precisely because `command` had to
        // stay a string and accept argv anyway.
        var offenders = new List<string>();
        foreach (ToolFunction tool in EveryDeclaration())
        {
            foreach (KeyValuePair<string, ToolParameter> parameter in tool.Parameters)
            {
                if (!ScalarTypes.Contains(parameter.Value.Type, StringComparer.Ordinal))
                    offenders.Add($"{tool.Name}.{parameter.Key} is declared '{parameter.Value.Type}'");
            }
        }

        Assert.Empty(offenders);
    }

    [Fact]
    public void EveryDeclaredParameterName_IsUnderscoreStyle_WithNoDots()
    {
        // Several chat templates splice a parameter name into markup unescaped, and some
        // parsers key on a dot. Lower-case with underscores is the one spelling every
        // template renders back the way it was written.
        foreach (ToolFunction tool in EveryDeclaration())
        {
            foreach (string name in tool.Parameters.Keys)
                Assert.Matches("^[a-z][a-z0-9_]*$", name);
        }
    }

    [Fact]
    public void EveryRequiredParameter_IsOneThatWasActuallyDeclared()
    {
        foreach (ToolFunction tool in EveryDeclaration())
        {
            foreach (string required in tool.Required)
                Assert.Contains(required, tool.Parameters.Keys);
        }
    }

    [Fact]
    public void TheDeclaredParameters_AreExactlyTheOnesTheHostReadsBack()
    {
        // Anti-vacuity for the three sweeps above, which all pass over an empty parameter
        // dictionary — and the pin that keeps declaration and reader in step, since
        // TryReadShell looks up these four names and nothing else is honoured.
        ToolFunction shell = ShellTools.DeclareShell(new CodeExecOptions { Enabled = true }, Stub("bash"));

        Assert.Equal(
            new[] { "command", "run_in_background", "timeout_ms", "workdir" },
            shell.Parameters.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());
        Assert.Equal(new[] { "command" }, shell.Required.ToArray());
        Assert.Equal("integer", shell.Parameters["timeout_ms"].Type);
        Assert.Equal("boolean", shell.Parameters["run_in_background"].Type);

        ToolFunction patch = ShellTools.DeclarePatch();
        Assert.Equal(new[] { "patch" }, patch.Parameters.Keys.ToArray());
        Assert.Equal(new[] { "patch" }, patch.Required.ToArray());

        ToolFunction read = ShellTools.DeclareRead();
        Assert.Equal(
            new[] { "limit", "offset", "path" },
            read.Parameters.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());
        Assert.Equal(new[] { "path" }, read.Required.ToArray());

        ToolFunction edit = ShellTools.DeclareEdit();
        Assert.Equal(
            new[] { "new_string", "old_string", "path", "replace_all" },
            edit.Parameters.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());
        Assert.Equal(new[] { "path", "old_string", "new_string" }, edit.Required.ToArray());
        Assert.Equal("boolean", edit.Parameters["replace_all"].Type);

        ToolFunction write = ShellTools.DeclareWrite();
        Assert.Equal(
            new[] { "content", "path" },
            write.Parameters.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());
        Assert.Equal(new[] { "path", "content" }, write.Required.ToArray());
    }

    [Fact]
    public void TheInventedSpellingsOfTheFileTools_ReachTheRightTool()
    {
        // A round spent on a spelling teaches nothing and fixes nothing. The str_replace
        // family is not a guess — those are the command names of Anthropic's published
        // str_replace_based_edit_tool, so a model trained near it reaches for them.
        Assert.Equal(SkillToolNames.ReadFile, SkillToolNames.ResolveFileTool("read"));
        Assert.Equal(SkillToolNames.ReadFile, SkillToolNames.ResolveFileTool("view_file"));
        Assert.Equal(SkillToolNames.EditFile, SkillToolNames.ResolveFileTool("str_replace"));
        Assert.Equal(SkillToolNames.EditFile, SkillToolNames.ResolveFileTool("str_replace_based_edit_tool"));
        Assert.Equal(SkillToolNames.EditFile, SkillToolNames.ResolveFileTool("edit"));
        Assert.Equal(SkillToolNames.WriteFile, SkillToolNames.ResolveFileTool("create_file"));
    }

    [Fact]
    public void TheFileToolAliases_DoNotStealTheNamesModelsUseForLookingAtAnImage()
    {
        // 'view', 'open', 'display', 'show', 'read_image' and 'look' belong to
        // SkillTools.LooksLikeAnImageTool, which answers them with the one message that
        // explains this host cannot show a picture and what to do instead. Capturing one
        // here would swap that answer for a file-not-found, and the model would keep
        // trying — the incident that message was written for cost four turns and ~27k
        // tokens.
        foreach (string claimed in new[]
        {
            "view", "open", "display", "show", "read_image", "look", "look_at", "see",
            "screenshot", "render", "preview", "image", "open_image", "show_image", "inspect_image",
        })
        {
            Assert.Null(SkillToolNames.ResolveFileTool(claimed));
            Assert.False(SkillToolNames.IsCodeTool(claimed), claimed + " was captured by the code tools");
        }
    }

    [Fact]
    public void TheFileToolAliases_DoNotStealTheSkillToolNames()
    {
        foreach (string skillTool in new[] { "skills_list", "skills_read", "skills_run" })
            Assert.Null(SkillToolNames.ResolveFileTool(skillTool));
    }

    // ---- reading back the call a model actually sent -------------------------

    [Fact]
    public void APlainStringCommand_IsTheCommand()
    {
        Assert.True(ShellTools.TryReadShell(Shell(("command", "ls -la")), out ShellRequest request, out string? error), error);

        Assert.Equal("ls -la", request.Command);
        Assert.Null(request.WorkDirectory);
        Assert.Null(request.Timeout);
        Assert.False(request.Background);
    }

    [Fact]
    public void AnArgvVectorWrappingAShell_YieldsTheScriptInsideIt()
    {
        // ["bash","-lc","<script>"] is what Codex-trained families emit whatever the
        // schema says. Re-quoting the vector would run a shell inside the shell and double
        // every layer of escaping the model already got right. Cloned so the element does
        // not depend on the JsonDocument staying alive.
        JsonElement argv = JsonDocument.Parse("[\"bash\",\"-lc\",\"echo hi && ls\"]").RootElement.Clone();

        Assert.True(ShellTools.TryReadShell(Shell(("command", argv)), out ShellRequest request, out string? error), error);
        Assert.Equal("echo hi && ls", request.Command);
    }

    [Fact]
    public void AJsonArrayThatArrivedAsAString_IsJoinedIntoACommandLine()
    {
        // The array is not always a JSON value: plenty of parsers hand the argument over
        // as the literal text the model typed.
        Assert.True(
            ShellTools.TryReadShell(Shell(("command", "[\"grep\", \"-rn\", \"needle\", \".\"]")),
                out ShellRequest request, out string? error),
            error);

        Assert.Equal("grep -rn needle .", request.Command);
    }

    [Fact]
    public void APlainListOfWords_IsJoinedIntoACommandLine()
    {
        Assert.True(
            ShellTools.TryReadShell(Shell(("command", new List<object> { "echo", "hi" })),
                out ShellRequest request, out string? error),
            error);

        Assert.Equal("echo hi", request.Command);
    }

    [Fact]
    public void ATimeoutSentAsAString_IsStillANumber()
    {
        // Every family that renders arguments as XML-ish markup sends integers as text.
        Assert.True(ShellTools.TryReadShell(Shell(("command", "sleep 1"), ("timeout_ms", "5000")),
            out ShellRequest request, out string? error), error);

        Assert.NotNull(request.Timeout);
        Assert.Equal(TimeSpan.FromSeconds(5), request.Timeout!.Value);
    }

    [Fact]
    public void ABareTimeoutOf60_IsReadAsSixtySeconds_NotSixtyMilliseconds()
    {
        // A model that writes `timeout: 60` means a minute. Sixty milliseconds is not a
        // deadline anyone asks for, and honouring it literally kills every command
        // instantly — which reads, in the transcript, as a host that is simply broken.
        Assert.True(ShellTools.TryReadShell(Shell(("command", "make build"), ("timeout", 60)),
            out ShellRequest request, out string? error), error);

        Assert.NotNull(request.Timeout);
        Assert.Equal(TimeSpan.FromSeconds(60), request.Timeout!.Value);
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData("true", true)]
    [InlineData("True", true)]
    [InlineData("yes", true)]
    [InlineData(1, true)]
    [InlineData("1", true)]
    [InlineData("false", false)]
    [InlineData(0, false)]
    public void RunInBackground_IsReadFromEveryShapeAModelSendsABooleanIn(object raw, bool expected)
    {
        Assert.True(ShellTools.TryReadShell(Shell(("command", "npm start"), ("run_in_background", raw)),
            out ShellRequest request, out string? error), error);

        Assert.Equal(expected, request.Background);
    }

    [Fact]
    public void AWorkdirUnderAnyOfItsUsualNames_IsRead()
    {
        Assert.True(ShellTools.TryReadShell(Shell(("command", "ls"), ("cwd", "src")),
            out ShellRequest request, out string? error), error);

        Assert.Equal("src", request.WorkDirectory);
    }

    [Fact]
    public void AMissingCommand_IsRefusedWithTheArgumentsName()
    {
        // The refusal is the model's only feedback, so it names the argument and shows
        // one. "Invalid arguments" costs a round and teaches nothing.
        Assert.False(ShellTools.TryReadShell(Shell(("workdir", "src")), out ShellRequest request, out string? error));

        Assert.NotNull(error);
        Assert.Contains("'command'", error, StringComparison.Ordinal);
        Assert.Contains("required", error, StringComparison.Ordinal);
        // Nothing half-read comes back with the refusal.
        Assert.Null(request.Command);
    }

    [Fact]
    public void AWhitespaceOnlyCommand_CountsAsMissing()
    {
        Assert.False(ShellTools.TryReadShell(Shell(("command", "   ")), out _, out string? error));
        Assert.Contains("'command'", error, StringComparison.Ordinal);
    }

    // ---- finding the patch envelope -----------------------------------------

    private const string Envelope =
        "*** Begin Patch\n*** Add File: notes.md\n+hello\n*** End Patch";

    [Theory]
    [InlineData("patch")]
    [InlineData("input")]
    [InlineData("diff")]
    [InlineData("content")]
    [InlineData("text")]
    public void ThePatchEnvelope_IsFoundUnderEveryNameModelsGiveIt(string key)
    {
        var call = new ToolCall { Name = SkillToolNames.ApplyPatch };
        call.Arguments[key] = Envelope;

        Assert.True(ShellTools.TryReadPatch(call, out string patch, out string? error), error);
        Assert.Equal(Envelope, patch);
    }

    [Fact]
    public void ThePatchEnvelope_IsFoundAsTheOnlyValueThatCouldBeOne()
    {
        // Some families send the envelope under the tool's own name, and some under no
        // sensible key at all. A string containing '*** Begin Patch' is unambiguous, so
        // accepting it costs nothing and refusing it costs a round.
        var call = new ToolCall { Name = SkillToolNames.ApplyPatch };
        call.Arguments["apply_patch"] = Envelope;

        Assert.True(ShellTools.TryReadPatch(call, out string patch, out string? error), error);
        Assert.Equal(Envelope, patch);
    }

    [Fact]
    public void AnArgumentThatIsNotAnEnvelope_IsNotMistakenForOne()
    {
        // The scan must key on the marker, not on "the only string there is" — otherwise a
        // model's stray note becomes the patch and the failure is a parse error about a
        // file it never mentioned.
        var call = new ToolCall { Name = SkillToolNames.ApplyPatch };
        call.Arguments["explanation"] = "I am about to fix the import in main.py";

        Assert.False(ShellTools.TryReadPatch(call, out string patch, out string? error));
        Assert.Equal(string.Empty, patch);
        Assert.Contains("'patch'", error, StringComparison.Ordinal);
        Assert.Contains("*** Begin Patch", error, StringComparison.Ordinal);
    }

    [Fact]
    public void ACallWithNoArgumentsAtAll_IsRefusedRatherThanThrowing()
    {
        // The shape a model emits when it writes the function tag with no parameter
        // children: Arguments is empty, and on some paths the call itself is bare.
        Assert.False(ShellTools.TryReadPatch(new ToolCall { Name = SkillToolNames.ApplyPatch }, out _, out string? error));
        Assert.Contains("'patch'", error, StringComparison.Ordinal);
    }

    // ---- the drift guard: declared == classified == dispatched ---------------

    [Fact]
    public void TheDeclaredSurface_IsReal_SoTheSweepsCannotPassVacuously()
    {
        // Every "for each declared tool" assertion below passes trivially over an empty
        // list, and the set-equality guard passes when BOTH sides are empty — which is
        // exactly what happens when the runner resolves no shell and DeclareTools returns
        // nothing. Pin the literal names so that regression fails loudly here instead of
        // turning the rest of this file green.
        Assert.Equal(
            new[] { "apply_patch", "edit_file", "read_file", "shell", "write_file" },
            _adapter.DeclareTools().Select(t => t.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray());
    }

    [Fact]
    public void TheDeclarationsComeInTheOrderTheModelShouldReadThem()
    {
        // ORDER, not just membership — and this assertion is the point of the test, not a
        // refinement of the one above it. Every other guard in this file compares SETS, so
        // all of them stayed green when three tools were inserted in front of the shell:
        // the two places that reach for "the shell" as declarations[0] silently started
        // reaching for read_file, one of them to append the conversation's attachment list
        // to its description. Nothing about that fails to compile and nothing about it
        // shows up anywhere except in what the model was told.
        //
        // The order is also a claim about behaviour. A declaration list is read top-down,
        // and the measured failure is a model reaching for the shell to do a job the
        // editor does better, so the editor comes first and the multi-file specialist
        // comes last.
        Assert.Equal(
            new[] { "read_file", "edit_file", "write_file", "shell", "apply_patch" },
            _adapter.DeclareTools().Select(t => t.Name).ToArray());
    }

    [Fact]
    public void TheSingleDeclaration_IsStillTheShell()
    {
        // ICodeRunner.Declare() means "the tool" for a host that only wants one, and that
        // has always meant the shell. It was written as DeclareTools()[0].
        Assert.Equal("shell", _adapter.Declare().Name);
    }

    [Fact]
    public void WithoutAPersistentWorkspace_OnlyTheShellIsDeclared()
    {
        // The stateless endpoints get a fresh empty directory per call, so a file tool
        // would offer capability the host cannot honour: nothing is there to read, and
        // nothing written survives the call. This branch previously had no test at all.
        Assert.Equal(
            new[] { "shell" },
            _adapter.DeclareTools(persists: false).Select(t => t.Name).ToArray());
    }

    [Fact]
    public void TheDeclaredToolNames_AreExactlyTheDispatchTablesCodeTools()
    {
        // One list, checked against the declarations rather than trusted. Adding a third
        // code tool is then one edit in SkillToolNames plus its declaration; forgetting
        // either half fails here instead of in a user's chat.
        Assert.Equal(
            SkillToolNames.CodeTools.OrderBy(n => n, StringComparer.Ordinal).ToArray(),
            _adapter.DeclareTools().Select(t => t.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray());
    }

    [Fact]
    public void EveryToolThisHostDeclares_IsClassifiedAsOneItAnswers()
    {
        // The assertion that was missing when apply_patch and list_files were declared and
        // then classified as the CLIENT's: forwarded to a Web UI that implements no tools,
        // the turn ended with nothing rendered at all.
        foreach (ToolFunction tool in _adapter.DeclareTools())
        {
            Assert.True(
                SkillTools.IsBuiltInTool(tool.Name),
                $"'{tool.Name}' is declared to the model but IsBuiltInTool says it is the caller's. "
                + "It will be forwarded to a client that has no implementation for it, and the turn "
                + "will end with nothing rendered.");
        }
    }

    [Fact]
    public void EveryToolThisHostDeclares_ReachesAnImplementation()
    {
        // Classification is only half of it: the dispatch switch was missing the same
        // names, so even a correctly classified call came back refused. Called with no
        // arguments, which reaches the argument readers and stops there — nothing is
        // executed, so this stays hermetic on any host.
        var context = new SkillToolContext(Array.Empty<Skill>()) { CodeRunner = _adapter };

        foreach (ToolFunction tool in _adapter.DeclareTools())
        {
            SkillToolResult result = SkillTools.Execute(
                new ToolCall { Name = tool.Name, Arguments = new Dictionary<string, object>() }, context);

            Assert.DoesNotContain(
                "is not a tool this host answers", result.Content ?? string.Empty, StringComparison.Ordinal);
        }
    }

    [Theory]
    [InlineData("apply-patch")]
    [InlineData("applypatch")]
    [InlineData("apply_patch_tool")]
    public void AnInventedSpellingOfApplyPatch_ReachesThePatcher(string name)
    {
        // Codex's own prompt has to warn against these spellings because models write them
        // anyway. Refusing the call teaches nothing and fixes nothing; the error naming
        // 'patch' rather than 'command' is what proves it landed on the patcher and not
        // the shell.
        Assert.True(SkillTools.IsBuiltInTool(name));

        SkillToolResult result = SkillTools.Execute(
            new ToolCall { Name = name, Arguments = new Dictionary<string, object>() },
            new SkillToolContext(Array.Empty<Skill>()) { CodeRunner = _adapter });

        Assert.False(result.Ok);
        Assert.DoesNotContain("is not a tool this host answers", result.Content ?? string.Empty, StringComparison.Ordinal);
        if (_runner.CanRun)
            Assert.Contains("'patch'", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUndeclaredName_IsStillRefused()
    {
        // The guard above must not have been satisfied by classifying everything built-in.
        Assert.False(SkillTools.IsBuiltInTool("get_weather"));
        Assert.False(SkillTools.IsBuiltInTool(null));
        Assert.False(SkillTools.IsBuiltInTool("shell "));    // ordinal, on purpose
        Assert.False(SkillTools.IsBuiltInTool("SHELL"));
    }

    [Fact]
    public void TheToolNameConstants_AgreeWithTheDispatchTable()
    {
        // ShellTools' static constructor throws when they drift, but a `const` is inlined
        // by the compiler and reading one does not run that constructor — so run it
        // explicitly rather than assuming some other test happened to touch the type
        // first. Then assert the same equality it checks, so a failure says WHICH name
        // disagreed instead of only that a type initializer threw.
        RuntimeHelpers.RunClassConstructor(typeof(ShellTools).TypeHandle);

        Assert.Equal(SkillToolNames.Shell, ShellTools.ShellToolName);
        Assert.Equal(SkillToolNames.ApplyPatch, ShellTools.PatchToolName);
        Assert.Equal(ShellTools.ShellToolName, ShellTools.DeclareShell(new CodeExecOptions(), Stub("bash")).Name);
        Assert.Equal(ShellTools.PatchToolName, ShellTools.DeclarePatch().Name);
    }
}
