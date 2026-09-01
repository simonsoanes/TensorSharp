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
using System.IO;
using System.Linq;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// A second sweep for the defect this codebase calls cardinal: the host reporting success
/// for something it did not do, or stating a constraint that is not the real one.
///
/// <para>
/// Every case here was found by an adversarial audit that VERIFIED each claim by running
/// it, and several were in code added the same day. They share the property that makes
/// them expensive: nothing in the result contradicts the false statement, so the model
/// cannot recover unaided — and the patch tool's own declaration tells it not to look.
/// </para>
/// </summary>
public class ResultTruthfulnessTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public ResultTruthfulnessTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-truth-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private ShellRunner Runner(Action<CodeExecOptions>? tweak = null)
    {
        var options = new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        };
        tweak?.Invoke(options);
        return new ShellRunner(options);
    }

    private static bool HavePosix =>
        ShellProgram.TryResolve(null, out ShellProgram? shell, out _) && shell is { Kind: ShellKind.Posix };

    // ---- the patch tool ------------------------------------------------------

    /// <summary>
    /// The JSON operation form carries the file under <c>content</c>, and reading only
    /// <c>diff</c> threw it away: an empty body is fine to the matcher, so a ZERO-BYTE file
    /// came back reported as "added a.py", with no counts printed for an Add to contradict
    /// it. The envelope parser already refuses a body-less Add; this path did not.
    /// </summary>
    [Fact]
    public void TheJsonPatchFormDoesNotDiscardTheFileItWasGiven()
    {
        Assert.True(CodePatch.TryParse(
            """{"type":"create_file","path":"a.py","content":"print(1)\nprint(2)\n"}""",
            out var sections, out string? error), error);

        Assert.Single(sections);
        Assert.Equal(CodePatch.FileOp.Add, sections[0].Op);
        Assert.Equal(2, sections[0].Body.Count);
        Assert.All(sections[0].Body, line => Assert.StartsWith("+", line, StringComparison.Ordinal));
    }

    [Fact]
    public void AJsonPatchWithNoContentAtAllIsRefusedRatherThanWritingNothing()
    {
        Assert.False(CodePatch.TryParse(
            """{"type":"create_file","path":"a.py"}""", out _, out string? error));
        Assert.Contains("no content", error!, StringComparison.Ordinal);
    }

    /// <summary>
    /// End to end: the JSON form must produce the file, not a zero-byte one reported as
    /// added.
    /// </summary>
    [Fact]
    public void TheJsonPatchFormActuallyWritesTheFile()
    {
        using ShellRunner runner = Runner();
        SessionWorkspace workspace = _workspaces.GetOrCreate("json");

        CodeExecResult result = runner.ApplyPatch(
            """{"type":"create_file","path":"a.py","content":"print(1)\nprint(2)\n"}""", workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Equal(
            "print(1)\nprint(2)\n",
            File.ReadAllText(Path.Combine(workspace.WorkDirectory, "a.py")).Replace("\r\n", "\n"));
    }

    // ---- installs ------------------------------------------------------------

    /// <summary>
    /// <c>Install</c> returns null for "no error", which is NOT the same as "installed".
    /// Conflating them made the auto-install loop report "installed and the command was run
    /// again" when nothing had been installed, and then coach the model to go looking for
    /// the package's real distribution name — a hunt for something already on disk.
    /// </summary>
    [Fact]
    public void AnInstallThatWasAlreadySatisfiedReportsThatItDidNothing()
    {
        var options = new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = _base,
            AllowInstall = true,
        };
        SessionWorkspace workspace = _workspaces.GetOrCreate("ledger");
        var installer = new PackageInstaller(options, sandbox: null);

        workspace.MarkInstalled("python", new[] { "python-pptx" });

        string? error = installer.Install(
            workspace, CodeLanguage.Python, new[] { "python-pptx" }, null, out bool performed);

        Assert.Null(error);          // no error...
        Assert.False(performed);     // ...and nothing installed. Two different facts.
    }

    /// <summary>
    /// <c>pip install a &amp;&amp; pip install b</c> was performed right-to-left so the
    /// spans stayed valid, which meant a failure in <c>b</c> marked <c>a</c> as skipped —
    /// and <c>a</c> was never attempted at all, silently. The model read one error about
    /// <c>b</c> and concluded from <c>&amp;&amp;</c> that <c>a</c> had run.
    /// </summary>
    [Fact]
    public void EveryInstallSkippedBecauseAnEarlierOneFailedIsNamed()
    {
        if (!HavePosix) return;

        using ShellRunner runner = Runner(o =>
        {
            o.AllowInstall = true;
            o.AllowedPackages = new[] { "an-allowed-package" };   // both installs are refused by name
        });

        CodeExecResult result = runner.Run(
            new ShellRequest("pip install alpha-pkg && pip install beta-pkg"),
            _workspaces.GetOrCreate("chain"));

        Assert.False(result.Ok);
        // The FIRST one in source order is the one that gets the real refusal...
        Assert.Contains("alpha-pkg", result.Content, StringComparison.Ordinal);
        // ...and the second must be named as not attempted, rather than passed over.
        Assert.Contains("beta-pkg", result.Content, StringComparison.Ordinal);
        Assert.Contains("NOT installed", result.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// A refusal must end in the one thing to do next. "This host cannot install that" on
    /// its own makes a model abandon the task; naming the alternative lets it finish.
    /// </summary>
    [Fact]
    public void ARefusedInstallerNamesWhatToDoInstead()
    {
        if (!HavePosix) return;

        using ShellRunner runner = Runner(o => o.AllowInstall = true);

        CodeExecResult npx = runner.Run(
            new ShellRequest("npx tsc --noEmit"), _workspaces.GetOrCreate("npx"));
        Assert.False(npx.Ok);
        Assert.Contains("node_modules/.bin", npx.Content, StringComparison.Ordinal);

        CodeExecResult apt = runner.Run(
            new ShellRequest("apt install ffmpeg"), _workspaces.GetOrCreate("apt"));
        Assert.False(apt.Ok);
        Assert.Contains("say so in your answer", apt.Content, StringComparison.Ordinal);
    }

    // ---- the shell -----------------------------------------------------------

    /// <summary>
    /// A trailing <c>&amp;</c> could not do what the model meant, and until the output
    /// drain was bounded the call did not return at all: the shell exited immediately while
    /// the grandchild held the pipe open, so the loop kept emitting heartbeats forever.
    /// </summary>
    [Fact]
    public void ATrailingAmpersandIsRefusedWithTheParameterThatWorks()
    {
        Assert.True(ShellCommand.EndsWithBackground("python3 server.py &"));
        Assert.True(ShellCommand.EndsWithBackground("nohup python3 s.py > log 2>&1 &"));
        Assert.False(ShellCommand.EndsWithBackground("a && b"));
        Assert.False(ShellCommand.EndsWithBackground("python3 x.py"));

        if (!HavePosix) return;

        using ShellRunner runner = Runner();
        CodeExecResult result = runner.Run(
            new ShellRequest("sleep 30 &"), _workspaces.GetOrCreate("amp"));

        Assert.False(result.Ok);
        Assert.Contains("run_in_background", result.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// <c>timeout_ms: 300</c> became a 300-SECOND deadline while <c>timeout_ms: 1000</c>
    /// stayed one second — two adjacent values a model might reasonably write, differing by
    /// 999x, on the parameter whose own description says milliseconds.
    /// </summary>
    [Fact]
    public void TimeoutMsIsReadAsMilliseconds()
    {
        Assert.True(ShellTools.TryReadShell(
            new ToolCall { Name = "shell", Arguments = Args(("command", "true"), ("timeout_ms", 300)) },
            out ShellRequest ms, out _));
        Assert.Equal(TimeSpan.FromMilliseconds(300), ms.Timeout);

        // The ambiguous spelling keeps the seconds reading, which is what it is for: a
        // model writing `timeout: 60` means a minute, and 60 ms is not a deadline anyone
        // asks for.
        Assert.True(ShellTools.TryReadShell(
            new ToolCall { Name = "shell", Arguments = Args(("command", "true"), ("timeout", 60)) },
            out ShellRequest bare, out _));
        Assert.Equal(TimeSpan.FromSeconds(60), bare.Timeout);
    }

    private static System.Collections.Generic.Dictionary<string, object> Args(
        params (string Key, object Value)[] pairs)
    {
        var arguments = new System.Collections.Generic.Dictionary<string, object>(StringComparer.Ordinal);
        foreach ((string key, object value) in pairs)
            arguments[key] = value;
        return arguments;
    }

    /// <summary>
    /// The declared ceiling has to be the enforced one. With <c>--code-exec-timeout 900</c>
    /// the declaration told the model the maximum was 600000 ms on a host that would have
    /// honoured 900000.
    /// </summary>
    [Fact]
    public void TheDeclaredTimeoutCeilingIsTheEnforcedOne()
    {
        var options = new CodeExecOptions { Enabled = true, Timeout = TimeSpan.FromSeconds(900) };
        if (!ShellProgram.TryResolve(null, out ShellProgram? shell, out _) || shell == null)
            return;

        ToolFunction declaration = ShellTools.DeclareShell(options, shell);
        string ceiling = ((int)options.EffectiveMaxTimeout.TotalMilliseconds).ToString();

        Assert.Contains(
            "maximum " + ceiling,
            declaration.Parameters["timeout_ms"].Description!,
            StringComparison.Ordinal);
    }

    /// <summary>
    /// A nested artifact's link was spelled with <c>%2F</c> while the endpoint that serves
    /// it spells the separator raw — and the result tells the model to copy these links
    /// verbatim, so whichever was wrong is what the user clicked.
    /// </summary>
    [Fact]
    public void ANestedArtifactLinkEscapesTheSegmentsAndNotTheSeparator()
    {
        if (!HavePosix) return;

        using ShellRunner runner = new ShellRunner(
            new CodeExecOptions
            {
                Enabled = true,
                Sandbox = SkillSandboxMode.Off,
                ScratchDirectory = _base,
                Timeout = TimeSpan.FromSeconds(30),
                ArtifactUriPrefix = "/api/code/artifacts",
            },
            artifacts: new CodeArtifactStore(Path.Combine(_base, "art")));

        CodeExecResult result = runner.Run(
            new ShellRequest("mkdir -p out && printf 'x' > out/report.txt"),
            _workspaces.GetOrCreate("nested"));

        Assert.True(result.Ok, result.Content);
        if (result.Artifacts.Count == 0)
            return;   // artifact capture is off on this host

        Assert.DoesNotContain("%2F", result.Content);
        Assert.Contains("out/report.txt", result.Content, StringComparison.Ordinal);
    }

    // ---- confinement ---------------------------------------------------------

    /// <summary>
    /// The confinement notice has to describe the sandbox that RAN, not the one this host
    /// is capable of.
    ///
    /// <para>
    /// Under <c>--code-exec-unconfined</c> the process starts raw while the sandbox object
    /// stays in place, so reading capabilities off the object reported an empty gap list —
    /// and the "Not confined on this host" line was omitted ENTIRELY. The model was told,
    /// by silence, that its command had been confined. Whether a command could have reached
    /// the network is the one fact it must be able to trust here.
    /// </para>
    /// </summary>
    /// <summary>
    /// The invariant, stated so it holds on every host: <b>a run that reports
    /// <c>sandbox: none</c> must carry the not-confined warning.</b>
    ///
    /// <para>
    /// The bug was that the warning came from the sandbox OBJECT's declared capabilities
    /// rather than from what ran. When a sandbox is present but its preparation fails, and
    /// the mode is not Required, the process starts raw with <c>sandboxName = "none"</c>
    /// while the object stays in place reporting no gaps — so the warning was omitted
    /// entirely and the model was told by silence that its command had been confined.
    /// </para>
    /// <para>
    /// Forcing a preparation failure needs a sandbox that cannot be prepared, which no
    /// portable test can arrange; asserting the one-way invariant instead covers the path
    /// on whichever host does hit it. A confined run may still report a narrower missing
    /// capability, so the reverse implication would be false.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(SkillSandboxMode.Off)]
    [InlineData(SkillSandboxMode.Preferred)]
    public void SayingSandboxNoneAlwaysReportsTheConfinementGaps(SkillSandboxMode mode)
    {
        if (!HavePosix) return;

        using ShellRunner runner = Runner(o =>
        {
            o.Sandbox = mode;
            o.Unconfined = mode == SkillSandboxMode.Off;
        });

        CodeExecResult result = runner.Run(
            new ShellRequest("echo hi"), _workspaces.GetOrCreate("confinement" + mode));

        Assert.True(result.Ok, result.Content);

        bool ranRaw = result.Content.Contains("sandbox: none", StringComparison.Ordinal);
        bool warned = result.Content.Contains("Not confined on this host", StringComparison.Ordinal);

        // "Ran raw and said nothing" is the defect. The reverse is intentionally not an
        // invariant: a real sandbox may enforce files/network while honestly reporting a
        // separate missing axis (Seatbelt cannot contain a setsid descendant on macOS).
        Assert.False(ranRaw && !warned);
    }

    /// <summary>
    /// A run that really had network confinement must not report that network was open.
    /// It may still report an independent platform gap such as process lifetime.
    /// </summary>
    [Fact]
    public void AConfinedRunDoesNotClaimToBeUnconfined()
    {
        if (!HavePosix) return;

        using ShellRunner runner = Runner(o => o.Sandbox = SkillSandboxMode.Preferred);
        CodeExecResult result = runner.Run(
            new ShellRequest("echo hi"), _workspaces.GetOrCreate("confined"));

        Assert.True(result.Ok, result.Content);
        if (result.Content.Contains("sandbox: none", StringComparison.Ordinal))
            return;   // this host has no sandbox; the warning is correct and expected

        Assert.DoesNotContain("may reach the network", result.Content);
    }

    [Fact]
    public void ANetworkEnabledRunSaysThatTheOperatorOpenedIt()
    {
        if (!HavePosix) return;

        using ShellRunner runner = Runner(o =>
        {
            o.AllowNetwork = true;
            o.Sandbox = SkillSandboxMode.Off;
            o.Unconfined = true;
        });
        CodeExecResult result = runner.Run(
            new ShellRequest("echo hi"), _workspaces.GetOrCreate("network-open"));

        Assert.True(result.Ok, result.Content);
        Assert.Contains("may reach the network", result.Content, StringComparison.Ordinal);
        Assert.Contains(CodeExecOptions.AllowNetworkFlag, result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("Commands here have no network", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnconfinedDeclarationDoesNotPromiseNetworkDenial()
    {
        using ShellRunner runner = Runner(o =>
        {
            o.AllowNetwork = false;
            o.Sandbox = SkillSandboxMode.Off;
            o.Unconfined = true;
        });
        if (runner.Shell == null) return;

        var adapter = new CodeRunnerAdapter(runner);
        ToolFunction declaration = adapter.DeclareTools()
            .Single(d => string.Equals(d.Name, ShellTools.ShellToolName, StringComparison.Ordinal));

        Assert.Contains("Network confinement: NOT GUARANTEED", declaration.Description,
            StringComparison.Ordinal);
        Assert.DoesNotContain("Internet/IP network access: BLOCKED", declaration.Description,
            StringComparison.Ordinal);
    }

    /// <summary>
    /// A sandbox that cannot ATTACH must fail where confinement was demanded and DEGRADE
    /// where the operator already accepted none — not die in both.
    ///
    /// <para>
    /// Making it fatal in every mode was a regression: the branch is Windows-only (only
    /// the job-object sandbox overrides <c>TryAttach</c>), and on Windows commands run at
    /// all only under <c>--code-exec-unconfined</c>. <c>AssignProcessToJobObject</c> fails
    /// when the parent is itself already in a job without breakaway — a CI agent, a
    /// Windows container — so refusing every command there takes away a capability the
    /// operator explicitly asked for, over a CPU bound. Degrading is only honest because
    /// the run then reports <c>sandbox: none</c>, which is what prints the full gap list.
    /// </para>
    /// </summary>
    [Fact]
    public void DegradingToNoConfinementStillReportsItself()
    {
        if (!HavePosix) return;

        // The invariant that holds on every platform, asserted through the real runner:
        // whatever the sandbox did or did not manage, "sandbox: none" implies the gap
        // list. A degraded attach reaches exactly this state.
        using ShellRunner runner = Runner(o =>
        {
            o.Sandbox = SkillSandboxMode.Preferred;
            o.Unconfined = true;
        });

        CodeExecResult result = runner.Run(
            new ShellRequest("echo hi"), _workspaces.GetOrCreate("degrade"));

        Assert.True(result.Ok, result.Content);
        bool ranRaw = result.Content.Contains("sandbox: none", StringComparison.Ordinal);
        bool warned = result.Content.Contains("Not confined on this host", StringComparison.Ordinal);
        Assert.False(ranRaw && !warned);
    }

    // ---- the environment the model actually writes for ----------------------

    /// <summary>
    /// <c>NODE_PATH</c> is consulted by CommonJS resolution only, so a program written with
    /// <c>import</c> could not see a package the host had just installed for it — and the
    /// error it got, "Cannot find package", is indistinguishable from not being installed.
    /// Verified on this host before the fix: <c>require</c> resolved, <c>import</c> failed
    /// with ERR_MODULE_NOT_FOUND. The fix is the mechanism Node itself uses — a
    /// <c>node_modules</c> in the directory the program is in.
    /// </summary>
    [Fact]
    public void BothImportAndRequireSeeTheSessionsInstalledPackages()
    {
        if (!HavePosix
            || !CodeEnvironment.TryResolveInterpreter(CodeLanguage.JavaScript, out string? node, out _)
            || node == null)
        {
            return;
        }

        using ShellRunner runner = Runner();
        SessionWorkspace workspace = _workspaces.GetOrCreate("esm");

        // Stand in for an installed package, in the exact place the host installs into.
        string installed = Path.Combine(workspace.EnvDirectory, "node_modules", "ts-probe-pkg");
        Directory.CreateDirectory(installed);
        File.WriteAllText(
            Path.Combine(installed, "package.json"),
            """{"name":"ts-probe-pkg","version":"1.0.0","main":"index.js"}""");
        File.WriteAllText(Path.Combine(installed, "index.js"), "module.exports = 42;\n");

        CodeExecResult required = runner.Run(
            new ShellRequest(
                "cat > a.cjs <<'EOF'\nconsole.log('cjs', require('ts-probe-pkg'));\nEOF\nnode a.cjs"),
            workspace);
        Assert.True(required.Ok, required.Content);
        Assert.Contains("cjs 42", required.Content, StringComparison.Ordinal);

        // The half that was broken.
        CodeExecResult imported = runner.Run(
            new ShellRequest(
                "cat > b.mjs <<'EOF'\nimport v from 'ts-probe-pkg';\nconsole.log('esm', v);\nEOF\nnode b.mjs"),
            workspace);
        Assert.True(imported.Ok, imported.Content);
        Assert.Contains("esm 42", imported.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// The link must be invisible to everything that reports to the user: artifact capture
    /// skips reparse points, and the workspace walk prunes node_modules by name. A
    /// thousand-file dependency tree offered as "files this command produced" would be the
    /// defect this repo already fixed once.
    /// </summary>
    [Fact]
    public void TheNodeModulesLinkIsNotOfferedAsAProducedFile()
    {
        if (!HavePosix
            || !CodeEnvironment.TryResolveInterpreter(CodeLanguage.JavaScript, out _, out _))
        {
            return;
        }

        using var runner = new ShellRunner(
            new CodeExecOptions
            {
                Enabled = true,
                Sandbox = SkillSandboxMode.Off,
                ScratchDirectory = _base,
                Timeout = TimeSpan.FromSeconds(30),
                ArtifactUriPrefix = "/api/code/artifacts",
            },
            artifacts: new CodeArtifactStore(Path.Combine(_base, "art-esm")));

        SessionWorkspace workspace = _workspaces.GetOrCreate("esmartifacts");
        string installed = Path.Combine(workspace.EnvDirectory, "node_modules", "ts-probe-pkg");
        Directory.CreateDirectory(installed);
        File.WriteAllText(Path.Combine(installed, "index.js"), "module.exports = 1;\n");

        CodeExecResult result = runner.Run(new ShellRequest("printf 'x' > out.txt"), workspace);

        Assert.True(result.Ok, result.Content);
        Assert.DoesNotContain("node_modules", result.Content);
        Assert.All(result.Artifacts, a => Assert.DoesNotContain("node_modules", a.Path));
    }

    // ---- the declaration -----------------------------------------------------

    /// <summary>
    /// The stateless endpoints patched the finished description, swapping eight words out
    /// of a paragraph whose other sixty went on asserting that files persist — ending with
    /// "Do not re-create what you already made", which is exactly what stops a model
    /// re-creating a file that is in fact gone.
    /// </summary>
    [Fact]
    public void ANonPersistingDeclarationDoesNotAlsoPromisePersistence()
    {
        if (!ShellProgram.TryResolve(null, out ShellProgram? shell, out _) || shell == null)
            return;

        string description = ShellTools
            .DeclareShell(new CodeExecOptions { Enabled = true }, shell, keepsArtifacts: false, persists: false)
            .Description!;

        Assert.Contains("NOTHING persists between calls", description, StringComparison.Ordinal);
        foreach (string contradiction in new[]
        {
            "files written earlier are still there",
            "stay installed",
            "already made",
            "PERSISTS for this whole conversation",
        })
        {
            Assert.DoesNotContain(contradiction, description);
        }
    }

    [Fact]
    public void APersistingDeclarationStillSaysSo()
    {
        if (!ShellProgram.TryResolve(null, out ShellProgram? shell, out _) || shell == null)
            return;

        string description = ShellTools
            .DeclareShell(new CodeExecOptions { Enabled = true }, shell, keepsArtifacts: false, persists: true)
            .Description!;

        Assert.Contains("PERSISTS for this whole conversation", description, StringComparison.Ordinal);
        Assert.DoesNotContain("NOTHING persists", description);
    }
}
