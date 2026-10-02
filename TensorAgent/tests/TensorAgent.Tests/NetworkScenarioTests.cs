// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Hosting;
using TensorAgent.Core.JavaScript;
using TensorAgent.Core.Python;
using TensorAgent.Core.Sandbox;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorAgent.Tests;

/// <summary>
/// Marks a test that talks to the real internet.
///
/// <para>
/// Off by default for the same reason <see cref="LivePythonFactAttribute"/> is: a
/// build machine without a route out would otherwise report a network feature as
/// working. Skipping with the command that enables it is the honest answer; passing
/// because nothing was tried is not.
/// </para>
/// </summary>
public sealed class NetworkFactAttribute : FactAttribute
{
    /// <summary>The environment variable that admits these tests to the network.</summary>
    public const string Variable = "TENSORAGENT_ALLOW_NETWORK_TESTS";

    public NetworkFactAttribute()
    {
        if (!Enabled)
            Skip = $"reaches the real internet: set {Variable}=1 and re-run";
    }

    internal static bool Enabled =>
        !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(Variable));
}

/// <summary>
/// Marks a test that needs BOTH the network and a real CPython, and names both when
/// either is missing — a skip reason that mentions one of two missing things sends
/// the reader to set it and watch the test skip again.
/// </summary>
public sealed class LiveNetworkPythonFactAttribute : FactAttribute
{
    public LiveNetworkPythonFactAttribute()
    {
        var missing = new List<string>();
        if (!NetworkFactAttribute.Enabled)
            missing.Add($"{NetworkFactAttribute.Variable}=1");
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(LivePythonFactAttribute.RootVariable)))
            missing.Add($"{LivePythonFactAttribute.RootVariable}=<a staged runtime or a CPython 3.13 prefix>");
        if (missing.Count > 0)
            Skip = "needs the internet and an embedded CPython: set " + string.Join(" and ", missing) + " and re-run";
    }
}

/// <summary>
/// The scenarios that only work if the network really works: fetching a page from
/// each of the three runtimes, the host allow-list actually holding in each of them,
/// and installing a package and importing it afterwards.
///
/// <para>
/// The refusals are tested here too, beside the successes, and deliberately without a
/// gate: a refusal needs no network, and testing it next to the thing it refuses is
/// what stops the two from drifting into different wordings. What the gate protects
/// is only the half that opens a socket.
/// </para>
/// <para>
/// Every network test names <c>example.com</c> or <c>pypi.org</c> and nothing else.
/// A test suite that reaches for an arbitrary site is a test suite that fails when
/// somebody else's blog is down.
/// </para>
/// </summary>
[Collection(LivePythonCollection.Name)]
public sealed class NetworkScenarioTests : IDisposable
{
    private const string ReachableUrl = "https://example.com";
    private const string ReachableHost = "example.com";

    /// <summary>A host that is real and is NOT the one an allow-list will name.</summary>
    private const string OtherUrl = "https://www.iana.org/help/example-domains";
    private const string OtherHost = "www.iana.org";

    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-net-" + Guid.NewGuid().ToString("N"));
    private readonly string _work;
    private readonly string _temp;
    private readonly string _packages;
    private readonly InProcessShell _shell = new();

    private static readonly object s_primeGate = new();
    private static bool s_primed;

    public NetworkScenarioTests()
    {
        _work = Path.Combine(_root, "work");
        _temp = Path.Combine(_root, "tmp");
        _packages = Path.Combine(_root, "packages");
        foreach (string directory in new[] { _work, _temp, _packages })
            Directory.CreateDirectory(directory);

        PrimePython();
    }

    /// <summary>
    /// Binds CPython before this class builds anything else, when a runtime root was
    /// configured.
    ///
    /// <para>
    /// .NET allows exactly one <c>DllImportResolver</c> per assembly, and
    /// JavaScriptCore's is registered the first time its engine is constructed. This
    /// class has both node tests and python tests in it, and xUnit picks the order —
    /// so without this, a run where a node test happens to go first leaves every later
    /// python test unable to bind CPython at all. Priming from the constructor, which
    /// runs before each test, makes the order the same every time.
    /// </para>
    /// </summary>
    private static void PrimePython()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(LivePythonFactAttribute.RootVariable)))
            return;
        lock (s_primeGate)
        {
            if (s_primed)
                return;
            s_primed = true;
            _ = LivePython().IsAvailable;
        }
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
    }

    // =====================================================================================
    // curl and wget
    // =====================================================================================

    [NetworkFact]
    public void CurlFetchesARealPageWhenTheNetworkIsOn()
    {
        ExecutionResult result = Run($"curl -s {ReachableUrl}", Context(Policy(network: true)));

        Assert.Equal(0, result.ExitCode);
        Assert.Contains("Example Domain", result.Stdout, StringComparison.Ordinal);
    }

    [NetworkFact]
    public void CurlWritesToTheFileItWasGivenAndTheFileLandsInTheWorkRoot()
    {
        ExecutionResult result = Run($"curl -s -o page.html {ReachableUrl}", Context(Policy(network: true)));

        Assert.Equal(0, result.ExitCode);
        // -o means the body does NOT go to stdout; a model that asked for a file and
        // got the page printed as well would have paid for it twice.
        Assert.Equal(string.Empty, result.Stdout);
        Assert.Contains("Example Domain", File.ReadAllText(Path.Combine(_work, "page.html")), StringComparison.Ordinal);
    }

    [Fact]
    public void CurlIsRefusedWithTheSharedWordingWhenTheNetworkIsOff()
    {
        ExecutionResult result = Run($"curl -s {ReachableUrl}");

        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, result.Stderr, StringComparison.Ordinal);
        Assert.Equal(string.Empty, result.Stdout);
    }

    [NetworkFact]
    public void WgetSavesThePageAndSaysHowBigItWas()
    {
        ExecutionResult result = Run($"wget -O saved.html {ReachableUrl}", Context(Policy(network: true)));

        Assert.Equal(0, result.ExitCode);
        Assert.Contains("200 OK", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("Example Domain", File.ReadAllText(Path.Combine(_work, "saved.html")), StringComparison.Ordinal);
    }

    [Fact]
    public void WgetIsRefusedWithTheSharedWordingWhenTheNetworkIsOff()
    {
        ExecutionResult result = Run($"wget -q -O - {ReachableUrl}");

        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, result.Stderr, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(_work, "index.html")));
    }

    // =====================================================================================
    // python
    // =====================================================================================

    [LiveNetworkPythonFact]
    public void PythonReachesTheNetworkThroughUrllibWhenTheNetworkIsOn()
    {
        ExecutionResult result = Run(
            $"python3 -c \"import urllib.request; print(urllib.request.urlopen('{ReachableUrl}').read().decode()[:2000])\"",
            Context(Policy(network: true), python: LivePython()));

        Assert.Equal(0, result.ExitCode);
        Assert.Contains("Example Domain", result.Stdout, StringComparison.Ordinal);
    }

    [LivePythonFact]
    public void PythonUrllibIsRefusedWithTheSharedWordingWhenTheNetworkIsOff()
    {
        ExecutionResult result = Run(
            $"python3 -c \"import urllib.request; urllib.request.urlopen('{ReachableUrl}')\"",
            Context(Policy(), python: LivePython()));

        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, result.Stderr, StringComparison.Ordinal);
    }

    // =====================================================================================
    // node
    // =====================================================================================

    [NetworkFact]
    public void NodeReachesTheNetworkThroughFetchWhenTheNetworkIsOn()
    {
        ExecutionResult result = Run(
            $"node -e \"fetch('{ReachableUrl}').then(r => r.text()).then(t => console.log(t.includes('Example Domain')))\"",
            Context(Policy(network: true), javaScript: new JavaScriptCoreEngine()));

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("true\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void NodeFetchIsRefusedWithTheSharedWordingWhenTheNetworkIsOff()
    {
        ExecutionResult result = Run(
            $"node -e \"try {{ fetch('{ReachableUrl}'); }} catch (e) {{ console.log(e.message); }}\"",
            Context(Policy(), javaScript: new JavaScriptCoreEngine()));

        Assert.Equal(0, result.ExitCode);
        Assert.Equal(ExecutionPolicy.NetworkDisabledMessage + "\n", result.Stdout);
    }

    // =====================================================================================
    // the host allow-list, in all three runtimes
    // =====================================================================================

    [NetworkFact]
    public void CurlFetchesAnAllowedHostAndIsRefusedAnyOther()
    {
        ExecutionPolicy policy = Policy(network: true) with { NetworkHosts = new[] { ReachableHost } };

        ExecutionResult allowed = Run($"curl -s {ReachableUrl}", Context(policy));
        Assert.Equal(0, allowed.ExitCode);
        Assert.Contains("Example Domain", allowed.Stdout, StringComparison.Ordinal);

        ExecutionResult refused = Run($"curl -s {OtherUrl}", Context(policy));
        Assert.NotEqual(0, refused.ExitCode);
        Assert.Contains(ExecutionPolicy.HostNotAllowedMessage(OtherHost, policy.NetworkHosts), refused.Stderr, StringComparison.Ordinal);
        Assert.Equal(string.Empty, refused.Stdout);
    }

    [NetworkFact]
    public void NodeFetchReachesAnAllowedHostAndIsRefusedAnyOther()
    {
        ExecutionPolicy policy = Policy(network: true) with { NetworkHosts = new[] { ReachableHost } };
        var engine = new JavaScriptCoreEngine();

        ExecutionResult allowed = Run(
            $"node -e \"fetch('{ReachableUrl}').then(r => console.log(r.status))\"", Context(policy, javaScript: engine));
        Assert.Equal(0, allowed.ExitCode);
        Assert.Equal("200\n", allowed.Stdout);

        ExecutionResult refused = Run(
            $"node -e \"try {{ fetch('{OtherUrl}'); }} catch (e) {{ console.log(e.message); }}\"",
            Context(policy, javaScript: engine));
        Assert.Equal(0, refused.ExitCode);
        Assert.Equal(ExecutionPolicy.HostNotAllowedMessage(OtherHost, policy.NetworkHosts) + "\n", refused.Stdout);
    }

    [LiveNetworkPythonFact]
    public void PythonReachesAnAllowedHostAndIsRefusedAnyOther()
    {
        // The one the audit hook, rather than a managed client, has to enforce: without
        // the hook checking the name, urllib walks straight past an allow-list that curl
        // and fetch both obey, and the session's list means nothing.
        ExecutionPolicy policy = Policy(network: true) with { NetworkHosts = new[] { ReachableHost } };
        IPythonRuntime python = LivePython();

        ExecutionResult allowed = Run(
            $"python3 -c \"import urllib.request; print(urllib.request.urlopen('{ReachableUrl}').status)\"",
            Context(policy, python: python));
        Assert.Equal(0, allowed.ExitCode);
        Assert.Equal("200\n", allowed.Stdout);

        ExecutionResult refused = Run(
            $"python3 -c \"import urllib.request; urllib.request.urlopen('{OtherUrl}')\"",
            Context(policy, python: python));
        Assert.NotEqual(0, refused.ExitCode);
        Assert.Contains(ExecutionPolicy.HostNotAllowedSuffix, refused.Stderr, StringComparison.Ordinal);
        Assert.Contains(OtherHost, refused.Stderr, StringComparison.Ordinal);
        // The refusal has to name what IS reachable, or the only move left is another guess.
        Assert.Contains(ReachableHost, refused.Stderr, StringComparison.Ordinal);
    }

    [LiveNetworkPythonFact]
    public void PythonWithNoAllowListReachesAnyHost()
    {
        // The other half of the rule, and the one a too-eager check would break: an
        // empty list means "any host", not "no host".
        ExecutionResult result = Run(
            $"python3 -c \"import urllib.request; print(urllib.request.urlopen('{OtherUrl}').status)\"",
            Context(Policy(network: true), python: LivePython()));

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("200\n", result.Stdout);
    }

    [Fact]
    public void TheGeneratedSandboxCarriesTheAllowListSoThePythonHookHasOneToEnforce()
    {
        // The plumbing, checked on a machine with no CPython at all. The live test
        // above proves the hook refuses; this proves the list reaches it, which is the
        // half that silently stops working if a caller forgets to pass the policy.
        ExecutionPolicy named = Policy(network: true) with { NetworkHosts = new[] { ReachableHost } };
        string source = PythonBootstrap.CreateSandboxSource(named);

        Assert.Contains($"        '{ReachableHost}',", source, StringComparison.Ordinal);
        // Escaped, not raw. The shared sentence has an apostrophe in it and lives
        // inside a Python string literal in generated source; pasted raw it ends the
        // string mid-sentence, and the whole bootstrap then fails to compile — which
        // is exactly how this was found.
        Assert.Contains(ExecutionPolicy.HostNotAllowedSuffix.Replace("'", "\\'"), source, StringComparison.Ordinal);

        // And the other direction: a session that named no hosts gets an empty list,
        // which the hook reads as "any host" rather than as "none".
        Assert.Contains("hosts=[],", PythonBootstrap.CreateSandboxSource(Policy(network: true)), StringComparison.Ordinal);
    }

    // =====================================================================================
    // installing a package, which is what "set up a virtual environment" means here
    // =====================================================================================

    [NetworkFact]
    public async Task TheInstallerFetchesAPureWheelAndUnpacksItIntoThePackageRoot()
    {
        ExecutionPolicy policy = Policy(network: true);
        var installer = new WheelInstaller(policy);
        Assert.True(installer.CanInstall, installer.UnavailableReason);

        var log = new List<string>();
        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", new[] { "six" }, _packages, policy) { OnOutputLine = log.Add },
            CancellationToken.None);

        Assert.True(result.ExitCode == 0, result.Stderr);
        Assert.Contains(log, line => line.StartsWith("downloading six-", StringComparison.Ordinal));
        Assert.True(File.Exists(Path.Combine(_packages, "six.py")), "six.py is not in the package root");
        // pip list reads dist-info; an install that skipped it is one the model cannot see.
        Assert.NotEmpty(Directory.GetDirectories(_packages, "six-*.dist-info"));
    }

    [Fact]
    public async Task TheInstallerRefusesWithoutTheNetworkAndNamesTheSwitchRatherThanFailingMidDownload()
    {
        ExecutionPolicy policy = Policy();
        var installer = new WheelInstaller(policy);

        Assert.False(installer.CanInstall);
        Assert.Equal(ExecutionPolicy.NetworkDisabledMessage, installer.UnavailableReason);

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", new[] { "six" }, _packages, policy), CancellationToken.None);

        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, result.Stderr, StringComparison.Ordinal);
        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [LiveNetworkPythonFact]
    public void APackageInstalledThroughPipIsImportableByTheNextPythonRun()
    {
        // The whole scenario, in the order a model would do it: install, check what is
        // installed, import it. The package root is the session's, not a virtualenv --
        // there is no virtualenv here, and this is what replaces one.
        ExecutionPolicy policy = Policy(network: true) with { PackageRoot = _packages };
        ShellContext context = Context(policy, python: LivePython(), installer: new WheelInstaller(policy));

        // Nothing is installed yet, so a later import can only have come from the
        // install. Checked on the file system rather than by importing and expecting a
        // failure: one CPython serves the whole process and its sys.modules outlives a
        // run, so `import six` in one test answers `import six` in the next.
        Assert.Empty(Directory.GetFileSystemEntries(_packages));

        ExecutionResult installed = Run("pip install six", context);
        Assert.True(installed.ExitCode == 0, installed.Stderr);

        ExecutionResult listed = Run("pip list", context);
        Assert.Equal(0, listed.ExitCode);
        Assert.StartsWith("six==", listed.Stdout, StringComparison.Ordinal);

        // __file__, not __version__: it is the one answer that proves the import
        // resolved to the file the installer wrote rather than to something the host
        // machine already had on a path.
        ExecutionResult imported = Run("python3 -c \"import six; print(six.__file__)\"", context);
        Assert.True(imported.ExitCode == 0, imported.Stderr);
        Assert.Equal(Path.Combine(_packages, "six.py"), imported.Stdout.Trim());
    }

    [LiveNetworkPythonFact]
    public void TheAssembledAppInstallsAndImportsAPackageAcrossModelToolCalls()
    {
        using var host = new AgentAppHost(Paths(network: true), python: LivePython());
        var workspace = host.Workspaces.GetOrCreate("live-package-install");

        SkillToolResult installed = host.CodeRunner!.Execute(
            new ToolCall
            {
                Name = "shell",
                Arguments = new Dictionary<string, object>(StringComparer.Ordinal)
                {
                    ["command"] = "python3 -m pip install six -q",
                },
            },
            workspace: workspace);

        Assert.True(installed.Ok, installed.Content);
        Assert.Contains("Installed: six", installed.Content, StringComparison.Ordinal);
        Assert.True(File.Exists(Path.Combine(workspace.EnvDirectory, "six.py")));

        SkillToolResult imported = host.CodeRunner.Execute(
            new ToolCall
            {
                Name = "shell",
                Arguments = new Dictionary<string, object>(StringComparer.Ordinal)
                {
                    ["command"] = "python3 -c \"import six; print(six.__file__)\"",
                },
            },
            workspace: workspace);

        Assert.True(imported.Ok, imported.Content);
        Assert.Contains("six.py", imported.Content, StringComparison.Ordinal);

        // The concrete package/spelling from the original report. Its current wheel is
        // pure Python; importing it on-device uses the Pillow copy staged in the app
        // runtime, while this macOS fixture intentionally supplies only bare CPython.
        SkillToolResult img2pdf = host.CodeRunner.Execute(
            new ToolCall
            {
                Name = "shell",
                Arguments = new Dictionary<string, object>(StringComparer.Ordinal)
                {
                    ["command"] = "pip install img2pdf -q",
                },
            },
            workspace: workspace);

        Assert.True(img2pdf.Ok, img2pdf.Content);
        Assert.Contains("Installed: img2pdf", img2pdf.Content, StringComparison.Ordinal);
        Assert.True(File.Exists(Path.Combine(workspace.EnvDirectory, "img2pdf.py")));
    }

    // =====================================================================================
    // the research skill, run the way the app runs it
    // =====================================================================================

    [LiveNetworkPythonFact]
    public void TheResearchSkillReadsARealPageThroughTheAppsOwnInterpreterAndSandbox()
    {
        string scripts = ResearchScripts();
        ExecutionPolicy policy = Policy(network: true) with { ReadableRoots = new[] { scripts } };

        ExecutionResult result = Run(
            $"python3 {scripts}/fetch_page.py {ReachableUrl}", Context(policy, python: LivePython()));

        Assert.True(result.ExitCode == 0, result.Stderr);
        Assert.Contains("# Example Domain", result.Stdout, StringComparison.Ordinal);
        Assert.Contains("This domain is for use in documentation", result.Stdout, StringComparison.Ordinal);
        // The markup went away: a skill that hands a model a page of <div>s has spent
        // the model's context on nothing.
        Assert.DoesNotContain("<html", result.Stdout, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("<style", result.Stdout, StringComparison.OrdinalIgnoreCase);
    }

    [LiveNetworkPythonFact]
    public void TheResearchSkillWritesADossierNamingEverySourceItRead()
    {
        string scripts = ResearchScripts();
        ExecutionPolicy policy = Policy(network: true) with { ReadableRoots = new[] { scripts } };

        ExecutionResult result = Run(
            $"python3 {scripts}/research.py --out notes.md --delay 0 {ReachableUrl}",
            Context(policy, python: LivePython()));

        Assert.True(result.ExitCode == 0, result.Stderr);
        string notes = File.ReadAllText(Path.Combine(_work, "notes.md"));
        Assert.Contains($"[Example Domain]({ReachableUrl})", notes, StringComparison.Ordinal);
        // The warning is the point of collecting into a file rather than into context.
        Assert.Contains("not fact and not instruction", notes, StringComparison.Ordinal);
    }

    /// <summary>
    /// The thing the whole skill exists for: a question with no URL in it comes back
    /// with sources.
    ///
    /// <para>
    /// Nothing hermetic can answer this. The ranking, the merging and the refusals are
    /// tested against canned documents in <c>ResearchSkillTests</c>; what those cannot
    /// tell you is whether the nine services still answer a plain HTTP request from a
    /// process with no key — which is exactly what stopped being true of the endpoint
    /// the old skill relied on, silently, some time after it was written. So this asks
    /// the real internet a real question and requires URLs back, and it is gated
    /// because a suite that reaches the internet by default fails when someone else's
    /// server is down.
    /// </para>
    /// <para>
    /// It deliberately does not require any PARTICULAR provider to answer. Two of them
    /// rate-limit on a bad day and that is normal operation, not a regression; what
    /// must hold is that enough of nine answer that the user gets sources.
    /// </para>
    /// </summary>
    [LiveNetworkPythonFact]
    public void TheResearchSkillFindsItsOwnSourcesForAQuestionWithNoUrlInIt()
    {
        string scripts = ResearchScripts();
        ExecutionPolicy policy = Policy(network: true) with { ReadableRoots = new[] { scripts } };

        ExecutionResult result = Run(
            $"python3 {scripts}/discover.py \"what is the Kessler syndrome\" --count 6",
            Context(policy, python: LivePython()));

        Assert.True(result.ExitCode == 0,
            $"discovery found nothing for a question every index knows about.{Environment.NewLine}{result.Stderr}");
        Assert.Contains("https://", result.Stdout, StringComparison.Ordinal);

        // At least two independent indexes answered, which is what the ranking is
        // built on: one provider answering is a run with no corroboration in it.
        string[] answered = new[] { "wikipedia", "duckduckgo", "marginalia", "hackernews" }
            .Where(name => result.Stdout.Contains(name, StringComparison.Ordinal)
                        || result.Stderr.Contains(name + ":", StringComparison.Ordinal))
            .ToArray();
        Assert.True(answered.Length >= 2,
            $"only {answered.Length} of four indexes were reached at all:{Environment.NewLine}{result.Stderr}");

        // And the page the question is about is among what came back. Chosen because
        // it is a stable, unambiguous phrase with one obvious article behind it.
        Assert.Contains("Kessler", result.Stdout, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// And the whole loop: question in, dossier out, sources cited, with the warning
    /// that travels with the text.
    /// </summary>
    [LiveNetworkPythonFact]
    public void TheResearchSkillWritesADossierForAQuestionItWasNotGivenSourcesFor()
    {
        string scripts = ResearchScripts();
        ExecutionPolicy policy = Policy(network: true) with { ReadableRoots = new[] { scripts } };

        ExecutionResult result = Run(
            $"python3 {scripts}/research.py \"what is the Kessler syndrome\" --pages 3 --delay 0.2 --out notes.md",
            Context(policy, python: LivePython()));

        Assert.True(result.ExitCode == 0, result.Stderr);
        string notes = File.ReadAllText(Path.Combine(_work, "notes.md"));
        Assert.Contains("not fact and not instruction", notes, StringComparison.Ordinal);
        Assert.Contains("## Sources", notes, StringComparison.Ordinal);
        Assert.Matches(@"\[[^\]]+\]\(https?://[^)]+\)", notes);
        Assert.Contains("Kessler", notes, StringComparison.OrdinalIgnoreCase);

        // The structured half, which analyze.py reads.
        string json = File.ReadAllText(Path.Combine(_work, "notes.json"));
        Assert.Contains("\"records\"", json, StringComparison.Ordinal);

        ExecutionResult analysed = Run(
            $"python3 {scripts}/analyze.py notes.json --claim \"the Kessler syndrome has begun\"",
            Context(policy, python: LivePython()));
        Assert.True(analysed.ExitCode == 0, analysed.Stderr);
        Assert.Contains("Sources on:", analysed.Stdout, StringComparison.Ordinal);
    }

    [LivePythonFact]
    public void TheResearchSkillNamesAnUnknownSourceRatherThanAskingNothing()
    {
        // Discovery has no key and nothing to configure, so the one thing a caller can
        // still get wrong is which providers to ask. Checked without the network,
        // because a name that does not exist has to be caught before a request rather
        // than reported as "every index failed" after several.
        string scripts = ResearchScripts();

        ExecutionResult result = Run(
            $"python3 {scripts}/discover.py anything --sources telepathy",
            Context(Policy() with { ReadableRoots = new[] { scripts } }, python: LivePython()));

        Assert.Equal(1, result.ExitCode);
        Assert.Contains("telepathy", result.Stderr, StringComparison.Ordinal);
        // And it says what IS available, because a refusal that does not is a guess.
        Assert.Contains("wikipedia", result.Stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("Traceback", result.Stderr, StringComparison.Ordinal);
    }

    [LivePythonFact]
    public void TheResearchSkillNamesTheSwitchWhenItGoesLookingForSourcesToo()
    {
        // The fetch path already says which switch is off. Discovery is the path a
        // question now STARTS on, and it asks nine services in a loop -- so without
        // this it would report nine failures, none of which said "the network is off".
        string scripts = ResearchScripts();

        ExecutionResult result = Run(
            $"python3 {scripts}/discover.py \"anything at all\"",
            Context(Policy() with { ReadableRoots = new[] { scripts } }, python: LivePython()));

        Assert.Equal(3, result.ExitCode);
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, result.Stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("Traceback", result.Stderr, StringComparison.Ordinal);
    }

    [LivePythonFact]
    public void TheResearchSkillNamesTheSwitchInsteadOfShowingATracebackWhenTheNetworkIsOff()
    {
        string scripts = ResearchScripts();

        ExecutionResult result = Run(
            $"python3 {scripts}/fetch_page.py {ReachableUrl}",
            Context(Policy() with { ReadableRoots = new[] { scripts } }, python: LivePython()));

        // 3 is the skill's own code for "the switch is off", which is what tells a
        // model to stop trying URLs and tell the user instead.
        Assert.Equal(3, result.ExitCode);
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, result.Stderr, StringComparison.Ordinal);
        Assert.Contains("Settings", result.Stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("Traceback", result.Stderr, StringComparison.Ordinal);
    }

    // =====================================================================================
    // what the model is told before it tries
    // =====================================================================================

    [Fact]
    public void TheShellDeclarationTellsTheModelTheNetworkIsBlockedBeforeItSpendsATurnOnIt()
    {
        using var host = new AgentAppHost(Paths());
        Assert.False(host.CodeExec.AllowNetwork);

        string description = ShellDeclaration(host);

        Assert.Contains("Internet/IP network access: BLOCKED for every command", description, StringComparison.Ordinal);
        // The alternative branch is worse than saying nothing: it tells the model the
        // opposite of the truth about this host, whose confinement is in the runtimes
        // and does not degrade.
        Assert.DoesNotContain("Network confinement: NOT GUARANTEED", description, StringComparison.Ordinal);
    }

    [Fact]
    public void TheShellDeclarationTellsTheModelTheNetworkIsOpenWhenTheUserOpenedIt()
    {
        using var host = new AgentAppHost(Paths(network: true));
        Assert.True(host.CodeExec.AllowNetwork);

        string description = ShellDeclaration(host);

        Assert.Contains("IP network access: ENABLED", description, StringComparison.Ordinal);
        Assert.DoesNotContain("BLOCKED for every command", description, StringComparison.Ordinal);
    }

    [Fact]
    public void TheEngineLineSaysWhichWayTheSwitchIsSetSoTheStatusBarDoesNotHaveToGuess()
    {
        using var off = new AgentAppHost(Paths());
        Assert.Contains("network off", off.DescribeEngine(), StringComparison.Ordinal);

        using var on = new AgentAppHost(Paths(network: true));
        Assert.Contains("network on", on.DescribeEngine(), StringComparison.Ordinal);
    }

    // =====================================================================================
    // helpers
    // =====================================================================================

    private ExecutionPolicy Policy(bool network = false) => new(
        AllowScripts: true,
        AllowNetwork: network,
        WorkRoot: _work,
        ReadableRoots: Array.Empty<string>(),
        TempRoot: _temp)
    {
        DefaultTimeout = TimeSpan.FromSeconds(60),
    };

    private ShellContext Context(
        ExecutionPolicy? policy = null,
        IPythonRuntime? python = null,
        IJavaScriptRuntime? javaScript = null,
        IInstallHook? installer = null) => new()
    {
        WorkingDirectory = _work,
        Policy = policy ?? Policy(),
        Environment = new Dictionary<string, string> { ["HOME"] = _work },
        Python = python,
        JavaScript = javaScript,
        Installer = installer,
    };

    /// <summary>
    /// Runs a command line, with the network OFF unless the caller passed a context
    /// that turns it on.
    ///
    /// <para>
    /// The default is off rather than "whatever the environment variable says". An
    /// earlier draft read the gate here, which made every refusal test pass on a
    /// machine without the variable and fail on one with it — a test whose meaning
    /// changes with the environment is not a test.
    /// </para>
    /// </summary>
    private ExecutionResult Run(string command, ShellContext? context = null)
        => _shell.RunCommandAsync(command, context ?? Context(), CancellationToken.None)
            .GetAwaiter().GetResult();

    private static EmbeddedPython LivePython()
        => new(Environment.GetEnvironmentVariable(LivePythonFactAttribute.RootVariable));

    /// <summary>
    /// The bundled research skill's scripts, found by walking up from the test binary.
    ///
    /// <para>
    /// It fails loudly rather than skipping when the tree is not there. A skill this
    /// project ships is not optional equipment, and a test that quietly passed because
    /// it could not find what it was testing would be the worst of both.
    /// </para>
    /// </summary>
    private static string ResearchScripts()
    {
        for (DirectoryInfo? at = new(AppContext.BaseDirectory); at is not null; at = at.Parent)
        {
            string candidate = Path.Combine(at.FullName, "skills", "research", "scripts");
            if (File.Exists(Path.Combine(candidate, "fetch_page.py")))
                return candidate;
        }
        Assert.Fail($"no skills/research/scripts above {AppContext.BaseDirectory}");
        return string.Empty;
    }

    /// <summary>A fresh installation, with the one setting these tests care about already written.</summary>
    private AgentPaths Paths(bool network = false)
    {
        string root = Path.Combine(_root, "host-" + Guid.NewGuid().ToString("N")[..8]);
        var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache"))
        {
            ExecutionMode = AgentExecutionMode.InProcess,
        };
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings loaded = settings.Load();
        loaded.AllowNetwork = network;
        settings.Save(loaded);
        return paths;
    }

    /// <summary>The <c>shell</c> tool exactly as the app declares it to a model.</summary>
    private static string ShellDeclaration(AgentAppHost host)
    {
        Assert.NotNull(host.CodeRunner);
        foreach (ToolFunction declared in host.CodeRunner!.DeclareTools())
        {
            if (string.Equals(declared.Name, ShellTools.ShellToolName, StringComparison.Ordinal))
                return declared.Description;
        }
        Assert.Fail("the app declares no shell tool at all");
        return string.Empty;
    }
}
