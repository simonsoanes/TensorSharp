using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorAgent.Tests;

[Collection(ProcessEnvironmentCollection.Name)]
public sealed class DesktopAgentHostTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-desktop-" + Guid.NewGuid().ToString("N"));
    private AgentAppHost? _host;

    private AgentAppHost CreateHost(bool network = false)
    {
        Skip.IfNot(DesktopShellBackend.IsSupported, "Native child processes are unavailable on this platform.");
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"));
        paths.EnsureCreated();
        var store = new SettingsStore(paths.SettingsFile);
        AppSettings settings = store.Load();
        settings.AllowNetwork = network;
        store.Save(settings);
        _host = new AgentAppHost(paths);
        return _host;
    }

    private static ToolCall Shell(string command) => new()
    {
        Name = ShellTools.ShellToolName,
        Arguments = new Dictionary<string, object?> { ["command"] = command },
    };

    [SkippableFact]
    public void DesktopDiscoversRealRuntimesAndDeclaresNativePackageInstallation()
    {
        AgentAppHost host = CreateHost(network: true);
        Assert.Equal("process", host.Backend.Name);
        Assert.True(host.Backend.UsesHostProcesses);
        Assert.Same(host.Backend, host.CodeRunner!.Backend);
        Assert.False(CodeEnvironment.IsConfigured);
        Assert.Null(host.JavaScript);
        Assert.Null(host.Python);
        string declaration = host.CodeRunner.DeclareTools().Single(t => t.Name == ShellTools.ShellToolName).Description;
        Assert.Contains("npm", declaration, StringComparison.Ordinal);
        Assert.DoesNotContain("pure-Python wheels", declaration, StringComparison.Ordinal);
        Assert.DoesNotContain("JavaScriptCore", host.DescribeEngine(), StringComparison.Ordinal);
    }

    [SkippableFact]
    public void DesktopBackendPreservesTheNativeSkillInterpreterGuard()
    {
        Skip.IfNot(OperatingSystem.IsWindows(), "Requires the Windows WSL launcher guard.");
        var backend = new DesktopShellBackend(new ProcessShellBackend(null, SkillSandboxMode.Off), Array.Empty<string>());
        string directory = Path.Combine(_root, "skills", "native-probe");
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "SKILL.md"), "---\nname: native-probe\ndescription: Native interpreter probe.\n---\n");
        File.WriteAllText(Path.Combine(directory, "probe.sh"), "exit 0\n");
        Skill skill = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { Path.Combine(_root, "skills") } }).Skills.Single();
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Off,
            Backend = backend,
            Interpreters = new() { [".sh"] = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "bash.exe") },
        });
        SkillToolResult result = runner.Run(skill, "probe.sh", Array.Empty<string>());
        Assert.False(result.Ok);
        Assert.Contains("WSL launcher", result.Content, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void NodeSkillCanSpawnAProcessWriteAnArtifactAndKeepSessionState()
    {
        AgentAppHost host = CreateHost();
        Skip.IfNot(host.CodeRunner!.CanRun, "A required OS sandbox and shell are not available.");
        Skip.IfNot(CodeEnvironment.TryResolveInterpreter(CodeLanguage.JavaScript, out _, out _), "Node.js is not installed.");
        SessionWorkspace workspace = host.Workspaces.GetOrCreate("node-session");
        string skill = Path.Combine(_root, "fixture-skill");
        Directory.CreateDirectory(skill);
        File.WriteAllText(Path.Combine(skill, "script.js"), """
            const fs = require('node:fs');
            const child = require('node:child_process').spawnSync(process.execPath, ['-e', 'process.stdout.write("native child")'], { encoding: 'utf8' });
            if (child.error || child.status !== 0) throw child.error || new Error(child.stderr);
            fs.writeFileSync('result.txt', child.stdout);
            console.log(child.stdout);
            """);
        string quote = "'" + Path.Combine(skill, "script.js").Replace("'", "'\\''") + "'";
        SkillToolResult result = host.CodeRunner.Execute(Shell($"node {quote}"), workspace: workspace, skillDirectories: new[] { skill });
        Assert.True(result.Ok, result.Content);
        Assert.Contains("native child", result.Content, StringComparison.Ordinal);
        Assert.Equal("native child", File.ReadAllText(Path.Combine(workspace.WorkDirectory, "result.txt")));
        Assert.Contains(result.Files, f => f.Name == "result.txt");

        SkillToolResult next = host.CodeRunner.Execute(Shell("node -e 'console.log(require(\"node:fs\").readFileSync(\"result.txt\", \"utf8\"))'"), workspace: workspace);
        Assert.True(next.Ok, next.Content);
        Assert.Contains("native child", next.Content, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void NativeProcessesCannotWriteToTheHostHomeDirectory()
    {
        AgentAppHost host = CreateHost();
        Skip.IfNot(host.CodeRunner!.CanRun, "A required OS sandbox and shell are not available.");
        Skip.IfNot(CodeEnvironment.TryResolveInterpreter(CodeLanguage.JavaScript, out _, out _), "Node.js is not installed.");
        SessionWorkspace workspace = host.Workspaces.GetOrCreate("write-boundary");
        // macOS allows its native temporary directory for executable compatibility.
        // The user's real home is outside that documented exception.
        string outside = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "tensoragent-write-denied-" + Guid.NewGuid().ToString("N") + ".txt");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "escape.js"),
            "require('node:fs').writeFileSync(" + JsonSerializer.Serialize(outside) + ", 'escape');");
        try
        {
            SkillToolResult result = host.CodeRunner.Execute(Shell("node escape.js"), workspace: workspace);
            Assert.False(result.Ok, result.Content);
            Assert.False(File.Exists(outside));
        }
        finally { File.Delete(outside); }
    }

    [SkippableFact]
    public void NativeNetworkPermissionChangesOnTheNextCommand()
    {
        AgentAppHost host = CreateHost();
        Skip.IfNot(host.CodeRunner!.CanRun, "A required OS sandbox and shell are not available.");
        Skip.IfNot(CodeEnvironment.TryResolveInterpreter(CodeLanguage.JavaScript, out _, out _), "Node.js is not installed.");
        host.Server.Start();
        SessionWorkspace workspace = host.Workspaces.GetOrCreate("network-switch");
        string url = host.Server.BaseUrl.TrimEnd('/') + "/api/agent/engine";
        string cookie = $"{LoopbackServer.TokenCookie}={host.Server.Token}";
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "fetch.js"),
            "fetch(" + JsonSerializer.Serialize(url) + ", { headers: { Cookie: " + JsonSerializer.Serialize(cookie)
            + " } }).then(async r => { if (!r.ok) throw new Error(r.status); console.log(await r.text()); })"
            + ".catch(e => { console.error(e.message); process.exitCode = 7; });");
        SkillToolResult denied = host.CodeRunner.Execute(Shell("node fetch.js"), workspace: workspace);
        Assert.False(denied.Ok, denied.Content);

        AppSettings enabled = host.Settings.Load();
        enabled.AllowNetwork = true;
        host.ApplySettings(enabled);
        SkillToolResult allowed = host.CodeRunner.Execute(Shell("node fetch.js"), workspace: workspace);
        Assert.True(allowed.Ok, allowed.Content);
        Assert.Contains("native process shell", allowed.Content, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void ADesktopHostDoesNotSilentlyIgnoreAnUnsupportedNetworkHostRestriction()
    {
        AgentAppHost host = CreateHost(network: true);
        AppSettings settings = host.Settings.Load();
        settings.NetworkHosts.Add("example.com");
        host.ApplySettings(settings);
        ConfinedResult refused = host.Backend.Run(new ShellLaunch
        {
            Argv = new[] { "node", "--version" },
            WorkingDirectory = host.Paths.ScratchDirectory,
            WriteDirectory = host.Paths.ScratchDirectory,
            ReadOnlyDirectory = host.Paths.ScratchDirectory,
            AllowNetwork = true,
        });
        Assert.False(refused.Started);
        Assert.Contains("networkHosts allow-list", refused.Error, StringComparison.Ordinal);
    }

    public void Dispose()
    {
        _host?.Dispose();
        CodeEnvironment.Reset();
        try { Directory.Delete(_root, true); } catch (IOException) { }
    }
}
