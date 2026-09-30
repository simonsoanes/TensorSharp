// Copyright (c) Zhongkai Fu. Licensed under the repository's BSD-3-Clause license.
using System.Text.Json;
using TensorSharp.AgentHost.CodeExec;

namespace InferenceWeb.Tests;

public sealed class WindowsSkillTheoryAttribute : TheoryAttribute
{
    public WindowsSkillTheoryAttribute()
    {
        if (!OperatingSystem.IsWindows()) Skip = "Requires Windows interpreter resolution.";
    }
}

public sealed class WindowsNodeSkillTheoryAttribute : TheoryAttribute
{
    public WindowsNodeSkillTheoryAttribute()
    {
        if (!OperatingSystem.IsWindows())
            Skip = "Requires a standalone Windows Node.js runtime.";
        else if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.JavaScript, out _, out _))
            Skip = "Requires native Node.js on PATH.";
    }
}

public sealed class WindowsPythonSkillTheoryAttribute : TheoryAttribute
{
    public WindowsPythonSkillTheoryAttribute()
    {
        if (!OperatingSystem.IsWindows()) Skip = "Requires Windows native known-folder APIs.";
        else if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _))
            Skip = "Requires native Python to call Windows known-folder APIs in a child process.";
    }
}

public sealed class SkillNativeInterpreterTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-native-skill-" + Guid.NewGuid().ToString("N"));
    private readonly SessionWorkspaceManager _workspaces;

    public SkillNativeInterpreterTests() => _workspaces = new SessionWorkspaceManager(_root);

    private Skill MakeSkill(string extension, string source)
    {
        string directory = Path.Combine(_root, "skills", "native-probe");
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "SKILL.md"), "---\nname: native-probe\ndescription: Native interpreter probe.\n---\n");
        File.WriteAllText(Path.Combine(directory, "probe" + extension), source);
        return new SkillRegistry(new SkillRegistryOptions { Roots = new[] { Path.Combine(_root, "skills") } }).Skills.Single();
    }

    [WindowsSkillTheory]
    [InlineData(".sh", "System32", "bash.exe", SkillSandboxMode.Preferred)]
    [InlineData(".bash", "System32", "bash.exe", SkillSandboxMode.Off)]
    [InlineData(".sh", "Sysnative", "bash.exe", SkillSandboxMode.Preferred)]
    [InlineData(".sh", "SysWOW64", "wsl.exe", SkillSandboxMode.Off)]
    public void WslSkillInterpretersAreRejectedBeforeLaunch(string extension, string directory, string name, SkillSandboxMode mode)
    {
        Skill skill = MakeSkill(extension, "exit 0\n");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = mode,
            Interpreters = new(StringComparer.OrdinalIgnoreCase)
            {
                [extension] = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), directory, name),
            },
        });
        SkillToolResult result = runner.Run(skill, "probe" + extension, Array.Empty<string>());
        Assert.False(result.Ok);
        Assert.Contains("WSL launcher", result.Content, StringComparison.Ordinal);
        Assert.Contains("Windows Node.js/npm", result.Content, StringComparison.Ordinal);
    }

    [WindowsSkillTheory]
    [InlineData("System32", "bash.exe", true)]
    [InlineData("Sysnative", "wsl.exe", true)]
    [InlineData("System32", "cmd.exe", false)]
    [InlineData("System32", "powershell.exe", false)]
    [InlineData("System32-tools", "bash.exe", false)]
    [InlineData("Git/bin", "bash.exe", false)]
    public void WslDetectionDistinguishesNativeToolsAndDirectoryBoundaries(string directory, string name, bool expected)
    {
        string path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), directory, name);
        Assert.Equal(expected, ShellProgram.IsWslLauncher(path));
    }

    [WindowsNodeSkillTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void JavaScriptSkillUsesSessionRuntimeAndPreservesArgumentBoundaries(bool underHome)
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("native");
        Assert.True(CodeEnvironment.TryResolveInterpreter(CodeLanguage.JavaScript, out string? node, out string? error), error);
        string prefix = underHome ? Path.Combine(workspace.WorkDirectory, ".home") : workspace.WorkDirectory;
        string runtime = Path.Combine(prefix, ".local", "bin", "node.exe");
        Directory.CreateDirectory(Path.GetDirectoryName(runtime)!);
        File.Copy(node!, runtime);
        if (!OperatingSystem.IsWindows()) File.SetUnixFileMode(runtime, File.GetUnixFileMode(node!));

        Skill skill = MakeSkill(".mjs", "console.log(JSON.stringify({runtime:process.execPath,args:process.argv.slice(2),platform:process.platform}));\n");
        string[] arguments = { "name with spaces", "中文", "a&b|c", "\"quoted\"", "$(ignored)", "%PATH%", "C:\\path with spaces\\" };
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Off, Workspace = workspace });
        SkillToolResult result = runner.Run(skill, "probe.mjs", arguments);
        Assert.True(result.Ok, result.Content);
        Assert.Contains(JsonSerializer.Serialize(runtime), result.Content, StringComparison.Ordinal);
        Assert.Contains(Path.GetDirectoryName(runtime)!, CodeEnvironment.SessionExecutablePath(workspace, null).Split(Path.PathSeparator));
        // Node emits Unicode directly; serialize with the same escaping policy.
        Assert.Contains(JsonSerializer.Serialize(arguments, new JsonSerializerOptions
        {
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        }), result.Content, StringComparison.Ordinal);
    }

    [WindowsPythonSkillTheory]
    [InlineData(false, SkillSandboxMode.Preferred)]
    [InlineData(true, SkillSandboxMode.Preferred)]
    [InlineData(false, SkillSandboxMode.Off)]
    [InlineData(true, SkillSandboxMode.Off)]
    public void FreshNativeFolderLookupsAgreeWithThePrivateProfile(bool viaShell, SkillSandboxMode mode)
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("native");
        Assert.True(CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out string? error), error);
        using Stream source = typeof(SkillNativeInterpreterTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.WindowsKnownFolders.probe.py")!;
        Assert.NotNull(source);
        using var reader = new StreamReader(source);
        string script = reader.ReadToEnd();
        if (viaShell)
        {
            File.WriteAllText(Path.Combine(workspace.WorkDirectory, "known_folders.py"), script);
            using var shell = new ShellRunner(new CodeExecOptions { Enabled = true, Sandbox = mode, MaxAutoInstalls = 0 });
            CodeExecResult result = shell.Run(new ShellRequest("& '" + python!.Replace("'", "''") + "' known_folders.py"), workspace);
            Assert.True(result.Ok, result.Content);
            Assert.Contains("native-known-folders-agree", result.Content, StringComparison.Ordinal);
        }
        else
        {
            Skill skill = MakeSkill(".py", script);
            var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
            {
                Sandbox = mode, Workspace = workspace, Interpreters = new() { [".py"] = python! },
            });
            SkillToolResult result = runner.Run(skill, "probe.py", Array.Empty<string>());
            Assert.True(result.Ok, result.Content);
            Assert.Contains("native-known-folders-agree", result.Content, StringComparison.Ordinal);
        }
        Assert.True(Directory.Exists(Path.Combine(workspace.WorkDirectory, ".home", "AppData", "Local")));
        Assert.True(Directory.Exists(Path.Combine(workspace.WorkDirectory, ".home", "AppData", "Roaming")));
    }

    public void Dispose()
    {
        _workspaces.Release("native");
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }
}
