using System.Diagnostics;
using System.Text;
using System.Text.Json;
using TensorAgent.Core.JavaScript;
using TensorAgent.Core.Sandbox;

namespace TensorAgent.Tests;

internal static class TestPlatforms
{
    internal static bool HasAppleJavaScriptCore => OperatingSystem.IsMacOS()
        || OperatingSystem.IsIOS() || OperatingSystem.IsMacCatalyst();

    internal static string? FindExecutable(string name)
    {
        string executable = OperatingSystem.IsWindows() ? name + ".exe" : name;
        return (Environment.GetEnvironmentVariable("PATH") ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Select(directory => Path.Combine(directory.Trim('"'), executable))
            .FirstOrDefault(File.Exists);
    }

    internal static void CreateFileSymlink(string path, string target)
    {
        try { File.CreateSymbolicLink(path, target); }
        catch (IOException ex) when (OperatingSystem.IsWindows() && (ex.HResult & 0xffff) == 1314)
        {
            Skip.If(true, "Windows symbolic-link creation requires Developer Mode or the symbolic-link privilege.");
        }
    }

    internal static void CreateDirectorySymlink(string path, string target)
    {
        try { Directory.CreateSymbolicLink(path, target); }
        catch (IOException ex) when (OperatingSystem.IsWindows() && (ex.HResult & 0xffff) == 1314)
        {
            Skip.If(true, "Windows symbolic-link creation requires Developer Mode or the symbolic-link privilege.");
        }
    }
}

public sealed class AppleJavaScriptFactAttribute : FactAttribute
{
    public AppleJavaScriptFactAttribute()
    {
        if (!TestPlatforms.HasAppleJavaScriptCore)
            Skip = "Exercises the Apple JavaScriptCore framework; Windows desktop uses native Node.js.";
    }
}

public sealed class AppleJavaScriptTheoryAttribute : TheoryAttribute
{
    public AppleJavaScriptTheoryAttribute()
    {
        if (!TestPlatforms.HasAppleJavaScriptCore)
            Skip = "Exercises the Apple JavaScriptCore framework; Windows desktop uses native Node.js.";
    }
}

public sealed class WebJavaScriptFactAttribute : FactAttribute
{
    public WebJavaScriptFactAttribute() => Skip = WebJavaScript.Unavailable;
}

public sealed class WebJavaScriptTheoryAttribute : TheoryAttribute
{
    public WebJavaScriptTheoryAttribute() => Skip = WebJavaScript.Unavailable;
}

/// <summary>Runs the same page fixtures in JavaScriptCore on Apple and Node's isolated VM elsewhere.</summary>
internal static class WebJavaScript
{
    private static string? Node => Environment.GetEnvironmentVariable("TENSORAGENT_TEST_NODE")
        ?? TestPlatforms.FindExecutable("node");

    internal static string? Unavailable => TestPlatforms.HasAppleJavaScriptCore || File.Exists(Node)
        ? null : "Page fixtures require Node.js on PATH or TENSORAGENT_TEST_NODE=<path to node>.";

    internal static ExecutionResult Run(string source, string root, InterpreterContext context)
    {
        if (TestPlatforms.HasAppleJavaScriptCore)
            return new JavaScriptCoreEngine().RunCodeAsync(source, [], context, CancellationToken.None)
                .GetAwaiter().GetResult();

        // A separate VM keeps Node's process/module globals out of the browser fixture.
        // Stop after the result has flushed: page watchdog intervals intentionally outlive a turn.
        string script = """
            const vm = require('node:vm');
            const fixtureConsole = Object.create(console);
            fixtureConsole.log = (...args) => {
              if (String(args[0]).startsWith('<<RESULT>>'))
                process.stdout.write(args.join(' ') + '\n', () => process.exit(0));
              else console.log(...args);
            };
            vm.runInNewContext(
            """ + JsonSerializer.Serialize(source) + """
            , { console: fixtureConsole, setTimeout, clearTimeout, setInterval, clearInterval },
              { filename: 'page-fixture.js' });
            """;
        string path = Path.Combine(root, "page-fixture.cjs");
        File.WriteAllText(path, script);
        return RunNode(root, path);
    }

    internal static async Task<SyntaxCheckResult> CheckSyntaxAsync(string path)
    {
        if (TestPlatforms.HasAppleJavaScriptCore)
            return await new JavaScriptCoreEngine().CheckSyntaxAsync(path, CancellationToken.None);
        ExecutionResult result = RunNode(Path.GetDirectoryName(path)!, "--check", path);
        return new SyntaxCheckResult(result.Ok, result.Stderr);
    }

    private static ExecutionResult RunNode(string root, params string[] arguments)
    {
        Assert.Null(Unavailable);
        var start = new ProcessStartInfo(Node!)
        {
            WorkingDirectory = root,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
            CreateNoWindow = true,
        };
        foreach (string argument in arguments)
            start.ArgumentList.Add(argument);
        var clock = Stopwatch.StartNew();
        using Process process = Process.Start(start)!;
        Task<string> stdout = process.StandardOutput.ReadToEndAsync();
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        bool timedOut = !process.WaitForExit(30_000);
        if (timedOut)
        {
            process.Kill(entireProcessTree: true);
            process.WaitForExit();
        }
        return new ExecutionResult(timedOut ? 124 : process.ExitCode,
            stdout.GetAwaiter().GetResult(), stderr.GetAwaiter().GetResult(), timedOut,
            root, new Dictionary<string, string>(), clock.Elapsed);
    }
}
