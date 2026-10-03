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
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Drift guards for how `dotnet build TensorSharp.slnx` builds TensorAgent.
///
/// <para>
/// TensorAgent's platform-neutral projects are listed in TensorSharp.slnx. Its app,
/// TensorAgent/src/TensorAgent.Maui, is not: the heads need the .NET MAUI workloads, and a
/// listed head would fail every restore on an SDK without them, which is every Linux build
/// of the solution. Directory.Solution.targets builds the app after the solution instead,
/// the way TensorAgent/scripts build it, and Directory.Build.props gives that build its own
/// restore directory. Each of those facts fails quietly when it drifts: a new
/// platform-neutral project that is never built, a new head the solution never builds, a
/// simulator app signed differently from build-sim.sh's, or a restore the two builds keep
/// overwriting, which recompiles TensorSharp.Models on every alternation. So these tests
/// read the files and pin them.
/// </para>
/// </summary>
public class TensorAgentSolutionBuildTests
{
    private static readonly XNamespace Ns = XNamespace.None;

    /// <summary>The app head and the extension it embeds: both need a platform workload.</summary>
    private static readonly string[] AppOnlyProjects =
    {
        "TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj",
        "TensorAgent/src/TensorAgent.ShareExtension/TensorAgent.ShareExtension.csproj",
    };

    private static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
                dir = dir.Parent;
            Assert.NotNull(dir);
            return dir!.FullName;
        }
    }

    private static string AppProjectPath =>
        Path.Combine(RepoRoot, "TensorAgent", "src", "TensorAgent.Maui", "TensorAgent.Maui.csproj");

    private static XDocument AppProject => XDocument.Load(AppProjectPath);

    private static XDocument SolutionTargets => XDocument.Load(Path.Combine(RepoRoot, "Directory.Solution.targets"));

    private static string ScriptText(string name) =>
        File.ReadAllText(Path.Combine(RepoRoot, "TensorAgent", "scripts", name));

    private static string Normalize(string path) => path.Replace('\\', '/');

    /// <summary>An element's own condition together with its group's.</summary>
    private static string ConditionOf(XElement element) =>
        $"{element.Attribute("Condition")?.Value} {element.Parent?.Attribute("Condition")?.Value}";

    private static string[] RootSolutionProjects() =>
        XDocument.Load(Path.Combine(RepoRoot, "TensorSharp.slnx"))
            .Descendants(Ns + "Project")
            .Select(p => Normalize(p.Attribute("Path")!.Value))
            .ToArray();

    /// <summary>Every project under TensorAgent/, as a repository-relative path, never entering build output or staged runtimes.</summary>
    private static string[] TensorAgentProjects()
    {
        var skipped = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "bin", "obj", "python-runtime", "node_modules" };
        var found = new List<string>();
        var pending = new Stack<DirectoryInfo>();
        pending.Push(new DirectoryInfo(Path.Combine(RepoRoot, "TensorAgent")));
        while (pending.Count > 0)
        {
            DirectoryInfo dir = pending.Pop();
            found.AddRange(dir.EnumerateFiles("*.csproj").Select(f => Normalize(Path.GetRelativePath(RepoRoot, f.FullName))));
            foreach (DirectoryInfo child in dir.EnumerateDirectories())
            {
                if (!skipped.Contains(child.Name) && child.LinkTarget == null)
                    pending.Push(child);
            }
        }
        return found.OrderBy(p => p, StringComparer.Ordinal).ToArray();
    }

    private static XElement[] Heads(string osPlatform) =>
        SolutionTargets.Descendants(Ns + "_TensorAgentAppHead")
            .Where(e => ConditionOf(e).Contains($"IsOSPlatform('{osPlatform}')", StringComparison.Ordinal))
            .ToArray();

    private static XElement Head(string framework) =>
        Assert.Single(SolutionTargets.Descendants(Ns + "_TensorAgentAppHead"), e => e.Attribute("Include")?.Value == framework);

    private static string[] Frameworks(string osPlatform) =>
        SolutionTargets.Descendants(Ns + "_TensorAgentAppTargetFramework")
            .Where(e => ConditionOf(e).Contains($"IsOSPlatform('{osPlatform}')", StringComparison.Ordinal))
            .SelectMany(e => e.Attribute("Include")!.Value.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToArray();

    private static XElement Target(string name) =>
        Assert.Single(SolutionTargets.Descendants(Ns + "Target"), t => t.Attribute("Name")?.Value == name);

    private static string PropertyValue(string name) =>
        SolutionTargets.Descendants(Ns + name).First().Value;

    [Fact]
    public void RootSolution_ListsEveryTensorAgentProjectButTheApp()
    {
        string[] projects = TensorAgentProjects();
        foreach (string app in AppOnlyProjects)
            Assert.Contains(app, projects);

        // The app head and its extension need a platform workload; listed, they would fail
        // the solution's restore with NETSDK1147 on every SDK without it, so
        // Directory.Solution.targets builds them instead. Everything else must be listed,
        // or nothing builds it. A new app-only project has to be added above deliberately.
        string[] listed = RootSolutionProjects()
            .Where(p => p.StartsWith("TensorAgent/", StringComparison.Ordinal))
            .OrderBy(p => p, StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(projects.Except(AppOnlyProjects).ToArray(), listed);
    }

    [Fact]
    public void SolutionBuild_ChecksAndBuildsEveryHeadTheAppTargets()
    {
        XElement[] frameworks = AppProject.Descendants(Ns + "TargetFrameworks").ToArray();
        string[] Targets(bool windows) => frameworks
            .Where(e => (e.Attribute("Condition")?.Value ?? string.Empty).StartsWith("!", StringComparison.Ordinal) != windows)
            .SelectMany(e => e.Value.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToArray();
        string[] Built(string osPlatform) => Heads(osPlatform)
            .Select(e => e.Attribute("Include")!.Value)
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToArray();

        // The head restores every target framework it has on the OS, so the workload check
        // covers all of them, and each is a head the solution builds. The project's
        // non-Windows heads are the Apple ones; only a Mac can build them.
        Assert.Equal(Targets(windows: false), Frameworks("OSX"));
        Assert.Equal(Targets(windows: true), Frameworks("Windows"));
        Assert.Equal(Targets(windows: false), Built("OSX"));
        Assert.Equal(Targets(windows: true), Built("Windows"));
        Assert.DoesNotContain(SolutionTargets.Descendants(Ns + "_TensorAgentAppHead"),
            e => !ConditionOf(e).Contains("IsOSPlatform('OSX')", StringComparison.Ordinal)
                 && !ConditionOf(e).Contains("IsOSPlatform('Windows')", StringComparison.Ordinal));

        // GgmlOps.xcframework's simulator slice is arm64 only, which is also the simulator
        // the iOS SDK picks on Apple silicon.
        Assert.Contains("'$(_TensorAgentAppMachineArch)' == 'arm64'", ConditionOf(Head("net10.0-ios")), StringComparison.Ordinal);
        Assert.Contains("OSArchitecture", PropertyValue("_TensorAgentAppMachineArch"), StringComparison.Ordinal);
    }

    [Fact]
    public void SolutionBuild_BuildsTheHeadsTheWayTheScriptsDo()
    {
        string build = Target("BuildTensorAgentAppHead").Elements(Ns + "Exec").Single().Attribute("Command")!.Value;
        string clean = Target("CleanTensorAgentAppHead").Elements(Ns + "Exec").Single().Attribute("Command")!.Value;
        string arguments = PropertyValue("_TensorAgentAppArguments");

        // The child runs on the SDK that answered the workload check: the dotnet that started
        // the build, else the one this MSBuild belongs to, never just the first on PATH.
        string[] dotnet = SolutionTargets.Descendants(Ns + "_TensorAgentDotnet").Select(e => e.Value).ToArray();
        Assert.Equal("$(DOTNET_HOST_PATH)", dotnet[0]);
        Assert.Contains("$([MSBuild]::NormalizePath('$(MSBuildBinPath)/../../dotnet'))", dotnet);
        Assert.Contains("$([MSBuild]::NormalizePath('$(MSBuildBinPath)/../../dotnet.exe'))", dotnet);
        Assert.Equal("dotnet", dotnet[^1]);

        // The scripts export the MLX skip (the app never uses MLX), and a native build skipped
        // for the solution stays skipped for the app; only the environment reaches it, since
        // TensorAgentBuildDesktopEngine passes TensorSharpSkipGgmlNative=false explicitly.
        Assert.Equal("$(_TensorAgentAppEnvironment)",
            Target("BuildTensorAgentAppHead").Elements(Ns + "Exec").Single().Attribute("EnvironmentVariables")?.Value);
        string[] environment = SolutionTargets.Descendants(Ns + "_TensorAgentAppEnvironment").Select(e => e.Value).ToArray();
        Assert.Equal("TENSORSHARP_MLX_NATIVE_SKIP=true", environment[0]);
        Assert.Contains("$(_TensorAgentAppEnvironment);TENSORSHARP_GGML_NATIVE_SKIP=true", environment);
        Assert.StartsWith("\"$(_TensorAgentDotnet)\" build \"$(_TensorAgentAppProject)\" -f %(_TensorAgentAppBuildHead.Identity) %(_TensorAgentAppBuildHead.Arguments)", build, StringComparison.Ordinal);
        Assert.StartsWith("\"$(_TensorAgentDotnet)\" clean \"$(_TensorAgentAppProject)\" -f %(_TensorAgentAppAvailableHead.Identity) %(_TensorAgentAppAvailableHead.Arguments)", clean, StringComparison.Ordinal);
        Assert.EndsWith("$(_TensorAgentAppArguments)", build, StringComparison.Ordinal);
        Assert.EndsWith("$(_TensorAgentAppArguments)", clean, StringComparison.Ordinal);

        Assert.Contains("-c \"$(Configuration)\"", arguments, StringComparison.Ordinal);
        // Several referenced projects share an output directory (build-sim.sh, build-device.sh).
        Assert.Contains("-m:1", arguments, StringComparison.Ordinal);
        Assert.Contains("-m:1", ScriptText("build-sim.sh"), StringComparison.Ordinal);
        // A Visual Studio x64 prompt exports Platform=x64, which the child would inherit and
        // which moves every output under bin/x64/.
        Assert.Contains("-p:Platform=AnyCPU", arguments, StringComparison.Ordinal);
        // Exec logs every warning the child prints. `dotnet build` appends
        // -consoleloggerparameters:Summary after its arguments, so -clp:NoSummary cannot stop
        // the summary that repeats them; the console logger is replaced instead.
        Assert.Contains("-noConsoleLogger", arguments, StringComparison.Ordinal);
        Assert.Contains("\"-logger:Microsoft.Build.Logging.ConsoleLogger,Microsoft.Build;Verbosity=minimal;NoSummary\"", arguments, StringComparison.Ordinal);

        var runtimeSwitch = new Regex(@"RuntimeIdentifier|(^|\s)(-r|--runtime|-a|--arch|--os)([\s:=]|$)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        foreach (XElement head in Heads("OSX"))
        {
            string headArguments = head.Attribute("Arguments")!.Value;
            // Restore reads TensorSharp.Models' target frameworks before a ProjectReference's
            // AdditionalProperties apply, so the property must be global, as in the scripts.
            Assert.Contains("-p:TensorSharpAppleTargets=true", headArguments, StringComparison.Ordinal);
            // A RuntimeIdentifier on the command line reaches the restore of every referenced
            // project, so the two Apple heads would rewrite each other's restore on every
            // solution build. The iOS SDK defaults `dotnet build` to this Mac's simulator.
            string effective = $"{build.Replace("%(_TensorAgentAppBuildHead.Arguments)", headArguments, StringComparison.Ordinal)} {arguments} {PropertyValue("_TensorAgentAppEnvironment")}";
            Assert.DoesNotMatch(runtimeSwitch, effective);
        }
        Assert.Contains("-p:TensorSharpAppleTargets=true", ScriptText("build-mac.sh"), StringComparison.Ordinal);
        Assert.Contains("-p:TensorSharpAppleTargets=true", ScriptText("build-sim.sh"), StringComparison.Ordinal);

        // build-sim.sh's app is ad-hoc signed by the head's simulator-only CodesignKey, whose
        // condition needs the RuntimeIdentifier on the command line. Without one it is never
        // set, and the SDK signs with whatever team profile the Mac has installed.
        Assert.Contains("<CodesignKey Condition=\"$(RuntimeIdentifier.Contains('simulator'))\">-</CodesignKey>",
            File.ReadAllText(AppProjectPath), StringComparison.Ordinal);
        Assert.Contains("-p:CodesignKey=-", Head("net10.0-ios").Attribute("Arguments")!.Value, StringComparison.Ordinal);
        Assert.Contains("-p:RuntimeIdentifier=iossimulator-arm64", ScriptText("build-sim.sh"), StringComparison.Ordinal);

        // The app the build leaves is where the scripts look for it.
        string mac = Normalize(Head("net10.0-maccatalyst").Attribute("App")!.Value);
        Assert.EndsWith("net10.0-maccatalyst/maccatalyst-$(_TensorAgentAppProcessArch)/TensorAgent.app", mac, StringComparison.Ordinal);
        Assert.Contains("net10.0-maccatalyst/\"*/TensorAgent.app", ScriptText("build-mac.sh"), StringComparison.Ordinal);
        string ios = Normalize(Head("net10.0-ios").Attribute("App")!.Value);
        Assert.EndsWith("net10.0-ios/iossimulator-arm64/TensorAgent.Maui.app", ios, StringComparison.Ordinal);
        Assert.Contains("net10.0-ios/iossimulator-arm64/TensorAgent.Maui.app", ScriptText("build-sim.sh"), StringComparison.Ordinal);

        // The engine library is x64 only on Windows, as in build-windows.ps1.
        XElement windows = Head("net10.0-windows10.0.19041.0");
        Assert.Equal("-r win-x64", windows.Attribute("Arguments")!.Value);
        Assert.Contains("'-r', 'win-x64'", ScriptText("build-windows.ps1"), StringComparison.Ordinal);
        Assert.EndsWith("net10.0-windows10.0.19041.0/win-x64/TensorAgent.Maui.exe", Normalize(windows.Attribute("App")!.Value), StringComparison.Ordinal);

        // Each head names the script that builds it on its own, for its failure message.
        Assert.Equal("TensorAgent/scripts/build-mac.sh", Head("net10.0-maccatalyst").Attribute("Script")?.Value);
        Assert.Equal("TensorAgent/scripts/build-sim.sh", Head("net10.0-ios").Attribute("Script")?.Value);
        Assert.Equal("TensorAgent/scripts/build-windows.ps1", windows.Attribute("Script")?.Value);
    }

    [Fact]
    public void SolutionBuild_AsksTheSdkWhichWorkloadsAreMissingAndSaysSo()
    {
        XDocument targets = SolutionTargets;
        // The SDK's own check: guessing from pack folders breaks on user-local and MSI
        // installs, and a missing workload must be named, not skipped silently.
        XElement probe = Assert.Single(targets.Descendants(Ns + "MSBuild"));
        Assert.Equal("$(_TensorAgentAppProject)", probe.Attribute("Projects")?.Value);
        Assert.Equal("GetSuggestedWorkloads", probe.Attribute("Targets")?.Value);
        Assert.Equal("TargetFramework=%(_TensorAgentAppTargetFramework.Identity)", probe.Attribute("Properties")?.Value);
        string[] warnings = targets.Descendants(Ns + "Warning").Select(w => w.Attribute("Text")!.Value).ToArray();
        Assert.Contains(warnings, w => w.Contains("dotnet workload install @(_TensorAgentAppMissingWorkloadId, ' ')", StringComparison.Ordinal));
        // A source-built SDK has no Apple manifests and suggests workloads that cannot help;
        // with MAUI already installed for one head, the SDK names the bare platform workload
        // (ios, maccatalyst) for the other, which an install does fix.
        Assert.Contains(warnings, w => w.Contains("cannot provide the .NET MAUI workloads", StringComparison.Ordinal));
        XElement foreign = Assert.Single(targets.Descendants(Ns + "_TensorAgentAppForeignWorkload"));
        Assert.Equal("@(_TensorAgentAppMissingWorkloadId)", foreign.Attribute("Include")?.Value);
        Assert.Equal("ios;maccatalyst", foreign.Attribute("Exclude")?.Value);
        Assert.Contains("StartsWith('maui-')", foreign.Attribute("Condition")?.Value ?? string.Empty, StringComparison.Ordinal);
        Assert.Contains(warnings, w => w.Contains("%(_TensorAgentAppMissingPrerequisite.MadeBy)", StringComparison.Ordinal));

        // The opt-out is what keeps every head out, not only the probe.
        XElement available = Assert.Single(targets.Descendants(Ns + "_TensorAgentAppAvailableHead"));
        Assert.Equal("@(_TensorAgentAppHead)", available.Attribute("Include")?.Value);
        Assert.Contains("'$(TensorSharpSkipTensorAgentApp)' != 'true'", available.Attribute("Condition")?.Value ?? string.Empty, StringComparison.Ordinal);
        Assert.Contains("'@(_TensorAgentAppMissingWorkloadId)' == ''", available.Attribute("Condition")?.Value ?? string.Empty, StringComparison.Ordinal);
        Assert.Contains("'$(TensorSharpSkipTensorAgentApp)' != 'true'", probe.Attribute("Condition")?.Value ?? string.Empty, StringComparison.Ordinal);

        // Only available heads are built, less those whose prerequisites are missing.
        XElement[] buildHeads = targets.Descendants(Ns + "_TensorAgentAppBuildHead").ToArray();
        Assert.Contains(buildHeads, e => e.Attribute("Include")?.Value == "@(_TensorAgentAppAvailableHead)");
        Assert.Contains(buildHeads, e => e.Attribute("Remove")?.Value == "@(_TensorAgentAppMissingPrerequisite->'%(Head)')");
        Assert.All(buildHeads.Where(e => e.Attribute("Include") != null),
            e => Assert.Equal("@(_TensorAgentAppAvailableHead)", e.Attribute("Include")!.Value));

        // A head failing to build fails the solution build; its siblings still build first,
        // and the failure names the head and the script that builds it alone.
        XElement build = Target("BuildTensorAgentAppHead").Elements(Ns + "Exec").Single();
        Assert.Equal("ErrorAndContinue", build.Attribute("ContinueOnError")?.Value);
        Assert.Equal("true", build.Attribute("IgnoreExitCode")?.Value);
        Assert.Contains(Target("BuildTensorAgentAppHead").Elements(Ns + "Error"),
            e => e.Attribute("Condition")?.Value == "'$(_TensorAgentAppExitCode)' != '0'"
                 && e.Attribute("Text")!.Value.Contains("%(_TensorAgentAppBuildHead.Script)", StringComparison.Ordinal)
                 && e.Attribute("Text")!.Value.Contains("-p:TensorSharpSkipTensorAgentApp=true", StringComparison.Ordinal));

        // TensorAgent.slnx builds the head itself and imports this file too (MSBuild looks
        // for it above the solution), so only TensorSharp.slnx may run the app build.
        foreach (XElement target in targets.Descendants(Ns + "Target")
                     .Where(t => t.Attribute("AfterTargets") != null || t.Attribute("BeforeTargets") != null))
            Assert.Contains("'$(SolutionFileName)' == 'TensorSharp.slnx'", target.Attribute("Condition")?.Value ?? string.Empty, StringComparison.Ordinal);
        foreach (XElement group in targets.Root!.Elements().Where(e => e.Name == Ns + "PropertyGroup" || e.Name == Ns + "ItemGroup"))
            Assert.Contains("'$(SolutionFileName)' == 'TensorSharp.slnx'", group.Attribute("Condition")?.Value ?? string.Empty, StringComparison.Ordinal);
    }

    [Fact]
    public void SolutionBuild_RebuildsAndCleansTheAppWithTheSolution()
    {
        XElement rebuild = Target("RebuildTensorAgentApp");
        Assert.Equal("Rebuild", rebuild.Attribute("AfterTargets")?.Value);
        Assert.Equal("_TensorAgentAppNoIncremental;BuildTensorAgentApp", rebuild.Attribute("DependsOnTargets")?.Value);
        Assert.Equal("--no-incremental", Target("_TensorAgentAppNoIncremental").Descendants(Ns + "_TensorAgentAppBuildArguments").Single().Value);
        Assert.Contains("$(_TensorAgentAppBuildArguments)",
            Target("BuildTensorAgentAppHead").Elements(Ns + "Exec").Single().Attribute("Command")!.Value, StringComparison.Ordinal);

        Assert.Equal("Build", Target("BuildTensorAgentApp").Attribute("AfterTargets")?.Value);
        Assert.Equal("Clean", Target("CleanTensorAgentApp").Attribute("AfterTargets")?.Value);
        Assert.Equal("_ResolveTensorAgentAppHeads;CleanTensorAgentAppHead", Target("CleanTensorAgentApp").Attribute("DependsOnTargets")?.Value);
        // A clean that fails, e.g. on a device-RID restore, must not fail the solution's clean.
        Assert.Equal("WarnAndContinue", Target("CleanTensorAgentAppHead").Elements(Ns + "Exec").Single().Attribute("ContinueOnError")?.Value);
    }

    [Fact]
    public void SolutionBuild_ChecksThePrerequisitesTheHeadsLink()
    {
        string appDir = Path.GetDirectoryName(AppProjectPath)!;
        string FromApp(string include) =>
            Normalize(Path.GetRelativePath(RepoRoot, Path.GetFullPath(Path.Combine(appDir, include.Replace('\\', '/')))));
        string FromTargets(string include) =>
            Normalize(include).Replace("$(MSBuildThisFileDirectory)", string.Empty, StringComparison.Ordinal);
        XElement[] prerequisites = SolutionTargets.Descendants(Ns + "_TensorAgentAppPrerequisite").ToArray();
        string[] CheckedFor(string head) => prerequisites
            .Where(e => e.Attribute("Head")?.Value == head)
            .Select(e => e.Attribute("Include")!.Value)
            .ToArray();

        // Neither is in git. Without the xcframework the simulator app has no engine to link,
        // and without the staged CPython its link fails on every _Py* symbol.
        string[] linked = AppProject.Descendants(Ns + "NativeReference")
            .Select(e => e.Attribute("Include")!.Value)
            .Where(path => Normalize(path).EndsWith("GgmlOps.xcframework", StringComparison.Ordinal)
                           || Normalize(path).Contains("python-runtime/simulator/", StringComparison.Ordinal))
            .Select(FromApp)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(2, linked.Length);
        Assert.Equal(linked, CheckedFor("net10.0-ios").Select(FromTargets).OrderBy(path => path, StringComparer.Ordinal).ToArray());

        // A prerequisite is checked only where its head is built; otherwise an Intel Mac would
        // be told to stage files that can never make the iOS head build there.
        foreach (XElement prerequisite in prerequisites)
        {
            string headCondition = Head(prerequisite.Attribute("Head")!.Value).Attribute("Condition")?.Value ?? string.Empty;
            Assert.Contains(headCondition, ConditionOf(prerequisite), StringComparison.Ordinal);
        }

        // With the native build skipped the desktop heads cannot make their engine library, and
        // the Mac head would otherwise fail inside the SDK on a missing NativeReference.
        Assert.Contains("<NativeReference Include=\"$(TensorAgentGgmlNativeDir)/libGgmlOps.dylib\" Kind=\"Dynamic\" />",
            File.ReadAllText(AppProjectPath), StringComparison.Ordinal);
        XElement macEngine = Assert.Single(prerequisites, e => e.Attribute("Head")?.Value == "net10.0-maccatalyst");
        Assert.Equal("TensorSharp.GGML.Native/build/libGgmlOps.dylib", FromTargets(macEngine.Attribute("Include")!.Value));
        Assert.Equal("'$(_TensorAgentAppNativeSkipped)' == 'true'", macEngine.Attribute("Condition")?.Value);
        XElement windowsEngine = Assert.Single(prerequisites, e => e.Attribute("Head")?.Value == "net10.0-windows10.0.19041.0");
        Assert.Equal("$(_TensorAgentWindowsEngine)", windowsEngine.Attribute("Include")?.Value);
        Assert.Equal("'$(_TensorAgentAppNativeSkipped)' == 'true'", windowsEngine.Attribute("Condition")?.Value);
        string[] windowsEngineCandidates = SolutionTargets.Descendants(Ns + "_TensorAgentWindowsEngine")
            .Select(e => FromTargets(e.Value))
            .ToArray();
        string app = File.ReadAllText(AppProjectPath);
        foreach (string candidate in new[] { "Release/GgmlOps.dll", "GgmlOps.dll" })
        {
            Assert.Contains($"TensorSharp.GGML.Native/build-windows/{candidate}", windowsEngineCandidates);
            Assert.Contains($"$(TensorAgentGgmlNativeDir)/{candidate}", app, StringComparison.Ordinal);
        }
        Assert.Contains("'$(TensorSharpSkipGgmlNative)' == 'true' Or '$(TENSORSHARP_GGML_NATIVE_SKIP)' == 'true'",
            SolutionTargets.Descendants(Ns + "_TensorAgentAppNativeSkipped").Single().Attribute("Condition")?.Value ?? string.Empty,
            StringComparison.Ordinal);
    }

    [Fact]
    public void AppleTargetsRestore_HasItsOwnRestoreDirectory()
    {
        XElement[] paths = XDocument.Load(Path.Combine(RepoRoot, "Directory.Build.props"))
            .Descendants(Ns + "MSBuildProjectExtensionsPath")
            .ToArray();

        // TensorSharpAppleTargets=true adds TensorSharp.Models' Apple target frameworks. In
        // the desktop build's restore directory, `dotnet build TensorSharp.slnx` and the
        // app's build would rewrite each other's project.assets.json and recompile
        // TensorSharp.Models every time; the split comes after the default it extends.
        Assert.Equal(2, paths.Length);
        Assert.Equal("$(BaseIntermediateOutputPath)", paths[0].Value.Trim());
        Assert.Equal("'$(TensorSharpAppleTargets)' == 'true'", paths[1].Attribute("Condition")?.Value);
        Assert.Equal("$(MSBuildProjectExtensionsPath)apple/", paths[1].Value.Trim());
    }
}
