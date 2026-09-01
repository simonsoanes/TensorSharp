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
using TensorSharp.Runtime;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins automatic dependency setup for skill scripts: a script that dies on a missing
/// import gets the module installed into the session environment and is run again,
/// without the model spending a round per import. Also pins the two up-front paths —
/// the caller's <c>packages</c> argument and a skill's own <c>requirements.txt</c>.
///
/// <para>
/// The installer here is a fake that "installs" by writing a module file into the
/// workspace env — the real pip path (wheels-only, confined) is exercised by
/// CodeExecInstallTests and the live end-to-end runs; these tests pin the retry LOOP's
/// decisions, which need no network.
/// </para>
/// </summary>
public class SkillScriptAutoInstallTests : IDisposable
{
    private readonly string _base;

    public SkillScriptAutoInstallTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-autoinstall-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static bool HavePython =>
        TensorSharp.AgentHost.CodeExec.CodeEnvironment.TryResolveInterpreter(
            TensorSharp.AgentHost.CodeExec.CodeLanguage.Python, out _, out _);

    private SessionWorkspace Workspace(string id) =>
        new SessionWorkspaceManager(Path.Combine(_base, "sessions")).GetOrCreate(id);

    private Skill MakeSkill(string script, string? requirements = null, string? scriptDirRequirements = null)
    {
        string root = Path.Combine(_base, "skill-" + Guid.NewGuid().ToString("N"));
        string dir = Path.Combine(root, "tester");
        Directory.CreateDirectory(Path.Combine(dir, "scripts"));
        File.WriteAllText(Path.Combine(dir, "SKILL.md"),
            "---\nname: tester\ndescription: auto-install test skill\n---\nBody.");
        File.WriteAllText(Path.Combine(dir, "scripts", "tool.py"), script);
        if (requirements != null)
            File.WriteAllText(Path.Combine(dir, "requirements.txt"), requirements);
        if (scriptDirRequirements != null)
            File.WriteAllText(Path.Combine(dir, "scripts", "requirements.txt"), scriptDirRequirements);
        return new SkillRegistry(new SkillRegistryOptions { Roots = new[] { root } }).Skills.Single();
    }

    /// <summary>Installs by writing <c>NAME.py</c> into the env; records every request.</summary>
    private sealed class FakeInstaller : ICodeRunner
    {
        public List<string> Installed { get; } = new();
        public string? FailWith { get; set; }
        public HashSet<string> Broken { get; } = new(StringComparer.OrdinalIgnoreCase);

        public bool CanRun => true;
        public string? UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = SkillToolNames.Shell };
        public bool CanInstallPackages => true;

        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null, SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null) =>
            SkillToolResult.Failure("not used in these tests");

        public string? InstallPackages(string language, IReadOnlyList<string> packages,
            SessionWorkspace workspace, Action<string>? onOutput = null)
        {
            if (FailWith != null)
                return FailWith;
            foreach (string package in packages)
            {
                Installed.Add(package);
                // A "broken" package installs but does not make the import work.
                if (!Broken.Contains(package))
                    File.WriteAllText(
                        Path.Combine(workspace.EnvDirectory, package + ".py"),
                        $"NAME = '{package}'");
                workspace.MarkInstalled("python", new[] { package });
            }
            return null;
        }
    }

    private static SkillScriptRunner Runner(SessionWorkspace workspace, FakeInstaller installer) =>
        new(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Off,
            Workspace = workspace,
            PackageInstaller = installer,
        });

    [Fact]
    public void AMissingImport_IsInstalledAndTheScriptRerun_InOneCall()
    {
        if (!HavePython) return;

        var installer = new FakeInstaller();
        Skill skill = MakeSkill("import neededmod\nprint('using', neededmod.NAME)");

        SkillToolResult result = Runner(Workspace("a"), installer)
            .Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.True(result.Ok, result.Content);
        Assert.Equal("neededmod", Assert.Single(installer.Installed));
        Assert.Contains("Auto-installed missing dependency: neededmod", result.Content, StringComparison.Ordinal);
        Assert.Contains("using neededmod", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AChainOfMissingImports_ResolvesOnePerRetry()
    {
        if (!HavePython) return;

        var installer = new FakeInstaller();
        Skill skill = MakeSkill("import first\nimport second\nprint(first.NAME, second.NAME)");

        SkillToolResult result = Runner(Workspace("b"), installer)
            .Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.True(result.Ok, result.Content);
        Assert.Equal(new[] { "first", "second" }, installer.Installed);
        Assert.Contains("first second", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInstallThatDoesNotFixTheImport_StopsAfterOneTry()
    {
        if (!HavePython) return;

        var installer = new FakeInstaller();
        installer.Broken.Add("ghostmod");
        Skill skill = MakeSkill("import ghostmod");

        SkillToolResult result = Runner(Workspace("c"), installer)
            .Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        // One install attempt, not a loop chasing an import that will never resolve.
        Assert.Equal("ghostmod", Assert.Single(installer.Installed));
    }

    [Fact]
    public void AFailedInstall_IsReportedAndNotRetried()
    {
        if (!HavePython) return;

        var installer = new FakeInstaller { FailWith = "no wheel for this platform" };
        Skill skill = MakeSkill("import unbuildable");

        SkillToolResult result = Runner(Workspace("d"), installer)
            .Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Contains("Could not auto-install 'unbuildable': no wheel for this platform",
            result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ExplicitPackages_AreSetUpBeforeTheFirstAttempt()
    {
        if (!HavePython) return;

        var installer = new FakeInstaller();
        Skill skill = MakeSkill("import upfront\nprint('ready', upfront.NAME)");

        SkillToolResult result = Runner(Workspace("e"), installer)
            .Run(skill, "scripts/tool.py", Array.Empty<string>(), packages: new[] { "upfront" });

        Assert.True(result.Ok, result.Content);
        Assert.Equal("upfront", Assert.Single(installer.Installed));   // no failed attempt needed
        Assert.Contains("Set up dependencies (requested): upfront", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ASkillsRequirementsFile_IsAppliedOncePerSession()
    {
        if (!HavePython) return;

        var installer = new FakeInstaller();
        SessionWorkspace workspace = Workspace("f");
        Skill skill = MakeSkill(
            "import reqroot, reqlocal\nprint(reqroot.NAME, reqlocal.NAME)",
            requirements: "# root deps\nreqroot\n",
            scriptDirRequirements: "reqlocal\n-r ../requirements.txt\nhttps://example/x.whl\n");

        SkillToolResult first = Runner(workspace, installer).Run(skill, "scripts/tool.py", Array.Empty<string>());
        Assert.True(first.Ok, first.Content);
        // Option lines and URLs from requirements.txt are never forwarded to pip.
        Assert.Equal(new[] { "reqroot", "reqlocal" }, installer.Installed.OrderBy(p => p).Reverse());

        SkillToolResult second = Runner(workspace, installer).Run(skill, "scripts/tool.py", Array.Empty<string>());
        Assert.True(second.Ok, second.Content);
        Assert.Equal(2, installer.Installed.Count);   // not applied again
    }

    [Fact]
    public void WithoutAnInstaller_TheOldCoachingStands()
    {
        if (!HavePython) return;

        // A name no host can have, like every other case in this file. It used to be
        // `defusedxml`, which is a REAL package: the test then asserted "this module is
        // missing" about a developer machine that may perfectly well have it installed -
        // and any machine set up to exercise the document skills certainly does, since
        // pptx and docx validation both import it. The assertion is about the COACHING,
        // not about defusedxml.
        Skill skill = MakeSkill("import ts_absent_module_xyz");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Off,
            Workspace = Workspace("g"),
        });

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        // The coaching still has to name the exact next action, not merely the module —
        // that is the whole lesson of the gemma-4-E4B incident. What changed with the
        // shell surface is only the SPELLING of that action: an install is now a command
        // the model types, not a 'packages' argument it fills in.
        Assert.Contains("ts_absent_module_xyz", result.Content, StringComparison.Ordinal);
        Assert.Contains("pip", result.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("install", result.Content, StringComparison.OrdinalIgnoreCase);
    }
}
