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
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the session workspace: one persistent sandbox per conversation, shared by the
/// <c>shell</c> tool and skill scripts, so a pipeline's steps see each other's files.
///
/// <para>
/// The incident this encodes: the pptx skill generated a presentation, and by the time
/// its own <c>validate.py</c> went looking, the file was gone — every call ran in a
/// scratch directory deleted on return. Real tasks are generate-then-validate-then-
/// convert pipelines; the workspace is what lets step N+1 find step N's output, and it
/// dies with the session instead of with the call.
/// </para>
/// </summary>
public class SessionWorkspaceTests : IDisposable
{
    private readonly string _base;

    public SessionWorkspaceTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-workspace-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static bool HavePython =>
        CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _);

    private SessionWorkspaceManager Manager() => new(Path.Combine(_base, "sessions"));

    private (ShellRunner Runner, CodeArtifactStore Store) BuildRunner()
    {
        string artifacts = Path.Combine(_base, "artifacts");
        Directory.CreateDirectory(artifacts);
        var store = new CodeArtifactStore(artifacts);
        var runner = new ShellRunner(
            new CodeExecOptions
            {
                Enabled = true,
                Sandbox = SkillSandboxMode.Off,
                Timeout = TimeSpan.FromSeconds(30),
                ScratchDirectory = Path.Combine(_base, "scratch"),
                ArtifactUriPrefix = "/api/code/artifacts",
            },
            logger: null,
            artifacts: store);
        return (runner, store);
    }

    private int _program;

    /// <summary>
    /// Run a Python program the way the model now would: put it in a file, then run the
    /// file. The program is written into the session's own working directory, which is
    /// exactly where a heredoc would have put it — so these tests exercise the same
    /// persistence they always did, through the surface that replaced run_code.
    /// </summary>
    private CodeExecResult RunPython(ShellRunner runner, SessionWorkspace workspace, string source)
    {
        string name = "prog" + (++_program).ToString(System.Globalization.CultureInfo.InvariantCulture) + ".py";
        Assert.True(workspace.TryWriteFile(name, source, out string? error), error);
        return runner.Run(new ShellRequest("python3 " + name), workspace);
    }

    // ---- files survive between calls ---------------------------------------

    [Fact]
    public void AFileWrittenByOneCall_IsReadableByTheNext()
    {
        if (!HavePython) return;

        SessionWorkspace workspace = Manager().GetOrCreate("s1");
        (ShellRunner runner, _) = BuildRunner();

        CodeExecResult first = RunPython(runner, workspace, "open('step1.txt','w').write('from step one')\nprint('wrote')");
        Assert.True(first.Ok, first.Content);

        CodeExecResult second = RunPython(runner, workspace, "print(open('step1.txt').read())");
        Assert.True(second.Ok, second.Content);
        Assert.Contains("from step one", second.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void TheSessionEnvironment_IsImportable_EvenByACallThatInstallsNothing()
    {
        if (!HavePython) return;

        SessionWorkspace workspace = Manager().GetOrCreate("s2");
        // Stand in for an earlier `pip install` without needing the network.
        File.WriteAllText(Path.Combine(workspace.EnvDirectory, "sessionmod.py"), "VALUE = 'installed earlier'");
        (ShellRunner runner, _) = BuildRunner();

        CodeExecResult result = RunPython(runner, workspace, "import sessionmod\nprint(sessionmod.VALUE)");

        Assert.True(result.Ok, result.Content);
        Assert.Contains("installed earlier", result.Content, StringComparison.Ordinal);
    }

    // ---- artifact capture reports the run, not the session ------------------

    [Fact]
    public void OnlyTheFilesARunAddedOrChanged_AreCapturedAsItsArtifacts()
    {
        if (!HavePython) return;

        SessionWorkspace workspace = Manager().GetOrCreate("s3");
        (ShellRunner runner, _) = BuildRunner();

        CodeExecResult first = RunPython(runner, workspace, "open('a.txt','w').write('A')\nprint('ok')");
        Assert.Equal("a.txt", Assert.Single(first.Artifacts).Path);

        CodeExecResult second = RunPython(runner, workspace, "print(open('a.txt').read())\nopen('b.txt','w').write('B')");
        // a.txt was read, not changed: presenting it again as this run's product
        // would bury the actual output in the session's history.
        Assert.Equal("b.txt", Assert.Single(second.Artifacts).Path);
    }

    // ---- skill scripts share the workspace ----------------------------------

    private Skill MakeSkill(string script)
    {
        string root = Path.Combine(_base, "skill-" + Guid.NewGuid().ToString("N"));
        string dir = Path.Combine(root, "tester");
        Directory.CreateDirectory(Path.Combine(dir, "scripts"));
        File.WriteAllText(Path.Combine(dir, "SKILL.md"),
            "---\nname: tester\ndescription: test skill for workspace runs\n---\nBody.");
        File.WriteAllText(Path.Combine(dir, "scripts", "tool.py"), script);
        return new SkillRegistry(new SkillRegistryOptions { Roots = new[] { root } }).Skills.Single();
    }

    private static SkillScriptRunner ScriptRunner(
        SessionWorkspace workspace, WorkspaceFileCapture? capture = null) =>
        new(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Off,
            Workspace = workspace,
            CaptureProducedFiles = capture,
        });

    [Fact]
    public void AScript_SeesFilesTheShellWrote_AndViceVersa()
    {
        if (!HavePython) return;

        SessionWorkspace workspace = Manager().GetOrCreate("s4");
        (ShellRunner runner, _) = BuildRunner();

        CodeExecResult produced = RunPython(runner, workspace, "open('input.txt','w').write('payload')\nprint('ok')");
        Assert.True(produced.Ok, produced.Content);

        Skill skill = MakeSkill(
            "text = open('input.txt').read()\n" +
            "open('validated.txt','w').write(text.upper())\n" +
            "print('validated', text)");
        SkillToolResult script = ScriptRunner(workspace).Run(skill, "scripts/tool.py", Array.Empty<string>());
        Assert.True(script.Ok, script.Content);
        Assert.Contains("validated payload", script.Content, StringComparison.Ordinal);

        CodeExecResult readBack = RunPython(runner, workspace, "print(open('validated.txt').read())");
        Assert.Contains("PAYLOAD", readBack.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AScript_CanImportPackagesTheSessionInstalled()
    {
        if (!HavePython) return;

        SessionWorkspace workspace = Manager().GetOrCreate("s5");
        File.WriteAllText(Path.Combine(workspace.EnvDirectory, "depmod.py"), "NAME = 'session dependency'");

        Skill skill = MakeSkill("import depmod\nprint('imported', depmod.NAME)");
        SkillToolResult result = ScriptRunner(workspace).Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.True(result.Ok, result.Content);
        Assert.Contains("imported session dependency", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AScriptMissingAModule_IsToldTheInstallCommandToRun()
    {
        if (!HavePython) return;

        SessionWorkspace workspace = Manager().GetOrCreate("s6");
        // A name no host can have. It was `defusedxml`, a REAL package - so the test
        // asserted "this module is missing" about a developer machine that may well have
        // it, and any machine set up to exercise the document skills certainly does,
        // since pptx and docx validation both import it. The claim under test is about
        // the COACHING, not about defusedxml.
        Skill skill = MakeSkill("import ts_absent_module_xyz\nprint('unreachable')");

        SkillToolResult result = ScriptRunner(workspace).Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        // The message must name the exact next command, not merely the module: a bare
        // traceback is what gemma-4-E4B re-ran verbatim before claiming success.
        Assert.Contains("ts_absent_module_xyz", result.Content, StringComparison.Ordinal);
        Assert.Contains("install", result.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AScriptsOutputFiles_AreCapturedForDownload_OnceEach()
    {
        if (!HavePython) return;

        SessionWorkspace workspace = Manager().GetOrCreate("s7");
        string artifacts = Path.Combine(_base, "script-artifacts");
        var store = new CodeArtifactStore(artifacts);
        WorkspaceFileCapture capture = (workDir, exclude) =>
        {
            string runId = Guid.NewGuid().ToString("N");
            IReadOnlyList<CodeArtifact> kept = store.Capture(
                runId, workDir, (id, rel, _) => "/api/code/artifacts/" + id + "/" + rel, out _, exclude);
            return kept.Select(a => new SkillProducedFile(a.Path, a.Bytes, a.Pointer)).ToList();
        };

        Skill skill = MakeSkill("open('deck.pptx','wb').write(b'PK fake pptx')\nprint('made deck')");
        SkillScriptRunner runner = ScriptRunner(workspace, capture);

        SkillToolResult first = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());
        Assert.True(first.Ok, first.Content);
        SkillProducedFile file = Assert.Single(first.Files);
        Assert.Equal("deck.pptx", file.Name);
        Assert.StartsWith("/api/code/artifacts/", file.Url, StringComparison.Ordinal);
        Assert.Contains("[deck.pptx](", first.Content, StringComparison.Ordinal);

        // A later script that only READS the deck must not present it as produced again.
        Skill checker = MakeSkill("print('checked', len(open('deck.pptx','rb').read()))");
        SkillToolResult second = ScriptRunner(workspace, capture).Run(checker, "scripts/tool.py", Array.Empty<string>());
        Assert.True(second.Ok, second.Content);
        Assert.Empty(second.Files);
    }

    [Fact]
    public void AScriptsLiveOutput_ReachesTheTap()
    {
        if (!HavePython) return;

        SessionWorkspace workspace = Manager().GetOrCreate("s8");
        Skill skill = MakeSkill("print('live line one')\nimport sys; print('live err', file=sys.stderr)");
        var lines = new List<string>();
        var gate = new object();

        SkillToolResult result = ScriptRunner(workspace).Run(
            skill, "scripts/tool.py", Array.Empty<string>(),
            line => { lock (gate) lines.Add(line); });

        Assert.True(result.Ok, result.Content);
        lock (gate)
        {
            Assert.Contains("live line one", lines);
            Assert.Contains("live err", lines);
        }
    }

    [Fact]
    public void RuntimeFallout_IsNeverCapturedAsOutput()
    {
        if (!HavePython) return;

        SessionWorkspace workspace = Manager().GetOrCreate("junk");
        (ShellRunner runner, _) = BuildRunner();

        // Simulate the mess HOME-redirected runtimes leave: Apple Python's bytecode
        // cache and LibreOffice's config tree, plus a legitimate output file.
        CodeExecResult result = RunPython(runner, workspace, 
            "import os\n" +
            "os.makedirs('Library/Caches/com.apple.python/x', exist_ok=True)\n" +
            "open('Library/Caches/com.apple.python/x/mod.cpython-39.pyc','wb').write(b'x')\n" +
            "os.makedirs('.config/libreoffice/4/user', exist_ok=True)\n" +
            "open('.config/libreoffice/4/user/registrymodifications.xcu','w').write('cfg')\n" +
            "os.makedirs('__pycache__', exist_ok=True)\n" +
            "open('__pycache__/t.cpython-312.pyc','wb').write(b'x')\n" +
            "open('real-output.txt','w').write('the answer')\n" +
            "print('done')");

        Assert.True(result.Ok, result.Content);
        CodeArtifact artifact = Assert.Single(result.Artifacts);
        Assert.Equal("real-output.txt", artifact.Path);
    }

    // ---- lifecycle ----------------------------------------------------------

    [Fact]
    public void TheWorkspace_IsPerSession_AndDiesWithIt()
    {
        SessionWorkspaceManager manager = Manager();

        SessionWorkspace a = manager.GetOrCreate("session-a");
        SessionWorkspace again = manager.GetOrCreate("session-a");
        SessionWorkspace b = manager.GetOrCreate("session-b");

        Assert.Same(a, again);
        Assert.NotEqual(a.Root, b.Root);

        File.WriteAllText(Path.Combine(a.WorkDirectory, "keep.txt"), "x");
        manager.Release("session-a");
        Assert.False(Directory.Exists(a.Root), "released workspace should be deleted");
        Assert.True(Directory.Exists(b.Root), "other sessions are untouched");
    }

    [Fact]
    public void Release_DetachesImmediately_ButWaitsForAnActiveOperationBeforeDeleting()
    {
        SessionWorkspaceManager manager = Manager();
        SessionWorkspace retiring = manager.GetOrCreate("reused-id");
        IDisposable operation = retiring.BeginOperation();

        manager.Release("reused-id");

        Assert.True(Directory.Exists(retiring.Root),
            "a worker still using the workspace must not have its directory deleted");

        // Release removes the map entry immediately. Reusing an id gets a separate
        // object and physical directory rather than attaching to the retiring worker.
        SessionWorkspace replacement = manager.GetOrCreate("reused-id");
        Assert.NotSame(retiring, replacement);
        Assert.NotEqual(retiring.Root, replacement.Root);

        operation.Dispose();

        Assert.False(Directory.Exists(retiring.Root));
        Assert.True(Directory.Exists(replacement.Root));
        manager.Release("reused-id");
    }

    [Fact]
    public async Task CleanupRegisteredWhileReleaseIsRunning_IsDisposedInsteadOfLeaked()
    {
        SessionWorkspaceManager manager = Manager();
        SessionWorkspace workspace = manager.GetOrCreate("cleanup-race");
        using var entered = new System.Threading.ManualResetEventSlim();
        using var resume = new System.Threading.ManualResetEventSlim();
        var blocking = new BlockingCleanup(entered, resume);
        var late = new CountingCleanup();
        workspace.RegisterCleanup(blocking);

        Task release = Task.Run(() => manager.Release("cleanup-race"));
        Assert.True(entered.Wait(TimeSpan.FromSeconds(5)), "release never entered cleanup");

        // RunCleanups has already copied and cleared its list. Registering now used to
        // append to an orphaned list that would never be drained again.
        workspace.RegisterCleanup(late);
        Assert.Equal(1, late.DisposeCount);

        resume.Set();
        await release.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(1, blocking.DisposeCount);
        Assert.False(Directory.Exists(workspace.Root));
    }

    [Fact]
    public void CleanupRegisteredByAnActiveOperationAfterRelease_IsDrainedAtOperationEnd()
    {
        SessionWorkspaceManager manager = Manager();
        SessionWorkspace workspace = manager.GetOrCreate("deferred-cleanup");
        IDisposable operation = workspace.BeginOperation();
        var cleanup = new CountingCleanup();

        manager.Release("deferred-cleanup");
        workspace.RegisterCleanup(cleanup);

        Assert.Equal(0, cleanup.DisposeCount);
        Assert.True(Directory.Exists(workspace.Root));

        operation.Dispose();

        Assert.Equal(1, cleanup.DisposeCount);
        Assert.False(Directory.Exists(workspace.Root));
    }

    [Fact]
    public void OrphanedWorkspaces_AreSweptAtStartup()
    {
        string root = Path.Combine(_base, "sessions");
        var before = new SessionWorkspaceManager(root);
        SessionWorkspace orphan = before.GetOrCreate("dead-session");
        File.WriteAllText(Path.Combine(orphan.WorkDirectory, "left.txt"), "over");

        // A new manager models a restarted server: every old session is unreachable.
        var after = new SessionWorkspaceManager(root);
        after.SweepOrphans();

        Assert.False(Directory.Exists(orphan.Root));
    }

    private sealed class CountingCleanup : IDisposable
    {
        private int _disposeCount;
        public int DisposeCount => _disposeCount;
        public void Dispose() => System.Threading.Interlocked.Increment(ref _disposeCount);
    }

    private sealed class BlockingCleanup : IDisposable
    {
        private readonly System.Threading.ManualResetEventSlim _entered;
        private readonly System.Threading.ManualResetEventSlim _resume;
        private int _disposeCount;

        public BlockingCleanup(
            System.Threading.ManualResetEventSlim entered,
            System.Threading.ManualResetEventSlim resume)
        {
            _entered = entered;
            _resume = resume;
        }

        public int DisposeCount => _disposeCount;

        public void Dispose()
        {
            System.Threading.Interlocked.Increment(ref _disposeCount);
            _entered.Set();
            _resume.Wait(TimeSpan.FromSeconds(5));
        }
    }
}
