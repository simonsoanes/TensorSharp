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
using System.Diagnostics;
using System.IO;
using System.Linq;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// What a trivial shell command is allowed to cost.
///
/// <para>
/// Every <c>shell</c> call walks the session's working directory twice — once before,
/// for the "what was here already" snapshot, and once after, to capture what the
/// command produced. A real session installs packages, so that directory ends up
/// holding a <c>node_modules</c> or a <c>.venv</c>: tens of thousands of files that
/// <c>echo hi</c> has no business looking at. Filtering them out AFTER enumeration
/// saved nothing, because enumeration had already stat'd every one; measured on a
/// 20,000-file install, the two walks cost 182 ms per command against 0.02 ms on an
/// empty workspace, and <c>echo hi</c> went from 213 ms to 383 ms end to end.
/// </para>
/// <para>
/// The fix is to prune those directories DURING the walk, from one shared list, and
/// these tests are what stop it regressing: the first pins the cost to be flat in the
/// size of the junk, and the rest pin the behaviour — that nothing outside the pruned
/// directories changed, and that the two walks prune identically. A snapshot that
/// descended where the capture does not would report every file under it as newly
/// produced and hand the user thousands of download links.
/// </para>
/// </summary>
public class WorkspaceWalkCostTests : IDisposable
{
    private readonly string _base;

    public WorkspaceWalkCostTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-walkcost-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private SessionWorkspace Workspace(string id) =>
        new SessionWorkspaceManager(Path.Combine(_base, "sessions")).GetOrCreate(id);

    private CodeArtifactStore Store(string name)
    {
        string root = Path.Combine(_base, name);
        Directory.CreateDirectory(root);
        return new CodeArtifactStore(root);
    }

    /// <summary>An installed dependency tree: many files, a few per directory.</summary>
    private static void FillNodeModules(string workDirectory, int files)
    {
        int perDirectory = 10;
        for (int i = 0; i < files; i++)
        {
            string directory = Path.Combine(
                workDirectory, "node_modules", "pkg-" + (i / perDirectory).ToString("D5"));
            if (i % perDirectory == 0)
                Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "f" + (i % perDirectory) + ".js"), "0");
        }
    }

    /// <summary>
    /// One command's fixed filesystem cost: the snapshot before it and the capture
    /// after it, exactly as ShellRunner pairs them. Best of several, because the
    /// question is what the work COSTS, not what the machine was doing at the time.
    /// </summary>
    private static double MillisecondsPerCall(SessionWorkspace workspace, CodeArtifactStore store)
    {
        double best = double.MaxValue;
        for (int i = 0; i < 6; i++)
        {
            var sw = Stopwatch.StartNew();
            Dictionary<string, (long Length, DateTime WriteTime)> before = workspace.SnapshotWorkFiles();
            store.Capture(
                Guid.NewGuid().ToString("N").Substring(0, 16),
                workspace.WorkDirectory,
                (id, relative, full) => full,
                out _,
                relative => workspace.IsUnchangedSince(before, relative));
            sw.Stop();
            if (i > 0)
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    [Fact]
    public void ATrivialCommandsCost_DoesNotGrowWithTheInstalledDependencies()
    {
        // Twenty times the junk. If the walks are proportional to the working
        // directory, so is this number — which is the regression being guarded.
        const int Small = 1_000;
        const int Big = 20_000;

        SessionWorkspace small = Workspace("small");
        File.WriteAllText(Path.Combine(small.WorkDirectory, "main.py"), "print(1)\n");
        FillNodeModules(small.WorkDirectory, Small);

        SessionWorkspace big = Workspace("big");
        File.WriteAllText(Path.Combine(big.WorkDirectory, "main.py"), "print(1)\n");
        FillNodeModules(big.WorkDirectory, Big);

        double smallCost = MillisecondsPerCall(small, Store("small-artifacts"));
        double bigCost = MillisecondsPerCall(big, Store("big-artifacts"));

        // Generous on purpose: a pruned walk makes both of these microseconds, so the
        // constant swallows every amount of machine noise a CI box can produce, while
        // an unpruned walk lands twenty times apart and cannot fit under it. Measured
        // unpruned on this shape: 9 ms and 182 ms.
        double budget = (smallCost * 4) + 20.0;
        Assert.True(
            bigCost <= budget,
            $"a trivial command's filesystem cost is growing with the working directory: "
            + $"{Small} junk files cost {smallCost:0.000} ms, {Big} cost {bigCost:0.000} ms "
            + $"(budget {budget:0.000} ms). Prune the directory during the walk, in "
            + "WorkspaceScan, rather than filtering paths after it.");
    }

    [Fact]
    public void APrunedDirectory_IsInvisibleToBothWalks_AndRealOutputIsNot()
    {
        SessionWorkspace workspace = Workspace("both");

        // One file inside every directory the walk prunes, taken from the list itself
        // so a name added there is covered here without anyone remembering to.
        foreach ((string name, bool _) in WorkspaceScan.PrunedNames())
        {
            string directory = Path.Combine(workspace.WorkDirectory, name);
            Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "junk.bin"), "junk");
        }
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "report.pdf"), "%PDF-1.4");

        Dictionary<string, (long Length, DateTime WriteTime)> snapshot = workspace.SnapshotWorkFiles();
        Assert.Equal(new[] { "report.pdf" }, snapshot.Keys.OrderBy(k => k, StringComparer.Ordinal));

        // And the capture agrees — it must, or the file the snapshot never saw looks new.
        IReadOnlyList<CodeArtifact> captured = Store("both-artifacts").Capture(
            "run0000000000000a", workspace.WorkDirectory, (id, relative, full) => full, out _);
        Assert.Equal(new[] { "report.pdf" }, captured.Select(a => a.Path).OrderBy(p => p, StringComparer.Ordinal));

        Assert.Equal(new[] { "report.pdf" }, workspace.ListFiles().Select(f => f.Path));
    }

    [Fact]
    public void ANestedDependencyTree_IsPrunedToo_AndIsNotOfferedAsOutput()
    {
        // The shape a session actually produces: `mkdir app && cd app && npm install`.
        // Nothing here is at the top level, which is the only place the old
        // path-shaped filter looked — so every one of these came back as a download.
        SessionWorkspace workspace = Workspace("nested");
        string nested = Path.Combine(workspace.WorkDirectory, "app", "node_modules", "left-pad");
        Directory.CreateDirectory(nested);
        File.WriteAllText(Path.Combine(nested, "index.js"), "module.exports=0");
        Directory.CreateDirectory(Path.Combine(workspace.WorkDirectory, "app"));
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "app", "server.js"), "listen()");

        Assert.Equal(
            new[] { "app/server.js" },
            workspace.SnapshotWorkFiles().Keys.OrderBy(k => k, StringComparer.Ordinal));

        IReadOnlyList<CodeArtifact> captured = Store("nested-artifacts").Capture(
            "run0000000000000b", workspace.WorkDirectory, (id, relative, full) => full, out _);
        Assert.Equal(new[] { "app/server.js" }, captured.Select(a => a.Path));

        Assert.True(CodeArtifactStore.IsRuntimeJunk("app/node_modules/left-pad/index.js"));
    }

    [Fact]
    public void ADirectoryCalledLibrary_IsOnlyHomeFalloutAtTheRoot()
    {
        // "Library" is macOS HOME fallout where HOME is the working directory, and an
        // ordinary English word anywhere else. A model asked to sort documents into a
        // library must still get its files back.
        SessionWorkspace workspace = Workspace("library");
        Directory.CreateDirectory(Path.Combine(workspace.WorkDirectory, "Library", "Caches"));
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "Library", "Caches", "x.bin"), "cache");
        Directory.CreateDirectory(Path.Combine(workspace.WorkDirectory, "docs", "Library"));
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "docs", "Library", "index.md"), "# books");

        Assert.Equal(
            new[] { "docs/Library/index.md" },
            workspace.SnapshotWorkFiles().Keys.OrderBy(k => k, StringComparer.Ordinal));
        Assert.False(CodeArtifactStore.IsRuntimeJunk("docs/Library/index.md"));
        Assert.True(CodeArtifactStore.IsRuntimeJunk("Library/Caches/x.bin"));
    }

    [Fact]
    public void ASymlinkLoopInTheWorkspace_DoesNotMultiplyTheWalk()
    {
        if (OperatingSystem.IsWindows())
            return; // creating a symbolic link needs a privilege the test host may not have

        // `ln -s . loop` is one ordinary command inside a directory the model is
        // allowed to write. A walk that follows it re-walks the workspace once per
        // level until the kernel's symlink limit stops it: measured at 3,747 entries
        // and 55 ms from a working directory holding a single file.
        SessionWorkspace workspace = Workspace("loop");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "main.py"), "print(1)\n");
        File.CreateSymbolicLink(Path.Combine(workspace.WorkDirectory, "loop"), workspace.WorkDirectory);

        Assert.Equal(
            new[] { "main.py" },
            workspace.SnapshotWorkFiles().Keys.OrderBy(k => k, StringComparer.Ordinal));
    }

    [Fact]
    public void AHiddenFile_IsStillPartOfTheBeforePicture()
    {
        // Pruning is about directories, not about dotfiles: the snapshot has always
        // seen `.env`, and a file it cannot see is a file the capture calls new.
        SessionWorkspace workspace = Workspace("hidden");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, ".env"), "KEY=1");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "app.py"), "x");

        Assert.Equal(
            new[] { ".env", "app.py" },
            workspace.SnapshotWorkFiles().Keys.OrderBy(k => k, StringComparer.Ordinal));
    }

    /// <summary>A store that already holds <paramref name="runs"/> runs of 20 files.</summary>
    private string SeededStoreRoot(string name, int runs)
    {
        string root = Path.Combine(_base, name);
        for (int r = 0; r < runs; r++)
        {
            string directory = Path.Combine(root, "seed" + r.ToString("D6"));
            Directory.CreateDirectory(directory);
            for (int i = 0; i < 20; i++)
                File.WriteAllText(Path.Combine(directory, "f" + i + ".bin"), "0123456789");
        }
        Directory.CreateDirectory(root);
        return root;
    }

    [Fact]
    public void KeepingAnOutputFile_DoesNotCostTheWholeArtifactStore()
    {
        // The budget check used to re-measure every run in the store on every command
        // that produced a file: 27.5 ms with 500 runs held, and growing, since the store
        // is bounded in bytes rather than in files. Counting it once and carrying the
        // total forward makes the check arithmetic; this pins that it stays so.
        double Cost(string name, int priorRuns)
        {
            var store = new CodeArtifactStore(SeededStoreRoot(name, priorRuns));
            SessionWorkspace workspace = Workspace(name + "-ws");
            double best = double.MaxValue;
            for (int i = 0; i < 12; i++)
            {
                File.WriteAllText(
                    Path.Combine(workspace.WorkDirectory, "out.txt"), new string('x', 64 + i));
                var sw = Stopwatch.StartNew();
                store.Capture(
                    Guid.NewGuid().ToString("N").Substring(0, 16), workspace.WorkDirectory,
                    (id, relative, full) => full, out _);
                sw.Stop();
                // The first call is allowed to count the store; it is the only one.
                if (i > 0)
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            return best;
        }

        double empty = Cost("budget-empty", 0);
        double full = Cost("budget-full", 400);

        Assert.True(
            full <= (empty * 4) + 10.0,
            $"capturing one file costs more as the artifact store fills: {empty:0.000} ms against "
            + $"an empty store, {full:0.000} ms against one holding 400 runs. The store's total is "
            + "meant to be carried forward, not re-measured from disk on every call.");
    }
}
