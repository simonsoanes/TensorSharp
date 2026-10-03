// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using TensorAgent.Core.Sandbox;
using TensorAgent.Core.Shell;

namespace TensorAgent.Tests;

/// <summary>
/// The <c>git</c> builtin, and the on-disk format underneath it.
///
/// <para>
/// The tests that matter most here are the two that cross the boundary: one builds a
/// repository with this implementation and hands it to <c>/usr/bin/git</c>, the other
/// builds one with <c>/usr/bin/git</c> and reads it back. Everything else — status codes,
/// diff text, log walking — is this implementation agreeing with itself, which proves
/// only that it is self-consistent. Real git agreeing is what proves the SHA-1 object
/// names, the tree sort order, the index checksum and the commit encoding are right.
/// </para>
/// <para>
/// The shipping code may not start a process; the test host is macOS and may. When the
/// system git is missing the cross-checks skip with a message rather than passing
/// vacuously.
/// </para>
/// </summary>
public sealed class ShellGitTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-git-" + Guid.NewGuid().ToString("N"));
    private readonly string _work;
    private readonly string _temp;
    private readonly string _outside;
    private readonly InProcessShell _shell = new();

    public ShellGitTests()
    {
        _work = Path.Combine(_root, "work");
        _temp = Path.Combine(_root, "tmp");
        _outside = Path.Combine(_root, "outside");
        Directory.CreateDirectory(_work);
        Directory.CreateDirectory(_temp);
        Directory.CreateDirectory(_outside);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { /* the temp tree is best-effort */ }
    }

    private ExecutionPolicy Policy() => new(
        AllowScripts: true,
        AllowNetwork: false,
        WorkRoot: _work,
        ReadableRoots: Array.Empty<string>(),
        TempRoot: _temp)
    { DefaultTimeout = TimeSpan.FromSeconds(60) };

    private ShellContext Context(string? cwd = null, IReadOnlyDictionary<string, string>? env = null) => new()
    {
        WorkingDirectory = cwd ?? _work,
        Policy = Policy(),
        Environment = env ?? new Dictionary<string, string>
        {
            ["GIT_AUTHOR_NAME"] = "Test Author",
            ["GIT_AUTHOR_EMAIL"] = "author@example.com",
            ["GIT_COMMITTER_NAME"] = "Test Author",
            ["GIT_COMMITTER_EMAIL"] = "author@example.com",
        },
    };

    private ExecutionResult Run(string command, ShellContext? context = null)
        => _shell.RunCommandAsync(command, context ?? Context(), CancellationToken.None).GetAwaiter().GetResult();

    private string Out(string command, ShellContext? context = null)
    {
        ExecutionResult result = Run(command, context);
        Assert.True(result.ExitCode == 0, $"`{command}` exited {result.ExitCode}: {result.Stderr}{result.Stdout}");
        return result.Stdout;
    }

    private void Write(string relative, string content)
    {
        string path = Path.Combine(_work, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }

    private string Read(string relative) => File.ReadAllText(Path.Combine(_work, relative));

    // =====================================================================================
    // init, add, commit, log
    // =====================================================================================

    [Fact]
    public void InitAddCommitThenLogShowsTheCommit()
    {
        Assert.Contains("Initialized empty Git repository", Out("git init"), StringComparison.Ordinal);
        Assert.True(Directory.Exists(Path.Combine(_work, ".git", "objects")));
        Assert.True(File.Exists(Path.Combine(_work, ".git", "HEAD")));

        Write("hello.txt", "hello world\n");
        Out("git add hello.txt");
        string commit = Out("git commit -m 'first commit'");
        Assert.Contains("(root-commit)", commit, StringComparison.Ordinal);
        Assert.Contains("first commit", commit, StringComparison.Ordinal);
        Assert.Contains("1 file changed", commit, StringComparison.Ordinal);

        string log = Out("git log");
        Assert.Contains("first commit", log, StringComparison.Ordinal);
        Assert.Contains("Author: Test Author <author@example.com>", log, StringComparison.Ordinal);
        Assert.Matches(@"commit [0-9a-f]{40}", log);

        string oneline = Out("git log --oneline");
        Assert.Single(oneline.TrimEnd('\n').Split('\n'));
        Assert.Contains("first commit", oneline, StringComparison.Ordinal);
    }

    [Fact]
    public void AuthorComesFromRepositoryConfigWhenTheEnvironmentIsSilent()
    {
        var context = Context(env: new Dictionary<string, string>());
        Out("git init", context);
        Out("git config user.name 'Config Person'", context);
        Out("git config user.email 'config@example.com'", context);
        Write("a.txt", "a\n");
        Out("git add a.txt", context);
        Out("git commit -m configured", context);
        Assert.Contains("Author: Config Person <config@example.com>", Out("git log", context), StringComparison.Ordinal);
    }

    [Fact]
    public void ASecondCommitProducesAParentChainThatLogWalks()
    {
        Out("git init");
        Write("a.txt", "one\n");
        Out("git add a.txt && git commit -m first");
        Write("a.txt", "one\ntwo\n");
        Out("git add a.txt && git commit -m second");

        string[] lines = Out("git log --oneline").TrimEnd('\n').Split('\n');
        Assert.Equal(2, lines.Length);
        Assert.Contains("second", lines[0], StringComparison.Ordinal);
        Assert.Contains("first", lines[1], StringComparison.Ordinal);

        // The chain is real: HEAD~1 resolves, and it is the commit `log` printed second.
        string parent = Out("git rev-parse HEAD~1").Trim();
        Assert.StartsWith(parent[..7], lines[1], StringComparison.Ordinal);
        Assert.NotEqual(Out("git rev-parse HEAD").Trim(), parent);

        // -n limits the walk.
        Assert.Single(Out("git log --oneline -n 1").TrimEnd('\n').Split('\n'));
    }

    [Fact]
    public void CommitWithNothingStagedFailsRatherThanRecordingNothing()
    {
        Out("git init");
        Write("a.txt", "a\n");
        Out("git add a.txt && git commit -m first");

        ExecutionResult result = Run("git commit -m 'nothing to do'");
        Assert.Equal(1, result.ExitCode);
        Assert.Contains("nothing to commit", result.Stdout, StringComparison.Ordinal);
        Assert.Single(Out("git log --oneline").TrimEnd('\n').Split('\n'));
    }

    [Fact]
    public void CommitWithoutAMessageIsRefusedBecauseThereIsNoEditor()
    {
        Out("git init");
        Write("a.txt", "a\n");
        Out("git add a.txt");
        ExecutionResult result = Run("git commit");
        Assert.Equal(128, result.ExitCode);
        Assert.Contains("no commit message given", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("no editor", result.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void AddDotStagesEverythingAndAddHonoursGitignore()
    {
        Out("git init");
        Write("keep.txt", "keep\n");
        Write("build/artifact.o", "binary\n");
        Write(".gitignore", "build/\n");
        Out("git add .");

        string staged = Out("git ls-files");
        Assert.Contains("keep.txt", staged, StringComparison.Ordinal);
        Assert.Contains(".gitignore", staged, StringComparison.Ordinal);
        Assert.DoesNotContain("artifact.o", staged, StringComparison.Ordinal);

        // Naming the ignored file directly is an error, not a silent skip.
        ExecutionResult ignored = Run("git add build/artifact.o");
        Assert.Equal(1, ignored.ExitCode);
        Assert.Contains("ignored by one of your .gitignore files", ignored.Stderr, StringComparison.Ordinal);
        Assert.Equal(0, Run("git add -f build/artifact.o").ExitCode);
        Assert.Contains("artifact.o", Out("git ls-files"), StringComparison.Ordinal);
    }

    [Fact]
    public void AddAcceptsGlobsAndReportsAPathspecThatMatchesNothing()
    {
        Out("git init");
        Write("a.md", "a\n");
        Write("b.md", "b\n");
        Write("c.txt", "c\n");
        Out("git add '*.md'");
        string staged = Out("git ls-files");
        Assert.Contains("a.md", staged, StringComparison.Ordinal);
        Assert.Contains("b.md", staged, StringComparison.Ordinal);
        Assert.DoesNotContain("c.txt", staged, StringComparison.Ordinal);

        ExecutionResult missing = Run("git add nosuchfile.txt");
        Assert.Equal(128, missing.ExitCode);
        Assert.Contains("did not match any files", missing.Stderr, StringComparison.Ordinal);
    }

    // =====================================================================================
    // status
    // =====================================================================================

    [Fact]
    public void StatusDistinguishesUntrackedModifiedAndStaged()
    {
        Out("git init");
        Write("tracked.txt", "original\n");
        Out("git add tracked.txt && git commit -m base");

        Write("tracked.txt", "edited\n");        // modified, not staged
        Write("staged.txt", "new\n");
        Out("git add staged.txt");                // staged, new file
        Write("untracked.txt", "loose\n");        // untracked

        string porcelain = Out("git status --porcelain");
        Assert.Contains(" M tracked.txt", porcelain, StringComparison.Ordinal);
        Assert.Contains("A  staged.txt", porcelain, StringComparison.Ordinal);
        Assert.Contains("?? untracked.txt", porcelain, StringComparison.Ordinal);

        string human = Out("git status");
        Assert.Contains("On branch main", human, StringComparison.Ordinal);
        Assert.Contains("Changes to be committed:", human, StringComparison.Ordinal);
        Assert.Contains("new file:   staged.txt", human, StringComparison.Ordinal);
        Assert.Contains("Changes not staged for commit:", human, StringComparison.Ordinal);
        Assert.Contains("modified:   tracked.txt", human, StringComparison.Ordinal);
        Assert.Contains("Untracked files:", human, StringComparison.Ordinal);
        Assert.Contains("untracked.txt", human, StringComparison.Ordinal);

        // A file staged and then edited again shows in both columns.
        Write("staged.txt", "new, then changed\n");
        Assert.Contains("AM staged.txt", Out("git status --porcelain"), StringComparison.Ordinal);
    }

    [Fact]
    public void StatusReportsAStagedDeletionAndACleanTree()
    {
        Out("git init");
        Write("gone.txt", "here\n");
        Out("git add gone.txt && git commit -m base");
        Assert.Contains("nothing to commit, working tree clean", Out("git status"), StringComparison.Ordinal);
        Assert.Equal(string.Empty, Out("git status --porcelain"));

        File.Delete(Path.Combine(_work, "gone.txt"));
        Assert.Contains(" D gone.txt", Out("git status --porcelain"), StringComparison.Ordinal);
        Out("git add gone.txt");
        Assert.Contains("D  gone.txt", Out("git status --porcelain"), StringComparison.Ordinal);
    }

    [Fact]
    public void StatusUntrackedFilesModeIsHonouredRatherThanIgnored()
    {
        Out("git init");
        Write("tracked.txt", "a\n");
        Out("git add tracked.txt && git commit -m base");
        Write("loose.txt", "b\n");

        Assert.Contains("?? loose.txt", Out("git status --porcelain"), StringComparison.Ordinal);
        Assert.Contains("?? loose.txt", Out("git status --porcelain -uall"), StringComparison.Ordinal);
        // -uno is a real option, not a flag to swallow: it must actually suppress them.
        Assert.DoesNotContain("loose.txt", Out("git status --porcelain -uno"), StringComparison.Ordinal);
        Assert.DoesNotContain("Untracked files", Out("git status -uno"), StringComparison.Ordinal);

        ExecutionResult nonsense = Run("git status -unonsense");
        Assert.Equal(128, nonsense.ExitCode);
        Assert.Contains("--untracked-files=nonsense", nonsense.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void StatusOnAFreshRepositorySaysThereAreNoCommitsYet()
    {
        Out("git init");
        Write("a.txt", "a\n");
        string human = Out("git status");
        Assert.Contains("No commits yet", human, StringComparison.Ordinal);
        Assert.Contains("nothing added to commit but untracked files present", human, StringComparison.Ordinal);
    }

    // =====================================================================================
    // diff
    // =====================================================================================

    [Fact]
    public void DiffShowsARealUnifiedDiffAndCachedDiffersFromIt()
    {
        Out("git init");
        Write("f.txt", "alpha\nbeta\ngamma\n");
        Out("git add f.txt && git commit -m base");

        Write("f.txt", "alpha\nBETA\ngamma\n");

        // Unstaged: index (alpha/beta/gamma) vs working tree (alpha/BETA/gamma).
        string unstaged = Out("git diff");
        Assert.Contains("diff --git a/f.txt b/f.txt", unstaged, StringComparison.Ordinal);
        Assert.Contains("--- a/f.txt", unstaged, StringComparison.Ordinal);
        Assert.Contains("+++ b/f.txt", unstaged, StringComparison.Ordinal);
        Assert.Contains("@@ -1,3 +1,3 @@", unstaged, StringComparison.Ordinal);
        Assert.Contains("-beta", unstaged, StringComparison.Ordinal);
        Assert.Contains("+BETA", unstaged, StringComparison.Ordinal);
        Assert.Contains(" alpha", unstaged, StringComparison.Ordinal);

        // --cached is HEAD vs index, which nothing has touched yet.
        Assert.Equal(string.Empty, Out("git diff --cached"));

        Out("git add f.txt");
        string cached = Out("git diff --cached");
        Assert.Contains("+BETA", cached, StringComparison.Ordinal);
        Assert.Contains("-beta", cached, StringComparison.Ordinal);
        // And now the unstaged diff is empty: staging moved the change from one view to
        // the other, which is the whole point of the distinction.
        Assert.Equal(string.Empty, Out("git diff"));
    }

    [Fact]
    public void DiffMarksNewAndDeletedFilesAndMissingTrailingNewlines()
    {
        Out("git init");
        Write("keep.txt", "keep\n");
        Write("drop.txt", "drop\n");
        Out("git add . && git commit -m base");

        File.Delete(Path.Combine(_work, "drop.txt"));
        Write("added.txt", "no trailing newline");
        Out("git add -A");

        string diff = Out("git diff --cached");
        Assert.Contains("new file mode 100644", diff, StringComparison.Ordinal);
        Assert.Contains("deleted file mode 100644", diff, StringComparison.Ordinal);
        Assert.Contains("--- /dev/null", diff, StringComparison.Ordinal);
        Assert.Contains("+++ /dev/null", diff, StringComparison.Ordinal);
        Assert.Contains("\\ No newline at end of file", diff, StringComparison.Ordinal);
    }

    [Fact]
    public void DiffExitCodeReportsWhetherAnythingChanged()
    {
        Out("git init");
        Write("f.txt", "a\n");
        Out("git add f.txt && git commit -m base");
        Assert.Equal(0, Run("git diff --exit-code").ExitCode);
        Write("f.txt", "b\n");
        Assert.Equal(1, Run("git diff --exit-code").ExitCode);
    }

    // =====================================================================================
    // checkout / restore
    // =====================================================================================

    [Fact]
    public void CheckoutRestoresAFileTheTestCorrupted()
    {
        Out("git init");
        Write("src.py", "def good():\n    return 1\n");
        Out("git add src.py && git commit -m good");

        Write("src.py", "this is not python at all");
        Assert.Contains(" M src.py", Out("git status --porcelain"), StringComparison.Ordinal);

        Out("git checkout -- src.py");
        Assert.Equal("def good():\n    return 1\n", Read("src.py"));
        Assert.Equal(string.Empty, Out("git status --porcelain"));
    }

    [Fact]
    public void CheckoutFromARevisionAlsoUpdatesTheIndex()
    {
        Out("git init");
        Write("f.txt", "v1\n");
        Out("git add f.txt && git commit -m v1");
        Write("f.txt", "v2\n");
        Out("git add f.txt && git commit -m v2");

        Out("git checkout HEAD~1 -- f.txt");
        Assert.Equal("v1\n", Read("f.txt"));
        // Restoring from a revision stages it too, so it shows as a staged modification.
        Assert.Contains("M  f.txt", Out("git status --porcelain"), StringComparison.Ordinal);
    }

    [Fact]
    public void RestoreStagedUnstagesWithoutTouchingTheWorkingTree()
    {
        Out("git init");
        Write("f.txt", "base\n");
        Out("git add f.txt && git commit -m base");
        Write("f.txt", "edited\n");
        Out("git add f.txt");
        Assert.Contains("M  f.txt", Out("git status --porcelain"), StringComparison.Ordinal);

        Out("git restore --staged f.txt");
        Assert.Contains(" M f.txt", Out("git status --porcelain"), StringComparison.Ordinal);
        Assert.Equal("edited\n", Read("f.txt"));
    }

    [Fact]
    public void CheckoutOfAPathGitDoesNotKnowFailsWithoutWritingAnything()
    {
        Out("git init");
        Write("f.txt", "x\n");
        Out("git add f.txt && git commit -m base");
        ExecutionResult result = Run("git checkout -- nosuch.txt");
        Assert.Equal(1, result.ExitCode);
        Assert.Contains("did not match any file(s) known to git", result.Stderr, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(_work, "nosuch.txt")));
    }

    [Fact]
    public void SwitchingBranchesIsRefusedRatherThanHalfDone()
    {
        Out("git init");
        Write("f.txt", "x\n");
        Out("git add f.txt && git commit -m base");
        Out("git branch feature");
        ExecutionResult result = Run("git checkout feature");
        Assert.Equal(128, result.ExitCode);
        Assert.Contains("not implemented", result.Stderr, StringComparison.Ordinal);
        // HEAD did not move.
        Assert.Equal("main", Out("git rev-parse --abbrev-ref HEAD").Trim());
    }

    // =====================================================================================
    // show, rm, mv
    // =====================================================================================

    [Fact]
    public void ShowPrintsACommitAndReadsAFileOutOfIt()
    {
        Out("git init");
        Write("f.txt", "committed content\n");
        Out("git add f.txt && git commit -m 'the message'");
        Write("f.txt", "working content\n");

        string show = Out("git show HEAD");
        Assert.Contains("the message", show, StringComparison.Ordinal);
        Assert.Contains("+committed content", show, StringComparison.Ordinal);

        // <rev>:<path> reads the committed version even though the file on disk differs.
        Assert.Equal("committed content\n", Out("git show HEAD:f.txt"));
    }

    [Fact]
    public void RmAndMvUpdateBothTheIndexAndTheWorkingTree()
    {
        Out("git init");
        Write("a.txt", "a\n");
        Write("b.txt", "b\n");
        Out("git add . && git commit -m base");

        Out("git rm a.txt");
        Assert.False(File.Exists(Path.Combine(_work, "a.txt")));
        Assert.Contains("D  a.txt", Out("git status --porcelain"), StringComparison.Ordinal);

        Out("git mv b.txt c.txt");
        Assert.False(File.Exists(Path.Combine(_work, "b.txt")));
        Assert.Equal("b\n", Read("c.txt"));
        string porcelain = Out("git status --porcelain");
        Assert.Contains("D  b.txt", porcelain, StringComparison.Ordinal);
        Assert.Contains("A  c.txt", porcelain, StringComparison.Ordinal);

        Out("git commit -m 'rm and mv'");
        Assert.Equal(string.Empty, Out("git status --porcelain"));
    }

    [Fact]
    public void RmRefusesToDiscardLocalModificationsWithoutForce()
    {
        Out("git init");
        Write("f.txt", "committed\n");
        Out("git add f.txt && git commit -m base");
        Write("f.txt", "edited but not staged\n");

        ExecutionResult result = Run("git rm f.txt");
        Assert.Equal(1, result.ExitCode);
        Assert.Contains("local modifications", result.Stderr, StringComparison.Ordinal);
        Assert.True(File.Exists(Path.Combine(_work, "f.txt")));
        Assert.Equal(0, Run("git rm -f f.txt").ExitCode);
        Assert.False(File.Exists(Path.Combine(_work, "f.txt")));
    }

    [Fact]
    public void AmendReplacesTheTipCommitInsteadOfAddingOne()
    {
        Out("git init");
        Write("f.txt", "content\n");
        Out("git add f.txt && git commit -m 'typo in mesage'");
        string before = Out("git rev-parse HEAD").Trim();

        Out("git commit --amend -m 'typo in message, fixed'");
        Assert.Single(Out("git log --oneline").TrimEnd('\n').Split('\n'));
        Assert.Contains("typo in message, fixed", Out("git log"), StringComparison.Ordinal);
        Assert.NotEqual(before, Out("git rev-parse HEAD").Trim());
    }

    [Fact]
    public void LogOnAnUnbornBranchSaysSoRatherThanPrintingNothing()
    {
        Out("git init");
        ExecutionResult result = Run("git log");
        Assert.Equal(128, result.ExitCode);
        Assert.Contains("does not have any commits yet", result.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void DiffCanBeLimitedToAPathspec()
    {
        Out("git init");
        Write("a.txt", "a\n");
        Write("b.txt", "b\n");
        Out("git add . && git commit -m base");
        Write("a.txt", "a changed\n");
        Write("b.txt", "b changed\n");

        string both = Out("git diff");
        Assert.Contains("a/a.txt", both, StringComparison.Ordinal);
        Assert.Contains("a/b.txt", both, StringComparison.Ordinal);

        string onlyA = Out("git diff -- a.txt");
        Assert.Contains("a/a.txt", onlyA, StringComparison.Ordinal);
        Assert.DoesNotContain("b.txt", onlyA, StringComparison.Ordinal);
    }

    [Fact]
    public void GitignoreHonoursNegationAndNestedFiles()
    {
        Out("git init");
        Write(".gitignore", "*.log\n!keep.log\n");
        Write("sub/.gitignore", "secret.txt\n");
        Write("drop.log", "x\n");
        Write("keep.log", "x\n");
        Write("sub/secret.txt", "x\n");
        Write("sub/public.txt", "x\n");

        string untracked = Out("git status --porcelain");
        Assert.DoesNotContain("drop.log", untracked, StringComparison.Ordinal);
        Assert.Contains("keep.log", untracked, StringComparison.Ordinal);
        Assert.DoesNotContain("secret.txt", untracked, StringComparison.Ordinal);
        Assert.Contains("sub/public.txt", untracked, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void ASymlinkIsRecordedAsItsTargetTextRatherThanFollowed()
    {
        Out("git init");
        Write("real.txt", "the real content\n");
        TestPlatforms.CreateFileSymlink(Path.Combine(_work, "link.txt"), "real.txt");
        Out("git add . && git commit -m 'with a link'");

        // Mode 120000 and a blob whose content is the target path, not the target's bytes.
        string staged = Out("git ls-files --stage");
        Assert.Contains("120000 ", staged, StringComparison.Ordinal);
        Assert.Equal("real.txt", Out("git show HEAD:link.txt"));
        Assert.Equal(string.Empty, Out("git status --porcelain"));
    }

    // =====================================================================================
    // refusals and confinement
    // =====================================================================================

    [Theory]
    [InlineData("git clone https://example.com/repo.git")]
    [InlineData("git fetch origin")]
    [InlineData("git push origin main")]
    [InlineData("git pull")]
    [InlineData("git remote -v")]
    public void NetworkSubcommandsAreRefusedByNameWithTheReason(string command)
    {
        Out("git init");
        ExecutionResult result = Run(command);
        Assert.NotEqual(0, result.ExitCode);
        Assert.Equal(128, result.ExitCode);
        Assert.Contains("needs the network", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("not supported", result.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnimplementedRealSubcommandSaysSoAndATypoDoesNot()
    {
        Out("git init");
        ExecutionResult rebase = Run("git rebase main");
        Assert.Equal(128, rebase.ExitCode);
        Assert.Contains("is a git command, but it is not implemented", rebase.Stderr, StringComparison.Ordinal);

        ExecutionResult typo = Run("git stauts");
        Assert.Equal(1, typo.ExitCode);
        Assert.Contains("is not a git command", typo.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnknownFlagIsAUsageErrorRatherThanASilentNoOp()
    {
        Out("git init");
        ExecutionResult result = Run("git status --invented-flag");
        Assert.Equal(129, result.ExitCode);
        Assert.Contains("unknown option `invented-flag'", result.Stderr, StringComparison.Ordinal);

        ExecutionResult shortFlag = Run("git add -Z x");
        Assert.Equal(129, shortFlag.ExitCode);
        Assert.Contains("unknown switch `Z'", shortFlag.Stderr, StringComparison.Ordinal);

        // A flag real git has but this does not is named, not ignored.
        ExecutionResult unimplemented = Run("git add -p");
        Assert.Equal(128, unimplemented.ExitCode);
        Assert.Contains("not implemented", unimplemented.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void ARepositoryOutsideTheSessionIsRefused()
    {
        // A real repository the session must not see, created by hand so this test does
        // not depend on the system git being installed.
        string outsideGit = Path.Combine(_outside, ".git");
        Directory.CreateDirectory(Path.Combine(outsideGit, "objects"));
        Directory.CreateDirectory(Path.Combine(outsideGit, "refs", "heads"));
        File.WriteAllText(Path.Combine(outsideGit, "HEAD"), "ref: refs/heads/main\n");

        ExecutionResult status = Run($"git -C {_outside} status");
        Assert.NotEqual(0, status.ExitCode);
        Assert.Contains("Permission denied", status.Stderr, StringComparison.Ordinal);

        ExecutionResult init = Run($"git init {Path.Combine(_outside, "fresh")}");
        Assert.NotEqual(0, init.ExitCode);
        Assert.Contains("Permission denied", init.Stderr, StringComparison.Ordinal);
        Assert.False(Directory.Exists(Path.Combine(_outside, "fresh")));

        // And a pathspec cannot reach out of the work tree either.
        Out("git init");
        ExecutionResult add = Run($"git add {Path.Combine(_outside, "anything")}");
        Assert.NotEqual(0, add.ExitCode);
        Assert.Contains("outside repository", add.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void CommandsOutsideARepositorySayThatPlainly()
    {
        ExecutionResult result = Run("git status");
        Assert.Equal(128, result.ExitCode);
        Assert.Contains("not a git repository", result.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void TheBuiltinTableAdvertisesGit()
    {
        Assert.Contains("git", InProcessShell.BuiltinNames);
    }

    // =====================================================================================
    // cross-checks against the real git binary
    // =====================================================================================

    private static readonly string? SystemGit = TestPlatforms.FindExecutable("git");

    private static bool SystemGitAvailable => File.Exists(SystemGit);

    /// <summary>
    /// Runs the real <c>git</c>. Only tests do this: shipping code cannot, which is the
    /// reason the builtin exists at all.
    /// </summary>
    private static (int ExitCode, string Stdout, string Stderr) RealGit(string directory, params string[] args)
    {
        var info = new ProcessStartInfo(SystemGit!)
        {
            WorkingDirectory = directory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (string arg in args)
            info.ArgumentList.Add(arg);

        // A deterministic identity and no user config, so the assertions do not depend on
        // whoever is running the suite.
        string nullDevice = OperatingSystem.IsWindows() ? "NUL" : "/dev/null";
        info.Environment["GIT_CONFIG_GLOBAL"] = nullDevice;
        info.Environment["GIT_CONFIG_SYSTEM"] = nullDevice;
        info.Environment["GIT_AUTHOR_NAME"] = "Real Git";
        info.Environment["GIT_AUTHOR_EMAIL"] = "real@example.com";
        info.Environment["GIT_COMMITTER_NAME"] = "Real Git";
        info.Environment["GIT_COMMITTER_EMAIL"] = "real@example.com";
        info.Environment["GIT_TERMINAL_PROMPT"] = "0";

        using Process process = Process.Start(info)!;
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit(60_000);
        return (process.ExitCode, stdout, stderr);
    }

    [SkippableFact]
    public void RealGitValidatesARepositoryThisImplementationWrote()
    {
        Skip.IfNot(SystemGitAvailable, "the system git is not installed at " + SystemGit + "; cross-check skipped");
        Skip.If(OperatingSystem.IsWindows(), "This cross-check requires Unix executable modes and symbolic links.");

        // Build a repository entirely with the builtin: two commits, a subdirectory, an
        // executable file, and a deletion, so the trees exercise the sort order and modes.
        Out("git init");
        Write("README.md", "# Title\n\nBody.\n");
        Write("src/app.py", "print('hi')\n");
        Write("src/lib/util.py", "X = 1\n");
        Write("scripts/run.sh", "#!/bin/sh\necho hi\n");
        File.SetUnixFileMode(
            Path.Combine(_work, "scripts", "run.sh"),
            UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute | UnixFileMode.GroupRead | UnixFileMode.GroupExecute | UnixFileMode.OtherRead | UnixFileMode.OtherExecute);
        TestPlatforms.CreateFileSymlink(Path.Combine(_work, "latest.md"), "README.md");
        Out("git add .");
        Out("git commit -m 'initial import'");

        Write("src/app.py", "print('hi')\nprint('again')\n");
        File.Delete(Path.Combine(_work, "src", "lib", "util.py"));
        Out("git add -A");
        Out("git commit -m 'second commit'");

        // 1. fsck: every object hashes to its name, every tree is sorted, every link resolves.
        (int fsckCode, string fsckOut, string fsckErr) = RealGit(_work, "fsck", "--strict", "--no-progress");
        Assert.True(fsckCode == 0, $"git fsck exited {fsckCode}:\n{fsckOut}{fsckErr}");
        Assert.DoesNotContain("error", fsckOut + fsckErr, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("missing", fsckOut + fsckErr, StringComparison.OrdinalIgnoreCase);

        // 2. log: real git walks the history we wrote, in order, with our messages and author.
        (int logCode, string logOut, string logErr) = RealGit(_work, "log", "--format=%H|%an|%ae|%s");
        Assert.True(logCode == 0, $"git log exited {logCode}: {logErr}");
        string[] logLines = logOut.TrimEnd('\n').Split('\n');
        Assert.Equal(2, logLines.Length);
        Assert.EndsWith("|Test Author|author@example.com|second commit", logLines[0], StringComparison.Ordinal);
        Assert.EndsWith("|Test Author|author@example.com|initial import", logLines[1], StringComparison.Ordinal);

        // The object names agree exactly: same content, same SHA-1, same tree encoding.
        Assert.Equal(Out("git rev-parse HEAD").Trim(), logLines[0].Split('|')[0]);
        Assert.Equal(Out("git rev-parse HEAD~1").Trim(), logLines[1].Split('|')[0]);

        // 3. status: real git reads our index and agrees the tree is clean.
        (int statusCode, string statusOut, string statusErr) = RealGit(_work, "status", "--porcelain");
        Assert.True(statusCode == 0, $"git status exited {statusCode}: {statusErr}");
        Assert.Equal(string.Empty, statusOut.Trim());

        // 4. The recorded modes and paths survive: the executable bit and the nesting.
        (int lsCode, string lsOut, _) = RealGit(_work, "ls-files", "--stage");
        Assert.Equal(0, lsCode);
        Assert.Contains("100755 ", lsOut, StringComparison.Ordinal);
        Assert.Contains("scripts/run.sh", lsOut, StringComparison.Ordinal);
        Assert.Contains("100644 ", lsOut, StringComparison.Ordinal);
        Assert.Contains("src/app.py", lsOut, StringComparison.Ordinal);
        // The symlink is a 120000 blob holding its target text, not a copy of the target.
        Assert.Contains("120000 ", lsOut, StringComparison.Ordinal);
        Assert.Equal("README.md", RealGit(_work, "show", "HEAD:latest.md").Stdout);
        // src/lib/util.py was deleted in the second commit, so its subtree is gone too.
        Assert.DoesNotContain("util.py", lsOut, StringComparison.Ordinal);

        // 5. The content real git checks out of our commit is the content we put in.
        (int showCode, string showOut, _) = RealGit(_work, "show", "HEAD:src/app.py");
        Assert.Equal(0, showCode);
        Assert.Equal("print('hi')\nprint('again')\n", showOut);

        // 6. Its own diff of the first commit against the second finds the change we made.
        (int diffCode, string diffOut, _) = RealGit(_work, "diff", "HEAD~1", "HEAD", "--", "src/app.py");
        Assert.Equal(0, diffCode);
        Assert.Contains("+print('again')", diffOut, StringComparison.Ordinal);

        // 7. The reflogs we wrote are real reflogs: `git reflog` is the way out of a mess,
        //    so a history with no reflog would strand an agent that made one.
        (int reflogCode, string reflogOut, _) = RealGit(_work, "reflog");
        Assert.Equal(0, reflogCode);
        Assert.Contains("commit (initial): initial import", reflogOut, StringComparison.Ordinal);
        Assert.Contains("commit: second commit", reflogOut, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void ARepositoryRealGitWroteIsReadableByThisImplementation()
    {
        Skip.IfNot(SystemGitAvailable, "the system git is not installed at " + SystemGit + "; cross-check skipped");

        Assert.Equal(0, RealGit(_work, "init", "-b", "main").ExitCode);
        Write("a.txt", "first line\n");
        Write("nested/deep/b.txt", "nested content\n");
        Assert.Equal(0, RealGit(_work, "add", ".").ExitCode);
        Assert.Equal(0, RealGit(_work, "commit", "-m", "real git commit one").ExitCode);

        Write("a.txt", "first line\nsecond line\n");
        Assert.Equal(0, RealGit(_work, "add", "a.txt").ExitCode);
        Assert.Equal(0, RealGit(_work, "commit", "-m", "real git commit two").ExitCode);

        string realHead = RealGit(_work, "rev-parse", "HEAD").Stdout.Trim();

        // The builtin reads the refs, the index and the objects real git wrote.
        Assert.Equal(realHead, Out("git rev-parse HEAD").Trim());
        string log = Out("git log --oneline");
        Assert.Contains("real git commit two", log, StringComparison.Ordinal);
        Assert.Contains("real git commit one", log, StringComparison.Ordinal);
        Assert.Contains("Author: Real Git <real@example.com>", Out("git log"), StringComparison.Ordinal);

        // Its index parses (checksum, entry layout, paths) and reports a clean tree.
        Assert.Equal(string.Empty, Out("git status --porcelain"));
        string tracked = Out("git ls-files");
        Assert.Contains("a.txt", tracked, StringComparison.Ordinal);
        Assert.Contains("nested/deep/b.txt", tracked, StringComparison.Ordinal);

        // Blobs inside real git's trees inflate correctly.
        Assert.Equal("nested content\n", Out("git show HEAD:nested/deep/b.txt"));

        // And the builtin can carry the history forward: a commit on top of real git's.
        Write("c.txt", "added by the builtin\n");
        Out("git add c.txt");
        Out("git commit -m 'builtin on top of real git'");
        (int fsckCode, string fsckOut, string fsckErr) = RealGit(_work, "fsck", "--strict", "--no-progress");
        Assert.True(fsckCode == 0, $"git fsck exited {fsckCode}:\n{fsckOut}{fsckErr}");
        Assert.Contains("builtin on top of real git", RealGit(_work, "log", "--format=%s").Stdout, StringComparison.Ordinal);
        Assert.Equal(realHead, Out("git rev-parse HEAD~1").Trim());
    }

    [SkippableFact]
    public void PackedObjectsWrittenByRealGitAreReadable()
    {
        Skip.IfNot(SystemGitAvailable, "the system git is not installed at " + SystemGit + "; cross-check skipped");

        Assert.Equal(0, RealGit(_work, "init", "-b", "main").ExitCode);

        // Enough related revisions of one file that `gc` has a reason to store deltas,
        // which is the pack path that a loose-only reader would silently fail on.
        for (int i = 0; i < 12; i++)
        {
            var body = new System.Text.StringBuilder();
            for (int line = 0; line < 200; line++)
                body.Append("line ").Append(line).Append(" revision ").Append(line == i ? i : 0).Append('\n');
            Write("big.txt", body.ToString());
            Write($"file{i}.txt", $"content {i}\n");
            Assert.Equal(0, RealGit(_work, "add", "-A").ExitCode);
            Assert.Equal(0, RealGit(_work, "commit", "-m", $"revision {i}").ExitCode);
        }

        // Pack everything and delete the loose copies, so the only way to read the history
        // is through the packfile reader.
        Assert.Equal(0, RealGit(_work, "gc", "--aggressive", "--prune=now").ExitCode);
        Assert.True(
            Directory.GetFiles(Path.Combine(_work, ".git", "objects", "pack"), "*.pack").Length > 0,
            "git gc produced no packfile, so this test would not exercise the pack reader");
        int looseBuckets = Directory.GetDirectories(Path.Combine(_work, ".git", "objects"))
            .Count(d => Path.GetFileName(d).Length == 2);
        Assert.True(looseBuckets == 0, $"{looseBuckets} loose object buckets survived gc; the pack path would not be exercised");

        // The whole history walks out of the pack, deltas and all.
        string[] lines = Out("git log --oneline").TrimEnd('\n').Split('\n');
        Assert.Equal(12, lines.Length);
        Assert.Contains("revision 11", lines[0], StringComparison.Ordinal);
        Assert.Contains("revision 0", lines[^1], StringComparison.Ordinal);

        // A delta-compressed blob inflates to exactly what real git says it is.
        string fromBuiltin = Out("git show HEAD~5:big.txt");
        Assert.Equal(RealGit(_work, "show", "HEAD~5:big.txt").Stdout, fromBuiltin);
        Assert.Contains("revision 6", fromBuiltin, StringComparison.Ordinal);

        // Status against a fully packed repository is clean, and a new commit on top of a
        // packed history is written loose and still validates.
        Assert.Equal(string.Empty, Out("git status --porcelain"));
        Write("after-pack.txt", "written after gc\n");
        Out("git add after-pack.txt && git commit -m 'after the pack'");
        (int fsckCode, string fsckOut, string fsckErr) = RealGit(_work, "fsck", "--strict", "--no-progress");
        Assert.True(fsckCode == 0, $"git fsck exited {fsckCode}:\n{fsckOut}{fsckErr}");
    }

    [SkippableFact]
    public void RealGitAcceptsAPatchThisImplementationProduced()
    {
        Skip.IfNot(SystemGitAvailable, "the system git is not installed at " + SystemGit + "; cross-check skipped");

        Out("git init");
        Write("f.txt", "one\ntwo\nthree\nfour\nfive\n");
        Out("git add f.txt && git commit -m base");
        Write("f.txt", "one\nTWO\nthree\nfour\nfive\nsix\n");

        string patch = Out("git diff");
        string patchFile = Path.Combine(_temp, "change.patch");
        File.WriteAllText(patchFile, patch);

        // Revert the working tree, then let real git apply our patch to it.
        Out("git checkout -- f.txt");
        Assert.Equal("one\ntwo\nthree\nfour\nfive\n", Read("f.txt"));

        (int applyCode, _, string applyErr) = RealGit(_work, "apply", patchFile);
        Assert.True(applyCode == 0, $"git apply rejected our patch ({applyCode}): {applyErr}\n--- patch ---\n{patch}");
        Assert.Equal("one\nTWO\nthree\nfour\nfive\nsix\n", Read("f.txt"));
    }
}
