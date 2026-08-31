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

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the one security boundary the whole skills feature rests on.
///
/// <para>
/// A skill is untrusted content. Somebody uploads a ZIP, or the operator points
/// <c>--skills-dir</c> at a tree pulled off GitHub, and the model is then invited to
/// name files inside it — so every path that reaches the filesystem is a string an
/// attacker wrote, either as a ZIP entry name, as a link in a <c>SKILL.md</c> body, or
/// as a <c>skills_read</c> argument the model was talked into producing. If any one of
/// those forms slips through, uploading a skill becomes an arbitrary host file read,
/// and the failure is completely silent: the request succeeds and returns the file.
/// </para>
/// <para>
/// The cases below are the three tiers the guard claims to close, tested separately
/// because closing only the obvious tier is what makes a guard like this look safe
/// while it is not. <b>Lexical</b> — <c>..</c>, <c>/etc/passwd</c>, <c>~</c>,
/// <c>C:\</c>, <c>\\server\share</c>, an embedded NUL, a segment Windows will silently
/// strip a trailing dot or space from. <b>Canonical</b> — the sibling-directory case,
/// where a root of <c>.../skills</c> must reject a path landing in
/// <c>.../skills-evil</c>; a plain <c>StartsWith</c> containment check passes every
/// other test in this file and fails that one, which is precisely why it is here.
/// <b>Symbolic</b> — a real symlink out of the skill directory, both as the leaf and,
/// the case a leaf-only check waves straight through, as an intermediate directory.
/// </para>
/// </summary>
public class SkillPathGuardTests : IDisposable
{
    private readonly string _baseDir;
    private readonly string _skillRoot;

    public SkillPathGuardTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-pathguard-" + Guid.NewGuid().ToString("N"));
        _skillRoot = Path.Combine(_baseDir, "skills", "pdf");
        Directory.CreateDirectory(_skillRoot);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- helpers -----------------------------------------------------------

    /// <summary>
    /// Create a symbolic link, reporting failure instead of throwing. Windows refuses
    /// symlink creation to an unprivileged process outside Developer Mode, and some
    /// CI filesystems refuse them outright; the cases that need one skip themselves
    /// there rather than turning an environment restriction into a red test.
    /// </summary>
    private static bool TryLinkDirectory(string path, string target)
    {
        try
        {
            Directory.CreateSymbolicLink(path, target);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            return false;
        }
    }

    private static bool TryLinkFile(string path, string target)
    {
        try
        {
            File.CreateSymbolicLink(path, target);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            return false;
        }
    }

    // ---- lexical escapes ---------------------------------------------------

    [Theory]
    [InlineData("../secrets.txt")]
    [InlineData("references/../../secrets.txt")]
    [InlineData("..")]
    [InlineData("../../../../etc/passwd")]
    public void TryResolve_ATraversalSegment_IsRefused(string path)
    {
        Assert.False(SkillPathGuard.TryResolve(_skillRoot, path, out string? full, out string? error));

        // "escapes", not "not found": the model — or the person who wrote the SKILL.md
        // it is reading — must be told it may not look there, not that the file is
        // missing, or it will retry with a different spelling of the same attack.
        Assert.Contains("escapes the skill directory", error, StringComparison.Ordinal);
        Assert.Null(full);
    }

    [Fact]
    public void TryResolve_ARootedPath_IsRefused()
    {
        Assert.False(SkillPathGuard.TryResolve(_skillRoot, "/etc/passwd", out _, out string? error));
        Assert.Contains("absolute path", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TryResolve_AHomeRelativePath_IsRefused()
    {
        // '~' is never expanded here, so letting it through would merely create a
        // directory literally called "~" — but a skill asking for it is asking for the
        // user's home, and answering that request at all is wrong.
        Assert.False(SkillPathGuard.TryResolve(_skillRoot, "~/.ssh/id_rsa", out _, out string? error));
        Assert.Contains("absolute path", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TryResolve_AUncPath_IsRefused()
    {
        // Refused on every platform, not only Windows: a skill authored to attack a
        // Windows host must not be quietly accepted while running on Linux, where the
        // same name would resolve to an ordinary relative directory.
        Assert.False(SkillPathGuard.TryResolve(_skillRoot, "//server/share/secret.txt", out _, out string? error));
        Assert.Contains("UNC path", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TryResolve_ADriveQualifiedPath_IsRefused()
    {
        Assert.False(SkillPathGuard.TryResolve(_skillRoot, "C:\\Windows\\win.ini", out _, out string? error));
        Assert.Contains("drive-qualified", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TryResolve_ANulByte_IsRefused()
    {
        // The classic truncation trick: everything after the NUL is dropped by a C
        // string API, so "safe.md\0/../../etc/passwd" passes a managed suffix check and
        // opens something else entirely.
        Assert.False(SkillPathGuard.TryResolve(_skillRoot, "references/api\0.md", out _, out string? error));
        Assert.Contains("NUL character", error, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("scripts/run.py.")]
    [InlineData("scripts /run.py")]
    public void TryResolve_ASegmentEndingInADotOrASpace_IsRefused(string path)
    {
        // Windows strips both when the file is opened, so "run.py." and "run.py" name
        // one file and compare as two strings. Refusing the shape is cheaper than
        // reasoning about which of the two an allowlist matched.
        Assert.False(SkillPathGuard.TryResolve(_skillRoot, path, out _, out string? error));
        Assert.Contains("ending in", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TryResolve_ARelativeSkillRoot_IsRefused()
    {
        // A relative root would be resolved against the process working directory,
        // which is not the skill's directory and is not something the caller controls.
        Assert.False(SkillPathGuard.TryResolve("skills/pdf", "SKILL.md", out _, out string? error));
        Assert.Contains("is not an absolute path", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TryResolve_NoSkillRoot_IsRefused()
    {
        Assert.False(SkillPathGuard.TryResolve("   ", "SKILL.md", out _, out string? error));
        Assert.Contains("no root directory", error, StringComparison.Ordinal);
    }

    // ---- accepted spellings ------------------------------------------------

    [Fact]
    public void TryResolve_BackslashSeparators_AreAccepted()
    {
        // A skill authored on Windows is routinely read on Linux and vice versa, so
        // the separator a SKILL.md happens to use must not decide whether it loads.
        Assert.True(SkillPathGuard.TryResolve(_skillRoot, "references\\api.md", out string? full, out string? error), error);
        Assert.Equal("references/api.md", SkillPathGuard.ToSkillRelative(_skillRoot, full!));
    }

    [Fact]
    public void TryResolve_ALeadingDotSlash_IsStripped()
    {
        // "./scripts/extract.py" is idiomatic in a Markdown link and means exactly the
        // same file as "scripts/extract.py".
        Assert.True(SkillPathGuard.TryResolve(_skillRoot, "./scripts/extract.py", out string? full, out string? error), error);
        Assert.Equal("scripts/extract.py", SkillPathGuard.ToSkillRelative(_skillRoot, full!));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(".")]
    [InlineData("./")]
    public void TryResolve_AnEmptyPath_MeansTheSkillDirectoryItself(string? path)
    {
        Assert.True(SkillPathGuard.TryResolve(_skillRoot, path, out string? full, out string? error), error);
        Assert.Equal(SkillPathGuard.NormalizeDirectory(_skillRoot), full);
    }

    [Fact]
    public void ToSkillRelative_AlwaysEmitsForwardSlashes()
    {
        // What the model is shown has to match what the skill author wrote, whichever
        // host the server runs on — the path comes straight back as a skills_read
        // argument, and a backslash there would not match the SKILL.md's own text.
        string nested = Path.Combine(_skillRoot, "references", "deep", "api.md");

        string relative = SkillPathGuard.ToSkillRelative(_skillRoot, nested);

        Assert.Equal("references/deep/api.md", relative);
        Assert.DoesNotContain('\\', relative);
    }

    // ---- containment -------------------------------------------------------

    [Fact]
    public void IsUnder_TheRootItself_IsUnderTheRoot()
    {
        // An empty relative path resolves to the root, and a read of the skill
        // directory itself has to be in bounds or every "" path would be refused.
        Assert.True(SkillPathGuard.IsUnder(_skillRoot, _skillRoot));
        Assert.True(SkillPathGuard.IsUnder(_skillRoot, _skillRoot + Path.DirectorySeparatorChar));
    }

    [Fact]
    public void IsUnder_ASiblingSharingTheRootsNamePrefix_IsNotUnderIt()
    {
        // THE classic containment bug. With root ".../skills", the path
        // ".../skills-evil/SKILL.md" starts with the root's characters, so a plain
        // string-prefix compare calls it contained — and an attacker who can create a
        // sibling directory (or a symlink named to produce one) reads out of it. The
        // separator has to be part of the comparison.
        string root = Path.Combine(_baseDir, "skills");
        string sibling = Path.Combine(_baseDir, "skills-evil", "SKILL.md");

        Assert.False(SkillPathGuard.IsUnder(root, sibling));
        Assert.True(SkillPathGuard.IsUnder(root, Path.Combine(root, "pdf", "SKILL.md")));
    }

    // ---- existence ---------------------------------------------------------

    [Fact]
    public void TryResolveExistingFile_ADirectory_SaysSoRatherThanNotFound()
    {
        Directory.CreateDirectory(Path.Combine(_skillRoot, "references"));

        Assert.False(SkillPathGuard.TryResolveExistingFile(_skillRoot, "references", out string? full, out string? error));

        // The distinction is what lets the model correct itself in one step: "that is a
        // directory" tells it to name a file inside, "does not exist" tells it to give up.
        Assert.Contains("is a directory, not a file", error, StringComparison.Ordinal);
        Assert.Null(full);
    }

    [Fact]
    public void TryResolveExistingFile_AMissingFile_SaysItDoesNotExist()
    {
        Assert.False(SkillPathGuard.TryResolveExistingFile(_skillRoot, "references/missing.md", out _, out string? error));

        // Deliberately NOT the security message: a skill that simply forgot to ship a
        // file it links to is a bug in the skill, not an attack, and reporting it as an
        // escape would send the operator hunting for an intrusion.
        Assert.Contains("does not exist in this skill", error, StringComparison.Ordinal);
        Assert.DoesNotContain("escapes", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TryResolveExistingFile_ARealFile_Resolves()
    {
        Directory.CreateDirectory(Path.Combine(_skillRoot, "references"));
        File.WriteAllText(Path.Combine(_skillRoot, "references", "api.md"), "# API\n");

        Assert.True(SkillPathGuard.TryResolveExistingFile(_skillRoot, "references/api.md", out string? full, out string? error), error);
        Assert.True(File.Exists(full));
    }

    // ---- symbolic escapes --------------------------------------------------

    [Fact]
    public void TryResolve_ASymlinkedLeafPointingOutOfTheSkill_IsRefused()
    {
        // Path.GetFullPath does not follow links, so this shape survives both the
        // lexical and the canonical checks: the name "leak.txt" contains nothing
        // suspicious and resolves to a path directly inside the skill directory.
        string outside = Path.Combine(_baseDir, "outside");
        Directory.CreateDirectory(outside);
        string secret = Path.Combine(outside, "secret.txt");
        File.WriteAllText(secret, "an API key the skill must not reach");

        if (!TryLinkFile(Path.Combine(_skillRoot, "leak.txt"), secret))
            return;   // this platform refuses symlinks; nothing to assert here

        Assert.False(SkillPathGuard.TryResolve(_skillRoot, "leak.txt", out string? full, out string? error));
        Assert.Contains("symbolic link", error, StringComparison.Ordinal);
        Assert.Null(full);
    }

    [Fact]
    public void TryResolve_ASymlinkedIntermediateDirectory_IsRefused()
    {
        // The case a leaf-only symlink check misses entirely, and the reason the guard
        // walks component by component: with "refs -> <outside>", the leaf
        // "refs/secret.txt" is an ordinary file and is not itself a link, so resolving
        // only the last component finds nothing to follow and lets the read through.
        // (The real-world shape is "refs -> /etc" and a read of "refs/passwd"; a
        // directory created here is the same shape and works on every platform.)
        string outside = Path.Combine(_baseDir, "outside");
        Directory.CreateDirectory(outside);
        File.WriteAllText(Path.Combine(outside, "secret.txt"), "an API key the skill must not reach");

        if (!TryLinkDirectory(Path.Combine(_skillRoot, "refs"), outside))
            return;   // this platform refuses symlinks; nothing to assert here

        Assert.False(SkillPathGuard.TryResolve(_skillRoot, "refs/secret.txt", out string? full, out string? error));
        Assert.Contains("symbolic link", error, StringComparison.Ordinal);
        Assert.Null(full);
    }

    [Fact]
    public void TryResolve_ASymlinkIntoASiblingSkillDirectory_IsRefused()
    {
        // The containment bug and the symlink bug in one shape: "evil -> ../pdf-evil"
        // lands on a directory whose full path begins with the root's own characters.
        string sibling = _skillRoot + "-evil";
        Directory.CreateDirectory(sibling);
        File.WriteAllText(Path.Combine(sibling, "SKILL.md"), "not this skill's file");

        if (!TryLinkDirectory(Path.Combine(_skillRoot, "evil"), sibling))
            return;   // this platform refuses symlinks; nothing to assert here

        Assert.False(SkillPathGuard.TryResolve(_skillRoot, "evil/SKILL.md", out _, out string? error));
        Assert.Contains("symbolic link", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TryResolve_ASymlinkThatStaysInsideTheSkill_IsAccepted()
    {
        // The control for the three cases above: the guard rejects links that LEAVE the
        // skill, not links as such. A skill that symlinks one of its own files into
        // scripts/ is doing nothing wrong, and refusing it would break real bundles.
        Directory.CreateDirectory(Path.Combine(_skillRoot, "references"));
        string real = Path.Combine(_skillRoot, "references", "api.md");
        File.WriteAllText(real, "# API\n");

        if (!TryLinkFile(Path.Combine(_skillRoot, "api-alias.md"), real))
            return;   // this platform refuses symlinks; nothing to assert here

        Assert.True(SkillPathGuard.TryResolve(_skillRoot, "api-alias.md", out string? full, out string? error), error);
        Assert.NotNull(full);
    }
}
