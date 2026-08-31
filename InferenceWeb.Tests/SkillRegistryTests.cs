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

namespace InferenceWeb.Tests;

/// <summary>
/// Covers discovery, precedence and file indexing.
///
/// <para>
/// The layout cases here are not hypothetical. A user points <c>--skills-dir</c> at
/// whatever directory they happen to have: the parent of several skills, one skill
/// directly, or a checkout of <see href="https://github.com/anthropics/skills"/>
/// whose skills live one level further down under <c>skills/</c>. A walker that
/// handled only the first of those would silently register nothing for the other two,
/// and "my skills do not show up" is the hardest kind of bug for a user to diagnose.
/// </para>
/// </summary>
public class SkillRegistryTests : IDisposable
{
    private readonly string _baseDir;

    public SkillRegistryTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-registry-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- helpers -----------------------------------------------------------

    private string WriteSkill(string relativeDirectory, string name, string description, string body = "# Instructions\nDo the thing.\n")
    {
        string dir = Path.Combine(_baseDir, relativeDirectory);
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, "SKILL.md"),
            $"---\nname: {name}\ndescription: {description}\n---\n\n{body}");
        return dir;
    }

    private SkillRegistry Registry(params string[] roots) => new(new SkillRegistryOptions
    {
        Roots = roots.Length > 0 ? roots : new[] { _baseDir },
    });

    // ---- discovery ---------------------------------------------------------

    [Fact]
    public void Scan_FindsSkillsOneLevelUnderTheRoot()
    {
        WriteSkill("pdf", "pdf", "does pdfs");
        WriteSkill("xlsx", "xlsx", "does spreadsheets");

        var registry = Registry();

        Assert.Equal(new[] { "pdf", "xlsx" }, registry.Skills.Select(s => s.Id));
    }

    [Fact]
    public void Scan_FindsASkillWhenTheRootIsTheSkillItself()
    {
        // "--skills-dir ./my-skill" is what a user types when they have exactly one.
        string dir = WriteSkill("solo", "solo", "does one thing");

        var registry = Registry(dir);

        Assert.Single(registry.Skills);
        Assert.Equal("solo", registry.Skills[0].Id);
    }

    [Fact]
    public void Scan_FindsSkillsNestedTwoLevelsDown()
    {
        // The anthropics/skills layout: pointing at the repository root has to reach
        // skills/<name>/SKILL.md.
        WriteSkill(Path.Combine("repo", "skills", "pdf"), "pdf", "does pdfs");
        WriteSkill(Path.Combine("repo", "skills", "docx"), "docx", "does documents");

        var registry = Registry(Path.Combine(_baseDir, "repo"));

        Assert.Equal(new[] { "docx", "pdf" }, registry.Skills.Select(s => s.Id));
    }

    [Fact]
    public void Scan_DoesNotDescendIntoASkillItAlreadyFound()
    {
        // A skill's own references/ may legitimately contain example skills — the
        // skill-creator skill ships exactly that. Registering them would put
        // documentation in the model's catalog.
        WriteSkill("outer", "outer", "the real skill");
        WriteSkill(Path.Combine("outer", "references", "example"), "example", "an example inside a skill");

        var registry = Registry();

        Assert.Single(registry.Skills);
        Assert.Equal("outer", registry.Skills[0].Id);
    }

    [Fact]
    public void Scan_SkipsDotDirectoriesAndNodeModules()
    {
        WriteSkill(".git/objects/pdf", "pdf", "should not be found");
        WriteSkill("node_modules/thing", "thing", "should not be found");
        WriteSkill("real", "real", "should be found");

        var registry = Registry();

        Assert.Single(registry.Skills);
        Assert.Equal("real", registry.Skills[0].Id);
    }

    [Fact]
    public void Scan_ReportsAnUnparseableSkillWithoutFailingTheRest()
    {
        WriteSkill("good", "good", "loads fine");
        string broken = Path.Combine(_baseDir, "broken");
        Directory.CreateDirectory(broken);
        File.WriteAllText(Path.Combine(broken, "SKILL.md"), "no frontmatter here\n");

        var registry = Registry();

        Assert.Single(registry.Skills);
        Assert.Contains(registry.Errors, e => e.Path.Contains("broken", StringComparison.Ordinal));
    }

    [Fact]
    public void Scan_MissingRoot_IsReportedRatherThanThrown()
    {
        var registry = Registry(Path.Combine(_baseDir, "nope"));

        Assert.Empty(registry.Skills);
        Assert.Contains(registry.Errors, e => e.Message.Contains("does not exist", StringComparison.Ordinal));
    }

    [Fact]
    public void Scan_ResultsAreSortedById()
    {
        // The catalog is rendered from this order and the KV prefix cache hashes the
        // rendered bytes from block 0, so an order that varied with the filesystem's
        // enumeration would change the prompt prefix between runs.
        WriteSkill("zebra", "zebra", "z");
        WriteSkill("alpha", "alpha", "a");
        WriteSkill("mid", "mid", "m");

        Assert.Equal(new[] { "alpha", "mid", "zebra" }, Registry().Skills.Select(s => s.Id));
    }

    // ---- precedence --------------------------------------------------------

    [Fact]
    public void Scan_FirstRootWinsAndTheShadowedCopyIsReported()
    {
        // Renaming the loser would be worse: the name is what a user types and what a
        // SKILL.md cross-references, and it would change when an unrelated root was
        // added. Refuse the duplicate and say which copy won.
        WriteSkill(Path.Combine("first", "pdf"), "pdf", "the winning copy");
        WriteSkill(Path.Combine("second", "pdf"), "pdf", "the shadowed copy");

        var registry = Registry(Path.Combine(_baseDir, "first"), Path.Combine(_baseDir, "second"));

        Assert.Single(registry.Skills);
        Assert.Equal("the winning copy", registry.Skills[0].Description);
        Assert.Contains(registry.Errors, e => e.Message.Contains("takes precedence", StringComparison.Ordinal));
    }

    [Fact]
    public void Scan_TheSameRootTwice_IsVisitedOnce()
    {
        WriteSkill("pdf", "pdf", "does pdfs");

        var registry = Registry(_baseDir, _baseDir + Path.DirectorySeparatorChar);

        Assert.Single(registry.Skills);
        Assert.Empty(registry.Errors);
    }

    // ---- file indexing -----------------------------------------------------

    [Fact]
    public void Scan_IndexesBundledFilesWithTheirConventionBucket()
    {
        string dir = WriteSkill("bundled", "bundled", "ships extras");
        Directory.CreateDirectory(Path.Combine(dir, "scripts"));
        Directory.CreateDirectory(Path.Combine(dir, "references"));
        Directory.CreateDirectory(Path.Combine(dir, "assets"));
        File.WriteAllText(Path.Combine(dir, "scripts", "extract.py"), "print('hi')\n");
        File.WriteAllText(Path.Combine(dir, "references", "api.md"), "# API\n");
        File.WriteAllBytes(Path.Combine(dir, "assets", "logo.png"), new byte[] { 0x89, 0x50, 0x4E, 0x47 });
        File.WriteAllText(Path.Combine(dir, "LICENSE.txt"), "MIT\n");

        Skill skill = Registry().Skills.Single();

        Assert.Equal(SkillFileKind.Script, skill.Files.Single(f => f.Path == "scripts/extract.py").Kind);
        Assert.Equal(SkillFileKind.Reference, skill.Files.Single(f => f.Path == "references/api.md").Kind);
        Assert.Equal(SkillFileKind.Asset, skill.Files.Single(f => f.Path == "assets/logo.png").Kind);
        Assert.False(skill.Files.Single(f => f.Path == "assets/logo.png").IsText);

        // Licence boilerplate is never task-relevant and is present in most published
        // skills, so it is indexed but not advertised to the model.
        Assert.Contains(skill.Files, f => f.Path == "LICENSE.txt");
        Assert.DoesNotContain(skill.BundledFiles, f => f.Path == "LICENSE.txt");
        Assert.DoesNotContain(skill.BundledFiles, f => f.Path == "SKILL.md");
    }

    [Fact]
    public void Scan_IndexesFilesOutsideTheThreeConventionalDirectories()
    {
        // The specification says a skill "may contain any files and directories", and
        // published skills use per-language folders, examples/, core/ and root-level
        // .md files. Indexing only scripts/references/assets would hide most of them.
        string dir = WriteSkill("polyglot", "polyglot", "ships several languages");
        Directory.CreateDirectory(Path.Combine(dir, "python"));
        File.WriteAllText(Path.Combine(dir, "python", "guide.md"), "# Python\n");
        File.WriteAllText(Path.Combine(dir, "forms.md"), "# Forms\n");

        Skill skill = Registry().Skills.Single();

        Assert.Contains(skill.BundledFiles, f => f.Path == "python/guide.md");
        Assert.Contains(skill.BundledFiles, f => f.Path == "forms.md");
    }

    [Fact]
    public void ReadResource_ReturnsTheSkillsOwnFile()
    {
        string dir = WriteSkill("readable", "readable", "has a reference");
        Directory.CreateDirectory(Path.Combine(dir, "references"));
        File.WriteAllText(Path.Combine(dir, "references", "api.md"), "# API\nEverything you need.\n");

        Skill skill = Registry().Skills.Single();

        Assert.True(skill.TryReadResource("references/api.md", 4096, 0, out var content, out string error), error);
        Assert.Contains("Everything you need.", content.Text, StringComparison.Ordinal);
        Assert.False(content.Truncated);
    }

    [Fact]
    public void ReadResource_PagesALongFileAndReportsWhereToContinue()
    {
        string dir = WriteSkill("long", "long", "has a long reference");
        File.WriteAllText(Path.Combine(dir, "big.md"), new string('a', 5000));

        Skill skill = Registry().Skills.Single();

        Assert.True(skill.TryReadResource("big.md", 1024, 0, out var first, out _));
        Assert.True(first.Truncated);
        Assert.Equal(1024, first.NextOffsetBytes);

        Assert.True(skill.TryReadResource("big.md", 4096, first.NextOffsetBytes, out var second, out _));
        Assert.False(second.Truncated);
        Assert.Equal(5000, second.NextOffsetBytes);
    }

    [Fact]
    public void ReadResource_RefusesToEscapeTheSkillDirectory()
    {
        WriteSkill("a", "a", "first skill");
        WriteSkill("b", "b", "second skill");

        Skill a = Registry().Skills.Single(s => s.Id == "a");

        // Not merely "not found": one skill must never be able to read another's
        // files, or a prompt injected into a's SKILL.md could exfiltrate b's.
        Assert.False(a.TryReadResource("../b/SKILL.md", 4096, 0, out _, out string error));
        Assert.Contains("escapes the skill directory", error, StringComparison.Ordinal);
    }

    [Fact]
    public void ReadResource_RefusesABinaryFileWithAUsableMessage()
    {
        string dir = WriteSkill("binary", "binary", "ships a font");
        File.WriteAllBytes(Path.Combine(dir, "font.woff2"), new byte[] { 0x77, 0x4F, 0x46, 0x32, 0x00, 0x01 });

        Skill skill = Registry().Skills.Single();

        Assert.False(skill.TryReadResource("font.woff2", 4096, 0, out _, out string error));
        Assert.Contains("binary", error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReadResource_SniffsAnExtensionlessFileRatherThanRefusingIt()
    {
        // A skill may reasonably ship a Makefile or an extension-less LICENSE.
        string dir = WriteSkill("sniff", "sniff", "ships a Makefile");
        File.WriteAllText(Path.Combine(dir, "Makefile"), "all:\n\techo hi\n");

        Skill skill = Registry().Skills.Single();

        Assert.True(skill.TryReadResource("Makefile", 4096, 0, out var content, out string error), error);
        Assert.Contains("echo hi", content.Text, StringComparison.Ordinal);
    }

    // ---- selection ---------------------------------------------------------

    [Fact]
    public void Resolve_ReturnsTheSelectionSortedAndReportsUnknownNames()
    {
        WriteSkill("pdf", "pdf", "does pdfs");
        WriteSkill("xlsx", "xlsx", "does spreadsheets");

        var registry = Registry();
        IReadOnlyList<Skill> resolved = registry.Resolve(new[] { "xlsx", "nope", "pdf", "xlsx" }, out var unknown);

        // Sorted regardless of the order the caller listed them, and de-duplicated:
        // the rendered catalog must be byte-identical for the same set however it
        // arrived, or the KV prefix cache misses from block 0 on every turn.
        Assert.Equal(new[] { "pdf", "xlsx" }, resolved.Select(s => s.Id));
        Assert.Equal(new[] { "nope" }, unknown);
    }

    [Fact]
    public void Resolve_IsCaseInsensitive()
    {
        WriteSkill("pdf", "pdf", "does pdfs");

        Assert.Single(Registry().Resolve(new[] { "PDF" }, out _));
    }

    // ---- installation ------------------------------------------------------

    [Fact]
    public void InstallFromDirectory_CopiesTheTreeAndRegistersIt()
    {
        string source = Path.Combine(_baseDir, "source", "importable");
        Directory.CreateDirectory(Path.Combine(source, "scripts"));
        File.WriteAllText(Path.Combine(source, "SKILL.md"),
            "---\nname: importable\ndescription: can be installed\n---\nbody\n");
        File.WriteAllText(Path.Combine(source, "scripts", "run.py"), "print(1)\n");

        string install = Path.Combine(_baseDir, "installed");
        var registry = new SkillRegistry(new SkillRegistryOptions { InstallDirectory = install });

        Skill skill = registry.InstallFromDirectory(source);

        Assert.Equal("importable", skill.Id);
        Assert.Equal(SkillOrigin.Installed, skill.Origin);
        Assert.True(File.Exists(Path.Combine(install, "importable", "scripts", "run.py")));
        Assert.Contains(registry.Skills, s => s.Id == "importable");
    }

    [Fact]
    public void InstallFromDirectory_RefusesToReplaceWithoutOverwrite()
    {
        string source = Path.Combine(_baseDir, "source", "dup");
        Directory.CreateDirectory(source);
        File.WriteAllText(Path.Combine(source, "SKILL.md"), "---\nname: dup\ndescription: a duplicate\n---\nbody\n");

        var registry = new SkillRegistry(new SkillRegistryOptions
        {
            InstallDirectory = Path.Combine(_baseDir, "installed"),
        });
        registry.InstallFromDirectory(source);

        Assert.Throws<SkillInstallException>(() => registry.InstallFromDirectory(source));
        registry.InstallFromDirectory(source, overwrite: true);   // succeeds
    }

    [Fact]
    public void Remove_RefusesToDeleteASkillTheOperatorPutThere()
    {
        // The management API must never delete files out of a directory the operator
        // configured by hand — that is their tree, not TensorSharp's.
        WriteSkill(Path.Combine("operator-root", "theirs"), "theirs", "operator owned");

        var registry = new SkillRegistry(new SkillRegistryOptions
        {
            Roots = new[] { Path.Combine(_baseDir, "operator-root") },
            InstallDirectory = Path.Combine(_baseDir, "installed"),
        });

        var ex = Assert.Throws<InvalidOperationException>(() => registry.Remove("theirs"));
        Assert.Contains("not managed by TensorSharp", ex.Message, StringComparison.Ordinal);
        Assert.True(Directory.Exists(Path.Combine(_baseDir, "operator-root", "theirs")));
    }

    [Fact]
    public void Remove_DeletesAnInstalledSkill()
    {
        string source = Path.Combine(_baseDir, "source", "removable");
        Directory.CreateDirectory(source);
        File.WriteAllText(Path.Combine(source, "SKILL.md"), "---\nname: removable\ndescription: can go\n---\nbody\n");

        string install = Path.Combine(_baseDir, "installed");
        var registry = new SkillRegistry(new SkillRegistryOptions { InstallDirectory = install });
        registry.InstallFromDirectory(source);

        Assert.True(registry.Remove("removable"));
        Assert.Empty(registry.Skills);
        Assert.False(Directory.Exists(Path.Combine(install, "removable")));
    }

    [Fact]
    public void InstallDirectory_TakesPrecedenceOverAConfiguredRoot()
    {
        // An uploaded skill must shadow a stale copy of the same name in a read-only
        // operator root, not the other way round — otherwise uploading a fix does
        // nothing and the user has no way to tell.
        WriteSkill(Path.Combine("operator-root", "pdf"), "pdf", "the old copy");

        string source = Path.Combine(_baseDir, "source", "pdf");
        Directory.CreateDirectory(source);
        File.WriteAllText(Path.Combine(source, "SKILL.md"), "---\nname: pdf\ndescription: the new copy\n---\nbody\n");

        var registry = new SkillRegistry(new SkillRegistryOptions
        {
            Roots = new[] { Path.Combine(_baseDir, "operator-root") },
            InstallDirectory = Path.Combine(_baseDir, "installed"),
        });
        registry.InstallFromDirectory(source, overwrite: true);

        Assert.Equal("the new copy", registry.Skills.Single(s => s.Id == "pdf").Description);
    }
}
