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
using System.IO.Compression;
using System.Linq;
using System.Text;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the upload path: what a ZIP is allowed to put on disk, and how much of it.
///
/// <para>
/// A ZIP entry name is attacker-controlled text. The classic "zip slip" ships an entry
/// called <c>../../../.ssh/authorized_keys</c>, and the framework's own
/// <c>ZipFile.ExtractToDirectory</c> has grown guards against the obvious spelling —
/// but which file actually lands still depends on how the name normalises on the host
/// platform, which is why every entry here goes through
/// <see cref="SkillPathGuard"/> instead: the extraction and the model's later reads then
/// agree on exactly one definition of "inside the skill". The escape cases below assert
/// not only that the upload is rejected but that nothing appeared outside the
/// destination, because "rejected after writing the file" is not rejected.
/// </para>
/// <para>
/// The ceilings exist because an upload is a small amount of attacker-controlled input
/// that decides how much disk and memory the host spends. Size is measured on the
/// DECOMPRESSED stream rather than on the entry's declared <c>Length</c>, which comes
/// from the archive's own headers and which a crafted ZIP simply lies about — so a
/// 40 KB upload that expands to fill the disk has to be caught while it expands, and
/// each ceiling has to say in its message which limit it was, or an operator cannot tell
/// a legitimately large skill from an attack.
/// </para>
/// </summary>
public class SkillArchiveTests : IDisposable
{
    private readonly string _baseDir;

    public SkillArchiveTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-archive-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- helpers -----------------------------------------------------------

    private static MemoryStream Zip(params (string Name, byte[] Content)[] entries)
    {
        var stream = new MemoryStream();
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach ((string name, byte[] content) in entries)
            {
                ZipArchiveEntry entry = zip.CreateEntry(name);
                using Stream target = entry.Open();
                target.Write(content, 0, content.Length);
            }
        }
        stream.Position = 0;
        return stream;
    }

    private static MemoryStream ZipText(params (string Name, string Content)[] entries) =>
        Zip(entries.Select(e => (e.Name, Encoding.UTF8.GetBytes(e.Content))).ToArray());

    private static (string Name, string Content) Manifest(string name = "SKILL.md") =>
        (name, "---\nname: pdf\ndescription: does pdfs\n---\n\nRead the form.\n");

    private string Destination(string name = "dest") => Path.Combine(_baseDir, name);

    private static SkillArchiveLimits Limits(
        long totalBytes = 1024 * 1024, long entryBytes = 1024 * 1024, int entries = 64, int ratio = 0) => new()
        {
            MaxTotalBytes = totalBytes,
            MaxEntryBytes = entryBytes,
            MaxEntries = entries,
            MaxCompressionRatio = ratio,
        };

    // ---- the happy path ----------------------------------------------------

    [Fact]
    public void Extract_ANormalBundle_WritesEveryFile()
    {
        using MemoryStream archive = ZipText(
            Manifest("pdf/SKILL.md"),
            ("pdf/scripts/extract.py", "print('hi')\n"),
            ("pdf/references/api.md", "# API\n"));
        string destination = Destination();

        SkillArchive.Extract(archive, destination, Limits());

        Assert.True(File.Exists(Path.Combine(destination, "pdf", "SKILL.md")));
        Assert.True(File.Exists(Path.Combine(destination, "pdf", "scripts", "extract.py")));
        Assert.Equal("# API\n", File.ReadAllText(Path.Combine(destination, "pdf", "references", "api.md")));
    }

    // ---- escapes -----------------------------------------------------------

    [Fact]
    public void Extract_AnEntryThatEscapesTheDestination_IsRefusedAndNothingIsWrittenOutside()
    {
        using MemoryStream archive = ZipText(("../evil.txt", "pwned"), Manifest());
        string destination = Destination();

        var ex = Assert.Throws<SkillInstallException>(() => SkillArchive.Extract(archive, destination, Limits()));

        Assert.Contains("../evil.txt", ex.Message, StringComparison.Ordinal);
        Assert.Contains("escapes the skill directory", ex.Message, StringComparison.Ordinal);

        // The point of the check: not merely that the call failed, but that the file
        // never appeared. A guard that throws after opening the FileStream has already
        // lost.
        Assert.False(File.Exists(Path.Combine(_baseDir, "evil.txt")));
    }

    [Theory]
    [InlineData("/evil.txt")]
    [InlineData("C:\\Windows\\evil.txt")]
    [InlineData("//server/share/evil.txt")]
    public void Extract_ARootedEntryName_IsRefused(string entryName)
    {
        using MemoryStream archive = ZipText((entryName, "pwned"), Manifest());
        string destination = Destination();

        var ex = Assert.Throws<SkillInstallException>(() => SkillArchive.Extract(archive, destination, Limits()));

        Assert.Contains("is rejected", ex.Message, StringComparison.Ordinal);
        Assert.False(File.Exists("/evil.txt"));
    }

    // ---- ceilings ----------------------------------------------------------

    [Fact]
    public void Extract_MoreEntriesThanTheCap_IsRefusedWithTheCapInTheMessage()
    {
        using MemoryStream archive = ZipText(
            Manifest(), ("a.txt", "a"), ("b.txt", "b"), ("c.txt", "c"));

        var ex = Assert.Throws<SkillInstallException>(
            () => SkillArchive.Extract(archive, Destination(), Limits(entries: 2)));

        Assert.Contains("more than 2 files", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Extract_AnEntryOverThePerFileCap_IsRefusedWithTheCapInTheMessage()
    {
        // Measured as it decompresses, not from the entry's declared Length: that field
        // is in the archive's own headers and a crafted upload sets it to anything.
        using MemoryStream archive = Zip(("big.bin", new byte[4096]));

        var ex = Assert.Throws<SkillInstallException>(
            () => SkillArchive.Extract(archive, Destination(), Limits(entryBytes: 1024)));

        Assert.Contains("per-file limit", ex.Message, StringComparison.Ordinal);
        Assert.Contains("1 KB", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Extract_MoreTotalBytesThanTheCap_IsRefusedWithTheCapInTheMessage()
    {
        // Neither entry is over the per-file cap; only their sum is. Checking one and
        // not the other lets an upload of a thousand legal files fill the disk.
        byte[] chunk = Encoding.UTF8.GetBytes(new string('a', 80));
        using MemoryStream archive = Zip(("one.txt", chunk), ("two.txt", chunk));

        var ex = Assert.Throws<SkillInstallException>(
            () => SkillArchive.Extract(archive, Destination(), Limits(totalBytes: 120, entryBytes: 1024)));

        Assert.Contains("expands to more than", ex.Message, StringComparison.Ordinal);
        Assert.Contains("120 B", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Extract_ADecompressionBomb_IsRefusedBeforeTheByteCeilingIsReached()
    {
        // A megabyte of zeros compresses to about a kilobyte. The byte ceilings above
        // would eventually stop a bomb, but the ratio catches it earlier and, more
        // usefully, says what it was — an operator seeing "expands to more than 256 MB"
        // has no idea whether they uploaded a big skill or were attacked.
        using MemoryStream archive = Zip(("bomb.bin", new byte[1024 * 1024]));

        var ex = Assert.Throws<SkillInstallException>(() => SkillArchive.Extract(
            archive, Destination(),
            new SkillArchiveLimits
            {
                MaxTotalBytes = 64L * 1024 * 1024,
                MaxEntryBytes = 64L * 1024 * 1024,
                MaxEntries = 64,
                MaxCompressionRatio = 100,
            }));

        Assert.Contains("decompression bomb", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Extract_AnEmptyArchive_IsRefused()
    {
        // Otherwise an empty upload installs an empty skill directory, and the user is
        // left wondering why the skill they "installed" never appears.
        using MemoryStream archive = ZipText();

        var ex = Assert.Throws<SkillInstallException>(
            () => SkillArchive.Extract(archive, Destination(), Limits()));

        Assert.Contains("the archive is empty", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Extract_AnArchiveOfNothingButDirectoryEntries_IsAlsoEmpty()
    {
        using MemoryStream archive = ZipText(("pdf/", ""), ("pdf/scripts/", ""));

        var ex = Assert.Throws<SkillInstallException>(
            () => SkillArchive.Extract(archive, Destination(), Limits()));

        Assert.Contains("the archive is empty", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Extract_SomethingThatIsNotAZip_IsRefusedWithAMessageSayingSo()
    {
        using var archive = new MemoryStream(Encoding.UTF8.GetBytes("this is a text file, not a zip"));

        var ex = Assert.Throws<SkillInstallException>(
            () => SkillArchive.Extract(archive, Destination(), Limits()));

        Assert.Contains("not a valid ZIP archive", ex.Message, StringComparison.Ordinal);
    }

    // ---- locating the skill inside a bundle ---------------------------------

    [Fact]
    public void LocateSkillRoot_FindsSkillMdAtTheArchiveRoot()
    {
        // A ZIP of the skill's CONTENTS, which is what you get by selecting the files
        // inside the folder rather than the folder itself.
        string root = Path.Combine(_baseDir, "flat");
        Directory.CreateDirectory(root);
        File.WriteAllText(Path.Combine(root, "SKILL.md"), "---\nname: pdf\ndescription: d\n---\n");

        Assert.Equal(root, SkillArchive.LocateSkillRoot(root));
    }

    [Fact]
    public void LocateSkillRoot_FindsSkillMdOneDirectoryDown()
    {
        // A ZIP of the skill DIRECTORY, which is what every archive tool produces when
        // you compress a folder. Both shapes are uploaded, so both have to work.
        string root = Path.Combine(_baseDir, "nested");
        Directory.CreateDirectory(Path.Combine(root, "pdf"));
        File.WriteAllText(Path.Combine(root, "pdf", "SKILL.md"), "---\nname: pdf\ndescription: d\n---\n");

        Assert.Equal(Path.Combine(root, "pdf"), SkillArchive.LocateSkillRoot(root));
    }

    [Fact]
    public void LocateSkillRoot_SkipsTheMacOsResourceDirectory()
    {
        // macOS Archive Utility adds a __MACOSX sibling to every ZIP it makes. Without
        // this it counts as a second candidate and every macOS upload becomes ambiguous.
        string root = Path.Combine(_baseDir, "macos");
        Directory.CreateDirectory(Path.Combine(root, "pdf"));
        Directory.CreateDirectory(Path.Combine(root, "__MACOSX"));
        File.WriteAllText(Path.Combine(root, "pdf", "SKILL.md"), "---\nname: pdf\ndescription: d\n---\n");
        File.WriteAllText(Path.Combine(root, "__MACOSX", "SKILL.md"), "resource fork noise");

        Assert.Equal(Path.Combine(root, "pdf"), SkillArchive.LocateSkillRoot(root));
    }

    [Fact]
    public void LocateSkillRoot_TwoCandidates_IsAmbiguousRatherThanAGuess()
    {
        // Several skills in one upload is a COLLECTION, which install-one cannot honour.
        // Silently taking the alphabetically first would install a skill the user did
        // not ask for and drop the rest without a word.
        string root = Path.Combine(_baseDir, "collection");
        foreach (string name in new[] { "pdf", "xlsx" })
        {
            Directory.CreateDirectory(Path.Combine(root, name));
            File.WriteAllText(Path.Combine(root, name, "SKILL.md"), $"---\nname: {name}\ndescription: d\n---\n");
        }

        Assert.Null(SkillArchive.LocateSkillRoot(root));
    }

    [Fact]
    public void LocateSkillRoot_NoSkillMdAnywhere_IsNull()
    {
        string root = Path.Combine(_baseDir, "empty");
        Directory.CreateDirectory(Path.Combine(root, "docs"));

        Assert.Null(SkillArchive.LocateSkillRoot(root));
    }

    [Fact]
    public void LocateAllSkillRoots_FindsEverySkillInACollection()
    {
        string root = Path.Combine(_baseDir, "collection");
        foreach (string name in new[] { "pdf", "xlsx", "docx" })
        {
            Directory.CreateDirectory(Path.Combine(root, name));
            File.WriteAllText(Path.Combine(root, name, "SKILL.md"), $"---\nname: {name}\ndescription: d\n---\n");
        }

        IReadOnlyList<string> found = SkillArchive.LocateAllSkillRoots(root);

        Assert.Equal(
            new[] { "docx", "pdf", "xlsx" },
            found.Select(Path.GetFileName).OrderBy(n => n, StringComparer.Ordinal));
    }

    [Fact]
    public void LocateAllSkillRoots_DoesNotDescendIntoASkillItAlreadyFound()
    {
        // A skill's own references/ may legitimately contain example skills — the
        // skill-creator skill ships exactly that — and installing documentation as a
        // skill puts it in the model's catalog.
        string root = Path.Combine(_baseDir, "outer");
        Directory.CreateDirectory(Path.Combine(root, "pdf", "references", "example"));
        File.WriteAllText(Path.Combine(root, "pdf", "SKILL.md"), "---\nname: pdf\ndescription: d\n---\n");
        File.WriteAllText(Path.Combine(root, "pdf", "references", "example", "SKILL.md"),
            "---\nname: example\ndescription: d\n---\n");

        Assert.Equal(new[] { Path.Combine(root, "pdf") }, SkillArchive.LocateAllSkillRoots(root));
    }
}
