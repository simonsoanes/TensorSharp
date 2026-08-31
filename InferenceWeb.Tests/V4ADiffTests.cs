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
using System.Linq;
using TensorSharp.AgentHost.CodeExec;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The deterministic editor, exercised at the only layer where nothing else can be
/// blamed: no filesystem, no shell, no interpreter, no model. <c>V4ADiff</c> decides
/// WHERE a change goes and <c>CodePatch</c> decides what the envelope around it means,
/// and everything built on top of them — the shell tool, the heredoc interception, the
/// all-or-nothing staging — is worth exactly as much as the answer these two give.
///
/// <para>
/// The property under test throughout is the one that justifies handing a model a patch
/// tool at all: a hunk either lands where it belongs or nothing is written. The
/// dangerous failure is not a refusal, it is a hunk that lands somewhere plausible and
/// wrong, because that is written silently and reads correctly in the diff. So every
/// rung of the matching ladder is pinned here BY VALUE — 0 exact, 1 trailing whitespace
/// ignored, 100 all surrounding whitespace ignored, +10000 an end-of-file hunk that had
/// to search forward — and a future "improvement" that adds a similarity score or an
/// edit distance breaks a test here rather than someone's file.
/// </para>
/// <para>
/// Three of these look like trivia and are the entire reason real patches apply.
/// Deleted lines join the text that is searched for, so a hunk that only deletes can be
/// found at all. A completely empty line in a hunk is read as a blank CONTEXT line,
/// because models strip the trailing space from those and without that one coercion
/// every patch touching a blank line fails. And a <c>@@</c> anchor is ADVISORY: an
/// anchor the model paraphrased must leave the hunk to resolve by context rather than
/// failing the patch, which is a deliberate departure from the behaviour the previous
/// apply_patch implementation had.
/// </para>
/// <para>
/// Everything here is pure string work, so these tests need nothing from the host and
/// there is nothing to gate on and nothing to clean up.
/// </para>
/// </summary>
public class V4ADiffTests
{
    /// <summary>Apply one '*** Update File' body to a file, the way the runner does.</summary>
    private static V4ADiff.DiffResult Update(string file, string diff) =>
        V4ADiff.Update(file, V4ADiff.SplitDiffLines(diff));

    // ---- the matching ladder, one rung at a time ---------------------------

    [Fact]
    public void AnExactMatch_LandsAtFuzzZero()
    {
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\ngamma\n",
            " alpha\n-beta\n+BETA\n gamma\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\nBETA\ngamma\n", result.Text);
        Assert.Equal(0, result.Fuzz);
        Assert.Equal(1, result.LinesAdded);
        Assert.Equal(1, result.LinesRemoved);
    }

    [Fact]
    public void TrailingWhitespaceOnly_MatchesAtFuzzOne()
    {
        // The file carries trailing spaces the model's copy of it does not. Rung two of
        // the ladder ignores them on both sides — and the untouched context line is
        // rewritten from the FILE, so the file's own trailing spaces survive.
        V4ADiff.DiffResult result = Update(
            "alpha   \nbeta\ngamma\n",
            " alpha\n-beta\n+BETA\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha   \nBETA\ngamma\n", result.Text);
        Assert.Equal(1, result.Fuzz);
    }

    [Fact]
    public void LeadingWhitespaceDifference_MatchesAtFuzz100()
    {
        // Rung three. Note what the result costs: the replacement is written with the
        // model's indentation, not the file's, so the hunk lands in the right place with
        // the wrong shape. That is exactly why fuzz 100 is reported to the model rather
        // than swallowed — see Describe_WarnsWhenAHunkOnlyMatchedOnFuzz100.
        V4ADiff.DiffResult result = Update(
            "def f():\n        return 1\n",
            " def f():\n-    return 1\n+    return 2\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("def f():\n    return 2\n", result.Text);
        Assert.Equal(100, result.Fuzz);
    }

    [Fact]
    public void AHunkThatMatchesNothing_ChangesNothing_AndSaysWhy()
    {
        // The bottom of the ladder is a refusal, never a best guess. The message has to
        // be self-correcting on its own, so it echoes the model's lines back WITH their
        // leading whitespace — usually the actual problem, and invisible in a summary —
        // and names how long the file really is.
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\ngamma",
            "-    ghost line\n+    replacement\n");

        Assert.False(result.Ok);
        Assert.Equal(string.Empty, result.Text);
        Assert.Equal(0, result.Fuzz);
        Assert.NotNull(result.Error);
        Assert.Contains("did not match", result.Error!, StringComparison.Ordinal);
        Assert.Contains("  |    ghost line", result.Error!, StringComparison.Ordinal);
        Assert.Contains("The file has 3 lines", result.Error!, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissWithALongContext_EchoesTheFirstEightLinesAndCountsTheRest()
    {
        // A failure message that reprints a fifty-line hunk buries the one line that
        // matters and costs the context window twice.
        V4ADiff.DiffResult result = Update(
            "only real line",
            " ghost 1\n ghost 2\n ghost 3\n ghost 4\n ghost 5\n ghost 6\n ghost 7\n ghost 8\n ghost 9\n"
            + "-ghost 10\n+replacement\n");

        Assert.False(result.Ok);
        Assert.NotNull(result.Error);
        Assert.Contains("  |ghost 8", result.Error!, StringComparison.Ordinal);
        Assert.DoesNotContain("ghost 9", result.Error!, StringComparison.Ordinal);
        Assert.Contains("and 2 more", result.Error!, StringComparison.Ordinal);
    }

    // ---- what actually forms the search key --------------------------------

    [Fact]
    public void AHunkOfOnlyDeletions_FindsItsPlace()
    {
        // Deleted lines are part of what is searched for as well as part of what is
        // removed. Were they not, this hunk's context would be EMPTY, empty context
        // matches wherever the cursor already is, and the replacement would be spliced
        // in at line 1 instead of over "two"/"three" — a silent wrong edit of exactly
        // the kind the whole design exists to make impossible.
        V4ADiff.DiffResult result = Update(
            "one\ntwo\nthree\nfour\n",
            "-two\n-three\n+2 and 3\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("one\n2 and 3\nfour\n", result.Text);
        Assert.Equal(0, result.Fuzz);
        Assert.Equal(1, result.LinesAdded);
        Assert.Equal(2, result.LinesRemoved);
    }

    [Fact]
    public void AnEntirelyEmptyDiffLine_IsABlankContextLine()
    {
        // One line of code in TryReadSection ("raw.Length == 0 ? " " : raw") and the
        // reason patches apply in practice: a context line that is blank has a single
        // trailing space, and every model strips it. Without the coercion this hunk has
        // no prefix character to read at all.
        V4ADiff.DiffResult result = Update(
            "header\n\nfooter\n",
            " header\n\n-footer\n+FOOTER\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("header\n\nFOOTER\n", result.Text);
        Assert.Equal(0, result.Fuzz);
    }

    // ---- @@ anchors ---------------------------------------------------------

    [Fact]
    public void AnAnchor_PicksBetweenTwoIdenticalBlocks()
    {
        // Without the anchor the context "    value = 0" matches inside first() and the
        // wrong function is edited.
        V4ADiff.DiffResult result = Update(
            "def first():\n    value = 0\n    return value\n\ndef second():\n    value = 0\n    return value\n",
            "@@ def second():\n-    value = 0\n+    value = 42\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal(
            "def first():\n    value = 0\n    return value\n\ndef second():\n    value = 42\n    return value\n",
            result.Text);
        Assert.Equal(0, result.Fuzz);
    }

    [Fact]
    public void StackedAnchors_NarrowIntoNestedCode()
    {
        // One anchor is not enough here: "class Beta:" only gets the cursor as far as
        // start(), whose body is identical to stop()'s. The second anchor is searched
        // FORWARD unconditionally, which is what lets a stack narrow rather than reset.
        V4ADiff.DiffResult result = Update(
            "class Alpha:\n    def run(self):\n        return 1\n\n"
            + "class Beta:\n    def start(self):\n        return 1\n    def stop(self):\n        return 1\n",
            "@@ class Beta:\n@@     def stop(self):\n-        return 1\n+        return 2\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal(
            "class Alpha:\n    def run(self):\n        return 1\n\n"
            + "class Beta:\n    def start(self):\n        return 1\n    def stop(self):\n        return 2\n",
            result.Text);
        Assert.Equal(0, result.Fuzz);
    }

    [Fact]
    public void AnAnchorThatMatchesNothing_IsAdvisory()
    {
        // The important one, and a deliberate change from the previous implementation,
        // which failed the patch outright. Models paraphrase the header line they are
        // anchoring on far more often than they get the surrounding context wrong; a
        // missed anchor leaves the cursor alone and the hunk still resolves by context.
        // Failing here taught models to stop using anchors, which made everything worse.
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\ngamma\n",
            "@@ def nonexistent():\n-beta\n+BETA\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\nBETA\ngamma\n", result.Text);
        // A miss is free: it is not charged as fuzz, because nothing about the match was
        // approximate.
        Assert.Equal(0, result.Fuzz);
    }

    // ---- *** End of File ----------------------------------------------------

    [Fact]
    public void AnEndOfFileHunk_AnchorsAtTheEnd_NotTheFirstIdenticalBlock()
    {
        // The whole point of the marker: the same line appears at the top of the file,
        // and an ordinary forward search would append after THAT one. Fuzz 0 proves the
        // end-anchored search hit, rather than the forward fallback below.
        V4ADiff.DiffResult result = Update(
            "print('done')\nmiddle\nprint('done')",
            " print('done')\n+print('extra')\n*** End of File\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("print('done')\nmiddle\nprint('done')\nprint('extra')", result.Text);
        Assert.Equal(0, result.Fuzz);
        Assert.Equal(1, result.LinesAdded);
        Assert.Equal(0, result.LinesRemoved);
    }

    [Fact]
    public void AnEndOfFileHunkThatIsNotAtTheEnd_AppliesButSaysSoLoudly()
    {
        // The model claimed end-of-file for context that is at the TOP. It still
        // applies, because refusing would be worse than a correct edit — but the +10000
        // is how the caller can tell the model's picture of the file is wrong.
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\ngamma",
            " alpha\n+inserted\n*** End of File\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\ninserted\nbeta\ngamma", result.Text);
        Assert.Equal(10000, result.Fuzz);
    }

    // ---- newlines -----------------------------------------------------------

    [Fact]
    public void ACrlfFile_StaysCrlf()
    {
        // A line-addressed edit once converted a whole CRLF file to LF and showed up as
        // a whole-file change in someone's diff the next morning.
        V4ADiff.DiffResult result = Update(
            "a = 1\r\nb = 2\r\nc = 3\r\n",
            "-b = 2\n+b = 99\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("a = 1\r\nb = 99\r\nc = 3\r\n", result.Text);
    }

    [Fact]
    public void AnLfFile_StaysLf_EvenWhenTheDiffIsCrlf()
    {
        // The reverse conversion, which is the one a Windows-hosted model produces. The
        // file's own style wins: CRs are stripped when the diff is split, and the
        // newline is taken from the input.
        V4ADiff.DiffResult result = Update(
            "a = 1\nb = 2\nc = 3\n",
            "-b = 2\r\n+b = 99\r\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("a = 1\nb = 99\nc = 3\n", result.Text);
        Assert.DoesNotContain("\r", result.Text, StringComparison.Ordinal);
    }

    [Fact]
    public void ACreatedFile_EndsWithANewline()
    {
        // A deliberate departure from the reference, which ends a new file exactly where
        // the last '+' line ends. Nothing tells a model to send a bare '+' as its final
        // line, so nothing ever does, and the result is a file that concatenates wrongly
        // and that git reports as "\ No newline at end of file".
        V4ADiff.DiffResult result = V4ADiff.Create(V4ADiff.SplitDiffLines("+alpha\n+beta\n"));

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\nbeta\n", result.Text);
        Assert.Equal(2, result.LinesAdded);
        Assert.Equal(0, result.LinesRemoved);
    }

    [Fact]
    public void ACreatedFileWhoseLastLineIsBare_DoesNotGetTwoNewlines()
    {
        // The control for the departure above: a model that DID send the bare '+' has
        // already produced the trailing newline, and appending must not double it up.
        V4ADiff.DiffResult result = V4ADiff.Create(new[] { "+alpha", "+beta", "+" });

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\nbeta\n", result.Text);
        Assert.Equal(3, result.LinesAdded);
    }

    [Fact]
    public void ACreateSection_StopsAtTheNextFileMarker()
    {
        V4ADiff.DiffResult result = V4ADiff.Create(new[] { "+alpha", "*** End Patch" });

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\n", result.Text);
    }

    // ---- overlap ------------------------------------------------------------

    [Fact]
    public void TwoHunksThatTouchTheSameLines_AreRefused()
    {
        // Ordinary hunks cannot overlap: the cursor only moves forward. An end-of-file
        // hunk is the exception, because it re-searches from the END of the file and so
        // can resolve BEHIND ground an earlier hunk already claimed. Applying both would
        // duplicate or drop lines depending on the order, so the whole patch is refused
        // instead — nothing is a better answer than something plausible.
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\ngamma",
            " beta\n-gamma\n+GAMMA\n@@\n-beta\n+BETA\n gamma\n*** End of File\n");

        Assert.False(result.Ok);
        Assert.Equal(string.Empty, result.Text);
        Assert.NotNull(result.Error);
        Assert.Contains("overlap", result.Error!, StringComparison.Ordinal);
        Assert.Contains("line 2", result.Error!, StringComparison.Ordinal);
    }

    // ---- CodePatch.TryParse: the envelope -----------------------------------

    [Fact]
    public void AMultiFileEnvelope_ParsesIntoItsSections()
    {
        Assert.True(CodePatch.TryParse(
            "*** Begin Patch\n"
            + "*** Add File: helpers.py\n"
            + "+def clean(s):\n"
            + "+    return s.strip()\n"
            + "*** Update File: main.py\n"
            + "@@ def process(rows):\n"
            + "-    return [r for r in rows]\n"
            + "+    return [clean(r) for r in rows]\n"
            + "*** Delete File: scratch.txt\n"
            + "*** End Patch\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error), error);

        Assert.Equal(3, sections.Count);

        Assert.Equal(CodePatch.FileOp.Add, sections[0].Op);
        Assert.Equal("helpers.py", sections[0].Path);
        Assert.Null(sections[0].MoveTo);
        // The body is handed on verbatim, prefixes and all: V4ADiff, not the parser,
        // is what understands '+', '-' and ' '.
        Assert.Equal(new[] { "+def clean(s):", "+    return s.strip()" }, sections[0].Body.ToArray());

        Assert.Equal(CodePatch.FileOp.Update, sections[1].Op);
        Assert.Equal("main.py", sections[1].Path);
        Assert.Equal(
            new[] { "@@ def process(rows):", "-    return [r for r in rows]", "+    return [clean(r) for r in rows]" },
            sections[1].Body.ToArray());

        Assert.Equal(CodePatch.FileOp.Delete, sections[2].Op);
        Assert.Equal("scratch.txt", sections[2].Path);
        Assert.Empty(sections[2].Body);
    }

    [Fact]
    public void AnEnvelopeWithoutItsBeginMarker_IsRefused()
    {
        Assert.False(CodePatch.TryParse(
            "*** Update File: m.py\n-x\n+y\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("must start with '*** Begin Patch'", error!, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEnvelopeWithoutItsEndMarker_IsRefused()
    {
        // Almost always a truncated generation rather than a format error, so the
        // message says what to do about that specifically.
        Assert.False(CodePatch.TryParse(
            "*** Begin Patch\n*** Update File: m.py\n-x\n+y\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("must end with '*** End Patch'", error!, StringComparison.Ordinal);
        Assert.Contains("nothing was written", error!, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAddSectionWithANonPlusLine_IsRefusedWhenTheFileIsBuilt()
    {
        // The parser deliberately does not police body prefixes — it collects lines
        // until the next header and lets V4ADiff.Create be the single place that knows
        // what an Add body means. This test pins the whole path, because a check that
        // exists in neither layer is how a file gets created with a stray line in it.
        Assert.True(CodePatch.TryParse(
            "*** Begin Patch\n"
            + "*** Add File: notes.txt\n"
            + "+first line\n"
            + "second line without a plus\n"
            + "*** End Patch\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error), error);

        CodePatch.FileSection only = Assert.Single(sections);
        V4ADiff.DiffResult created = V4ADiff.Create(only.Body);

        Assert.False(created.Ok);
        Assert.Equal(string.Empty, created.Text);
        Assert.NotNull(created.Error);
        Assert.Contains("must start with '+'", created.Error!, StringComparison.Ordinal);
        Assert.Contains("second line without a plus", created.Error!, StringComparison.Ordinal);
    }

    [Fact]
    public void ADeleteSectionWithABody_IsRefused()
    {
        // A delete removes the whole file, so a diff after it means the model meant
        // something else and guessing which is not the parser's job.
        Assert.False(CodePatch.TryParse(
            "*** Begin Patch\n*** Delete File: scratch.txt\n-something\n*** End Patch\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("must not be followed by a diff", error!, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("/etc/passwd")]
    [InlineData("C:\\Windows\\System32\\drivers\\etc\\hosts")]
    public void AnAbsolutePath_IsRefused(string path)
    {
        // Refused by SPELLING here, before any filesystem is touched, so the message
        // names the rule rather than a resolved path the model never wrote. The Windows
        // form is caught by the colon on every platform, not only on Windows.
        Assert.False(CodePatch.TryParse(
            "*** Begin Patch\n*** Add File: " + path + "\n+x\n*** End Patch\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("is an absolute path", error!, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("```")]
    [InlineData("```diff")]
    [InlineData("```patch")]
    public void AMarkdownFencedEnvelope_IsAccepted(string fence)
    {
        // Several families wrap a patch in a fence however firmly the prompt says not
        // to. Unwrapping costs nothing and cannot change what the patch means; refusing
        // costs a whole round.
        Assert.True(CodePatch.TryParse(
            fence + "\n*** Begin Patch\n*** Delete File: a.txt\n*** End Patch\n```\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error), error);

        CodePatch.FileSection only = Assert.Single(sections);
        Assert.Equal(CodePatch.FileOp.Delete, only.Op);
        Assert.Equal("a.txt", only.Path);
    }

    [Fact]
    public void TheJsonOperationsForm_IsAccepted()
    {
        // Models trained on the Agents SDK send this instead of the envelope. Accepting
        // it costs forty lines; refusing it costs a round every time.
        const string json =
            "{\"operations\":["
            + "{\"type\":\"create_file\",\"path\":\"a.py\",\"diff\":\"+x = 1\\n\"},"
            + "{\"type\":\"update_file\",\"path\":\"b.py\",\"move_to\":\"c.py\",\"diff\":\"@@\\n-a\\n+b\\n\"},"
            + "{\"type\":\"delete_file\",\"path\":\"d.py\"}]}";

        Assert.True(CodePatch.TryParse(
            json, out IReadOnlyList<CodePatch.FileSection> sections, out string? error), error);

        Assert.Equal(3, sections.Count);

        Assert.Equal(CodePatch.FileOp.Add, sections[0].Op);
        Assert.Equal("a.py", sections[0].Path);
        Assert.Equal(new[] { "+x = 1" }, sections[0].Body.ToArray());

        Assert.Equal(CodePatch.FileOp.Update, sections[1].Op);
        Assert.Equal("c.py", sections[1].MoveTo);
        Assert.Equal(new[] { "@@", "-a", "+b" }, sections[1].Body.ToArray());

        Assert.Equal(CodePatch.FileOp.Delete, sections[2].Op);
        Assert.Equal("d.py", sections[2].Path);
        Assert.Empty(sections[2].Body);
    }

    [Fact]
    public void AnUnknownJsonOperation_NamesWhatItShouldHaveBeen()
    {
        Assert.False(CodePatch.TryParse(
            "{\"type\":\"frobnicate\",\"path\":\"a.py\"}",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("frobnicate", error!, StringComparison.Ordinal);
        Assert.Contains("create_file", error!, StringComparison.Ordinal);
    }

    [Fact]
    public void AJsonObjectThatIsNotAPatch_FallsThroughToTheEnvelopeError()
    {
        // The JSON path must not hijack every input that happens to start with '{'; a
        // document with no operations in it is simply not a patch, and the model needs
        // to be told about the envelope rather than about JSON.
        Assert.False(CodePatch.TryParse(
            "{\"foo\":1}", out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("must start with '*** Begin Patch'", error!, StringComparison.Ordinal);
    }

    [Fact]
    public void MoveTo_IsCarriedOnTheUpdateSection()
    {
        Assert.True(CodePatch.TryParse(
            "*** Begin Patch\n"
            + "*** Update File: old.py\n"
            + "*** Move to: new.py\n"
            + "-print('hi')\n"
            + "+print('hello')\n"
            + "*** End Patch\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error), error);

        CodePatch.FileSection only = Assert.Single(sections);
        Assert.Equal(CodePatch.FileOp.Update, only.Op);
        Assert.Equal("old.py", only.Path);
        Assert.Equal("new.py", only.MoveTo);
        Assert.Equal(new[] { "-print('hi')", "+print('hello')" }, only.Body.ToArray());
    }

    [Fact]
    public void AnUpdateThatOnlyRenames_NeedsNoHunks()
    {
        // A rename with no content change is a legitimate patch, and the "no hunks"
        // refusal must not swallow it.
        Assert.True(CodePatch.TryParse(
            "*** Begin Patch\n*** Update File: old.py\n*** Move to: new.py\n*** End Patch\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error), error);

        CodePatch.FileSection only = Assert.Single(sections);
        Assert.Equal("new.py", only.MoveTo);
        Assert.Empty(only.Body);
    }

    [Fact]
    public void AnUpdateWithNeitherHunksNorARename_IsRefused()
    {
        Assert.False(CodePatch.TryParse(
            "*** Begin Patch\n*** Update File: m.py\n*** End Patch\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("has no hunks", error!, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEnvelopeThatNamesNoFiles_IsRefused()
    {
        Assert.False(CodePatch.TryParse(
            "*** Begin Patch\n*** End Patch\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("names no files", error!, StringComparison.Ordinal);
    }

    [Fact]
    public void ALineOutsideEverySection_IsRefused()
    {
        Assert.False(CodePatch.TryParse(
            "*** Begin Patch\nrandom prose the model wrote\n*** End Patch\n",
            out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("not a file header", error!, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData((string?)null)]
    [InlineData("")]
    [InlineData("   \n\t\n")]
    public void AnEmptyPatch_IsRefused(string? patch)
    {
        // A malformed patch is a recoverable tool error, never an exception — including
        // the degenerate case where the argument never arrived at all.
        Assert.False(CodePatch.TryParse(
            patch, out IReadOnlyList<CodePatch.FileSection> sections, out string? error));

        Assert.Empty(sections);
        Assert.NotNull(error);
        Assert.Contains("the patch was empty", error!, StringComparison.Ordinal);
    }

    // ---- CodePatch.Describe -------------------------------------------------

    [Fact]
    public void Describe_WarnsWhenAHunkOnlyMatchedOnFuzz100()
    {
        // The patch applied, so this is not a failure — but the model's copy of the file
        // disagrees with the real one about indentation, and the NEXT hunk it writes
        // from that copy will miss. Saying so now is cheaper than the retry loop later.
        string text = CodePatch.Describe(new[]
        {
            new CodePatch.FileOutcome(CodePatch.FileOp.Update, "main.py", null, 2, 1, 100),
        });

        Assert.Contains("Applied the patch to 1 file:", text, StringComparison.Ordinal);
        Assert.Contains("updated main.py  (+2 -1)", text, StringComparison.Ordinal);
        Assert.Contains("ignoring leading whitespace", text, StringComparison.Ordinal);
    }

    [Fact]
    public void Describe_StaysQuietBelowFuzz100()
    {
        // Trailing whitespace is not worth a paragraph; a model told about every
        // harmless approximation starts ignoring the notes that matter.
        string text = CodePatch.Describe(new[]
        {
            new CodePatch.FileOutcome(CodePatch.FileOp.Update, "main.py", null, 1, 1, 1),
        });

        Assert.DoesNotContain("ignoring leading whitespace", text, StringComparison.Ordinal);
    }

    [Fact]
    public void Describe_NamesTheRenameAndOnlyCountsLinesForUpdates()
    {
        string text = CodePatch.Describe(new[]
        {
            new CodePatch.FileOutcome(CodePatch.FileOp.Add, "helpers.py", null, 4, 0, 0),
            new CodePatch.FileOutcome(CodePatch.FileOp.Update, "old.py", "new.py", 1, 1, 0),
            new CodePatch.FileOutcome(CodePatch.FileOp.Delete, "scratch.txt", null, 0, 0, 0),
        });

        Assert.Contains("Applied the patch to 3 files:", text, StringComparison.Ordinal);
        Assert.Contains("added   helpers.py\n", text, StringComparison.Ordinal);
        Assert.Contains("updated old.py -> new.py  (+1 -1)", text, StringComparison.Ordinal);
        Assert.Contains("deleted scratch.txt\n", text, StringComparison.Ordinal);
        Assert.DoesNotContain("ignoring leading whitespace", text, StringComparison.Ordinal);
    }
    // ---- pinned against the reference, by differential test -----------------
    //
    // Everything below survived a differential harness: 7,280 hand-written and generated
    // cases -- every case from the reference's own tests/test_apply_diff.py, plus sweeps
    // over newline styles, the whitespace ladder, anchors, end-of-file markers and CR
    // mangling, plus four seeded fuzz rounds -- run through both this file and
    // openai-agents-python's src/agents/apply_diff.py, with the outcome compared field by
    // field: applied-or-refused, resulting text, fuzz, lines added, lines removed. These
    // are the cases where the two came apart, or came within one character of it.
    //
    // Three deviations from the reference survive ON PURPOSE and are pinned below AS
    // deviations, so that "fixing" one of them fails a test here instead of surprising
    // someone later: a created file gains a trailing newline, a created file is LF, and a
    // file with no newline of its own is written LF where the reference would copy the
    // diff's style. Nothing else differs, over all 7,280 cases.

    [Fact]
    public void RungTwo_IgnoresTheAsciiInformationSeparators()
    {
        // U+001C-U+001F are whitespace to Python's str.rstrip() and are NOT whitespace to
        // .NET's TrimEnd() -- enumerating both classifications over every code point, in
        // both directions, says those four characters are the whole difference between
        // them. It is not cosmetic, because rstrip is literally what rung two of the
        // ladder compares with: a file carrying a separator character (a paste out of a
        // record-oriented export, a fixture) resolved its hunk in the reference and was
        // REFUSED here. The same patch applying on one implementation and failing on the
        // other is the one property a line-for-line port must not have.
        V4ADiff.DiffResult result = Update("alpha\u001C\nbeta\n", " alpha\n-beta\n+B\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\u001C\nB\n", result.Text);
        Assert.Equal(1, result.Fuzz);
    }

    [Fact]
    public void RungThree_IgnoresTheAsciiInformationSeparators()
    {
        // The same class on the leading edge, where str.strip() removes it and Trim()
        // does not. Rung three, so fuzz 100.
        V4ADiff.DiffResult result = Update("\u001Dalpha\nbeta\n", " alpha\n-beta\n+B\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("\u001Dalpha\nB\n", result.Text);
        Assert.Equal(100, result.Fuzz);
    }

    [Fact]
    public void AnAnchor_AlsoMatchesAcrossTheInformationSeparators()
    {
        // The anchor search has its own strip-equal rung, which charges 1 fuzz. Pinned
        // separately because it is a different call site: a fix applied only to the
        // context ladder leaves the anchors disagreeing with the reference.
        V4ADiff.DiffResult result = Update("\u001Ealpha\nbeta\n", "@@ alpha\n-beta\n+B\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("\u001Ealpha\nB\n", result.Text);
        Assert.Equal(1, result.Fuzz);
    }

    [Fact]
    public void TheWhitespaceClass_IsUnicodes_NotJustSpaceAndTab()
    {
        // The other half of the same contract, and the reason the fix could not be a
        // hand-rolled Trim(' ', '\t'): a no-break space is rung two and an ideographic
        // space is rung three, in the reference and here.
        V4ADiff.DiffResult trailing = Update("alpha\u00A0\nbeta\n", " alpha\n-beta\n+B\n");
        Assert.True(trailing.Ok, trailing.Error);
        Assert.Equal(1, trailing.Fuzz);

        V4ADiff.DiffResult leading = Update("\u3000alpha\nbeta\n", " alpha\n-beta\n+B\n");
        Assert.True(leading.Ok, leading.Error);
        Assert.Equal(100, leading.Fuzz);
    }

    [Fact]
    public void AFileWithNoNewlineOfItsOwn_IsWrittenWithLf()
    {
        // DEVIATION, deliberate. The reference, having no newline in the input to copy,
        // falls back to the newline style of the DIFF -- so this CRLF patch would make a
        // CRLF file there. It cannot here: the section body reaches V4ADiff already split
        // by SplitDiffLines, which strips the CRs, and nothing downstream can recover
        // them. Rather than a detection that silently always answers "\n", Update says LF
        // outright -- the same call Create already makes and documents. A caller that
        // wants the reference's answer has to carry the envelope's newline style down.
        V4ADiff.DiffResult result = Update("solo", "-solo\r\n+SOLO\r\n+second\r\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("SOLO\nsecond", result.Text);
        Assert.DoesNotContain("\r", result.Text, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEmptyFile_IsWrittenWithLf_EvenWhenTheDiffIsCrlf()
    {
        // The same deviation at its most visible: the reference's own test suite pins
        // "hello\r\nworld\r\n" for exactly this input.
        V4ADiff.DiffResult result = Update("", "@@\r\n+hello\r\n+world");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("hello\nworld\n", result.Text);
    }

    // ---- the reference's own cases, applied through this port ---------------

    [Fact]
    public void TheToolDescriptionsWorkedExample_Applies()
    {
        // The example the apply_patch tool description shows the model. If stacked
        // anchors ever stop working, the models are being taught a syntax that fails.
        V4ADiff.DiffResult result = Update(
            "class BaseClass\n    def search():\n        pass\n\n"
            + "class Subclass\n    def search():\n        pass\n",
            "@@ class BaseClass\n@@     def search():\n-        pass\n"
            + "+        raise NotImplementedError()\n\n"
            + "@@ class Subclass\n@@     def search():\n-        pass\n"
            + "+        raise NotImplementedError()");

        Assert.True(result.Ok, result.Error);
        Assert.Equal(
            "class BaseClass\n    def search():\n        raise NotImplementedError()\n\n"
            + "class Subclass\n    def search():\n        raise NotImplementedError()\n",
            result.Text);
        Assert.Equal(0, result.Fuzz);
        Assert.Equal(2, result.LinesAdded);
        Assert.Equal(2, result.LinesRemoved);
    }

    [Fact]
    public void AWholeStackOfAnchorsThatAllMiss_StillResolvesByContext()
    {
        // Advisory means advisory all the way down: the second anchor is searched FORWARD
        // unconditionally, and a forward search that finds nothing must still leave the
        // cursor alone rather than parking it at the end of the file.
        V4ADiff.DiffResult result = Update("a\nb\n", "@@ nope\n@@ also-nope\n-b\n+B");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("a\nB\n", result.Text);
        Assert.Equal(0, result.Fuzz);
    }

    [Fact]
    public void AStackEndingInABareAnchor_IsAccepted()
    {
        // A bare "@@" contributes no text to search for but still counts as "a header was
        // here". Reading it as an anchor of "" would send the cursor to the first empty
        // line in the file.
        V4ADiff.DiffResult result = Update(
            "class Only\n    def run():\n        pass\n",
            "@@ class Only\n@@\n-        pass\n+        return 1");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("class Only\n    def run():\n        return 1\n", result.Text);
        Assert.Equal(0, result.Fuzz);
    }

    [Fact]
    public void AHunkOfPureAdditionsWithNoContext_LandsAtTheCursor()
    {
        // Empty context matches where the cursor already is -- the early return in
        // FindContextCore. It is what makes "@@" plus a run of "+" lines a legal way to
        // write into an empty file, which is the shape a model reaches for first.
        V4ADiff.DiffResult result = Update("", "@@\n+hello\n+world");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("hello\nworld\n", result.Text);
        Assert.Equal(2, result.LinesAdded);
    }

    [Fact]
    public void AnEndOfFileMarkerWithNothingElse_IsAnEmptyHunkThatChangesNothing()
    {
        // The EOF check in TryReadSection runs BEFORE the "nothing in this section"
        // check, so this parses rather than failing. It has to stay that way: swapping
        // the two turns a harmless no-op section into a refused patch.
        V4ADiff.DiffResult result = Update("one\ntwo\n", "*** End of File");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("one\ntwo\n", result.Text);
        Assert.Equal(0, result.Fuzz);
        Assert.Equal(0, result.LinesAdded);
        Assert.Equal(0, result.LinesRemoved);
    }

    [Fact]
    public void AnEndOfFileHunkWhoseContextIsTheWholeFile_AnchorsAtLineOne()
    {
        // The end-anchored search starts at len(file) - len(context), which for a
        // whole-file hunk is zero. Fuzz 0 proves it matched there rather than falling
        // through to the forward search and its +10000.
        V4ADiff.DiffResult result = Update(
            "p\nq\nr\np\nq\n",
            " p\n q\n r\n p\n-q\n+Q\n \n*** End of File");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("p\nq\nr\np\nQ\n", result.Text);
        Assert.Equal(0, result.Fuzz);
    }

    // ---- fuzz is a running total, not a high-water mark ---------------------

    [Fact]
    public void TwoHunksThatEachMatchAtRungTwo_ReportFuzzTwo()
    {
        // Fuzz is summed across hunks, so the caller can tell one approximate match from
        // two. A max() here would read the same for both and hide the second.
        V4ADiff.DiffResult result = Update(
            "a  \nb\nc  \nd\n",
            "@@ a\n-b\n+B\n@@ c\n-d\n+D\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("a  \nB\nc  \nD\n", result.Text);
        Assert.Equal(2, result.Fuzz);
    }

    [Fact]
    public void TwoEndOfFileHunksThatBothFallBack_ReportTwentyThousand()
    {
        // Both claimed end-of-file for context that is not at the end, so both took the
        // forward fallback and both were charged. 20000 is the number that tells a caller
        // the model's picture of this file is wrong twice over.
        V4ADiff.DiffResult result = Update(
            "l1\nl2\nl3\nl4\nl5\nl6\n",
            "-l2\n+L2\n*** End of File\n@@\n-l4\n+L4\n*** End of File");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("l1\nL2\nl3\nL4\nl5\nl6\n", result.Text);
        Assert.Equal(20000, result.Fuzz);
    }

    [Fact]
    public void AnEndOfFileHunkThatOnlyMatchedOnIndentation_ReportsBothPenalties()
    {
        // 10000 for the fallback plus 100 for the rung-three match. Pinned as the sum
        // because the two are added, not merged: a caller reading 10100 knows the hunk
        // was wrong about where it was AND wrong about the indentation.
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\ngamma",
            "   alpha\n+inserted\n*** End of File\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\ninserted\nbeta\ngamma", result.Text);
        Assert.Equal(10100, result.Fuzz);
    }

    // ---- lines that look like syntax and are not ----------------------------

    [Fact]
    public void ATripleAtSign_IsNotAnAnchorAndDoesNotLoop()
    {
        // "@@@" starts with "@@" so TryReadSection stops at it, but ReadAnchors will not
        // consume it -- it is neither "@@ " nor exactly "@@". That leaves the parser
        // pointing at a line it cannot make progress on, and the ONLY thing standing
        // between that and an infinite loop is the empty-section refusal.
        V4ADiff.DiffResult result = Update("a\nb\n", "@@@\n-b\n+B\n");

        Assert.False(result.Ok);
        Assert.NotNull(result.Error);
        Assert.Contains("no lines in it", result.Error!, StringComparison.Ordinal);
    }

    [Fact]
    public void ABareTripleStar_EndsTheSectionAndIsThenRefused()
    {
        // "***" on its own ends a section without being a marker, so the parser comes
        // back around to a line it cannot start a hunk from. Refused, not silently
        // treated as the end of the patch -- a patch that stops early writes half an edit.
        V4ADiff.DiffResult result = Update("a\nb\n", " a\n***\n");

        Assert.False(result.Ok);
        Assert.NotNull(result.Error);
        Assert.Contains("not part of a hunk", result.Error!, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnknownTripleStarMarker_NamesTheMarkersThatExist()
    {
        V4ADiff.DiffResult result = Update("a\nb\n", " a\n*** Whatever\n");

        Assert.False(result.Ok);
        Assert.NotNull(result.Error);
        Assert.Contains("*** End of File", result.Error!, StringComparison.Ordinal);
        Assert.Contains("*** End Patch", result.Error!, StringComparison.Ordinal);
    }

    [Fact]
    public void MoveToInTheMiddleOfAHunk_SaysWhereItBelongs()
    {
        // The reference calls this "Invalid Line" like any other stray marker. Naming the
        // rule instead is the difference between the model moving the line and the model
        // deleting it.
        V4ADiff.DiffResult result = Update("a\nb\n", " a\n*** Move to: other.txt\n");

        Assert.False(result.Ok);
        Assert.NotNull(result.Error);
        Assert.Contains("*** Move to:", result.Error!, StringComparison.Ordinal);
        Assert.Contains("*** Update File:", result.Error!, StringComparison.Ordinal);
    }

    // ---- splitting the diff into lines --------------------------------------

    [Theory]
    // A lone CR is NOT a line break: re.split(r"\r?\n") leaves it inside the line, and so
    // must this. Reading it as one shifts every line number after it.
    [InlineData("a\rb\nc", new[] { "a\rb", "c" })]
    // Trailing CRs are stripped from each line, however many there are, because the
    // reference rstrips them rather than removing one.
    [InlineData("a\r\r\nb", new[] { "a", "b" })]
    // Exactly one trailing blank line is dropped, so "a\n" is one line and not two.
    [InlineData("a\n", new[] { "a" })]
    // And only one: a deliberate blank last line survives.
    [InlineData("a\n\n", new[] { "a", "" })]
    [InlineData("", new string[0])]
    [InlineData("\n", new[] { "" })]
    public void SplitDiffLines_MatchesTheReferenceSplit(string diff, string[] expected)
    {
        Assert.Equal(expected, V4ADiff.SplitDiffLines(diff));
    }

    [Fact]
    public void ADiffWrittenEntirelyInCrlf_AppliesToAnLfFileUnchanged()
    {
        // The CR stripping is what makes this work at all: without it every context line
        // carries a trailing CR the file does not have, and the ladder drops to rung two
        // for the whole patch -- or misses.
        V4ADiff.DiffResult result = Update("a\nb\n", " a\r\n-b\r\n+B\r\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("a\nB\n", result.Text);
        Assert.Equal(0, result.Fuzz);
    }

    // ---- text that is not ASCII, and text that is very long -----------------

    [Fact]
    public void NonAsciiText_MatchesAndIsRewrittenIntact()
    {
        V4ADiff.DiffResult cjk = Update(
            "行一\n行二\n",
            " 行一\n-行二\n+行三\n");
        Assert.True(cjk.Ok, cjk.Error);
        Assert.Equal("行一\n行三\n", cjk.Text);

        // An astral character costs two UTF-16 units, so anything that indexed into a
        // line by code unit other than at position 0 would split it in half here.
        V4ADiff.DiffResult emoji = Update(
            "a\n\U0001F389 party\nb\n",
            " a\n-\U0001F389 party\n+\U0001F38A party\n");
        Assert.True(emoji.Ok, emoji.Error);
        Assert.Equal("a\n\U0001F38A party\nb\n", emoji.Text);
    }

    [Fact]
    public void AVeryLongLine_MatchesExactlyAndOnRungTwo()
    {
        string line = new string('L', 5000);

        V4ADiff.DiffResult exact = Update(line + "\nb\n", " " + line + "\n-b\n+B\n");
        Assert.True(exact.Ok, exact.Error);
        Assert.Equal(0, exact.Fuzz);

        V4ADiff.DiffResult rungTwo = Update(line + "   \nb\n", " " + line + "\n-b\n+B\n");
        Assert.True(rungTwo.Ok, rungTwo.Error);
        Assert.Equal(1, rungTwo.Fuzz);
        Assert.Equal(line + "   \nB\n", rungTwo.Text);
    }

    // ---- duplicate blocks ---------------------------------------------------

    [Fact]
    public void TwoIdenticalBlocks_TakeTheFirstUnlessAnchored()
    {
        const string file = "x\nblock\nend\ny\nblock\nend\n";

        V4ADiff.DiffResult first = Update(file, "-block\n+BLOCK\n end\n");
        Assert.True(first.Ok, first.Error);
        Assert.Equal("x\nBLOCK\nend\ny\nblock\nend\n", first.Text);

        V4ADiff.DiffResult second = Update(file, "@@ y\n-block\n+BLOCK\n end\n");
        Assert.True(second.Ok, second.Error);
        Assert.Equal("x\nblock\nend\ny\nBLOCK\nend\n", second.Text);
    }

    [Fact]
    public void AHunkAfterTheFirstWithNoAnchor_IsRefusedRatherThanGuessed()
    {
        // Once the cursor has moved, a hunk with no "@@" of its own is read as a
        // continuation of the previous one -- its lines join that hunk's context, which
        // then matches nothing. Refused, which is the right answer: the model wrote two
        // hunks and only marked one.
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\ngamma\ndelta\n",
            " alpha\n-beta\n+B\n-delta\n+D\n");

        Assert.False(result.Ok);
        Assert.Equal(string.Empty, result.Text);
        Assert.NotNull(result.Error);
        Assert.Contains("did not match", result.Error!, StringComparison.Ordinal);
    }

    // ---- the anchor that is also the hunk's first line ---------------------
    //
    // The one apply_patch failure in this server's logs that was the HOST's fault, and
    // the reason the model that hit it never called apply_patch again in that
    // conversation. Every test in this block guards one half of the fix: that it fires
    // when it must, and that it cannot fire anywhere else.

    [Fact]
    public void AnAnchorThatIsAlsoTheHunksFirstLine_Applies()
    {
        // The logged transcript, reproduced. The model ran `head -5 create_slides.py`,
        // so the only lines it had ever seen were the first five -- and the line it
        // picked as the '@@' header was necessarily also the line it quoted as the
        // hunk's first line, because there was no other line to pick. The anchor matched
        // at index 0, the cursor stepped to 1, and the context search began past the
        // only place it could match.
        const string file =
            "from pptx import ExcelWriter\n"
            + "from pptx.util import Inches\n"
            + "\n"
            + "prs = Presentation()\n"
            + "slide = prs.slides.add_slide(prs.slide_layouts[0])\n";

        V4ADiff.DiffResult result = Update(
            file,
            "@@ from pptx import ExcelWriter\n"
            + "-from pptx import ExcelWriter\n"
            + "+from pptx import Presentation\n"
            + " from pptx.util import Inches\n");

        Assert.True(result.Ok, result.Error);
        Assert.StartsWith("from pptx import Presentation\n", result.Text, StringComparison.Ordinal);
        Assert.DoesNotContain("ExcelWriter", result.Text, StringComparison.Ordinal);

        // Said out loud, because the hunk only landed after the host relaxed something.
        Assert.NotNull(result.Note);
        Assert.Contains("also this hunk's first line", result.Note!, StringComparison.Ordinal);
    }

    [Fact]
    public void TheAnchorRetry_AlsoWorksWhenTheFirstLineIsContextRatherThanDeleted()
    {
        // Same shape, but the anchor line is quoted as CONTEXT and the change is below
        // it. The guard compares the hunk's first line to the anchor and does not care
        // which kind of line it is.
        V4ADiff.DiffResult result = Update(
            "def solve():\n    return 1\n",
            "@@ def solve():\n def solve():\n-    return 1\n+    return 2\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("def solve():\n    return 2\n", result.Text);
    }

    [Fact]
    public void TheAnchorRetry_DoesNotFireWhenTheHunkDoesNotStartAtTheAnchor()
    {
        // The ordinary, correct shape: '@@' names the enclosing declaration and the hunk
        // body is the lines under it. Nothing here needs re-seeking, and a hunk that
        // genuinely does not match must still be refused -- the retry must not become a
        // second chance for every miss.
        V4ADiff.DiffResult result = Update(
            "def f():\n    return 1\n\ndef g():\n    return 2\n",
            "@@ def g():\n-    return 99\n+    return 3\n");

        Assert.False(result.Ok);
        Assert.Equal(string.Empty, result.Text);
        Assert.Contains("did not match", result.Error!, StringComparison.Ordinal);
    }

    [Fact]
    public void TheAnchorRetry_CannotLandBeforeTheAnchor()
    {
        // The forward-only contract, which is what keeps a multi-hunk patch from
        // reordering itself. 'block' appears before the anchor and after it; the hunk's
        // first line is the anchor, so the retry runs -- and must still refuse to match
        // the earlier occurrence, because the retry is required to land ON the anchor.
        const string file =
            "target\n"
            + "block\n"
            + "end\n"
            + "@@marker\n"
            + "other\n";

        V4ADiff.DiffResult result = Update(
            file,
            "@@ target\n target\n-block\n+BLOCK\n end\n");

        Assert.True(result.Ok, result.Error);

        // It landed on the anchor at line 1, not somewhere earlier: line 1 is still
        // 'target' and the edit is the block directly under it.
        Assert.Equal("target\nBLOCK\nend\n@@marker\nother\n", result.Text);
    }

    [Fact]
    public void TheAnchorRetry_DoesNotRescueAHunkAimedAtAnEarlierCopy()
    {
        // Two identical regions, an anchor selecting the SECOND. Under the retry the
        // hunk resolves at the second copy -- never the first -- so an anchor still
        // means what it has always meant.
        const string file =
            "def go():\n    return 1\n\ndef go():\n    return 1\n";

        V4ADiff.DiffResult result = Update(
            file,
            "@@ def go():\n def go():\n-    return 1\n+    return 7\n");

        Assert.True(result.Ok, result.Error);

        // The FIRST copy -- the one the anchor names. Without the retry this landed on
        // the SECOND, silently and at fuzz 0: the anchor matched at line 0, the cursor
        // stepped to line 1, and the identical block further down was the first thing
        // the context could match. A wrong edit reported as a clean one is the exact
        // failure this file is built to refuse, so it is pinned here by value.
        Assert.Equal("def go():\n    return 7\n\ndef go():\n    return 1\n", result.Text);
    }

    [Fact]
    public void AnUnanchoredHunk_IsUnaffectedByTheRetry()
    {
        // No '@@' at all means no landing to retry from, so the path is not reachable
        // and the ordinary first-match behaviour stands.
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\n",
            " alpha\n-beta\n+BETA\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\nBETA\n", result.Text);
        Assert.Null(result.Note);
    }
}
