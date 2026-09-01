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
using System.Linq;
using TensorSharp.AgentHost.CodeExec;

namespace InferenceWeb.Tests;

/// <summary>
/// The one way a hunk could go wrong with no symptom, and the message that ends the
/// guess-again cycle.
///
/// <para>
/// Every other failure in <see cref="V4ADiff"/> ends in a refusal the model can read.
/// Two did not. A hunk whose context fits in three places was applied to whichever came
/// first, at fuzz 0 — reported as "updated main.py (+1 -1)", indistinguishable from a
/// perfect edit, with the change made to the wrong function. And a hunk that missed was
/// told only what IT had looked for, never what was actually in the file, so its
/// replacement was drawn from the same distribution and missed the same way.
/// </para>
/// </summary>
public class V4AAmbiguityTests
{
    private static V4ADiff.DiffResult Update(string input, string diff, string path = "main.py") =>
        V4ADiff.Update(input, V4ADiff.SplitDiffLines(diff), "\n", path);

    private const string TwoIdenticalMethods =
        "class Alpha:\n"
        + "    def run(self):\n"
        + "        total = 0\n"
        + "        return total\n"
        + "\n"
        + "class Beta:\n"
        + "    def run(self):\n"
        + "        total = 0\n"
        + "        return total\n";

    // ---- ambiguity -----------------------------------------------------------

    /// <summary>
    /// It APPLIES — taking the first match is the reference's behaviour, is pinned by
    /// <c>V4ADiffTests.TwoIdenticalBlocks_TakeTheFirstUnlessAnchored</c>, and is usually
    /// what the model meant, since it read the file top-down. What must not happen is that
    /// it applies SILENTLY: this was the only way a hunk could land wrongly with nothing
    /// in the result to contradict it.
    /// </summary>
    [Fact]
    public void AHunkThatFitsInTwoPlaces_AppliesToTheFirstAndSaysSo()
    {
        V4ADiff.DiffResult result = Update(
            TwoIdenticalMethods,
            "-        total = 0\n+        total = 1\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal(
            "class Alpha:\n    def run(self):\n        total = 1\n        return total\n\n"
            + "class Beta:\n    def run(self):\n        total = 0\n        return total\n",
            result.Text);

        Assert.NotNull(result.Note);
        // The line numbers are the actionable part: the model has to be able to see WHICH
        // places, or "be more specific" is not a thing it can do.
        Assert.Contains("line 3", result.Note!, StringComparison.Ordinal);
        Assert.Contains("line 8", result.Note!, StringComparison.Ordinal);
        Assert.Contains("main.py", result.Note!, StringComparison.Ordinal);
        Assert.Contains("@@", result.Note!, StringComparison.Ordinal);
    }

    /// <summary>An unambiguous hunk says nothing — the note is for the coin toss only.</summary>
    [Fact]
    public void AnUnambiguousHunkCarriesNoNote()
    {
        V4ADiff.DiffResult result = Update("alpha\nbeta\ngamma\n", "-beta\n+BETA\n");

        Assert.True(result.Ok, result.Error);
        Assert.Null(result.Note);
    }

    /// <summary>
    /// An anchor that LANDED has positioned the hunk deliberately, so a second match
    /// elsewhere is not ambiguity — it is the thing the anchor was written to skip.
    /// </summary>
    [Fact]
    public void AnAnchorThatLands_MakesTheSameHunkUnambiguous()
    {
        V4ADiff.DiffResult result = Update(
            TwoIdenticalMethods,
            "@@ class Beta:\n-        total = 0\n+        total = 1\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal(
            "class Alpha:\n    def run(self):\n        total = 0\n        return total\n\n"
            + "class Beta:\n    def run(self):\n        total = 1\n        return total\n",
            result.Text);
        Assert.Equal(0, result.Fuzz);
        Assert.Null(result.Note);
    }

    /// <summary>
    /// The recorded finding this codebase already paid for: a missed anchor stays
    /// ADVISORY. Failing outright taught models to stop writing '@@' headers at all. So
    /// when the context is unique, a paraphrased anchor still applies — the anchor added
    /// nothing the context had not already settled.
    /// </summary>
    [Fact]
    public void AMissedAnchorStillApplies_WhenTheContextIsUnique()
    {
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\ngamma\n",
            "@@ def nonexistent():\n-beta\n+BETA\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("alpha\nBETA\ngamma\n", result.Text);
        Assert.Equal(0, result.Fuzz);
    }

    /// <summary>
    /// And the case those two meet: a missed anchor over an AMBIGUOUS context. Nothing
    /// narrowed the hunk, so nothing decided where it goes.
    /// </summary>
    [Fact]
    public void AMissedAnchorOverAnAmbiguousContext_IsNoted()
    {
        V4ADiff.DiffResult result = Update(
            TwoIdenticalMethods,
            "@@ class Gamma:\n-        total = 0\n+        total = 1\n");

        Assert.True(result.Ok, result.Error);
        Assert.NotNull(result.Note);
        Assert.Contains("more than one place", result.Note!, StringComparison.Ordinal);
    }

    /// <summary>
    /// A later hunk positioned by the cursor of an earlier one is not ambiguous: the
    /// earlier hunk moved the search past the first occurrence, which is how a multi-hunk
    /// patch addresses repeated blocks at all.
    /// </summary>
    [Fact]
    public void ASecondHunkPositionedByTheFirst_IsNotCalledAmbiguous()
    {
        V4ADiff.DiffResult result = Update(
            TwoIdenticalMethods,
            "@@ class Alpha:\n-        total = 0\n+        total = 1\n"
            + "@@ class Beta:\n-        total = 0\n+        total = 2\n");

        Assert.True(result.Ok, result.Error);
        Assert.Contains("total = 1", result.Text, StringComparison.Ordinal);
        Assert.Contains("total = 2", result.Text, StringComparison.Ordinal);
        Assert.Null(result.Note);
    }

    /// <summary>
    /// Only the winning rung counts. A block that matches EXACTLY once and loosely
    /// elsewhere is not ambiguous — the ladder already decided, and the exact match is the
    /// one the model meant.
    /// </summary>
    [Fact]
    public void AnExactMatchIsNotAmbiguousBecauseOtherPlacesMatchLoosely()
    {
        V4ADiff.DiffResult result = Update(
            "    total = 0\nmiddle\n        total = 0   \n",
            "-    total = 0\n+    total = 1\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal(0, result.Fuzz);
        Assert.Null(result.Note);
        Assert.StartsWith("    total = 1\n", result.Text, StringComparison.Ordinal);
    }

    /// <summary>
    /// An end-of-file hunk is anchored at the end by definition, so an identical block
    /// earlier in the file is exactly what that anchoring exists to skip.
    /// </summary>
    [Fact]
    public void AnEndOfFileHunkIsNeverCalledAmbiguous_BecauseTheEndIsTheDisambiguation()
    {
        // Written without a trailing newline: splitting "…\n" leaves an empty element
        // after the last line, which is a position in the array and not a line, and the
        // end-anchored search starts one line too high because of it. That is existing
        // behaviour with its own coverage; what matters here is only that an end-of-file
        // hunk is never REFUSED for ambiguity.
        V4ADiff.DiffResult result = Update(
            "tail\nmiddle\ntail",
            "-tail\n+TAIL\n*** End of File\n");

        Assert.True(result.Ok, result.Error);
        Assert.Equal("tail\nmiddle\nTAIL", result.Text);
        Assert.Null(result.Note);
    }

    // ---- showing the real file ----------------------------------------------

    /// <summary>
    /// The message that has to carry new information. Echoing the model's own search text
    /// back at it is the definition of a result it cannot act on.
    /// </summary>
    [Fact]
    public void AMissedHunkIsShownTheFilesRealBytesWithLineNumbers()
    {
        V4ADiff.DiffResult result = Update(
            "def build():\n    layout = 'wide'\n    return layout\n",
            "@@ def build():\n-    layout = {width: 10}\n+    layout = 'LAYOUT_16x9'\n");

        Assert.False(result.Ok);
        Assert.Contains("did not match", result.Error!, StringComparison.Ordinal);
        Assert.Contains("main.py", result.Error!, StringComparison.Ordinal);
        // The actual line, with its number, is what the next hunk gets rebuilt from.
        Assert.Contains("layout = 'wide'", result.Error!, StringComparison.Ordinal);
        Assert.Contains(" | ", result.Error!, StringComparison.Ordinal);
    }

    /// <summary>
    /// When nothing in the file resembles the hunk, no "closest match" is invented — a
    /// made-up region would send the model to rebuild against code that has nothing to do
    /// with what it wanted.
    /// </summary>
    [Fact]
    public void AHunkWithNothingLikeItInTheFile_GetsNoInventedRegion()
    {
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\ngamma\n",
            "-zzzzz-nothing-like-this\n+replacement\n");

        Assert.False(result.Ok);
        Assert.Contains("not there at all", result.Error!, StringComparison.Ordinal);
        Assert.DoesNotContain("closest place", result.Error!);
    }

    // ---- the reference's own test matrix -------------------------------------

    /// <summary>
    /// A trailing SPACE after '*** End of File' silently discarded every hunk that came
    /// after it, and the patch reported the first hunk's counts as though it were the whole
    /// patch. The loop that ends a hunk tests <c>StartsWith</c> while the code that
    /// CONSUMES the marker tested <c>==</c>, and <c>SplitDiffLines</c> strips carriage
    /// returns but not spaces — so the marker fell between the two.
    /// </summary>
    [Fact]
    public void ATrailingSpaceAfterTheEndOfFileMarkerDoesNotDiscardLaterHunks()
    {
        const string file = "def f():\n    x = 1\ndef g():\n    y = 1\n";

        V4ADiff.DiffResult result = Update(
            file,
            "@@ def f():\n-    x = 1\n+    x = 2\n*** End of File \n"
            + "@@ def g():\n-    y = 1\n+    y = 2\n");

        // Either both hunks land, or the whole thing is refused. What must not happen is
        // one hunk applying while the result reports success for the patch.
        if (result.Ok)
        {
            Assert.Contains("x = 2", result.Text, StringComparison.Ordinal);
            Assert.Contains("y = 2", result.Text, StringComparison.Ordinal);
            Assert.Equal(2, result.LinesAdded);
        }
        else
        {
            Assert.NotNull(result.Error);
        }
    }


    /// <summary>
    /// Case B from the reference implementation, executed against it: a hunk with context
    /// lines and no '+' or '-' at all produces `chunks == []`, writes the file back
    /// byte-identical, and reports `Updated &lt;path&gt;`. A model instructed not to re-read
    /// after a patch — which the reference's own prompt instructs, on the grounds that
    /// "the tool call will fail if it didn't work" — has no way to discover that.
    /// </summary>
    [Fact]
    public void AHunkThatChangesNothingIsNotReportedAsAChange()
    {
        V4ADiff.DiffResult result = Update(
            "def build():\n    total = 0\n    return total\n",
            "@@ def build():\n     total = 0\n     return total\n");

        // Either it refuses, or it applies and says plainly that nothing changed. What it
        // must not do is report an edit.
        if (result.Ok)
        {
            Assert.Equal(0, result.LinesAdded);
            Assert.Equal(0, result.LinesRemoved);
            Assert.Equal("def build():\n    total = 0\n    return total\n", result.Text);
        }
        else
        {
            Assert.Contains("nothing", result.Error!, StringComparison.OrdinalIgnoreCase);
        }
    }

    /// <summary>
    /// Case C/D: a hunk that matched only by ignoring whitespace lands the MODEL's
    /// indentation, which in Python changes the program. The reference computes a fuzz
    /// score of 100 for exactly this and then discards it — <c>apply_diff()</c> returns a
    /// bare string and the score has no consumer anywhere in that repo.
    /// </summary>
    [Fact]
    public void AWhitespaceOnlyMatchIsReportedAsFuzz_NotSilentlyApplied()
    {
        V4ADiff.DiffResult result = Update(
            "class A:\n        pass\n",
            "@@ class A:\n-    pass\n+    return 1\n");

        if (result.Ok)
            Assert.True(result.Fuzz >= 100, $"fuzz was {result.Fuzz}, so nothing warns about the re-indent");
        else
            Assert.NotNull(result.Error);
    }

    // ---- the defect that numbered reads create -------------------------------

    /// <summary>
    /// A model told to read with <c>nl -ba</c> — which is the right advice, because a hunk
    /// built from unnumbered output is a hunk built from a guess about indentation — then
    /// copies the numbers into the patch. The generic message sends it to check its
    /// indentation, which was never the problem.
    /// </summary>
    [Fact]
    public void ContextLinesCarryingNlLineNumbersAreNamedAsTheCause()
    {
        V4ADiff.DiffResult result = Update(
            "def build():\n    total = 0\n    return total\n",
            // As a paste of `nl -ba` output actually looks: the deleted line carries a
            // number too, because the model copied the whole listing.
            "     2\t    total = 0\n-     3\t    return total\n+    return total + 1\n");

        Assert.False(result.Ok);
        Assert.Contains("LINE NUMBERS", result.Error!, StringComparison.Ordinal);
        Assert.Contains("`nl`", result.Error!, StringComparison.Ordinal);
        Assert.Contains("Nothing was written", result.Error!, StringComparison.Ordinal);
    }

    [Fact]
    public void GrepNAndPipeSeparatedNumbersAreRecognisedToo()
    {
        foreach (string numbered in new[] { "2:    total = 0", "     2 |     total = 0" })
        {
            V4ADiff.DiffResult result = Update(
                "def build():\n    total = 0\n    return total\n",
                " " + numbered + "\n-    return total\n+    return total + 1\n");

            Assert.False(result.Ok);
            Assert.Contains("LINE NUMBERS", result.Error!, StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// Detected and REPORTED, never stripped and applied. Auto-stripping would be a fourth
    /// rung of the fuzz ladder under another name, and the standing decision in
    /// <see cref="V4ADiff"/> is that the ladder does not grow.
    /// </summary>
    [Fact]
    public void NumberedContextIsNeverSilentlyRepairedAndApplied()
    {
        const string file = "def build():\n    total = 0\n    return total\n";
        V4ADiff.DiffResult result = Update(
            file, "     2\t    total = 0\n-     3\t    return total\n+    return total + 1\n");

        Assert.False(result.Ok);
        Assert.Equal(string.Empty, result.Text);
    }

    /// <summary>
    /// A file that genuinely contains numbered lines — a changelog, a data table — must not
    /// be told its numbers are the problem. The diagnosis fires only when removing them
    /// actually makes the hunk match.
    /// </summary>
    [Fact]
    public void AFileWhoseLinesReallyStartWithNumbersIsNotMisdiagnosed()
    {
        V4ADiff.DiffResult result = Update(
            "1: alpha\n2: beta\n3: gamma\n",
            "-9: nothing-like-this\n+replacement\n");

        Assert.False(result.Ok);
        Assert.DoesNotContain("LINE NUMBERS", result.Error!);
    }

    /// <summary>
    /// The numbers must be visibly NOT part of the line. A tab is what <c>nl</c> and
    /// <c>grep -n</c> emit, and a model that copies numbered output straight back into a
    /// hunk writes context lines with a number glued to the front — the failure this
    /// excerpt exists to prevent, not to cause.
    /// </summary>
    [Fact]
    public void TheExcerptsLineNumbersAreNotSeparatedByATab()
    {
        V4ADiff.DiffResult result = Update(
            "def build():\n    layout = 'wide'\n    return layout\n",
            "-    layout = {width: 10}\n+    layout = 'x'\n");

        Assert.False(result.Ok);
        Assert.DoesNotContain('\t', result.Error!);
    }

    /// <summary>
    /// Splitting "a\nb\n" leaves a trailing empty element that is a position in the array,
    /// not a line of the file. The excerpt must not print it as one — the same off-by-one
    /// the line COUNT in this message was already fixed for once.
    /// </summary>
    [Fact]
    public void TheExcerptStopsAtTheLastRealLine()
    {
        V4ADiff.DiffResult result = Update(
            "alpha\nbeta\n",
            "-alpha\n-beta\n-omega\n+x\n");

        Assert.False(result.Ok);
        string[] numbered = result.Error!
            .Split('\n')
            .Where(line => line.Contains(" | ", StringComparison.Ordinal))
            .ToArray();
        Assert.All(numbered, line => Assert.DoesNotContain("    3 | ", line));
    }
}
