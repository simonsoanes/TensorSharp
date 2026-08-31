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
using TensorSharp.AgentHost.CodeExec;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The string matcher behind <c>edit_file</c>, at the layer where nothing else can be
/// blamed: no filesystem, no shell, no model.
///
/// <para>
/// The property under test is the same one that justifies handing a model an editor at
/// all: a replacement either lands on the text the model named or nothing happens. The
/// dangerous failure is not a refusal — it is a match that lands somewhere plausible and
/// wrong, because that is written silently and reads correctly afterwards. So the ladder
/// is pinned RUNG BY RUNG and in ORDER, and the tolerant rungs are pinned to be
/// reachable only when the exact one has already missed.
/// </para>
/// <para>
/// The restyle cases look cosmetic and are not. Matching leniently and then writing the
/// model's literal bytes converts a file's typography behind its author's back — a
/// change to bytes nobody asked about, in a diff nobody reviewed. Tolerance without
/// restyle is not tolerance, it is corruption committed on the model's behalf.
/// </para>
/// </summary>
public class FileEditTests
{
    // ---- the ladder, one rung at a time ------------------------------------

    [Fact]
    public void AnExactMatch_IsFoundAtTheFirstRung()
    {
        FileEdit.MatchResult match = FileEdit.Find("alpha\nbeta\ngamma\n", "beta");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.Exact, match.Rung);
        Assert.Equal(1, match.Count);
        Assert.Equal("beta", match.Matched);
        Assert.Null(FileEdit.Describe(match.Rung));
    }

    [Fact]
    public void TextThatIsNotThere_IsNotFound()
    {
        FileEdit.MatchResult match = FileEdit.Find("alpha\nbeta\n", "delta");

        Assert.False(match.Found);
        Assert.Equal(0, match.Count);
    }

    [Fact]
    public void MoreIndentationThanTheFileHas_IsNotFound()
    {
        // No rung ignores leading whitespace, and that is deliberate. It is semantic in
        // Python and meaningful everywhere else, so a rung that folded it would let an
        // edit land inside the wrong block at the same nesting depth — written silently,
        // reading correctly. The patch side of this host DOES have such a rung; it can
        // afford one because a hunk carries several lines of surrounding context to
        // disambiguate with, and a single string does not.
        FileEdit.MatchResult match = FileEdit.Find("def f():\n    return 1\n", "        return 1");

        Assert.False(match.Found);
    }

    [Fact]
    public void AStringThatSitsInsideDeeperIndentation_IsFoundThere()
    {
        // The honest consequence of replacing a SUBSTRING rather than whole lines, and
        // the reference behaves identically — Claude Code's Edit compiles to a plain
        // String.replace. A model that sends less indentation than the line has still
        // matches, at an offset inside that line, and the replacement is spliced in
        // exactly there.
        //
        // This is not the silent-wrong-edit hazard: the text the model named genuinely is
        // at that position, the splice is textually exact, and anything ambiguous is
        // refused by the uniqueness rule before it can be written. Pinned so that a future
        // "fix" that anchors matches to line starts is a decision, not an accident.
        FileEdit.MatchResult match = FileEdit.Find("def f():\n        return 1\n", "    return 1");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.Exact, match.Rung);
        Assert.Equal(1, match.Count);

        // Inside the line, not at its start: index 13 is four spaces into the eight.
        Assert.Equal("def f():\n    ".Length, match.Index);
    }

    [Fact]
    public void CurlyQuotesInTheFile_MatchPlainOnesFromTheModel_AndSaySo()
    {
        FileEdit.MatchResult match = FileEdit.Find("say(\u2018HELLO\u2019)\n", "say('HELLO')");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.Punctuation, match.Rung);

        // The file's OWN bytes come back, which is what makes the restyle possible.
        Assert.Equal("say(\u2018HELLO\u2019)", match.Matched);
        Assert.NotNull(FileEdit.Describe(match.Rung));
    }

    [Fact]
    public void AnEmDashInTheFile_MatchesAHyphenFromTheModel()
    {
        // Codex's fourth rung, added there to "mirror the fuzzy behaviour of git apply".
        FileEdit.MatchResult match = FileEdit.Find("total = a \u2014 b\n", "total = a - b");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.Punctuation, match.Rung);
    }

    [Fact]
    public void ANonBreakingSpaceInTheFile_MatchesAnOrdinarySpace()
    {
        FileEdit.MatchResult match = FileEdit.Find("x =\u00a01\n", "x = 1");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.Punctuation, match.Rung);
    }

    [Fact]
    public void TheExactRungWins_EvenWhenATolerantOneWouldAlsoMatchElsewhere()
    {
        // A string that appears once exactly and once approximately is NOT ambiguous: the
        // ladder has already decided, and the exact match is the one the model meant.
        // Refusing here would teach the model that being precise is punished.
        FileEdit.MatchResult match = FileEdit.Find("a = 'x'\nb = \u2018x\u2019\n", "'x'");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.Exact, match.Rung);
        Assert.Equal(1, match.Count);
        Assert.Equal(4, match.Index);
    }

    [Fact]
    public void ALiteralUnicodeEscape_IsDecodedBeforeMatching()
    {
        FileEdit.MatchResult match = FileEdit.Find("label = \"caf\u00e9\"\n", @"label = ""caf\u00e9""");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.UnicodeEscape, match.Rung);
    }

    [Fact]
    public void ABackslashInOrdinaryCode_DoesNotTakeTheEscapeRung()
    {
        // The gate is that the string actually contains the '\uXXXX' form. A file full of
        // Windows paths and regexes must not be walked down a rung that rewrites its text.
        FileEdit.MatchResult match = FileEdit.Find(@"path = 'C:\users\x'" + "\n", @"path = 'C:\users\x'");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.Exact, match.Rung);
    }

    [Fact]
    public void LineNumbersCopiedInWithTheText_AreStripped()
    {
        // The failure this host manufactures for itself: it renders numbered listings and
        // recommends `grep -n`, so a model pastes the numbers back. The patch side already
        // carries a detector for exactly this; here it is absorbed instead.
        FileEdit.MatchResult match = FileEdit.Find(
            "def f():\n    return 1\n",
            "    12 | def f():\n    13 |     return 1");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.LineNumbers, match.Rung);
        Assert.Contains("line numbers", FileEdit.Describe(match.Rung)!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void NumbersThatAreActuallyData_AreNotStripped()
    {
        // A changelog, a table, generated output: the numbers are the content. The rung
        // fires only when stripping them makes the string match, so this one must not.
        const string file = "1: alpha\n2: beta\n";
        FileEdit.MatchResult match = FileEdit.Find(file, "1: alpha");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.Exact, match.Rung);
    }

    [Fact]
    public void ABlockWhereOnlySomeLinesAreNumbered_IsNotTreatedAsAListing()
    {
        FileEdit.MatchResult match = FileEdit.Find(
            "alpha\nbeta\n", "    1 | alpha\nbeta");

        Assert.False(match.Found);
    }

    // ---- counting ----------------------------------------------------------

    [Fact]
    public void EveryOccurrenceIsCounted_SoTheCallerCanRefuseAnAmbiguousEdit()
    {
        FileEdit.MatchResult match = FileEdit.Find("x\nx\nx\n", "x");

        Assert.True(match.Found);
        Assert.Equal(3, match.Count);
        Assert.Equal(3, match.Offsets.Count);
    }

    [Fact]
    public void OccurrencesDoNotOverlap()
    {
        // "aa" in "aaaa" is two, not three: a replacement consumes what it matched.
        FileEdit.MatchResult match = FileEdit.Find("aaaa", "aa");

        Assert.Equal(2, match.Count);
    }

    [Fact]
    public void AnEmptySearch_FindsNothing()
    {
        // Claude Code uses an empty old_string as an undocumented create-file sentinel.
        // Here creation is write_file's job — one spelling per operation — so this is a
        // miss and the caller turns it into a pointer at the right tool.
        Assert.False(FileEdit.Find("anything", string.Empty).Found);
    }

    // ---- restyle -----------------------------------------------------------

    [Fact]
    public void ATolerantMatch_WritesTheReplacementInTheFilesOwnPunctuation()
    {
        // Without this, matching leniently and writing literally would silently convert
        // the file's typography to ASCII — bytes nobody asked to change.
        FileEdit.MatchResult match = FileEdit.Find("say(\u2018HELLO\u2019)\n", "say('HELLO')");
        string restyled = FileEdit.Restyle("say('GOODBYE')", match.Matched, match.Search);

        Assert.Equal("say(\u2018GOODBYE\u2019)", restyled);
    }

    [Fact]
    public void AnApostropheBetweenLetters_IsLeftAlone()
    {
        // The reference's carve-out. A quote flanked by letters is an apostrophe, not a
        // closing quote, and turning "don't" into "don\u2019t" would be the restyle
        // introducing an error of its own.
        FileEdit.MatchResult match = FileEdit.Find("msg = \u2018hi\u2019\n", "msg = 'hi'");
        string restyled = FileEdit.Restyle("msg = 'don't'", match.Matched, match.Search);

        Assert.Contains("don't", restyled, StringComparison.Ordinal);
    }

    [Fact]
    public void AnExactMatch_IsNeverRestyled()
    {
        // Nothing was relaxed, so nothing may be rewritten. The model's bytes go in as
        // the model wrote them.
        const string replacement = "b = 'x'";
        Assert.Equal(replacement, FileEdit.Restyle(replacement, "a = 'x'", "a = 'x'"));
    }

    [Fact]
    public void RestyleRefusesToInventAMappingItDidNotMake()
    {
        // If the difference between the file's bytes and the model's is not one this
        // matcher folded, nothing is substituted. Guessing here would corrupt text on a
        // coincidence.
        const string replacement = "totally different";
        Assert.Equal(replacement, FileEdit.Restyle(replacement, "abc", "xyz"));
    }

    // ---- what tolerance must NOT do ---------------------------------------
    //
    // Every case below was a real defect found by adversarially reviewing this file and
    // reproduced by running it. They share one shape: a rung relaxed the match and then
    // the host wrote bytes the model never asked for, under a result line promising it
    // had not. Tolerance that corrupts is worse than a refusal, because a refusal is
    // something the model can read and act on.

    [Fact]
    public void AFileThatSpellsAQuoteBothWays_LeavesTheModelsBytesAlone()
    {
        // A file with ASCII quotes in the code and one curly apostrophe in a comment —
        // which is what an editor with smart quotes produces. The plain spelling is never
        // recorded as a "form" (the loop skips positions that already agree), so the one
        // typographic form used to win and BOTH string delimiters were converted, writing
        // invalid Python.
        const string matched = "label = 'x'  # don’t touch";
        const string search  = "label = 'x'  # don't touch";

        Assert.Equal(
            "label = 'y'  # don't touch",
            FileEdit.Restyle("label = 'y'  # don't touch", matched, search));
    }

    [Fact]
    public void AModelsSmartQuotes_AreWrittenBackAsTheFilesAscii()
    {
        // The reverse fold direction, which is the COMMON one: models emit typographic
        // quotes unprompted. The ladder matches it (both sides are folded), and the
        // restyle guard used to be one-directional — so it bailed and wrote the model's
        // curly quotes into an ASCII source file, turning what would have been a clean
        // "not found" into a written syntax error.
        FileEdit.MatchResult match = FileEdit.Find("msg = 'hi'\n", "msg = ‘hi’");

        Assert.True(match.Found);
        Assert.Equal(FileEdit.Rung.Punctuation, match.Rung);
        Assert.Equal("msg = 'bye'", FileEdit.Restyle("msg = ‘bye’", match.Matched, match.Search));
    }

    [Fact]
    public void TheReplacementGoesThroughTheSameTransformAsTheSearch()
    {
        // The line-number rung stripped prefixes from old_string and wrote new_string
        // verbatim, so a model that copied a read_file block into BOTH arguments — exactly
        // the mistake the rung exists to absorb — had "   42 | " written into its source.
        Assert.Equal(
            "    total = 1",
            FileEdit.ApplyRungTo("    42 |     total = 1", FileEdit.Rung.LineNumbers));

        Assert.Equal(
            "cafés",
            FileEdit.ApplyRungTo(@"cafés", FileEdit.Rung.UnicodeEscape));

        // An exact match relaxed nothing, so nothing may be rewritten.
        Assert.Equal(@"aé | b", FileEdit.ApplyRungTo(@"aé | b", FileEdit.Rung.Exact));
    }

    [Fact]
    public void TheBacktickIsNotAnApostrophe()
    {
        // A backtick is a template literal in JavaScript, command substitution in a shell
        // and a code span in Markdown. Folding it onto an apostrophe let an edit match
        // text that means something else — and, because it was the only ASCII character
        // the fold touched, it also made the fold gate answer differently for a substring
        // than for the whole string, which stopped a replace_all walk early.
        Assert.False(FileEdit.Find("const s = `hi`;\n", "const s = 'hi';").Found);
    }

    [Fact]
    public void ReplaceAllFindsEveryOccurrence_EvenPastTheLastNonAsciiCharacter()
    {
        // The gate that stopped the walk. `Find` is re-run on a shrinking substring, so a
        // gate keyed on "is anything non-ASCII" answered true for the whole file and false
        // once the walk passed the file's only non-ASCII character — abandoning the
        // remaining matches while the result reported them all as edited.
        const string file = "— title\nx = 'a'\ny = 'a'\nz = 'a'\n";

        FileEdit.MatchResult match = FileEdit.Find(file, "'a'");
        Assert.True(match.Found);
        Assert.Equal(3, match.Count);

        // Every one of them is still findable from a position past the em dash.
        FileEdit.MatchResult later = FileEdit.Find(file.Substring(file.IndexOf("y =", StringComparison.Ordinal)), "'a'");
        Assert.True(later.Found);
    }

    [Fact]
    public void AnEnormousNeedleAgainstAnEnormousFile_GivesUpRatherThanGrinding()
    {
        // Both strings come from the model and the folded rung cannot use an index, so it
        // is a multiplication with no deadline anywhere on the path. Giving up degrades to
        // "not found", which is a refusal the model can read.
        string content = new string('a', 400_000) + "—";
        string needle = new string('b', 400_000);

        Assert.False(FileEdit.Find(content, needle).Found);
    }
}
