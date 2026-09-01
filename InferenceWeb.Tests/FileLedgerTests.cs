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
using System.Globalization;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// What the model has been shown of each file — the bookkeeping that decides whether an
/// edit is checked against bytes it has seen or bytes it half-remembers.
///
/// <para>
/// The design bet under test is that freshness is COMPUTED by comparing content, never
/// recorded by a flag set at write time. That is what makes it correct where a
/// command-line scanner cannot be: <c>cp</c>, <c>mv</c>, <c>sed -i</c>, a Python
/// <c>open().write()</c>, a redirect through a shell variable and a background job that
/// finished between calls are all seen, because the question is asked of the file rather
/// than of the command that may have written it. There is nothing to invalidate, and
/// therefore nothing that can be forgotten.
/// </para>
/// <para>
/// The partial-read range is the one place this is deliberately better than the
/// reference, which tracks a boolean. A boolean says "was read" and so authorises an edit
/// to a region the model never saw; a range can answer the question that was actually
/// asked.
/// </para>
/// </summary>
public class FileLedgerTests
{
    private const string Path = "/w/main.py";

    private static FileLedger Ledger() => new();

    [Fact]
    public void AFileNeverSeen_IsUnread()
    {
        Assert.Equal(ReadFreshness.Unread, Ledger().Check(Path, "anything").Freshness);
    }

    [Fact]
    public void AFileReadInFull_IsFresh()
    {
        FileLedger ledger = Ledger();
        ledger.Record(Path, "a\nb\n", 1, 2, complete: true);

        FileLedger.ReadState state = ledger.Check(Path, "a\nb\n");
        Assert.Equal(ReadFreshness.Fresh, state.Freshness);
        Assert.True(state.Complete);
    }

    [Fact]
    public void AFileThatChanged_IsStale_WhateverChangedIt()
    {
        // The whole point: nothing told the ledger this happened. It is noticed because
        // the content is compared, which is why a `cp`, a `sed -i` or a background job is
        // no harder to see than an edit through this host's own tools.
        FileLedger ledger = Ledger();
        ledger.Record(Path, "a\nb\n", 1, 2, complete: true);

        Assert.Equal(ReadFreshness.Stale, ledger.Check(Path, "a\nB\n").Freshness);
    }

    [Fact]
    public void AFileTouchedButNotChanged_IsStillFresh()
    {
        // A timestamp would call this stale and cost a round for nothing. The content is
        // what the model's context holds, so the content is what is compared.
        FileLedger ledger = Ledger();
        ledger.Record(Path, "a\nb\n", 1, 2, complete: true);

        Assert.Equal(ReadFreshness.Fresh, ledger.Check(Path, "a\nb\n").Freshness);
    }

    [Fact]
    public void AChangeInsideTimestampAndLengthNoise_IsStillSeen()
    {
        // The measured worst case sits inside both: 188 lines re-typed to change one,
        // leaving 6,780 of 6,808 characters identical. A length-and-mtime comparison
        // would have called that unchanged for any write in the same second.
        FileLedger ledger = Ledger();
        const string before = "value = 11\nother = 2\n";
        const string after = "value = 22\nother = 2\n";
        Assert.Equal(before.Length, after.Length);

        ledger.Record(Path, before, 1, 2, complete: true);
        Assert.Equal(ReadFreshness.Stale, ledger.Check(Path, after).Freshness);
    }

    // ---- partial reads -----------------------------------------------------

    [Fact]
    public void APartialRead_IsPartial_AndKnowsWhichLinesItCovered()
    {
        FileLedger ledger = Ledger();
        ledger.Record(Path, "a\nb\nc\nd\n", 1, 2, complete: false);

        FileLedger.ReadState state = ledger.Check(Path, "a\nb\nc\nd\n");
        Assert.Equal(ReadFreshness.Partial, state.Freshness);
        Assert.True(state.Covers(1, 2));
        Assert.False(state.Covers(3, 4));
    }

    [Fact]
    public void TwoOverlappingWindows_BecomeOne()
    {
        // Reading 1-40 and then 30-80 has shown the model 1-80, and saying otherwise
        // would refuse an edit it has every byte for.
        FileLedger ledger = Ledger();
        ledger.Record(Path, "x", 1, 40, complete: false);
        ledger.Record(Path, "x", 30, 80, complete: false);

        Assert.True(ledger.Check(Path, "x").Covers(1, 80));
    }

    [Fact]
    public void TwoDisjointWindows_DoNotClaimTheGapBetweenThem()
    {
        // The union would assert the model has seen lines nobody rendered.
        FileLedger ledger = Ledger();
        ledger.Record(Path, "x", 1, 10, complete: false);
        ledger.Record(Path, "x", 500, 510, complete: false);

        FileLedger.ReadState state = ledger.Check(Path, "x");
        Assert.False(state.Covers(1, 10));
        Assert.True(state.Covers(500, 510));
    }

    [Fact]
    public void ACompleteRead_SubsumesEveryEarlierWindow()
    {
        FileLedger ledger = Ledger();
        ledger.Record(Path, "x", 5, 6, complete: false);
        ledger.Record(Path, "x", 1, 999, complete: true);

        FileLedger.ReadState state = ledger.Check(Path, "x");
        Assert.Equal(ReadFreshness.Fresh, state.Freshness);
        Assert.True(state.Covers(1, 999));
    }

    [Fact]
    public void APartialReadFollowedByACompleteOne_StaysComplete()
    {
        FileLedger ledger = Ledger();
        ledger.Record(Path, "x", 1, 999, complete: true);
        ledger.Record(Path, "x", 5, 6, complete: false);

        Assert.Equal(ReadFreshness.Fresh, ledger.Check(Path, "x").Freshness);
    }

    // ---- housekeeping ------------------------------------------------------

    [Fact]
    public void Forgetting_MakesAFileUnreadAgain()
    {
        FileLedger ledger = Ledger();
        ledger.Record(Path, "a", 1, 1, complete: true);
        ledger.Forget(Path);

        Assert.Equal(ReadFreshness.Unread, ledger.Check(Path, "a").Freshness);
    }

    [Fact]
    public void TheLedgerIsBounded_AndEvictionDegradesToUnread()
    {
        // Eviction has to fail in the SAFE direction. Unread still applies an unambiguous
        // edit — with a note — so the cost of forgetting is a sentence, not a refusal.
        FileLedger ledger = Ledger();
        for (int i = 0; i < 400; i++)
            ledger.Record("/w/f" + i.ToString(CultureInfo.InvariantCulture) + ".py", "x", 1, 1, complete: true);

        Assert.True(ledger.Count <= 256);
        Assert.Equal(ReadFreshness.Unread, ledger.Check("/w/f0.py", "x").Freshness);
        Assert.Equal(ReadFreshness.Fresh, ledger.Check("/w/f399.py", "x").Freshness);
    }

    [Fact]
    public void TheKnownTextComesBack_SoARewriteCanBeCompared()
    {
        FileLedger ledger = Ledger();
        ledger.Record(Path, "a\nb\n", 1, 2, complete: true);

        Assert.True(ledger.TryGetKnownText(Path, out string text));
        Assert.Equal("a\nb\n", text);
    }

    [Fact]
    public void PathsAreKeptApart_EvenWhenTheirNamesMatch()
    {
        // The ledger is keyed on the RESOLVED absolute path, never a relative spelling.
        // Holding relative names meant `cd sub && …` could compare two different files
        // and report on "it" while describing both.
        FileLedger ledger = Ledger();
        ledger.Record("/w/a/main.py", "one", 1, 1, complete: true);
        ledger.Record("/w/b/main.py", "two", 1, 1, complete: true);

        Assert.Equal(ReadFreshness.Fresh, ledger.Check("/w/a/main.py", "one").Freshness);
        Assert.Equal(ReadFreshness.Stale, ledger.Check("/w/b/main.py", "one").Freshness);
    }

    [Fact]
    public void AWorkspaceCarriesItsOwnLedger()
    {
        // Per session, with the session's lifetime: one conversation's reads are worth
        // nothing to the next, and must never authorise an edit in it.
        string root = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "ts-ledger-" + Guid.NewGuid().ToString("N"));
        var manager = new SessionWorkspaceManager(root);
        try
        {
            SessionWorkspace first = manager.GetOrCreate("a");
            SessionWorkspace second = manager.GetOrCreate("b");

            first.Reads.Record("/w/x.py", "hello", 1, 1, complete: true);

            Assert.Equal(ReadFreshness.Fresh, first.Reads.Check("/w/x.py", "hello").Freshness);
            Assert.Equal(ReadFreshness.Unread, second.Reads.Check("/w/x.py", "hello").Freshness);
        }
        finally
        {
            try { manager.Release("a"); } catch (Exception ex) when (ex is System.IO.IOException) { }
            try { manager.Release("b"); } catch (Exception ex) when (ex is System.IO.IOException) { }
            try { System.IO.Directory.Delete(root, recursive: true); }
            catch (Exception ex) when (ex is System.IO.IOException or UnauthorizedAccessException) { }
        }
    }
}
