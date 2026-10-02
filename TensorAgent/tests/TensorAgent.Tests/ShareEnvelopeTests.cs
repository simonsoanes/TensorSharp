// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;
using System.Text.Json;
using TensorAgent.Sharing;

namespace TensorAgent.Tests;

/// <summary>
/// The drop box between the "Ask TensorAgent" share extension and the app.
///
/// <para>
/// Every one of these is a two-process failure that has no other way of being caught.
/// The extension and the app are built together and never run together; the only thing
/// they agree on is a directory of files, and the app reads envelopes the extension
/// wrote seconds — or an app update — earlier.
/// </para>
/// </summary>
public sealed class ShareEnvelopeTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "tensoragent-share-" + Guid.NewGuid().ToString("N"));

    public ShareEnvelopeTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch (IOException) { }
    }

    private ShareEnvelopeStore Store(long? maxBytes = null) => maxBytes is { } cap
        ? new ShareEnvelopeStore(_root) { MaxTotalBytes = cap }
        : new ShareEnvelopeStore(_root);

    private static SharePayload Text(string text = "hello", string prompt = "") =>
        new() { Prompt = prompt, Items = { ShareItem.ForText(text) } };

    // =====================================================================================
    // writing and claiming
    // =====================================================================================

    [Fact]
    public void AnEnvelopeIsInvisibleUntilItIsWhole()
    {
        ShareEnvelopeStore store = Store();
        ShareEnvelopeWriter writer = store.BeginWrite();
        (string absolute, string relative) = writer.ReserveFile("notes.txt");
        File.WriteAllText(absolute, "the file");

        // Half written: the app must not be able to see this, or it reads a share.json
        // naming files that are still being copied.
        Assert.Empty(store.ListReady());

        writer.Commit(new SharePayload { Items = { ShareItem.ForFile(relative, "notes.txt", 8) } });
        Assert.Equal(new[] { writer.Id }, store.ListReady());
    }

    [Fact]
    public void AClaimedEnvelopeIsGoneFromTheBoxAndItsFilesAreStillThere()
    {
        ShareEnvelopeStore store = Store();
        ShareEnvelopeWriter writer = store.BeginWrite();
        (string absolute, string relative) = writer.ReserveFile("report.pdf");
        File.WriteAllBytes(absolute, new byte[] { 1, 2, 3, 4 });
        string id = writer.Commit(new SharePayload { Items = { ShareItem.ForFile(relative, "report.pdf", 4) } });

        ClaimedShare claim = Assert.IsType<ClaimedShare>(store.TryClaim(id, out string? why));
        Assert.Null(why);
        Assert.Empty(store.ListReady());

        // The files have to outlive the claim: the app copies them into its own upload
        // directory afterwards.
        string? resolved = ShareEnvelopeStore.ResolveFile(claim.Directory, claim.Payload.Items[0]);
        Assert.NotNull(resolved);
        Assert.Equal(4, new FileInfo(resolved!).Length);

        store.Discard(claim.Directory);
        Assert.False(Directory.Exists(claim.Directory));
    }

    [Fact]
    public void TwoReadersCannotBothClaimTheSameShareOrDeleteTheWinner()
    {
        ShareEnvelopeStore store = Store();
        string id = store.BeginWrite().Commit(Text());

        ClaimedShare winner = Assert.IsType<ClaimedShare>(store.TryClaim(id, out _));
        string sentinel = Path.Combine(winner.Directory, "still-importing");
        File.WriteAllText(sentinel, "owned by the winning reader");
        // The second claim is the app's own belt-and-braces foreground drain racing
        // for the same envelope. It must lose quietly, not deliver the share twice or
        // delete files the winner is still importing.
        Assert.Null(store.TryClaim(id, out string? why));
        Assert.Contains("already taken", why!, StringComparison.Ordinal);
        Assert.Equal("owned by the winning reader", File.ReadAllText(sentinel));
    }

    [Fact]
    public async Task ManyConcurrentReadersProduceExactlyOneIntactClaim()
    {
        // The Windows rename race is intermittent; exercise independent envelopes
        // repeatedly, with independent store instances as separate importers use.
        for (int iteration = 0; iteration < 20; iteration++)
        {
            ShareEnvelopeWriter writer = Store().BeginWrite();
            (string absolute, string relative) = writer.ReserveFile("race.txt");
            File.WriteAllText(absolute, "the bytes must survive the race");
            string id = writer.Commit(new SharePayload
            {
                Items = { ShareItem.ForFile(relative, "race.txt", new FileInfo(absolute).Length) },
            });

            var start = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            Task<ClaimedShare?>[] readers = Enumerable.Range(0, 32).Select(readerIndex => Task.Run(async () =>
            {
                await start.Task;
                return new ShareEnvelopeStore(_root).TryClaim(id, out _);
            })).ToArray();

            start.SetResult();
            ClaimedShare?[] results = await Task.WhenAll(readers);
            ClaimedShare claim = Assert.Single(results.OfType<ClaimedShare>());
            string? resolved = ShareEnvelopeStore.ResolveFile(claim.Directory, claim.Payload.Items[0]);
            Assert.NotNull(resolved);
            Assert.Equal("the bytes must survive the race", File.ReadAllText(resolved!));
        }
    }

    [Fact]
    public void ATransientlyFailedClaimCanBeReleasedAndClaimedAgain()
    {
        ShareEnvelopeStore store = Store();
        string id = store.BeginWrite().Commit(Text("retry me"));
        ClaimedShare first = Assert.IsType<ClaimedShare>(store.TryClaim(id, out _));

        Assert.True(store.Release(first));
        Assert.Equal(new[] { id }, store.ListReady());

        ClaimedShare retry = Assert.IsType<ClaimedShare>(store.TryClaim(id, out _));
        Assert.Equal("retry me", retry.Payload.Items[0].Text);
    }

    [Fact]
    public void AClaimLeftByATerminatedAppIsRecoveredOnTheNextStart()
    {
        ShareEnvelopeStore firstProcess = Store();
        string id = firstProcess.BeginWrite().Commit(Text("survive a crash"));
        Assert.NotNull(firstProcess.TryClaim(id, out _));

        var nextProcess = new ShareEnvelopeStore(_root);
        Assert.Equal(1, nextProcess.RecoverClaims());
        Assert.Equal(new[] { id }, nextProcess.ListReady());
        ClaimedShare recovered = Assert.IsType<ClaimedShare>(nextProcess.TryClaim(id, out _));
        Assert.Equal("survive a crash", recovered.Payload.Items[0].Text);
    }

    [Fact]
    public void SharesAreDeliveredOldestFirst()
    {
        ShareEnvelopeStore store = Store();
        var ids = new List<string>();
        for (int i = 0; i < 3; i++)
        {
            ShareEnvelopeWriter writer = store.BeginWrite();
            ids.Add(writer.Commit(Text("share " + i)));
            // The list is ordered by the payload file's write time, which has to be
            // distinguishable for the order to mean anything.
            File.SetLastWriteTimeUtc(
                Path.Combine(_root, ids[^1], "share.json"),
                new DateTime(2026, 1, 1, 0, 0, i, DateTimeKind.Utc));
        }
        Assert.Equal(ids, store.ListReady());
    }

    // =====================================================================================
    // what a share must never be able to do
    // =====================================================================================

    [Theory]
    [InlineData("../../../settings.json")]
    [InlineData("files/../../conversations/a.json")]
    [InlineData("/etc/passwd")]
    public void AnEnvelopeCannotNameAFileOutsideItself(string path)
    {
        // The share.json comes from another process. The directory two levels above an
        // envelope holds every conversation the user has ever had.
        string envelope = Path.Combine(_root, "envelope");
        Directory.CreateDirectory(envelope);
        Assert.Null(ShareEnvelopeStore.ResolveFile(
            envelope, new ShareItem { Kind = ShareItemKinds.File, File = path }));
    }

    [Theory]
    [InlineData("../../settings.json", "settings.json")]
    [InlineData("/tmp/evil.sh", "evil.sh")]
    [InlineData("", "shared")]
    [InlineData("...", "shared")]
    [InlineData(".hidden", "hidden")]
    public void AFileNameFromAnotherAppIsOnlyEverAFileName(string given, string expected)
        => Assert.Equal(expected, ShareEnvelopeWriter.SafeFileName(given));

    [Fact]
    public void TwoPhotosCalledTheSameThingBothSurvive()
    {
        // Sharing several photos out of Messages routinely produces IMG_0001 twice.
        ShareEnvelopeWriter writer = Store().BeginWrite();
        (_, string first) = writer.ReserveFile("IMG_0001.jpg");
        (_, string second) = writer.ReserveFile("IMG_0001.jpg");
        Assert.Equal("files/IMG_0001.jpg", first);
        Assert.Equal("files/IMG_0001-2.jpg", second);
    }

    [Fact]
    public void ALongUnicodeFileNameFitsAnIosPathComponentAndKeepsItsExtension()
    {
        string safe = ShareEnvelopeWriter.SafeFileName(new string('界', 120) + ".png");

        Assert.True(Encoding.UTF8.GetByteCount(safe) <= 255, $"name used {Encoding.UTF8.GetByteCount(safe)} UTF-8 bytes");
        Assert.EndsWith(".png", safe, StringComparison.Ordinal);
        _ = new UTF8Encoding(false, true).GetBytes(safe);
    }

    [Fact]
    public void ACollisionSuffixDoesNotPushALongUnicodeNamePastTheIosLimit()
    {
        ShareEnvelopeWriter writer = Store().BeginWrite();
        string source = new string('界', 120) + ".jpeg";
        (_, string first) = writer.ReserveFile(source);
        (_, string second) = writer.ReserveFile(source);
        string firstName = Path.GetFileName(first);
        string secondName = Path.GetFileName(second);

        Assert.True(Encoding.UTF8.GetByteCount(firstName) <= 255);
        Assert.True(Encoding.UTF8.GetByteCount(secondName) <= 255);
        Assert.EndsWith("-2.jpeg", secondName, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("not-hex-at-all")]
    [InlineData("../..")]
    [InlineData("0123456789abcdef0123456789abcdeF")]   // upper case is not the id format
    public void AShareIdThatIsNotAShareIdIsRefused(string id)
    {
        Assert.False(ShareIds.IsValid(id));
        Assert.Null(Store().TryClaim(id, out _));
    }

    // =====================================================================================
    // versions, and the app update that happens while an envelope is waiting
    // =====================================================================================

    [Fact]
    public void AnEnvelopeFromANewerBuildIsRefusedRatherThanReadHalfWay()
    {
        ShareEnvelopeStore store = Store();
        string id = ShareIds.New();
        string directory = Path.Combine(_root, id);
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "share.json"), """
            { "version": 99, "id": "x", "items": [ { "kind": "text", "text": "hello" } ] }
            """);

        Assert.Null(store.TryClaim(id, out string? why));
        Assert.Contains("format 99", why!, StringComparison.Ordinal);
        // And it is thrown away rather than re-offered on every launch forever.
        Assert.Empty(store.ListReady());
    }

    [Fact]
    public void AnUnreadableEnvelopeIsDiscardedRatherThanRetriedForever()
    {
        ShareEnvelopeStore store = Store();
        string id = ShareIds.New();
        Directory.CreateDirectory(Path.Combine(_root, id));
        File.WriteAllText(Path.Combine(_root, id, "share.json"), "{ this is not json");

        Assert.Null(store.TryClaim(id, out string? why));
        Assert.Contains("could not be read", why!, StringComparison.Ordinal);
        Assert.False(Directory.Exists(Path.Combine(_root, id)));
    }

    [Fact]
    public void TheIdInTheFileNeverOverridesTheDirectoryItWasFoundIn()
    {
        ShareEnvelopeStore store = Store();
        ShareEnvelopeWriter writer = store.BeginWrite();
        var payload = new SharePayload { Id = "ffffffffffffffffffffffffffffffff", Items = { ShareItem.ForText("x") } };
        string id = writer.Commit(payload);

        ClaimedShare claim = Assert.IsType<ClaimedShare>(store.TryClaim(id, out _));
        Assert.Equal(id, claim.Payload.Id);
    }

    // =====================================================================================
    // housekeeping
    // =====================================================================================

    [Fact]
    public void OldSharesAndInterruptedWritesArePurged()
    {
        ShareEnvelopeStore store = new(_root) { MaxAge = TimeSpan.FromHours(1) };
        string fresh = store.BeginWrite().Commit(Text("recent"));
        string stale = store.BeginWrite().Commit(Text("old"));
        Directory.SetLastWriteTimeUtc(Path.Combine(_root, stale), DateTime.UtcNow.AddDays(-3));

        // An extension the system killed mid-copy leaves this behind.
        string abandoned = Path.Combine(_root, ShareIds.New() + ".partial");
        Directory.CreateDirectory(abandoned);
        Directory.SetLastWriteTimeUtc(abandoned, DateTime.UtcNow.AddDays(-3));

        Assert.Equal(2, store.Purge(DateTimeOffset.UtcNow));
        Assert.Equal(new[] { fresh }, store.ListReady());
    }

    [Fact]
    public void APartialDirectoryBeingWrittenRightNowIsNotPurgedUnderTheExtension()
    {
        ShareEnvelopeStore store = new(_root) { MaxAge = TimeSpan.FromHours(1) };
        ShareEnvelopeWriter writer = store.BeginWrite();

        // The app launching at the moment the extension is copying a video must not
        // delete the directory out from under it.
        Assert.Equal(0, store.Purge(DateTimeOffset.UtcNow));
        Assert.True(Directory.Exists(writer.PartialDirectory));
    }

    [Fact]
    public void AClaimBackingAnOldUnsentDraftIsNeverPurged()
    {
        ShareEnvelopeStore store = new(_root) { MaxAge = TimeSpan.FromHours(1) };
        string id = store.BeginWrite().Commit(Text("still editing"));
        ClaimedShare claim = Assert.IsType<ClaimedShare>(store.TryClaim(id, out _));
        Directory.SetLastWriteTimeUtc(claim.Directory, DateTime.UtcNow.AddDays(-3));

        Assert.Equal(0, store.Purge(DateTimeOffset.UtcNow));
        Assert.True(Directory.Exists(claim.Directory));
    }

    [Fact]
    public void TheBoxRefusesToGrowWithoutLimit()
    {
        // Everything in the envelope counts, the payload JSON as well as the file: the
        // cap is about the user's storage, not about one number in it.
        ShareEnvelopeStore store = Store();
        ShareEnvelopeWriter writer = store.BeginWrite();
        (string absolute, string relative) = writer.ReserveFile("big.bin");
        File.WriteAllBytes(absolute, new byte[900]);
        writer.Commit(new SharePayload { Items = { ShareItem.ForFile(relative, "big.bin", 900) } });

        long held = store.TotalBytes();
        Assert.True(held >= 900, $"the box holds {held} bytes, expected at least the 900-byte file");

        ShareEnvelopeStore capped = Store(maxBytes: held + 100);
        Assert.True(capped.HasRoomFor(100));
        Assert.False(capped.HasRoomFor(101));
    }

    [Fact]
    public void CommitUsesActualPartialBytesAndLeavesNoReadyEnvelopeWhenFull()
    {
        ShareEnvelopeStore store = Store(maxBytes: 900);
        ShareEnvelopeWriter writer = store.BeginWrite();
        (string absolute, string relative) = writer.ReserveFile("actual.bin");
        File.WriteAllBytes(absolute, new byte[900]);

        // The payload deliberately lies about the size. Capacity is a disk invariant,
        // so Commit must measure the partial directory rather than trust wire metadata.
        ShareInboxFullException error = Assert.Throws<ShareInboxFullException>(() =>
            writer.Commit(new SharePayload { Items = { ShareItem.ForFile(relative, "actual.bin", 1) } }));

        Assert.True(error.IncomingBytes > 900);
        Assert.Empty(store.ListReady());
        Assert.True(Directory.Exists(writer.PartialDirectory));
        writer.Abandon();
    }

    [Fact]
    public async Task ConcurrentCommitsCannotTogetherCrossTheInboxCap()
    {
        var store = new ShareEnvelopeStore(_root) { MaxTotalBytes = 4096 };
        ShareEnvelopeWriter first = store.BeginWrite();
        ShareEnvelopeWriter second = store.BeginWrite();
        (string firstPath, string firstRelative) = first.ReserveFile("first.bin");
        (string secondPath, string secondRelative) = second.ReserveFile("second.bin");
        File.WriteAllBytes(firstPath, new byte[3000]);
        File.WriteAllBytes(secondPath, new byte[3000]);

        var start = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<Exception?> CommitAsync(ShareEnvelopeWriter writer, string relative, string name) => Task.Run(async () =>
        {
            await start.Task;
            try
            {
                writer.Commit(new SharePayload { Items = { ShareItem.ForFile(relative, name, 3000) } });
                return null;
            }
            catch (Exception ex)
            {
                return ex;
            }
        });

        Task<Exception?> one = CommitAsync(first, firstRelative, "first.bin");
        Task<Exception?> two = CommitAsync(second, secondRelative, "second.bin");
        start.SetResult();
        Exception?[] outcomes = await Task.WhenAll(one, two);

        Assert.Single(outcomes, outcome => outcome is null);
        Assert.Single(outcomes, outcome => outcome is ShareInboxFullException);
        Assert.Single(store.ListReady());
        Assert.True(store.TotalBytes() <= store.MaxTotalBytes,
            $"the inbox holds {store.TotalBytes()} bytes past its {store.MaxTotalBytes}-byte cap");
        first.Abandon();
        second.Abandon();
    }

    // =====================================================================================
    // the format itself
    // =====================================================================================

    [Fact]
    public void AnEnvelopeRoundTripsThroughItsOwnJson()
    {
        var payload = new SharePayload
        {
            Id = ShareIds.New(),
            CreatedAt = new DateTimeOffset(2026, 9, 7, 12, 0, 0, TimeSpan.Zero),
            SourceApp = "Safari",
            Prompt = "What is this about?\nSecond line's apostrophe",
            NewChat = false,
            AutoSend = false,
            Items =
            {
                ShareItem.ForPage("https://example.com/a", "A title", "body", "selected"),
                ShareItem.ForFile("files/x.png", "x.png", 12, "public.png", "image/png"),
            },
            Notes = { "one note" },
        };

        string json = JsonSerializer.Serialize(payload, SharePayload.JsonOptions);
        SharePayload back = JsonSerializer.Deserialize<SharePayload>(json, SharePayload.JsonOptions)!;

        Assert.Equal(payload.Prompt, back.Prompt);
        Assert.Equal(payload.SourceApp, back.SourceApp);
        Assert.False(back.NewChat);
        Assert.False(back.AutoSend);
        Assert.Equal(2, back.Items.Count);
        Assert.Equal("selected", back.Items[0].Selection);
        Assert.Equal("image/png", back.Items[1].MimeType);
        Assert.Equal(new[] { "one note" }, back.Notes);
    }

    [Fact]
    public void ASharePayloadSerializesThroughTheSourceGeneratedResolver()
    {
        // The extension is a TRIMMED assembly, where reflection-based serialization
        // fails at run time with no build warning. This asserts the options carry a
        // resolver at all -- the one thing that would silently stop being true if
        // SharePayloadJsonContext were removed or the property reset.
        Assert.NotNull(SharePayload.JsonOptions.TypeInfoResolver);
        Assert.Contains(
            "SharePayloadJsonContext",
            SharePayload.JsonOptions.TypeInfoResolver!.GetType().FullName,
            StringComparison.Ordinal);
    }

    [Fact]
    public void AnEmptyOrUnversionedShareIsNotUsable()
    {
        Assert.False(new SharePayload { Id = ShareIds.New() }.IsUsable(out string why));
        Assert.Contains("carried nothing", why, StringComparison.Ordinal);

        Assert.False(new SharePayload { Id = "nope", Items = { ShareItem.ForText("x") } }.IsUsable(out why));
        Assert.Contains("no valid id", why, StringComparison.Ordinal);
    }
}

/// <summary>
/// The drop box's two other jobs — refusing what does not belong to it, and making
/// room — both of which a review found were not doing what their comments said.
/// </summary>
public sealed class ShareEnvelopeHousekeepingTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "tensoragent-house-" + Guid.NewGuid().ToString("N"));

    public ShareEnvelopeHousekeepingTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch (IOException) { }
    }

    [SkippableFact]
    public void AFileInsideTheEnvelopeThatIsALinkOutOfItIsRefused()
    {
        // Path.GetFullPath resolves "..", not symlinks, while every reader afterwards
        // follows them. A files/photo.png that is a link to the user's conversations
        // passed a purely textual containment check and was then read into the chat.
        string outside = Path.Combine(_root, "settings.json");
        File.WriteAllText(outside, "{\"secret\":true}");

        string envelope = Path.Combine(_root, "envelope");
        Directory.CreateDirectory(Path.Combine(envelope, "files"));
        string link = Path.Combine(envelope, "files", "photo.png");
        TestPlatforms.CreateFileSymlink(link, outside);

        var item = new ShareItem { Kind = ShareItemKinds.File, File = "files/photo.png", FileName = "photo.png" };
        Assert.Null(ShareEnvelopeStore.ResolveFile(envelope, item));
    }

    [Fact]
    public void AnOrdinaryFileInsideTheEnvelopeIsStillFound()
    {
        string envelope = Path.Combine(_root, "ok");
        Directory.CreateDirectory(Path.Combine(envelope, "files"));
        File.WriteAllText(Path.Combine(envelope, "files", "note.txt"), "hello");

        var item = new ShareItem { Kind = ShareItemKinds.File, File = "files/note.txt", FileName = "note.txt" };
        Assert.NotNull(ShareEnvelopeStore.ResolveFile(envelope, item));
    }

    [Fact]
    public void MakingRoomLowersTheAgeRatherThanMovingTheClock()
    {
        // Purge(now.AddDays(-1)) compares against now - 1 day - MaxAge, which deletes
        // strictly LESS than an ordinary purge — so the one call that existed to free
        // space could never free any.
        var store = new ShareEnvelopeStore(_root) { MaxAge = TimeSpan.FromDays(2) };
        string id = store.BeginWrite().Commit(
            new SharePayload { Items = { ShareItem.ForText("a day old") } });
        Directory.SetLastWriteTimeUtc(Path.Combine(_root, id), DateTime.UtcNow.AddHours(-12));

        // The routine purge leaves it: it is well inside the two-day grace.
        Assert.Equal(0, store.Purge(DateTimeOffset.UtcNow));
        // The make-room purge takes it, because it lowers the age.
        Assert.Equal(1, store.Purge(DateTimeOffset.UtcNow, TimeSpan.FromHours(6)));
        Assert.Empty(store.ListReady());
    }
}
