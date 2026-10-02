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

namespace TensorAgent.Sharing;

/// <summary>
/// The drop box between the share extension and the app: a directory of envelopes,
/// each one directory holding a <c>share.json</c> and the files it names.
///
/// <para>
/// Two processes, no lock. The extension is started by iOS inside whatever app the
/// user was in, writes an envelope and dies; the app reads it, perhaps seconds later,
/// perhaps after a launch. They never overlap on one envelope, but they can easily
/// overlap on the DIRECTORY, and the app must never see a half-written share — a
/// <c>share.json</c> that names three files while the third is still being copied is
/// exactly the kind of failure that reproduces once a week and never in a test.
/// </para>
/// <para>
/// So writing is two-phase and finishes with a rename. Everything is written under
/// <c>&lt;id&gt;.partial</c>, and the last thing the extension does is rename that
/// directory to <c>&lt;id&gt;</c>. A rename within one filesystem is atomic, so an
/// envelope is either absent or complete, and a partial directory left behind by an
/// extension the system killed mid-copy is visibly not an envelope. The reader claims
/// one by renaming again, to <c>&lt;id&gt;.claimed</c>, so two readers racing for the
/// same share cannot both get it: the loser's rename fails.
/// </para>
/// <para>
/// It is a plain directory path rather than anything iOS-specific on purpose. On the
/// device that path is inside the App Group container both processes can see; in a
/// test it is a temporary directory; and neither this class nor anything it does
/// knows the difference.
/// </para>
/// </summary>
public sealed class ShareEnvelopeStore
{
    private const string PayloadFileName = "share.json";
    private const string FilesDirectoryName = "files";
    private const string PartialSuffix = ".partial";
    private const string ClaimedSuffix = ".claimed";
    private const string AcknowledgedSuffix = ".acknowledged";

    /// <summary>JSON is metadata, never the shared file itself; bound it before parsing.</summary>
    public const long MaxPayloadBytes = 4L * 1024 * 1024;

    /// <summary>
    /// How much the drop box may hold before new shares are refused.
    ///
    /// <para>
    /// It is the user's storage and the app is not the only thing using it. The cap is
    /// generous — a shared video is the big case and one of those fits comfortably —
    /// but it has to exist: an envelope nobody claims lives until the purge, and a user
    /// who shares repeatedly into an app they then never open would otherwise fill the
    /// device with copies of files they still have.
    /// </para>
    /// </summary>
    public long MaxTotalBytes { get; init; } = 512L * 1024 * 1024;

    /// <summary>How long an unclaimed envelope survives. See <see cref="Purge"/>.</summary>
    public TimeSpan MaxAge { get; init; } = TimeSpan.FromDays(2);

    /// <summary>
    /// A half-written extension handoff is never usable. Give an active copy a short
    /// grace period, then reclaim it much sooner than a completed user share.
    /// </summary>
    public TimeSpan PartialMaxAge { get; init; } = TimeSpan.FromMinutes(15);

    /// <summary>The directory the envelopes live in. Created on first use.</summary>
    public string Root { get; }

    public ShareEnvelopeStore(string root)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        Root = Path.GetFullPath(root);
    }

    /// <summary>Start writing an envelope. See <see cref="ShareEnvelopeWriter"/>.</summary>
    public ShareEnvelopeWriter BeginWrite(string? id = null)
    {
        string shareId = id ?? ShareIds.New();
        if (!ShareIds.IsValid(shareId))
            throw new ArgumentException("A share id is 32 lowercase hex characters.", nameof(id));

        Directory.CreateDirectory(Root);
        string partial = Path.Combine(Root, shareId + PartialSuffix);
        if (Directory.Exists(partial))
            Directory.Delete(partial, recursive: true);
        Directory.CreateDirectory(Path.Combine(partial, FilesDirectoryName));
        return new ShareEnvelopeWriter(this, shareId, partial);
    }

    /// <summary>
    /// The ids of the envelopes that are complete and unclaimed, oldest first.
    ///
    /// <para>
    /// Oldest first because they are then delivered in the order the user made them,
    /// which is the order they will make sense in. Two shares in a row from the same
    /// page is not a hypothetical: the share sheet is a fast thing to tap twice.
    /// </para>
    /// </summary>
    public IReadOnlyList<string> ListReady()
    {
        if (!Directory.Exists(Root))
            return Array.Empty<string>();

        var ready = new List<(string Id, DateTime When)>();
        foreach (string directory in Directory.EnumerateDirectories(Root))
        {
            string name = Path.GetFileName(directory);
            if (!ShareIds.IsValid(name) || !File.Exists(Path.Combine(directory, PayloadFileName)))
                continue;
            DateTime when;
            try { when = File.GetLastWriteTimeUtc(Path.Combine(directory, PayloadFileName)); }
            catch (IOException) { continue; }
            ready.Add((name, when));
        }
        ready.Sort((a, b) => a.When.CompareTo(b.When));
        return ready.Select(r => r.Id).ToArray();
    }

    /// <summary>
    /// Take one envelope, so that nobody else can.
    ///
    /// <para>
    /// The rename is the claim. It is also what makes a crash mid-import survivable in
    /// the right direction: a claimed envelope is no longer offered to a second reader.
    /// At the next containing-app start <see cref="RecoverClaims"/> returns a claim left
    /// by a terminated process to the ready queue, where deterministic upload staging
    /// lets it be imported again without accumulating duplicate file families.
    /// </para>
    /// </summary>
    /// <returns>The claim, or null when the envelope is gone, unreadable, or a version this build cannot read.</returns>
    public ClaimedShare? TryClaim(string id, out string? reason)
    {
        reason = null;
        if (!ShareIds.IsValid(id))
        {
            reason = "the share id is not a share id";
            return null;
        }

        string ready = Path.Combine(Root, id);
        string claimed = Path.Combine(Root, id + ClaimedSuffix);
        FileStream? claimLease = null;
        try
        {
            // Windows can complete two renames from handles opened on the same
            // directory before either caller moves it. Serialize claimants with
            // a lease outside the envelope (an open child would prevent its move).
            // DeleteOnClose also releases the lease after a process crash.
            if (OperatingSystem.IsWindows())
                claimLease = new FileStream(Path.Combine(Root, id + ".claim-lock"),
                    FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose);
            // Never delete an existing claim here. Another reader may still be
            // importing it; deleting that directory was a race that removed its files
            // while it had them open. A claim left by a crashed process is recovered
            // explicitly at the next app start by RecoverClaims().
            if (Directory.Exists(claimed))
            {
                claimLease?.Dispose();
                reason = "the share was already taken or is still being imported";
                return null;
            }
            Directory.Move(ready, claimed);
        }
        catch (Exception ex) when (ex is IOException or DirectoryNotFoundException or UnauthorizedAccessException)
        {
            claimLease?.Dispose();
            // Lost the race, or it was never there. Both are "somebody else has it".
            reason = "the share was already taken or is no longer there";
            return null;
        }

        using (claimLease)
            return ReadClaim(id, claimed, out reason);
    }

    private ClaimedShare? ReadClaim(string id, string claimed, out string? reason)
    {
        reason = null;
        SharePayload? payload;
        try
        {
            string payloadPath = Path.Combine(claimed, PayloadFileName);
            long payloadBytes = new FileInfo(payloadPath).Length;
            if (payloadBytes <= 0 || payloadBytes > MaxPayloadBytes)
            {
                reason = payloadBytes <= 0
                    ? "the share metadata was empty"
                    : $"the share metadata was too large ({payloadBytes:N0} bytes)";
                Discard(claimed);
                return null;
            }
            using FileStream stream = File.OpenRead(payloadPath);
            payload = JsonSerializer.Deserialize(stream, SharePayloadJsonContext.Default.SharePayload);
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            reason = "the share could not be read: " + ex.Message;
            Discard(claimed);
            return null;
        }

        string why = string.Empty;
        if (payload is null || !payload.IsUsable(out why))
        {
            reason = payload is null ? "the share was empty" : why;
            Discard(claimed);
            return null;
        }

        // The id in the file is not trusted over the id of the directory it was found
        // in: the directory name is what this process resolved, and the field is what
        // some other process wrote into it.
        payload.Id = id;
        return new ClaimedShare(payload, claimed);
    }

    /// <summary>
    /// Put claims left by a terminated app process back into the ready queue.
    /// Call once when the containing app constructs its host, before any import starts;
    /// the extension must not call it while an app could be processing a claim.
    /// </summary>
    public int RecoverClaims()
    {
        if (!Directory.Exists(Root))
            return 0;

        int recovered = 0;
        foreach (string claimed in Directory.EnumerateDirectories(Root, "*" + ClaimedSuffix))
        {
            string name = Path.GetFileName(claimed);
            string id = name[..^ClaimedSuffix.Length];
            if (!ShareIds.IsValid(id))
                continue;
            string ready = Path.Combine(Root, id);
            if (Directory.Exists(ready))
                continue;
            try
            {
                Directory.Move(claimed, ready);
                recovered++;
            }
            catch (Exception ex) when (ex is IOException or DirectoryNotFoundException or UnauthorizedAccessException)
            {
                // A second startup/drain won the rename. The remaining directory is
                // preserved; it is never safe to delete somebody else's claim.
            }
        }
        return recovered;
    }

    /// <summary>
    /// Return a transiently failed claim to the ready queue. If a ready copy already
    /// exists, preserve both paths and let age-based cleanup handle the stale claim.
    /// </summary>
    public bool Release(ClaimedShare claim)
    {
        ArgumentNullException.ThrowIfNull(claim);
        string id = claim.Payload.Id;
        if (!ShareIds.IsValid(id))
            return false;
        string expected = Path.Combine(Root, id + ClaimedSuffix);
        if (!string.Equals(Path.GetFullPath(claim.Directory), Path.GetFullPath(expected), StringComparison.Ordinal)
            || !Directory.Exists(expected))
            return false;
        string ready = Path.Combine(Root, id);
        if (Directory.Exists(ready))
            return false;
        try
        {
            Directory.Move(expected, ready);
            return true;
        }
        catch (Exception ex) when (ex is IOException or DirectoryNotFoundException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// Atomically mark an applied claim as acknowledged before deleting it. If deletion
    /// is interrupted, the tombstone is never recovered as a new share and a later
    /// purge can remove it safely.
    /// </summary>
    public bool Acknowledge(string id, string claimedDirectory)
    {
        if (!ShareIds.IsValid(id) || string.IsNullOrWhiteSpace(claimedDirectory))
            return false;
        string expected = Path.Combine(Root, id + ClaimedSuffix);
        string acknowledged = Path.Combine(Root, id + AcknowledgedSuffix);
        try
        {
            if (!string.Equals(Path.GetFullPath(claimedDirectory), Path.GetFullPath(expected), StringComparison.Ordinal))
                return false;
            if (!Directory.Exists(acknowledged))
            {
                if (!Directory.Exists(expected))
                    return false;
                Directory.Move(expected, acknowledged);
            }

            try { Directory.Delete(acknowledged, recursive: true); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // The atomic rename is the acknowledgement. A failed cleanup leaves a
                // tombstone, not something RecoverClaims can redeliver.
            }
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return false;
        }
    }

    /// <summary>Delete a claimed envelope and everything in it. Safe to call twice.</summary>
    public void Discard(string claimedDirectory)
    {
        if (string.IsNullOrEmpty(claimedDirectory))
            return;
        try
        {
            if (Directory.Exists(claimedDirectory))
                Directory.Delete(claimedDirectory, recursive: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // A share that will not delete is worth nothing and costs a few kilobytes;
            // the purge will try again. Never worth failing an import over.
        }
    }

    /// <summary>
    /// Remove what nobody is coming back for: ready envelopes and interrupted
    /// <c>.partial</c> writes older than <see cref="MaxAge"/>. Claimed envelopes are
    /// never purged here because one can be backing a draft the user is still editing;
    /// <see cref="RecoverClaims"/> handles process-crash claims at app startup.
    /// </summary>
    /// <returns>How many directories were removed.</returns>
    public int Purge(DateTimeOffset now) => Purge(now, MaxAge);

    /// <summary>
    /// The same, with an explicit age, for making room rather than tidying up.
    ///
    /// <para>
    /// An overload rather than a shifted clock, because a shifted clock is the wrong way
    /// round and reads as though it is right: <c>Purge(now.AddDays(-1))</c> compares
    /// against <c>now - 1 day - MaxAge</c>, which deletes strictly LESS than an ordinary
    /// purge, so the one call that existed to free space could never free any.
    /// </para>
    /// </summary>
    public int Purge(DateTimeOffset now, TimeSpan maxAge)
    {
        if (!Directory.Exists(Root))
            return 0;

        int removed = 0;
        foreach (string directory in Directory.EnumerateDirectories(Root))
        {
            string name = Path.GetFileName(directory);
            if (name.EndsWith(ClaimedSuffix, StringComparison.Ordinal))
                continue;
            bool leftover = name.EndsWith(PartialSuffix, StringComparison.Ordinal);
            bool stranger = !leftover && !ShareIds.IsValid(name);

            DateTime written;
            try { written = Directory.GetLastWriteTimeUtc(directory); }
            catch (IOException) { continue; }

            TimeSpan ageLimit = leftover ? PartialMaxAge : maxAge;
            bool old = now - new DateTimeOffset(written, TimeSpan.Zero) > ageLimit;
            // A partial is given the same grace as an envelope rather than deleted on
            // sight: the extension writing one RIGHT NOW has a .partial directory, and
            // an app that launches at that moment must not delete it out from under it.
            if (!old && !stranger)
                continue;

            try
            {
                Directory.Delete(directory, recursive: true);
                removed++;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
            }
        }
        return removed;
    }

    /// <summary>
    /// Total bytes the drop box is HOLDING, for the size guard.
    ///
    /// <para>
    /// Completed/claimed envelopes and stale partial directories count. A fresh partial
    /// is a writer with a short lease: excluding concurrent writers lets the commit lock
    /// choose one winner, while a killed writer starts counting after
    /// <see cref="PartialMaxAge"/> instead of bypassing the cap for the two-day inbox age.
    /// </para>
    /// </summary>
    public long TotalBytes() => TotalBytesExcept(null);

    private long TotalBytesExcept(string? excludedDirectory)
    {
        if (!Directory.Exists(Root))
            return 0;
        string? excluded = excludedDirectory is null ? null : Path.GetFullPath(excludedDirectory);
        long total = 0;
        foreach (string directory in Directory.EnumerateDirectories(Root))
        {
            if (excluded is not null
                && string.Equals(Path.GetFullPath(directory), excluded, StringComparison.Ordinal))
                continue;
            if (Path.GetFileName(directory).EndsWith(PartialSuffix, StringComparison.Ordinal))
            {
                DateTime written;
                try { written = Directory.GetLastWriteTimeUtc(directory); }
                catch (IOException) { continue; }
                if (DateTimeOffset.UtcNow - new DateTimeOffset(written, TimeSpan.Zero) <= PartialMaxAge)
                    continue;
            }
            foreach (string file in Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories))
            {
                try { total += new FileInfo(file).Length; }
                catch (IOException) { }
            }
        }
        return total;
    }

    /// <summary>Whether one more share of <paramref name="bytes"/> would fit.</summary>
    public bool HasRoomFor(long bytes) => TotalBytes() + Math.Max(0, bytes) <= MaxTotalBytes;

    internal bool HasCommitRoomFor(long bytes, string ownPartialDirectory, DateTimeOffset now)
    {
        if (!Directory.Exists(Root))
            return Math.Max(0, bytes) <= MaxTotalBytes;
        string own = Path.GetFullPath(ownPartialDirectory);
        long held = 0;
        foreach (string directory in Directory.EnumerateDirectories(Root))
        {
            string full = Path.GetFullPath(directory);
            if (string.Equals(full, own, StringComparison.Ordinal))
                continue;
            string name = Path.GetFileName(directory);
            if (name.EndsWith(PartialSuffix, StringComparison.Ordinal))
            {
                DateTime written;
                try { written = Directory.GetLastWriteTimeUtc(directory); }
                catch (IOException) { continue; }
                // Fresh partials are concurrent writers. Excluding both lets the commit
                // lock choose one winner; the second then observes the winner's final
                // directory. Stale partials are abandoned storage and do count.
                if (now - new DateTimeOffset(written, TimeSpan.Zero) <= PartialMaxAge)
                    continue;
            }
            foreach (string file in Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories))
            {
                try { held = checked(held + new FileInfo(file).Length); }
                catch (IOException) { }
            }
        }
        return held + Math.Max(0, bytes) <= MaxTotalBytes;
    }

    internal static string PayloadPathIn(string directory) => Path.Combine(directory, PayloadFileName);

    internal static string FilesDirectoryIn(string directory) => Path.Combine(directory, FilesDirectoryName);

    /// <summary>
    /// Where an envelope's file item actually is on disk, or null when the item names
    /// something outside the envelope.
    ///
    /// <para>
    /// The containment check is not paranoia about a hostile extension; it is the same
    /// rule <c>WebUiRoutes.ServeUpload</c> applies to an upload name, and for the same
    /// reason. This path comes out of a JSON file in a directory another process wrote,
    /// and the directory above it holds every conversation the user has ever had.
    /// </para>
    /// </summary>
    public static string? ResolveFile(string envelopeDirectory, ShareItem item)
    {
        if (item is null || !item.IsFile || string.IsNullOrWhiteSpace(item.File))
            return null;
        // A rooted or drive-qualified path would make Path.Combine discard the envelope
        // directory entirely, so it is refused before the combine rather than after.
        if (Path.IsPathRooted(item.File))
            return null;

        string root;
        string resolved;
        try
        {
            root = Path.GetFullPath(envelopeDirectory);
            resolved = Path.GetFullPath(Path.Combine(root, item.File));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            return null;
        }
        string prefix = root.EndsWith(Path.DirectorySeparatorChar) ? root : root + Path.DirectorySeparatorChar;
        if (!resolved.StartsWith(prefix, StringComparison.Ordinal))
            return null;
        if (!File.Exists(resolved))
            return null;

        // And again after following links. GetFullPath resolves "..", not symlinks,
        // while every reader afterwards — File.OpenRead, the upload service — follows
        // them: a files/photo.png that is a link to somewhere else passes a purely
        // textual containment check and is then read and put into the chat under a name
        // the envelope chose. The writer only ever creates regular files, so a link here
        // is by definition not something this app put there.
        try
        {
            if (new FileInfo(resolved).ResolveLinkTarget(returnFinalTarget: true) is { } target
                && !Path.GetFullPath(target.FullName).StartsWith(prefix, StringComparison.Ordinal))
            {
                return null;
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
        }
        return resolved;
    }
}

/// <summary>One envelope, taken out of the drop box and nobody else's.</summary>
/// <param name="Payload">What the extension wrote.</param>
/// <param name="Directory">Where its files are, until <see cref="ShareEnvelopeStore.Discard"/>.</param>
public sealed record ClaimedShare(SharePayload Payload, string Directory);

/// <summary>
/// Writes one envelope, and makes it visible only when it is whole.
///
/// <para>
/// Used from the share extension, where the memory limit is the constraint that
/// shapes everything: <see cref="ReserveFile"/> hands back a PATH so a shared video
/// is copied file-to-file by the system rather than read into a buffer this process
/// cannot afford. Nothing here ever holds file content.
/// </para>
/// </summary>
public sealed class ShareEnvelopeWriter
{
    // APFS, like the POSIX APIs underneath it, limits one path component to 255 UTF-8
    // bytes. Leave a little room for the collision suffix ReserveFile adds and for file
    // providers that normalize a name while copying it.
    private const int MaxFileNameUtf8Bytes = 240;
    private const int MaxPreservedExtensionUtf8Bytes = 32;

    private readonly ShareEnvelopeStore _store;
    private readonly HashSet<string> _taken = new(StringComparer.OrdinalIgnoreCase);
    private bool _done;

    internal ShareEnvelopeWriter(ShareEnvelopeStore store, string id, string partialDirectory)
    {
        _store = store;
        Id = id;
        PartialDirectory = partialDirectory;
    }

    /// <summary>The id this envelope will have.</summary>
    public string Id { get; }

    /// <summary>The directory being written into, before the rename.</summary>
    public string PartialDirectory { get; }

    /// <summary>
    /// Reserve a place for one file and return where to put it, plus the
    /// envelope-relative path to record in the item.
    ///
    /// <para>
    /// The name is sanitised here rather than trusted, because it comes from another
    /// app: iOS hands over whatever the source called the file, which has included a
    /// leading dot, a slash, and on one memorable occasion a newline. Collisions are
    /// resolved by numbering rather than overwriting, because two photos shared
    /// together are routinely both called IMG_0001.
    /// </para>
    /// </summary>
    public (string AbsolutePath, string RelativePath) ReserveFile(string suggestedName)
    {
        string safe = SafeFileName(suggestedName);
        string candidate = safe;
        for (int n = 2; !_taken.Add(candidate); n++)
        {
            candidate = FitFileName(safe, "-" + n.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }
        return (
            Path.Combine(ShareEnvelopeStore.FilesDirectoryIn(PartialDirectory), candidate),
            "files/" + candidate);
    }

    /// <summary>
    /// Write <c>share.json</c> and make the envelope visible, atomically.
    /// </summary>
    /// <returns>The id of the finished envelope.</returns>
    public string Commit(SharePayload payload)
    {
        ArgumentNullException.ThrowIfNull(payload);
        if (_done)
            throw new InvalidOperationException("This envelope has already been committed.");

        payload.Id = Id;
        payload.Version = SharePayload.CurrentVersion;
        if (payload.CreatedAt == default)
            payload.CreatedAt = DateTimeOffset.UtcNow;
        payload.ConstrainForWire();
        if (!payload.IsUsable(out string reason))
            throw new InvalidDataException("The share cannot be committed: " + reason + ".");

        byte[] json = JsonSerializer.SerializeToUtf8Bytes(payload, SharePayloadJsonContext.Default.SharePayload);
        if (json.LongLength > ShareEnvelopeStore.MaxPayloadBytes)
            throw new InvalidDataException($"The share metadata is too large ({json.LongLength:N0} bytes).");
        File.WriteAllBytes(ShareEnvelopeStore.PayloadPathIn(PartialDirectory), json);

        string final = Path.Combine(_store.Root, Id);

        // Commit capacity and the final rename under one cross-process file lock.
        // Measuring before a copy is useful for avoiding work, but two extensions can
        // finish at the same time; only this last check makes the storage cap a rule.
        using FileStream commitLock = AcquireCommitLock(_store.Root);
        long incoming = DirectoryBytes(PartialDirectory);
        if (!_store.HasCommitRoomFor(incoming, PartialDirectory, DateTimeOffset.UtcNow))
            throw new ShareInboxFullException(incoming, _store.MaxTotalBytes);
        if (Directory.Exists(final))
            throw new IOException($"A share envelope named {Id} already exists.");
        Directory.Move(PartialDirectory, final);
        _done = true;
        return Id;
    }

    private static FileStream AcquireCommitLock(string root)
    {
        string path = Path.Combine(root, ".commit.lock");
        IOException? last = null;
        for (int attempt = 0; attempt < 50; attempt++)
        {
            try
            {
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException ex)
            {
                last = ex;
                Thread.Sleep(10);
            }
        }
        throw new IOException("The share inbox is busy.", last);
    }

    private static long DirectoryBytes(string directory)
    {
        long total = 0;
        foreach (string file in Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories))
            total = checked(total + new FileInfo(file).Length);
        return total;
    }

    /// <summary>Throw the half-written envelope away. Safe after a commit, where it does nothing.</summary>
    public void Abandon()
    {
        if (_done)
            return;
        try
        {
            if (Directory.Exists(PartialDirectory))
                Directory.Delete(PartialDirectory, recursive: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
        }
    }

    /// <summary>
    /// A file name that is only a file name: no separators, no leading dot, no control
    /// characters, and never empty.
    ///
    /// <para>
    /// Public because both sides need the SAME answer. The extension names the file on
    /// disk with it; the app shows that name to the user and puts it in the message it
    /// gives the model. Two spellings of "the shared file's name" would be two names for
    /// one file in a chat where the user can see both.
    /// </para>
    /// </summary>
    public static string SafeFileName(string? suggested)
    {
        string name = suggested ?? string.Empty;
        // Take the last component only; a name arriving as "../../settings.json" must
        // become "settings.json" and not keep any of its journey.
        int cut = name.LastIndexOfAny(new[] { '/', '\\' });
        if (cut >= 0)
            name = name[(cut + 1)..];

        var cleaned = new StringBuilder(name.Length);
        foreach (Rune rune in name.EnumerateRunes())
        {
            int value = rune.Value;
            bool bad = Rune.IsControl(rune) || value is ':' or '"' or '*' or '?' or '<' or '>' or '|';
            cleaned.Append(bad ? "_" : rune.ToString());
        }
        name = cleaned.ToString().Normalize(NormalizationForm.FormC).Trim().Trim('.');
        if (name.Length == 0)
            name = "shared";
        return FitFileName(name, string.Empty);
    }

    private static string FitFileName(string name, string suffix)
    {
        string extension = Path.GetExtension(name);
        if (Encoding.UTF8.GetByteCount(extension) > MaxPreservedExtensionUtf8Bytes)
            extension = string.Empty;
        string stem = extension.Length == 0 ? name : name[..^extension.Length];
        int fixedBytes = Encoding.UTF8.GetByteCount(suffix) + Encoding.UTF8.GetByteCount(extension);
        int stemBudget = Math.Max(0, MaxFileNameUtf8Bytes - fixedBytes);
        string fittedStem = Utf8Prefix(stem, stemBudget).TrimEnd(' ', '.');
        if (fittedStem.Length == 0)
            fittedStem = Utf8Prefix("shared", stemBudget);
        return fittedStem + suffix + extension;
    }

    private static string Utf8Prefix(string value, int maximumBytes)
    {
        if (maximumBytes <= 0 || value.Length == 0)
            return string.Empty;
        if (Encoding.UTF8.GetByteCount(value) <= maximumBytes)
            return value;

        var result = new StringBuilder(Math.Min(value.Length, maximumBytes));
        int used = 0;
        foreach (Rune rune in value.EnumerateRunes())
        {
            if (used + rune.Utf8SequenceLength > maximumBytes)
                break;
            result.Append(rune.ToString());
            used += rune.Utf8SequenceLength;
        }
        return result.ToString();
    }
}

/// <summary>The durable share inbox has reached its configured storage limit.</summary>
public sealed class ShareInboxFullException : IOException
{
    public ShareInboxFullException(long incomingBytes, long maximumBytes)
        : base($"The share needs {incomingBytes:N0} bytes and the inbox limit is {maximumBytes:N0} bytes.")
    {
        IncomingBytes = incomingBytes;
        MaximumBytes = maximumBytes;
    }

    public long IncomingBytes { get; }
    public long MaximumBytes { get; }
}
