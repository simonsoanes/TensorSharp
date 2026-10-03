// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Localization;
using TensorAgent.Sharing;
using TensorSharp.Chat;

namespace TensorAgent.Core.Sharing;

public enum ShareImportStatus
{
    Offered,
    RetryLater,
    PermanentRefusal,
    Duplicate,
}

public sealed record ShareImportResult(ShareImportStatus Status, PendingShare? Share);

/// <summary>
/// Turns a share envelope into something the chat can hold: one message and a set of
/// ordinary attachments.
///
/// <para>
/// The whole design of this feature is in one sentence here: a shared page becomes an
/// ORDINARY chat message with ORDINARY attachments, and nothing downstream of this
/// class knows a share happened. Files go through <see cref="UploadNaming"/> and
/// <see cref="WebUiChatService.UploadAsync"/> — the same two calls the photo picker
/// uses — rather than through a second copy of the naming and classification rules,
/// and the service's answer is passed on untouched. That is what makes it impossible
/// for this feature to regress a conversation that has nothing to do with it: there is
/// no share-shaped path through the inference pipeline to get wrong.
/// </para>
/// </summary>
public sealed class ShareImporter
{
    private readonly WebUiChatService _chat;
    private readonly ShareIntake _intake;
    private readonly ILogger _log;

    /// <summary>
    /// The largest single file that will be imported.
    ///
    /// <para>
    /// Above this it is refused with a sentence rather than attached, because the
    /// alternative is a phone spending a minute copying a video into the chat's own
    /// storage and then failing at the model instead. The extension applies its own,
    /// lower limit first; this is the one that holds when a share arrives from anywhere
    /// else.
    /// </para>
    /// </summary>
    public long MaxFileBytes { get; init; } = 256L * 1024 * 1024;

    /// <summary>How much of a share the model is shown. See <see cref="ShareCompositionOptions"/>.</summary>
    public ShareCompositionOptions Composition { get; init; } = ShareCompositionOptions.Default;

    public ShareImporter(WebUiChatService chat, ShareIntake intake, ILoggerFactory? loggerFactory = null)
    {
        _chat = chat ?? throw new ArgumentNullException(nameof(chat));
        _intake = intake ?? throw new ArgumentNullException(nameof(intake));
        _log = (loggerFactory ?? NullLoggerFactory.Instance).CreateLogger("TensorAgent.Share");
    }

    /// <summary>
    /// Import one share and offer it to the page.
    /// </summary>
    /// <param name="payload">What the extension wrote.</param>
    /// <param name="envelopeDirectory">
    /// Where the payload's files are, or null for an in-memory/text-only caller that
    /// has none.
    /// </param>
    /// <returns>The pending share, or null when it was empty, unusable or a duplicate.</returns>
    public async Task<PendingShare?> ImportAsync(
        SharePayload payload, string? envelopeDirectory, CancellationToken cancellationToken = default)
        => (await ImportWithOutcomeAsync(payload, envelopeDirectory, cancellationToken).ConfigureAwait(false)).Share;

    /// <summary>Import a share while preserving the distinction between retry and refusal.</summary>
    public async Task<ShareImportResult> ImportWithOutcomeAsync(
        SharePayload payload, string? envelopeDirectory, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(payload);
        if (!payload.IsUsable(out string why))
        {
            _log.LogWarning("Share {Id} refused: {Reason}", payload.Id, why);
            return new ShareImportResult(ShareImportStatus.PermanentRefusal, null);
        }

        var importNotices = new List<string>();
        var attachments = new List<object>();
        var attachedFiles = new List<ShareItem>();
        var preparedFiles = new List<PreparedFile>();

        // Validate every envelope path before the first upload. That makes a transient
        // stat/access failure retryable without first creating successful uploads for
        // earlier items in the same share.
        foreach (ShareItem item in payload.FileItems)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string shown = ShareEnvelopeWriter.SafeFileName(item.FileName);
            string? source = envelopeDirectory is null
                ? null
                : ShareEnvelopeStore.ResolveFile(envelopeDirectory, item);
            if (source is null)
            {
                // Either the envelope did not survive, or the item named a path outside
                // it. Both are "that file did not arrive", and both are worth a sentence:
                // silently dropping an attachment is the failure this codebase has
                // already paid for once, with photos.
                importNotices.Add(Loc.T("host.share.unreadable", ("file", shown)));
                _log.LogWarning("Share {Id}: {File} did not resolve inside {Directory}",
                    payload.Id, item.File, envelopeDirectory ?? "(no envelope)");
                continue;
            }

            long length;
            try { length = new FileInfo(source).Length; }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _log.LogWarning(ex, "Share {Id}: could not stat {Path}; retaining it for retry", payload.Id, source);
                return new ShareImportResult(ShareImportStatus.RetryLater, null);
            }

            if (length == 0)
            {
                importNotices.Add(Loc.T("host.share.empty", ("file", shown)));
                continue;
            }
            if (length > MaxFileBytes)
            {
                importNotices.Add(Loc.T("host.share.tooLarge",
                    ("file", shown), ("size", (length / (1024.0 * 1024)).ToString("0.#", Loc.Culture))));
                continue;
            }

            string name = UploadNaming.ResolveFileName(shown, item.MimeType, source);
            preparedFiles.Add(new PreparedFile(item, shown, source, name, length));
        }

        int inlineTextBudget = Math.Max(1, Composition.MaxTotalChars);
        int inlineTextPerFile = preparedFiles.Count == 0
            ? inlineTextBudget
            : Math.Max(1, inlineTextBudget / preparedFiles.Count);
        for (int fileIndex = 0; fileIndex < preparedFiles.Count; fileIndex++)
        {
            PreparedFile file = preparedFiles[fileIndex];
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                // The name the service classifies by, resolved from what actually
                // arrived: a share can hand over a file with no extension at all (a
                // Mail attachment, an image dropped out of Messages), and the service
                // decides what an upload IS from the extension alone.
                // The SANITISED name, not the raw one. UploadNaming returns a name whose
                // extension already classifies completely untouched, WebUiChatService
                // echoes it back as fileName, and the page splices that into the
                // "[File: name] … [End of file]" envelope it builds for the model — so a
                // newline in a name chosen by another app would end that envelope early.
                await using FileStream content = File.OpenRead(file.Source);
                // A stable key bounds crash/restart staging: recovering the same durable
                // envelope replaces this file's earlier app-side copy and derived frames
                // instead of minting another GUID family on every launch. The text cap is
                // share-specific; ordinary uploads retain their existing full-text contract,
                // while a 128 MB .txt share must not become several hundred MB of UTF-16 and
                // JSON inside the phone's WebView.
                object result = await _chat.UploadAsync(
                        content, file.UploadName, file.Length, cancellationToken,
                        stableStorageKey: StableUploadKey(payload.Id, fileIndex),
                        maxInlineTextChars: inlineTextPerFile)
                    .ConfigureAwait(false);
                attachments.Add(result);
                attachedFiles.Add(file.Item);
                _log.LogInformation("Share {Id}: attached {Name} ({Bytes} bytes)",
                    payload.Id, file.UploadName, file.Length);
            }
            catch (WebUiRequestRejectedException rejected) when (IsTransient(rejected.StatusCode))
            {
                RollBack(attachments, payload.Id);
                _log.LogWarning("Share {Id}: {Name} temporarily refused ({Status}); retaining the envelope: {Reason}",
                    payload.Id, file.ShownName, rejected.StatusCode, ReasonOf(rejected));
                return new ShareImportResult(ShareImportStatus.RetryLater, null);
            }
            catch (WebUiRequestRejectedException rejected)
            {
                importNotices.Add(Loc.T("host.share.refused", ("file", file.ShownName), ("reason", ReasonOf(rejected))));
                _log.LogWarning("Share {Id}: {Name} rejected", payload.Id, file.ShownName);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                RollBack(attachments, payload.Id);
                _log.LogWarning(ex, "Share {Id}: {Name} hit transient I/O; retaining the envelope",
                    payload.Id, file.ShownName);
                return new ShareImportResult(ShareImportStatus.RetryLater, null);
            }
            catch (NotSupportedException ex)
            {
                importNotices.Add(Loc.T("host.share.refused", ("file", file.ShownName), ("reason", ex.Message)));
                _log.LogWarning(ex, "Share {Id}: {Name} is not supported", payload.Id, file.ShownName);
            }
            catch
            {
                RollBack(attachments, payload.Id);
                throw;
            }
        }

        // Compose only after upload classification, so a refused file is named in the
        // notice but not misleadingly described as an attachment in the draft.
        ShareComposition composed = ShareComposer.Compose(payload, Composition, attachedFiles);
        var notices = new List<string>(composed.Notices);
        notices.AddRange(importNotices);

        // Did anything the question is ABOUT actually arrive? Not "is the message
        // empty" — the message always carries a question, typed or default — and not the
        // envelope's file list either, because those are uploaded here and any of them
        // can be refused. Text that reached the message, or a file that reached the
        // upload directory.
        bool arrived = composed.HasSharedContent || attachments.Count > 0;

        if (!arrived && notices.Count == 0)
        {
            _log.LogWarning("Share {Id} produced nothing to send", payload.Id);
            return new ShareImportResult(ShareImportStatus.PermanentRefusal, null);
        }

        // A share where everything was refused is still an EVENT the user has to be told
        // about: they tapped Ask, the sheet closed, and the app came forward. Returning
        // null here — which is what this did — threw away the notes, which were the only
        // thing there was to say and the only explanation of why nothing happened. So a
        // share carrying nothing but notes is still offered, with no message and nothing
        // to send.
        var pending = new PendingShare(
            payload.Id,
            arrived ? composed.Message : string.Empty,
            attachments,
            notices,
            composed.SuggestedTitle,
            // One durable envelope is one user share action and therefore one chat.
            // Keeping this invariant here (instead of trusting a legacy wire flag)
            // prevents separately shared items from ever being folded into the same
            // conversation by an older extension or a hand-written envelope.
            true,
            // A durable envelope cannot prove that an earlier model turn was accepted
            // if the app/WebView died before ACK. Always leave the requested draft for
            // the user to inspect and send exactly once themselves.
            false);

        ShareOfferStatus offer = _intake.TryOffer(pending);
        if (offer != ShareOfferStatus.Offered)
        {
            // CanAccept is deliberately only a fast pre-check in the host. Another
            // concurrent offer can fill the queue between that check and this atomic
            // TryOffer, and retrying must not leave another GUID family behind each time.
            RollBack(attachments, payload.Id);
        }

        return offer switch
        {
            ShareOfferStatus.Offered => new ShareImportResult(ShareImportStatus.Offered, pending),
            ShareOfferStatus.Full => new ShareImportResult(ShareImportStatus.RetryLater, null),
            _ => new ShareImportResult(ShareImportStatus.Duplicate, null),
        };
    }

    private static string StableUploadKey(string shareId, int fileIndex)
    {
        byte[] bytes = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(shareId + ":" + fileIndex.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        return Convert.ToHexString(bytes.AsSpan(0, 16)).ToLowerInvariant();
    }

    /// <summary>The sentence inside a refusal payload, or the exception's own message.</summary>
    private static string ReasonOf(WebUiRequestRejectedException rejected)
    {
        try
        {
            System.Text.Json.JsonElement payload = System.Text.Json.JsonSerializer
                .SerializeToElement(rejected.Payload, SseFraming.JsonOptions);
            if (payload.ValueKind == System.Text.Json.JsonValueKind.Object
                && payload.TryGetProperty("error", out System.Text.Json.JsonElement error)
                && error.GetString() is { Length: > 0 } sentence)
            {
                return sentence;
            }
        }
        catch (Exception ex) when (ex is System.Text.Json.JsonException or NotSupportedException)
        {
        }
        return rejected.Message;
    }

    private static bool IsTransient(int statusCode) =>
        statusCode == 408 || statusCode == 429 || statusCode == 507 || statusCode >= 500;

    private void RollBack(List<object> uploads, string shareId)
    {
        foreach (object upload in uploads)
        {
            if (!_chat.DiscardUpload(upload))
                _log.LogWarning("Share {Id}: a staged upload could not be rolled back cleanly", shareId);
        }
        uploads.Clear();
    }

    private sealed record PreparedFile(
        ShareItem Item,
        string ShownName,
        string Source,
        string UploadName,
        long Length);
}
