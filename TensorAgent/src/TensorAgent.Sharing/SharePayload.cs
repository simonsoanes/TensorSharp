// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using System.Text.Json.Serialization;
using TensorAgent.Sharing.Localization;

namespace TensorAgent.Sharing;

/// <summary>
/// What one "Ask TensorAgent" share carries, as it is written by the extension and
/// read by the app.
///
/// <para>
/// This type is a WIRE FORMAT between two processes that are built together but do
/// not run together, and it is versioned for a reason that is easy to miss: an
/// envelope written by the extension survives on disk until the app claims it, and
/// the user can update the app while one is sitting there. Every field therefore has
/// a safe default, unknown fields are ignored, and <see cref="Version"/> is checked
/// before anything else is trusted.
/// </para>
/// <para>
/// The kinds are deliberately few. Images, videos, audio, PDFs and documents are all
/// <see cref="ShareItemKinds.File"/>: the app already classifies an upload by its
/// file name alone (<c>UploadNaming</c> and <c>WebUiChatService.UploadAsync</c>), and
/// a second classification here could only ever disagree with the first.
/// </para>
/// </summary>
public sealed class SharePayload
{
    /// <summary>The format this envelope was written in. See <see cref="CurrentVersion"/>.</summary>
    [JsonPropertyName("version")] public int Version { get; set; } = CurrentVersion;

    /// <summary>Identifies the envelope directory; 32 lowercase hex characters.</summary>
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;

    /// <summary>When the extension finished writing it, for the age-based purge.</summary>
    [JsonPropertyName("createdAt")] public DateTimeOffset CreatedAt { get; set; }

    /// <summary>
    /// A human-readable name for where the share came from ("Safari", "Mail"), or
    /// empty when the system did not say. It is shown to the user and given to the
    /// model as context; it is never used to decide anything.
    /// </summary>
    [JsonPropertyName("sourceApp")] public string SourceApp { get; set; } = string.Empty;

    /// <summary>What the user typed into the share sheet, or the preset they chose. May be empty.</summary>
    [JsonPropertyName("prompt")] public string Prompt { get; set; } = string.Empty;

    /// <summary>
    /// Compatibility field written by version-1 extensions. The containing app opens
    /// every independently committed envelope in its own fresh chat, regardless of an
    /// old envelope's value, so two share actions can never be merged together.
    /// </summary>
    [JsonPropertyName("newChat")] public bool NewChat { get; set; } = true;

    /// <summary>
    /// Reserved for compatibility with version-1 envelopes written by development
    /// builds. Current builds always leave a reviewable draft and ignore this value;
    /// a durable handoff cannot prove across an app/WebView crash that an earlier send
    /// was accepted exactly once.
    /// </summary>
    [JsonPropertyName("autoSend")] public bool AutoSend { get; set; }

    /// <summary>The shared content, in the order the source app offered it.</summary>
    [JsonPropertyName("items")] public List<ShareItem> Items { get; set; } = new();

    /// <summary>
    /// What the extension could not do, in sentences meant for the user.
    ///
    /// <para>
    /// A share extension runs under a memory limit far below an app's and gets no
    /// second chance: a file it refuses is simply not in the envelope. Saying so here
    /// is the difference between "TensorAgent dropped my video" and "that video was
    /// too large to share". The app shows these once, with the share.
    /// </para>
    /// </summary>
    [JsonPropertyName("notes")] public List<string> Notes { get; set; } = new();

    /// <summary>The version this build writes, and the highest it will read.</summary>
    public const int CurrentVersion = 1;

    /// <summary>Hard structural bounds for an untrusted/corrupt envelope.</summary>
    public const int MaxItems = 64;
    public const int MaxNotes = 128;
    public const int MaxSourceAppChars = 256;
    public const int MaxPromptChars = 16_000;
    public const int MaxTitleChars = 1_024;
    public const int MaxUrlChars = 8_192;
    public const int MaxTextualChars = 1_000_000;
    public const int MaxNoteChars = 2_000;
    public const int MaxFileFieldChars = 512;
    public const int MaxKindChars = 16;
    // System.Text.Json may escape one UTF-16 code unit as six ASCII bytes. Keeping the
    // sum below this value proves the serialized envelope remains under 4 MiB even for
    // CJK/control-heavy input, with ample room for JSON structure.
    public const int MaxAggregateChars = 400_000;

    /// <summary>
    /// The one serializer both sides use.
    ///
    /// <para>
    /// Not <c>JsonSerializerDefaults.Web</c>: every property here already names itself
    /// with <see cref="JsonPropertyNameAttribute"/>, and a naming policy on top of that
    /// is a second opinion about the same thing. Case-insensitive reading is on so that
    /// an envelope hand-written by a test or a script is not rejected over a capital
    /// letter.
    /// </para>
    /// </summary>
    // Trim safety, and the reason SharePayloadJsonContext exists at all: the share
    // extension is a trimmed assembly and reflection-based serialization there fails
    // at run time rather than at build time. The options are generated with the
    // context, making this property a single contract rather than a second copy.
    public static JsonSerializerOptions JsonOptions => SharePayloadJsonContext.Default.Options;

    /// <summary>Everything that is not a file: the text, links and pages that become the message.</summary>
    public IEnumerable<ShareItem> TextualItems => Items.Where(i => !i.IsFile);

    /// <summary>The file items, which become ordinary chat attachments.</summary>
    public IEnumerable<ShareItem> FileItems => Items.Where(i => i.IsFile);

    /// <summary>
    /// Repair nullable fields that JSON is allowed to contain despite the C# nullable
    /// annotations. Kept public because the composer is also a public entry point and
    /// must be safe when called without first going through the envelope reader.
    /// </summary>
    public void NormalizeNulls()
    {
        SourceApp ??= string.Empty;
        Prompt ??= string.Empty;
        Items ??= new List<ShareItem>();
        Notes ??= new List<string>();
        for (int i = 0; i < Notes.Count; i++)
            Notes[i] ??= string.Empty;
        for (int i = 0; i < Items.Count; i++)
        {
            ShareItem? item = Items[i];
            if (item is null)
                continue;
            item.Kind ??= ShareItemKinds.Text;
            item.Text ??= string.Empty;
            item.Title ??= string.Empty;
            item.Url ??= string.Empty;
            item.Selection ??= string.Empty;
            item.File ??= string.Empty;
            item.FileName ??= string.Empty;
            item.TypeIdentifier ??= string.Empty;
            item.MimeType ??= string.Empty;
        }
    }

    /// <summary>
    /// Bound extension-authored metadata before it crosses the App Group boundary.
    /// Shared file bytes live beside this JSON and are unaffected; this protects the
    /// extension and app from a source handing over megabytes of title/prompt metadata.
    /// </summary>
    /// <returns>Whether any user-visible string was shortened.</returns>
    public bool ConstrainForWire()
    {
        NormalizeNulls();
        bool shortened = false;
        const int noteReserve = 32_768;
        long immutableChars = Items.Where(static item => item is not null)
            .Sum(static item => (long)item!.Kind.Length + item.File.Length);
        int remainingMetadata = (int)Math.Max(
            0, MaxAggregateChars - noteReserve - immutableChars);
        SourceApp = TakeBounded(SourceApp, MaxSourceAppChars, ref remainingMetadata, ref shortened);
        Prompt = TakeBounded(Prompt, MaxPromptChars, ref remainingMetadata, ref shortened);

        int remainingText = MaxTextualChars;
        foreach (ShareItem? item in Items)
        {
            if (item is null)
                continue;
            item.Title = TakeBounded(item.Title, MaxTitleChars, ref remainingMetadata, ref shortened);
            item.Url = TakeBounded(item.Url, MaxUrlChars, ref remainingMetadata, ref shortened);
            item.Text = TakeText(item.Text, ref remainingText, ref remainingMetadata, ref shortened);
            item.Selection = TakeText(item.Selection, ref remainingText, ref remainingMetadata, ref shortened);
            // File is an envelope-relative identity and truncating it would point at a
            // different path. Writer-produced values are tiny; malformed callers are
            // rejected by IsUsable rather than silently rewritten.
            item.FileName = TakeBounded(item.FileName, MaxFileFieldChars, ref remainingMetadata, ref shortened);
            item.TypeIdentifier = TakeBounded(item.TypeIdentifier, MaxFileFieldChars, ref remainingMetadata, ref shortened);
            item.MimeType = TakeBounded(item.MimeType, 128, ref remainingMetadata, ref shortened);
        }

        remainingMetadata += noteReserve;
        for (int i = 0; i < Notes.Count; i++)
            Notes[i] = TakeBounded(Notes[i], MaxNoteChars, ref remainingMetadata, ref shortened);
        if (Notes.Count > MaxNotes)
        {
            Notes.RemoveRange(MaxNotes - 1, Notes.Count - (MaxNotes - 1));
            Notes.Add(ShareStrings.T("share.note.warningsOmitted"));
            shortened = true;
        }
        if (shortened && Notes.Count < MaxNotes)
        {
            string note = ShareStrings.T("share.note.metadataShortened");
            if (!Notes.Contains(note, StringComparer.Ordinal))
                Notes.Add(note);
        }
        return shortened;

        static string TakeText(
            string value, ref int remainingText, ref int remainingMetadata, ref bool wasShortened)
        {
            int allowance = Math.Max(0, Math.Min(remainingText, remainingMetadata));
            string limited = Limit(value, allowance, ref wasShortened);
            remainingText -= limited.Length;
            remainingMetadata -= limited.Length;
            return limited;
        }

        static string TakeBounded(
            string value, int fieldMaximum, ref int remaining, ref bool wasShortened)
        {
            int allowance = Math.Max(0, Math.Min(fieldMaximum, remaining));
            string limited = Limit(value, allowance, ref wasShortened);
            remaining -= limited.Length;
            return limited;
        }

        static string Limit(string value, int maximum, ref bool wasShortened)
        {
            if (value.Length <= maximum)
                return value;
            wasShortened = true;
            return ShareText.Shorten(value, maximum).Text;
        }
    }

    /// <summary>
    /// Whether this envelope is one this build understands and can act on.
    ///
    /// <para>
    /// A newer version is refused rather than read leniently. The alternative — read
    /// what is recognised and drop the rest — turns "this app is too old for that
    /// share" into a share that silently arrives with half its content, which is the
    /// worse of the two failures by a distance.
    /// </para>
    /// </summary>
    public bool IsUsable(out string reason)
    {
        NormalizeNulls();
        if (Version <= 0 || Version > CurrentVersion)
        {
            reason = $"the share was written in format {Version}, and this build reads up to {CurrentVersion}";
            return false;
        }
        if (!ShareIds.IsValid(Id))
        {
            reason = "the share has no valid id";
            return false;
        }
        if (Items.Count > MaxItems || Notes.Count > MaxNotes)
        {
            reason = "the share contains too many items";
            return false;
        }
        if (Items.Any(static item => item is null))
        {
            reason = "the share contains an empty item";
            return false;
        }
        if (Items.Any(static item => item!.Kind.Length > MaxKindChars
            || item.Kind is not (ShareItemKinds.Text or ShareItemKinds.Url or ShareItemKinds.Page or ShareItemKinds.File)))
        {
            reason = "the share contains an unsupported item kind";
            return false;
        }
        if (SourceApp.Length > MaxSourceAppChars || Prompt.Length > MaxPromptChars
            || Notes.Any(static note => note.Length > MaxNoteChars)
            || Items.Any(static item =>
                item!.Title.Length > MaxTitleChars
                || item.Url.Length > MaxUrlChars
                || item.File.Length > MaxFileFieldChars
                || item.FileName.Length > MaxFileFieldChars
                || item.TypeIdentifier.Length > MaxFileFieldChars
                || item.MimeType.Length > 128)
            || Items.Sum(static item => (long)item!.Text.Length + item.Selection.Length) > MaxTextualChars)
        {
            reason = "the share metadata exceeds its safety bounds";
            return false;
        }
        long aggregate = SourceApp.Length + Prompt.Length
            + Notes.Sum(static note => (long)note.Length)
            + Items.Sum(static item => (long)item!.Kind.Length + item.Text.Length + item.Title.Length
                + item.Url.Length + item.Selection.Length + item.File.Length + item.FileName.Length
                + item.TypeIdentifier.Length + item.MimeType.Length);
        if (aggregate > MaxAggregateChars)
        {
            reason = "the share metadata is too large";
            return false;
        }
        if (Items.Count == 0 && !Notes.Any(static note => !string.IsNullOrWhiteSpace(note)))
        {
            reason = "the share carried nothing";
            return false;
        }
        reason = string.Empty;
        return true;
    }
}

/// <summary>The four things a share can carry. Strings rather than an enum so the
/// on-disk envelope stays readable and an unknown kind is data rather than a throw.</summary>
public static class ShareItemKinds
{
    /// <summary>Plain text: a selection, a note, the body of a message.</summary>
    public const string Text = "text";

    /// <summary>A link, with no page content behind it.</summary>
    public const string Url = "url";

    /// <summary>A web page the browser extracted for us: title, URL, selection and readable text.</summary>
    public const string Page = "page";

    /// <summary>A file copied into the envelope: an image, a clip, a PDF, a document.</summary>
    public const string File = "file";
}

/// <summary>One piece of shared content.</summary>
public sealed class ShareItem
{
    /// <summary>One of <see cref="ShareItemKinds"/>.</summary>
    [JsonPropertyName("kind")] public string Kind { get; set; } = ShareItemKinds.Text;

    /// <summary>The text, for <c>text</c>; the page's readable body, for <c>page</c>. Empty otherwise.</summary>
    [JsonPropertyName("text")] public string Text { get; set; } = string.Empty;

    /// <summary>The page or link title, when the source app offered one.</summary>
    [JsonPropertyName("title")] public string Title { get; set; } = string.Empty;

    /// <summary>The address, for <c>url</c> and <c>page</c>.</summary>
    [JsonPropertyName("url")] public string Url { get; set; } = string.Empty;

    /// <summary>What the user had selected on the page, when they shared from a selection.</summary>
    [JsonPropertyName("selection")] public string Selection { get; set; } = string.Empty;

    /// <summary>Envelope-relative path of the copied file, for <c>file</c>. Always <c>files/&lt;name&gt;</c>.</summary>
    [JsonPropertyName("file")] public string File { get; set; } = string.Empty;

    /// <summary>The name the file had where it came from, which is what the user recognises.</summary>
    [JsonPropertyName("fileName")] public string FileName { get; set; } = string.Empty;

    /// <summary>The uniform type identifier the system reported, for the log. Never used to decide anything.</summary>
    [JsonPropertyName("typeIdentifier")] public string TypeIdentifier { get; set; } = string.Empty;

    /// <summary>
    /// The MIME type for <see cref="TypeIdentifier"/>, when the system could give one.
    ///
    /// <para>
    /// Recorded separately because the app's <c>UploadNaming</c> classifies by
    /// extension and falls back to a MIME type, not to a uniform type identifier — and
    /// a shared file arriving with no extension at all is routine: a Mail attachment,
    /// an image lifted out of Messages, a photo the picker names <c>IMG_0004</c>.
    /// </para>
    /// </summary>
    [JsonPropertyName("mimeType")] public string MimeType { get; set; } = string.Empty;

    /// <summary>Size of the copied file in bytes.</summary>
    [JsonPropertyName("bytes")] public long Bytes { get; set; }

    [JsonIgnore] public bool IsFile => string.Equals(Kind, ShareItemKinds.File, StringComparison.Ordinal);

    public static ShareItem ForText(string text, string title = "") =>
        new() { Kind = ShareItemKinds.Text, Text = text ?? string.Empty, Title = title ?? string.Empty };

    public static ShareItem ForUrl(string url, string title = "") =>
        new() { Kind = ShareItemKinds.Url, Url = url ?? string.Empty, Title = title ?? string.Empty };

    public static ShareItem ForPage(string url, string title, string text, string selection = "") => new()
    {
        Kind = ShareItemKinds.Page,
        Url = url ?? string.Empty,
        Title = title ?? string.Empty,
        Text = text ?? string.Empty,
        Selection = selection ?? string.Empty,
    };

    public static ShareItem ForFile(
        string relativePath, string fileName, long bytes, string typeIdentifier = "", string mimeType = "") => new()
    {
        Kind = ShareItemKinds.File,
        File = relativePath ?? string.Empty,
        FileName = fileName ?? string.Empty,
        Bytes = bytes,
        TypeIdentifier = typeIdentifier ?? string.Empty,
        MimeType = mimeType ?? string.Empty,
    };
}
