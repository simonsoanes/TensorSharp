// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json.Serialization;

namespace TensorAgent.Core.Sessions;

/// <summary>
/// One message as the Web UI holds it in its <c>chatHistory</c> array and sends it in
/// every <c>/api/chat</c> body. The property names ARE the Web UI's, so a saved
/// conversation can be handed straight back to the page (and the page's own array
/// saved straight to disk) without a mapping layer.
/// </summary>
public sealed class StoredMessage
{
    [JsonPropertyName("role")] public string Role { get; set; } = "user";
    [JsonPropertyName("content")] public string Content { get; set; } = string.Empty;
    [JsonPropertyName("thinking")] public string? Thinking { get; set; }
    [JsonPropertyName("imagePaths")] public List<string>? ImagePaths { get; set; }
    [JsonPropertyName("stillImagePaths")] public List<string>? StillImagePaths { get; set; }
    [JsonPropertyName("maskPath")] public string? MaskPath { get; set; }
    [JsonPropertyName("maskMode")] public string? MaskMode { get; set; }
    [JsonPropertyName("maskInvert")] public bool? MaskInvert { get; set; }
    [JsonPropertyName("maskFeather")] public int? MaskFeather { get; set; }
    [JsonPropertyName("maskCrop")] public bool? MaskCrop { get; set; }
    [JsonPropertyName("maskCropPadding")] public int? MaskCropPadding { get; set; }
    [JsonPropertyName("videoFilePaths")] public List<string>? VideoFilePaths { get; set; }
    [JsonPropertyName("audioPaths")] public List<string>? AudioPaths { get; set; }
    [JsonPropertyName("textFilePaths")] public List<string>? TextFilePaths { get; set; }
    [JsonPropertyName("textFileNames")] public List<string>? TextFileNames { get; set; }
    [JsonPropertyName("isVideo")] public bool? IsVideo { get; set; }
    /// <summary>Attachment chips the page showed for this message (display names and
    /// preview URLs), so a resumed conversation renders the same bubbles.</summary>
    [JsonPropertyName("attachments")] public List<StoredAttachment>? Attachments { get; set; }
    /// <summary>Files a tool produced during this assistant turn (download chips).</summary>
    [JsonPropertyName("artifacts")] public List<StoredArtifact>? Artifacts { get; set; }
    /// <summary>Generated image URL for an image-edit turn.</summary>
    [JsonPropertyName("imageUrl")] public string? ImageUrl { get; set; }
    /// <summary>Generated clip URL for a video turn.</summary>
    [JsonPropertyName("videoUrl")] public string? VideoUrl { get; set; }
    /// <summary>The clip's soundtrack, kept only when it is a separate file the page plays
    /// beside the clip (when the sound is inside the MP4 there is nothing to keep).</summary>
    [JsonPropertyName("audioUrl")] public string? AudioUrl { get; set; }

    /// <summary>Every upload file name (bare names under the uploads root) this message references.</summary>
    [JsonIgnore]
    public IEnumerable<string> ReferencedUploads
    {
        get
        {
            foreach (var list in new[] { ImagePaths, StillImagePaths, VideoFilePaths, AudioPaths, TextFilePaths })
                if (list is not null)
                    foreach (string p in list)
                        yield return Path.GetFileName(p);
            if (!string.IsNullOrEmpty(MaskPath)) yield return Path.GetFileName(MaskPath);
            if (Attachments is not null)
                foreach (StoredAttachment a in Attachments)
                {
                    if (!string.IsNullOrEmpty(a.File)) yield return Path.GetFileName(a.File);
                    if (!string.IsNullOrEmpty(a.PreviewFile)) yield return Path.GetFileName(a.PreviewFile);
                    if (!string.IsNullOrEmpty(a.EditFile)) yield return Path.GetFileName(a.EditFile);
                    if (!string.IsNullOrEmpty(a.MaskPath)) yield return Path.GetFileName(a.MaskPath);
                    if (a.Frames is not null) foreach (string f in a.Frames) yield return Path.GetFileName(f);
                }
            if (!string.IsNullOrEmpty(ImageUrl)) yield return Path.GetFileName(ImageUrl);
            if (!string.IsNullOrEmpty(VideoUrl)) yield return Path.GetFileName(VideoUrl);
            if (!string.IsNullOrEmpty(AudioUrl)) yield return Path.GetFileName(AudioUrl);
        }
    }
}

/// <summary>What the Web UI's attachment chip knows about an upload.</summary>
public sealed class StoredAttachment
{
    [JsonPropertyName("file")] public string File { get; set; } = string.Empty;
    [JsonPropertyName("fileName")] public string FileName { get; set; } = string.Empty;
    [JsonPropertyName("mediaType")] public string MediaType { get; set; } = "text";
    /// <summary>True when the complete text stays in the upload and is read through
    /// file/code tools instead of being copied into every prompt.</summary>
    [JsonPropertyName("fileBacked")] public bool? FileBacked { get; set; }
    [JsonPropertyName("previewFile")] public string? PreviewFile { get; set; }
    /// <summary>Browser-decodable original-size image for editing HEIC/HEIF uploads.</summary>
    [JsonPropertyName("editFile")] public string? EditFile { get; set; }
    [JsonPropertyName("editUnavailableReason")] public string? EditUnavailableReason { get; set; }
    [JsonPropertyName("maskPath")] public string? MaskPath { get; set; }
    [JsonPropertyName("maskMode")] public string? MaskMode { get; set; }
    [JsonPropertyName("maskInvert")] public bool? MaskInvert { get; set; }
    [JsonPropertyName("maskFeather")] public int? MaskFeather { get; set; }
    [JsonPropertyName("maskCrop")] public bool? MaskCrop { get; set; }
    [JsonPropertyName("maskCropPadding")] public int? MaskCropPadding { get; set; }
    [JsonPropertyName("frames")] public List<string>? Frames { get; set; }
    [JsonPropertyName("pageCount")] public int? PageCount { get; set; }
    [JsonPropertyName("extractedPageCount")] public int? ExtractedPageCount { get; set; }
    [JsonPropertyName("renderedAsImages")] public bool? RenderedAsImages { get; set; }
}

/// <summary>A produced file kept in the artifact store.</summary>
public sealed class StoredArtifact
{
    [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
    [JsonPropertyName("bytes")] public long Bytes { get; set; }
    [JsonPropertyName("url")] public string Url { get; set; } = string.Empty;
}

/// <summary>A saved chat session: the messages plus what the page needs to resume it.</summary>
public sealed class Conversation
{
    [JsonPropertyName("id")] public string Id { get; set; } = Guid.NewGuid().ToString("N");
    [JsonPropertyName("title")] public string Title { get; set; } = string.Empty;
    [JsonPropertyName("createdAt")] public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
    [JsonPropertyName("updatedAt")] public DateTimeOffset UpdatedAt { get; set; } = DateTimeOffset.UtcNow;
    /// <summary>Catalog id of the model the conversation was held with.</summary>
    [JsonPropertyName("modelId")] public string? ModelId { get; set; }
    [JsonPropertyName("think")] public bool Think { get; set; }
    [JsonPropertyName("skills")] public List<string> Skills { get; set; } = new();
    /// <summary>
    /// Whether <see cref="Skills"/> is a deliberate per-chat selection. An empty
    /// list without this bit means "let skill discovery decide"; an empty list with
    /// it means the user explicitly deselected every skill.
    /// </summary>
    [JsonPropertyName("skillsExplicit")] public bool SkillsExplicit { get; set; }
    [JsonPropertyName("messages")] public List<StoredMessage> Messages { get; set; } = new();
    /// <summary>Whether the engine session was reset after the last saved turn (a resumed
    /// conversation always starts with newChat=true, this records the page's own flag).</summary>
    [JsonPropertyName("needsCacheReset")] public bool NeedsCacheReset { get; set; }

    [JsonIgnore] public bool IsEmpty => Messages.Count == 0;

    /// <summary>The title the list shows: the first user line, trimmed, or a date.</summary>
    public static string DeriveTitle(IEnumerable<StoredMessage> messages, DateTimeOffset createdAt)
    {
        foreach (StoredMessage m in messages)
        {
            if (m.Role != "user")
                continue;
            string text = StripFileEnvelopes(m.Content).Trim();
            if (text.Length == 0)
            {
                if (m.Attachments is { Count: > 0 })
                    return m.Attachments[0].FileName;
                continue;
            }
            string firstLine = text.Split('\n', 2)[0].Trim();
            return firstLine.Length <= 60 ? firstLine : firstLine[..57].TrimEnd() + "…";
        }
        return "Chat " + createdAt.ToLocalTime().ToString("MMM d, HH:mm");
    }

    /// <summary>Removes the "[File: name] … [End of file]" blocks the page prepends for
    /// inlined text uploads, so titles show what the user typed.</summary>
    public static string StripFileEnvelopes(string content)
    {
        if (string.IsNullOrEmpty(content) || !content.StartsWith("[File: ", StringComparison.Ordinal))
            return content;
        const string end = "[End of file]";
        int last = content.LastIndexOf(end, StringComparison.Ordinal);
        return last < 0 ? content : content[(last + end.Length)..].TrimStart();
    }
}

/// <summary>Row of the sessions list; cheap to build without loading messages.</summary>
public sealed record ConversationSummary(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("title")] string Title,
    [property: JsonPropertyName("createdAt")] DateTimeOffset CreatedAt,
    [property: JsonPropertyName("updatedAt")] DateTimeOffset UpdatedAt,
    [property: JsonPropertyName("modelId")] string? ModelId,
    [property: JsonPropertyName("messageCount")] int MessageCount);
