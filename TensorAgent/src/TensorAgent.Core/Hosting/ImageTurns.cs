// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorSharp.Chat;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// A chat turn when the loaded model makes pictures instead of text.
///
/// <para>
/// The page has one route for a turn, <c>/api/chat</c>, and on this host that route runs
/// under the turn manager and the conversation recorder: the turn survives the page being
/// hidden, and what it produced is written down when it ends. An image model answers the
/// same route rather than a second one the page would have to drive on its own. The newest
/// user message is the request: with a photo attached it is an edit of that photo,
/// otherwise a picture made from the words.
/// </para>
/// <para>
/// The image service's frames are translated into the chat stream's vocabulary:
/// <c>image_step</c> while the picture denoises (some carry a small <c>preview</c>), one
/// <c>imageUrl</c> for the finished picture, and the usual <c>done</c> with the session,
/// which is what the recorder keys the transcript on.
/// </para>
/// </summary>
public static class ImageTurns
{
    /// <summary>
    /// The output area a turn asks for: 1024 x 1024 for a picture made from words, and the
    /// same area at the photo's own shape for an edit. The model's native area is 2048 x
    /// 2048, which has four times the image tokens and costs several times as long per
    /// step; the page offers no size setting, so this is the size every turn gets.
    /// </summary>
    public const long DefaultTargetArea = 1024L * 1024;

    /// <summary>
    /// Where a chat turn's frames come from: the image service when the loaded model
    /// makes pictures, the video service (<see cref="VideoTurns"/>) when it makes clips,
    /// the chat service otherwise.
    ///
    /// <para>
    /// One decision for every caller that answers <c>/api/chat</c> — the route's own
    /// default and the app host's GPU gate, which supplies its frames in place of that
    /// default. The first version of this decided inside the route, the app passed its
    /// gate, and a picture request in the app went to the text pipeline and failed on its
    /// first frame; every unit test passed, because none of them ran the app's wiring.
    /// </para>
    /// </summary>
    public static IAsyncEnumerable<object> FramesFor(
        WebUiChatService chat, JsonElement body, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(chat);
        return chat.LoadedModelMakesImages ? StreamAsync(chat, body, cancellationToken)
            : chat.LoadedModelMakesVideo ? VideoTurns.StreamAsync(chat, body, cancellationToken)
            : chat.ChatStreamAsync(body, cancellationToken);
    }

    /// <summary>
    /// Run the turn <paramref name="body"/> describes and yield its chat frames.
    /// </summary>
    /// <param name="chat">The chat service, with an image model loaded.</param>
    /// <param name="body">The <c>/api/chat</c> request the page sent.</param>
    /// <param name="cancellationToken">Ends the turn; the picture is abandoned.</param>
    public static async IAsyncEnumerable<object> StreamAsync(
        WebUiChatService chat, JsonElement body, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(chat);

        string? sessionId = body.ValueKind == JsonValueKind.Object
            && body.TryGetProperty("sessionId", out JsonElement id) && id.ValueKind == JsonValueKind.String
                ? id.GetString()
                : null;
        // Held for the whole turn, as a text turn holds it: a shared photo being sent here
        // must not be discarded underneath the edit. A refusal is thrown before the first
        // frame, which is what turns it into a status code rather than a stream.
        using IDisposable? lease = chat.AcquireChatRequestLease?.Invoke(body);
        ImageRequest? request = Read(body);
        if (request is null)
        {
            yield return new
            {
                done = true,
                error = "Describe the picture you want, or attach a photo and say how to change it.",
                sessionId,
            };
            yield break;
        }

        // Recorded exactly as a text turn is: the host's handler writes the user's
        // message into the conversation and settles a shared item the turn consumed.
        if (!string.IsNullOrEmpty(sessionId))
            chat.OnChatRequest?.Invoke(sessionId, body);

        using JsonDocument payload = JsonDocument.Parse(JsonSerializer.Serialize(request.Payload));
        JsonElement service = payload.RootElement.Clone();
        IAsyncEnumerable<object> frames = request.Editing
            ? chat.ImageEditStreamAsync(service, cancellationToken)
            : chat.ImageGenerateStreamAsync(service, cancellationToken);

        await foreach (object frame in Translate(frames, sessionId, cancellationToken).ConfigureAwait(false))
            yield return frame;
    }

    /// <summary>The image service's frames, as the chat stream's.</summary>
    internal static async IAsyncEnumerable<object> Translate(
        IAsyncEnumerable<object> frames, string? sessionId,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (object frame in frames.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            using JsonDocument document = JsonDocument.Parse(JsonSerializer.Serialize(frame));
            JsonElement f = document.RootElement;
            if (f.TryGetProperty("done", out _))
            {
                if (f.TryGetProperty("error", out JsonElement error) && error.ValueKind == JsonValueKind.String)
                {
                    yield return new { done = true, error = error.GetString(), sessionId };
                    yield break;
                }
                yield return new { imageUrl = Text(f, "url"), width = Number(f, "width"), height = Number(f, "height") };
                yield return new
                {
                    done = true,
                    sessionId,
                    tokenCount = 0,
                    elapsed = f.TryGetProperty("elapsedSeconds", out JsonElement s) && s.TryGetDouble(out double seconds) ? seconds : 0,
                    tokPerSec = 0.0,
                    truncated = false,
                };
                yield break;
            }

            yield return new
            {
                image_step = Number(f, "step"),
                image_steps = Number(f, "total"),
                preview = Text(f, "image"),
            };
        }

        // The service ends without a terminal frame only when the turn was stopped.
        yield return new { done = true, aborted = true, sessionId };
    }

    /// <summary>What the newest user message asks for, or null when it asks for nothing.</summary>
    internal static ImageRequest? Read(JsonElement body)
    {
        if (body.ValueKind != JsonValueKind.Object
            || !body.TryGetProperty("messages", out JsonElement messages)
            || messages.ValueKind != JsonValueKind.Array)
            return null;

        JsonElement? last = null;
        foreach (JsonElement message in messages.EnumerateArray())
        {
            if (message.ValueKind == JsonValueKind.Object
                && message.TryGetProperty("role", out JsonElement role)
                && role.ValueKind == JsonValueKind.String
                && role.GetString() == "user")
                last = message;
        }
        if (last is not { } user)
            return null;

        string prompt = Text(user, "content")?.Trim() ?? string.Empty;
        // Stills only: a video's sampled frames are not a photo to edit.
        string[] photos = Paths(user, "stillImagePaths");
        if (photos.Length > 0)
            return new ImageRequest(true, new { prompt, imagePaths = photos, targetArea = DefaultTargetArea });
        return prompt.Length == 0 ? null : new ImageRequest(false, new { prompt, targetArea = DefaultTargetArea });
    }

    private static string? Text(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static int Number(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.TryGetInt32(out int n) ? n : 0;

    private static string[] Paths(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.Array
            ? value.EnumerateArray()
                .Where(p => p.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(p.GetString()))
                .Select(p => p.GetString()!)
                .ToArray()
            : Array.Empty<string>();

    /// <summary>An edit (with the photos) or a picture made from words, as the service's body.</summary>
    internal sealed record ImageRequest(bool Editing, object Payload);
}
