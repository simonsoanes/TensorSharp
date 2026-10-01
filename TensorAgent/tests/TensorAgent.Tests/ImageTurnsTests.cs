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
using System.Text.RegularExpressions;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;

namespace TensorAgent.Tests;

/// <summary>
/// A chat turn when the loaded model makes pictures (<see cref="ImageTurns"/>): which
/// request a message becomes, how the image service's frames reach the page, and that the
/// picture is what the conversation keeps.
/// </summary>
public sealed class ImageTurnsTests
{
    private static JsonElement Body(string json) => JsonDocument.Parse(json).RootElement.Clone();

    private static JsonElement PayloadOf(ImageTurns.ImageRequest request) =>
        JsonSerializer.SerializeToElement(request.Payload);

    [Fact]
    public void WordsAloneMakeAPictureAtTheDefaultArea()
    {
        ImageTurns.ImageRequest request = Assert.IsType<ImageTurns.ImageRequest>(ImageTurns.Read(Body("""
            { "sessionId": "s1", "messages": [ { "role": "user", "content": "  a lighthouse at dusk  " } ] }
            """)));

        Assert.False(request.Editing);
        JsonElement payload = PayloadOf(request);
        Assert.Equal("a lighthouse at dusk", payload.GetProperty("prompt").GetString());
        Assert.Equal(1024L * 1024, payload.GetProperty("targetArea").GetInt64());
        Assert.False(payload.TryGetProperty("imagePaths", out _));
    }

    [Fact]
    public void AnAttachedPhotoMakesItAnEditOfThatPhotoAndNotOfAVideosFrames()
    {
        ImageTurns.ImageRequest request = Assert.IsType<ImageTurns.ImageRequest>(ImageTurns.Read(Body("""
            { "messages": [ {
                "role": "user", "content": "make the sky purple",
                "imagePaths": ["photo.png", "clip-frame-1.png"],
                "stillImagePaths": ["photo.png"]
            } ] }
            """)));

        Assert.True(request.Editing);
        JsonElement payload = PayloadOf(request);
        Assert.Equal("make the sky purple", payload.GetProperty("prompt").GetString());
        Assert.Equal(new[] { "photo.png" },
            payload.GetProperty("imagePaths").EnumerateArray().Select(p => p.GetString()).ToArray());
    }

    [Fact]
    public void OnlyTheNewestUserMessageIsTheRequest()
    {
        ImageTurns.ImageRequest request = Assert.IsType<ImageTurns.ImageRequest>(ImageTurns.Read(Body("""
            { "messages": [
                { "role": "user", "content": "a red bicycle", "stillImagePaths": ["old.png"] },
                { "role": "assistant", "content": "", "imageUrl": "/uploads/a.png" },
                { "role": "user", "content": "a blue boat" }
            ] }
            """)));

        Assert.False(request.Editing);
        Assert.Equal("a blue boat", PayloadOf(request).GetProperty("prompt").GetString());
    }

    [Theory]
    [InlineData("""{ "messages": [] }""")]
    [InlineData("""{ "messages": [ { "role": "user", "content": "   " } ] }""")]
    [InlineData("""{ "messages": [ { "role": "assistant", "content": "hello" } ] }""")]
    [InlineData("""{ "prompt": "no messages at all" }""")]
    public void NothingToDrawIsNoRequest(string json)
    {
        Assert.Null(ImageTurns.Read(Body(json)));
    }

    private static async IAsyncEnumerable<object> ServiceFrames(params object[] frames)
    {
        foreach (object frame in frames)
        {
            await Task.Yield();
            yield return frame;
        }
    }

    private static async Task<List<JsonElement>> CollectAsync(IAsyncEnumerable<object> frames)
    {
        var all = new List<JsonElement>();
        await foreach (object frame in frames)
            all.Add(JsonSerializer.SerializeToElement(frame));
        return all;
    }

    [Fact]
    public async Task TheServicesFramesBecomeStepsThenThePictureThenDone()
    {
        List<JsonElement> frames = await CollectAsync(ImageTurns.Translate(ServiceFrames(
            new { imageGenerate = true, step = 1, total = 2, image = (string?)null, width = 0, height = 0 },
            new { imageGenerate = true, step = 2, total = 2, image = "data:image/png;base64,AAAA", width = 64, height = 64 },
            new { done = true, url = "/uploads/picture.png", width = 1024, height = 1024, elapsedSeconds = 12.5 }),
            "session-1"));

        Assert.Equal(4, frames.Count);
        Assert.Equal(1, frames[0].GetProperty("image_step").GetInt32());
        Assert.Equal(2, frames[0].GetProperty("image_steps").GetInt32());
        Assert.Equal(JsonValueKind.Null, frames[0].GetProperty("preview").ValueKind);
        Assert.Equal("data:image/png;base64,AAAA", frames[1].GetProperty("preview").GetString());
        // A preview is never mistaken for the result: the page and the recorder read
        // `imageUrl`, and only the finished picture carries it.
        Assert.False(frames[1].TryGetProperty("imageUrl", out _));

        Assert.Equal("/uploads/picture.png", frames[2].GetProperty("imageUrl").GetString());
        Assert.Equal(1024, frames[2].GetProperty("width").GetInt32());
        Assert.True(frames[3].GetProperty("done").GetBoolean());
        Assert.Equal("session-1", frames[3].GetProperty("sessionId").GetString());
        Assert.Equal(12.5, frames[3].GetProperty("elapsed").GetDouble());
        Assert.False(frames[3].TryGetProperty("error", out _));
    }

    [Fact]
    public async Task AnErrorEndsTheTurnWithTheServicesOwnWords()
    {
        List<JsonElement> frames = await CollectAsync(ImageTurns.Translate(ServiceFrames(
            new { imageEdit = true, step = 1, total = 40, image = (string?)null, width = 0, height = 0 },
            new { done = true, error = "Image editing needs the vision file." }),
            "session-2"));

        JsonElement last = frames[^1];
        Assert.True(last.GetProperty("done").GetBoolean());
        Assert.Equal("Image editing needs the vision file.", last.GetProperty("error").GetString());
        Assert.DoesNotContain(frames, f => f.TryGetProperty("imageUrl", out _));
    }

    [Fact]
    public async Task AStreamThatEndsWithoutAResultIsAStoppedTurn()
    {
        // The service ends its stream without a terminal frame only when it was cancelled.
        List<JsonElement> frames = await CollectAsync(ImageTurns.Translate(ServiceFrames(
            new { imageGenerate = true, step = 1, total = 40, image = (string?)null, width = 0, height = 0 }),
            "session-3"));

        JsonElement last = frames[^1];
        Assert.True(last.GetProperty("done").GetBoolean());
        Assert.True(last.GetProperty("aborted").GetBoolean());
        Assert.Equal("session-3", last.GetProperty("sessionId").GetString());
    }

    /// <summary>
    /// The picture is what an image model's turn produced, and often all it produced; the
    /// recorder used to save a turn only when it had text, thinking or files, so a picture
    /// made while the page was hidden was gone when the chat was reopened.
    /// </summary>
    [Fact]
    public async Task ThePictureIsWhatTheConversationKeeps()
    {
        string root = Path.Combine(Path.GetTempPath(), "image-turns-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            var store = new ConversationStore(Path.Combine(root, "conversations"));
            Conversation conversation = store.Create();
            conversation.Messages.Add(new StoredMessage { Role = "user", Content = "a lighthouse at dusk" });
            store.Save(conversation);
            var recorder = new ConversationRecorder(store);
            recorder.Bind("session-picture", conversation.Id);
            using var turns = new ChatTurnManager(recorder);

            string turn = turns.Start(conversation.Id, _ => ImageTurns.Translate(ServiceFrames(
                new { imageGenerate = true, step = 1, total = 1, image = (string?)null, width = 0, height = 0 },
                new { done = true, url = "/uploads/lighthouse.png", width = 1024, height = 1024, elapsedSeconds = 3.0 }),
                "session-picture"));
            for (int i = 0; i < 200 && turns.StatusOfId(turn)!.IsRunning; i++)
                await Task.Delay(25);

            Conversation reloaded = Assert.IsType<Conversation>(
                new ConversationStore(Path.Combine(root, "conversations")).Load(conversation.Id));
            StoredMessage assistant = Assert.Single(reloaded.Messages, m => m.Role == "assistant");
            Assert.Equal("/uploads/lighthouse.png", assistant.ImageUrl);
            Assert.Contains("lighthouse.png", reloaded.Messages.SelectMany(m => m.ReferencedUploads));
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }
    }

    /// <summary>
    /// Every chat turn asks <see cref="ImageTurns.FramesFor"/> which service answers it.
    ///
    /// <para>
    /// The app host does not use the route's default frame source: it passes its GPU
    /// gate instead, and the first version of this feature decided only in the default,
    /// so in the app a picture request went straight to the text pipeline and failed on
    /// its first frame. No unit test could see it, because the decision depends on a real
    /// image model being loaded. This keeps the decision in one place: a direct use of the
    /// chat stream anywhere in the app is either listed here with the reason it is not a
    /// user's turn, or it is that bug again.
    /// </para>
    /// </summary>
    [Fact]
    public void EveryChatTurnAsksImageTurnsWhichServiceAnswersIt()
    {
        var allowed = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            // The decision itself.
            ["ImageTurns.cs"] = 1,
            // The prefix warm-up: a chat prompt by definition, and skipped for an image model.
            ["AgentAppHost.cs"] = 1,
            // The decode benchmark: measures text decoding, which an image model does not do.
            ["SpeculationBench.cs"] = 1,
        };

        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !Directory.Exists(Path.Combine(directory.FullName, "TensorAgent", "src")))
            directory = directory.Parent;
        Assert.NotNull(directory);
        string sources = Path.Combine(directory!.FullName, "TensorAgent", "src");

        var found = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string file in Directory.EnumerateFiles(sources, "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
                continue;
            int uses = File.ReadLines(file)
                .Where(line => !line.TrimStart().StartsWith("//", StringComparison.Ordinal))
                .Sum(line => Regex.Matches(line, @"\bChatStreamAsync\b").Count);
            if (uses > 0)
                found[Path.GetFileName(file)] = found.GetValueOrDefault(Path.GetFileName(file)) + uses;
        }

        Assert.Equal(
            allowed.OrderBy(p => p.Key).Select(p => $"{p.Key}={p.Value}"),
            found.OrderBy(p => p.Key).Select(p => $"{p.Key}={p.Value}"));
    }
}
