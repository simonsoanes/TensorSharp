using System.Text.Json;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

public sealed class ConversationStoreTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "tensoragent-conv-" + Guid.NewGuid().ToString("N"));

    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    [Fact]
    public void RoundTripsMessagesWithTheWebUiPropertyNames()
    {
        var store = new ConversationStore(_dir);
        Conversation c = store.Create(modelId: "gemma-4-e4b-iq4xs", think: true, skills: new[] { "pdf" });
        c.Messages.Add(new StoredMessage
        {
            Role = "user",
            Content = "[File: notes.md]\nhello\n[End of file]\n\nSummarise this",
            TextFilePaths = new() { "abc.md" },
            TextFileNames = new() { "notes.md" },
            ImagePaths = new() { "img1.png" },
            Attachments = new() { new StoredAttachment { File = "abc.md", FileName = "notes.md", MediaType = "text" } },
        });
        c.Messages.Add(new StoredMessage { Role = "assistant", Content = "Done.", Thinking = "hmm",
            Artifacts = new() { new StoredArtifact { Name = "out.pdf", Bytes = 10, Url = "/api/code/artifacts/r/out.pdf" } } });
        store.Save(c);

        string json = File.ReadAllText(Path.Combine(_dir, c.Id + ".json"));
        using var doc = JsonDocument.Parse(json);
        JsonElement first = doc.RootElement.GetProperty("messages")[0];
        Assert.Equal("user", first.GetProperty("role").GetString());
        Assert.Equal("abc.md", first.GetProperty("textFilePaths")[0].GetString());
        Assert.Equal("notes.md", first.GetProperty("textFileNames")[0].GetString());
        Assert.False(first.TryGetProperty("audioPaths", out _));   // nulls are omitted, like the page's payload
        Assert.False(first.TryGetProperty("stats", out _));

        Conversation? loaded = store.Load(c.Id);
        Assert.NotNull(loaded);
        Assert.Equal("Summarise this", loaded!.Title);
        Assert.Equal(2, loaded.Messages.Count);
        Assert.Equal("hmm", loaded.Messages[1].Thinking);
        Assert.True(loaded.Think);
        Assert.Equal(new[] { "pdf" }, loaded.Skills);
        Assert.True(loaded.SkillsExplicit);
        Assert.Equal("gemma-4-e4b-iq4xs", loaded.ModelId);

        var refs = store.ReferencedUploads();
        Assert.Contains("abc.md", refs);
        Assert.Contains("img1.png", refs);
    }

    [Fact]
    public void TheNextRequestPreservesThePreviousAnswersStats()
    {
        var store = new ConversationStore(_dir);
        Conversation conversation = store.Create();
        var recorder = new ConversationRecorder(store);
        recorder.Bind("session-stats", conversation.Id);

        recorder.Complete("session-stats", "Done.", stats: new StoredTurnStats
        {
            TokenCount = 1200, Elapsed = 42.25, TokensPerSecond = 100.5,
            PromptTokens = 2000, KvReusedTokens = 1500, KvReusePercent = 75,
            Truncated = true,
        });
        Conversation prior = Assert.IsType<Conversation>(new ConversationStore(_dir).Load(conversation.Id));
        prior.Messages.Add(new StoredMessage { Role = "user", Content = "Continue." });

        // The page carries the saved counters in its history array. Record replaces
        // that array wholesale, so the new optional field must survive deserialization.
        recorder.Record("session-stats", JsonSerializer.SerializeToElement(new { messages = prior.Messages }));
        StoredMessage assistant = Assert.Single(new ConversationStore(_dir).Load(conversation.Id)!.Messages,
            message => message.Role == "assistant");
        StoredTurnStats stats = Assert.IsType<StoredTurnStats>(assistant.Stats);
        Assert.Equal(1200, stats.TokenCount);
        Assert.Equal(42.25, stats.Elapsed);
        Assert.Equal(100.5, stats.TokensPerSecond);
        Assert.Equal(2000, stats.PromptTokens);
        Assert.Equal(1500, stats.KvReusedTokens);
        Assert.Equal(75.0, stats.KvReusePercent);
        Assert.True(stats.Truncated);
    }

    [Fact]
    public void RoundTripsAFileBackedCsvMarkerWithoutStoringItsRows()
    {
        var store = new ConversationStore(_dir);
        Conversation conversation = store.Create();
        conversation.Messages.Add(new StoredMessage
        {
            Role = "user",
            Content = "Please analyze this form.",
            TextFilePaths = new() { "a5.csv" },
            TextFileNames = new() { "responses.csv" },
            Attachments = new()
            {
                new StoredAttachment
                {
                    File = "a5.csv",
                    FileName = "responses.csv",
                    MediaType = "text",
                    FileBacked = true,
                },
            },
        });

        store.Save(conversation);

        string json = File.ReadAllText(Path.Combine(_dir, conversation.Id + ".json"));
        Assert.DoesNotContain("[File:", json, StringComparison.Ordinal);
        Conversation loaded = Assert.IsType<Conversation>(store.Load(conversation.Id));
        StoredAttachment attachment = Assert.Single(Assert.Single(loaded.Messages).Attachments!);
        Assert.True(attachment.FileBacked);
    }

    [Fact]
    public void RecorderPersistsTheDifferenceBetweenDiscoveryAndDeselectAll()
    {
        var store = new ConversationStore(_dir);
        Conversation conversation = store.Create();
        var recorder = new ConversationRecorder(store);
        recorder.Bind("engine-session", conversation.Id);

        recorder.Record("engine-session", JsonSerializer.Deserialize<JsonElement>(
            """{"messages":[{"role":"user","content":"discover for me"}]}"""));
        Conversation discovered = Assert.IsType<Conversation>(store.Load(conversation.Id));
        Assert.Empty(discovered.Skills);
        Assert.False(discovered.SkillsExplicit);

        recorder.Record("engine-session", JsonSerializer.Deserialize<JsonElement>(
            """{"messages":[{"role":"user","content":"no tools"}],"skills":[]}"""));
        Conversation deselected = Assert.IsType<Conversation>(store.Load(conversation.Id));
        Assert.Empty(deselected.Skills);
        Assert.True(deselected.SkillsExplicit);
    }

    [Fact]
    public void ListIsNewestFirstAndSurvivesRestart()
    {
        var store = new ConversationStore(_dir);
        Conversation a = store.Create();
        a.Messages.Add(new StoredMessage { Role = "user", Content = "first" });
        store.Save(a);
        Thread.Sleep(20);
        Conversation b = store.Create();
        b.Messages.Add(new StoredMessage { Role = "user", Content = "second" });
        store.Save(b);

        var again = new ConversationStore(_dir);
        var list = again.List();
        Assert.Equal(2, list.Count);
        Assert.Equal(b.Id, list[0].Id);
        Assert.Equal("second", list[0].Title);
        Assert.Equal(1, list[0].MessageCount);

        Assert.True(again.Rename(a.Id, "My chat"));
        Assert.Equal("My chat", again.Load(a.Id)!.Title);
        Assert.True(again.Delete(a.Id));
        Assert.Single(again.List());
        Assert.Null(again.Load(a.Id));
    }

    [Fact]
    public void ATruncatedFileDoesNotBreakTheList()
    {
        var store = new ConversationStore(_dir);
        Conversation a = store.Create();
        File.WriteAllText(Path.Combine(_dir, a.Id + ".json"), "{\"id\":\"" + a.Id + "\",\"messages\":[");
        var again = new ConversationStore(_dir);
        // includeEmpty, because a file that will not parse contributes no messages and
        // the ordinary listing hides those; what is being tested is that it does not
        // throw or take the rest of the index with it.
        Assert.Single(again.List(includeEmpty: true));
        Assert.Null(again.Load(a.Id));
    }

    [Fact]
    public void TitlesComeFromTheFirstTypedLine()
    {
        var created = new DateTimeOffset(2026, 9, 1, 10, 0, 0, TimeSpan.Zero);
        Assert.Equal("What is in this image?", Conversation.DeriveTitle(new[] { new StoredMessage { Role = "user", Content = "What is in this image?\nmore" } }, created));
        string longLine = new string('x', 100);
        Assert.Equal(57, Conversation.DeriveTitle(new[] { new StoredMessage { Role = "user", Content = longLine } }, created).Length - 1);
        Assert.Equal("report.pdf", Conversation.DeriveTitle(new[] { new StoredMessage { Role = "user", Content = "[File: report.pdf]\ntext\n[End of file]\n\n", Attachments = new() { new StoredAttachment { FileName = "report.pdf" } } } }, created));
        Assert.StartsWith("Chat ", Conversation.DeriveTitle(Array.Empty<StoredMessage>(), created));
    }

    [Fact]
    public void IdsMustBeHexGuids()
    {
        Assert.True(ConversationStore.IsValidId(Guid.NewGuid().ToString("N")));
        Assert.False(ConversationStore.IsValidId("../etc/passwd"));
        Assert.False(ConversationStore.IsValidId(""));
        Assert.Null(new ConversationStore(_dir).Load("../x"));
    }

    [Fact]
    public void SettingsRoundTripAndDefaultWhenMissingOrCorrupt()
    {
        var store = new SettingsStore(Path.Combine(_dir, "settings.json"));
        AppSettings defaults = store.Load();
        Assert.True(defaults.AllowCodeExecution);
        Assert.False(defaults.AllowNetwork);
        Assert.Equal(2048, defaults.MaxTokens);

        defaults.SelectedModelId = "gemma-4-e2b-q8";
        defaults.AllowNetwork = true;
        defaults.DefaultSkills.Add("pdf");
        store.Save(defaults);
        AppSettings loaded = store.Load();
        Assert.Equal("gemma-4-e2b-q8", loaded.SelectedModelId);
        Assert.True(loaded.AllowNetwork);
        Assert.Equal(new[] { "pdf" }, loaded.DefaultSkills);

        File.WriteAllText(store.Path, "{ not json");
        Assert.Equal(2048, store.Load().MaxTokens);
    }
}
