using System.Net;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text.Json;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.CodeExec;

namespace TensorAgent.Tests;

/// <summary>
/// The model directory is one persisted choice shared by the downloader, catalog and
/// loading paths. These tests use small synthetic files rather than loading an engine.
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class ModelCacheDirectoryTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "tensoragent-model-directory-" + Guid.NewGuid().ToString("N"));
    private readonly List<AgentAppHost> _hosts = new();

    private AgentPaths Paths => new(Path.Combine(_root, "data"), Path.Combine(_root, "cache"))
    {
        ExecutionMode = AgentExecutionMode.InProcess,
    };

    private AgentAppHost CreateHost()
    {
        var host = new AgentAppHost(Paths);
        _hosts.Add(host);
        return host;
    }

    private static HttpClient StartClient(AgentAppHost host)
    {
        host.Start();
        var client = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={host.Server.Token}");
        return client;
    }

    public void Dispose()
    {
        foreach (AgentAppHost host in _hosts)
            host.Dispose();
        CodeEnvironment.Reset();
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best-effort scratch cleanup */ }
    }

    [Fact]
    public void OlderSettingsAndBlankOverridesUseTheDefaultModelDirectory()
    {
        Paths.EnsureCreated();
        File.WriteAllText(Paths.SettingsFile, "{\"maxTokens\":1024}");
        AppSettings settings = new SettingsStore(Paths.SettingsFile).Load();

        Assert.Equal(string.Empty, settings.ModelCacheDirectory);
        Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), Paths.ResolveModelsDirectory(settings));
        Assert.Equal(Path.Combine(Path.GetFullPath(Paths.ModelsDirectory), "no-model-selected.gguf"),
            Paths.SelectedModelPath(settings));

        settings.ModelCacheDirectory = "   ";
        Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), Paths.ResolveModelsDirectory(settings));
    }

    [Fact]
    public void TheSelectedWeightsAndProjectorResolveInsideTheConfiguredDirectory()
    {
        CatalogModel model = ModelCatalog.BuiltIn.First(m => m.Projector is not null);
        string chosen = Path.Combine(_root, "downloaded models");
        var settings = new AppSettings
        {
            SelectedModelId = model.Id,
            ModelCacheDirectory = chosen,
        };

        Assert.Equal(chosen, Paths.ResolveModelsDirectory(settings));
        Assert.Equal(Path.Combine(chosen, model.Id, model.Weights.FileName), Paths.SelectedModelPath(settings));
        Assert.Equal(Path.Combine(chosen, model.Id, model.Projector!.FileName), Paths.SelectedProjectorPath(settings));

        settings.SelectedModelId = null;
        Assert.Equal(Path.Combine(chosen, "no-model-selected.gguf"), Paths.SelectedModelPath(settings));
        Assert.Null(Paths.SelectedProjectorPath(settings));
    }

    [Fact]
    public void ChangingAndResettingTheDirectorySurvivesRestartAndKeepsExistingFiles()
    {
        AgentAppHost host = CreateHost();
        ModelStore store = host.Models;
        CatalogModel model = Entry(new byte[] { 1, 2, 3 });
        Directory.CreateDirectory(store.DirectoryFor(model));
        string original = store.PathFor(model, model.Weights);
        File.WriteAllBytes(original, new byte[] { 1, 2, 3 });
        string chosen = Path.Combine(_root, "downloaded models");

        host.SetModelCacheDirectory(chosen);

        Assert.Same(store, host.Models);
        Assert.Equal(chosen, store.Root);
        Assert.Equal(chosen, new SettingsStore(Paths.SettingsFile).Load().ModelCacheDirectory);
        Assert.True(File.Exists(original));
        Assert.False(File.Exists(store.PathFor(model, model.Weights)));
        Directory.CreateDirectory(store.DirectoryFor(model));
        File.WriteAllBytes(store.PathFor(model, model.Weights), new byte[] { 4, 5, 6 });
        string looseFile = Path.Combine(chosen, "user-model-index.json");
        File.WriteAllText(looseFile, "user-owned model metadata");
        host.Dispose();
        _hosts.Remove(host);

        AgentAppHost restarted = CreateHost();
        Assert.Equal(chosen, restarted.Models.Root);
        Assert.Equal(InstallState.Installed, restarted.Models.StateOf(model));
        Assert.Equal(new byte[] { 4, 5, 6 }, File.ReadAllBytes(restarted.Models.WeightsPath(model)!));
        Assert.Equal("user-owned model metadata", File.ReadAllText(looseFile));

        restarted.SetModelCacheDirectory("   ");
        Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), restarted.Models.Root);
        Assert.Equal(string.Empty, new SettingsStore(Paths.SettingsFile).Load().ModelCacheDirectory);
        Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(restarted.Models.WeightsPath(model)!));
        Assert.True(File.Exists(Path.Combine(chosen, model.Id, model.Weights.FileName)));
        restarted.Dispose();
        _hosts.Remove(restarted);

        Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), CreateHost().Models.Root);
    }

    [Fact]
    public async Task DownloadsUseTheNewDirectoryAndCompletedJobsAllowAnotherChange()
    {
        AgentAppHost host = CreateHost();
        string chosen = Path.Combine(_root, "downloads");
        host.SetModelCacheDirectory(chosen);
        byte[] bytes = Enumerable.Range(0, 4096).Select(i => (byte)(i * 37)).ToArray();
        using var server = new RangeServer(bytes);
        CatalogModel model = Entry(bytes, server.Url);

        host.Downloads.Start(model);
        await WaitUntil(() => !host.Downloads.IsBusy);

        Assert.Equal(DownloadState.Completed, host.Downloads.StatusOf(model.Id)?.State);
        Assert.Equal(InstallState.Installed, host.Models.StateOf(model));
        Assert.Equal(Path.Combine(chosen, model.Id, model.Weights.FileName), host.Models.WeightsPath(model));
        Assert.Equal(bytes, await File.ReadAllBytesAsync(host.Models.WeightsPath(model)!));
        Assert.False(Directory.Exists(Path.Combine(Paths.ModelsDirectory, model.Id)));
        Assert.Equal(1, server.Requests);

        host.SetModelCacheDirectory(Path.Combine(_root, "next-downloads"));
        Assert.Equal(InstallState.NotInstalled, host.Models.StateOf(model));
        Assert.True(File.Exists(Path.Combine(chosen, model.Id, model.Weights.FileName)));
    }

    [Fact]
    public async Task SettingsApiReturnsTheEffectiveDirectoryAndPreservesItAgainstAStaleSave()
    {
        AgentAppHost host = CreateHost();
        using HttpClient client = StartClient(host);
        JsonElement initial = await client.GetFromJsonAsync<JsonElement>("/api/agent/settings");
        Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), initial.GetProperty("modelCacheDirectory").GetString());
        string chosen = Path.Combine(_root, "custom-models");

        using HttpResponseMessage changed = await client.PostAsJsonAsync(
            "/api/agent/settings/model-cache-directory", new { modelCacheDirectory = chosen });
        Assert.Equal(HttpStatusCode.OK, changed.StatusCode);
        Assert.Equal(chosen, host.Models.Root);
        Assert.Equal(chosen, new SettingsStore(Paths.SettingsFile).Load().ModelCacheDirectory);

        // The chat page may still hold the settings read before the directory changed.
        using HttpResponseMessage stale = await client.PostAsJsonAsync("/api/agent/settings", new
        {
            modelCacheDirectory = initial.GetProperty("modelCacheDirectory").GetString(),
            maxTokens = 4096,
        });
        Assert.Equal(HttpStatusCode.OK, stale.StatusCode);
        AppSettings saved = new SettingsStore(Paths.SettingsFile).Load();
        Assert.Equal(4096, saved.MaxTokens);
        Assert.Equal(chosen, saved.ModelCacheDirectory);
        Assert.Equal(chosen, host.Models.Root);
        JsonElement current = await client.GetFromJsonAsync<JsonElement>("/api/agent/settings");
        Assert.Equal(chosen, current.GetProperty("modelCacheDirectory").GetString());

        using HttpResponseMessage reset = await client.PostAsJsonAsync(
            "/api/agent/settings/model-cache-directory", new { modelCacheDirectory = " " });
        Assert.Equal(HttpStatusCode.OK, reset.StatusCode);
        Assert.Equal(string.Empty, host.Settings.Load().ModelCacheDirectory);
        Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), host.Models.Root);
        current = await client.GetFromJsonAsync<JsonElement>("/api/agent/settings");
        Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), current.GetProperty("modelCacheDirectory").GetString());
    }

    [Fact]
    public async Task InvalidDirectoriesCannotChangeThePersistedOrActiveLocation()
    {
        AgentAppHost host = CreateHost();
        string chosen = Path.Combine(_root, "custom-models");
        host.SetModelCacheDirectory(chosen);
        using HttpClient client = StartClient(host);
        string file = Path.Combine(_root, "a-file");
        await File.WriteAllTextAsync(file, "a directory cannot replace this file");

        foreach (string invalid in new[] { "relative-models", file, Path.Combine(file, "child") })
        {
            using HttpResponseMessage response = await client.PostAsJsonAsync(
                "/api/agent/settings/model-cache-directory", new { modelCacheDirectory = invalid });
            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
            JsonElement body = await response.Content.ReadFromJsonAsync<JsonElement>();
            Assert.False(string.IsNullOrWhiteSpace(body.GetProperty("error").GetString()));
            Assert.Equal(chosen, host.Models.Root);
            Assert.Equal(chosen, new SettingsStore(Paths.SettingsFile).Load().ModelCacheDirectory);
        }

        Assert.Equal("a directory cannot replace this file", await File.ReadAllTextAsync(file));
    }

    [Fact]
    public async Task AnyActiveDownloadJobRejectsADirectoryChangeThroughTheHostAndApi()
    {
        AgentAppHost host = CreateHost();
        using HttpClient client = StartClient(host);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        host.Downloads.Start("lora:busy-test", async (_, ct) => await release.Task.WaitAsync(ct));
        try
        {
            Assert.True(host.Downloads.IsBusy);
            string chosen = Path.Combine(_root, "other-models");
            Assert.Throws<InvalidOperationException>(() => host.SetModelCacheDirectory(chosen));
            using HttpResponseMessage response = await client.PostAsJsonAsync(
                "/api/agent/settings/model-cache-directory", new { modelCacheDirectory = chosen });
            Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
            Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), host.Models.Root);
            Assert.Equal(string.Empty, host.Settings.Load().ModelCacheDirectory);
        }
        finally
        {
            release.TrySetResult();
            await WaitUntil(() => !host.Downloads.IsBusy);
        }

        host.SetModelCacheDirectory(Path.Combine(_root, "after-job"));
        Assert.Equal(Path.Combine(_root, "after-job"), host.Models.Root);
    }

    [Fact]
    public async Task ADirectStoreDownloadPreventsChangingItsDestinationMidTransfer()
    {
        AgentAppHost host = CreateHost();
        byte[] bytes = new byte[2 * 1024 * 1024];
        using var server = new RangeServer(bytes) { DelayPerChunkMs = 25 };
        using var cancellation = new CancellationTokenSource();
        CatalogModel model = Entry(bytes, server.Url);
        string original = host.Models.PathFor(model, model.Weights);
        Task download = host.Models.DownloadAsync(model, progress: null, cancellation.Token);
        try
        {
            await WaitUntil(() => File.Exists(ResumableDownloader.PartPath(original)));
            Assert.False(host.Downloads.IsBusy);
            Assert.Throws<InvalidOperationException>(() =>
                host.SetModelCacheDirectory(Path.Combine(_root, "other-models")));
            Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), host.Models.Root);
            Assert.Equal(string.Empty, host.Settings.Load().ModelCacheDirectory);
        }
        finally
        {
            await cancellation.CancelAsync();
            try { await download; }
            catch (OperationCanceledException) { }
        }
    }

    [Fact]
    public async Task ALocalImportPreventsChangingItsDestinationUntilItFinishes()
    {
        AgentAppHost host = CreateHost();
        byte[] bytes = new byte[] { 10, 20, 30, 40 };
        CatalogModel model = Entry(bytes) with { SideloadOnly = true };
        await using var source = new BlockingReadStream(bytes);
        string original = host.Models.PathFor(model, model.Weights);
        Task import = host.Models.ImportAsync(model, source);
        try
        {
            await source.Entered.Task.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.False(host.Downloads.IsBusy);
            Assert.Throws<InvalidOperationException>(() =>
                host.SetModelCacheDirectory(Path.Combine(_root, "other-models")));
            Assert.Equal(Path.GetFullPath(Paths.ModelsDirectory), host.Models.Root);
            Assert.Equal(string.Empty, host.Settings.Load().ModelCacheDirectory);
        }
        finally
        {
            source.Release.TrySetResult();
            await import;
        }

        Assert.Equal(bytes, await File.ReadAllBytesAsync(original));
        host.SetModelCacheDirectory(Path.Combine(_root, "after-import"));
        Assert.Equal(Path.Combine(_root, "after-import"), host.Models.Root);
    }

    private static CatalogModel Entry(byte[] bytes, string url = "") => new()
    {
        Id = "cache-directory-test",
        DisplayName = "Model directory test",
        Family = CatalogFamily.Qwen35,
        Kind = CatalogArchitectureKind.Dense,
        Parameters = "test",
        Quantization = "test",
        Files = new[]
        {
            new CatalogFile(CatalogFileRole.Weights, "model.gguf", url, bytes.LongLength,
                Convert.ToHexStringLower(SHA256.HashData(bytes))),
        },
        Modalities = CatalogModalities.Text,
        MinDeviceMemoryGB = 1,
        ContextLength = 128,
        KvCacheDtype = "f16",
        Sampling = new CatalogSampling(1, 1, 1, 0),
        License = "test",
    };

    private static async Task WaitUntil(Func<bool> condition)
    {
        DateTime deadline = DateTime.UtcNow.AddSeconds(10);
        while (!condition())
        {
            Assert.True(DateTime.UtcNow < deadline, "timed out waiting for the model transfer");
            await Task.Delay(20);
        }
    }

    private sealed class BlockingReadStream(byte[] bytes) : MemoryStream(bytes, writable: false)
    {
        public TaskCompletionSource Entered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default)
        {
            Entered.TrySetResult();
            await Release.Task.WaitAsync(ct);
            return await base.ReadAsync(buffer, ct);
        }
    }
}
