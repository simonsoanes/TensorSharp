// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorSharp.Models.MiniMaxH3;

namespace TensorAgent.Tests;

/// <summary>
/// The two MiniMax-H3 entries: one model in two checkpoints, which share every file but
/// the denoiser, and whose files have to be the ones the engine looks for under the
/// variables it reads - a companion it does not find is a clip that fails, or one that is
/// silently made with another model's network.
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class MiniMaxH3CatalogTests
{
    private static CatalogModel Keyframes => ModelCatalog.Find("minimax-h3-fl2va-q4k")!;
    private static CatalogModel References => ModelCatalog.Find("minimax-h3-ref2va-q4k")!;

    private static readonly string[] VideoVariables =
    [
        "TS_VIDEO_TEXT_ENCODER", "TS_VIDEO_VAE", "TS_VIDEO_AUDIO_VAE", "TS_VIDEO_TOKENIZER",
    ];

    private static readonly string[] ImageVariables =
    [
        "TS_QWEN_IMAGE_VAE", "TS_QWEN_IMAGE_TE", "TS_QWEN_IMAGE_MMPROJ",
    ];

    [Fact]
    public void TheTwoEntriesAreTheTwoCheckpointsOfOneModel()
    {
        foreach (CatalogModel model in new[] { Keyframes, References })
        {
            Assert.Equal(CatalogFamily.MiniMaxH3, model.Family);
            Assert.Equal(CatalogArchitectureKind.Diffusion, model.Kind);
            Assert.True(model.IsVideoGenerator);
            Assert.False(model.IsImageGenerator);
            Assert.True(model.Modalities.HasFlag(CatalogModalities.VideoOutput | CatalogModalities.AudioOutput));
            Assert.Equal(0, model.ContextLength);
            // Every file is required: an optional file of a role the Models page never
            // offers could not be added later, and without the audio VAE every clip is mute.
            Assert.All(model.Files, f => Assert.False(f.Optional, $"{model.Id}/{f.FileName} is optional"));
        }

        // Everything but the denoiser is the same file, byte for byte, which is what lets
        // the second entry link the first one's copies instead of downloading them.
        static string Key(CatalogFile f) => $"{f.Role}|{f.FileName}|{f.Url}|{f.Bytes}|{f.Sha256}";
        Assert.Equal(
            Keyframes.Files.Where(f => f.Role != CatalogFileRole.Weights).Select(Key),
            References.Files.Where(f => f.Role != CatalogFileRole.Weights).Select(Key));
        Assert.NotEqual(Keyframes.Weights.Sha256, References.Weights.Sha256);

        // Photos are keyframes on one and references, beside clips and recordings, on the other.
        Assert.Equal(CatalogModalities.Image, Keyframes.Modalities & (CatalogModalities.Image | CatalogModalities.Video | CatalogModalities.Audio));
        Assert.Equal(CatalogModalities.Image | CatalogModalities.Video | CatalogModalities.Audio,
            References.Modalities & (CatalogModalities.Image | CatalogModalities.Video | CatalogModalities.Audio));
    }

    [Fact]
    public void EachEntryCarriesEveryFileTheDenoiserDoesNot()
    {
        foreach (CatalogModel model in new[] { Keyframes, References })
        {
            Assert.Single(model.Files, f => f.Role == CatalogFileRole.TextEncoder);
            Assert.Single(model.Files, f => f.Role == CatalogFileRole.Vae);
            Assert.Single(model.Files, f => f.Role == CatalogFileRole.AudioVae);
            // The encoder's GGUF has no tokenizer, and the vision markers live only in
            // tokenizer_config.json: without it a photo cannot be placed in the prompt.
            Assert.Equal(
                new[] { "merges.txt", "tokenizer_config.json", "vocab.json" },
                model.Files.Where(f => f.Role == CatalogFileRole.Tokenizer).Select(f => f.FileName).Order(StringComparer.Ordinal));
        }
    }

    [Fact]
    public void TheCheckpointIsTheOneItsFileNameSays()
    {
        // The engine reads the partition off the file name (the GGUFs carry no metadata),
        // and the app stores the file under the catalog's name.
        Assert.Equal(MiniMaxH3Partition.FirstLastFrame, MiniMaxH3Config.PartitionFromFileName(Keyframes.Weights.FileName));
        Assert.Equal(MiniMaxH3Partition.Reference, MiniMaxH3Config.PartitionFromFileName(References.Weights.FileName));
    }

    /// <summary>
    /// The engine resolves its companions from these variables and, failing that, from
    /// preferred file names - stated here, because they are private to the model, and
    /// kept honest by reading its source.
    /// </summary>
    [Fact]
    public void TheNamesAndVariablesThisCatalogReliesOnAreTheOnesTheEngineUses()
    {
        string model = ReadSource("TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3Model.cs");
        foreach (string literal in new[]
                 {
                     "\"TS_VIDEO_TEXT_ENCODER\"", "\"TS_VIDEO_VAE\"", "\"TS_VIDEO_AUDIO_VAE\"",
                     $"\"{Keyframes.Files.Single(f => f.Role == CatalogFileRole.TextEncoder).FileName}\"",
                     $"\"{Keyframes.Files.Single(f => f.Role == CatalogFileRole.Vae).FileName}\"",
                     $"\"{Keyframes.Files.Single(f => f.Role == CatalogFileRole.AudioVae).FileName}\"",
                 })
            Assert.Contains(literal, model, StringComparison.Ordinal);

        string encoder = ReadSource("TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3TextEncoder.cs");
        foreach (string literal in new[] { "\"TS_VIDEO_TOKENIZER\"", "\"vocab.json\"", "\"merges.txt\"", "\"tokenizer_config.json\"" })
            Assert.Contains(literal, encoder, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryInstalledCompanionIsPublishedAndTheTokenizerAsItsFolder()
    {
        using var installation = new FakeInstall(Keyframes, install: m => m.Files);

        IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(Keyframes, installation.Store);

        Assert.Equal(installation.PathOf(CatalogFileRole.TextEncoder), published["TS_VIDEO_TEXT_ENCODER"]);
        Assert.Equal(installation.PathOf(CatalogFileRole.Vae), published["TS_VIDEO_VAE"]);
        Assert.Equal(installation.PathOf(CatalogFileRole.AudioVae), published["TS_VIDEO_AUDIO_VAE"]);
        Assert.Equal(installation.Store.DirectoryFor(Keyframes), published["TS_VIDEO_TOKENIZER"]);
        Assert.Equal(VideoVariables.Order(StringComparer.Ordinal), published.Keys.Order(StringComparer.Ordinal));
        foreach (string variable in VideoVariables)
            Assert.Equal(published[variable], Environment.GetEnvironmentVariable(variable));
    }

    [Fact]
    public void ATokenizerFolderMissingAFileIsNotPublished()
    {
        Environment.SetEnvironmentVariable("TS_VIDEO_TOKENIZER", "/somewhere/from/before");
        using var installation = new FakeInstall(Keyframes,
            install: m => m.Files.Where(f => f.FileName != "tokenizer_config.json"));

        IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(Keyframes, installation.Store);

        Assert.False(published.ContainsKey("TS_VIDEO_TOKENIZER"));
        Assert.Null(Environment.GetEnvironmentVariable("TS_VIDEO_TOKENIZER"));
        Assert.True(published.ContainsKey("TS_VIDEO_TEXT_ENCODER"));
    }

    [Fact]
    public void SwitchingFamiliesLeavesNoOtherModelsPathsBehind()
    {
        using var video = new FakeInstall(Keyframes, install: m => m.Files);
        using var image = new FakeInstall(DiffusionModelFixture.QwenImage21, install: m => m.Files);

        DiffusionCompanions.Publish(DiffusionModelFixture.QwenImage21, image.Store);
        DiffusionCompanions.Publish(Keyframes, video.Store);
        foreach (string variable in ImageVariables)
            Assert.Null(Environment.GetEnvironmentVariable(variable));
        Assert.NotNull(Environment.GetEnvironmentVariable("TS_VIDEO_TEXT_ENCODER"));

        DiffusionCompanions.Publish(DiffusionModelFixture.QwenImage21, image.Store);
        foreach (string variable in VideoVariables)
            Assert.Null(Environment.GetEnvironmentVariable(variable));
        Assert.NotNull(Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_TE"));
    }

    [Fact]
    public void EachCheckpointIsPointedAtItsOwnFolder()
    {
        using var keyframes = new FakeInstall(Keyframes, install: m => m.Files);
        using var references = new FakeInstall(References, install: m => m.Files);

        Assert.Equal(references.PathOf(CatalogFileRole.TextEncoder),
            DiffusionCompanions.Publish(References, references.Store)["TS_VIDEO_TEXT_ENCODER"]);
        Assert.Equal(keyframes.PathOf(CatalogFileRole.TextEncoder),
            DiffusionCompanions.Publish(Keyframes, keyframes.Store)["TS_VIDEO_TEXT_ENCODER"]);
    }

    /// <summary>
    /// A half-downloaded set is refused before the engine sees it. The pipeline would find
    /// a missing companion by its name somewhere in the model store - for the text
    /// encoder, Qwen-Image's - and the weights file existing is all the old check asked.
    /// </summary>
    [Fact]
    public void AnIncompleteSetIsNotLoaded()
    {
        string root = Path.Combine(Path.GetTempPath(), "tensoragent-h3-" + Guid.NewGuid().ToString("N"));
        var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache")) { DeviceMemoryGB = 48 };
        paths.EnsureCreated();
        try
        {
            using var host = new AgentAppHost(paths);
            var store = new ModelStore(paths.ModelsDirectory);
            Directory.CreateDirectory(store.DirectoryFor(Keyframes));
            // The weights exist (empty, as a .part rename would leave a short file) and
            // nothing else does.
            File.WriteAllBytes(store.PathFor(Keyframes, Keyframes.Weights), []);

            var ex = Assert.Throws<FileNotFoundException>(() => host.UseModel(Keyframes, warmAfterwards: false));
            Assert.Equal("MiniMax-H3 is not completely downloaded yet.", ex.Message);
            Assert.Equal(AgentAppHost.ModelLoadState.Failed, host.ModelLoad);
        }
        finally
        {
            DiffusionCompanions.Publish(null, new ModelStore(paths.ModelsDirectory));
            try { Directory.Delete(root, true); } catch (Exception) { /* scratch */ }
        }
    }

    /// <summary>The repo's own copy of a file, so a test can read the source it mirrors.</summary>
    private static string ReadSource(string relativePath)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
            directory = directory.Parent;
        Assert.True(directory is not null, $"no repository root above {AppContext.BaseDirectory}");
        string full = Path.Combine(directory!.FullName, relativePath);
        Assert.True(File.Exists(full), $"{full} is missing");
        return File.ReadAllText(full);
    }

    /// <summary>An entry's chosen files, empty, in a scratch store: which paths exist is the point.</summary>
    private sealed class FakeInstall : IDisposable
    {
        private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-h3-install-" + Guid.NewGuid().ToString("N"));
        private readonly CatalogModel _model;

        public FakeInstall(CatalogModel model, Func<CatalogModel, IEnumerable<CatalogFile>> install)
        {
            _model = model;
            Store = new ModelStore(_root);
            Directory.CreateDirectory(Store.DirectoryFor(model));
            foreach (CatalogFile file in install(model))
                File.WriteAllBytes(Store.PathFor(model, file), []);
        }

        public ModelStore Store { get; }

        public string PathOf(CatalogFileRole role) =>
            Store.PathFor(_model, _model.Files.Single(f => f.Role == role));

        public void Dispose()
        {
            DiffusionCompanions.Publish(null, Store);
            try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
        }
    }
}
