// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorAgent.Core.Catalog;

/// <summary>What a file in a catalog entry is for. The app loads the weights and hands the
/// companions to the engine by role (the projector to <c>--mmproj</c>, a diffusion model's
/// companions to the environment variables its family reads; see DiffusionCompanions).</summary>
public enum CatalogFileRole
{
    /// <summary>The GGUF the model is loaded from.</summary>
    Weights,
    /// <summary>Vision/audio projector (mmproj) for a multimodal model.</summary>
    Projector,
    /// <summary>A diffusion model's text encoder GGUF (Qwen3-VL-8B for Qwen-Image-2.1,
    /// Qwen3-VL-32B for MiniMax-H3).</summary>
    TextEncoder,
    /// <summary>Qwen-Image-2.1 vision mmproj (image-grounded conditioning; required for editing).</summary>
    VisionProjector,
    /// <summary>A diffusion model's image or video VAE safetensors.</summary>
    Vae,
    /// <summary>Speculative-decoding draft head.</summary>
    Draft,
    /// <summary>MiniMax-H3's audio VAE safetensors, which turns the soundtrack it generates
    /// into sound (and encodes a reference recording).</summary>
    AudioVae,
    /// <summary>One of the loose tokenizer files a text encoder whose GGUF carries no
    /// tokenizer needs beside it (MiniMax-H3: vocab.json, merges.txt, tokenizer_config.json).</summary>
    Tokenizer,
    /// <summary>A later shard of a split weights GGUF (<c>-0000N-of-0000M.gguf</c>). The
    /// <see cref="Weights"/> file is shard 1, the one the engine is pointed at; it opens the
    /// others from the same folder by their exact gguf-split names, so every shard is required
    /// and stored under its published name.</summary>
    WeightsShard,
}

/// <summary>One downloadable artifact of a catalog entry.</summary>
/// <param name="Role">What the engine uses it for.</param>
/// <param name="FileName">The name it is stored under, inside the entry's folder.</param>
/// <param name="Url">Where it is fetched from (a plain Hugging Face resolve URL), or
/// empty for an artifact whose publisher did not embed a verifiable repository and
/// which must therefore be imported by the user.</param>
/// <param name="Bytes">Exact expected artifact size (from the Hugging Face tree API for downloads,
/// or from the inspected source file for a local import).</param>
/// <param name="Sha256">Lower-case expected SHA-256 (also the LFS object id for downloads), verified
/// before a download or import is published.</param>
/// <param name="Optional">True when the model works without it (e.g. a draft head).</param>
public sealed record CatalogFile(
    CatalogFileRole Role,
    string FileName,
    string Url,
    long Bytes,
    string Sha256,
    bool Optional = false);

/// <summary>Model families the catalog knows; used for grouping in the UI and for
/// family-specific defaults (thinking, sampling).</summary>
public enum CatalogFamily { Gemma4, Qwen35, Qwen36, Qwen38, QwenImage, GptOss, Bonsai, MuseGlimmer, MiniMaxH3, Qwen38FlashNext }

/// <summary>Dense or mixture-of-experts.</summary>
public enum CatalogArchitectureKind { Dense, MixtureOfExperts, Diffusion }

/// <summary>Input/output modalities an entry supports once its projector is installed.
/// <see cref="Image"/>, <see cref="Audio"/> and <see cref="Video"/> are what it takes in;
/// the <c>Output</c> flags are what it makes.</summary>
[Flags]
public enum CatalogModalities
{
    Text = 0,
    Image = 1,
    Audio = 2,
    Video = 4,
    ImageOutput = 8,
    VideoOutput = 16,
    AudioOutput = 32,
}

/// <summary>Sampling defaults the model card recommends; the app sends them with every
/// chat request the way the Web UI sends the server's configured defaults.</summary>
public sealed record CatalogSampling(float Temperature, int TopK, float TopP, float MinP);

/// <summary>
/// A built-in model the user can pick. Everything the app needs to obtain, size and
/// load it lives here so the catalog is data, not code: the UI shows it, the store
/// downloads or verifies an import, and the engine host turns it into load arguments.
/// </summary>
public sealed record CatalogModel
{
    public required string Id { get; init; }
    public required string DisplayName { get; init; }
    public required CatalogFamily Family { get; init; }
    public required CatalogArchitectureKind Kind { get; init; }
    /// <summary>Human-readable parameter count, e.g. "4B effective" or "26B (4B active)".</summary>
    public required string Parameters { get; init; }
    public required string Quantization { get; init; }
    public required IReadOnlyList<CatalogFile> Files { get; init; }
    public required CatalogModalities Modalities { get; init; }
    /// <summary>Smallest device memory class this entry is offered on. Weights are wired by
    /// Metal, so resident memory is roughly the GGUF size plus KV cache, the F32 projector
    /// and compute buffers; the class keeps a model that cannot fit off the picker.</summary>
    public required int MinDeviceMemoryGB { get; init; }
    /// <summary>Context length the app configures (MAX_CONTEXT); bounds the KV cache.</summary>
    public required int ContextLength { get; init; }
    /// <summary>KV cache dtype to request ("f16", "q8_0"); block-quantised caches halve KV memory
    /// where the family's fused paths accept them.</summary>
    public required string KvCacheDtype { get; init; }
    /// <summary>
    /// Keep the phone's cache budget on the desktop as well (see
    /// <see cref="Hosting.EngineMemoryPolicy"/>): caches that start small and grow, at most a
    /// little of the reply reserved ahead, one finished conversation kept, nothing parked.
    /// The engine's desktop defaults are written for a machine with memory to spare; for a
    /// model whose weights take most of a Mac's memory there is none, and they are what
    /// fills it.
    /// </summary>
    public bool LeanCaches { get; init; }
    public required CatalogSampling Sampling { get; init; }
    /// <summary>Whether the family has a thinking channel the app may enable.</summary>
    public bool SupportsThinking { get; init; }
    /// <summary>
    /// The card describes an exact, hash-pinned artifact but has no publisher URL the
    /// app can verify. The Models page offers a file picker instead of a download and
    /// <see cref="ModelStore.ImportAsync"/> verifies the selected bytes before exposing
    /// them to the engine.
    /// </summary>
    public bool SideloadOnly { get; init; }
    /// <summary>Marked in the UI: fits only with reduced context or has not been validated on
    /// a phone yet.</summary>
    public bool Experimental { get; init; }
    public string? Notes { get; init; }
    public required string License { get; init; }
    /// <summary>
    /// Bytes of the weights the engine reads from the SSD on demand instead of holding them
    /// resident: zero for every entry whose weights a token reads in full. Qwen3.8 Flash Next
    /// declares its n-gram table (a token reads 16 of its 320 M rows) and the routed experts
    /// the engine offloads on the entry's smallest device (a token reads 10 of each layer's
    /// 512), which is what lets a 79 GB file run on a 48 GB Mac. The residency checks hold
    /// <see cref="ResidentWeightsBytes"/> to the device instead of the whole file.
    /// </summary>
    public long WeightsPagedFromDiskBytes { get; init; }

    public long TotalBytes => Files.Where(f => !f.Optional).Sum(f => f.Bytes);
    /// <summary>The weights GGUF and its later shards, in shard order.</summary>
    public IEnumerable<CatalogFile> WeightFiles =>
        Files.Where(f => f.Role is CatalogFileRole.Weights or CatalogFileRole.WeightsShard);
    /// <summary>Every byte of the weights, all shards.</summary>
    public long WeightsBytes => WeightFiles.Sum(f => f.Bytes);
    /// <summary>The weights a token reads in full: everything not paged from disk on demand.</summary>
    public long ResidentWeightsBytes => WeightsBytes - WeightsPagedFromDiskBytes;
    public long TotalBytesWithOptional => Files.Sum(f => f.Bytes);
    public CatalogFile Weights => Files.First(f => f.Role == CatalogFileRole.Weights);
    public CatalogFile? Projector => Files.FirstOrDefault(f => f.Role == CatalogFileRole.Projector);
    public bool IsImageGenerator => Family == CatalogFamily.QwenImage;
    public bool IsVideoGenerator => Modalities.HasFlag(CatalogModalities.VideoOutput);
}
