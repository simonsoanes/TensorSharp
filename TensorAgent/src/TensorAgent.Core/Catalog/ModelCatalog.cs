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

/// <summary>
/// The built-in model list. Downloadable entries' sizes and hashes were read from the
/// Hugging Face tree API (LFS object ids) on 2026-09-01 (Bonsai 2 27B on 2026-09-28;
/// Qwen3.8 27B, Muse-Glimmer 30B and Qwen-Image 2.1 on 2026-09-30; MiniMax-H3 on
/// 2026-09-30 from bare clones of unsloth/MiniMax-H3-GGUF at d629413c and
/// MiniMaxAI/MiniMax-H3 at 42ed227e, their current heads, whose three tokenizer files are
/// plain git blobs hashed from their bytes),
/// so a download is verified against the exact bytes the publisher uploaded. A
/// sideload-only entry would carry the same immutable size/hash identity but
/// deliberately no URL, for a GGUF that embeds no publisher repository; the app then
/// verifies a user-selected local file instead. No built-in entry needs that today.
///
/// <para>
/// Sizing rule (why these quantizations): on GGML Metal the quantized weights are wrapped
/// as MTLBuffers straight over the GGUF mmap, and Metal keeps those pages resident, so
/// the resident set is about the file size plus the KV cache, the projector (materialised
/// as F32, so roughly twice its file size) and ~0.5 GB of compute buffers. An iPhone with
/// 12 GB grants an app with the increased-memory entitlement roughly 8.5 GB, which is why
/// the 12 GB tier tops out around 7.5 GB of weights and everything larger is offered only
/// to 16 GB iPads.
/// </para>
/// <para>
/// The 24 and 32 GB tiers are the desktop app's (a Mac today; no iPhone or iPad has that
/// much memory). Their entries are the models a phone can only run as one- or two-bit
/// quantizations that cost real quality, so they are offered at four bits where the
/// memory exists rather than squeezed onto a phone. A Mac has no jetsam, so the tier is
/// what the weights (which a dense model reads in full for every token) plus the app's
/// measured footprint plus macOS need; below that every token pages the model in from
/// disk. The two large chat models also keep the phone's cache budget
/// (<see cref="CatalogModel.LeanCaches"/>), without which the engine's desktop defaults
/// alone grew the app to 18.8 GB beside Qwen3.8 27B's weights.
/// </para>
/// </summary>
public static class ModelCatalog
{
    private const string GemmaLicense = "Gemma Terms of Use";
    private const string ApacheLicense = "Apache-2.0";
    // The denoisers' license. Its "Applicable Territory" excludes the EU, the UK, the
    // Republic of Korea and the US, which the entries' notes say; the Qwen3-VL text
    // encoder is Apache-2.0.
    private const string MiniMaxH3License = "MiniMax H3 Community License";

    private static string Hf(string repo, string file) => $"https://huggingface.co/{repo}/resolve/main/{file}";

    public static IReadOnlyList<CatalogModel> BuiltIn { get; } = new[]
    {
        new CatalogModel
        {
            Id = "gemma-4-e2b-q8",
            DisplayName = "Gemma 4 E2B",
            Family = CatalogFamily.Gemma4,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "2B effective (5B with per-layer embeddings)",
            Quantization = "Q8_0",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "gemma-4-E2B-it-Q8_0.gguf",
                    Hf("ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf"),
                    4_967_497_152, "996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-gemma-4-E2B-it-Q8_0.gguf",
                    Hf("ggml-org/gemma-4-E2B-it-GGUF", "mmproj-gemma-4-E2B-it-Q8_0.gguf"),
                    557_368_064, "9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292"),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Audio | CatalogModalities.Video,
            MinDeviceMemoryGB = 12,
            ContextLength = 8192,
            // f16, not q8_0: Gemma 4 declines a block-quantized cache
            // (Gemma4Model.SupportsBlockQuantizedKvCache). Its sliding-window layers
            // use a circular cache whose managed helpers are float-only, and the 26B
            // MoE reaches them on an ordinary prompt: setting q8_0 here crashed with
            // "Requires a Float32 tensor, but found Q8_0" out of CopyToCacheCircular.
            KvCacheDtype = "f16",
            Sampling = new CatalogSampling(1.0f, 64, 0.95f, 0.0f),
            SupportsThinking = true,
            License = GemmaLicense,
            Notes = "Fastest option. Sees images and video frames, hears audio, thinks when asked.",
        },
        new CatalogModel
        {
            Id = "gemma-4-e4b-iq4xs",
            DisplayName = "Gemma 4 E4B",
            Family = CatalogFamily.Gemma4,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "4B effective (8B with per-layer embeddings)",
            Quantization = "IQ4_XS",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "gemma-4-E4B-it-IQ4_XS.gguf",
                    Hf("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-IQ4_XS.gguf"),
                    4_715_416_704, "0847f7300471e9a61abaeb46b45b3d61e393af8fa6d6aa70c563623773670dd9"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-gemma-4-E4B-it-Q8_0.gguf",
                    Hf("ggml-org/gemma-4-E4B-it-GGUF", "mmproj-gemma-4-E4B-it-Q8_0.gguf"),
                    559_874_816, "197f49a93027f9843772bd24a6a9e0be2a32a788de5a3def330e9c585d86edd1"),
                new CatalogFile(CatalogFileRole.Draft, "mtp-gemma-4-E4B-it-Q8_0.gguf",
                    Hf("ggml-org/gemma-4-E4B-it-GGUF", "mtp-gemma-4-E4B-it-Q8_0.gguf"),
                    98_653_280, "f38ae62962657c7a6303c49bbb147e9ae23634e911cfa532fac0818c2e18b665", Optional: true),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Audio | CatalogModalities.Video,
            MinDeviceMemoryGB = 12,
            ContextLength = 8192,
            KvCacheDtype = "f16",
            Sampling = new CatalogSampling(1.0f, 64, 0.95f, 0.0f),
            SupportsThinking = true,
            License = GemmaLicense,
            Notes = "The recommended default: TensorSharp's verified fast-path tier, multimodal, with an optional speculative draft head.",
        },
        new CatalogModel
        {
            Id = "gemma-4-12b-iq2m",
            DisplayName = "Gemma 4 12B",
            Family = CatalogFamily.Gemma4,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "12B",
            Quantization = "UD-IQ2_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "gemma-4-12b-it-UD-IQ2_M.gguf",
                    Hf("unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-UD-IQ2_M.gguf"),
                    4_213_353_280, "4bd2461d35398dbcf5f3d5f0c9ad91cac78ae35b556e3a81f315a0cc0815ae8c"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-F16.gguf",
                    Hf("unsloth/gemma-4-12b-it-GGUF", "mmproj-F16.gguf"),
                    175_115_840, "91f086971e56d7a7d8d39e271873fccdb49541bd259d6e02c401a4f1cb7a219e", Optional: true),
                // The per-token assistant head, the same shape gemma-4-e4b-iq4xs carries.
                new CatalogFile(CatalogFileRole.Draft, "mtp-gemma-4-12b-it.gguf",
                    Hf("unsloth/gemma-4-12b-it-GGUF", "mtp-gemma-4-12b-it.gguf"),
                    465_109_248, "145db9094bc0f85f1701e255a2ed216dcc9800fc8bc8631ad00905b456bd451b", Optional: true),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Video,
            MinDeviceMemoryGB = 12,
            // 32768, not 8192. The window was small because the LOAD was expensive, not
            // because the cache was: fusing this model's 48 ffn_gate/ffn_up pairs cost
            // gigabytes of anonymous memory duplicating bytes already mapped from the
            // GGUF, which is what jetsam killed the app for. Gemma 4 now runs the pair
            // as two matmuls instead (Gemma4Model.SupportsSplitGateUpFfn), and the
            // window is affordable. MEASURED on the Q4_K_XL build of this same model,
            // ggml_metal, where the fusion was 3,108 MB:
            //   fused,  8192   peak footprint 4,534 MB   31.7 tok/s
            //   split,  8192                   1,423 MB   31.5 tok/s
            //   split, 32768                   2,187 MB   31.3 tok/s
            // Output is byte-identical between the first two, and a 14,294-token needle
            // prompt returns the planted value on both. THIS entry is the UD-IQ2_M
            // build, and it reaches the split path by the same route the measurement
            // above did: on iOS ModelBase.AllowWeightFusionCopies is false, so the
            // fused copy is never made whatever the types are. Nothing here depends on
            // ggml declining to requantize -- VERIFIED against the pinned file, whose
            // 48 ffn_gate/ffn_up pairs match on BOTH sides (43 IQ2_S pairs and five
            // IQ3_XXS pairs), so the mismatch branch that would call for a requantize
            // is never entered at all.
            ContextLength = 32768,
            // f16, like every Gemma entry: Gemma 4 refuses a block-quantized cache
            // (Gemma4Model.SupportsBlockQuantizedKvCache) because its sliding-window
            // layers use a circular cache whose managed helpers are float-only.
            KvCacheDtype = "f16",
            Sampling = new CatalogSampling(1.0f, 64, 0.95f, 0.0f),
            SupportsThinking = true,
            License = GemmaLicense,
            Notes = "The dense Gemma between E4B and the 26B mixture of experts, in the 2.7 bpw IQ2_M recipe. "
                + "4.2 GB of weights, mapped from the file rather than copied, so the model itself "
                + "costs the phone almost nothing and the 32k window is the larger part of its "
                + "footprint. Vision and a speculative draft head are optional downloads.",
        },
        new CatalogModel
        {
            Id = "bonsai-2-27b-ptq1-0",
            DisplayName = "Bonsai 2 27B",
            Family = CatalogFamily.Bonsai,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "27B",
            Quantization = "PTQ1_0",
            Files = new[]
            {
                // qwen35 hybrid (48 Gated DeltaNet + 16 full-attention layers) with ternary
                // g128 weights in PRISM's rotated basis. Sizes and LFS ids were read from
                // the tree API on 2026-09-28 (revision b072e1d3b35a0a630cece372c2127528e0994386).
                // PTQ1_0 and PQ2_0 hold the same trits, and the loader repacks either one
                // losslessly to GGML Q2_0, so the dense PTQ1_0 packing is simply the
                // smaller download (5.95 GB against 7.21 GB) for the same resident model.
                new CatalogFile(CatalogFileRole.Weights, "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
                    Hf("prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
                    5_946_648_928, "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"),
                new CatalogFile(CatalogFileRole.Projector, "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
                    Hf("prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"),
                    629_246_976, "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903", Optional: true),
            },
            Modalities = CatalogModalities.Image,
            // 16, not 12: unlike the entries above, these weights are not served from the
            // GGUF mapping. The Q2_0 repack lives in anonymous memory, ~7.7 GB for this
            // file (PTQ1_0 grows by about 29%), which with the KV cache and compute
            // buffers is more than the ~8.5 GB a 12 GB iPhone grants the app.
            MinDeviceMemoryGB = 16,
            ContextLength = 32768,
            KvCacheDtype = "q8_0",
            // The publisher's thinking-mode recommendation (the GGUF does not carry min_p).
            Sampling = new CatalogSampling(1.0f, 20, 0.95f, 0.05f),
            SupportsThinking = true,
            // Bonsai2 is validated on desktop Metal/CPU, not yet on an iPad.
            Experimental = true,
            License = ApacheLicense,
            Notes = "PrismML's ternary Qwen 3.5 hybrid with Hadamard-rotated weights. The 5.9 GB download is "
                + "repacked losslessly to about 7.7 GB when it loads, so it is offered only on 16 GB devices. "
                + "Vision is an optional download.",
        },
        new CatalogModel
        {
            Id = "qwen3.5-9b-iq4xs",
            DisplayName = "Qwen3.5 9B",
            Family = CatalogFamily.Qwen35,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "9B",
            Quantization = "IQ4_XS",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Qwen3.5-9B-IQ4_XS.gguf",
                    Hf("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-IQ4_XS.gguf"),
                    5_168_653_536, "7e918aeca06c52bcb528ea6b04b4ec957e75ee8c0a73138854c0dfcf371ea429"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-F16.gguf",
                    Hf("unsloth/Qwen3.5-9B-GGUF", "mmproj-F16.gguf"),
                    918_166_080, "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f", Optional: true),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Video,
            MinDeviceMemoryGB = 12,
            // 32768, not 8192: the reply-length setting is bounded by the CONTEXT
            // (ChatGenerationPipeline.ClampGenerationReserve trims the generation
            // reserve to what the window leaves after the prompt, and the thinking
            // budget is 75% of THAT), so an 8192 window capped a reply at ~7.7k
            // tokens however high the user set the limit -- and a reasoning model
            // that spent it produced no answer at all. A quantized cache pays for
            // the bigger window: MEASURED on Qwen3.5-9B UD-Q4_K_XL, ggml_metal,
            // peak physical footprint is 1697 MB at q8_0/32768 against the 1118 MB
            // that f16/8192 already cost, and q8_0/16384 (1152 MB) is a wash.
            ContextLength = 32768,
            // q8_0, not f16: the KV cache is the only thing that grows with the
            // conversation, and on Metal it is charged twice. MEASURED on ggml_metal
            // after the fused graphs learned block-quantized K/V: 22.4 KiB/token
            // against f16's 41.8, at decode parity (Qwen3.6-35B-A3B 75.5 vs 76.0
            // tok/s, within noise), with a two-needle recall test at 7,490 tokens
            // returning both planted values. Qwen3.5/3.6 take the fused graph, whose
            // native side is dtype-generic; Gemma 4 does not and stays on f16.
            KvCacheDtype = "q8_0",
            Sampling = new CatalogSampling(0.7f, 20, 0.8f, 0.0f),
            SupportsThinking = true,
            License = ApacheLicense,
            Notes = "Strong general model with vision; the projector is optional and costs ~1.8 GB of memory when loaded.",
        },
        new CatalogModel
        {
            Id = "qwen3.8-27b-q4kxl",
            DisplayName = "Qwen3.8 27B",
            Family = CatalogFamily.Qwen38,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "27B",
            Quantization = "UD-Q4_K_XL",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Qwen3.8-27B-UD-Q4_K_XL.gguf",
                    Hf("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_XL.gguf"),
                    17_559_178_144, "3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-F16.gguf",
                    Hf("unsloth/Qwen3.8-27B-GGUF", "mmproj-F16.gguf"),
                    927_607_488, "cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e", Optional: true),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Video,
            // 32, not 24: the weights are mapped, not charged, but a dense model reads all
            // of them for every token, so they must stay resident beside what the app
            // allocates. MEASURED in the Mac app (M5 Pro, ggml_metal, chat-e2e.py's seven
            // scenarios with the projector): the footprint peaked at 8.8 GB with
            // LeanCaches, so 17.6 + 8.8 GB plus macOS is about 31 GB. (The 1-bit build that
            // fits a phone, UD-IQ1_S at 6.2 GB, was withdrawn on 2026-09-07 for quality.)
            MinDeviceMemoryGB = 32,
            // A qwen35 hybrid: one layer in four is full attention and the rest keep a
            // fixed linear-attention state, so a token costs about 35 KB of K/V at q8_0,
            // charged twice on Metal. The qwen35 fused graph reads q8_0 at decode parity.
            ContextLength = 32768,
            KvCacheDtype = "q8_0",
            // MEASURED: the engine's desktop defaults grew the app's footprint to 18.8 GB
            // over the same seven scenarios (the machine fell to 0.1 GB free with 10.6 GB
            // compressed); the phone's budget held it at 8.8 GB with the same 99.6-99.7%
            // prompt reuse on follow-ups and new chats.
            LeanCaches = true,
            // The publisher's non-thinking settings, as for Qwen3.5 9B: the app starts
            // with thinking off.
            Sampling = new CatalogSampling(0.7f, 20, 0.8f, 0.0f),
            SupportsThinking = true,
            License = ApacheLicense,
            Notes = "Qwen's dense 27B at four bits: stronger than the 9B at reasoning, coding and agent work, "
                + "and slower, since all 27B parameters run for every token. Needs 32 GB of memory (a Mac). "
                + "Vision is an optional download.",
        },
        new CatalogModel
        {
            Id = "muse-glimmer-30b-q4kxl",
            DisplayName = "Muse-Glimmer 30B",
            Family = CatalogFamily.MuseGlimmer,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "30B",
            Quantization = "UD-Q4_K_XL",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
                    Hf("unsloth/Muse-Glimmer-30B-GGUF", "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"),
                    15_878_222_368, "82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-Muse-Glimmer-30B-Q8_0.gguf",
                    Hf("unsloth/Muse-Glimmer-30B-GGUF", "mmproj-Muse-Glimmer-30B-Q8_0.gguf"),
                    2_051_685_088, "01ff73c95108e1754a4c145176c6d3ba44338942285cb87dcac7f4f193192ea2", Optional: true),
                // No draft entry: its DFlash drafter verifies greedily against the model,
                // and the app samples at the card's temperature, so the drafter would buy
                // nothing here.
            },
            Modalities = CatalogModalities.Image,
            // 32 for the same reason as Qwen3.8 27B: a dense 15.9 GB read for every token
            // beside the app. MEASURED in the Mac app (M5 Pro, ggml_metal, chat-e2e.py's
            // seven scenarios): footprint 10.9 GB at its highest, about 8 GB of it the
            // vision tower, which loads whenever its file is installed.
            MinDeviceMemoryGB = 32,
            ContextLength = 32768,
            // q8_0 MEASURED on ggml_metal: a 1,542-word answer (2,404 tokens) that grew the
            // full-attention layers past 8,192 rows mid-reply came out clean, at 17.0 tok/s
            // against the CLI's 17.2.
            KvCacheDtype = "q8_0",
            LeanCaches = true,
            // The model card's recommendation (the GGUF carries no sampling metadata).
            Sampling = new CatalogSampling(1.0f, 64, 0.95f, 0.0f),
            SupportsThinking = true,
            License = ApacheLicense,
            Notes = "A dense 30B that reads images and calls tools. It reasons in its own channel before "
                + "answering, even with thinking off. Needs 32 GB of memory (a Mac). Vision is an optional "
                + "download and takes about 8 GB more memory when it is installed.",
        },
        new CatalogModel
        {
            // The set config/qwen-image-2.1.json downloads: the DiT is only the diffusion
            // transformer, and the VAE, the Qwen3-VL-8B text encoder and its vision
            // projector are separate files (see DiffusionCompanions).
            Id = "qwen-image-2.1-q4km",
            DisplayName = "Qwen-Image 2.1",
            Family = CatalogFamily.QwenImage,
            Kind = CatalogArchitectureKind.Diffusion,
            Parameters = "Qwen-Image-2.1 DiT + Qwen3-VL-8B text encoder",
            Quantization = "Q4_K_M (DiT) / Q4_K_M (text encoder)",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "qwen_image_2.1_Q4_K_M.gguf",
                    Hf("Abiray/Qwen-Image-2.1-GGUF", "qwen_image_2.1_Q4_K_M.gguf"),
                    4_189_343_904, "dc956c958fbfa1d5c64ec316d7e865283d17d97a9eb332a4a74a4d63afaae9a5"),
                new CatalogFile(CatalogFileRole.TextEncoder, "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                    Hf("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
                    5_027_784_800, "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"),
                new CatalogFile(CatalogFileRole.Vae, "qwen_image_2.1_vae_bf16.safetensors",
                    Hf("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors"),
                    675_509_688, "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"),
                // Required, unlike a chat model's projector. Making a picture from words
                // does not use it, but editing a photo refuses to run without it, and an
                // optional file here could never be added later: the Models page's "add
                // vision" action fetches a chat model's Projector, not this role. Optional,
                // a download with optional files switched off made a model that advertises
                // photo editing and cannot do it.
                new CatalogFile(CatalogFileRole.VisionProjector, "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
                    Hf("Qwen/Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
                    1_159_029_824, "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38"),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.ImageOutput,
            MinDeviceMemoryGB = 24,
            // A diffusion model holds no KV cache; generation size and steps are chosen
            // per request (ImageTurns).
            ContextLength = 0,
            KvCacheDtype = "f16",
            Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
            License = ApacheLicense,
            Notes = "Makes a picture from a description, or edits an attached photo. Needs 24 GB of memory "
                + "(a Mac).",
        },
        MiniMaxH3(
            id: "minimax-h3-fl2va-q4k",
            displayName: "MiniMax-H3",
            denoiser: new CatalogFile(CatalogFileRole.Weights, "minimax_h3_fl2va_pruned-Q4_K.gguf",
                Hf("unsloth/MiniMax-H3-GGUF", "minimax_h3_fl2va_pruned-Q4_K.gguf"),
                11_420_663_904, "dd948e08ad0ba3c71bd42f368e283dd82e790f5122a63b276e22a3e0283d0c10"),
            // Photos are keyframes here: one is the clip's first frame, two its first and last.
            modalities: CatalogModalities.Image,
            notes: "Makes a short video with its own soundtrack from a description, brings an attached photo to "
                + "life, or runs between two photos (the first and the last frame)."),
        MiniMaxH3(
            id: "minimax-h3-ref2va-q4k",
            displayName: "MiniMax-H3 References",
            // The checkpoint is chosen by the "ref2va" in this name (MiniMaxH3Config.PartitionFromFileName).
            denoiser: new CatalogFile(CatalogFileRole.Weights, "minimax_h3_ref2va_pruned-Q4_K.gguf",
                Hf("unsloth/MiniMax-H3-GGUF", "minimax_h3_ref2va_pruned-Q4_K.gguf"),
                11_381_096_544, "2fa5840021cf6967843eaeefde9aaa277e540de02986d5ee3d5b0e6a7a8c9dec"),
            // Photos, clips and recordings are references for a new scene, up to nine.
            modalities: CatalogModalities.Image | CatalogModalities.Video | CatalogModalities.Audio,
            notes: "Makes a short video with its own soundtrack that features the people, things, places or "
                + "sounds in up to nine attached photos, clips and recordings. Say who or what is in the shot: "
                + "a reference gives their look, the description puts them in the scene."),
    };

    /// <summary>
    /// A MiniMax-H3 entry: one of its two denoisers (it is two checkpoints, not a setting)
    /// plus the files both share. The shared ones are byte-identical between the entries,
    /// so a second entry links the first one's copies instead of downloading 24 GB again
    /// (ModelStore), and only its denoiser is new.
    /// </summary>
    private static CatalogModel MiniMaxH3(
        string id, string displayName, CatalogFile denoiser, CatalogModalities modalities, string notes) => new()
    {
        Id = id,
        DisplayName = displayName,
        Family = CatalogFamily.MiniMaxH3,
        Kind = CatalogArchitectureKind.Diffusion,
        Parameters = "33B audio-video DiT (20B with its AdaLN branches pruned) + Qwen3-VL-32B text encoder",
        Quantization = "Q4_K (DiT) / Q4_K_M (text encoder)",
        Files = new[]
        {
            denoiser,
            // Truncated to the 50 layers H3 reads, and carrying the Qwen3-VL vision tower that
            // presents a photo to the prompt.
            new CatalogFile(CatalogFileRole.TextEncoder, "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                Hf("unsloth/MiniMax-H3-GGUF", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"),
                18_218_065_024, "11e6efe70a57ce7f4838c47bdbd1a1c4b8ce10e2b7747f1b065990b70f4b05fc"),
            new CatalogFile(CatalogFileRole.Vae, "minimax_h3_video_vae_fp16.safetensors",
                Hf("unsloth/MiniMax-H3-GGUF", "vae/minimax_h3_video_vae_fp16.safetensors"),
                5_207_808_496, "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"),
            // Required, not optional: without it every clip is silent, and an optional file of
            // a role the Models page never offers could not be added later.
            new CatalogFile(CatalogFileRole.AudioVae, "minimax_h3_audio_vae_fp32.safetensors",
                Hf("unsloth/MiniMax-H3-GGUF", "vae/minimax_h3_audio_vae_fp32.safetensors"),
                605_254_808, "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"),
            // The text encoder's GGUF carries no tokenizer. tokenizer_config.json is the only
            // place its vision markers are defined; without it a photo cannot be placed in the
            // prompt (MiniMaxH3Pipeline.RequireVisionTokens).
            new CatalogFile(CatalogFileRole.Tokenizer, "vocab.json",
                Hf("MiniMaxAI/MiniMax-H3", "processor/vocab.json"),
                2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
            new CatalogFile(CatalogFileRole.Tokenizer, "merges.txt",
                Hf("MiniMaxAI/MiniMax-H3", "processor/merges.txt"),
                1_671_839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
            new CatalogFile(CatalogFileRole.Tokenizer, "tokenizer_config.json",
                Hf("MiniMaxAI/MiniMax-H3", "processor/tokenizer_config.json"),
                11_003, "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68"),
        },
        Modalities = modalities | CatalogModalities.VideoOutput | CatalogModalities.AudioOutput,
        MinDeviceMemoryGB = 32,
        // A diffusion model holds no KV cache; the clip's size and length are chosen per
        // request (VideoTurns).
        ContextLength = 0,
        KvCacheDtype = "f16",
        Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
        License = MiniMaxH3License,
        Notes = notes + " Needs 32 GB of memory (a Mac). MiniMax's license does not cover use in the EU, the UK, "
            + "South Korea or the US.",
    };

    /// <summary>
    /// The ids earlier versions of this catalog shipped and this one no longer does: an
    /// entry re-pointed at another file (the id names the quantization, so
    /// <c>gemma-4-12b-q4kxl</c> became <c>gemma-4-12b-iq2m</c>) or withdrawn. What is stored
    /// under one of these names will never be loaded again, and is what the launch sweeps
    /// reclaim (<see cref="ModelStore.SweepOrphanedModels"/> and the prefix checkpoints).
    ///
    /// <para>
    /// An id in neither list is not reclaimed, because it is not necessarily an old one.
    /// On a Mac the Debug and Release builds share one models directory, and a build
    /// older than the catalog that installed a model does not know its id: a Release
    /// build from the day before, treating every id it did not know as retired, deleted
    /// the folders of the five entries the Debug build had just installed. So an id
    /// leaves <see cref="BuiltIn"/> by moving here, and <c>CatalogTests</c> holds every id
    /// the catalog has shipped so that it fails until one does. A retired id is never
    /// given to a new entry: every build that retired it would delete that entry's files.
    /// </para>
    /// </summary>
    public static IReadOnlyList<string> Retired { get; } = new[]
    {
        // Re-pointed at another file on 2026-09-05; gemma-4-12b-iq3xxs was the 12B between
        // the Q4_K_XL and today's UD-IQ2_M.
        "gemma-4-e4b-q4kxl",
        "gemma-4-12b-q4kxl",
        "gemma-4-12b-iq3xxs",
        "qwen3.5-9b-q4kxl",
        "qwen3.8-27b-iq2xxs",
        // Withdrawn on 2026-09-07.
        "gemma-4-e4b-q8",
        "gemma-4-26b-a4b-iq2xxs",
        "gpt-oss-20b-q8",
        "qwen3.6-35b-a3b-iq1m",
        "qwen3.8-27b-iq1s",
        "qwen-image-edit-2511-q2k",
        // Bonsai 1, replaced by Bonsai 2 27B on 2026-09-28.
        "bonsai-8b-q1-0",
        "bonsai-27b-q1-0",
    };

    /// <summary>Whether <paramref name="id"/> is one of the <see cref="Retired"/> ids.</summary>
    public static bool IsRetired(string id) => Retired.Contains(id, StringComparer.OrdinalIgnoreCase);

    public static CatalogModel? Find(string id) =>
        BuiltIn.FirstOrDefault(m => string.Equals(m.Id, id, StringComparison.OrdinalIgnoreCase));

    /// <summary>The entries a device with <paramref name="physicalMemoryGB"/> of RAM is offered.</summary>
    public static IReadOnlyList<CatalogModel> ForDevice(int physicalMemoryGB) =>
        BuiltIn.Where(m => m.MinDeviceMemoryGB <= physicalMemoryGB).ToList();

    /// <summary>
    /// Rounds a reported physical-memory figure (iOS reports slightly under the marketing
    /// number, e.g. 11.6 GB for a "12 GB" phone) to the marketing tier used by
    /// <see cref="CatalogModel.MinDeviceMemoryGB"/>.
    /// </summary>
    public static int DeviceMemoryTier(long physicalMemoryBytes)
    {
        double gb = physicalMemoryBytes / 1_000_000_000.0;
        int[] tiers = { 4, 6, 8, 12, 16, 24, 32, 48, 64, 128 };
        int best = tiers[0];
        foreach (int t in tiers)
        {
            if (gb + 0.75 >= t)
                best = t;
        }
        return best;
    }
}
