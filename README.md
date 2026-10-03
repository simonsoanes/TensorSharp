# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

<p align="center"><a href="https://buymeacoffee.com/zhongkaifu"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>TensorSharp/TensorAgent is free. If you like it, a coffee keeps the work on it going.</sub></p>

**Native .NET AI inference engine for GGUF models** — text, reasoning, multimodal input, embeddings, image generation and editing, and video with audio. Run it from the CLI, browser chat, Ollama/OpenAI-compatible APIs, or [TensorAgent](TensorAgent/README.md), the local app for iPhone, iPad, Mac and Windows. The .NET runtime offers managed CPU and native accelerator backends; published comparisons use identical GGUF files and hardware. The optional `TensorSharp.AgentHost` layer adds Agent Skills, a bounded, in-process model-to-tool loop for file and shell work, and bounded automatic subagent delegation.

## Highlights

- **Local, native .NET inference.** Run GGUF text and multimodal models from the CLI, browser UI, or Ollama/OpenAI-compatible APIs.
- **Broad model and media support.** Current source covers modern text models, DiffusionGemma text diffusion, vision/audio input, PDF, [Qwen-Image-2.1 image generation and editing](docs/models/qwenimage21.md) with masks and LoRA plug-ins, MiniMax-H3 video with native 32 kHz stereo audio, and Wan 2.1/2.2 video. See the [model cards](docs/models/README.md).
- **Text and code embeddings.** GGUF BERT/XLM-R encoders with OpenAI/Ollama batch embedding APIs for Snowflake Arctic Embed and MiniLM; see the [embedding guide](docs/embeddings.md).
- **Measured performance.** TensorSharp is benchmarked against `llama.cpp` on identical models and hardware. Results are specific to the measured model, backend, and workload. See [Benchmarks](docs/benchmarks.md).
- **Agentic work.** `TensorSharp.AgentHost` adds bounded Agent Skills, code tools, and [automatic subagent delegation](docs/multi_agent.md) with independent contexts, private workspaces, dependency scheduling, and read-only defaults.
- **TensorAgent for phones and desktops.** One app for local chat, multimodal input, code and document work, image generation/editing, and short video with audio. Its twelve-model catalog is gated by system RAM tier; image/video generation models need desktop memory tiers. Qwen3.8 Flash Next offers experimental UD-IQ1_M with optional vision from 32 GB system RAM alongside text-only UD-Q2_K_XL from 48 GB. The interface supports English, Simplified and Traditional Chinese, Japanese, Korean, Spanish, French and German. See [TensorAgent](TensorAgent/README.md) for source builds, platform differences and measured coverage.
- **Production-friendly building blocks.** Continuous batching and the paged, Radix prefix-shared KV cache are on by default; speculative decoding, tensor parallelism, and configurable security boundaries are available when you need them. See [Features](FEATURES.md), [Usage](USAGE.md), and the [current project status](docs/PROJECT_STATUS.md).

## Supported model families at a glance

- **Text, reasoning, and multimodal LLMs:** [DeepSeek V4 Flash](docs/models/deepseek4.md) / [V4.1 Flash](docs/models/deepseek41.md), [GLM 5.x](docs/models/glm.md), [Gemma 4](docs/models/gemma4.md), [Qwen 3.5 / 3.6 / 3.8 27B](docs/models/qwen35.md), [Qwen 3.8 Flash Next](docs/models/qwen38-flash-next.md), [Bonsai2](docs/models/bonsai2.md) (Qwen family), [GPT OSS](docs/models/gptoss.md), [Nemotron-H](docs/models/nemotron.md), [Mistral 3](docs/models/mistral3.md), [Hunyuan Dense](docs/models/hunyuan-dense.md), and [Muse-Glimmer](docs/models/muse-glimmer.md).
- **Text diffusion:** [DiffusionGemma](docs/models/diffusiongemma.md), including [Jev typed decision inference](docs/models/jev.md) at `/v1/systemone`, over text, images, uploaded documents, sampled video frames and audio transcripts (configured ASR companion).
- **Image generation/editing and video generation:** [Qwen-Image-2.1](docs/models/qwenimage21.md), [MiniMax-H3 (video + stereo audio)](docs/models/minimax-h3.md), and [Wan 2.1 / 2.2](docs/models/wan.md).
- **Text and code embeddings:** BERT / XLM-R encoders — [Snowflake Arctic Embed L v2.0 and all-MiniLM-L6-v2](docs/embeddings.md).

Backend, modality, feature support, and validation coverage vary by model. See the [supported models](docs/supported_models.md) tables, the [model cards](docs/models/README.md), and the [embedding guide](docs/embeddings.md) for details.

Recent source additions include Qwen-Image-2.1 masked edits with exact protected pixels and optional processing of the selected region, twelve TensorAgent LoRA plug-ins for speed, style and editing, and Qwen3.8 Flash Next on a 48 GB Mac using SSD-backed weights. Multi-GPU `--layer-split` and `--tp` are separate controls; support and performance depend on the architecture and quantization. These source features may be ahead of the published CLI/server packages; TensorAgent currently requires a source build.

## Learn with the books

| Qwen inference and agentic runtimes | Gemma 4 and multimodal inference |
|---|---|
| <a href="https://www.amazon.com/dp/B0HJQ4VQ31"><img src="website/assets/building-llm-inference-engines-cover.jpg" alt="Building LLM Inference Engines and Agentic Runtimes from Scratch: Qwen Dense and MoE Models with TensorSharp and TensorAgent" width="190"></a> | <a href="https://www.amazon.com/dp/B0H9P44QZZ"><img src="website/assets/from-tensors-to-tokens-cover.jpg" alt="From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B" width="190"></a> |
| **[Building LLM Inference Engines and Agentic Runtimes from Scratch: Qwen Dense and MoE Models with TensorSharp and TensorAgent](https://www.amazon.com/dp/B0HJQ4VQ31)** | **[From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B](https://www.amazon.com/dp/B0H9P44QZZ)** |
| Build Qwen dense/MoE inference and controlled agent workflows in C#. Follow tensors, tokenization, attention, expert routing, quantization, and caching through GPU acceleration, multimodal execution, tools, skills, sandboxed code, and desktop/mobile deployment with TensorSharp and TensorAgent. | Build a multimodal inference engine in C#/.NET with Gemma 4 E4B, from tensors, GGUF model loading, quantization, and tokenization to text, image, video, and audio execution. Connect correctness checks and serving optimizations to the running TensorSharp code. |
| **[Buy on Amazon](https://www.amazon.com/dp/B0HJQ4VQ31)** | **[Buy on Amazon](https://www.amazon.com/dp/B0H9P44QZZ)** |

**[Explore both books and their repository reading paths](docs/BOOK.md)**

## Quick Start

Prefer a prebuilt application? The [Releases page](https://github.com/zhongkaifu/TensorSharp/releases) provides self-contained CLI and Server archives for Windows x64 (CPU/CUDA), Linux x64 (CPU/CUDA), and macOS arm64.

To build from source you need the full **.NET 10 SDK** ([how to install it](docs/getting_started.md#install-and-first-run)), `git`, `curl`, [CMake](https://cmake.org/download/) 3.20+, and the toolchain for your GPU. Then run the verified [Gemma 4 E4B](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf) model (7.48 GiB). On Windows with an NVIDIA GPU (PowerShell):

```powershell
git clone https://github.com/zhongkaifu/TensorSharp.git; Set-Location TensorSharp
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L --fail "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models\gemma-4-E4B-it-Q8_0.gguf
'Answer in one short sentence: what is TensorSharp?' | Set-Content prompt.txt
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA = 'ON'
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models\gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 128 --backend ggml_cuda
```

On other machines, change the backend (see [Pick a backend](#pick-a-backend)):

- **macOS (Apple Silicon):** drop the CUDA environment variable and use `--backend ggml_metal`.
- **Linux + NVIDIA:** prefix the `dotnet run` with `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` and use `--backend ggml_cuda`.
- **AMD / Intel / NVIDIA Vulkan:** set `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON` and use `--backend ggml_vulkan`.

Host the same model as a server: a browser chat at <http://localhost:5000> plus Ollama- and OpenAI-compatible APIs.

```bash
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 512
```

> The server listens on `0.0.0.0:5000` with no built-in authentication or TLS; keep it behind a firewall or an authenticated HTTPS reverse proxy.

### Pick a backend

| Your hardware | Backend |
|---|---|
| Apple Silicon (Mac) | `--backend ggml_metal` |
| Windows / Linux + NVIDIA GPU | `--backend ggml_cuda` |
| Windows / Linux + AMD / Intel / NVIDIA GPU | `--backend ggml_vulkan` |
| No GPU | `--backend ggml_cpu` (native kernels), or `--backend cpu` (pure C#, no native dependencies) |

The [Getting started guide](docs/getting_started.md) has the rest: installing the SDK on each platform, multi-GPU and multi-node runs, NVIDIA DGX Spark, multimodal input, embeddings, and making it fast. Every option is in the [CLI](USAGE.md#console-application) and [Server](USAGE.md#web-application) references, and both programs print them with `--help`.

`dotnet build TensorSharp.slnx` also builds TensorAgent's available desktop heads and the iOS simulator head on Apple Silicon when the selected SDK has the required MAUI workloads and staged native/Python files. Missing prerequisites skip the affected app head with a warning; see [TensorAgent build instructions](TensorAgent/README.md#build-and-run).

## See it in action

One engine, four ways to use it, each an unedited capture of a real run.

<table>
  <tr>
    <td align="center" width="50%"><img src="website/assets/screenshots/tensorsharp-cli.png" alt="TensorSharp.Cli in a terminal: an interactive chat with Gemma 4 E4B that reads this README and answers questions about it" width="250"><br><b>TensorSharp.Cli</b><br>Models in your terminal</td>
    <td align="center" width="50%"><img src="website/assets/screenshots/tensorsharp-webui.png" alt="The TensorSharp Web UI: Qwen3.8 27B compared two mortgages by writing and running a Python script" width="400"><br><b>TensorSharp.Server.Host</b><br>Web UI chat and Ollama/OpenAI-compatible APIs</td>
  </tr>
  <tr>
    <td align="center"><img src="website/assets/screenshots/tensoragent-iphone.png" alt="TensorAgent on an iPhone: Gemma 4 E2B scaled a recipe by running a Python script on the phone" width="140"><br><b>TensorAgent on iPhone</b><br>A private agent that runs the model on the phone</td>
    <td align="center"><img src="website/assets/screenshots/tensoragent-mac.png" alt="TensorAgent on a Mac: a saved Qwen-Image 2.1 edit changes the TensorSharp banner background to a starry blue night sky" width="400"><br><b>TensorAgent on the desktop</b><br>Edit images with Qwen-Image 2.1, alongside text and agentic work</td>
  </tr>
</table>

What each run shows, step by step: [Screenshots](docs/showcase.md).

## Benchmarks

TensorSharp and `llama.cpp` run identical GGUF files on the same NVIDIA RTX 3080 Laptop GPU (16 GB), each on its GGML CUDA and Vulkan builds. Each number is TensorSharp's speedup over llama.cpp on the same backend (geomean, single-stream, greedy, MTP off); above 1.0× means TensorSharp is faster.

| Model | Backend | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it (Q8_0, dense multimodal) | CUDA | 1.02× | **1.28×** | **1.27×** |
| Gemma 4 E4B it (Q8_0, dense multimodal) | Vulkan | 1.00× | 1.05× | 1.03× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | CUDA | 1.04× | **1.17×** | **1.16×** |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | Vulkan | **1.21×** | 1.04× | 1.03× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | CUDA | 0.98× | **1.28×** | **1.27×** |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | Vulkan | 0.87× | 1.04× | 1.03× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | CUDA | **1.07×** | 0.96× | 0.95× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | Vulkan | 1.02× | 0.85× | 0.84× |

What these numbers mean, how to rerun them, and the head-to-heads of models too large for this GPU: [Benchmarks](docs/benchmarks.md).

## Documentation

New here? The sections above are all you need to get running. Everything else is detailed reference:

| Doc | What's inside |
|---|---|
| [TensorSharp and TensorAgent book guide](docs/BOOK.md) | Building LLM Inference Engines and Agentic Runtimes from Scratch, plus From Tensors to Tokens: introductions, Amazon links, and repository reading paths |
| [Getting started](docs/getting_started.md) | The full first-run guide: the .NET SDK on each platform, every backend, multi-GPU and multi-node runs, NVIDIA DGX Spark, embeddings, choosing a backend, and making it fast |
| [Supported models](docs/supported_models.md) | Implemented model families and their validation scope: example GGUFs, modalities, thinking, tools, and speculative decoding |
| [Benchmarks](docs/benchmarks.md) | TensorSharp against llama.cpp on the same GPU and files, and the head-to-heads of larger models |
| [Screenshots](docs/showcase.md) | The CLI, the Web UI, and TensorAgent on iPhone and Mac at work, with what each run did |
| [Model Downloads](MODEL_DOWNLOADS.md) | Per-model `huggingface-cli` download + run quick reference (quant tiers, projectors, companions) |
| [Usage](USAGE.md) | Full CLI reference (options, interactive REPL, JSONL batch), server hosting, logging, HTTP API examples, backends, and the env-var matrix |
| [Features](FEATURES.md) | Deep dives on continuous batching, speculative decoding, tool calling, thinking mode, multimodal, MoE, KV codecs, and more |
| [Configuration files](config/README.md) | Put options in a reusable JSON file with `${variables}` and auto-downloading models |
| [Development](DEVELOPMENT.md) | Prerequisites, building the native GGML/MLX libraries, repository layout, package boundaries, internal architecture, and the test harness |
| [Per-model architecture cards](docs/models/README.md) | End-to-end docs of each architecture (forward graph, components, parameters, prefill/decode optimizations) |
| [Paged attention & continuous batching](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | The vLLM-style paged KV cache, prefix sharing, and iteration-level scheduler |
| [Agent Skills & agentic work](docs/agent_skills.md) | The `SKILL.md` format, progressive disclosure and its budget, the in-process tool loop, sandboxed code execution, workspaces and artifacts, the path/ZIP/exec security model, and the HTTP + C# surfaces |
| [Multiple agents](docs/multi_agent.md) | Automatic task delegation, private child workspaces, dependency scheduling, permission limits, server controls, and reproducible evaluation |
| [Browser automation skill (Playwright)](docs/playwright_agent.md) | Running the bundled `playwright` skill, which drives a browser through `@playwright/cli` via `skills_run`: the flags it needs, the macOS Chromium-sandbox config, account handoff, and TensorAgent desktop hosting (not iOS) |
| [Speculative decoding](docs/speculative_decoding.md) | The three-layer design (model adapter / algorithm / speculator weights), the shipped `auto` / `draft-head` / `block` / `ngram` algorithms, and what to write to add a new one |
| [Environment variable feature matrix](docs/env_var_feature_matrix.md) | Which high-impact runtime flags affect which models, backends, and prompt types |
| [Engine comparison report](docs/engine_comparison_report.md) | Full per-scenario TensorSharp vs llama.cpp tables |
| [ggml_metal vs llama.cpp](docs/perf/metal-vs-llama-cpp.md) | Head-to-head prefill/decode on Apple Silicon, the four graph-construction gaps it found, and what each was worth |
| [Test/benchmark matrix runner](TensorSharp.TestMatrix/README.md) | Sweep model × backend × feature × env-var cells and generate regression reports |
| [Server API examples](TensorSharp.Server.Host/API_EXAMPLES.md) | Complete curl and Python examples for the server surface |

## Current Status

Actively developed, and the source tree runs ahead of the published packages.

| Area | Where it stands |
|---|---|
| Models | A dozen autoregressive families plus text diffusion, image generation and editing, and video with audio. See [Supported models](docs/supported_models.md). |
| Inference hosts | CLI, Web UI, Ollama- and OpenAI-compatible APIs, and the TensorAgent app for iPhone, iPad, Mac and Windows. TensorAgent is source-only. |
| Backends | Pure C# CPU, direct CUDA/cuBLAS, MLX Metal, and GGML CPU/Metal/CUDA/Vulkan, with per-architecture exceptions. |
| Serving features | Continuous batching with a shared prefix cache, speculative decoding, tensor parallelism, structured output, and tool calling. |
| Agentic work | Agent Skills, sandboxed file and shell tools, and bounded sub-agents. See [Agent Skills](docs/agent_skills.md) and [Multiple agents](docs/multi_agent.md). |
| TensorAgent | Twelve catalog entries, saved chats and artifacts, masked image edits and LoRA choices, eight interface languages, and persisted text-turn statistics. Media generation has been measured on a Mac; iOS media generation and Windows image/audio/video generation remain unverified. |

Per-area detail (which architecture runs on which backend, which features each family supports, and the known limits) is in the [status matrix](docs/PROJECT_STATUS.md#status-matrix).

## Author

Zhongkai Fu

## License

See [LICENSE](LICENSE) for details.
