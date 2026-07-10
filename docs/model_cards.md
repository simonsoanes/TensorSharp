# Model Architecture Cards

[English](model_cards.md) | [中文](model_cards_zh-cn.md)

> This file has been split. Each model now has a dedicated card under
> [`docs/models/`](models/README.md), and there is a per-model 中文 version
> alongside each English card.

Use the index below to jump straight into a specific architecture, or read
[`docs/models/README.md`](models/README.md) for the full implementation
matrix and feature comparison.

Every current card begins with checked Hugging Face download pointers, exact
example filenames, and copy/paste `TensorSharp.Cli` and `TensorSharp.Server`
commands. The verified quick-start lane (see the repository
[README](../README.md#quick-start)) is TensorSharp's Gemma 4 E4B Q8_0 native
GGML family/path tier: use `gemma-4-E4B-it-Q8_0.gguf` from the recommended
public
[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)
with `ggml_cuda`, `ggml_metal`, or `ggml_vulkan`; see the
[Gemma 4 card](models/gemma4.md#verified-gemma-4-e4b-native-ggml-fast-path).
The matching `mmproj` is optional for text and required for image, video, or
audio input; no particular public-file checksum is asserted as the benchmark
input.

| Architecture | English card | 中文卡片 |
|---|---|---|
| Gemma 3 | [models/gemma3.md](models/gemma3.md) | [models/gemma3_zh-cn.md](models/gemma3_zh-cn.md) |
| Gemma 4 | [models/gemma4.md](models/gemma4.md) | [models/gemma4_zh-cn.md](models/gemma4_zh-cn.md) |
| DiffusionGemma | [models/diffusiongemma.md](models/diffusiongemma.md) | [models/diffusiongemma_zh-cn.md](models/diffusiongemma_zh-cn.md) |
| Qwen 3 | [models/qwen3.md](models/qwen3.md) | [models/qwen3_zh-cn.md](models/qwen3_zh-cn.md) |
| Qwen 3.5 / 3.6 family | [models/qwen35.md](models/qwen35.md) | [models/qwen35_zh-cn.md](models/qwen35_zh-cn.md) |
| GPT OSS | [models/gptoss.md](models/gptoss.md) | [models/gptoss_zh-cn.md](models/gptoss_zh-cn.md) |
| Nemotron-H | [models/nemotron.md](models/nemotron.md) | [models/nemotron_zh-cn.md](models/nemotron_zh-cn.md) |
| Mistral 3 | [models/mistral3.md](models/mistral3.md) | [models/mistral3_zh-cn.md](models/mistral3_zh-cn.md) |
| Qwen-Image-Edit | [models/qwenimage.md](models/qwenimage.md) | [models/qwenimage_zh-cn.md](models/qwenimage_zh-cn.md) |

Each card walks an engineer or researcher from "I have never heard of this
model" to "I can explain the forward graph and reproduce the inference path
in TensorSharp", covering:

1. Checked downloads and runnable CLI/server commands
2. Origin and intent (provider, GGUF arch keys, modalities, thinking, tools)
3. Model architecture (high-level block diagram, layer counts, per-layer heterogeneity)
4. Forward graph (the exact ordered list of ops, per-token decode and multi-token prefill)
5. Components in detail (attention, FFN/SSM, routing, normalization, RoPE flavor, vision/audio encoder)
6. Parameters and settings (GGUF metadata keys, weight tensor naming, dtype expectations)
7. TensorSharp implementation walkthrough
8. Prefill optimization
9. Decode optimization
10. Memory and KV cache strategy
11. Multimodal pipeline (when applicable)
12. Output parser and chat template
13. Optimization opportunities

When adding a new architecture, follow the checklist in
[`docs/models/README.md`](models/README.md#adding-a-new-architecture).
