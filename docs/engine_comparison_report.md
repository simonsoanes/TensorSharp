# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on the selected compute backends (ggml_cuda / ggml_vulkan / ggml_metal / ggml_cpu / cpu / ...).

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

## Software / hardware

| Component | Version / detail |
|---|---|
| TensorSharp | git `0789c8d`, .NET 10.0.204 (backends: ggml_cuda / ggml_vulkan / ggml_metal / cuda / mlx / ggml_cpu / cpu) |
| llama.cpp | `C:\Works\llama.cpp\build-cuda\bin\Release\llama-server.exe` |
| vLLM | endpoint `http://127.0.0.1:8000` (connect-only) |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16384 MiB |


## Methodology

- Each `(engine, backend, model)` group launches its server once; all of that group's scenarios run against it, so per-scenario timings exclude model-load cost.
- Metrics come from the **streamed** response: `ttft` is time-to-first-token (prefill latency proxy), `prefill_tps = prompt_tokens / ttft`, and `decode_tps = (completion_tokens - 1) / (t_last - t_first)`.
- DiffusionGemma denoises whole blocks (no token stream), so it is run non-streaming and its `decode_tps` is wall-clock tokens/second.
- Greedy sampling (`temperature=0`); one warmup request per server is discarded.
- The headline per-engine tables are the **single-stream, MTP-off** baseline. MTP on/off and parallel-request scaling are reported in their own sections below.

## Performance ratio — TensorSharp vs reference engines

Geomean of TensorSharp's per-scenario speedup over each reference engine on the **same backend**, across every scenario both engines ran (single-stream, MTP-off). A value **> 1.0× means TensorSharp is faster** (for decode / prefill throughput) or lower-latency (for TTFT); `—` = no overlapping cells. Per-scenario ratios are in each model's section below.

| Model | Comparison | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it (Q8_0, dense multimodal) | vs llama.cpp · Vulkan | 0.94× | 0.99× | 0.98× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | vs llama.cpp · Vulkan | 1.18× | 0.95× | 0.94× |

## Gemma 4 E4B it (Q8_0, dense multimodal)  (`gemma4-e4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 44.4 | 47.0 |
| text_long | 42.9 | 46.4 |
| multi_turn | 43.6 | 46.4 |
| function_call | 43.7 | 46.6 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1755.6 | 1731.6 |
| text_long | 1290.4 | 1763.9 |
| multi_turn | 1811.4 | 1538.7 |
| function_call | 1851.0 | 1668.6 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1125.0 | 1125.0 |
| text_long | 2438.0 | 1766.0 |
| multi_turn | 1156.0 | 1344.0 |
| function_call | 1094.0 | 1219.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 0.94× |
| text_long | 0.92× |
| multi_turn | 0.94× |
| function_call | 0.94× |

_Prefill throughput_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 1.01× |
| text_long | 0.73× |
| multi_turn | 1.18× |
| function_call | 1.11× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 1.00× |
| text_long | 0.72× |
| multi_turn | 1.16× |
| function_call | 1.11× |

## Gemma 4 12B it (QAT UD-Q4_K_XL, dense)  (`gemma4-12b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 32.7 | 33.4 |
| text_long | 31.8 | 32.0 |
| multi_turn | 32.7 | 32.5 |
| function_call | 65.7 | 33.5 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 820.9 | 769.7 |
| text_long | 625.3 | 758.3 |
| multi_turn | 657.0 | 652.3 |
| function_call | 641.1 | 700.3 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 2406.0 | 2532.0 |
| text_long | 5031.0 | 4109.0 |
| multi_turn | 3187.0 | 3172.0 |
| function_call | 3235.0 | 2906.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 0.98× |
| text_long | 0.99× |
| multi_turn | 1.01× |
| function_call | 1.96× |

_Prefill throughput_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 1.07× |
| text_long | 0.82× |
| multi_turn | 1.01× |
| function_call | 0.92× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 1.05× |
| text_long | 0.82× |
| multi_turn | 1.00× |
| function_call | 0.90× |

## Image editing (stable-diffusion)

Same input image, prompt, resolution, step count, cfg and seed for every engine. Timings are each engine's **own pipeline timers** (TensorSharp's `[pipe-timing]` phases + server `elapsedSeconds`; sd.cpp's phase logs + `generate_image` total), so weight-file loading and HTTP/process overhead are excluded on both sides. `total (warm)` is the steady-state request on an already-running server; `first request (cold)` additionally pays TensorSharp's per-request DiT rebuild + graph capture on a fresh server (a CLI engine has no such distinction). Lower is better.

### Qwen-Image-Edit 2511 (Q2_K DiT + Lightning 4-step LoRA) — `image_edit` on CUDA, 544x1184, 4 steps

| Engine | total (warm) | per step | sampling | text encode | VAE encode | VAE decode | first request (cold) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorSharp | 40.44 s | 7.57 s | 30.27 s | 7.45 s | 0.54 s | 1.51 s | 54.11 s |
| stable-diffusion.cpp | 48.16 s | 9.43 s | 37.73 s | 4.47 s | 1.92 s | 2.57 s | — |

**TensorSharp vs stable-diffusion.cpp** (ratio = stable-diffusion.cpp time / TensorSharp time; > 1.0× = TensorSharp faster): total (warm) **1.19×**, per step **1.25×**, sampling **1.25×**, text encode **0.60×**, VAE encode **3.56×**, VAE decode **1.70×**


## MTP / NextN speculative decoding (on vs off)

Single-stream decode tok/s with MTP/NextN speculative decoding off vs on (TensorSharp only). Speedup `< 1.0×` means speculation cost more than it saved for that cell — expected when the fused full-model decode path is already the fast path.

_No MTP on/off pairs were run (use `--mtp off,on`)._

## Parallel-request scaling (concurrency)

`decode/req` is the mean per-request decode tok/s; `aggregate` is the system-wide decode throughput (total generated tokens / the wall window during which any sequence was decoding) when N identical requests are fired at one server at once.

_No parallel-request cells were run (use `--concurrency 1,4,8`)._

## Function-calling correctness

| Engine · Backend · Model | tool_call emitted |
|---|:---:|
| llamacpp · ggml_vulkan · gemma4-12b | yes |
| llamacpp · ggml_vulkan · gemma4-e4b | yes |
| tensorsharp · ggml_vulkan · gemma4-12b | yes |
| tensorsharp · ggml_vulkan · gemma4-e4b | yes |
