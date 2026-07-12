# Qwen-Image-Edit

[← back to model index](README.md)

## Status snapshot

| Field | Status |
|---|---|
| GGUF architecture key | `qwen_image` (the MMDiT diffusion transformer) |
| Source class | [`QwenImageModel`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) (+ internal [`QwenImagePipeline`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs)) |
| Task | Image editing — prompt + input image(s) → edited image |
| Modalities | Image(s) in + text in → image out (multi-image composition à la `QwenImageEditPlusPipeline`: "Picture 1", "Picture 2", ...) |
| Thinking / tools | Not applicable |
| Generation mode | FlowMatch-Euler diffusion denoise (true-CFG), **not** autoregressive token decode |
| CLI support | `TensorSharp.Cli` detects `QwenImageModel` and runs image-edit mode (`--image` repeatable for multi-image edits, `--prompt`, `--output`, `--cfg`, `--diffusion-steps`, `--diffusion-seed`, `--width`/`--height` to force the output size, `--offload-cpu` to always stream DiT weights from RAM) |
| Server support | Web UI image-edit flow: `POST /api/image-edit` and `POST /api/image-edit/stream` (SSE with live denoising previews); both accept multiple images (multipart `image` parts / JSON `imagePaths[]`). Startup `--width`/`--height` set a fixed output size for every edit (per-request sizes still override). |
| Continuous batching | None — image edits are serialized (the diffusion nets are not thread-safe); concurrent requests run one at a time |

## Downloads

Image editing needs a **four-component set**. The DiT GGUF is the `--model`; the
companions auto-resolve from the same directory, or can be pointed at explicitly
with `--qwen-image-vae` / `--qwen-image-vl` / `--qwen-image-mmproj` /
`--qwen-image-lora` (CLI and server) — equivalently the `TS_QWEN_IMAGE_VAE` /
`TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ` / `TS_QWEN_IMAGE_LORA` env vars:

| Component | HF repo | File |
|---|---|---|
| MMDiT DiT (the `--model` GGUF) | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) | `qwen-image-edit-2511-Q4_K_M.gguf` (13.245 GB); `qwen-image-edit-2511-Q2_K.gguf` (7.468 GB) is the smallest published option and fits 16 GB VRAM |
| Qwen-Image VAE | [QuantStack/Qwen-Image-Edit-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) | `VAE/Qwen_Image-VAE.safetensors` (0.254 GB) |
| Qwen2.5-VL-7B text encoder | [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) | **`Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (4.683 GB) — strongly recommended.** The text/vision encoders are **freed before the denoise loop**, so they never compete with the DiT for VRAM: a bigger TE quant costs *nothing* at denoise time but drives the whole edit's fidelity. A ~2-bit `…-UD-IQ2_XXS.gguf` (2.398 GB) markedly softens faces and lowers overall quality — avoid it even on low-VRAM cards. Auto-resolution now prefers the **highest-quality** VL GGUF present. |
| Vision projector (optional — image-grounded conditioning) | same repo | `mmproj-BF16.gguf` (1.354 GB) |
| Lightning LoRA (optional — 4/8-step editing) | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` (0.850 GB) |

All conversion cards above identify official Qwen base models and declare
Apache-2.0. The DiT and Lightning files derive from
[Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).

```bash
python -m pip install -U huggingface_hub
hf download unsloth/Qwen-Image-Edit-2511-GGUF qwen-image-edit-2511-Q4_K_M.gguf --local-dir models
hf download QuantStack/Qwen-Image-Edit-GGUF VAE/Qwen_Image-VAE.safetensors --local-dir models
hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --local-dir models
hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF mmproj-BF16.gguf --local-dir models
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --local-dir models
```

`hf download` keeps the repo's `VAE/` subfolder — move `Qwen_Image-VAE.safetensors`
up next to the DiT GGUF (auto-resolution scans the DiT's directory for that exact
file name), or pass `--qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors`.

CLI edit (image-edit mode engages automatically when `--model` is a `qwen_image`
GGUF; `--image` is required and the edit instruction goes in `--prompt`):

```bash
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/qwen-image-edit-2511-Q4_K_M.gguf --backend ggml_cuda \
  --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors \
  --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --image input.png --prompt "Replace the sky with a golden sunset" \
  --output edited.png --cfg 2.5 --diffusion-steps 30
```

(Leave `--cfg` / `--diffusion-steps` off for auto: 30 steps / cfg 2.5, or the
Lightning LoRA's trained step count / cfg 1.0 when one is loaded via
`--qwen-image-lora`.)

**Multi-image editing** — repeat `--image` to compose several inputs; the first
image drives the output geometry and each image becomes a "Picture N" reference
(in listed order) that the prompt can point at:

```bash
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/qwen-image-edit-2511-Q4_K_M.gguf --backend ggml_cuda \
  --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors \
  --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --image model-photo.heic --image dress.png \
  --prompt "请为模特换上图中的衣服" --output edited.png
```

Server (the Web UI at `http://localhost:5000/index.html` routes image + prompt to
`POST /api/image-edit/stream` — attach an image and type the edit instruction):

```bash
dotnet run --project TensorSharp.Server -c Release -- \
  --model models/qwen-image-edit-2511-Q4_K_M.gguf --backend ggml_cuda \
  --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors \
  --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```

## 1. Origin and intent

Qwen-Image-Edit is an instruction-driven image editor. Unlike the autoregressive
LLMs, the loaded `qwen_image` GGUF is **only** the MMDiT (multimodal diffusion
transformer); image editing additionally needs two networks the DiT GGUF does
not contain:

- the **Qwen-Image VAE** — image ↔ 16-channel latent (8× spatial downsample), and
- the **Qwen2.5-VL-7B text encoder** — prompt → 3584-dim conditioning, with an
  optional `mmproj` vision tower for image-grounded conditioning.

`ModelBase.Create()` routes `general.architecture = qwen_image` to
`QwenImageModel`. The companion GGUFs are resolved from the DiT GGUF's directory,
or via the `TS_QWEN_IMAGE_VAE` / `TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ`
environment variables (the VAE may be the original `.safetensors`, loaded
directly with BF16→F32 upcast). `QwenImageModel` is not an
`IModelArchitecture` text generator: `Forward()` throws and editing is driven
through [`EditImage()`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs).

## 2. Pipeline (the denoise loop)

[`QwenImagePipeline.Edit`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs)
orchestrates the full edit:

```text
input image(s)   [first image drives the output geometry]
  -> preprocess (aspect-preserving; dims multiple of 16; area clamped to VRAM budget, up to native ~1 MP;
     extra reference images keep their own aspect at min(native, output area))
  -> VAE encode each -> normalize (diffusers latents mean/std) -> pack [refSeq_i, 64]
  -> text conditioning: Qwen2.5-VL encode(prompt [+ "Picture N" vision-grounded tokens per image])
       (+ negative prompt when CfgScale > 1, for true-CFG; vision embeds cached across both passes)
  -> free text + vision encoders (reclaim VRAM for the DiT)
  -> noise latents (seeded Gaussian) packed to [genSeq, 64]
  -> concatenate [generated | ref 1 | ref 2 | ...] tokens (modulateIndex marks all ref tokens;
     DiT RoPE gives each stream its own frame index 0,1,2,... with centered spatial grids)
  -> FlowMatch-Euler scheduler (timestep == sigma), First-Block-Cache reset
  -> for each step:
       DiT velocity prediction (cond) [+ neg -> true-CFG combine + per-row renorm]
       scheduler.Step(latents, v, step)
       optional decoded RGB preview (downsampled) via OnStep callback
  -> unpack -> denormalize -> VAE decode -> output image
```

True-CFG combines the conditional and unconditional velocity per packed token
row (`comb = neg + scale·(cond − neg)`) and renormalizes each row back to the
conditional norm. With `CfgScale <= 1` the negative pass is skipped (one DiT
forward per step instead of two).

## 3. MMDiT architecture constants

The `qwen_image` GGUF carries no hyperparameters; they are pinned in
[`QwenImageModel`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs):

| Constant | Value |
|---|---|
| Hidden size | 3072 |
| Layers (double-stream blocks) | 60 |
| Attention heads | 24 (head dim 128) |
| DiT input channels | 64 (16 latent channels × 2×2 patch) |
| Text conditioning dim | 3584 (Qwen2.5-VL hidden) |
| VAE latent channels | 16 |
| VAE scale factor | 8 |
| RoPE axes dims | {16, 56, 56} (sum = head dim 128) |
| Eps | 1e-6 |

Each MMDiT block jointly attends image and text tokens with multimodal RoPE;
timestep/text modulation is applied via the per-token `modulateIndex`
(0 = generated half, 1 = reference half). The reference latent is concatenated
into the token sequence so the edit is grounded on the original image.

## 4. TensorSharp implementation

| Component | File |
|---|---|
| Model entry / companion resolution | [`QwenImageModel.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) |
| Denoise orchestration | [`QwenImagePipeline.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs) |
| Generation params | [`QwenImageParams.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageParams.cs) |
| MMDiT (graph, cache, CFG-batch, native, whole-model) | `QwenImageDiT*.cs` |
| FlowMatch-Euler scheduler | [`QwenImageScheduler.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageScheduler.cs) |
| VAE (+ reference math) | [`QwenImageVae.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageVae.cs) |
| Qwen2.5-VL text encoder | [`QwenImageTextEncoder.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageTextEncoder.cs) |
| Vision tower (grounding) | [`QwenImageVisionEncoder.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageVisionEncoder.cs) |
| Image decode / encode (PNG) | [`ImageIO.cs`](../../TensorSharp.Models/Models/QwenImage/ImageIO.cs) |
| Fused native kernels | [`ggml_ops_qwen_image.cpp`](../../TensorSharp.GGML.Native/ggml_ops_qwen_image.cpp) |

## 5. Acceleration status

- **CUDA-graph-captured whole-DiT forward** (`TSGgml_QwenImageForward`): the
  weights are made resident once and the whole 60-block forward is captured into
  a single replayable graph with a dedicated `gallocr`. This turned the denoise
  from launch-bound (~40% GPU) into compute-bound (~100% GPU) — the per-forward
  cost dropped ~2.9× and the 8-step denoise fell from ~153 s to ~63 s on the
  measured CUDA box. Toggle off with `TS_QWEN_DIT_WHOLE_CAPTURE=0`.
- **Flash attention by default** on the ggml-cuda path (`TS_QWEN_DIT_FLASH=0`
  forces the explicit-scores path): O(n) attention memory, so higher resolutions
  fit before the VAE decode's im2col scratch becomes the limit.
- **CFG-batching**: both guidance branches run in one launch-amortized fused
  pass when the combined token block fits VRAM (`TS_QWEN_DIT_CFG_BATCH_MAXTOK`);
  larger images fall back to two forwards per step.
- **Lightning distillation LoRA (4/8-step editing)** — `--qwen-image-lora
  <lora.safetensors>` (CLI + server) / `TS_QWEN_IMAGE_LORA` applies a DiT LoRA as
  a **runtime side-path** next to each targeted projection:
  `y = W_quant·x + b + (alpha/rank)·up·(down·x)` in F32, with the quantized base
  weights untouched. (Merging into the weights — the stable-diffusion.cpp
  dequantize→add→requantize path — is only sound for F16/Q8_0 storage: the
  Lightning deltas are ~1e-4 RMS, far below a Q2_K quantization step, so a merge
  replaces them with requantization noise.) The factors are resident like the
  weights, capture-safe, and cost a few % extra per forward + ~1.6 GB VRAM. With
  a lightx2v
  [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
  checkpoint the sampling defaults switch automatically (parsed from the
  filename): its trained step count (4 or 8), cfg 1.0 (single forward per step),
  and the fixed timestep shift 3 the distillation was trained with — cutting the
  default 60 DiT forwards to 4–8. Explicit `steps`/`cfg` still win;
  `TS_QWEN_IMAGE_LORA_SCALE` adjusts the LoRA multiplier (default 1.0). The
  side-path is implemented in the whole-model forward (the default CUDA path)
  **and** the fused per-block kernels — the path CPU offload streams through —
  so Lightning keeps working at offloaded high resolutions; only the 3-call /
  managed fallbacks drop the LoRA (and log/throw accordingly).
- **Whole-step denoise cache (EasyCache)** — **opt-in** port of
  stable-diffusion.cpp's `easycache.hpp` (sd.cpp also ships it strictly opt-in
  via `--cache-mode`). It predicts each step's output change
  from the measured input-latent change (times an empirically tracked
  input→output transformation rate) and, while the accumulated prediction stays
  below a threshold, skips the **entire DiT forward for both CFG branches**,
  reconstructing each branch's velocity as `input + cached(output − input)`.
  The first ~15% and last ~5% of steps always compute. Typically skips 40–55%
  of steps at the default threshold (a 30-step run computes only ~19), which
  measurably softens fine detail (faces) on edit workloads — hence **off by
  default**: quality is the default, speed is the opt-in
  (`TS_QWEN_DIT_CACHE_MODE=easycache`).
  Knobs: `TS_QWEN_DIT_CACHE_MODE` (`off` default /`easycache`/`fbc`/`both`),
  `TS_QWEN_DIT_EASYCACHE_THRESHOLD` (default 0.2 — lower = closer to no-cache),
  `TS_QWEN_DIT_EASYCACHE_START`/`_END` (window fractions, 0.15/0.95),
  `TS_QWEN_DIT_CACHE_DEBUG=1` (per-step decision trace).
- **First-Block-Cache** across denoise steps (reset per generation); selected
  with `TS_QWEN_DIT_CACHE_MODE=fbc`. It always computes
  block 0 per branch and skips blocks 1..59 on low-change steps — strictly less
  saving than the whole-step cache, but its decision uses the actual block-0
  residual rather than a prediction.
- **Fused conditioning-encoder trunks** (`TSGgml_QwenTeTrunk`, default on): the
  Qwen2.5-VL text-encoder LLM (28 layers, GQA, causal) and its vision tower
  (32 layers, MHA, window/full attention as an additive mask) each run their whole
  layer stack as ONE device graph — the per-op path paid ~10 device⇄host round
  trips per layer (M-RoPE, bias adds, SiLU, block-mask softmax as host loops).
  RoPE cos/sin tables are host-precomputed; weights bind resident by GGUF mmap
  pointer. At the default edit: vision 4.7 s → **1.6 s**, LLM prefill 2.8 s →
  **0.8 s** (text-conditioning phase 11.2 s → 2.5 s). Verified vs the numpy
  oracles (`te-verify` cosine 0.9989, `vis-verify` 0.9994 — both ≥ the per-op
  path's own scores). Falls back to the per-op path when the backend can't run
  it; `TS_QWEN_TE_FUSED=0` disables both.
- **VRAM budgeting + CPU offload**: the text + vision encoders are freed after
  conditioning so the DiT (and its attention scratch) own the working set, and the
  persistent reuse `gallocr` is handed back before the final VAE decode. The auto
  output size is **quality-first**: it targets the model's native ~1 MP training
  resolution (what diffusers `QwenImageEditPlusPipeline` and sd.cpp render — the
  old default scaled the area down by step/reference count as a speed budget,
  which rendered a 30-step 2-image edit at ~0.4 MP with visibly soft faces).
  When that resolution does not fit beside the resident DiT weights, **CPU
  offload** (the sd.cpp `--offload-to-cpu` equivalent) engages automatically:
  the resident whole-model graph is bypassed and the 60 blocks run as a few
  **chunked whole-model graphs** (`TS_QWEN_DIT_OFFLOAD_CHUNK` blocks each,
  default 10) whose weights live in per-call input slots of the shared reuse
  `gallocr` — the buffer is re-planned chunk over chunk, so VRAM holds only ONE
  chunk's weights at a time while the AdaLN modulation stays in-graph (no
  host-expanded per-block modulation uploads). A device-copy residency budget
  keeps in VRAM whatever fits *after* the activations are budgeted (the flash
  masks first, then weights — the more VRAM, the less PCIe traffic, degrading
  gracefully to full streaming). Measured on a 16 GB RTX 3080 Laptop at a
  912×1136 2-image edit (12.7k tokens): ~19 s per forward / ~41 s per true-CFG
  step — on par with sd.cpp's `--offload-to-cpu` — and it is the difference
  between a ~0.4 MP and a native ~1 MP edit on that card. `--offload-cpu`
  (`TS_QWEN_IMAGE_OFFLOAD_CPU=1`) forces it always on;
  `TS_QWEN_IMAGE_OFFLOAD_CPU=0` restores the old clamp-resolution behavior.
  `--width`/`--height` are still capped at the hardware memory ceiling (offload
  raises that ceiling) — an oversized request (e.g. 2048×2048 on a 16 GB card)
  is clamped down to the largest size that fits, with a warning, rather than
  OOMing into a garbled/noise result.
- **Fused whole-VAE graph** (`TSGgml_QwenVaeRun`, default on): the entire VAE
  encode/decode runs as ONE device-resident ggml graph — the C# side emits a flat
  op list mirroring the verified `VaeReferenceMath` topology, features stay on the
  GPU end-to-end, weights bind resident from stable buffers (uploaded once), and
  each conv picks im2col+GEMM (tensor cores) while its transient F16 im2col fits a
  budget (`TS_QWEN_VAE_FUSED_IM2COL_BUDGET`, default 2 GiB; the gallocr reuses the
  scratch so the peak is one conv's im2col) or `ggml_conv_2d_direct` above it.
  Replaces the per-conv path's GBs of PCIe round-trips + CPU SiLU/norm loops:
  at 928×688 encode 19.5 s → **0.95 s**, decode 22.8 s → **1.35 s**, bit-equivalent
  to the diffusers oracle (PSNR 99 dB, same as the legacy path). Falls back to the
  per-conv path when the backend can't run it; `TS_QWEN_VAE_FUSED=0` disables.
- **Band-tiled VAE conv** (`VaeReferenceMath.TryGpuConv2dMaybeTiled`): `ggml_conv_2d`
  materializes an F16 im2col of `~IC·KH·KW·OH·OW·2` bytes, which is several GB at
  high resolution and used to spill into shared VRAM (≈3× slower VAE) or OOM — it
  was the real ceiling that forced the output down to ~0.55 MP. The conv is now
  split into horizontal output bands whose im2col fits a budget
  (`TS_QWEN_VAE_CONV_TILE_BYTES`, default 1 GiB); each band re-runs the same conv
  on a manually-padded input slice so the result is **bit-identical** (no tile
  seams — only the transient im2col is bounded). This lets the area clamp target
  the model's **native ~1 MP**, which materially improves face/fine detail.

Head-to-head: on the project's CUDA `image_edit` benchmark scenario (reproducible
via [`benchmarks/engine_comparison`](../../benchmarks/engine_comparison); 4-step
Lightning edit at 544×1184), TensorSharp completes a warm edit in **40.44 s** vs
stable-diffusion.cpp's 48.16 s (~1.19× faster); the cold first request is 54.11 s.

Important toggles:

| Variable | Effect |
|---|---|
| `TS_QWEN_IMAGE_VAE` / `TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ` | Override the resolved companion GGUFs (the CLI exposes these as `--qwen-image-vae` / `--qwen-image-vl` / `--qwen-image-mmproj`) |
| `TS_QWEN_IMAGE_NO_VISION=1` | Skip vision grounding (faster, ungrounded text-only conditioning) |
| `TS_QWEN_IMAGE_REF_AREA` | Reference-latent area in pixels for `inputs[1..]` — the detail source the DiT copies faces/textures from. Default (CUDA): the native ~1 MP with each ref's own aspect, **independent of the output size** (diffusers Edit Plus `VAE_IMAGE_SIZE`; the old rule shrank refs with a smaller output, silently discarding input detail). Clamped to [65536, 4 MP]; above ~1 MP keeps more input detail but is outside the training distribution, and every ref costs area/256 attention tokens |
| `TS_QWEN_IMAGE_VISION_MIN_PIXELS` / `TS_QWEN_IMAGE_VISION_MAX_PIXELS` | Vision-tower condition-image pixel band (defaults 384², 560² — the Qwen-Image-Edit-2511 reference sizing): images keep their OWN resolution snapped to /28 inside the band, one Lanczos resample from the original. The old flow squashed every input to exactly 384² and re-upscaled it to the processor minimum — two resamples and ≤ half the pixels for high-res inputs |
| `TS_QWEN_IMAGE_LORA` | DiT LoRA safetensors applied as a runtime F32 side-path (CLI/server: `--qwen-image-lora`); a Lightning checkpoint also switches the sampling defaults (its steps, cfg 1.0, fixed shift 3) |
| `TS_QWEN_IMAGE_LORA_SCALE` | LoRA multiplier (default 1.0; 0 = structurally on but zero effect) |
| `TS_QWEN_IMAGE_FLOW_SHIFT` | FlowMatch time shift for the base (non-Lightning) path. Default **3** — the Qwen-Image-Edit value (matches stable-diffusion.cpp `default_flow_shift=3`); a lower shift under-noises the trajectory and softens the edit. Set `<= 0` for the old diffusers *dynamic* resolution-dependent shift. Lightning always pins shift 3. |
| `TS_QWEN_IMAGE_MAX_AREA` | Cap the auto target area (Metal speed clamp; on CUDA an optional smaller-than-native quality/speed trade) |
| `TS_QWEN_IMAGE_OFFLOAD_CPU` | CPU offload (CLI/server: `--offload-cpu`): `1` always streams the DiT weights from RAM; `0` never (clamps resolution instead); unset = **auto** (engages exactly when the target resolution doesn't fit beside the resident weights) |
| `TS_QWEN_DIT_OFFLOAD_CHUNK` | Blocks per chunked whole-model graph on the offload path (default 10; smaller = less VRAM for weight slots, more chunk-boundary PCIe) |
| `TS_QWEN_DIT_WHOLE_CAPTURE=0` | Disable the CUDA-graph-captured whole-DiT forward |
| `TS_QWEN_DIT_FLASH=0` | Force the explicit-scores attention path (tighter quadratic VRAM budget) |
| `TS_QWEN_DIT_CFG_BATCH_MAXTOK` | Token budget under which CFG-batching stays enabled |
| `TS_QWEN_DIT_CACHE_MODE` | Denoise cache: `off` (**default** — quality first, like sd.cpp), `easycache` (whole-step skip), `fbc` (First-Block-Cache), `both` |
| `TS_QWEN_DIT_EASYCACHE_THRESHOLD` | EasyCache accumulated-change threshold (default 0.2; lower = closer to no-cache, higher = more skips) |
| `TS_QWEN_DIT_CACHE=0` | Legacy master switch — disables all denoise caching |
| `TS_QIMG_DEBUG=1` | Per-step velocity / latent statistics |

## 6. Generation parameters (`QwenImageParams`)

| Property | Default | Meaning |
|---|---:|---|
| `Steps` | 0 (auto) | FlowMatch-Euler denoising steps. Auto = 30, or the step count of a loaded Lightning LoRA (4/8) |
| `CfgScale` | 0 (auto) | True-CFG guidance scale; `<= 1` disables the negative pass. Auto = 2.5 (the Qwen-Image-Edit-2511 recommendation — 4.0 over-guides: distorted faces, over-saturated color), or 1.0 with a Lightning LoRA. Raise toward 3.5–4 for stronger stylization at the cost of face fidelity |
| `NegativePrompt` | `" "` | Negative prompt for the CFG pass (used only when `CfgScale > 1`) |
| `Seed` | 0 | Deterministic initial-noise seed |
| `TargetArea` | 1024×1024 | Output area in pixels (aspect follows the input; dims snapped to /16) |
| `Width` / `Height` | 0 | Explicit output size (bypasses the VRAM area clamp) |
| `OnStep` | null | Per-step callback `(step, totalSteps, preview)` for live UI feedback |
| `PreviewCount` | 0 | Number of decoded RGB previews emitted across the loop |

## 7. Server behavior

When the Web UI hosts a `qwen_image` DiT GGUF, the upload + edit endpoints take
over:

- `POST /api/image-edit` runs a single edit and returns the output image.
  Multipart: one or more `image` file parts + `prompt`/`steps`/`cfg`/`seed`/
  `targetArea` fields. JSON: `{ imagePaths: [...], prompt, ... }` (or legacy
  single `imagePath`), where paths come from `POST /api/upload`.
- `POST /api/image-edit/stream` streams SSE frames: progress ticks plus
  throttled decoded previews (`{ imageEdit: true, step, total, image, width,
  height }`), then a final full-resolution image before `done`. Previews also
  work when the request leaves the step count at auto (0) — up to 8 evenly
  spaced preview frames; a failed preview encode degrades to a progress-only
  tick instead of aborting the edit.
- Edits are serialized behind a shared lock (the diffusion nets are not
  thread-safe), so concurrent requests run one at a time.
- The Ollama / OpenAI chat adapters are autoregressive and do not expose image
  editing.

## 8. Remaining work

- Concurrent edits are serialized; batched / pipelined editing across requests
  is not implemented.
- Vision-grounded conditioning is accurate but heavy; the fastest practical
  full-quality edits are on the CUDA path with the whole-DiT capture.
