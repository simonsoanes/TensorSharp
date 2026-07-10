# Gemma 3

[← back to model index](README.md)

| Property | Value |
|---|---|
| Provider | Google |
| GGUF architecture key | `gemma3` |
| Source class | [`Gemma3Model`](../../TensorSharp.Models/Models/Gemma3/Gemma3Model.cs) |
| Vision encoder | [`Gemma3VisionEncoder`](../../TensorSharp.Models/Models/Gemma3/Gemma3VisionEncoder.cs) |
| Image processor | [`Gemma3ImageProcessor`](../../TensorSharp.Models/Models/Gemma3/ImageProcessor.cs) |
| Example models | gemma-3-4b, gemma-3-12b, gemma-3-27b |
| Modalities | Text, image |
| Thinking mode | No |
| Tool calling | No |
| Batched / paged forward | Not implemented. Runs through the per-sequence KV-swap fallback inside `BatchExecutor` when the continuous-batching engine is active. See §11. |
| Output parser | `PassthroughOutputParser` |

## Downloads

Verified GGUF pointers:

| Model | HF repo | Recommended file | mmproj |
|---|---|---|---|
| gemma-3-4b-it (official QAT) | [google/gemma-3-4b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) | `gemma-3-4b-it-q4_0.gguf` (3.155 GB) | `mmproj-model-f16-4B.gguf` (0.851 GB; same repo) |
| gemma-3-4b-it (non-gated alternative) | [ggml-org/gemma-3-4b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) | `gemma-3-4b-it-Q4_K_M.gguf` (2.490 GB) or `gemma-3-4b-it-Q8_0.gguf` (4.130 GB) | `mmproj-model-f16.gguf` (0.851 GB; same repo) |

The official `google/...` repo is **gated**: downloading its files requires a
Hugging Face login and accepting Google's Gemma license. The `ggml-org` repo
downloads anonymously, but the weights remain derived from Gemma and its model
card declares the Gemma license.

Command-line download (one line per file; requires `pip install -U huggingface_hub`):

```bash
python -m pip install -U huggingface_hub
hf download ggml-org/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir models
hf download ggml-org/gemma-3-4b-it-GGUF mmproj-model-f16.gguf --local-dir models
```

CLI one-shot (the text prompt comes from a file via `--input`; with `--image`
and no `--input` a default describe-the-image prompt is used; CLI sampling
defaults to greedy and `--max-tokens` defaults to 100):

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/gemma-3-4b-it-Q4_K_M.gguf \
  --mmproj models/mmproj-model-f16.gguf \
  --image photo.png --max-tokens 300 --backend ggml_cpu
```

Server (chat Web UI plus OpenAI/Ollama-compatible APIs on `http://localhost:5000`):

```bash
dotnet run --project TensorSharp.Server -c Release -- --model models/gemma-3-4b-it-Q4_K_M.gguf \
  --mmproj models/mmproj-model-f16.gguf --backend ggml_cuda --max-tokens 4096
```

## 1. Origin and intent

Gemma 3 is Google's third-generation open-weights LLM family, distilled from
larger Gemini-class models. The defining architectural choices are:

- **Hybrid attention pattern** — every sixth layer is a full causal attention
  layer; the other five are sliding-window attention (SWA) layers. SWA bounds
  the per-layer attention cost on long contexts.
- **Per-head QK normalization** — both Q and K are RMSNorm-ed per head before
  the attention dot-product, which stabilizes attention scores in the wide
  head dimension Gemma 3 uses (256 by default).
- **GeGLU FFN** with a heavy intermediate-to-hidden ratio — uses a fused
  `ffn_gate_up` weight and the GELU non-linearity, giving slightly stronger
  signal preservation than SwiGLU at the cost of a small additional FLOP.
- **Embedding scaling** — token embeddings are multiplied by `sqrt(hidden_size)`
  before the first layer, which keeps the residual stream norm independent of
  hidden size.
- **Optional logit softcap** — `tanh(logits / cap) * cap` clips extreme logits.

Gemma 3 is a vision-text model: a separate `Gemma3VisionEncoder` (loaded from a
multimodal projector GGUF, typically `mmproj-gemma3-4b-f16.gguf`) produces
fixed-length image token embeddings that are spliced into the residual stream
at `<start_of_image>` token positions.

## 2. Model architecture

```
                              ┌──────────────────────────┐
                              │ token_embd.weight        │
        tokens (int[]) ──────►│  (× sqrt(hidden_size))   │
                              └────────────┬─────────────┘
                                           │
                              [optional] InjectVisionEmbeddings
                                           │
                                           ▼
                  ┌────────────── × NumLayers ──────────────┐
                  │  RMSNorm (attn_norm)                    │
                  │  Q, K, V projections                    │
                  │  per-head RMSNorm (attn_q_norm/k_norm)  │
                  │  RoPE (NeoX, local OR global base)      │
                  │  scale Q                                 │
                  │  Attention (SWA window OR full causal)  │
                  │  Output projection                       │
                  │  RMSNorm (post_attention_norm) + residual│
                  │  RMSNorm (ffn_norm)                     │
                  │  GeGLU: GELU(gate) * up → down           │
                  │  RMSNorm (post_ffw_norm) + residual      │
                  └─────────────────────────────────────────┘
                                           │
                              RMSNorm (output_norm)
                                           │
                              LM head (output.weight or tied)
                                           │
                              [optional] tanh-softcap
                                           │
                                           ▼
                                       logits
```

`IsGlobalLayer(layer)` is true when `(layer + 1) % 6 == 0`. Global layers use
full causal attention; the other five-of-six layers attend only to the trailing
`sliding_window` tokens.

## 3. Forward graph

Token-by-token (decode) and per-chunk (prefill) flow through the same
`TransformerBlock`. The single token position is the only difference: prefill
broadcasts `[seqLen, hidden]` through the GEMMs and applies a triangular causal
mask, while decode runs a `[1, hidden]` row through the cached attention
window.

For each layer L:

```
hidden ─► RMSNorm(attn_norm.weight, eps)
       ─► attn_q.weight ─► Q          (per-head reshape)
       ─► attn_k.weight ─► K          (per-head reshape)
       ─► attn_v.weight ─► V          (per-head reshape)
       ─► RMSNorm(attn_q_norm.weight) on Q (per head)
       ─► RMSNorm(attn_k_norm.weight) on K (per head)
       ─► RoPE_NeoX(Q, K, freqs[layer])     // local OR global base
       ─► Q ← Q * (1/sqrt(headDim))
       ─► append K, V to KV cache (positions startPos..startPos+seqLen-1)
       ─► attention(Q, KCache, VCache, window=W if SWA else totalSeq)
       ─► attn_output.weight matmul → o
       ─► RMSNorm(post_attention_norm.weight, eps) on o
       ─► hidden = hidden + o
       ─► h2 = RMSNorm(ffn_norm.weight, eps) on hidden
       ─► ffn_gate_up.weight matmul → [gate ‖ up]   (fused projection)
       ─► g = GELU(gate)
       ─► h3 = ffn_down.weight × (g * up)
       ─► RMSNorm(post_ffw_norm.weight, eps) on h3
       ─► hidden = hidden + h3
```

After all layers:

```
hidden ─► narrow(seq_len-1) if prefill          // only the last row matters for the LM head
       ─► RMSNorm(output_norm.weight, eps)
       ─► matmul against output.weight (or token_embd.weight when tied)
       ─► [optional] tanh(logits/cap) * cap     // _finalLogitSoftcap > 0
       ─► copy to float[VocabSize] for the sampler
```

## 4. Components in detail

### 4.1 Attention

- **GQA** with separate `key_length` and `value_length` (default 256/256).
- **Pattern**: `IsGlobalLayer(layer)` returns true when `(layer + 1) % 6 == 0`.
  Global layers are full causal; the other five layers in every six attend only
  to positions in `[totalSeqLen − slidingWindow, totalSeqLen)`.
- **QK normalization**: per-head RMSNorm with weights
  `attn_q_norm.weight` / `attn_k_norm.weight`.
- **RoPE**: NeoX-style. Local layers use `_ropeFreqsLocal[]` derived from
  `gemma3.rope.local.freq_base` (default 10000). Global layers use
  `_ropeFreqsGlobal[]` from `gemma3.rope.freq_base` divided by `_ropeScale`.
  The 27B variant (`NumLayers == 34`) hardcodes `_ropeScale = 8.0`.
- **Decode SWA bounding**: `AttentionDecodeWithWindow()` reads only positions
  `[max(0, totalSeqLen − slidingWindow), totalSeqLen)` from the cache.

### 4.2 FFN — GeGLU

`ffn_gate_up.weight` packs `[gate ‖ up]` along the row dimension so a single
matmul produces both halves. `Ops.GELUMul` then computes `GELU(gate) * up`
(the activation is the Gaussian Error Linear Unit, not the cheaper SiLU used by
Qwen / Mistral). The `ffn_down.weight` matmul follows.

### 4.3 Normalization

Four RMSNorms per block — `attn_norm`, `post_attention_norm`, `ffn_norm`,
`post_ffw_norm` — plus the per-head `attn_q_norm` / `attn_k_norm`. The final
`output_norm` is shared across layers. Epsilon comes from `general` GGUF
metadata (`Config.Eps`, typically `1e-6`).

### 4.4 Embedding and LM head

The token embedding tensor `token_embd.weight` is also reused as the LM head
when `output.weight` is absent (`_hasTiedOutput`). Embeddings are scaled by
`sqrt(Config.HiddenSize)` in `ScaleEmbedding()`.

### 4.5 Logit softcap

If `gemma3.final_logit_softcapping > 0`, the LM head output is clipped via
`logits = tanh(logits / cap) * cap` to keep extreme logits from dominating the
softmax. Skipped otherwise.

### 4.6 Vision pipeline

`Gemma3VisionEncoder` is a CLIP-style ViT that produces a fixed
`TokensPerImage` (256 by default) embeddings of size hidden. The image
processor:

1. Composites RGBA over white.
2. Resizes the image to `image_size × image_size` (typically 896×896).
3. Normalizes with the encoder's mean/std.
4. Converts to NCHW float tensor.

Embeddings are injected into the residual stream by
`InjectVisionEmbeddings()`, which copies the `[256, hidden]` block over the
positions occupied by `<start_of_image>` placeholders. The input prompt is
expanded ahead of time by `ChatTemplate.ExpandGemma3ImageTokens()` so that
each image becomes one start sentinel + 256 padding tokens + one end sentinel
inside the tokenized prompt.

## 5. Parameters and settings (GGUF metadata)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `gemma3.attention.sliding_window` | uint32 | 1024 | SWA window length for non-global layers |
| `gemma3.attention.key_length` | uint32 | 256 | Per-head key dim |
| `gemma3.attention.value_length` | uint32 | 256 | Per-head value dim |
| `gemma3.rope.local.freq_base` | float32 | 10000 | RoPE base for SWA layers |
| `gemma3.rope.freq_base` | float32 | from `general` | RoPE base for global layers |
| `gemma3.final_logit_softcapping` | float32 | 0 (disabled) | tanh-softcap level for the LM head |

Plus the standard `general.*` keys (`hidden_length`, `block_count`,
`attention.head_count`, `attention.head_count_kv`, `embedding_length`, etc.)
that `ParseBaseConfig` reads.

## 6. Weight naming convention

```
token_embd.weight                          # [vocab, hidden]
blk.{L}.attn_norm.weight                   # pre-attention RMSNorm
blk.{L}.attn_q.weight                      # Q projection
blk.{L}.attn_k.weight                      # K projection
blk.{L}.attn_v.weight                      # V projection
blk.{L}.attn_q_norm.weight                 # per-head Q RMSNorm
blk.{L}.attn_k_norm.weight                 # per-head K RMSNorm
blk.{L}.attn_output.weight                 # output projection
blk.{L}.post_attention_norm.weight         # post-attention RMSNorm
blk.{L}.ffn_norm.weight                    # pre-FFN RMSNorm
blk.{L}.ffn_gate.weight }                  # before fusion: separate gate/up
blk.{L}.ffn_up.weight   }
blk.{L}.ffn_gate_up.weight                 # after fusion: [2*intermediate, hidden]
blk.{L}.ffn_down.weight                    # FFN down projection
blk.{L}.post_ffw_norm.weight               # post-FFN RMSNorm
output_norm.weight                         # final RMSNorm
output.weight                              # LM head (optional if tied)
```

`Gemma3Model` calls `FuseGateUpWeights()` at load time to concatenate
`ffn_gate.weight` and `ffn_up.weight` into a single `ffn_gate_up.weight`. When
the GGUF already ships fused weights, the call is a no-op.

## 7. TensorSharp implementation walkthrough

The constructor (`Gemma3Model(string ggufPath, BackendType backend)`) does
exactly the steps a generic `ModelBase` workflow expects:

1. `ParseBaseConfig()` — reads `general.*` keys and fills `Config`.
2. Reads the Gemma 3-specific metadata, decides the per-head dims, the SWA
   window, both RoPE bases, the optional softcap, and detects the 27B variant
   to override `_ropeScale`.
3. `ParseTokenizer()` — builds the SentencePiece tokenizer from GGUF
   metadata.
4. `LoadWeights()` — streams F32 norms / embeddings into managed tensors and
   leaves quantized matmul weights in the `_quantWeights` dictionary.
5. `_hasTiedOutput = !_weights.ContainsKey("output.weight") && !_quantWeights.ContainsKey("output.weight")`.
6. `FuseGateUpWeights()` — concatenates gate / up if the GGUF stored them
   separately.
7. `PrepareCudaQuantizedWeightsForInference()` — uploads and reorders quant
   blobs for the direct CUDA backend (no-op on other backends).
8. `PrecomputeRoPE()` — fills `_ropeFreqsLocal` / `_ropeFreqsGlobal`.
9. `InitKVCache(maxSeqLen)` — allocates per-layer K and V tensors of shape
   `[NumKVHeads, maxSeqLen, headDim]` in the configured KV-cache dtype
   (`f32`, `f16`, or `q8_0`).

`Forward(int[] tokens)` then runs the transformer block-by-block. There is no
fused single-graph kernel for Gemma 3 in this branch — each op dispatches
independently — which is one of the optimization opportunities listed below.

### 7.1 Multimodal injection

`InjectVisionEmbeddings()` is called before the loop when
`_pendingVisionEmbeddingsList` is non-empty. The list is populated by the
multimodal injector (`TensorSharp.Models/ModelMultimodalInjector.cs`) which
runs the image processor + vision encoder once per unique image path and caches
the resulting `[TokensPerImage, hidden]` embedding tensor.

## 8. Prefill path

Prefill walks `seqLen > 1` tokens through the same `TransformerBlock` and falls
back to the standard managed code path. The two notable details are:

- **SWA mask cache** — `_cachedSWAMaskWidths` stores per-row sliding-window
  mask widths for the current `(queryLen, startPos)`. All five SWA layers in
  any block-of-six share the same mask, so the mask is built once and reused.
  When chunked KV-cache reuse changes `startPos`, the cache invalidates.
- **Single-pass last-row narrow** — the LM head only ever runs on the last
  row, so prefill keeps the full `[seqLen, hidden]` tensor only until the last
  layer, then narrows to `[1, hidden]` before `output_norm` and the LM head.

## 9. Decode path

`Forward([token])` dispatches each op individually because Gemma 3 does not
yet have a fused single-graph kernel (unlike Gemma 4). The decode hot path
relies on:

- Pre-resolved per-layer weight name strings (no string interpolation in the
  hot loop after init).
- The KV-cache append being a `Narrow + Copy` rather than a full slab rewrite.
- `AttentionDecodeWithWindow()` which restricts the `K^T Q` and softmax window
  for SWA layers to the trailing `slidingWindow` positions instead of the full
  cache.

For the GGML CUDA / Metal backends, every matmul, RMSNorm, RoPE, softmax, and
attention op runs through the registered native kernel; for the direct CUDA
backend it goes through cuBLAS GEMM and PTX kernels; for `cpu` it goes through
the SIMD-optimized managed kernels in `TensorSharp.Core`.

## 10. Memory and KV cache strategy

- Per-layer K and V tensors are sized for the full `maxSeqLen` and *every*
  layer (both SWA and global) gets the same capacity. SWA bounding is a
  per-decode windowing operation, not a smaller allocation. (This is one of
  the listed optimization opportunities below.)
- `ResetKVCache()` zeroes every cache tensor and calls
  `InvalidateTensorDeviceCache()` so the GGML / CUDA copies see the reset.
- `TruncateKVCache(int tokenCount)` keeps the first `tokenCount` positions
  (used by the multi-turn KV reuse path in the server).

Quantized weights load through `LoadWeights()` and stay quantized in the
`_quantWeights` dictionary; matmul calls run through the backend's native
quantized matmul without dequantizing to FP32.

## 11. Batched / paged forward (continuous batching)

Gemma 3 does **not** implement `IBatchedPagedModel.ForwardBatch`. When
the continuous-batching engine is active (default in
`TensorSharp.Server`), Gemma 3 sequences run through
`BatchExecutor.ExecuteStepPerSequence` — the per-sequence KV-swap
fallback that uses the model's `TryExtractKVBlock` /
`TryInjectKVBlock` contract to move KV state in and out as the scheduler
hands the model different sequences. The scheduler, paged block pool,
prefix-cache index, and request streaming all still apply; the only
difference vs. a true batched port (Mistral 3 / Gemma 4 / Qwen 3 / etc.)
is that the model runs one sequence at a time inside each scheduler
step instead of packing N sequences into one forward.

Porting Gemma 3 to `IBatchedPagedModel` would mostly mirror Gemma 4's
batched port, with simpler requirements (no PLE, no KV-donor aliasing,
no per-layer head dim heterogeneity) but with the same alternating
SWA / global attention dispatch and circular-cache considerations.
It's a feasible follow-up, listed under §13 Optimization opportunities.

## 12. Output parser and chat template

- `Gemma3OutputParser` is the same as `PassthroughOutputParser` — Gemma 3 has
  no thinking / tool-call wire format.
- Chat template falls back to the hardcoded Gemma chat template when the GGUF
  does not ship a Jinja2 template.

## 13. Optimization opportunities

- **Batched / paged forward port** — implementing
  `IBatchedPagedModel.ForwardBatch` would let Gemma 3 use the same
  continuous-batching engine path that Mistral 3 / Gemma 4 / Qwen 3 use,
  with the per-layer SWA dispatch already supported by the native paged
  attention kernel.
- **Fused QKV** — Q, K, V are still three separate projections. Concatenating
  them into a single matmul (the way Gemma 4 / Qwen 3 do) would halve the
  attention dispatch count.
- **Fused single-graph decode** — adopting a `Gemma3ModelDecode` kernel
  analogous to `Gemma4ModelDecode` would eliminate per-op CPU/GPU round-trips
  on Metal / CUDA.
- **Circular SWA cache** — SWA layers could allocate `slidingWindow` slots
  instead of `maxSeqLen`, mirroring Gemma 4's `CopyToCacheCircular()` path.
- **Chunked prefill** — long prompts currently materialize the full
  `[seqLen × seqLen]` mask for global layers. Chunking (as in Gemma 4) would
  bound the per-step memory usage.
