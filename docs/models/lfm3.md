# LFM3 (`lfm3`)

[← back to model index](README.md)

TensorSharp runs Liquid AI's LFM3 decoders, and the text tower of LFM3-VL, from
GGUFs that declare `general.architecture = lfm3`. That is the spelling llama.cpp's
`convert_hf_to_gguf.py` writes (`LFM3Model`), and the one AMQL's `to-gguf`
reproduces byte for byte.

| Property | Value |
|---|---|
| GGUF architecture | `lfm3` |
| Source class | [`Lfm3Model`](../../TensorSharp.Models/Models/Lfm3/Lfm3Model.cs) |
| Architecture plug-in | [`Lfm3Architecture`](../../TensorSharp.Models/Models/Lfm3/Lfm3Architecture.cs) |
| Modalities | Text. LFM3-VL's vision tower lives in a separate mmproj and is not loaded |
| Chat template | The GGUF's embedded `tokenizer.chat_template` (no dedicated chat protocol) |
| Serving | The continuous-batching engine through the KV-state snapshot (conv state included); no batched paged forward yet |
| Multi-GPU | Tensor parallelism (`--tp N`), local and multi-node |
| Speculative decoding | Not supported |
| LoRA | Not supported |
| Backends | Every backend, through the generic per-op path |

## Run it

```bash
echo "Explain a short convolution in one paragraph." > prompt.txt
dotnet run --project TensorSharp.Cli -- \
    --model models/LFM3-VL-2B-Q8_0.gguf \
    --backend ggml_cuda \
    --input prompt.txt
```

```bash
dotnet run --project TensorSharp.Server.Host -- \
    --model models/LFM3-VL-2B-Q8_0.gguf \
    --backend ggml_cuda --tp 2 --port 5000
```

## Architecture

Every layer is the same block around one of three mixers:

```text
tokens -> embedding
  -> N x [ RMSNorm -> mixer -> (post_attention_norm) -> + residual
           RMSNorm -> SwiGLU(gate, up) -> down -> (post_ffw_norm) -> + residual ]
  -> final RMSNorm -> LM head (tied to the embedding when there is no output.weight)
```

The `post_*` "sandwich" norms are present on every layer or on none.

| Layer kind | How the GGUF marks it | Mixer |
|---|---|---|
| conv | `attention.head_count_kv[l] == 0` | Gated short conv: `in_proj` gives `[B \| C \| x]`; a causal depthwise conv of width `shortconv.l_cache` runs over `B·x`; `C` gates its output before `out_proj` |
| full attention | KV heads > 0, `sliding_window_pattern[l] == false` | GQA with per-head Q/K RMSNorm **before** NeoX RoPE, zero-tail partial rope, per-head sigmoid output gate (`attn_gate`) |
| sliding attention | KV heads > 0, `sliding_window_pattern[l] == true` | The same over the last `attention.sliding_window` positions, rope base `rope.freq_base_swa`, every pair rotating |

**Zero-tail partial rope.** LFM3's full-attention layers take the whole head
dimension's frequency progression and give the trailing pairs frequency zero
(they do not rotate) — not llama-style partial rope, which shortens the
progression. The GGUF expresses it as `rope_freqs.weight`, 1 for rotating pairs
and 1e30 for the rest; the model divides the full-attention frequencies by it
exactly as llama.cpp does, and leaves the sliding layers alone.

**Tokenizer.** `tokenizer.ggml.pre = lfm2` selects llama.cpp's Llama-3
pre-tokenizer regex (digits in runs of up to three).

## Serving

A sequence carries the attention K/V and, per conv layer, the last
`l_cache - 1` values of `B·x` per channel. The KV-state snapshot holds both, the
way Nemotron-H's holds its Mamba2 state:

- each attention layer contributes its K then V rows for the block
  (`[kvHeads, tokens, row]`, the cache's own dtype);
- each conv layer contributes its state at the **end** of the block, which is
  why the model asks for per-block capture (`RequiresPerBlockCapture`) and offers
  the smaller K/V-only form for blocks that are not restore points.

The conv state cannot be rewound, so the cache does not truncate: a shorter
prefix is reached by restoring a snapshot or re-prefilling. In the radix prefix
cache LFM3 is a class-R page family (pages carry the state at restorable
boundaries), like Nemotron-H.

Sliding layers keep a full linear cache rather than a ring, so a snapshot of any
range is restorable; attention reads only the window.

## Tensor parallelism

Megatron-style, per layer:

- attention: Q, the gate, K and V split by heads; each rank keeps its own K/V
  shard; `attn_output` is row-parallel;
- short conv: `in_proj` is sharded segment-aware (rank g gets `[B_g | C_g | x_g]`
  for its `hidden/N` channels), each rank convolves and keeps the state of its own
  channels, `out_proj` is row-parallel;
- FFN: gate and up column-parallel, down row-parallel.

`--tp N` needs N to divide the attention heads, the KV heads, the hidden size and
the FFN width; a quantized row-parallel projection additionally needs whole
quantization blocks per rank. The snapshot layout is the single-device one byte
for byte (heads are split contiguously, the conv state is channel-major), so a
block captured on one device restores under TP and back. Across nodes, extract
and inject are mirrored model-control operations.

## Validation

`InferenceWeb.Tests/Lfm3ModelTests.cs` runs a synthetic six-layer checkpoint
(conv / full / conv / sliding / conv / full) written in the converter's layout
(`Lfm3SyntheticModelBuilder`) against a float64 forward written from Liquid's
`modeling_lfm3.py`, which takes the zero-tail rope from the partial rotary factor
rather than from `rope_freqs`:

| Covered | Backend |
|---|---|
| Prefill, chunked prefill and decode against the reference (with and without sandwich norms, tied and untied head); F16 KV; Q8_0 projections | `cpu` |
| Snapshot round trip (K/V-only + full blocks), refusal of truncation | `cpu` |
| Three concurrent greedy requests through `InferenceEngine` with forced ownership swaps, token-for-token against serving each alone | `cpu` |
| `--tp 2` (F32 and Q8_0) against the reference / single device, TP snapshot equal to the single-device one, single-device snapshot restored under TP | `cpu` (in-process group) |
| Chunked prefill, decode and snapshot restore through the native masked softmax | `ggml_cpu` |
| `--tp 2` against one device | `ggml_cuda`, two GPUs (gated; skipped without them) |

Not yet validated: a real LFM3 / LFM3-VL GGUF end to end, the GPU backends'
throughput (the short conv, rope, gate and window masks run on the host), and
multi-node TP.
