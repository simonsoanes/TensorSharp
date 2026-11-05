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
| Chat template | The GGUF's embedded `tokenizer.chat_template`, parsed by the `lfm3` chat protocol (reasoning, Qwen-style XML tool calls) |
| Serving | The continuous-batching engine through the KV-state snapshot (conv state included); no batched paged forward yet |
| Multi-GPU | Tensor parallelism (`--tp N`), local and multi-node |
| Verified checkpoints | LFM3-VL-2B (text tower) as llama.cpp's BF16 conversion and a Q4_K_M quant of it |
| Speculative decoding | Not supported |
| LoRA | Not supported |
| Backends | Every backend through the generic per-op path; `ggml_cuda`, on one device or tensor parallel, through the [native forward](#cuda) |

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

## Chat

The embedded template is rendered as shipped. With thinking on, its generation
prompt ends `<|im_start|>assistant\n<think>`, so the reply starts inside the
reasoning block; with thinking off it ends with the closed `<think></think>`.
The `lfm3` chat protocol (`ChatProtocolRegistry`) parses the reply with the
ChatML parser: reasoning up to `</think>` goes to the reasoning field, and
tool calls in the template's XML form
(`<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`)
become structured calls. It also tells the prompt renderer what the cache holds
in front of each past reply, so a follow-up or a tool round extends the cached
conversation instead of re-prefilling it - which matters here, because the conv
state cannot be rewound.

As for every family, the server's `think` flag defaults to off; llama.cpp's
default follows the template, which is on.

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

On `ggml_cuda` each rank runs the [native forward](#cuda) over its shard. The graph
is cut after each row-parallel projection, 2 per layer, and TensorSharp's fused TP
executor runs the ranks segment by segment, reducing in exact F32 across a node's
GPUs and then across nodes. Only the driver's first rank carries the LM head. Nodes
vote on the native path at load: a node that cannot take it (a Q8 cache,
`TS_LFM3_NATIVE=0`) sends them all per-op, because the two paths' reductions do not
pair up.

## CUDA

On one `ggml_cuda` device a forward is a single native graph
(`TSGgml_Lfm3Forward`, [`ggml_ops_lfm3.cpp`](../../TensorSharp.GGML.Native/ggml_ops_lfm3.cpp))
instead of an operation at a time, with three TensorSharp kernels
([`ggml_ops_lfm3.cu`](../../TensorSharp.GGML.Native/ggml_ops_lfm3.cu)) run as custom
nodes through the fused backend. ggml itself is unchanged.

| Kernel | Does |
|---|---|
| Short conv | `C * conv(B * x)` for every channel and token, and the conv state's update, in place |
| Attention prep | Q/K head RMSNorm, NeoX RoPE and the K/V cache write, in one launch for all heads |
| Decode attention | One query over the layer's visible keys, split across the KV range and folded with the sigmoid head gate |

The position is a device input the kernels read, so the one-token graph has no
shape that changes between tokens: it is built once and replayed until the cache
moves. Several tokens get a graph of their own and attend through ggml's flash
attention, over the exact key range each layer can see.

The K/V caches and conv state then live on the device. The snapshot contract
still works on host bytes, so extracting pulls the rows it needs across (all of a
forward's new rows at once, not block by block) and injecting pushes them back;
nothing else moves.

The native path needs an F16 or F32 KV cache, a head size up to 256 and a GQA ratio
of 1, 2, 4 or 8; anything else, or `TS_LFM3_NATIVE=0`, keeps the per-op path, and the
model says which at load.

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
| Native forward: prefill and single tokens interleaved past the window and the first cache capacity, reset, snapshots from the device in both block forms, restore into a reset model; F32 and F16 KV; head size 64 (flash attention) and 16 (explicit attention) | `ggml_cuda` |
| Native single-token forward held to the reference at 1e-4 | `ggml_cuda` |
| `--tp 2` against one device, per-op and native | `ggml_cuda`, two GPUs (gated; skipped without them; **not run**, this machine has one) |

`InferenceWeb.Tests/Lfm3ChatProtocolTests.cs` covers the chat protocol: the
generation suffix in both thinking modes, splitting reasoning from the answer,
the XML tool call, and follow-up / tool-round prompts that start with everything
the cache holds.

### Real checkpoint

LFM3-VL-2B's text tower, converted by llama.cpp's `convert_hf_to_gguf.py`
(BF16, 3.3 GB) and a Q4_K_M quant of it, against llama.cpp's own tools on the same
files, on one RTX 5090 (2026-11-04, CUDA 13.3, unmodified ggml at `facdc44a`). The
probes were throwaway; the logs are in the ignored `artifacts/lfm3-cuda/`.

| Check | Backend | Result |
|---|---|---|
| Perplexity, 4 x 512-token chunks of a Gutenberg text, scored as `llama-perplexity` scores it (BF16 / Q4_K_M) | `ggml_cuda` native | 234.89 vs 234.60 (+0.12%) / 214.42 vs 215.17 (-0.35%) |
| The same at 16- and 32-token contexts, ~3,700 predictions each, so every one comes from a short prefill (BF16 / Q4_K_M) | `ggml_cuda` native | 2100.73 vs 2100.32, 1118.03 vs 1117.77 / 1918.14 vs 1935.66 (+/- 129), 986.25 vs 986.45 |
| Top-10 next-token log-probs and 64 greedy tokens vs `llama-server`, prompts of 8, 300, 700 and 2000 tokens, BF16 | `ggml_cuda` native | greedy 64/64 at every length; log-probs within 0.06 |
| Chunked prefill (96-token chunks) and a 2600-token prompt, past the 2048-token split, 16 greedy tokens, BF16 | `ggml_cuda` native | 16/16; within 0.06 |
| OpenAI chat, greedy, repeat penalty off: three prompts, thinking on and off, vs `llama-server` | server, native | identical prompt token counts and replies, reasoning included, 6/6 |
| Twelve concurrent requests, each vs the same request alone | server, native | 12/12 identical |
| A `tools` request (thinking on), then the tool result | server, native | the same `get_weather` call and final answer as llama.cpp; the follow-up reused 148 of 174 prompt tokens |
| Multi-node TP: the server as node 0 and the CLI as node 1 (`--tp 1 --tp-peers 127.0.0.1:9500,127.0.0.1:9501`, both on the one GPU), BF16: three prompts, thinking on and off, against `llama-server` | `ggml_cuda` native | 6/6 identical, reasoning included |
| The same cluster: a `tools` round (thinking on); six concurrent requests against each alone | `ggml_cuda` native | the same call and answer as llama.cpp, 148 of 174 tokens reused through the mirrored snapshot; 6/6 identical |
| One node with `TS_LFM3_NATIVE=0` | `ggml_cuda` | both nodes take the per-op path and answer correctly |
| Two ranks in one process ([`Lfm3TensorParallelProbe`](../../eng/validation/Lfm3TensorParallelProbe/Program.cs)) against one device: 700-token chunked prefill, 64 greedy tokens replayed, a snapshot round trip, cache growth | `ggml_cuda` native, loopback ranks on one GPU, host reduction | argmax 65/65 at BF16 and Q4_K_M; top token within 0.02 / 0.09 |
| The same on the per-op TP path, which has none of the native code: the rounding floor of sharding | `ggml_cuda` per-op, loopback | argmax 65/65; top token within 0.02 / 0.12 |
| The same, teacher-forced over 1,500 corpus tokens | `ggml_cuda` native, loopback | perplexity 147.90 vs 147.93 (BF16), 134.13 vs 134.49 (Q4_K_M) |
| Top-10 log-probs, 300 tokens in 128-token chunks, BF16 (2026-11-03) | `cpu` | greedy 8/8; within 0.04 |

Q4_K_M greedy output can differ from llama.cpp's on a single prompt, and its
log-probs can differ by up to about 0.5 on one with a sharp distribution: its
activations are rounded to 8 bits before each matmul, so a tiny difference upstream
can flip a rounding. Prompts split into two-token chunks agreed with llama.cpp at
micro-batch 2 to 0.07, and this path's own results then move with chunk size where
llama.cpp's stayed put up to 8 tokens. The aggregate rows above show no bias from
it. BF16 has no such step.
Managed-`cpu` Q4_K_M drifts further (greedy 5/8, within 0.62), from the managed
K-quant kernels.

**Speed**, RTX 5090, `llama-bench` for llama.cpp (tokens/s):

| | Decode | Prefill, 2000 tokens |
|---|---|---|
| BF16, native forward | 315-334 | 48,000 (14-17k through the server) |
| BF16, llama.cpp | 348 | 39,000 |
| Q4_K_M, native forward | 560-590 | 54,000 |
| Q4_K_M, llama.cpp | 475 | 37,000 |
| BF16, per-op `ggml_cuda` | 12 | 300 |

Sharding moves single values more than the averages suggest: BF16 log-probs in the
top 10 differ by up to 0.14 from one device (0.11 per-op), and over a 576-token greedy
run one flat position (top token p 0.15, against 0.24 here and 0.17 per-op) stood
out and the next tokens were back within 0.05. The perplexity rows are the check
that this is rounding: a fault would bias them.

Through the server, BF16 decode has a median token gap of 3.1 ms (about 320 tokens/s),
but sustained decode is 115-130 tokens/s: about every 16th token, one KV block, takes
75-115 ms. The model's block extract is 6 ms of that; the rest is not yet explained.
`Forward` alone sustains 307, including a snapshot round trip.

Tensor parallelism, BF16, tokens/s:

| | Decode | Prefill |
|---|---|---|
| Two nodes, TCP over loopback, one GPU between them, through the server | 28-32 | 500-530 |
| The same, per-op | about 8 | not measured |
| Two loopback ranks in one process, one GPU, no server | 197 | not measured |

These say little about real hardware, which this machine does not have. Prefill is
bound by the transport: each 2048-token reduction carries 16 MB and loopback
`TcpCommunicator` moved 70-140 MiB/s. Decode's reductions cost about 2 ms a token
(0.04 ms each) and its execution about 4.5 ms, which leaves most of its 33 ms
unmeasured; two processes taking turns on one GPU under WDDM is the likely cause, not
a measured one.

Not yet validated: `--tp 2` across two real GPUs, including its device collective and
each rank's device selection (`GgmlCudaTensorParallelNativeForwardMatchesOneDevice`
needs a second GPU; the loopback ranks share device 0 and reduce through the host);
multi-node TP across real machines; Q4_K_M through a multi-node server (loopback
ranks only); and LFM3-VL's vision tower (not implemented). Only `ggml_cuda` has the
kernels, so other GPU backends keep the per-op speed.
