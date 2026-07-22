# Qwen3.5 Tensor Parallelism — Design Proposal

Companion to `TENSOR_PARALLELISM_PLAN.md`, which lists Qwen3.5 as
`⏸ Deferred (GatedDeltaNet SSM + MoE + MRoPE)`. This document proposes a concrete
design and argues that the deferral is no longer necessary: the hard part
(GatedDeltaNet) turns out to shard more cleanly than full attention, and the
MoE part reduces to the existing column/row machinery.

---

## 1. Why Qwen3.5 is different

Qwen3 (the reference TP implementation) is homogeneous: 36 identical
dense transformer blocks. Every layer is `attn_qkv → attention → attn_output`
plus `ffn_gate_up → SiLU·mul → ffn_down`, so one `TransformerBlockTP` covers
the whole model.

Qwen3.5 is a **hybrid, heterogeneous** stack. Four axes of variation:

| Axis | Values | Source |
|------|--------|--------|
| Mixer type | GatedDeltaNet (recurrent) vs FullAttention | `_isRecurrent[l]` — [Qwen35Model.cs:391](TensorSharp.Models/Models/Qwen35/Qwen35Model.cs#L391) |
| FFN type | Dense vs MoE (+optional shared experts, +optional shared-expert gate) | `_isMoeLayer[l]`, `_hasSharedExperts[l]` — [Qwen35Model.cs:577](TensorSharp.Models/Models/Qwen35/Qwen35Model.cs#L577) |
| Extra blocks | NextN/MTP draft block outside the main stack | `_numNextnLayers`, `_mtpLayerIdx` |
| Modality | Optional vision encoder | `Qwen35VisionEncoder.cs` |

The three genuinely new problems versus Qwen3:

1. **GDN carries mutable recurrent state.** The KV cache is append-only — a
   rank that computes a slightly wrong value corrupts one row. The GDN delta
   state (`_deltaStateTensor[l]`, `[numVHeads, headVDim, headKDim]`) and conv
   ring buffer (`_convState[l]`) are **read-modify-write every token**, so any
   per-rank divergence compounds geometrically and silently.
2. **The K→V head mapping is interleaved, not blocked.** Every GDN kernel
   computes `src_h = h % num_k_heads`
   ([tensorsharp_kernels.cu:913](TensorSharp.Backends.Cuda/native/kernels/tensorsharp_kernels.cu#L913),
   `:1073`, `:4655`; mirrored in managed code at
   [Qwen35Model.GatedDeltaNet.cs:2809](TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs#L2809)).
   A naive contiguous split of V heads forces **full replication of all K heads** —
   see §3.2. This is the single most important detail in the design.
3. **The GDN block has five execution paths** (fused dense FFN, fused MoE router,
   chunked prefill, CUDA-native, managed per-token, MLX). TP must not try to
   support all five.

---

## 2. The load-bearing observation

**GatedDeltaNet has no cross-head reduction anywhere.** Inspecting the kernel:

- One CUDA block per V head (`int h = blockIdx.x`), state staged into shared
  memory per head, written back per head.
- `dt_bias[h]`, `a_log[h]` — indexed per V head.
- `ssm_norm[r]` where `r < head_v_dim` — **shared across heads**, so replicated.
- The RMS reduction (`warp_allreduce_sum`) is over `head_v_dim` *within one head*.
- The `conv1d` is depthwise — per-channel, no channel mixing.
- L2-normalisation of q/k is per head.

So the entire GDN mixer is embarrassingly head-parallel. In one respect it is
*easier* to shard than GQA attention: because the recurrent state is keyed by
**V** head, there is no "replicate KV heads when `numKVHeads < tp`" problem.

The mixer is therefore a textbook Megatron block:

```
replicated hidden
      │
      ▼  RMSNorm (replicated)
  ssm_in_proj      ← COLUMN-parallel (packed; see §3.1)
      │
      ▼  per-rank GDN: conv1d → L2norm → delta-rule scan → gated RMSNorm
      │              (own conv state + own delta state, never communicated)
  ssm_out           ← ROW-parallel + AllReduce
      │
      ▼
replicated hidden
```

**One AllReduce per mixer, identical to Qwen3.** The recurrent state itself is
never communicated — each rank owns the state for its own V heads for the whole
sequence. This is the property that makes the whole thing viable.

---

## 3. Sharding the GDN layer

### 3.1 The packed `ssm_in_proj` layout

`FuseRecurrentInputWeights` ([Qwen35Model.cs:463](TensorSharp.Models/Models/Qwen35/Qwen35Model.cs#L463))
fuses four projections into one weight, and the kernel's offset arithmetic
(`:911–918`) pins the layout to:

```
packed_dim = qkv_dim + v_dim + 2*nV
             where qkv_dim = 2*qk_dim + v_dim, qk_dim = nK*dK, v_dim = nV*dV

┌──────────┬──────────┬──────────┬──────────┬────────┬─────────┐
│ Q        │ K        │ V        │ Z (gate) │ beta   │ alpha   │
│ nK·dK    │ nK·dK    │ nV·dV    │ nV·dV    │ nV     │ nV      │
└──────────┴──────────┴──────────┴──────────┴────────┴─────────┘
   ↑contiguous↑          ↑─────── strided by V head ────────↑
```

A TP shard must take a **contiguous** slice of Q and K but a **strided** gather
of V/Z/beta/alpha. That is why `ShardWeightsForTensorParallelism`
([ModelBase.cs:2215](TensorSharp.Models/ModelBase.cs#L2215)) cannot be reused
unmodified — its column path assumes one contiguous run of output rows.

### 3.2 Head assignment: block-cyclic on V, contiguous on K

Write `h = j·nK + i` with `i = h % nK` (the K head) and `j ∈ [0, nV/nK)`.

- **Naive contiguous V split is wrong.** For `nV=32, nK=16, tp=2`, rank 0 gets
  V heads `[0,16)`, whose `h % 16` covers *all 16* K heads. Both ranks then need
  every K head — no K sharding, wasted bandwidth and memory.
- **Correct scheme:** rank `r` owns K heads `[r·nK/tp, (r+1)·nK/tp)` and the V
  heads that map to them, i.e. `{ h : (h % nK) ∈ that range }`. This is a
  block-cyclic set in `h`, stride `nK`, block width `nK/tp`.

The payoff is that this scheme is **invariant under the kernel's own indexing**.
Enumerate rank `r`'s V heads in increasing `h`: they arrive in groups of
`nK/tp`, and group `j` uses local K heads `0 .. nK/tp-1`. So
`local_h % (nK/tp)` yields exactly the right local K head.

> **The existing GDN kernels need no modification.** Launch them per rank with
> `num_v_heads = nV/tp`, `num_k_heads = nK/tp`, `qk_dim`, `v_dim`, `qkv_dim`,
> `packed_dim` all recomputed for the shard, and every offset in the kernel
> stays correct. The kernel is already fully parameterised by these values.

This is the difference between "a week of CUDA work" and "a weight-gather
function". It is worth writing a unit test that asserts this mapping property
directly, independent of any GPU.

### 3.3 Per-weight shard map

| Weight | Shape | Treatment |
|--------|-------|-----------|
| `ssm_in_proj.weight` (fused) | `[packed_dim, hidden]` | **Column, segmented**: contiguous for Q/K, block-cyclic for V/Z/beta/alpha |
| `ssm_conv1d.weight` | `[qkv_dim, convKernel]` | Shard dim 0 with the same Q\|K\|V segmentation (depthwise) |
| `ssm_dt_bias`, `ssm_a` | `[nV]` | Block-cyclic per V head |
| `ssm_norm.weight` | `[headVDim]` | **Replicated** (shared across heads) |
| `ssm_out.weight` | `[hidden, v_dim]` | **Row**-parallel; input rows permuted to match the V-head shard |
| `attn_norm`, `post_attn_norm` | `[hidden]` | Replicated |

The `ssm_out` row split is the subtle one: its input dim is indexed by
`(v_head, v_dim)`, so its column ordering must use the *same* block-cyclic
permutation as the V shard, otherwise the AllReduce sums mismatched channels.
One helper should compute the permutation once and drive both.

### 3.4 Per-rank recurrent state

```csharp
private Tensor[][] _tpDeltaState;   // [layer][rank] : [nV/tp, headVDim, headKDim]
private Tensor[][] _tpConvState;    // [layer][rank] : [convKernel-1, qkvDim/tp]
private int[]      _tpConvWriteIdx; // [layer] — identical on all ranks, no per-rank copy
```

`_convStateWriteIdx` advances deterministically with token count, so it stays a
single shared value. `Reset`/`DisposeGdnState`/`InitGdnLayerCache` grow a
rank loop. Note the existing managed path keeps `_convState` as a **host**
`float[]` — TP should not use that path at all (§5).

---

## 4. Sharding the MoE layer

**Recommendation: tensor-parallel experts, not expert parallelism.**

Rank `r` holds a `1/tp` *slice of every expert* rather than a whole subset of
experts:

| Weight | Treatment |
|--------|-----------|
| `ffn_gate_inp.weight` (router) | **Replicated**, evaluated identically on every rank |
| `ffn_gate_exps.N` / `ffn_up_exps.N` | Column-parallel (split `_expertFfnLength`) |
| `ffn_down_exps.N` | Row-parallel |
| `ffn_*_shexp` (shared experts) | Same column/row split on `_sharedExpertFfnLength` |
| `ffn_gate_inp_shexp` | Replicated |

Why this over classic expert parallelism:

- **No dynamic communication.** EP needs an all-to-all token dispatch whose
  volume depends on routing; TP-MoE needs the same single AllReduce a dense
  layer needs. The current `CudaP2PCommunicator` does reduce-to-0-plus-broadcast
  and has no all-to-all primitive — EP would require building one.
- **Perfect load balance by construction.** EP suffers when a hot expert lands
  on one rank; TP-MoE has every rank do exactly `1/tp` of the work regardless of
  routing.
- **Identical memory saving.** Each rank still holds `1/tp` of the expert
  weights.
- **Reuses `TpColumnParallelLinear`/`TpRowParallelLinear` verbatim** per selected
  expert.

The cost is that each rank touches all top-k experts (at `1/tp` width) instead of
a subset at full width — same total FLOPs, same per-rank FLOPs, more kernel
launches. For `numExpertsUsed` in the 8–10 range that is an acceptable trade,
and EP remains available later as an optimisation if launch overhead dominates.

**Correctness invariant to enforce:** every rank must select the *same* top-k.
This holds only because the current AllReduce is reduce-to-rank-0 followed by
broadcast, so the post-AllReduce hidden state is **bitwise** identical on all
ranks. If the communicator is ever changed to a ring/butterfly AllReduce (where
each rank sums in a different order), routers can diverge on near-tie logits and
ranks will silently compute different experts. This must be a documented,
asserted invariant on `CudaP2PCommunicator` — it is the kind of bug that shows
up as "slightly worse quality at tp=4" and takes a week to find.

A cheap debug guard: under `TS_TP_VERIFY=1`, hash the selected expert indices per
layer on every rank and compare.

---

## 5. Scope: one execution path only

`RecurrentBlock` ([Qwen35Model.GatedDeltaNet.cs:731](TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs#L731))
currently has five paths. Supporting all of them under TP is where this task
becomes a multi-month project. Proposal:

| Path | TP support |
|------|-----------|
| CUDA-native GDN (`TS_CUDA_QWEN35_GDN_NATIVE`) | ✅ **The only supported path** |
| GGML fused dense FFN / fused MoE router | ❌ GGML backend is single-device by design |
| Fused recurrent prefill (`TS_QWEN35_FUSED_REC_PREFILL`) | ❌ GGML-only |
| Managed per-token loop | ❌ host-side state, would need host round-trips per rank per token |
| MLX | ❌ Apple-only, single device |

TP already requires `backend == Cuda` ([ModelBase.cs:485](TensorSharp.Models/ModelBase.cs#L485)),
which excludes the GGML/MLX paths automatically. The constructor should
**throw** if `tpDegree > 1` and the CUDA-native GDN path is disabled, rather than
silently falling back to a path that would produce wrong results.

---

## 6. Constraints to validate at construction

```
nKHeads            % tp == 0        // GDN K heads
nVHeads            % tp == 0        // GDN V heads and recurrent state
NumHeads           % tp == 0        // full-attention layers
NumKVHeads         % tp == 0
IntermediateSize   % tp == 0        // dense FFN layers
_expertFfnLength   % tp == 0        // MoE experts
_sharedExpertFfnLength % tp == 0    // shared experts
(nVHeads / nKHeads) integral        // model invariant, assert it
quantized row-parallel: ne0 % (tp * blockSize) == 0
```

For a typical `nV=32, nK=16` configuration this admits `tp ∈ {1,2,4,8,16}`.
Fail fast with a message naming the offending dimension — a silent wrong answer
here is far worse than a refusal to start.

---

## 7. Phasing

Each phase is independently shippable and independently verifiable.

**Phase 0 — Harness (do this first).**
`TS_TP_VERIFY=1` runs tp=1 and tp=N side by side on the same prompt and compares
per-layer hidden states, reporting first-divergence layer and max abs/rel delta.
This mirrors the existing `GDN_VERIFY_CHUNKED` pattern
([Qwen35Model.GatedDeltaNet.cs:92](TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs#L92)),
which is already the right idea. Without this, debugging a stateful recurrent
divergence at layer 30 of 48 is guesswork.

**Phase 1 — Head-mapping unit tests, no GPU.**
Assert the block-cyclic permutation property from §3.2 for every valid
`(nV, nK, tp)` triple; assert `local_h % (nK/tp)` recovers the correct global K
head. Pure arithmetic, runs in CI without CUDA. This locks in the design's
central claim.

**Phase 2 — Segmented sharding primitives.**
Add `ShardPackedColumnParallel(name, segments[])` to `ModelBase`, where a segment
is `(offset, elementsPerHead, headCount, stride)`. Covers the packed
`ssm_in_proj`, the conv weight, and `dt_bias`/`a` in one mechanism. Extend the
quantized path with the same block-aligned copy the row-parallel branch already
uses.

**Phase 3 — Full-attention layers only.**
Run Qwen3.5 with TP where recurrent layers execute on rank 0 replicated
(gathering/broadcasting around them). Slow, but it isolates the attention+MoE
path and proves the MoE design before GDN state enters the picture.

**Phase 4 — GDN mixer under TP.** The core work. Phase 0's harness carries this.

**Phase 5 — MoE under TP**, with the router-determinism guard from §4.

**Phase 6 — MTP/NextN draft block.** Shard like a full-attention layer.
`Qwen35Model.Mtp.cs` has its own forward; defer until the main stack is green.

**Deferred indefinitely:** vision encoder (replicate on rank 0 — it runs once per
image, not per token, so TP buys almost nothing), batched/continuous-batching TP
(`BatchExecutor` is single-GPU today — already a known Stage 1 limitation),
CUDA graph capture under TP.

---

## 8. Honest risk assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Router divergence across ranks on near-tie logits | **High** — silent quality loss | §4 invariant + `TS_TP_VERIFY` expert-index hash |
| Recurrent state divergence compounding over a long generation | **High** — looks like "gets incoherent after 500 tokens" | Phase 0 harness; compare state tensors, not just logits |
| `ssm_out` column permutation mismatched with V shard | **High** — plausible-looking garbage | Derive both from one permutation helper; Phase 1 test |
| AllReduce latency: 2 per layer × ~48 layers at decode | Medium — may erase the speedup at tp=2 on PCIe | Measure before optimising; NCCL is already on the Stage 1 TODO list |
| Quantized block alignment on the strided V gather | Medium — throws at load | Validate in §6 checks |
| Five-path `RecurrentBlock` complexity | Medium | §5: support exactly one, throw otherwise |

The latency risk deserves emphasis. Qwen3.5's appeal is cheap long-context
decode via linear attention; if per-layer AllReduce latency dominates, TP could
be *slower* than single-GPU for models that already fit in one card. **The
honest success criterion for this work is enabling models that do not fit at
all, not accelerating ones that do.** Worth benchmarking a 2-GPU tp=2 run
against 1-GPU on a model that fits both ways, early, and being willing to
report that it is slower.

---

## 9. Estimated footprint

| File | Change |
|------|--------|
| `Qwen35Model.cs` | `tpDegree` ctor param, `IsTensorParallel` dispatch, TP constraint validation |
| `Qwen35Model.TensorParallel.cs` | **New.** `ForwardTP`, per-layer-type dispatch, TP state init/dispose |
| `Qwen35Model.GatedDeltaNet.TensorParallel.cs` | **New.** `RecurrentBlockTP`, per-rank state, block-cyclic permutation helper |
| `ModelBase.cs` | `ShardPackedColumnParallel`, permutation helper, TP-MoE expert linear helpers |
| `CudaP2PCommunicator.cs` | Document + assert the bitwise-determinism invariant |
| `ModelBase.Create` | Pass `tpDegree` to the `qwen35`/`qwen35moe`/`qwen3next` arm ([ModelBase.cs:4280](TensorSharp.Models/ModelBase.cs#L4280)) |
| `InferenceWeb.Tests` | Phase 1 head-mapping tests; sharding round-trip tests |

Roughly comparable in size to the Qwen3 TP work plus the MoE helpers — the GDN
mixer itself is *smaller* than `Qwen3Model.TensorParallel.cs`'s attention path,
because there is no KV cache expansion, no RoPE per rank, and no GQA head
replication.
