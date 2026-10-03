# DeepSeek V4.1 Flash at Q4_K_M: throughput work on eight A40s

Measured 2026-09-12 on the eight-A40 benchmark VM against
`DeepSeek-V4.1-Flash-Q4_K_M` (414.2 GiB, eleven shards) on a MooseFS mount.
Every prefill/decode row is a cold-prompt measurement: each repeat sends a
different prompt body, so its n-grams select Engram rows that run has not
touched. Re-sending one prompt measures the page-cache-warm path only, which is
not what a server sees.

## Result

| | before | after |
|---|---:|---:|
| Prefill, 4,924-token prompt (tok/s) | 199.7 / 252.5 / 250.9 | 451.8 / 463.9 / 492.1 |
| Decode, single stream (tok/s) | 23.5 / 25.8 / 25.9 | 31.9 / 31.0 / 32.5 |
| Decode aggregate, 2 concurrent (tok/s) | 24.8 | 39.3 |
| Decode aggregate, 4 concurrent (tok/s) | 24.3 | 48.9 |
| Decode aggregate, 8 concurrent (tok/s) | 26.5 | 48.5 |
| Four concurrent 10,836-token documents | server aborted | 4/4 answered |
| Routed-expert layers kept on the host | 3 of 40 | 1 of 40 |

**1.9x prefill, 1.3x single-stream decode, 2.0x aggregate decode at four
concurrent requests**, and a crash under concurrent long prompts is gone.

Both columns are the same binary on the same host, same checkpoint and same
profile (`MAX_CONTEXT` 65536, `TS_DSV4_UBATCH` 1024, F16 KV,
`TS_SCHED_MAX_RUNNING_SEQS` 4, `TS_SCHED_PREFILL_CHUNK` 1024, sparse flash
attention and compact raw gather on). "before" sets
`TS_DSV41_ENGRAM_WARM=0`, `TS_DSV4_VRAM_RESERVE_MB=5240`,
`TS_BATCHED_FUSED_DECODE=0` and `TS_DSV4_GRAPH_CACHE_HEADROOM_MB=0`, which
reproduces the previous behaviour on one build.

## Where the time went

From `TS_DSV4_PERF=2`, per 1024-token prefill chunk and per decoded token:

| | before | after |
|---|---:|---:|
| Prefill input preparation | 1,435-2,565 ms | 56-81 ms |
| Prefill graph compute | 2,326-2,750 ms | 2,139-2,154 ms |
| Decode input preparation | 5-15 ms | 0.86-0.95 ms |
| Decode graph compute | 36-39 ms | 35.6-35.9 ms |
| Scheduler splits per prefill chunk | 14 | 10 |

Input preparation is the host Engram lookup. Graph compute and the split count
move with how many layers keep their routed experts on the host.

## The four changes

### 1. Host Engram tables are warmed automatically

At Q4_K_M the two Engram tables are `Q4_K` at 51.5 GiB each, so unlike Q2_K they
do not fit the devices and the loader keeps both as host mappings. Each token
then hashes 24 row ids per table, and a row that is not already page cache is
one storage round trip — about a millisecond on this mount. That is the
1,435-2,565 ms of input preparation above, and it is 37% of a prefill chunk.

Warming the mapping removes it. That was already implemented and opt-in
(`TS_DSV41_ENGRAM_WARM=1`), and it cost 110-210 s of startup, so nobody turned it
on. It is now the default and runs on its own thread once the model is serving:
startup is unchanged, requests work (more slowly) while it proceeds, and it is
skipped with a diagnostic when the host could not keep the tables cached anyway.

A model load reads the whole checkpoint through the page cache and evicts the
previous run's Engram pages, which is why this has to happen after every load
rather than once per machine.

The sparse-read mapping advice (`MADV_RANDOM`) now waits for warming to finish.
It turns off exactly the readahead a warm pass depends on, so applying it first
would have made a cold warm pass fault one 4 KiB page at a time.

### 2. The device-memory reserve was holding back twice what it needs

The layer-split packer leaves per-device headroom for the transients a graph
needs at run time — mainly the lightning indexer's top-k, whose CUB sort
workspace comes from the CUDA VMM pool rather than from any graph buffer. The
estimate multiplied that term by four, partly because the reserve also had to
cover a graph cache capped by entry *count* (see below).

Measured at `TS_DSV4_UBATCH` 1024 and a 65,536-token context: the largest
graph's compute buffers are 1,336 MiB on a device, and a 57,424-token prefill
peaks with 1,522 MiB still free on the tightest device against a 3,072 MiB
reserve. The multiplier is now 1.25, which lands at 3,128 MiB.

That is not cosmetic. 5,240 MiB forced three layers of routed experts onto the
host; 3,128 MiB needs one. Each host layer costs a CPU `mul_mat_id` over 128.6
MiB of expert rows on every token.

### 3. Concurrent long prompts no longer kill the server

Four concurrent 10,836-token prefills used to end like this:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 165.36 MiB on device 1: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate CUDA1 buffer of size 173394304
```

and then the process was gone. This is not a survivable error in ggml's
allocator: `ggml_gallocr_reserve_n_impl` frees the old buffer before allocating
the new one, so a failed allocation leaves a null buffer behind, and the retry
inside `ggml_backend_sched_alloc_graph` dereferences it. The request does not
fail — the server dies.

The cause was the graph cache. It was capped at twelve entries, which is blind
to what an entry costs: a decode graph's compute buffers are ~2 MiB and a
1024-token prefill graph's are 464-1,336 MiB per device, and concurrent
sequences sitting at different positions produce many distinct shapes. Twelve
prefill entries do not fit any headroom.

The cache is now bounded by bytes as well. Before building a new entry,
least-recently-used entries are freed until every device has room for another
entry as large as the largest one cached, plus a floor
(`TS_DSV4_GRAPH_CACHE_HEADROOM_MB`, default 1024). Devices are drained before
anything is freed, because a pipelined prefill submits without waiting.

Same test after: 4/4 documents answered with their own secret, no answer
containing another document's secret, 18 cache trims, zero allocation failures.

### 4. Token-batched decode for V4.1

`TSGgml_Dsv4ForwardBatchedDecode` refused V4.1 outright — "until its
token-batched graph is implemented, use the caller's per-slot forward" — so four
concurrent sequences ran four separate single-token forwards and four concurrent
requests were no faster in aggregate than one.

V4.1 now has that graph. The dense projections, the compressor and indexer
projections, the whole MoE, the shared expert and the output head run once over
all N tokens; only the parts that touch a sequence's own state fork per slot:
the sliding-window ring write, the compressor state read/write, the lightning
indexer's selection and top-k, the sparse gather and attention itself. The
Engram lookup needs no fork — its staged rows are already one column per token,
so a slot is just a column, hashed against that slot's own history.

The saving is bounded by routing. Each token picks its own 6 of 384 experts, so
the routed-expert reads do not overlap between slots; only the ~98 MiB a layer
of dense and shared weights and the 543 MiB output head are shared. That is why
four concurrent requests are 2.0x and not 4x, and why eight are no better than
four.

## What a decode step is actually made of

`TS_DSV4_PERF=3` now reports how a device subgraph is submitted. A single-token
V4.1 decode graph is 3,235-3,352 nodes, 2,194-2,269 of which compute something
(~55 a layer), submitted as 59 device subgraphs, 331-337 forwarded runs of
ordinary nodes and 400-406 fused kernel launches. The submitting thread spends
3.6-3.7 ms on all of that and then waits ~31 ms, so the step is bound by the
device, not by submission.

One token's weights are 9.78 GiB at Q4_K_M — 128.6 MiB of routed experts, 21.4
MiB of shared expert and 76.5 MiB of attention per layer, plus a 543 MiB output
head — which is 14.0 ms at the A40's 696 GB/s. Summed GPU utilization across the
eight cards during decode is 58%, so about 40% of the step has no kernel running
at all. The gap is the floor under ~2,200 small kernels, not bandwidth. Raising
single-stream decode further needs fewer kernels (CUDA-graph capture of a whole
device subgraph, or fusing the 235 concat/cont copies a token spends on rotary
splits), not fewer bytes.

## Tensor parallelism

`--tp 8` shards the routed-expert gate/up/down matrices along the FFN
intermediate dimension. At Q4_K_M it removes the capacity cliff completely —
38.3 GiB of shards a rank and zero CPU-offloaded layers — and is still slower:

| eight A40s, Q4_K_M | Prefill tok/s | Decode tok/s | CPU-MoE layers |
|---|---:|---:|---:|
| Layer split (default) | 451.8-492.1 | 31.0-32.5 | 1 |
| `--tp 8` | 391.9-410.4 | 21.4-22.0 | 0 |

Attention, the shared expert and the caches keep their layer placement, and the
partial sums reduce through host-staged F32 buffers. Decode splits rise from 62
to 132 a step. These A40s have no NVLink and straddle two NUMA nodes, which is
the same reason `GGML_CUDA_P2P=1` is ~150x slower here. This reproduces the
earlier Q2_K conclusion on a checkpoint where TP has a real placement advantage,
so it is the placement-independent result.

## Correctness

Batching changes GEMM shapes, so a batched step and a solo step are not
bit-identical and a near-tie in the logits can pick a different token. Over six
greedy prompts at 200 tokens each, four concurrent copies per prompt, solo
decode reproduced its own output 6/6 while batched output matched the solo text
2-3/6, diverging mid-answer and continuing coherently.

That is the graph's shape, not the per-slot wiring, and the discriminating test
says so: running the **same** prompt at batch width 2 and at batch width 4 —
both the batched path, identical wiring, only the GEMM widths differ — the two
widths disagree on 1 of 4 prompts, the same rate as batched-against-solo. Output
therefore depends on which requests happen to share a step, which is a property
of batched serving rather than of this graph. `TS_BATCHED_FUSED_DECODE=0`
restores the serial path and its determinism.

The test that matters more is per-slot state separation: four concurrent
10,836-token documents, each hiding a different secret, answered 4/4 with their
own secret and none with another slot's. That exercises the per-slot sliding
window, compressed cache and sparse selection, which is what the batched graph
forks.

The managed suite is 3,897 passed / 181 skipped / 1 failed, and the one failure
(`ManagedQuantizedOpsTests.Q2KMatmul_IntegerPathIsNotSlowerThanDequantizing`, a
timing assertion in the managed Q2_K kernels) fails identically without these
changes.

## Also fixed

The native macOS build was broken at `3eee2a5`: the checkpoint loader's
`TS_DSV4_LOAD_DROP_CACHE` path called `posix_fadvise(..., POSIX_FADV_DONTNEED)`
under `#if !defined(_WIN32)`, and macOS does not declare that constant. It now
uses `posix_fadvise` on Linux, whole-descriptor `F_NOCACHE` on Apple, and warns
once elsewhere.
