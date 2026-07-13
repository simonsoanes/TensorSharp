// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#include "ggml_ops_internal.h"
#include "ggml_ops_transformer_common.h"
#include <chrono>
#include <cstdio>

using namespace tsg;

// ============================================================================
// TSGgml_Qwen35ModelVerify  --  the N-token sibling of TSGgml_Qwen35ModelDecode.
//
// MTP speculative decoding verifies a window of (1 + draft) tokens in one trunk
// pass. The per-op SpecForward fallback runs that pass op-by-op (~1 s/step on
// WDDM ggml_cuda); this kernel runs the WHOLE hybrid transformer over N tokens as
// ONE ggml graph: prefill-style causal flash attention (read cache prefix [0,
// start_pos) + the N fresh K/V, causal F16 mask, append the N rows to the cache),
// GDN recurrence over the N tokens via ggml_gated_delta_net(K=N) (the op also
// emits a state snapshot per prefix length so partial-acceptance rollback can pick
// the committed state without a re-forward), batched MoE/dense FFN, and a folded
// final-norm that outputs BOTH the per-row logits [vocab, N] AND the post-norm
// hidden [hidden, N] the MTP draft head consumes.
//
// GDN state is passed per-layer via host in/out pointers (conv_state_in/out in
// ggml [time, channel] layout, delta_state_in/out as [S, S, H]); the kernel reads
// the pre-window state and writes the post-window state. Non-persistent (rebuilt
// per call) -- correctness first; CUDA-graph capture is a follow-up that needs a
// fixed-N persistent cache (cf. g_q35dc). Returns 0 on anything it cannot handle
// so the C# caller falls back to the per-op SpecForward.
// ============================================================================
namespace
{
    // Persistent verify-graph cache (multi-entry, keyed by N + window stride). With
    // TS_Q35_VERIFY_PERSIST (default ON) the per-(N,window) graph is built ONCE in a
    // raw ggml ctx with ggml_backend_alloc_ctx_tensors (each tensor its own slot =
    // stable addresses), so subsequent verify steps of the same shape only upload the
    // per-call inputs (hidden / pos / kv_index / mask / GDN state) + recompute -> the
    // ~150-580 ms C++ build is amortized AND ggml-cuda can CUDA-graph-capture the
    // replay. Multi-entry because spec alternates N=draft+1 (speculative) and
    // N=accepted+1 (rollback re-forward); a single entry would evict + rebuild every
    // call. Small GDN weights (dt_bias/ssm_a/...) get their own stable alloc_ctx slot
    // uploaded once, so they don't go stale on reuse (the gallocr-recycle bug).
    struct Q35VerifyCache
    {
        bool valid = false;
        int n = 0, window = 0, num_layers = 0, out_vocab = 0, n_logits = 0;
        bool has_normed = false;
        const void* sig = nullptr;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* pos_t = nullptr;
        ggml_tensor* kv_index = nullptr;
        ggml_tensor* mask_t = nullptr;
        ggml_tensor* logits_out = nullptr;
        ggml_tensor* normed_out = nullptr;
        std::vector<ggml_tensor*> conv_in, delta_in, conv_out, delta_out;
        std::uint64_t lru = 0;
        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_t = pos_t = kv_index = mask_t = logits_out = normed_out = nullptr;
            conv_in.clear(); delta_in.clear(); conv_out.clear(); delta_out.clear();
            n = window = num_layers = out_vocab = n_logits = 0; has_normed = false; sig = nullptr;
        }
    };
    Q35VerifyCache g_q35vc[16];
    std::uint64_t g_q35vc_clock = 0;

    void fill_verify_causal_mask(std::vector<ggml_fp16_t>& mask, int window, int n, int start_pos, int total_len)
    {
        mask.resize(static_cast<std::size_t>(window) * n);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
        for (int qi = 0; qi < n; qi++)
        {
            int threshold = start_pos + qi;
            ggml_fp16_t* row = &mask[static_cast<std::size_t>(qi) * window];
            for (int ki = 0; ki < window; ki++)
                row[ki] = (ki <= threshold && ki < total_len) ? zero_val : neg_inf;
        }
    }

    int qwen35_model_verify_impl(
        const TSGgmlQwen35LayerDesc* layers, int num_layers,
        void* hidden_data, int hidden_size, int start_pos, int num_tokens,
        int num_heads, int num_kv_heads, int head_dim, int cache_size,
        int rope_n_dims, int rope_mode, int kv_cache_type,
        int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
        float eps, float rope_base, float rope_freq_scale,
        int num_experts, int num_experts_used, int expert_ff, int shared_ff,
        int norm_topk, float expert_weights_scale,
        void* logits_data, int vocab_size,
        const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
        const void* final_norm_data, void* normed_out, int n_logit_rows,
        const std::int32_t* mrope_pos, const std::int32_t* mrope_sections)
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr || num_tokens < 1)
        {
            set_last_error("Qwen3.5 model verify: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen35LayerDesc)))
        {
            set_last_error("Qwen3.5 model verify: descriptor size mismatch.");
            return 0;
        }
        // gated_delta_net requires S_k == S_v (state is [S_v, S_v, H]).
        if (head_k_dim != head_v_dim)
        {
            set_last_error("Qwen3.5 model verify: head_k_dim != head_v_dim unsupported.");
            return 0;
        }

        const int N = num_tokens;
        const int H = hidden_size;
        const int totalSeqLen = start_pos + N;
        const int qDim = num_heads * head_dim;
        const int qFullDim = qDim * 2;            // Q + gate interleaved per head
        const int kDim = num_kv_heads * head_dim;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const int head_tile = (num_k_heads > 0) ? (num_v_heads / num_k_heads) : 1;
        const ggml_type kvType = static_cast<ggml_type>(kv_cache_type);
        if (convDim <= 0 || totalSeqLen > cache_size)
        {
            set_last_error("Qwen3.5 model verify: bad conv dim or sequence exceeds cache.");
            return 0;
        }
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        if (!fold)
        {
            set_last_error("Qwen3.5 model verify: folded lm_head required.");
            return 0;
        }

        // Prefill only needs the LAST token's logits (to sample the first decode
        // token); MTP verify needs all N rows. n_logit_rows in [1, N) computes the
        // lm_head only over the last n_logit_rows columns of the post-norm hidden,
        // so a 2048-token prefill writes vocab*1 floats (not vocab*2048 ~ 2 GB) and
        // skips the lm_head matmul over the first N-1 tokens. <=0 or >=N => all N.
        const int n_logits = (n_logit_rows > 0 && n_logit_rows < N) ? n_logit_rows : N;

        // Persistent per-(N,window) graph cache (build amortization + CUDA-graph
        // capture). DEFAULT ON: the earlier reuse access-violation (0xC0000005) was the
        // 3D N-row set_rows (heads in ne2) faulting on cgraph reuse; replacing it with a
        // 2D set_rows PER HEAD (llama.cpp's proven KV-write shape) made reuse stable
        // (validated 252 reuses, no crash). Reuse: setup ~8 ms + compute ~12-20 ms vs
        // ~61 ms non-persist build. TS_Q35_VERIFY_PERSIST=0 forces the rebuild path.
        static const bool fv_persist_cfg = []{ const char* e = std::getenv("TS_Q35_VERIFY_PERSIST"); return e == nullptr || e[0] != '0'; }();
        // Multimodal MRoPE: per-axis positions (T/H/W/E axis-concatenated, [4N] I32)
        // route the attention RoPE through ggml_rope_multi (interleaved MRoPE, the
        // Qwen3-VL LLM rope). Text prompts keep the plain sequential NeoX rope.
        const bool use_mrope = mrope_pos != nullptr && mrope_sections != nullptr;

        // PREFILL (n_logits < N) processes a long prompt one-shot, so it never reuses
        // the cached graph; force the NON-PERSIST path (pooled ctx + gallocr lifetime-
        // packing) which reuses activation buffers across the graph. The persist path
        // gives every intermediate its own slot (no reuse) — for a 40-layer × N-token
        // graph on the VRAM-tight 35B that thrashes WDDM paging (N=512) or OOMs (N>=1024).
        // MTP verify (n_logits == N) keeps the persist+capture fast-replay reuse.
        // MRoPE calls are prefill-only; keep them off the persist cache too.
        const bool fv_persist = fv_persist_cfg && (n_logits >= N) && !use_mrope;

        const std::size_t convStateBytes = static_cast<std::size_t>(convDim) * conv_dim * sizeof(float);
        const std::size_t deltaStateBytes = static_cast<std::size_t>(head_k_dim) * head_v_dim * num_v_heads * sizeof(float);
        constexpr int kVerifyKvStride = 256;
        // Persist mode pads the attention window to a fixed stride so one cached graph
        // serves every start_pos in that stride (the mask masks the unused tail).
        const int window = fv_persist
            ? std::min(cache_size, ((totalSeqLen + kVerifyKvStride - 1) / kVerifyKvStride) * kVerifyKvStride)
            : std::min(cache_size, totalSeqLen);
        const void* sig = layers[0].attn_norm_w;

        // Device-resident GDN state: when the C# caller points each recurrent layer's
        // conv_state_in and conv_state_out at the SAME buffer (the decode's device-
        // resident _fdConvScratch slot + _deltaStateTensor), the verify reads/writes the
        // GDN state IN-PLACE on the device (cacheable COMPUTE binding) instead of
        // uploading + downloading it every call (~60 MB delta + 3 MB conv). The state
        // persists across verify/plain steps exactly like the captured decode's; the C#
        // snapshots it (drain) only before a draft-verify for rollback.
        bool resident_state = false;
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_recurrent != 0)
            {
                resident_state = (layers[l].conv_state_in == layers[l].conv_state_out) && layers[l].conv_state_in != nullptr;
                break;
            }
        }

        // ===== Persist reuse fast-path: upload the per-call inputs + replay =====
        if (fv_persist)
        {
            for (auto& c : g_q35vc)
            {
                if (!c.valid || c.n != N || c.window != window || c.sig != sig ||
                    c.num_layers != num_layers || c.out_vocab != vocab_size ||
                    c.n_logits != n_logits ||
                    c.has_normed != (normed_out != nullptr))
                    continue;
                // llama.cpp pattern (llama-context.cpp): before re-setting the inputs of
                // a REUSED graph we must fully synchronize, else we overwrite input
                // tensors the previous (async) graph_compute is still reading -> the
                // pipeline accumulates across reuses and faults. host_read_barrier()
                // only syncs conditionally, so force a full backend sync here.
                ggml_backend_synchronize(g_backend);
                ggml_backend_tensor_set(c.hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * N * sizeof(float));
                std::vector<std::int32_t> pv(N);
                std::vector<std::int64_t> kv(N);
                for (int i = 0; i < N; i++) { pv[i] = start_pos + i; kv[i] = start_pos + i; }
                ggml_backend_tensor_set(c.pos_t, pv.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int32_t));
                ggml_backend_tensor_set(c.kv_index, kv.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int64_t));
                std::vector<ggml_fp16_t> mk;
                fill_verify_causal_mask(mk, window, N, start_pos, totalSeqLen);
                ggml_backend_tensor_set(c.mask_t, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
                // Host mode uploads the per-call GDN state; resident keeps it device-
                // resident (cacheable, in-place), so no upload/download here.
                if (!resident_state)
                {
                    int gi = 0;
                    for (int l = 0; l < num_layers; l++)
                    {
                        if (layers[l].is_recurrent == 0) continue;
                        ggml_backend_tensor_set(c.conv_in[gi], layers[l].conv_state_in, 0, convStateBytes);
                        ggml_backend_tensor_set(c.delta_in[gi], layers[l].delta_state_in, 0, deltaStateBytes);
                        gi++;
                    }
                }
                if (ggml_backend_graph_compute(g_backend, c.graph) != GGML_STATUS_SUCCESS) { c.reset(); break; }
                if (!resident_state)
                {
                    int gi = 0;
                    for (int l = 0; l < num_layers; l++)
                    {
                        if (layers[l].is_recurrent == 0) continue;
                        finalize_compute_with_download(c.conv_out[gi], layers[l].conv_state_out, convStateBytes);
                        finalize_compute_with_download(c.delta_out[gi], layers[l].delta_state_out, deltaStateBytes);
                        gi++;
                    }
                }
                if (normed_out != nullptr && c.normed_out != nullptr)
                    finalize_compute_with_download(c.normed_out, normed_out, static_cast<std::size_t>(H) * N * sizeof(float));
                finalize_compute_with_download(c.logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_logits * sizeof(float));
                host_read_barrier();
                c.lru = ++g_q35vc_clock;
                clear_last_error();
                return 1;
            }
        }

        // ===== Build a fresh graph. Persist: raw ctx + alloc_ctx_tensors (each tensor
        // its OWN slot = stable addresses, required for reuse/capture). Non-persist:
        // pooled ctx + gallocr lifetime-packing. =====
        ggml_context* ctx = nullptr;
        PooledContextHandle context;
        if (fv_persist)
        {
            ggml_init_params ip = { 32 * 1024 * 1024, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Qwen3.5 model verify: failed to init persist ctx."); return 0; }
        }
        else
        {
            if (!context.init(32 * 1024 * 1024)) { set_last_error("Qwen3.5 model verify: failed to acquire ggml context."); return 0; }
            ctx = context.value;
        }

        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
        // MRoPE positions are axis-concatenated [T0..Tn-1, H.., W.., E..] = 4N ints
        // (ggml_rope_multi asserts pos->ne[0] == 4 * a->ne[2]).
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, use_mrope ? 4 * N : N);
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        ggml_tensor* kv_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N);
        ggml_tensor* attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, window, N, 1, 1);
        ggml_set_input(hidden_t);
        ggml_set_input(pos_tensor);
        ggml_set_input(kv_index);
        ggml_set_input(attn_mask);
        std::vector<std::int64_t> kv_index_data(N);
        for (int i = 0; i < N; i++) kv_index_data[i] = start_pos + i;
        std::vector<ggml_fp16_t> attn_mask_data;
        fill_verify_causal_mask(attn_mask_data, window, N, start_pos, totalSeqLen);

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w; ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w; ggml_tensor* o_w;
            ggml_tensor* k_cache_base; ggml_tensor* v_cache_base;
            ggml_tensor* k_cpy; ggml_tensor* v_cpy;
            std::vector<ggml_fp16_t> mask_data;
            ggml_tensor* mask_t;
            // gdn
            ggml_tensor* gdn_qkv_w; ggml_tensor* gdn_gate_w;
            ggml_tensor* ssm_beta_w; ggml_tensor* ssm_alpha_w;
            ggml_tensor* conv1d_w; ggml_tensor* ssm_dt_w; ggml_tensor* ssm_a_w;
            ggml_tensor* ssm_norm_w; ggml_tensor* ssm_out_w;
            ggml_tensor* conv_state_in; ggml_tensor* delta_state_in;
            ggml_tensor* conv_state_out; ggml_tensor* delta_state_out;
            // ffn
            ggml_tensor* post_attn_norm_w; ggml_tensor* gu_w; ggml_tensor* down_w;
            ggml_tensor* gate_inp_w; ggml_tensor* gate_exps; ggml_tensor* up_exps; ggml_tensor* down_exps;
            ggml_tensor* shexp_gate_w; ggml_tensor* shexp_up_w; ggml_tensor* shexp_down_w; ggml_tensor* shexp_gate_inp_w;
        };
        std::vector<LayerTensors> lt(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            if (d.is_recurrent == 0)
            {
                t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
                if (d.separate_qkv != 0)
                {
                    t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                    t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                }
                t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
                t.k_cache_base = ggml_new_tensor_3d(ctx, kvType, head_dim, cache_size, num_kv_heads);
                t.v_cache_base = ggml_new_tensor_3d(ctx, kvType, head_dim, cache_size, num_kv_heads);
            }
            else
            {
                t.gdn_qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_qkv_type), d.gdn_qkv_ne0, d.gdn_qkv_ne1);
                t.gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_gate_type), d.gdn_gate_ne0, d.gdn_gate_ne1);
                t.ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_beta_type), d.ssm_beta_ne0, d.ssm_beta_ne1);
                t.ssm_alpha_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_alpha_type), d.ssm_alpha_ne0, d.ssm_alpha_ne1);
                t.conv1d_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
                t.ssm_dt_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_a_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
                t.ssm_out_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_out_type), d.ssm_out_ne0, d.ssm_out_ne1);
                t.conv_state_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                t.delta_state_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);
                if (resident_state)
                {
                    // Resident: conv/delta state lives in a device-resident cacheable
                    // COMPUTE buffer (bound below); the post-window state is written
                    // IN-PLACE to conv_state_in / delta_state_in (no separate out tensor,
                    // no per-call upload/download). Saves the ~60 MB delta out alloc too.
                    t.conv_state_out = nullptr;
                    t.delta_state_out = nullptr;
                }
                else
                {
                    t.conv_state_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                    t.delta_state_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);
                    // Per-call GDN state inputs: preserved (set_input) + uploaded each call.
                    ggml_set_input(t.conv_state_in);
                    ggml_set_input(t.delta_state_in);
                }
            }
            if (d.is_moe == 0)
            {
                t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
                t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            }
            else
            {
                t.gate_inp_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gate_inp_type), d.gate_inp_ne0, d.gate_inp_ne1);
                t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gate_exps_type), hidden_size, expert_ff, num_experts);
                t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.up_exps_type), hidden_size, expert_ff, num_experts);
                t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.down_exps_type), expert_ff, hidden_size, num_experts);
                t.shexp_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_gate_type), d.shexp_gate_ne0, d.shexp_gate_ne1);
                t.shexp_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_up_type), d.shexp_up_ne0, d.shexp_up_ne1);
                t.shexp_down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_down_type), d.shexp_down_ne0, d.shexp_down_ne1);
                t.shexp_gate_inp_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // --- build the chained graph over N tokens ---
        std::vector<ggml_tensor*> state_writes;
        std::vector<ggml_tensor*> kv_writes;   // per-head 2D set_rows results (all attn layers)
        ggml_tensor* hidden = hidden_t;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];

            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w); // [H, N]
            ggml_tensor* block_out;

            if (d.is_recurrent == 0)
            {
                // ===== Full attention (prefill-style causal flash attention) =====
                ggml_tensor* qg_part; ggml_tensor* k_raw; ggml_tensor* v_raw;
                if (d.separate_qkv != 0)
                {
                    qg_part = ggml_mul_mat(ctx, t.qkv_w, normed);  // [qFullDim, N]
                    k_raw = ggml_mul_mat(ctx, t.k_w, normed);      // [kDim, N]
                    v_raw = ggml_mul_mat(ctx, t.v_w, normed);
                }
                else
                {
                    ggml_tensor* qkv_out = ggml_mul_mat(ctx, t.qkv_w, normed); // [qFullDim+2kDim, N]
                    qg_part = ggml_view_2d(ctx, qkv_out, qFullDim, N, qkv_out->nb[1], 0);
                    k_raw = ggml_view_2d(ctx, qkv_out, kDim, N, qkv_out->nb[1], static_cast<std::size_t>(qFullDim) * sizeof(float));
                    v_raw = ggml_view_2d(ctx, qkv_out, kDim, N, qkv_out->nb[1], static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));
                }

                ggml_tensor* qg_4d = ggml_reshape_4d(ctx, ggml_cont(ctx, qg_part), head_dim, 2, num_heads, N);
                ggml_tensor* q_view = ggml_view_3d(ctx, qg_4d, head_dim, num_heads, N, qg_4d->nb[2], qg_4d->nb[3], 0);
                ggml_tensor* gate_view = ggml_view_3d(ctx, qg_4d, head_dim, num_heads, N, qg_4d->nb[2], qg_4d->nb[3], qg_4d->nb[1]);
                ggml_tensor* q_cont = ggml_cont(ctx, q_view);       // [head_dim, num_heads, N]
                ggml_tensor* gate_cont = ggml_cont(ctx, gate_view); // [head_dim, num_heads, N]
                ggml_tensor* k_3d_raw = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), head_dim, num_kv_heads, N);
                ggml_tensor* v_3d_raw = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), head_dim, num_kv_heads, N);

                ggml_tensor* q_norm_in = ggml_reshape_2d(ctx, q_cont, head_dim, num_heads * N);
                ggml_tensor* k_norm_in = ggml_reshape_2d(ctx, k_3d_raw, head_dim, num_kv_heads * N);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_norm_in, eps), t.q_norm_w);
                ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_norm_in, eps), t.k_norm_w);

                ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_normed, head_dim, num_heads, N, 1);
                ggml_tensor* k_4d = ggml_reshape_4d(ctx, k_normed, head_dim, num_kv_heads, N, 1);
                ggml_tensor* q_roped;
                ggml_tensor* k_roped;
                if (use_mrope)
                {
                    // Interleaved MRoPE (Qwen3-VL LLM): per-pair modality assignment
                    // comes from the GGUF's mrope sections; positions per token are
                    // the (T,H,W,E) axes uploaded into pos_tensor [4N].
                    int sections_local[4] = { mrope_sections[0], mrope_sections[1], mrope_sections[2], mrope_sections[3] };
                    q_roped = ggml_rope_multi(ctx, q_4d, pos_tensor, nullptr, rope_n_dims, sections_local, GGML_ROPE_TYPE_IMROPE, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                    k_roped = ggml_rope_multi(ctx, k_4d, pos_tensor, nullptr, rope_n_dims, sections_local, GGML_ROPE_TYPE_IMROPE, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                }
                else
                {
                    q_roped = ggml_rope_ext(ctx, q_4d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                    k_roped = ggml_rope_ext(ctx, k_4d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                }

                ggml_tensor* q_attn = ggml_permute(ctx, q_roped, 0, 2, 1, 3); // [head_dim, N, num_heads]
                ggml_tensor* k_fresh = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, k_roped, 0, 2, 1, 3)), head_dim, N, num_kv_heads);
                ggml_tensor* v_3d_pre = ggml_reshape_4d(ctx, v_3d_raw, head_dim, num_kv_heads, N, 1);
                ggml_tensor* v_fresh = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, v_3d_pre, 0, 2, 1, 3)), head_dim, N, num_kv_heads);

                // KV write: a 2D ggml_set_rows PER HEAD — dst [head_dim, cache_size]
                // (a contiguous ne2 slice of the head-major cache), src [head_dim, N],
                // idx [N]. This is EXACTLY llama.cpp's proven 2D set_rows shape
                // (n_embd_gqa == head_dim for one head). The single 3D N-row set_rows
                // (heads broadcast via ne2) is the untested combo that faults on
                // cgraph reuse (the decode is fine only because it writes N=1). The
                // write position is an I64 INPUT (kv_index), keeping the graph constant
                // within a window stride for reuse/capture.
                for (int h = 0; h < num_kv_heads; h++)
                {
                    ggml_tensor* k_dst_h = ggml_view_2d(ctx, t.k_cache_base, head_dim, cache_size,
                        t.k_cache_base->nb[1], static_cast<std::size_t>(h) * t.k_cache_base->nb[2]);
                    ggml_tensor* v_dst_h = ggml_view_2d(ctx, t.v_cache_base, head_dim, cache_size,
                        t.v_cache_base->nb[1], static_cast<std::size_t>(h) * t.v_cache_base->nb[2]);
                    ggml_tensor* k_src_h = ggml_view_2d(ctx, k_fresh, head_dim, N,
                        k_fresh->nb[1], static_cast<std::size_t>(h) * k_fresh->nb[2]);
                    ggml_tensor* v_src_h = ggml_view_2d(ctx, v_fresh, head_dim, N,
                        v_fresh->nb[1], static_cast<std::size_t>(h) * v_fresh->nb[2]);
                    kv_writes.push_back(ggml_set_rows(ctx, k_dst_h, k_src_h, kv_index));
                    kv_writes.push_back(ggml_set_rows(ctx, v_dst_h, v_src_h, kv_index));
                }

                // Attend over the fixed window [0, window) (now holds the N fresh rows);
                // the shared causal mask zeroes valid keys and -inf's the rest.
                ggml_tensor* k_full = view_kv_cache_window(ctx, t.k_cache_base, head_dim, cache_size, num_kv_heads, 0, window, kv_cache_type);
                ggml_tensor* v_full = view_kv_cache_window(ctx, t.v_cache_base, head_dim, cache_size, num_kv_heads, 0, window, kv_cache_type);
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Qwen3.5 model verify: failed to build KV cache views.");
                    return 0;
                }

                ggml_tensor* attn_flat;
                ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, attn_mask, attn_scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                if (backend_supports_op(fa))
                {
                    attn_flat = ggml_reshape_2d(ctx, fa, qDim, N);
                }
                else
                {
                    ggml_tensor* q_attn_cont = ggml_cont(ctx, q_attn);
                    ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q_attn_cont);
                    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                    ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, attn_mask, attn_scale, 0.0f);
                    ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));
                    ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, probs);
                    ggml_tensor* attn_perm = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
                    attn_flat = ggml_reshape_2d(ctx, attn_perm, qDim, N);
                }

                ggml_tensor* gate_flat = ggml_reshape_2d(ctx, gate_cont, qDim, N);
                ggml_tensor* attn_gated = ggml_mul(ctx, attn_flat, ggml_sigmoid(ctx, gate_flat));
                block_out = ggml_mul_mat(ctx, t.o_w, attn_gated); // [H, N]
            }
            else
            {
                // ===== Gated Delta Net over N tokens (one ggml_gated_delta_net, K=N) =====
                ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, t.gdn_qkv_w, normed);  // [conv_dim, N]
                ggml_tensor* z_all = ggml_mul_mat(ctx, t.gdn_gate_w, normed);     // [value_dim, N]
                ggml_tensor* beta_all = ggml_sigmoid(ctx, ggml_mul_mat(ctx, t.ssm_beta_w, normed)); // [num_v_heads, N]
                ggml_tensor* alpha_all = ggml_mul_mat(ctx, t.ssm_alpha_w, normed); // [num_v_heads, N]
                ggml_tensor* g_all = ggml_softplus(ctx, ggml_add(ctx, alpha_all, t.ssm_dt_w));
                g_all = ggml_mul(ctx, g_all, t.ssm_a_w); // [num_v_heads, N]

                // conv over the N new timesteps prepended with the conv ring state.
                ggml_tensor* qkv_T = ggml_cont(ctx, ggml_transpose(ctx, qkv_mixed));  // [N, conv_dim]
                ggml_tensor* conv_input = ggml_concat(ctx, t.conv_state_in, qkv_T, 0); // [convDim+N, conv_dim]
                ggml_tensor* conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, t.conv1d_w)); // [conv_dim, N]
                // new conv state = the last convDim timesteps (rows [N, N+convDim)).
                ggml_tensor* new_conv = ggml_cont(ctx, ggml_view_2d(ctx, conv_input, convDim, conv_dim, conv_input->nb[1], static_cast<std::size_t>(N) * conv_input->nb[0]));
                // Resident: write the post-window conv state IN-PLACE to conv_state_in
                // (the device-resident buffer); host mode: to the separate out tensor.
                t.conv_state_out = ggml_cpy(ctx, new_conv, resident_state ? t.conv_state_in : t.conv_state_out);

                ggml_tensor* q_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, key_dim, N, conv_out->nb[1], 0));
                ggml_tensor* k_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, key_dim, N, conv_out->nb[1], static_cast<std::size_t>(key_dim) * sizeof(float)));
                ggml_tensor* v_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, value_dim, N, conv_out->nb[1], static_cast<std::size_t>(2 * key_dim) * sizeof(float)));

                // l2-norm over head_k_dim, then tile q/k from num_k_heads -> num_v_heads.
                ggml_tensor* q_hn = ggml_l2_norm(ctx, ggml_reshape_2d(ctx, q_part, head_k_dim, num_k_heads * N), eps);
                ggml_tensor* k_hn = ggml_l2_norm(ctx, ggml_reshape_2d(ctx, k_part, head_k_dim, num_k_heads * N), eps);
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_hn, head_k_dim, num_k_heads, N);
                ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_hn, head_k_dim, num_k_heads, N);
                ggml_tensor* q_tl = q_3d; ggml_tensor* k_tl = k_3d;
                for (int r = 1; r < head_tile; r++) { q_tl = ggml_concat(ctx, q_tl, q_3d, 1); k_tl = ggml_concat(ctx, k_tl, k_3d, 1); }
                ggml_tensor* q4 = ggml_reshape_4d(ctx, ggml_cont(ctx, q_tl), head_k_dim, num_v_heads, N, 1);
                ggml_tensor* k4 = ggml_reshape_4d(ctx, ggml_cont(ctx, k_tl), head_k_dim, num_v_heads, N, 1);
                ggml_tensor* v4 = ggml_reshape_4d(ctx, v_part, head_v_dim, num_v_heads, N, 1);
                ggml_tensor* g4 = ggml_reshape_4d(ctx, ggml_cont(ctx, g_all), 1, num_v_heads, N, 1);
                ggml_tensor* beta4 = ggml_reshape_4d(ctx, ggml_cont(ctx, beta_all), 1, num_v_heads, N, 1);
                ggml_tensor* state4 = ggml_reshape_4d(ctx, t.delta_state_in, head_k_dim, head_v_dim, num_v_heads, 1);

                // K=1: the op recurs over all N tokens internally and emits the per-
                // token outputs (rows [0,N)) + ONLY the FINAL state snapshot (we roll
                // back via host snapshot/re-forward, not the per-prefix snapshots, so
                // requesting K=N would waste ~19 MB/layer of VRAM on unused states).
                ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g4, beta4, state4, 1);
                // Per-token outputs occupy the first N rows ([S_v*H] each).
                ggml_tensor* gdn_out = ggml_view_2d(ctx, gdn, value_dim, N, ggml_row_size(gdn->type, value_dim), 0);
                // Final state snapshot (slot 0, most-recent) at offset N * (S_v*H).
                ggml_tensor* new_state = ggml_view_4d(ctx, gdn, head_k_dim, head_v_dim, num_v_heads, 1,
                    ggml_row_size(gdn->type, head_k_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim * num_v_heads),
                    ggml_row_size(gdn->type, value_dim) * static_cast<std::size_t>(N));
                // Resident: write the post-window delta state IN-PLACE to delta_state_in
                // (state4 aliases it); host mode: to the separate out tensor.
                t.delta_state_out = ggml_cpy(ctx, new_state, resident_state ? state4 : t.delta_state_out);

                // gated RMSNorm with z, per token: rms_norm(out) * ssm_norm * silu(z).
                ggml_tensor* out_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, gdn_out), head_v_dim, num_v_heads * N);
                ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_2d, eps), t.ssm_norm_w);
                ggml_tensor* out_n_3d = ggml_reshape_3d(ctx, out_n, head_v_dim, num_v_heads, N);
                ggml_tensor* z_3d = ggml_reshape_3d(ctx, z_all, head_v_dim, num_v_heads, N);
                ggml_tensor* gated = ggml_mul(ctx, out_n_3d, ggml_silu(ctx, z_3d));
                ggml_tensor* gated_flat = ggml_reshape_2d(ctx, gated, value_dim, N);
                block_out = ggml_mul_mat(ctx, t.ssm_out_w, gated_flat); // [H, N]

                state_writes.push_back(t.conv_state_out);
                state_writes.push_back(t.delta_state_out);
            }

            ggml_tensor* residual1 = ggml_add(ctx, hidden, block_out); // [H, N]

            // ===== FFN =====
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.post_attn_norm_w); // [H, N]
            ggml_tensor* ffn_out;
            if (d.is_moe == 0)
            {
                const std::int64_t ffDense = d.ff_dense;
                ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed); // [2*ffDense, N]
                ggml_tensor* g_part = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, N, gu->nb[1], 0));
                ggml_tensor* u_part = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, N, gu->nb[1], static_cast<std::size_t>(ffDense) * sizeof(float)));
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_part), u_part);
                ffn_out = ggml_mul_mat(ctx, t.down_w, act); // [H, N]
            }
            else
            {
                ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, ffn_normed); // [num_experts, N]
                ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
                ggml_tensor* sel = ggml_top_k(ctx, probs, num_experts_used);              // [num_used, N]
                ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, num_experts, N);
                ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                         // [1, num_used, N]
                ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, num_experts_used, N);
                if (norm_topk != 0)
                {
                    ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
                    w_2d = ggml_div(ctx, w_2d, w_sum);
                }
                if (expert_weights_scale != 1.0f)
                    w_2d = ggml_scale(ctx, w_2d, expert_weights_scale);
                ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, num_experts_used, N);

                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, ffn_normed, H, 1, N);
                ggml_tensor* g_exp = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);     // [expert_ff, num_used, N]
                ggml_tensor* u_exp = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_exp), u_exp);
                ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);        // [H, num_used, N]
                ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);
                ggml_tensor* moe_out = ggml_cont(ctx, ggml_view_3d(ctx, weighted, H, 1, N, weighted->nb[1], weighted->nb[2], 0));
                for (int u = 1; u < num_experts_used; ++u)
                {
                    ggml_tensor* vu = ggml_view_3d(ctx, weighted, H, 1, N, weighted->nb[1], weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                    moe_out = ggml_add(ctx, moe_out, vu);
                }
                ggml_tensor* moe_out_2d = ggml_reshape_2d(ctx, moe_out, H, N);

                ggml_tensor* sh_g = ggml_mul_mat(ctx, t.shexp_gate_w, ffn_normed); // [shared_ff, N]
                ggml_tensor* sh_u = ggml_mul_mat(ctx, t.shexp_up_w, ffn_normed);
                ggml_tensor* sh_act = ggml_mul(ctx, ggml_silu(ctx, sh_g), sh_u);
                ggml_tensor* sh_down = ggml_mul_mat(ctx, t.shexp_down_w, sh_act); // [H, N]
                ggml_tensor* sh_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, t.shexp_gate_inp_w, H, 1), ffn_normed)); // [1, N]
                ggml_tensor* sh_out = ggml_mul(ctx, sh_down, sh_gate);
                ffn_out = ggml_add(ctx, moe_out_2d, sh_out);
            }

            hidden = ggml_add(ctx, residual1, ffn_out); // [H, N]
        }

        // Final norm over all N rows -> the MTP head's input AND the LM head.
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t); // [H, N]
        ggml_tensor* normed_out_t = nullptr;
        ggml_tensor* normed_cpy = nullptr;
        if (normed_out != nullptr)
        {
            normed_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
            normed_cpy = ggml_cpy(ctx, fn, normed_out_t);
            ggml_set_output(normed_cpy);
        }
        // Prefill (n_logits < N) folds the lm_head over only the LAST n_logits
        // columns of the post-norm hidden — the trailing token(s) we sample from.
        ggml_tensor* fn_head_in = fn;                                  // [H, N]
        if (n_logits < N)
        {
            ggml_tensor* fn_last = ggml_view_2d(ctx, fn, H, n_logits, fn->nb[1],
                static_cast<std::size_t>(N - n_logits) * fn->nb[1]);
            fn_head_in = ggml_cont(ctx, fn_last);                      // [H, n_logits]
        }
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn_head_in); // [vocab, n_logits]
        ggml_tensor* logits_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_logits);
        ggml_tensor* logits_cpy = ggml_cpy(ctx, logits, logits_out_t);
        ggml_set_output(logits_cpy);

        // Per-head set_rows adds ~2*num_kv_heads nodes per attention layer, so size
        // the graph generously to avoid GGML_ASSERT(n_nodes < size).
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (260 + 2 * num_kv_heads) + 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (ggml_tensor* w : kv_writes)
            ggml_build_forward_expand(graph, w);
        for (ggml_tensor* w : state_writes)
        {
            ggml_set_output(w);
            ggml_build_forward_expand(graph, w);
        }
        if (normed_cpy != nullptr)
            ggml_build_forward_expand(graph, normed_cpy);
        ggml_build_forward_expand(graph, logits_cpy);

        // --- bind tensors ---
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tgt, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, addr) == GGML_STATUS_SUCCESS)
                    { if (needs_upload) upload_list.push_back({tgt, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                { if (!cacheable) ephemeral_bufs.emplace_back(buf);
                  if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            if (d.is_moe == 0)
            {
                bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
                bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            }
            else
            {
                bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(d.gate_inp_bytes), true);
                bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.gate_exps_bytes), true);
                bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.up_exps_bytes), true);
                bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.down_exps_bytes), true);
                bind_or_mark(t.shexp_gate_w, d.shexp_gate_w, static_cast<std::size_t>(d.shexp_gate_bytes), true);
                bind_or_mark(t.shexp_up_w, d.shexp_up_w, static_cast<std::size_t>(d.shexp_up_bytes), true);
                bind_or_mark(t.shexp_down_w, d.shexp_down_w, static_cast<std::size_t>(d.shexp_down_bytes), true);
                bind_or_mark(t.shexp_gate_inp_w, d.shexp_gate_inp_w, static_cast<std::size_t>(H) * sizeof(float), true);
            }
            if (d.is_recurrent == 0)
            {
                bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
                if (d.separate_qkv != 0)
                {
                    bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                    bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
                }
                bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
                bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_cache_base, d.k_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_cache_base, d.v_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (!t.mask_data.empty())
                    bind_or_mark(t.mask_t, t.mask_data.data(), t.mask_data.size() * sizeof(ggml_fp16_t), false);
            }
            else
            {
                bind_or_mark(t.gdn_qkv_w, d.gdn_qkv_w, static_cast<std::size_t>(d.gdn_qkv_bytes), true);
                bind_or_mark(t.gdn_gate_w, d.gdn_gate_w, static_cast<std::size_t>(d.gdn_gate_bytes), true);
                bind_or_mark(t.ssm_beta_w, d.ssm_beta_w, static_cast<std::size_t>(d.ssm_beta_bytes), true);
                bind_or_mark(t.ssm_alpha_w, d.ssm_alpha_w, static_cast<std::size_t>(d.ssm_alpha_bytes), true);
                bind_or_mark(t.conv1d_w, d.conv1d_w, static_cast<std::size_t>(conv_kernel) * conv_dim * sizeof(float), true);
                bind_or_mark(t.ssm_dt_w, d.ssm_dt_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_a_w, d.ssm_a_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_norm_w, d.ssm_norm_w, static_cast<std::size_t>(head_v_dim) * sizeof(float), true);
                bind_or_mark(t.ssm_out_w, d.ssm_out_w, static_cast<std::size_t>(d.ssm_out_bytes), true);
                if (resident_state)
                {
                    // Device-resident GDN state: bind cacheable COMPUTE (keyed by the
                    // decode's _fdConvScratch / _deltaStateTensor host ptrs) so the
                    // buffer persists across calls and is updated in-place. The cacheable
                    // path uploads only when the host key is invalidated (the C# seed);
                    // subsequent calls cache-hit (no upload).
                    bind_or_mark(t.conv_state_in, d.conv_state_in, convStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    bind_or_mark(t.delta_state_in, d.delta_state_in, deltaStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                }
                // else: conv_state_in / delta_state_in are per-call set_input inputs,
                // uploaded each call below.
            }
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);

        // Persist: give every still-unbound tensor (intermediates, inputs, outputs,
        // small weights) its OWN stable slot via alloc_ctx_tensors -> reuse/capture.
        ggml_backend_buffer_t persist_buf = nullptr;
        if (fv_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Qwen3.5 model verify: failed to allocate persist buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else if (!alloc_graph_reuse_gallocr(graph))
        {
            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (buffer.value == nullptr)
            {
                set_last_error("Qwen3.5 model verify: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);
        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * N * sizeof(float));
        if (use_mrope)
        {
            ggml_backend_tensor_set(pos_tensor, mrope_pos, 0, static_cast<std::size_t>(4) * N * sizeof(std::int32_t));
        }
        else
        {
            std::vector<std::int32_t> pos_vals(N);
            for (int i = 0; i < N; i++) pos_vals[i] = start_pos + i;
            ggml_backend_tensor_set(pos_tensor, pos_vals.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int32_t));
        }
        ggml_backend_tensor_set(kv_index, kv_index_data.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int64_t));
        ggml_backend_tensor_set(attn_mask, attn_mask_data.data(), 0, attn_mask_data.size() * sizeof(ggml_fp16_t));
        if (!resident_state)
        {
            // Host mode: upload the per-call GDN state. Resident mode skips this — the
            // state is device-resident (cacheable), seeded only on invalidation.
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent == 0) continue;
                ggml_backend_tensor_set(lt[l].conv_state_in, layers[l].conv_state_in, 0, convStateBytes);
                ggml_backend_tensor_set(lt[l].delta_state_in, layers[l].delta_state_in, 0, deltaStateBytes);
            }
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3.5 model verify: graph execution failed.");
            return 0;
        }

        // Download the post-window GDN state (per recurrent layer) + outputs. Resident
        // mode skips the state download (it stays device-resident, updated in-place).
        if (!resident_state)
        {
            for (int l = 0; l < num_layers; l++)
            {
                const TSGgmlQwen35LayerDesc& d = layers[l];
                if (d.is_recurrent == 0) continue;
                if (d.conv_state_out != nullptr)
                    finalize_compute_with_download(lt[l].conv_state_out, d.conv_state_out, convStateBytes);
                if (d.delta_state_out != nullptr)
                    finalize_compute_with_download(lt[l].delta_state_out, d.delta_state_out, deltaStateBytes);
            }
        }
        if (normed_out != nullptr && normed_out_t != nullptr)
            finalize_compute_with_download(normed_out_t, normed_out, static_cast<std::size_t>(H) * N * sizeof(float));
        finalize_compute_with_download(logits_out_t, logits_data, static_cast<std::size_t>(vocab_size) * n_logits * sizeof(float));
        host_read_barrier();


        // Persist: keep ctx/graph/buffer alive + record tensor handles so later steps
        // of the same (N, window) shape just upload inputs + replay (capturable).
        if (fv_persist)
        {
            Q35VerifyCache* slot = nullptr;
            for (auto& c : g_q35vc) { if (!c.valid) { slot = &c; break; } }
            if (slot == nullptr)
            {
                slot = &g_q35vc[0];
                for (auto& c : g_q35vc) if (c.lru < slot->lru) slot = &c;
                slot->reset();
            }
            slot->valid = true;
            slot->n = N; slot->window = window; slot->sig = sig;
            slot->num_layers = num_layers; slot->out_vocab = vocab_size;
            slot->n_logits = n_logits;
            slot->has_normed = (normed_out != nullptr);
            slot->ctx = ctx; slot->buffer = persist_buf; slot->graph = graph;
            slot->hidden_t = hidden_t; slot->pos_t = pos_tensor;
            slot->kv_index = kv_index; slot->mask_t = attn_mask;
            slot->logits_out = logits_out_t; slot->normed_out = normed_out_t;
            slot->conv_in.clear(); slot->delta_in.clear(); slot->conv_out.clear(); slot->delta_out.clear();
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent == 0) continue;
                slot->conv_in.push_back(lt[l].conv_state_in);
                slot->delta_in.push_back(lt[l].delta_state_in);
                slot->conv_out.push_back(lt[l].conv_state_out);
                slot->delta_out.push_back(lt[l].delta_state_out);
            }
            slot->lru = ++g_q35vc_clock;
        }
        clear_last_error();
        return 1;
    }

    void reset_qwen35_verify_cache()
    {
        for (auto& c : g_q35vc) c.reset();
    }
}

TSG_EXPORT int TSGgml_Qwen35ModelVerify(
    const TSGgmlQwen35LayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int start_pos, int num_tokens,
    int num_heads, int num_kv_heads, int head_dim, int cache_size,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, void* normed_out, int n_logit_rows,
    const std::int32_t* mrope_pos, const std::int32_t* mrope_sections)
{
    try
    {
        int r = qwen35_model_verify_impl(
            layers, num_layers, hidden_data, hidden_size, start_pos, num_tokens,
            num_heads, num_kv_heads, head_dim, cache_size,
            rope_n_dims, rope_mode, kv_cache_type,
            conv_kernel, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
            eps, rope_base, rope_freq_scale,
            num_experts, num_experts_used, expert_ff, shared_ff,
            norm_topk, expert_weights_scale,
            logits_data, vocab_size,
            lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
            final_norm_data, normed_out, n_logit_rows, mrope_pos, mrope_sections);
        return r;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 model verify."); return 0; }
}

// Drop the persistent verify-graph cache. Called from C# whenever the attention KV
// device buffer or GDN-state buffers may have moved (KV cache grow / reset), since
// the cached graphs pin those addresses.
TSG_EXPORT void TSGgml_Qwen35ResetVerifyCache()
{
    reset_qwen35_verify_cache();
}

