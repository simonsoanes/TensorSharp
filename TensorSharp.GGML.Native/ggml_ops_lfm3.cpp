// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// LFM3's whole-model forward on ggml-cuda: every layer of a forward as ONE graph,
// with the short conv, the Q/K preparation and the decode attention as TensorSharp
// kernels (ggml_ops_lfm3.cu) and the matmuls, norms and prefill flash attention as
// ordinary ggml nodes.
//
// One token uses a graph built once and replayed. Its position is a device input
// read by the kernels, so nothing changes shape from token to token; it is rebuilt
// only when the caches move. Several tokens get a graph of their own, which attends
// through ggml's flash attention over the exact key range each layer can see.
//
// The K/V caches and conv state are the managed tensors' device copies, bound by
// their host pointers and advanced in place. After a forward the device holds the
// newest bytes: the managed side moves the rows it needs with
// TSGgml_SyncHostBufferRanges and TSGgml_UploadHostBufferRanges.
//
// Under tensor parallelism each rank builds this same graph over its own shard
// (TSGgml_Lfm3ForwardPlan): the head and channel counts in the arguments are the
// rank's, and the graph comes back unexecuted as a TpRankPlan cut after each
// row-parallel projection, for tp_execute_plans to run and reduce. The sandwich
// norms follow those cuts, so they read the summed value.

#include "ggml_ops_internal.h"
#include "ggml_ops_transformer_common.h"
#include "ggml_ops_flash_attn_guard.h"
#include "ggml_ops_lfm3.h"

#ifdef TSG_GGML_USE_CUDA
#include "ggml_ops_dsv4_fused.h"
#include "ggml-impl.h"
#endif

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

using namespace tsg;

// Field order and types MUST match GgmlNative.Lfm3Weight / Lfm3Layer / Lfm3Args.
struct TSGgmlLfm3Weight
{
    const void* data;
    std::int32_t type;
    std::int32_t reserved;
    std::int64_t ne0;
    std::int64_t ne1;
    std::int64_t bytes;
};

struct TSGgmlLfm3Layer
{
    std::int32_t struct_bytes;
    std::int32_t kind;            // 0 conv, 1 full attention, 2 sliding attention
    const void* attn_norm;
    const void* ffn_norm;
    const void* post_attn_norm;   // null without sandwich norms
    const void* post_ffn_norm;
    TSGgmlLfm3Weight ffn_gate;
    TSGgmlLfm3Weight ffn_up;
    TSGgmlLfm3Weight ffn_down;
    // conv
    TSGgmlLfm3Weight in_proj;
    TSGgmlLfm3Weight out_proj;
    const void* conv_taps;
    void* conv_state;
    // attention
    TSGgmlLfm3Weight q;
    TSGgmlLfm3Weight k;
    TSGgmlLfm3Weight v;
    TSGgmlLfm3Weight gate;
    TSGgmlLfm3Weight o;
    const void* q_norm;
    const void* k_norm;
    void* k_cache;
    void* v_cache;
};

struct TSGgmlLfm3Args
{
    std::int32_t struct_bytes;
    std::int32_t hidden;
    std::int32_t num_heads;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    std::int32_t cache_capacity;
    std::int32_t kv_type;
    std::int32_t conv_width;
    std::int32_t window;
    std::int32_t rope_half;
    std::int32_t vocab;
    std::int32_t conv_state_floats;  // each conv_state's length, padding included
    std::int32_t conv_channels;      // this rank's short-conv channels; hidden on one device
    std::int64_t cache_bytes;
    float eps;
    float position_scale;
    const void* freqs_full;
    const void* freqs_sliding;
    const void* final_norm;
    TSGgmlLfm3Weight lm_head;        // data null on a rank that does not produce logits
};

#ifdef TSG_GGML_USE_CUDA
namespace
{
    constexpr int kFullSplits = 32;
    constexpr int kSlidingSplits = 8;
    constexpr int kFlashKvPad = 256;
    constexpr std::size_t kGraphContextBytes = 32 * 1024 * 1024;
    constexpr std::size_t kGraphNodes = 16384;

    // The kernels run through the fused backend; a CPU backend never gets these nodes.
    void no_cpu_kernel(ggml_tensor*, int, int, void*)
    {
        GGML_ABORT("LFM3 kernels have no CPU implementation");
    }

    tsg_dsv4_fused_desc* new_desc(ggml_context* ctx, int kind)
    {
        auto* desc = new (ggml_new_buffer(ctx, sizeof(tsg_dsv4_fused_desc))) tsg_dsv4_fused_desc;
        desc->kind = kind;
        desc->cuda_compute = tsg_lfm3_cuda_compute;
        return desc;
    }

    struct Lfm3Graph
    {
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* position = nullptr;
        ggml_tensor* logits = nullptr;
        // Held here because a plan's upload is still queued when its builder returns.
        std::int32_t position_value = 0;
        ggml_tensor* full_mask = nullptr;
        ggml_tensor* sliding_mask = nullptr;
        // Keys each attention reads (padded) and the ones it may see (valid).
        int full_kv = 0;
        int full_valid = 0;
        int sliding_start = 0;
        int sliding_kv = 0;
        int sliding_valid = 0;
    };

    struct PendingUpload
    {
        ggml_tensor* tensor;
        const void* data;
        std::size_t bytes;
    };

    struct Binder
    {
        ggml_backend_dev_t device;
        std::vector<PendingUpload> uploads;
        bool failed = false;

        // Weights and norms: resident when they can be, else part of this graph's buffer.
        void constant(ggml_tensor* tensor, const void* data, std::size_t bytes)
        {
            bool needs_upload = false;
            if (bytes >= 4096 &&
                try_bind_cached_tensor(g_backend, device, tensor, const_cast<void*>(data), bytes, needs_upload,
                    GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
            {
                if (needs_upload)
                    uploads.push_back({tensor, data, bytes});
                return;
            }
            // An input: the graph allocator gives other leaves their memory just before
            // first use, so one uploaded ahead of the compute could be overwritten by a
            // node that ran earlier in the same buffer.
            ggml_set_input(tensor);
            uploads.push_back({tensor, data, bytes});
        }

        // Caches and conv state must be the same device bytes in every graph, so a
        // private copy would be wrong and there is no fallback from here.
        void state(ggml_tensor* tensor, void* data, std::size_t bytes)
        {
            bool needs_upload = false;
            if (bytes < 4096 ||
                !try_bind_cached_tensor(g_backend, device, tensor, data, bytes, needs_upload,
                    GGML_BACKEND_BUFFER_USAGE_COMPUTE))
            {
                failed = true;
                return;
            }
            if (needs_upload)
                uploads.push_back({tensor, data, bytes});
        }
    };

    ggml_tensor* new_weight(ggml_context* ctx, const TSGgmlLfm3Weight& w, Binder& binder)
    {
        ggml_tensor* tensor = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(w.type), w.ne0, w.ne1);
        binder.constant(tensor, w.data, static_cast<std::size_t>(w.bytes));
        return tensor;
    }

    ggml_tensor* new_vector(ggml_context* ctx, const void* data, int length, Binder& binder)
    {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length);
        binder.constant(tensor, data, static_cast<std::size_t>(length) * sizeof(float));
        return tensor;
    }

    ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps)
    {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
    }

    ggml_tensor* short_conv(ggml_context* ctx, const TSGgmlLfm3Args& a, const TSGgmlLfm3Layer& d,
        ggml_tensor* normed, int tokens, Binder& binder)
    {
        ggml_tensor* bcx = ggml_mul_mat(ctx, new_weight(ctx, d.in_proj, binder), normed);
        ggml_tensor* taps = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a.conv_width, a.conv_channels);
        binder.constant(taps, d.conv_taps, static_cast<std::size_t>(a.conv_width) * a.conv_channels * sizeof(float));
        ggml_tensor* state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a.conv_state_floats);
        binder.state(state, d.conv_state, static_cast<std::size_t>(a.conv_state_floats) * sizeof(float));

        tsg_dsv4_fused_desc* desc = new_desc(ctx, TSG_LFM3_SHORTCONV);
        desc->i0 = a.conv_channels;
        desc->i1 = a.conv_width;
        ggml_tensor* args[] = {bcx, taps, state};
        ggml_tensor* gated = ggml_custom_4d(ctx, GGML_TYPE_F32, a.conv_channels, tokens, 1, 1, args, 3, no_cpu_kernel, 1, desc);
        return ggml_mul_mat(ctx, new_weight(ctx, d.out_proj, binder), gated);
    }

    ggml_tensor* attention(ggml_context* ctx, const TSGgmlLfm3Args& a, const TSGgmlLfm3Layer& d,
        ggml_tensor* normed, int tokens, int start_pos, Lfm3Graph& g, ggml_tensor* freqs, Binder& binder)
    {
        const bool sliding = d.kind == 2;
        const int H = a.num_heads, KV = a.num_kv_heads, D = a.head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));

        ggml_tensor* q = ggml_mul_mat(ctx, new_weight(ctx, d.q, binder), normed);
        ggml_tensor* k = ggml_mul_mat(ctx, new_weight(ctx, d.k, binder), normed);
        ggml_tensor* v = ggml_mul_mat(ctx, new_weight(ctx, d.v, binder), normed);
        ggml_tensor* gate = ggml_mul_mat(ctx, new_weight(ctx, d.gate, binder), normed);
        ggml_tensor* k_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(a.kv_type), D, a.cache_capacity, KV);
        ggml_tensor* v_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(a.kv_type), D, a.cache_capacity, KV);
        binder.state(k_cache, d.k_cache, static_cast<std::size_t>(a.cache_bytes));
        binder.state(v_cache, d.v_cache, static_cast<std::size_t>(a.cache_bytes));

        tsg_dsv4_fused_desc* prep_desc = new_desc(ctx, TSG_LFM3_ATTN_PREP);
        prep_desc->i0 = H;
        prep_desc->i1 = KV;
        prep_desc->i2 = a.cache_capacity;
        prep_desc->i3 = a.rope_half;
        prep_desc->f0 = a.eps;
        prep_desc->f1 = a.position_scale;
        ggml_tensor* prep_args[] = {
            q, k, v, new_vector(ctx, d.q_norm, D, binder), new_vector(ctx, d.k_norm, D, binder),
            freqs, g.position, k_cache, v_cache,
        };
        ggml_tensor* prepared = ggml_custom_4d(ctx, GGML_TYPE_F32, D, H, tokens, 1, prep_args, 9, no_cpu_kernel, 1, prep_desc);
        // It writes this forward's K/V rows as a side effect, which the graph cannot see:
        // anything reading the cache only through a view, such as the copy of K explicit
        // attention makes, would otherwise be free to run first and read the old rows.
        ggml_build_forward_expand(g.graph, prepared);

        ggml_tensor* attended;
        if (tokens == 1)
        {
            const int splits = sliding ? kSlidingSplits : kFullSplits;
            tsg_dsv4_fused_desc* desc = new_desc(ctx, TSG_LFM3_ATTN);
            desc->i0 = KV;
            desc->i1 = a.cache_capacity;
            desc->i2 = sliding ? a.window : 0;
            desc->i3 = splits;
            desc->f0 = scale;
            ggml_tensor* args[] = {prepared, gate, k_cache, v_cache, g.position};
            const std::int64_t floats = static_cast<std::int64_t>(H) * D +
                static_cast<std::int64_t>(tsg_lfm3_attn_workspace_floats(D, H, splits));
            ggml_tensor* node = ggml_custom_4d(ctx, GGML_TYPE_F32, floats, 1, 1, 1, args, 5, no_cpu_kernel, 1, desc);
            attended = ggml_view_2d(ctx, node, static_cast<std::int64_t>(H) * D, 1,
                static_cast<std::size_t>(H) * D * sizeof(float), 0);
        }
        else
        {
            const int first = sliding ? g.sliding_start : 0;
            const int kv = sliding ? g.sliding_kv : g.full_kv;
            ggml_tensor* mask = sliding ? g.sliding_mask : g.full_mask;
            const std::size_t offset = static_cast<std::size_t>(first) * k_cache->nb[1];
            ggml_tensor* keys = ggml_view_3d(ctx, k_cache, D, kv, KV, k_cache->nb[1], k_cache->nb[2], offset);
            ggml_tensor* values = ggml_view_3d(ctx, v_cache, D, kv, KV, v_cache->nb[1], v_cache->nb[2], offset);
            ggml_tensor* queries = ggml_permute(ctx, prepared, 0, 2, 1, 3);
            ggml_tensor* flash = tsg_flash_attn_ext_guarded(ctx, g_backend, "lfm3 prefill",
                queries, keys, values, mask, scale, 0.0f, 0.0f, nullptr, GGML_PREC_F32);
            ggml_tensor* heads = ggml_reshape_3d(ctx, flash, D, H, tokens);
            ggml_tensor* sigmoid = ggml_sigmoid(ctx, ggml_reshape_3d(ctx, gate, 1, H, tokens));
            attended = ggml_reshape_2d(ctx, ggml_mul(ctx, heads, sigmoid), static_cast<std::int64_t>(H) * D, tokens);
        }
        return ggml_mul_mat(ctx, new_weight(ctx, d.o, binder), attended);
    }

    // partials, when given, collects the row-parallel outputs a plan has to reduce.
    bool build_graph(Lfm3Graph& g, const TSGgmlLfm3Args& a, const TSGgmlLfm3Layer* layers, int num_layers,
        int tokens, int start_pos, Binder& binder, std::vector<ggml_tensor*>* partials)
    {
        ggml_init_params params = { kGraphContextBytes, nullptr, /*no_alloc=*/true };
        g.ctx = ggml_init(params);
        if (g.ctx == nullptr)
            return false;
        ggml_context* ctx = g.ctx;
        g.graph = ggml_new_graph_custom(ctx, kGraphNodes, false);

        g.hidden_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a.hidden, tokens);
        g.position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(g.hidden_in);
        ggml_set_input(g.position);

        ggml_tensor* freqs_full = nullptr;
        ggml_tensor* freqs_sliding = nullptr;
        bool any_full = false, any_sliding = false;
        for (int l = 0; l < num_layers; l++)
        {
            any_full |= layers[l].kind == 1;
            any_sliding |= layers[l].kind == 2;
        }
        if (any_full)
            freqs_full = new_vector(ctx, a.freqs_full, a.rope_half, binder);
        if (any_sliding)
            freqs_sliding = new_vector(ctx, a.freqs_sliding, a.rope_half, binder);

        if (tokens > 1)
        {
            // Padded and masked as llama.cpp pads its KV length: ggml-cuda chooses its
            // flash kernel by this, and an exact length selected a different one.
            const auto padded = [&](int valid, int first)
            {
                const int rounded = (valid + kFlashKvPad - 1) / kFlashKvPad * kFlashKvPad;
                return rounded < a.cache_capacity - first ? rounded : a.cache_capacity - first;
            };
            const int total = start_pos + tokens;
            g.full_valid = total;
            g.full_kv = padded(total, 0);
            g.sliding_start = start_pos - a.window + 1 > 0 ? start_pos - a.window + 1 : 0;
            g.sliding_valid = total - g.sliding_start;
            g.sliding_kv = padded(g.sliding_valid, g.sliding_start);
            if (any_full)
            {
                g.full_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, g.full_kv, tokens);
                ggml_set_input(g.full_mask);
            }
            if (any_sliding)
            {
                g.sliding_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, g.sliding_kv, tokens);
                ggml_set_input(g.sliding_mask);
            }
        }

        ggml_tensor* hidden = g.hidden_in;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlLfm3Layer& d = layers[l];
            ggml_tensor* normed = rms_norm(ctx, hidden, new_vector(ctx, d.attn_norm, a.hidden, binder), a.eps);
            ggml_tensor* mixed = d.kind == 0
                ? short_conv(ctx, a, d, normed, tokens, binder)
                : attention(ctx, a, d, normed, tokens, start_pos, g, d.kind == 2 ? freqs_sliding : freqs_full, binder);
            if (partials != nullptr)
                partials->push_back(mixed);
            if (d.post_attn_norm != nullptr)
                mixed = rms_norm(ctx, mixed, new_vector(ctx, d.post_attn_norm, a.hidden, binder), a.eps);
            hidden = ggml_add(ctx, hidden, mixed);

            ggml_tensor* ffn_in = rms_norm(ctx, hidden, new_vector(ctx, d.ffn_norm, a.hidden, binder), a.eps);
            ggml_tensor* activated = ggml_swiglu_split(ctx,
                ggml_mul_mat(ctx, new_weight(ctx, d.ffn_gate, binder), ffn_in),
                ggml_mul_mat(ctx, new_weight(ctx, d.ffn_up, binder), ffn_in));
            ggml_tensor* down = ggml_mul_mat(ctx, new_weight(ctx, d.ffn_down, binder), activated);
            if (partials != nullptr)
                partials->push_back(down);
            if (d.post_ffn_norm != nullptr)
                down = rms_norm(ctx, down, new_vector(ctx, d.post_ffn_norm, a.hidden, binder), a.eps);
            hidden = ggml_add(ctx, hidden, down);
        }

        if (a.lm_head.data == nullptr)
        {
            ggml_build_forward_expand(g.graph, hidden);
            return !binder.failed;
        }

        ggml_tensor* last = tokens == 1
            ? hidden
            : ggml_view_2d(ctx, hidden, a.hidden, 1, hidden->nb[1], static_cast<std::size_t>(tokens - 1) * hidden->nb[1]);
        ggml_tensor* normed = rms_norm(ctx, last, new_vector(ctx, a.final_norm, a.hidden, binder), a.eps);
        g.logits = ggml_mul_mat(ctx, new_weight(ctx, a.lm_head, binder), normed);
        ggml_set_output(g.logits);

        ggml_build_forward_expand(g.graph, g.logits);
        return !binder.failed;
    }

    void upload_constants(const Binder& binder)
    {
        for (const PendingUpload& u : binder.uploads)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);
    }

    // The decode graph and the backend that runs LFM3's nodes, one of each per rank.
    // Models take turns on these slots: a different model or a moved cache rebuilds them.
    struct DecodeSlot
    {
        Lfm3Graph graph;
        TpRankPlan plan;
        ggml_backend_buffer_t buffer = nullptr;
        const void* model = nullptr;
        const void* cache = nullptr;
        int capacity = 0;

        void reset()
        {
            plan.clear();
            if (buffer != nullptr)
                ggml_backend_buffer_free(buffer);
            buffer = nullptr;
            if (graph.ctx != nullptr)
                ggml_free(graph.ctx);
            graph = Lfm3Graph();
            model = cache = nullptr;
            capacity = 0;
        }
    };

    // A multi-token graph lives until the next one: a plan outlives its builder.
    struct PrefillSlot
    {
        Lfm3Graph graph;
        TpRankPlan plan;

        void reset()
        {
            plan.clear();
            if (graph.ctx != nullptr)
                ggml_free(graph.ctx);
            graph = Lfm3Graph();
        }
    };

    DecodeSlot g_decode[TSG_MAX_DEVICES];
    PrefillSlot g_prefill[TSG_MAX_DEVICES];
    ggml_backend_t g_kernel_backend[TSG_MAX_DEVICES] = {};
    ggml_backend_t g_kernel_backend_owner[TSG_MAX_DEVICES] = {};
    std::atomic<std::uint64_t> g_graph_uid{0};

    std::recursive_mutex& lfm3_mutex()
    {
        static std::recursive_mutex mutex;
        return mutex;
    }

    ggml_backend_t kernel_backend()
    {
        const int rank = g_active_rank;
        if (g_kernel_backend[rank] == nullptr || g_kernel_backend_owner[rank] != g_backend)
        {
            g_kernel_backend[rank] = tsg_dsv4_fused_backend_init(g_backend);
            g_kernel_backend_owner[rank] = g_backend;
        }
        return g_kernel_backend[rank];
    }

    const void* first_cache(const TSGgmlLfm3Layer* layers, int num_layers)
    {
        for (int l = 0; l < num_layers; l++)
            if (layers[l].kind != 0)
                return layers[l].k_cache;
        return nullptr;
    }

    void set_inputs(Lfm3Graph& g, const float* hidden, int tokens, int start_pos)
    {
        decode_input_set_async(g.hidden_in, hidden, static_cast<std::size_t>(tokens) * g.hidden_in->ne[0] * sizeof(float));
        g.position_value = start_pos;
        decode_input_set_async(g.position, &g.position_value, sizeof(g.position_value));
    }

    int run(Lfm3Graph& g, ggml_backend_t backend, const float* hidden, int tokens, int start_pos, float* logits, int vocab)
    {
        set_inputs(g, hidden, tokens, start_pos);
        if (ggml_backend_graph_compute_async(backend, g.graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("LFM3 forward: graph execution failed.");
            return 0;
        }
        sync_backend(backend);
        ggml_backend_tensor_get(g.logits, logits, 0, static_cast<std::size_t>(vocab) * sizeof(float));
        clear_last_error();
        return 1;
    }

    // Cut the graph after each partial. Exact F32: the reduction feeds a norm directly,
    // and the upstream collective would narrow a prefill-sized partial to BF16.
    bool cut_plan(TpRankPlan& plan, const Lfm3Graph& g, const std::vector<ggml_tensor*>& partials, ggml_backend_t backend)
    {
        plan.clear();
        plan.backend = backend;
        plan.allreduce_f32 = true;
        plan.graph = g.graph;
        plan.ar_tensor = partials;
        return tp_plan_segments(plan, partials);
    }

    // Stage this forward's inputs and hand the plan over; the executor runs it.
    int submit_plan(Lfm3Graph& g, TpRankPlan& plan, const float* hidden, int tokens, int start_pos,
        float* logits, int vocab, void** plan_out)
    {
        set_inputs(g, hidden, tokens, start_pos);
        plan.out_tensor = g.logits;
        plan.out_host = g.logits != nullptr ? logits : nullptr;
        plan.out_bytes = g.logits != nullptr ? static_cast<std::size_t>(vocab) * sizeof(float) : 0;
        *plan_out = &plan;
        clear_last_error();
        return 1;
    }

    int decode(const TSGgmlLfm3Args& a, const TSGgmlLfm3Layer* layers, int num_layers,
        const float* hidden, int start_pos, float* logits, ggml_backend_t backend, void** plan_out)
    {
        DecodeSlot& slot = g_decode[g_active_rank];
        const void* cache = first_cache(layers, num_layers);
        const bool reusable = slot.graph.graph != nullptr && slot.model == layers[0].attn_norm &&
            slot.cache == cache && slot.capacity == a.cache_capacity &&
            slot.plan.valid() == (plan_out != nullptr);
        if (!reusable)
        {
            slot.reset();
            Binder binder{ggml_backend_get_device(g_backend)};
            std::vector<ggml_tensor*> partials;
            if (!build_graph(slot.graph, a, layers, num_layers, 1, start_pos, binder, plan_out != nullptr ? &partials : nullptr))
            {
                set_last_error("LFM3 forward: a cache or conv state has no device copy.");
                slot.reset();
                return 0;
            }
            slot.buffer = ggml_backend_alloc_ctx_tensors(slot.graph.ctx, g_backend);
            if (slot.buffer == nullptr)
            {
                set_last_error("LFM3 forward: failed to allocate the decode graph.");
                slot.reset();
                return 0;
            }
            upload_constants(binder);
            if (plan_out != nullptr)
            {
                if (!cut_plan(slot.plan, slot.graph, partials, backend))
                {
                    slot.reset();
                    return 0;
                }
            }
            else
            {
                // ggml-cuda skips comparing a captured graph's nodes when the uid repeats.
                // A plan runs as views of this graph, which carry no uid.
                slot.graph.graph->uid = (0x4C46ull << 40) | g_graph_uid.fetch_add(1, std::memory_order_relaxed);
            }
            slot.model = layers[0].attn_norm;
            slot.cache = cache;
            slot.capacity = a.cache_capacity;
        }
        return plan_out != nullptr
            ? submit_plan(slot.graph, slot.plan, hidden, 1, start_pos, logits, a.vocab, plan_out)
            : run(slot.graph, backend, hidden, 1, start_pos, logits, a.vocab);
    }

    int prefill(const TSGgmlLfm3Args& a, const TSGgmlLfm3Layer* layers, int num_layers,
        const float* hidden, int tokens, int start_pos, float* logits, ggml_backend_t backend, void** plan_out)
    {
        PrefillSlot& slot = g_prefill[g_active_rank];
        slot.reset();
        Lfm3Graph& g = slot.graph;
        Binder binder{ggml_backend_get_device(g_backend)};
        std::vector<ggml_tensor*> partials;
        if (!build_graph(g, a, layers, num_layers, tokens, start_pos, binder, plan_out != nullptr ? &partials : nullptr))
        {
            set_last_error("LFM3 forward: a cache or conv state has no device copy.");
            return 0;
        }
        SuppressGraphReorder keep_order(true);
        if (!alloc_graph_reuse_gallocr(g.graph))
        {
            set_last_error("LFM3 forward: failed to allocate the prefill graph.");
            return 0;
        }
        upload_constants(binder);

        // Written straight into the graph's buffers, from the same rule the host fill uses,
        // on stream 0 of whichever device is current: under TP it has to be this rank's.
        tsg_lfm3_select_device(g_backend);
        bool masks_ok = true;
        if (g.full_mask != nullptr)
            masks_ok &= tsg_cuda_fill_causal_mask_f16(g.full_mask->data, g.full_kv, tokens, start_pos, 0, g.full_valid);
        if (g.sliding_mask != nullptr)
            masks_ok &= tsg_cuda_fill_causal_mask_f16(g.sliding_mask->data, g.sliding_kv, tokens,
                start_pos - g.sliding_start, a.window, g.sliding_valid);
        if (!masks_ok || !tsg_cuda_sync_stream0())
        {
            set_last_error("LFM3 forward: failed to fill the attention masks.");
            return 0;
        }
        if (plan_out != nullptr)
        {
            return cut_plan(slot.plan, g, partials, backend)
                ? submit_plan(g, slot.plan, hidden, tokens, start_pos, logits, a.vocab, plan_out)
                : 0;
        }
        const int result = run(g, backend, hidden, tokens, start_pos, logits, a.vocab);
        slot.reset();
        return result;
    }
}
#endif

namespace
{
    int forward(const TSGgmlLfm3Args* args, const TSGgmlLfm3Layer* layers, int num_layers,
        const float* hidden, int tokens, int start_pos, float* logits, void** plan_out)
    {
#ifdef TSG_GGML_USE_CUDA
        try
        {
            if (!ensure_backend())
                return 0;
            if (g_backend_type != BACKEND_TYPE_CUDA)
            {
                set_last_error("LFM3 forward: the fused path is CUDA only.");
                return 0;
            }
            if (args == nullptr || layers == nullptr || hidden == nullptr ||
                num_layers <= 0 || tokens <= 0 || start_pos < 0 ||
                start_pos + tokens > args->cache_capacity)
            {
                set_last_error("LFM3 forward: invalid arguments.");
                return 0;
            }
            if (args->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlLfm3Args)) ||
                layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlLfm3Layer)))
            {
                set_last_error("LFM3 forward: descriptor size mismatch between C# and native.");
                return 0;
            }
            if ((args->lm_head.data != nullptr) != (logits != nullptr) ||
                (plan_out == nullptr && logits == nullptr))
            {
                set_last_error("LFM3 forward: logits and the LM head go together.");
                return 0;
            }
            if (args->kv_type != GGML_TYPE_F16 && args->kv_type != GGML_TYPE_F32)
            {
                set_last_error("LFM3 forward: the kernels read F16 and F32 caches only.");
                return 0;
            }
            if (!tsg_lfm3_attn_supported(args->head_dim, args->num_heads / args->num_kv_heads))
            {
                set_last_error("LFM3 forward: no decode attention for this head size and GQA ratio.");
                return 0;
            }

            std::lock_guard<std::recursive_mutex> lock(lfm3_mutex());
            ggml_backend_t backend = kernel_backend();
            if (backend == nullptr)
            {
                set_last_error("LFM3 forward: cannot create the kernel backend.");
                return 0;
            }
            host_read_barrier();
            return tokens == 1
                ? decode(*args, layers, num_layers, hidden, start_pos, logits, backend, plan_out)
                : prefill(*args, layers, num_layers, hidden, tokens, start_pos, logits, backend, plan_out);
        }
        catch (const std::exception& ex)
        {
            set_last_error(ex.what());
            return 0;
        }
#else
        (void) args; (void) layers; (void) num_layers; (void) hidden; (void) tokens; (void) start_pos; (void) logits; (void) plan_out;
        set_last_error("LFM3 forward: this build has no CUDA backend.");
        return 0;
#endif
    }
}

TSG_EXPORT int TSGgml_Lfm3Forward(
    const TSGgmlLfm3Args* args, const TSGgmlLfm3Layer* layers, int num_layers,
    const float* hidden, int tokens, int start_pos, float* logits)
{
    return forward(args, layers, num_layers, hidden, tokens, start_pos, logits, nullptr);
}

// The active rank's graph for this forward, unexecuted: every rank's plan then goes to
// TSGgml_TensorParallelExecutePlans (or ...Distributed). hidden and logits are read and
// written during that execution, so they must outlive it. logits is null on a rank
// whose lm_head is.
TSG_EXPORT int TSGgml_Lfm3ForwardPlan(
    const TSGgmlLfm3Args* args, const TSGgmlLfm3Layer* layers, int num_layers,
    const float* hidden, int tokens, int start_pos, float* logits, void** plan_out)
{
    if (plan_out == nullptr)
    {
        set_last_error("LFM3 forward: no plan slot.");
        return 0;
    }
    *plan_out = nullptr;
    return forward(args, layers, num_layers, hidden, tokens, start_pos, logits, plan_out);
}

TSG_EXPORT void TSGgml_Lfm3ResetDecodeCache()
{
#ifdef TSG_GGML_USE_CUDA
    std::lock_guard<std::recursive_mutex> lock(lfm3_mutex());
    for (int rank = 0; rank < TSG_MAX_DEVICES; rank++)
    {
        g_decode[rank].reset();
        g_prefill[rank].reset();
        if (g_kernel_backend[rank] != nullptr)
        {
            ggml_backend_free(g_kernel_backend[rank]);
            g_kernel_backend[rank] = nullptr;
            g_kernel_backend_owner[rank] = nullptr;
        }
    }
#endif
}
