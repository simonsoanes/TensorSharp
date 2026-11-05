// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// LFM3's CUDA kernels. ggml_ops_lfm3.h says what each node computes.

#include "ggml_ops_lfm3.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "ggml_ops_dsv4_fused.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"   // ggml_backend_cuda_context (device, stream)

namespace
{
    constexpr int kWarp = 32;
    constexpr int kPrepRowsPerBlock = 4;
    constexpr int kAttnWarps = 8;
    constexpr int kMaxConvHistory = 31;

    __device__ inline void store_cache(float* dst, float value) { *dst = value; }
    __device__ inline void store_cache(__half* dst, float value) { *dst = __float2half(value); }

    __device__ inline float warp_sum(float value)
    {
#pragma unroll
        for (int offset = kWarp / 2; offset > 0; offset >>= 1)
            value += __shfl_xor_sync(0xffffffffu, value, offset);
        return value;
    }

    // ---- short conv ---------------------------------------------------------

    // One thread per (channel, token): tap j reads position t - history + j, which
    // before the block starts is the state's window. With one token the same
    // thread owns the channel's whole state and advances it here.
    __global__ void lfm3_shortconv_kernel(
        const float* __restrict__ bcx, const float* __restrict__ taps, float* state,
        float* __restrict__ out, int channels, int width, int tokens)
    {
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        const int t = blockIdx.y;
        if (c >= channels)
            return;

        const int history = width - 1;
        const long long stride = 3LL * channels;
        const float* channel_taps = taps + static_cast<long long>(c) * width;
        float* window = state + static_cast<long long>(c) * history;

        float acc = 0.0f;
        float newest = 0.0f;
        for (int j = 0; j < width; j++)
        {
            const int position = t - history + j;
            float value;
            if (position >= 0)
            {
                const float* row = bcx + position * stride;
                value = row[c] * row[2 * channels + c];
                newest = value;
            }
            else
            {
                value = window[history + position];
            }
            acc += channel_taps[j] * value;
        }
        out[static_cast<long long>(t) * channels + c] = bcx[t * stride + channels + c] * acc;

        if (tokens == 1 && history > 0)
        {
            for (int j = 0; j + 1 < history; j++)
                window[j] = window[j + 1];
            window[history - 1] = newest;
        }
    }

    // After a multi-token block: the state is the last `history` values of B*x, the
    // block's own where it has them and the old window's before that. Separate from
    // the kernel above, whose first tokens are still reading the old window.
    __global__ void lfm3_shortconv_state_kernel(
        const float* __restrict__ bcx, float* state, int channels, int width, int tokens)
    {
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= channels)
            return;

        const int history = width - 1;
        const long long stride = 3LL * channels;
        float* window = state + static_cast<long long>(c) * history;
        float next[kMaxConvHistory];
        for (int j = 0; j < history; j++)
        {
            const int position = tokens - history + j;
            if (position >= 0)
            {
                const float* row = bcx + position * stride;
                next[j] = row[c] * row[2 * channels + c];
            }
            else
            {
                next[j] = window[history + position];
            }
        }
        for (int j = 0; j < history; j++)
            window[j] = next[j];
    }

    // ---- attention prep -----------------------------------------------------

    // One warp per row of one token. A row is a Q head, a K head or a V head, in
    // that order. Q and K go through head RMSNorm and NeoX RoPE (pair j is
    // (j, j + rope_half), the rows past 2*rope_half pass through); V is a copy.
    template <typename KV>
    __global__ void lfm3_attn_prep_kernel(
        const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
        const float* __restrict__ q_norm, const float* __restrict__ k_norm,
        const float* __restrict__ freqs, const int* __restrict__ pos,
        KV* k_cache, KV* v_cache, float* __restrict__ q_out,
        int num_heads, int num_kv_heads, int head_dim, int capacity, int rope_half,
        float eps, float position_scale)
    {
        extern __shared__ float rows[];
        const int lane = threadIdx.x;
        const int slot = threadIdx.y;
        const int row = blockIdx.x * kPrepRowsPerBlock + slot;
        const int t = blockIdx.y;
        const int total_rows = num_heads + 2 * num_kv_heads;
        if (row >= total_rows)
            return;

        const int position = pos[0] + t;
        const bool is_q = row < num_heads;
        const bool is_k = !is_q && row < num_heads + num_kv_heads;
        const int head = is_q ? row : is_k ? row - num_heads : row - num_heads - num_kv_heads;
        const float* src = is_q
            ? q + (static_cast<long long>(t) * num_heads + head) * head_dim
            : (is_k ? k : v) + (static_cast<long long>(t) * num_kv_heads + head) * head_dim;
        const long long cache_row = (static_cast<long long>(head) * capacity + position) * head_dim;

        if (!is_q && !is_k)
        {
            for (int d = lane; d < head_dim; d += kWarp)
                store_cache(v_cache + cache_row + d, src[d]);
            return;
        }

        float squares = 0.0f;
        for (int d = lane; d < head_dim; d += kWarp)
            squares += src[d] * src[d];
        const float scale = rsqrtf(warp_sum(squares) / head_dim + eps);
        const float* weight = is_q ? q_norm : k_norm;
        float* staged = rows + slot * head_dim;
        for (int d = lane; d < head_dim; d += kWarp)
            staged[d] = src[d] * scale * weight[d];
        __syncwarp();

        const float scaled_position = position * position_scale;
        for (int d = lane; d < head_dim; d += kWarp)
        {
            float value = staged[d];
            if (d < 2 * rope_half)
            {
                const int pair = d < rope_half ? d : d - rope_half;
                // Double precision on purpose: --use_fast_math's sinf is meant for
                // small arguments and a position times a low-order frequency is not.
                double sine, cosine;
                sincos(static_cast<double>(scaled_position * freqs[pair]), &sine, &cosine);
                const float first = staged[pair];
                const float second = staged[pair + rope_half];
                value = d < rope_half
                    ? first * static_cast<float>(cosine) - second * static_cast<float>(sine)
                    : second * static_cast<float>(cosine) + first * static_cast<float>(sine);
            }
            if (is_q)
                q_out[(static_cast<long long>(t) * num_heads + head) * head_dim + d] = value;
            else
                store_cache(k_cache + cache_row + d, value);
        }
    }

    // ---- decode attention ---------------------------------------------------

    // `Dims` consecutive elements of a row for this lane, vectorized when the row is
    // exactly 32 * Dims wide (the real head sizes; a test's narrower head takes the
    // guarded loop).
    template <int Dims>
    __device__ inline void load_row(const __half* row, int lane, int head_dim, float (&out)[Dims])
    {
        if constexpr (Dims == 4)
        {
            if (head_dim == kWarp * Dims)
            {
                const uint2 raw = reinterpret_cast<const uint2*>(row)[lane];
                const float2 low = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
                const float2 high = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
                out[0] = low.x; out[1] = low.y; out[2] = high.x; out[3] = high.y;
                return;
            }
        }
#pragma unroll
        for (int i = 0; i < Dims; i++)
        {
            const int d = lane * Dims + i;
            out[i] = d < head_dim ? __half2float(row[d]) : 0.0f;
        }
    }

    template <int Dims>
    __device__ inline void load_row(const float* row, int lane, int head_dim, float (&out)[Dims])
    {
        if constexpr (Dims == 4)
        {
            if (head_dim == kWarp * Dims)
            {
                const float4 raw = reinterpret_cast<const float4*>(row)[lane];
                out[0] = raw.x; out[1] = raw.y; out[2] = raw.z; out[3] = raw.w;
                return;
            }
        }
#pragma unroll
        for (int i = 0; i < Dims; i++)
        {
            const int d = lane * Dims + i;
            out[i] = d < head_dim ? row[d] : 0.0f;
        }
    }

    // One block per (split, KV head). The block's warps walk the split's keys in
    // turn, each with its own online softmax over all `Group` query heads, so a K/V
    // row is read once for the whole group. The warps then merge through shared
    // memory into the split's partial result.
    template <typename KV, int Dims, int Group>
    __global__ void __launch_bounds__(kAttnWarps * kWarp) lfm3_attn_partial_kernel(
        const float* __restrict__ q, const KV* __restrict__ k_cache, const KV* __restrict__ v_cache,
        const int* __restrict__ pos, float* __restrict__ workspace,
        int num_heads, int head_dim, int capacity, int window, float scale)
    {
        extern __shared__ float merge[];
        const int split = blockIdx.x;
        const int splits = gridDim.x;
        const int kv_head = blockIdx.y;
        const int warp = threadIdx.x / kWarp;
        const int lane = threadIdx.x % kWarp;

        const int total = pos[0] + 1;
        const int first = window > 0 && total > window ? total - window : 0;
        const int per_split = (total - first + splits - 1) / splits;
        const int begin = first + split * per_split;
        const int end = min(total, begin + per_split);

        float query[Group][Dims];
#pragma unroll
        for (int g = 0; g < Group; g++)
        {
            load_row<Dims>(q + static_cast<long long>(kv_head * Group + g) * head_dim, lane, head_dim, query[g]);
#pragma unroll
            for (int i = 0; i < Dims; i++)
                query[g][i] *= scale;
        }

        float running_max[Group];
        float running_sum[Group];
        float acc[Group][Dims];
#pragma unroll
        for (int g = 0; g < Group; g++)
        {
            running_max[g] = -INFINITY;
            running_sum[g] = 0.0f;
#pragma unroll
            for (int i = 0; i < Dims; i++)
                acc[g][i] = 0.0f;
        }

        const long long head_base = static_cast<long long>(kv_head) * capacity;
        for (int p = begin + warp; p < end; p += kAttnWarps)
        {
            float key[Dims];
            float value[Dims];
            load_row<Dims>(k_cache + (head_base + p) * head_dim, lane, head_dim, key);
            load_row<Dims>(v_cache + (head_base + p) * head_dim, lane, head_dim, value);
#pragma unroll
            for (int g = 0; g < Group; g++)
            {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < Dims; i++)
                    dot += query[g][i] * key[i];
                const float score = warp_sum(dot);
                const float next_max = fmaxf(running_max[g], score);
                const float correction = __expf(running_max[g] - next_max);
                const float weight = __expf(score - next_max);
                running_max[g] = next_max;
                running_sum[g] = running_sum[g] * correction + weight;
#pragma unroll
                for (int i = 0; i < Dims; i++)
                    acc[g][i] = acc[g][i] * correction + weight * value[i];
            }
        }

        // One group member at a time keeps shared memory at warps * (D + 2) floats.
        const int stride = head_dim + 2;
        float* mine = merge + warp * stride;
#pragma unroll
        for (int g = 0; g < Group; g++)
        {
#pragma unroll
            for (int i = 0; i < Dims; i++)
            {
                const int d = lane * Dims + i;
                if (d < head_dim)
                    mine[d] = acc[g][i];
            }
            if (lane == 0)
            {
                mine[head_dim] = running_max[g];
                mine[head_dim + 1] = running_sum[g];
            }
            __syncthreads();

            float block_max = -INFINITY;
            for (int w = 0; w < kAttnWarps; w++)
                block_max = fmaxf(block_max, merge[w * stride + head_dim]);
            float* partial = workspace +
                (static_cast<long long>(split) * num_heads + kv_head * Group + g) * stride;
            float block_sum = 0.0f;
            float block_acc = 0.0f;
            const int d = threadIdx.x;
            for (int w = 0; w < kAttnWarps; w++)
            {
                const float warp_max = merge[w * stride + head_dim];
                if (warp_max == -INFINITY)
                    continue;
                const float weight = __expf(warp_max - block_max);
                block_sum += merge[w * stride + head_dim + 1] * weight;
                if (d < head_dim)
                    block_acc += merge[w * stride + d] * weight;
            }
            if (d < head_dim)
                partial[d] = block_acc;
            if (d == 0)
            {
                partial[head_dim] = block_max;
                partial[head_dim + 1] = block_sum;
            }
            __syncthreads();
        }
    }

    // One block per query head: fold the splits and apply the head's sigmoid gate.
    __global__ void lfm3_attn_finish_kernel(
        const float* __restrict__ workspace, const float* __restrict__ gate, float* __restrict__ out,
        int num_heads, int head_dim, int splits)
    {
        const int head = blockIdx.x;
        const int d = threadIdx.x;
        const int stride = head_dim + 2;

        float best = -INFINITY;
        for (int s = 0; s < splits; s++)
            best = fmaxf(best, workspace[(static_cast<long long>(s) * num_heads + head) * stride + head_dim]);

        float sum = 0.0f;
        float acc = 0.0f;
        for (int s = 0; s < splits; s++)
        {
            const float* partial = workspace + (static_cast<long long>(s) * num_heads + head) * stride;
            const float split_max = partial[head_dim];
            if (split_max == -INFINITY)
                continue;
            const float weight = __expf(split_max - best);
            sum += partial[head_dim + 1] * weight;
            acc += partial[d] * weight;
        }
        const float sigmoid = 1.0f / (1.0f + __expf(-gate[head]));
        out[static_cast<long long>(head) * head_dim + d] = acc / sum * sigmoid;
    }

    template <typename KV, int Dims, int Group>
    void launch_attn_partial(
        const float* q, const void* k_cache, const void* v_cache, const int* pos, float* workspace,
        int num_heads, int num_kv_heads, int head_dim, int capacity, int window, int splits, float scale,
        cudaStream_t stream)
    {
        const dim3 grid(static_cast<unsigned>(splits), static_cast<unsigned>(num_kv_heads));
        const std::size_t shared = static_cast<std::size_t>(kAttnWarps) * (head_dim + 2) * sizeof(float);
        lfm3_attn_partial_kernel<KV, Dims, Group><<<grid, kAttnWarps * kWarp, shared, stream>>>(
            q, static_cast<const KV*>(k_cache), static_cast<const KV*>(v_cache), pos, workspace,
            num_heads, head_dim, capacity, window, scale);
    }

    template <typename KV, int Dims>
    bool dispatch_group(
        int group, const float* q, const void* k_cache, const void* v_cache, const int* pos, float* workspace,
        int num_heads, int num_kv_heads, int head_dim, int capacity, int window, int splits, float scale,
        cudaStream_t stream)
    {
        switch (group)
        {
            case 1: launch_attn_partial<KV, Dims, 1>(q, k_cache, v_cache, pos, workspace, num_heads, num_kv_heads, head_dim, capacity, window, splits, scale, stream); return true;
            case 2: launch_attn_partial<KV, Dims, 2>(q, k_cache, v_cache, pos, workspace, num_heads, num_kv_heads, head_dim, capacity, window, splits, scale, stream); return true;
            case 4: launch_attn_partial<KV, Dims, 4>(q, k_cache, v_cache, pos, workspace, num_heads, num_kv_heads, head_dim, capacity, window, splits, scale, stream); return true;
            case 8: launch_attn_partial<KV, Dims, 8>(q, k_cache, v_cache, pos, workspace, num_heads, num_kv_heads, head_dim, capacity, window, splits, scale, stream); return true;
            default: return false;
        }
    }

    template <typename KV>
    bool dispatch_dims(
        int group, const float* q, const void* k_cache, const void* v_cache, const int* pos, float* workspace,
        int num_heads, int num_kv_heads, int head_dim, int capacity, int window, int splits, float scale,
        cudaStream_t stream)
    {
        if (head_dim <= 32)
            return dispatch_group<KV, 1>(group, q, k_cache, v_cache, pos, workspace, num_heads, num_kv_heads, head_dim, capacity, window, splits, scale, stream);
        if (head_dim <= 64)
            return dispatch_group<KV, 2>(group, q, k_cache, v_cache, pos, workspace, num_heads, num_kv_heads, head_dim, capacity, window, splits, scale, stream);
        if (head_dim <= 128)
            return dispatch_group<KV, 4>(group, q, k_cache, v_cache, pos, workspace, num_heads, num_kv_heads, head_dim, capacity, window, splits, scale, stream);
        if (head_dim <= 256)
            return dispatch_group<KV, 8>(group, q, k_cache, v_cache, pos, workspace, num_heads, num_kv_heads, head_dim, capacity, window, splits, scale, stream);
        return false;
    }

    tsg_dsv4_fused_desc node_desc(const ggml_tensor* dst)
    {
        ggml_custom_op_params params;
        std::memcpy(&params, dst->op_params, sizeof(params));
        return *static_cast<const tsg_dsv4_fused_desc*>(params.userdata);
    }

    void compute_shortconv(const ggml_tensor* dst, const tsg_dsv4_fused_desc& d, cudaStream_t stream)
    {
        const ggml_tensor* bcx = dst->src[0];
        const int channels = d.i0;
        const int width = d.i1;
        const int tokens = static_cast<int>(bcx->ne[1]);
        const unsigned blocks = static_cast<unsigned>((channels + 255) / 256);
        float* state = static_cast<float*>(dst->src[2]->data);
        lfm3_shortconv_kernel<<<dim3(blocks, static_cast<unsigned>(tokens)), 256, 0, stream>>>(
            static_cast<const float*>(bcx->data), static_cast<const float*>(dst->src[1]->data),
            state, static_cast<float*>(dst->data), channels, width, tokens);
        if (tokens > 1)
        {
            lfm3_shortconv_state_kernel<<<blocks, 256, 0, stream>>>(
                static_cast<const float*>(bcx->data), state, channels, width, tokens);
        }
    }

    template <typename KV>
    void launch_attn_prep(const ggml_tensor* dst, const tsg_dsv4_fused_desc& d, cudaStream_t stream)
    {
        const int head_dim = static_cast<int>(dst->ne[0]);
        const int tokens = static_cast<int>(dst->ne[2]);
        const int rows = d.i0 + 2 * d.i1;
        const dim3 grid(static_cast<unsigned>((rows + kPrepRowsPerBlock - 1) / kPrepRowsPerBlock), static_cast<unsigned>(tokens));
        const dim3 block(kWarp, kPrepRowsPerBlock);
        const std::size_t shared = static_cast<std::size_t>(kPrepRowsPerBlock) * head_dim * sizeof(float);
        lfm3_attn_prep_kernel<KV><<<grid, block, shared, stream>>>(
            static_cast<const float*>(dst->src[0]->data), static_cast<const float*>(dst->src[1]->data),
            static_cast<const float*>(dst->src[2]->data), static_cast<const float*>(dst->src[3]->data),
            static_cast<const float*>(dst->src[4]->data), static_cast<const float*>(dst->src[5]->data),
            static_cast<const int*>(dst->src[6]->data),
            static_cast<KV*>(dst->src[7]->data), static_cast<KV*>(dst->src[8]->data),
            static_cast<float*>(dst->data),
            d.i0, d.i1, head_dim, d.i2, d.i3, d.f0, d.f1);
    }

    void compute_attn(const ggml_tensor* dst, const tsg_dsv4_fused_desc& d, cudaStream_t stream)
    {
        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k_cache = dst->src[2];
        const int head_dim = static_cast<int>(q->ne[0]);
        const int num_heads = static_cast<int>(q->ne[1]);
        const int num_kv_heads = d.i0;
        const int splits = d.i3;
        float* out = static_cast<float*>(dst->data);
        float* workspace = out + static_cast<std::size_t>(num_heads) * head_dim;
        const float* q_data = static_cast<const float*>(q->data);
        const void* k_data = k_cache->data;
        const void* v_data = dst->src[3]->data;
        const int* pos = static_cast<const int*>(dst->src[4]->data);
        const int group = num_heads / num_kv_heads;
        const bool launched = k_cache->type == GGML_TYPE_F16
            ? dispatch_dims<__half>(group, q_data, k_data, v_data, pos, workspace, num_heads, num_kv_heads, head_dim, d.i1, d.i2, splits, d.f0, stream)
            : dispatch_dims<float>(group, q_data, k_data, v_data, pos, workspace, num_heads, num_kv_heads, head_dim, d.i1, d.i2, splits, d.f0, stream);
        GGML_ASSERT(launched && "tsg_lfm3_attn_supported admitted a shape with no instantiation");
        lfm3_attn_finish_kernel<<<static_cast<unsigned>(num_heads), static_cast<unsigned>(head_dim), 0, stream>>>(
            workspace, static_cast<const float*>(dst->src[1]->data), out, num_heads, head_dim, splits);
    }
}

bool tsg_lfm3_attn_supported(int head_dim, int group_size)
{
    return head_dim > 0 && head_dim <= 256 &&
        (group_size == 1 || group_size == 2 || group_size == 4 || group_size == 8);
}

void tsg_lfm3_select_device(ggml_backend_t backend)
{
    CUDA_CHECK(cudaSetDevice(static_cast<ggml_backend_cuda_context*>(backend->context)->device));
}

void tsg_lfm3_cuda_compute(ggml_tensor* dst, ggml_backend_t backend)
{
    auto* context = static_cast<ggml_backend_cuda_context*>(backend->context);
    CUDA_CHECK(cudaSetDevice(context->device));
    const cudaStream_t stream = context->stream(context->device, 0);
    const tsg_dsv4_fused_desc d = node_desc(dst);
    switch (d.kind)
    {
        case TSG_LFM3_SHORTCONV:
            GGML_ASSERT(d.i1 - 1 <= kMaxConvHistory);
            compute_shortconv(dst, d, stream);
            break;
        case TSG_LFM3_ATTN_PREP:
            if (dst->src[7]->type == GGML_TYPE_F16)
                launch_attn_prep<__half>(dst, d, stream);
            else
                launch_attn_prep<float>(dst, d, stream);
            break;
        case TSG_LFM3_ATTN:
            compute_attn(dst, d, stream);
            break;
        default:
            GGML_ABORT("tsg_lfm3_cuda_compute: not an LFM3 node");
    }
    CUDA_CHECK(cudaGetLastError());
}
