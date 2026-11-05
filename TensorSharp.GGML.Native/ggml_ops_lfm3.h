// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// LFM3's CUDA kernels, injected into the whole-model graph of
// ggml_ops_lfm3.cpp as GGML_OP_CUSTOM nodes the fused backend launches on
// ggml-cuda's stream (see ggml_ops_dsv4_fused.h). Nothing here touches ggml.
//
//   TSG_LFM3_SHORTCONV  src bcx F32 [3C, T], taps F32 [K, C], state F32 [K-1, C]
//                       -> dst F32 [C, T] = C * conv(B * x). Advances the state
//                       in place. i0 C, i1 K.
//   TSG_LFM3_ATTN_PREP  src q F32 [H*D, T], k/v F32 [KV*D, T], q_norm/k_norm F32
//                       [D], freqs F32 [rope_half], pos I32 [1], k/v cache
//                       [D, capacity, KV] -> dst F32 [D, H, T]: Q after RMSNorm
//                       and NeoX RoPE. K (the same) and V land in cache rows
//                       pos..pos+T-1. i0 H, i1 KV, i2 capacity, i3 rope_half,
//                       f0 eps, f1 position scale.
//   TSG_LFM3_ATTN       src q F32 [D, H, 1], gate F32 [H, 1], k/v cache, pos ->
//                       dst F32 1-D: [H*D] of gated attention for the one query
//                       at pos, followed by the split-KV workspace. i0 KV, i1
//                       capacity, i2 sliding window (0 = none), i3 splits, f0 kq
//                       scale.
//
// pos is read on the device, so a decode graph has no input that changes its
// shape: it replays unchanged until the cache moves.
// ---------------------------------------------------------------------------

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>

#ifdef TSG_GGML_USE_CUDA
void tsg_lfm3_cuda_compute(ggml_tensor* dst, ggml_backend_t cuda_backend);

// Make the backend's device current on this thread, for launches outside its stream.
void tsg_lfm3_select_device(ggml_backend_t cuda_backend);

// Whether TSG_LFM3_ATTN has an instantiation for this head size and GQA ratio.
bool tsg_lfm3_attn_supported(int head_dim, int group_size);
#endif

// Bytes of TSG_LFM3_ATTN's workspace: (acc[D], max, sum) per split and head.
inline std::size_t tsg_lfm3_attn_workspace_floats(int head_dim, int num_heads, int splits)
{
    return static_cast<std::size_t>(splits) * num_heads * (head_dim + 2);
}
