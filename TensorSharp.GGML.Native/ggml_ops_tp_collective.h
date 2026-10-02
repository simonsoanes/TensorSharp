// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdint>
#include <string>

namespace tsg
{
struct TpF32Gather;

// Owns a communicator for these raw CUDA backends. Returns null when the
// optional transport is unavailable; callers can choose host staging before
// starting a forward. The handle must be freed before any backend is freed.
// Calls on a handle are serialized by its owning model executor.
TpF32Gather * tp_cuda_f32_gather_create(ggml_backend_t * backends, int count);

// Gather a complete, disjoint tiling of rows, retaining F32 on the wire.
// Each source is [row_counts[r], columns], each distinct destination is
// [full_rows, columns]; contiguous F32 tensors may have additional flattened
// dimensions. Destinations are ready on the corresponding backend streams.
// A false result aborts this forward: never retry with another transport after
// buffers may have been modified. Shape failures are reported before mutation.
bool tp_cuda_f32_gather(TpF32Gather * handle,
    ggml_tensor * const * sources, ggml_tensor * const * destinations,
    const int64_t * first_rows, const int64_t * row_counts,
    int64_t full_rows, int64_t columns, std::string & error);

void tp_cuda_f32_gather_free(TpF32Gather * handle);
}
