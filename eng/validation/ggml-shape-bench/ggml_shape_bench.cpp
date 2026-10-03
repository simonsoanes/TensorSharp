// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// ggml's own decode kernels on the exact shapes a model runs, for comparison with TensorSharp's
// direct-CUDA kernels (benchmarks/CudaQuantizedMatmulBench --decode-kernels). ggml's
// test-backend-ops perf mode times a fixed shape list; this times a dense mul_mat or an expert
// mul_mat_id at the shapes given on the command line, against the unchanged ggml build.
//
// Build (against a TensorSharp.GGML.Native build tree's static ggml libraries):
//   g++ -O2 -std=c++17 -fopenmp -I$GGML/include ggml_shape_bench.cpp -o ggml-shape-bench \
//     -L$BUILD -Wl,--start-group -lggml -lggml-cpu -lggml-cuda -lggml-base -Wl,--end-group \
//     -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -lnccl -L/usr/local/cuda/lib64/stubs -lcuda -lpthread -ldl
//
// Usage: ggml-shape-bench <case>...
//   mm:<type>:<out>:<in>:<rows>                     dense  y[rows, out] = W[out, in] x[rows, in]
//   id:<type>:<experts>:<used>:<out>:<in>:<tokens>  experts y[tokens, used, out]
// <type> is a ggml type name (q8_0, q4_K, q5_K, q6_K, q2_K, q3_K, iq3_s, iq4_nl, f32, ...).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static ggml_type type_from_name(const std::string& name)
{
    for (int t = 0; t < GGML_TYPE_COUNT; ++t)
    {
        const char* n = ggml_type_name((ggml_type)t);
        if (n && name == n)
            return (ggml_type)t;
    }
    fprintf(stderr, "unknown ggml type %s\n", name.c_str());
    exit(2);
}

static std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ':'))
        parts.push_back(item);
    return parts;
}

static void fill_random(ggml_tensor* t, std::mt19937& rng)
{
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32)
    {
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        float* f = reinterpret_cast<float*>(bytes.data());
        for (size_t i = 0; i < bytes.size() / 4; ++i)
            f[i] = dist(rng);
    }
    else if (t->type == GGML_TYPE_I32)
    {
        return;   // filled by the caller
    }
    else
    {
        // Random block bytes: a quantized block's value decoding does not depend on what it holds.
        for (auto& b : bytes)
            b = (uint8_t)(rng() & 0xFF);
    }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

// Mean microseconds of one graph compute, after warmup; the weight bytes it streams.
static void run_case(ggml_backend_t backend, const std::string& spec, int warmup, int iterations)
{
    auto p = split(spec);
    const bool id = p[0] == "id";
    if ((!id && p.size() != 5) || (id && p.size() != 7))
    {
        fprintf(stderr, "bad case %s\n", spec.c_str());
        exit(2);
    }
    const ggml_type type = type_from_name(p[1]);

    ggml_init_params params{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* out = nullptr;
    ggml_tensor* ids = nullptr;
    ggml_tensor* w = nullptr;
    ggml_tensor* x = nullptr;
    double bytes = 0;
    int experts = 0, used = 0, tokens = 0;
    if (!id)
    {
        const int o = std::atoi(p[2].c_str()), in = std::atoi(p[3].c_str()), rows = std::atoi(p[4].c_str());
        w = ggml_new_tensor_2d(ctx, type, in, o);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, rows);
        out = ggml_mul_mat(ctx, w, x);
        bytes = (double)ggml_nbytes(w);
    }
    else
    {
        experts = std::atoi(p[2].c_str());
        used = std::atoi(p[3].c_str());
        const int o = std::atoi(p[4].c_str()), in = std::atoi(p[5].c_str());
        tokens = std::atoi(p[6].c_str());
        w = ggml_new_tensor_3d(ctx, type, in, o, experts);
        x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in, 1, tokens);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
        out = ggml_mul_mat_id(ctx, w, x, ids);
        // The distinct experts one step reads (a token's selections are distinct).
        bytes = (double)ggml_nbytes(w) / experts * std::min(experts, used * tokens);
    }
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::mt19937 rng(7);
    fill_random(w, rng);
    fill_random(x, rng);
    std::vector<std::vector<int32_t>> idSets;
    if (id)
    {
        // Several distinct selections, rotated per iteration so the experts are not all in L2.
        for (int v = 0; v < 16; ++v)
        {
            std::vector<int32_t> sel((size_t)used * tokens);
            for (int t = 0; t < tokens; ++t)
            {
                std::vector<int32_t> pool(experts);
                for (int e = 0; e < experts; ++e) pool[e] = e;
                std::shuffle(pool.begin(), pool.end(), rng);
                for (int j = 0; j < used; ++j) sel[(size_t)t * used + j] = pool[j];
            }
            idSets.push_back(sel);
        }
        ggml_backend_tensor_set(ids, idSets[0].data(), 0, idSets[0].size() * sizeof(int32_t));
    }

    for (int i = 0; i < warmup; ++i)
        ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    double total = 0;
    for (int i = 0; i < iterations; ++i)
    {
        if (id)
            ggml_backend_tensor_set(ids, idSets[i % idSets.size()].data(), 0, idSets[0].size() * sizeof(int32_t));
        ggml_backend_synchronize(backend);
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        total += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    const double us = total / iterations;
    printf("ggml %-40s %9.2f us  %7.1f GB/s\n", spec.c_str(), us, bytes / (us * 1e3));
    fflush(stdout);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: %s <case>...  (mm:<type>:<out>:<in>:<rows> | id:<type>:<experts>:<used>:<out>:<in>:<tokens>)\n", argv[0]);
        return 2;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend)
    {
        fprintf(stderr, "no CUDA backend\n");
        return 1;
    }
    for (int i = 1; i < argc; ++i)
        run_case(backend, argv[i], 20, 200);
    ggml_backend_free(backend);
    return 0;
}
