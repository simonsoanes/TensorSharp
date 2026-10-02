// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone physical-CUDA test. No model checkpoint or modified ggml required.
#include "ggml_ops_tp_collective.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// The probe's only dependency on TensorSharp's global error service.
namespace tsg { void set_last_error(const std::string & error) { std::fprintf(stderr, "%s\n", error.c_str()); } }

namespace {
size_t checks = 0, exact_values = 0;
void require(bool ok, const std::string & message)
{
    if (!ok) throw std::runtime_error(message);
    ++checks;
}

void environment(const char * key, const char * value)
{
#if defined(_WIN32)
    _putenv_s(key, value ? value : "");
#else
    if (value) setenv(key, value, 1); else unsetenv(key);
#endif
}

struct tensor
{
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * value = nullptr;
    tensor(ggml_backend_t backend, int64_t rows, int64_t columns, bool flattened = false)
    {
        context = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
        require(context != nullptr, "Cannot create tensor metadata");
        value = flattened && columns % 3 == 0
            ? ggml_new_tensor_3d(context, GGML_TYPE_F32, rows, 3, columns / 3)
            : ggml_new_tensor_2d(context, GGML_TYPE_F32, rows, columns);
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        require(buffer != nullptr, "Cannot allocate device tensor");
    }
    ~tensor() { ggml_backend_buffer_free(buffer); ggml_free(context); }
    tensor(const tensor &) = delete;
    void put(const std::vector<uint32_t> & bits)
    {
        require(bits.size() * sizeof(uint32_t) == ggml_nbytes(value), "Upload shape mismatch");
        ggml_backend_tensor_set(value, bits.data(), 0, ggml_nbytes(value));
    }
    std::vector<uint32_t> get() const
    {
        std::vector<uint32_t> bits(ggml_nelements(value));
        ggml_backend_tensor_get(value, bits.data(), 0, ggml_nbytes(value));
        return bits;
    }
};

using handle_ptr = std::unique_ptr<tsg::TpF32Gather, decltype(&tsg::tp_cuda_f32_gather_free)>;

uint32_t pattern(size_t index, unsigned pass)
{
    // All finite: signed zeros/subnormals, extrema, and mantissas that cannot
    // survive BF16 narrowing. Bit comparisons never apply numerical tolerance.
    static const uint32_t special[] = {0u, 0x80000000u, 1u, 0x80000001u,
        0x00800000u, 0x80800000u, 0x7f7fffffu, 0xff7fffffu,
        0x3f800001u, 0xbf800001u, 0x3f807fffu, 0xbf807fffu};
    if (index < sizeof(special) / sizeof(special[0])) return special[(index + pass) % 12];
    return 0x3e800000u | (static_cast<uint32_t>(index * 2654435761u + pass * 1777u) & 0x807fffffu);
}

void exact_case(ggml_backend_t * backends, tsg::TpF32Gather * handle,
                int64_t a, int64_t b, int64_t columns, bool flattened)
{
    const int64_t rows = a + b;
    std::array<int64_t, 2> starts{0, a}, counts{a, b};
    tensor source0(backends[0], a, columns, flattened), source1(backends[1], b, columns, flattened);
    tensor destination0(backends[0], rows, columns, flattened), destination1(backends[1], rows, columns, flattened);
    std::array<ggml_tensor *, 2> sources{source0.value, source1.value}, destinations{destination0.value, destination1.value};
    for (unsigned pass = 0; pass < 3; ++pass)
    {
        std::vector<uint32_t> expected(rows * columns), local0(a * columns), local1(b * columns);
        for (int64_t c = 0; c < columns; ++c)
            for (int64_t r = 0; r < rows; ++r)
            {
                const auto bits = pattern(c * rows + r, pass);
                expected[c * rows + r] = bits;
                if (r < a) local0[c * a + r] = bits; else local1[c * b + r - a] = bits;
            }
        source0.put(local0); source1.put(local1);
        const std::vector<uint32_t> sentinel(expected.size(), 0x42f6e979u);
        destination0.put(sentinel); destination1.put(sentinel);
        std::string error;
        require(tsg::tp_cuda_f32_gather(handle, sources.data(), destinations.data(), starts.data(), counts.data(), rows, columns, error),
                "Valid gather failed: " + error);
        for (auto backend : {backends[0], backends[1]}) ggml_backend_synchronize(backend);
        for (const tensor * output : {&destination0, &destination1})
        {
            const auto actual = output->get();
            for (size_t i = 0; i < actual.size(); ++i)
                if (actual[i] != expected[i])
                {
                    char message[256];
                    std::snprintf(message, sizeof(message), "Gather bits differ rows=%lld+%lld columns=%lld pass=%u element=%zu: %08x != %08x",
                        (long long)a, (long long)b, (long long)columns, pass, i, actual[i], expected[i]);
                    require(false, message);
                }
            exact_values += actual.size();
            ++checks;
        }
        require(source0.get() == local0 && source1.get() == local1, "Successful gather mutated a source");
    }
}

void rejection_cases(ggml_backend_t * backends, tsg::TpF32Gather * handle)
{
    tensor source0(backends[0], 4, 5), source1(backends[1], 4, 5);
    tensor destination0(backends[0], 8, 5), destination1(backends[1], 8, 5);
    const std::vector<uint32_t> input0(20, 0x3f812345u), input1(20, 0xbf854321u), sentinel(40, 0x42123456u);
    source0.put(input0); source1.put(input1);
    destination0.put(sentinel); destination1.put(sentinel);
    using mutate = std::function<void(std::array<ggml_tensor *, 2> &, std::array<ggml_tensor *, 2> &,
                                     std::array<int64_t, 2> &, std::array<int64_t, 2> &, int64_t &, int64_t &)>;
    auto reject = [&](const char * label, mutate change) {
        std::array<ggml_tensor *, 2> sources{source0.value, source1.value}, destinations{destination0.value, destination1.value};
        std::array<int64_t, 2> first{0, 4}, counts{4, 4};
        int64_t rows = 8, columns = 5;
        change(sources, destinations, first, counts, rows, columns);
        std::string error;
        require(!tsg::tp_cuda_f32_gather(handle, sources.data(), destinations.data(), first.data(), counts.data(), rows, columns, error),
                std::string("Accepted invalid gather: ") + label);
        require(!error.empty(), std::string("Missing rejection reason: ") + label);
        for (auto backend : {backends[0], backends[1]}) ggml_backend_synchronize(backend);
        require(destination0.get() == sentinel && destination1.get() == sentinel &&
                source0.get() == input0 && source1.get() == input1,
                std::string("Rejected gather mutated buffers: ") + label);
    };
    reject("row overlap", [](auto &, auto &, auto & first, auto &, auto &, auto &) { first[1] = 3; });
    reject("row gap", [](auto &, auto &, auto & first, auto &, auto &, auto &) { first[1] = 5; });
    reject("empty rank", [](auto &, auto &, auto &, auto & counts, auto &, auto &) { counts[1] = 0; });
    reject("negative rows", [](auto &, auto &, auto &, auto &, auto & rows, auto &) { rows = -1; });
    reject("zero columns", [](auto &, auto &, auto &, auto &, auto &, auto & columns) { columns = 0; });
    reject("dimension overflow", [](auto &, auto &, auto &, auto &, auto & rows, auto & columns) {
        rows = std::numeric_limits<int64_t>::max(); columns = 2;
    });
    reject("null source", [](auto & src, auto &, auto &, auto &, auto &, auto &) { src[1] = nullptr; });
    reject("null destination", [](auto &, auto & dst, auto &, auto &, auto &, auto &) { dst[1] = nullptr; });
    reject("duplicate destination", [](auto &, auto & dst, auto &, auto &, auto &, auto &) { dst[1] = dst[0]; });
    reject("wrong source device", [](auto & src, auto &, auto &, auto &, auto &, auto &) { src[1] = src[0]; });
    reject("swapped destination devices", [](auto &, auto & dst, auto &, auto &, auto &, auto &) { std::swap(dst[0], dst[1]); });
    ggml_tensor malformed = *source0.value;
    malformed.type = GGML_TYPE_I32;
    reject("wrong type", [&](auto & src, auto &, auto &, auto &, auto &, auto &) { src[0] = &malformed; });
    malformed = *source0.value; malformed.nb[1] += sizeof(float);
    reject("strided source", [&](auto & src, auto &, auto &, auto &, auto &, auto &) { src[0] = &malformed; });
    malformed = *destination0.value; malformed.data = source0.value->data; malformed.buffer = source0.value->buffer;
    reject("local source destination overlap", [&](auto &, auto & dst, auto &, auto &, auto &, auto &) { dst[0] = &malformed; });
}

void many_rank_case(ggml_backend_t * backends, tsg::TpF32Gather * handle, int ranks, bool large)
{
    const int64_t columns = large ? 129 : 3;
    std::vector<int64_t> counts(ranks), first(ranks);
    int64_t rows = 0;
    for (int r = 0; r < ranks; ++r)
    {
        first[r] = rows;
        counts[r] = large ? 128 * (1 + r % 3) : r + 1;
        rows += counts[r];
    }
    std::vector<std::unique_ptr<tensor>> input, output;
    std::vector<ggml_tensor *> sources, destinations;
    for (int r = 0; r < ranks; ++r)
    {
        input.emplace_back(std::make_unique<tensor>(backends[r], counts[r], columns, true));
        output.emplace_back(std::make_unique<tensor>(backends[r], rows, columns, true));
        sources.push_back(input.back()->value);
        destinations.push_back(output.back()->value);
    }
    for (unsigned pass = 0; pass < 3; ++pass)
    {
        std::vector<uint32_t> expected(rows * columns);
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = pattern(i, pass);
        std::vector<std::vector<uint32_t>> locals(ranks);
        for (int r = 0; r < ranks; ++r)
        {
            locals[r].resize(counts[r] * columns);
            for (int64_t c = 0; c < columns; ++c)
                for (int64_t row = 0; row < counts[r]; ++row)
                    locals[r][c * counts[r] + row] = expected[c * rows + first[r] + row];
            input[r]->put(locals[r]);
            output[r]->put(std::vector<uint32_t>(expected.size(), 0x42f6e979u));
        }
        std::string error;
        require(tsg::tp_cuda_f32_gather(handle, sources.data(), destinations.data(), first.data(), counts.data(), rows, columns, error),
                "Full-rank gather failed: " + error);
        for (int r = 0; r < ranks; ++r) ggml_backend_synchronize(backends[r]);
        for (int r = 0; r < ranks; ++r)
        {
            require(output[r]->get() == expected, "Full-rank uneven gather changed finite F32 bits on rank " + std::to_string(r));
            require(input[r]->get() == locals[r], "Full-rank gather changed a source");
            exact_values += expected.size();
        }
    }
}
}

int main(int argc, char ** argv)
{
    try
    {
        int ranks = 2;
        if (argc == 3 && std::strcmp(argv[1], "--ranks") == 0)
        {
            char * end = nullptr;
            const long parsed = std::strtol(argv[2], &end, 10);
            require(end != argv[2] && *end == '\0' && parsed >= 2 && parsed <= 8, "--ranks must be2..8");
            ranks = static_cast<int>(parsed);
        }
        else require(argc == 1, "Usage: GgmlOpsTpCollectiveGatherTest [--ranks2..8]");
        if (ggml_backend_cuda_get_device_count() < ranks)
        { std::printf("SKIP: needs %d physical CUDA devices; no passing coverage claimed.\n", ranks); return 77; }
        using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
        std::vector<backend_ptr> owned_backends;
        std::vector<ggml_backend_t> backends;
        for (int r = 0; r < ranks; ++r)
        {
            owned_backends.emplace_back(ggml_backend_cuda_init(r), ggml_backend_free);
            require(owned_backends.back() != nullptr, "Cannot initialize CUDA backend " + std::to_string(r));
            backends.push_back(owned_backends.back().get());
        }
        environment("TS_GGML_TP_F32_NCCL", "0");
        require(tsg::tp_cuda_f32_gather_create(backends.data(), 2) == nullptr, "Disabled factory created a communicator");
        environment("TS_GGML_TP_F32_NCCL", "1");
        environment("GGML_CUDA_ALLREDUCE", "internal");
        require(tsg::tp_cuda_f32_gather_create(backends.data(), 2) == nullptr, "Non-NCCL policy created a communicator");
        environment("GGML_CUDA_ALLREDUCE", "nccl");
        require(tsg::tp_cuda_f32_gather_create(backends.data(), 1) == nullptr, "Single-rank factory was accepted");
        require(tsg::tp_cuda_f32_gather_create(nullptr, 2) == nullptr, "Null backend array was accepted");
        std::array<ggml_backend_t, 2> duplicate{backends[0], backends[0]};
        require(tsg::tp_cuda_f32_gather_create(duplicate.data(), 2) == nullptr, "Duplicate CUDA device factory was accepted");
        backend_ptr cpu(ggml_backend_cpu_init(), ggml_backend_free);
        std::array<ggml_backend_t, 2> mixed{backends[0], cpu.get()};
        require(tsg::tp_cuda_f32_gather_create(mixed.data(), 2) == nullptr, "Mixed CPU/CUDA factory was accepted");
        tsg::tp_cuda_f32_gather_free(nullptr);
        for (int lifecycle = 0; lifecycle < 2; ++lifecycle)
        {
            handle_ptr handle(tsg::tp_cuda_f32_gather_create(backends.data(), 2), tsg::tp_cuda_f32_gather_free);
            if (!handle)
            { std::puts("SKIP: optional NCCL gather unavailable; enabled path not validated."); return 77; }
            exact_case(backends.data(), handle.get(), 3, 5, 1, false);
            exact_case(backends.data(), handle.get(), 7, 2, 39, true);
            exact_case(backends.data(), handle.get(), 128, 129, 129, true);
            exact_case(backends.data(), handle.get(), 768, 896, 37, false);
            exact_case(backends.data(), handle.get(), 2560, 2560, 17, false);
            rejection_cases(backends.data(), handle.get());
            // Valid work after all refusals proves validation did not poison it.
            exact_case(backends.data(), handle.get(), 5, 3, 3, true);
        }
        if (ranks > 2)
            for (int lifecycle = 0; lifecycle < 2; ++lifecycle)
            {
                handle_ptr handle(tsg::tp_cuda_f32_gather_create(backends.data(), ranks), tsg::tp_cuda_f32_gather_free);
                if (!handle)
                { std::puts("SKIP: full-rank NCCL gather unavailable; full-rank path not validated."); return 77; }
                many_rank_case(backends.data(), handle.get(), ranks, false);
                many_rank_case(backends.data(), handle.get(), ranks, true);
            }
        std::printf("Passed %zu CUDA gather checks and %zu exact finite F32 values on %d physical ranks; two create/free lifecycles per tested communicator size.\n",
            checks, exact_values, ranks);
        return 0;
    }
    catch (const std::exception & error)
    { std::fprintf(stderr, "FAILED: %s\n", error.what()); return 1; }
}
