// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

// A double-precision, two-pass softmax reference checks the streaming kernel's
// geometry and arithmetic independently, including long bidirectional rows.
static void check_vision(ggml_backend_t allocator, ggml_backend_t backend,
                         int queries, int keys, int heads, int batches, bool padded,
                         bool interleaved, bool large_values = false) {
    constexpr int width = 72;
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot initialize vision attention test context");
    input_tensor q(ctx, GGML_TYPE_F32, {width, queries, heads, batches}, padded, interleaved);
    input_tensor k(ctx, GGML_TYPE_F32, {width, keys, heads, batches}, padded, interleaved);
    input_tensor v(ctx, GGML_TYPE_F32, {width, keys, heads, batches}, padded, interleaved);
    const float scale = 1.0f / std::sqrt(float(width));
    auto * output = tsg_vision_attention_f32(ctx, q.tensor, k.tensor, v.tensor, scale);
    auto * graph = ggml_new_graph(ctx);
    // An ordinary downstream node exercises the paired backend's stream order.
    auto * result_tensor = ggml_scale(ctx, output, 0.5f);
    ggml_build_forward_expand(graph, result_tensor);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, allocator);
    require(buffer != nullptr, "Cannot allocate vision attention test graph");
    std::mt19937 random(46581 + keys + queries);
    std::uniform_real_distribution<float> uniform(-1.5f, 1.5f);
    for (auto & value : q.values) value = uniform(random);
    for (auto & value : k.values) value = uniform(random);
    for (auto & value : v.values) value = large_values ? 70000.125f + uniform(random) : uniform(random);
    q.upload(); k.upload(); v.upload();
    timed_compute(backend, graph, 1);
    std::vector<float> result(ggml_nelements(result_tensor));
    ggml_backend_tensor_get(result_tensor, result.data(), 0, result.size() * sizeof(float));
    double maximum_error = 0, squared_error = 0, squared_reference = 0;
    std::vector<double> probabilities(keys);
    for (int batch = 0; batch < batches; ++batch)
    for (int query = 0; query < queries; ++query)
    for (int head = 0; head < heads; ++head) {
        double maximum = -INFINITY, sum = 0;
        for (int key = 0; key < keys; ++key) {
            double dot = 0;
            for (int x = 0; x < width; ++x)
                dot += double(q.at(x, query, head, batch)) * k.at(x, key, head, batch);
            probabilities[key] = dot * scale;
            maximum = std::max(maximum, probabilities[key]);
        }
        for (auto & value : probabilities) { value = std::exp(value - maximum); sum += value; }
        for (int x = 0; x < width; ++x) {
            double expected = 0;
            for (int key = 0; key < keys; ++key)
                expected += probabilities[key] * v.at(x, key, head, batch);
            expected *= 0.5 / sum;
            const size_t index = x + width * (head + heads * (query + queries * batch));
            const double error = std::abs(double(result[index]) - expected);
            require(std::isfinite(result[index]) && error <= 8e-5 + 2e-5 * std::abs(expected),
                "F32 vision attention differs from independent double-precision reference");
            maximum_error = std::max(maximum_error, error);
            squared_error += error * error;
            squared_reference += expected * expected;
        }
    }
    q.check_unchanged(); k.check_unchanged(); v.check_unchanged();
    std::printf("VISION_F32_TEST queries=%d keys=%d heads=%d batches=%d padded=%d interleaved=%d large_values=%d max_abs=%.9g rel_l2=%.9g\n",
        queries, keys, heads, batches, padded, interleaved, large_values, maximum_error,
        std::sqrt(squared_error / std::max(1e-30, squared_reference)));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void run_vision(ggml_backend_t allocator, ggml_backend_t backend) {
    check_vision(allocator, backend, 1, 1, 1, 1, false, false);
    check_vision(allocator, backend, 37, 37, 3, 2, true, false);
    check_vision(allocator, backend, 65, 129, 2, 1, true, true);
    check_vision(allocator, backend, 129, 521, 4, 1, false, false);
    check_vision(allocator, backend, 3, 8193, 2, 1, true, false);
    check_vision(allocator, backend, 65, 257, 2, 1, false, false, true);
}

#ifdef TSG_GGML_USE_CUDA
static void benchmark_vision(ggml_backend_t allocator, ggml_backend_t backend,
                             int queries, int keys, int heads, int repeats) {
    constexpr int width = 72;
    require(queries > 0 && keys > 0 && heads > 0 && repeats > 0, "Invalid vision attention benchmark dimensions");
    std::vector<float> baseline;
    double baseline_us = 0;
    for (int mode = 0; mode < 3; ++mode) {
        auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
        auto * input_ctx = ggml_init({1024 * 1024, nullptr, true});
        require(ctx && input_ctx, "Cannot create vision benchmark context");
        auto * q = ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, width, queries, heads);
        auto * k = ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, width, keys, heads);
        auto * v = ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, width, keys, heads);
        const float scale = 1.0f / std::sqrt(float(width));
        ggml_tensor * output = nullptr;
        if (mode == 2) output = tsg_vision_attention_f32(ctx, q, k, v, scale);
        else {
            // The production CUDA baseline bounds score tiles to 256 MiB and
            // concatenates them. Retain its exact F32-QK / default-PV settings.
            auto * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
            const int64_t chunk = std::min<int64_t>(queries,
                std::max<int64_t>(256, (int64_t(256) * 1024 * 1024) / (int64_t(keys) * heads * sizeof(float))));
            ggml_tensor * accumulated = mode == 1
                ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, queries, heads) : nullptr;
            for (int64_t first = 0; first < queries; first += chunk) {
                const int64_t count = std::min<int64_t>(chunk, queries - first);
                auto * qt = ggml_cont(ctx, ggml_view_3d(ctx, q, width, count, heads, q->nb[1], q->nb[2], first * q->nb[1]));
                auto * scores = ggml_mul_mat(ctx, k, qt);
                ggml_prec_set_acc(scores, GGML_PREC_F32);
                auto * probabilities = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
                auto * weighted = ggml_mul_mat(ctx, vt, probabilities);
                if (mode == 1)
                    accumulated = ggml_set_inplace(ctx, accumulated, weighted, accumulated->nb[1],
                        accumulated->nb[2], accumulated->nb[3], first * accumulated->nb[1]);
                else accumulated = accumulated ? ggml_concat(ctx, accumulated, weighted, 1) : weighted;
            }
            output = ggml_cont(ctx, ggml_permute(ctx, accumulated, 0, 2, 1, 3));
        }
        auto * graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, output);
        ggml_set_output(output);
        auto * input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx, allocator);
        auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(allocator));
        require(input_buffer && galloc && ggml_gallocr_alloc_graph(galloc, graph), "Cannot allocate vision benchmark graph");
        std::mt19937 random(37121);
        std::uniform_real_distribution<float> uniform(-1.5f, 1.5f);
        for (auto * tensor : {q, k, v}) {
            std::vector<float> data(ggml_nelements(tensor));
            for (auto & value : data) value = uniform(random);
            ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
        }
        timed_compute(backend, graph, 2);
        std::vector<double> samples(repeats);
        for (auto & sample : samples) sample = timed_compute(backend, graph, 1);
        std::sort(samples.begin(), samples.end());
        const double median = (samples[(repeats - 1) / 2] + samples[repeats / 2]) / 2;
        std::vector<float> result(ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        const size_t scratch = ggml_gallocr_get_buffer_size(galloc, 0);
        std::printf("VISION_F32_BENCH implementation=%s queries=%d keys=%d heads=%d median_us=%.1f min_us=%.1f max_us=%.1f scratch_bytes=%zu",
            mode == 2 ? "owned-f32" : mode == 1 ? "ordered-f32" : "decomposed-f32",
            queries, keys, heads, median, samples.front(), samples.back(), scratch);
        if (mode) {
            double max_abs = 0, squared_error = 0, squared_reference = 0;
            require(result.size() == baseline.size(), "Vision benchmark output shape mismatch");
            for (size_t i = 0; i < result.size(); ++i) {
                require(std::isfinite(result[i]) && std::isfinite(baseline[i]), "Nonfinite vision benchmark output");
                const double error = double(result[i]) - baseline[i];
                max_abs = std::max(max_abs, std::abs(error));
                squared_error += error * error;
                squared_reference += double(baseline[i]) * baseline[i];
            }
            std::printf(" max_abs=%.9g rel_l2=%.9g speedup=%.4f", max_abs,
                std::sqrt(squared_error / std::max(1e-30, squared_reference)), baseline_us / median);
            if (mode == 1) require(result == baseline, "Ordered vision attention writes must preserve every output bit");
        } else { baseline = std::move(result); baseline_us = median; }
        std::printf("\n");
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(input_buffer);
        ggml_free(input_ctx); ggml_free(ctx);
    }
}
#endif
