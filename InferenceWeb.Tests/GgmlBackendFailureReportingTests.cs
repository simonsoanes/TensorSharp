// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>
/// How a GPU execution failure is reported.
///
/// A command buffer that dies on the GPU is discovered inside
/// ggml_backend_synchronize, which returns void, and ggml_backend_graph_compute
/// returns the status it captured BEFORE that synchronize — so the graph whose
/// buffer died still reports success and only the NEXT graph fails. That is how a
/// Metal out-of-memory during prefill came back as
/// "Native GGML get_rows_quant failed", naming the embedding lookup that happened
/// to open the following forward.
///
/// Two things have to hold for the fix to be worth anything: a healthy backend
/// must never report a failure (a false positive latches for the life of the
/// process and breaks every model), and a failing op must carry the native
/// detail rather than a bare "it failed".
///
/// Runs on whichever backend TS_TEST_GGML_BACKEND selects (default cpu).
/// </summary>
public class GgmlBackendFailureReportingTests
{
    private static GgmlBackendType ConfiguredBackend() =>
        (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu").Trim().ToLowerInvariant() switch
        {
            "cuda" => GgmlBackendType.Cuda,
            "metal" => GgmlBackendType.Metal,
            "vulkan" => GgmlBackendType.Vulkan,
            _ => GgmlBackendType.Cpu,
        };

    private static GgmlAllocator NewAllocator()
    {
        var context = new GgmlContext(new[] { 0 }, ConfiguredBackend());
        return new GgmlAllocator(context, 0);
    }

    [Fact]
    public void HealthyBackendNeverReportsAFailure()
    {
        var allocator = NewAllocator();

        // Enough real work to exercise the compute + synchronize wrappers on
        // whichever backend is selected. ggml logs plenty at INFO/WARN while doing
        // it; none of that may be mistaken for a dead command buffer.
        for (int i = 0; i < 8; i++)
        {
            using var a = Tensor.FromArray(allocator, BuildMatrix(32, 64, 0.011f * (i + 1)));
            using var b = Tensor.FromArray(allocator, BuildMatrix(32, 64, 0.017f));
            using var result = new Tensor(allocator, DType.Float32, 32, 64);
            using var product = Ops.Mul(result, a, b);
            Assert.Equal(
                a.GetElementAsFloat(3, 5) * b.GetElementAsFloat(3, 5),
                product.GetElementAsFloat(3, 5),
                3);
        }

        Assert.False(GgmlBasicOps.HasBackendFailure());
        Assert.Equal(string.Empty, GgmlBasicOps.BackendFailureText());
    }

    [Fact]
    public void FailingOpCarriesTheNativeReasonAndDoesNotLatchTheBackend()
    {
        var allocator = NewAllocator();

        const int embeddingDim = 16;
        long weightBytes = embeddingDim * 4L * sizeof(float);
        IntPtr weights = GgmlBasicOps.AlignedAlloc(weightBytes);
        Assert.NotEqual(IntPtr.Zero, weights);
        try
        {
            // Well-formed enough to reach the native validator, wrong enough to
            // fail it: the result is [2, 8] while the source rows are 16 wide.
            using var result = new Tensor(allocator, DType.Float32, 2, 8);
            using var indices = Tensor.FromArray(allocator, new[] { 1, 3 });

            var ex = Assert.Throws<InvalidOperationException>(() =>
                GgmlBasicOps.GetRowsQuant(
                    result, weights, (int)GgmlTensorType.F32,
                    ne0: embeddingDim, ne1: 4, rawBytes: weightBytes, indices));

            // The native reason, not just the op name.
            Assert.Contains("get_rows_quant", ex.Message);
            Assert.Contains("Shape mismatch", ex.Message);
            // ggml itself said nothing here, so nothing is appended.
            Assert.DoesNotContain("ggml:", ex.Message);

            // A rejected argument is not a dead GPU: the process stays usable.
            Assert.False(GgmlBasicOps.HasBackendFailure());
        }
        finally
        {
            GgmlBasicOps.AlignedFree(weights);
        }
    }

    private static float[,] BuildMatrix(int rows, int cols, float step)
    {
        var data = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                data[r, c] = MathF.Sin((r * cols + c) * step);
        return data;
    }
}
