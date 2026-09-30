// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

/// <summary>
/// <c>--decode-kernels</c>: the weight-streaming kernels a whole-model engine runs per decode step,
/// timed with CUDA events. The dense matvec goes through the same routing the engines use
/// (<see cref="CudaQuantizedOps.AddmmResidentToFloat32"/> with row-invariant kernels); the routed
/// experts run the per-slot decode kernels. The first shapes are the ones ggml's
/// <c>test-backend-ops perf</c> times (MUL_MAT 4096 x 14336, MUL_MAT_ID 128 experts top-8
/// 768 x 2048), so each line compares with ggml's kernel directly; the last are GLM-5.3-Flash's
/// expert widths. Weights hold random bytes: the time does not depend on the values.
/// </summary>
internal static unsafe class DecodeKernelBench
{
    private readonly record struct QuantType(string Name, int Type, int BlockValues, int BlockBytes)
    {
        public long RowBytes(int values) => (long)values / BlockValues * BlockBytes;
    }

    private static readonly QuantType Q8_0 = new("q8_0", 8, 32, 34);
    private static readonly QuantType Q2_K = new("q2_K", 10, 256, 84);
    private static readonly QuantType Q3_K = new("q3_K", 11, 256, 110);
    private static readonly QuantType Q4_K = new("q4_K", 12, 256, 144);
    private static readonly QuantType Q5_K = new("q5_K", 13, 256, 176);
    private static readonly QuantType Q6_K = new("q6_K", 14, 256, 210);

    private static readonly QuantType IQ4_NL = new("iq4_nl", 20, 32, 18);
    private static readonly QuantType IQ3_S = new("iq3_s", 21, 256, 110);
    private static readonly QuantType IQ4_XS = new("iq4_xs", 23, 256, 136);

    private const int Q81BlockBytes = 36;
    private const int SelectionVariants = 16;

    public static void Run(CudaAllocator allocator, int warmup, int iterations)
    {
        allocator.Context.MakeCurrent();
        IntPtr stream = allocator.Stream.Handle;
        CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out IntPtr end, 0).ThrowOnError();
        var timer = new Timer(stream, start, end, warmup, iterations);
        try
        {
            LaunchGaps(allocator, start, end);

            Console.WriteLine("dense matvec, ggml's MUL_MAT shape (out 4096, in 14336):");
            foreach (int rows in new[] { 1, 4, 8 })
                foreach (QuantType t in new[] { Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K })
                    Dense(allocator, timer, t, rows, 14336, 4096);

            Console.WriteLine("Q8_0 matvec, GLM-5.3's dense shapes (out x in):");
            foreach ((int outDim, int inDim) in new[] { (8192, 4096), (4096, 8192), (2048, 4096), (24, 16384) })
                foreach (int rows in new[] { 1, 4, 8 })
                    Q80Shape(allocator, timer, rows, inDim, outDim);

            using Dsv4Kernels dk = Dsv4Kernels.Create();
            Console.WriteLine("MoE router, one token: the F32 projection (split-K GEMV, splits as given) and the top-k:");
            foreach ((int experts, int used) in new[] { (288, 8), (384, 6) })
                Router(allocator, dk, timer, 4096, experts, used);

            Console.WriteLine("routed experts, ggml's MUL_MAT_ID shape (128 experts, top-8, out 768, in 2048):");
            foreach (int nt in new[] { 1, 4, 8 })
                foreach (QuantType t in new[] { Q8_0, Q4_K, Q6_K })
                    Experts(allocator, dk, timer, t, t, nt, nExpert: 128, nUsed: 8, e: 2048, ff: 768);

            Console.WriteLine("prefill, GLM-5.3-Flash's shapes: int8 MMQ, dequantize-per-call F16 cuBLAS, cached F16 cuBLAS:");
            foreach ((int outDim, int inDim) in new[] { (8192, 4096), (4096, 8192), (2048, 4096) })
                foreach (int rows in new[] { 32, 64, 128, 256, 512, 1024 })
                    PrefillDenseRoutes(allocator, timer, rows, inDim, outDim);
            PrefillExperts(allocator, dk, timer, Q4_K, Q5_K, nt: 512, nExpert: 288, nUsed: 8, e: 4096, ff: 2048);

            Console.WriteLine("routed experts, GLM-5.3-Flash's widths (gate/up 4096 -> 2048, down 2048 -> 4096):");
            foreach (int nt in new[] { 1, 4 })
            {
                Experts(allocator, dk, timer, Q4_K, Q5_K, nt, nExpert: 128, nUsed: 8, e: 4096, ff: 2048);
                Experts(allocator, dk, timer, Q4_K, Q6_K, nt, nExpert: 128, nUsed: 8, e: 4096, ff: 2048);
            }
        }
        finally
        {
            CudaDriverApi.cuEventDestroy(start);
            CudaDriverApi.cuEventDestroy(end);
        }
    }

    /// <summary>Qwen3.8-Flash-Next's decode kernels at its shapes, for ggml-shape-bench's
    /// mm/id cases of the same shapes.</summary>
    public static void RunQwen(CudaAllocator allocator, int warmup, int iterations)
    {
        allocator.Context.MakeCurrent();
        IntPtr stream = allocator.Stream.Handle;
        CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out IntPtr end, 0).ThrowOnError();
        var timer = new Timer(stream, start, end, warmup, iterations);
        try
        {
            Console.WriteLine("Q8_0 matvec, Qwen3.8-Flash-Next's dense shapes (out x in):");
            foreach ((int outDim, int inDim) in new[]
                {
                    (10240, 2560), (6144, 2560), (2560, 6144), (12288, 2560), (320, 10240), (10240, 320), (640, 2560),
                    (2560, 640), (512, 2560),
                })
                foreach (int rows in new[] { 1, 4 })
                    Q80Shape(allocator, timer, rows, inDim, outDim);
            Console.WriteLine("LM head, Q6_K 248320 x 2560:");
            foreach (int rows in new[] { 1, 4 })
                Dense(allocator, timer, Q6_K, rows, 2560, 248320);

            using Dsv4Kernels dk = Dsv4Kernels.Create();
            Console.WriteLine("MoE router, one token (512 experts, top-10):");
            Router(allocator, dk, timer, 2560, 512, 10);

            Console.WriteLine("routed experts, Qwen3.8-Flash-Next's widths (gate/up 2560 -> 640, down 640 -> 2560, 512 experts, top-10):");
            foreach (int nt in new[] { 1, 4 })
            {
                Experts(allocator, dk, timer, IQ3_S, IQ4_NL, nt, nExpert: 512, nUsed: 10, e: 2560, ff: 640);
                Experts(allocator, dk, timer, IQ4_XS, Q8_0, nt, nExpert: 512, nUsed: 10, e: 2560, ff: 640);
            }
        }
        finally
        {
            CudaDriverApi.cuEventDestroy(start);
            CudaDriverApi.cuEventDestroy(end);
        }
    }

    private static void Dense(CudaAllocator allocator, Timer timer, QuantType t, int rows, int inDim, int outDim)
    {
        long bytes = outDim * t.RowBytes(inDim);
        IntPtr w = Upload(allocator, RandomBytes(bytes, 1));
        try
        {
            using var input = Tensor.FromArray(allocator, RandomMatrix(rows, inDim, 2));
            using var output = new Tensor(allocator, DType.Float32, rows, outDim);
            double us = timer.Time(() => CudaQuantizedOps.AddmmResidentToFloat32(
                output, input, w, t.Type, inDim, outDim, rowInvariant: true));
            Report($"  {t.Name} rows={rows}", us, bytes);
        }
        finally
        {
            CudaDriverApi.cuMemFree(w);
        }
    }

    private static void Q80Shape(CudaAllocator allocator, Timer timer, int rows, int inDim, int outDim)
    {
        long bytes = outDim * Q8_0.RowBytes(inDim);
        IntPtr w = Upload(allocator, RandomBytes(bytes, 9));
        try
        {
            using var x = Device(Tensor.FromArray(allocator, RandomMatrix(rows, inDim, 10)));
            using var xq = new Tensor(allocator, DType.UInt8, rows, (long)(inDim / 32) * Q81BlockBytes);
            using var y = new Tensor(allocator, DType.Float32, rows, outDim);
            IntPtr stream = allocator.Stream.Handle;
            allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(x), Ptr(xq), inDim, rows, stream, warpCooperative: true);
            double us = timer.Time(() => allocator.Kernels.LaunchQuantMatmulQ80VecRows(
                w, Ptr(xq), Ptr(y), inDim, outDim, rows, stream));
            Report($"  {outDim}x{inDim} rows={rows}", us, bytes);
        }
        finally
        {
            CudaDriverApi.cuMemFree(w);
        }
    }

    /// <summary>What a kernel launch costs on the device when kernels run back to back: a one-float
    /// fill and a real small kernel (the 24 x 16384 hyper-connection mix), queued on the stream one
    /// after another, then the same sequence captured once and replayed as a CUDA graph. A decode
    /// step is ~1500 such launches.</summary>
    private static void LaunchGaps(CudaAllocator allocator, IntPtr start, IntPtr end)
    {
        IntPtr stream = allocator.Stream.Handle;
        const int n = 500, reps = 10;
        using var buf = new Tensor(allocator, DType.Float32, 64);
        int inDim = 16384, outDim = 24;
        IntPtr w = Upload(allocator, RandomBytes(outDim * Q8_0.RowBytes(inDim), 13));
        using var x = Device(Tensor.FromArray(allocator, RandomMatrix(1, inDim, 14)));
        using var xq = new Tensor(allocator, DType.UInt8, 1, (long)(inDim / 32) * Q81BlockBytes);
        using var y = new Tensor(allocator, DType.Float32, 1, outDim);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(x), Ptr(xq), inDim, 1, stream, warpCooperative: true);
        try
        {
            var kernels = new (string Name, Action Launch)[]
            {
                ("fill 1 float", () => allocator.Kernels.LaunchFillF32(Ptr(buf), 1, 0f, stream)),
                ("q8_0 matvec 24x16384", () => allocator.Kernels.LaunchQuantMatmulQ80VecRows(w, Ptr(xq), Ptr(y), inDim, outDim, 1, stream)),
            };
            Console.WriteLine($"launch gaps ({n} back-to-back launches, stream vs one captured graph):");
            foreach (var (name, launch) in kernels)
            {
                for (int i = 0; i < 50; i++) launch();
                CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
                for (int r = 0; r < reps; r++)
                    for (int i = 0; i < n; i++)
                        launch();
                CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float streamMs, start, end).ThrowOnError();

                // Pre-queued: the device is parked behind ~2 ms of real work while the host queues the
                // launches, so the span after it is the device's own cost per launch, host excluded.
                for (int i = 0; i < 400; i++)
                    allocator.Kernels.LaunchQuantMatmulQ80VecRows(w, Ptr(xq), Ptr(y), inDim, outDim, 1, stream);
                CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
                for (int i = 0; i < n; i++)
                    launch();
                CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float queuedMs, start, end).ThrowOnError();

                CudaDriverApi.cuStreamBeginCapture(stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED).ThrowOnError();
                for (int i = 0; i < n; i++)
                    launch();
                CudaDriverApi.cuStreamEndCapture(stream, out IntPtr graph).ThrowOnError();
                CudaDriverApi.cuGraphInstantiateWithFlags(out IntPtr exec, graph, 0).ThrowOnError();
                CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
                for (int r = 0; r < reps; r++)
                    CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float graphMs, start, end).ThrowOnError();
                CudaDriverApi.cuGraphExecDestroy(exec);
                CudaDriverApi.cuGraphDestroy(graph);
                Console.WriteLine($"  {name,-24} stream {streamMs * 1000.0 / (n * reps),6:F2} us/launch   pre-queued {queuedMs * 1000.0 / n,6:F2} us/launch   graph {graphMs * 1000.0 / (n * reps),6:F2} us/node");
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree(w);
        }
    }

    private static void Router(CudaAllocator allocator, Dsv4Kernels dk, Timer timer, int e, int experts, int used)
    {
        IntPtr stream = allocator.Stream.Handle;
        using var w = Device(Tensor.FromArray(allocator, RandomMatrix(experts, e, 11)));
        using var x = Device(Tensor.FromArray(allocator, RandomMatrix(1, e, 12)));
        using var logits = new Tensor(allocator, DType.Float32, 1, experts);
        using var scratch = new Tensor(allocator, DType.Float32, (long)Dsv4Kernels.GemvScratchFloats);
        using var sel = new Tensor(allocator, DType.Int32, 1, used);
        using var selW = new Tensor(allocator, DType.Float32, 1, used);
        foreach (int splits in new[] { 1, 2, 4, 8 })
        {
            double us = timer.Time(() => dk.Gemv(Ptr(x), Ptr(w), Ptr(logits), e, experts, 1, stream, Ptr(scratch), splits));
            Report($"  gemv f32 {e} -> {experts}, {splits} split(s)", us, (long)experts * e * 4);
        }
        double sus = timer.Time(() => dk.MoeSelect(logits, IntPtr.Zero, IntPtr.Zero, null, sel, selW, experts, used, 1, 1f, 1,
            stream, Dsv4Kernels.RouterSigmoid));
        Report($"  select top-{used} of {experts}", sus, (long)experts * 4);
    }

    /// <summary>One decode step's routed experts for <paramref name="nt"/> tokens: the gate/up
    /// kernel (both matrices of each selected expert) and the down kernel, timed separately. The
    /// expert selection rotates through <see cref="SelectionVariants"/> sets so successive
    /// iterations do not find the same experts in L2 (ggml's perf mode redraws its ids too).</summary>
    private static void Experts(CudaAllocator allocator, Dsv4Kernels dk, Timer timer, QuantType gu, QuantType down,
        int nt, int nExpert, int nUsed, int e, int ff)
    {
        int slots = nt * nUsed;
        long guRow = gu.RowBytes(e), downRow = down.RowBytes(ff);
        IntPtr gate = Upload(allocator, RandomBytes(nExpert * ff * guRow, 3));
        IntPtr up = Upload(allocator, RandomBytes(nExpert * ff * guRow, 4));
        IntPtr downW = Upload(allocator, RandomBytes(nExpert * e * downRow, 5));
        var sels = new Tensor[SelectionVariants];
        try
        {
            var rng = new Random(6);
            for (int v = 0; v < sels.Length; v++)
            {
                var ids = new int[slots];
                for (int tok = 0; tok < nt; tok++)
                {
                    var chosen = new HashSet<int>();
                    for (int j = 0; j < nUsed; j++)
                    {
                        int id;
                        do id = rng.Next(nExpert); while (!chosen.Add(id));
                        ids[tok * nUsed + j] = id;
                    }
                }
                sels[v] = Device(Tensor.FromArray(allocator, ids));
            }
            using var x = Device(Tensor.FromArray(allocator, RandomMatrix(nt, e, 7)));
            using var h = Device(Tensor.FromArray(allocator, RandomMatrix(slots, ff, 8)));
            using var actA = new Tensor(allocator, DType.UInt8, nt, (long)(e / 32) * Q81BlockBytes);
            using var actB = new Tensor(allocator, DType.UInt8, slots, (long)(ff / 32) * Q81BlockBytes);
            using var gateOut = new Tensor(allocator, DType.Float32, slots, ff);
            using var upOut = new Tensor(allocator, DType.Float32, slots, ff);
            using var downOut = new Tensor(allocator, DType.Float32, slots, e);
            IntPtr stream = allocator.Stream.Handle;
            allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(x), Ptr(actA), e, nt, stream, warpCooperative: true);
            allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(h), Ptr(actB), ff, slots, stream, warpCooperative: true);

            int variant = 0;
            double guUs = timer.Time(() => dk.MoeGateUpDecode(gate, up, actA, sels[variant++ % sels.Length],
                gateOut, upOut, gu.Type, ff, e, guRow, nt, nUsed, stream));
            variant = 0;
            double downUs = timer.Time(() => dk.MoeDownDecode(downW, actB, sels[variant++ % sels.Length], downOut,
                down.Type, e, ff, downRow, slots, stream));
            // Bytes of the distinct experts one step reads (a token's selections are distinct; across
            // tokens they may repeat, so this is an upper bound past one token).
            Report($"  gate+up {gu.Name} nt={nt} ({e} -> {ff})", guUs, slots * 2L * ff * guRow);
            Report($"  down    {down.Name} nt={nt} ({ff} -> {e})", downUs, slots * (long)e * downRow);
        }
        finally
        {
            foreach (Tensor s in sels)
                s?.Dispose();
            CudaDriverApi.cuMemFree(gate);
            CudaDriverApi.cuMemFree(up);
            CudaDriverApi.cuMemFree(downW);
        }
    }

    private static void PrefillDenseRoutes(CudaAllocator allocator, Timer timer, int rows, int inDim, int outDim)
    {
        long bytes = outDim * Q8_0.RowBytes(inDim);
        IntPtr w = Upload(allocator, RandomBytes(bytes, 17));
        try
        {
            using var input = Device(Tensor.FromArray(allocator, RandomMatrix(rows, inDim, 18)));
            using var output = new Tensor(allocator, DType.Float32, rows, outDim);
            using var wF16 = new Tensor(allocator, DType.Float16, (long)outDim * inDim);
            IntPtr stream = allocator.Stream.Handle;
            allocator.Kernels.LaunchDequantWeightQ80F16(w, Ptr(wF16), (long)outDim * inDim, stream);
            double mmq = timer.Time(() => CudaQuantizedOps.AddmmResidentToFloat32(output, input, w, Q8_0.Type, inDim, outDim));
            bool saved = CudaQuantizedOps.Q80MmqEnabled;
            CudaQuantizedOps.Q80MmqEnabled = false;
            double perCall;
            try
            {
                perCall = timer.Time(() => CudaQuantizedOps.AddmmResidentToFloat32(output, input, w, Q8_0.Type, inDim, outDim));
            }
            finally
            {
                CudaQuantizedOps.Q80MmqEnabled = saved;
            }
            double cached = timer.Time(() => CudaQuantizedOps.AddmmResidentToFloat32(output, input, Ptr(wF16), 1, inDim, outDim));
            Report($"  q8_0 {outDim}x{inDim} rows={rows} int8 mmq", mmq, bytes);
            Report($"  q8_0 {outDim}x{inDim} rows={rows} f16 per call", perCall, bytes);
            Report($"  q8_0 {outDim}x{inDim} rows={rows} f16 cached", cached, bytes);
        }
        finally
        {
            CudaDriverApi.cuMemFree(w);
        }
    }

    /// <summary>A prefill ubatch's routed experts through the engines' own path (grouping plan,
    /// tensor-core gate/up and down, clamped SwiGLU, weighted scatter-add): uniformly random
    /// top-k selections over all experts.</summary>
    private static void PrefillExperts(CudaAllocator allocator, Dsv4Kernels dk, Timer timer, QuantType gu, QuantType down,
        int nt, int nExpert, int nUsed, int e, int ff)
    {
        long guRow = gu.RowBytes(e), downRow = down.RowBytes(ff);
        IntPtr gate = Upload(allocator, RandomBytes(nExpert * ff * guRow, 3));
        IntPtr up = Upload(allocator, RandomBytes(nExpert * ff * guRow, 4));
        IntPtr downW = Upload(allocator, RandomBytes(nExpert * e * downRow, 5));
        var owned = new List<Tensor>();
        try
        {
            var scratch = new CudaMoeScratch((type, sizes) =>
            {
                var t = new Tensor(allocator, type, sizes);
                owned.Add(t);
                return t;
            }, nt, nExpert, nUsed, e, ff);
            var rng = new Random(15);
            var ids = new int[nt * nUsed];
            var weights = new float[nt * nUsed];
            for (int t = 0; t < nt; t++)
            {
                var chosen = new HashSet<int>();
                for (int j = 0; j < nUsed; j++)
                {
                    int id;
                    do id = rng.Next(nExpert); while (!chosen.Add(id));
                    ids[t * nUsed + j] = id;
                    weights[t * nUsed + j] = 1f / nUsed;
                }
            }
            IntPtr stream = allocator.Stream.Handle;
            fixed (int* pi = ids)
                CudaDriverApi.cuMemcpyHtoD(Ptr(scratch.Sel), (IntPtr)pi, (UIntPtr)(ulong)(ids.Length * 4)).ThrowOnError();
            fixed (float* pw = weights)
                CudaDriverApi.cuMemcpyHtoD(Ptr(scratch.SelW), (IntPtr)pw, (UIntPtr)(ulong)(weights.Length * 4)).ThrowOnError();
            using var cur = Device(Tensor.FromArray(allocator, RandomMatrix(nt, e, 16)));
            using var shDown = new Tensor(allocator, DType.Float32, nt, e);
            using var ffnOut = new Tensor(allocator, DType.Float32, nt, e);
            var g = new DeviceWeight { Ptr = gate, Type = gu.Type, Ne0 = e, Ne1 = ff, RowBytes = guRow };
            var u = new DeviceWeight { Ptr = up, Type = gu.Type, Ne0 = e, Ne1 = ff, RowBytes = guRow };
            var d = new DeviceWeight { Ptr = downW, Type = down.Type, Ne0 = ff, Ne1 = e, RowBytes = downRow };
            double us = timer.Time(() => CudaMoe.Experts(dk, allocator.Kernels, scratch, g, u, d, cur, shDown, ffnOut,
                nt, nUsed, nExpert, e, ff, 10f, 16, stream));
            // Every expert is picked at 512 tokens x 8: all of them stream once.
            long bytes = nExpert * (2L * ff * guRow + (long)e * downRow);
            Report($"  experts {gu.Name}/{down.Name} nt={nt} ({nExpert} experts, top-{nUsed})", us, bytes);
        }
        finally
        {
            foreach (Tensor t in owned)
                t.Dispose();
            CudaDriverApi.cuMemFree(gate);
            CudaDriverApi.cuMemFree(up);
            CudaDriverApi.cuMemFree(downW);
        }
    }

    private static void Report(string label, double us, long bytes)
        => Console.WriteLine($"{label,-44} {us,9:F2} us  {bytes / (us * 1e3),7:F1} GB/s");

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

    /// <summary>FromArray fills only the host side of a CUDA tensor; the raw-pointer kernels need
    /// the device copy.</summary>
    private static Tensor Device(Tensor t)
    {
        t.Storage.EnsureDeviceCurrent();
        return t;
    }

    private static IntPtr Upload(CudaAllocator allocator, byte[] data)
    {
        allocator.Context.MakeCurrent();
        // 16 slack bytes, as the engines' weight arenas carry, for the kernels' widest tail loads.
        CudaDriverApi.cuMemAlloc(out IntPtr p, (UIntPtr)(ulong)(data.LongLength + 16)).ThrowOnError();
        fixed (byte* src = data)
            CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)src, (UIntPtr)(ulong)data.LongLength).ThrowOnError();
        return p;
    }

    private static byte[] RandomBytes(long count, int seed)
    {
        var b = new byte[count];
        new Random(seed).NextBytes(b);
        return b;
    }

    private static float[,] RandomMatrix(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var m = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m[r, c] = (float)(rng.NextDouble() * 2 - 1);
        return m;
    }

    private sealed class Timer
    {
        private readonly IntPtr _stream, _start, _end;
        private readonly int _warmup, _iterations;

        public Timer(IntPtr stream, IntPtr start, IntPtr end, int warmup, int iterations)
        {
            _stream = stream;
            _start = start;
            _end = end;
            _warmup = warmup;
            _iterations = iterations;
        }

        /// <summary>Mean microseconds per call, measured on the stream between two events.</summary>
        public double Time(Action call)
        {
            for (int i = 0; i < _warmup; i++)
                call();
            CudaDriverApi.cuStreamSynchronize(_stream).ThrowOnError();
            CudaDriverApi.cuEventRecord(_start, _stream).ThrowOnError();
            for (int i = 0; i < _iterations; i++)
                call();
            CudaDriverApi.cuEventRecord(_end, _stream).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(_end).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float ms, _start, _end).ThrowOnError();
            return ms * 1000.0 / _iterations;
        }
    }
}
