// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Parity battery for the matmul module (GgmlManagedOps.Matmul.cs vs
// ggml_ops_matmul.cpp): Addmm, AddmmQuant, GetRowsQuant, AddmmQuantBatch,
// AddmmBatch, MulMatId and AddId.
using System;
using System.Runtime.InteropServices;
using TensorSharp.GGML;

internal static unsafe class CasesMatmul
{
    private const string DllName = "GgmlOps";

    static CasesMatmul()
    {
        // The locally declared ggml imports below live in this test assembly,
        // which has no DllImport resolver of its own. Delegate resolution to
        // the TensorSharp.Backends.GGML assembly so "GgmlOps" maps to the same
        // module the backend loaded. If registration is not possible (another
        // resolver already owns this assembly), the default loaded-module
        // lookup still finds GgmlOps because the GgmlContext created it first.
        GgmlNative.EnsureImportResolverRegistered();
        try
        {
            NativeLibrary.SetDllImportResolver(
                typeof(CasesMatmul).Assembly,
                (name, _, _) => name == DllName
                    ? NativeLibrary.Load(DllName, typeof(GgmlNative).Assembly, null)
                    : IntPtr.Zero);
        }
        catch (InvalidOperationException)
        {
            // A resolver is already registered for this assembly; rely on it.
        }
    }

    // ggml quantization entry point (statically linked ggml re-exported by
    // GgmlOps). imatrix is unused (null) for the legacy quant formats.
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern nuint ggml_quantize_chunk(int type, float* src, void* dst, long start, long nrows, long n_per_row, float* imatrix);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern nuint ggml_row_size(int type, long ne);

    private const int GGML_TYPE_F32 = 0;
    private const int GGML_TYPE_Q4_0 = 2;
    private const int GGML_TYPE_Q8_0 = 8;

    public static void Run(Harness.Dump dump)
    {
        RunAddmmCases(dump);
        RunAddmmQuantCases(dump);
        RunGetRowsQuantCases(dump);
        RunAddmmQuantBatchCases(dump);
        RunAddmmBatchCases(dump);
        RunMulMatIdCases(dump);
        RunAddIdCases(dump);
    }

    // Quantizes deterministic float data (ne1 rows of ne0 values) into a fresh
    // aligned buffer. Row length must be a multiple of the quant block size
    // (32 for Q8_0/Q4_0).
    private static Harness.AlignedBuffer BuildQuantWeight(int ggmlType, int ne0, int ne1, out long rawBytes)
    {
        long rowSize = (long)ggml_row_size(ggmlType, ne0);
        rawBytes = rowSize * ne1;

        var floats = new float[(long)ne1 * ne0];
        Harness.FillDeterministic(floats);

        var buffer = new Harness.AlignedBuffer(rawBytes);
        fixed (float* src = floats)
        {
            ggml_quantize_chunk(ggmlType, src, (void*)buffer.Ptr, 0, ne1, ne0, null);
        }
        return buffer;
    }

    // ------------------------------------------------------------------
    // Addmm: result = beta * src + alpha * (m1 x m2)
    // ------------------------------------------------------------------

    public static void RunAddmmCases(Harness.Dump dump)
    {
        const int Rows = 7, Cols = 12, Shared = 16;
        long m1Bytes = (long)Rows * Shared * sizeof(float);
        long m2Bytes = (long)Shared * Cols * sizeof(float);
        long srcBytes = (long)Rows * Cols * sizeof(float);
        long dstBytes = srcBytes;

        using var m1 = new Harness.AlignedBuffer(m1Bytes);
        using var m2 = new Harness.AlignedBuffer(m2Bytes);
        using var src = new Harness.AlignedBuffer(srcBytes);
        using var dst = new Harness.AlignedBuffer(dstBytes);

        // m2 in the direct (transposed) layout the runtime expects for
        // zero-copy weights: element (s, c) at s * 1 + c * Shared.
        var m1View = new GgmlTensorView2D(m1.Ptr, Rows, Shared, Shared, 1, m1Bytes);
        var m2View = new GgmlTensorView2D(m2.Ptr, Shared, Cols, 1, Shared, m2Bytes);
        var srcView = new GgmlTensorView2D(src.Ptr, Rows, Cols, Cols, 1, srcBytes);
        var dstView = new GgmlTensorView2D(dst.Ptr, Rows, Cols, Cols, 1, dstBytes);

        // No bias (beta == 0 skips the src descriptor entirely).
        {
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2.Floats);
            dst.Floats.Clear();
            double ns = Harness.TimeCase(() => GgmlNative.Addmm(dstView, default, m1View, m2View, 0f, 1f));
            dump.Add("matmul/addmm-nobias", dst.RawBytes, ns);
        }

        // Bias with beta == 1 (no scale node on src).
        {
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2.Floats);
            Harness.FillDeterministic(src.Floats);
            dst.Floats.Clear();
            double ns = Harness.TimeCase(() => GgmlNative.Addmm(dstView, srcView, m1View, m2View, 1f, 1f));
            dump.Add("matmul/addmm-bias", dst.RawBytes, ns);
        }

        // Row-vector bias broadcast over all result rows.
        {
            long rowBytes = (long)Cols * sizeof(float);
            using var srcRow = new Harness.AlignedBuffer(rowBytes);
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2.Floats);
            Harness.FillDeterministic(srcRow.Floats);
            dst.Floats.Clear();
            var srcRowView = new GgmlTensorView2D(srcRow.Ptr, 1, Cols, Cols, 1, rowBytes);
            double ns = Harness.TimeCase(() => GgmlNative.Addmm(dstView, srcRowView, m1View, m2View, 1f, 1f));
            dump.Add("matmul/addmm-bias-broadcast", dst.RawBytes, ns);
        }

        // alpha != 1 and beta != 1 exercise both ggml_scale nodes.
        {
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2.Floats);
            Harness.FillDeterministic(src.Floats);
            dst.Floats.Clear();
            double ns = Harness.TimeCase(() => GgmlNative.Addmm(dstView, srcView, m1View, m2View, 2f, 0.5f));
            dump.Add("matmul/addmm-alphabeta", dst.RawBytes, ns);
        }

        // Column-major m1 (stride1 != 1) forces the packed-standard m1 path.
        {
            using var m1T = new Harness.AlignedBuffer(m1Bytes);
            Harness.FillDeterministic(m1T.Floats);
            Harness.FillDeterministic(m2.Floats);
            dst.Floats.Clear();
            var m1TView = new GgmlTensorView2D(m1T.Ptr, Rows, Shared, 1, Rows, m1Bytes);
            double ns = Harness.TimeCase(() => GgmlNative.Addmm(dstView, default, m1TView, m2View, 0f, 1f));
            dump.Add("matmul/addmm-m1-transposed", dst.RawBytes, ns);
        }

        // Row-major m2 (stride0 != 1) forces the packed-m2 path.
        {
            using var m2RowMajor = new Harness.AlignedBuffer(m2Bytes);
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2RowMajor.Floats);
            dst.Floats.Clear();
            var m2RowMajorView = new GgmlTensorView2D(m2RowMajor.Ptr, Shared, Cols, Cols, 1, m2Bytes);
            double ns = Harness.TimeCase(() => GgmlNative.Addmm(dstView, default, m1View, m2RowMajorView, 0f, 1f));
            dump.Add("matmul/addmm-m2-rowmajor", dst.RawBytes, ns);
        }

        // Direct m2 with a padded column stride (stride1 > dim0).
        {
            const int PaddedStride = Shared + 4;
            long paddedBytes = (((Cols - 1L) * PaddedStride) + Shared) * sizeof(float);
            using var m2Padded = new Harness.AlignedBuffer(paddedBytes);
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2Padded.Floats);
            dst.Floats.Clear();
            var m2PaddedView = new GgmlTensorView2D(m2Padded.Ptr, Shared, Cols, 1, PaddedStride, paddedBytes);
            double ns = Harness.TimeCase(() => GgmlNative.Addmm(dstView, default, m1View, m2PaddedView, 0f, 1f));
            dump.Add("matmul/addmm-m2-padded", dst.RawBytes, ns);
        }

        // Padded result rows (raw span larger than the logical bytes).
        {
            const int PaddedStride = Cols + 4;
            long paddedBytes = (((Rows - 1L) * PaddedStride) + Cols) * sizeof(float);
            using var dstPadded = new Harness.AlignedBuffer(paddedBytes);
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2.Floats);
            dstPadded.Floats.Clear();
            var dstPaddedView = new GgmlTensorView2D(dstPadded.Ptr, Rows, Cols, PaddedStride, 1, paddedBytes);
            double ns = Harness.TimeCase(() => GgmlNative.Addmm(dstPaddedView, default, m1View, m2View, 0f, 1f));
            dump.Add("matmul/addmm-result-padded", dstPadded.RawBytes, ns);
        }
    }

    // ------------------------------------------------------------------
    // AddmmQuant: result = m1 x dequant(m2)
    // ------------------------------------------------------------------

    public static void RunAddmmQuantCases(Harness.Dump dump)
    {
        // Q8_0 weight above the 4096-byte cache gate: the repeated TimeCase
        // calls make every call after the first bind through the cached
        // weight buffer, so the dumped result comes from the cache-hit path.
        {
            const int Rows = 5, In = 64, Out = 64;
            using var weight = BuildQuantWeight(GGML_TYPE_Q8_0, In, Out, out long weightBytes);
            long m1Bytes = (long)Rows * In * sizeof(float);
            long dstBytes = (long)Rows * Out * sizeof(float);
            using var m1 = new Harness.AlignedBuffer(m1Bytes);
            using var dst = new Harness.AlignedBuffer(dstBytes);

            Harness.FillDeterministic(m1.Floats);
            dst.Floats.Clear();
            var m1View = new GgmlTensorView2D(m1.Ptr, Rows, In, In, 1, m1Bytes);
            var dstView = new GgmlTensorView2D(dst.Ptr, Rows, Out, Out, 1, dstBytes);
            double ns = Harness.TimeCase(() => GgmlNative.AddmmQuant(dstView, m1View, weight.Ptr, GGML_TYPE_Q8_0, In, Out, weightBytes));
            dump.Add("matmul/addmmquant-q8_0", dst.RawBytes, ns);

            // Same weight pointer again with fresh activations: every call in
            // this case starts from an already-populated cache entry.
            Harness.FillDeterministic(m1.Floats, offset: 0.25f);
            dst.Floats.Clear();
            double nsRepeat = Harness.TimeCase(() => GgmlNative.AddmmQuant(dstView, m1View, weight.Ptr, GGML_TYPE_Q8_0, In, Out, weightBytes));
            dump.Add("matmul/addmmquant-q8_0-repeat", dst.RawBytes, nsRepeat);
        }

        // Q4_0 weight above the cache gate.
        {
            const int Rows = 3, In = 64, Out = 128;
            using var weight = BuildQuantWeight(GGML_TYPE_Q4_0, In, Out, out long weightBytes);
            long m1Bytes = (long)Rows * In * sizeof(float);
            long dstBytes = (long)Rows * Out * sizeof(float);
            using var m1 = new Harness.AlignedBuffer(m1Bytes);
            using var dst = new Harness.AlignedBuffer(dstBytes);

            Harness.FillDeterministic(m1.Floats);
            dst.Floats.Clear();
            var m1View = new GgmlTensorView2D(m1.Ptr, Rows, In, In, 1, m1Bytes);
            var dstView = new GgmlTensorView2D(dst.Ptr, Rows, Out, Out, 1, dstBytes);
            double ns = Harness.TimeCase(() => GgmlNative.AddmmQuant(dstView, m1View, weight.Ptr, GGML_TYPE_Q4_0, In, Out, weightBytes));
            dump.Add("matmul/addmmquant-q4_0", dst.RawBytes, ns);
        }

        // Q8_0 weight below the 4096-byte cache gate: per-call weight upload.
        {
            const int Rows = 4, In = 32, Out = 48;
            using var weight = BuildQuantWeight(GGML_TYPE_Q8_0, In, Out, out long weightBytes);
            long m1Bytes = (long)Rows * In * sizeof(float);
            long dstBytes = (long)Rows * Out * sizeof(float);
            using var m1 = new Harness.AlignedBuffer(m1Bytes);
            using var dst = new Harness.AlignedBuffer(dstBytes);

            Harness.FillDeterministic(m1.Floats);
            dst.Floats.Clear();
            var m1View = new GgmlTensorView2D(m1.Ptr, Rows, In, In, 1, m1Bytes);
            var dstView = new GgmlTensorView2D(dst.Ptr, Rows, Out, Out, 1, dstBytes);
            double ns = Harness.TimeCase(() => GgmlNative.AddmmQuant(dstView, m1View, weight.Ptr, GGML_TYPE_Q8_0, In, Out, weightBytes));
            dump.Add("matmul/addmmquant-q8_0-small", dst.RawBytes, ns);

            // Column-major m1 forces the packed-standard fallback path.
            using var m1T = new Harness.AlignedBuffer(m1Bytes);
            Harness.FillDeterministic(m1T.Floats);
            dst.Floats.Clear();
            var m1TView = new GgmlTensorView2D(m1T.Ptr, Rows, In, 1, Rows, m1Bytes);
            double nsT = Harness.TimeCase(() => GgmlNative.AddmmQuant(dstView, m1TView, weight.Ptr, GGML_TYPE_Q8_0, In, Out, weightBytes));
            dump.Add("matmul/addmmquant-m1-transposed", dst.RawBytes, nsT);
        }
    }

    // ------------------------------------------------------------------
    // GetRowsQuant: result[i] = dequant(src[indices[i]])
    // ------------------------------------------------------------------

    public static void RunGetRowsQuantCases(Harness.Dump dump)
    {
        // Q8_0 source above the cache gate (I32 indices).
        {
            const int Ne0 = 64, Ne1 = 64, Picks = 7;
            using var srcQuant = BuildQuantWeight(GGML_TYPE_Q8_0, Ne0, Ne1, out long srcBytes);
            long dstBytes = (long)Picks * Ne0 * sizeof(float);
            using var dst = new Harness.AlignedBuffer(dstBytes);
            using var indices = new Harness.AlignedBuffer(Picks * sizeof(int));

            Span<int> indexValues = new Span<int>((void*)indices.Ptr, Picks);
            int[] picks = { 5, 1, 63, 0, 17, 5, 42 };
            picks.CopyTo(indexValues);

            dst.Floats.Clear();
            var dstView = new GgmlTensorView2D(dst.Ptr, Picks, Ne0, Ne0, 1, dstBytes);
            var indicesDesc = new GgmlContiguousTensor(indices.Ptr, Picks, TensorSharp.DType.Int32);
            double ns = Harness.TimeCase(() => GgmlNative.GetRowsQuant(dstView, srcQuant.Ptr, GGML_TYPE_Q8_0, Ne0, Ne1, srcBytes, indicesDesc));
            dump.Add("matmul/getrowsquant-q8_0", dst.RawBytes, ns);
        }

        // Q4_0 source above the cache gate.
        {
            const int Ne0 = 64, Ne1 = 128, Picks = 5;
            using var srcQuant = BuildQuantWeight(GGML_TYPE_Q4_0, Ne0, Ne1, out long srcBytes);
            long dstBytes = (long)Picks * Ne0 * sizeof(float);
            using var dst = new Harness.AlignedBuffer(dstBytes);
            using var indices = new Harness.AlignedBuffer(Picks * sizeof(int));

            Span<int> indexValues = new Span<int>((void*)indices.Ptr, Picks);
            int[] picks = { 127, 0, 64, 3, 3 };
            picks.CopyTo(indexValues);

            dst.Floats.Clear();
            var dstView = new GgmlTensorView2D(dst.Ptr, Picks, Ne0, Ne0, 1, dstBytes);
            var indicesDesc = new GgmlContiguousTensor(indices.Ptr, Picks, TensorSharp.DType.Int32);
            double ns = Harness.TimeCase(() => GgmlNative.GetRowsQuant(dstView, srcQuant.Ptr, GGML_TYPE_Q4_0, Ne0, Ne1, srcBytes, indicesDesc));
            dump.Add("matmul/getrowsquant-q4_0", dst.RawBytes, ns);
        }

        // F32 (unquantized) source through the same entry point.
        {
            const int Ne0 = 48, Ne1 = 16, Picks = 6;
            long srcBytes = (long)Ne1 * Ne0 * sizeof(float);
            long dstBytes = (long)Picks * Ne0 * sizeof(float);
            using var src = new Harness.AlignedBuffer(srcBytes);
            using var dst = new Harness.AlignedBuffer(dstBytes);
            using var indices = new Harness.AlignedBuffer(Picks * sizeof(int));

            Harness.FillDeterministic(src.Floats);
            Span<int> indexValues = new Span<int>((void*)indices.Ptr, Picks);
            int[] picks = { 15, 2, 0, 9, 2, 7 };
            picks.CopyTo(indexValues);

            dst.Floats.Clear();
            var dstView = new GgmlTensorView2D(dst.Ptr, Picks, Ne0, Ne0, 1, dstBytes);
            var indicesDesc = new GgmlContiguousTensor(indices.Ptr, Picks, TensorSharp.DType.Int32);
            double ns = Harness.TimeCase(() => GgmlNative.GetRowsQuant(dstView, src.Ptr, GGML_TYPE_F32, Ne0, Ne1, srcBytes, indicesDesc));
            dump.Add("matmul/getrowsquant-f32", dst.RawBytes, ns);
        }

        // F32 index tensor (values converted to int inside the op).
        {
            const int Ne0 = 32, Ne1 = 8, Picks = 4;
            using var srcQuant = BuildQuantWeight(GGML_TYPE_Q8_0, Ne0, Ne1, out long srcBytes);
            long dstBytes = (long)Picks * Ne0 * sizeof(float);
            using var dst = new Harness.AlignedBuffer(dstBytes);
            using var indices = new Harness.AlignedBuffer(Picks * sizeof(float));

            Span<float> indexValues = indices.Floats;
            indexValues[0] = 7f;
            indexValues[1] = 0f;
            indexValues[2] = 3f;
            indexValues[3] = 3f;

            dst.Floats.Clear();
            var dstView = new GgmlTensorView2D(dst.Ptr, Picks, Ne0, Ne0, 1, dstBytes);
            var indicesDesc = new GgmlContiguousTensor(indices.Ptr, Picks, TensorSharp.DType.Float32);
            double ns = Harness.TimeCase(() => GgmlNative.GetRowsQuant(dstView, srcQuant.Ptr, GGML_TYPE_Q8_0, Ne0, Ne1, srcBytes, indicesDesc));
            dump.Add("matmul/getrowsquant-f32indices", dst.RawBytes, ns);
        }
    }

    // ------------------------------------------------------------------
    // AddmmQuantBatch: result[b] = m1[b] x dequant(weights[b])
    // ------------------------------------------------------------------

    public static void RunAddmmQuantBatchCases(Harness.Dump dump)
    {
        // Q8_0: one blob holding batchCount weights back to back. Quantizing
        // Batch*Out rows of In values in one chunk produces exactly the
        // concatenated per-batch weights.
        {
            const int Batch = 3, In = 32, Out = 48;
            using var blob = BuildQuantWeight(GGML_TYPE_Q8_0, In, Batch * Out, out long blobBytes);
            long perWeightBytes = (long)ggml_row_size(GGML_TYPE_Q8_0, In) * Out;
            long[] offsets = { 0, perWeightBytes, 2 * perWeightBytes };
            long[] ne1Arr = { Out, Out, Out };

            long m1Bytes = (long)Batch * In * sizeof(float);
            long dstBytes = (long)Batch * Out * sizeof(float);
            using var m1 = new Harness.AlignedBuffer(m1Bytes);
            using var dst = new Harness.AlignedBuffer(dstBytes);

            Harness.FillDeterministic(m1.Floats);
            dst.Floats.Clear();
            var m1View = new GgmlTensorView2D(m1.Ptr, Batch, In, In, 1, m1Bytes);
            var dstView = new GgmlTensorView2D(dst.Ptr, Batch, Out, Out, 1, dstBytes);
            double ns = Harness.TimeCase(() => GgmlNative.AddmmQuantBatch(dstView, m1View, blob.Ptr, GGML_TYPE_Q8_0, In, blobBytes, Batch, offsets, ne1Arr));
            dump.Add("matmul/addmmquantbatch-q8_0", dst.RawBytes, ns);
        }

        // Q4_0 variant.
        {
            const int Batch = 2, In = 32, Out = 32;
            using var blob = BuildQuantWeight(GGML_TYPE_Q4_0, In, Batch * Out, out long blobBytes);
            long perWeightBytes = (long)ggml_row_size(GGML_TYPE_Q4_0, In) * Out;
            long[] offsets = { 0, perWeightBytes };
            long[] ne1Arr = { Out, Out };

            long m1Bytes = (long)Batch * In * sizeof(float);
            long dstBytes = (long)Batch * Out * sizeof(float);
            using var m1 = new Harness.AlignedBuffer(m1Bytes);
            using var dst = new Harness.AlignedBuffer(dstBytes);

            Harness.FillDeterministic(m1.Floats);
            dst.Floats.Clear();
            var m1View = new GgmlTensorView2D(m1.Ptr, Batch, In, In, 1, m1Bytes);
            var dstView = new GgmlTensorView2D(dst.Ptr, Batch, Out, Out, 1, dstBytes);
            double ns = Harness.TimeCase(() => GgmlNative.AddmmQuantBatch(dstView, m1View, blob.Ptr, GGML_TYPE_Q4_0, In, blobBytes, Batch, offsets, ne1Arr));
            dump.Add("matmul/addmmquantbatch-q4_0", dst.RawBytes, ns);
        }
    }

    // ------------------------------------------------------------------
    // AddmmBatch: result[b] = beta * src[b] + alpha * (m1[b] x m2[b])
    // ------------------------------------------------------------------

    public static void RunAddmmBatchCases(Harness.Dump dump)
    {
        const int Batches = 2, Rows = 5, Cols = 6, Shared = 8;
        long m1Bytes = (long)Batches * Rows * Shared * sizeof(float);
        long m2Bytes = (long)Batches * Shared * Cols * sizeof(float);
        long srcBytes = (long)Batches * Rows * Cols * sizeof(float);
        long dstBytes = srcBytes;

        using var m1 = new Harness.AlignedBuffer(m1Bytes);
        using var m2 = new Harness.AlignedBuffer(m2Bytes);
        using var src = new Harness.AlignedBuffer(srcBytes);
        using var dst = new Harness.AlignedBuffer(dstBytes);

        var m1View = new GgmlTensorView3D(m1.Ptr, Batches, Rows, Shared, Rows * Shared, Shared, 1, m1Bytes);
        // Direct (per-batch transposed) m2: element (b, s, c) at
        // b * Shared * Cols + c * Shared + s.
        var m2View = new GgmlTensorView3D(m2.Ptr, Batches, Shared, Cols, Shared * Cols, 1, Shared, m2Bytes);
        var srcView = new GgmlTensorView3D(src.Ptr, Batches, Rows, Cols, Rows * Cols, Cols, 1, srcBytes);
        var dstView = new GgmlTensorView3D(dst.Ptr, Batches, Rows, Cols, Rows * Cols, Cols, 1, dstBytes);

        // No bias.
        {
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2.Floats);
            dst.Floats.Clear();
            double ns = Harness.TimeCase(() => GgmlNative.AddmmBatch(dstView, default, m1View, m2View, 0f, 1f));
            dump.Add("matmul/addmmbatch-nobias", dst.RawBytes, ns);
        }

        // Full-shape bias with beta == 1.
        {
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2.Floats);
            Harness.FillDeterministic(src.Floats);
            dst.Floats.Clear();
            double ns = Harness.TimeCase(() => GgmlNative.AddmmBatch(dstView, srcView, m1View, m2View, 1f, 1f));
            dump.Add("matmul/addmmbatch-bias", dst.RawBytes, ns);
        }

        // Row-vector bias broadcast over batches and rows.
        {
            long rowBytes = (long)Cols * sizeof(float);
            using var srcRow = new Harness.AlignedBuffer(rowBytes);
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2.Floats);
            Harness.FillDeterministic(srcRow.Floats);
            dst.Floats.Clear();
            var srcRowView = new GgmlTensorView3D(srcRow.Ptr, 1, 1, Cols, Cols, Cols, 1, rowBytes);
            double ns = Harness.TimeCase(() => GgmlNative.AddmmBatch(dstView, srcRowView, m1View, m2View, 1f, 1f));
            dump.Add("matmul/addmmbatch-bias-broadcast", dst.RawBytes, ns);
        }

        // alpha != 1 and beta != 1 exercise both ggml_scale nodes.
        {
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2.Floats);
            Harness.FillDeterministic(src.Floats);
            dst.Floats.Clear();
            double ns = Harness.TimeCase(() => GgmlNative.AddmmBatch(dstView, srcView, m1View, m2View, 2f, 0.5f));
            dump.Add("matmul/addmmbatch-alphabeta", dst.RawBytes, ns);
        }

        // Row-major m2 (stride1 != 1) forces the packed-m2 3D path.
        {
            using var m2RowMajor = new Harness.AlignedBuffer(m2Bytes);
            Harness.FillDeterministic(m1.Floats);
            Harness.FillDeterministic(m2RowMajor.Floats);
            dst.Floats.Clear();
            var m2RowMajorView = new GgmlTensorView3D(m2RowMajor.Ptr, Batches, Shared, Cols, Shared * Cols, Cols, 1, m2Bytes);
            double ns = Harness.TimeCase(() => GgmlNative.AddmmBatch(dstView, default, m1View, m2RowMajorView, 0f, 1f));
            dump.Add("matmul/addmmbatch-m2-rowmajor", dst.RawBytes, ns);
        }

        // Per-batch transposed m1 (stride2 != 1) forces the packed-standard
        // 3D path.
        {
            using var m1T = new Harness.AlignedBuffer(m1Bytes);
            Harness.FillDeterministic(m1T.Floats);
            Harness.FillDeterministic(m2.Floats);
            dst.Floats.Clear();
            var m1TView = new GgmlTensorView3D(m1T.Ptr, Batches, Rows, Shared, Rows * Shared, 1, Rows, m1Bytes);
            double ns = Harness.TimeCase(() => GgmlNative.AddmmBatch(dstView, default, m1TView, m2View, 0f, 1f));
            dump.Add("matmul/addmmbatch-m1-transposed", dst.RawBytes, ns);
        }
    }

    // ------------------------------------------------------------------
    // MulMatId (MoE expert-routed matmul)
    // ------------------------------------------------------------------

    public static void RunMulMatIdCases(Harness.Dump dump)
    {
        const int Experts = 4, OutRows = 16, Inner = 32, Tokens = 5, Used = 2;
        long expertBytes = (long)Experts * OutRows * Inner * sizeof(float);
        long inputBytes = (long)Tokens * Used * Inner * sizeof(float);
        long dstBytes = (long)Tokens * Used * OutRows * sizeof(float);

        using var experts = new Harness.AlignedBuffer(expertBytes);
        using var input = new Harness.AlignedBuffer(inputBytes);
        using var dst = new Harness.AlignedBuffer(dstBytes);
        using var ids = new Harness.AlignedBuffer(Tokens * Used * sizeof(int));

        Span<int> idValues = new Span<int>((void*)ids.Ptr, Tokens * Used);
        int[] routing = { 0, 3, 1, 2, 2, 0, 1, 3, 0, 2 };
        routing.CopyTo(idValues);

        var expertView = new GgmlTensorView3D(experts.Ptr, Experts, OutRows, Inner, OutRows * Inner, Inner, 1, expertBytes);
        var dstView = new GgmlTensorView3D(dst.Ptr, Tokens, Used, OutRows, Used * OutRows, OutRows, 1, dstBytes);
        var idsDesc = new GgmlContiguousTensor(ids.Ptr, Tokens * Used, TensorSharp.DType.Int32);

        // Per-slot input rows (input dim1 == idsCols).
        {
            Harness.FillDeterministic(experts.Floats);
            Harness.FillDeterministic(input.Floats);
            dst.Floats.Clear();
            var inputView = new GgmlTensorView3D(input.Ptr, Tokens, Used, Inner, Used * Inner, Inner, 1, inputBytes);
            double ns = Harness.TimeCase(() => GgmlNative.MulMatId(dstView, expertView, inputView, idsDesc, Tokens, Used));
            dump.Add("matmul/mulmatid-basic", dst.RawBytes, ns);
        }

        // Shared input row broadcast over the expert slots (input dim1 == 1,
        // idsCols % 1 == 0).
        {
            long sharedInputBytes = (long)Tokens * Inner * sizeof(float);
            using var sharedInput = new Harness.AlignedBuffer(sharedInputBytes);
            Harness.FillDeterministic(experts.Floats);
            Harness.FillDeterministic(sharedInput.Floats);
            dst.Floats.Clear();
            var sharedInputView = new GgmlTensorView3D(sharedInput.Ptr, Tokens, 1, Inner, Inner, Inner, 1, sharedInputBytes);
            double ns = Harness.TimeCase(() => GgmlNative.MulMatId(dstView, expertView, sharedInputView, idsDesc, Tokens, Used));
            dump.Add("matmul/mulmatid-broadcast", dst.RawBytes, ns);
        }
    }

    // ------------------------------------------------------------------
    // AddId (MoE expert-routed bias add)
    // ------------------------------------------------------------------

    public static void RunAddIdCases(Harness.Dump dump)
    {
        const int Experts = 4, Embd = 32, Tokens = 5, Used = 2;
        long srcBytes = (long)Tokens * Used * Embd * sizeof(float);
        long biasBytes = (long)Experts * Embd * sizeof(float);
        long dstBytes = srcBytes;

        using var src = new Harness.AlignedBuffer(srcBytes);
        using var bias = new Harness.AlignedBuffer(biasBytes);
        using var dst = new Harness.AlignedBuffer(dstBytes);

        var srcView = new GgmlTensorView3D(src.Ptr, Tokens, Used, Embd, Used * Embd, Embd, 1, srcBytes);
        var biasView = new GgmlTensorView2D(bias.Ptr, Experts, Embd, Embd, 1, biasBytes);
        var dstView = new GgmlTensorView3D(dst.Ptr, Tokens, Used, Embd, Used * Embd, Embd, 1, dstBytes);

        // I32 ids.
        {
            using var ids = new Harness.AlignedBuffer(Tokens * Used * sizeof(int));
            Span<int> idValues = new Span<int>((void*)ids.Ptr, Tokens * Used);
            int[] routing = { 1, 0, 3, 2, 0, 0, 2, 1, 3, 3 };
            routing.CopyTo(idValues);

            Harness.FillDeterministic(src.Floats);
            Harness.FillDeterministic(bias.Floats);
            dst.Floats.Clear();
            var idsDesc = new GgmlContiguousTensor(ids.Ptr, Tokens * Used, TensorSharp.DType.Int32);
            double ns = Harness.TimeCase(() => GgmlNative.AddId(dstView, srcView, biasView, idsDesc, Tokens, Used));
            dump.Add("matmul/addid-basic", dst.RawBytes, ns);
        }

        // F32 id tensor (values converted to int inside the op).
        {
            using var ids = new Harness.AlignedBuffer(Tokens * Used * sizeof(float));
            Span<float> idValues = ids.Floats;
            int[] routing = { 2, 2, 0, 1, 3, 0, 1, 3, 2, 0 };
            for (int i = 0; i < routing.Length; ++i)
                idValues[i] = routing[i];

            Harness.FillDeterministic(src.Floats);
            Harness.FillDeterministic(bias.Floats);
            dst.Floats.Clear();
            var idsDesc = new GgmlContiguousTensor(ids.Ptr, Tokens * Used, TensorSharp.DType.Float32);
            double ns = Harness.TimeCase(() => GgmlNative.AddId(dstView, srcView, biasView, idsDesc, Tokens, Used));
            dump.Add("matmul/addid-f32ids", dst.RawBytes, ns);
        }
    }
}
