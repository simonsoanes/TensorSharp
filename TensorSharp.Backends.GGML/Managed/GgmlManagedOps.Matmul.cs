// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Runtime.InteropServices;
using TensorSharp.GGML.Interop;
using static TensorSharp.GGML.Interop.GgmlApi;
using Runtime = TensorSharp.GGML.Interop.GgmlManagedRuntime;

namespace TensorSharp.GGML
{
    // Managed port of TensorSharp.GGML.Native/ggml_ops_matmul.cpp. Each method
    // mirrors the corresponding *_impl: validate the descriptors, bind the
    // caller's host buffers (zero-copy where the backend allows it, direct m2
    // views for transposed weights, packed staging otherwise), build a small
    // ggml graph, compute on the shared backend, and download the result when
    // it was not host-mapped. Quantized weights bind through the managed
    // weight cache (GgmlManagedWeightCache) exactly where the native impls use
    // try_get_cacheable_tensor_buffer. Returns 1/0 with a thread-local
    // last-error exactly like the native entry points so the GgmlNative
    // wrappers can route to either implementation.
    internal static unsafe partial class GgmlManagedOps
    {
        private const string DllName = "GgmlOps";

        // ggml_add_id is not bound in GgmlApi yet; declare it locally (same
        // module, statically linked ggml re-exported by GgmlOps).
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern GgmlTensor* ggml_add_id(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, GgmlTensor* ids);

        // ------------------------------------------------------------------
        // Helpers ported from ggml_ops_core.cpp (matmul-specific: direct/packed
        // m2 bindings, packed standard bindings and the CUDA row-chunking
        // arithmetic). Kept private here per the module-port pattern.
        // ------------------------------------------------------------------

        // Mirrors tsg::k_ggml_cuda_max_copy_bytes (size_t)INT_MAX.
        private const ulong KCudaMaxCopyBytes = int.MaxValue;

        private static bool IsNonOverlappingFastToSlow(ReadOnlySpan<long> sizes, ReadOnlySpan<long> strides)
        {
            long requiredStride = 1;
            for (int i = 0; i < sizes.Length; ++i)
            {
                if (sizes[i] <= 0 || strides[i] < 0)
                    return false;
                if (sizes[i] == 1)
                    continue;
                if (strides[i] < requiredStride)
                    return false;
                requiredStride = strides[i] * sizes[i];
            }
            return true;
        }

        // Mirrors can_map_m2_direct (2D): the weight is stored transposed
        // (stride0 == 1) so ggml can view it directly as [dim0, dim1].
        private static bool CanMapM2Direct(in GgmlTensorView2D d)
        {
            return d.Stride0 == 1 &&
                d.Stride1 >= d.Dim0 &&
                IsNonOverlappingFastToSlow(
                    stackalloc long[] { d.Dim0, d.Dim1 },
                    stackalloc long[] { d.Stride0, d.Stride1 });
        }

        // Mirrors can_map_m2_direct (3D).
        private static bool CanMapM2Direct(in GgmlTensorView3D d)
        {
            return d.Stride1 == 1 &&
                d.Stride2 >= d.Dim1 &&
                IsNonOverlappingFastToSlow(
                    stackalloc long[] { d.Dim1, d.Dim2, d.Dim0 },
                    stackalloc long[] { d.Stride1, d.Stride2, d.Stride0 });
        }

        // Mirrors logical_row_bytes (2D).
        private static ulong LogicalRowBytes(in GgmlTensorView2D d) =>
            (ulong)d.Dim1 * sizeof(float);

        // Mirrors raw_row_bytes (2D): required_raw_bytes of a single-row slice.
        private static ulong RawRowBytes(in GgmlTensorView2D d) =>
            (ulong)(((d.Dim1 - 1L) * d.Stride1) + 1) * sizeof(float);

        // Mirrors slice_rows_2d.
        private static GgmlTensorView2D SliceRows2D(in GgmlTensorView2D desc, int rowStart, int rowCount)
        {
            IntPtr data = desc.Data + (nint)((long)rowStart * desc.Stride0 * sizeof(float));
            long rawBytes = (((rowCount - 1L) * desc.Stride0) + ((desc.Dim1 - 1L) * desc.Stride1) + 1) * sizeof(float);
            return new GgmlTensorView2D(data, rowCount, desc.Dim1, desc.Stride0, desc.Stride1, rawBytes);
        }

        // Mirrors limit_rows_for_cuda_copy.
        private static int LimitRowsForCudaCopy(int currentLimit, in GgmlTensorView2D desc)
        {
            if (currentLimit <= 0)
                return 0;
            ulong perRowBytes = Math.Max(LogicalRowBytes(desc), RawRowBytes(desc));
            if (perRowBytes == 0 || perRowBytes > KCudaMaxCopyBytes)
                return 0;
            int limit = (int)(KCudaMaxCopyBytes / perRowBytes);
            return Math.Min(currentLimit, Math.Max(1, limit));
        }

        // Mirrors create_direct_m2_binding (2D).
        private static Runtime.TensorBinding CreateDirectM2Binding(IntPtr ctx, in GgmlTensorView2D d)
        {
            GgmlTensor* baseTensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.RawBytes / sizeof(float));
            if (baseTensor == null)
                return default;
            GgmlTensor* view = ggml_view_2d(ctx, baseTensor, d.Dim0, d.Dim1,
                (nuint)((long)d.Stride1 * sizeof(float)), 0);
            if (view == null)
                return default;
            return new Runtime.TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = (nuint)d.RawBytes };
        }

        // Mirrors create_direct_m2_binding (3D).
        private static Runtime.TensorBinding CreateDirectM2Binding(IntPtr ctx, in GgmlTensorView3D d)
        {
            GgmlTensor* baseTensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.RawBytes / sizeof(float));
            if (baseTensor == null)
                return default;
            GgmlTensor* view = ggml_view_3d(ctx, baseTensor, d.Dim1, d.Dim2, d.Dim0,
                (nuint)((long)d.Stride2 * sizeof(float)),
                (nuint)((long)d.Stride0 * sizeof(float)), 0);
            if (view == null)
                return default;
            return new Runtime.TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = (nuint)d.RawBytes };
        }

        // Mirrors pack_m2 (2D): gather into transposed [dim0, dim1] ggml layout.
        private static float[] PackM2(in GgmlTensorView2D d)
        {
            float* data = (float*)d.Data;
            var packed = new float[(long)d.Dim0 * d.Dim1];
            for (int row = 0; row < d.Dim0; ++row)
                for (int col = 0; col < d.Dim1; ++col)
                    packed[((long)col * d.Dim0) + row] =
                        data[((long)row * d.Stride0) + ((long)col * d.Stride1)];
            return packed;
        }

        // Mirrors pack_m2 (3D).
        private static float[] PackM2(in GgmlTensorView3D d)
        {
            float* data = (float*)d.Data;
            var packed = new float[(long)d.Dim0 * d.Dim1 * d.Dim2];
            for (int batch = 0; batch < d.Dim0; ++batch)
                for (int row = 0; row < d.Dim1; ++row)
                    for (int col = 0; col < d.Dim2; ++col)
                        packed[(((long)batch * d.Dim2 + col) * d.Dim1) + row] =
                            data[((long)batch * d.Stride0) +
                                 ((long)row * d.Stride1) +
                                 ((long)col * d.Stride2)];
            return packed;
        }

        // Mirrors pack_standard (2D): gather into contiguous row-major layout.
        private static float[] PackStandard(in GgmlTensorView2D d)
        {
            float* data = (float*)d.Data;
            var packed = new float[(long)d.Dim0 * d.Dim1];
            for (int row = 0; row < d.Dim0; ++row)
                for (int col = 0; col < d.Dim1; ++col)
                    packed[((long)row * d.Dim1) + col] =
                        data[((long)row * d.Stride0) + ((long)col * d.Stride1)];
            return packed;
        }

        // Mirrors pack_standard (3D).
        private static float[] PackStandard(in GgmlTensorView3D d)
        {
            float* data = (float*)d.Data;
            var packed = new float[(long)d.Dim0 * d.Dim1 * d.Dim2];
            for (int batch = 0; batch < d.Dim0; ++batch)
                for (int row = 0; row < d.Dim1; ++row)
                    for (int col = 0; col < d.Dim2; ++col)
                        packed[(((long)batch * d.Dim1 + row) * d.Dim2) + col] =
                            data[((long)batch * d.Stride0) +
                                 ((long)row * d.Stride1) +
                                 ((long)col * d.Stride2)];
            return packed;
        }

        // Mirrors create_packed_m2_binding (2D).
        private static Runtime.TensorBinding CreatePackedM2Binding(IntPtr ctx, in GgmlTensorView2D d, out float[] packed)
        {
            packed = PackM2(d);
            GgmlTensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.Dim0, d.Dim1);
            return new Runtime.TensorBinding { Storage = tensor, Tensor = tensor, RawBytes = (nuint)packed.LongLength * sizeof(float) };
        }

        // Mirrors create_packed_m2_binding (3D).
        private static Runtime.TensorBinding CreatePackedM2Binding(IntPtr ctx, in GgmlTensorView3D d, out float[] packed)
        {
            packed = PackM2(d);
            GgmlTensor* tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.Dim1, d.Dim2, d.Dim0);
            return new Runtime.TensorBinding { Storage = tensor, Tensor = tensor, RawBytes = (nuint)packed.LongLength * sizeof(float) };
        }

        // Mirrors create_packed_standard_binding (2D).
        private static Runtime.TensorBinding CreatePackedStandardBinding(IntPtr ctx, in GgmlTensorView2D d, out float[] packed)
        {
            packed = PackStandard(d);
            GgmlTensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.Dim1, d.Dim0);
            return new Runtime.TensorBinding { Storage = tensor, Tensor = tensor, RawBytes = (nuint)packed.LongLength * sizeof(float) };
        }

        // Mirrors create_packed_standard_binding (3D).
        private static Runtime.TensorBinding CreatePackedStandardBinding(IntPtr ctx, in GgmlTensorView3D d, out float[] packed)
        {
            packed = PackStandard(d);
            GgmlTensor* tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.Dim2, d.Dim1, d.Dim0);
            return new Runtime.TensorBinding { Storage = tensor, Tensor = tensor, RawBytes = (nuint)packed.LongLength * sizeof(float) };
        }

        // Mirrors create_binding_from_host_ptr_direct_m2_2d.
        private static bool CreateBindingFromHostPtrDirectM2(IntPtr ctx, in GgmlTensorView2D d, out Runtime.TensorBinding binding, out IntPtr buffer)
        {
            binding = default;
            buffer = IntPtr.Zero;

            nuint rawBytes = (nuint)d.RawBytes;
            IntPtr dev = ggml_backend_get_device(Runtime.Backend);
            if (!Runtime.CanUseHostPtrBuffer(dev, d.Data, rawBytes))
                return false;
            buffer = ggml_backend_dev_buffer_from_host_ptr(dev, d.Data, rawBytes, rawBytes);
            if (buffer == IntPtr.Zero)
                return false;

            GgmlTensor* baseTensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (long)(rawBytes / sizeof(float)));
            if (baseTensor == null)
            {
                ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }

            if (ggml_backend_tensor_alloc(buffer, baseTensor, d.Data) != GGML_STATUS_SUCCESS)
            {
                ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }

            GgmlTensor* view = ggml_view_2d(ctx, baseTensor, d.Dim0, d.Dim1,
                (nuint)((long)d.Stride1 * sizeof(float)), 0);
            if (view == null)
            {
                ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }

            binding = new Runtime.TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = rawBytes };
            return true;
        }

        // Mirrors create_binding_from_host_ptr_direct_m2_3d.
        private static bool CreateBindingFromHostPtrDirectM2(IntPtr ctx, in GgmlTensorView3D d, out Runtime.TensorBinding binding, out IntPtr buffer)
        {
            binding = default;
            buffer = IntPtr.Zero;

            nuint rawBytes = (nuint)d.RawBytes;
            IntPtr dev = ggml_backend_get_device(Runtime.Backend);
            if (!Runtime.CanUseHostPtrBuffer(dev, d.Data, rawBytes))
                return false;
            buffer = ggml_backend_dev_buffer_from_host_ptr(dev, d.Data, rawBytes, rawBytes);
            if (buffer == IntPtr.Zero)
                return false;

            GgmlTensor* baseTensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (long)(rawBytes / sizeof(float)));
            if (baseTensor == null)
            {
                ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }

            if (ggml_backend_tensor_alloc(buffer, baseTensor, d.Data) != GGML_STATUS_SUCCESS)
            {
                ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }

            GgmlTensor* view = ggml_view_3d(ctx, baseTensor, d.Dim1, d.Dim2, d.Dim0,
                (nuint)((long)d.Stride2 * sizeof(float)),
                (nuint)((long)d.Stride0 * sizeof(float)), 0);
            if (view == null)
            {
                ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }

            binding = new Runtime.TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = rawBytes };
            return true;
        }

        // Uploads either the caller's raw view bytes or the packed staging copy
        // (mirrors the packed_x.empty() ? desc.data : packed.data() pattern).
        private static void UploadPackedOrRaw(in Runtime.TensorBinding binding, IntPtr rawData, float[] packed)
        {
            if (packed == null)
            {
                Runtime.UploadBinding(binding, rawData, binding.RawBytes);
            }
            else
            {
                fixed (float* packedPtr = packed)
                {
                    Runtime.UploadBinding(binding, (IntPtr)packedPtr, binding.RawBytes);
                }
            }
        }

        // --------------------------------------------------------------
        // Addmm (result = beta * src + alpha * (m1 x m2))
        // --------------------------------------------------------------

        internal static int AddmmF32(
            in GgmlTensorView2D resultDesc,
            in GgmlTensorView2D srcDesc,
            in GgmlTensorView2D m1Desc,
            in GgmlTensorView2D m2Desc,
            float beta,
            float alpha)
        {
            if (!Runtime.EnsureBackend())
                return 0;

            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(m1Desc, "m1") || !Runtime.ValidateDesc(m2Desc, "m2"))
                return 0;

            if (beta != 0.0f && !Runtime.ValidateDesc(srcDesc, "src"))
                return 0;

            int rows = resultDesc.Dim0;
            int cols = resultDesc.Dim1;
            int shared = m1Desc.Dim1;

            if (m1Desc.Dim0 != rows || m2Desc.Dim0 != shared || m2Desc.Dim1 != cols)
            {
                Runtime.SetLastError("Size mismatch passed to ggml addmm.");
                return 0;
            }

            if (beta != 0.0f && ((rows % srcDesc.Dim0) != 0 || (cols % srcDesc.Dim1) != 0))
            {
                Runtime.SetLastError("Source tensor shape cannot be broadcast to result shape for ggml addmm.");
                return 0;
            }

            if (Runtime.BackendType == Runtime.BackendTypeCuda)
            {
                bool needsChunking =
                    (ulong)Runtime.LogicalBytes(resultDesc) > KCudaMaxCopyBytes ||
                    (ulong)Runtime.LogicalBytes(m1Desc) > KCudaMaxCopyBytes ||
                    (ulong)resultDesc.RawBytes > KCudaMaxCopyBytes ||
                    (ulong)m1Desc.RawBytes > KCudaMaxCopyBytes ||
                    (beta != 0.0f && (
                        (ulong)Runtime.LogicalBytes(srcDesc) > KCudaMaxCopyBytes ||
                        (ulong)srcDesc.RawBytes > KCudaMaxCopyBytes));

                if (needsChunking)
                {
                    int chunkRows = rows;
                    chunkRows = LimitRowsForCudaCopy(chunkRows, resultDesc);
                    chunkRows = LimitRowsForCudaCopy(chunkRows, m1Desc);
                    if (beta != 0.0f)
                    {
                        chunkRows = LimitRowsForCudaCopy(chunkRows, srcDesc);
                        if (srcDesc.Dim0 != rows)
                            chunkRows = (chunkRows / srcDesc.Dim0) * srcDesc.Dim0;
                    }

                    if (chunkRows <= 0)
                    {
                        Runtime.SetLastError("GGML CUDA addmm received a row slice larger than the backend copy limit.");
                        return 0;
                    }

                    if (chunkRows < rows)
                    {
                        for (int rowStart = 0; rowStart < rows; rowStart += chunkRows)
                        {
                            int rowCount = Math.Min(chunkRows, rows - rowStart);
                            GgmlTensorView2D resultSlice = SliceRows2D(resultDesc, rowStart, rowCount);
                            GgmlTensorView2D m1Slice = SliceRows2D(m1Desc, rowStart, rowCount);
                            GgmlTensorView2D srcSlice = beta == 0.0f
                                ? default(GgmlTensorView2D)
                                : (srcDesc.Dim0 == rows ? SliceRows2D(srcDesc, rowStart, rowCount) : srcDesc);

                            if (AddmmF32(resultSlice, srcSlice, m1Slice, m2Desc, beta, alpha) == 0)
                                return 0;
                        }

                        Runtime.ClearLastError();
                        return 1;
                    }
                }
            }

            if (!Runtime.CanMapStandardView(resultDesc))
            {
                Runtime.SetLastError("Result tensor layout is not supported by the ggml addmm Metal path.");
                return 0;
            }

            if (beta != 0.0f && !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Source tensor layout is not supported by the ggml addmm Metal path.");
                return 0;
            }

            bool useZeroCopy = Runtime.CanMapStandardView(m1Desc) && CanMapM2Direct(m2Desc);

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = default;
                Runtime.TensorBinding m1Binding = default;
                Runtime.TensorBinding srcBinding = default;
                float[] packedM1 = null;
                float[] packedM2 = null;

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out IntPtr resultBuf))
                        buffers.Add(resultBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, m1Desc, out m1Binding, out IntPtr m1Buf))
                        buffers.Add(m1Buf);
                    else
                        useZeroCopy = false;
                }
                else
                {
                    m1Binding = Runtime.CanMapStandardView(m1Desc)
                        ? Runtime.CreateStandardBinding(context.Ctx, m1Desc)
                        : CreatePackedStandardBinding(context.Ctx, m1Desc, out packedM1);
                }

                if (useZeroCopy && beta != 0.0f)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, srcDesc, out srcBinding, out IntPtr srcBuf))
                        buffers.Add(srcBuf);
                    else
                        useZeroCopy = false;
                }
                else if (beta != 0.0f)
                {
                    srcBinding = Runtime.CreateStandardBinding(context.Ctx, srcDesc);
                }

                Runtime.TensorBinding m2Binding = default;
                bool m2ZeroCopy = false;
                if (useZeroCopy && CanMapM2Direct(m2Desc))
                {
                    if (CreateBindingFromHostPtrDirectM2(context.Ctx, m2Desc, out m2Binding, out IntPtr m2Buf))
                    {
                        m2ZeroCopy = true;
                        buffers.Add(m2Buf);
                    }
                }
                if (!m2ZeroCopy)
                {
                    m2Binding = CanMapM2Direct(m2Desc)
                        ? CreateDirectM2Binding(context.Ctx, m2Desc)
                        : CreatePackedM2Binding(context.Ctx, m2Desc, out packedM2);
                }

                if (!resultBinding.IsValid ||
                    !m1Binding.IsValid ||
                    !m2Binding.IsValid ||
                    (beta != 0.0f && !srcBinding.IsValid))
                {
                    Runtime.SetLastError("Failed to allocate ggml tensor views.");
                    return 0;
                }

                GgmlTensor* mmTensor = ggml_mul_mat(context.Ctx, m2Binding.Tensor, m1Binding.Tensor);
                if (mmTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml matmul node.");
                    return 0;
                }

                GgmlTensor* combinedTensor = mmTensor;
                if (alpha != 1.0f)
                {
                    combinedTensor = ggml_scale(context.Ctx, combinedTensor, alpha);
                    if (combinedTensor == null)
                    {
                        Runtime.SetLastError("Failed to scale ggml matmul output.");
                        return 0;
                    }
                }

                if (beta != 0.0f)
                {
                    GgmlTensor* scaledSrc = srcBinding.Tensor;
                    if (beta != 1.0f)
                    {
                        scaledSrc = ggml_scale(context.Ctx, srcBinding.Tensor, beta);
                        if (scaledSrc == null)
                        {
                            Runtime.SetLastError("Failed to scale ggml source tensor.");
                            return 0;
                        }
                    }

                    combinedTensor = ggml_add(context.Ctx, combinedTensor, scaledSrc);
                    if (combinedTensor == null)
                    {
                        Runtime.SetLastError("Failed to create ggml add node.");
                        return 0;
                    }
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, combinedTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml output copy node.");
                    return 0;
                }

                ggml_set_output(outputTensor);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }

                ggml_build_forward_expand(graph, outputTensor);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!useZeroCopy)
                {
                    if (beta != 0.0f)
                        Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);
                    UploadPackedOrRaw(m1Binding, m1Desc.Data, packedM1);
                    if (resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                }
                if (!m2ZeroCopy)
                {
                    UploadPackedOrRaw(m2Binding, m2Desc.Data, packedM2);
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                Runtime.FinalizeCompute(useZeroCopy, resultBinding, resultDesc.Data, resultBinding.RawBytes);
                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // AddmmQuant (result = m1 x dequant(m2))
        // --------------------------------------------------------------

        internal static int AddmmQuantF32(
            in GgmlTensorView2D resultDesc,
            in GgmlTensorView2D m1Desc,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2Ne1,
            long m2RawBytes)
        {
            var m2Quant = new GgmlQuantizedWeight(m2Data, m2GgmlType, m2Ne0, m2Ne1, m2RawBytes);
            return AddmmQuantImpl(resultDesc, m1Desc, m2Quant);
        }

        private static int AddmmQuantImpl(
            in GgmlTensorView2D resultDesc,
            in GgmlTensorView2D m1Desc,
            in GgmlQuantizedWeight m2Quant)
        {
            if (!Runtime.EnsureBackend())
                return 0;

            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(m1Desc, "m1"))
                return 0;

            if (m2Quant.Data == IntPtr.Zero || m2Quant.Ne0 <= 0 || m2Quant.Ne1 <= 0 || m2Quant.RawBytes <= 0)
            {
                Runtime.SetLastError("Invalid quantized weight descriptor.");
                return 0;
            }

            int rows = resultDesc.Dim0;   // seqLen
            int cols = resultDesc.Dim1;   // outDim
            int shared = m1Desc.Dim1;     // inDim

            if (m1Desc.Dim0 != rows)
            {
                Runtime.SetLastError("Size mismatch: m1.dim0 != result.dim0 in addmm_quant.");
                return 0;
            }

            // m2_quant: ne0 = inDim (shared), ne1 = outDim
            if (m2Quant.Ne0 != shared || m2Quant.Ne1 != cols)
            {
                Runtime.SetLastError("Size mismatch: quantized weight dims don't match in addmm_quant.");
                return 0;
            }

            if (Runtime.BackendType == Runtime.BackendTypeCuda)
            {
                bool needsChunking =
                    (ulong)Runtime.LogicalBytes(resultDesc) > KCudaMaxCopyBytes ||
                    (ulong)Runtime.LogicalBytes(m1Desc) > KCudaMaxCopyBytes ||
                    (ulong)resultDesc.RawBytes > KCudaMaxCopyBytes ||
                    (ulong)m1Desc.RawBytes > KCudaMaxCopyBytes;

                if (needsChunking)
                {
                    int chunkRows = rows;
                    chunkRows = LimitRowsForCudaCopy(chunkRows, resultDesc);
                    chunkRows = LimitRowsForCudaCopy(chunkRows, m1Desc);

                    if (chunkRows <= 0)
                    {
                        Runtime.SetLastError("GGML CUDA addmm_quant received a row slice larger than the backend copy limit.");
                        return 0;
                    }

                    if (chunkRows < rows)
                    {
                        for (int rowStart = 0; rowStart < rows; rowStart += chunkRows)
                        {
                            int rowCount = Math.Min(chunkRows, rows - rowStart);
                            GgmlTensorView2D resultSlice = SliceRows2D(resultDesc, rowStart, rowCount);
                            GgmlTensorView2D m1Slice = SliceRows2D(m1Desc, rowStart, rowCount);

                            if (AddmmQuantImpl(resultSlice, m1Slice, m2Quant) == 0)
                                return 0;
                        }

                        Runtime.ClearLastError();
                        return 1;
                    }
                }
            }

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                bool useZeroCopy = Runtime.CanMapStandardView(m1Desc);

                // Result/input bindings. If zero-copy host binding fails for
                // either tensor, fall back to standard ggml-owned buffers for
                // both.
                Runtime.TensorBinding resultBinding = default;
                Runtime.TensorBinding m1Binding = default;
                float[] packedM1 = null;
                if (useZeroCopy)
                {
                    IntPtr resultBuf = IntPtr.Zero;
                    IntPtr m1Buf = IntPtr.Zero;
                    bool resultOk = Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out resultBuf);
                    bool m1Ok = resultOk && Runtime.CreateBindingFromHostPtr(context.Ctx, m1Desc, out m1Binding, out m1Buf);

                    if (resultOk && m1Ok)
                    {
                        buffers.Add(resultBuf);
                        buffers.Add(m1Buf);
                    }
                    else
                    {
                        if (m1Buf != IntPtr.Zero)
                            ggml_backend_buffer_free(m1Buf);
                        if (resultBuf != IntPtr.Zero)
                            ggml_backend_buffer_free(resultBuf);

                        useZeroCopy = false;
                        resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);
                        m1Binding = Runtime.CanMapStandardView(m1Desc)
                            ? Runtime.CreateStandardBinding(context.Ctx, m1Desc)
                            : CreatePackedStandardBinding(context.Ctx, m1Desc, out packedM1);
                    }
                }
                else
                {
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);
                    m1Binding = Runtime.CanMapStandardView(m1Desc)
                        ? Runtime.CreateStandardBinding(context.Ctx, m1Desc)
                        : CreatePackedStandardBinding(context.Ctx, m1Desc, out packedM1);
                }

                // m2 (quantized weight) binding: create ggml tensor with actual quantized type
                GgmlTensor* m2Tensor = ggml_new_tensor_2d(context.Ctx, m2Quant.GgmlType, m2Quant.Ne0, m2Quant.Ne1);
                var m2Binding = new Runtime.TensorBinding { Storage = m2Tensor, Tensor = m2Tensor, RawBytes = (nuint)m2Quant.RawBytes };

                if (!resultBinding.IsValid || !m1Binding.IsValid || m2Tensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensor views for addmm_quant.");
                    return 0;
                }

                // Try cached host_ptr binding for quantized weight (stable pointer across calls)
                bool m2Bound = false;
                bool m2NeedsUpload = false;
                {
                    IntPtr dev = ggml_backend_get_device(Runtime.Backend);
                    if (dev != IntPtr.Zero && m2Quant.RawBytes >= 4096)
                    {
                        if (GgmlManagedWeightCache.TryGetCacheableTensorBuffer(
                                m2Tensor,
                                m2Quant.Data,
                                (nuint)m2Quant.RawBytes,
                                out IntPtr cacheBuf,
                                out IntPtr cacheAddr,
                                out m2NeedsUpload))
                        {
                            m2Bound = ggml_backend_tensor_alloc(cacheBuf, m2Tensor, cacheAddr) == GGML_STATUS_SUCCESS;
                            if (!m2Bound)
                                GgmlManagedWeightCache.Invalidate(m2Quant.Data);
                        }
                    }
                }

                GgmlTensor* mmTensor = ggml_mul_mat(context.Ctx, m2Binding.Tensor, m1Binding.Tensor);
                if (mmTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml matmul node for addmm_quant.");
                    return 0;
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, mmTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml output copy node for addmm_quant.");
                    return 0;
                }

                ggml_set_output(outputTensor);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph for addmm_quant.");
                    return 0;
                }

                ggml_build_forward_expand(graph, outputTensor);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer for addmm_quant.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                // Upload data
                if (!useZeroCopy)
                {
                    UploadPackedOrRaw(m1Binding, m1Desc.Data, packedM1);
                }

                if (!m2Bound || m2NeedsUpload)
                    Runtime.UploadBinding(m2Binding, m2Quant.Data, m2Binding.RawBytes);

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed for addmm_quant.");
                    return 0;
                }

                Runtime.FinalizeCompute(useZeroCopy, resultBinding, resultDesc.Data, resultBinding.RawBytes);
                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // GetRowsQuant: result[i] = dequant(src[indices[i]])
        // --------------------------------------------------------------

        internal static int GetRowsQuantF32(
            in GgmlTensorView2D resultDesc,
            IntPtr srcData,
            int srcGgmlType,
            long srcNe0,
            long srcNe1,
            long srcRawBytes,
            in GgmlContiguousTensor indicesDesc)
        {
            var srcQuant = new GgmlQuantizedWeight(srcData, srcGgmlType, srcNe0, srcNe1, srcRawBytes);

            if (!Runtime.EnsureBackend())
                return 0;

            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(indicesDesc, "indices"))
                return 0;

            if (srcQuant.Data == IntPtr.Zero || srcQuant.Ne0 <= 0 || srcQuant.Ne1 <= 0 || srcQuant.RawBytes <= 0)
            {
                Runtime.SetLastError("Invalid quantized weight descriptor for get_rows_quant.");
                return 0;
            }

            int numIndices = (int)indicesDesc.ElementCount;
            int embeddingDim = (int)srcQuant.Ne0;

            if (resultDesc.Dim0 != numIndices || resultDesc.Dim1 != embeddingDim)
            {
                Runtime.SetLastError("Shape mismatch in get_rows_quant: result must be [num_indices, ne0].");
                return 0;
            }

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(2 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context for get_rows_quant.");
                return 0;
            }
            using (context)
            {
                bool useZeroCopy = Runtime.CanMapStandardView(resultDesc);

                // Result binding (F32 output)
                Runtime.TensorBinding resultBinding = default;
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out IntPtr resultBuf))
                        buffers.Add(resultBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);

                // Source tensor: quantized type
                GgmlTensor* srcTensor = ggml_new_tensor_2d(context.Ctx, srcQuant.GgmlType, srcQuant.Ne0, srcQuant.Ne1);

                // Index tensor: I32
                GgmlTensor* indexTensor = ggml_new_tensor_1d(context.Ctx, GGML_TYPE_I32, numIndices);

                if (!resultBinding.IsValid || srcTensor == null || indexTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors for get_rows_quant.");
                    return 0;
                }

                var srcBinding = new Runtime.TensorBinding { Storage = srcTensor, Tensor = srcTensor, RawBytes = (nuint)srcQuant.RawBytes };

                // Cache quantized source buffer (same as addmm_quant)
                bool srcBound = false;
                bool srcNeedsUpload = false;
                {
                    IntPtr dev = ggml_backend_get_device(Runtime.Backend);
                    if (dev != IntPtr.Zero && srcQuant.RawBytes >= 4096)
                    {
                        if (GgmlManagedWeightCache.TryGetCacheableTensorBuffer(
                                srcTensor,
                                srcQuant.Data,
                                (nuint)srcQuant.RawBytes,
                                out IntPtr cacheBuf,
                                out IntPtr cacheAddr,
                                out srcNeedsUpload))
                        {
                            srcBound = ggml_backend_tensor_alloc(cacheBuf, srcTensor, cacheAddr) == GGML_STATUS_SUCCESS;
                            if (!srcBound)
                                GgmlManagedWeightCache.Invalidate(srcQuant.Data);
                        }
                    }
                }

                // Build graph: get_rows(src, indices) -> copy -> result
                GgmlTensor* rowsTensor = ggml_get_rows(context.Ctx, srcTensor, indexTensor);
                if (rowsTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml get_rows node for get_rows_quant.");
                    return 0;
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, rowsTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml output copy node for get_rows_quant.");
                    return 0;
                }

                ggml_set_output(outputTensor);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph for get_rows_quant.");
                    return 0;
                }

                ggml_build_forward_expand(graph, outputTensor);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer for get_rows_quant.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                // Upload quantized source if not zero-copy bound
                if (!srcBound || srcNeedsUpload)
                    Runtime.UploadBinding(srcBinding, srcQuant.Data, srcBinding.RawBytes);

                // Upload indices
                int[] indices = new int[numIndices];
                if (!ReadI32Values(indices, indicesDesc))
                    return 0;
                fixed (int* indicesPtr = indices)
                {
                    ggml_backend_tensor_set(indexTensor, (IntPtr)indicesPtr, 0, (nuint)(indices.Length * sizeof(int)));
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed for get_rows_quant.");
                    return 0;
                }

                Runtime.FinalizeCompute(useZeroCopy, resultBinding, resultDesc.Data, resultBinding.RawBytes);
                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // AddmmQuantBatch: result[b] = input[b] * quantWeights[b]^T
        // (each batch uses a separate quantized weight at its byte offset)
        // --------------------------------------------------------------

        internal static int AddmmQuantBatchF32(
            in GgmlTensorView2D resultDesc,
            in GgmlTensorView2D m1Desc,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2RawBytes,
            int batchCount,
            long[] weightOffsets,
            long[] weightNe1Arr)
        {
            if (!Runtime.EnsureBackend())
                return 0;

            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(m1Desc, "m1"))
                return 0;

            if (m2Data == IntPtr.Zero || batchCount <= 0)
            {
                Runtime.SetLastError("Invalid arguments for addmm_quant_batch.");
                return 0;
            }

            // Process each batch sequentially using the existing single-batch impl
            int resultRow = 0;
            int m1Row = 0;

            for (int b = 0; b < batchCount; b++)
            {
                long ne1B = weightNe1Arr[b];
                long offsetB = weightOffsets[b];

                var rDesc = new GgmlTensorView2D(
                    resultDesc.Data + (nint)((long)resultRow * resultDesc.Stride0 * sizeof(float)),
                    1,
                    resultDesc.Dim1,
                    resultDesc.Stride0,
                    resultDesc.Stride1,
                    (long)resultDesc.Dim1 * sizeof(float));

                var inputDesc = new GgmlTensorView2D(
                    m1Desc.Data + (nint)((long)m1Row * m1Desc.Stride0 * sizeof(float)),
                    1,
                    m1Desc.Dim1,
                    m1Desc.Stride0,
                    m1Desc.Stride1,
                    (long)m1Desc.Dim1 * sizeof(float));

                long rowSize = (long)ggml_row_size(m2GgmlType, m2Ne0);
                var wDesc = new GgmlQuantizedWeight(m2Data + (nint)offsetB, m2GgmlType, m2Ne0, ne1B, ne1B * rowSize);

                if (AddmmQuantImpl(rDesc, inputDesc, wDesc) == 0)
                    return 0;

                resultRow++;
                m1Row++;
            }

            Runtime.ClearLastError();
            return 1;
        }

        // --------------------------------------------------------------
        // AddmmBatch (3D batched addmm)
        // --------------------------------------------------------------

        internal static int AddmmBatchF32(
            in GgmlTensorView3D resultDesc,
            in GgmlTensorView3D srcDesc,
            in GgmlTensorView3D m1Desc,
            in GgmlTensorView3D m2Desc,
            float beta,
            float alpha)
        {
            if (!Runtime.EnsureBackend())
                return 0;

            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(m1Desc, "m1") || !Runtime.ValidateDesc(m2Desc, "m2"))
                return 0;

            if (beta != 0.0f && !Runtime.ValidateDesc(srcDesc, "src"))
                return 0;

            int batches = resultDesc.Dim0;
            int rows = resultDesc.Dim1;
            int cols = resultDesc.Dim2;
            int shared = m1Desc.Dim2;

            if (m1Desc.Dim0 != batches || m2Desc.Dim0 != batches || m1Desc.Dim1 != rows || m2Desc.Dim1 != shared || m2Desc.Dim2 != cols)
            {
                Runtime.SetLastError("Size mismatch passed to ggml addmmbatch.");
                return 0;
            }

            if (beta != 0.0f && ((batches % srcDesc.Dim0) != 0 || (rows % srcDesc.Dim1) != 0 || (cols % srcDesc.Dim2) != 0))
            {
                Runtime.SetLastError("Source tensor shape cannot be broadcast to result shape for ggml addmmbatch.");
                return 0;
            }

            if (!Runtime.CanMapStandardView(resultDesc))
            {
                Runtime.SetLastError("Result tensor layout is not supported by the ggml addmmbatch Metal path.");
                return 0;
            }

            if (beta != 0.0f && !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Source tensor layout is not supported by the ggml addmmbatch Metal path.");
                return 0;
            }

            bool useZeroCopy = Runtime.CanMapStandardView(m1Desc) && CanMapM2Direct(m2Desc);

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = default;
                Runtime.TensorBinding m1Binding = default;
                Runtime.TensorBinding srcBinding = default;
                float[] packedM1 = null;
                float[] packedM2 = null;
                Runtime.TensorBinding m2Binding = default;
                bool m2ZeroCopy = false;

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out IntPtr resultBuf))
                        buffers.Add(resultBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, m1Desc, out m1Binding, out IntPtr m1Buf))
                        buffers.Add(m1Buf);
                    else
                        useZeroCopy = false;
                }
                else
                {
                    m1Binding = Runtime.CanMapStandardView(m1Desc)
                        ? Runtime.CreateStandardBinding(context.Ctx, m1Desc)
                        : CreatePackedStandardBinding(context.Ctx, m1Desc, out packedM1);
                }

                if (useZeroCopy && beta != 0.0f)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, srcDesc, out srcBinding, out IntPtr srcBuf))
                        buffers.Add(srcBuf);
                    else
                        useZeroCopy = false;
                }
                else if (beta != 0.0f)
                {
                    srcBinding = Runtime.CreateStandardBinding(context.Ctx, srcDesc);
                }

                if (useZeroCopy && CanMapM2Direct(m2Desc))
                {
                    if (CreateBindingFromHostPtrDirectM2(context.Ctx, m2Desc, out m2Binding, out IntPtr m2Buf))
                    {
                        m2ZeroCopy = true;
                        buffers.Add(m2Buf);
                    }
                }
                if (!m2ZeroCopy)
                {
                    m2Binding = CanMapM2Direct(m2Desc)
                        ? CreateDirectM2Binding(context.Ctx, m2Desc)
                        : CreatePackedM2Binding(context.Ctx, m2Desc, out packedM2);
                }

                if (!resultBinding.IsValid ||
                    !m1Binding.IsValid ||
                    !m2Binding.IsValid ||
                    (beta != 0.0f && !srcBinding.IsValid))
                {
                    Runtime.SetLastError("Failed to allocate ggml tensor views.");
                    return 0;
                }

                GgmlTensor* mmTensor = ggml_mul_mat(context.Ctx, m2Binding.Tensor, m1Binding.Tensor);
                if (mmTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml batched matmul node.");
                    return 0;
                }

                GgmlTensor* combinedTensor = mmTensor;
                if (alpha != 1.0f)
                {
                    combinedTensor = ggml_scale(context.Ctx, combinedTensor, alpha);
                    if (combinedTensor == null)
                    {
                        Runtime.SetLastError("Failed to scale ggml batched matmul output.");
                        return 0;
                    }
                }

                if (beta != 0.0f)
                {
                    GgmlTensor* scaledSrc = srcBinding.Tensor;
                    if (beta != 1.0f)
                    {
                        scaledSrc = ggml_scale(context.Ctx, srcBinding.Tensor, beta);
                        if (scaledSrc == null)
                        {
                            Runtime.SetLastError("Failed to scale ggml batched source tensor.");
                            return 0;
                        }
                    }

                    combinedTensor = ggml_add(context.Ctx, combinedTensor, scaledSrc);
                    if (combinedTensor == null)
                    {
                        Runtime.SetLastError("Failed to create ggml batched add node.");
                        return 0;
                    }
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, combinedTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml batched output copy node.");
                    return 0;
                }

                ggml_set_output(outputTensor);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }

                ggml_build_forward_expand(graph, outputTensor);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!useZeroCopy)
                {
                    if (beta != 0.0f)
                        Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);
                    UploadPackedOrRaw(m1Binding, m1Desc.Data, packedM1);
                    if (resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                }
                if (!m2ZeroCopy)
                {
                    UploadPackedOrRaw(m2Binding, m2Desc.Data, packedM2);
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                Runtime.FinalizeCompute(useZeroCopy, resultBinding, resultDesc.Data, resultBinding.RawBytes);
                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // MulMatId (MoE expert-routed matmul)
        // --------------------------------------------------------------

        internal static int MulMatIdF32(
            in GgmlTensorView3D resultDesc,
            in GgmlTensorView3D expertDesc,
            in GgmlTensorView3D inputDesc,
            in GgmlContiguousTensor idsDesc,
            int idsRows,
            int idsCols)
        {
            if (!Runtime.EnsureBackend())
                return 0;

            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(expertDesc, "expertWeights") || !Runtime.ValidateDesc(inputDesc, "input") || !Runtime.ValidateDesc(idsDesc, "ids"))
                return 0;

            if (idsRows <= 0 || idsCols <= 0)
            {
                Runtime.SetLastError("mulmatid requires positive id matrix dimensions.");
                return 0;
            }

            if (expertDesc.Dim2 != inputDesc.Dim2)
            {
                Runtime.SetLastError("mulmatid expects expert weights and input to share the inner dimension.");
                return 0;
            }

            if (inputDesc.Dim0 != idsRows || (idsCols % inputDesc.Dim1) != 0)
            {
                Runtime.SetLastError("mulmatid expects ids rows to match input tokens and ids cols to broadcast over input expert slots.");
                return 0;
            }

            if (resultDesc.Dim0 != inputDesc.Dim0 || resultDesc.Dim1 != idsCols || resultDesc.Dim2 != expertDesc.Dim1)
            {
                Runtime.SetLastError("mulmatid expects result shape [tokens, expert_used, rows].");
                return 0;
            }

            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(expertDesc) || !Runtime.CanMapStandardView(inputDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml mulmatid path.");
                return 0;
            }

            bool useZeroCopy = Runtime.CanMapStandardView(resultDesc) && Runtime.CanMapStandardView(expertDesc) && Runtime.CanMapStandardView(inputDesc);
            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(2 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = default;
                Runtime.TensorBinding expertBinding = default;
                Runtime.TensorBinding inputBinding = default;

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out IntPtr resultBuf))
                        buffers.Add(resultBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, expertDesc, out expertBinding, out IntPtr expertBuf))
                        buffers.Add(expertBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                    expertBinding = Runtime.CreateStandardBinding(context.Ctx, expertDesc);

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, inputDesc, out inputBinding, out IntPtr inputBuf))
                        buffers.Add(inputBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                    inputBinding = Runtime.CreateStandardBinding(context.Ctx, inputDesc);

                GgmlTensor* idsTensor = ggml_new_tensor_2d(context.Ctx, GGML_TYPE_I32, idsCols, idsRows);
                if (!resultBinding.IsValid ||
                    !expertBinding.IsValid ||
                    !inputBinding.IsValid ||
                    idsTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml mulmatid tensors.");
                    return 0;
                }

                GgmlTensor* valueTensor = ggml_mul_mat_id(context.Ctx, expertBinding.Tensor, inputBinding.Tensor, idsTensor);
                if (valueTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml mul_mat_id node.");
                    return 0;
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, valueTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml mulmatid output copy node.");
                    return 0;
                }

                ggml_set_output(outputTensor);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }

                ggml_build_forward_expand(graph, outputTensor);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(expertBinding, expertDesc.Data, expertBinding.RawBytes);
                    Runtime.UploadBinding(inputBinding, inputDesc.Data, inputBinding.RawBytes);
                }

                int[] ids = new int[(int)idsDesc.ElementCount];
                if (!ReadI32Values(ids, idsDesc))
                    return 0;
                fixed (int* idsPtr = ids)
                {
                    ggml_backend_tensor_set(idsTensor, (IntPtr)idsPtr, 0, (nuint)(ids.Length * sizeof(int)));
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                Runtime.FinalizeCompute(useZeroCopy, resultBinding, resultDesc.Data, resultBinding.RawBytes);
                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // AddId (MoE expert-routed bias add)
        // --------------------------------------------------------------

        internal static int AddIdF32(
            in GgmlTensorView3D resultDesc,
            in GgmlTensorView3D srcDesc,
            in GgmlTensorView2D biasDesc,
            in GgmlContiguousTensor idsDesc,
            int idsRows,
            int idsCols)
        {
            if (!Runtime.EnsureBackend())
                return 0;

            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src") || !Runtime.ValidateDesc(biasDesc, "bias") || !Runtime.ValidateDesc(idsDesc, "ids"))
                return 0;

            if (idsRows <= 0 || idsCols <= 0)
            {
                Runtime.SetLastError("addid requires positive id matrix dimensions.");
                return 0;
            }

            if (resultDesc.Dim0 != srcDesc.Dim0 || resultDesc.Dim1 != srcDesc.Dim1 || resultDesc.Dim2 != srcDesc.Dim2)
            {
                Runtime.SetLastError("addid expects result and src to have the same shape.");
                return 0;
            }

            if (srcDesc.Dim0 != idsRows || srcDesc.Dim1 != idsCols || srcDesc.Dim2 != biasDesc.Dim1)
            {
                Runtime.SetLastError("addid expects src shape [tokens, expert_used, rows], bias shape [experts, rows], and ids shape [tokens, expert_used].");
                return 0;
            }

            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc) || !Runtime.CanMapStandardView(biasDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml addid path.");
                return 0;
            }

            bool useZeroCopy = Runtime.CanMapStandardView(resultDesc) && Runtime.CanMapStandardView(srcDesc) && Runtime.CanMapStandardView(biasDesc);
            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(2 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = default;
                Runtime.TensorBinding srcBinding = default;
                Runtime.TensorBinding biasBinding = default;

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out IntPtr resultBuf))
                        buffers.Add(resultBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, srcDesc, out srcBinding, out IntPtr srcBuf))
                        buffers.Add(srcBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                    srcBinding = Runtime.CreateStandardBinding(context.Ctx, srcDesc);

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, biasDesc, out biasBinding, out IntPtr biasBuf))
                        buffers.Add(biasBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                    biasBinding = Runtime.CreateStandardBinding(context.Ctx, biasDesc);

                GgmlTensor* idsTensor = ggml_new_tensor_2d(context.Ctx, GGML_TYPE_I32, idsCols, idsRows);
                if (!resultBinding.IsValid ||
                    !srcBinding.IsValid ||
                    !biasBinding.IsValid ||
                    idsTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml addid tensors.");
                    return 0;
                }

                GgmlTensor* valueTensor = ggml_add_id(context.Ctx, srcBinding.Tensor, biasBinding.Tensor, idsTensor);
                if (valueTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml add_id node.");
                    return 0;
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, valueTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml addid output copy node.");
                    return 0;
                }

                ggml_set_output(outputTensor);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }

                ggml_build_forward_expand(graph, outputTensor);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);
                    Runtime.UploadBinding(biasBinding, biasDesc.Data, biasBinding.RawBytes);
                }

                int[] ids = new int[(int)idsDesc.ElementCount];
                if (!ReadI32Values(ids, idsDesc))
                    return 0;
                fixed (int* idsPtr = ids)
                {
                    ggml_backend_tensor_set(idsTensor, (IntPtr)idsPtr, 0, (nuint)(ids.Length * sizeof(int)));
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                Runtime.FinalizeCompute(useZeroCopy, resultBinding, resultDesc.Data, resultBinding.RawBytes);
                Runtime.ClearLastError();
                return 1;
            }
        }
    }
}
