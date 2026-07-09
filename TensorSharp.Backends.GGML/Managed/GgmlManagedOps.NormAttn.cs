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
    // Managed port of TensorSharp.GGML.Native/ggml_ops_norm_attn.cpp. Each
    // method mirrors the corresponding *_impl / TSGgml_* entry point: validate
    // the descriptors, bind the caller's host buffers (zero-copy where the
    // backend allows it, staged upload otherwise), build the same ggml graph
    // the native kernel builds, compute on the shared backend, and download
    // the result when it was not host-mapped. The only deliberate deviation is
    // finalize_compute_with_download's Metal lazy-sync branch, which the
    // managed layer replaces with the eager synchronize+download model (see
    // GgmlManagedRuntime.FinalizeCompute for the rationale).
    internal static unsafe partial class GgmlManagedOps
    {
        static GgmlManagedOps()
        {
            // Ensure the assembly-wide DllImport resolver is registered before
            // this class issues its first P/Invoke into the GgmlOps module.
            GgmlNative.EnsureImportResolverRegistered();
        }

        // ------------------------------------------------------------------
        // ggml P/Invoke bindings used only by this module (same style as
        // Interop/GgmlApi.cs; the symbols are exported from GgmlOps via
        // ggml_api_exports.def / default visibility).
        // ------------------------------------------------------------------

        private const string NormAttnDllName = "GgmlOps";
        private const CallingConvention NormAttnCdecl = CallingConvention.Cdecl;

        [DllImport(NormAttnDllName, CallingConvention = NormAttnCdecl)] private static extern void ggml_set_param(GgmlTensor* tensor);
        [DllImport(NormAttnDllName, CallingConvention = NormAttnCdecl)] private static extern void ggml_set_loss(GgmlTensor* tensor);
        [DllImport(NormAttnDllName, CallingConvention = NormAttnCdecl)] private static extern void ggml_build_backward_expand(IntPtr ctx, IntPtr cgraph, GgmlTensor** grad_accs);
        [DllImport(NormAttnDllName, CallingConvention = NormAttnCdecl)] private static extern GgmlTensor* ggml_graph_get_grad(IntPtr cgraph, GgmlTensor* node);
        [DllImport(NormAttnDllName, CallingConvention = NormAttnCdecl)] private static extern void ggml_graph_reset(IntPtr cgraph);
        [DllImport(NormAttnDllName, CallingConvention = NormAttnCdecl)] private static extern void ggml_mul_mat_set_prec(GgmlTensor* a, int prec);
        [DllImport(NormAttnDllName, CallingConvention = NormAttnCdecl)] private static extern void ggml_soft_max_add_sinks(GgmlTensor* a, GgmlTensor* sinks);
        [DllImport(NormAttnDllName, CallingConvention = NormAttnCdecl)] private static extern ushort ggml_fp32_to_fp16(float value);

        // enum ggml_prec (ggml.h)
        private const int GGML_PREC_F32 = 10;

        // ------------------------------------------------------------------
        // Module-private helpers (ports of ggml_ops_internal.h /
        // ggml_ops_core.cpp helpers that GgmlManagedRuntime does not carry)
        // ------------------------------------------------------------------

        private static bool BackendSupportsOp(GgmlTensor* op) =>
            op != null && Runtime.Backend != IntPtr.Zero && ggml_backend_supports_op(Runtime.Backend, op);

        // Mirrors tsg::create_scalar_binding.
        private static Runtime.TensorBinding CreateScalarBinding(IntPtr ctx)
        {
            GgmlTensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            return new Runtime.TensorBinding { Storage = tensor, Tensor = tensor, RawBytes = sizeof(float) };
        }

        // Mirrors tsg::is_vector_like.
        private static bool IsVectorLike(in GgmlTensorView4D desc, long width) =>
            desc.Ne0 == width && desc.Ne1 == 1 && desc.Ne2 == 1 && desc.Ne3 == 1;

        // Mirrors tsg::make_norm_tensor.
        private static GgmlTensor* MakeNormTensor(IntPtr ctx, int op, GgmlTensor* src, float eps)
        {
            switch ((GgmlNormOp)op)
            {
                case GgmlNormOp.LayerNorm: return ggml_norm(ctx, src, eps);
                case GgmlNormOp.RmsNorm: return ggml_rms_norm(ctx, src, eps);
                default:
                    Runtime.SetLastError("Unsupported norm ggml op code.");
                    return null;
            }
        }

        // Mirrors tsg::flatten_to_rows.
        private static GgmlTensor* FlattenToRows(IntPtr ctx, GgmlTensor* tensor, long cols, long rows) =>
            tensor == null ? null : ggml_reshape_2d(ctx, tensor, cols, rows);

        // Mirrors tsg::sum_rows_to_feature_vector.
        private static GgmlTensor* SumRowsToFeatureVector(IntPtr ctx, GgmlTensor* tensor)
        {
            GgmlTensor* transposed = ggml_transpose(ctx, tensor);
            GgmlTensor* transposedContiguous = transposed == null ? null : ggml_cont(ctx, transposed);
            GgmlTensor* summed = transposedContiguous == null ? null : ggml_sum_rows(ctx, transposedContiguous);
            GgmlTensor* restored = summed == null ? null : ggml_transpose(ctx, summed);
            return restored == null ? null : ggml_cont(ctx, restored);
        }

        // Mirrors tsg::read_i32_values (named variant so the error string
        // matches the native "Unsupported element type for positions.").
        private static bool ReadI32Values(int[] output, in GgmlContiguousTensor desc, string name)
        {
            if (desc.ElementType == 3 /* I32 */)
            {
                new ReadOnlySpan<int>((void*)desc.Data, output.Length).CopyTo(output);
                return true;
            }
            if (desc.ElementType == 0 /* F32 */)
            {
                var raw = new ReadOnlySpan<float>((void*)desc.Data, output.Length);
                for (int i = 0; i < output.Length; ++i)
                    output[i] = (int)raw[i];
                return true;
            }
            Runtime.SetLastError($"Unsupported element type for {name}.");
            return false;
        }

        // Mirrors tsg::finalize_compute_with_download for results that live in
        // a ggml-owned backend buffer and must always be copied back. The
        // native Metal lazy-sync branch (async blit download) is intentionally
        // simplified to the eager branch, exactly like Runtime.FinalizeCompute.
        private static void FinalizeComputeWithDownload(GgmlTensor* resultStorage, IntPtr resultData, nuint resultBytes)
        {
            ggml_backend_synchronize(Runtime.Backend);
            if (resultStorage != null && resultData != IntPtr.Zero && resultBytes > 0)
            {
                ggml_backend_tensor_get(resultStorage, resultData, 0, resultBytes);
            }
        }

        // Shared causal (+ optional sliding window) F16 additive mask builder.
        // Row q attends key k iff winStart <= k <= maskStartPos + q; everything
        // else gets -inf. Identical loop to the four copies in the native file.
        private static void FillCausalSwaMask(ushort[] mask, int width, int seqLen, int maskStartPos, int slidingWindow)
        {
            ushort negInf = ggml_fp32_to_fp16(float.NegativeInfinity);
            ushort zeroVal = ggml_fp32_to_fp16(0.0f);
            for (int qIdx = 0; qIdx < seqLen; qIdx++)
            {
                int threshold = maskStartPos + qIdx;
                int winStart = (slidingWindow > 0) ? Math.Max(0, threshold - slidingWindow + 1) : 0;
                int rowBase = qIdx * width;
                for (int kvIdx = 0; kvIdx < width; kvIdx++)
                    mask[rowBase + kvIdx] = (kvIdx > threshold || kvIdx < winStart) ? negInf : zeroVal;
            }
        }

        // --------------------------------------------------------------
        // Norm (LayerNorm / RmsNorm forward)
        // --------------------------------------------------------------

        // Builds norm(srcPart) * gamma (+ beta) copied into dstPart and adds
        // the chain to the graph (the native build_norm_chain lambda).
        private static bool BuildNormChain(
            IntPtr ctx,
            IntPtr graph,
            int op,
            float eps,
            GgmlTensor* srcPart,
            GgmlTensor* dstPart,
            GgmlTensor* contiguousGamma,
            GgmlTensor* contiguousBeta,
            bool hasBeta)
        {
            GgmlTensor* valueTensor = MakeNormTensor(ctx, op, srcPart, eps);
            if (valueTensor == null)
            {
                // MakeNormTensor already set the op-code error (mirrors the
                // native empty-last-error fallback to "Failed to create ggml
                // norm node.", which is unreachable in practice).
                return false;
            }

            valueTensor = ggml_mul(ctx, valueTensor, contiguousGamma);
            if (valueTensor == null)
            {
                Runtime.SetLastError("Failed to create ggml norm scale node.");
                return false;
            }

            if (hasBeta)
            {
                valueTensor = ggml_add(ctx, valueTensor, contiguousBeta);
                if (valueTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml norm bias node.");
                    return false;
                }
            }

            GgmlTensor* outputTensor = ggml_cpy(ctx, valueTensor, dstPart);
            if (outputTensor == null)
            {
                Runtime.SetLastError("Failed to create ggml norm output copy node.");
                return false;
            }

            ggml_set_output(outputTensor);
            ggml_build_forward_expand(graph, outputTensor);
            return true;
        }

        internal static int NormF32(
            int op,
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D srcDesc,
            in GgmlTensorView4D gammaDesc,
            in GgmlTensorView4D betaDesc,
            bool hasBeta,
            float eps)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src") || !Runtime.ValidateDesc(gammaDesc, "gamma"))
                return 0;
            if (hasBeta && !Runtime.ValidateDesc(betaDesc, "beta"))
                return 0;
            if (!Runtime.SameShape(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Result tensor shape does not match source tensor shape for ggml norm op.");
                return 0;
            }
            if (!CanRepeat(gammaDesc, srcDesc) || (hasBeta && !CanRepeat(betaDesc, srcDesc)))
            {
                Runtime.SetLastError("gamma/beta tensor shape cannot be broadcast to source tensor for ggml norm op.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc) || !Runtime.CanMapStandardView(gammaDesc) ||
                (hasBeta && !Runtime.CanMapStandardView(betaDesc)))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml norm Metal path.");
                return 0;
            }

            bool useZeroCopy = Runtime.CanMapStandardView(resultDesc) && Runtime.CanMapStandardView(srcDesc) &&
                Runtime.CanMapStandardView(gammaDesc) && (!hasBeta || Runtime.CanMapStandardView(betaDesc));
            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(3 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = default;
                Runtime.TensorBinding srcBinding = default;
                Runtime.TensorBinding gammaBinding = default;
                Runtime.TensorBinding betaBinding = default;
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, srcDesc, out srcBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, gammaDesc, out gammaBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy && hasBeta)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, betaDesc, out betaBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                {
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);
                    srcBinding = Runtime.CreateStandardBinding(context.Ctx, srcDesc);
                    gammaBinding = Runtime.CreateStandardBinding(context.Ctx, gammaDesc);
                    if (hasBeta)
                        betaBinding = Runtime.CreateStandardBinding(context.Ctx, betaDesc);
                }
                if (!resultBinding.IsValid || !srcBinding.IsValid || !gammaBinding.IsValid)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors.");
                    return 0;
                }
                if (hasBeta && !betaBinding.IsValid)
                {
                    Runtime.SetLastError("Failed to allocate ggml beta tensor.");
                    return 0;
                }

                GgmlTensor* contiguousSrc = ggml_cont(context.Ctx, srcBinding.Tensor);
                GgmlTensor* contiguousGamma = ggml_cont(context.Ctx, gammaBinding.Tensor);
                if (contiguousSrc == null || contiguousGamma == null)
                {
                    Runtime.SetLastError("Failed to create ggml contiguous norm inputs.");
                    return 0;
                }

                GgmlTensor* contiguousBeta = null;
                if (hasBeta)
                {
                    contiguousBeta = ggml_cont(context.Ctx, betaBinding.Tensor);
                    if (contiguousBeta == null)
                    {
                        Runtime.SetLastError("Failed to create ggml contiguous beta tensor.");
                        return 0;
                    }
                }

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }

                // ggml-vulkan dispatches the row-wise norm family with one
                // workgroup per dim-1 row and only guarantees 65535 workgroups
                // per dispatch dimension; split row-heavy 2D norms into
                // <=32768-row sub-norms inside the same graph (see the native
                // comment for the full story).
                const long kMaxNormRowsPerOp = 32768;
                GgmlTensor* resultTensor = resultBinding.Tensor;
                bool splitRows =
                    contiguousSrc->ne[1] > kMaxNormRowsPerOp &&
                    contiguousSrc->ne[2] == 1 && contiguousSrc->ne[3] == 1 &&
                    resultTensor->ne[2] == 1 && resultTensor->ne[3] == 1 &&
                    resultTensor->ne[0] == contiguousSrc->ne[0] &&
                    resultTensor->ne[1] == contiguousSrc->ne[1];

                if (!splitRows)
                {
                    if (!BuildNormChain(context.Ctx, graph, op, eps, contiguousSrc, resultTensor, contiguousGamma, contiguousBeta, hasBeta))
                        return 0;
                }
                else
                {
                    long totalRows = contiguousSrc->ne[1];
                    for (long row0 = 0; row0 < totalRows; row0 += kMaxNormRowsPerOp)
                    {
                        long chunkRows = Math.Min(kMaxNormRowsPerOp, totalRows - row0);
                        GgmlTensor* srcPart = ggml_view_2d(
                            context.Ctx, contiguousSrc,
                            contiguousSrc->ne[0], chunkRows,
                            (nuint)contiguousSrc->nb[1],
                            (nuint)((ulong)row0 * contiguousSrc->nb[1]));
                        GgmlTensor* dstPart = ggml_view_2d(
                            context.Ctx, resultTensor,
                            resultTensor->ne[0], chunkRows,
                            (nuint)resultTensor->nb[1],
                            (nuint)((ulong)row0 * resultTensor->nb[1]));
                        if (srcPart == null || dstPart == null)
                        {
                            Runtime.SetLastError("Failed to create ggml norm row-chunk views.");
                            return 0;
                        }
                        if (!BuildNormChain(context.Ctx, graph, op, eps, srcPart, dstPart, contiguousGamma, contiguousBeta, hasBeta))
                            return 0;
                    }
                }

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
                    Runtime.UploadBinding(gammaBinding, gammaDesc.Data, gammaBinding.RawBytes);
                    if (hasBeta)
                        Runtime.UploadBinding(betaBinding, betaDesc.Data, betaBinding.RawBytes);
                    if (resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
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
        // NormGrad (LayerNorm via ggml autodiff, RmsNorm via explicit graph)
        // --------------------------------------------------------------

        internal static int NormGradF32(
            int op,
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D gradGammaDesc,
            in GgmlTensorView4D gradBetaDesc,
            in GgmlTensorView4D adjDesc,
            in GgmlTensorView4D xDesc,
            in GgmlTensorView4D gammaDesc,
            bool hasGradBeta,
            float eps)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result")
                || !Runtime.ValidateDesc(gradGammaDesc, "gradGamma")
                || !Runtime.ValidateDesc(adjDesc, "adj")
                || !Runtime.ValidateDesc(xDesc, "x")
                || !Runtime.ValidateDesc(gammaDesc, "gamma"))
            {
                return 0;
            }
            if (hasGradBeta && !Runtime.ValidateDesc(gradBetaDesc, "gradBeta"))
                return 0;
            if (!Runtime.SameShape(resultDesc, adjDesc) || !Runtime.SameShape(adjDesc, xDesc))
            {
                Runtime.SetLastError("Tensor shape mismatch passed to ggml norm grad.");
                return 0;
            }
            if (!IsVectorLike(gammaDesc, xDesc.Ne0) || !IsVectorLike(gradGammaDesc, xDesc.Ne0) || (hasGradBeta && !IsVectorLike(gradBetaDesc, xDesc.Ne0)))
            {
                Runtime.SetLastError("gamma/gradGamma/gradBeta must match the last source dimension for ggml norm grad.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc)
                || !Runtime.CanMapStandardView(gradGammaDesc)
                || !Runtime.CanMapStandardView(adjDesc)
                || !Runtime.CanMapStandardView(xDesc)
                || !Runtime.CanMapStandardView(gammaDesc)
                || (hasGradBeta && !Runtime.CanMapStandardView(gradBetaDesc)))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml norm-grad Metal path.");
                return 0;
            }

            bool useZeroCopy = Runtime.CanMapStandardView(resultDesc) && Runtime.CanMapStandardView(gradGammaDesc) &&
                Runtime.CanMapStandardView(adjDesc) && Runtime.CanMapStandardView(xDesc) && Runtime.CanMapStandardView(gammaDesc) &&
                (!hasGradBeta || Runtime.CanMapStandardView(gradBetaDesc));
            using var buffers = new Runtime.BufferList();
            const int graphCapacity = 512;
            nuint ctxSize = 16 * 1024 * 1024 + ggml_graph_overhead_custom(graphCapacity, true);

            var context = new Runtime.PooledContext();
            if (!context.Init(ctxSize))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = default;
                Runtime.TensorBinding gradGammaBinding = default;
                Runtime.TensorBinding adjBinding = default;
                Runtime.TensorBinding xBinding = default;
                Runtime.TensorBinding gammaBinding = default;
                Runtime.TensorBinding gradBetaBinding = default;
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, gradGammaDesc, out gradGammaBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, adjDesc, out adjBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, xDesc, out xBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, gammaDesc, out gammaBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy && hasGradBeta)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, gradBetaDesc, out gradBetaBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                {
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);
                    gradGammaBinding = Runtime.CreateStandardBinding(context.Ctx, gradGammaDesc);
                    adjBinding = Runtime.CreateStandardBinding(context.Ctx, adjDesc);
                    xBinding = Runtime.CreateStandardBinding(context.Ctx, xDesc);
                    gammaBinding = Runtime.CreateStandardBinding(context.Ctx, gammaDesc);
                    if (hasGradBeta)
                        gradBetaBinding = Runtime.CreateStandardBinding(context.Ctx, gradBetaDesc);
                }
                Runtime.TensorBinding epsBinding = CreateScalarBinding(context.Ctx);

                if (!resultBinding.IsValid || !gradGammaBinding.IsValid || !adjBinding.IsValid ||
                    !xBinding.IsValid || !gammaBinding.IsValid || !epsBinding.IsValid)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors.");
                    return 0;
                }
                if (hasGradBeta && !gradBetaBinding.IsValid)
                {
                    Runtime.SetLastError("Failed to allocate ggml gradBeta tensor.");
                    return 0;
                }

                GgmlTensor* contiguousResult = ggml_cont(context.Ctx, resultBinding.Tensor);
                GgmlTensor* contiguousGradGamma = ggml_cont(context.Ctx, gradGammaBinding.Tensor);
                GgmlTensor* contiguousAdj = ggml_cont(context.Ctx, adjBinding.Tensor);
                GgmlTensor* contiguousX = ggml_cont(context.Ctx, xBinding.Tensor);
                GgmlTensor* contiguousGamma = ggml_cont(context.Ctx, gammaBinding.Tensor);
                GgmlTensor* contiguousGradBeta = null;
                if (hasGradBeta)
                {
                    contiguousGradBeta = ggml_cont(context.Ctx, gradBetaBinding.Tensor);
                }

                if (contiguousResult == null || contiguousGradGamma == null || contiguousAdj == null || contiguousX == null || contiguousGamma == null ||
                    (hasGradBeta && contiguousGradBeta == null))
                {
                    Runtime.SetLastError("Failed to create ggml contiguous norm-grad inputs.");
                    return 0;
                }

                if ((GgmlNormOp)op == GgmlNormOp.LayerNorm)
                {
                    ggml_set_param(xBinding.Storage);

                    GgmlTensor* normValue = ggml_norm(context.Ctx, contiguousX, eps);
                    GgmlTensor* scaledValue = normValue == null ? null : ggml_mul(context.Ctx, normValue, contiguousGamma);
                    GgmlTensor* weightedValue = scaledValue == null ? null : ggml_mul(context.Ctx, scaledValue, contiguousAdj);
                    GgmlTensor* lossTensor = weightedValue == null ? null : ggml_sum(context.Ctx, weightedValue);
                    if (lossTensor == null)
                    {
                        Runtime.SetLastError("Failed to create ggml layernorm backward loss graph.");
                        return 0;
                    }
                    ggml_set_loss(lossTensor);

                    IntPtr graph = ggml_new_graph_custom(context.Ctx, graphCapacity, true);
                    if (graph == IntPtr.Zero)
                    {
                        Runtime.SetLastError("Failed to create ggml backward graph.");
                        return 0;
                    }

                    ggml_build_forward_expand(graph, lossTensor);
                    ggml_build_backward_expand(context.Ctx, graph, null);

                    GgmlTensor* dxDelta = ggml_graph_get_grad(graph, contiguousX);
                    if (dxDelta == null)
                    {
                        Runtime.SetLastError("Failed to obtain ggml layernorm input gradient.");
                        return 0;
                    }

                    long rows = Runtime.FlatRowCount(xDesc);
                    GgmlTensor* flatAdj = FlattenToRows(context.Ctx, contiguousAdj, xDesc.Ne0, rows);
                    GgmlTensor* flatNorm = normValue == null ? null : FlattenToRows(context.Ctx, normValue, xDesc.Ne0, rows);
                    GgmlTensor* flatGradGamma = FlattenToRows(context.Ctx, contiguousGradGamma, xDesc.Ne0, 1);
                    GgmlTensor* flatGradBeta = hasGradBeta ? FlattenToRows(context.Ctx, contiguousGradBeta, xDesc.Ne0, 1) : null;
                    if (flatAdj == null || flatNorm == null || flatGradGamma == null || (hasGradBeta && flatGradBeta == null))
                    {
                        Runtime.SetLastError("Failed to reshape ggml layernorm gradient tensors.");
                        return 0;
                    }

                    GgmlTensor* adjNorm = ggml_mul(context.Ctx, flatAdj, flatNorm);
                    GgmlTensor* gradGammaDelta = adjNorm == null ? null : SumRowsToFeatureVector(context.Ctx, adjNorm);
                    GgmlTensor* gradBetaDelta = hasGradBeta ? SumRowsToFeatureVector(context.Ctx, flatAdj) : null;
                    if (gradGammaDelta == null || (hasGradBeta && gradBetaDelta == null))
                    {
                        Runtime.SetLastError("Failed to create ggml layernorm parameter gradients.");
                        return 0;
                    }

                    GgmlTensor* dxValue = ggml_add(context.Ctx, contiguousResult, dxDelta);
                    GgmlTensor* gradGammaValue = ggml_add(context.Ctx, flatGradGamma, gradGammaDelta);
                    GgmlTensor* gradGammaView = gradGammaValue == null ? null : ggml_reshape_4d(context.Ctx, gradGammaValue, gradGammaDesc.Ne0, gradGammaDesc.Ne1, gradGammaDesc.Ne2, gradGammaDesc.Ne3);
                    GgmlTensor* gradBetaValue = hasGradBeta ? ggml_add(context.Ctx, flatGradBeta, gradBetaDelta) : null;
                    GgmlTensor* gradBetaView = hasGradBeta && gradBetaValue != null
                        ? ggml_reshape_4d(context.Ctx, gradBetaValue, gradBetaDesc.Ne0, gradBetaDesc.Ne1, gradBetaDesc.Ne2, gradBetaDesc.Ne3)
                        : null;
                    GgmlTensor* dxOutput = dxValue == null ? null : ggml_cpy(context.Ctx, dxValue, resultBinding.Tensor);
                    GgmlTensor* gradGammaOutput = gradGammaView == null ? null : ggml_cpy(context.Ctx, gradGammaView, gradGammaBinding.Tensor);
                    GgmlTensor* gradBetaOutput = hasGradBeta
                        ? (gradBetaView == null ? null : ggml_cpy(context.Ctx, gradBetaView, gradBetaBinding.Tensor))
                        : null;
                    if (dxOutput == null || gradGammaOutput == null || (hasGradBeta && gradBetaOutput == null))
                    {
                        Runtime.SetLastError("Failed to create ggml layernorm output copy nodes.");
                        return 0;
                    }

                    ggml_set_output(dxOutput);
                    ggml_set_output(gradGammaOutput);
                    if (hasGradBeta)
                    {
                        ggml_set_output(gradBetaOutput);
                    }

                    ggml_build_forward_expand(graph, dxOutput);
                    ggml_build_forward_expand(graph, gradGammaOutput);
                    if (hasGradBeta)
                    {
                        ggml_build_forward_expand(graph, gradBetaOutput);
                    }

                    IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                    if (computeBuffer == IntPtr.Zero)
                    {
                        Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                        return 0;
                    }
                    buffers.Add(computeBuffer);

                    if (!useZeroCopy)
                    {
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                        Runtime.UploadBinding(gradGammaBinding, gradGammaDesc.Data, gradGammaBinding.RawBytes);
                        Runtime.UploadBinding(adjBinding, adjDesc.Data, adjBinding.RawBytes);
                        Runtime.UploadBinding(xBinding, xDesc.Data, xBinding.RawBytes);
                        Runtime.UploadBinding(gammaBinding, gammaDesc.Data, gammaBinding.RawBytes);
                        if (hasGradBeta)
                            Runtime.UploadBinding(gradBetaBinding, gradBetaDesc.Data, gradBetaBinding.RawBytes);
                    }

                    ggml_graph_reset(graph);

                    if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                    {
                        Runtime.SetLastError("ggml backend graph execution failed.");
                        return 0;
                    }

                    // Multi-tensor download path (gradients). Drain pending
                    // async work so the explicit downloads below run on a
                    // quiesced backend.
                    GgmlNative.HostReadBarrier();
                    ggml_backend_synchronize(Runtime.Backend);
                    if (!useZeroCopy)
                    {
                        ggml_backend_tensor_get(resultBinding.Storage, resultDesc.Data, 0, resultBinding.RawBytes);
                        ggml_backend_tensor_get(gradGammaBinding.Storage, gradGammaDesc.Data, 0, gradGammaBinding.RawBytes);
                        if (hasGradBeta)
                            ggml_backend_tensor_get(gradBetaBinding.Storage, gradBetaDesc.Data, 0, gradBetaBinding.RawBytes);
                    }

                    Runtime.ClearLastError();
                    return 1;
                }

                long rmsRows = Runtime.FlatRowCount(xDesc);
                float invCols = 1.0f / xDesc.Ne0;
                float colsValue = xDesc.Ne0;

                GgmlTensor* rmsFlatAdj = FlattenToRows(context.Ctx, contiguousAdj, xDesc.Ne0, rmsRows);
                GgmlTensor* flatX = FlattenToRows(context.Ctx, contiguousX, xDesc.Ne0, rmsRows);
                GgmlTensor* flatGamma = FlattenToRows(context.Ctx, contiguousGamma, xDesc.Ne0, 1);
                GgmlTensor* rmsFlatGradGamma = FlattenToRows(context.Ctx, contiguousGradGamma, xDesc.Ne0, 1);
                GgmlTensor* rmsFlatGradBeta = hasGradBeta ? FlattenToRows(context.Ctx, contiguousGradBeta, xDesc.Ne0, 1) : null;
                if (rmsFlatAdj == null || flatX == null || flatGamma == null || rmsFlatGradGamma == null || (hasGradBeta && rmsFlatGradBeta == null))
                {
                    Runtime.SetLastError("Failed to reshape ggml norm-grad tensors.");
                    return 0;
                }

                GgmlTensor* dxDeltaFlat = null;
                GgmlTensor* rmsGradGammaDelta = null;
                GgmlTensor* rmsGradBetaDelta = null;

                if ((GgmlNormOp)op == GgmlNormOp.RmsNorm)
                {
                    GgmlTensor* nativeAdj = ggml_mul(context.Ctx, contiguousAdj, contiguousGamma);
                    GgmlTensor* nativeDx = nativeAdj == null ? null : ggml_rms_norm_back(context.Ctx, nativeAdj, contiguousX, eps);
                    if (BackendSupportsOp(nativeDx))
                    {
                        dxDeltaFlat = FlattenToRows(context.Ctx, nativeDx, xDesc.Ne0, rmsRows);
                    }

                    GgmlTensor* sq = ggml_mul(context.Ctx, flatX, flatX);
                    GgmlTensor* sqSum = sq == null ? null : ggml_sum_rows(context.Ctx, sq);
                    GgmlTensor* meanSq = sqSum == null ? null : ggml_scale(context.Ctx, sqSum, invCols);
                    GgmlTensor* epsFull = meanSq == null ? null : ggml_repeat(context.Ctx, epsBinding.Tensor, meanSq);
                    GgmlTensor* rmsSq = (meanSq == null || epsFull == null) ? null : ggml_add(context.Ctx, meanSq, epsFull);
                    GgmlTensor* rms = rmsSq == null ? null : ggml_sqrt(context.Ctx, rmsSq);
                    GgmlTensor* rmsFull = rms == null ? null : ggml_repeat(context.Ctx, rms, flatX);
                    GgmlTensor* rmsNorm = rmsFull == null ? null : ggml_div(context.Ctx, flatX, rmsFull);
                    GgmlTensor* adjRmsNorm = rmsNorm == null ? null : ggml_mul(context.Ctx, rmsFlatAdj, rmsNorm);
                    GgmlTensor* sumAdjRmsNorm = adjRmsNorm == null ? null : ggml_sum_rows(context.Ctx, adjRmsNorm);
                    GgmlTensor* sumAdjRmsNormFull = sumAdjRmsNorm == null ? null : ggml_repeat(context.Ctx, sumAdjRmsNorm, flatX);
                    GgmlTensor* weighted = (rmsNorm == null || sumAdjRmsNormFull == null) ? null : ggml_mul(context.Ctx, rmsNorm, sumAdjRmsNormFull);
                    GgmlTensor* scaledAdj = ggml_scale(context.Ctx, rmsFlatAdj, colsValue);
                    GgmlTensor* dxNumerator = (scaledAdj == null || weighted == null) ? null : ggml_sub(context.Ctx, scaledAdj, weighted);
                    GgmlTensor* dxDenominator = rmsFull == null ? null : ggml_scale(context.Ctx, rmsFull, colsValue);
                    GgmlTensor* dxCore = (dxNumerator == null || dxDenominator == null) ? null : ggml_div(context.Ctx, dxNumerator, dxDenominator);
                    GgmlTensor* unclamped = dxCore == null ? null : ggml_mul(context.Ctx, dxCore, flatGamma);

                    if (dxDeltaFlat == null)
                    {
                        dxDeltaFlat = unclamped == null ? null : ggml_clamp(context.Ctx, unclamped, -1000.0f, 1000.0f);
                    }
                    rmsGradGammaDelta = adjRmsNorm == null ? null : SumRowsToFeatureVector(context.Ctx, adjRmsNorm);
                    if (hasGradBeta)
                    {
                        rmsGradBetaDelta = SumRowsToFeatureVector(context.Ctx, rmsFlatAdj);
                    }
                }
                else
                {
                    Runtime.SetLastError("Unsupported norm-grad ggml op code.");
                    return 0;
                }

                if (dxDeltaFlat == null || rmsGradGammaDelta == null || (hasGradBeta && rmsGradBetaDelta == null))
                {
                    Runtime.SetLastError("Failed to create ggml norm-grad intermediate tensors.");
                    return 0;
                }

                GgmlTensor* rmsDxDelta = ggml_reshape_4d(context.Ctx, dxDeltaFlat, resultDesc.Ne0, resultDesc.Ne1, resultDesc.Ne2, resultDesc.Ne3);
                GgmlTensor* rmsDxValue = rmsDxDelta == null ? null : ggml_add(context.Ctx, contiguousResult, rmsDxDelta);
                GgmlTensor* rmsGradGammaValue = ggml_add(context.Ctx, rmsFlatGradGamma, rmsGradGammaDelta);
                GgmlTensor* rmsGradGammaView = rmsGradGammaValue == null ? null : ggml_reshape_4d(context.Ctx, rmsGradGammaValue, gradGammaDesc.Ne0, gradGammaDesc.Ne1, gradGammaDesc.Ne2, gradGammaDesc.Ne3);
                GgmlTensor* rmsGradBetaValue = null;
                GgmlTensor* rmsGradBetaView = null;
                if (hasGradBeta)
                {
                    rmsGradBetaValue = ggml_add(context.Ctx, rmsFlatGradBeta, rmsGradBetaDelta);
                    rmsGradBetaView = rmsGradBetaValue == null ? null : ggml_reshape_4d(context.Ctx, rmsGradBetaValue, gradBetaDesc.Ne0, gradBetaDesc.Ne1, gradBetaDesc.Ne2, gradBetaDesc.Ne3);
                }

                if (rmsDxValue == null || rmsGradGammaView == null || (hasGradBeta && rmsGradBetaView == null))
                {
                    Runtime.SetLastError("Failed to create ggml norm-grad accumulation tensors.");
                    return 0;
                }

                GgmlTensor* rmsDxOutput = ggml_cpy(context.Ctx, rmsDxValue, resultBinding.Tensor);
                GgmlTensor* rmsGradGammaOutput = ggml_cpy(context.Ctx, rmsGradGammaView, gradGammaBinding.Tensor);
                GgmlTensor* rmsGradBetaOutput = hasGradBeta ? ggml_cpy(context.Ctx, rmsGradBetaView, gradBetaBinding.Tensor) : null;
                if (rmsDxOutput == null || rmsGradGammaOutput == null || (hasGradBeta && rmsGradBetaOutput == null))
                {
                    Runtime.SetLastError("Failed to create ggml norm-grad output copy nodes.");
                    return 0;
                }

                ggml_set_output(rmsDxOutput);
                ggml_set_output(rmsGradGammaOutput);
                if (hasGradBeta)
                {
                    ggml_set_output(rmsGradBetaOutput);
                }

                IntPtr rmsGraph = ggml_new_graph(context.Ctx);
                if (rmsGraph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }

                ggml_build_forward_expand(rmsGraph, rmsDxOutput);
                ggml_build_forward_expand(rmsGraph, rmsGradGammaOutput);
                if (hasGradBeta)
                {
                    ggml_build_forward_expand(rmsGraph, rmsGradBetaOutput);
                }

                IntPtr rmsComputeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (rmsComputeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                    return 0;
                }
                buffers.Add(rmsComputeBuffer);

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                    Runtime.UploadBinding(gradGammaBinding, gradGammaDesc.Data, gradGammaBinding.RawBytes);
                    Runtime.UploadBinding(adjBinding, adjDesc.Data, adjBinding.RawBytes);
                    Runtime.UploadBinding(xBinding, xDesc.Data, xBinding.RawBytes);
                    Runtime.UploadBinding(gammaBinding, gammaDesc.Data, gammaBinding.RawBytes);
                    if (hasGradBeta)
                        Runtime.UploadBinding(gradBetaBinding, gradBetaDesc.Data, gradBetaBinding.RawBytes);
                }
                ggml_backend_tensor_set(epsBinding.Storage, (IntPtr)(&eps), 0, sizeof(float));

                if (ggml_backend_graph_compute(Runtime.Backend, rmsGraph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                // Multi-tensor download path (gradients). Drain pending async
                // work first so the explicit downloads below run on a quiesced
                // backend.
                GgmlNative.HostReadBarrier();
                ggml_backend_synchronize(Runtime.Backend);
                if (!useZeroCopy)
                {
                    ggml_backend_tensor_get(resultBinding.Storage, resultDesc.Data, 0, resultBinding.RawBytes);
                    ggml_backend_tensor_get(gradGammaBinding.Storage, gradGammaDesc.Data, 0, gradGammaBinding.RawBytes);
                    if (hasGradBeta)
                        ggml_backend_tensor_get(gradBetaBinding.Storage, gradBetaDesc.Data, 0, gradBetaBinding.RawBytes);
                }

                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // RoPE (fixed 500000-base rope with generated positions; called by
        // both the RoPE and RoPEGrad wrappers)
        // --------------------------------------------------------------

        internal static int RoPEF32(
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D srcDesc,
            int seqLen,
            int rowOffset,
            bool addToResult,
            bool invertPositions)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src"))
                return 0;
            if (seqLen <= 0)
            {
                Runtime.SetLastError("seqLen must be positive for ggml rope.");
                return 0;
            }
            if (!Runtime.SameShape(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Source tensor shape does not match result shape for ggml rope.");
                return 0;
            }
            if ((srcDesc.Ne0 % 2) != 0)
            {
                Runtime.SetLastError("ggml rope requires an even embedding dimension.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml rope Metal path.");
                return 0;
            }

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(4 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);
                Runtime.TensorBinding srcBinding = Runtime.CreateStandardBinding(context.Ctx, srcDesc);
                GgmlTensor* positionTensor = ggml_new_tensor_1d(context.Ctx, GGML_TYPE_I32, Runtime.FlatRowCount(srcDesc));
                if (!resultBinding.IsValid || !srcBinding.IsValid || positionTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors.");
                    return 0;
                }

                GgmlTensor* contiguousSrc = ggml_cont(context.Ctx, srcBinding.Tensor);
                GgmlTensor* contiguousResult = addToResult ? ggml_cont(context.Ctx, resultBinding.Tensor) : null;
                if (contiguousSrc == null || (addToResult && contiguousResult == null))
                {
                    Runtime.SetLastError("Failed to create ggml contiguous rope inputs.");
                    return 0;
                }

                long rows = Runtime.FlatRowCount(srcDesc);
                GgmlTensor* ropeInput = ggml_reshape_4d(context.Ctx, contiguousSrc, srcDesc.Ne0, 1, rows, 1);
                GgmlTensor* ropeTensor = null;
                bool useNativeBackward = false;
                if (ropeInput != null && invertPositions)
                {
                    GgmlTensor* nativeBackward = ggml_rope_ext_back(
                        context.Ctx,
                        ropeInput,
                        positionTensor,
                        null,
                        srcDesc.Ne0,
                        0,
                        0,
                        500000.0f,
                        1.0f,
                        0.0f,
                        1.0f,
                        0.0f,
                        0.0f);
                    if (BackendSupportsOp(nativeBackward))
                    {
                        ropeTensor = nativeBackward;
                        useNativeBackward = true;
                    }
                }

                if (ropeTensor == null)
                {
                    ropeTensor = ropeInput == null ? null : ggml_rope_ext(
                        context.Ctx,
                        ropeInput,
                        positionTensor,
                        null,
                        srcDesc.Ne0,
                        0,
                        0,
                        500000.0f,
                        1.0f,
                        0.0f,
                        1.0f,
                        0.0f,
                        0.0f);
                }
                GgmlTensor* restored = ropeTensor == null ? null : ggml_reshape_4d(context.Ctx, ropeTensor, resultDesc.Ne0, resultDesc.Ne1, resultDesc.Ne2, resultDesc.Ne3);
                GgmlTensor* valueTensor = restored;
                if (addToResult)
                {
                    valueTensor = restored == null ? null : ggml_add(context.Ctx, contiguousResult, restored);
                }

                if (ropeInput == null || ropeTensor == null || valueTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml rope node.");
                    return 0;
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, valueTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml rope output copy node.");
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

                Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);

                int[] positions = new int[rows];
                for (long i = 0; i < rows; ++i)
                {
                    int position = rowOffset + (int)(i % seqLen);
                    positions[i] = (invertPositions && !useNativeBackward) ? -position : position;
                }
                fixed (int* positionsPtr = positions)
                {
                    ggml_backend_tensor_set(positionTensor, (IntPtr)positionsPtr, 0, (nuint)positions.Length * sizeof(int));
                }

                if (addToResult || resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                {
                    Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                FinalizeComputeWithDownload(resultBinding.Storage, resultDesc.Data, resultBinding.RawBytes);

                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // RoPEEx (explicit positions, full ggml_rope_ext parameter set,
        // optional freq-factors tensor)
        // --------------------------------------------------------------

        internal static int RoPEExF32(
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D srcDesc,
            in GgmlContiguousTensor positionsDesc,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            bool addToResult,
            bool invertPositions)
        {
            return RoPEExImpl(
                resultDesc, srcDesc, positionsDesc,
                ropeDim, mode, originalContextLength,
                freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow,
                addToResult, invertPositions,
                IntPtr.Zero, 0);
        }

        internal static int RoPEExFreqFactorsF32(
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D srcDesc,
            in GgmlContiguousTensor positionsDesc,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            bool addToResult,
            bool invertPositions,
            IntPtr freqFactors,
            int freqFactorsLen)
        {
            return RoPEExImpl(
                resultDesc, srcDesc, positionsDesc,
                ropeDim, mode, originalContextLength,
                freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow,
                addToResult, invertPositions,
                freqFactors, freqFactorsLen);
        }

        private static int RoPEExImpl(
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D srcDesc,
            in GgmlContiguousTensor positionsDesc,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            bool addToResult,
            bool invertPositions,
            IntPtr freqFactors,
            int freqFactorsLen)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src") || !Runtime.ValidateDesc(positionsDesc, "positions"))
                return 0;
            if (!Runtime.SameShape(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Source tensor shape does not match result shape for ggml rope_ex.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml rope_ex Metal path.");
                return 0;
            }
            if (ropeDim <= 0 || ropeDim > srcDesc.Ne0 || (ropeDim % 2) != 0)
            {
                Runtime.SetLastError("rope_dim must be positive, even, and within the source embedding dimension.");
                return 0;
            }

            long rows = Runtime.FlatRowCount(srcDesc);
            if (positionsDesc.ElementCount != rows)
            {
                Runtime.SetLastError("rope_ex expects one position per logical row.");
                return 0;
            }

            bool hasFreqFactors = freqFactors != IntPtr.Zero && freqFactorsLen > 0;
            if (hasFreqFactors && freqFactorsLen != ropeDim / 2)
            {
                Runtime.SetLastError("rope_ex freq_factors length must equal rope_dim/2.");
                return 0;
            }

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(4 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);
                Runtime.TensorBinding srcBinding = Runtime.CreateStandardBinding(context.Ctx, srcDesc);
                GgmlTensor* positionTensor = ggml_new_tensor_1d(context.Ctx, GGML_TYPE_I32, rows);
                GgmlTensor* freqFactorsTensor = hasFreqFactors
                    ? ggml_new_tensor_1d(context.Ctx, GGML_TYPE_F32, freqFactorsLen)
                    : null;
                if (!resultBinding.IsValid || !srcBinding.IsValid || positionTensor == null ||
                    (hasFreqFactors && freqFactorsTensor == null))
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors.");
                    return 0;
                }

                GgmlTensor* contiguousSrc = ggml_cont(context.Ctx, srcBinding.Tensor);
                GgmlTensor* contiguousResult = addToResult ? ggml_cont(context.Ctx, resultBinding.Tensor) : null;
                if (contiguousSrc == null || (addToResult && contiguousResult == null))
                {
                    Runtime.SetLastError("Failed to create ggml contiguous rope_ex inputs.");
                    return 0;
                }

                GgmlTensor* ropeInput = ggml_reshape_4d(context.Ctx, contiguousSrc, srcDesc.Ne0, 1, rows, 1);
                if (ropeInput == null)
                {
                    Runtime.SetLastError("Failed to reshape ggml rope_ex input.");
                    return 0;
                }

                int[] positions = new int[rows];
                if (!ReadI32Values(positions, positionsDesc, "positions"))
                {
                    return 0;
                }

                if (invertPositions)
                {
                    for (int i = 0; i < positions.Length; ++i)
                    {
                        positions[i] = -positions[i];
                    }
                }

                GgmlTensor* ropeTensor = ggml_rope_ext(
                    context.Ctx,
                    ropeInput,
                    positionTensor,
                    freqFactorsTensor,
                    ropeDim,
                    mode,
                    originalContextLength,
                    freqBase,
                    freqScale,
                    extFactor,
                    attnFactor,
                    betaFast,
                    betaSlow);
                GgmlTensor* restored = ropeTensor == null ? null : ggml_reshape_4d(context.Ctx, ropeTensor, resultDesc.Ne0, resultDesc.Ne1, resultDesc.Ne2, resultDesc.Ne3);
                GgmlTensor* valueTensor = restored;
                if (addToResult)
                {
                    valueTensor = restored == null ? null : ggml_add(context.Ctx, contiguousResult, restored);
                }

                if (ropeTensor == null || restored == null || valueTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml rope_ex node.");
                    return 0;
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, valueTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml rope_ex output copy node.");
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

                Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);
                fixed (int* positionsPtr = positions)
                {
                    ggml_backend_tensor_set(positionTensor, (IntPtr)positionsPtr, 0, (nuint)positions.Length * sizeof(int));
                }
                if (hasFreqFactors)
                {
                    ggml_backend_tensor_set(freqFactorsTensor, freqFactors, 0, (nuint)freqFactorsLen * sizeof(float));
                }

                if (addToResult || resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                {
                    Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                FinalizeComputeWithDownload(resultBinding.Storage, resultDesc.Data, resultBinding.RawBytes);

                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // RoPE MRoPE (ggml_rope_multi with per-axis position table; input
        // shape [headDim, numHeads, seqLen, 1], positions length 4*seqLen)
        // --------------------------------------------------------------

        internal static int RoPEMRoPEF32(
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D srcDesc,
            in GgmlContiguousTensor positionsDesc,
            int ropeDim,
            int mode,
            int sect0, int sect1, int sect2, int sect3,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src") || !Runtime.ValidateDesc(positionsDesc, "positions"))
                return 0;
            if (!Runtime.SameShape(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Source tensor shape does not match result shape for ggml rope_mrope.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml rope_mrope Metal path.");
                return 0;
            }
            if (ropeDim <= 0 || ropeDim > srcDesc.Ne0 || (ropeDim % 2) != 0)
            {
                Runtime.SetLastError("rope_dim must be positive, even, and within the source embedding dimension.");
                return 0;
            }

            // Expect 4-D input shape [headDim, numHeads, seqLen, 1].
            long seqLen = srcDesc.Ne2;
            long batch = srcDesc.Ne3;
            if (batch != 1)
            {
                Runtime.SetLastError("rope_mrope expects batch=1 input.");
                return 0;
            }
            if (positionsDesc.ElementCount != 4 * seqLen)
            {
                Runtime.SetLastError("rope_mrope positions length must be 4 * seqLen.");
                return 0;
            }

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(4 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);
                Runtime.TensorBinding srcBinding = Runtime.CreateStandardBinding(context.Ctx, srcDesc);
                GgmlTensor* positionTensor = ggml_new_tensor_1d(context.Ctx, GGML_TYPE_I32, 4 * seqLen);
                if (!resultBinding.IsValid || !srcBinding.IsValid || positionTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors.");
                    return 0;
                }

                // Use the existing 4D shape (headDim, numHeads, seqLen, 1). The
                // ggml_rope_multi assert is a->ne[2] * 4 == b->ne[0], i.e.
                // seqLen * 4 == positions length, which we already enforce above.
                GgmlTensor* contiguousSrc = ggml_cont(context.Ctx, srcBinding.Tensor);
                if (contiguousSrc == null)
                {
                    Runtime.SetLastError("Failed to make ggml rope_mrope source contiguous.");
                    return 0;
                }

                int[] positions = new int[4 * seqLen];
                if (!ReadI32Values(positions, positionsDesc, "positions"))
                    return 0;

                int* sectionsLocal = stackalloc int[4] { sect0, sect1, sect2, sect3 };

                GgmlTensor* ropeTensor = ggml_rope_multi(
                    context.Ctx,
                    contiguousSrc,
                    positionTensor,
                    /*freq_factors=*/null,
                    ropeDim,
                    sectionsLocal,
                    mode,
                    originalContextLength,
                    freqBase,
                    freqScale,
                    extFactor,
                    attnFactor,
                    betaFast,
                    betaSlow);

                if (ropeTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml rope_multi node.");
                    return 0;
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, ropeTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml rope_mrope output copy node.");
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

                Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);
                fixed (int* positionsPtr = positions)
                {
                    ggml_backend_tensor_set(positionTensor, (IntPtr)positionsPtr, 0, (nuint)positions.Length * sizeof(int));
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                FinalizeComputeWithDownload(resultBinding.Storage, resultDesc.Data, resultBinding.RawBytes);
                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // ScaledDotProductAttention
        // --------------------------------------------------------------

        internal static int ScaledDotProductAttentionF32(
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D queryDesc,
            in GgmlTensorView4D keyDesc,
            in GgmlTensorView4D valueDesc,
            in GgmlTensorView4D maskDesc,
            bool hasMask,
            float scale)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result")
                || !Runtime.ValidateDesc(queryDesc, "query")
                || !Runtime.ValidateDesc(keyDesc, "key")
                || !Runtime.ValidateDesc(valueDesc, "value")
                || (hasMask && !Runtime.ValidateDesc(maskDesc, "mask")))
            {
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc)
                || !Runtime.CanMapStandardView(queryDesc)
                || !Runtime.CanMapStandardView(keyDesc)
                || !Runtime.CanMapStandardView(valueDesc)
                || (hasMask && !Runtime.CanMapStandardView(maskDesc)))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml scaled_dot_product_attention path.");
                return 0;
            }
            if (queryDesc.Ne3 != keyDesc.Ne3 || queryDesc.Ne3 != valueDesc.Ne3)
            {
                Runtime.SetLastError("scaled_dot_product_attention expects matching batch dimensions.");
                return 0;
            }
            if (queryDesc.Ne2 != keyDesc.Ne2 || queryDesc.Ne2 != valueDesc.Ne2)
            {
                Runtime.SetLastError("scaled_dot_product_attention expects matching head dimensions.");
                return 0;
            }
            if (queryDesc.Ne0 != keyDesc.Ne0)
            {
                Runtime.SetLastError("scaled_dot_product_attention expects query and key to share the key dimension.");
                return 0;
            }
            if (resultDesc.Ne3 != queryDesc.Ne3 || resultDesc.Ne1 != queryDesc.Ne1 || resultDesc.Ne2 != queryDesc.Ne2 || resultDesc.Ne0 != valueDesc.Ne0)
            {
                Runtime.SetLastError("scaled_dot_product_attention expects result shape [value_dim, heads, seq_q, batch].");
                return 0;
            }
            if (hasMask)
            {
                if (maskDesc.Ne3 != queryDesc.Ne3 || maskDesc.Ne2 != queryDesc.Ne1 || maskDesc.Ne1 != queryDesc.Ne2 || maskDesc.Ne0 != keyDesc.Ne2)
                {
                    Runtime.SetLastError("scaled_dot_product_attention expects mask shape [seq_k, seq_q, heads, batch].");
                    return 0;
                }
            }

            bool useZeroCopy = true;
            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(4 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding resultBinding = default;
                Runtime.TensorBinding queryBinding = default;
                Runtime.TensorBinding keyBinding = default;
                Runtime.TensorBinding valueBinding = default;
                Runtime.TensorBinding maskBinding = default;

                // Per-tensor fallback (mirrors the native structure exactly):
                // once one binding fails, the remaining ones become staged, but
                // earlier host-ptr bindings are kept.
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy && !resultBinding.IsValid)
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, queryDesc, out queryBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy && !queryBinding.IsValid)
                    queryBinding = Runtime.CreateStandardBinding(context.Ctx, queryDesc);

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, keyDesc, out keyBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy && !keyBinding.IsValid)
                    keyBinding = Runtime.CreateStandardBinding(context.Ctx, keyDesc);

                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, valueDesc, out valueBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy && !valueBinding.IsValid)
                    valueBinding = Runtime.CreateStandardBinding(context.Ctx, valueDesc);

                if (hasMask && useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, maskDesc, out maskBinding, out IntPtr buf))
                        buffers.Add(buf);
                    else
                        useZeroCopy = false;
                }
                if (hasMask && !useZeroCopy && !maskBinding.IsValid)
                    maskBinding = Runtime.CreateStandardBinding(context.Ctx, maskDesc);

                if (!resultBinding.IsValid || !queryBinding.IsValid || !keyBinding.IsValid || !valueBinding.IsValid ||
                    (hasMask && !maskBinding.IsValid))
                {
                    Runtime.SetLastError("Failed to allocate ggml scaled_dot_product_attention tensors.");
                    return 0;
                }

                GgmlTensor* queryPerm = ggml_permute(context.Ctx, queryBinding.Tensor, 0, 2, 1, 3);
                GgmlTensor* keyPerm = ggml_permute(context.Ctx, keyBinding.Tensor, 0, 2, 1, 3);
                GgmlTensor* valuePerm = ggml_permute(context.Ctx, valueBinding.Tensor, 1, 2, 0, 3);
                valuePerm = valuePerm == null ? null : ggml_cont(context.Ctx, valuePerm);
                if (queryPerm == null || keyPerm == null || valuePerm == null)
                {
                    Runtime.SetLastError("Failed to create ggml attention permutation nodes.");
                    return 0;
                }

                GgmlTensor* scores = ggml_mul_mat(context.Ctx, keyPerm, queryPerm);
                if (scores == null)
                {
                    Runtime.SetLastError("Failed to create ggml attention score node.");
                    return 0;
                }
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);

                GgmlTensor* probs = ggml_soft_max_ext(context.Ctx, scores, hasMask ? maskBinding.Tensor : null, scale, 0.0f);
                if (probs == null)
                {
                    Runtime.SetLastError("Failed to create ggml soft_max_ext node.");
                    return 0;
                }

                GgmlTensor* contextTensor = ggml_mul_mat(context.Ctx, valuePerm, probs);
                contextTensor = contextTensor == null ? null : ggml_permute(context.Ctx, contextTensor, 0, 2, 1, 3);
                contextTensor = contextTensor == null ? null : ggml_cont(context.Ctx, contextTensor);
                if (contextTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml attention output node.");
                    return 0;
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, contextTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml attention output copy node.");
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
                    Runtime.UploadBinding(queryBinding, queryDesc.Data, queryBinding.RawBytes);
                    Runtime.UploadBinding(keyBinding, keyDesc.Data, keyBinding.RawBytes);
                    Runtime.UploadBinding(valueBinding, valueDesc.Data, valueBinding.RawBytes);
                    if (hasMask)
                    {
                        Runtime.UploadBinding(maskBinding, maskDesc.Data, maskBinding.RawBytes);
                    }
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
        // Softmax
        // --------------------------------------------------------------

        internal static int SoftmaxF32(in GgmlTensorView4D resultDesc, in GgmlTensorView4D srcDesc)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src"))
                return 0;
            if (!Runtime.SameShape(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Source tensor shape does not match result shape for ggml softmax.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml softmax Metal path.");
                return 0;
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
                Span<GgmlTensorView4D> descs = stackalloc GgmlTensorView4D[] { resultDesc, srcDesc };
                Span<Runtime.TensorBinding> bindings = stackalloc Runtime.TensorBinding[2];
                bool bound = BindAll4D(context.Ctx, descs, bindings, buffers, out bool useZeroCopy);
                if (!bound)
                    return 0;
                Runtime.TensorBinding resultBinding = bindings[0], srcBinding = bindings[1];

                GgmlTensor* contiguousSrc = ggml_cont(context.Ctx, srcBinding.Tensor);
                if (contiguousSrc == null)
                {
                    Runtime.SetLastError("Failed to create ggml contiguous softmax input.");
                    return 0;
                }

                GgmlTensor* softmaxTensor = ggml_soft_max(context.Ctx, contiguousSrc);
                if (softmaxTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml softmax node.");
                    return 0;
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, softmaxTensor, resultBinding.Tensor);
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
                    Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);
                    if (resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
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
        // AttentionSoftmaxWithSinks (in-place fused causal+SWA mask softmax
        // with optional attention sinks; scores is C# [numHeads, seqLen,
        // kvLen] == GGML [kvLen, seqLen, numHeads])
        // --------------------------------------------------------------

        internal static int AttentionSoftmaxWithSinksF32(
            in GgmlTensorView3D scoresDesc,
            IntPtr sinksData,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStartPos,
            int slidingWindow,
            float scale)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(scoresDesc, "scores"))
                return 0;

            if (scoresDesc.Dim0 != numHeads ||
                scoresDesc.Dim1 != seqLen ||
                scoresDesc.Dim2 != kvLen)
            {
                Runtime.SetLastError("scores tensor shape doesn't match (num_heads, seq_len, kv_len) in SoftmaxWithSinks.");
                return 0;
            }

            if (numHeads <= 0 || seqLen <= 0 || kvLen <= 0)
            {
                Runtime.SetLastError("Invalid (num_heads, seq_len, kv_len) for SoftmaxWithSinks.");
                return 0;
            }

            if (!Runtime.CanMapStandardView(scoresDesc))
            {
                Runtime.SetLastError("scores layout is not supported by the SoftmaxWithSinks Metal path.");
                return 0;
            }

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(4 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context for SoftmaxWithSinks.");
                return 0;
            }
            using (context)
            {
                IntPtr ctx = context.Ctx;

                // Bind scores zero-copy (in-place).
                bool useZeroCopy = true;
                Runtime.TensorBinding scoresBinding;
                if (Runtime.CreateBindingFromHostPtr(ctx, scoresDesc, out scoresBinding, out IntPtr scoresBuf))
                    buffers.Add(scoresBuf);
                else
                    useZeroCopy = false;
                if (!useZeroCopy)
                    scoresBinding = Runtime.CreateStandardBinding(ctx, scoresDesc);

                if (!scoresBinding.IsValid)
                {
                    Runtime.SetLastError("Failed to allocate scores binding in SoftmaxWithSinks.");
                    return 0;
                }

                // Build the causal+SWA mask on host as F16 [kv_len, seq_len, 1, 1].
                // ggml_soft_max_ext broadcasts the mask across the head dimension.
                GgmlTensor* maskTensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, seqLen, 1, 1);
                if (maskTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate mask tensor in SoftmaxWithSinks.");
                    return 0;
                }

                ushort[] maskData = new ushort[(long)kvLen * seqLen];
                FillCausalSwaMask(maskData, kvLen, seqLen, maskStartPos, slidingWindow);

                // Optional sinks tensor [num_heads]. ggml_soft_max_add_sinks
                // treats it as an extra column in the per-row softmax denominator.
                GgmlTensor* sinksTensor = null;
                if (sinksData != IntPtr.Zero)
                {
                    sinksTensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, numHeads);
                    if (sinksTensor == null)
                    {
                        Runtime.SetLastError("Failed to allocate sinks tensor in SoftmaxWithSinks.");
                        return 0;
                    }
                }

                // soft_max_ext requires a contiguous source tensor; copy first
                // because the host_ptr binding may be a view with non-trivial
                // strides.
                GgmlTensor* scoresInput = ggml_cont(ctx, scoresBinding.Tensor);
                GgmlTensor* sm = ggml_soft_max_ext(ctx, scoresInput, maskTensor, scale, 0.0f);
                if (sm == null)
                {
                    Runtime.SetLastError("Failed to create soft_max_ext node in SoftmaxWithSinks.");
                    return 0;
                }
                if (sinksTensor != null)
                    ggml_soft_max_add_sinks(sm, sinksTensor);

                GgmlTensor* output = ggml_cpy(ctx, sm, scoresBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create output cpy node in SoftmaxWithSinks.");
                    return 0;
                }
                ggml_set_output(output);

                IntPtr graph = ggml_new_graph(ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create graph in SoftmaxWithSinks.");
                    return 0;
                }
                ggml_build_forward_expand(graph, output);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate backend buffer in SoftmaxWithSinks.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!useZeroCopy)
                    Runtime.UploadBinding(scoresBinding, scoresDesc.Data, scoresBinding.RawBytes);

                // Drain pending async work so the upcoming CPU memcpys are safe
                // (sinks_data is a host array that a previous async op may
                // still be writing to).
                GgmlNative.HostReadBarrier();

                fixed (ushort* maskPtr = maskData)
                {
                    ggml_backend_tensor_set(maskTensor, (IntPtr)maskPtr, 0, (nuint)maskData.Length * sizeof(ushort));
                }
                if (sinksTensor != null)
                    ggml_backend_tensor_set(sinksTensor, sinksData, 0, (nuint)numHeads * sizeof(float));

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("graph compute failed in SoftmaxWithSinks.");
                    return 0;
                }

                Runtime.FinalizeCompute(useZeroCopy, scoresBinding, scoresDesc.Data, scoresBinding.RawBytes);

                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // SoftmaxGrad
        // --------------------------------------------------------------

        internal static int SoftmaxGradF32(
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D adjDesc,
            in GgmlTensorView4D valDesc,
            bool addGrad)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(adjDesc, "adj") || !Runtime.ValidateDesc(valDesc, "val"))
                return 0;
            if (!Runtime.SameShape(resultDesc, adjDesc) || !Runtime.SameShape(resultDesc, valDesc))
            {
                Runtime.SetLastError("Tensor shape mismatch passed to ggml softmaxgrad.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(adjDesc) || !Runtime.CanMapStandardView(valDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml softmaxgrad Metal path.");
                return 0;
            }

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(2 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Span<GgmlTensorView4D> descs = stackalloc GgmlTensorView4D[] { resultDesc, adjDesc, valDesc };
                Span<Runtime.TensorBinding> bindings = stackalloc Runtime.TensorBinding[3];
                bool bound = BindAll4D(context.Ctx, descs, bindings, buffers, out bool useZeroCopy);
                if (!bound)
                    return 0;
                Runtime.TensorBinding resultBinding = bindings[0], adjBinding = bindings[1], valBinding = bindings[2];

                GgmlTensor* contiguousAdj = ggml_cont(context.Ctx, adjBinding.Tensor);
                GgmlTensor* contiguousVal = ggml_cont(context.Ctx, valBinding.Tensor);
                if (contiguousAdj == null || contiguousVal == null)
                {
                    Runtime.SetLastError("Failed to create ggml contiguous softmaxgrad inputs.");
                    return 0;
                }

                GgmlTensor* gradTensor = ggml_soft_max_ext_back(context.Ctx, contiguousAdj, contiguousVal, 1.0f, 0.0f);
                if (!BackendSupportsOp(gradTensor))
                {
                    GgmlTensor* weightedAdj = ggml_mul(context.Ctx, contiguousVal, contiguousAdj);
                    if (weightedAdj == null)
                    {
                        Runtime.SetLastError("Failed to create ggml softmaxgrad mul node.");
                        return 0;
                    }

                    GgmlTensor* rowSum = ggml_sum_rows(context.Ctx, weightedAdj);
                    if (rowSum == null)
                    {
                        Runtime.SetLastError("Failed to create ggml softmaxgrad sum_rows node.");
                        return 0;
                    }

                    GgmlTensor* centeredAdj = ggml_sub(context.Ctx, contiguousAdj, rowSum);
                    if (centeredAdj == null)
                    {
                        Runtime.SetLastError("Failed to create ggml softmaxgrad subtract node.");
                        return 0;
                    }

                    gradTensor = ggml_mul(context.Ctx, contiguousVal, centeredAdj);
                }

                if (gradTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml softmaxgrad output node.");
                    return 0;
                }

                if (addGrad)
                {
                    GgmlTensor* contiguousResult = ggml_cont(context.Ctx, resultBinding.Tensor);
                    if (contiguousResult == null)
                    {
                        Runtime.SetLastError("Failed to create ggml contiguous softmaxgrad accumulation input.");
                        return 0;
                    }

                    gradTensor = ggml_add(context.Ctx, gradTensor, contiguousResult);
                    if (gradTensor == null)
                    {
                        Runtime.SetLastError("Failed to create ggml softmaxgrad accumulation node.");
                        return 0;
                    }
                }

                GgmlTensor* outputTensor = ggml_cpy(context.Ctx, gradTensor, resultBinding.Tensor);
                if (outputTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml softmaxgrad output copy node.");
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
                    Runtime.UploadBinding(adjBinding, adjDesc.Data, adjBinding.RawBytes);
                    Runtime.UploadBinding(valBinding, valDesc.Data, valBinding.RawBytes);
                    if (addGrad || resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
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

        // ==================================================================
        // Fused prefill attention (session-cached / flash / materialized),
        // mirroring the native session cache in ggml_ops_norm_attn.cpp.
        // ==================================================================

        // TS_PREFILL_ATTN_CACHE=0 disables the session cache;
        // TS_PREFILL_ATTN_FLASH=0 reverts large-kvLen calls to the
        // materialized graph. Both are read once, like the native
        // function-local statics.
        private static readonly bool s_prefillAttnCacheEnabled =
            Environment.GetEnvironmentVariable("TS_PREFILL_ATTN_CACHE") != "0";
        private static readonly bool s_prefillAttnFlashEnabled =
            Environment.GetEnvironmentVariable("TS_PREFILL_ATTN_FLASH") != "0";

        private static uint PrefillFloatBits(float v) => unchecked((uint)BitConverter.SingleToInt32Bits(v));

        private static int PrefillKvBucket(int kvLen)
        {
            if (kvLen <= 64)
                return 64;
            int v = kvLen - 1;
            v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
            return v + 1;
        }

        // Only cache shapes whose retained per-session buffer stays modest
        // (see the native comment for the full rationale).
        private static bool PrefillShouldCache(int kvBucket, int seqLen, int numHeads)
        {
            long scoresBytes = (long)kvBucket * seqLen * numHeads * 4;
            return scoresBytes <= (192L << 20); // 192 MiB per scores tensor
        }

        private sealed class PrefillAttnSession
        {
            public bool Valid;
            public int NumQ;
            public int KvBucket;
            public int NumHeads;
            public int NumKvHeads;
            public int HeadDim;
            public uint ScaleBits;
            public bool KvF16;

            public IntPtr CtxMem;
            public IntPtr Ctx;
            public IntPtr Buffer;

            public GgmlTensor* QIn;
            public GgmlTensor* KIn;
            public GgmlTensor* VIn;
            public GgmlTensor* Mask;
            public GgmlTensor* Result;
            public IntPtr Graph;

            public ulong Lru;
            public int KvZeroCoveredFrom; // max kvLen written into the K/V padding region

            public void Destroy()
            {
                if (Buffer != IntPtr.Zero) { ggml_backend_buffer_free(Buffer); Buffer = IntPtr.Zero; }
                if (Ctx != IntPtr.Zero) { ggml_free(Ctx); Ctx = IntPtr.Zero; }
                if (CtxMem != IntPtr.Zero) { Marshal.FreeHGlobal(CtxMem); CtxMem = IntPtr.Zero; }
                QIn = KIn = VIn = Mask = Result = null;
                Graph = IntPtr.Zero;
                Valid = false;
                KvZeroCoveredFrom = 0;
            }
        }

        private const int PrefillAttnCacheSize = 16;
        [ThreadStatic] private static PrefillAttnSession[] t_prefillAttnCache;
        [ThreadStatic] private static ulong t_prefillAttnLru;
        [ThreadStatic] private static ushort[] t_prefillMaskScratch;
        [ThreadStatic] private static byte[] t_prefillZeroScratch;

        private static PrefillAttnSession[] PrefillAttnCache
        {
            get
            {
                PrefillAttnSession[] cache = t_prefillAttnCache;
                if (cache == null)
                {
                    cache = new PrefillAttnSession[PrefillAttnCacheSize];
                    for (int i = 0; i < cache.Length; ++i)
                        cache[i] = new PrefillAttnSession();
                    t_prefillAttnCache = cache;
                }
                return cache;
            }
        }

        // Frees the calling thread's cached prefill-attention sessions.
        // Called from GgmlNative.Shutdown (managed branch) so the sessions'
        // backend buffers are released before the backend is torn down —
        // mirrors free_prefill_attn_sessions in ggml_ops_norm_attn.cpp.
        internal static void FreePrefillSessions()
        {
            PrefillAttnSession[] cache = t_prefillAttnCache;
            if (cache == null)
                return;
            foreach (PrefillAttnSession s in cache)
                s?.Destroy();
        }

        private static PrefillAttnSession FindPrefillSession(
            int numQ, int kvBucket, int numHeads, int numKvHeads, int headDim,
            uint scaleBits, bool kvF16)
        {
            foreach (PrefillAttnSession s in PrefillAttnCache)
            {
                if (s.Valid && s.NumQ == numQ && s.KvBucket == kvBucket &&
                    s.NumHeads == numHeads && s.NumKvHeads == numKvHeads &&
                    s.HeadDim == headDim && s.ScaleBits == scaleBits && s.KvF16 == kvF16)
                {
                    s.Lru = ++t_prefillAttnLru;
                    return s;
                }
            }
            return null;
        }

        private static PrefillAttnSession AcquirePrefillSessionSlot()
        {
            PrefillAttnSession[] cache = PrefillAttnCache;
            foreach (PrefillAttnSession s in cache)
            {
                if (!s.Valid)
                {
                    s.Lru = ++t_prefillAttnLru;
                    return s;
                }
            }
            PrefillAttnSession victim = cache[0];
            foreach (PrefillAttnSession s in cache)
            {
                if (s.Lru < victim.Lru)
                    victim = s;
            }
            victim.Destroy();
            victim.Lru = ++t_prefillAttnLru;
            return victim;
        }

        private static bool BuildPrefillSession(
            PrefillAttnSession s,
            int numQ, int kvBucket, int numHeads, int numKvHeads, int headDim,
            float scale, bool kvF16)
        {
            const int kCtxMemSize = 1024 * 1024;
            s.CtxMem = Marshal.AllocHGlobal(kCtxMemSize);
            if (s.CtxMem == IntPtr.Zero)
            {
                Runtime.SetLastError("prefill-attn session: alloc ctx mem failed.");
                return false;
            }

            var initParams = new ggml_init_params
            {
                mem_size = kCtxMemSize,
                mem_buffer = s.CtxMem,
                no_alloc = true,
            };
            s.Ctx = ggml_init(initParams);
            if (s.Ctx == IntPtr.Zero)
            {
                Runtime.SetLastError("prefill-attn session: ggml_init failed.");
                Marshal.FreeHGlobal(s.CtxMem);
                s.CtxMem = IntPtr.Zero;
                return false;
            }

            int kvType = kvF16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
            // Head-first: GGML [headDim, seq, heads].
            s.QIn = ggml_new_tensor_3d(s.Ctx, GGML_TYPE_F32, headDim, numQ, numHeads);
            s.KIn = ggml_new_tensor_3d(s.Ctx, kvType, headDim, kvBucket, numKvHeads);
            s.VIn = ggml_new_tensor_3d(s.Ctx, kvType, headDim, kvBucket, numKvHeads);
            s.Mask = ggml_new_tensor_4d(s.Ctx, GGML_TYPE_F16, kvBucket, numQ, 1, 1);
            s.Result = ggml_new_tensor_2d(s.Ctx, GGML_TYPE_F32, (long)numHeads * headDim, numQ);
            if (s.QIn == null || s.KIn == null || s.VIn == null || s.Mask == null || s.Result == null)
            {
                Runtime.SetLastError("prefill-attn session: tensor alloc failed.");
                ggml_free(s.Ctx);
                s.Ctx = IntPtr.Zero;
                Marshal.FreeHGlobal(s.CtxMem);
                s.CtxMem = IntPtr.Zero;
                return false;
            }

            // scores = mul_mat(K, Q) with GQA broadcast; F32 accumulation.
            GgmlTensor* scores = ggml_mul_mat(s.Ctx, s.KIn, s.QIn);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            GgmlTensor* probs = ggml_soft_max_ext(s.Ctx, scores, s.Mask, scale, 0.0f);
            GgmlTensor* vPerm = ggml_cont(s.Ctx, ggml_permute(s.Ctx, s.VIn, 1, 0, 2, 3));
            GgmlTensor* attnOut = ggml_mul_mat(s.Ctx, vPerm, probs);
            GgmlTensor* attnPerm = ggml_permute(s.Ctx, attnOut, 0, 2, 1, 3);
            GgmlTensor* attnCont = ggml_cont(s.Ctx, attnPerm);
            GgmlTensor* attnFlat = ggml_reshape_2d(s.Ctx, attnCont, (long)numHeads * headDim, numQ);
            GgmlTensor* output = ggml_cpy(s.Ctx, attnFlat, s.Result);
            ggml_set_output(output);

            s.Graph = ggml_new_graph(s.Ctx);
            ggml_build_forward_expand(s.Graph, output);

            s.Buffer = ggml_backend_alloc_ctx_tensors(s.Ctx, Runtime.Backend);
            if (s.Buffer == IntPtr.Zero)
            {
                Runtime.SetLastError("prefill-attn session: backend buffer alloc failed.");
                ggml_free(s.Ctx);
                s.Ctx = IntPtr.Zero;
                Marshal.FreeHGlobal(s.CtxMem);
                s.CtxMem = IntPtr.Zero;
                return false;
            }

            // Backend buffers are NOT zero-initialised; the KvZeroCoveredFrom
            // tracking needs the K/V padding [kvLen, kvBucket) to start zero
            // or NaN garbage leaks through the masked softmax (same fix as
            // build_prefill_session in ggml_ops_norm_attn.cpp).
            GgmlApi.ggml_backend_buffer_clear(s.Buffer, 0);

            s.NumQ = numQ; s.KvBucket = kvBucket; s.NumHeads = numHeads;
            s.NumKvHeads = numKvHeads; s.HeadDim = headDim;
            s.ScaleBits = PrefillFloatBits(scale); s.KvF16 = kvF16;
            s.KvZeroCoveredFrom = 0; // buffer explicitly cleared above
            s.Valid = true;
            return true;
        }

        // Cached head-first fused prefill attention. qData is F32 [numHeads,
        // seqLen, headDim]; k/vData are F32 or F16 [numKVHeads, kvStride,
        // headDim] (the leading kvLen rows are read per head). outData is F32
        // [seqLen, numHeads*headDim].
        private static int FusedPrefillAttnCached(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen, int kvStride,
            int maskStartPos, int slidingWindow, float scale, bool kvF16)
        {
            if (!Runtime.EnsureBackend())
                return 0;

            int kvBucket = PrefillKvBucket(kvLen);
            uint scaleBits = PrefillFloatBits(scale);

            PrefillAttnSession sess = FindPrefillSession(
                seqLen, kvBucket, numHeads, numKvHeads, headDim, scaleBits, kvF16);
            if (sess == null)
            {
                PrefillAttnSession slot = AcquirePrefillSessionSlot();
                if (!BuildPrefillSession(slot, seqLen, kvBucket, numHeads, numKvHeads,
                                         headDim, scale, kvF16))
                    return 0;
                sess = slot;
            }

            nuint kvElem = kvF16 ? (nuint)sizeof(ushort) : sizeof(float);
            nuint qBytes = (nuint)numHeads * (nuint)seqLen * (nuint)headDim * sizeof(float);

            // The previous graph_compute on this cached session may still be
            // reading q_in/k_in/v_in; drain before overwriting them.
            GgmlNative.HostReadBarrier();

            ggml_backend_tensor_set(sess.QIn, qData, 0, qBytes);

            // Upload K/V: leading kv_len rows per head into the bucket-sized tensor.
            nuint srcHeadElems = (nuint)kvStride * (nuint)headDim;
            nuint dstHeadElems = (nuint)kvBucket * (nuint)headDim;
            nuint rowBytes = (nuint)kvLen * (nuint)headDim * kvElem;
            byte* kb = (byte*)kData;
            byte* vb = (byte*)vData;
            for (int h = 0; h < numKvHeads; ++h)
            {
                nuint srcOff = (nuint)h * srcHeadElems * kvElem;
                nuint dstOff = (nuint)h * dstHeadElems * kvElem;
                ggml_backend_tensor_set(sess.KIn, (IntPtr)(kb + srcOff), dstOff, rowBytes);
                ggml_backend_tensor_set(sess.VIn, (IntPtr)(vb + srcOff), dstOff, rowBytes);
            }

            // Zero the K/V padding [kv_len, covered) per head only when this
            // call is shorter than a previous one (otherwise the tail is still
            // zero from the freshly-allocated buffer / never written).
            if (kvLen < sess.KvZeroCoveredFrom)
            {
                int zeroRows = sess.KvZeroCoveredFrom - kvLen;
                nuint zeroBytes = (nuint)zeroRows * (nuint)headDim * kvElem;
                if (t_prefillZeroScratch == null || (nuint)t_prefillZeroScratch.Length < zeroBytes)
                    t_prefillZeroScratch = new byte[zeroBytes];
                fixed (byte* zeroPtr = t_prefillZeroScratch)
                {
                    for (int h = 0; h < numKvHeads; ++h)
                    {
                        nuint padOff = ((nuint)h * dstHeadElems + (nuint)kvLen * (nuint)headDim) * kvElem;
                        ggml_backend_tensor_set(sess.KIn, (IntPtr)zeroPtr, padOff, zeroBytes);
                        ggml_backend_tensor_set(sess.VIn, (IntPtr)zeroPtr, padOff, zeroBytes);
                    }
                }
            }
            if (kvLen > sess.KvZeroCoveredFrom)
                sess.KvZeroCoveredFrom = kvLen;

            // Build + upload the causal (+ optional sliding window) mask over
            // the full bucket. threshold < kv_len, so the [kv_len, bucket)
            // tail is auto-masked.
            int maskCount = kvBucket * seqLen;
            if (t_prefillMaskScratch == null || t_prefillMaskScratch.Length < maskCount)
                t_prefillMaskScratch = new ushort[maskCount];
            FillCausalSwaMask(t_prefillMaskScratch, kvBucket, seqLen, maskStartPos, slidingWindow);
            fixed (ushort* maskPtr = t_prefillMaskScratch)
            {
                ggml_backend_tensor_set(sess.Mask, (IntPtr)maskPtr, 0, (nuint)maskCount * sizeof(ushort));
            }

            if (ggml_backend_graph_compute(Runtime.Backend, sess.Graph) != GGML_STATUS_SUCCESS)
            {
                Runtime.SetLastError("ggml graph compute failed for cached fused prefill attention.");
                return 0;
            }

            FinalizeComputeWithDownload(sess.Result, outData, qBytes);
            Runtime.ClearLastError();
            return 1;
        }

        // Flash-attention prefill: same contract as the materialized paths
        // below but runs ggml_flash_attn_ext. Returns 1 on success, 0 on hard
        // failure, -1 when the backend has no flash kernel for this shape
        // (caller falls back to the materialized graph).
        private static int FusedPrefillAttnFlash(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen, int kvStride,
            int maskStartPos, int slidingWindow, float scale, bool kvF16)
        {
            if (!Runtime.EnsureBackend())
                return 0;

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(4 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context for flash prefill attention.");
                return 0;
            }
            using (context)
            {
                IntPtr ctx = context.Ctx;

                int kvType = kvF16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

                // flash_attn_ext wants q=[headDim, seqLen, numHeads],
                // k/v=[headDim, kvLen, numKVHeads] — exactly the head-first
                // layout, so no permutes are needed.
                GgmlTensor* qIn = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, seqLen, numHeads);
                GgmlTensor* kIn = ggml_new_tensor_3d(ctx, kvType, headDim, kvLen, numKvHeads);
                GgmlTensor* vIn = ggml_new_tensor_3d(ctx, kvType, headDim, kvLen, numKvHeads);
                GgmlTensor* maskTensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kvLen, seqLen);
                GgmlTensor* attnResult = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (long)numHeads * headDim, seqLen);
                if (qIn == null || kIn == null || vIn == null || maskTensor == null || attnResult == null)
                {
                    Runtime.SetLastError("Failed to create ggml tensors for flash prefill attention.");
                    return 0;
                }

                GgmlTensor* attnOut = ggml_flash_attn_ext(ctx, qIn, kIn, vIn, maskTensor, scale, 0.0f, 0.0f);
                if (attnOut == null)
                {
                    Runtime.SetLastError("Failed flash_attn_ext node.");
                    return 0;
                }
                ggml_flash_attn_ext_set_prec(attnOut, GGML_PREC_F32);
                if (!BackendSupportsOp(attnOut))
                    return -1; // no flash kernel for this geometry; let the caller fall back

                // flash_attn_ext returns [headDim, numHeads, seqLen, 1] —
                // contiguous, exactly the flat [numHeads*headDim, seqLen] output.
                GgmlTensor* attnFlat = ggml_reshape_2d(ctx, attnOut, (long)numHeads * headDim, seqLen);
                GgmlTensor* output = ggml_cpy(ctx, attnFlat, attnResult);
                if (output == null)
                {
                    Runtime.SetLastError("Failed flash output cpy node.");
                    return 0;
                }
                ggml_set_output(output);

                IntPtr graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, output);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate backend buffer for flash prefill attention.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                GgmlNative.HostReadBarrier();
                ggml_backend_tensor_set(qIn, qData, 0,
                    (nuint)numHeads * (nuint)seqLen * (nuint)headDim * sizeof(float));

                // Upload the leading kvLen rows of each head from the
                // [numKVHeads, kvStride, headDim] cache. Contiguous single
                // upload when the cache is exactly sized.
                nuint kvElem = kvF16 ? (nuint)sizeof(ushort) : sizeof(float);
                nuint dstHeadElems = (nuint)kvLen * (nuint)headDim;
                if (kvStride == kvLen)
                {
                    nuint bytes = (nuint)numKvHeads * dstHeadElems * kvElem;
                    ggml_backend_tensor_set(kIn, kData, 0, bytes);
                    ggml_backend_tensor_set(vIn, vData, 0, bytes);
                }
                else
                {
                    byte* kb = (byte*)kData;
                    byte* vb = (byte*)vData;
                    nuint srcHeadElems = (nuint)kvStride * (nuint)headDim;
                    nuint headBytes = dstHeadElems * kvElem;
                    for (int h = 0; h < numKvHeads; ++h)
                    {
                        nuint srcOff = (nuint)h * srcHeadElems * kvElem;
                        nuint dstOff = (nuint)h * dstHeadElems * kvElem;
                        ggml_backend_tensor_set(kIn, (IntPtr)(kb + srcOff), dstOff, headBytes);
                        ggml_backend_tensor_set(vIn, (IntPtr)(vb + srcOff), dstOff, headBytes);
                    }
                }

                // Causal (+ optional sliding-window) additive mask.
                int maskCount = kvLen * seqLen;
                if (t_prefillMaskScratch == null || t_prefillMaskScratch.Length < maskCount)
                    t_prefillMaskScratch = new ushort[maskCount];
                FillCausalSwaMask(t_prefillMaskScratch, kvLen, seqLen, maskStartPos, slidingWindow);
                fixed (ushort* maskPtr = t_prefillMaskScratch)
                {
                    ggml_backend_tensor_set(maskTensor, (IntPtr)maskPtr, 0, (nuint)maskCount * sizeof(ushort));
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml graph compute failed for flash prefill attention.");
                    return 0;
                }

                FinalizeComputeWithDownload(attnResult, outData,
                    (nuint)numHeads * (nuint)seqLen * (nuint)headDim * sizeof(float));
                Runtime.ClearLastError();
                return 1;
            }
        }

        // Fused prefill attention: Q*K^T -> causal mask -> softmax -> *V as
        // one GGML graph. inputFormat: 0 = head-first [numHeads, seqLen,
        // headDim], 1 = flat [seqLen, numHeads*headDim]. Output is always the
        // flat layout.
        internal static int FusedPrefillAttentionF32(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int maskStartPos,
            int slidingWindow,
            float scale,
            int inputFormat)
        {
            if (!Runtime.EnsureBackend())
            {
                return 0;
            }

            // Session-cached fast path (head-first only): reuse the graph +
            // backend buffer across the per-layer/per-chunk prefill calls.
            if (inputFormat == 0 && s_prefillAttnCacheEnabled
                && PrefillShouldCache(PrefillKvBucket(kvLen), seqLen, numHeads))
            {
                return FusedPrefillAttnCached(
                    qData, kData, vData, outData,
                    numHeads, numKVHeads, headDim, seqLen, kvLen, /*kvStride*/ kvLen,
                    maskStartPos, slidingWindow, scale, /*kvF16*/ false);
            }

            // Large-kvLen head-first path: flash attention streams K/V instead
            // of materializing the O(N^2) scores+softmax. The flat
            // (inputFormat==1) layout stays on the materialized graph below.
            if (inputFormat == 0 && s_prefillAttnFlashEnabled)
            {
                int fr = FusedPrefillAttnFlash(
                    qData, kData, vData, outData,
                    numHeads, numKVHeads, headDim, seqLen, kvLen, /*kvStride*/ kvLen,
                    maskStartPos, slidingWindow, scale, /*kvF16*/ false);
                if (fr >= 0)
                    return fr;
            }

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(4 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context for fused prefill attention.");
                return 0;
            }
            using (context)
            {
                IntPtr ctx = context.Ctx;
                int qSize = numHeads * seqLen * headDim;
                int kvSize = numKVHeads * kvLen * headDim;

                // Create input tensors matching the C# memory layout.
                GgmlTensor* qIn;
                GgmlTensor* kIn;
                GgmlTensor* vIn;
                if (inputFormat == 1)
                {
                    // Flat layout: C# [seqLen, numHeads*headDim] == GGML [numHeads*headDim, seqLen]
                    qIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (long)numHeads * headDim, seqLen);
                    kIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (long)numKVHeads * headDim, kvLen);
                    vIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (long)numKVHeads * headDim, kvLen);
                }
                else
                {
                    // Head-first: C# [numHeads, seqLen, headDim] == GGML [headDim, seqLen, numHeads]
                    qIn = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, seqLen, numHeads);
                    kIn = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, kvLen, numKVHeads);
                    vIn = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, kvLen, numKVHeads);
                }
                // Output in flat layout [numHeads*headDim, seqLen] (GGML).
                GgmlTensor* attnResult = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (long)numHeads * headDim, seqLen);

                // Build causal + optional sliding window mask in FP16: [kvLen, seqLen]
                GgmlTensor* maskTensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, seqLen, 1, 1);
                ushort[] maskData = new ushort[(long)kvLen * seqLen];
                FillCausalSwaMask(maskData, kvLen, seqLen, maskStartPos, slidingWindow);

                if (qIn == null || kIn == null || vIn == null || attnResult == null || maskTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml tensors for fused prefill attention.");
                    return 0;
                }

                // For flat input format, reshape [dim, seqLen] -> [headDim,
                // numHeads, seqLen] then permute to [headDim, seqLen, numHeads].
                GgmlTensor* qAttn = qIn;
                GgmlTensor* kAttn = kIn;
                GgmlTensor* vAttn = vIn;
                if (inputFormat == 1)
                {
                    GgmlTensor* q3d = ggml_reshape_3d(ctx, qIn, headDim, numHeads, seqLen);
                    qAttn = ggml_permute(ctx, q3d, 0, 2, 1, 3);
                    qAttn = ggml_cont(ctx, qAttn);
                    GgmlTensor* k3d = ggml_reshape_3d(ctx, kIn, headDim, numKVHeads, kvLen);
                    kAttn = ggml_permute(ctx, k3d, 0, 2, 1, 3);
                    kAttn = ggml_cont(ctx, kAttn);
                    GgmlTensor* v3d = ggml_reshape_3d(ctx, vIn, headDim, numKVHeads, kvLen);
                    vAttn = ggml_permute(ctx, v3d, 0, 2, 1, 3);
                    vAttn = ggml_cont(ctx, vAttn);
                }

                // mul_mat(K, Q) with GQA broadcast.
                GgmlTensor* scores = ggml_mul_mat(ctx, kAttn, qAttn);
                if (scores == null)
                {
                    Runtime.SetLastError("Failed to create Q*K^T matmul node.");
                    return 0;
                }
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);

                // Softmax with mask: softmax(scores * scale + mask)
                GgmlTensor* probs = ggml_soft_max_ext(ctx, scores, maskTensor, scale, 0.0f);
                if (probs == null)
                {
                    Runtime.SetLastError("Failed to create softmax node.");
                    return 0;
                }

                // V permute for value matmul: [headDim, kvLen, numKVHeads] -> [kvLen, headDim, numKVHeads]
                GgmlTensor* vPerm = ggml_permute(ctx, vAttn, 1, 0, 2, 3);
                vPerm = ggml_cont(ctx, vPerm);

                // attn = probs * V_perm -> [headDim, seqLen, numHeads]
                GgmlTensor* attnOut = ggml_mul_mat(ctx, vPerm, probs);
                if (attnOut == null)
                {
                    Runtime.SetLastError("Failed to create scores*V matmul node.");
                    return 0;
                }

                // Permute to flat layout [headDim*numHeads, seqLen] (GGML).
                GgmlTensor* attnPerm = ggml_permute(ctx, attnOut, 0, 2, 1, 3);
                GgmlTensor* attnCont = ggml_cont(ctx, attnPerm);
                GgmlTensor* attnFlat = ggml_reshape_2d(ctx, attnCont, (long)numHeads * headDim, seqLen);

                GgmlTensor* output = ggml_cpy(ctx, attnFlat, attnResult);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create output copy node.");
                    return 0;
                }
                ggml_set_output(output);

                IntPtr graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, output);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate backend buffer for fused prefill attention.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                // Upload input data. q/k/v may have pending GPU writes from a
                // previous async op; drain first.
                GgmlNative.HostReadBarrier();
                ggml_backend_tensor_set(qIn, qData, 0, (nuint)qSize * sizeof(float));
                ggml_backend_tensor_set(kIn, kData, 0, (nuint)kvSize * sizeof(float));
                ggml_backend_tensor_set(vIn, vData, 0, (nuint)kvSize * sizeof(float));
                fixed (ushort* maskPtr = maskData)
                {
                    ggml_backend_tensor_set(maskTensor, (IntPtr)maskPtr, 0, (nuint)maskData.Length * sizeof(ushort));
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml graph compute failed for fused prefill attention.");
                    return 0;
                }

                FinalizeComputeWithDownload(attnResult, outData, (nuint)qSize * sizeof(float));

                Runtime.ClearLastError();
                return 1;
            }
        }

        // Fused prefill attention reading K/V straight out of an F16 cache
        // ([numKVHeads, kvCacheLen, headDim] head-first; the leading kvLen
        // rows of each head are read). Q stays F32 head-first; output is flat
        // F32 [seqLen, numHeads*headDim].
        internal static int FusedPrefillAttentionF16KV(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int kvCacheLen,
            int maskStartPos,
            int slidingWindow,
            float scale)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || seqLen <= 0 || kvLen <= 0 ||
                kvCacheLen < kvLen || numHeads % numKVHeads != 0)
            {
                Runtime.SetLastError("Invalid dimensions for fused prefill attention (F16 KV).");
                return 0;
            }

            // Session-cached fast path: reuse the graph + backend buffer across calls.
            if (s_prefillAttnCacheEnabled
                && PrefillShouldCache(PrefillKvBucket(kvLen), seqLen, numHeads))
            {
                return FusedPrefillAttnCached(
                    qData, kData, vData, outData,
                    numHeads, numKVHeads, headDim, seqLen, kvLen, /*kvStride*/ kvCacheLen,
                    maskStartPos, slidingWindow, scale, /*kvF16*/ true);
            }

            // Large-kvLen path: flash attention streams K/V instead of
            // materializing the O(N^2) scores+softmax. Falls through to the
            // materialized graph below only if the backend has no flash kernel.
            if (s_prefillAttnFlashEnabled)
            {
                int fr = FusedPrefillAttnFlash(
                    qData, kData, vData, outData,
                    numHeads, numKVHeads, headDim, seqLen, kvLen, /*kvStride*/ kvCacheLen,
                    maskStartPos, slidingWindow, scale, /*kvF16*/ true);
                if (fr >= 0)
                    return fr;
            }

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(4 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context for fused prefill attention (F16 KV).");
                return 0;
            }
            using (context)
            {
                IntPtr ctx = context.Ctx;

                // Head-first layout: C# [numHeads, seqLen, headDim] == GGML [headDim, seqLen, numHeads].
                GgmlTensor* qIn = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, seqLen, numHeads);
                GgmlTensor* kIn = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, headDim, kvLen, numKVHeads);
                GgmlTensor* vIn = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, headDim, kvLen, numKVHeads);
                GgmlTensor* attnResult = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (long)numHeads * headDim, seqLen);
                GgmlTensor* maskTensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, seqLen, 1, 1);

                if (qIn == null || kIn == null || vIn == null || attnResult == null || maskTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml tensors for fused prefill attention (F16 KV).");
                    return 0;
                }

                ushort[] maskData = new ushort[(long)kvLen * seqLen];
                FillCausalSwaMask(maskData, kvLen, seqLen, maskStartPos, slidingWindow);

                // scores = mul_mat(K_f16, Q_f32); GQA broadcast; F32 accumulation.
                GgmlTensor* scores = ggml_mul_mat(ctx, kIn, qIn);
                if (scores == null)
                {
                    Runtime.SetLastError("Failed Q*K^T (F16 KV).");
                    return 0;
                }
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);

                GgmlTensor* probs = ggml_soft_max_ext(ctx, scores, maskTensor, scale, 0.0f);
                if (probs == null)
                {
                    Runtime.SetLastError("Failed softmax (F16 KV).");
                    return 0;
                }

                // V permute [headDim, kvLen, numKVHeads] -> [kvLen, headDim, numKVHeads] (stays F16).
                GgmlTensor* vPerm = ggml_cont(ctx, ggml_permute(ctx, vIn, 1, 0, 2, 3));
                GgmlTensor* attnOut = ggml_mul_mat(ctx, vPerm, probs);
                if (attnOut == null)
                {
                    Runtime.SetLastError("Failed scores*V (F16 KV).");
                    return 0;
                }

                // [headDim, seqLen, numHeads] -> flat [headDim*numHeads, seqLen].
                GgmlTensor* attnPerm = ggml_permute(ctx, attnOut, 0, 2, 1, 3);
                GgmlTensor* attnCont = ggml_cont(ctx, attnPerm);
                GgmlTensor* attnFlat = ggml_reshape_2d(ctx, attnCont, (long)numHeads * headDim, seqLen);
                GgmlTensor* output = ggml_cpy(ctx, attnFlat, attnResult);
                if (output == null)
                {
                    Runtime.SetLastError("Failed output cpy (F16 KV).");
                    return 0;
                }
                ggml_set_output(output);

                IntPtr graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, output);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate backend buffer for fused prefill attention (F16 KV).");
                    return 0;
                }
                buffers.Add(computeBuffer);

                GgmlNative.HostReadBarrier();
                ggml_backend_tensor_set(qIn, qData, 0,
                    (nuint)numHeads * (nuint)seqLen * (nuint)headDim * sizeof(float));

                // Upload K/V F16 directly from the cache, reading the leading
                // kvLen rows of each head. Single contiguous upload when the
                // cache is exactly sized; otherwise per-head to skip the
                // unused [kvLen, kvCacheLen) tail.
                nuint dstHeadElems = (nuint)kvLen * (nuint)headDim;
                if (kvCacheLen == kvLen)
                {
                    nuint bytes = (nuint)numKVHeads * dstHeadElems * sizeof(ushort);
                    ggml_backend_tensor_set(kIn, kData, 0, bytes);
                    ggml_backend_tensor_set(vIn, vData, 0, bytes);
                }
                else
                {
                    byte* ksrc = (byte*)kData;
                    byte* vsrc = (byte*)vData;
                    nuint srcHeadElems = (nuint)kvCacheLen * (nuint)headDim;
                    nuint headBytes = dstHeadElems * sizeof(ushort);
                    for (int h = 0; h < numKVHeads; ++h)
                    {
                        nuint dstOff = (nuint)h * dstHeadElems * sizeof(ushort);
                        nuint srcOff = (nuint)h * srcHeadElems * sizeof(ushort);
                        ggml_backend_tensor_set(kIn, (IntPtr)(ksrc + srcOff), dstOff, headBytes);
                        ggml_backend_tensor_set(vIn, (IntPtr)(vsrc + srcOff), dstOff, headBytes);
                    }
                }
                fixed (ushort* maskPtr = maskData)
                {
                    ggml_backend_tensor_set(maskTensor, (IntPtr)maskPtr, 0, (nuint)maskData.Length * sizeof(ushort));
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml graph compute failed for fused prefill attention (F16 KV).");
                    return 0;
                }

                FinalizeComputeWithDownload(attnResult, outData,
                    (nuint)numHeads * (nuint)seqLen * (nuint)headDim * sizeof(float));

                Runtime.ClearLastError();
                return 1;
            }
        }
    }
}
