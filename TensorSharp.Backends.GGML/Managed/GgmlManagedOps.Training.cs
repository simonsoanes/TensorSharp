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
    // Managed port of TensorSharp.GGML.Native/ggml_ops_training.cpp
    // (TSGgml_CrossEntropyLossF32 / TSGgml_CrossEntropyLossBackwardF32 /
    // TSGgml_AdamF32). Each method mirrors the corresponding *_impl: validate
    // the descriptors, bind the caller's host buffers (zero-copy where the
    // backend allows it, staged upload otherwise), build the ggml graph,
    // compute on the shared backend, and download / read back the results
    // when they were not host-mapped. Returns 1/0 with a thread-local
    // last-error exactly like the native entry points.
    internal static unsafe partial class GgmlManagedOps
    {
        // ------------------------------------------------------------------
        // ggml bindings used by the training module that GgmlApi does not
        // declare yet. Same module ("GgmlOps") and conventions as GgmlApi.
        // ------------------------------------------------------------------

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern GgmlTensor* ggml_cross_entropy_loss(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);

        // NOTE: in the vendored ggml, `a` is the scalar loss gradient and
        // `b`/`c` are logits/labels (see ggml.c: GGML_ASSERT(ggml_is_scalar(a))).
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern GgmlTensor* ggml_cross_entropy_loss_back(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, GgmlTensor* c);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern GgmlTensor* ggml_opt_step_adamw(IntPtr ctx, GgmlTensor* a, GgmlTensor* grad, GgmlTensor* m, GgmlTensor* v, GgmlTensor* adamw_params);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void ggml_backend_tensor_memset(GgmlTensor* tensor, byte value, nuint offset, nuint size);

        // ------------------------------------------------------------------
        // Small binding helpers (mirror create_scalar_binding /
        // create_matrix_binding in ggml_ops_internal.h)
        // ------------------------------------------------------------------

        private static Runtime.TensorBinding CreateMatrixBinding(IntPtr ctx, long cols, long rows)
        {
            GgmlTensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
            return new Runtime.TensorBinding { Storage = tensor, Tensor = tensor, RawBytes = (nuint)(cols * rows * sizeof(float)) };
        }

        // Port of build_cross_entropy_label_buffer in ggml_ops_core.cpp:
        // expands the per-row target indices into a dense one-hot (optionally
        // label-smoothed) label matrix that is uploaded next to the
        // probabilities.
        private static bool BuildCrossEntropyLabelBuffer(
            out float[] labels,
            in GgmlContiguousTensor targetIndicesDesc,
            long rows,
            long cols,
            float labelSmooth)
        {
            labels = null;
            if (targetIndicesDesc.ElementCount != rows)
            {
                Runtime.SetLastError("Target index count must match the number of probability rows for ggml crossentropyloss.");
                return false;
            }

            float baseValue = labelSmooth > 0.0f
                ? (labelSmooth / (float)cols)
                : 0.0f;
            float targetValue = 1.0f - labelSmooth + (labelSmooth / (float)cols);

            labels = new float[rows * cols];
            Array.Fill(labels, baseValue);

            int[] targetIndices = new int[rows];
            if (!ReadI32Values(targetIndices, targetIndicesDesc, "targetIndices"))
                return false;

            for (long row = 0; row < rows; ++row)
            {
                long targetIndex = targetIndices[row];
                if (targetIndex < 0 || targetIndex >= cols)
                {
                    Runtime.SetLastError("Target index out of range for ggml crossentropyloss.");
                    return false;
                }
                labels[row * cols + targetIndex] = targetValue;
            }

            return true;
        }

        // --------------------------------------------------------------
        // CrossEntropyLoss (forward, scalar loss read back to the caller)
        // --------------------------------------------------------------

        internal static int CrossEntropyLossF32(
            out float lossValue,
            in GgmlTensorView4D probsDesc,
            in GgmlContiguousTensor targetIndicesDesc,
            float smooth,
            float labelSmooth)
        {
            // The native impl null-checks the loss_value pointer here; a
            // managed out-parameter cannot be null, so only assignment
            // remains.
            lossValue = 0f;
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(probsDesc, "probs") || !Runtime.ValidateDesc(targetIndicesDesc, "targetIndices"))
                return 0;
            if (!Runtime.CanMapStandardView(probsDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml crossentropyloss Metal path.");
                return 0;
            }
            if (labelSmooth < 0.0f || labelSmooth > 1.0f)
            {
                Runtime.SetLastError("labelSmooth must be in [0, 1] for ggml crossentropyloss.");
                return 0;
            }

            long rows = Runtime.FlatRowCount(probsDesc);
            long cols = probsDesc.Ne0;
            if (targetIndicesDesc.ElementCount != rows)
            {
                Runtime.SetLastError("Target index count must match the number of probability rows for ggml crossentropyloss.");
                return 0;
            }

            bool probsZeroCopy = Runtime.CanMapStandardView(probsDesc);
            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(4 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding probsBinding = default;
                if (probsZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, probsDesc, out probsBinding, out IntPtr probsBuf))
                        buffers.Add(probsBuf);
                    else
                        probsZeroCopy = false;
                }
                if (!probsZeroCopy)
                    probsBinding = Runtime.CreateStandardBinding(context.Ctx, probsDesc);
                Runtime.TensorBinding labelsBinding = CreateMatrixBinding(context.Ctx, cols, rows);

                if (!probsBinding.IsValid || !labelsBinding.IsValid)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors for crossentropyloss.");
                    return 0;
                }

                GgmlTensor* contiguousProbs = ggml_cont(context.Ctx, probsBinding.Tensor);
                GgmlTensor* flatProbs = contiguousProbs == null ? null : ggml_reshape_2d(context.Ctx, contiguousProbs, cols, rows);
                GgmlTensor* logitsTensor = flatProbs == null ? null : ggml_log(context.Ctx, flatProbs);
                if (contiguousProbs == null || flatProbs == null || logitsTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml crossentropyloss logits tensor.");
                    return 0;
                }

                GgmlTensor* lossTensor = ggml_cross_entropy_loss(context.Ctx, logitsTensor, labelsBinding.Tensor);
                if (lossTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml_cross_entropy_loss node.");
                    return 0;
                }

                if (!ggml_backend_supports_op(Runtime.Backend, lossTensor))
                {
                    Runtime.SetLastError("ggml_cross_entropy_loss is not supported by the active backend.");
                    return 0;
                }

                ggml_set_output(lossTensor);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }
                ggml_build_forward_expand(graph, lossTensor);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!BuildCrossEntropyLabelBuffer(out float[] labels, targetIndicesDesc, rows, cols, labelSmooth))
                    return 0;

                if (!probsZeroCopy)
                    Runtime.UploadBinding(probsBinding, probsDesc.Data, probsBinding.RawBytes);
                fixed (float* labelsPtr = labels)
                {
                    Runtime.UploadBinding(labelsBinding, (IntPtr)labelsPtr, labelsBinding.RawBytes);
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                ggml_backend_synchronize(Runtime.Backend);
                float loss;
                ggml_backend_tensor_get(lossTensor, (IntPtr)(&loss), 0, sizeof(float));
                lossValue = loss;

                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // CrossEntropyLossBackward (gradient w.r.t. the probabilities,
        // optionally accumulated into the caller's grad buffer)
        // --------------------------------------------------------------

        internal static int CrossEntropyLossBackwardF32(
            in GgmlTensorView4D gradDesc,
            in GgmlTensorView4D probsDesc,
            in GgmlContiguousTensor targetIndicesDesc,
            float lossGradient,
            float smooth,
            float labelSmooth,
            bool addGrad)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(gradDesc, "grad") || !Runtime.ValidateDesc(probsDesc, "probs") || !Runtime.ValidateDesc(targetIndicesDesc, "targetIndices"))
                return 0;
            if (!Runtime.SameShape(gradDesc, probsDesc))
            {
                Runtime.SetLastError("Gradient tensor shape must match probability tensor shape for ggml crossentropyloss backward.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(gradDesc) || !Runtime.CanMapStandardView(probsDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml crossentropyloss backward Metal path.");
                return 0;
            }
            if (labelSmooth < 0.0f || labelSmooth > 1.0f)
            {
                Runtime.SetLastError("labelSmooth must be in [0, 1] for ggml crossentropyloss backward.");
                return 0;
            }

            long rows = Runtime.FlatRowCount(probsDesc);
            long cols = probsDesc.Ne0;
            if (targetIndicesDesc.ElementCount != rows)
            {
                Runtime.SetLastError("Target index count must match the number of probability rows for ggml crossentropyloss backward.");
                return 0;
            }

            bool useZeroCopy = Runtime.CanMapStandardView(gradDesc) && Runtime.CanMapStandardView(probsDesc);
            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(6 * 1024 * 1024))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                Runtime.TensorBinding gradBinding = default;
                Runtime.TensorBinding probsBinding = default;
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, gradDesc, out gradBinding, out IntPtr gradBuf))
                        buffers.Add(gradBuf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, probsDesc, out probsBinding, out IntPtr probsBuf))
                        buffers.Add(probsBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                {
                    gradBinding = Runtime.CreateStandardBinding(context.Ctx, gradDesc);
                    probsBinding = Runtime.CreateStandardBinding(context.Ctx, probsDesc);
                }
                Runtime.TensorBinding labelsBinding = CreateMatrixBinding(context.Ctx, cols, rows);
                Runtime.TensorBinding lossGradBinding = CreateScalarBinding(context.Ctx);

                if (!gradBinding.IsValid || !probsBinding.IsValid ||
                    !labelsBinding.IsValid || !lossGradBinding.IsValid)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors for crossentropyloss backward.");
                    return 0;
                }

                GgmlTensor* contiguousProbs = ggml_cont(context.Ctx, probsBinding.Tensor);
                GgmlTensor* flatProbs = contiguousProbs == null ? null : ggml_reshape_2d(context.Ctx, contiguousProbs, cols, rows);
                GgmlTensor* logitsTensor = flatProbs == null ? null : ggml_log(context.Ctx, flatProbs);
                if (contiguousProbs == null || flatProbs == null || logitsTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml crossentropyloss backward logits tensor.");
                    return 0;
                }

                GgmlTensor* gradTensor = ggml_cross_entropy_loss_back(context.Ctx, lossGradBinding.Tensor, logitsTensor, labelsBinding.Tensor);
                if (gradTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml_cross_entropy_loss_back node.");
                    return 0;
                }

                if (!ggml_backend_supports_op(Runtime.Backend, gradTensor))
                {
                    Runtime.SetLastError("ggml_cross_entropy_loss_back is not supported by the active backend.");
                    return 0;
                }

                GgmlTensor* reshapedGrad = ggml_reshape_4d(context.Ctx, gradTensor, gradDesc.Ne0, gradDesc.Ne1, gradDesc.Ne2, gradDesc.Ne3);
                if (reshapedGrad == null)
                {
                    Runtime.SetLastError("Failed to reshape ggml crossentropyloss backward tensor.");
                    return 0;
                }

                if (addGrad)
                {
                    GgmlTensor* contiguousGrad = ggml_cont(context.Ctx, gradBinding.Tensor);
                    reshapedGrad = contiguousGrad == null ? null : ggml_add(context.Ctx, contiguousGrad, reshapedGrad);
                    if (reshapedGrad == null)
                    {
                        Runtime.SetLastError("Failed to create ggml crossentropyloss backward accumulation node.");
                        return 0;
                    }
                }

                GgmlTensor* output = ggml_cpy(context.Ctx, reshapedGrad, gradBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml crossentropyloss backward output copy node.");
                    return 0;
                }

                ggml_set_output(output);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }
                ggml_build_forward_expand(graph, output);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!BuildCrossEntropyLabelBuffer(out float[] labels, targetIndicesDesc, rows, cols, labelSmooth))
                    return 0;

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(probsBinding, probsDesc.Data, probsBinding.RawBytes);
                    if (addGrad || gradBinding.RawBytes > Runtime.LogicalBytes(gradDesc))
                        Runtime.UploadBinding(gradBinding, gradDesc.Data, gradBinding.RawBytes);
                }
                fixed (float* labelsPtr = labels)
                {
                    Runtime.UploadBinding(labelsBinding, (IntPtr)labelsPtr, labelsBinding.RawBytes);
                }
                ggml_backend_tensor_set(lossGradBinding.Storage, (IntPtr)(&lossGradient), 0, sizeof(float));

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                Runtime.FinalizeCompute(useZeroCopy, gradBinding, gradDesc.Data, gradBinding.RawBytes);
                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // Adam (AdamW optimizer step; updates weight/v/m in place and
        // zeroes the gradient afterwards)
        // --------------------------------------------------------------

        internal static int AdamF32(
            in GgmlContiguousTensor weightDesc,
            in GgmlContiguousTensor gradientDesc,
            in GgmlContiguousTensor vDesc,
            in GgmlContiguousTensor mDesc,
            float gradNormFactor,
            float stepSize,
            float clipValue,
            float regc,
            float decayRateV,
            float decayRateM,
            int iter,
            float eps)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(weightDesc, "weight")
                || !Runtime.ValidateDesc(gradientDesc, "gradient")
                || !Runtime.ValidateDesc(vDesc, "v")
                || !Runtime.ValidateDesc(mDesc, "m"))
            {
                return 0;
            }

            if (weightDesc.ElementCount != gradientDesc.ElementCount
                || weightDesc.ElementCount != vDesc.ElementCount
                || weightDesc.ElementCount != mDesc.ElementCount)
            {
                Runtime.SetLastError("Tensor shape mismatch passed to ggml adam.");
                return 0;
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
                Runtime.TensorBinding weightBinding = default;
                Runtime.TensorBinding gradientBinding = default;
                Runtime.TensorBinding vBinding = default;
                Runtime.TensorBinding mBinding = default;
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, weightDesc, out weightBinding, out IntPtr weightBuf))
                        buffers.Add(weightBuf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, gradientDesc, out gradientBinding, out IntPtr gradientBuf))
                        buffers.Add(gradientBuf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, vDesc, out vBinding, out IntPtr vBuf))
                        buffers.Add(vBuf);
                    else
                        useZeroCopy = false;
                }
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, mDesc, out mBinding, out IntPtr mBuf))
                        buffers.Add(mBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                {
                    weightBinding = Runtime.CreateContiguousBinding(context.Ctx, weightDesc);
                    gradientBinding = Runtime.CreateContiguousBinding(context.Ctx, gradientDesc);
                    vBinding = Runtime.CreateContiguousBinding(context.Ctx, vDesc);
                    mBinding = Runtime.CreateContiguousBinding(context.Ctx, mDesc);
                }
                GgmlTensor* adamwParamsTensor = ggml_new_tensor_1d(context.Ctx, GGML_TYPE_F32, 7);

                if (!weightBinding.IsValid || !gradientBinding.IsValid ||
                    !vBinding.IsValid || !mBinding.IsValid ||
                    adamwParamsTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors for adam.");
                    return 0;
                }

                GgmlTensor* gradTensor = gradientBinding.Tensor;
                if (gradNormFactor != 1.0f)
                {
                    gradTensor = ggml_scale(context.Ctx, gradTensor, gradNormFactor);
                    if (gradTensor == null)
                    {
                        Runtime.SetLastError("Failed to create ggml adam grad scaling node.");
                        return 0;
                    }
                }

                GgmlTensor* clippedGrad = ggml_clamp(context.Ctx, gradTensor, -clipValue, clipValue);
                if (clippedGrad == null)
                {
                    Runtime.SetLastError("Failed to create ggml adam clamp node.");
                    return 0;
                }

                float biasCorrectionM = (float)(1.0 / (1.0 - Math.Pow(decayRateM, iter)));
                float biasCorrectionV = (float)(1.0 / (1.0 - Math.Pow(decayRateV, iter)));
                float* adamwParams = stackalloc float[7]
                {
                    stepSize,
                    decayRateM,
                    decayRateV,
                    eps,
                    regc,
                    biasCorrectionM,
                    biasCorrectionV,
                };

                ggml_set_param(weightBinding.Tensor);

                GgmlTensor* adamwStep = ggml_opt_step_adamw(
                    context.Ctx,
                    weightBinding.Tensor,
                    clippedGrad,
                    mBinding.Tensor,
                    vBinding.Tensor,
                    adamwParamsTensor);
                if (adamwStep == null)
                {
                    Runtime.SetLastError("Failed to create ggml adamw optimizer node.");
                    return 0;
                }

                ggml_set_output(adamwStep);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }
                ggml_build_forward_expand(graph, adamwStep);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(weightBinding, weightDesc.Data, weightBinding.RawBytes);
                    Runtime.UploadBinding(gradientBinding, gradientDesc.Data, gradientBinding.RawBytes);
                    Runtime.UploadBinding(vBinding, vDesc.Data, vBinding.RawBytes);
                    Runtime.UploadBinding(mBinding, mDesc.Data, mBinding.RawBytes);
                }
                ggml_backend_tensor_set(adamwParamsTensor, (IntPtr)adamwParams, 0, 7 * sizeof(float));

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                ggml_backend_synchronize(Runtime.Backend);
                ggml_backend_tensor_memset(gradientBinding.Storage, 0, 0, gradientBinding.RawBytes);
                ggml_backend_synchronize(Runtime.Backend);

                if (!useZeroCopy)
                {
                    ggml_backend_tensor_get(weightBinding.Storage, weightDesc.Data, 0, weightBinding.RawBytes);
                    ggml_backend_tensor_get(mBinding.Storage, mDesc.Data, 0, mBinding.RawBytes);
                    ggml_backend_tensor_get(vBinding.Storage, vDesc.Data, 0, vBinding.RawBytes);
                    ggml_backend_tensor_get(gradientBinding.Storage, gradientDesc.Data, 0, gradientBinding.RawBytes);
                }

                Runtime.ClearLastError();
                return 1;
            }
        }
    }
}
