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
using System.Collections.Generic;
using TensorSharp.GGML.Interop;
using static TensorSharp.GGML.Interop.GgmlApi;
using Runtime = TensorSharp.GGML.Interop.GgmlManagedRuntime;

namespace TensorSharp.GGML
{
    // Managed port of TensorSharp.GGML.Native/ggml_ops_elementwise.cpp. Each
    // method mirrors the corresponding *_impl: validate the descriptors, bind
    // the caller's host buffers (zero-copy where the backend allows it, staged
    // upload otherwise), build a small ggml graph, compute on the shared
    // backend, and download the result when it was not host-mapped. Returns
    // 1/0 with a thread-local last-error exactly like the native entry points
    // so the GgmlNative wrappers can route to either implementation.
    internal static unsafe partial class GgmlManagedOps
    {
        internal static void Check(int result, string opName)
        {
            if (result != 0)
                return;
            Runtime.ThrowLastError(opName);
        }

        // Shared prologue: backend + zero-copy-or-standard binding of 4D descs
        // in the caller's order (result first), mirroring the native fallback
        // semantics (any zero-copy failure falls everything back to staged).
        private static bool BindAll4D(
            IntPtr ctx,
            ReadOnlySpan<GgmlTensorView4D> descs,
            Span<Runtime.TensorBinding> bindings,
            Runtime.BufferList buffers,
            out bool useZeroCopy)
        {
            useZeroCopy = true;
            for (int i = 0; i < descs.Length && useZeroCopy; ++i)
            {
                if (Runtime.CreateBindingFromHostPtr(ctx, descs[i], out bindings[i], out IntPtr buf))
                    buffers.Add(buf);
                else
                    useZeroCopy = false;
            }
            if (!useZeroCopy)
            {
                for (int i = 0; i < descs.Length; ++i)
                    bindings[i] = Runtime.CreateStandardBinding(ctx, descs[i]);
            }
            for (int i = 0; i < descs.Length; ++i)
            {
                if (!bindings[i].IsValid)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensor views.");
                    return false;
                }
            }
            return true;
        }

        private static GgmlTensor* MakeUnaryTensor(IntPtr ctx, int op, GgmlTensor* src)
        {
            switch ((GgmlUnaryOp)op)
            {
                case GgmlUnaryOp.Neg: return ggml_neg(ctx, src);
                case GgmlUnaryOp.Exp: return ggml_exp(ctx, src);
                case GgmlUnaryOp.Log: return ggml_log(ctx, src);
                case GgmlUnaryOp.Sqrt: return ggml_sqrt(ctx, src);
                case GgmlUnaryOp.Relu: return ggml_relu(ctx, src);
                case GgmlUnaryOp.Sigmoid: return ggml_sigmoid(ctx, src);
                case GgmlUnaryOp.Tanh: return ggml_tanh(ctx, src);
                case GgmlUnaryOp.SiLU: return ggml_silu(ctx, src);
                case GgmlUnaryOp.Step: return ggml_step(ctx, src);
                case GgmlUnaryOp.Abs: return ggml_abs(ctx, src);
                case GgmlUnaryOp.Sign: return ggml_sgn(ctx, src);
                case GgmlUnaryOp.GELU: return ggml_gelu(ctx, src);
                default:
                    Runtime.SetLastError("Unsupported unary ggml op code.");
                    return null;
            }
        }

        private static GgmlTensor* MakeBinaryTensor(IntPtr ctx, int op, GgmlTensor* lhs, GgmlTensor* rhs)
        {
            switch ((GgmlBinaryTensorOp)op)
            {
                case GgmlBinaryTensorOp.Add: return ggml_add(ctx, lhs, rhs);
                case GgmlBinaryTensorOp.Sub: return ggml_sub(ctx, lhs, rhs);
                case GgmlBinaryTensorOp.Mul: return ggml_mul(ctx, lhs, rhs);
                case GgmlBinaryTensorOp.Div: return ggml_div(ctx, lhs, rhs);
                default:
                    Runtime.SetLastError("Unsupported binary ggml op code.");
                    return null;
            }
        }

        private static GgmlTensor* MakeFusedActMulTensor(IntPtr ctx, int op, GgmlTensor* a, GgmlTensor* b)
        {
            switch ((GgmlFusedActMulOp)op)
            {
                case GgmlFusedActMulOp.SiLUMul: return ggml_mul(ctx, ggml_silu(ctx, a), b);
                case GgmlFusedActMulOp.GELUMul: return ggml_mul(ctx, ggml_gelu(ctx, a), b);
                case GgmlFusedActMulOp.SigmoidMul: return ggml_mul(ctx, a, ggml_sigmoid(ctx, b));
                default:
                    Runtime.SetLastError("Unsupported fused activation-multiply ggml op code.");
                    return null;
            }
        }

        private static GgmlTensor* MakeReductionTensor(IntPtr ctx, int op, GgmlTensor* src)
        {
            switch ((GgmlReductionOp)op)
            {
                case GgmlReductionOp.Sum: return ggml_sum_rows(ctx, src);
                case GgmlReductionOp.Mean: return ggml_mean(ctx, src);
                default:
                    Runtime.SetLastError("Unsupported reduction ggml op code.");
                    return null;
            }
        }

        private static bool CanRepeat(in GgmlTensorView4D repeated, in GgmlTensorView4D target) =>
            (target.Ne0 % repeated.Ne0) == 0 &&
            (target.Ne1 % repeated.Ne1) == 0 &&
            (target.Ne2 % repeated.Ne2) == 0 &&
            (target.Ne3 % repeated.Ne3) == 0;

        private static bool ReadI32Values(int[] output, in GgmlContiguousTensor desc)
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
            Runtime.SetLastError("Unsupported element type for indices.");
            return false;
        }

        // --------------------------------------------------------------
        // ReduceLastDim (sum / mean over the fastest dimension)
        // --------------------------------------------------------------

        internal static int ReduceLastDimF32(int op, in GgmlTensorView4D resultDesc, in GgmlTensorView4D srcDesc)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src"))
                return 0;
            if (!Runtime.SameShapeWithLastDimReduced(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Result tensor shape must match source shape with the last dimension reduced to 1.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml reduction Metal path.");
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
                    Runtime.SetLastError("Failed to create ggml contiguous reduction input.");
                    return 0;
                }

                GgmlTensor* reduced = MakeReductionTensor(context.Ctx, op, contiguousSrc);
                if (reduced == null)
                    return 0;

                GgmlTensor* output = ggml_cpy(context.Ctx, reduced, resultBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml reduction output copy node.");
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
        // IndexReduction (argmin / argmax over the fastest dimension)
        // --------------------------------------------------------------

        internal static int IndexReductionF32(int op, in GgmlTensorView4D resultDesc, in GgmlTensorView4D srcDesc)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src"))
                return 0;
            if (!Runtime.SameShapeWithLastDimReduced(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Result tensor shape must match source shape with the last dimension reduced to 1.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml index-reduction Metal path.");
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
                bool srcZeroCopy = true;
                Runtime.TensorBinding srcBinding;
                if (Runtime.CreateBindingFromHostPtr(context.Ctx, srcDesc, out srcBinding, out IntPtr srcBuf))
                    buffers.Add(srcBuf);
                else
                    srcZeroCopy = false;
                if (!srcZeroCopy)
                    srcBinding = Runtime.CreateStandardBinding(context.Ctx, srcDesc);
                if (!srcBinding.IsValid)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensor views.");
                    return 0;
                }

                GgmlTensor* contiguousSrc = ggml_cont(context.Ctx, srcBinding.Tensor);
                if (contiguousSrc == null)
                {
                    Runtime.SetLastError("Failed to create ggml contiguous index-reduction input.");
                    return 0;
                }

                GgmlTensor* reductionInput = contiguousSrc;
                if ((GgmlIndexReductionOp)op == GgmlIndexReductionOp.Argmin)
                {
                    reductionInput = ggml_neg(context.Ctx, contiguousSrc);
                    if (reductionInput == null)
                    {
                        Runtime.SetLastError("Failed to create ggml argmin preprocessing node.");
                        return 0;
                    }
                }
                else if ((GgmlIndexReductionOp)op != GgmlIndexReductionOp.Argmax)
                {
                    Runtime.SetLastError("Unsupported index-reduction ggml op code.");
                    return 0;
                }

                long rows = Runtime.FlatRowCount(srcDesc);
                GgmlTensor* flatInput = ggml_reshape_2d(context.Ctx, reductionInput, srcDesc.Ne0, rows);
                GgmlTensor* argTensor = flatInput == null ? null : ggml_argmax(context.Ctx, flatInput);
                if (flatInput == null || argTensor == null)
                {
                    Runtime.SetLastError("Failed to create ggml index-reduction node.");
                    return 0;
                }
                ggml_set_output(argTensor);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph.");
                    return 0;
                }
                ggml_build_forward_expand(graph, argTensor);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!srcZeroCopy)
                    Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed.");
                    return 0;
                }

                // Host readback of the i32 indices; converted to F32 into the
                // caller's result buffer, exactly like the native impl.
                GgmlNative.HostReadBarrier();
                ggml_backend_synchronize(Runtime.Backend);
                int[] hostIndices = new int[rows];
                fixed (int* indicesPtr = hostIndices)
                {
                    ggml_backend_tensor_get(argTensor, (IntPtr)indicesPtr, 0, (nuint)(rows * sizeof(int)));
                }
                float* resultData = (float*)resultDesc.Data;
                for (long i = 0; i < rows; ++i)
                    resultData[i] = hostIndices[i];

                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // Copy
        // --------------------------------------------------------------

        internal static int CopyF32(in GgmlTensorView4D resultDesc, in GgmlTensorView4D srcDesc)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src"))
                return 0;
            if (!Runtime.SameShape(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Source tensor shape does not match result shape for ggml copy.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml copy Metal path.");
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
                    Runtime.SetLastError("Failed to create ggml contiguous copy input.");
                    return 0;
                }

                GgmlTensor* output = ggml_cpy(context.Ctx, contiguousSrc, resultBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml copy node.");
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
        // Unary
        // --------------------------------------------------------------

        internal static int UnaryF32(int op, in GgmlTensorView4D resultDesc, in GgmlTensorView4D srcDesc)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src"))
                return 0;
            if (!Runtime.SameShape(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Source tensor shape does not match result shape for unary ggml op.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the unary ggml Metal path.");
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
                    Runtime.SetLastError("Failed to create ggml contiguous unary input.");
                    return 0;
                }

                GgmlTensor* value = MakeUnaryTensor(context.Ctx, op, contiguousSrc);
                if (value == null)
                    return 0;

                GgmlTensor* output = ggml_cpy(context.Ctx, value, resultBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml unary output copy node.");
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
        // BinaryTensor (add / sub / mul / div with rhs broadcast)
        // --------------------------------------------------------------

        internal static int BinaryTensorF32(int op, in GgmlTensorView4D resultDesc, in GgmlTensorView4D lhsDesc, in GgmlTensorView4D rhsDesc)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(lhsDesc, "lhs") || !Runtime.ValidateDesc(rhsDesc, "rhs"))
                return 0;
            if (!Runtime.SameShape(resultDesc, lhsDesc))
            {
                Runtime.SetLastError("Result tensor shape does not match lhs tensor shape.");
                return 0;
            }
            if (!CanRepeat(rhsDesc, lhsDesc))
            {
                Runtime.SetLastError("rhs tensor shape cannot be broadcast to lhs for ggml binary op.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(lhsDesc) || !Runtime.CanMapStandardView(rhsDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the binary ggml Metal path.");
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
                Span<GgmlTensorView4D> descs = stackalloc GgmlTensorView4D[] { resultDesc, lhsDesc, rhsDesc };
                Span<Runtime.TensorBinding> bindings = stackalloc Runtime.TensorBinding[3];
                bool bound = BindAll4D(context.Ctx, descs, bindings, buffers, out bool useZeroCopy);
                if (!bound)
                    return 0;
                Runtime.TensorBinding resultBinding = bindings[0], lhsBinding = bindings[1], rhsBinding = bindings[2];

                GgmlTensor* value = MakeBinaryTensor(context.Ctx, op, lhsBinding.Tensor, rhsBinding.Tensor);
                if (value == null)
                    return 0;

                GgmlTensor* output = ggml_cpy(context.Ctx, value, resultBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml binary output copy node.");
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

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(lhsBinding, lhsDesc.Data, lhsBinding.RawBytes);
                    Runtime.UploadBinding(rhsBinding, rhsDesc.Data, rhsBinding.RawBytes);
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
        // FusedActMul (act(a) * b)
        // --------------------------------------------------------------

        internal static int FusedActMulF32(int op, in GgmlTensorView4D resultDesc, in GgmlTensorView4D aDesc, in GgmlTensorView4D bDesc)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(aDesc, "a") || !Runtime.ValidateDesc(bDesc, "b"))
                return 0;
            if (!Runtime.SameShape(resultDesc, aDesc) || !Runtime.SameShape(resultDesc, bDesc))
            {
                Runtime.SetLastError("All tensor shapes must match for fused activation-multiply op.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(aDesc) || !Runtime.CanMapStandardView(bDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the fused activation-multiply ggml path.");
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
                Span<GgmlTensorView4D> descs = stackalloc GgmlTensorView4D[] { resultDesc, aDesc, bDesc };
                Span<Runtime.TensorBinding> bindings = stackalloc Runtime.TensorBinding[3];
                bool bound = BindAll4D(context.Ctx, descs, bindings, buffers, out bool useZeroCopy);
                if (!bound)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensor views for fused op.");
                    return 0;
                }
                Runtime.TensorBinding resultBinding = bindings[0], aBinding = bindings[1], bBinding = bindings[2];

                GgmlTensor* value = MakeFusedActMulTensor(context.Ctx, op, aBinding.Tensor, bBinding.Tensor);
                if (value == null)
                    return 0;

                GgmlTensor* output = ggml_cpy(context.Ctx, value, resultBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml fused output copy node.");
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

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(aBinding, aDesc.Data, aBinding.RawBytes);
                    Runtime.UploadBinding(bBinding, bDesc.Data, bBinding.RawBytes);
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
        // FusedActMulSplit (act(gate_up[:, :H]) * gate_up[:, H:2H])
        // --------------------------------------------------------------

        internal static int FusedActMulSplitF32(int op, in GgmlTensorView2D resultDesc, in GgmlTensorView2D gateUpDesc, int halfDim)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(gateUpDesc, "gate_up"))
                return 0;
            if (halfDim <= 0)
            {
                Runtime.SetLastError("fused_act_mul_split: half_dim must be positive.");
                return 0;
            }
            if (resultDesc.Stride1 != 1 || gateUpDesc.Stride1 != 1)
            {
                Runtime.SetLastError("fused_act_mul_split: tensors must be row-major (stride1 == 1).");
                return 0;
            }
            if (resultDesc.Dim0 != gateUpDesc.Dim0 || resultDesc.Dim1 != halfDim || gateUpDesc.Dim1 != 2 * halfDim)
            {
                Runtime.SetLastError("fused_act_mul_split: shape mismatch.");
                return 0;
            }
            if (gateUpDesc.Stride0 < gateUpDesc.Dim1)
            {
                Runtime.SetLastError("fused_act_mul_split: gate_up row stride must be >= dim1.");
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
                // Promote to 4D so the shared zero-copy helper applies (mirrors
                // the native to_4d lambda).
                static GgmlTensorView4D To4D(in GgmlTensorView2D d)
                {
                    long nb1 = (long)d.Stride0 * sizeof(float);
                    return new GgmlTensorView4D(d.Data, d.Dim1, d.Dim0, 1, 1, nb1, nb1 * d.Dim0, nb1 * d.Dim0, d.RawBytes);
                }

                GgmlTensorView4D result4D = To4D(resultDesc);
                GgmlTensorView4D gateUp4D = To4D(gateUpDesc);

                Span<GgmlTensorView4D> descs = stackalloc GgmlTensorView4D[] { result4D, gateUp4D };
                Span<Runtime.TensorBinding> bindings = stackalloc Runtime.TensorBinding[2];
                bool bound = BindAll4D(context.Ctx, descs, bindings, buffers, out bool useZeroCopy);
                if (!bound)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors for fused_act_mul_split.");
                    return 0;
                }
                Runtime.TensorBinding resultBinding = bindings[0], gateUpBinding = bindings[1];

                nuint rowBytes = (nuint)((long)gateUpDesc.Stride0 * sizeof(float));
                nuint halfBytes = (nuint)((long)halfDim * sizeof(float));

                GgmlTensor* gateView = ggml_view_2d(context.Ctx, gateUpBinding.Tensor, halfDim, gateUpDesc.Dim0, rowBytes, 0);
                GgmlTensor* upView = ggml_view_2d(context.Ctx, gateUpBinding.Tensor, halfDim, gateUpDesc.Dim0, rowBytes, halfBytes);
                if (gateView == null || upView == null)
                {
                    Runtime.SetLastError("Failed to create gate/up views for fused_act_mul_split.");
                    return 0;
                }

                GgmlTensor* gateCont = ggml_cont(context.Ctx, gateView);
                GgmlTensor* upCont = ggml_cont(context.Ctx, upView);
                if (gateCont == null || upCont == null)
                {
                    Runtime.SetLastError("Failed to create gate/up cont nodes for fused_act_mul_split.");
                    return 0;
                }

                GgmlTensor* value = MakeFusedActMulTensor(context.Ctx, op, gateCont, upCont);
                if (value == null)
                    return 0;

                GgmlTensor* output = ggml_cpy(context.Ctx, value, resultBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml fused_act_mul_split output copy node.");
                    return 0;
                }
                ggml_set_output(output);

                IntPtr graph = ggml_new_graph(context.Ctx);
                if (graph == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to create ggml graph for fused_act_mul_split.");
                    return 0;
                }
                ggml_build_forward_expand(graph, output);

                IntPtr computeBuffer = ggml_backend_alloc_ctx_tensors(context.Ctx, Runtime.Backend);
                if (computeBuffer == IntPtr.Zero)
                {
                    Runtime.SetLastError("Failed to allocate ggml backend buffer for fused_act_mul_split.");
                    return 0;
                }
                buffers.Add(computeBuffer);

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(gateUpBinding, gateUpDesc.Data, gateUpBinding.RawBytes);
                    if (resultBinding.RawBytes > Runtime.LogicalBytes(result4D))
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                }

                if (ggml_backend_graph_compute(Runtime.Backend, graph) != GGML_STATUS_SUCCESS)
                {
                    Runtime.SetLastError("ggml backend graph execution failed for fused_act_mul_split.");
                    return 0;
                }

                Runtime.FinalizeCompute(useZeroCopy, resultBinding, resultDesc.Data, resultBinding.RawBytes);
                Runtime.ClearLastError();
                return 1;
            }
        }

        // --------------------------------------------------------------
        // BinaryScalar
        // --------------------------------------------------------------

        internal static int BinaryScalarF32(int op, in GgmlTensorView4D resultDesc, in GgmlTensorView4D srcDesc, float scalar)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src"))
                return 0;
            if (!Runtime.SameShape(resultDesc, srcDesc))
            {
                Runtime.SetLastError("Result tensor shape does not match source tensor shape for scalar ggml op.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the scalar ggml Metal path.");
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
                Span<GgmlTensorView4D> descs = stackalloc GgmlTensorView4D[] { resultDesc, srcDesc };
                Span<Runtime.TensorBinding> bindings = stackalloc Runtime.TensorBinding[2];
                bool bound = BindAll4D(context.Ctx, descs, bindings, buffers, out bool useZeroCopy);
                if (!bound)
                    return 0;
                Runtime.TensorBinding resultBinding = bindings[0], srcBinding = bindings[1];

                GgmlTensor* scalarTensor = ggml_new_tensor_1d(context.Ctx, GGML_TYPE_F32, 1);
                if (scalarTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors.");
                    return 0;
                }

                GgmlTensor* contiguousSrc = ggml_cont(context.Ctx, srcBinding.Tensor);
                if (contiguousSrc == null)
                {
                    Runtime.SetLastError("Failed to create ggml contiguous scalar-op input.");
                    return 0;
                }

                var opCode = (GgmlBinaryScalarOp)op;
                GgmlTensor* value = null;
                if (opCode == GgmlBinaryScalarOp.Mul)
                {
                    value = ggml_scale(context.Ctx, contiguousSrc, scalar);
                }
                else
                {
                    GgmlTensor* repeatedScalar = ggml_repeat(context.Ctx, scalarTensor, contiguousSrc);
                    if (repeatedScalar == null)
                    {
                        Runtime.SetLastError("Failed to create repeated scalar tensor.");
                        return 0;
                    }

                    switch (opCode)
                    {
                        case GgmlBinaryScalarOp.Add:
                            value = ggml_add(context.Ctx, contiguousSrc, repeatedScalar);
                            break;
                        case GgmlBinaryScalarOp.Sub:
                            value = ggml_sub(context.Ctx, contiguousSrc, repeatedScalar);
                            break;
                        case GgmlBinaryScalarOp.ReverseSub:
                            value = ggml_sub(context.Ctx, repeatedScalar, contiguousSrc);
                            break;
                        case GgmlBinaryScalarOp.Div:
                            value = ggml_div(context.Ctx, contiguousSrc, repeatedScalar);
                            break;
                        case GgmlBinaryScalarOp.ReverseDiv:
                            value = ggml_div(context.Ctx, repeatedScalar, contiguousSrc);
                            break;
                        default:
                            Runtime.SetLastError("Unsupported scalar ggml op code.");
                            return 0;
                    }
                }

                if (value == null)
                {
                    Runtime.SetLastError("Failed to create ggml scalar op node.");
                    return 0;
                }

                GgmlTensor* output = ggml_cpy(context.Ctx, value, resultBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml scalar-op output copy node.");
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

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);
                    if (resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                }
                if (opCode != GgmlBinaryScalarOp.Mul)
                {
                    ggml_backend_tensor_set(scalarTensor, (IntPtr)(&scalar), 0, sizeof(float));
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
        // ActivationGrad
        // --------------------------------------------------------------

        internal static int ActivationGradF32(
            int op,
            in GgmlTensorView4D resultDesc,
            in GgmlTensorView4D srcDesc,
            in GgmlTensorView4D gradDesc,
            in GgmlTensorView4D accumulationDesc,
            bool hasAccumulation)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src") || !Runtime.ValidateDesc(gradDesc, "grad"))
                return 0;
            if (hasAccumulation && !Runtime.ValidateDesc(accumulationDesc, "accumulation"))
                return 0;
            if (!Runtime.SameShape(resultDesc, srcDesc) || !Runtime.SameShape(srcDesc, gradDesc) ||
                (hasAccumulation && !Runtime.SameShape(srcDesc, accumulationDesc)))
            {
                Runtime.SetLastError("Tensor shape mismatch passed to ggml activation grad.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc) || !Runtime.CanMapStandardView(gradDesc) ||
                (hasAccumulation && !Runtime.CanMapStandardView(accumulationDesc)))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml activation-grad Metal path.");
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
                int descCount = hasAccumulation ? 4 : 3;
                Span<GgmlTensorView4D> descs = stackalloc GgmlTensorView4D[4];
                descs[0] = resultDesc;
                descs[1] = srcDesc;
                descs[2] = gradDesc;
                if (hasAccumulation)
                    descs[3] = accumulationDesc;
                Span<Runtime.TensorBinding> bindings = stackalloc Runtime.TensorBinding[4];
                bool bound = BindAll4D(context.Ctx, descs.Slice(0, descCount), bindings.Slice(0, descCount), buffers, out bool useZeroCopy);
                if (!bound)
                    return 0;
                Runtime.TensorBinding resultBinding = bindings[0], srcBinding = bindings[1], gradBinding = bindings[2];
                Runtime.TensorBinding accumulationBinding = hasAccumulation ? bindings[3] : default;

                GgmlTensor* oneTensor = ggml_new_tensor_1d(context.Ctx, GGML_TYPE_F32, 1);
                if (oneTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors.");
                    return 0;
                }

                GgmlTensor* contiguousSrc = ggml_cont(context.Ctx, srcBinding.Tensor);
                GgmlTensor* contiguousGrad = ggml_cont(context.Ctx, gradBinding.Tensor);
                if (contiguousSrc == null || contiguousGrad == null)
                {
                    Runtime.SetLastError("Failed to create ggml contiguous activation-grad inputs.");
                    return 0;
                }

                GgmlTensor* value = null;
                switch ((GgmlActivationGradOp)op)
                {
                    case GgmlActivationGradOp.Relu:
                    {
                        GgmlTensor* step = ggml_step(context.Ctx, contiguousSrc);
                        if (step != null)
                            value = ggml_mul(context.Ctx, step, contiguousGrad);
                        break;
                    }
                    case GgmlActivationGradOp.Sigmoid:
                    {
                        GgmlTensor* one = ggml_repeat(context.Ctx, oneTensor, contiguousSrc);
                        GgmlTensor* oneMinus = one == null ? null : ggml_sub(context.Ctx, one, contiguousSrc);
                        GgmlTensor* deriv = oneMinus == null ? null : ggml_mul(context.Ctx, contiguousSrc, oneMinus);
                        value = deriv == null ? null : ggml_mul(context.Ctx, deriv, contiguousGrad);
                        break;
                    }
                    case GgmlActivationGradOp.Tanh:
                    {
                        GgmlTensor* one = ggml_repeat(context.Ctx, oneTensor, contiguousSrc);
                        GgmlTensor* sq = ggml_mul(context.Ctx, contiguousSrc, contiguousSrc);
                        GgmlTensor* oneMinus = (one == null || sq == null) ? null : ggml_sub(context.Ctx, one, sq);
                        value = oneMinus == null ? null : ggml_mul(context.Ctx, oneMinus, contiguousGrad);
                        break;
                    }
                    case GgmlActivationGradOp.SiLU:
                    {
                        value = ggml_silu_back(context.Ctx, contiguousGrad, contiguousSrc);
                        if (value == null || !ggml_backend_supports_op(Runtime.Backend, value))
                        {
                            GgmlTensor* one = ggml_repeat(context.Ctx, oneTensor, contiguousSrc);
                            GgmlTensor* sig = ggml_sigmoid(context.Ctx, contiguousSrc);
                            GgmlTensor* oneMinusSig = (one == null || sig == null) ? null : ggml_sub(context.Ctx, one, sig);
                            GgmlTensor* weighted = oneMinusSig == null ? null : ggml_mul(context.Ctx, contiguousSrc, oneMinusSig);
                            GgmlTensor* inner = (one == null || weighted == null) ? null : ggml_add(context.Ctx, one, weighted);
                            GgmlTensor* deriv = (sig == null || inner == null) ? null : ggml_mul(context.Ctx, sig, inner);
                            value = deriv == null ? null : ggml_mul(context.Ctx, deriv, contiguousGrad);
                        }
                        break;
                    }
                    default:
                        Runtime.SetLastError("Unsupported activation-grad ggml op code.");
                        return 0;
                }

                if (value == null)
                {
                    Runtime.SetLastError("Failed to create ggml activation-grad node.");
                    return 0;
                }

                if (hasAccumulation)
                {
                    GgmlTensor* contiguousAccumulation = ggml_cont(context.Ctx, accumulationBinding.Tensor);
                    if (contiguousAccumulation == null)
                    {
                        Runtime.SetLastError("Failed to create ggml contiguous accumulation input.");
                        return 0;
                    }
                    value = ggml_add(context.Ctx, contiguousAccumulation, value);
                    if (value == null)
                    {
                        Runtime.SetLastError("Failed to create ggml activation-grad accumulation node.");
                        return 0;
                    }
                }

                GgmlTensor* output = ggml_cpy(context.Ctx, value, resultBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml activation-grad output copy node.");
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

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);
                    Runtime.UploadBinding(gradBinding, gradDesc.Data, gradBinding.RawBytes);
                    if (hasAccumulation)
                        Runtime.UploadBinding(accumulationBinding, accumulationDesc.Data, accumulationBinding.RawBytes);
                    if (resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                }
                float oneValue = 1.0f;
                ggml_backend_tensor_set(oneTensor, (IntPtr)(&oneValue), 0, sizeof(float));

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
        // IndexSelect (get_rows, optionally accumulated into result)
        // --------------------------------------------------------------

        internal static int IndexSelectF32(in GgmlTensorView2D resultDesc, in GgmlTensorView2D srcDesc, in GgmlContiguousTensor indicesDesc, bool addToResult)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(resultDesc, "result") || !Runtime.ValidateDesc(srcDesc, "src") || !Runtime.ValidateDesc(indicesDesc, "indices"))
                return 0;
            if (resultDesc.Dim1 != srcDesc.Dim1 || indicesDesc.ElementCount != resultDesc.Dim0)
            {
                Runtime.SetLastError("Tensor shape mismatch passed to ggml indexselect.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(resultDesc) || !Runtime.CanMapStandardView(srcDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml indexselect Metal path.");
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
                bool useZeroCopy = true;
                Runtime.TensorBinding resultBinding = default, srcBinding = default;
                if (Runtime.CreateBindingFromHostPtr(context.Ctx, resultDesc, out resultBinding, out IntPtr resultBuf))
                    buffers.Add(resultBuf);
                else
                    useZeroCopy = false;
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, srcDesc, out srcBinding, out IntPtr srcBuf))
                        buffers.Add(srcBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                {
                    resultBinding = Runtime.CreateStandardBinding(context.Ctx, resultDesc);
                    srcBinding = Runtime.CreateStandardBinding(context.Ctx, srcDesc);
                }
                GgmlTensor* indexTensor = ggml_new_tensor_1d(context.Ctx, GGML_TYPE_I32, indicesDesc.ElementCount);
                if (!resultBinding.IsValid || !srcBinding.IsValid || indexTensor == null)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors.");
                    return 0;
                }

                GgmlTensor* contiguousSrc = ggml_cont(context.Ctx, srcBinding.Tensor);
                if (contiguousSrc == null)
                {
                    Runtime.SetLastError("Failed to create ggml contiguous indexselect input.");
                    return 0;
                }

                GgmlTensor* value = ggml_get_rows(context.Ctx, contiguousSrc, indexTensor);
                if (value == null)
                {
                    Runtime.SetLastError("Failed to create ggml get_rows node.");
                    return 0;
                }

                if (addToResult)
                {
                    GgmlTensor* contiguousResult = ggml_cont(context.Ctx, resultBinding.Tensor);
                    if (contiguousResult == null)
                    {
                        Runtime.SetLastError("Failed to create ggml contiguous indexselect accumulation input.");
                        return 0;
                    }
                    value = ggml_add(context.Ctx, value, contiguousResult);
                    if (value == null)
                    {
                        Runtime.SetLastError("Failed to create ggml indexselect accumulation node.");
                        return 0;
                    }
                }

                GgmlTensor* output = ggml_cpy(context.Ctx, value, resultBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml indexselect output copy node.");
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

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(srcBinding, srcDesc.Data, srcBinding.RawBytes);
                    if (addToResult || resultBinding.RawBytes > Runtime.LogicalBytes(resultDesc))
                        Runtime.UploadBinding(resultBinding, resultDesc.Data, resultBinding.RawBytes);
                }

                int[] indices = new int[indicesDesc.ElementCount];
                if (!ReadI32Values(indices, indicesDesc))
                    return 0;
                fixed (int* indicesPtr = indices)
                {
                    ggml_backend_tensor_set(indexTensor, (IntPtr)indicesPtr, 0, (nuint)(indices.Length * sizeof(int)));
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
        // IndexSelectGrad (scatter-add of adj rows into grad)
        // --------------------------------------------------------------

        internal static int IndexSelectGradF32(in GgmlTensorView2D gradDesc, in GgmlTensorView2D adjDesc, in GgmlContiguousTensor indicesDesc)
        {
            if (!Runtime.EnsureBackend())
                return 0;
            if (!Runtime.ValidateDesc(gradDesc, "grad") || !Runtime.ValidateDesc(adjDesc, "adj") || !Runtime.ValidateDesc(indicesDesc, "indices"))
                return 0;
            if (adjDesc.Dim0 != indicesDesc.ElementCount || gradDesc.Dim1 != adjDesc.Dim1)
            {
                Runtime.SetLastError("Tensor shape mismatch passed to ggml indexselectgrad.");
                return 0;
            }
            if (!Runtime.CanMapStandardView(gradDesc) || !Runtime.CanMapStandardView(adjDesc))
            {
                Runtime.SetLastError("Tensor layout is not supported by the ggml indexselectgrad Metal path.");
                return 0;
            }

            int[] indices = new int[indicesDesc.ElementCount];
            if (!ReadI32Values(indices, indicesDesc))
                return 0;
            long activeRowCount = 0;
            foreach (int index in indices)
            {
                if (index >= 0)
                    ++activeRowCount;
            }

            nuint minGraphCapacity = (nuint)GGML_DEFAULT_GRAPH_SIZE * 8;
            nuint estimatedGraphCapacity = (nuint)(activeRowCount * 6 + 64);
            nuint graphCapacity = estimatedGraphCapacity > minGraphCapacity ? estimatedGraphCapacity : minGraphCapacity;
            nuint ctxSize = 16 * 1024 * 1024 + ggml_graph_overhead_custom(graphCapacity, false);

            using var buffers = new Runtime.BufferList();
            var context = new Runtime.PooledContext();
            if (!context.Init(ctxSize))
            {
                Runtime.SetLastError("Failed to create ggml context.");
                return 0;
            }
            using (context)
            {
                bool useZeroCopy = true;
                Runtime.TensorBinding gradBinding = default, adjBinding = default;
                if (Runtime.CreateBindingFromHostPtr(context.Ctx, gradDesc, out gradBinding, out IntPtr gradBuf))
                    buffers.Add(gradBuf);
                else
                    useZeroCopy = false;
                if (useZeroCopy)
                {
                    if (Runtime.CreateBindingFromHostPtr(context.Ctx, adjDesc, out adjBinding, out IntPtr adjBuf))
                        buffers.Add(adjBuf);
                    else
                        useZeroCopy = false;
                }
                if (!useZeroCopy)
                {
                    gradBinding = Runtime.CreateStandardBinding(context.Ctx, gradDesc);
                    adjBinding = Runtime.CreateStandardBinding(context.Ctx, adjDesc);
                }
                if (!gradBinding.IsValid || !adjBinding.IsValid)
                {
                    Runtime.SetLastError("Failed to allocate ggml tensors.");
                    return 0;
                }

                GgmlTensor* workingGrad = ggml_cont(context.Ctx, gradBinding.Tensor);
                GgmlTensor* contiguousAdj = ggml_cont(context.Ctx, adjBinding.Tensor);
                if (workingGrad == null || contiguousAdj == null)
                {
                    Runtime.SetLastError("Failed to create ggml contiguous indexselectgrad inputs.");
                    return 0;
                }

                var pendingIndexTensors = new List<(IntPtr Tensor, int Value)>(indices.Length);

                nuint rowBytes = (nuint)((long)adjDesc.Dim1 * sizeof(float));
                for (int row = 0; row < indices.Length; ++row)
                {
                    if (indices[row] < 0)
                        continue;

                    GgmlTensor* indexTensor = ggml_new_tensor_1d(context.Ctx, GGML_TYPE_I32, 1);
                    GgmlTensor* currentRow = indexTensor == null ? null : ggml_get_rows(context.Ctx, workingGrad, indexTensor);
                    GgmlTensor* adjRow = currentRow == null ? null : ggml_view_2d(
                        context.Ctx, contiguousAdj, adjDesc.Dim1, 1, rowBytes, (nuint)row * rowBytes);
                    GgmlTensor* updatedRow = (currentRow == null || adjRow == null) ? null : ggml_add(context.Ctx, currentRow, adjRow);
                    GgmlTensor* updatedGrad = updatedRow == null ? null : ggml_set_rows(context.Ctx, workingGrad, updatedRow, indexTensor);

                    if (indexTensor == null || currentRow == null || adjRow == null || updatedRow == null || updatedGrad == null)
                    {
                        Runtime.SetLastError("Failed to create ggml indexselectgrad scatter-add node.");
                        return 0;
                    }

                    pendingIndexTensors.Add(((IntPtr)indexTensor, indices[row]));
                    workingGrad = updatedGrad;
                }

                GgmlTensor* output = ggml_cpy(context.Ctx, workingGrad, gradBinding.Tensor);
                if (output == null)
                {
                    Runtime.SetLastError("Failed to create ggml indexselectgrad output copy node.");
                    return 0;
                }
                ggml_set_output(output);

                IntPtr graph = ggml_new_graph_custom(context.Ctx, graphCapacity, false);
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

                if (!useZeroCopy)
                {
                    Runtime.UploadBinding(gradBinding, gradDesc.Data, gradBinding.RawBytes);
                    Runtime.UploadBinding(adjBinding, adjDesc.Data, adjBinding.RawBytes);
                }
                foreach ((IntPtr tensor, int value) in pendingIndexTensors)
                {
                    int v = value;
                    ggml_backend_tensor_set((GgmlTensor*)tensor, (IntPtr)(&v), 0, sizeof(int));
                }

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
    }
}
