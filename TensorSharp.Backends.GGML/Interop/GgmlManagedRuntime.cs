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
using System.Runtime.InteropServices;
using System.Threading;

namespace TensorSharp.GGML.Interop
{
    // Managed port of the shared infrastructure in
    // TensorSharp.GGML.Native/ggml_ops_core.cpp + ggml_ops_internal.h: backend
    // access, the pooled graph-context allocator, tensor-descriptor validation,
    // and the host-pointer (zero-copy) / staged-upload binding helpers that the
    // managed op implementations (GgmlManagedOps.*) build ggml graphs with.
    //
    // The backend instance is SHARED with the remaining native kernels (fetched
    // via TSGgml_GetBackendHandle), so managed-built graphs run on the same
    // CUDA context / command queue and interleave correctly with native ops.
    internal static unsafe class GgmlManagedRuntime
    {
        // ------------------------------------------------------------------
        // Enable switch
        // ------------------------------------------------------------------

        // TS_GGML_MANAGED_OPS=1 routes ported ops through the managed graph
        // builders; unset/0 keeps the legacy native TSGgml_* path. Read once.
        internal static readonly bool OpsEnabled = IsTruthy(Environment.GetEnvironmentVariable("TS_GGML_MANAGED_OPS"));

        private static bool IsTruthy(string v) =>
            v == "1" ||
            string.Equals(v, "true", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(v, "on", StringComparison.OrdinalIgnoreCase);

        // ------------------------------------------------------------------
        // Error handling (mirrors tsg::g_last_error)
        // ------------------------------------------------------------------

        [ThreadStatic] private static string t_lastError;

        internal static void SetLastError(string message) => t_lastError = message;
        internal static void ClearLastError() => t_lastError = null;
        internal static string LastError => string.IsNullOrEmpty(t_lastError) ? "Unknown managed GGML error." : t_lastError;

        internal static void ThrowLastError(string opName) =>
            throw new InvalidOperationException($"Managed GGML {opName} failed. {LastError}");

        // ------------------------------------------------------------------
        // Backend (shared singleton with the native layer)
        // ------------------------------------------------------------------

        private static IntPtr s_backend;
        private static int s_backendType;
        private static bool s_abiValidated;
        private static readonly object s_backendLock = new object();

        internal static IntPtr Backend => s_backend;
        internal static int BackendType => s_backendType;

        internal const int BackendTypeMetal = 1;
        internal const int BackendTypeCpu = 2;
        internal const int BackendTypeCuda = 3;
        internal const int BackendTypeVulkan = 4;

        // Mirrors tsg::ensure_backend but delegates the singleton's creation to
        // the native layer so both layers use the same instance.
        internal static bool EnsureBackend()
        {
            if (s_backend != IntPtr.Zero)
                return true;

            lock (s_backendLock)
            {
                if (s_backend != IntPtr.Zero)
                    return true;

                int backendType = GgmlNative.InitializedBackendType;
                if (backendType == 0)
                {
                    SetLastError("The GGML backend has not been initialized yet (no GgmlContext was created).");
                    return false;
                }

                IntPtr backend = GgmlNative.GetBackendHandle(backendType);
                if (backend == IntPtr.Zero)
                {
                    SetLastError("Failed to obtain the native GGML backend handle.");
                    return false;
                }

                ValidateAbi();
                s_backendType = backendType;
                s_backend = backend;
                return true;
            }
        }

        // One-time functional validation that the GgmlTensor struct layout in
        // GgmlApi matches the native ggml build. Creates a small context, makes
        // a 3x5 F32 tensor and checks type/ne/nb/nbytes through the managed
        // struct view. Throws (rather than corrupting memory later) on drift.
        private static void ValidateAbi()
        {
            if (s_abiValidated)
                return;

            byte* mem = stackalloc byte[8192];
            var p = new GgmlApi.ggml_init_params
            {
                mem_size = 8192,
                mem_buffer = (IntPtr)mem,
                no_alloc = true,
            };
            IntPtr ctx = GgmlApi.ggml_init(p);
            if (ctx == IntPtr.Zero)
                throw new InvalidOperationException("Managed GGML ABI validation failed: ggml_init returned null.");
            try
            {
                GgmlApi.GgmlTensor* t = GgmlApi.ggml_new_tensor_2d(ctx, GgmlApi.GGML_TYPE_F32, 3, 5);
                bool ok = t != null &&
                    t->type == GgmlApi.GGML_TYPE_F32 &&
                    t->ne[0] == 3 && t->ne[1] == 5 && t->ne[2] == 1 && t->ne[3] == 1 &&
                    t->nb[0] == 4 && t->nb[1] == 12 &&
                    GgmlApi.ggml_nbytes(t) == 60;
                if (!ok)
                    throw new InvalidOperationException(
                        "Managed GGML ABI validation failed: the GgmlTensor struct layout does not match the native ggml build. " +
                        "Update TensorSharp.Backends.GGML/Interop/GgmlApi.cs to match ExternalProjects/ggml/include/ggml.h.");
                s_abiValidated = true;
            }
            finally
            {
                GgmlApi.ggml_free(ctx);
            }
        }

        // ------------------------------------------------------------------
        // Context memory pool (mirrors ggml_pool in ggml_ops_core.cpp)
        // ------------------------------------------------------------------

        internal const nuint PoolBufferSize = 32 * 1024 * 1024;
        private const int PoolMaxCount = 32;

        private static readonly object s_poolLock = new object();
        private static readonly Stack<IntPtr> s_pool = new Stack<IntPtr>();

        internal struct PoolEntry
        {
            public IntPtr Ptr;
            public nuint Size;
        }

        internal static PoolEntry AcquirePoolEntry(nuint requiredSize)
        {
            if (requiredSize == 0 || requiredSize > PoolBufferSize)
                return default;
            lock (s_poolLock)
            {
                if (s_pool.Count > 0)
                    return new PoolEntry { Ptr = s_pool.Pop(), Size = PoolBufferSize };
            }
            IntPtr ptr = Marshal.AllocHGlobal((nint)PoolBufferSize);
            return new PoolEntry { Ptr = ptr, Size = PoolBufferSize };
        }

        internal static void ReleasePoolEntry(PoolEntry e)
        {
            if (e.Ptr == IntPtr.Zero)
                return;
            lock (s_poolLock)
            {
                if (s_pool.Count < PoolMaxCount)
                {
                    s_pool.Push(e.Ptr);
                    return;
                }
            }
            Marshal.FreeHGlobal(e.Ptr);
        }

        // RAII wrapper mirroring tsg::PooledContextHandle: a no-alloc ggml
        // context whose metadata arena comes from the block pool.
        internal struct PooledContext : IDisposable
        {
            public IntPtr Ctx;
            private PoolEntry _entry;

            public bool Init(nuint requiredSize)
            {
                _entry = AcquirePoolEntry(requiredSize);
                if (_entry.Ptr == IntPtr.Zero)
                    return false;
                var p = new GgmlApi.ggml_init_params
                {
                    mem_size = _entry.Size,
                    mem_buffer = _entry.Ptr,
                    no_alloc = true,
                };
                Ctx = GgmlApi.ggml_init(p);
                if (Ctx == IntPtr.Zero)
                {
                    ReleasePoolEntry(_entry);
                    _entry = default;
                    return false;
                }
                return true;
            }

            public void Dispose()
            {
                if (Ctx != IntPtr.Zero)
                {
                    GgmlApi.ggml_free(Ctx);
                    Ctx = IntPtr.Zero;
                }
                if (_entry.Ptr != IntPtr.Zero)
                {
                    ReleasePoolEntry(_entry);
                    _entry = default;
                }
            }
        }

        // Tracks per-op host-pointer wrapper buffers so they are freed after
        // compute (mirrors the std::vector<BufferHandle> host_ptr_buffers
        // pattern in the native impls).
        internal sealed class BufferList : IDisposable
        {
            private readonly List<IntPtr> _buffers = new List<IntPtr>(4);

            public void Add(IntPtr buffer)
            {
                _buffers.Add(buffer);
            }

            public void Dispose()
            {
                foreach (IntPtr b in _buffers)
                {
                    if (b != IntPtr.Zero)
                        GgmlApi.ggml_backend_buffer_free(b);
                }
                _buffers.Clear();
            }
        }

        // ------------------------------------------------------------------
        // Descriptor validation + layout queries (mirrors ggml_ops_core.cpp)
        // ------------------------------------------------------------------

        internal static nuint RequiredRawBytes(in GgmlTensorView4D d)
        {
            long maxOffset =
                (d.Ne0 - 1L) +
                (d.Ne1 - 1L) * (d.Nb1 / sizeof(float)) +
                (d.Ne2 - 1L) * (d.Nb2 / sizeof(float)) +
                (d.Ne3 - 1L) * (d.Nb3 / sizeof(float));
            return (nuint)((maxOffset + 1) * sizeof(float));
        }

        internal static nuint RequiredRawBytes(in GgmlTensorView2D d)
        {
            long maxOffset = (d.Dim0 - 1L) * d.Stride0 + (d.Dim1 - 1L) * d.Stride1;
            return (nuint)((maxOffset + 1) * sizeof(float));
        }

        internal static nuint RequiredRawBytes(in GgmlTensorView3D d)
        {
            long maxOffset = (d.Dim0 - 1L) * d.Stride0 + (d.Dim1 - 1L) * d.Stride1 + (d.Dim2 - 1L) * d.Stride2;
            return (nuint)((maxOffset + 1) * sizeof(float));
        }

        internal static nuint LogicalBytes(in GgmlTensorView4D d) =>
            (nuint)((long)d.Ne0 * d.Ne1 * d.Ne2 * d.Ne3 * sizeof(float));

        internal static nuint LogicalBytes(in GgmlTensorView2D d) =>
            (nuint)((long)d.Dim0 * d.Dim1 * sizeof(float));

        internal static nuint LogicalBytes(in GgmlTensorView3D d) =>
            (nuint)((long)d.Dim0 * d.Dim1 * d.Dim2 * sizeof(float));

        internal static bool ValidateDesc(in GgmlTensorView4D d, string name)
        {
            if (d.Data == IntPtr.Zero)
            {
                SetLastError($"Null pointer passed for {name}.");
                return false;
            }
            if (d.Ne0 <= 0 || d.Ne1 <= 0 || d.Ne2 <= 0 || d.Ne3 <= 0)
            {
                SetLastError($"Invalid tensor shape passed for {name}.");
                return false;
            }
            if (d.Nb1 <= 0 || d.Nb2 <= 0 || d.Nb3 <= 0)
            {
                SetLastError($"Invalid tensor strides passed for {name}.");
                return false;
            }
            if ((d.Nb1 % sizeof(float)) != 0 || (d.Nb2 % sizeof(float)) != 0 || (d.Nb3 % sizeof(float)) != 0)
            {
                SetLastError($"Tensor byte strides must be multiples of sizeof(float) for {name}.");
                return false;
            }
            if (d.RawBytes <= 0 || (d.RawBytes % sizeof(float)) != 0)
            {
                SetLastError($"Invalid raw byte size passed for {name}.");
                return false;
            }
            if ((nuint)d.RawBytes < RequiredRawBytes(d))
            {
                SetLastError($"Raw byte span is too small for {name}.");
                return false;
            }
            return true;
        }

        internal static bool ValidateDesc(in GgmlTensorView2D d, string name)
        {
            if (d.Data == IntPtr.Zero)
            {
                SetLastError($"Null pointer passed for {name}.");
                return false;
            }
            if (d.Dim0 <= 0 || d.Dim1 <= 0)
            {
                SetLastError($"Invalid tensor shape passed for {name}.");
                return false;
            }
            if (d.Stride0 < 0 || d.Stride1 < 0)
            {
                SetLastError($"Negative tensor strides are not supported for {name}.");
                return false;
            }
            if (d.RawBytes <= 0 || (d.RawBytes % sizeof(float)) != 0)
            {
                SetLastError($"Invalid raw byte size passed for {name}.");
                return false;
            }
            if ((nuint)d.RawBytes < RequiredRawBytes(d))
            {
                SetLastError($"Raw byte span is too small for {name}.");
                return false;
            }
            return true;
        }

        internal static bool ValidateDesc(in GgmlTensorView3D d, string name)
        {
            if (d.Data == IntPtr.Zero)
            {
                SetLastError($"Null pointer passed for {name}.");
                return false;
            }
            if (d.Dim0 <= 0 || d.Dim1 <= 0 || d.Dim2 <= 0)
            {
                SetLastError($"Invalid tensor shape passed for {name}.");
                return false;
            }
            if (d.Stride0 < 0 || d.Stride1 < 0 || d.Stride2 < 0)
            {
                SetLastError($"Negative tensor strides are not supported for {name}.");
                return false;
            }
            if (d.RawBytes <= 0 || (d.RawBytes % sizeof(float)) != 0)
            {
                SetLastError($"Invalid raw byte size passed for {name}.");
                return false;
            }
            if ((nuint)d.RawBytes < RequiredRawBytes(d))
            {
                SetLastError($"Raw byte span is too small for {name}.");
                return false;
            }
            return true;
        }

        internal static bool ValidateDesc(in GgmlContiguousTensor d, string name)
        {
            if (d.Data == IntPtr.Zero)
            {
                SetLastError($"Null pointer passed for {name}.");
                return false;
            }
            if (d.ElementCount <= 0)
            {
                SetLastError($"Invalid element count passed for {name}.");
                return false;
            }
            if (d.ElementType != 0 /* F32 */ && d.ElementType != 3 /* I32 */)
            {
                SetLastError($"Unsupported contiguous tensor element type passed for {name}.");
                return false;
            }
            return true;
        }

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

        internal static bool CanMapStandardView(in GgmlTensorView4D d)
        {
            long stride1 = d.Nb1 / sizeof(float);
            long stride2 = d.Nb2 / sizeof(float);
            long stride3 = d.Nb3 / sizeof(float);
            return IsNonOverlappingFastToSlow(
                stackalloc long[] { d.Ne0, d.Ne1, d.Ne2, d.Ne3 },
                stackalloc long[] { 1, stride1, stride2, stride3 });
        }

        internal static bool CanMapStandardView(in GgmlTensorView2D d)
        {
            return d.Stride1 == 1 && IsNonOverlappingFastToSlow(
                stackalloc long[] { d.Dim1, d.Dim0 },
                stackalloc long[] { d.Stride1, d.Stride0 });
        }

        internal static bool CanMapStandardView(in GgmlTensorView3D d)
        {
            return d.Stride2 == 1 && IsNonOverlappingFastToSlow(
                stackalloc long[] { d.Dim2, d.Dim1, d.Dim0 },
                stackalloc long[] { d.Stride2, d.Stride1, d.Stride0 });
        }

        // ------------------------------------------------------------------
        // Device props cache + host-pointer capability
        // (mirrors get_device_static_props / can_use_host_ptr_buffer)
        // ------------------------------------------------------------------

        private struct DeviceStaticProps
        {
            public int Type;
            public bool BufferFromHostPtr;
        }

        private static readonly Dictionary<IntPtr, DeviceStaticProps> s_devPropsCache = new Dictionary<IntPtr, DeviceStaticProps>();
        private static readonly object s_devPropsLock = new object();

        private static DeviceStaticProps GetDeviceStaticProps(IntPtr dev)
        {
            lock (s_devPropsLock)
            {
                if (s_devPropsCache.TryGetValue(dev, out DeviceStaticProps cached))
                    return cached;
                GgmlApi.ggml_backend_dev_props props = default;
                GgmlApi.ggml_backend_dev_get_props(dev, &props);
                var result = new DeviceStaticProps
                {
                    Type = props.type,
                    BufferFromHostPtr = props.caps.buffer_from_host_ptr,
                };
                s_devPropsCache[dev] = result;
                return result;
            }
        }

        internal static bool PrefersDeviceLocalCache(IntPtr dev)
        {
            if (dev == IntPtr.Zero)
                return false;
            int type = GetDeviceStaticProps(dev).Type;
            return type == GgmlApi.GGML_BACKEND_DEVICE_TYPE_GPU || type == GgmlApi.GGML_BACKEND_DEVICE_TYPE_IGPU;
        }

        private static nuint GetHostPtrAlignment(IntPtr dev)
        {
            if (dev != IntPtr.Zero)
            {
                IntPtr buft = GgmlApi.ggml_backend_dev_buffer_type(dev);
                if (buft != IntPtr.Zero)
                    return GgmlApi.ggml_backend_buft_get_alignment(buft);
            }
            return 16384;
        }

        private static bool IsPointerAligned(IntPtr ptr, nuint alignment) =>
            ptr != IntPtr.Zero && (alignment <= 1 || ((nuint)ptr % alignment) == 0);

        internal static bool CanUseHostPtrBuffer(IntPtr dev, IntPtr ptr, nuint size)
        {
            if (PrefersDeviceLocalCache(dev))
                return false;
            if (dev == IntPtr.Zero || ptr == IntPtr.Zero || size == 0)
                return false;
            if (!GetDeviceStaticProps(dev).BufferFromHostPtr)
                return false;
            return IsPointerAligned(ptr, GetHostPtrAlignment(dev));
        }

        // ------------------------------------------------------------------
        // Tensor bindings (mirrors create_standard_binding /
        // create_binding_from_host_ptr_* in ggml_ops_core.cpp)
        // ------------------------------------------------------------------

        internal struct TensorBinding
        {
            public GgmlApi.GgmlTensor* Storage;
            public GgmlApi.GgmlTensor* Tensor;
            public nuint RawBytes;

            public bool IsValid => Storage != null && Tensor != null;
        }

        internal static TensorBinding CreateStandardBinding(IntPtr ctx, in GgmlTensorView4D d)
        {
            GgmlApi.GgmlTensor* baseTensor = GgmlApi.ggml_new_tensor_1d(ctx, GgmlApi.GGML_TYPE_F32, d.RawBytes / sizeof(float));
            if (baseTensor == null)
                return default;
            GgmlApi.GgmlTensor* view = GgmlApi.ggml_view_4d(ctx, baseTensor, d.Ne0, d.Ne1, d.Ne2, d.Ne3,
                (nuint)d.Nb1, (nuint)d.Nb2, (nuint)d.Nb3, 0);
            if (view == null)
                return default;
            return new TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = (nuint)d.RawBytes };
        }

        internal static TensorBinding CreateStandardBinding(IntPtr ctx, in GgmlTensorView2D d)
        {
            GgmlApi.GgmlTensor* baseTensor = GgmlApi.ggml_new_tensor_1d(ctx, GgmlApi.GGML_TYPE_F32, d.RawBytes / sizeof(float));
            if (baseTensor == null)
                return default;
            GgmlApi.GgmlTensor* view = GgmlApi.ggml_view_2d(ctx, baseTensor, d.Dim1, d.Dim0,
                (nuint)((long)d.Stride0 * sizeof(float)), 0);
            if (view == null)
                return default;
            return new TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = (nuint)d.RawBytes };
        }

        internal static TensorBinding CreateStandardBinding(IntPtr ctx, in GgmlTensorView3D d)
        {
            GgmlApi.GgmlTensor* baseTensor = GgmlApi.ggml_new_tensor_1d(ctx, GgmlApi.GGML_TYPE_F32, d.RawBytes / sizeof(float));
            if (baseTensor == null)
                return default;
            GgmlApi.GgmlTensor* view = GgmlApi.ggml_view_3d(ctx, baseTensor, d.Dim2, d.Dim1, d.Dim0,
                (nuint)((long)d.Stride1 * sizeof(float)),
                (nuint)((long)d.Stride0 * sizeof(float)), 0);
            if (view == null)
                return default;
            return new TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = (nuint)d.RawBytes };
        }

        internal static TensorBinding CreateContiguousBinding(IntPtr ctx, in GgmlContiguousTensor d)
        {
            GgmlApi.GgmlTensor* tensor = GgmlApi.ggml_new_tensor_1d(ctx, GgmlApi.GGML_TYPE_F32, d.ElementCount);
            if (tensor == null)
                return default;
            return new TensorBinding { Storage = tensor, Tensor = tensor, RawBytes = (nuint)(d.ElementCount * sizeof(float)) };
        }

        private static bool TryWrapHostPtr(IntPtr ctx, IntPtr data, nuint rawBytes, out GgmlApi.GgmlTensor* baseTensor, out IntPtr buffer)
        {
            baseTensor = null;
            buffer = IntPtr.Zero;

            IntPtr dev = GgmlApi.ggml_backend_get_device(s_backend);
            if (!CanUseHostPtrBuffer(dev, data, rawBytes))
                return false;

            buffer = GgmlApi.ggml_backend_dev_buffer_from_host_ptr(dev, data, rawBytes, rawBytes);
            if (buffer == IntPtr.Zero)
                return false;

            baseTensor = GgmlApi.ggml_new_tensor_1d(ctx, GgmlApi.GGML_TYPE_F32, (long)(rawBytes / sizeof(float)));
            if (baseTensor == null)
            {
                GgmlApi.ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }

            if (GgmlApi.ggml_backend_tensor_alloc(buffer, baseTensor, data) != GgmlApi.GGML_STATUS_SUCCESS)
            {
                GgmlApi.ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                baseTensor = null;
                return false;
            }

            return true;
        }

        internal static bool CreateBindingFromHostPtr(IntPtr ctx, in GgmlTensorView4D d, out TensorBinding binding, out IntPtr buffer)
        {
            binding = default;
            if (!TryWrapHostPtr(ctx, d.Data, (nuint)d.RawBytes, out GgmlApi.GgmlTensor* baseTensor, out buffer))
                return false;
            GgmlApi.GgmlTensor* view = GgmlApi.ggml_view_4d(ctx, baseTensor, d.Ne0, d.Ne1, d.Ne2, d.Ne3,
                (nuint)d.Nb1, (nuint)d.Nb2, (nuint)d.Nb3, 0);
            if (view == null)
            {
                GgmlApi.ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }
            binding = new TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = (nuint)d.RawBytes };
            return true;
        }

        internal static bool CreateBindingFromHostPtr(IntPtr ctx, in GgmlTensorView2D d, out TensorBinding binding, out IntPtr buffer)
        {
            binding = default;
            if (!TryWrapHostPtr(ctx, d.Data, (nuint)d.RawBytes, out GgmlApi.GgmlTensor* baseTensor, out buffer))
                return false;
            GgmlApi.GgmlTensor* view = GgmlApi.ggml_view_2d(ctx, baseTensor, d.Dim1, d.Dim0,
                (nuint)((long)d.Stride0 * sizeof(float)), 0);
            if (view == null)
            {
                GgmlApi.ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }
            binding = new TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = (nuint)d.RawBytes };
            return true;
        }

        internal static bool CreateBindingFromHostPtr(IntPtr ctx, in GgmlTensorView3D d, out TensorBinding binding, out IntPtr buffer)
        {
            binding = default;
            if (!TryWrapHostPtr(ctx, d.Data, (nuint)d.RawBytes, out GgmlApi.GgmlTensor* baseTensor, out buffer))
                return false;
            GgmlApi.GgmlTensor* view = GgmlApi.ggml_view_3d(ctx, baseTensor, d.Dim2, d.Dim1, d.Dim0,
                (nuint)((long)d.Stride1 * sizeof(float)),
                (nuint)((long)d.Stride0 * sizeof(float)), 0);
            if (view == null)
            {
                GgmlApi.ggml_backend_buffer_free(buffer);
                buffer = IntPtr.Zero;
                return false;
            }
            binding = new TensorBinding { Storage = baseTensor, Tensor = view, RawBytes = (nuint)d.RawBytes };
            return true;
        }

        internal static bool CreateBindingFromHostPtr(IntPtr ctx, in GgmlContiguousTensor d, out TensorBinding binding, out IntPtr buffer)
        {
            binding = default;
            nuint rawBytes = (nuint)(d.ElementCount * sizeof(float));
            if (!TryWrapHostPtr(ctx, d.Data, rawBytes, out GgmlApi.GgmlTensor* baseTensor, out buffer))
                return false;
            binding = new TensorBinding { Storage = baseTensor, Tensor = baseTensor, RawBytes = rawBytes };
            return true;
        }

        // ------------------------------------------------------------------
        // Upload / finalize (mirrors upload_binding / finalize_compute)
        // ------------------------------------------------------------------

        internal static void UploadBinding(in TensorBinding binding, IntPtr data, nuint size)
        {
            // Mirror the native path: drain any deferred (Metal lazy-sync) GPU
            // work before a host->buffer memcpy that could race a pending
            // command buffer. Cross-path safe because the barrier and the
            // pending-work flag live in the shared native layer.
            GgmlNative.HostReadBarrier();
            GgmlApi.ggml_backend_tensor_set(binding.Storage, data, 0, size);
        }

        // Standard end-of-op finalisation. The managed layer always uses the
        // eager model: synchronize, then download when the result was not
        // bound zero-copy. The native Metal lazy-sync optimisation (deferring
        // the sync and marking g_pending_gpu_work) is not ported yet — that
        // flag lives native-side with no setter export, so managed ops on
        // Metal pay one extra sync per op until the Metal pass ports it.
        // Correctness is unaffected (eager is strictly stronger ordering).
        internal static void FinalizeCompute(bool resultUsedZeroCopy, in TensorBinding resultBinding, IntPtr resultData, nuint resultBytes)
        {
            GgmlApi.ggml_backend_synchronize(s_backend);
            if (!resultUsedZeroCopy && resultBinding.Storage != null && resultData != IntPtr.Zero && resultBytes > 0)
            {
                GgmlApi.ggml_backend_tensor_get(resultBinding.Storage, resultData, 0, resultBytes);
            }
        }

        // ------------------------------------------------------------------
        // Shape helpers (mirrors ggml_ops_internal.h)
        // ------------------------------------------------------------------

        internal static bool SameShape(in GgmlTensorView4D a, in GgmlTensorView4D b) =>
            a.Ne0 == b.Ne0 && a.Ne1 == b.Ne1 && a.Ne2 == b.Ne2 && a.Ne3 == b.Ne3;

        internal static bool SameShapeWithLastDimReduced(in GgmlTensorView4D result, in GgmlTensorView4D src) =>
            result.Ne0 == 1 && result.Ne1 == src.Ne1 && result.Ne2 == src.Ne2 && result.Ne3 == src.Ne3;

        internal static long FlatRowCount(in GgmlTensorView4D d) => (long)d.Ne1 * d.Ne2 * d.Ne3;
    }
}
