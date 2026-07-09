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
using System.Threading;

namespace TensorSharp.GGML.Interop
{
    // Managed port of the weight-binding caches in ggml_ops_core.cpp:
    // g_host_buffer_cache (host-pointer wraps and device-local copies keyed by
    // the host data pointer), g_preloaded_buffer_cache (device copies uploaded
    // at model load, keyed by an opaque cache key), and the device-copy VRAM
    // budget. Managed op implementations bind stable weight pointers through
    // TryGetCacheableTensorBuffer so repeated calls reuse the same backend
    // buffer instead of re-uploading per graph.
    //
    // Transition note: while native TSGgml_* kernels coexist with managed ops,
    // this cache is separate from the native one. A weight bound by BOTH paths
    // on a discrete-GPU backend is resident twice; the invalidation entry
    // points (Invalidate/Clear) are routed to both caches so correctness is
    // unaffected. The duplication disappears as modules finish migrating.
    // The native MoE offloadable LRU (TSGgml_RegisterOffloadable) is not
    // ported yet: entries here are treated as non-offloadable, which only
    // affects the Metal MoE offload-budget optimisation, not correctness.
    internal static unsafe class GgmlManagedWeightCache
    {
        private enum CachedBufferMode
        {
            HostPtr,
            DeviceCopy,
        }

        private struct CachedHostBuffer
        {
            public IntPtr Buffer;
            public nuint Bytes;
            public nuint BufferSize;
            public CachedBufferMode Mode;
        }

        private static readonly object s_lock = new object();
        private static readonly Dictionary<IntPtr, CachedHostBuffer> s_hostBufferCache = new Dictionary<IntPtr, CachedHostBuffer>();
        private static readonly Dictionary<IntPtr, CachedHostBuffer> s_preloadedBufferCache = new Dictionary<IntPtr, CachedHostBuffer>();

        private static long s_deviceCopyResidentBytes;
        private static long s_deviceCopyBudgetBytes;

        internal static void SetDeviceCopyBudget(long budgetBytes)
        {
            Interlocked.Exchange(ref s_deviceCopyBudgetBytes, budgetBytes);
        }

        // Mirrors try_get_cacheable_tensor_buffer (ggml_ops_core.cpp:980).
        // On success the caller binds `tensor` at `addr` inside `buffer` via
        // ggml_backend_tensor_alloc; `needsUpload` means the caller must
        // upload the host bytes into the (new device-copy) buffer once.
        internal static bool TryGetCacheableTensorBuffer(
            GgmlApi.GgmlTensor* tensor,
            IntPtr data,
            nuint bytes,
            out IntPtr buffer,
            out IntPtr addr,
            out bool needsUpload)
        {
            buffer = IntPtr.Zero;
            addr = IntPtr.Zero;
            needsUpload = false;

            IntPtr backend = GgmlManagedRuntime.Backend;
            if (backend == IntPtr.Zero || tensor == null || data == IntPtr.Zero || bytes == 0)
                return false;

            IntPtr dev = GgmlApi.ggml_backend_get_device(backend);
            if (dev == IntPtr.Zero)
                return false;

            // The unified-memory Metal weight wrap (unified_weight) is not
            // ported yet — on Metal the host-ptr capability check below covers
            // the zero-copy case for the per-op path.
            bool useDeviceCopy = GgmlManagedRuntime.PrefersDeviceLocalCache(dev);

            lock (s_lock)
            {
                if (s_preloadedBufferCache.TryGetValue(data, out CachedHostBuffer preloaded))
                {
                    nuint requiredSize = GgmlApi.ggml_backend_buffer_get_alloc_size(preloaded.Buffer, tensor);
                    if (preloaded.Bytes == bytes && requiredSize <= preloaded.BufferSize)
                    {
                        buffer = preloaded.Buffer;
                        addr = GgmlApi.ggml_backend_buffer_get_base(buffer);
                        return true;
                    }
                    GgmlApi.ggml_backend_buffer_free(preloaded.Buffer);
                    s_preloadedBufferCache.Remove(data);
                }

                if (s_hostBufferCache.TryGetValue(data, out CachedHostBuffer cached))
                {
                    bool modeMatches =
                        (useDeviceCopy && cached.Mode == CachedBufferMode.DeviceCopy) ||
                        (!useDeviceCopy && cached.Mode == CachedBufferMode.HostPtr);
                    nuint requiredSize = GgmlApi.ggml_backend_buffer_get_alloc_size(cached.Buffer, tensor);
                    if (modeMatches && cached.Bytes == bytes && requiredSize <= cached.BufferSize)
                    {
                        buffer = cached.Buffer;
                        addr = useDeviceCopy ? GgmlApi.ggml_backend_buffer_get_base(buffer) : data;
                        return true;
                    }
                    RemoveHostCacheEntryLocked(data, cached);
                }
            }

            if (useDeviceCopy)
            {
                IntPtr buft = GgmlApi.ggml_backend_get_default_buffer_type(backend);
                if (buft == IntPtr.Zero)
                    return false;
                nuint allocSize = GgmlApi.ggml_backend_buft_get_alloc_size(buft, tensor);

                lock (s_lock)
                {
                    long budget = Interlocked.Read(ref s_deviceCopyBudgetBytes);
                    if (budget > 0 && s_deviceCopyResidentBytes + (long)allocSize > budget)
                        return false;
                }

                buffer = GgmlApi.ggml_backend_buft_alloc_buffer(buft, allocSize);
                if (buffer == IntPtr.Zero)
                    return false;
                GgmlApi.ggml_backend_buffer_set_usage(buffer, GgmlApi.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                addr = GgmlApi.ggml_backend_buffer_get_base(buffer);
                needsUpload = true;

                lock (s_lock)
                {
                    nuint bufferSize = GgmlApi.ggml_backend_buffer_get_size(buffer);
                    s_hostBufferCache[data] = new CachedHostBuffer
                    {
                        Buffer = buffer,
                        Bytes = bytes,
                        BufferSize = bufferSize,
                        Mode = CachedBufferMode.DeviceCopy,
                    };
                    s_deviceCopyResidentBytes += (long)bufferSize;
                }
                return true;
            }

            // Host-pointer wrap path (mirrors try_get_host_ptr_buffer with
            // cacheable=true).
            if (!GgmlManagedRuntime.CanUseHostPtrBuffer(dev, data, bytes))
                return false;

            buffer = GgmlApi.ggml_backend_dev_buffer_from_host_ptr(dev, data, bytes, bytes);
            if (buffer == IntPtr.Zero)
                return false;

            lock (s_lock)
            {
                s_hostBufferCache[data] = new CachedHostBuffer
                {
                    Buffer = buffer,
                    Bytes = bytes,
                    BufferSize = GgmlApi.ggml_backend_buffer_get_size(buffer),
                    Mode = CachedBufferMode.HostPtr,
                };
            }
            addr = data;
            return true;
        }

        // Mirrors TSGgml_PreloadQuantizedWeight (ggml_ops_core.cpp:2218).
        // Returns 1 on success (or when the backend keeps weights host-side),
        // 2 when the weight exceeds the backend's max single-buffer size (the
        // caller keeps its host copy), 0 on failure.
        internal static int PreloadQuantizedWeight(IntPtr cacheKey, IntPtr hostData, int ggmlType, long ne0, long ne1, long rawBytes)
        {
            if (!GgmlManagedRuntime.EnsureBackend())
                return 0;

            if (cacheKey == IntPtr.Zero || hostData == IntPtr.Zero || ne0 <= 0 || ne1 <= 0 || rawBytes <= 0)
            {
                GgmlManagedRuntime.SetLastError("Invalid arguments for quantized weight preload.");
                return 0;
            }

            IntPtr backend = GgmlManagedRuntime.Backend;
            IntPtr dev = GgmlApi.ggml_backend_get_device(backend);
            if (dev == IntPtr.Zero)
            {
                GgmlManagedRuntime.SetLastError("No GGML backend device is available for quantized weight preload.");
                return 0;
            }

            if (!GgmlManagedRuntime.PrefersDeviceLocalCache(dev))
            {
                GgmlManagedRuntime.ClearLastError();
                return 1;
            }

            nuint bytes = (nuint)rawBytes;
            var context = new GgmlManagedRuntime.PooledContext();
            if (!context.Init(64 * 1024))
            {
                GgmlManagedRuntime.SetLastError("Failed to create GGML context for quantized weight preload.");
                return 0;
            }
            using (context)
            {
                GgmlApi.GgmlTensor* tensor = GgmlApi.ggml_new_tensor_2d(context.Ctx, ggmlType, ne0, ne1);
                if (tensor == null)
                {
                    GgmlManagedRuntime.SetLastError("Failed to create GGML tensor for quantized weight preload.");
                    return 0;
                }

                lock (s_lock)
                {
                    if (s_preloadedBufferCache.TryGetValue(cacheKey, out CachedHostBuffer existing))
                    {
                        nuint requiredSize = GgmlApi.ggml_backend_buffer_get_alloc_size(existing.Buffer, tensor);
                        if (existing.Bytes == bytes && requiredSize <= existing.BufferSize)
                        {
                            GgmlManagedRuntime.ClearLastError();
                            return 1;
                        }
                        GgmlApi.ggml_backend_buffer_free(existing.Buffer);
                        s_preloadedBufferCache.Remove(cacheKey);
                    }
                }

                IntPtr buft = GgmlApi.ggml_backend_get_default_buffer_type(backend);
                if (buft == IntPtr.Zero)
                {
                    GgmlManagedRuntime.SetLastError("Failed to get GGML backend buffer type for quantized weight preload.");
                    return 0;
                }

                nuint allocSize = GgmlApi.ggml_backend_buft_get_alloc_size(buft, tensor);
                IntPtr buffer = GgmlApi.ggml_backend_buft_alloc_buffer(buft, allocSize);
                if (buffer == IntPtr.Zero)
                {
                    if (allocSize > GgmlApi.ggml_backend_buft_get_max_size(buft))
                    {
                        GgmlManagedRuntime.ClearLastError();
                        return 2;
                    }
                    GgmlManagedRuntime.SetLastError("Failed to allocate GGML backend buffer for quantized weight preload.");
                    return 0;
                }

                GgmlApi.ggml_backend_buffer_set_usage(buffer, GgmlApi.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                IntPtr addr = GgmlApi.ggml_backend_buffer_get_base(buffer);
                if (addr == IntPtr.Zero)
                {
                    GgmlApi.ggml_backend_buffer_free(buffer);
                    GgmlManagedRuntime.SetLastError("Failed to get GGML backend buffer base for quantized weight preload.");
                    return 0;
                }

                if (GgmlApi.ggml_backend_tensor_alloc(buffer, tensor, addr) != GgmlApi.GGML_STATUS_SUCCESS)
                {
                    GgmlApi.ggml_backend_buffer_free(buffer);
                    GgmlManagedRuntime.SetLastError("Failed to bind GGML tensor to backend buffer during quantized weight preload.");
                    return 0;
                }

                GgmlApi.ggml_backend_tensor_set(tensor, hostData, 0, bytes);
                GgmlApi.ggml_backend_synchronize(backend);

                lock (s_lock)
                {
                    s_preloadedBufferCache[cacheKey] = new CachedHostBuffer
                    {
                        Buffer = buffer,
                        Bytes = bytes,
                        BufferSize = GgmlApi.ggml_backend_buffer_get_size(buffer),
                        Mode = CachedBufferMode.DeviceCopy,
                    };
                }

                GgmlManagedRuntime.ClearLastError();
                return 1;
            }
        }

        // Mirrors invalidate_cached_buffer: drops any cache entry keyed by the
        // given host pointer (called when the managed side frees or reuses the
        // memory).
        internal static void Invalidate(IntPtr data)
        {
            if (data == IntPtr.Zero)
                return;
            lock (s_lock)
            {
                if (s_preloadedBufferCache.TryGetValue(data, out CachedHostBuffer preloaded))
                {
                    GgmlApi.ggml_backend_buffer_free(preloaded.Buffer);
                    s_preloadedBufferCache.Remove(data);
                    return;
                }
                if (s_hostBufferCache.TryGetValue(data, out CachedHostBuffer cached))
                {
                    RemoveHostCacheEntryLocked(data, cached);
                }
            }
        }

        internal static void Clear()
        {
            lock (s_lock)
            {
                foreach (CachedHostBuffer entry in s_hostBufferCache.Values)
                    GgmlApi.ggml_backend_buffer_free(entry.Buffer);
                s_hostBufferCache.Clear();
                foreach (CachedHostBuffer entry in s_preloadedBufferCache.Values)
                    GgmlApi.ggml_backend_buffer_free(entry.Buffer);
                s_preloadedBufferCache.Clear();
                s_deviceCopyResidentBytes = 0;
            }
        }

        private static void RemoveHostCacheEntryLocked(IntPtr data, in CachedHostBuffer entry)
        {
            if (entry.Mode == CachedBufferMode.DeviceCopy)
            {
                s_deviceCopyResidentBytes = s_deviceCopyResidentBytes >= (long)entry.BufferSize
                    ? s_deviceCopyResidentBytes - (long)entry.BufferSize
                    : 0;
            }
            GgmlApi.ggml_backend_buffer_free(entry.Buffer);
            s_hostBufferCache.Remove(data);
        }
    }
}
