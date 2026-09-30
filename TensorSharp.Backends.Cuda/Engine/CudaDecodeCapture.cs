// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Stream capture of one device's share of a whole-model engine's decode step. The engine
    /// replays the graph with its inputs refreshed in pinned memory, so the graph must reference
    /// only buffers the engine owns for as long as it keeps the graph. Capture fails, with nothing
    /// executed, when the step does anything else inside it: synchronizes, allocates or returns
    /// device memory through the allocator (the pool could hand the block to someone else), or
    /// takes a route that reads shared scratch (<see cref="Refuse"/>).
    /// </summary>
    internal static class CudaDecodeCapture
    {
        private const int ModeThreadLocal = 1;

        [ThreadStatic]
        private static bool t_active;

        /// <summary>Capture what <paramref name="issue"/> queues on <paramref name="alloc"/>'s stream and
        /// instantiate it. Throws when the step is not capturable.</summary>
        public static IntPtr Capture(CudaAllocator alloc, Action issue)
        {
            CudaGraphCapture.CaptureContext ctx = CudaGraphCapture.Begin(alloc)
                ?? throw new InvalidOperationException("another CUDA graph capture is in progress");
            IntPtr stream = alloc.Stream.Handle;
            IntPtr graph;
            try
            {
                CudaDriverApi.cuStreamBeginCapture(stream, ModeThreadLocal).ThrowOnError();
                t_active = true;
                try
                {
                    issue();
                }
                catch
                {
                    t_active = false;
                    if (CudaDriverApi.cuStreamEndCapture(stream, out IntPtr broken) == 0 && broken != IntPtr.Zero)
                        CudaDriverApi.cuGraphDestroy(broken);
                    throw;
                }
                t_active = false;
                CudaDriverApi.cuStreamEndCapture(stream, out graph).ThrowOnError();
            }
            finally
            {
                CudaGraphCapture.End(ctx);
            }
            try
            {
                if (ctx.TrackedBlocks.Count > 0 || ctx.QuarantinedBlocks.Count > 0)
                    throw new InvalidOperationException(
                        $"the step allocated device memory ({ctx.TrackedBlocks.Count + ctx.QuarantinedBlocks.Count} block(s))");
                CudaDriverApi.cuGraphInstantiateWithFlags(out IntPtr exec, graph, 0).ThrowOnError();
                return exec;
            }
            finally
            {
                CudaDriverApi.cuGraphDestroy(graph);
            }
        }

        /// <summary>Throw inside a capture: <paramref name="route"/> would bake shared scratch into it.</summary>
        public static void Refuse(string route)
        {
            if (t_active)
                throw new InvalidOperationException($"{route} is not capturable (it may use shared scratch)");
        }
    }
}
