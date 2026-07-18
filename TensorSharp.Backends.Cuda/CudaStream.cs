using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaStream : IDisposable
    {
        private IntPtr stream;

        private CudaStream(IntPtr stream)
        {
            this.stream = stream;
        }

        public IntPtr Handle => stream;

        public static CudaStream Create()
        {
            CudaDriverApi.cuStreamCreate(out IntPtr stream, 0).ThrowOnError();
            return new CudaStream(stream);
        }

        public void Synchronize()
        {
            if (stream == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaStream));

            // A synchronize on a stream that is capturing a CUDA graph means some
            // op needs device data on the host mid-capture; that cannot be part of
            // a graph. Abort the capture (the site catches this, re-runs plainly).
            CudaGraphCapture.OnStreamSynchronize(stream);

            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
        }

        public void Dispose()
        {
            IntPtr handle = Interlocked.Exchange(ref stream, IntPtr.Zero);
            if (handle != IntPtr.Zero)
                CudaDriverApi.cuStreamDestroy(handle);
        }
    }
}
