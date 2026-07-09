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
// Shared utilities for the parity harness: result dump writer, aligned native
// buffers, deterministic fills, timing, and descriptor construction. Each op
// module contributes a Cases*.cs file built on these.
using System;
using System.Diagnostics;
using System.IO;
using TensorSharp.GGML;

internal static class Harness
{
    public const int TimingIterations = 20;
    public const int WarmupIterations = 3;

    public sealed class Dump : IDisposable
    {
        private readonly BinaryWriter _writer;

        public Dump(string path) => _writer = new BinaryWriter(File.Create(path));

        public void Add(string name, ReadOnlySpan<byte> data, double ns)
        {
            _writer.Write(name);
            _writer.Write(ns);
            _writer.Write(data.Length);
            _writer.Write(data);
        }

        public void Dispose() => _writer.Dispose();
    }

    // A page-aligned native buffer wrapper (64-byte alignment satisfies the
    // CPU backend's host-pointer requirements so the zero-copy path engages).
    public sealed unsafe class AlignedBuffer : IDisposable
    {
        public IntPtr Ptr { get; }
        public long Bytes { get; }

        public AlignedBuffer(long bytes)
        {
            Bytes = bytes;
            Ptr = (IntPtr)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)bytes, 4096);
        }

        public Span<float> Floats => new Span<float>((void*)Ptr, (int)(Bytes / sizeof(float)));
        public Span<byte> RawBytes => new Span<byte>((void*)Ptr, (int)Bytes);

        public void Dispose() => System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)Ptr);
    }

    private static int _seed = 12345;

    public static void FillDeterministic(Span<float> data, float offset = 0f, float scale = 1f)
    {
        // xorshift-based deterministic fill, identical across both child modes.
        uint state = (uint)System.Threading.Interlocked.Increment(ref _seed) * 2654435761u;
        for (int i = 0; i < data.Length; ++i)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            float unit = (state & 0xFFFFFF) / (float)0x1000000; // [0, 1)
            data[i] = ((unit * 2f) - 1f) * scale + offset;
        }
    }

    internal static GgmlTensorView4D Contiguous4D(IntPtr data, int ne0, int ne1, int ne2, int ne3)
    {
        long nb1 = (long)ne0 * sizeof(float);
        long nb2 = nb1 * ne1;
        long nb3 = nb2 * ne2;
        long rawBytes = nb3 * ne3;
        return new GgmlTensorView4D(data, ne0, ne1, ne2, ne3, nb1, nb2, nb3, rawBytes);
    }

    public static long Bytes4D(int ne0, int ne1, int ne2, int ne3) => (long)ne0 * ne1 * ne2 * ne3 * sizeof(float);

    public static double TimeCase(Action action)
    {
        for (int i = 0; i < WarmupIterations; ++i)
            action();
        var laps = new double[TimingIterations];
        for (int i = 0; i < TimingIterations; ++i)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            laps[i] = (Stopwatch.GetTimestamp() - start) * 1e9 / Stopwatch.Frequency;
        }
        Array.Sort(laps);
        return laps[laps.Length / 2];
    }

}
