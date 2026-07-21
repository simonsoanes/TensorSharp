using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;

namespace TensorSharp.Cuda
{
    // Lightweight env-gated diagnostics (TS_CUDA_PROFILE=1): counts CPU-fallback op
    // invocations and host<->device mirror syncs so backend-level stalls in the
    // per-op model loop are attributable without an external profiler.
    internal static class CudaProfileCounters
    {
        internal static readonly int Level =
            int.TryParse(Environment.GetEnvironmentVariable("TS_CUDA_PROFILE"), out int lvl) ? lvl : 0;

        internal static readonly bool Enabled = Level >= 1;

        private sealed class Stat { public long Count; public long Ticks; public long Bytes; }

        private static readonly ConcurrentDictionary<string, Stat> Fallbacks = new();
        private static readonly ConcurrentDictionary<string, Stat> Syncs = new();

        static CudaProfileCounters()
        {
            if (Enabled)
                AppDomain.CurrentDomain.ProcessExit += (_, _) => Dump();
        }

        internal static void RecordFallback(string opName, long ticks)
        {
            var s = Fallbacks.GetOrAdd(opName, _ => new Stat());
            System.Threading.Interlocked.Increment(ref s.Count);
            System.Threading.Interlocked.Add(ref s.Ticks, ticks);
        }

        internal static void RecordSync(string kind, long bytes, long ticks)
        {
            if (Level >= 2)
            {
                var trace = new StackTrace(2, false);
                int max = Math.Min(trace.FrameCount, 14);
                int taken = 0;
                for (int i = 0; i < max && taken < 3; i++)
                {
                    var m = trace.GetFrame(i)?.GetMethod();
                    string typeName = m?.DeclaringType?.Name ?? "?";
                    if (typeName.Contains("Storage") || typeName.Contains("Tensor") || typeName.Contains("ComputePrimitives"))
                        continue;
                    kind = kind + " <- " + typeName + "." + (m?.Name ?? "?");
                    taken++;
                }
            }
            var s = Syncs.GetOrAdd(kind, _ => new Stat());
            System.Threading.Interlocked.Increment(ref s.Count);
            System.Threading.Interlocked.Add(ref s.Ticks, ticks);
            System.Threading.Interlocked.Add(ref s.Bytes, bytes);
        }

        internal static void Dump()
        {
            Console.WriteLine("[cuda-profile] CPU fallback ops:");
            foreach (var kv in Fallbacks.OrderByDescending(kv => kv.Value.Ticks))
                Console.WriteLine($"  {kv.Key,-32} {kv.Value.Count,8} calls  {kv.Value.Ticks * 1000.0 / Stopwatch.Frequency,10:F1} ms");
            Console.WriteLine("[cuda-profile] host<->device syncs:");
            foreach (var kv in Syncs.OrderByDescending(kv => kv.Value.Ticks))
                Console.WriteLine($"  {kv.Key,-32} {kv.Value.Count,8} calls  {kv.Value.Ticks * 1000.0 / Stopwatch.Frequency,10:F1} ms  {kv.Value.Bytes / (1024.0 * 1024.0),10:F1} MB");
        }
    }
}
