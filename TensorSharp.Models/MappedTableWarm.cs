// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>
    /// Warms host-mapped lookup tables into the page cache. A table read a few scattered rows per
    /// token (DeepSeek V4.1's Engram tables, Qwen3.8-Flash-Next's PLE n-gram table) pays a storage
    /// round trip for every row whose page is not cached: about a millisecond on a network
    /// filesystem, so a cold table can cost several times the decode step. Reading the tables once,
    /// on a background thread after the model is ready, takes that off the requests.
    /// </summary>
    internal static class MappedTableWarm
    {
        private const long Block = 64L << 20;
        private const long Headroom = 8L << 30;

        /// <summary>
        /// Start reading <paramref name="ranges"/> (shard path, byte offset, byte count), or return null
        /// with a note when the host has no room to keep them cached. The task completes when every
        /// block has been read, <paramref name="cancel"/> fires, or a read fails (reported, not thrown).
        /// </summary>
        public static Task Start(IReadOnlyList<(string Path, long Offset, long Bytes)> ranges, string tag, string what,
            CancellationToken cancel)
        {
            if (ranges == null || ranges.Count == 0)
                return null;
            long bytes = 0;
            foreach (var range in ranges)
                bytes += range.Bytes;
            long available = HostMemoryAvailable();
            if (available > 0 && available < bytes + Headroom)
            {
                Console.Error.WriteLine($"[{tag}] {what} warming skipped: {available / (double)(1L << 30):F1} GiB " +
                    $"of host memory available would not keep {bytes / (double)(1L << 30):F1} GiB of tables cached");
                return null;
            }
            return Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                var blocks = new List<(string Path, long Offset, long Bytes)>();
                foreach (var (path, offset, length) in ranges)
                    for (long at = 0; at < length; at += Block)
                        blocks.Add((path, offset + at, Math.Min(Block, length - at)));
                using var handles = new ThreadLocal<Dictionary<string, Microsoft.Win32.SafeHandles.SafeFileHandle>>(
                    () => new Dictionary<string, Microsoft.Win32.SafeHandles.SafeFileHandle>(), trackAllValues: true);
                using var buffers = new ThreadLocal<byte[]>(() => new byte[Block]);
                try
                {
                    Parallel.ForEach(blocks, new ParallelOptions { MaxDegreeOfParallelism = 16, CancellationToken = cancel }, block =>
                    {
                        var open = handles.Value;
                        if (!open.TryGetValue(block.Path, out var handle))
                            open[block.Path] = handle = File.OpenHandle(block.Path, FileMode.Open, FileAccess.Read, FileShare.Read);
                        RandomAccess.Read(handle, new Span<byte>(buffers.Value, 0, (int)block.Bytes), block.Offset);
                    });
                    Console.Error.WriteLine($"[{tag}] warmed {bytes / (double)(1L << 30):F1} GiB of {what} in {sw.Elapsed.TotalSeconds:F1}s");
                }
                catch (OperationCanceledException) { }
                catch (IOException ex)
                {
                    Console.Error.WriteLine($"[{tag}] {what} warming stopped: {ex.Message}");
                }
                finally
                {
                    foreach (var open in handles.Values)
                        foreach (var handle in open.Values)
                            handle.Dispose();
                }
            });
        }

        /// <summary>MemAvailable from /proc/meminfo in bytes, or 0 where there is none.</summary>
        public static long HostMemoryAvailable()
        {
            try
            {
                foreach (string line in File.ReadLines("/proc/meminfo"))
                {
                    if (!line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                        continue;
                    string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return long.Parse(parts[1]) * 1024;
                }
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            return 0;
        }
    }
}
