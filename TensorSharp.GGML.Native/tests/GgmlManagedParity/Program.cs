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
// Parity + performance harness for the managed re-implementations of the
// GgmlOps kernels (TS_GGML_MANAGED_OPS). Because the managed/native switch is
// read once per process, the parent process spawns itself twice per backend
// (native mode and managed mode); each child runs an identical deterministic
// op battery, writing every result buffer and per-case timing to a dump file.
// The parent then compares the dumps byte-for-byte and reports timing ratios.
//
// Usage:
//   GgmlManagedParity <cpu|cuda|vulkan>            parent: run both modes + compare
//   GgmlManagedParity capture <backend> <outfile>  child: run battery, write dump
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using TensorSharp.GGML;

internal static class Program
{
    private const int TimingIterations = 20;
    private const int WarmupIterations = 3;

    private static int Main(string[] args)
    {
        if (args.Length >= 3 && args[0] == "capture")
        {
            return Capture(args[1], args[2]);
        }

        string backend = args.Length > 0 ? args[0] : "cpu";
        return RunParent(backend);
    }

    // ------------------------------------------------------------------
    // Parent: spawn native + managed children, compare dumps
    // ------------------------------------------------------------------

    private static int RunParent(string backend)
    {
        string exePath = Environment.ProcessPath;
        string dir = Path.Combine(Path.GetTempPath(), "ggml-managed-parity");
        Directory.CreateDirectory(dir);
        string nativeDump = Path.Combine(dir, $"native-{backend}.bin");
        string managedDump = Path.Combine(dir, $"managed-{backend}.bin");

        if (!RunChild(exePath, backend, nativeDump, managedOps: false) ||
            !RunChild(exePath, backend, managedDump, managedOps: true))
        {
            return 2;
        }

        Dictionary<string, (byte[] Data, double Ns)> native = ReadDump(nativeDump);
        Dictionary<string, (byte[] Data, double Ns)> managed = ReadDump(managedDump);

        int failures = 0;
        double ratioSum = 0;
        int ratioCount = 0;
        Console.WriteLine();
        Console.WriteLine($"=== Parity report ({backend}): {native.Count} cases ===");
        foreach (KeyValuePair<string, (byte[] Data, double Ns)> kv in native)
        {
            if (!managed.TryGetValue(kv.Key, out (byte[] Data, double Ns) m))
            {
                Console.WriteLine($"FAIL {kv.Key}: missing from managed dump");
                ++failures;
                continue;
            }

            bool equal = kv.Value.Data.AsSpan().SequenceEqual(m.Data);
            double ratio = m.Ns / Math.Max(kv.Value.Ns, 1.0);
            ratioSum += ratio;
            ++ratioCount;
            if (!equal)
            {
                int firstDiff = FirstDifference(kv.Value.Data, m.Data);
                Console.WriteLine($"FAIL {kv.Key}: results differ at byte {firstDiff} (native {kv.Value.Data.Length} B, managed {m.Data.Length} B)");
                ++failures;
            }
            else
            {
                Console.WriteLine($"PASS {kv.Key}: bit-exact; native {kv.Value.Ns / 1000.0:F1} us, managed {m.Ns / 1000.0:F1} us (x{ratio:F2})");
            }
        }
        foreach (string key in managed.Keys)
        {
            if (!native.ContainsKey(key))
            {
                Console.WriteLine($"FAIL {key}: present only in managed dump");
                ++failures;
            }
        }

        Console.WriteLine();
        Console.WriteLine($"Cases: {native.Count}, failures: {failures}, mean managed/native time ratio: {(ratioCount > 0 ? ratioSum / ratioCount : 0):F3}");
        Console.WriteLine(failures == 0 ? $"PARITY OK ({backend})" : $"PARITY FAILED ({backend})");
        return failures == 0 ? 0 : 1;
    }

    private static bool RunChild(string exePath, string backend, string dumpPath, bool managedOps)
    {
        var psi = new ProcessStartInfo
        {
            FileName = exePath,
            UseShellExecute = false,
        };
        // When launched as `dotnet GgmlManagedParity.dll`, ProcessPath is the
        // dotnet host, so the child needs the assembly path as its first arg.
        string hostName = Path.GetFileNameWithoutExtension(exePath);
        if (string.Equals(hostName, "dotnet", StringComparison.OrdinalIgnoreCase))
        {
            psi.ArgumentList.Add(typeof(Program).Assembly.Location);
        }
        psi.ArgumentList.Add("capture");
        psi.ArgumentList.Add(backend);
        psi.ArgumentList.Add(dumpPath);
        psi.Environment["TS_GGML_MANAGED_OPS"] = managedOps ? "1" : "0";

        using Process child = Process.Start(psi);
        child.WaitForExit();
        if (child.ExitCode != 0)
        {
            Console.Error.WriteLine($"Child ({(managedOps ? "managed" : "native")}, {backend}) failed with exit code {child.ExitCode}.");
            return false;
        }
        return true;
    }

    private static int FirstDifference(byte[] a, byte[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; ++i)
        {
            if (a[i] != b[i])
                return i;
        }
        return n;
    }

    // ------------------------------------------------------------------
    // Child: runs every module's case battery
    // ------------------------------------------------------------------

    private static Dictionary<string, (byte[] Data, double Ns)> ReadDump(string path)
    {
        var result = new Dictionary<string, (byte[], double)>(StringComparer.Ordinal);
        using var reader = new BinaryReader(File.OpenRead(path));
        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            string name = reader.ReadString();
            double ns = reader.ReadDouble();
            int length = reader.ReadInt32();
            byte[] data = reader.ReadBytes(length);
            result[name] = (data, ns);
        }
        return result;
    }

    private static int Capture(string backendName, string dumpPath)
    {
        GgmlBackendType backendType = backendName switch
        {
            "cpu" => GgmlBackendType.Cpu,
            "cuda" => GgmlBackendType.Cuda,
            "vulkan" => GgmlBackendType.Vulkan,
            "metal" => GgmlBackendType.Metal,
            _ => throw new ArgumentException($"Unknown backend '{backendName}'."),
        };

        _ = new GgmlContext(new[] { 0 }, backendType);

        using var dump = new Harness.Dump(dumpPath);

        CasesElementwise.Run(dump);
        CasesTraining.Run(dump);
        CasesNormAttn.Run(dump);
        CasesMatmul.Run(dump);

        // Tear down the backend + caches (incl. the prefill-attention session
        // cache) before process exit, exactly like the production server does:
        // leaked CUDA buffers freed by static destructors after driver
        // shutdown abort the process.
        GgmlNative.Shutdown();

        Console.WriteLine($"capture complete: {dumpPath}");
        return 0;
    }
}
