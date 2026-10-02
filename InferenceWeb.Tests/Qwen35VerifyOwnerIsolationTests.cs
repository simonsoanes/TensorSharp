// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class Qwen35VerifyOwnerIsolationTests
{
    [Fact]
    public void VerifyOwnerIds_AreNonZeroAndUniqueUnderConcurrency()
    {
        var ids = new ConcurrentBag<long>();

        Parallel.For(0, 512, _ => ids.Add(Qwen35Model.AllocateVerifyOwnerId()));

        Assert.Equal(512, ids.Count);
        Assert.All(ids, id => Assert.True(id > 0));
        Assert.Equal(512, ids.Distinct().Count());
    }

    [Fact]
    public void OwnedInterop_AppendsOwnerToEveryDeferredStateOperation()
    {
        Type native = typeof(GgmlBasicOps).Assembly.GetType("TensorSharp.GGML.GgmlNative", throwOnError: true)!;

        foreach (string name in new[]
                 {
                     "TSGgml_Qwen35ModelVerifyOwned",
                     "TSGgml_Qwen35CommitStateSnapshotOwned",
                     "TSGgml_Qwen35FetchStateSnapshotOwned",
                     "TSGgml_Qwen35DrainDeviceStateOwned",
                 })
        {
            MethodInfo method = Assert.Single(
                native.GetMethods(BindingFlags.NonPublic | BindingFlags.Static),
                candidate => candidate.Name == name);
            ParameterInfo owner = Assert.Single(method.GetParameters().TakeLast(1));
            Assert.Equal(typeof(long), owner.ParameterType);
            Assert.Equal("ownerId", owner.Name);
        }
    }

    [Fact]
    public void TwoOwnerResetRelease_InterleavesAndToleratesMissingOrRepeatedOwners()
    {
        long ownerA = Qwen35Model.AllocateVerifyOwnerId();
        long ownerB = Qwen35Model.AllocateVerifyOwnerId();

        GgmlBasicOps.Qwen35ResetVerifyCache(ownerA);
        GgmlBasicOps.Qwen35ResetVerifyCache(ownerB);
        GgmlBasicOps.Qwen35ReleaseVerifyOwner(ownerB);
        GgmlBasicOps.Qwen35ResetVerifyCache(ownerA);
        GgmlBasicOps.Qwen35ReleaseVerifyOwner(ownerB);
        GgmlBasicOps.Qwen35ReleaseVerifyOwner(ownerA);
    }

    [Fact]
    public void NativeLibrary_ExportsTheOwnedLifecycle()
    {
        string libraryName = OperatingSystem.IsMacOS() ? "libGgmlOps.dylib"
            : OperatingSystem.IsWindows() ? "GgmlOps.dll"
            : "libGgmlOps.so";
        string libraryPath = Path.Combine(AppContext.BaseDirectory, libraryName);
        Assert.True(File.Exists(libraryPath), $"Native test library not found: {libraryPath}");

        IntPtr handle = NativeLibrary.Load(libraryPath);
        try
        {
            foreach (string symbol in new[]
                     {
                         "TSGgml_Qwen35ModelVerifyOwned",
                         "TSGgml_Qwen35CommitStateSnapshotOwned",
                         "TSGgml_Qwen35FetchStateSnapshotOwned",
                         "TSGgml_Qwen35DrainDeviceStateOwned",
                         "TSGgml_Qwen35ResetVerifyCacheOwner",
                         "TSGgml_Qwen35ReleaseVerifyOwner",
                         "TSGgml_Qwen35ResetVerifyCacheForHostPointer",
                         "TSGgml_Qwen35ReleaseVerifyGraphsPreserveState",
                     })
            {
                Assert.True(NativeLibrary.TryGetExport(handle, symbol, out _), $"Missing native export {symbol}");
            }
        }
        finally
        {
            NativeLibrary.Free(handle);
        }
    }
}
