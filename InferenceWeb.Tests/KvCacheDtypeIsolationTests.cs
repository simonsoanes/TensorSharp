// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// KvCacheDtypeConfig is process-wide and every model reads it at construction, so a test
// that sets it must put back both the value and the "explicitly set" flag (which turns off
// the model-aligned f16 default) - KvCacheDtypeConfig.RestoreForTests does both. Three
// tests did not, and every model a later test in the same process created ran with a
// cache nobody asked for: a Qwen 3.5 exactness test failed only inside a group run.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public class KvCacheDtypeIsolationTests
{
    [Fact]
    public void EveryTestFileThatSetsTheKvDtype_RestoresItExactly()
    {
        var here = new DirectoryInfo(AppContext.BaseDirectory);
        while (here != null && !File.Exists(Path.Combine(here.FullName, "InferenceWeb.Tests", "InferenceWeb.Tests.csproj")))
            here = here.Parent;
        Assert.NotNull(here);

        string setCall = nameof(KvCacheDtypeConfig) + "." + nameof(KvCacheDtypeConfig.Set) + "(";
        string restoreCall = nameof(KvCacheDtypeConfig) + ".RestoreForTests(";
        var offenders = Directory.EnumerateFiles(Path.Combine(here.FullName, "InferenceWeb.Tests"), "*.cs")
            .Where(file =>
            {
                string text = File.ReadAllText(file);
                return text.Contains(setCall, StringComparison.Ordinal) && !text.Contains(restoreCall, StringComparison.Ordinal);
            })
            .Select(Path.GetFileName)
            .ToList();
        Assert.True(offenders.Count == 0,
            $"these files call {setCall}...) without {restoreCall}...): " + string.Join(", ", offenders));
    }
}
