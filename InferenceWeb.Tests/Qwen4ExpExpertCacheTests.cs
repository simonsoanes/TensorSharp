// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection("MoeCpuOffloadConfig")]
public sealed class Qwen4ExpExpertCacheTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-q4e-expert-cache-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _environment = new();
    private readonly ITestOutputHelper _output;

    public Qwen4ExpExpertCacheTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "256");
        _environment.ClearSpeculationVars();
        MoeCpuOffloadConfig.Reset();
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(false)]
    [InlineData(true)]
    public void ChangingRoutes_ResetAndReload_KeepCompleteLogits(bool q2kxl)
    {
        bool cached = int.TryParse(Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_MB"), out int budget) && budget > 1;
        string a = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "a.gguf"), q2kxlExperts: q2kxl);
        string b = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "b.gguf"), q2kxlExperts: !q2kxl);
        var device = Qwen4ExpExpertCacheScenario.Exercise(a, b, BackendType.GgmlCuda, host: false);
        var active = new List<Qwen4ExpExpertCacheScenario.Stats>();
        var host = Qwen4ExpExpertCacheScenario.Exercise(a, b, BackendType.GgmlCuda, host: true,
            observeActiveCache: cached ? (before, after) => active.Add(new Qwen4ExpExpertCacheScenario.Stats(
                after.ReservedBytes, after.BudgetBytes, after.Hits - before.Hits,
                after.Misses - before.Misses, after.Calls - before.Calls)) : null);
        string file = q2kxl ? "q2kxl.json" : "mixed.json";
        string? capture = Environment.GetEnvironmentVariable("TS_Q4E_EXPERT_CACHE_CAPTURE_DIR");
        if (!string.IsNullOrEmpty(capture))
        {
            Directory.CreateDirectory(capture);
            File.WriteAllText(Path.Combine(capture, file), JsonSerializer.Serialize(host));
        }
        if (cached)
        {
            var stats = Qwen4ExpExpertCacheScenario.CacheStats();
            _output.WriteLine(JsonSerializer.Serialize(stats));
            Assert.Equal(3, active.Count);
            Assert.All(active, snapshot =>
            {
                _output.WriteLine(JsonSerializer.Serialize(snapshot));
                Assert.True(snapshot.Calls > 0 && snapshot.Hits > 0 && snapshot.Misses > 0,
                    "The expert cache did not engage and reuse weights in this model.");
                Assert.InRange(snapshot.ReservedBytes, 1, snapshot.BudgetBytes);
            });
            Assert.Equal(0, stats.ReservedBytes);
            Assert.Equal((long)budget * 1024 * 1024, stats.BudgetBytes);
        }
        Check("reset A-B-A", host["a2"], host["a1"], 0);
        Check("dispose different model and reload A", host["reloaded-a"], host["a1"], 0);
        foreach (string scenario in device.Keys)
            Check("all-CUDA " + scenario, host[scenario], device[scenario], cached ? 1e-6 : 2.5e-2, requireEqualArgmax: cached);
        string? reference = Environment.GetEnvironmentVariable("TS_Q4E_EXPERT_CACHE_REFERENCE_DIR");
        if (!string.IsNullOrEmpty(reference))
        {
            var expected = JsonSerializer.Deserialize<Dictionary<string, float[][]>>(
                File.ReadAllText(Path.Combine(reference, file)))!;
            foreach (string scenario in host.Keys)
                Check("fresh-process cached reference " + scenario, host[scenario], expected[scenario], 1e-6);
        }
    }

    private void Check(string name, float[][] actual, float[][] expected, double bound, bool requireEqualArgmax = true)
    {
        var difference = Qwen4ExpExpertCacheScenario.Difference(actual, expected);
        _output.WriteLine($"{name}: relative L2={difference.RelativeL2:E6}, max absolute={difference.MaxAbsolute:E6}, argmax mismatches={difference.ArgmaxMismatches}");
        if (requireEqualArgmax) Assert.Equal(0, difference.ArgmaxMismatches);
        Assert.True(difference.RelativeL2 <= bound, $"{name}: relative L2 {difference.RelativeL2:E6} exceeds {bound:E6}.");
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void ShortPrefillAndTargetVerification_KeepRowsAndRollback(int width)
    {
        bool cached = int.TryParse(Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_MB"), out int budget) && budget > 1;
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "verify.gguf"));
        float[][] device;
        using (var model = (Qwen4ExpModel)ModelBase.Create(path, BackendType.GgmlCuda))
            device = Qwen4ExpExpertCacheScenario.Verify(model, width);
        MoeCpuOffloadConfig.SetAllLayers();
        using (var model = (Qwen4ExpModel)ModelBase.Create(path, BackendType.GgmlCuda))
        {
            var host = Qwen4ExpExpertCacheScenario.Verify(model, width, cached ? (phase, before, after) =>
            {
                _output.WriteLine($"{phase}, width={width}: " + JsonSerializer.Serialize(after));
                Assert.True(after.Calls - before.Calls >= width * Qwen4ExpSyntheticModelBuilder.Layers,
                    $"{phase} silently fell back instead of caching every target row.");
                Assert.InRange(after.ReservedBytes, 1, after.BudgetBytes);
            } : null);
            Check($"short prefill + target verification width={width}", host, device, cached ? 1e-6 : 2.5e-2, requireEqualArgmax: cached);
        }
        if (cached) Assert.Equal(0, Qwen4ExpExpertCacheScenario.CacheStats().ReservedBytes);
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        try { Directory.Delete(_directory, recursive: true); } catch (IOException) { }
    }
}
