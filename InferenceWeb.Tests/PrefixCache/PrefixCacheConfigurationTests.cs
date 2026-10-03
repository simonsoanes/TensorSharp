// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests.PrefixCache;

[Collection(EngineEnvironmentCollection.Name)]

public class PrefixCacheConfigurationTests
{
    [Fact]
    public void PrefixCachingIsOnByDefaultAndSurvivesASpeculationChange()
    {
        Assert.True(SchedulerConfig.Default.EnablePrefixCaching);
        Assert.True(SchedulerConfig.Default.WithSpeculation(SpeculationOptions.Disabled).EnablePrefixCaching);
        Assert.False(new SchedulerConfig { EnablePrefixCaching = false }
            .WithSpeculation(SpeculationOptions.Disabled).EnablePrefixCaching);
    }

    [Fact]
    public void TheOffSwitchTurnsPrefixCachingOff()
    {
        using var env = new EnvScope();
        env.Set("TS_SCHED_PREFIX_CACHE", "0");
        Assert.False(SchedulerConfig.FromEnvironment().EnablePrefixCaching);
    }

    /// <summary>The radix cache is the only prefix cache. A deployment still asking for the
    /// removed legacy mode hears so instead of running on something it did not ask for.</summary>
    [Theory]
    [InlineData("legacy")]
    [InlineData("tree")]
    public void TheRemovedModeVariable_ErrorsNamingWhatReplacesIt(string value)
    {
        using var env = new EnvScope();
        env.Set("TS_PREFIX_CACHE_MODE", value);
        var ex = Assert.Throws<ArgumentException>(() => SchedulerConfig.FromEnvironment());
        Assert.Contains("TS_PREFIX_CACHE_MODE", ex.Message);
        Assert.Contains("TS_SCHED_PREFIX_CACHE", ex.Message);
    }
}
