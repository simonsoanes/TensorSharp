// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace InferenceWeb.Tests;

/// <summary>
/// Tests that set the engine's option variables (TS_RETAINED_*, TS_PREFIX_*, TS_PER_SEQ_*, TS_SCHED_*,
/// TS_BATCHED_*, TS_KV_*, TS_THINKING_*). The engine reads them while it runs, so a test that set one while
/// another test's engine was running changed that test's behaviour (a model without retained holders was
/// seen reusing 16 tokens it must not). They run alone.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class EngineEnvironmentCollection
{
    public const string Name = "Engine environment";
}
