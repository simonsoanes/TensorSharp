// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp

using Xunit;

// Tests exercise process-global CUDA contexts, native graph caches, and TS_*
// environment overrides. Cross-class parallelism makes those shared resources
// race (CUDA error 700 or a request observing another test's override), so run
// test collections serially for deterministic correctness.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
