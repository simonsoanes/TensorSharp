// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Operator-level execution-path overrides for the engine's step routing,
    /// read from <c>TS_*</c> environment variables in ONE place instead of
    /// scattered <c>Environment.GetEnvironmentVariable</c> checks at each
    /// decision point. <see cref="BatchExecutor"/> materialises a snapshot per
    /// step (env reads are cheap and tests toggle these variables at runtime,
    /// so the values are deliberately NOT cached for the process lifetime) and
    /// hands it to <see cref="ExecutionPlanner"/> together with the model's
    /// <see cref="ExecutionCapabilities"/>.
    ///
    /// Every flag keeps the exact parse semantics of the check it replaced so
    /// existing deployments and A/B scripts behave identically.
    /// </summary>
    public sealed record ExecutionOptions
    {
        /// <summary>Force the per-sequence KV-swap fallback even when the model
        /// implements <see cref="IBatchedPagedModel"/>. Used to A/B the batched
        /// and per-sequence paths on the same workload.
        /// Env: <c>TS_SCHED_DISABLE_BATCHED</c> (default off).</summary>
        public bool BatchedPathDisabled { get; init; }

        /// <summary>Serve a solo scheduled sequence through the model's fused
        /// single-graph <c>Forward</c> (linear KV cache) instead of the op-by-op
        /// batched path — dramatically faster on models with a fused decode
        /// kernel. Env: <c>TS_BATCHED_N1_FAST_PATH</c> (default on; set 0 to A/B
        /// the fully-batched path).</summary>
        public bool BatchedN1FastPathEnabled { get; init; } = true;

        /// <summary>Serve concurrent (N&gt;=2) sequences on fused-capable models
        /// by running each through its own fused Forward with a per-request KV
        /// cache. Env: <c>TS_PER_SEQ_FUSED</c> (default on; set 0 to force the
        /// op-by-op batched paged path for A/B or debugging).</summary>
        public bool PerSeqFusedEnabled { get; init; } = true;

        /// <summary>TRUE token-batched fused decode inside the per-sequence
        /// fused path: decode one token for each of N sequences in a single
        /// fused graph. Env: <c>TS_BATCHED_FUSED_DECODE</c> (default off,
        /// correctness-first; strict opt-in "1"/"true").</summary>
        public bool BatchedFusedDecodeEnabled { get; init; }

        /// <summary>Retain finished fused-path KV holders for cross-request
        /// prefix reuse. Env: <c>TS_RETAINED_FUSED_CACHE</c> (default on;
        /// kill-switch for A/B or to cap VRAM use).</summary>
        public bool RetainedFusedCacheEnabled { get; init; } = true;

        /// <summary>How many finished fused holders to keep alive for
        /// cross-request prefix reuse; each pins a full per-request KV cache.
        /// Env: <c>TS_RETAINED_FUSED_CACHE_MAX</c> (default 4).</summary>
        public int RetainedFusedCacheBudget { get; init; } = 4;

        /// <summary>All defaults — the configuration used when no TS_* override
        /// is set. Handy for tests.</summary>
        public static ExecutionOptions Default { get; } = new();

        /// <summary>Read the current override state from the environment.</summary>
        public static ExecutionOptions FromEnvironment() => new()
        {
            BatchedPathDisabled = ReadFlag("TS_SCHED_DISABLE_BATCHED", false),
            BatchedN1FastPathEnabled = ReadFlag("TS_BATCHED_N1_FAST_PATH", true),
            PerSeqFusedEnabled = ReadFlag("TS_PER_SEQ_FUSED", true),
            BatchedFusedDecodeEnabled = ReadStrictOptIn("TS_BATCHED_FUSED_DECODE"),
            RetainedFusedCacheEnabled = ReadFlag("TS_RETAINED_FUSED_CACHE", true),
            RetainedFusedCacheBudget = ReadNonNegativeInt("TS_RETAINED_FUSED_CACHE_MAX", 4),
        };

        /// <summary>One-line summary of the non-default overrides in effect
        /// (empty string when everything is at its default).</summary>
        public string DescribeOverrides()
        {
            var parts = new System.Collections.Generic.List<string>();
            if (BatchedPathDisabled) parts.Add("TS_SCHED_DISABLE_BATCHED");
            if (!BatchedN1FastPathEnabled) parts.Add("TS_BATCHED_N1_FAST_PATH=0");
            if (!PerSeqFusedEnabled) parts.Add("TS_PER_SEQ_FUSED=0");
            if (BatchedFusedDecodeEnabled) parts.Add("TS_BATCHED_FUSED_DECODE=1");
            if (!RetainedFusedCacheEnabled) parts.Add("TS_RETAINED_FUSED_CACHE=0");
            if (RetainedFusedCacheBudget != 4) parts.Add($"TS_RETAINED_FUSED_CACHE_MAX={RetainedFusedCacheBudget}");
            return string.Join(", ", parts);
        }

        // Loose boolean: unset -> default; "0"/"false" -> false; anything else -> true.
        // (The historical parse rule of the flags this type replaced.)
        private static bool ReadFlag(string name, bool fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrEmpty(raw)) return fallback;
            return raw != "0" && !string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase);
        }

        // Strict opt-in: only "1" or "true" enables; everything else stays off.
        private static bool ReadStrictOptIn(string name)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return raw == "1" || string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase);
        }

        private static int ReadNonNegativeInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int v) && v >= 0)
                return v;
            return fallback;
        }
    }
}
