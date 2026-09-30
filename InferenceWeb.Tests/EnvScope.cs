// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace InferenceWeb.Tests;

/// <summary>
/// Disposable helper that snapshots and restores environment variables
/// touched during a test. Without this, the env vars set by one test could
/// leak into another test that runs in the same process.
/// </summary>
internal sealed class EnvScope : IDisposable
{
    private readonly Dictionary<string, string?> _originals = new();

    public void Set(string name, string value)
    {
        if (!_originals.ContainsKey(name))
            _originals[name] = Environment.GetEnvironmentVariable(name);
        Environment.SetEnvironmentVariable(name, value);
    }

    /// <summary>
    /// Clear every speculative-decoding variable, and the removed names a
    /// developer's shell may still export (reading any of them is an error).
    /// </summary>
    public void ClearSpeculationVars()
    {
        foreach ((string removed, _) in TensorSharp.Runtime.Speculative.SpeculationEnvVars.RemovedNames)
            Set(removed, null);
        foreach (string name in new[]
                 {
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.Enabled,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.Type,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.Draft,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.PMin,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.DraftModel,
                 })
        {
            Set(name, null);
        }
    }

    public void Dispose()
    {
        foreach (var kv in _originals)
            Environment.SetEnvironmentVariable(kv.Key, kv.Value);
        _originals.Clear();
    }
}

/// <summary>
/// <see cref="EnvScope"/> for variables the NATIVE library reads: sets each one in
/// the managed environment and in the C runtime's table (libc <c>setenv</c>, the UCRT
/// <c>_putenv_s</c>), which .NET may keep separately, and restores both on dispose.
/// </summary>
internal sealed class NativeEnvScope : IDisposable
{
    private readonly Dictionary<string, string?> _originals = new();

    [DllImport("libc", EntryPoint = "setenv", CharSet = CharSet.Ansi)]
    private static extern int SetEnvUnix(string name, string value, int overwrite);

    [DllImport("libc", EntryPoint = "unsetenv", CharSet = CharSet.Ansi)]
    private static extern int UnsetEnvUnix(string name);

    [DllImport("ucrtbase", EntryPoint = "_putenv_s", CharSet = CharSet.Ansi)]
    private static extern int PutEnvWindows(string name, string value);

    public void Set(string name, string? value)
    {
        if (!_originals.ContainsKey(name))
            _originals[name] = Environment.GetEnvironmentVariable(name);
        SetBoth(name, value);
    }

    public void Dispose()
    {
        foreach (var pair in _originals)
            SetBoth(pair.Key, pair.Value);
        _originals.Clear();
    }

    private static void SetBoth(string name, string? value)
    {
        Environment.SetEnvironmentVariable(name, value);
        int result = OperatingSystem.IsWindows()
            ? PutEnvWindows(name, value ?? string.Empty)
            : value == null ? UnsetEnvUnix(name) : SetEnvUnix(name, value, 1);
        if (result != 0)
            throw new InvalidOperationException($"Failed to update the native environment variable '{name}'.");
    }
}
