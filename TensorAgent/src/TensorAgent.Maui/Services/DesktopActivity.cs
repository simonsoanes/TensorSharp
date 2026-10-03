// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

#if MACCATALYST || WINDOWS
using TensorAgent.Core.Hosting;

namespace TensorAgent.Maui.Services;

/// <summary>
/// Keeps a desktop from throttling or sleeping the model while it works: App Nap on a Mac,
/// power throttling and idle sleep on Windows (see <see cref="DeviceState.HoldActivity"/>).
///
/// <para>
/// The phone does the opposite -- it hands the GPU back the moment the app leaves the
/// screen, because iOS refuses GPU work from the background -- and that is
/// Platforms/iOS/BackgroundGeneration. A desktop keeps working behind other windows, so
/// all it has to do is say that it is working.
/// </para>
/// <para>
/// Two signals, because neither alone sees everything. The host's turn signal says at
/// once when a user's answer starts, and covers the stretches a turn spends running a
/// command rather than the engine. A look at the engine's own counters twice a second
/// also sees the work no page asked for: the prefix-cache warm-up after a load,
/// sub-agents, a benchmark. The activity is held only while one of them says so.
/// </para>
/// </summary>
internal sealed class DesktopActivity : IDisposable
{
    private static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(500);

    private readonly AgentAppHost _host;
    private readonly Timer _poll;
    private readonly object _gate = new();
    private bool _held;
    private bool _disposed;

    public DesktopActivity(AgentAppHost host)
    {
        _host = host ?? throw new ArgumentNullException(nameof(host));
        _host.Turns.BusyChanged += OnBusyChanged;
        _poll = new Timer(_ => Update(), null, PollInterval, PollInterval);
    }

    private void OnBusyChanged(bool busy) => Update();

    private void Update()
    {
        bool working;
        try { working = _host.Turns.IsBusy || _host.IsEngineWorking; }
        catch (Exception) { return; }
        lock (_gate)
        {
            if (_disposed || working == _held)
                return;
            _held = working;
            DeviceState.HoldActivity(working);
        }
    }

    public void Dispose()
    {
        _host.Turns.BusyChanged -= OnBusyChanged;
        _poll.Dispose();
        lock (_gate)
        {
            _disposed = true;
            if (_held)
                DeviceState.HoldActivity(false);
            _held = false;
        }
    }
}
#endif
