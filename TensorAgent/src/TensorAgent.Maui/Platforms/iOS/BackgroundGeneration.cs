// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Foundation;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Settings;
using TensorAgent.Maui.Services;
using UIKit;

namespace TensorAgent.Maui.Platforms.iOS;

/// <summary>
/// Keeps a generation alive across the three things that used to end it: the display
/// going to sleep, the user leaving the app, and — the one that actually broke answers —
/// the GPU being taken away while the app is not in front.
///
/// <para>
/// The fourth, leaving the chat for another screen, is <see cref="ChatTurnManager"/>'s
/// job and needs nothing from iOS. These are the ones only the platform can answer. A
/// phone dims and then locks after a minute or two of no touches, and a locked phone
/// suspends the app; a turn that takes ninety seconds is therefore routinely killed by
/// the user doing nothing at all. Switching to another app suspends it immediately.
/// </para>
/// <para>
/// THE GPU. An app that is not frontmost may not submit Metal work — no entitlement
/// grants it, and there is no background mode for compute. A command buffer committed
/// anyway comes back <c>MTLCommandBufferStatusError</c> with "Insufficient Permission (to
/// submit GPU work from background)", and ggml-metal's answer to that is a sticky flag:
/// "once set, graph_compute will return GGML_STATUS_FAILED until the backend is
/// recreated". So switching away mid-answer did not lose a token, it lost the model —
/// that turn failed and so did every message after it, for the life of the process.
/// That is the bug this class now exists for more than any other.
/// </para>
/// <para>
/// The fix is to stop submitting rather than to submit and fail, and the moment to stop
/// is <c>WillResignActive</c>, NOT <c>DidEnterBackground</c>: resign arrives while the
/// app is still frontmost and several hundred milliseconds of animation before the
/// background transition, which is far longer than the decode step that would otherwise
/// have been in flight when the GPU was withdrawn. Coming back is
/// <c>DidBecomeActive</c>. The pair also covers the transient interruptions — Control
/// Centre, an incoming call banner, the app switcher — where the app never backgrounds
/// at all; a generation pauses for as long as the sheet is up and then carries on, which
/// is both correct and unnoticeable.
/// </para>
/// <para>
/// It follows the HOST's turns rather than the page's idea of them, which is the whole
/// reason it can be written at all: the page stops reading whenever it is navigated
/// away from, and asking it whether the model is working would answer "no" at exactly
/// the moments this exists for.
/// </para>
/// </summary>
internal sealed class BackgroundGeneration : IDisposable
{
    private readonly ChatTurnManager _turns;
    private readonly SettingsStore _settings;
    private readonly AgentAppHost _host;
    private readonly BackgroundAssertion _assertion = new("TensorAgent.Generation");
    private readonly List<NSObject> _observers = new();

    public BackgroundGeneration(AgentAppHost host)
    {
        _host = host ?? throw new ArgumentNullException(nameof(host));
        _turns = host.Turns;
        _settings = host.Settings;

        _turns.BusyChanged += OnBusyChanged;

        // The GPU goes away here, not at DidEnterBackground. See the class comment: this
        // is the earliest notification, and the several hundred milliseconds it buys
        // before the app actually leaves the screen is what makes it likely that no
        // decode step is still in flight when Metal stops accepting work.
        _observers.Add(NSNotificationCenter.DefaultCenter.AddObserver(
            UIApplication.WillResignActiveNotification, _ => Suspend()));
        _observers.Add(NSNotificationCenter.DefaultCenter.AddObserver(
            UIApplication.DidBecomeActiveNotification, _ => Resume()));

        // Asked for again on the way out as well as on the first token: a turn that was
        // already running when the user left never raises BusyChanged, and that is
        // precisely the case this exists for.
        _observers.Add(NSNotificationCenter.DefaultCenter.AddObserver(
            UIApplication.DidEnterBackgroundNotification, _ => { if (_turns.IsBusy) _assertion.Acquire(); }));

        // An app that is launched into the background — which iOS does, for a
        // background URL session among other things — must not start out believing it
        // owns the GPU. Read once, on the main thread, where reading it is allowed.
        if (UIApplication.SharedApplication.ApplicationState != UIApplicationState.Active)
            _host.Compute.Close();
    }

    /// <summary>
    /// Stop the model. Not "cancel": the turn keeps its place and its cache, and the
    /// engine's next step simply does not start until the app is in front again.
    /// </summary>
    private void Suspend()
    {
        _host.TraceBackground(_turns.IsBusy
            ? "leaving the foreground with a turn running: pausing the model"
            : "leaving the foreground, nothing running");
        if (_turns.IsBusy)
            _assertion.Acquire();
        _host.Compute.Close();
    }

    /// <summary>
    /// Back in front of the user: the model may run again — and if a turn died while we
    /// were away, rebuild the engine before anything else tries to use it.
    /// </summary>
    private void Resume()
    {
        _host.TraceBackground(_turns.IsBusy
            ? "back in front with a turn still running: letting the model go on"
            : "back in front, nothing running");
        _host.Compute.Open();

        if (!_host.EngineNeedsReload)
            return;
        // Off the UI thread: a reload reads gigabytes of weights and takes seconds. The
        // page is told nothing here on purpose -- refreshModel on the way back in
        // already asks what the model is doing, and it will report "loading".
        //
        // Not while a turn is running. Recovery UNLOADS the model and frees the
        // backend; doing that underneath a live generation pulls the weights out from
        // under it. A turn that needs a rebuilt engine asks for one itself, at a point
        // where it has stopped reading -- see AgentAppHost.GatedChatFrames.
        if (_turns.IsBusy)
        {
            _host.TraceBackground("the engine needs rebuilding, but a turn is running: leaving it to the turn");
            return;
        }

        _ = Task.Run(() =>
        {
            try { _host.RecoverEngineIfNeeded(); }
            catch (Exception ex) { _host.TraceBackground("rebuilding the engine threw: " + ex.Message); }
        });
    }

    private void OnBusyChanged(bool busy)
    {
        // The display is held awake for exactly the stretch the model is working, and
        // only when the user has left that setting on. It is a real cost to their
        // battery, so it is never held a moment longer than the answer takes.
        bool keepAwake = busy && _settings.Load().KeepAwakeWhileGenerating;
        DeviceState.KeepAwake(keepAwake);
        // Traced, because the display going to sleep mid-answer is not a cosmetic
        // problem on iOS: it resigns the app exactly as switching apps does, and is
        // by far the likeliest way a real user reaches the refused-GPU failure. When
        // the log shows a turn dying without a preceding "holding the display awake",
        // this is where to look rather than at the gate.
        _host.TraceBackground(busy
            ? (keepAwake ? "a turn started: holding the display awake" : "a turn started: NOT holding the display awake (the setting is off)")
            : "the turn ended: letting the display sleep again");
        if (busy)
            _assertion.Acquire();
        else
            _assertion.Release();
    }

    public void Dispose()
    {
        _turns.BusyChanged -= OnBusyChanged;
        foreach (NSObject observer in _observers)
            NSNotificationCenter.DefaultCenter.RemoveObserver(observer);
        _observers.Clear();
        DeviceState.KeepAwake(false);
        _assertion.Dispose();
    }
}
