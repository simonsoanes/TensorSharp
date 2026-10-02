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
using System.Runtime.InteropServices;
using UIKit;

namespace TensorAgent.Maui.Services;

/// <summary>
/// The two things the app has to ask the device about, both of which exist because
/// a setting would otherwise be a switch that does nothing.
/// </summary>
internal static class DeviceState
{
    /// <summary>
    /// How much memory iOS will still let this process take, in bytes.
    ///
    /// <para>
    /// This is the number that actually kills the app, and until now nothing asked for
    /// it. NSProcessInfo.PhysicalMemory reports the DEVICE's RAM -- 12 GB -- while the
    /// per-process jetsam budget is roughly two thirds of that, and it is against the
    /// budget that a KV cache is charged. os_proc_available_memory() is the supported
    /// way to ask, and it accounts for the increased-memory-limit entitlement the app
    /// already carries.
    /// </para>
    ///
    /// <para>
    /// Returns 0 where the call is unavailable (the simulator returns 0, and so does any
    /// process without the entitlement on older systems), which every caller must read
    /// as "unknown" rather than "none left".
    /// </para>
    /// </summary>
    [DllImport("__Internal", EntryPoint = "os_proc_available_memory")]
    private static extern nint OsProcAvailableMemory();

    /// <summary>Installed physical memory, in bytes.</summary>
    public static long PhysicalMemoryBytes() => (long)NSProcessInfo.ProcessInfo.PhysicalMemory;

    public static long AvailableMemoryBytes()
    {
        try
        {
            return (long)OsProcAvailableMemory();
        }
        catch (EntryPointNotFoundException)
        {
            return 0;
        }
        catch (DllNotFoundException)
        {
            return 0;
        }
    }

    /// <summary>
    /// One line naming the headroom, for the log that a jetsam kill leaves behind.
    /// A kill writes no stack and no message of its own, so the last thing the app
    /// said about its own budget is the only evidence there is.
    /// </summary>
    public static string DescribeMemory()
    {
        long available = AvailableMemoryBytes();
        double physical = NSProcessInfo.ProcessInfo.PhysicalMemory / 1_000_000_000.0;
        string headroom = available > 0
            ? $"device {physical:0.0} GB, this process may still take {available / 1_000_000_000.0:0.00} GB"
            : $"device {physical:0.0} GB, per-process headroom unavailable";
        // The per-process figure above is the one the app used to trust, and it is the
        // wrong one on a phone: the weights are wired outside the footprint it is judged
        // against, so it can read "4 GB left" on a device with none. The kernel's own
        // numbers -- what this process is charged and what the device has wired and
        // free -- are the ones a jetsam report will show. See ProcessMemory.
        return headroom + "; " + TensorAgent.Core.Hosting.ProcessMemory.Describe();
    }

    /// <summary>
    /// Whether the only route to the internet right now is the cellular radio.
    ///
    /// <para>
    /// A catalog entry is five to ten gigabytes. Starting one of those on a metered
    /// connection because nobody checked is a real cost to a real person, so the
    /// "download over cellular" setting is enforced here rather than merely stored.
    /// MAUI's own connectivity profile is enough for this: it reports the active
    /// profiles, and cellular-only means cellular is present and WiFi and Ethernet
    /// are not.
    /// </para>
    /// </summary>
    public static bool IsOnCellularOnly()
    {
        try
        {
            IEnumerable<ConnectionProfile> profiles = Connectivity.Current.ConnectionProfiles;
            bool cellular = profiles.Contains(ConnectionProfile.Cellular);
            bool unmetered = profiles.Contains(ConnectionProfile.WiFi) || profiles.Contains(ConnectionProfile.Ethernet);
            return cellular && !unmetered;
        }
        catch (Exception)
        {
            // If the answer cannot be determined, do not block the download: a user
            // who asked for a model and got silence would have no way to find out why.
            return false;
        }
    }

#if MACCATALYST
    private static readonly object s_activityGate = new();
    private static NSObject? s_activity;

    /// <summary>
    /// Tell macOS the model is working, for exactly as long as it is.
    ///
    /// <para>
    /// App Nap throttles an app none of whose windows can be seen -- minimised, covered by
    /// another window, or behind a locked screen -- by lowering its threads' priority and
    /// coalescing its timers, and the engine's host threads are what feed the GPU. MEASURED
    /// on an M5 Pro with the screen locked, the in-process benchmark on Gemma 4 E2B: stretches
    /// of 47-58 tok/s where the unthrottled rate is 72-75, in five runs of eight, while the
    /// same code on CoreCLR in a terminal (never napped) was never slower than 73. A
    /// user-initiated activity is the documented way to say "this is work the user is waiting
    /// for"; it also keeps the Mac from idle-sleeping halfway through an answer. The display
    /// may still sleep: nothing about a generation needs it.
    /// </para>
    /// </summary>
    public static void HoldActivity(bool on)
    {
        lock (s_activityGate)
        {
            if (on && s_activity is null)
            {
                s_activity = NSProcessInfo.ProcessInfo.BeginActivity(
                    NSActivityOptions.UserInitiated, "TensorAgent is generating a reply");
            }
            else if (!on && s_activity is not null)
            {
                NSProcessInfo.ProcessInfo.EndActivity(s_activity);
                s_activity = null;
            }
        }
    }
#endif

    /// <summary>
    /// Keep the screen on, or stop keeping it on.
    ///
    /// <para>
    /// Generation on a phone is slow enough that the display sleeps partway through,
    /// and on iOS that suspends the app and stops the work. The alternative to this
    /// is a user watching a half-finished answer stop moving whenever they look away.
    /// It is only ever held while a reply is actually being generated.
    /// </para>
    /// </summary>
    public static void KeepAwake(bool on)
    {
        try
        {
            MainThread.BeginInvokeOnMainThread(() => UIApplication.SharedApplication.IdleTimerDisabled = on);
        }
        catch (Exception)
        {
            // Not worth failing a generation over.
        }
    }
}
