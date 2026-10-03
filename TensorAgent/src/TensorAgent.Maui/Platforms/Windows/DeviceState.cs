// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.InteropServices;

namespace TensorAgent.Maui.Services;

/// <summary>
/// The Windows answers to what Services/Apple/DeviceState.cs asks an Apple device.
/// </summary>
internal static class DeviceState
{
    /// <summary>Installed physical memory, in bytes.</summary>
    public static long PhysicalMemoryBytes()
    {
        var status = new MemoryStatusEx { Length = (uint)Marshal.SizeOf<MemoryStatusEx>() };
        return GlobalMemoryStatusEx(ref status)
            ? (long)status.TotalPhysical
            : GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
    }

    /// <summary>
    /// How much more this process could take, in bytes. Windows has no per-process
    /// budget of the kind iOS enforces, so this is the system's available physical
    /// memory, which is what a model load competes for.
    /// </summary>
    public static long AvailableMemoryBytes()
    {
        var status = new MemoryStatusEx { Length = (uint)Marshal.SizeOf<MemoryStatusEx>() };
        return GlobalMemoryStatusEx(ref status) ? (long)status.AvailablePhysical : 0;
    }

    /// <summary>One line naming the memory, for the startup log.</summary>
    public static string DescribeMemory()
    {
        double physical = PhysicalMemoryBytes() / 1_000_000_000.0;
        long available = AvailableMemoryBytes();
        return available > 0
            ? $"device {physical:0.0} GB, {available / 1_000_000_000.0:0.00} GB available"
            : $"device {physical:0.0} GB";
    }

    /// <summary>
    /// Whether the only route to the internet is a metered cellular connection, the same
    /// question and the same answer as on the phone: a model is gigabytes.
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
            return false;
        }
    }

    /// <summary>
    /// Keep Windows from sleeping or throttling the app while the model works, and stop
    /// when it is done (see Services/DesktopActivity). Two requests: the machine must not
    /// idle-sleep halfway through an answer, and the process must not be put into
    /// EcoQoS, the power-throttled mode Windows 11 gives a process whose windows are not
    /// in front, which lowers the clock the engine's host threads run at. The display may
    /// still sleep. Execution state belongs to the thread that asks, so it is asked from
    /// the UI thread, which lives as long as the app. UNVERIFIED on Windows hardware: the
    /// Mac's equivalent was measured (Services/Apple/DeviceState.HoldActivity).
    /// </summary>
    public static void HoldActivity(bool on)
    {
        try
        {
            uint state = on ? EsContinuous | EsSystemRequired : EsContinuous;
            MainThread.BeginInvokeOnMainThread(() => SetThreadExecutionState(state));

            // ControlMask names the policy this process decides for itself; StateMask 0 with
            // the bit controlled means "never throttle", and ControlMask 0 hands the choice
            // back to Windows.
            var throttling = new ProcessPowerThrottlingState
            {
                Version = ProcessPowerThrottlingCurrentVersion,
                ControlMask = on ? ProcessPowerThrottlingExecutionSpeed : 0,
                StateMask = 0,
            };
            SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                ref throttling, (uint)Marshal.SizeOf<ProcessPowerThrottlingState>());
        }
        catch (Exception)
        {
            // Not worth failing a generation over.
        }
    }

    private const uint EsSystemRequired = 0x00000001;
    private const uint EsContinuous = 0x80000000;

    [DllImport("kernel32.dll")]
    private static extern uint SetThreadExecutionState(uint esFlags);

    private const int ProcessPowerThrottling = 4;
    private const uint ProcessPowerThrottlingCurrentVersion = 1;
    private const uint ProcessPowerThrottlingExecutionSpeed = 0x1;

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessPowerThrottlingState
    {
        public uint Version;
        public uint ControlMask;
        public uint StateMask;
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentProcess();

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetProcessInformation(
        IntPtr process, int informationClass, ref ProcessPowerThrottlingState information, uint size);

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatusEx
    {
        public uint Length;
        public uint MemoryLoad;
        public ulong TotalPhysical;
        public ulong AvailablePhysical;
        public ulong TotalPageFile;
        public ulong AvailablePageFile;
        public ulong TotalVirtual;
        public ulong AvailableVirtual;
        public ulong AvailableExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx buffer);
}
