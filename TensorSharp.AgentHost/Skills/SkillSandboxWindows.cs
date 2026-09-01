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
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

using TensorSharp.AgentHost.CodeExec;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// Windows containment through a job object.
    ///
    /// <para>
    /// <b>This is weaker than the macOS and Linux sandboxes, and says so.</b> A job
    /// object bounds a process TREE — how many processes it may spawn, how much memory
    /// it may commit, and that every one of them dies when the job handle closes — but
    /// TensorSharp's current Windows backend offers no filesystem or network confinement through it. Real isolation
    /// on Windows means an AppContainer or a low-integrity token, both of which require
    /// launching the child through <c>CreateProcessAsUser</c> with a hand-built
    /// security-capability attribute list and hand-plumbed stdio handles; that is a
    /// large piece of interop to get right, and getting it subtly wrong would produce
    /// something that reports "sandboxed" while confining nothing, which is worse than
    /// this.
    /// </para>
    /// <para>
    /// So the guarantee is stated honestly instead of inflated:
    /// <see cref="Capabilities"/> reports writes, network and home reads as NOT
    /// confined, <see cref="SkillScriptRunner"/> repeats those gaps in the result the
    /// model sees and in the startup log, and the documentation's platform table is
    /// written from the same record. On Windows the protections that actually carry the
    /// weight are the portable ones — the path guard, the interpreter allow-list, the
    /// scrubbed environment, the scratch working directory, no shell, and the time and
    /// output bounds — plus the fact that <c>skills_run</c> is off unless an operator
    /// turns it on.
    /// </para>
    /// </summary>
    [SupportedOSPlatform("windows")]
    internal sealed class WindowsJobObjectSandbox : ISkillSandbox
    {
        public string Name => "windows-job-object";

        public bool IsAvailable => OperatingSystem.IsWindows();

        public SkillSandboxCapabilities Capabilities => new(
            // A job object cannot restrict file or socket access. Claiming otherwise
            // would be the one failure mode this whole design exists to avoid.
            ConfinesWrites: false,
            ConfinesNetwork: false,
            ConfinesHomeReads: false,
            BoundsProcessTree: true);

        /// <summary>
        /// What this sandbox enforces AND what it does not, in one line.
        ///
        /// <para>
        /// The negative half is not padding. This string is what an operator reads in
        /// the startup banner and in <c>--list-skills</c>, and it used to name only the
        /// process-tree bound - so the line said "sandbox: bounds the process tree" and
        /// left the reader to infer that files and sockets were bounded too, which is
        /// the single inference this whole area is written to prevent.
        /// <c>SkillSandboxCapabilities</c> already reported both gaps; the sentence a
        /// human actually reads did not.
        /// </para>
        /// </summary>
        public string Describe() =>
            "bounds the process tree: kills every child when the request ends, caps the process count and "
            + "committed memory. It does NOT confine writes and does NOT block the network - TensorSharp's "
            + "current Windows backend has no shell-confinement mechanism, so a script writes and reaches the network "
            + "with this process's own access";

        /// <summary>
        /// Nothing to rewrite — a job object attaches to a process that is already
        /// running, through <see cref="TryAttach"/>.
        /// </summary>
        public bool TryWrap(
            SkillSandboxRequest request,
            out string fileName,
            out IReadOnlyList<string> arguments,
            out IDisposable cleanup,
            out string error)
        {
            fileName = request.Interpreter;
            arguments = request.Arguments;
            cleanup = null!;
            error = null!;
            return true;
        }

        /// <inheritdoc />
        public bool TryAttach(SpawnedProcess process, out string error)
        {
            error = null!;
            if (!OperatingSystem.IsWindows())
                return true;

            IntPtr job = IntPtr.Zero;
            try
            {
                job = CreateJobObjectW(IntPtr.Zero, null);
                if (job == IntPtr.Zero)
                {
                    error = $"CreateJobObject failed (Win32 error {Marshal.GetLastWin32Error()})";
                    return false;
                }

                var limits = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION
                {
                    BasicLimitInformation = new JOBOBJECT_BASIC_LIMIT_INFORMATION
                    {
                        // Killing the tree when the handle closes is the property worth
                        // having: without it a script that forks and returns leaves its
                        // children running after the request that started them is gone.
                        LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                                   | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                                   | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION,
                        ActiveProcessLimit = MaxProcesses,
                    },
                    ProcessMemoryLimit = (UIntPtr)MaxProcessMemoryBytes,
                };
                limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;

                int size = Marshal.SizeOf<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
                IntPtr buffer = Marshal.AllocHGlobal(size);
                try
                {
                    Marshal.StructureToPtr(limits, buffer, fDeleteOld: false);
                    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, buffer, (uint)size))
                    {
                        error = $"SetInformationJobObject failed (Win32 error {Marshal.GetLastWin32Error()})";
                        return false;
                    }
                }
                finally
                {
                    Marshal.FreeHGlobal(buffer);
                }

                // SpawnedProcess.Handle is the Windows process handle, and non-zero only
                // where the child was started through System.Diagnostics.Process — which on
                // Windows it always is, because there is no posix_spawn and no fork to
                // avoid. A zero handle means this is not the platform this sandbox is for.
                if (process.Handle == IntPtr.Zero)
                {
                    error = "the process was not started with a Windows process handle";
                    return false;
                }

                if (!AssignProcessToJobObject(job, process.Handle))
                {
                    error = $"AssignProcessToJobObject failed (Win32 error {Marshal.GetLastWin32Error()})";
                    return false;
                }

                // The handle is deliberately leaked into the process's lifetime: the
                // job dies with the handle, so closing it here would kill the child we
                // just started. It is released when this process exits, and
                // SkillScriptRunner kills the tree explicitly on timeout regardless.
                job = IntPtr.Zero;
                return true;
            }
            catch (Exception ex) when (ex is EntryPointNotFoundException or DllNotFoundException
                                          or InvalidOperationException or ExternalException)
            {
                error = $"the job object could not be created ({ex.Message})";
                return false;
            }
            finally
            {
                if (job != IntPtr.Zero)
                    CloseHandle(job);
            }
        }

        /// <summary>Most processes the script's tree may hold open at once.</summary>
        private const int MaxProcesses = 32;

        /// <summary>Committed-memory ceiling per process.</summary>
        private const ulong MaxProcessMemoryBytes = 2UL * 1024 * 1024 * 1024;

        private const int JobObjectExtendedLimitInformation = 9;
        private const uint JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008;
        private const uint JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100;
        private const uint JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400;
        private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObjectW(IntPtr lpJobAttributes, string? lpName);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetInformationJobObject(
            IntPtr hJob, int jobObjectInfoClass, IntPtr lpJobObjectInfo, uint cbJobObjectInfoLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CloseHandle(IntPtr hObject);

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IO_COUNTERS
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }
    }
}
