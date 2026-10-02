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

namespace TensorAgent.Core.Interop;

/// <summary>
/// Gives an existing file a second name: a hard link, which is how two catalog entries
/// that list the same artifact hold one copy of its bytes between them (see
/// <see cref="Catalog.ModelStore"/>).
///
/// <para>
/// .NET can make a symbolic link (<see cref="File.CreateSymbolicLink"/>) but not a hard
/// one, and a symbolic link is the wrong tool here: it names the other entry's PATH, so
/// deleting that entry would leave this one with a dangling name and a model that no
/// longer loads. A hard link is a second directory entry for the same file, and the bytes
/// stay on disk until the last name is gone, so either entry can be deleted without
/// touching the other.
/// </para>
/// <para>
/// <c>link(2)</c> is in the C library of every system this runs on that is not Windows --
/// macOS, iOS, Mac Catalyst and Linux -- and the runtime resolves <c>"libc"</c> to each
/// one's own, so one import serves them all. Windows has <c>CreateHardLinkW</c>.
/// </para>
/// </summary>
internal static class HardLinks
{
    /// <summary>
    /// Make <paramref name="newName"/> a second name for <paramref name="existingFile"/>.
    /// Throws <see cref="IOException"/> with the system's reason when it cannot: the two
    /// are on different volumes, the file system has no hard links, or something already
    /// has that name, which is refused rather than replaced.
    /// </summary>
    public static void Create(string existingFile, string newName)
    {
        ArgumentException.ThrowIfNullOrEmpty(existingFile);
        ArgumentException.ThrowIfNullOrEmpty(newName);

        bool linked;
        try
        {
            linked = OperatingSystem.IsWindows()
                ? CreateHardLink(newName, existingFile, IntPtr.Zero)
                : Link(existingFile, newName) == 0;
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            throw new IOException($"this system offers no hard links ({ex.Message})", ex);
        }

        if (!linked)
        {
            int error = Marshal.GetLastPInvokeError();
            throw new IOException($"{Marshal.GetPInvokeErrorMessage(error)} (error {error})");
        }
    }

    [DllImport("libc", EntryPoint = "link", SetLastError = true)]
    private static extern int Link(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string existing,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string newName);

    [DllImport("kernel32.dll", EntryPoint = "CreateHardLinkW", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateHardLink(string newName, string existing, IntPtr securityAttributes);
}
