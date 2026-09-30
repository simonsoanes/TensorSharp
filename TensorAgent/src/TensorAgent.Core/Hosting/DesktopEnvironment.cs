// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// What a desktop app does not inherit that a host started from a terminal does.
///
/// <para>
/// A Mac app launched from the Finder or the Dock gets launchd's PATH
/// (<c>/usr/bin:/bin:/usr/sbin:/sbin</c>), not the one the user's shell profile builds.
/// The desktop backend runs node, npm, npx and python3 as ordinary child processes, so
/// with that PATH Homebrew's copies are simply not found and the Playwright skill fails
/// on its first step. The login shell is asked once, at startup, for the PATH it would
/// give a terminal; its entries go first and the app's own stay after them.
/// </para>
/// </summary>
public static class DesktopEnvironment
{
    /// <summary>Where a Mac usually keeps user-installed tools, tried when the shell cannot answer.</summary>
    internal static readonly string[] WellKnownMacDirectories =
    {
        "/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin",
    };

    /// <summary>
    /// On macOS (including a Mac Catalyst app), put the login shell's PATH in front of this
    /// process's own and return the result; elsewhere, or when nothing is found to add,
    /// leave PATH alone and return null.
    /// </summary>
    public static string? ImportLoginShellPath(TimeSpan? timeout = null)
    {
        if (!OperatingSystem.IsMacOS() && !OperatingSystem.IsMacCatalyst())
            return null;

        string current = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        string? fromShell = ReadLoginShellPath(timeout ?? TimeSpan.FromSeconds(5));
        IEnumerable<string> fallback = fromShell is null
            ? WellKnownMacDirectories.Where(Directory.Exists)
            : Array.Empty<string>();
        string merged = MergePath(fromShell, fallback, current);
        if (string.Equals(merged, current, StringComparison.Ordinal))
            return null;
        Environment.SetEnvironmentVariable("PATH", merged);
        return merged;
    }

    /// <summary>
    /// The login shell's entries, then <paramref name="extra"/>, then the current ones, each
    /// directory once and in its first position. Empty entries are dropped.
    /// </summary>
    internal static string MergePath(string? loginShellPath, IEnumerable<string> extra, string current)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var result = new List<string>();
        void Add(IEnumerable<string> entries)
        {
            foreach (string entry in entries)
            {
                string trimmed = entry.Trim();
                if (trimmed.Length > 0 && seen.Add(trimmed))
                    result.Add(trimmed);
            }
        }

        Add((loginShellPath ?? string.Empty).Split(Path.PathSeparator));
        Add(extra);
        Add(current.Split(Path.PathSeparator));
        return string.Join(Path.PathSeparator, result);
    }

    /// <summary>The PATH a login shell would set, or null when the shell cannot be asked in time.</summary>
    private static string? ReadLoginShellPath(TimeSpan timeout)
    {
        string shell = Environment.GetEnvironmentVariable("SHELL") is { Length: > 0 } configured && File.Exists(configured)
            ? configured
            : "/bin/zsh";
        try
        {
            var start = new ProcessStartInfo(shell)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                RedirectStandardInput = true,
                UseShellExecute = false,
            };
            // A login shell reads the profile files (Homebrew's shellenv lives there); the
            // marker keeps anything the profile itself prints out of the answer.
            start.ArgumentList.Add("-l");
            start.ArgumentList.Add("-c");
            start.ArgumentList.Add("printf '__TENSORAGENT_PATH__%s' \"$PATH\"");
            using var process = Process.Start(start);
            if (process is null)
                return null;
            process.StandardInput.Close();
            Task<string> output = process.StandardOutput.ReadToEndAsync();
            _ = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit((int)timeout.TotalMilliseconds))
            {
                try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                return null;
            }
            const string Marker = "__TENSORAGENT_PATH__";
            string text = output.Result;
            int at = text.LastIndexOf(Marker, StringComparison.Ordinal);
            if (process.ExitCode != 0 || at < 0)
                return null;
            string path = text[(at + Marker.Length)..].Trim();
            return path.Length > 0 ? path : null;
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException or InvalidOperationException)
        {
            return null;
        }
    }
}
