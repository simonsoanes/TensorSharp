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
using System.IO;
using System.Linq;
using System.Text;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>How hard a host insists on OS-level isolation for a skill's scripts.</summary>
    public enum SkillSandboxMode
    {
        /// <summary>
        /// Run the script directly, with only the in-process limits
        /// (<see cref="SkillScriptRunner"/>'s working directory, argument vector,
        /// timeout and output cap). No filesystem or network confinement.
        /// </summary>
        Off,

        /// <summary>
        /// Sandbox when the platform provides one, run unsandboxed when it does not.
        /// A developer's own machine is the case this is for; it is the wrong choice
        /// for anything that accepts skill uploads, because "no sandbox available"
        /// silently becomes "no sandbox".
        /// </summary>
        Preferred,

        /// <summary>
        /// Sandbox or refuse. The default whenever script execution is enabled: a
        /// host that cannot isolate a script should say so rather than run it, and the
        /// model is told plainly that the tool is unavailable here.
        /// </summary>
        Required,
    }

    /// <summary>What a sandbox is asked to confine one script run to.</summary>
    /// <param name="Interpreter">Absolute path or PATH name of the interpreter to launch.</param>
    /// <param name="Arguments">The interpreter's argument vector, script path first.</param>
    /// <param name="SkillDirectory">The skill's own directory. Readable, never writable.</param>
    /// <param name="WorkDirectory">
    /// A scratch directory that becomes the process's working directory and is the only
    /// place it may write. Anything the script produces lands here, where the caller can
    /// find it — and a script that tries to edit its own skill, the model's uploads, or
    /// anything else on the host fails instead.
    /// </param>
    /// <param name="AllowNetwork">Let the script reach the network. Off by default.</param>
    /// <param name="ReadablePaths">Extra paths the script may read, beyond the system and its own skill.</param>
    public readonly record struct SkillSandboxRequest(
        string Interpreter,
        IReadOnlyList<string> Arguments,
        string SkillDirectory,
        string WorkDirectory,
        bool AllowNetwork,
        IReadOnlyList<string> ReadablePaths)
    {
        /// <summary>
        /// When set (and <see cref="AllowNetwork"/> is false), the one TCP port on
        /// localhost the process may connect to. This is how an install phase is
        /// given a package registry without the whole internet: the host runs an
        /// egress proxy with a domain allowlist on this port, HTTPS_PROXY points the
        /// installer at it, and the sandbox admits exactly that loopback port —
        /// every other destination stays denied at the OS level.
        /// </summary>
        public int? AllowLoopbackPort { get; init; }
    }

    /// <summary>
    /// What a sandbox actually enforces.
    ///
    /// <para>
    /// The three platforms do not offer the same primitives, and pretending otherwise
    /// would be the worst outcome: an operator who reads "sandboxed" and gets only
    /// process-lifetime bounds has been misled. Every sandbox states its guarantees
    /// here, the runner reports the ones that are MISSING in the result the model sees
    /// and in the startup log, and the documentation is generated from the same record
    /// so it cannot drift.
    /// </para>
    /// </summary>
    /// <param name="ConfinesWrites">The script can only write to its scratch directory.</param>
    /// <param name="ConfinesNetwork">The script cannot open a socket.</param>
    /// <param name="ConfinesHomeReads">The script cannot read the user's home directory — credentials, keys, other skills.</param>
    /// <param name="BoundsProcessTree">Children are killed with the parent, so nothing outlives the request.</param>
    public readonly record struct SkillSandboxCapabilities(
        bool ConfinesWrites,
        bool ConfinesNetwork,
        bool ConfinesHomeReads,
        bool BoundsProcessTree)
    {
        /// <summary>The properties this sandbox does NOT provide, phrased for a human.</summary>
        public IReadOnlyList<string> Gaps()
        {
            var gaps = new List<string>();
            if (!ConfinesWrites) gaps.Add("the script may write anywhere the host process can");
            if (!ConfinesNetwork) gaps.Add("the script may reach the network");
            if (!ConfinesHomeReads) gaps.Add("the script may read the user's home directory");
            if (!BoundsProcessTree) gaps.Add("a child process may outlive the request");
            return gaps;
        }
    }

    /// <summary>An OS mechanism that can confine a child process.</summary>
    public interface ISkillSandbox
    {
        /// <summary>Short name for logs and for the result the model sees (<c>sandbox-exec</c>, <c>bubblewrap</c>).</summary>
        string Name { get; }

        /// <summary>False when the mechanism is not present or not usable on this host.</summary>
        bool IsAvailable { get; }

        /// <summary>What this sandbox enforces. See <see cref="SkillSandboxCapabilities"/>.</summary>
        SkillSandboxCapabilities Capabilities { get; }

        /// <summary>
        /// Called after the child has started, for mechanisms that attach to a running
        /// process rather than wrapping its command line (a Windows job object). A
        /// wrapper-style sandbox does nothing here.
        /// </summary>
        /// <returns>False when the child could not be confined and must be killed.</returns>
        bool TryAttach(Process process, out string error) { error = null!; return true; }

        /// <summary>
        /// One line describing what this sandbox actually enforces, so a host can log
        /// it and the docs cannot drift from the implementation.
        /// </summary>
        string Describe();

        /// <summary>
        /// Rewrite <paramref name="request"/> into the command that runs it confined.
        /// </summary>
        /// <param name="fileName">The executable to launch (the sandbox helper, not the interpreter).</param>
        /// <param name="arguments">Its full argument vector.</param>
        /// <param name="cleanup">
        /// Disposed after the run — a generated profile file, a temporary mount point.
        /// Null when the sandbox needs no scratch state.
        /// </param>
        /// <param name="error">Why the request could not be wrapped, or null.</param>
        bool TryWrap(
            SkillSandboxRequest request,
            out string fileName,
            out IReadOnlyList<string> arguments,
            out IDisposable cleanup,
            out string error);
    }

    /// <summary>
    /// Picks the strongest sandbox this host actually provides.
    ///
    /// <para>
    /// Detection is by probing, not by guessing from the OS: <c>sandbox-exec</c> is on
    /// every macOS but has been deprecated for years and could be removed, and
    /// <c>bwrap</c> is on most Linux desktops but almost no containers. A host that
    /// reports which sandbox is in force — and refuses to run scripts when the answer
    /// is "none" — is the difference between a security property and a hope.
    /// </para>
    /// </summary>
    public static class SkillSandboxFactory
    {
        private static readonly object Gate = new();
        private static ISkillSandbox? _detected;
        private static bool _probed;

        /// <summary>The available sandbox, or null when this host provides none.</summary>
        public static ISkillSandbox? Detect()
        {
            lock (Gate)
            {
                if (_probed)
                    return _detected;

                _probed = true;
                foreach (ISkillSandbox candidate in Candidates())
                {
                    if (!candidate.IsAvailable)
                        continue;
                    _detected = candidate;
                    break;
                }
                return _detected;
            }
        }

        private static IEnumerable<ISkillSandbox> Candidates()
        {
            if (OperatingSystem.IsMacOS())
                yield return new SeatbeltSandbox();
            if (OperatingSystem.IsLinux())
                yield return new BubblewrapSandbox();
            if (OperatingSystem.IsWindows())
                yield return new WindowsJobObjectSandbox();
        }

        /// <summary>
        /// A one-line summary of this host's isolation, for the startup banner and the
        /// <c>--list-skills</c> footer.
        /// </summary>
        public static string DescribeHost()
        {
            ISkillSandbox? sandbox = Detect();
            if (sandbox == null)
                return "no OS sandbox available on this platform";

            IReadOnlyList<string> gaps = sandbox.Capabilities.Gaps();
            return gaps.Count == 0
                ? $"{sandbox.Name}: {sandbox.Describe()}"
                : $"{sandbox.Name}: {sandbox.Describe()}. NOT confined: {string.Join("; ", gaps)}";
        }
    }

    /// <summary>
    /// macOS Seatbelt, driven through <c>/usr/bin/sandbox-exec</c> and a generated
    /// SBPL profile.
    ///
    /// <para>
    /// The profile denies everything, then re-allows the narrowest set a scripting
    /// interpreter actually needs. Verified against a probe script that tries each
    /// escape: reading <c>~/.ssh</c> fails, writing anywhere but the scratch directory
    /// fails, and opening a socket fails, while <c>python3</c> still starts and imports
    /// its standard library.
    /// </para>
    /// <para>
    /// Reads of system paths stay allowed, because narrowing them further stops the
    /// interpreter from loading at all (an earlier profile that whitelisted
    /// <c>/usr</c>, <c>/System</c> and <c>/Library</c> by subpath aborted CPython with
    /// SIGABRT before it reached <c>main</c>). What matters is closed: the user's home
    /// directory — where credentials, SSH keys and every other skill live — is denied
    /// wholesale, and only the running skill's own directory is punched back through.
    /// </para>
    /// </summary>
    internal sealed class SeatbeltSandbox : ISkillSandbox
    {
        private const string Helper = "/usr/bin/sandbox-exec";

        public string Name => "sandbox-exec";

        public bool IsAvailable => OperatingSystem.IsMacOS() && File.Exists(Helper);

        public SkillSandboxCapabilities Capabilities => new(
            ConfinesWrites: true,
            ConfinesNetwork: true,
            ConfinesHomeReads: true,
            BoundsProcessTree: true);

        public string Describe() =>
            "denies network (unix sockets scoped to /tmp and the scratch directory), denies reads and " +
            "file metadata of the user's home directory, and confines writes to the run's scratch directory";

        public bool TryWrap(
            SkillSandboxRequest request,
            out string fileName,
            out IReadOnlyList<string> arguments,
            out IDisposable cleanup,
            out string error)
        {
            fileName = null!;
            arguments = Array.Empty<string>();
            cleanup = null!;
            error = null!;

            if (!IsAvailable)
            {
                error = "sandbox-exec is not present on this host";
                return false;
            }

            // The profile goes somewhere the confined process CANNOT reach, under a name
            // it cannot predict. It used to live in the write directory — which is, by
            // definition, the one place the process being confined is allowed to write —
            // at a fixed name. sandbox-exec reads the file at exec time, so anything
            // already running inside that directory could overwrite it between this write
            // and that read and be governed by `(allow default)` instead. That was only
            // theoretical while every run was a single short-lived process; a shell that
            // can leave a background job running across later calls supplies exactly the
            // process needed to sit in that loop.
            string profilePath;
            try
            {
                profilePath = Path.Combine(
                    Path.GetTempPath(),
                    ".tensorsharp-sandbox-" + Guid.NewGuid().ToString("N") + ".sb");
                File.WriteAllText(profilePath, BuildProfile(request));
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"the sandbox profile could not be written: {ex.Message}";
                return false;
            }

            var argv = new List<string> { "-f", profilePath, request.Interpreter };
            argv.AddRange(request.Arguments);

            fileName = Helper;
            arguments = argv;
            cleanup = new FileCleanup(profilePath);
            return true;
        }

        /// <summary>
        /// Build the SBPL profile. Rules are evaluated in order with the LAST match
        /// winning, which is what lets a broad allow be carved back by a narrower deny
        /// and then punched through again for one directory.
        /// </summary>
        private static string BuildProfile(SkillSandboxRequest request)
        {
            var sb = new StringBuilder();
            sb.AppendLine("(version 1)");
            sb.AppendLine("(deny default)");
            if (request.AllowNetwork)
            {
                sb.AppendLine("(allow network*)");
            }
            else
            {
                // Deny the NETWORK, but not local IPC. A blanket `(deny network*)`
                // also blocks AF_UNIX sockets, which are how a local tool coordinates
                // with itself — LibreOffice's headless bootstrap opens a unix-domain
                // singleton pipe and, denied it, exits non-zero with a half-built
                // profile, so the xlsx skill's recalc.py could never recalculate a
                // sheet under the sandbox.
                //
                // But "unix sockets" is not one thing: a host unix socket is a door
                // into whatever service listens on it — /var/run/docker.sock is root
                // in a trench coat, and launchd keeps the ssh-agent listener under
                // /private/tmp. So the allowance is scoped BY SOCKET PATH (the same
                // shape Anthropic's sandbox-runtime uses): sockets under the shared
                // system temp and the run's own scratch directory work, which is
                // exactly what self-coordinating tools create, and every other
                // socket on the host stays out of reach. The launchd listeners that
                // live under /private/tmp are carved back out explicitly.
                sb.AppendLine("(deny network*)");
                sb.AppendLine("(allow system-socket (socket-domain AF_UNIX))");
                var socketRoots = new List<string> { "/private/tmp" };
                socketRoots.AddRange(Forms(request.WorkDirectory));
                foreach (string root in socketRoots)
                {
                    sb.Append("(allow network-bind (local unix-socket (subpath ").Append(Quote(root)).AppendLine(")))");
                    sb.Append("(allow network-outbound (remote unix-socket (subpath ").Append(Quote(root)).AppendLine(")))");
                }
                sb.AppendLine("(deny network-outbound (remote unix-socket (regex #\"^/private/tmp/com\\.apple\\.launchd\")))");

                // The install phase's registry proxy: one loopback TCP port, and
                // nothing else. The proxy on the host side enforces the domain
                // allowlist; this rule is what makes the proxy the ONLY way out.
                if (request.AllowLoopbackPort is int port and > 0 and <= 65535)
                {
                    sb.Append("(allow network-outbound (remote ip \"localhost:")
                      .Append(port).AppendLine("\"))");
                }
            }

            // The interpreter needs to fork/exec, read sysctls, talk to the bootstrap
            // server for dyld, and signal itself. Without these CPython aborts before
            // running a single line of the script.
            sb.AppendLine("(allow process-fork process-exec)");
            sb.AppendLine("(allow sysctl-read mach-lookup ipc-posix-shm)");
            sb.AppendLine("(allow signal (target self))");

            // Reads: broad, then the user's home carved out. Narrowing reads to a
            // system whitelist kills the interpreter (see the class remarks); what has
            // to be closed is the home directory, and it is.
            sb.AppendLine("(allow file-read*)");
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (!string.IsNullOrEmpty(home))
            {
                foreach (string form in Forms(home))
                    sb.Append("(deny file-read* (subpath ").Append(Quote(form)).AppendLine("))");
            }

            // Metadata: readable everywhere OUTSIDE the home directory (a denial on
            // system paths surfaces as a confusing "file not found" from deep inside
            // a library rather than a permission error) — but hidden for FILES under
            // home. Existence is information: a script that can os.path.exists() its
            // way through ~/.ssh and ~/.aws learns which credentials this machine
            // holds even though every open() is denied. Directories stay visible
            // (vnode-type DIRECTORY) because the kernel checks each ancestor on the
            // way into the re-allowed subtrees below, and Seatbelt gives the more
            // specific file-read-metadata operation precedence over the general
            // file-read* allows — which is also why each re-allowed subtree needs
            // its own explicit metadata rule.
            sb.AppendLine("(allow file-read-metadata)");
            if (!string.IsNullOrEmpty(home))
            {
                foreach (string form in Forms(home))
                    sb.Append("(deny file-read-metadata (subpath ").Append(Quote(form)).AppendLine("))");
                sb.AppendLine("(allow file-read-metadata (vnode-type DIRECTORY))");
            }

            var readableRoots = new List<string>();
            readableRoots.AddRange(Forms(request.SkillDirectory));
            foreach (string extra in request.ReadablePaths ?? Array.Empty<string>())
            {
                if (string.IsNullOrWhiteSpace(extra))
                    continue;
                readableRoots.AddRange(Forms(extra));
            }
            foreach (string form in readableRoots)
            {
                sb.Append("(allow file-read* (subpath ").Append(Quote(form)).AppendLine("))");
                sb.Append("(allow file-read-metadata (subpath ").Append(Quote(form)).AppendLine("))");
            }

            // Writes: the scratch directory and the null device, nothing else.
            foreach (string form in Forms(request.WorkDirectory))
            {
                sb.Append("(allow file-read* file-write* (subpath ").Append(Quote(form)).AppendLine("))");
                sb.Append("(allow file-read-metadata (subpath ").Append(Quote(form)).AppendLine("))");
            }
            sb.AppendLine("(allow file-write-data (literal \"/dev/null\"))");
            sb.AppendLine("(allow file-ioctl (literal \"/dev/null\") (literal \"/dev/urandom\"))");

            // Apple's /usr/bin/python3 is an xcrun shim that writes a cache database
            // into the per-user Darwin temp directory, which it locates through
            // confstr() rather than TMPDIR, so no environment scrubbing can redirect
            // it. Denying it costs nothing functionally — the interpreter still runs —
            // but it prints a permission error to stderr on EVERY run, and that stderr
            // goes to the model as part of the tool result, where it reads as the
            // script having failed. Allowing exactly that file name keeps the result
            // honest without opening the directory.
            sb.AppendLine("(allow file-read* file-write* (regex #\"^/private/var/folders/[^/]+/[^/]+/T/xcrun_db\"))");

            // The shared system temp, /tmp. A whole class of tools a skill invokes
            // keeps a FIXED-path scratch or singleton-IPC node here, independent of
            // TMPDIR and HOME (both already redirected into the workdir): LibreOffice's
            // headless converter opens /private/tmp/OSL_PIPE_<uid>_SingleOfficeIPC_<hash>
            // and, denied it, exits without building its profile — which is why the xlsx
            // skill's recalc.py could never recalculate a sheet under the sandbox. This
            // is not a hole: /private/tmp is world-writable already (mode 1777 — every
            // process on the host shares it), the confinement's guarantees that matter
            // stay intact — the home directory is still unreadable and the network is
            // still closed — and the session's own files live under the server's scratch
            // root, never here. The per-user Darwin temp (/var/folders/.../T) is left
            // denied; nothing needed it once /private/tmp was open.
            sb.AppendLine("(allow file-read* file-write* (subpath \"/private/tmp\"))");

            return sb.ToString();
        }

        /// <summary>
        /// Every spelling of <paramref name="path"/> a <c>subpath</c> rule might need to
        /// match: the path as given and, when they differ, the path with every symlinked
        /// ancestor resolved.
        ///
        /// <para>
        /// Resolving only the leaf is not enough, and getting this wrong silently
        /// TIGHTENS the sandbox rather than loosening it — which is how it was found.
        /// On macOS the system temp directory is <c>/var/folders/...</c> and <c>/var</c>
        /// is a symlink to <c>/private/var</c>; the scratch directory itself is not a
        /// link, so a leaf-only resolve returns the <c>/var</c> spelling, the kernel
        /// checks the <c>/private/var</c> one, no rule matches, and the script cannot
        /// write to its own working directory.
        /// </para>
        /// </summary>
        private static IReadOnlyList<string> Forms(string path)
        {
            string full;
            try
            {
                full = Path.GetFullPath(path);
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                return new[] { path };
            }

            string resolved = ResolveThroughAncestors(full);
            return string.Equals(resolved, full, StringComparison.Ordinal)
                ? new[] { full }
                : new[] { full, resolved };
        }

        /// <summary>
        /// Walk from the filesystem root, following each component that is a symlink, so
        /// the result is the path the kernel sees.
        /// </summary>
        private static string ResolveThroughAncestors(string fullPath)
        {
            try
            {
                string current = Path.DirectorySeparatorChar.ToString();
                foreach (string part in fullPath.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries))
                {
                    current = Path.Combine(current, part);
                    FileSystemInfo info = Directory.Exists(current) ? new DirectoryInfo(current) : new FileInfo(current);
                    if (!info.Exists)
                        continue;
                    FileSystemInfo? target = info.ResolveLinkTarget(returnFinalTarget: true);
                    if (target != null)
                        current = target.FullName;
                }
                return current;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException or PathTooLongException)
            {
                return fullPath;
            }
        }

        /// <summary>
        /// Quote a path for SBPL. A path is host-controlled rather than model-controlled
        /// (it is the skill root the operator configured plus a GUID), but a stray quote
        /// would silently truncate the profile and widen the sandbox, so it is escaped
        /// rather than trusted.
        /// </summary>
        private static string Quote(string path) =>
            "\"" + path.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

        private sealed class FileCleanup : IDisposable
        {
            private readonly string _path;

            public FileCleanup(string path) => _path = path;

            public void Dispose()
            {
                try { File.Delete(_path); }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
            }
        }
    }

    /// <summary>
    /// Linux namespaces, driven through <c>bwrap</c> (bubblewrap).
    ///
    /// <para>
    /// Bubblewrap is the unprivileged sandbox Flatpak is built on, and it is the only
    /// widely deployed mechanism a .NET process can drive without root or a helper
    /// daemon. It is present on most desktop distributions and almost no container
    /// images, which is exactly why <see cref="SkillSandboxMode.Required"/> is the
    /// default: a server that cannot isolate a script refuses instead of running it.
    /// </para>
    /// </summary>
    internal sealed class BubblewrapSandbox : ISkillSandbox
    {
        private static readonly Lazy<string?> Located = new(Locate);

        public string Name => "bubblewrap";

        public bool IsAvailable => OperatingSystem.IsLinux() && Located.Value != null;

        public SkillSandboxCapabilities Capabilities => new(
            ConfinesWrites: true,
            ConfinesNetwork: true,
            ConfinesHomeReads: true,
            BoundsProcessTree: true);

        public string Describe() =>
            "unshares the network and PID namespaces, mounts the filesystem read-only, and confines writes to the run's scratch directory";

        public bool TryWrap(
            SkillSandboxRequest request,
            out string fileName,
            out IReadOnlyList<string> arguments,
            out IDisposable cleanup,
            out string error)
        {
            fileName = null!;
            arguments = Array.Empty<string>();
            cleanup = null!;
            error = null!;

            string? bwrap = Located.Value;
            if (bwrap == null)
            {
                error = "bwrap (bubblewrap) is not installed on this host";
                return false;
            }

            var argv = new List<string>
            {
                // Everything read-only, then the pieces that must be writable bound
                // back over it. --die-with-parent is what stops an abandoned script
                // outliving the request that started it.
                "--ro-bind", "/", "/",
                "--dev", "/dev",
                "--proc", "/proc",
                "--tmpfs", "/tmp",
                "--die-with-parent",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
            };

            if (!request.AllowNetwork)
                argv.Add("--unshare-net");

            // The user's home is replaced by an empty tmpfs rather than merely made
            // read-only: credentials and every other installed skill live there, and a
            // script that can read them can put them in its stdout and thus in the
            // model's context.
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (!string.IsNullOrEmpty(home) && Directory.Exists(home))
            {
                argv.Add("--tmpfs");
                argv.Add(home);
            }

            argv.Add("--ro-bind");
            argv.Add(request.SkillDirectory);
            argv.Add(request.SkillDirectory);

            foreach (string extra in request.ReadablePaths ?? Array.Empty<string>())
            {
                if (string.IsNullOrWhiteSpace(extra) || !Directory.Exists(extra))
                    continue;
                argv.Add("--ro-bind");
                argv.Add(extra);
                argv.Add(extra);
            }

            argv.Add("--bind");
            argv.Add(request.WorkDirectory);
            argv.Add(request.WorkDirectory);
            argv.Add("--chdir");
            argv.Add(request.WorkDirectory);

            argv.Add("--");
            argv.Add(request.Interpreter);
            argv.AddRange(request.Arguments);

            fileName = bwrap;
            arguments = argv;
            cleanup = NullCleanup.Instance;
            return true;
        }

        private static string? Locate()
        {
            if (!OperatingSystem.IsLinux())
                return null;
            foreach (string candidate in new[] { "/usr/bin/bwrap", "/bin/bwrap", "/usr/local/bin/bwrap" })
            {
                if (File.Exists(candidate))
                    return candidate;
            }
            return null;
        }

        private sealed class NullCleanup : IDisposable
        {
            public static readonly NullCleanup Instance = new();

            public void Dispose() { }
        }
    }
}
