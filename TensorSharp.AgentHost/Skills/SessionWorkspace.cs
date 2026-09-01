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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Enumeration;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// The one walk of a session's working directory, and the one list of directories
    /// it does not descend into.
    ///
    /// <para>
    /// Every shell command walks this tree twice — once before, for the "what was here
    /// already" snapshot, and once after, to capture what the command produced — so the
    /// cost of the walk is paid by <c>echo hi</c> exactly as much as by a build. A real
    /// session installs packages, and then the working directory holds a
    /// <c>node_modules</c> or a <c>.venv</c>: tens of thousands of files that no filter
    /// applied AFTER enumeration can save, because the enumeration has already stat'd
    /// every one of them. Measured on a 20k-file <c>node_modules</c>, the two walks cost
    /// 182 ms per command against 0.02 ms on an empty workspace — a trivial command
    /// spending a fifth of a second looking at files nobody asked about.
    /// </para>
    /// <para>
    /// So the junk is skipped at the only place where skipping is free: the recursion
    /// itself. <see cref="Files"/> never opens a pruned directory, and the two call
    /// sites share this list rather than each keeping their own — a snapshot that
    /// descends where the capture does not would report every file under it as newly
    /// produced.
    /// </para>
    /// </summary>
    internal static class WorkspaceScan
    {
        /// <summary>
        /// Directory names that are a tool's own storage wherever they appear.
        ///
        /// <para>
        /// At ANY depth, because that is where they actually occur: npm nests a
        /// <c>node_modules</c> inside a package whose dependency versions conflict, a
        /// virtualenv lives at <c>myproject/.venv</c> rather than at the workspace root,
        /// and CPython drops a <c>__pycache__</c> beside every package it imports.
        /// </para>
        /// </summary>
        private static readonly HashSet<string> PrunedAnywhere = new(StringComparer.Ordinal)
        {
            "node_modules", "__pycache__", ".venv", ".git",
            ".npm", ".cargo", ".gradle", ".m2",
            ".config", ".cache", ".local", ".fontconfig",
            ".pytest_cache", ".ruff_cache", ".mypy_cache",
        };

        /// <summary>
        /// Directory names pruned only directly under the working directory.
        ///
        /// <para>
        /// These are fallout from HOME pointing at the working directory, so the root is
        /// the only place they can be that. <c>Library</c> in particular is an ordinary
        /// English word — a model asked to organise documents may well create one — and
        /// pruning it at depth would silently swallow the user's own output.
        /// </para>
        /// <para>
        /// <c>AppData</c> and <c>pip</c> are the Windows half of the same list, and it was
        /// missing: PowerShell creates <c>AppData\Roaming</c> the moment it starts, and
        /// pip writes a cache tree, both directly in the working directory. Two commands
        /// into a session the model's own <c>Get-ChildItem</c> showed them beside its
        /// files, and artifact capture counted them as things the run had produced. Both
        /// are ordinary-looking names, so like <c>Library</c> they are pruned at the root
        /// only — a <c>pip</c> directory the model deliberately creates inside a project it
        /// is building is its own output and must survive.
        /// </para>
        /// </summary>
        private static readonly HashSet<string> PrunedAtRoot = new(StringComparer.Ordinal)
        {
            "Library", ".jobs", "AppData", "pip",
        };

        private static readonly HashSet<string>.AlternateLookup<ReadOnlySpan<char>> AnywhereLookup =
            PrunedAnywhere.GetAlternateLookup<ReadOnlySpan<char>>();

        private static readonly HashSet<string>.AlternateLookup<ReadOnlySpan<char>> AtRootLookup =
            PrunedAtRoot.GetAlternateLookup<ReadOnlySpan<char>>();

        /// <summary>
        /// The enumeration terms of the "what was here before" snapshot: everything,
        /// hidden files included, but never through a symbolic link.
        ///
        /// <para>
        /// The link rule is not tidiness. <c>ln -s . loop</c> is one ordinary command a
        /// model may run inside its own workspace, and a walk that follows it re-walks
        /// the workspace once per level until the kernel's symlink limit stops it —
        /// measured at 3,747 entries and 55 ms from a working directory holding a single
        /// file, and multiplying with every file added after. <c>ln -s / x</c> is worse.
        /// Capture already skips reparse points for a different reason (it copies as the
        /// unsandboxed host, so following a link out of the workspace would hand the user
        /// a download link to the host's home), and the two walks must agree.
        /// </para>
        /// <para>
        /// A directory that cannot be read is stepped over rather than ending the walk,
        /// which is what the capture side already does. The old snapshot let the
        /// exception out and kept whatever it had gathered so far, so one unreadable
        /// directory silently truncated the "what was here before" picture — and every
        /// file the walk never reached then looked newly produced.
        /// </para>
        /// </summary>
        internal static readonly EnumerationOptions SnapshotOptions = new()
        {
            RecurseSubdirectories = true,
            AttributesToSkip = FileAttributes.ReparsePoint,
            IgnoreInaccessible = true,
        };

        /// <summary>
        /// Every pruned name and whether it counts only at the root — so a test can pin
        /// the two walks to this one list rather than to a copy of it that drifts.
        /// </summary>
        internal static IEnumerable<(string Name, bool RootOnly)> PrunedNames()
        {
            foreach (string name in PrunedAnywhere)
                yield return (name, false);
            foreach (string name in PrunedAtRoot)
                yield return (name, true);
        }

        /// <summary>True for a directory the walk does not descend into.</summary>
        /// <param name="atRoot">Whether it sits directly in the working directory.</param>
        internal static bool IsPrunedDirectory(ReadOnlySpan<char> name, bool atRoot) =>
            AnywhereLookup.Contains(name) || (atRoot && AtRootLookup.Contains(name));

        /// <summary>The same question for a caller that has a string and a relative path.</summary>
        internal static bool IsPrunedDirectory(string name, bool atRoot) =>
            IsPrunedDirectory(name.AsSpan(), atRoot);

        /// <summary>
        /// Every file under <paramref name="root"/> that is not inside a pruned
        /// directory, as full paths.
        ///
        /// <para>
        /// Identical in what it yields to
        /// <c>Directory.EnumerateFiles(root, "*", options)</c> for everything outside the
        /// pruned directories — same options, same order, same treatment of hidden files
        /// and links — and it simply never opens the ones inside them.
        /// </para>
        /// </summary>
        internal static IEnumerable<string> Files(string root, EnumerationOptions options)
        {
            string normalizedRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
            return new FileSystemEnumerable<string>(
                root,
                static (ref FileSystemEntry entry) => entry.ToFullPath(),
                options)
            {
                ShouldIncludePredicate = static (ref FileSystemEntry entry) => !entry.IsDirectory,
                ShouldRecursePredicate = (ref FileSystemEntry entry) =>
                    !IsPrunedDirectory(entry.FileName, entry.Directory.SequenceEqual(normalizedRoot)),
            };
        }
    }

    /// <summary>
    /// One chat session's persistent execution workspace: the working directory every
    /// <c>shell</c> command and every skill script of that session runs in, and the
    /// package environment their installs accumulate into.
    ///
    /// <para>
    /// The per-call scratch that preceded this was correct for isolation but wrong for
    /// WORK: a pptx generated by one script was deleted before <c>validate.py</c> could
    /// check it, and every call re-installed its packages from scratch. Real tasks are
    /// pipelines — generate, then validate, then convert — so the pipeline's files must
    /// outlive each step. The unit of trust is the SESSION: everything in one
    /// conversation already shares a context, so its steps sharing a disk is no new
    /// exposure, while two different sessions never share a workspace.
    /// </para>
    /// <para>
    /// Lifecycle: created on first use, deleted when the session is disposed or reset
    /// (a new chat starts clean), and swept at server startup — a server restart orphans
    /// every session, so anything left under the root is finished business.
    /// </para>
    /// </summary>
    public sealed class SessionWorkspace
    {
        internal const string DirectoryPrefix = "ts-session-";

        private readonly HashSet<string> _installedPackages = new(StringComparer.OrdinalIgnoreCase);
        private readonly object _gate = new();

        internal SessionWorkspace(string root, string sessionId = "")
        {
            Id = sessionId;
            Root = root;
            WorkDirectory = Path.Combine(root, "work");
            EnvDirectory = Path.Combine(root, "env");
            StateDirectory = Path.Combine(root, "state");
            TempDirectory = Path.Combine(root, "tmp");
            Directory.CreateDirectory(WorkDirectory);
            Directory.CreateDirectory(EnvDirectory);
            Directory.CreateDirectory(StateDirectory);
            Directory.CreateDirectory(TempDirectory);
        }

        /// <summary>
        /// The chat session this workspace belongs to — the key
        /// <see cref="SessionWorkspaceManager.Release"/> takes, so a holder of the
        /// workspace can release it without separately remembering the id.
        /// </summary>
        public string Id { get; }

        /// <summary>The workspace's own directory, holding the three below.</summary>
        public string Root { get; }

        /// <summary>
        /// What the model has actually been shown of each file in this workspace.
        ///
        /// <para>
        /// It hangs off the WORKSPACE rather than off the shell session because the
        /// workspace is what every entry point already receives — the tool dispatch, the
        /// patcher and the editor all take one — while a shell session is reachable only
        /// through a resolved shell that a host without one does not have. It also has
        /// exactly the right lifetime: a conversation's reads are worth nothing to the
        /// next conversation, and the workspace is released with the session.
        /// </para>
        /// </summary>
        public FileLedger Reads { get; } = new();

        /// <summary>
        /// Where everything runs and every file lives: the shell's starting directory,
        /// shared by <c>shell</c> commands and skill scripts alike so one step's output
        /// is the next step's input.
        /// </summary>
        public string WorkDirectory { get; }

        /// <summary>
        /// The session's package environment (<c>pip install --target</c>,
        /// <c>npm --prefix</c>). Reached read-only at run time via PYTHONPATH /
        /// NODE_PATH — by model-written code and by skill scripts equally, which is what
        /// lets a script's dependencies be installed once from the shell and then found
        /// by every later step.
        /// </summary>
        public string EnvDirectory { get; }

        /// <summary>
        /// The host's own bookkeeping for this session: the shell's persisted working
        /// directory and exported environment, the script each command is handed to the
        /// shell as, and the log of every background job.
        ///
        /// <para>
        /// A sibling of <see cref="WorkDirectory"/> rather than a hidden folder inside it,
        /// so that <c>ls</c> shows the model its own files and nothing else, and so that
        /// artifact capture — which scans only the work directory — can never hand the
        /// user a download link to the host's scratch.
        /// </para>
        /// </summary>
        public string StateDirectory { get; }

        /// <summary>
        /// Where <c>TMPDIR</c> points, so a tool's scratch file is not mistaken for the
        /// user's output.
        ///
        /// <para>
        /// Temp used to be the work directory, which is change-based capture's worst
        /// case: every intermediate a converter writes and deletes looks exactly like a
        /// file the user asked for. A sibling directory is writable, is still inside the
        /// sandbox, and is not scanned.
        /// </para>
        /// </summary>
        public string TempDirectory { get; }

        /// <summary>
        /// The packages already installed into <see cref="EnvDirectory"/> this session,
        /// so a repeated request skips pip entirely. Names only — a request that pins a
        /// different version is not filtered and pip resolves it.
        ///
        /// <para>
        /// Keyed by LANGUAGE, because one workspace holds a pip environment and an npm
        /// one and plenty of names exist in both registries. A flat ledger made an npm
        /// install of <c>markitdown</c> suppress the later pip install of the Python
        /// package with that name, and the model was told "Already installed this
        /// session: markitdown" while <c>import markitdown</c> kept failing — a false
        /// statement from the host, which is the one kind of tool result a model has no
        /// way to recover from. Observed on exactly this path: the pptx skill asks for
        /// <c>pptxgenjs</c> (npm) and <c>markitdown</c> (pip) in the same task.
        /// </para>
        /// </summary>
        /// <param name="language">The installer's language, e.g. "python" or "javascript".</param>
        public bool IsInstalled(string language, string package)
        {
            lock (_gate)
                return _installedPackages.Contains(InstallKey(language, package));
        }

        /// <summary>Record a successful install, against the language that performed it.</summary>
        public void MarkInstalled(string language, IEnumerable<string> packages)
        {
            lock (_gate)
            {
                foreach (string package in packages)
                    _installedPackages.Add(InstallKey(language, package));
            }
        }

        private static string InstallKey(string language, string package) =>
            (language ?? string.Empty) + "\u0000" + package;

        /// <summary>
        /// Read a file from the session's working directory, for the tools that address
        /// code BY PATH rather than as "the program I just ran".
        ///
        /// <para>
        /// Confined to <see cref="WorkDirectory"/>: the path comes from the model, so it
        /// is resolved and checked to be inside, and a traversal out of the workspace is
        /// refused rather than followed.
        /// </para>
        /// </summary>
        public bool TryReadFile(string relativePath, out string content, out string? error)
        {
            content = string.Empty;
            error = null;
            if (!TryResolve(relativePath, out string full, out error))
                return false;
            if (!File.Exists(full))
            {
                error = $"'{relativePath}' does not exist in this conversation's working directory."
                        + DescribeWhatIsHere();
                return false;
            }
            try
            {
                content = File.ReadAllText(full);
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"'{relativePath}' could not be read: {ex.Message}";
                return false;
            }
        }

        /// <summary>Write a file into the session's working directory, creating directories as needed.</summary>
        public bool TryWriteFile(string relativePath, string content, out string? error)
        {
            error = null;
            if (!TryResolve(relativePath, out string full, out error))
                return false;
            try
            {
                string? parent = Path.GetDirectoryName(full);
                if (!string.IsNullOrEmpty(parent))
                    Directory.CreateDirectory(parent);
                File.WriteAllText(full, content);
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"'{relativePath}' could not be written: {ex.Message}";
                return false;
            }
        }

        /// <summary>
        /// The files in the working directory, relative and sorted, excluding the
        /// runtime fallout no one asked for (bytecode caches and the like).
        /// </summary>
        public IReadOnlyList<(string Path, long Bytes)> ListFiles(int max = 200)
        {
            var found = new List<(string, long)>();
            try
            {
                foreach (string file in WorkspaceScan.Files(WorkDirectory, WorkspaceScan.SnapshotOptions))
                {
                    string relative = Path.GetRelativePath(WorkDirectory, file).Replace('\\', '/');
                    if (relative.StartsWith(".tensorsharp-", StringComparison.Ordinal)
                        || relative.Contains("__pycache__/", StringComparison.Ordinal)
                        || relative.EndsWith(".pyc", StringComparison.Ordinal)
                        || relative.StartsWith("Library/", StringComparison.Ordinal)
                        || relative.StartsWith("node_modules/", StringComparison.Ordinal)
                        || relative.StartsWith(".", StringComparison.Ordinal))
                    {
                        continue;
                    }
                    try { found.Add((relative, new FileInfo(file).Length)); }
                    catch (IOException) { /* vanished mid-scan */ }
                    if (found.Count >= max)
                        break;
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* partial is fine */ }
            found.Sort((a, b) => string.CompareOrdinal(a.Item1, b.Item1));
            return found;
        }

        /// <summary>
        /// Resolve a model-supplied relative path inside <see cref="WorkDirectory"/>.
        /// The containment check is the point: "../../.ssh/id_rsa" is a path the model
        /// can write, and only the resolved comparison catches it.
        ///
        /// <para>
        /// Public because the deterministic editor is what needs it most: every path in a
        /// patch comes from the model, and every one of them is resolved through here
        /// before a single byte is written.
        /// </para>
        /// </summary>
        public bool TryResolve(string relativePath, out string fullPath, out string? error) =>
            TryResolveFrom(WorkDirectory, relativePath, out fullPath, out error);

        /// <summary>
        /// The same resolution, but relative to <paramref name="baseDirectory"/> —
        /// wherever the shell currently is.
        ///
        /// <para>
        /// Needed because the two halves of the tool surface have to agree about what a
        /// relative path means. A model that <c>cd</c>s into <c>build/</c> and then
        /// patches <c>main.c</c> means <c>build/main.c</c>; resolving from the work
        /// directory instead would silently patch a different file, or refuse a file that
        /// is plainly right there. Containment is still checked against
        /// <see cref="WorkDirectory"/>, so a base of the model's choosing cannot widen it.
        /// </para>
        /// </summary>
        public bool TryResolveFrom(
            string baseDirectory, string relativePath, out string fullPath, out string? error)
        {
            fullPath = string.Empty;
            error = null;
            if (string.IsNullOrWhiteSpace(relativePath))
            {
                error = "a file path is required.";
                return false;
            }
            try
            {
                string root = Path.GetFullPath(WorkDirectory);
                string from = string.IsNullOrWhiteSpace(baseDirectory)
                    ? root
                    : Path.GetFullPath(baseDirectory);

                // An absolute path is followed when it lands INSIDE the working
                // directory and refused when it does not. A model reading a traceback
                // sees absolute paths and repeats them, and the containment check below
                // is what makes the path safe — not its spelling. Refusing an absolute
                // path that names a file the model may already open relatively taught it
                // nothing except to go looking, which is how a fixable typo became a
                // hunt through tools.
                string candidate = Path.IsPathRooted(relativePath)
                    ? Path.GetFullPath(relativePath)
                    : Path.GetFullPath(Path.Combine(from, relativePath));
                string rootWithSeparator = root.EndsWith(Path.DirectorySeparatorChar)
                    ? root
                    : root + Path.DirectorySeparatorChar;
                if (!candidate.StartsWith(rootWithSeparator, StringComparison.Ordinal))
                {
                    error = $"'{relativePath}' is outside this conversation's working directory."
                            + DescribeWhatIsHere();
                    return false;
                }

                // The lexical check above collapses "..", and that is all it does. It does
                // NOT follow symbolic links, and the caller is the HOST process, which is
                // not sandboxed — so `ln -s ~/.ssh/id_rsa notes.txt` (one ordinary shell
                // command, entirely permitted inside the workspace) would otherwise turn a
                // patch of "notes.txt" into a write through the link, and a read of it into
                // a read of the target. The component-by-component walk is the primitive
                // this codebase already uses for exactly this, and it is what makes the
                // sandbox's two central promises — home unreadable, writes confined — hold
                // for the file operations the host performs on the model's behalf.
                if (!SkillPathGuard.TryResolveSymlinks(root, candidate, out string? _, out string? linkError))
                {
                    error = $"'{relativePath}' {linkError}. Paths must stay inside this "
                            + "conversation's working directory.";
                    return false;
                }

                fullPath = candidate;
                return true;
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                error = $"'{relativePath}' is not a usable file name: {ex.Message}";
                return false;
            }
        }

        /// <summary>
        /// The tail of a path error: what the working directory actually holds, and the
        /// fact that the program just run needs no path at all.
        ///
        /// <para>
        /// A path error that only says the path is wrong leaves the model guessing, and
        /// it guesses the same wrong path again — observed exactly: an edit addressed at
        /// an invented <c>/tmp/Untitled.mjs</c> was refused, and the next round sent the
        /// identical path. Naming the real files, in the result, is what turns a refusal
        /// into a correction; small models act on tool results far more reliably than on
        /// parameter documentation they read thousands of tokens earlier.
        /// </para>
        /// </summary>
        public string DescribeWhatIsHere(int max = 20)
        {
            var sb = new StringBuilder();

            IReadOnlyList<(string Path, long Bytes)> files = ListFiles(max + 1);
            if (files.Count == 0)
            {
                sb.Append(" The working directory is empty.");
            }
            else
            {
                sb.Append(" It holds: ");
                for (int i = 0; i < files.Count && i < max; i++)
                {
                    if (i > 0)
                        sb.Append(", ");
                    sb.Append(files[i].Path);
                }
                if (files.Count > max)
                    sb.Append(", and more");
                sb.Append('.');
            }

            return sb.ToString();
        }

        private readonly List<IDisposable> _cleanups = new();

        /// <summary>
        /// Register something to shut down when this session ends.
        ///
        /// <para>
        /// A background job outlives the call that started it — that is the point of one —
        /// so something has to own the moment it stops, and the only honest owner is the
        /// session whose files it is writing into. Without this the workspace directory is
        /// deleted out from under a process that is still running in it.
        /// </para>
        /// </summary>
        public void RegisterCleanup(IDisposable cleanup)
        {
            ArgumentNullException.ThrowIfNull(cleanup);
            lock (_gate)
                _cleanups.Add(cleanup);
        }

        /// <summary>Run every registered cleanup. Called once, before the directory is deleted.</summary>
        internal void RunCleanups()
        {
            IDisposable[] pending;
            lock (_gate)
            {
                pending = _cleanups.ToArray();
                _cleanups.Clear();
            }
            foreach (IDisposable cleanup in pending)
            {
                // One misbehaving cleanup must not strand the rest, and must not leave the
                // workspace undeleted — the disk is the resource that actually accumulates.
                try { cleanup.Dispose(); }
                catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException)) { }
            }
        }

        private readonly HashSet<string> _appliedSetups = new(StringComparer.Ordinal);

        /// <summary>
        /// True exactly once per <paramref name="key"/> for this workspace's lifetime.
        /// Used for one-time setup steps — a skill's <c>requirements.txt</c> is applied
        /// on the session's first script run, not on every one.
        /// </summary>
        public bool TryMarkApplied(string key)
        {
            lock (_gate)
                return _appliedSetups.Add(key);
        }

        /// <summary>
        /// The state of every file under <see cref="WorkDirectory"/>, for telling what a
        /// run changed: capture-worthy output is what is NEW or MODIFIED relative to the
        /// snapshot taken before the run, never the accumulated history of the session.
        /// </summary>
        public Dictionary<string, (long Length, DateTime WriteTime)> SnapshotWorkFiles()
        {
            var snapshot = new Dictionary<string, (long, DateTime)>(StringComparer.Ordinal);
            try
            {
                foreach (string file in WorkspaceScan.Files(WorkDirectory, WorkspaceScan.SnapshotOptions))
                {
                    var info = new FileInfo(file);
                    string relative = Path.GetRelativePath(WorkDirectory, file).Replace('\\', '/');
                    snapshot[relative] = (info.Length, info.LastWriteTimeUtc);
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* partial is fine */ }
            return snapshot;
        }

        /// <summary>True when <paramref name="relative"/> is unchanged from <paramref name="snapshot"/>.</summary>
        public bool IsUnchangedSince(
            Dictionary<string, (long Length, DateTime WriteTime)> snapshot, string relative)
        {
            if (!snapshot.TryGetValue(relative.Replace('\\', '/'), out (long Length, DateTime WriteTime) before))
                return false;
            try
            {
                var now = new FileInfo(Path.Combine(WorkDirectory, relative));
                return now.Exists && now.Length == before.Length && now.LastWriteTimeUtc == before.WriteTime;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }
    }

    /// <summary>
    /// Owns the map from session id to <see cref="SessionWorkspace"/> and their disk
    /// lifecycle. One per host; sessions get workspaces lazily and lose them exactly
    /// when the session itself ends.
    /// </summary>
    public sealed class SessionWorkspaceManager
    {
        private readonly string _root;
        private readonly ILogger _logger;
        private readonly ConcurrentDictionary<string, SessionWorkspace> _workspaces = new(StringComparer.Ordinal);

        /// <param name="root">Parent directory the workspaces live under.</param>
        public SessionWorkspaceManager(string root, ILogger? logger = null)
        {
            _root = root ?? throw new ArgumentNullException(nameof(root));
            _logger = logger ?? NullLogger.Instance;
        }

        /// <summary>
        /// Delete every workspace a previous server run left behind. A restart orphans
        /// all sessions, so anything under the root is unreachable by construction.
        /// </summary>
        public void SweepOrphans()
        {
            try
            {
                if (!Directory.Exists(_root))
                    return;
                foreach (string dir in Directory.EnumerateDirectories(_root, SessionWorkspace.DirectoryPrefix + "*"))
                    TryDelete(dir);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                    "workspace.sweep-failed root={Root} reason={Reason}", _root, ex.Message);
            }
        }

        /// <summary>The session's workspace, created on first use.</summary>
        public SessionWorkspace GetOrCreate(string sessionId)
        {
            ArgumentException.ThrowIfNullOrEmpty(sessionId);
            return _workspaces.GetOrAdd(sessionId, id =>
            {
                var workspace = new SessionWorkspace(
                    Path.Combine(_root, SessionWorkspace.DirectoryPrefix + Sanitize(id)), id);
                _logger.LogInformation(LogEventIds.SkillScriptExecuted,
                    "workspace.created session={SessionId} root={Root}", id, workspace.Root);
                return workspace;
            });
        }

        /// <summary>Delete the session's workspace and everything in it, if one exists.</summary>
        public void Release(string sessionId)
        {
            if (string.IsNullOrEmpty(sessionId))
                return;
            if (_workspaces.TryRemove(sessionId, out SessionWorkspace? workspace))
            {
                _logger.LogInformation(LogEventIds.SkillScriptExecuted,
                    "workspace.released session={SessionId} root={Root}", sessionId, workspace.Root);
                // Stop anything still running in the directory BEFORE deleting it: a
                // background job holding a file open turns the delete into a partial one,
                // and the leftovers are what SweepOrphans has to clean up next boot.
                workspace.RunCleanups();
                TryDelete(workspace.Root);
            }
        }

        /// <summary>Workspace directory names come from session ids we generate, but a
        /// path separator in one must never choose where the delete lands.</summary>
        private static string Sanitize(string sessionId) =>
            new(sessionId.Where(char.IsAsciiLetterOrDigit).Take(64).ToArray());

        private void TryDelete(string directory)
        {
            try
            {
                if (Directory.Exists(directory))
                    Directory.Delete(directory, recursive: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                    "workspace.delete-failed dir={Dir} reason={Reason}", directory, ex.Message);
            }
        }
    }
}
