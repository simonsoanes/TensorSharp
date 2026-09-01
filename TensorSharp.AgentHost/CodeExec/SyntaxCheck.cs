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
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// After a file is written or patched, check that it still parses — and say so when
    /// it does not.
    ///
    /// <para>
    /// <b>The defect class this closes is the one this codebase keeps finding: reporting
    /// SUCCESS for something that was not done.</b> <c>apply_patch</c> is
    /// all-or-nothing about MATCHING, and that is not the same as being right. Every
    /// anchor can resolve, every byte can land where it was asked to, and the result can
    /// truthfully say "updated deck.py (+3 -1)" while the file it just produced no longer
    /// compiles — because a hunk that deletes a line can delete the one that closed a
    /// block, and a hunk that fuzz-matched on whitespace can land correct text at the
    /// wrong indentation. The model reads "applied", believes the file, and its own tool
    /// declaration tells it — correctly, for matching — <i>"do not re-read a file to check
    /// after a patch succeeded"</i>. So it does not look, and the defect surfaces a round
    /// or two later as something that looks unrelated, or never surfaces at all and the
    /// user is handed broken code.
    /// </para>
    /// <para>
    /// On the shell side the gate is deliberately the opposite of the obvious one: the
    /// check runs when a command SUCCEEDED and left a source file behind. A command that
    /// failed has already put its own <c>SyntaxError</c> in the output, and repeating it
    /// would be noise; a command that exited 0 having written a file it never ran is the
    /// case where nothing in the result contradicts "done", which is the only kind of
    /// wrong result a model cannot recover from.
    /// </para>
    /// <para>
    /// This is what Claude Code gets from an editor's live diagnostics after every edit,
    /// done the way a host with no editor attached can do it: with the compiler that is
    /// already installed. It is a PARSE, not a lint and not a type-check — nothing here
    /// has an opinion about style, and nothing here fails a file for anything a language
    /// would actually run.
    /// </para>
    /// <para>
    /// <b>Cost.</b> One process per language, not per file: every Python file in the
    /// batch is compiled by one interpreter launch (~25 ms measured), every JavaScript
    /// file by one Node launch. JSON needs no process at all. Nothing runs when nothing
    /// checkable was touched, which is the common case.
    /// </para>
    /// </summary>
    public sealed class SyntaxCheck
    {
        private readonly CodeExecOptions _options;
        private readonly ISkillSandbox? _sandbox;

        /// <summary>Most files checked per language. A patch that touches more has bigger problems.</summary>
        private const int MaxFiles = 24;

        private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(20);

        /// <summary>Create a checker running under the same terms as the run it follows.</summary>
        public SyntaxCheck(CodeExecOptions options, ISkillSandbox? sandbox)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _sandbox = sandbox;
        }

        /// <summary>
        /// Check every checkable file among <paramref name="relativePaths"/> and describe
        /// what no longer parses, or return null when everything does.
        /// </summary>
        /// <param name="relativePaths">Paths relative to the work directory, as the model knows them.</param>
        /// <param name="workspace">The session workspace the paths belong to.</param>
        /// <param name="ranIn">The directory the paths are relative to; defaults to the work directory.</param>
        public string? Verify(
            IReadOnlyList<string> relativePaths, SessionWorkspace workspace, string? ranIn = null)
        {
            ArgumentNullException.ThrowIfNull(workspace);
            if (relativePaths == null || relativePaths.Count == 0)
                return null;

            string from = ranIn is { Length: > 0 } ? ranIn : workspace.WorkDirectory;
            var python = new List<string>();
            var node = new List<string>();
            var json = new List<string>();

            foreach (string relative in relativePaths.Distinct(StringComparer.Ordinal))
            {
                // Model-supplied paths, resolved by the workspace rather than combined
                // lexically: this read happens in the HOST, which is not sandboxed, and a
                // symlink planted inside the workspace is the escape this codebase has
                // already been bitten by once.
                if (!workspace.TryResolveFrom(from, relative, out string full, out _) || !File.Exists(full))
                    continue;

                List<string>? bucket = Path.GetExtension(relative).ToLowerInvariant() switch
                {
                    ".py" or ".pyw" => python,
                    ".js" or ".mjs" or ".cjs" => node,
                    ".json" => json,
                    _ => null,
                };
                if (bucket != null && bucket.Count < MaxFiles)
                    bucket.Add(full);
            }

            var problems = new List<string>();
            problems.AddRange(CheckJson(json, workspace, from));
            problems.AddRange(Check(CodeLanguage.Python, PythonChecker, python, workspace, from));
            problems.AddRange(Check(CodeLanguage.JavaScript, NodeChecker, node, workspace, from));

            if (problems.Count == 0)
                return null;

            var sb = new StringBuilder();
            // Counted by FILE, not by line: a single broken file reports two lines — the
            // compiler's message and the source it pointed at — and counting those made
            // one bad file announce itself as several.
            int broken = problems.Count(line => line.Length > 0 && !char.IsWhiteSpace(line[0]));
            // Phrased so it cannot be misread as "the write did not happen". It did; the
            // file on disk is the one described below, and that is the whole point.
            sb.Append(broken <= 1
                ? "The file was written, but it does not parse:\n"
                : "The files were written, but they do not parse:\n");
            foreach (string problem in problems)
                sb.Append("  ").Append(problem).Append('\n');
            sb.Append("Read the lines named above and fix them before running anything else — the code on "
                    + "disk right now cannot start.\n");
            return sb.ToString();
        }

        /// <summary>
        /// The files a command REDIRECTED into, read straight out of the command line.
        ///
        /// <para>
        /// This is how the shell side learns what to check without paying for a second
        /// walk of the work directory. The alternative — diffing a before-and-after
        /// snapshot — is the exact cost that was measured down from 392 ms to 27 ms by
        /// removing a redundant walk, and putting one back to serve a diagnostic would be
        /// taxing every command to help the ones that write code. A redirect target is
        /// already written down in the command, exactly and unambiguously, and a heredoc
        /// into a redirect is how every file gets written here.
        /// </para>
        /// <para>
        /// Only the shapes that mean "a file now contains this text". <c>2&gt;&amp;1</c>
        /// and <c>&gt;&amp;2</c> are descriptor plumbing, <c>/dev/null</c> is a sink, and
        /// a target built out of a variable cannot be resolved without running the
        /// command — all three are skipped rather than guessed at.
        /// </para>
        /// </summary>
        public static IReadOnlyList<RedirectTarget> RedirectTargets(string? command)
        {
            if (string.IsNullOrEmpty(command))
                return Array.Empty<RedirectTarget>();

            // Scanned per SEGMENT, never over the raw command, because a heredoc BODY is
            // not shell — it is data. `cat > README.md <<'EOF' / Run it with: python
            // gen.py > out.py / EOF` writes README.md and nothing else, and scanning the
            // whole string found `out.py` in the prose and reported on a file the command
            // never opened. SplitSegments already skips heredoc bodies whole, which is
            // exactly the distinction needed here.
            // One segment at a time, never a rejoined string: the operator and its target
            // have to be adjacent, and joining segments let the pattern's `\s*` cross the
            // join — `echo hi >&2` splits on `&` into "echo hi >" and "2", which read
            // together as a redirection into a file called "2".
            var targets = new List<RedirectTarget>();
            foreach (ShellSegment segment in ShellCommand.SplitSegments(command))
            foreach (System.Text.RegularExpressions.Match match in Redirect.Matches(segment.Text))
            {
                // Ten capture groups: the operator, then three per alternative for the
                // three ways a path can be quoted. Reading only the first three found
                // `> a.py` and silently missed every `tee` and every PowerShell spelling
                // — a check that quietly covers one third of the ways a file gets written
                // is worse than none, because its silence reads as "this file is fine".
                // Named groups, not indices. Three alternatives each need an operator
                // slot and three quoting slots, and a positional scan across ten groups
                // reads the SECOND alternative's optional "-a " as the first's path.
                // Only one alternative can match, so a repeated name is unambiguous.
                string path = match.Groups["path"].Value.Trim();
                if (path.Length == 0
                    || path.StartsWith('$') || path.Contains('$', StringComparison.Ordinal)
                    || path.StartsWith("/dev/", StringComparison.Ordinal)
                    || path.Equals("NUL", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                string @operator = match.Groups["op"].Value;
                bool appends = @operator.Contains(">>", StringComparison.Ordinal)
                    || @operator.Contains("-a", StringComparison.OrdinalIgnoreCase);

                if (!targets.Any(t => string.Equals(t.Path, path, StringComparison.Ordinal)))
                    targets.Add(new RedirectTarget(path, appends));
            }
            return targets;
        }

        /// <summary>A file a command writes into, and whether it ADDS to it or replaces it.</summary>
        /// <param name="Path">The path as the command wrote it, relative to where it ran.</param>
        /// <param name="Appends">
        /// True for <c>&gt;&gt;</c>, <c>tee -a</c> and <c>-Append</c>.
        ///
        /// <para>
        /// The distinction is not decorative, and leaving it out produced a host that
        /// stated a falsehood: <see cref="RewriteWatch"/> saw an append of five lines to a
        /// 190-line file as "this command replaced all 195 lines of it", and told the
        /// model its correct action had been wasteful. A syntax check does not care how
        /// the bytes got there; anything reasoning about what the command DID does.
        /// </para>
        /// </param>
        public readonly record struct RedirectTarget(string Path, bool Appends);

        // `> a.py`, `>> a.py`, `| tee a.py`, and PowerShell's two spellings. The leading
        // (?<![0-9&>]) keeps `2>` and `>&` out: those are descriptors, not files.
        private static readonly System.Text.RegularExpressions.Regex Redirect = new(
            @"(?<![0-9&>])(?<op>>>?)\s*(?:'(?<path>[^']+)'|""(?<path>[^""]+)""|(?<path>[A-Za-z0-9_./\\-]+))"
            + @"|\btee\s+(?<op>-a\s+)?(?:'(?<path>[^']+)'|""(?<path>[^""]+)""|(?<path>[A-Za-z0-9_./\\-]+))"
            + @"|\b(?:Set-Content|Out-File)\s+(?<op>-Append\s+)?(?:-(?:LiteralPath|Path|FilePath)\s+)?"
            + @"(?:'(?<path>[^']+)'|""(?<path>[^""]+)""|(?<path>[A-Za-z0-9_./\\-]+))",
            System.Text.RegularExpressions.RegexOptions.Compiled
            | System.Text.RegularExpressions.RegexOptions.CultureInvariant
            | System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        /// <summary>
        /// JSON, in-process. A parser is already linked in, so spending a subprocess to
        /// find a missing comma would be the wrong trade.
        /// </summary>
        private static IEnumerable<string> CheckJson(
            IReadOnlyList<string> files, SessionWorkspace workspace, string from)
        {
            foreach (string file in files)
            {
                string? problem = null;
                try
                {
                    using var stream = File.OpenRead(file);
                    using JsonDocument _ = JsonDocument.Parse(
                        stream, new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip });
                }
                catch (JsonException ex)
                {
                    problem = $"{Label(workspace, from, file)} line {ex.LineNumber + 1}: {ex.Message}";
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    // Unreadable is not unparseable, and guessing which would be a false
                    // statement about the model's own file.
                }
                if (problem != null)
                    yield return problem;
            }
        }

        private IReadOnlyList<string> Check(
            CodeLanguage language, string checker, IReadOnlyList<string> files,
            SessionWorkspace workspace, string from)
        {
            if (files.Count == 0
                || !CodeEnvironment.TryResolveInterpreter(language, out string? interpreter, out _)
                || interpreter == null)
            {
                return Array.Empty<string>();
            }

            var arguments = new List<string>
            {
                language == CodeLanguage.Python ? "-c" : "-e",
                checker,
            };
            arguments.AddRange(files);

            ConfinedResult result = ConfinedProcess.Run(
                new ConfinedLaunch
                {
                    Interpreter = interpreter,
                    Arguments = arguments,
                    WriteDirectory = workspace.TempDirectory,
                    WorkingDirectory = from,
                    ReadOnlyDirectory = workspace.Root,
                    ReadablePaths = new[] { workspace.EnvDirectory },
                    AllowNetwork = false,
                    Timeout = Deadline,
                    MaxOutputBytes = 16 * 1024,
                    EnvironmentVariables = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["HOME"] = workspace.WorkDirectory,
                        ["TMPDIR"] = workspace.TempDirectory,
                        // A parse must not leave .pyc files behind. They would be captured
                        // as produced files and handed to the user as downloads.
                        ["PYTHONDONTWRITEBYTECODE"] = "1",
                        ["PYTHONUNBUFFERED"] = "1",
                        ["NO_COLOR"] = "1",
                    },
                },
                _sandbox,
                _options.Unconfined ? SkillSandboxMode.Preferred : _options.Sandbox);

            // A checker that could not run says nothing. It is a diagnostic, and a
            // diagnostic that invents a problem is worse than one that stays quiet.
            if (!result.Started || result.TimedOut || result.ExitCode == 0)
                return Array.Empty<string>();

            // Indentation is preserved on purpose. A continuation line — the source the
            // compiler pointed at — is indented by the checker, and that is what
            // distinguishes it from the next FILE's report both visually and when the
            // header counts how many files are broken.
            return OutputPaths.Scrub(result.Stdout, workspace, from)
                .Replace("\r", string.Empty, StringComparison.Ordinal)
                .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Take(MaxFiles * 2)
                .ToList();
        }

        private static string Label(SessionWorkspace workspace, string from, string full)
        {
            string relative = Path.GetRelativePath(from, full).Replace('\\', '/');
            return relative.Length == 0 ? Path.GetFileName(full) : relative;
        }

        /// <summary>
        /// Compile every named file and report each failure, rather than dying on the
        /// first one: a patch across three files that broke two of them should cost one
        /// round to learn about, not two.
        /// </summary>
        private const string PythonChecker =
            "import sys\n"
            + "bad = 0\n"
            + "for path in sys.argv[1:]:\n"
            + "    try:\n"
            + "        with open(path, 'rb') as handle:\n"
            + "            compile(handle.read(), path, 'exec')\n"
            + "    except SyntaxError as error:\n"
            + "        bad += 1\n"
            + "        print('%s line %s: %s' % (path, error.lineno, error.msg))\n"
            + "        if error.text:\n"
            + "            print('    %s' % error.text.rstrip())\n"
            + "    except Exception:\n"
            + "        pass\n"
            + "sys.exit(1 if bad else 0)\n";

        /// <summary>
        /// <c>vm.Script</c> rather than <c>node --check</c>, which takes one file per
        /// invocation. Compiling does not run the code: a module's top level is not
        /// executed, so checking a script cannot have side effects.
        /// </summary>
        private const string NodeChecker =
            "const fs = require('fs'), vm = require('vm');\n"
            + "let bad = 0;\n"
            + "for (const path of process.argv.slice(1)) {\n"
            + "  try { new vm.Script(fs.readFileSync(path, 'utf8'), { filename: path }); }\n"
            + "  catch (error) {\n"
            + "    if (error instanceof SyntaxError) { bad++; console.log(path + ': ' + error.message); }\n"
            + "  }\n"
            + "}\n"
            + "process.exit(bad ? 1 : 0);\n";
    }
}
