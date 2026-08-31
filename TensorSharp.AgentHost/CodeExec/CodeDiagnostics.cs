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
using System.Linq;
using System.Text.RegularExpressions;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// Reading a failure's output well enough to say what to do about it.
    ///
    /// <para>
    /// This exists because of a measured incident, not a hunch. Asked to convert an
    /// attached markdown file to PDF, gemma-4-E4B ran <c>import reportlab</c>, was handed
    /// the bare traceback, ran the IDENTICAL code again, and then told the user the PDF
    /// existed. A traceback teaches a small model nothing on its own; a result that names
    /// the next command — <c>pip install reportlab</c>, in this host's spelling — changes
    /// the behaviour. Every failure path here therefore ends in an action, not a
    /// description.
    /// </para>
    /// <para>
    /// Shared by the shell tool and by skill scripts on purpose: they fail in the same
    /// ways and the advice is the same, and two copies of a regex like these is two
    /// copies that will stop agreeing.
    /// </para>
    /// </summary>
    public static class CodeDiagnostics
    {
        /// <summary>The module a failed run could not import, or null.</summary>
        public static string? MissingModule(CodeLanguage language, string? stderr)
        {
            if (string.IsNullOrEmpty(stderr))
                return null;

            Match match = language switch
            {
                CodeLanguage.Python => PythonMissingModule.Match(stderr!),
                CodeLanguage.JavaScript => NodeMissingModule.Match(stderr!),
                _ => Match.Empty,
            };
            if (!match.Success)
                return null;

            // "No module named 'yaml.parser'" names the submodule; the thing to install
            // is the top-level package.
            string name = match.Groups[1].Value;
            int dot = name.IndexOf('.');
            return dot > 0 ? name.Substring(0, dot) : name;
        }

        /// <summary>
        /// The module a failed run could not import, in whichever language's shape the
        /// output matches — for a shell, where the host did not choose the interpreter.
        /// </summary>
        public static bool TryFindMissingModule(
            string? output, out CodeLanguage language, out string module)
        {
            language = CodeLanguage.Unknown;
            module = string.Empty;

            if (MissingModule(CodeLanguage.Python, output) is { } python)
            {
                language = CodeLanguage.Python;
                module = python;
                return true;
            }
            if (MissingModule(CodeLanguage.JavaScript, output) is { } node)
            {
                language = CodeLanguage.JavaScript;
                module = node;
                return true;
            }
            return false;
        }

        /// <summary>
        /// The command a run could not find, or null.
        ///
        /// <para>
        /// Read from stderr rather than from the exit code: 127 is the shell's own
        /// convention and a program is free to return it for something else entirely,
        /// while the message names what is actually missing — which is the part the
        /// model needs. Undiagnosed, a model retries the same line forever, because it
        /// cannot tell "this host has no pdftoppm" from "I typed it wrong", and neither
        /// pip nor npm can supply that one.
        /// </para>
        /// </summary>
        public static string? MissingCommand(string? stderr)
        {
            if (string.IsNullOrEmpty(stderr))
                return null;

            // zsh trails the name ("zsh: command not found: ffmpeg"); sh and bash lead
            // with it ("bash: line 1: pandoc: command not found"). Checked in this order
            // because in the zsh form the leading shell name also matches the other
            // shape, and leftmost-wins would report the shell as the missing command.
            Match trailing = ShellMissingCommandTrailing.Match(stderr!);
            if (trailing.Success)
                return trailing.Groups[1].Value;

            Match leading = ShellMissingCommandLeading.Match(stderr!);
            if (leading.Success)
                return leading.Groups[1].Value;

            // PowerShell's spelling, which names the command and then explains at length.
            Match powershell = PowerShellMissingCommand.Match(stderr!);
            return powershell.Success ? powershell.Groups[1].Value : null;
        }

        private static readonly Regex ShellMissingCommandTrailing = new(
            @"command not found: ([^\s]+)",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex ShellMissingCommandLeading = new(
            @"([^\s:]+): (?:command not found|not found)\b",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex PowerShellMissingCommand = new(
            @"The term '([^']+)' is not recognized as (?:the name of )?a cmdlet",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex PythonMissingModule = new(
            @"(?:ModuleNotFoundError|ImportError): No module named '?([A-Za-z0-9_.]+)'?",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex NodeMissingModule = new(
            @"Cannot find (?:module|package) '([^']+)'",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        /// <summary>
        /// Import names that differ from the name the installer knows. Only the ones a
        /// model actually runs into; anything absent installs under its own name.
        /// </summary>
        private static readonly Dictionary<string, string> PipNameForImport = new(StringComparer.Ordinal)
        {
            ["PIL"] = "pillow",
            ["cv2"] = "opencv-python-headless",
            ["sklearn"] = "scikit-learn",
            ["yaml"] = "PyYAML",
            ["bs4"] = "beautifulsoup4",
            ["docx"] = "python-docx",
            ["pptx"] = "python-pptx",
            ["fitz"] = "PyMuPDF",
            ["Crypto"] = "pycryptodome",
            ["dateutil"] = "python-dateutil",
        };

        /// <summary>The name the installer knows <paramref name="module"/> by.</summary>
        public static string InstallNameFor(CodeLanguage language, string module) =>
            language == CodeLanguage.Python && PipNameForImport.TryGetValue(module, out string? pip)
                ? pip
                : module;

        /// <summary>The command that installs <paramref name="module"/> from this host's shell.</summary>
        public static string InstallCommandFor(CodeLanguage language, string module)
        {
            string package = InstallNameFor(language, module);
            return language == CodeLanguage.JavaScript
                ? $"npm install {package}"
                : $"{PythonInstallPrefix()} {package}";
        }

        /// <summary>
        /// How to spell a pip install ON THIS HOST.
        ///
        /// <para>
        /// Plenty of machines have <c>python3</c> and no <c>pip</c> at all — Apple's
        /// system Python is one, and a Homebrew install without the shim is another. A
        /// model told "run pip install X" there gets "pip: command not found" and, having
        /// followed the instruction it was given, has nowhere to go. Probing once and
        /// naming the spelling that exists costs a directory lookup.
        /// </para>
        /// </summary>
        public static string PythonInstallPrefix()
        {
            if (_pythonInstallPrefix != null)
                return _pythonInstallPrefix;

            if (CodeEnvironment.Which("pip") != null)
                return _pythonInstallPrefix = "pip install";
            if (CodeEnvironment.Which("pip3") != null)
                return _pythonInstallPrefix = "pip3 install";
            if (CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _) && python != null)
                return _pythonInstallPrefix = System.IO.Path.GetFileName(python) + " -m pip install";
            return _pythonInstallPrefix = "pip install";
        }

        private static string? _pythonInstallPrefix;

        // ---- environment or code? -------------------------------------------

        /// <summary>Where the blame for a failed run lies.</summary>
        public enum FailureSource
        {
            /// <summary>Nothing here says. Make no claim.</summary>
            Unknown,

            /// <summary>
            /// The HOST or the machine, not the program. The program either never ran or
            /// was stopped by something outside it, so re-typing it cannot help.
            /// </summary>
            Environment,

            /// <summary>The program itself. Editing it is the fix.</summary>
            Code,
        }

        /// <summary>What a failure was caused by, and the sentence that says so.</summary>
        /// <param name="Source">Where the blame lies.</param>
        /// <param name="Reason">
        /// The cause in the model's terms, or empty. Never a restatement of the output —
        /// the output is already right there.
        /// </param>
        /// <param name="HostCanFix">
        /// True when the HOST is able to correct this itself. Those cases do not need a
        /// message at all; they need the host to act, which it does elsewhere.
        /// </param>
        public readonly record struct FailureCause(FailureSource Source, string Reason, bool HostCanFix);

        /// <summary>
        /// Decide whether a failed run failed because of the ENVIRONMENT or because of the
        /// code, and say which.
        ///
        /// <para>
        /// <b>Why this is worth a classifier.</b> The reference implementation's standing
        /// instruction to its own model is <i>"fix the problem at the root cause rather
        /// than applying surface-level patches"</i> — and in this server's logs the model
        /// could not tell what the root cause WAS. Handed a failure that came from the
        /// host, it rewrote the program: one turn responded to a missing dependency and a
        /// sandbox denial by re-emitting 15,000 characters and then switching language
        /// entirely. Re-typing a program costs about 24 times what re-reading it from the
        /// prompt costs, so a wrong diagnosis here is the most expensive wrong turn the
        /// loop can take.
        /// </para>
        /// <para>
        /// <b>What it is careful never to say.</b> It does not say "your code is correct".
        /// It cannot know that — a program can have a bug AND hit a missing library on the
        /// same run — and a host that says so wrongly has told the model to re-run
        /// something broken, which is worse than saying nothing. What it says is the thing
        /// that IS knowable from the output: this failure did not come from the code, so
        /// changing the code will not change it. That is true whether or not the code also
        /// has a defect, and it points at the only next step that can make progress.
        /// </para>
        /// <para>
        /// Everything not recognised is <see cref="FailureSource.Unknown"/> and produces
        /// no claim at all. The classifier's value is entirely in being right, so the
        /// default is silence.
        /// </para>
        /// </summary>
        /// <param name="networkConfined">
        /// Whether this host's sandbox actually blocks the network.
        ///
        /// <para>
        /// False on Windows, where <c>WindowsJobObjectSandbox.Capabilities</c> reports
        /// <c>ConfinesNetwork: false</c> — so there the network genuinely works, a DNS
        /// failure is an ordinary transient failure, and answering it with "nothing here
        /// can reach the network, not this one, not any of them" is a false statement of a
        /// constraint in the one place a model most needs the truth.
        /// </para>
        /// </param>
        public static FailureCause ClassifyFailure(
            string? output,
            CodeLanguage language = CodeLanguage.Unknown,
            bool networkConfined = true)
        {
            if (string.IsNullOrEmpty(output))
                return new FailureCause(FailureSource.Unknown, string.Empty, false);

            string text = output!;

            // ---- the environment, and the host can fix it ----

            if (TryFindMissingModule(text, out CodeLanguage moduleLanguage, out string module))
            {
                return new FailureCause(
                    FailureSource.Environment,
                    $"the module '{module}' is not installed in this session",
                    HostCanFix: true);
            }

            if (MissingCommand(text) is { } absent)
            {
                bool spelled = SpelledDifferentlyHere(absent) != null;
                return new FailureCause(
                    FailureSource.Environment,
                    spelled
                        ? $"'{absent}' is spelled differently on this host"
                        : $"'{absent}' is not installed on this host",
                    HostCanFix: spelled);
            }

            // ---- the environment, and the host cannot ----

            if (networkConfined && LooksLikeNetworkAttempt(text))
            {
                return new FailureCause(
                    FailureSource.Environment,
                    "nothing here can reach the network, so this could never have worked",
                    HostCanFix: false);
            }

            if (LooksLikeSandboxDenial(text))
            {
                return new FailureCause(
                    FailureSource.Environment,
                    "the sandbox refused this, so the program was stopped rather than being wrong",
                    HostCanFix: false);
            }

            if (LooksLikeOldInterpreter(text, language))
            {
                return new FailureCause(
                    FailureSource.Environment,
                    "this host's interpreter is too old for the syntax used",
                    HostCanFix: false);
            }

            // A missing native library. The wheel installed and is unusable: no package
            // manager here can supply a system library, so this is as far as it goes, and
            // a model that reads only the ctypes noise will keep reinstalling the package.
            if (NativeLibraryMissing.IsMatch(text))
            {
                return new FailureCause(
                    FailureSource.Environment,
                    "the package installed but needs a system library this host does not have, "
                    + "which no package manager here can supply",
                    HostCanFix: false);
            }

            // A package that failed to IMPORT, from inside its own files.
            //
            // Narrowed to import failures on purpose, and the narrowing is the whole
            // safety of it. "The deepest frame is in a library" is a fine reason to stop
            // routing something to the API probe — the model did not guess a NAME — but it
            // is NOT a reason to tell the model its code is not at fault: a TypeError
            // raised deep inside a library is very often caused by the argument the model
            // passed in, and saying "not the code you wrote" there is the one claim this
            // classifier must never make. An IMPORT that fails inside a package cannot
            // have been caused by the caller's arguments, because the caller has not run
            // yet.
            if (ImportFailure.IsMatch(text) && FailedInsideALibrary(text))
            {
                return new FailureCause(
                    FailureSource.Environment,
                    "the package is installed but cannot be loaded on this host — it failed inside its "
                    + "own files, before your program ran",
                    HostCanFix: false);
            }

            // ---- the code ----
            //
            // Ordered after every environmental shape on purpose. A SyntaxError raised by
            // an interpreter too old for a `match` statement is an ENVIRONMENT failure
            // wearing a code failure's clothes, and it is checked above for exactly that
            // reason.

            if (CodeFailure.IsMatch(text))
            {
                return new FailureCause(
                    FailureSource.Code, "the program itself raised this", HostCanFix: false);
            }

            return new FailureCause(FailureSource.Unknown, string.Empty, false);
        }

        /// <summary>
        /// A run stopped by the sandbox rather than by its own logic.
        ///
        /// <para>
        /// Deliberately narrow. A <c>PermissionError</c> on a path the program itself
        /// chose inside the work directory is a program bug, and only the sandbox's own
        /// vocabulary — Seatbelt's <c>deny(1)</c> lines, bubblewrap's refusals, the
        /// "Operation not permitted" a confined process gets — is treated as the sandbox
        /// speaking.
        /// </para>
        /// </summary>
        private static bool LooksLikeSandboxDenial(string text) =>
            // Only what the CHILD itself printed.
            //
            // `deny(1)` and the "sandbox denials observed" header come from
            // SandboxViolationMonitor, which is a SYSTEM-WIDE `log stream` filtered by
            // substring against a list that includes "sh" — a substring of almost any
            // path. Its block is appended to stderr for the model to read, and it used to
            // be part of the text this classifier reasoned over, which meant another
            // process's log lines could produce the strongest environmental verdict here.
            // In the corpus that block is attached to a `pandoc: command not found` it had
            // nothing to do with.
            //
            // The lines are still shown — "the sandbox denied ~/.config" is worth reading
            // — they are just no longer evidence. See ConfinedResult.Denials.
            text.Contains("System Policy:", StringComparison.Ordinal)
            || text.Contains("bwrap:", StringComparison.Ordinal)
            || (text.Contains("Operation not permitted", StringComparison.Ordinal)
                && !text.Contains("No such file", StringComparison.Ordinal));

        /// <summary>
        /// An interpreter older than the syntax it was handed. Apple's frozen
        /// <c>/usr/bin/python3</c> is 3.9, and a <c>match</c> statement or a PEP 604 union
        /// dies there as a bare "SyntaxError: invalid syntax" — which reads as a broken
        /// program when the program is fine and the host is old.
        /// </summary>
        private static bool LooksLikeOldInterpreter(string text, CodeLanguage language)
        {
            if (language == CodeLanguage.JavaScript)
                return false;
            if (!text.Contains("SyntaxError", StringComparison.Ordinal)
                && !text.Contains("unsupported operand type(s) for |: 'type'", StringComparison.Ordinal))
            {
                return false;
            }
            return CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
                && python != null
                && CodeEnvironment.PythonVersionOf(python) is { } version
                && version < new Version(3, 10);
        }

        /// <summary>
        /// A wheel that installed and cannot load, because it needs a system library.
        /// Reproduced with weasyprint, which needs libgobject: the wheel is present, the
        /// import fails, and no installer available here can fix it.
        /// </summary>
        private static readonly Regex NativeLibraryMissing = new(
            @"cannot load library|ctypes\.util\.find_library|could not import some external libraries"
            + @"|ImportError: lib[\w.+-]*\.so|image not found|incompatible architecture",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        /// <summary>
        /// A failure that happened while LOADING a module, before the model's own code ran.
        /// </summary>
        private static readonly Regex ImportFailure = new(
            @"ImportError|ModuleNotFoundError|ERR_MODULE_NOT_FOUND|Cannot find (?:module|package)",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        /// <summary>An exception the program raised about its own logic.</summary>
        private static readonly Regex CodeFailure = new(
            @"SyntaxError|IndentationError|TabError|NameError|UnboundLocalError|TypeError|ValueError"
            + @"|KeyError|IndexError|AttributeError|ZeroDivisionError|AssertionError|RecursionError"
            + @"|ReferenceError|Unexpected token|Unexpected identifier|is not a function|is not a constructor"
            + @"|Cannot read propert",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        /// <summary>Renamed from the shell runner's private copy so both callers share one answer.</summary>
        public static bool LooksLikeNetworkAttempt(string? text) =>
            text != null
            && (text.Contains("Temporary failure in name resolution", StringComparison.Ordinal)
                || text.Contains("Could not resolve host", StringComparison.Ordinal)
                || text.Contains("Network is unreachable", StringComparison.Ordinal)
                || text.Contains("nodename nor servname provided", StringComparison.Ordinal)
                || text.Contains("getaddrinfo", StringComparison.Ordinal)
                // musl and non-Darwin glibc spell it this way; the Darwin and Windows
                // spellings are already covered above and by `getaddrinfo`.
                || text.Contains("Name or service not known", StringComparison.Ordinal)
                || (text.Contains("Operation not permitted", StringComparison.Ordinal)
                    && text.Contains("socket", StringComparison.OrdinalIgnoreCase)));

        // ---- guessed APIs ---------------------------------------------------

        /// <summary>Which kind of "you guessed the API" failure an output shows.</summary>
        public enum ApiMissKind
        {
            /// <summary>An attribute an object or class does not have.</summary>
            PythonAttribute,

            /// <summary>An attribute a MODULE does not have.</summary>
            PythonModuleAttribute,

            /// <summary>A name a module does not export, so <c>from x import y</c> failed.</summary>
            PythonImportName,

            /// <summary>A Node value used as a constructor or a function when it is neither.</summary>
            NodeExport,
        }

        /// <summary>What the run got wrong about a library's API.</summary>
        /// <param name="Kind">Which shape of mistake it is.</param>
        /// <param name="Subject">The type, module or expression the member was looked for on.</param>
        /// <param name="Member">The member that does not exist.</param>
        public readonly record struct ApiMiss(ApiMissKind Kind, string Subject, string Member);

        /// <summary>
        /// Read a guessed-API failure out of a run's output.
        ///
        /// <para>
        /// These four shapes are not a guess at what models get wrong; they are what this
        /// server's own logs contain. Across the recoverable tool results, API-shape
        /// errors outnumber every other semantic failure — 33 bare
        /// <c>TypeError</c>s, 13 "is not a function", 9 "is not a constructor", plus the
        /// import-name and attribute forms — and unlike a missing module (already
        /// answered by <see cref="TryFindMissingModule"/>) a traceback gives the model
        /// nothing at all to act on. See <c>ApiProbe</c> for what is done about it.
        /// </para>
        /// </summary>
        public static bool TryFindApiMiss(string? output, out ApiMiss miss)
        {
            miss = default;
            if (string.IsNullOrEmpty(output))
                return false;

            // A failure raised INSIDE a library is not the model guessing an API.
            //
            // This distinction was missing and it produced a confidently wrong diagnosis.
            // The host installs wheels with the newest Python it can find; if the command
            // then runs under an older one, `from PIL import Image` dies as
            // `ImportError: cannot import name '_imaging' from 'PIL' (../env/PIL/__init__.py)`
            // — an ABI mismatch, in Pillow's own file, about Pillow's own C extension. The
            // import-name pattern below matches it, so the probe fired, ran under the
            // NEWER interpreter where the module is perfectly importable, and reported
            // "you guessed the API" about code that was correct.
            //
            // The discriminator is where the deepest traceback frame lives. Model code is
            // in the work directory (rendered as a bare or ./ path, or as the word
            // "command"); a library is under the session's env. If the last frame is a
            // library's own file, the package is broken or mismatched and the model's
            // spelling was right.
            if (FailedInsideALibrary(output!))
                return false;

            // Ordered most specific first. "module 'x' has no attribute 'y'" and
            // "type object 'X' has no attribute 'y'" are both attribute errors and
            // neither is matched by the plain object form, so each needs its own shape —
            // and the module form needs a different probe, because a module is imported
            // by name while a type has to be hunted for.
            Match match = PythonModuleAttribute.Match(output!);
            if (match.Success)
            {
                miss = new ApiMiss(ApiMissKind.PythonModuleAttribute, match.Groups[1].Value, match.Groups[2].Value);
                return true;
            }

            match = PythonImportName.Match(output!);
            if (match.Success)
            {
                miss = new ApiMiss(ApiMissKind.PythonImportName, match.Groups[2].Value, match.Groups[1].Value);
                return true;
            }

            match = PythonTypeAttribute.Match(output!);
            if (!match.Success)
                match = PythonObjectAttribute.Match(output!);
            if (match.Success)
            {
                miss = new ApiMiss(ApiMissKind.PythonAttribute, match.Groups[1].Value, match.Groups[2].Value);
                return true;
            }

            match = NodeNotConstructible.Match(output!);
            if (!match.Success)
                match = NodeNotCallable.Match(output!);
            if (match.Success)
            {
                string expression = match.Groups[1].Value;
                int dot = expression.LastIndexOf('.');
                miss = new ApiMiss(
                    ApiMissKind.NodeExport, expression,
                    dot >= 0 ? expression.Substring(dot + 1) : expression);
                return true;
            }

            return false;
        }

        /// <summary>
        /// True when the deepest Python traceback frame is a file belonging to an INSTALLED
        /// PACKAGE rather than to the program the model wrote.
        ///
        /// <para>
        /// Frames are rendered relative to where the command ran (see
        /// <see cref="OutputPaths"/>), so a package under the session's environment appears
        /// as <c>../env/…</c> and the model's own program as a bare name or the word
        /// <c>command</c>. Only the LAST frame matters: a traceback that starts in the
        /// model's file and ends inside a library is the library failing.
        /// </para>
        /// </summary>
        private static bool FailedInsideALibrary(string output)
        {
            string? deepest = null;
            foreach (Match frame in PythonFrame.Matches(output))
                deepest = frame.Groups[1].Value;

            if (deepest == null)
                return false;

            string path = deepest.Replace('\\', '/');
            return path.Contains("/env/", StringComparison.Ordinal)
                || path.StartsWith("../env", StringComparison.Ordinal)
                || path.Contains("/site-packages/", StringComparison.Ordinal)
                || path.Contains("/dist-packages/", StringComparison.Ordinal)
                || path.Contains("/node_modules/", StringComparison.Ordinal);
        }

        private static readonly Regex PythonFrame = new(
            @"^\s*File ""([^""]+)"", line \d+",
            RegexOptions.Compiled | RegexOptions.CultureInvariant | RegexOptions.Multiline);

        private static readonly Regex PythonObjectAttribute = new(
            @"AttributeError: '([A-Za-z_][\w\.]*)' object has no attribute '(\w+)'",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex PythonTypeAttribute = new(
            @"AttributeError: type object '([A-Za-z_][\w\.]*)' has no attribute '(\w+)'",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex PythonModuleAttribute = new(
            @"AttributeError: module '([A-Za-z_][\w\.]*)' has no attribute '(\w+)'",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex PythonImportName = new(
            @"ImportError: cannot import name '(\w+)' from '([A-Za-z_][\w\.]*)'",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        // `new pptxgen.JSLib()` -> "pptxgen.JSLib is not a constructor". Deliberately
        // narrow: a dotted identifier only, so a V8 message about a subscript or a call
        // expression is left alone rather than probed for a package named "rows[0]".
        private static readonly Regex NodeNotConstructible = new(
            @"TypeError: ([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*) is not a constructor",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex NodeNotCallable = new(
            @"TypeError: ([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+) is not a function",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        /// <summary>
        /// The top-level packages a failing Python command had in play, most likely
        /// candidate first.
        ///
        /// <para>
        /// Needed because a Python <c>AttributeError</c> names the type and never its
        /// module: <c>'Slide' object has no attribute 'notes_page'</c> is the whole
        /// message. The import lines say where to look, and a shell command carries its
        /// own program in a heredoc so they are usually right there in the command; when
        /// it merely RUNS a file written by an earlier call, the file is read instead.
        /// </para>
        /// <para>
        /// Third-party roots are returned before standard-library ones. Not by excluding
        /// the standard library — <c>'Path' object has no attribute</c> is a real failure
        /// and <c>pathlib</c> is where the answer is — but by ordering, so that a bounded
        /// search spends its budget on the package the model was guessing at.
        /// </para>
        /// </summary>
        /// <param name="command">The failing command, heredoc body included.</param>
        /// <param name="readFile">
        /// Reads a workspace-relative path, or returns null. Supplied by the caller
        /// because only it knows how to resolve a model-written path safely.
        /// </param>
        public static IReadOnlyList<string> PythonImportRoots(
            string? command, Func<string, string?>? readFile = null)
        {
            var roots = new List<string>();
            void Collect(string? text)
            {
                if (string.IsNullOrEmpty(text))
                    return;
                foreach (Match match in PythonImportLine.Matches(text!))
                {
                    string list = match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value;
                    foreach (string item in list.Split(','))
                    {
                        string name = item.Trim();
                        int space = name.IndexOf(' ');
                        if (space > 0)
                            name = name.Substring(0, space);   // "import numpy as np"
                        int dot = name.IndexOf('.');
                        if (dot > 0)
                            name = name.Substring(0, dot);
                        if (name.Length > 0 && !roots.Contains(name, StringComparer.Ordinal))
                            roots.Add(name);
                    }
                }
            }

            Collect(command);
            if (readFile != null && command != null)
            {
                foreach (Match match in PythonFileName.Matches(command))
                {
                    Collect(readFile(match.Groups[1].Value));
                    if (roots.Count > 12)
                        break;
                }
            }

            roots.Sort((a, b) => StandardLibrary.Contains(a).CompareTo(StandardLibrary.Contains(b)));
            return roots;
        }

        /// <summary>
        /// The npm package a Node identifier was bound to, or null when it was not bound
        /// to one.
        ///
        /// <para>
        /// <c>pptxgen.JSLib is not a constructor</c> names a local variable, and a probe
        /// handed "pptxgen" would look for a package that does not exist. The binding is
        /// in the same command: <c>const pptxgen = require("pptxgenjs")</c>. Returning
        /// null when there is no such line is the point — an identifier that is an
        /// ordinary object (<c>pres.addSlide is not a function</c>) has no package behind
        /// it and must not be probed for one.
        /// </para>
        /// </summary>
        public static string? NodeRequireFor(
            string identifier, string? command, Func<string, string?>? readFile = null)
        {
            if (string.IsNullOrEmpty(identifier))
                return null;

            string? Search(string? text)
            {
                if (string.IsNullOrEmpty(text))
                    return null;
                var pattern = new Regex(
                    @"(?:const|let|var)\s+" + Regex.Escape(identifier) + @"\s*=\s*require\s*\(\s*['""]([^'""]+)['""]"
                    + @"|import\s+(?:\*\s+as\s+)?" + Regex.Escape(identifier) + @"\s+from\s+['""]([^'""]+)['""]",
                    RegexOptions.CultureInvariant);
                Match match = pattern.Match(text!);
                if (!match.Success)
                    return null;
                string package = match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value;
                // A relative import is the model's own file, not a package, and reading
                // its exports would answer a question nobody asked.
                return package.StartsWith('.') || package.StartsWith('/') ? null : package;
            }

            string? found = Search(command);
            if (found != null || readFile == null || command == null)
                return found;

            foreach (Match match in NodeFileName.Matches(command))
            {
                found = Search(readFile(match.Groups[1].Value));
                if (found != null)
                    return found;
            }
            return null;
        }

        // Not anchored at the start of a LINE, because a one-liner is a real and common
        // shape: `python3 -c 'import pathlib; pathlib.Path(".").parrent'` has its import
        // after a quote and its second statement after a semicolon, and a line-anchored
        // pattern finds neither — which meant no roots, and so no probe at all, for
        // exactly the commands a model types when it is checking something quickly.
        private static readonly Regex PythonImportLine = new(
            @"(?:^|[;'""\n])[ \t]*(?:from[ \t]+([A-Za-z_][\w\.]*)[ \t]+import\b|import[ \t]+([A-Za-z_][\w\., \t]*))",
            RegexOptions.Compiled | RegexOptions.CultureInvariant | RegexOptions.Multiline);

        private static readonly Regex PythonFileName = new(
            @"([A-Za-z0-9_./\\-]+\.py)\b",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex NodeFileName = new(
            @"([A-Za-z0-9_./\\-]+\.(?:js|mjs|cjs))\b",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        /// <summary>
        /// Standard-library roots, used only to push them DOWN the search order. Not
        /// exhaustive and does not need to be: a name missing from it is merely searched
        /// as eagerly as a third-party package.
        /// </summary>
        private static readonly HashSet<string> StandardLibrary = new(StringComparer.Ordinal)
        {
            "abc", "argparse", "asyncio", "base64", "collections", "contextlib", "copy", "csv",
            "dataclasses", "datetime", "decimal", "difflib", "enum", "functools", "glob", "hashlib",
            "http", "importlib", "io", "itertools", "json", "logging", "math", "os", "pathlib",
            "pickle", "pprint", "random", "re", "shutil", "socket", "sqlite3", "statistics",
            "string", "struct", "subprocess", "sys", "tempfile", "textwrap", "threading", "time",
            "traceback", "typing", "unittest", "urllib", "uuid", "warnings", "xml", "zipfile",
        };

        /// <summary>
        /// The name this host actually knows <paramref name="command"/> by, or null.
        ///
        /// <para>
        /// A missing PROGRAM and a program spelled differently are opposite situations
        /// with opposite advice, and getting them confused made the host argue against
        /// itself: <c>python</c> is not a package manager, so it fell through to "no
        /// package manager here can supply it — do the step another way", on a machine
        /// whose shell description had already told the model it has
        /// <c>python3 (3.9.6)</c>. Five incidents and nine rounds in the measured logs,
        /// every one of them a single character.
        /// </para>
        /// <para>
        /// PROBED, not assumed. Answering "use python3" on a host that has no python3
        /// either would be a second wrong answer on top of the first.
        /// </para>
        /// </summary>
        public static string? SpelledDifferentlyHere(string? command)
        {
            if (string.IsNullOrEmpty(command))
                return null;

            foreach (string alternative in Alternatives(command!))
            {
                if (CodeEnvironment.Which(alternative) != null)
                    return alternative;
            }
            return null;
        }

        /// <summary>Names the same tool goes by, most likely first.</summary>
        private static IEnumerable<string> Alternatives(string command)
        {
            switch (command.ToLowerInvariant())
            {
                case "python":
                    yield return "python3";
                    break;
                case "python3":
                    yield return "python";
                    break;
                case "node":
                    yield return "nodejs";
                    break;
                case "nodejs":
                    yield return "node";
                    break;
            }
        }

        /// <summary>
        /// True when the command a shell could not find is a package manager, which is a
        /// different problem from a missing program: the model was told to install
        /// something and this host spells the installer another way.
        /// </summary>
        public static bool IsPackageManager(string? command) =>
            command != null
            && (command.Equals("pip", StringComparison.OrdinalIgnoreCase)
                || command.Equals("pip3", StringComparison.OrdinalIgnoreCase)
                || command.Equals("npm", StringComparison.OrdinalIgnoreCase)
                || command.Equals("uv", StringComparison.OrdinalIgnoreCase)
                || command.Equals("yarn", StringComparison.OrdinalIgnoreCase)
                || command.Equals("pnpm", StringComparison.OrdinalIgnoreCase));
    }
}
