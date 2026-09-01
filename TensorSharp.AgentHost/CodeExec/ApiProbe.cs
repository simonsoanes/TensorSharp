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
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// When a run fails because the model guessed a library's API, go and read the real
    /// API out of the installed package and put it in the tool result.
    ///
    /// <para>
    /// <b>This is the most expensive failure in this server's logs, by a wide margin.</b>
    /// Rounds are not lost to patching or to shell syntax; they are lost to a model
    /// inventing an attribute, a class name or a constructor and then discovering one
    /// wrong guess per round. One logged session spent rounds 5 through 16 — eleven
    /// consecutive failures — on <c>pptxgen.JSLib()</c>, then
    /// <c>from pptx import Image</c>, then <c>slide.notes_page</c>, then
    /// <c>notes_slide.shapes.add_textbox</c>, then <c>notes_slide.body</c>, and ran out
    /// of its round budget before it could check its own output. Every one of those is a
    /// question the sandbox could have answered in a tenth of a second, because the
    /// package was already installed and sitting right there.
    /// </para>
    /// <para>
    /// A traceback is not the answer. <c>'Slide' object has no attribute 'notes_page'</c>
    /// says what is wrong and nothing at all about what is right, so the model's only
    /// move is another guess — and a 4B model's second guess is drawn from the same
    /// distribution as its first. What ends the loop is
    /// <c>did you mean: notes_slide</c> followed by the eleven names <c>Slide</c> actually
    /// has. Measured on the real failing script: 100 ms, and the correct answer first.
    /// </para>
    /// <para>
    /// It follows the rule this codebase already learned twice over — see
    /// <see cref="CodeDiagnostics"/> — that <b>every failure result must end in an
    /// action, not a description</b>, and that corrective guidance only lands when it is
    /// attached to the failing result. This is that idiom extended from "the module is
    /// not installed" to "the module is installed and does not work the way you think".
    /// </para>
    /// <para>
    /// <b>Cost, and what gates it.</b> Nothing on a successful run and nothing on a
    /// failure whose output does not match one of the shapes in
    /// <see cref="CodeDiagnostics.TryFindApiMiss"/>. When it does match, one extra
    /// confined process — the same sandbox, the same workspace, the same closed network
    /// as the run that just failed, never anything more. A round has already been lost by
    /// the time this runs, so a tenth of a second to stop losing the next ten is not a
    /// trade that needs thinking about.
    /// </para>
    /// <para>
    /// <b>It is diagnostics, so it fails silently.</b> If the interpreter cannot be
    /// resolved, the probe times out, or the package will not import, nothing is
    /// appended. That is deliberate and it is not a hidden fallback: the model still has
    /// the real traceback, unmodified, and is no worse off than before this existed. The
    /// one thing this must never do is replace a true error with a story about why the
    /// host could not investigate it.
    /// </para>
    /// </summary>
    public sealed class ApiProbe
    {
        private readonly CodeExecOptions _options;
        private readonly ISkillSandbox? _sandbox;

        /// <summary>How long the probe may spend reading the package. Beyond this, say nothing.</summary>
        private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(8);

        /// <summary>Create a probe that runs under the same terms as the run it explains.</summary>
        public ApiProbe(CodeExecOptions options, ISkillSandbox? sandbox)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _sandbox = sandbox;
        }

        /// <summary>
        /// Read the real API and describe it, or return null when there is nothing
        /// useful to say.
        /// </summary>
        /// <param name="miss">What the run got wrong, as read out of its output.</param>
        /// <param name="command">
        /// The command that failed. This is where the IMPORTS come from: a Python
        /// traceback names the type (<c>'Slide'</c>) but never the module it lives in, so
        /// the probe has to be told which packages are in play. A shell command carries
        /// its own program in a heredoc, so its import lines are right there.
        /// </param>
        /// <param name="workspace">The session workspace — its env directory is where the package is.</param>
        /// <param name="ranIn">The directory the failing command ran in.</param>
        public string? Explain(
            CodeDiagnostics.ApiMiss miss, string command, SessionWorkspace workspace, string? ranIn)
        {
            ArgumentNullException.ThrowIfNull(workspace);

            try
            {
                return miss.Kind == CodeDiagnostics.ApiMissKind.NodeExport
                    ? ExplainNode(miss, command, workspace, ranIn)
                    : ExplainPython(miss, command, workspace, ranIn);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return null;
            }
        }

        // ---- Python ---------------------------------------------------------

        private string? ExplainPython(
            CodeDiagnostics.ApiMiss miss, string command, SessionWorkspace workspace, string? ranIn)
        {
            if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
                || python == null)
            {
                return null;
            }

            // The type name in an AttributeError is unqualified, so the probe is handed
            // the packages the failing program imported and told to go looking. A module
            // attribute or a failed `from x import y` already names its own module, so
            // that one is searched first and the parsed roots are only a fallback.
            var roots = new List<string>();
            if (miss.Kind != CodeDiagnostics.ApiMissKind.PythonAttribute)
                roots.Add(RootOf(miss.Subject));
            roots.AddRange(CodeDiagnostics.PythonImportRoots(command, path => ReadWorkspaceText(workspace, ranIn, path)));
            roots = roots
                .Where(r => r.Length > 0)
                .Distinct(StringComparer.Ordinal)
                .Take(6)
                .ToList();
            if (roots.Count == 0)
                return null;

            string kind = miss.Kind switch
            {
                CodeDiagnostics.ApiMissKind.PythonAttribute => "attr",
                CodeDiagnostics.ApiMissKind.PythonModuleAttribute => "modattr",
                _ => "importname",
            };

            // No -I and no -S. Both were tried and both are wrong here: -I implies -E,
            // which drops PYTHONPATH and so hides the session's installed packages —
            // exactly the packages this exists to read — and -S skips site-packages,
            // which hides the host's. The probe has to resolve imports the way the
            // program that failed resolved them, or it reports a different library.
            var arguments = new List<string>
            {
                ScriptPath(workspace, "ts-apiprobe.py", PythonProbe), kind, miss.Subject, miss.Member,
            };
            arguments.AddRange(roots);
            return Run(python, arguments, workspace, ranIn);
        }

        // ---- Node -----------------------------------------------------------

        private string? ExplainNode(
            CodeDiagnostics.ApiMiss miss, string command, SessionWorkspace workspace, string? ranIn)
        {
            if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.JavaScript, out string? node, out _)
                || node == null)
            {
                return null;
            }

            // `pptxgen.JSLib is not a constructor` names a LOCAL VARIABLE, which tells the
            // probe nothing on its own. What it needs is the package that variable was
            // bound to, and that is written down in the same command:
            // `const pptxgen = require("pptxgenjs")`.
            string? package = CodeDiagnostics.NodeRequireFor(
                RootOf(miss.Subject), command, path => ReadWorkspaceText(workspace, ranIn, path));
            if (package == null)
                return null;

            var arguments = new List<string>
            {
                ScriptPath(workspace, "ts-apiprobe.js", NodeProbe), package, miss.Subject, miss.Member,
            };
            return Run(node, arguments, workspace, ranIn);
        }

        // ---- running it -----------------------------------------------------

        private string? Run(
            string interpreter, IReadOnlyList<string> arguments, SessionWorkspace workspace, string? ranIn)
        {
            ConfinedResult result = ConfinedProcess.Run(
                new ConfinedLaunch
                {
                    Interpreter = interpreter,
                    Arguments = arguments,
                    WriteDirectory = workspace.TempDirectory,
                    WorkingDirectory = ranIn is { Length: > 0 } ? ranIn : workspace.WorkDirectory,
                    ReadOnlyDirectory = workspace.Root,
                    ReadablePaths = new[] { workspace.EnvDirectory },
                    // The same closed network as the run being explained. A probe is not a
                    // privileged context; it reads what is already on disk.
                    AllowNetwork = false,
                    Timeout = Deadline,
                    MaxOutputBytes = 8 * 1024,
                    EnvironmentVariables = ProbeEnvironment(workspace),
                },
                _sandbox,
                _options.Unconfined ? SkillSandboxMode.Preferred : _options.Sandbox);

            if (!result.Started || result.TimedOut)
                return null;

            string text = result.Stdout.Trim();
            // The probe prints nothing when it could not find the thing. Its stderr is
            // never shown: a probe that itself blew up would otherwise read as a second
            // error in the model's own program.
            return text.Length == 0 ? null : OutputPaths.Scrub(text, workspace, ranIn);
        }

        /// <summary>
        /// Just enough environment to find the session's packages. Deliberately NOT the
        /// full shell environment: the probe is the host's own program, and every extra
        /// variable is a way for it to behave differently from the run it is explaining
        /// in some way that matters.
        /// </summary>
        private static Dictionary<string, string> ProbeEnvironment(SessionWorkspace workspace) => new(StringComparer.Ordinal)
        {
            ["HOME"] = workspace.WorkDirectory,
            ["USERPROFILE"] = workspace.WorkDirectory,
            ["TMPDIR"] = workspace.TempDirectory,
            ["PYTHONPATH"] = workspace.EnvDirectory,
            ["NODE_PATH"] = Path.Combine(workspace.EnvDirectory, "node_modules"),
            ["PYTHONDONTWRITEBYTECODE"] = "1",
            ["PYTHONUNBUFFERED"] = "1",
            ["NO_COLOR"] = "1",
        };

        /// <summary>
        /// The probe script on disk, written into the host's own state directory.
        ///
        /// <para>
        /// State, not the work directory, for the reason a sandbox profile is not kept
        /// there either: the work directory is the one place the confined process may
        /// write, so a script it can rewrite is a script it can replace. It is also the
        /// directory <c>ls</c> shows the model and the one artifact capture scans, and a
        /// host tool appearing in either would be a file the model did not create being
        /// offered to the user as its output.
        /// </para>
        /// </summary>
        private static string ScriptPath(SessionWorkspace workspace, string name, string body)
        {
            string path = Path.Combine(workspace.StateDirectory, name);
            // Rewritten every time rather than cached: the cost is one small write against
            // a probe that is about to launch a process anyway, and a stale copy from an
            // older build of the host would be a probe that answers questions this one no
            // longer asks.
            File.WriteAllText(path, body, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            return path;
        }

        /// <summary>
        /// A file the failing command named, read from the workspace so its imports can be
        /// parsed.
        ///
        /// <para>
        /// Routed through the workspace's own resolver, never through
        /// <see cref="Path.Combine(string,string)"/>: the path comes from a command the
        /// MODEL wrote, and this read happens in the HOST, which is not sandboxed. That is
        /// the shape of the symlink escape this codebase has already been bitten by — one
        /// permitted <c>ln -s</c> inside the workspace and a lexical containment check
        /// reads whatever it points at.
        /// </para>
        /// </summary>
        private static string? ReadWorkspaceText(SessionWorkspace workspace, string? ranIn, string relative)
        {
            string from = ranIn is { Length: > 0 } ? ranIn : workspace.WorkDirectory;
            if (!workspace.TryResolveFrom(from, relative, out string full, out _))
                return null;
            try
            {
                var info = new FileInfo(full);
                // A whole repository does not need parsing to find an import line, and a
                // binary the model happened to name is not a program.
                if (!info.Exists || info.Length > 512 * 1024)
                    return null;
                return File.ReadAllText(full);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return null;
            }
        }

        private static string RootOf(string dotted)
        {
            int dot = dotted.IndexOf('.');
            return dot > 0 ? dotted.Substring(0, dot) : dotted;
        }

        // ---- the probes -----------------------------------------------------

        /// <summary>
        /// Read the real API out of an installed Python package.
        ///
        /// <para>
        /// Every import is done with stdout and stderr swallowed. Walking a package
        /// imports its submodules, and a package that prints a banner or a deprecation
        /// warning at import time would otherwise put that text in the middle of the
        /// answer, where it reads as part of the model's own failure.
        /// </para>
        /// <para>
        /// Written for Python 3.9, which is the oldest interpreter this host resolves
        /// (Apple's <c>/usr/bin/python3</c>) — no <c>match</c>, no PEP 604 unions.
        /// </para>
        /// </summary>
        private const string PythonProbe = """"
import contextlib
import difflib
import importlib
import io
import pkgutil
import sys
import time

KIND = sys.argv[1]
SUBJECT = sys.argv[2]
MEMBER = sys.argv[3]
ROOTS = sys.argv[4:]

DEADLINE = time.time() + 6.0
MAX_MODULES = 400
MAX_NAMES = 80


@contextlib.contextmanager
def quiet():
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = out, err


def load(name):
    with quiet():
        return importlib.import_module(name)


def public(obj):
    try:
        return sorted(n for n in dir(obj) if not n.startswith("_"))
    except Exception:
        return []


def near(name, names):
    """Closest real names, by edit distance and then by containment.

    Containment matters as much as distance and difflib does not do it:
    'notes_page' -> 'has_notes_slide' is a poor ratio and exactly the answer.
    """
    hits = difflib.get_close_matches(name, names, n=3, cutoff=0.4)
    lowered = name.lower()
    for candidate in names:
        if len(hits) >= 5:
            break
        low = candidate.lower()
        if candidate not in hits and (lowered in low or low in lowered):
            hits.append(candidate)
    return hits


def walk(root):
    """The root package, then its submodules, bounded in count and in time."""
    try:
        package = load(root)
    except Exception:
        return
    yield root, package
    paths = getattr(package, "__path__", None)
    if not paths:
        return
    try:
        walker = pkgutil.walk_packages(paths, root + ".")
    except Exception:
        return
    seen = 0
    while True:
        if time.time() > DEADLINE or seen >= MAX_MODULES:
            return
        try:
            info = next(walker)
        except StopIteration:
            return
        except Exception:
            return
        seen += 1
        try:
            yield info.name, load(info.name)
        except Exception:
            continue


def say(lines):
    print("\n".join(lines))
    sys.exit(0)


def names_line(label, names):
    shown = names[:MAX_NAMES]
    text = ", ".join(shown)
    if len(names) > len(shown):
        text += ", ... and %d more" % (len(names) - len(shown))
    return "%s %s" % (label, text)


def suggestion(names):
    hits = near(MEMBER, names)
    return ["did you mean: " + ", ".join(hits)] if hits else []


if KIND == "attr":
    wanted = SUBJECT.rsplit(".", 1)[-1]
    for root in ROOTS:
        for modname, module in walk(root):
            found = getattr(module, wanted, None)
            if isinstance(found, type) and found.__name__ == wanted:
                names = public(found)
                if MEMBER in names:
                    # It does have it. Then the object was not what the model
                    # thought it was, and saying so is the whole finding.
                    say([
                        "%s (in %s) does have '%s', so the object you called it on is "
                        "not a %s. Print type(x) to see what it really is."
                        % (wanted, modname, MEMBER, wanted),
                    ])
                say(
                    ["%s (in %s) has no '%s'." % (wanted, modname, MEMBER)]
                    + suggestion(names)
                    + [names_line("%s really has:" % wanted, names)]
                )
    sys.exit(0)

if KIND == "modattr":
    try:
        module = load(SUBJECT)
    except Exception as error:
        say(["%s did not import here either: %s" % (SUBJECT, error)])
    names = public(module)
    say(
        ["module %s has no '%s'." % (SUBJECT, MEMBER)]
        + suggestion(names)
        + [names_line("%s really has:" % SUBJECT, names)]
    )

# KIND == "importname": `from SUBJECT import MEMBER` failed.
try:
    module = load(SUBJECT)
except Exception as error:
    say(["%s did not import here either: %s" % (SUBJECT, error)])

names = public(module)
lines = ["%s does not export '%s'." % (SUBJECT, MEMBER)]

# Where the name really lives is the answer that ends the round, so it is
# hunted for across the whole package before anything else is reported.
elsewhere = []
for root in [SUBJECT.split(".")[0]] + ROOTS:
    if root in ("", SUBJECT):
        continue
    for modname, candidate in walk(root):
        if modname != SUBJECT and getattr(candidate, MEMBER, None) is not None:
            line = "from %s import %s" % (modname, MEMBER)
            if line not in elsewhere:
                elsewhere.append(line)
        if len(elsewhere) >= 3:
            break
    if elsewhere:
        break

if elsewhere:
    lines.append("'%s' is really in: %s" % (MEMBER, "; ".join(elsewhere)))
else:
    lines += suggestion(names)
lines.append(names_line("%s exports:" % SUBJECT, names))
say(lines)
"""";

        /// <summary>
        /// Read the real shape of an installed npm package.
        ///
        /// <para>
        /// The question a Node <c>TypeError</c> raises is almost always the same one —
        /// is the module's export the constructor itself or an object holding it — and it
        /// is the question the model cannot answer by guessing, because both spellings are
        /// common in the wild and the docs of any given package show only one.
        /// <c>pptxgenjs</c> exports the constructor directly, which is why
        /// <c>new pptxgen.JSLib()</c> and <c>new pptxgen.Presentation()</c> both failed.
        /// </para>
        /// </summary>
        private const string NodeProbe = """"
'use strict';
const target = process.argv[2];
const expression = process.argv[3] || '';
const member = process.argv[4] || '';

let loaded;
try {
    loaded = require(target);
} catch (error) {
    console.log('require("' + target + '") failed here too: ' + error.message);
    process.exit(0);
}

const MAX = 40;
const list = (names) => names.length > MAX
    ? names.slice(0, MAX).join(', ') + ', ... and ' + (names.length - MAX) + ' more'
    : names.join(', ');

const own = (value) => {
    try {
        return Object.getOwnPropertyNames(value || {})
            .filter((name) => !['length', 'name', 'prototype', 'caller', 'arguments', 'constructor'].includes(name));
    } catch (error) {
        return [];
    }
};

const lines = [];
const kind = typeof loaded;

if (kind === 'function') {
    lines.push('require("' + target + '") returns the CONSTRUCTOR itself'
        + (loaded.name ? ' (' + loaded.name + ')' : '') + ', not an object holding one.');
    lines.push('Use it exactly like this:  const ' + (loaded.name || 'Lib') + ' = require("' + target
        + '");  const instance = new ' + (loaded.name || 'Lib') + '();');
    const methods = own(loaded.prototype);
    if (methods.length) {
        lines.push('an instance has: ' + list(methods.sort()));
    }
    const statics = own(loaded);
    if (statics.length) {
        lines.push('on the constructor itself: ' + list(statics.sort()));
    }
} else if (loaded && kind === 'object') {
    const names = Object.keys(loaded).sort();
    lines.push('require("' + target + '") returns an OBJECT, not a constructor.');
    const constructors = names.filter((name) => typeof loaded[name] === 'function' && /^[A-Z]/.test(name));
    if (constructors.length) {
        lines.push('the constructors on it are: ' + list(constructors)
            + '  — e.g. new (require("' + target + '").' + constructors[0] + ')()');
    }
    if (loaded.default !== undefined) {
        lines.push('it also has a .default, so this package wants: '
            + 'const X = require("' + target + '").default;');
    }
    lines.push('everything it exports: ' + list(names));
} else {
    lines.push('require("' + target + '") returned a ' + kind + ', which cannot be constructed or called.');
}

if (member && expression) {
    lines.push('So ' + expression + ' does not exist; that is why "' + member + '" was not found on it.');
}

console.log(lines.join('\n'));
"""";
    }
}
