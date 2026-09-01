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
using System.IO;
using System.Linq;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Reading the real API out of an installed package when the model guessed it wrong.
///
/// <para>
/// Every case here is taken verbatim from this server's own logs. The session that
/// motivated the feature spent rounds 5 through 16 on five consecutive API guesses and
/// then ran out of its round budget, so the assertion that matters is not "the probe
/// produced text" but "the probe produced the NAME the model should have used" — the
/// first line of a self-correcting result.
/// </para>
/// </summary>
public class ApiProbeTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _manager;
    private readonly SessionWorkspace _workspace;

    public ApiProbeTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-apiprobe-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _manager = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
        _workspace = _manager.GetOrCreate("s");
    }

    public void Dispose()
    {
        try { _manager.Release("s"); } catch { /* best effort */ }
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- reading the failure -------------------------------------------------

    [Fact]
    public void TheAttributeErrorFromTheLogIsRecognised()
    {
        const string output = """
            Traceback (most recent call last):
              File "command", line 86, in <module>
                slide.notes_page.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(4))
            AttributeError: 'Slide' object has no attribute 'notes_page'
            """;

        Assert.True(CodeDiagnostics.TryFindApiMiss(output, out CodeDiagnostics.ApiMiss miss));
        Assert.Equal(CodeDiagnostics.ApiMissKind.PythonAttribute, miss.Kind);
        Assert.Equal("Slide", miss.Subject);
        Assert.Equal("notes_page", miss.Member);
    }

    [Fact]
    public void TheImportErrorFromTheLogIsRecognised()
    {
        Assert.True(CodeDiagnostics.TryFindApiMiss(
            "ImportError: cannot import name 'MSO_SHAPE' from 'pptx.enum.text' (/x/y.py)",
            out CodeDiagnostics.ApiMiss miss));
        Assert.Equal(CodeDiagnostics.ApiMissKind.PythonImportName, miss.Kind);
        Assert.Equal("pptx.enum.text", miss.Subject);
        Assert.Equal("MSO_SHAPE", miss.Member);
    }

    [Fact]
    public void TheNodeConstructorErrorFromTheLogIsRecognised()
    {
        Assert.True(CodeDiagnostics.TryFindApiMiss(
            "TypeError: pptxgen.JSLib is not a constructor", out CodeDiagnostics.ApiMiss miss));
        Assert.Equal(CodeDiagnostics.ApiMissKind.NodeExport, miss.Kind);
        Assert.Equal("pptxgen.JSLib", miss.Subject);
        Assert.Equal("JSLib", miss.Member);
    }

    [Fact]
    public void AModuleAttributeIsNotConfusedWithAnObjectAttribute()
    {
        Assert.True(CodeDiagnostics.TryFindApiMiss(
            "AttributeError: module 'markitdown' has no attribute '__version__'",
            out CodeDiagnostics.ApiMiss miss));
        Assert.Equal(CodeDiagnostics.ApiMissKind.PythonModuleAttribute, miss.Kind);
        Assert.Equal("markitdown", miss.Subject);
    }

    /// <summary>
    /// The anti-vacuity case. A missing MODULE has its own coaching already
    /// (<see cref="CodeDiagnostics.TryFindMissingModule"/>) and must not be turned into a
    /// probe of a package that is not installed — the probe would find nothing, take a
    /// process to do it, and say nothing, which is strictly worse than the advice that
    /// branch already gives.
    /// </summary>
    [Fact]
    public void AMissingModuleIsNotAnApiMiss()
    {
        Assert.False(CodeDiagnostics.TryFindApiMiss(
            "ModuleNotFoundError: No module named 'reportlab'", out _));
        Assert.False(CodeDiagnostics.TryFindApiMiss("Cannot find module 'docx'", out _));
    }

    /// <summary>
    /// A <c>TypeError</c> about something that is not an identifier must not be read as
    /// one. V8 says "rows[0].x is not a function" and "(intermediate value) is not a
    /// function"; neither names a package, and probing for one would spend a process to
    /// learn nothing.
    /// </summary>
    [Fact]
    public void ATypeErrorThatNamesNoIdentifierIsIgnored()
    {
        Assert.False(CodeDiagnostics.TryFindApiMiss(
            "TypeError: (intermediate value) is not a function", out _));
        Assert.False(CodeDiagnostics.TryFindApiMiss(
            "TypeError: Cannot read properties of undefined (reading 'x')", out _));
    }

    // ---- finding the packages in play ---------------------------------------

    [Fact]
    public void ImportRootsComeOutOfTheHeredocTheCommandCarries()
    {
        const string command = """
            cat > deck.py <<'EOF'
            from pptx import Presentation
            from pptx.util import Inches, Pt
            import numpy as np
            import os, sys
            EOF
            python3 deck.py
            """;

        var roots = CodeDiagnostics.PythonImportRoots(command);
        Assert.Contains("pptx", roots);
        Assert.Contains("numpy", roots);
        // Ordering is the useful part: the third-party package is searched before the
        // standard library, because a bounded search spends its budget on the first ones.
        Assert.True(roots.ToList().IndexOf("pptx") < roots.ToList().IndexOf("os"));
    }

    /// <summary>
    /// The case the heredoc parse cannot answer: a later call merely RUNS a file an
    /// earlier one wrote, so the imports are on disk rather than in the command.
    /// </summary>
    [Fact]
    public void ImportRootsAreReadFromAFileTheCommandOnlyRuns()
    {
        File.WriteAllText(Path.Combine(_workspace.WorkDirectory, "deck.py"),
            "from pptx import Presentation\nprint(1)\n");

        var roots = CodeDiagnostics.PythonImportRoots(
            "python3 deck.py",
            path => path == "deck.py"
                ? File.ReadAllText(Path.Combine(_workspace.WorkDirectory, path))
                : null);

        Assert.Contains("pptx", roots);
    }

    [Fact]
    public void ANodeIdentifierIsResolvedToThePackageItWasRequiredFrom()
    {
        Assert.Equal("pptxgenjs", CodeDiagnostics.NodeRequireFor(
            "pptxgen", "const pptxgen = require(\"pptxgenjs\");\nconst p = new pptxgen.JSLib();"));
        Assert.Equal("pptxgenjs", CodeDiagnostics.NodeRequireFor(
            "pptxgen", "import pptxgen from 'pptxgenjs';"));
    }

    /// <summary>
    /// An identifier that is an ordinary object has no package behind it. Returning null
    /// is what stops the probe from being run for <c>pres.addSlide is not a function</c>,
    /// where there is nothing to read.
    /// </summary>
    [Fact]
    public void ANodeIdentifierWithNoRequireBehindItIsNotResolved()
    {
        Assert.Null(CodeDiagnostics.NodeRequireFor("pres", "const pres = new pptxgen();"));
        Assert.Null(CodeDiagnostics.NodeRequireFor("helper", "const helper = require('./helper');"));
    }

    // ---- the probe itself ----------------------------------------------------

    /// <summary>
    /// The whole point, end to end, against a package that is certainly present: the
    /// standard library. <c>pathlib.Path</c> has no <c>parrent</c>, and a result that
    /// merely repeats that is a result the model cannot act on — so the assertion is that
    /// the real name comes back.
    /// </summary>
    [Fact]
    public void TheProbeNamesTheAttributeThatReallyExists()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;   // No Python on this host; the probe correctly says nothing.
        }

        var probe = new ApiProbe(
            new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off }, sandbox: null);

        string? report = probe.Explain(
            new CodeDiagnostics.ApiMiss(CodeDiagnostics.ApiMissKind.PythonAttribute, "Path", "parrent"),
            "python3 -c 'import pathlib; pathlib.Path(\".\").parrent'",
            _workspace,
            _workspace.WorkDirectory);

        Assert.NotNull(report);
        Assert.Contains("did you mean", report!, StringComparison.Ordinal);
        Assert.Contains("parent", report!, StringComparison.Ordinal);
        // And the full surface, which is what lets the model pick a different member
        // rather than guess a second time at the one it wanted.
        Assert.Contains("really has:", report!, StringComparison.Ordinal);
    }

    /// <summary>
    /// Where a name really lives is the answer that ends the round. <c>datetime</c> is
    /// the reliable standard-library shape of the logged <c>MSO_SHAPE</c> case: asking
    /// <c>datetime.timezone</c> for <c>timedelta</c> should point at the module that has it.
    /// </summary>
    [Fact]
    public void TheProbeSaysWhereAnImportedNameReallyLives()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;
        }

        var probe = new ApiProbe(
            new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off }, sandbox: null);

        string? report = probe.Explain(
            new CodeDiagnostics.ApiMiss(CodeDiagnostics.ApiMissKind.PythonImportName, "json.decoder", "dumps"),
            "python3 -c 'from json.decoder import dumps'",
            _workspace,
            _workspace.WorkDirectory);

        Assert.NotNull(report);
        Assert.Contains("does not export 'dumps'", report!, StringComparison.Ordinal);
        Assert.Contains("json", report!, StringComparison.Ordinal);
    }

    /// <summary>
    /// The probe is the HOST's program, so it must not be written where the confined
    /// process can replace it — the same rule the seatbelt profile had to learn. It goes
    /// in the state directory, which is also not the directory <c>ls</c> shows the model
    /// nor the one artifact capture scans.
    /// </summary>
    [Fact]
    public void TheProbeScriptIsNotWrittenWhereTheModelCanSeeOrReplaceIt()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;
        }

        var probe = new ApiProbe(
            new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off }, sandbox: null);
        probe.Explain(
            new CodeDiagnostics.ApiMiss(CodeDiagnostics.ApiMissKind.PythonAttribute, "Path", "parrent"),
            "import pathlib", _workspace, _workspace.WorkDirectory);

        Assert.Empty(Directory.GetFiles(_workspace.WorkDirectory, "ts-apiprobe.*"));
        Assert.NotEmpty(Directory.GetFiles(_workspace.StateDirectory, "ts-apiprobe.*"));
    }
}
