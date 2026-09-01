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
using TensorSharp.Runtime;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Covers the two halves of getting a script's arguments to it: the declaration that
/// tells the model where they go, and the error that corrects it when it puts them
/// somewhere else.
///
/// <para>
/// Both exist because of one observed failure. Asked to run a script with an argument,
/// gemma-4-E4B called <c>skills_run(path: "scripts/budget.py 2400")</c> — the whole
/// command line in <c>path</c> — because that is the shape the skill's own SKILL.md
/// writes ("RUN <c>scripts/budget.py &lt;payload_kg&gt;</c>"), and its reasoning said
/// out loud: "skills_run … takes <c>skill</c> and <c>path</c>". The reply it got,
/// "'scripts/budget.py 2400' does not exist in this skill", was true and told it
/// nothing, so it read the script to check the filename, retried the identical call,
/// and spent the whole round budget without answering. Four other families called the
/// same tool correctly, which makes it a declaration problem rather than a model one.
/// </para>
/// </summary>
public class SkillRunArgumentShapeTests : IDisposable
{
    private readonly string _baseDir;

    public SkillRunArgumentShapeTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-args-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private Skill WriteSkill()
    {
        string dir = Path.Combine(_baseDir, "orbital");
        Directory.CreateDirectory(Path.Combine(dir, "scripts"));
        File.WriteAllText(Path.Combine(dir, "SKILL.md"),
            "---\nname: orbital\ndescription: Computes fuel budgets.\n---\n\nRUN `scripts/budget.py <payload_kg>`.\n");
        File.WriteAllText(Path.Combine(dir, "scripts", "budget.py"), "print('ok')\n");
        return new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } })
            .Skills.Single(s => s.Id == "orbital");
    }

    // ---- the declaration: get it right first time ---------------------------

    [Fact]
    public void TheRunToolTellsTheModelWhereArgumentsGo()
    {
        ToolFunction run = SkillTools.BuiltIn(allowScripts: true)
            .Single(t => t.Name == SkillTools.RunToolName);

        // The tool's own description must mention args — a model that reads only the
        // summary line still has to learn that arguments are possible at all.
        Assert.Contains("args", run.Description, StringComparison.OrdinalIgnoreCase);

        // And `path` must say what does NOT belong in it, because the natural mistake is
        // to paste the whole command line.
        string path = run.Parameters["path"].Description;
        Assert.Contains("never append arguments", path, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("args", path, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TheArgsParameterLeadsWithPurposeAndShowsAnExample()
    {
        ToolFunction run = SkillTools.BuiltIn(allowScripts: true)
            .Single(t => t.Name == SkillTools.RunToolName);
        string args = run.Parameters["args"].Description;

        // A concrete worked example beats a format grammar for a model that is about to
        // guess. This is the exact case that failed.
        Assert.Contains("2400", args, StringComparison.Ordinal);
    }

    // ---- the error: recover when it is still wrong --------------------------

    [Fact]
    public void ACrammedCommandLine_IsAnsweredWithTheCorrectedCall()
    {
        Skill skill = WriteSkill();
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Off });

        SkillToolResult result = runner.Run(skill, "scripts/budget.py 2400", Array.Empty<string>());

        Assert.False(result.Ok);
        // It must name the mistake, not merely report a missing file.
        Assert.Contains("arguments were included in 'path'", result.Content, StringComparison.Ordinal);
        // And hand back the call that would have worked, both halves spelled out.
        Assert.Contains("path=\"scripts/budget.py\"", result.Content, StringComparison.Ordinal);
        Assert.Contains("args=\"2400\"", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AGenuinelyMissingFile_StillGetsThePlainError_WithNoMisleadingAdvice()
    {
        Skill skill = WriteSkill();
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Off });

        SkillToolResult result = runner.Run(skill, "scripts/nope.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Contains("does not exist in this skill", result.Content, StringComparison.Ordinal);
        // Nothing was crammed, so offering a "corrected" call would be nonsense.
        Assert.DoesNotContain("arguments were included", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingFileWhoseLeadingTokenAlsoMisses_GetsNoInventedAdvice()
    {
        Skill skill = WriteSkill();
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Off });

        // Whitespace present, but the head does not resolve either — so this is a wrong
        // filename, not a crammed argument, and must not be diagnosed as one.
        SkillToolResult result = runner.Run(skill, "scripts/absent.py 2400", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.DoesNotContain("arguments were included", result.Content, StringComparison.Ordinal);
    }
}
