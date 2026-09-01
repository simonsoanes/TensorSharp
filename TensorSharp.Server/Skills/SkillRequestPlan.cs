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
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Logging;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.Skills
{
    /// <summary>
    /// Everything one chat request needs in order to be answered with skills:
    /// which skills it selected, the text to put in front of the model, the tools to
    /// offer alongside the caller's, and the sandbox those tools run in.
    ///
    /// <para>
    /// One type shared by all four chat surfaces — <c>/v1/chat/completions</c>,
    /// <c>/v1/responses</c>, Ollama's <c>/api/chat</c> and the Web UI's — because the
    /// only thing that differs between them is where the <c>skills</c> array is read
    /// from. Duplicating the resolution, the budget and the capability check per
    /// protocol is exactly how three of the four end up subtly different, and a
    /// mismatch here is invisible: the request still succeeds, the model just never
    /// sees the skill.
    /// </para>
    /// </summary>
    internal sealed class SkillRequestPlan
    {
        private SkillRequestPlan(
            SkillPlan prompt,
            IReadOnlyList<Skill> selected,
            SkillToolContext toolContext,
            List<ToolFunction> tools,
            SkillAgentLoopOptions loopOptions,
            bool toolsOffered,
            IReadOnlyCollection<ToolFunction>? clientTools)
        {
            Prompt = prompt;
            Selected = selected;
            ToolContext = toolContext;
            Tools = tools;
            LoopOptions = loopOptions;
            ToolsOffered = toolsOffered;
            ClientTools = clientTools ?? Array.Empty<ToolFunction>();
        }

        /// <summary>The rendered instruction block and what went into it.</summary>
        public SkillPlan Prompt { get; }

        /// <summary>The skills the request named, resolved and sorted.</summary>
        public IReadOnlyList<Skill> Selected { get; }

        /// <summary>The sandbox the built-in tools resolve paths in.</summary>
        public SkillToolContext ToolContext { get; }

        /// <summary>The caller's tools plus the built-in skill tools, or the caller's alone.</summary>
        public List<ToolFunction> Tools { get; }

        /// <summary>
        /// The caller's OWN tools, kept apart from <see cref="Tools"/>.
        ///
        /// <para>
        /// This is what the loop partitions on, and it must not be the merged list. A
        /// tool THIS host declares but does not recognise has to end up in the loop's
        /// "nobody can answer this" bucket, where the model is told so and gets another
        /// round; matching it against the merged roster would call it the caller's and
        /// hand it to a client with no implementation — which is precisely how
        /// <c>list_files</c> ended turns with nothing rendered at all.
        /// </para>
        /// </summary>
        public IReadOnlyCollection<ToolFunction> ClientTools { get; }

        /// <summary>Bounds for the disclosure loop.</summary>
        public SkillAgentLoopOptions LoopOptions { get; }

        /// <summary>
        /// False when the model's chat format cannot carry tool declarations, so the
        /// skill body was written into the prompt instead and no loop will run.
        /// </summary>
        public bool ToolsOffered { get; }

        /// <summary>
        /// Every skill file the model actually read, appended as the loop runs. The Web
        /// UI drains this to stream a trace; the audit log reports it at the end.
        /// </summary>
        public List<SkillToolInvocation> Invocations { get; } = new();

        /// <summary>
        /// True when this plan changes nothing about the request.
        ///
        /// <para>
        /// No longer the same question as "is the prompt empty". A code-only plan renders
        /// no instruction block at all — its <see cref="Prompt"/> is
        /// <see cref="SkillPlan.Empty"/> — while still declaring <c>shell</c> and
        /// carrying the runner that answers it. Reading the prompt alone would report such
        /// a plan as inert and invite a caller to drop it, silently disabling the feature.
        /// </para>
        /// </summary>
        public bool IsEmpty => Prompt.IsEmpty && !ToolsOffered;

        /// <summary>
        /// Default rounds for a plan that can also RUN code.
        ///
        /// <para>
        /// Eight was sized for progressive disclosure — read a skill, read two of its
        /// references, page through a long one — and it is generous for that. It is not
        /// the same activity as writing a program, running it, reading the traceback and
        /// fixing it, and the same counter gates both. Measured on the failure that
        /// prompted this: converting one README into an internal-comms document and then
        /// into a slide deck spent three rounds reading skills, two producing the
        /// document, and three on a deck the model was still debugging when the budget
        /// ran out — the very next generation held a correct program that was never run.
        /// </para>
        /// <para>
        /// Still bounded, and still one generation per round. An operator who names a
        /// number keeps it, in both directions.
        /// </para>
        /// </summary>
        private static int RoundsFor(ServerHostingOptions? options, bool offerCode)
        {
            int configured = options?.SkillsMaxRounds ?? 8;

            // No options at all is a host with no skills configuration — which is a
            // perfectly ordinary --code-exec deployment, and one with no operator
            // preference to respect. Only a number somebody actually set is left alone.
            if (!offerCode || options is { SkillsMaxRoundsSpecified: true })
                return configured;
            return Math.Max(configured, SkillHostOptions.CodeExecutionRounds);
        }

        /// <summary>Names the model may reach, for logging.</summary>
        public string DescribeSelection() =>
            Selected.Count == 0 ? "(none)" : string.Join(",", Selected.Select(s => s.Id));

        /// <summary>
        /// Build the plan for one request, or null when neither skills nor code execution
        /// changes anything about it.
        /// </summary>
        /// <param name="registry">The server's skills. Null when the feature is off.</param>
        /// <param name="requestedSkills">The request's <c>skills</c> array, or null.</param>
        /// <param name="discovery">
        /// The request's <c>skills_discovery</c> flag. Null follows the server default,
        /// which advertises the rest of the registry so the model can pick up a skill
        /// the caller did not think to name.
        /// </param>
        /// <param name="clientTools">The caller's own tools. Not modified.</param>
        /// <param name="architecture">
        /// The loaded model's <c>general.architecture</c>. Decides whether tools can be
        /// offered at all — Mistral 3 discards tool declarations and tool results.
        /// </param>
        /// <param name="contextTokens">The model's context length, for budgeting. Zero when unknown.</param>
        /// <param name="options">Server configuration.</param>
        /// <param name="allowTools">
        /// False to deliver skills WITHOUT the built-in tools, writing the selected
        /// bodies into the prompt instead.
        ///
        /// <para>
        /// The caller that needs this is a structured-output request. Its grammar
        /// constrains every generated token to the schema, so a round in which the model
        /// wanted to call <c>skills_read</c> could not produce the call, and the loop
        /// would never terminate usefully. Refusing the combination outright would be
        /// the easy answer; delivering the instructions up front instead means
        /// <c>response_format</c> and <c>skills</c> compose, at the cost of progressive
        /// disclosure for that one request.
        /// </para>
        /// </param>
        /// <param name="unknown">Names the registry does not have. A caller should reject the request.</param>
        /// <param name="codeInputFiles">
        /// Files the user attached to the conversation, for the shell tool to stage into
        /// the program's working directory. The declaration names them so the model opens
        /// the real attachment instead of re-typing its content into the source.
        /// </param>
        /// <param name="workspace">
        /// The session's persistent workspace, shared by <c>shell</c> and skill
        /// scripts. Null on hosts (or protocols) that keep the call-scoped scratch.
        /// </param>
        /// <param name="captureProducedFiles">
        /// How files a skill script produced become user-downloadable. Only used
        /// together with <paramref name="workspace"/>.
        /// </param>
        public static SkillRequestPlan Create(
            SkillRegistry registry,
            IReadOnlyList<string> requestedSkills,
            bool? discovery,
            List<ToolFunction> clientTools,
            string architecture,
            int contextTokens,
            ServerHostingOptions options,
            out IReadOnlyList<string> unknown,
            bool allowTools = true,
            ICodeRunner? codeRunner = null,
            IReadOnlyList<CodeInputFile>? codeInputFiles = null,
            SessionWorkspace? workspace = null,
            WorkspaceFileCapture? captureProducedFiles = null,
            ILogger? logger = null)
        {
            unknown = Array.Empty<string>();

            // Code execution is its own feature and must work with skills switched off
            // entirely, so it is decided before any of the skills gates below.
            bool offerCode = codeRunner is { CanRun: true } && allowTools;

            // The operator enabled --code-exec and this request could use it, but the
            // loaded family cannot carry tool declarations, so the shell tool is quietly not
            // offered. Say so once per family instead of never.
            if (offerCode && !SkillCapabilities.For(architecture).ToolsRendered)
                WarnCodeToolUnrenderable(logger, architecture);

            if (registry == null || options == null || !options.SkillsEnabled)
                return offerCode
                    ? CodeOnly(clientTools, architecture, codeRunner!, options, codeInputFiles, workspace)
                    : null;

            // A request that names no skills inherits the operator's --skill selection.
            // One that names skills REPLACES it rather than adding to it, so a client can
            // always narrow the selection - and an explicit empty array means "none",
            // which is the only way to opt out of an operator default per request.
            if (requestedSkills == null && options.DefaultSkills.Count > 0)
                requestedSkills = options.DefaultSkills;

            bool anythingRequested = requestedSkills != null && requestedSkills.Count > 0;
            bool advertise = discovery ?? options.SkillsDiscovery;

            // No selection and nothing to advertise means skills are simply not part of
            // this request: no block, no tools, no behaviour change at all. That matters
            // beyond tidiness — declaring tools flips every adapter's output parser on
            // and is rejected alongside a structured-output format.
            if (!anythingRequested && (!advertise || registry.Skills.Count == 0))
            {
                return offerCode
                    ? CodeOnly(clientTools, architecture, codeRunner!, options, codeInputFiles, workspace)
                    : null;
            }

            IReadOnlyList<Skill> selected = registry.Resolve(requestedSkills, out unknown);
            if (unknown.Count > 0)
                return null;

            IReadOnlyList<Skill> catalog = advertise ? registry.Skills : Array.Empty<Skill>();

            SkillModelCapabilities capabilities = SkillCapabilities.For(architecture);
            bool offerTools = allowTools && capabilities.ToolsRendered;

            SkillPlan prompt = SkillPrompt.Plan(selected, catalog, new SkillPromptOptions
            {
                ContextTokens = contextTokens,
                ToolsAvailable = offerTools,
            });

            if (prompt.IsEmpty && !offerCode)
                return null;

            List<ToolFunction> tools = clientTools;
            if (offerTools)
            {
                tools = SkillTools.Merge(clientTools, options.SkillsAllowScripts, out IReadOnlyList<string> shadowed);

                // A caller tool that shadows one of ours changes who answers a name the
                // model will use, and discarding this list meant nobody could see it had
                // happened. The CLI has always logged it; the server was silent.
                foreach (string name in shadowed)
                {
                    logger?.LogWarning(LogEventIds.HostConfiguration,
                        "skills.tool-shadowed name={ToolName} - the request's own tool definition wins", name);
                }
            }

            // the shell tool rides alongside the skills tools rather than replacing them: a
            // skill's instructions may well tell the model to compute something, and the
            // two are useful in the same turn.
            if (offerCode && offerTools)
                tools = AppendCodeTool(tools, codeRunner!, codeInputFiles, workspace, prompt.Reachable.Count());

            if (workspace != null && offerTools)
                DescribeSharedWorkspace(tools, codeRunner);

            var context = new SkillToolContext(prompt.Reachable.ToList())
            {
                ScriptRunner = options.SkillsAllowScripts && offerTools
                    ? new SkillScriptRunner(new SkillScriptRunnerOptions
                    {
                        Sandbox = options.SkillsSandbox,
                        AllowNetwork = options.SkillsAllowNetwork,
                        Workspace = workspace,
                        CaptureProducedFiles = captureProducedFiles,
                        PackageInstaller = codeRunner,
                    }, logger)
                    : null,
                CodeRunner = offerCode && offerTools ? codeRunner : null,
                CodeInputFiles = codeInputFiles ?? Array.Empty<CodeInputFile>(),
                Workspace = workspace,
            };

            var loopOptions = new SkillAgentLoopOptions
            {
                MaxRounds = RoundsFor(options, offerCode),
                ToolResultsAreRendered = capabilities.ToolResultsRendered,
            };

            // The editing rules ride along with the skills block rather than replacing it.
            // A skill's instructions are about a task; these are about how to change a
            // file, and a turn that does both needs both. Appended AFTER the skills block
            // so a skill's own wording is what the model reads first.
            if (offerCode && offerTools && CodePlan(tools).Instructions is { Length: > 0 } editing)
                prompt = prompt with { Instructions = prompt.Instructions + "\n" + editing };

            // Just offerTools: the shell tool rides the same declaration channel as skills_read,
            // so a family that cannot carry declarations carries neither. There is no case
            // where code execution is offered while the skill tools are withheld.
            return new SkillRequestPlan(
                prompt, selected, context, tools, loopOptions, offerTools, clientTools);
        }

        /// <summary>
        /// The plan for a request that has no skills but may run code.
        ///
        /// <para>
        /// Code execution was asked for as its own feature, so it cannot depend on the
        /// skills machinery being configured — a host with <c>--code-exec</c> and no
        /// <c>--skills-dir</c> is a perfectly ordinary deployment.
        /// </para>
        /// <para>
        /// This used to inject NOTHING, on the reasoning that "the tool declaration says
        /// everything the model needs". That was a bet and it lost: with the patch tool
        /// declared and an emphatic prefer-a-patch paragraph in its description, models
        /// used it zero times across the runs where both it and the shell were offered,
        /// and re-typed whole files instead. <see cref="CodePrompt"/> is the experiment
        /// that had never been run — six lines, in the channel both reference
        /// implementations use heavily and this one had left empty.
        /// </para>
        /// </summary>
        private static SkillRequestPlan? CodeOnly(
            List<ToolFunction> clientTools,
            string architecture,
            ICodeRunner codeRunner,
            ServerHostingOptions? options,
            IReadOnlyList<CodeInputFile>? codeInputFiles = null,
            SessionWorkspace? workspace = null)
        {
            SkillModelCapabilities capabilities = SkillCapabilities.For(architecture);
            if (!capabilities.ToolsRendered)
                return null;                       // nothing could parse the call back out

            List<ToolFunction> tools = AppendCodeTool(
                clientTools != null ? new List<ToolFunction>(clientTools) : new List<ToolFunction>(),
                codeRunner,
                codeInputFiles,
                workspace);

            if (workspace != null)
                DescribeSharedWorkspace(tools, codeRunner);

            var context = new SkillToolContext(Array.Empty<Skill>())
            {
                CodeRunner = codeRunner,
                CodeInputFiles = codeInputFiles ?? Array.Empty<CodeInputFile>(),
                Workspace = workspace,
            };

            return new SkillRequestPlan(
                CodePlan(tools),
                Array.Empty<Skill>(),
                context,
                tools,
                new SkillAgentLoopOptions
                {
                    // --skills-max-rounds bounds this too, on a host that may have no
                    // skills at all. One knob for "rounds the server answers itself" is
                    // the right shape; the flag's name is just narrower than its job.
                    MaxRounds = RoundsFor(options, offerCode: true),
                    ToolResultsAreRendered = capabilities.ToolResultsRendered,
                },
                toolsOffered: true,
                clientTools);
        }

        /// <summary>
        /// Tell the model, on the declarations themselves, that this conversation's
        /// tools share one persistent working directory. Without this it re-generates
        /// files it already has and re-installs packages every call.
        /// </summary>
        private static void DescribeSharedWorkspace(List<ToolFunction> tools, ICodeRunner? codeRunner)
        {
            // The shell declaration already states that the directory persists — it is
            // the shell's own central fact and belongs in its own description, not bolted
            // on per request. What still has to be said here is that SKILL SCRIPTS share
            // that directory, which the shell's declaration cannot know.
            ToolFunction scriptTool = tools.FirstOrDefault(t =>
                string.Equals(t?.Name, SkillTools.RunToolName, StringComparison.Ordinal));
            if (scriptTool == null)
                return;

            scriptTool.Description +=
                " Scripts run in this conversation's persistent working directory, the same one the "
                + SkillToolNames.Shell + " tool works in: files produced earlier are available by name.";

            // Dependencies take care of themselves on this host, and the model should
            // know that BEFORE it spends a round installing things by hand.
            if (codeRunner is { CanInstallPackages: true })
            {
                scriptTool.Description +=
                    " A script's missing Python/JavaScript dependencies are installed automatically "
                    + "and the script re-run - just run it. You may also name dependencies up front "
                    + "in 'packages' to skip the first failed attempt.";

                scriptTool.Parameters ??= new Dictionary<string, ToolParameter>();
                if (!scriptTool.Parameters.ContainsKey("packages"))
                {
                    scriptTool.Parameters["packages"] = new ToolParameter
                    {
                        Type = "string",
                        Description =
                            "Optional: packages to install into the session environment before the "
                            + "script runs, separated by commas - for example \"defusedxml, lxml\". "
                            + "Omit it to rely on automatic installation of whatever the script "
                            + "fails to import.",
                    };
                }
            }
        }

        /// <summary>
        /// Splice the code-execution tools into a tool list, without disturbing what is
        /// there. Both of them: the program runner, and the delta editor that fixes the
        /// last program without resending it.
        /// </summary>
        /// <summary>
        /// The editing rules as a plan, derived from what was actually declared.
        ///
        /// <para>
        /// Read off the finished tool list rather than from the options that produced it,
        /// so the block can never name a tool the model was not given — the one failure
        /// worse than saying nothing, because a model cannot tell an inapplicable
        /// instruction from its own misreading of its tool list.
        /// </para>
        /// </summary>
        private static SkillPlan CodePlan(IReadOnlyList<ToolFunction> tools)
        {
            bool Declared(string name) =>
                tools.Any(t => string.Equals(t?.Name, name, StringComparison.Ordinal));

            string block = CodePrompt.Block(
                fileTools: Declared(SkillToolNames.EditFile) && Declared(SkillToolNames.ReadFile),
                hasPatch: Declared(SkillToolNames.ApplyPatch));

            return block.Length == 0
                ? SkillPlan.Empty
                : SkillPlan.Empty with { Instructions = block, ApproximateTokens = block.Length / 4 };
        }

        private static List<ToolFunction> AppendCodeTool(
            List<ToolFunction>? tools, ICodeRunner runner, IReadOnlyList<CodeInputFile>? inputFiles = null,
            SessionWorkspace? workspace = null, int reachableSkills = 0)
        {
            var merged = tools != null ? new List<ToolFunction>(tools) : new List<ToolFunction>();
            // Whether anything the model does survives between calls on this request. It
            // decides both how the shell is described and whether apply_patch is offered
            // at all, so it is one value read once.
            bool persists = workspace != null;
            IReadOnlyList<ToolFunction> declarations = runner.DeclareTools(persists);
            if (declarations.Count == 0)
                return merged;

            // BY NAME, never by index. Everything below patches this declaration's
            // description — the conversation's attachments, the skills on the module
            // path — and it was written as declarations[0] back when the shell was the
            // first thing declared. The moment read_file/edit_file/write_file went in
            // front of it, every attachment note and skill-import note would have landed
            // on read_file's description instead: it compiles, and the only place it
            // shows up is in what the model was told.
            ToolFunction? shell = declarations.FirstOrDefault(
                d => string.Equals(d?.Name, SkillToolNames.Shell, StringComparison.Ordinal));
            if (shell == null)
                return Merge(merged, declarations, persists);

            // Without a session workspace — every stateless API surface: /v1/chat/completions,
            // /v1/responses, Ollama — each command gets a throwaway directory that is deleted
            // when the call returns. The declaration is written for the persistent case and
            // says so at length, and a model that believes it writes a file in one call and
            // reads it back in the next, finds nothing, and concludes its own program is
            // broken. Say what is actually true here, and do not offer apply_patch at all:
            // it needs a workspace and would refuse every call.
            // The declaration is BUILT for this case rather than edited afterwards — see
            // ShellTools.DeclareShell's `persists` parameter for what patching the prose
            // produced.

            // The attachments are per request, so the declaration learns about them here
            // rather than in the runner. Named explicitly - "this file exists, open it" -
            // because a model that only saw the attachment's CONTENT inlined into the
            // conversation otherwise re-types that content into its program, abridged.
            // Repeated on the 'command' parameter: that description is what the model is
            // reading while it writes the command, and observed on gemma-4-E4B, a note
            // only at the tail of the tool description was not enough to stop the
            // re-typing.
            // A skill's bundled package is on the module search path, which the model
            // cannot discover by trying: observed spending a thousand seconds copying
            // slack-gif-creator's core/ package into its workspace file by file because
            // nothing said `import core.gif` would simply work.
            // Only when a skill is actually reachable. CodeOnly requests carry an empty
            // skill set, so SkillTools.ExecuteCode passes no skill directories and nothing
            // is on the module path — the promise was false for every code-only request,
            // and it is the kind of promise a model spends a round acting on.
            if (reachableSkills > 0)
            {
                shell.Description +=
                    " The bundled Python modules of the skills available here can be imported directly by "
                    + "a program you write (both a skill's own directory and its scripts/ folder are on "
                    + "the module path), so use them rather than copying their code.";
            }

            if (inputFiles is { Count: > 0 })
            {
                string names = string.Join(", ", inputFiles.Select(f => "'" + f.Name + "'"));
                shell.Description +=
                    " The user's attached files are already in the working directory - open them by these exact names: "
                    + names + ".";

                if (shell.Parameters != null
                    && shell.Parameters.TryGetValue("command", out ToolParameter commandParameter))
                {
                    commandParameter.Description +=
                        " The user's attached files are in the working directory: read "
                        + names + " from there instead of pasting their content into a command.";
                }
            }

            // A caller's own tool of the same name wins: it is theirs, they can service it,
            // and quietly shadowing it would break their request in order to add ours.
            // Shadowing is decided PER NAME. An early return on a clash with the first
            // declaration - as this once did - also withheld every other one, none of
            // which the caller had asked for: a client that happened to own a tool called
            // run_code lost four unrelated ones, silently.
            return Merge(merged, declarations, persists);
        }

        /// <summary>
        /// Add each declaration the caller does not already own.
        ///
        /// <para>
        /// The workspace-required tools are no longer filtered here by name. The RUNNER
        /// decides what a non-persistent endpoint gets, because it is the half that knows
        /// which of its tools need a workspace — a name-by-name skip in the caller could
        /// not survive a fourth such tool, and three were about to be added.
        /// </para>
        /// </summary>
        private static List<ToolFunction> Merge(
            List<ToolFunction> merged, IReadOnlyList<ToolFunction> declarations, bool persists)
        {
            foreach (ToolFunction declaration in declarations)
            {
                if (!merged.Any(t => string.Equals(t?.Name, declaration.Name, StringComparison.OrdinalIgnoreCase)))
                    merged.Add(declaration);
            }
            return merged;
        }

        /// <summary>
        /// Families that already got the shell-suppressed warning. Keyed by
        /// architecture so a model switch onto another declaration-less family reports
        /// again, while a busy server says it once rather than per request.
        /// </summary>
        private static readonly HashSet<string> WarnedCodeUnrenderable = new(StringComparer.OrdinalIgnoreCase);

        private static void WarnCodeToolUnrenderable(ILogger? logger, string architecture)
        {
            if (logger == null)
                return;
            lock (WarnedCodeUnrenderable)
            {
                if (!WarnedCodeUnrenderable.Add(architecture ?? string.Empty))
                    return;
            }
            logger.LogWarning(LogEventIds.HostConfiguration,
                "--code-exec is on, but the '{Architecture}' model family cannot carry tool declarations, " +
                "so the shell tool is NOT offered on its requests: the model answers without executing code. " +
                "Load a tool-capable model to use code execution. Reported once.",
                string.IsNullOrEmpty(architecture) ? "(unknown)" : architecture);
        }

        /// <summary>
        /// Put the instruction block in front of <paramref name="messages"/>, returning
        /// a new list. Composes with <see cref="StructuredOutputPrompt.Apply"/>: both
        /// merge into the leading system message, and both leave their input alone.
        /// </summary>
        public List<ChatMessage> Apply(List<ChatMessage> messages) => SkillPrompt.Apply(messages, Prompt);
    }
}
