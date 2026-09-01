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
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// Binds <see cref="ShellRunner"/> to the tool interface every host talks to.
    ///
    /// <para>
    /// The seam exists so that the skills namespace — and through it the servers and the
    /// CLI — knows only <see cref="ICodeRunner"/>, and code execution knows about skills
    /// rather than the other way round. Without it the two would reference each other and
    /// a host that only wanted skills would drag the whole execution stack in with them.
    /// </para>
    /// </summary>
    public sealed class CodeRunnerAdapter : ICodeRunner
    {
        private readonly ShellRunner _runner;
        private readonly CodeExecOptions _options;
        private readonly Action<CodeExecResult>? _onCompleted;

        /// <param name="runner">The engine.</param>
        /// <param name="options">The host's terms, for the declaration.</param>
        /// <param name="onCompleted">
        /// Observer for each finished call, so a host can record what a command produced
        /// without parsing the model's prose about it.
        /// </param>
        public CodeRunnerAdapter(
            ShellRunner runner,
            CodeExecOptions? options = null,
            Action<CodeExecResult>? onCompleted = null)
        {
            _runner = runner ?? throw new ArgumentNullException(nameof(runner));
            _options = options ?? runner.Options;
            _onCompleted = onCompleted;
        }

        /// <inheritdoc/>
        public bool CanRun => _runner.CanRun;

        /// <inheritdoc/>
        public string? UnavailableReason => _runner.UnavailableReason;

        /// <inheritdoc/>
        public SamplingConfig ForCodingTurn(SamplingConfig requested) =>
            // Two different operations, and only one of them is opt-in.
            //
            // REMOVING the repetition penalty is on by default: it is an Ollama
            // chat-compatibility default that neither reference has any analogue of —
            // Codex sends no penalty at all — and on code it penalises the indentation,
            // the `return` and the closing delimiters against each other. Taking it off
            // for coding turns moves toward both references, not away.
            //
            // SETTING a temperature is opt-in, because that would ADD something neither
            // reference sets: Codex leaves it None and omits it from the wire, and Claude
            // Code exposes no sampling setting at all.
            requested == null || !CanRun ? requested : requested.ForCodingTurn(_options.Temperature);

        /// <inheritdoc/>
        public ToolFunction Declare()
        {
            // BY NAME, never by index. This returned declarations[0] until the file tools
            // were added in front of the shell, at which point "the tool" silently became
            // read_file — the kind of change that compiles, passes a set-equality test,
            // and is only visible in what the model was told. The same hazard is fixed in
            // SkillRequestPlan, which patched declarations[0]'s description to mention the
            // conversation's attachments.
            IReadOnlyList<ToolFunction> declarations = DeclareTools();
            foreach (ToolFunction declaration in declarations)
            {
                if (string.Equals(declaration.Name, ShellTools.ShellToolName, StringComparison.Ordinal))
                    return declaration;
            }
            return declarations.Count > 0 ? declarations[0] : new ToolFunction();
        }

        /// <inheritdoc/>
        public IReadOnlyList<ToolFunction> DeclareTools() => DeclareTools(persists: true);

        /// <inheritdoc/>
        public IReadOnlyList<ToolFunction> DeclareTools(bool persists)
        {
            if (_runner.Shell is not { } shell)
            {
                // Declaring a tool this host cannot answer is strictly worse than staying
                // quiet: the model emits the call, nothing services it, and the raw tool
                // markup reaches the user as the answer.
                return Array.Empty<ToolFunction>();
            }

            // The file tools need a workspace: they read and write files that have to
            // outlive the call, and the read ledger that authorises an edit lives on the
            // session. A stateless endpoint gets a fresh empty directory per call, so
            // there is nothing there to read and nothing that would survive being
            // written — declaring them would offer capability the host cannot honour.
            if (!persists)
            {
                return new[]
                {
                    ShellTools.DeclareShell(
                        _options, shell, _runner.KeepsArtifacts, persists: false, fileTools: false,
                        networkConfinementGuaranteed: _runner.NetworkConfinementGuaranteed),
                };
            }

            // ORDER MATTERS, and this is the order. read_file and edit_file come first
            // because a declaration list is read top-down and the measured failure is a
            // model reaching for the shell to do a job the editor does better; apply_patch
            // stays last, now correctly, as the specialist for a change that spans several
            // files atomically.
            return new[]
            {
                ShellTools.DeclareRead(),
                ShellTools.DeclareEdit(),
                ShellTools.DeclareWrite(),
                ShellTools.DeclareShell(
                    _options, shell, _runner.KeepsArtifacts, persists, fileTools: true,
                    networkConfinementGuaranteed: _runner.NetworkConfinementGuaranteed),
                ShellTools.DeclarePatch(),
            };
        }

        /// <inheritdoc/>
        public SkillToolResult Execute(
            ToolCall call,
            IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null,
            SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null)
        {
            if (!_runner.CanRun)
                return SkillToolResult.Failure(_runner.UnavailableReason ?? "code execution is unavailable");

            // Patching is a workspace operation, not a run: it returns without entering
            // the execution pipeline at all, and nothing is launched.
            if (ShellTools.IsPatchTool(call?.Name))
            {
                if (!ShellTools.TryReadPatch(call!, out string patch, out string? patchError))
                    return SkillToolResult.Failure(patchError!);
                return Finish(_runner.ApplyPatch(patch, workspace));
            }

            // The file tools, likewise: nothing is launched, no sandbox is entered, and
            // the host places the bytes. Dispatched through the shared resolver so an
            // invented spelling costs nothing — a round spent on `str_replace` instead of
            // `edit_file` teaches the model nothing and fixes nothing.
            switch (ShellTools.ResolveFileTool(call?.Name))
            {
                case SkillToolNames.ReadFile:
                    if (!ShellTools.TryReadRead(call!, out ShellTools.ReadRequest read, out string? readError))
                        return SkillToolResult.Failure(readError!);
                    return Finish(_runner.ReadFile(read, workspace));

                case SkillToolNames.EditFile:
                    if (!ShellTools.TryReadEdit(call!, out ShellTools.EditRequest edit, out string? editError))
                        return SkillToolResult.Failure(editError!);
                    return Finish(_runner.EditFile(edit, workspace));

                case SkillToolNames.WriteFile:
                    if (!ShellTools.TryReadWrite(call!, out ShellTools.WriteRequest write, out string? writeError))
                        return SkillToolResult.Failure(writeError!);
                    return Finish(_runner.WriteFile(write, workspace));
            }

            if (!ShellTools.TryReadShell(call!, out ShellRequest request, out string? error))
                return SkillToolResult.Failure(error!);

            if (workspace != null)
                StageInputFiles(inputFiles, workspace);

            CodeExecResult result = _runner.Run(
                request with { ReadablePaths = skillDirectories ?? Array.Empty<string>() },
                workspace,
                onOutput);

            return Finish(result);
        }

        /// <summary>
        /// Turn an engine result into a tool result.
        ///
        /// <para>
        /// A failure carries the FULL output with no "Error:" prefix, because the output
        /// is the useful part — the model is about to fix its own command with it. The
        /// produced files ride along structurally as well as in the text, so a host's UI
        /// can offer the downloads itself whatever the model's answer ends up saying
        /// about them.
        /// </para>
        /// </summary>
        private SkillToolResult Finish(CodeExecResult result)
        {
            _onCompleted?.Invoke(result);

            var files = new List<SkillProducedFile>(result.Artifacts.Count);
            foreach (CodeArtifact artifact in result.Artifacts)
                files.Add(new SkillProducedFile(artifact.Path, artifact.Bytes, artifact.Pointer));

            return new SkillToolResult(result.Ok, result.Content, null, null) { Files = files };
        }

        /// <summary>
        /// Put the conversation's attachments in the working directory under the names
        /// the model was told.
        ///
        /// <para>
        /// An attachment's inlined CONTENT is not a file on disk: asked to "convert this
        /// md file", a model with only the inline copy re-types it into its program,
        /// abridged. The real file has to be there under its display name.
        /// </para>
        /// <para>
        /// In a persistent workspace the file may already be there from an earlier call —
        /// possibly EDITED since. Re-copying would silently revert that work, so an
        /// up-to-date copy stands and only a genuinely newer source replaces it.
        /// </para>
        /// </summary>
        private static void StageInputFiles(IReadOnlyList<CodeInputFile>? inputFiles, SessionWorkspace workspace)
        {
            if (inputFiles == null || inputFiles.Count == 0)
                return;

            foreach (CodeInputFile input in inputFiles)
            {
                // The name is what the model was told; the flattening keeps a name like
                // "../x" from writing outside the directory the sandbox will confine.
                string name = Path.GetFileName(input.Name ?? string.Empty);
                if (name.Length == 0 || string.IsNullOrEmpty(input.SourcePath))
                    continue;

                string destination = Path.Combine(workspace.WorkDirectory, name);
                if (!SkillPathGuard.IsUnder(workspace.WorkDirectory, destination))
                    continue;

                try
                {
                    var existing = new FileInfo(destination);
                    var source = new FileInfo(input.SourcePath);
                    if (!source.Exists)
                        continue;
                    if (existing.Exists && existing.LastWriteTimeUtc >= source.LastWriteTimeUtc)
                        continue;
                    File.Copy(input.SourcePath, destination, overwrite: true);

                    // The HOST just replaced a file behind the model's back — not the
                    // model, and not anything it ran. "Stale" would overstate what it
                    // knows, because it never saw this happen at all, so the path is
                    // dropped entirely and the next edit is checked from scratch.
                    workspace.Reads.Forget(destination);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    // Best effort: a file that cannot be staged is reported by the command
                    // that goes looking for it, which is a better place to learn about it
                    // than a note attached to an unrelated call.
                }
            }
        }

        /// <inheritdoc/>
        public bool CanInstallPackages => _options.AllowInstall;

        /// <inheritdoc/>
        public string? InstallPackages(
            string language, IReadOnlyList<string> packages,
            SessionWorkspace workspace, Action<string>? onOutput = null)
        {
            return _runner.Installer.Install(
                workspace, CodeExecOptions.ParseLanguage(language), packages, onOutput);
        }
    }
}
