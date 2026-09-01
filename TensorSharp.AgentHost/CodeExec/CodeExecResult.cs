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

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// What one call did: what the model is told, and what the host's own UI needs.
    /// </summary>
    /// <param name="Ok">
    /// Whether it succeeded. A false here still carries the full output rather than an
    /// error summary — the output is what the model fixes the command with.
    /// </param>
    /// <param name="Content">The tool result text.</param>
    /// <param name="Artifacts">
    /// Files the command wrote that were kept and can be fetched. Empty when it wrote
    /// none, or when the host keeps none. Carried structurally as well as named in
    /// <paramref name="Content"/>, because a user's download must never depend on the
    /// model remembering to repeat a link.
    /// </param>
    /// <param name="RunId">Identifies the artifact directory this call owns, when it has one.</param>
    public readonly record struct CodeExecResult(
        bool Ok, string Content, IReadOnlyList<CodeArtifact> Artifacts, string RunId)
    {
        /// <summary>Nothing was run, and the model is told plainly why.</summary>
        public static CodeExecResult Refused(string reason) =>
            new(false, "The command was not run: " + reason, Array.Empty<CodeArtifact>(), string.Empty);

        /// <summary>
        /// Nothing was WRITTEN, and the model is told plainly why.
        ///
        /// <para>
        /// A separate opening from <see cref="Refused"/> because the file tools are not
        /// commands, and saying "The command was not run" about an <c>edit_file</c> call
        /// costs the model a round working out which of its tools that sentence is
        /// describing. With five code tools instead of two, mis-framing which one refused
        /// is a real cost; the reference implementation frames its own the same way
        /// ("apply_patch verification failed").
        /// </para>
        /// <para>
        /// It also states the thing a model most needs to know before it decides what to
        /// do next: the file is exactly as it was, so there is nothing to undo and no
        /// half-applied change to reason about — which is precisely the belief that
        /// otherwise leads to rewriting the file from scratch.
        /// </para>
        /// </summary>
        public static CodeExecResult NoChange(string reason) =>
            new(false, "No change was made: " + reason, Array.Empty<CodeArtifact>(), string.Empty);
    }
}
