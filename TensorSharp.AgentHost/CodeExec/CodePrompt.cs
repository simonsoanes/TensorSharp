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
using System.Text;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// The six rules about editing that have to be known BEFORE the first tool call, put
    /// in the system prompt.
    ///
    /// <para>
    /// <b>This channel had never been used.</b> Nothing anywhere in this host's system
    /// prompt said a word about files, editing, patching or rewriting: the whole surface
    /// was three constants about skills, and the code-only request path was documented as
    /// deliberately empty — "there is no instruction block here because there is nothing
    /// to disclose: the tool declaration says everything the model needs". That was a bet,
    /// it was never tested, and the measurement is in <see cref="RewriteWatch"/>'s own
    /// docstring: on runs where the patch tool was declared with an emphatic
    /// prefer-a-patch paragraph, models used it zero times and re-typed whole files
    /// instead. Both reference implementations use the prompt channel heavily. This is the
    /// experiment that was skipped.
    /// </para>
    /// <para>
    /// <b>Six lines, and no syntax teaching.</b> Everything about HOW to call a tool lives
    /// in that tool's declaration, and everything about recovering from a failure is
    /// attached to the failing result — which is where this codebase has its one existence
    /// proof that guidance changes behaviour. What is left is only what must be known
    /// before there is any result to attach anything to: which tool to reach for, and what
    /// not to do.
    /// </para>
    /// <para>
    /// <b>Every byte is a pure function of the options.</b> No timestamps, no paths, no
    /// counters, nothing per-conversation. This block sits at the FRONT of the prompt, and
    /// the KV prefix cache chains its hashes from block zero and stops adopting at the
    /// first mismatch — so a single varying byte here drops prefix reuse to zero for the
    /// whole conversation, not just for the part that changed, and it would never show up
    /// as a test failure. It is pinned by a test for exactly that reason.
    /// </para>
    /// </summary>
    public static class CodePrompt
    {
        /// <summary>The heading the block opens with. Also how tests find it.</summary>
        public const string Heading = "## Working with files";

        /// <summary>
        /// The block, or the empty string when this host has no file tools to talk about.
        /// </summary>
        /// <param name="fileTools">
        /// Whether <c>read_file</c>, <c>edit_file</c> and <c>write_file</c> are declared.
        /// A caller with no workspace has none of them, and telling such a model to "use
        /// edit_file" names a tool it was never given — the one failure worse than saying
        /// nothing, because the model cannot tell that the instruction is inapplicable
        /// rather than that it has misread its own tool list.
        /// </param>
        /// <param name="hasPatch">Whether <c>apply_patch</c> is declared.</param>
        public static string Block(bool fileTools, bool hasPatch)
        {
            if (!fileTools)
                return string.Empty;

            var sb = new StringBuilder();
            sb.Append(Heading).Append('\n');

            sb.Append("- To change part of a file that already exists, use `")
              .Append(ShellTools.EditToolName)
              .Append("`. Never rewrite a whole file to change part of it: it costs you every line "
                    + "that was already right, re-rolls each one, and is how a second bug appears in "
                    + "code that worked.\n");

            sb.Append("- Use `").Append(ShellTools.WriteToolName)
              .Append("` only to create a new file, or when a file genuinely should be thrown away "
                    + "and written again.\n");

            if (hasPatch)
            {
                sb.Append("- Use `").Append(ShellTools.PatchToolName)
                  .Append("` when one change spans several files and all of it has to land or none "
                        + "of it.\n");
            }

            sb.Append("- Read a file with `").Append(ShellTools.ReadToolName)
              .Append("` before an edit that depends on lines you cannot already see. An edit has to "
                    + "match the file exactly, so copy the text out of what you read rather than "
                    + "recalling it.\n");

            sb.Append("- After an edit or a write reports success, do not read the file back to check "
                    + "it. The result is authoritative — if it had not applied, it would have said so.\n");

            sb.Append("- Search with `rg` (or `grep -rn`) through the `")
              .Append(ShellTools.ShellToolName)
              .Append("`; use the shell to run programs, and to move, copy and delete files.\n");

            return sb.ToString();
        }
    }
}
