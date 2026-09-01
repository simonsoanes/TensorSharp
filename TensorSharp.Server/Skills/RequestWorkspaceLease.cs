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
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.Server.Skills
{
    /// <summary>
    /// Gives one stateless HTTP request a private persistent workspace for all of its
    /// internally serviced tool rounds, then releases it with the request.
    /// </summary>
    /// <remarks>
    /// The protocol adapters are singletons, so the workspace must never live on an
    /// adapter field. A unique manager key keeps concurrent requests isolated, while a
    /// method-scope <c>using</c> keeps the directory alive through both buffered and
    /// streaming response writers and releases it on cancellation or failure.
    /// </remarks>
    internal sealed class RequestWorkspaceLease : IDisposable
    {
        private SessionWorkspaceManager _manager;

        private RequestWorkspaceLease(SessionWorkspaceManager manager, SessionWorkspace workspace)
        {
            _manager = manager;
            Workspace = workspace;
        }

        /// <summary>The request's private workspace.</summary>
        public SessionWorkspace Workspace { get; }

        /// <summary>
        /// Acquire a workspace only when this request can actually be offered built-in
        /// code tools. Other requests retain their existing allocation-free path.
        /// </summary>
        public static RequestWorkspaceLease Acquire(
            SessionWorkspaceManager manager,
            ICodeRunner codeRunner,
            string architecture,
            bool allowTools = true)
        {
            if (manager == null || codeRunner is not { CanRun: true } || !allowTools
                || !SkillCapabilities.For(architecture).ToolsRendered)
            {
                return null;
            }

            string id = "request-" + Guid.NewGuid().ToString("N");
            return new RequestWorkspaceLease(manager, manager.GetOrCreate(id));
        }

        public void Dispose()
        {
            SessionWorkspaceManager manager = System.Threading.Interlocked.Exchange(ref _manager, null);
            if (manager != null)
                manager.Release(Workspace.Id);
        }
    }
}
