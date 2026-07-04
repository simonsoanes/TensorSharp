// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Concurrent;

namespace TensorSharp.Server.Responses
{
    /// <summary>
    /// Process-lifetime <see cref="IResponsesStore"/>. Responses vanish on
    /// restart and there is no cross-request conversation chaining
    /// (<c>previous_response_id</c> is rejected by the adapter); this only
    /// backs <c>GET /v1/responses/{id}</c> for as long as the server runs.
    /// </summary>
    internal sealed class InMemoryResponsesStore : IResponsesStore
    {
        private readonly ConcurrentDictionary<string, StoredResponse> _responses = new();

        public void Store(StoredResponse response)
        {
            _responses[response.Id] = response;
        }

        public bool TryGet(string id, out StoredResponse response)
        {
            return _responses.TryGetValue(id, out response);
        }
    }
}
