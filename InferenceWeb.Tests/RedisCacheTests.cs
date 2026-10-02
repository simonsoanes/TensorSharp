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
using System.Text;
using System.Threading;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Redis;
using TensorSharp.Server.Responses;

namespace InferenceWeb.Tests;

/// <summary>
/// In-memory fake of <see cref="IRedisKeyValueStore"/> for unit-testing the
/// Redis-backed tiers without a live Redis server.
/// </summary>
internal sealed class FakeRedisKeyValueStore : IRedisKeyValueStore
{
    private readonly Dictionary<string, byte[]> _store = new();
    private readonly object _gate = new();

    public bool IsConnected => true;
    public int SetCount { get; private set; }
    public int GetCount { get; private set; }

    public bool StringSet(string key, byte[] value, TimeSpan? expiry)
    {
        lock (_gate)
        {
            _store[key] = value;
            SetCount++;
            return true;
        }
    }

    public byte[] StringGet(string key)
    {
        lock (_gate)
        {
            GetCount++;
            return _store.TryGetValue(key, out var value) ? value : null;
        }
    }

    public void KeyDelete(string[] keys)
    {
        lock (_gate)
        {
            foreach (var key in keys)
                _store.Remove(key);
        }
    }

    public string[] ScanKeys(string pattern)
    {
        lock (_gate)
        {
            // Simple prefix match: strip trailing '*'
            string prefix = pattern.TrimEnd('*');
            return _store.Keys.Where(k => k.StartsWith(prefix, StringComparison.Ordinal)).ToArray();
        }
    }

    public int Count
    {
        get { lock (_gate) return _store.Count; }
    }

    public void Dispose() { }
}

public class RedisResponsesStoreTests
{
    [Fact]
    public void Store_And_TryGet_RoundTrips()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        var response = new StoredResponse
        {
            Id = "resp_abc123",
            Json = """{"id":"resp_abc123","output":[]}""",
        };
        store.Store(response);

        Assert.True(store.TryGet("resp_abc123", out var retrieved));
        Assert.Equal("resp_abc123", retrieved.Id);
        Assert.Equal(response.Json, retrieved.Json);
    }

    [Fact]
    public void TryGet_MissingKey_ReturnsFalse()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        Assert.False(store.TryGet("resp_nonexistent", out var retrieved));
        Assert.Null(retrieved);
    }

    [Fact]
    public void Store_OverwritesExistingEntry()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        store.Store(new StoredResponse { Id = "resp_1", Json = """{"v":1}""" });
        store.Store(new StoredResponse { Id = "resp_1", Json = """{"v":2}""" });

        Assert.True(store.TryGet("resp_1", out var retrieved));
        Assert.Equal("""{"v":2}""", retrieved.Json);
    }

    [Fact]
    public void Store_UsesCorrectKeyPrefix()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        store.Store(new StoredResponse { Id = "resp_xyz", Json = "{}" });

        Assert.Equal(1, fake.Count);
        // The key should be prefixed with "tsresp:"
        Assert.True(fake.StringGet("tsresp:resp_xyz") != null);
    }

    [Fact]
    public void Store_MultipleEntries_AllRetrievable()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        for (int i = 0; i < 10; i++)
            store.Store(new StoredResponse { Id = $"resp_{i}", Json = $$"""{"i":{{i}}}""" });

        for (int i = 0; i < 10; i++)
        {
            Assert.True(store.TryGet($"resp_{i}", out var retrieved));
            Assert.Equal($$"""{"i":{{i}}}""", retrieved.Json);
        }
    }
}
