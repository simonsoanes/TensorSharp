// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using System.Text.Json.Serialization;
using TensorAgent.Core.Localization;

namespace TensorAgent.Core.Sessions;

/// <summary>
/// Persists conversations as one JSON file each under a root directory, plus a small
/// index so the sessions list opens without parsing every file.
///
/// <para>
/// Uploads referenced by messages live under the app's upload root (the same directory
/// the loopback server serves as <c>/uploads/</c>); <see cref="ReferencedUploads"/> lets
/// the app's cleanup sweep keep every file a saved conversation still points at.
/// </para>
/// </summary>
public sealed class ConversationStore
{
    /// <summary>
    /// The serializer the saved transcript uses. Public because the recorder reads the
    /// page's own message objects with it: the two must agree exactly, or a message
    /// saved by one and read by the other loses its attachments.
    /// </summary>
    internal static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private readonly object _lock = new();
    private readonly Dictionary<string, ConversationSummary> _index = new(StringComparer.Ordinal);
    /// <summary>The indexed files that could not be read, whose rows have no title of their own.</summary>
    private readonly HashSet<string> _unreadable = new(StringComparer.Ordinal);
    private bool _indexed;

    public string Root { get; }

    public ConversationStore(string root)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        Root = Path.GetFullPath(root);
        Directory.CreateDirectory(Root);
    }

    private string PathFor(string id) => Path.Combine(Root, id + ".json");

    /// <summary>
    /// The saved chats, newest first, omitting any that never got a message.
    ///
    /// <para>
    /// The page creates a session on every load, and a session binds a conversation,
    /// so an app that is opened and closed without anything being typed would leave a
    /// row behind each time. Twenty launches, twenty "Chat Sep 2, 07:38" entries and
    /// the real conversations pushed off the screen. An empty one is not a chat the
    /// user had; it is a chat they were about to have.
    /// </para>
    /// </summary>
    /// <param name="includeEmpty">Include conversations with no messages. For tests and diagnostics.</param>
    public IReadOnlyList<ConversationSummary> List(bool includeEmpty = false)
    {
        lock (_lock)
        {
            EnsureIndex();
            IEnumerable<ConversationSummary> rows = _index.Values;
            if (!includeEmpty)
                rows = rows.Where(s => s.MessageCount > 0);
            return rows.OrderByDescending(s => s.UpdatedAt).Select(Shown).ToList();
        }
    }

    /// <summary>
    /// A row as the lists show it. The titles the store writes itself -- a chat's date, and a
    /// file it cannot read -- read in the interface language, and are never stored in it: a
    /// saved chat is the same file whichever language it was saved under.
    /// </summary>
    private ConversationSummary Shown(ConversationSummary row)
    {
        string title = _unreadable.Contains(row.Id)
            ? Loc.T("host.conversations.unreadable")
            : Conversation.DisplayTitle(row.Title, row.CreatedAt);
        return ReferenceEquals(title, row.Title) ? row : row with { Title = title };
    }

    /// <summary>
    /// The most recent conversation that has no messages, or null.
    ///
    /// <para>
    /// Handing this back to the next session instead of minting another is what stops
    /// the empty rows accumulating in the first place: there is at most one at a time,
    /// and the moment it gets a message the next launch starts a new one.
    /// </para>
    /// </summary>
    public Conversation? MostRecentEmpty()
    {
        lock (_lock)
        {
            EnsureIndex();
            ConversationSummary? candidate = _index.Values
                .Where(s => s.MessageCount == 0)
                .OrderByDescending(s => s.UpdatedAt)
                .FirstOrDefault();
            return candidate is null ? null : Load(candidate.Id);
        }
    }

    public Conversation? Load(string id)
    {
        if (!IsValidId(id))
            return null;
        string path = PathFor(id);
        if (!File.Exists(path))
            return null;
        try
        {
            using FileStream stream = File.OpenRead(path);
            return JsonSerializer.Deserialize<Conversation>(stream, Json);
        }
        catch (JsonException)
        {
            // A truncated write (the app was killed mid-save) must not brick the list;
            // the row stays but opens empty and the next save repairs the file.
            return null;
        }
    }

    public Conversation Create(string? modelId = null, bool think = false, IEnumerable<string>? skills = null)
    {
        var conversation = new Conversation { ModelId = modelId, Think = think };
        if (skills is not null)
        {
            conversation.Skills.AddRange(skills);
            conversation.SkillsExplicit = true;
        }
        conversation.Title = Conversation.DeriveTitle(conversation.Messages, conversation.CreatedAt);
        Save(conversation);
        return conversation;
    }

    /// <summary>Writes atomically (temp file + rename) and refreshes the title/index.</summary>
    public void Save(Conversation conversation)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        if (!IsValidId(conversation.Id))
            throw new ArgumentException("conversation id must be a hex GUID", nameof(conversation));

        conversation.UpdatedAt = DateTimeOffset.UtcNow;
        if (string.IsNullOrWhiteSpace(conversation.Title) || conversation.Title.StartsWith("Chat ", StringComparison.Ordinal))
            conversation.Title = Conversation.DeriveTitle(conversation.Messages, conversation.CreatedAt);

        string path = PathFor(conversation.Id);
        string tmp = path + ".tmp";
        lock (_lock)
        {
            using (FileStream stream = File.Create(tmp))
                JsonSerializer.Serialize(stream, conversation, Json);
            File.Move(tmp, path, overwrite: true);
            EnsureIndex();
            _index[conversation.Id] = Summarize(conversation);
            _unreadable.Remove(conversation.Id);
        }
    }

    public bool Delete(string id)
    {
        if (!IsValidId(id))
            return false;
        lock (_lock)
        {
            EnsureIndex();
            _index.Remove(id);
            _unreadable.Remove(id);
            string path = PathFor(id);
            if (!File.Exists(path))
                return false;
            File.Delete(path);
            return true;
        }
    }

    public bool Rename(string id, string title)
    {
        Conversation? c = Load(id);
        if (c is null)
            return false;
        c.Title = string.IsNullOrWhiteSpace(title) ? Conversation.DeriveTitle(c.Messages, c.CreatedAt) : title.Trim();
        Save(c);
        return true;
    }

    /// <summary>Every upload file name any saved conversation still references.</summary>
    public HashSet<string> ReferencedUploads()
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (ConversationSummary s in List())
        {
            Conversation? c = Load(s.Id);
            if (c is null) continue;
            foreach (StoredMessage m in c.Messages)
                foreach (string f in m.ReferencedUploads)
                    set.Add(f);
        }
        return set;
    }

    private void EnsureIndex()
    {
        if (_indexed)
            return;
        _indexed = true;
        foreach (string file in Directory.EnumerateFiles(Root, "*.json"))
        {
            string id = Path.GetFileNameWithoutExtension(file);
            if (!IsValidId(id))
                continue;
            Conversation? c = Load(id);
            if (c is null)
            {
                // Titled where it is shown (Shown), in the interface language.
                var info = new FileInfo(file);
                _index[id] = new ConversationSummary(id, string.Empty, info.CreationTimeUtc, info.LastWriteTimeUtc, null, 0);
                _unreadable.Add(id);
                continue;
            }
            _index[id] = Summarize(c);
        }
    }

    private static ConversationSummary Summarize(Conversation c) =>
        new(c.Id, c.Title, c.CreatedAt, c.UpdatedAt, c.ModelId, c.Messages.Count);

    /// <summary>Ids are 32 hex characters (Guid "N"), which also keeps them path-safe.</summary>
    public static bool IsValidId(string? id) =>
        id is { Length: 32 } && id.All(ch => ch is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F');
}
