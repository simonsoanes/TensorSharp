// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    private const string DraftStorage = """
        document.getElementById('text').value = '';
        var storedDrafts = {};
        window.sessionStorage = {
          setItem: function (key, value) { storedDrafts[key] = value; },
          getItem: function (key) { return storedDrafts[key] || null; },
          removeItem: function (key) { delete storedDrafts[key]; }
        };
        """;

    [WebJavaScriptFact]
    public void SwitchingLanguageKeepsTheUnsentTextAttachmentsAndImageSelection()
    {
        JsonElement captured = Run(DraftStorage, """
            window.TensorAgent.insertText("用户的草稿\nL'utilisateur's draft");
            window.TensorAgent.addAttachment({ ok: true, file: 'photo.png', mediaType: 'image', maskPath: 'selection.png', maskCrop: true, maskFeather: 5 });
            window.TensorAgent.setSkills(['documents']);
            window.TensorAgent.setThink(true);
            var kept = window.TensorAgent.prepareLanguageReload();
            return { kept: kept, draft: JSON.parse(storedDrafts['tensoragent-language-draft']) };
            """);
        Assert.True(captured.GetProperty("kept").GetBoolean());
        string draft = captured.GetProperty("draft").GetRawText();

        JsonElement restored = Run(DraftStorage + "\nstoredDrafts['tensoragent-language-draft'] = " + JsonSerializer.Serialize(draft) + ";", """
            return {
              text: __page.byId['text'].value,
              count: window.TensorAgent.attachmentCount(),
              cleared: !storedDrafts['tensoragent-language-draft'],
              captured: window.TensorAgent.prepareLanguageReload() && JSON.parse(storedDrafts['tensoragent-language-draft']),
              sends: __page.requests('/api/chat').length
            };
            """);
        Assert.Equal("用户的草稿\nL'utilisateur's draft", restored.GetProperty("text").GetString());
        Assert.Equal(1, restored.GetProperty("count").GetInt32());
        Assert.True(restored.GetProperty("cleared").GetBoolean());
        Assert.Equal(0, restored.GetProperty("sends").GetInt32());
        JsonElement snapshot = restored.GetProperty("captured");
        JsonElement attachment = Assert.Single(snapshot.GetProperty("attachments").EnumerateArray());
        Assert.Equal("selection.png", attachment.GetProperty("maskPath").GetString());
        Assert.True(attachment.GetProperty("maskCrop").GetBoolean());
        Assert.Equal(5, attachment.GetProperty("maskFeather").GetInt32());
        Assert.Equal("documents", Assert.Single(snapshot.GetProperty("skills").EnumerateArray()).GetString());
        Assert.True(snapshot.GetProperty("think").GetBoolean());
    }

    [WebJavaScriptFact]
    public void ALanguageReloadDoesNotRestoreADraftIntoAnotherConversation()
    {
        JsonElement result = Run(DraftStorage + """
            storedDrafts['tensoragent-language-draft'] = JSON.stringify({ conversation: 'deleted-chat', text: 'private draft', attachments: [{ ok: true, file: 'private.pdf' }] });
            """, """
            return { text: __page.byId['text'].value, count: window.TensorAgent.attachmentCount(), cleared: !storedDrafts['tensoragent-language-draft'] };
            """);
        Assert.Equal(string.Empty, result.GetProperty("text").GetString());
        Assert.Equal(0, result.GetProperty("count").GetInt32());
        Assert.True(result.GetProperty("cleared").GetBoolean());
    }

    [WebJavaScriptFact]
    public void StorageFailureDefersTheLanguageReloadWithoutDiscardingTheDraft()
    {
        JsonElement result = Run("document.getElementById('text').value = ''; window.sessionStorage = { setItem: function () { throw new Error('storage full'); } };", """
            window.TensorAgent.insertText('keep this');
            window.TensorAgent.addAttachment({ ok: true, file: 'notes.pdf', mediaType: 'document' });
            return { kept: window.TensorAgent.prepareLanguageReload(), text: __page.byId['text'].value, count: window.TensorAgent.attachmentCount() };
            """);
        Assert.False(result.GetProperty("kept").GetBoolean());
        Assert.Equal("keep this", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("count").GetInt32());
    }

    [WebJavaScriptFact]
    public void ALanguageReloadWaitsUntilASentShareIsAcknowledged()
    {
        JsonElement result = Run(DraftStorage + $$"""
            var shareTaken = false, releaseHeaders;
            R['/api/agent/share/claim'] = function () {
              return shareTaken ? { share: null }
                : { share: { id: 'share-language', text: 'look at this', newChat: false, attachments: [] } };
            };
            R['/api/chat'] = function () {
              shareTaken = true;
              return new Promise(function (resolve) { releaseHeaders = resolve; });
            };
            """, $$"""
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              var before = window.TensorAgent.prepareLanguageReload();
              var savedBefore = storedDrafts['tensoragent-language-draft'] || null;
              releaseHeaders({ __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Done.' }, { done: true }] } });
              return settle(40).then(function () {
                return {
                  before: before, savedBefore: savedBefore,
                  after: window.TensorAgent.prepareLanguageReload(),
                  draft: JSON.parse(storedDrafts['tensoragent-language-draft']),
                  sends: __page.requests('/api/chat').length
                };
              });
            });
            """);
        Assert.Equal("busy", result.GetProperty("before").GetString());
        Assert.Equal(JsonValueKind.Null, result.GetProperty("savedBefore").ValueKind);
        Assert.True(result.GetProperty("after").GetBoolean());
        Assert.Empty(result.GetProperty("draft").GetProperty("shareOrder").EnumerateArray());
        Assert.Equal(1, result.GetProperty("sends").GetInt32());
    }
}
