using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptFact]
    public void MaskedEditWaitsForTheChosenLorasBeforeSending()
    {
        JsonElement result = Run(MaskImageModel + TwoStyles + Held + """
            R['/api/chat'] = { __sse: [
              { image_step: 1, image_steps: 6, image_loras: ['Film Stills'], preview: 'data:image/png;base64,AAAA' },
              { imageUrl: '/uploads/result.png' }, { done: true, sessionId: 's1' }
            ] };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'mask.png' });
            __page.byId['text'].value = 'make only the selected area blue';
            return openLoras().then(function () {
              hold('/api/agent/loras/choice');
              var film = loraRow('film').querySelectorAll('INPUT')[0];
              film.checked = true; film.dispatch('change');
              __page.byId['sheet-bg'].dispatch('click');
              __page.byId['send'].dispatch('click');
              var early = __page.requests('/api/chat').length;
              var draft = __page.byId['text'].value;
              return settle(20).then(function () {
                var pending = heldFor('/api/agent/loras/choice')[0];
                pending.answer(loraRows(pending.call.body.loras));
                return settle(20).then(function () {
                  __page.byId['send'].dispatch('click');
                  return settle(30).then(function () { return {
                    early: early, draft: draft, sent: __page.requests('/api/chat'), progress: __page.progress(),
                    media: __page.transcript().slice(-1)[0].media, errors: __page.errorNotices()
                  }; });
                });
              });
            });
            """);
        Assert.Equal(0, result.GetProperty("early").GetInt32());
        Assert.Equal("make only the selected area blue", result.GetProperty("draft").GetString());
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal("mask.png", message.GetProperty("maskPath").GetString());
        Assert.Equal(new[] { "source.png" }, Strings(message, "stillImagePaths"));
        Assert.Contains("Drawing with Film Stills… step 1 of 6", Strings(result, "progress"));
        Assert.Equal("/uploads/result.png", Assert.Single(result.GetProperty("media").EnumerateArray()).GetProperty("src").GetString());
        Assert.Empty(Strings(result, "errors"));
    }

    private const string MaskImageModel = """
        R['/api/models'] = { loaded: 'qwen-image.gguf', architecture: 'qwen_image', visionReady: true };
        R['/api/upload'] = { ok: true, file: 'selection.png', mediaType: 'image' };
        R['/api/chat'] = { __sse: [{ imageUrl: '/uploads/result.png' }, { done: true, sessionId: 's1' }] };
        """;

    [WebJavaScriptFact]
    public void SwitchingToAnImageModelOffersSelectionForAnAlreadyAttachedPhoto()
    {
        JsonElement result = Run("", """
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image' });
            var before = __page.byId['chips'].querySelectorAll('.filechip').length;
            R['/api/models'] = { loaded: 'qwen-image.gguf', architecture: 'qwen_image' };
            window.TensorAgent.refreshModel();
            return settle(20).then(function () { return {
              before: before, after: __page.byId['chips'].querySelectorAll('.filechip').map(function (b) { return b.textContent; })
            }; });
            """);
        Assert.Equal(0, result.GetProperty("before").GetInt32());
        Assert.Equal("Select area", Assert.Single(Strings(result, "after")));
    }

    [WebJavaScriptFact]
    public void SelectedAreaIsUploadedAndSentWithOnlyTheFirstSourcePhoto()
    {
        JsonElement result = Run(MaskImageModel, """
            var editorOptions;
            window.TensorSharpMaskEditor = { open: function (options) {
              editorOptions = options;
              return Promise.resolve({ blob: { png: true }, maskFeather: 3, maskCrop: true });
            } };
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', url: '/uploads/source.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'reference.png', mediaType: 'image', url: '/uploads/reference.png' });
            var controls = __page.byId['chips'].querySelectorAll('.filechip');
            var controlCount = controls.length;
            controls[0].dispatch('click');
            return settle(20).then(function () {
              __page.byId['text'].value = 'make the scarf red'; __page.byId['send'].dispatch('click');
              return settle(20).then(function () {
                return { controlCount: controlCount, options: editorOptions,
                  uploads: __page.requests('/api/upload'), sent: __page.requests('/api/chat'),
                  errors: __page.errorNotices() };
              });
            });
            """);
        Assert.Equal(1, result.GetProperty("controlCount").GetInt32());
        Assert.Equal("/uploads/source.png", result.GetProperty("options").GetProperty("sourceUrl").GetString());
        Assert.Equal("selection.png", Assert.Single(result.GetProperty("uploads").EnumerateArray()).GetProperty("parts")[0].GetProperty("fileName").GetString());
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal(new[] { "source.png", "reference.png" }, Strings(message, "stillImagePaths"));
        Assert.Equal("selection.png", message.GetProperty("maskPath").GetString());
        Assert.Equal("grayscale", message.GetProperty("maskMode").GetString());
        Assert.Equal(3, message.GetProperty("maskFeather").GetInt32());
        Assert.True(message.GetProperty("maskCrop").GetBoolean());
        Assert.Equal("selection.png", message.GetProperty("attachments")[0].GetProperty("maskPath").GetString());
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptFact]
    public void FailedSelectionUploadKeepsThePreviousSelectionAndDraft()
    {
        JsonElement result = Run(MaskImageModel + """
            R['/api/upload'] = { __status: 500, body: { error: 'disk full' } };
            """, """
            window.TensorSharpMaskEditor = { open: function () {
              return Promise.resolve({ blob: {}, maskFeather: 2, maskCrop: false });
            } };
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'previous.png' });
            __page.byId['text'].value = 'change only this area';
            __page.byId['chips'].querySelector('.filechip').dispatch('click');
            __page.byId['send'].dispatch('click');
            var during = __page.requests('/api/chat').length;
            return settle(20).then(function () {
              var draft = __page.byId['text'].value;
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return {
                during: during, draft: draft, sent: __page.requests('/api/chat'), errors: __page.errorNotices()
              }; });
            });
            """);
        Assert.Equal(0, result.GetProperty("during").GetInt32());
        Assert.Equal("change only this area", result.GetProperty("draft").GetString());
        Assert.Equal("previous.png", Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0].GetProperty("maskPath").GetString());
        Assert.Contains("disk full", Assert.Single(Strings(result, "errors")), StringComparison.Ordinal);
    }

    [WebJavaScriptFact]
    public void RemovingSelectionRestoresWholeImageEditing()
    {
        JsonElement result = Run(MaskImageModel, """
            window.TensorSharpMaskEditor = { open: function () { return Promise.resolve({ remove: true }); } };
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'previous.png' });
            __page.byId['chips'].querySelector('.filechip').dispatch('click');
            return settle(20).then(function () {
              __page.byId['text'].value = 'change the style'; __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return { sent: __page.requests('/api/chat'), uploads: __page.requests('/api/upload') }; });
            });
            """);
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.False(message.TryGetProperty("maskPath", out _));
        Assert.Empty(result.GetProperty("uploads").EnumerateArray());
    }

    [WebJavaScriptFact]
    public void ResultCanCompareAndRepeatTheOriginalSelectionAndPrompt()
    {
        JsonElement result = Run(MaskImageModel, """
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'mask.png', maskFeather: 5, maskCrop: true });
            __page.byId['text'].value = 'make the scarf red'; __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var actions = __page.byId['chat'].querySelector('.image-edit-actions');
              var buttons = actions.querySelectorAll('button');
              buttons[0].dispatch('click');
              var original = __page.transcript().slice(-1)[0].media[0].src;
              buttons[0].dispatch('click');
              var edited = __page.transcript().slice(-1)[0].media[0].src;
              buttons[1].dispatch('click');
              var prompt = __page.byId['text'].value;
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return { original: original, edited: edited, prompt: prompt, sent: __page.requests('/api/chat') }; });
            });
            """);
        Assert.Equal("/uploads/source.png", result.GetProperty("original").GetString());
        Assert.Equal("/uploads/result.png", result.GetProperty("edited").GetString());
        Assert.Equal("make the scarf red", result.GetProperty("prompt").GetString());
        JsonElement message = result.GetProperty("sent")[1].GetProperty("body").GetProperty("messages")[2];
        Assert.Equal("source.png", Assert.Single(Strings(message, "stillImagePaths")));
        Assert.Equal("mask.png", message.GetProperty("maskPath").GetString());
        Assert.Equal(5, message.GetProperty("maskFeather").GetInt32());
        Assert.True(message.GetProperty("maskCrop").GetBoolean());
    }
}
