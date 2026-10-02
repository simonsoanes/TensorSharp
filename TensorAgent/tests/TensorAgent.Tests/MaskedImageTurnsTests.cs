using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;

namespace TensorAgent.Tests;

public sealed class MaskedImageTurnsTests
{
    [Fact]
    public void SelectedAreaIsForwardedSeparatelyFromSourceAndReferences()
    {
        using JsonDocument body = JsonDocument.Parse("""
            {"messages":[{"role":"user","content":"change the scarf",
              "stillImagePaths":["source.png","reference.png"],"maskPath":"selection.png",
              "maskMode":"grayscale","maskInvert":true,"maskFeather":4,"maskCrop":true,"maskCropPadding":96}]}
            """);
        ImageTurns.ImageRequest request = Assert.IsType<ImageTurns.ImageRequest>(ImageTurns.Read(body.RootElement));
        JsonElement payload = JsonSerializer.SerializeToElement(request.Payload);
        Assert.True(request.Editing);
        Assert.Equal(new[] { "source.png", "reference.png" }, payload.GetProperty("imagePaths").EnumerateArray().Select(x => x.GetString()));
        Assert.Equal("selection.png", payload.GetProperty("maskPath").GetString());
        Assert.Equal("grayscale", payload.GetProperty("maskMode").GetString());
        Assert.True(payload.GetProperty("maskInvert").GetBoolean());
        Assert.Equal(4, payload.GetProperty("maskFeather").GetInt32());
        Assert.True(payload.GetProperty("maskCrop").GetBoolean());
        Assert.Equal(96, payload.GetProperty("maskCropPadding").GetInt32());
    }

    [Fact]
    public void OldSelectionsDoNotLeakIntoNewImageRequests()
    {
        using JsonDocument body = JsonDocument.Parse("""
            {"messages":[{"role":"user","content":"edit","stillImagePaths":["old.png"],"maskPath":"old-mask.png"},
              {"role":"assistant","imageUrl":"/uploads/old-result.png"},
              {"role":"user","content":"new edit","stillImagePaths":["new.png"]}]}
            """);
        var request = Assert.IsType<ImageTurns.ImageRequest>(ImageTurns.Read(body.RootElement));
        Assert.False(JsonSerializer.SerializeToElement(request.Payload).TryGetProperty("maskPath", out _));
    }

    [Theory]
    [InlineData("\"mask.png\"")]
    [InlineData("123")]
    public void MissingSourceAndInvalidSelectionAreSentToServiceValidation(string mask)
    {
        using JsonDocument body = JsonDocument.Parse("""
            {"messages":[{"role":"user","content":"edit","maskPath":
            """ + mask + "}]}");
        var request = Assert.IsType<ImageTurns.ImageRequest>(ImageTurns.Read(body.RootElement));
        Assert.True(request.Editing);
        JsonElement payload = JsonSerializer.SerializeToElement(request.Payload);
        Assert.Empty(payload.GetProperty("imagePaths").EnumerateArray());
        Assert.Equal(mask, payload.GetProperty("maskPath").GetRawText());
    }

    [Fact]
    public void SavedConversationKeepsSelectionSettingsAndProtectsItsUpload()
    {
        var message = new StoredMessage
        {
            Content = "make the scarf red", StillImagePaths = ["source.png"],
            MaskPath = "mask.png", MaskMode = "grayscale", MaskFeather = 3, MaskCrop = true,
            Attachments = [new StoredAttachment { File = "source.png", MediaType = "image", MaskPath = "mask.png", MaskFeather = 3, MaskCrop = true }],
        };
        var loaded = JsonSerializer.Deserialize<StoredMessage>(JsonSerializer.Serialize(message))!;
        Assert.Equal("mask.png", loaded.MaskPath);
        Assert.Equal("grayscale", loaded.MaskMode);
        Assert.Equal(3, loaded.MaskFeather);
        Assert.True(loaded.MaskCrop);
        Assert.Equal("mask.png", Assert.Single(loaded.Attachments!).MaskPath);
        Assert.Contains("mask.png", loaded.ReferencedUploads);
    }
}
