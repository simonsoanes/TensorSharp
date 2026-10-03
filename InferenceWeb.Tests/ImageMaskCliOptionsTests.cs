// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cli;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

public sealed class ImageMaskCliOptionsTests
{
    private static ImageMaskCliOptions Parse(params string[] args)
    {
        args = CliUsage.NormalizeOptionSpellings(args);
        var result = new ImageMaskCliOptions();
        for (int i = 0; i < args.Length; i++) result.Read(args, ref i);
        return result;
    }

    [Fact]
    public void ExplicitOptionsSurviveNormalizedSpellings()
    {
        var options = Parse("--MASK=mask.png", "--mask-mode=alpha", "--mask-invert", "--mask-feather=8", "--mask-crop", "--mask-crop-padding", "96");
        Assert.Equal("mask.png", options.Path);
        Assert.Equal(QwenImageMaskMode.Alpha, options.Mode);
        Assert.True(options.Invert);
        Assert.Equal(8, options.Feather);
        Assert.True(options.Crop);
        Assert.Equal(96, options.CropPadding);
        Assert.Equal(QwenImageMaskMode.Grayscale, Parse("--mask", "mask.png").Mode);
    }

    [Theory]
    [InlineData("--mask-mode", "edit")]
    [InlineData("--mask-feather", "-1")]
    [InlineData("--mask-feather", "1025")]
    [InlineData("--mask-feather", "1.5")]
    [InlineData("--mask-crop-padding", "16385")]
    [InlineData("--mask", "")]
    public void InvalidValuesAreConfigurationErrors(string flag, string value) =>
        Assert.Throws<ArgumentException>(() => Parse(flag, value));

    [Fact]
    public void MaskRequiresImageAndQwenModel()
    {
        Assert.Contains("--image", Assert.Throws<ArgumentException>(() => Parse("--mask", "mask.png").Validate(0)).Message);
        Assert.Contains("--mask", Assert.Throws<ArgumentException>(() => Parse("--mask-feather", "8").Validate(1)).Message);
        Assert.Contains("Qwen-Image-2.1", Assert.Throws<ArgumentException>(() => Parse("--mask", "mask.png").ValidateModel(false)).Message);
        Parse().Validate(0);
        Parse().ValidateModel(false);
    }

    [Theory]
    [InlineData("--mask")]
    [InlineData("--mask-mode")]
    [InlineData("--mask-feather")]
    [InlineData("--mask-crop-padding")]
    public void MissingValuesAreConfigurationErrors(string flag) => Assert.Throws<ArgumentException>(() => Parse(flag));

    [Fact]
    public void MissingFileIsAConfigurationError() => Assert.Contains("not found", Assert.Throws<ArgumentException>(() =>
        Parse("--mask", System.IO.Path.Combine(System.IO.Path.GetTempPath(), Guid.NewGuid() + ".png")).Validate(1)).Message);
}
