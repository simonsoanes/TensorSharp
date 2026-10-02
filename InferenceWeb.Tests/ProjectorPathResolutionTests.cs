// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// <c>--mmproj</c> may name the directory the projector sits in. The DeepSeek V4.1 CLI opened
/// <c>--mmproj /workspace/models/deepseek/</c> as a GGUF, failed on the magic bytes, and aborted with a
/// core dump (exit 134); a directory now resolves to the family's own companion file, and a directory
/// without one is a load refusal naming what was looked for.
/// </summary>
public sealed class ProjectorPathResolutionTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-mmproj-" + Guid.NewGuid().ToString("N"));

    public ProjectorPathResolutionTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    [Fact]
    public void ADirectory_ResolvesToTheFamilysCompanionInside_AndAFileIsUsedAsGiven()
    {
        // The V4.1 directory beside the shards holds the text model, a drafter and the vision companion.
        File.WriteAllBytes(Path.Combine(_dir, "DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf"), Array.Empty<byte>());
        File.WriteAllBytes(Path.Combine(_dir, "DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf"), Array.Empty<byte>());
        string vision = Path.Combine(_dir, "deepseek41.vision.gguf");
        File.WriteAllBytes(vision, Array.Empty<byte>());

        Assert.Equal(vision, ModelArchitectureRegistry.ResolveProjectorPath("deepseek41", _dir));
        Assert.Equal(vision, ModelArchitectureRegistry.ResolveProjectorPath("deepseek41", _dir + Path.DirectorySeparatorChar));
        Assert.Equal(vision, ModelArchitectureRegistry.ResolveProjectorPath("deepseek41", vision));
        // A path that is not a directory is the loader's to judge, as before.
        string missing = Path.Combine(_dir, "nope.gguf");
        Assert.Equal(missing, ModelArchitectureRegistry.ResolveProjectorPath("deepseek41", missing));
        Assert.Null(ModelArchitectureRegistry.ResolveProjectorPath("deepseek41", null));

        // A wildcard hint (Qwen 3.8, GLM) picks the same file the beside-the-model lookup would.
        string mmproj = Path.Combine(_dir, "mmproj-BF16.gguf");
        File.WriteAllBytes(mmproj, Array.Empty<byte>());
        Assert.Equal(mmproj, ModelArchitectureRegistry.ResolveProjectorPath("qwen4exp", _dir));
        Assert.Equal(ModelArchitectureRegistry.FindCompanionProjector("qwen4exp", Path.Combine(_dir, "model.gguf")),
            ModelArchitectureRegistry.ResolveProjectorPath("qwen4exp", _dir));
    }

    [Fact]
    public void ADirectoryWithoutTheCompanion_IsALoadRefusalNamingWhatWasLookedFor()
    {
        File.WriteAllBytes(Path.Combine(_dir, "DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf"), Array.Empty<byte>());
        var ex = Assert.Throws<FileNotFoundException>(() => ModelArchitectureRegistry.ResolveProjectorPath("deepseek41", _dir));
        Assert.Contains("deepseek41.vision.gguf", ex.Message, StringComparison.Ordinal);
        Assert.Contains(_dir, ex.Message, StringComparison.Ordinal);
        Assert.True(ModelLoadRefusal.TryDescribe(ex, out string reason));
        Assert.DoesNotContain('\n', ModelLoadRefusal.FormatErrorLine(reason));
    }
}
