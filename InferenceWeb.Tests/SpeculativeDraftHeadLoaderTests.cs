// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public sealed class SpeculativeDraftHeadLoaderTests : IDisposable
{
    private readonly string _dir;
    private readonly EnvScope _env = new();

    public SpeculativeDraftHeadLoaderTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), $"ts-draft-loader-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
        _env.ClearSpeculationVars();
    }

    public void Dispose()
    {
        _env.Dispose();
        Directory.Delete(_dir, recursive: true);
    }

    [Theory]
    [InlineData(SpeculationEnvVars.DraftModel)]
    [InlineData(SpeculationEnvVars.LegacyDraftModel)]
    public void ConfiguredDraftHeadPath_TrimsEitherEnvironmentSpelling(string variable)
    {
        string path = Path.Combine(_dir, "draft model.gguf");
        _env.Set(variable, $"  {path}  ");

        Assert.Equal(path, SpeculativeDraftHeadLoader.ConfiguredDraftHeadPath());
    }

    [Fact]
    public void TryAttachConfiguredDraftHead_FactoryLoadedBlockHeadIsAlreadyComplete()
    {
        string targetPath = WriteMinimalGguf("target.gguf");
        string draftPath = Path.Combine(_dir, "dspark.gguf");
        // Deliberately absent. Once the factory has made the head resident, the
        // attach-after-load phase must neither reopen nor reclassify its source
        // file (which may have been moved after loading).
        _env.Set(SpeculationEnvVars.DraftModel, draftPath);
        using var model = new FakeBlockModel(targetPath);

        bool attached = SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(
            model, out string error);

        Assert.True(attached);
        Assert.Null(error);
    }

    private string WriteMinimalGguf(string name)
    {
        string path = Path.Combine(_dir, name);
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);          // version
        writer.Write(0UL);         // tensor count
        writer.Write(0UL);         // metadata count
        writer.Write(new byte[8]); // 32-byte alignment
        return path;
    }

    private sealed class FakeBlockModel : ModelBase, IDraftHead
    {
        public FakeBlockModel(string ggufPath)
            : base(ggufPath, BackendType.Cpu)
        {
        }

        public DraftHeadKind DraftHeadKind => DraftHeadKind.Block;

        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
        {
        }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();

        protected override void ResetKVCacheCore()
        {
        }
    }
}
