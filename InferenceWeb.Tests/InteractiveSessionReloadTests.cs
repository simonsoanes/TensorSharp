// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.Cli;
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public sealed class InteractiveSessionReloadTests : IDisposable
{
    private readonly string _dir;
    private readonly EnvScope _env = new();

    public InteractiveSessionReloadTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), $"ts-interactive-reload-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
        _env.ClearSpeculationVars();
    }

    public void Dispose()
    {
        _env.Dispose();
        Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void CreateModelForReload_ForwardsConfiguredDraftAndRunsPostLoadAttachment()
    {
        string targetPath = WriteMinimalGguf("target.gguf");
        string draftPath = Path.Combine(_dir, "draft.gguf");
        File.WriteAllBytes(draftPath, new byte[] { 1, 2, 3 });
        _env.Set(SpeculationEnvVars.DraftModel, draftPath);

        FakeModel created = null;
        ModelBase attachedTo = null;
        InteractiveSession.DraftHeadAttacher attach =
            (ModelBase model, out string error) =>
            {
                attachedTo = model;
                error = "draft checkpoint is incompatible";
                return false;
            };

        var loaded = InteractiveSession.CreateModelForReload(
            targetPath,
            BackendType.Cpu,
            (path, backend, configuredDraft) =>
            {
                Assert.Equal(targetPath, path);
                Assert.Equal(BackendType.Cpu, backend);
                Assert.Equal(draftPath, configuredDraft);
                return created = new FakeModel(path);
            },
            attach);

        try
        {
            Assert.Same(created, loaded.Model);
            Assert.Same(created, attachedTo);
            Assert.Equal("draft checkpoint is incompatible", loaded.DraftHeadError);
            Assert.False(created.Disposed);
        }
        finally
        {
            loaded.Model.Dispose();
        }
    }

    [Fact]
    public void CreateModelForReload_AttachmentExceptionDisposesUnadoptedReplacement()
    {
        string targetPath = WriteMinimalGguf("target.gguf");
        FakeModel created = null;
        InteractiveSession.DraftHeadAttacher attach =
            (ModelBase _, out string error) =>
            {
                error = null;
                throw new InvalidOperationException("attachment failed unexpectedly");
            };

        var ex = Assert.Throws<InvalidOperationException>(() =>
            InteractiveSession.CreateModelForReload(
                targetPath,
                BackendType.Cpu,
                (path, _, _) => created = new FakeModel(path),
                attach));

        Assert.Equal("attachment failed unexpectedly", ex.Message);
        Assert.True(created.Disposed);
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

    private sealed class FakeModel : ModelBase
    {
        public bool Disposed { get; private set; }

        public FakeModel(string ggufPath)
            : base(ggufPath, BackendType.Cpu)
        {
        }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();

        protected override void ResetKVCacheCore()
        {
        }

        public override void Dispose()
        {
            Disposed = true;
            base.Dispose();
        }
    }
}
