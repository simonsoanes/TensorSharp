// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace TensorAgent.Maui.Hosting;

/// <summary>
/// What the engine probe found; serialised verbatim by <c>GET /api/engine</c> and
/// summarised in the native status bar.
/// </summary>
/// <param name="Backend">The backend <see cref="Compute"/> selected (GgmlMetal / GgmlCpu).</param>
/// <param name="GgmlCpuAvailable"><c>TSGgml_CanInitializeBackend(Cpu)</c> through the public managed API.</param>
/// <param name="GgmlMetalAvailable"><c>TSGgml_CanInitializeBackend(Metal)</c> through the public managed API.</param>
/// <param name="MainProgramHandleResolved">Whether <c>TSGgml_CanInitializeBackend</c> can be found by dlsym in the
/// main program image, i.e. the static link plus the exported_symbol linker flags worked. Only the phone links
/// the engine statically; the desktop heads load it as a library and report <paramref name="NativeLibraryLoaded"/>.</param>
/// <param name="NativeLibraryLoaded">Whether the GgmlOps resolver has loaded the engine, from the main program
/// image on the phone or from the library shipped beside the desktop app.</param>
/// <param name="GgmlVersionOrError">A one-line description of the successful native round trip, or the
/// exception (type and message) the first P/Invoke threw.</param>
/// <param name="EngineAssemblies">Name and version of every TensorSharp engine assembly that loaded.</param>
/// <param name="GpuName">The Metal device name, or null.</param>
/// <param name="Reason">Why <see cref="Backend"/> was chosen.</param>
public sealed record EngineProbeResult(
    string Backend,
    bool GgmlCpuAvailable,
    bool GgmlMetalAvailable,
    bool MainProgramHandleResolved,
    bool NativeLibraryLoaded,
    string GgmlVersionOrError,
    IReadOnlyList<string> EngineAssemblies,
    string? GpuName,
    string Reason);

/// <summary>
/// Proves, cheaply and without touching GGML's backend slot, that the engine (the
/// statically linked GgmlOps archive on the phone, the GgmlOps library beside a
/// desktop app) is reachable from managed code and that the engine assemblies load
/// under the trimmed runtime.
/// </summary>
public static class EngineProbe
{
    private const string ProbeExport = "TSGgml_CanInitializeBackend";

    public static EngineProbeResult Run()
    {
        // Assemblies first: if one of them fails to load (trimmed away, missing
        // dependency) that is the error to report, not a downstream P/Invoke.
        var assemblies = new List<string>();
        foreach (Assembly assembly in EngineAssemblies())
        {
            AssemblyName name = assembly.GetName();
            assemblies.Add($"{name.Name} {name.Version}");
        }

        bool handleResolved = false;
        try
        {
            IntPtr main = NativeLibrary.GetMainProgramHandle();
            handleResolved = main != IntPtr.Zero && NativeLibrary.TryGetExport(main, ProbeExport, out _);
        }
        catch (Exception ex)
        {
            // Diagnostic only: the P/Invoke below is the real test, and its
            // exception is the one reported in GgmlVersionOrError.
            Console.WriteLine($"TensorAgent: GetMainProgramHandle probe failed: {ex.GetType().Name}: {ex.Message}");
        }

        bool cpu = false;
        bool metal = false;
        string ggmlVersionOrError;
        ComputeSelection selection;
        try
        {
            // These are the first P/Invokes into GgmlOps in the process. They go
            // through GgmlNative.ImportResolver (main program handle on iOS) and
            // the native side is a compile-flag check, so nothing is initialised.
            cpu = GgmlBasicOps.CanInitializeBackend(GgmlBackendType.Cpu);
            metal = GgmlBasicOps.CanInitializeBackend(GgmlBackendType.Metal);
            selection = Compute.Selection;
            ggmlVersionOrError =
                $"ok: {ProbeExport}(Cpu)={cpu}, (Metal)={metal}; " +
#if IOS
                (handleResolved
                    ? "TSGgml_* resolved from the main program image."
                    : "resolver reached GgmlOps although dlsym on the main program image did not find " + ProbeExport + ".");
#else
                (GgmlBasicOps.IsNativeLibraryLoaded
                    ? "TSGgml_* resolved from the GgmlOps library shipped with the app."
                    : "the P/Invokes answered, but the resolver never recorded loading GgmlOps.");
#endif
        }
        catch (Exception ex)
        {
            ggmlVersionOrError = $"{ex.GetType().Name}: {ex.Message}";
            // Compute.Selection would throw the same way; report the failure once.
            return new EngineProbeResult(
                Backend: "unavailable",
                GgmlCpuAvailable: false,
                GgmlMetalAvailable: false,
                MainProgramHandleResolved: handleResolved,
                NativeLibraryLoaded: GgmlBasicOps.IsNativeLibraryLoaded,
                GgmlVersionOrError: ggmlVersionOrError,
                EngineAssemblies: assemblies,
                GpuName: null,
                Reason: "The native engine could not be reached, see ggmlVersionOrError.");
        }

        return new EngineProbeResult(
            Backend: selection.Backend.ToString(),
            GgmlCpuAvailable: cpu,
            GgmlMetalAvailable: metal,
            MainProgramHandleResolved: handleResolved,
            NativeLibraryLoaded: GgmlBasicOps.IsNativeLibraryLoaded,
            GgmlVersionOrError: ggmlVersionOrError,
            EngineAssemblies: assemblies,
            GpuName: selection.GpuName,
            Reason: selection.Reason);
    }

    // One public type per engine assembly. Touching the type forces the assembly
    // to load, which is the whole point.
    private static IEnumerable<Assembly> EngineAssemblies()
    {
        yield return typeof(TensorSharp.Tensor).Assembly;                                 // TensorSharp.Core
        yield return typeof(TensorSharp.Runtime.BackendType).Assembly;                    // TensorSharp.Runtime
        yield return typeof(TensorSharp.Models.Video.IVideoGenerationModel).Assembly;     // TensorSharp.Models
        yield return typeof(GgmlBasicOps).Assembly;                                       // TensorSharp.Backends.GGML
        yield return typeof(TensorSharp.AgentHost.Skills.SkillRegistry).Assembly;         // TensorSharp.AgentHost
    }
}
