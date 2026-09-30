using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Validation;

string Required(string name) => Environment.GetEnvironmentVariable(name)
    ?? throw new ArgumentException($"Set {name}.");
int Number(string name, int fallback) => int.TryParse(Environment.GetEnvironmentVariable(name), out int value) ? value : fallback;
string modelPath = Required("TS_TEST_GEMMA4_MODEL");
string output = Environment.GetEnvironmentVariable("TS_GEMMA4_PROBE_OUT") ?? "artifacts/gemma4-batched-cache/report.json";
bool allowFallback = args.Contains("--allow-fallback", StringComparer.Ordinal);
bool uniformControl = args.Contains("--uniform-control", StringComparer.Ordinal);
bool diagnosticOnly = args.Contains("--diagnostic-only", StringComparer.Ordinal);
int steps = Number("TS_GEMMA4_PROBE_STEPS", 12);
int pairs = Number("TS_GEMMA4_PROBE_PAIRS", 3);
if (steps is < 4 or > 512 || pairs is < 1 or > 100) throw new ArgumentOutOfRangeException("Use 4..512 steps and 1..100 pairs.");
string backendName = Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cuda";
BackendType backend = backendName switch { "cuda" => BackendType.GgmlCuda, "metal" => BackendType.GgmlMetal, "vulkan" => BackendType.GgmlVulkan, "cpu" => BackendType.GgmlCpu, _ => throw new ArgumentException("Unknown GGML backend.") };
Environment.SetEnvironmentVariable("MAX_CONTEXT", "16384");
Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", "0");
using var model = ModelBase.Create(modelPath, backend);
if (model is not Gemma4Model gemma) throw new InvalidOperationException("Requires Gemma 4.");
var runs = new List<CacheProbeComparison>();
string widthsText = Environment.GetEnvironmentVariable("TS_GEMMA4_PROBE_WIDTHS") ?? "2,3,4";
foreach (int width in widthsText.Split(',').Select(int.Parse))
{
    Console.WriteLine($"[cache-probe] warmup width={width}");
    _ = Gemma4CacheProbe.Compare(gemma, width, Math.Min(steps, 6), batchedFirst: false, requireFused: !allowFallback,
        compactClones: !uniformControl, requireNumericalParity: !diagnosticOnly);
    for (int pair = 0; pair < pairs; pair++)
    {
        var row = Gemma4CacheProbe.Compare(gemma, width, steps, pair % 2 != 0, requireFused: !allowFallback,
            compactClones: !uniformControl, requireNumericalParity: !diagnosticOnly);
        runs.Add(row);
        Console.WriteLine($"[cache-probe] width={width} pair={pair + 1} serial_ms={row.Serial.Milliseconds:F2} batch_ms={row.Batched.Milliseconds:F2} accepted={row.Batched.FusedSteps}/{steps} capacities={string.Join(',', row.Batched.InitialCapacities)} cosine={row.MinCosine:F8} rmse={row.MaxRmse:F6}");
    }
}
var module = Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Single(m => Path.GetFileName(m.FileName).Equals(OperatingSystem.IsWindows() ? "GgmlOps.dll" : OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so", StringComparison.OrdinalIgnoreCase));
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
File.WriteAllText(output, JsonSerializer.Serialize(new
{
    Model = Path.GetFullPath(modelPath), ModelBytes = new FileInfo(modelPath).Length, Backend = backendName, Context = 16384, Steps = steps, Pairs = pairs,
    AllowFallback = allowFallback, UniformControl = uniformControl, DiagnosticOnly = diagnosticOnly,
    ValidationPassed = !allowFallback && !diagnosticOnly, WarmupsExcluded = true,
    NativePath = module.FileName, NativeSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(module.FileName))),
    ManagedSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(Gemma4Model).Assembly.Location))),
    Runs = runs,
    Summary = runs.GroupBy(r => r.Width).Select(g => new { Width = g.Key, SerialMedianMs = Median(g.Select(r => r.Serial.Milliseconds)), BatchedMedianMs = Median(g.Select(r => r.Batched.Milliseconds)), Speedup = Median(g.Select(r => r.Serial.Milliseconds)) / Median(g.Select(r => r.Batched.Milliseconds)), AcceptedSteps = g.Sum(r => r.Batched.FusedSteps), FallbackSteps = g.Sum(r => r.Batched.FallbackSteps) }),
    Limitations = "Direct teacher-forced decode microbenchmark; model loading, prefill/checkpoint allocation and solo continuation excluded from timing. Does not measure end-to-end agent performance. Each timed batch includes any cache growth and graph capture. Only listed model/backend scenarios are validated."
}, new JsonSerializerOptions { WriteIndented = true }));
static double Median(IEnumerable<double> values) { double[] sorted = values.Order().ToArray(); return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2; }
