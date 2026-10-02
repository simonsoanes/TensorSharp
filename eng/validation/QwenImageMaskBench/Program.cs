// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// dotnet run -c Release --project eng/validation/QwenImageMaskBench -- [--sizes 512,1024,2048] [--reps 10] [--output artifacts/mask-bench.json]
// Synthetic CPU mask overhead only. No model weights, transformer, VAE, image encoding or device are exercised.
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Models.QwenImage;

string Option(string name, string fallback)
{
    int index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : fallback;
}

int[] sizes = Option("--sizes", "512,1024,2048").Split(',').Select(int.Parse).ToArray();
int repetitions = int.Parse(Option("--reps", "10"));
if (repetitions < 1 || sizes.Any(s => s < 32 || s % 32 != 0))
    throw new ArgumentException("Repetitions must be positive and sizes positive multiples of 32.");
var results = new List<object>();
foreach (int size in sizes)
{
    var rgb = new float[checked(size * size * 3)];
    var maskRgb = new float[rgb.Length];
    var alpha = new float[checked(size * size)];
    for (int i = 0; i < alpha.Length; i++)
    {
        alpha[i] = (i % 255) / 254f;
        int x = i % size, y = i / size;
        float selected = x >= size * 3 / 8 && x < size * 5 / 8 && y >= size * 3 / 8 && y < size * 5 / 8 ? 1f : 0f;
        for (int c = 0; c < 3; c++)
        {
            rgb[i * 3 + c] = ((i * 17 + c * 31) % 255) / 254f;
            maskRgb[i * 3 + c] = selected;
        }
    }
    var source = new RgbImage(size, size, rgb, alpha);
    var inputMask = new RgbImage(size, size, maskRgb);
    foreach (bool crop in new[] { false, true })
    {
        var parameters = new QwenImageParams { Mask = inputMask, MaskCrop = crop, MaskCropPadding = 32, MaskFeather = 8 };
        var plan = QwenImageEditMask.Create(parameters, source);
        var dimensions = plan.SamplingDimensions(size, size);
        var mask = plan.LatentWeights(dimensions.Width / 16, dimensions.Height / 16);
        var clean = Enumerable.Range(0, mask.Length * 64).Select(i => (i % 17 - 8) / 8f).ToArray();
        var noise = Enumerable.Range(0, clean.Length).Select(i => (i % 11 - 5) / 5f).ToArray();
        var state = Enumerable.Repeat(.125f, clean.Length).ToArray();
        var reference = (float[])state.Clone();
        ScalarReinject(reference, clean, noise, mask, .375f);
        QwenImageEditMask.Reinject(state, clean, noise, mask, .375f);
        float maxError = state.Zip(reference, (a, b) => Math.Abs(a - b)).Max();
        if (maxError > 1e-6f) throw new InvalidOperationException($"SIMD/scalar error {maxError}");

        var generated = new RgbImage(plan.Width, plan.Height, Enumerable.Repeat(1f, plan.Width * plan.Height * 3).ToArray());
        var output = plan.Composite(generated);
        int protectedPixels = 0, changedPixels = 0;
        for (int i = 0; i < plan.Weights.Length; i++)
        {
            if (plan.Weights[i] == 0)
            {
                protectedPixels++;
                if (output.Alpha[i] != source.Alpha[i]) throw new InvalidOperationException("Protected alpha changed.");
                for (int c = 0; c < 3; c++)
                    if (output.Pixels[i * 3 + c] != source.Pixels[i * 3 + c]) throw new InvalidOperationException("Protected RGB changed.");
            }
            else if (output.Pixels[i * 3] != source.Pixels[i * 3]) changedPixels++;
        }
        if (protectedPixels == 0 || changedPixels == 0) throw new InvalidOperationException("Incomplete synthetic composite coverage.");
        var preparation = Measure(() => QwenImageEditMask.Create(parameters, source), repetitions);
        var reduction = Measure(() => plan.LatentWeights(dimensions.Width / 16, dimensions.Height / 16), repetitions);
        var reinjection = Measure(() => QwenImageEditMask.Reinject(state, clean, noise, mask, .375f), repetitions * 100);
        var scalar = Measure(() => ScalarReinject(state, clean, noise, mask, .375f), repetitions * 100);
        var composite = Measure(() => plan.Composite(generated), repetitions);
        if (reinjection.AllocatedBytesPerCall != 0) throw new InvalidOperationException("Reinjection allocated per-step memory.");
        results.Add(new
        {
            sourceWidth = size, sourceHeight = size, crop, feather = parameters.MaskFeather,
            region = new { plan.X, plan.Y, plan.Width, plan.Height },
            sampleWidth = dimensions.Width, sampleHeight = dimensions.Height, latentTokens = mask.Length,
            fullCanvasTokens = size * size / 256,
            tokenReduction = 1.0 - (double)mask.Length / (size * size / 256),
            preparation, reduction, reinjection, scalar, composite, maxSimdScalarError = maxError,
            protectedPixels, changedPixels, protectedRgbAndAlphaExact = true
        });
    }
}
var report = new
{
    kind = "Synthetic CPU mask preparation, latent constraint, and compositing microbenchmark",
    limitations = "No real model/VAE/transformer inference or GPU speedup measured. Token reduction is geometric, not an end-to-end latency claim. Composite timing excludes resizing and image encoding.",
    timestampUtc = DateTimeOffset.UtcNow, runtime = RuntimeInformation.FrameworkDescription,
    os = RuntimeInformation.OSDescription, cpuCount = Environment.ProcessorCount,
    vectorLanes = Vector<float>.Count, vectorAccelerated = Vector.IsHardwareAccelerated,
    repetitions, results
};
string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
string outputPath = Option("--output", "");
if (outputPath.Length > 0)
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
    File.WriteAllText(outputPath, json);
}
Console.WriteLine(json);

static Measurement Measure(Action action, int repetitions)
{
    // Exercise fast methods long enough for tiered JIT/PGO to settle before comparing
    // SIMD and scalar. A fixed handful of calls mostly measures tier-0 code at 512px.
    long warmupStart = Stopwatch.GetTimestamp();
    int warmups = 0;
    do { action(); warmups++; }
    while (warmups < 5 || Stopwatch.GetElapsedTime(warmupStart).TotalMilliseconds < 250);
    long startBytes = GC.GetAllocatedBytesForCurrentThread();
    long startTime = Stopwatch.GetTimestamp();
    for (int i = 0; i < repetitions; i++) action();
    double elapsed = Stopwatch.GetElapsedTime(startTime).TotalMilliseconds;
    long bytes = GC.GetAllocatedBytesForCurrentThread() - startBytes;
    return new Measurement(elapsed / repetitions, (double)bytes / repetitions);
}

static void ScalarReinject(float[] state, float[] source, float[] noise, float[] mask, float sigma)
{
    for (int token = 0; token < mask.Length; token++)
    {
        float amount = mask[token];
        if (amount == 1f) continue;
        for (int c = 0; c < 64; c++)
        {
            int at = token * 64 + c;
            float original = (1f - sigma) * source[at] + sigma * noise[at];
            state[at] = amount == 0f ? original : amount * state[at] + (1f - amount) * original;
        }
    }
}

internal sealed record Measurement(double MillisecondsPerCall, double AllocatedBytesPerCall);
