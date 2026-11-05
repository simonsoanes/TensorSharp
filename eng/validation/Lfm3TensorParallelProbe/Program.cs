// Checks LFM3's tensor-parallel forward against one device, on one GPU: the ranks are
// backend instances on device 0 reducing through host memory, so this runs the real
// per-rank graphs, segment schedule and reduction, but not a device collective, and its
// speed means nothing. Loopback lasts for the life of the process, which is why this is
// a tool and not a test.
//
// Both models read the same tokens: chunked prefill, then the one-device model's greedy
// tokens, with the state extracted and restored halfway. A small initial cache makes
// both grow during prefill and again during decode.
//
//   Lfm3TensorParallelProbe MODEL.gguf TEXT.txt [--ranks 2] [--prompt 700] [--chunk 256]
//       [--gen 64] [--tolerance 0.1] [--per-op]   (tolerance: top-1 log-prob)
//
// --per-op puts the tensor-parallel model on the per-op path instead: the rounding floor
// of a sharded forward, measured without any of the native code.
// --teacher feeds the text's own next tokens instead of greedy ones and judges the mean
// negative log-likelihood of them: greedy text passes through flat distributions where one
// token can swing on rounding, but a numerical fault biases this average.
// Exit code 0 only when the tensor-parallel run took the expected path and matched.
using System.Diagnostics;
using System.Reflection;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

string modelPath = args[0], textPath = args[1];
int Option(string name, int fallback)
{
    int i = Array.IndexOf(args, name);
    return i < 0 ? fallback : int.Parse(args[i + 1]);
}
int ranks = Option("--ranks", 2), promptLength = Option("--prompt", 700), chunk = Option("--chunk", 256), gen = Option("--gen", 64);
int toleranceIndex = Array.IndexOf(args, "--tolerance");
double tolerance = toleranceIndex < 0 ? 0.1 : double.Parse(args[toleranceIndex + 1]);
bool expectNative = !args.Contains("--per-op");
bool teacher = args.Contains("--teacher");

Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", "64");
KvCacheDtypeConfig.ConfigureFromEnvironment();

int[] prompt;
int[] continuation = new int[gen];
List<float[]> expected;
long blockBytes;
using (var single = new Lfm3Model(modelPath, BackendType.GgmlCuda))
{
    if (!UsesNative(single))
        throw new InvalidOperationException("The one-device reference did not take the native forward.");
    int[] text = single.Tokenizer.Encode(File.ReadAllText(textPath), addSpecial: false).Skip(1000).Take(promptLength + gen).ToArray();
    prompt = text[..promptLength];
    if (teacher)
    {
        text[promptLength..].CopyTo(continuation, 0);
    }
    else
    {
        // This pass only discovers the greedy tokens the next one replays.
        Trace(single, prompt, chunk, continuation, discover: true);
        single.ResetKVCache();
    }
    expected = Trace(single, prompt, chunk, continuation, discover: false);
    blockBytes = single.ComputeKVBlockByteSize(prompt.Length);
}

GgmlBasicOps.ClearHostBufferCache();
if (!expectNative)
    Environment.SetEnvironmentVariable("TS_LFM3_NATIVE", "0");
var context = GgmlContext.CreateTensorParallelLoopback(GgmlBackendType.Cuda, ranks);
using var group = new GgmlTensorParallelGroup(context);
using var sharded = new Lfm3Model(modelPath, BackendType.GgmlCuda, ranks, group);
bool native = UsesNative(sharded);
Console.WriteLine($"{ranks} ranks: {(native ? "native plans" : "per-op")}");
var watch = Stopwatch.StartNew();
List<float[]> actual = Trace(sharded, prompt, chunk, continuation, discover: false);
Console.WriteLine($"tensor-parallel trace: {watch.Elapsed.TotalSeconds:F1} s (loopback; not a speed measurement)");

double worst = 0, worstTop = 0, total = 0;
int worstStep = -1, argmaxDifferences = 0, compared = 0;
for (int step = 0; step < expected.Count; step++)
{
    double[] e = LogSoftmax(expected[step]), a = LogSoftmax(actual[step]);
    foreach (int id in Enumerable.Range(0, e.Length).OrderByDescending(i => e[i]).Take(10))
    {
        double difference = Math.Abs(e[id] - a[id]);
        if (difference > worst) (worst, worstStep) = (difference, step);
        total += difference;
        compared++;
    }
    int top = ArgMax(expected[step]);
    worstTop = Math.Max(worstTop, Math.Abs(e[top] - a[top]));
    if (Math.Abs(e[top] - a[top]) > 0.05)
        Console.WriteLine($"  step {step} (position {prompt.Length + step}): top token p {Math.Exp(e[top]):F4} vs {Math.Exp(a[top]):F4}");
    if (ArgMax(expected[step]) != ArgMax(actual[step]))
        argmaxDifferences++;
}
long shardedBlockBytes = sharded.ComputeKVBlockByteSize(prompt.Length);
Console.WriteLine($"{expected.Count} steps: top-10 |dlogprob| max {worst:F4} at step {worstStep}, mean {total / compared:F4}, top-1 max {worstTop:F4}, argmax differs at {argmaxDifferences}, " +
    $"block bytes {shardedBlockBytes} vs {blockBytes}");

// Gated on the top token: the tail of the top 10 moves by more on this path's own
// rounding floor (--per-op, 0.11 BF16 and 0.60 Q4_K_M) than a bug would need to hide in.
double expectedNll = 0, actualNll = 0;
for (int step = 0; step < continuation.Length; step++)
{
    expectedNll -= LogSoftmax(expected[step])[continuation[step]];
    actualNll -= LogSoftmax(actual[step])[continuation[step]];
}
expectedNll /= continuation.Length;
actualNll /= continuation.Length;
Console.WriteLine($"mean NLL of the {continuation.Length} continuation tokens: {expectedNll:F5} vs {actualNll:F5} " +
    $"(perplexity {Math.Exp(expectedNll):F3} vs {Math.Exp(actualNll):F3})");

bool passed = native == expectNative && shardedBlockBytes == blockBytes && (teacher
    ? Math.Abs(expectedNll - actualNll) <= 0.005
    : worstTop <= tolerance && argmaxDifferences == 0);
Console.WriteLine(passed ? "PASS" : "FAIL");
return passed ? 0 : 1;

static bool UsesNative(Lfm3Model model) => (bool)typeof(Lfm3Model)
    .GetProperty("UsesNativeForward", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

// Logits after the prefill and after every continuation token.
static List<float[]> Trace(Lfm3Model model, int[] prompt, int chunk, int[] continuation, bool discover)
{
    var logits = new List<float[]>();
    float[] last = [];
    for (int start = 0; start < prompt.Length; start += chunk)
        last = model.Forward(prompt[start..Math.Min(prompt.Length, start + chunk)]);
    logits.Add((float[])last.Clone());
    var decode = Stopwatch.StartNew();
    for (int step = 0; step < continuation.Length; step++)
    {
        if (discover)
            continuation[step] = ArgMax(last);
        if (step == continuation.Length / 2)
            RoundTrip(model, prompt.Length + step);
        last = model.Forward([continuation[step]]);
        logits.Add((float[])last.Clone());
    }
    Console.WriteLine($"decode: {continuation.Length / decode.Elapsed.TotalSeconds:F1} tok/s, the snapshot round trip included");
    return logits;
}

// Everything out, a reset, everything back: what the server does between requests.
static void RoundTrip(Lfm3Model model, int tokens)
{
    var block = new byte[model.ComputeKVBlockByteSize(tokens)];
    if (!model.TryExtractKVBlock(0, tokens, block))
        throw new InvalidOperationException("Snapshot extract refused.");
    model.ResetKVCache();
    if (!model.TryInjectKVBlock(0, tokens, block))
        throw new InvalidOperationException("Snapshot inject refused.");
}

static int ArgMax(float[] values)
{
    int best = 0;
    for (int i = 1; i < values.Length; i++)
        if (values[i] > values[best]) best = i;
    return best;
}

static double[] LogSoftmax(float[] logits)
{
    double max = logits.Max(), sum = 0;
    foreach (float value in logits)
        sum += Math.Exp(value - max);
    double log = Math.Log(sum) + max;
    return logits.Select(value => value - log).ToArray();
}
