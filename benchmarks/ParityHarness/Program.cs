// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Cross-engine parity + throughput harness. Feeds RAW token ids (so the model
// is isolated from tokenizer/template differences) and compares against
// llama.cpp greedy goldens, measures llama-bench-shaped throughput, and checks
// that batched (continuous-batching) decode reproduces serial decode.
//
// Modes:
//   parity <model.gguf> --ref <golden.json> [backend] [max_new]
//   parity <model.gguf> --bench <backend> <pp1,pp2,...> <tg> [reps]
//   parity <model.gguf> --batched <backend> <steps> <promptA> <promptB> [...]
//       prompts are comma-separated token ids; each runs on its own sequence
//       slot serially first, then all together through the fused batched
//       decode; the two must agree token for token.
//   parity <model.gguf> <tok0,tok1,...> [n_predict] [backend]   raw greedy
//   parity <model.gguf> --ppl <text-file> [backend] [n_ctx] [max_chunks]
//       teacher-forced perplexity over non-overlapping n_ctx windows,
//       scoring the SECOND half of each window (llama.cpp's
//       `llama-perplexity` protocol: first = n_ctx/2), so the numbers are
//       directly comparable to its "Final estimate: PPL = ..." line.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

public static class Program
{
    private sealed class RefRecord
    {
        public string prompt { get; set; }
        public int[] prompt_tokens { get; set; }
        public int[] generated_tokens { get; set; }
    }

    private static BackendType ResolveBackend(string s) => s switch
    {
        "ggmlcuda" or "ggml_cuda" => BackendType.GgmlCuda,
        "ggmlcpu" or "ggml_cpu" => BackendType.GgmlCpu,
        "ggmlvulkan" or "ggml_vulkan" => BackendType.GgmlVulkan,
        "cuda" => BackendType.Cuda,
        _ => BackendType.Cpu,
    };

    private static int ResolveTp()
    {
        string raw = Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE");
        return int.TryParse(raw, out int v) && v > 1 ? v : 1;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    public static int Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("usage: parity <model.gguf> --ref|--bench|--batched|<tokens> ...");
            return 1;
        }

        string modelPath = args[0];
        if (args[1] == "--ref") return RunReference(modelPath, args);
        if (args[1] == "--bench") return RunBench(modelPath, args);
        if (args[1] == "--batched") return RunBatched(modelPath, args);
        if (args[1] == "--ppl") return RunPerplexity(modelPath, args);
        return RunRaw(modelPath, args);
    }

    /// <summary>
    /// Teacher-forced perplexity, mirroring llama.cpp's `llama-perplexity`
    /// protocol: the text is tokenized once, split into non-overlapping n_ctx
    /// windows, and only the SECOND half of each window is scored (llama.cpp
    /// uses `first = n_ctx/2`) so every scored token has at least n_ctx/2 tokens
    /// of context. The window is fed one token at a time, which is what makes
    /// each position's logits available; that also means this measures the
    /// DECODE kernels, whereas llama.cpp scores a batched prefill.
    /// </summary>
    private static int RunPerplexity(string modelPath, string[] args)
    {
        string textPath = args[2];
        BackendType backend = ResolveBackend(args.Length > 3 ? args[3] : "ggmlcuda");
        int nCtx = args.Length > 4 ? int.Parse(args[4], CultureInfo.InvariantCulture) : 512;
        int maxChunks = args.Length > 5 ? int.Parse(args[5], CultureInfo.InvariantCulture) : int.MaxValue;

        var sw = Stopwatch.StartNew();
        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        Console.WriteLine($"[ppl] model loaded in {sw.Elapsed.TotalSeconds:F1}s, backend={backend}, arch={model.Config.Architecture}");

        List<int> ids = model.Tokenizer.Encode(File.ReadAllText(textPath), addSpecial: true);
        int chunks = Math.Min(maxChunks, ids.Count / nCtx);
        if (chunks <= 0)
        {
            Console.Error.WriteLine($"[ppl] text has {ids.Count} tokens, need at least n_ctx={nCtx}");
            return 1;
        }
        int first = nCtx / 2;
        Console.WriteLine($"[ppl] {ids.Count} tokens, n_ctx={nCtx}, scoring tokens [{first},{nCtx}) of {chunks} chunks");

        double nllSum = 0.0;
        long scored = 0;
        var swAll = Stopwatch.StartNew();
        for (int c = 0; c < chunks; c++)
        {
            model.ResetKVCache();
            int baseIdx = c * nCtx;
            float[] logits = null;
            for (int i = 0; i < nCtx - 1; i++)
            {
                logits = model.Forward(new[] { ids[baseIdx + i] });
                if (i + 1 < first)
                    continue;   // context-only positions are not scored

                int target = ids[baseIdx + i + 1];
                // log_softmax(logits)[target], computed in the numerically stable way.
                float max = float.NegativeInfinity;
                for (int v = 0; v < logits.Length; v++)
                    if (logits[v] > max) max = logits[v];
                double sumExp = 0.0;
                for (int v = 0; v < logits.Length; v++)
                    sumExp += Math.Exp(logits[v] - max);
                nllSum += -(logits[target] - max - Math.Log(sumExp));
                scored++;
            }
            double running = Math.Exp(nllSum / Math.Max(1, scored));
            Console.WriteLine($"[ppl] chunk {c + 1}/{chunks}  scored={scored}  running PPL = {running:F4}  ({swAll.Elapsed.TotalSeconds:F0}s)");
        }

        double ppl = Math.Exp(nllSum / Math.Max(1, scored));
        Console.WriteLine($"[ppl] Final estimate: PPL = {ppl:F4}  over {scored} tokens in {chunks} chunks of {nCtx}");
        return 0;
    }

    private static int RunRaw(string modelPath, string[] args)
    {
        int[] tokens = args[1].Split(',', StringSplitOptions.RemoveEmptyEntries)
                              .Select(t => int.Parse(t.Trim(), CultureInfo.InvariantCulture)).ToArray();
        int nPredict = args.Length > 2 ? int.Parse(args[2]) : 0;
        BackendType backend = ResolveBackend(args.Length > 3 ? args[3] : "ggmlcuda");

        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        float[] logits = model.Forward(tokens);
        Console.WriteLine($"n_vocab {logits.Length}");
        Console.WriteLine("logits " + string.Join(' ', logits.Select(v => v.ToString("F6", CultureInfo.InvariantCulture))));
        if (nPredict > 0)
        {
            var gen = new List<int>();
            int tok = ArgMax(logits);
            gen.Add(tok);
            for (int i = 1; i < nPredict; i++)
            {
                logits = model.Forward(new[] { tok });
                tok = ArgMax(logits);
                gen.Add(tok);
            }
            Console.WriteLine("generated " + string.Join(' ', gen));
        }
        return 0;
    }

    private static int RunReference(string modelPath, string[] args)
    {
        string refPath = args[2];
        BackendType backend = ResolveBackend(args.Length > 3 ? args[3] : "ggmlcuda");
        int maxNew = args.Length > 4 ? int.Parse(args[4]) : int.MaxValue;

        var records = JsonSerializer.Deserialize<List<RefRecord>>(File.ReadAllText(refPath));
        var sw = Stopwatch.StartNew();
        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        Console.WriteLine($"[parity] model loaded in {sw.Elapsed.TotalSeconds:F1}s, backend={backend}, arch={model.Config.Architecture}");

        int total = 0, matched = 0;
        foreach (var rec in records)
        {
            if (rec.prompt_tokens == null || rec.prompt_tokens.Length == 0) continue;
            int want = Math.Min(rec.generated_tokens.Length, maxNew);

            model.ResetKVCache();
            var swPrefill = Stopwatch.StartNew();
            float[] logits = model.ForwardRefill(rec.prompt_tokens);
            swPrefill.Stop();

            var produced = new int[want];
            var swDecode = Stopwatch.StartNew();
            for (int i = 0; i < want; i++)
            {
                produced[i] = ArgMax(logits);
                if (i + 1 < want)
                    logits = model.Forward(new[] { produced[i] });
            }
            swDecode.Stop();

            int firstDiff = -1;
            for (int i = 0; i < want; i++)
                if (produced[i] != rec.generated_tokens[i]) { firstDiff = i; break; }

            total++;
            string label = rec.prompt != null && rec.prompt.Length > 44 ? rec.prompt.Substring(0, 44) + "..." : rec.prompt;
            double pp = rec.prompt_tokens.Length / Math.Max(1e-9, swPrefill.Elapsed.TotalSeconds);
            double tg = Math.Max(0, want - 1) / Math.Max(1e-9, swDecode.Elapsed.TotalSeconds);
            if (firstDiff < 0)
            {
                matched++;
                Console.WriteLine($"[MATCH ] {label}  ({want} tokens)  prefill {pp:F1} tok/s  decode {tg:F2} tok/s");
            }
            else
            {
                Console.WriteLine($"[DIFF  ] {label}  diverges at {firstDiff}/{want}  prefill {pp:F1} tok/s  decode {tg:F2} tok/s");
                Console.WriteLine($"          ref: {string.Join(' ', rec.generated_tokens.Take(want))}");
                Console.WriteLine($"          ts : {string.Join(' ', produced)}");
            }
        }
        Console.WriteLine($"[parity] {matched}/{total} prompts reproduce llama.cpp token-for-token");
        return matched == total ? 0 : 2;
    }

    /// <summary>llama-bench-shaped throughput: synthetic prompt of P tokens
    /// (prefill t/s), then TG greedy decode steps (decode t/s), best of reps.</summary>
    private static int RunBench(string modelPath, string[] args)
    {
        BackendType backend = ResolveBackend(args[2]);
        int[] ppLens = args[3].Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        int tg = args.Length > 4 ? int.Parse(args[4]) : 64;
        int reps = args.Length > 5 ? int.Parse(args[5]) : 2;

        var sw = Stopwatch.StartNew();
        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        Console.WriteLine($"[bench] loaded in {sw.Elapsed.TotalSeconds:F1}s, backend={backend}, arch={model.Config.Architecture}");

        var rng = new Random(42);
        int vocab = Math.Max(1000, model.Config.VocabSize - 1000);

        foreach (int pp in ppLens)
        {
            double best = 0;
            var prompt = new int[pp];
            for (int i = 0; i < pp; i++) prompt[i] = 1000 + rng.Next(vocab - 1000);
            for (int r = 0; r < reps; r++)
            {
                model.ResetKVCache();
                var t = Stopwatch.StartNew();
                model.ForwardRefill(prompt);
                t.Stop();
                best = Math.Max(best, pp / t.Elapsed.TotalSeconds);
            }
            Console.WriteLine($"[bench] pp{pp,-8} {best,10:F2} tok/s");
        }

        {
            double best = 0;
            var prompt = new int[32];
            for (int i = 0; i < 32; i++) prompt[i] = 1000 + rng.Next(vocab - 1000);
            for (int r = 0; r < reps; r++)
            {
                model.ResetKVCache();
                float[] logits = model.ForwardRefill(prompt);
                int tok = ArgMax(logits);
                var t = Stopwatch.StartNew();
                for (int i = 0; i < tg; i++)
                {
                    logits = model.Forward(new[] { tok });
                    tok = ArgMax(logits);
                }
                t.Stop();
                best = Math.Max(best, tg / t.Elapsed.TotalSeconds);
            }
            Console.WriteLine($"[bench] tg{tg,-8} {best,10:F2} tok/s");
        }
        return 0;
    }

    /// <summary>Continuous-batching equivalence: each prompt decodes serially on
    /// its own sequence slot, then all together through the fused batched-decode
    /// step. Batching changes when the weights are read, not what the model
    /// computes, so the streams must agree token for token.</summary>
    private static int RunBatched(string modelPath, string[] args)
    {
        BackendType backend = ResolveBackend(args.Length > 2 ? args[2] : "ggmlcuda");
        int steps = args.Length > 3 ? int.Parse(args[3]) : 8;
        var prompts = new List<int[]>();
        for (int i = 4; i < args.Length; i++)
            prompts.Add(args[i].Split(',', StringSplitOptions.RemoveEmptyEntries)
                               .Select(t => int.Parse(t.Trim(), CultureInfo.InvariantCulture)).ToArray());
        if (prompts.Count < 2) { Console.Error.WriteLine("need at least two prompts"); return 1; }

        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        var seq = model as IBatchedPagedModel;
        if (seq == null || !seq.SupportsPerSequenceFusedForward)
        {
            Console.Error.WriteLine("[batched] model has no per-sequence slots");
            return 1;
        }

        int n = prompts.Count;
        var ids = new string[n];
        for (int i = 0; i < n; i++) ids[i] = "req" + i;

        // --- serial: each sequence decoded on its own slot ---
        var serial = new List<List<int>>();
        for (int i = 0; i < n; i++)
        {
            seq.BindSequenceCache(ids[i]);
            float[] lg = model.Forward(prompts[i]);
            var outs = new List<int>();
            int tok = ArgMax(lg);
            outs.Add(tok);
            for (int s = 1; s < steps; s++)
            {
                lg = model.Forward(new[] { tok });
                tok = ArgMax(lg);
                outs.Add(tok);
            }
            serial.Add(outs);
            seq.OnSequenceReleased(ids[i]);
        }

        // --- batched: same sequences, one fused step per token ---
        var lastTok = new int[n];
        var pos = new int[n];
        var batched = new List<List<int>>();
        for (int i = 0; i < n; i++)
        {
            seq.BindSequenceCache(ids[i]);
            float[] lg = model.Forward(prompts[i]);
            lastTok[i] = ArgMax(lg);
            pos[i] = prompts[i].Length;
            batched.Add(new List<int> { lastTok[i] });
        }

        var outLogits = new float[n][];
        int fusedSteps = 0, fallbackSteps = 0;
        for (int s = 1; s < steps; s++)
        {
            if (seq.TryForwardBatchedFusedDecode(ids, lastTok, pos, outLogits))
            {
                fusedSteps++;
                for (int i = 0; i < n; i++)
                {
                    lastTok[i] = ArgMax(outLogits[i]);
                    pos[i]++;
                    batched[i].Add(lastTok[i]);
                }
            }
            else
            {
                // round-robin fallback, as the engine would
                fallbackSteps++;
                for (int i = 0; i < n; i++)
                {
                    seq.BindSequenceCache(ids[i]);
                    float[] lg = model.Forward(new[] { lastTok[i] });
                    lastTok[i] = ArgMax(lg);
                    pos[i]++;
                    batched[i].Add(lastTok[i]);
                }
            }
        }
        for (int i = 0; i < n; i++) seq.OnSequenceReleased(ids[i]);

        bool allMatch = true;
        for (int i = 0; i < n; i++)
        {
            bool same = serial[i].SequenceEqual(batched[i]);
            allMatch &= same;
            Console.WriteLine($"[batched] seq{i}: {(same ? "MATCH" : "DIFF")}");
            if (!same)
            {
                Console.WriteLine($"          serial : {string.Join(' ', serial[i])}");
                Console.WriteLine($"          batched: {string.Join(' ', batched[i])}");
            }
        }
        Console.WriteLine($"[batched] fused steps={fusedSteps} fallback steps={fallbackSteps}");
        Console.WriteLine(allMatch ? "[batched] CONCURRENT_MATCH" : "[batched] CONCURRENT_DIFFERS");
        return allMatch ? 0 : 2;
    }
}
