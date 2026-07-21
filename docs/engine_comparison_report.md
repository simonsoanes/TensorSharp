# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on the selected compute backends (ggml_cuda / ggml_vulkan / ggml_metal / ggml_cpu / cpu / ...).

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

## Software / hardware

| Component | Version / detail |
|---|---|
| TensorSharp | git `8a13852`, .NET 10.0.204 (backends: ggml_cuda / ggml_vulkan / ggml_metal / cuda / mlx / ggml_cpu / cpu) |
| llama.cpp | `C:\Works\llama.cpp\build-cuda\bin\Release\llama-server.exe` |
| vLLM | endpoint `http://127.0.0.1:8000` (connect-only) |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16384 MiB |


## Methodology

- Each `(engine, backend, model)` group launches its server once; all of that group's scenarios run against it, so per-scenario timings exclude model-load cost.
- Metrics come from the **streamed** response: `ttft` is time-to-first-token (prefill latency proxy), `prefill_tps = prompt_tokens / ttft`, and `decode_tps = (completion_tokens - 1) / (t_last - t_first)`.
- DiffusionGemma denoises whole blocks (no token stream), so it is run non-streaming and its `decode_tps` is wall-clock tokens/second.
- Greedy sampling (`temperature=0`); one warmup request per server is discarded.
- The headline per-engine tables are the **single-stream, MTP-off** baseline. MTP on/off and parallel-request scaling are reported in their own sections below.

## Performance ratio — TensorSharp vs reference engines

Geomean of TensorSharp's per-scenario speedup over each reference engine on the **same backend**, across every scenario both engines ran (single-stream, MTP-off). A value **> 1.0× means TensorSharp is faster** (for decode / prefill throughput) or lower-latency (for TTFT); `—` = no overlapping cells. Per-scenario ratios are in each model's section below.

| Model | Comparison | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it (Q8_0, dense multimodal) | vs llama.cpp · CUDA | 1.02× | 1.28× | 1.27× |
| Gemma 4 E4B it (Q8_0, dense multimodal) | vs llama.cpp · Vulkan | 1.00× | 1.05× | 1.03× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | vs llama.cpp · CUDA | 1.04× | 1.17× | 1.16× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | vs llama.cpp · Vulkan | 1.21× | 1.04× | 1.03× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | vs llama.cpp · CUDA | 0.98× | 1.28× | 1.27× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | vs llama.cpp · Vulkan | 0.87× | 1.04× | 1.03× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | vs llama.cpp · CUDA | 1.07× | 0.96× | 0.95× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | vs llama.cpp · Vulkan | 1.02× | 0.85× | 0.84× |

## Gemma 4 E4B it (Q8_0, dense multimodal)  (`gemma4-e4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 51.6 | 44.1 | 50.6 | 45.4 |
| text_long | 51.4 | 42.5 | 49.9 | 43.3 |
| multi_turn | 50.7 | 42.4 | 50.2 | 40.1 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 2633.3 | 1780.9 | 1889.4 | 1558.4 |
| text_long | 2615.1 | 1163.9 | 2190.6 | 1647.3 |
| multi_turn | 2627.4 | 1740.6 | 2068.0 | 1225.1 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 750.0 | 1109.0 | 1031.0 | 1250.0 |
| text_long | 1203.0 | 2703.0 | 1422.0 | 1891.0 |
| multi_turn | 797.0 | 1203.0 | 1000.0 | 1688.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1.02× | 0.97× |
| text_long | 1.03× | 0.98× |
| multi_turn | 1.01× | 1.06× |

_Prefill throughput_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1.39× | 1.14× |
| text_long | 1.19× | 0.71× |
| multi_turn | 1.27× | 1.42× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1.37× | 1.13× |
| text_long | 1.18× | 0.70× |
| multi_turn | 1.25× | 1.40× |

## Gemma 4 12B it (QAT UD-Q4_K_XL, dense)  (`gemma4-12b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 38.1 | 32.2 | 38.2 | 30.1 |
| text_long | 38.4 | 32.0 | 36.0 | 24.2 |
| multi_turn | 38.7 | 32.4 | 36.6 | 26.0 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 1118.3 | 804.8 | 1104.2 | 685.5 |
| text_long | 1144.0 | 564.0 | 1007.1 | 586.6 |
| multi_turn | 1155.6 | 560.8 | 832.9 | 558.7 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 1766.0 | 2454.0 | 1765.0 | 2843.0 |
| text_long | 2750.0 | 5578.0 | 3094.0 | 5312.0 |
| multi_turn | 1812.0 | 3734.0 | 2484.0 | 3703.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1.00× | 1.07× |
| text_long | 1.07× | 1.32× |
| multi_turn | 1.06× | 1.25× |

_Prefill throughput_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1.01× | 1.17× |
| text_long | 1.14× | 0.96× |
| multi_turn | 1.39× | 1.00× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1.00× | 1.16× |
| text_long | 1.13× | 0.95× |
| multi_turn | 1.37× | 0.99× |

## Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE)  (`gemma4-26b-a4b`)

**Decode throughput (tok/s)**

| Scenario | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 52.1 | fail |
| text_long | 50.0 | fail |
| multi_turn | 50.6 | fail |

**Prefill throughput (tok/s)**

| Scenario | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 166.5 | fail |
| text_long | 165.4 | fail |
| multi_turn | 162.9 | fail |

**Time to first token (ms, lower is better)**

| Scenario | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 11703.0 | fail |
| text_long | 18844.0 | fail |
| multi_turn | 12703.0 | fail |

## Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE)  (`qwen36-35b-a3b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 67.9 | 39.0 | 77.5 | 49.5 |
| text_long | 73.5 | 42.9 | 70.8 | 46.3 |
| multi_turn | 73.9 | 41.6 | 71.3 | 45.9 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 1212.3 | 853.5 | 1113.5 | 928.1 |
| text_long | 1388.3 | 895.5 | 1064.7 | 896.7 |
| multi_turn | 1377.5 | 931.2 | 922.5 | 757.2 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 1672.0 | 2375.0 | 1797.0 | 2156.0 |
| text_long | 2328.0 | 3609.0 | 3000.0 | 3562.0 |
| multi_turn | 1563.0 | 2312.0 | 2296.0 | 2797.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 0.88× | 0.79× |
| text_long | 1.04× | 0.93× |
| multi_turn | 1.04× | 0.91× |

_Prefill throughput_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1.09× | 0.92× |
| text_long | 1.30× | 1.00× |
| multi_turn | 1.49× | 1.23× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1.07× | 0.91× |
| text_long | 1.29× | 0.99× |
| multi_turn | 1.47× | 1.21× |

## Qwen 3.6 27B (UD-IQ2_XXS, dense)  (`qwen36-27b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 18.9 | 13.2 | 19.2 | 14.0 |
| text_long | 19.2 | 14.7 | 17.1 | 14.0 |
| multi_turn | 19.2 | 14.9 | 17.2 | 14.0 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 432.5 | 217.7 | 498.3 | 298.5 |
| text_long | 423.9 | 232.7 | 461.4 | 274.8 |
| multi_turn | 397.2 | 251.0 | 357.6 | 253.4 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | TensorSharp · Vulkan | llama.cpp · CUDA | llama.cpp · Vulkan |
|---|---:|---:|---:|---:|
| text_short | 4687.0 | 9313.0 | 4016.0 | 6704.0 |
| text_long | 7625.0 | 13891.0 | 6922.0 | 11625.0 |
| multi_turn | 5421.0 | 8578.0 | 5922.0 | 8359.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 0.98× | 0.94× |
| text_long | 1.12× | 1.05× |
| multi_turn | 1.12× | 1.06× |

_Prefill throughput_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 0.87× | 0.73× |
| text_long | 0.92× | 0.85× |
| multi_turn | 1.11× | 0.99× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · CUDA | vs llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 0.86× | 0.72× |
| text_long | 0.91× | 0.84× |
| multi_turn | 1.09× | 0.97× |

## Output quality — TensorSharp vs llama.cpp

Both engines decode the **same GGUF greedily** (temperature=0) on the same backend, so their outputs should agree closely. `similarity` is a whitespace-normalized SequenceMatcher ratio between the two outputs (1.00 = identical); low similarity, an invalid JSON object in `json_mode`, or a missing tool call in `function_call` flags an output-quality problem on one side. Prefill scenarios (8-token outputs) are excluded. Side-by-side excerpts follow the table, lowest agreement first.

| Model | Backend | Scenario | similarity | verdict | tokens (TS / ref) | finish (TS / ref) | checks |
|---|---|---|---:|---|---|---|---|
| gemma4-e4b | CUDA | text_short | 0.02 | diverged | 284 / 522 | stop / stop | — |
| gemma4-e4b | Vulkan | text_short | 0.01 | diverged | 239 / 527 | stop / stop | — |
| gemma4-e4b | CUDA | text_long | 0.01 | diverged | 392 / 600 | stop / stop | — |
| gemma4-e4b | Vulkan | text_long | 0.01 | diverged | 320 / 578 | stop / stop | — |
| gemma4-e4b | CUDA | multi_turn | 0.03 | diverged | 1206 / 1280 | stop / length | — |
| gemma4-e4b | Vulkan | multi_turn | 0.02 | diverged | 1169 / 1280 | stop / length | — |
| gemma4-12b | CUDA | text_short | 0.15 | diverged | 79 / 493 | stop / stop | — |
| gemma4-12b | Vulkan | text_short | 0.08 | diverged | 71 / 557 | stop / stop | — |
| gemma4-12b | CUDA | text_long | 0.18 | diverged | 76 / 761 | stop / stop | — |
| gemma4-12b | Vulkan | text_long | 0.27 | diverged | 76 / 737 | stop / stop | — |
| gemma4-12b | CUDA | multi_turn | 0.06 | diverged | 619 / 1280 | stop / length | — |
| gemma4-12b | Vulkan | multi_turn | 0.03 | diverged | 500 / 1280 | stop / length | — |
| qwen36-35b-a3b | CUDA | text_short | 0.10 | diverged | 72 / 1233 | stop / stop | — |
| qwen36-35b-a3b | Vulkan | text_short | 0.11 | diverged | 87 / 1159 | stop / stop | — |
| qwen36-35b-a3b | CUDA | text_long | 0.13 | diverged | 71 / 1066 | stop / stop | — |
| qwen36-35b-a3b | Vulkan | text_long | 0.10 | diverged | 68 / 667 | stop / stop | — |
| qwen36-35b-a3b | CUDA | multi_turn | 0.02 | diverged | 46 / 1280 | stop / length | — |
| qwen36-35b-a3b | Vulkan | multi_turn | 0.03 | diverged | 174 / 1280 | stop / length | — |
| qwen36-27b | CUDA | text_short | 0.25 | diverged | 85 / 651 | stop / stop | — |
| qwen36-27b | Vulkan | text_short | 0.20 | diverged | 82 / 707 | stop / stop | — |
| qwen36-27b | CUDA | text_long | 0.11 | diverged | 66 / 1280 | stop / length | — |
| qwen36-27b | Vulkan | text_long | 0.06 | diverged | 74 / 892 | stop / stop | — |
| qwen36-27b | CUDA | multi_turn | 0.10 | diverged | 37 / 1280 | stop / length | — |
| qwen36-27b | Vulkan | multi_turn | 0.06 | diverged | 40 / 1130 | stop / stop | — |

Mean similarity across 24 compared cells: **0.09** (min 0.01, max 0.27).

<details><summary>gemma4-e4b · ggml_vulkan · text_long (similarity 0.01)</summary>

**TensorSharp**

```
The user wants a two-sentence summary of the provided text, which details the history of computing.

1.  **Analyze the source material:** The text traces computing from ancient mechanical aids (abacus) through theoretical breakthroughs (Turing machine), hardware revolutions (vacuum tubes to transistors/Moore's Law), architectural shifts (stored program concept), and connectivity advancements (Internet/Web). It concludes with the modern era of machine learning as the latest layer of abstraction.
2.  **Identify key themes for condensation:** Mechanical calculation $\rightarrow$ Theoretical found
```

**llama.cpp**

```
The history of computing progressed from ancient mechanical aids to theoretical breakthroughs like the Turing machine, which defined the limits of computation, and the stored-program concept, which made machines flexible and general-purpose. Driven by the transistor and Moore's Law, this evolution accelerated through layers of software abstraction and global networking, ultimately transforming computers from specialized tools into an essential, ambient utility woven into daily life.
```

</details>
<details><summary>gemma4-e4b · ggml_vulkan · text_short (similarity 0.01)</summary>

**TensorSharp**

```
The user wants a three-sentence explanation of a transformer neural network based on the provided text.

1.  **Scan the text for "transformer architecture."**
2.  **Identify key characteristics:** It was introduced in 2017, it replaced sequential reading with an attention mechanism, and this allows every position in a sequence to directly consult every other position.
3.  **Draft the sentences, ensuring conciseness.**

*Self-Correction/Refinement:* The text mentions that earlier networks read input one element at a time (slow/forgetful), and the transformer uses *attention* so every position c
```

**llama.cpp**

```
The transformer architecture was developed to overcome the limitations of earlier sequence networks, which processed input one element at a time, leading to slow training and forgetfulness over long distances. It replaces this sequential reading with a mechanism called attention. This attention allows every position in a sequence to directly consult every other position, enabling the network to decide how much weight to give each element and effectively manage long-distance dependencies.
```

</details>
<details><summary>gemma4-e4b · ggml_cuda · text_long (similarity 0.01)</summary>

**TensorSharp**

```
The user wants a two-sentence summary of the provided text, which details the history of computing.

1.  **Analyze the source material:** The text traces computing from ancient mechanical aids (abacus) through theoretical breakthroughs (Turing machine), hardware revolutions (vacuum tubes to transistors/Moore's Law), architectural shifts (von Neumann stored program), and connectivity (Internet/Web). It concludes with the modern era of machine learning built upon these foundations.
2.  **Identify key themes for Sentence 1 (Early/Mid History):** Focus on the foundational concepts—from mechanical 
```

**llama.cpp**

```
The history of computing progressed from ancient mechanical aids and prophetic designs like Babbage's Analytical Engine to the theoretical foundation laid by the Turing machine and the practical shift enabled by the stored-program concept. Miniaturization via the transistor and integrated circuit, coupled with the rise of high-level software and the global Internet, transformed computing into an ambient utility, a process defined by continuous layers of abstraction that now supports modern machine learning.
```

</details>
<details><summary>qwen36-35b-a3b · ggml_cuda · multi_turn (similarity 0.02)</summary>

**TensorSharp**

```
For a $3,000 budget, I'd suggest splitting it roughly 60/40 between Kyoto and Tokyo. This reflects the fact that Kyoto often has higher accommodation costs due to its popularity during peak season.
```

**llama.cpp**

```
Here's a thinking process:

1.  **Analyze User Input:**
   - User wants to split a $3000 budget between two cities: Kyoto and Tokyo.
   - Context: Cherry blossom season trip to Japan.
   - Need to provide a reasonable spending split/recommendation based on typical costs, travel patterns, and the nature of each city.

2.  **Identify Key Factors:**
   - Total budget: $3000
   - Cities: Tokyo and Kyoto
   - Trip duration: Not specified, but typically cherry blossom trips are 7-10 days. I'll assume a standard ~7-8 day trip split between the two.
   - Costs in Japan vary by city, accommodation, foo
```

</details>
<details><summary>gemma4-e4b · ggml_vulkan · multi_turn (similarity 0.02)</summary>

**TensorSharp**

```
Here's a thinking process to arrive at the suggested budget split:

1.  **Analyze the Request:** The user is asking for a rough spending allocation between Kyoto and Tokyo, given a total budget of $3000 for a trip during cherry blossom season.
2.  **Identify Key Variables/Assumptions (Implicit):**
    *   **Trip Duration:** Not specified, but since they are prioritizing two major cities, I should assume a standard 7-10 day trip structure.
    *   **Travel Style:** "Comfortable" was mentioned earlier, suggesting mid-range accommodation, reasonable dining, and some activities, not extreme backpa
```

**llama.cpp**

```
This is a tricky question because the split depends heavily on **how long** you plan to spend in each city, and whether you prioritize high-end experiences (like luxury hotels or fine dining) or budget travel.

However, assuming you are planning a balanced trip (e.g., 4-5 days in each city), here is a general guideline for splitting your *on-the-ground* spending:

### ð¯ðµ Tokyo: ~55% of the Budget
### ð¸ Kyoto: ~45% of the Budget

***

### Why this split?

**Tokyo (The Higher Cost/Higher Variety Hub):**
Tokyo is a massive, modern metropolis and generally has a higher cost of living, esp
```

</details>
<details><summary>gemma4-e4b · ggml_cuda · text_short (similarity 0.02)</summary>

**TensorSharp**

```
The user wants a three-sentence explanation of a transformer neural network based on the provided text.

1.  **Scan the text for "transformer architecture."**
2.  **Identify key characteristics:** It was introduced in 2017, it replaced sequential reading with an attention mechanism, and this allows every position in a sequence to directly consult every other position.
3.  **Draft the sentences, ensuring conciseness.**

*Self-Correction/Refinement:* Ensure the explanation focuses on *what* it is and *how* it works, as described in the context.

**Drafting:**
1. The transformer architecture, int
```

**llama.cpp**

```
The transformer architecture was developed to overcome the limitations of earlier sequence-processing networks, which were slow and forgetful over long distances because they read input one element at a time. It replaces this sequential reading with a mechanism called attention. This attention allows every position in a sequence to directly consult every other position, enabling the network to decide how much weight to give each element and effectively manage long-range dependencies.
```

</details>
<details><summary>gemma4-12b · ggml_vulkan · multi_turn (similarity 0.03)</summary>

**TensorSharp**

```
Since you have a $3000 budget (which is quite healthy for a standard trip to Japan), the best way to split your spending depends on whether you want a "luxury" experience in one city or a balanced experience across both.

However, because **Tokyo** generally has a higher cost of living and more diverse dining/entertainment options, while **Kyoto** is often slightly cheaper for accommodation but can be pricey during peak blossom season, here is a recommended breakdown:

### 1. The Balanced Split (Recommended)
*   **Tokyo: $1,600 (approx. 55%)**
*   **Kyoto: $1,400 (approx. 45%)**

**Why this sp
```

**llama.cpp**

```
With a $3,000 budget for a week-long
```

</details>
<details><summary>qwen36-35b-a3b · ggml_vulkan · multi_turn (similarity 0.03)</summary>

**TensorSharp**

```
For a $3,000 budget, I recommend splitting it roughly **60/40** between Kyoto and Tokyo.

Here is why:
*   **Tokyo (approx. $1,800):** As Japan’s largest metropolis, accommodation and dining costs are generally higher. You’ll likely spend more on hotels and varied activities here.
*   **Kyoto (approx. $1,200):** While still popular, you can find excellent value in traditional *ryokan* (inns) or mid-range hotels, especially if you stay slightly outside the main tourist hubs like Gion.

This split allows you to enjoy Tokyo's vibrant energy while ensuring your time in Kyoto—where the cherry bloss
```

**llama.cpp**

```
Here's a thinking process:

1.  **Analyze User Input:**
   - User wants to split a $3000 budget between two cities: Kyoto and Tokyo.
   - Context: Cherry blossom season trip to Japan.
   - Need to provide a reasonable spending split/recommendation based on typical costs, travel patterns, and the nature of the two cities.

2.  **Identify Key Factors:**
   - Total budget: $3000
   - Cities: Tokyo and Kyoto
   - Trip duration: Not specified, but typically cherry blossom trips are 7-10 days. I'll assume a standard ~7-8 day trip for budgeting purposes, but I should clarify or make reasonable assump
```

</details>
<details><summary>gemma4-e4b · ggml_cuda · multi_turn (similarity 0.03)</summary>

**TensorSharp**

```
Here's a thinking process to arrive at the suggested budget split:

1.  **Analyze the Request:** The user is asking for a rough spending allocation between Kyoto and Tokyo, given a total budget of $3000 for a trip during cherry blossom season.
2.  **Identify Key Variables/Assumptions (Implicit):**
    *   **Duration:** Not specified, but assuming a standard 7-10 day trip to cover both cities reasonably.
    *   **Travel Style:** Since the budget is $3000, it suggests a mid-range traveler (not ultra-luxury, but not backpacker level either). Cherry blossom season often means higher accommodation
```

**llama.cpp**

```
This is highly dependent on how long you plan to stay in each city, but generally, a **balanced split (around 50/50)** is recommended, with a slight lean toward Tokyo if you plan on doing more high-end dining or nightlife.

However, since the total budget of $3000 must cover *everything* (flights, accommodation, food, transport), it's more helpful to look at how the budget should be allocated across the *categories* of spending, rather than just the two cities.

Here is a suggested framework for how your $3000 might be distributed for a 7-10 day trip:

### âï¸ Fixed Costs (The Non-Negotiabl
```

</details>
<details><summary>qwen36-27b · ggml_vulkan · multi_turn (similarity 0.06)</summary>

**TensorSharp**

```
A 60/40 split toward Tokyo makes sense. Kyoto’s charm is concentrated, so you can cover it in three days; Tokyo offers more flexibility for dining and shopping within your budget.
```

**llama.cpp**

```
A **50/50 split ($1,500 per city)** works best for a $3,000 budget. Tokyo and Kyoto have surprisingly similar baseline costs for mid-range travel, and the bullet train between them is only ~$30â50 round trip, so you won't lose much to intercity transit.

**Typical per-city breakdown ($1,500):**
- **Accommodation:** $600â700 (4â5 nights in a solid business hotel or ryokan)
- **Food & drinks:** $300â400 (mix of convenience store breakfasts, ramen/ramen shops, and 1â2 nice meals)
- **Local transit & activities:** $200â300 (day passes, temple/shrine fees, museum tickets, occasional tax
```

</details>
<details><summary>gemma4-12b · ggml_cuda · multi_turn (similarity 0.06)</summary>

**TensorSharp**

```
Since you have a $3000 budget (which is quite healthy for a standard trip to Japan), the best way to split your spending depends on whether you want a "luxury" experience in one city or a balanced experience across both.

However, because **Kyoto** generally has a higher concentration of traditional experiences and **Tokyo** offers a wider variety of dining and shopping options, here is a recommended breakdown:

### 1. The Balanced Split (Recommended)
*   **Tokyo: $1,600 (approx. 55%)**
*   **Kyoto: $1,400 (approx. 45%)**

**Why this split?**
*   **Tokyo's Scale:** Tokyo is massive. Between hi
```

**llama.cpp**

```
Since you have a $3,000 budget for a cherry blossom trip, you are in a good position to have a high-quality experience, but keep in mind that **cherry blossom season is peak travel time**, which can drive up prices for accommodation and popular dining.

A balanced split for your budget would be roughly **60% for Tokyo and 40% for Kyoto.**

Here is the reasoning behind that split:

### 1. Tokyo (The "High-Cost" Hub)
Tokyo is generally more expensive for daily expenses, especially if you enjoy the "city" side of the trip.
*   **Accommodation:** Hotels in popular areas (like Shinjuku or Shibuya) 
```

</details>
<details><summary>qwen36-27b · ggml_vulkan · text_long (similarity 0.06)</summary>

**TensorSharp**

```
The history of computing is a progression from early mechanical calculators to the foundational theoretical and architectural breakthroughs established by figures like Turing and von Neumann, which enabled the transition to electronic processing. This evolution was accelerated by Moore's Law and the development of high-level software abstractions, ultimately transforming computers from specialized tools into ubiquitous networks that support modern machine learning and artificial intelligence.
```

**llama.cpp**

```
The history of computing traces a continuous evolution from early mechanical calculators and foundational theoretical breakthroughs to the ubiquitous, highly abstracted digital infrastructure of today. Driven by relentless advances in hardware miniaturization, software abstraction, and global networking, computing transformed from a specialized, physical endeavor into an ambient utility that seamlessly underpins modern life.
```

</details>
<details><summary>gemma4-12b · ggml_vulkan · text_short (similarity 0.08)</summary>

**TensorSharp**

```
A transformer is a neural network architecture that processes sequences by using a mechanism called "attention" instead of reading inputs one element at a time. This allows every position in a sequence to directly consult and weigh the importance of every other position simultaneously. By replacing sequential processing, it enables the model to better handle long-distance relationships within data and train more efficiently.
```

**llama.cpp**

```
Introduced in 2017, the transformer architecture serves as a catalyst for the current era of machine learning. It replaces older methods of processing sequences one element at a time with a mechanism called "attention." This mechanism allows every position in a sequence to directly consult every other position to determine how much weight to give it.
```

</details>
<details><summary>qwen36-27b · ggml_cuda · multi_turn (similarity 0.10)</summary>

**TensorSharp**

```
A 60/40 split toward Tokyo makes sense. Kyoto’s charm is concentrated, so you can cover it in three days; Tokyo offers more variety for the remaining four.
```

**llama.cpp**

```
A **60/40 split** works well: roughly **$1,800 for Tokyo** and **$1,200 for Kyoto**. 

This accounts for Tokyoâs slightly higher baseline costs, its role as your arrival/departure hub, and includes your round-trip Shinkansen fare (~$50â60). Kyotoâs prices are comparable during cherry blossom season, but youâll stretch your budget further there with free temple grounds, street food, and park picnics. 

If youâre staying around a week, **4 nights in Tokyo and 3
```

</details>
<details><summary>qwen36-35b-a3b · ggml_cuda · text_short (similarity 0.10)</summary>

**TensorSharp**

```
A transformer is a neural network architecture that processes sequences by using an attention mechanism, allowing every position in the input to directly consult all other positions simultaneously. This design overcomes the limitations of earlier sequential models by enabling efficient training and better handling of long-range dependencies. Introduced in 2017, this approach became the foundational catalyst for modern artificial intelligence advancements.
```

**llama.cpp**

```
A transformer neural network is a deep learning architecture that processes entire sequences of data in parallel rather than step-by-step, using a mechanism called "attention" to dynamically weigh the relevance of each input element. This design allows the model to capture long-range contextual dependencies across the entire sequence, overcoming the memory and training bottlenecks of earlier sequential networks. By efficiently learning complex patterns from vast datasets, transformers have become the foundational architecture powering modern artificial intelligence, including large language mo
```

</details>
<details><summary>qwen36-35b-a3b · ggml_vulkan · text_long (similarity 0.10)</summary>

**TensorSharp**

```
The history of computing traces a trajectory from early mechanical aids like the abacus to modern digital systems, driven by key conceptual leaps such as Babbage’s general-purpose architecture and Turing’s theoretical framework. This evolution is characterized by continuous abstraction, where each technological advancement hides underlying complexity to create more flexible, powerful, and ubiquitous tools for human use.
```

**llama.cpp**

```
The history of computing traces a centuries-long evolution from ancient mechanical aids to modern digital systems, driven by foundational theoretical breakthroughs, hardware miniaturization, and layered software abstractions. This continuous process of abstraction has transformed computers from specialized calculating machines into ubiquitous, intelligent utilities that now power everything from everyday applications to advanced artificial intelligence.
```

</details>
<details><summary>qwen36-35b-a3b · ggml_vulkan · text_short (similarity 0.11)</summary>

**TensorSharp**

```
A transformer is a neural network architecture that processes sequential data by using an attention mechanism, allowing every position in the input to directly consult and weigh all other positions simultaneously. This design overcomes the limitations of earlier models that processed data sequentially, enabling faster training and better retention of long-range dependencies. Introduced in 2017 as a catalyst for modern machine learning, it has become foundational for advanced artificial intelligence applications like large language models.
```

**llama.cpp**

```
A transformer neural network is a deep learning architecture introduced in 2017 that replaces sequential data processing with an attention mechanism. This allows every element in a sequence to directly interact with all other elements and dynamically assign importance weights to them. By efficiently capturing long-range dependencies, this design enables highly scalable models that power modern artificial intelligence.
```

</details>
<details><summary>qwen36-27b · ggml_cuda · text_long (similarity 0.11)</summary>

**TensorSharp**

```
The history of computing is a progression from early mechanical calculators to the foundational theoretical and architectural breakthroughs established by pioneers like Babbage, Turing, and von Neumann. This evolution was accelerated by technological advancements such as transistors and networking, ultimately transforming computers into ubiquitous utilities that rely on layered abstractions and modern machine learning.
```

**llama.cpp**

```
The history of computing traces a continuous evolution from early mechanical calculators to ubiquitous digital infrastructure, driven by foundational breakthroughs in theoretical computation, hardware mini
```

</details>
<details><summary>qwen36-35b-a3b · ggml_cuda · text_long (similarity 0.13)</summary>

**TensorSharp**

```
The history of computing traces the evolution from early mechanical aids like the abacus to modern digital systems, driven by key theoretical breakthroughs such as Turing’s universal machine and von Neumann’s stored-program architecture. This progression is characterized by a continuous process of abstraction that has transformed computers from specialized, room-sized instruments into ubiquitous, ambient utilities woven into daily life.
```

**llama.cpp**

```
The history of computing traces a centuries-long evolution from simple mechanical aids to sophisticated electronic systems, unified by a relentless drive toward higher levels of abstraction. This progression has transformed the computer from a rare, specialized instrument into an essential global utility that now underpins modern artificial intelligence and digital connectivity.
```

</details>
<details><summary>gemma4-12b · ggml_cuda · text_short (similarity 0.15)</summary>

**TensorSharp**

```
A transformer is a neural network architecture that processes sequences by using a mechanism called "attention" instead of reading inputs one element at a time. This allows every position in a sequence to directly consult and weigh the importance of every other position simultaneously. By replacing sequential processing, this design enables the model to handle long-distance relationships more efficiently and serves as the catalyst for the current era of machine learning.
```

**llama.cpp**

```
Introduced in 2017, the transformer architecture is a neural network designed to process sequences more efficiently than previous models. It replaces sequential reading with a mechanism called "attention," which allows every position in a sequence to directly consult every other position. This enables the model to decide how much weight to give each part of the sequence, overcoming the speed and memory limitations of earlier systems.
```

</details>
<details><summary>gemma4-12b · ggml_cuda · text_long (similarity 0.18)</summary>

**TensorSharp**

```
The history of computing is a progression from ancient mechanical aids and 19th-century conceptual designs to the development of universal theoretical models, electronic architectures, and exponential hardware scaling. This evolution has been defined by a continuous "ladder of abstraction," where each technological leap—from stored programs and networking to modern machine learning—has simplified complexity to turn computation into an ambient global utility.
```

**llama.cpp**

```
The history of computing traces a trajectory from ancient mechanical aids and 19th-century designs to the theoretical foundations of universal machines and the practical innovations of the von Neumann architecture. This evolution is characterized by a continuous climb of abstraction, where hardware revolutions and software layers have transformed computing from a physical process into a global, intelligent utility.
```

</details>
<details><summary>qwen36-27b · ggml_vulkan · text_short (similarity 0.20)</summary>

**TensorSharp**

```
A transformer neural network is a deep learning architecture that processes input sequences in parallel rather than sequentially, allowing for faster training and better retention of long-range dependencies. It utilizes an attention mechanism where every position in the sequence directly consults every other position to determine how much weight to assign to each element based on relevance. This design enables the model to efficiently capture complex relationships within data, fundamentally transforming natural language processing capabilities.
```

**llama.cpp**

```
A transformer neural network is a deep learning architecture that processes entire input sequences simultaneously rather than step-by-step, drastically accelerating training and improving memory of distant information. It relies on an attention mechanism that allows every position in the data to directly evaluate and assign dynamic weight to its relationship with every other position. This parallel, context-aware design enables the model to capture complex long-range dependencies and patterns far more effectively than earlier sequential networks.
```

</details>
<details><summary>qwen36-27b · ggml_cuda · text_short (similarity 0.25)</summary>

**TensorSharp**

```
A transformer neural network is a deep learning architecture that processes entire sequences of data simultaneously rather than sequentially, allowing for faster training and better retention of long-range dependencies. It utilizes an "attention" mechanism where every position in the input sequence directly consults every other position to determine their relative importance or weight. This parallel processing capability enables the model to efficiently capture complex relationships across vast amounts of information, revolutionizing natural language processing tasks.
```

**llama.cpp**

```
A transformer neural network is a deep learning architecture that processes entire sequences of data simultaneously rather than step-by-step, enabling highly efficient parallel training. It relies on a self-attention mechanism that allows every element in an input to dynamically weigh its relevance to all other elements, capturing long-range dependencies and complex contextual relationships. This design overcomes the sequential bottlenecks of earlier models and serves as the foundational architecture behind modern large language models and advanced AI systems.
```

</details>
<details><summary>gemma4-12b · ggml_vulkan · text_long (similarity 0.27)</summary>

**TensorSharp**

```
The history of computing is a progression from ancient mechanical aids and 19th-century conceptual designs to the development of universal theoretical models, electronic architectures, and exponential hardware scaling. This evolution has been defined by a continuous "ladder of abstraction," where each technological leap—from stored programs and networking to modern machine learning—has simplified complexity to turn computation into an ambient global utility.
```

**llama.cpp**

```
The history of computing traces a progression from ancient mechanical aids and 19th-century conceptual designs to the theoretical foundations and hardware innovations that established the modern computer. This evolution is characterized by a continuous layering of abstraction, transforming computation from a specialized manual task into a pervasive global utility that powers today's networks and machine learning systems.
```

</details>

## Image editing (stable-diffusion)

Same input image, prompt, resolution, step count, cfg and seed for every engine. Timings are each engine's **own pipeline timers** (TensorSharp's `[pipe-timing]` phases + server `elapsedSeconds`; sd.cpp's phase logs + `generate_image` total), so weight-file loading and HTTP/process overhead are excluded on both sides. `total (warm)` is the steady-state request on an already-running server; `first request (cold)` additionally pays TensorSharp's per-request DiT rebuild + graph capture on a fresh server (a CLI engine has no such distinction). Lower is better.

_No image-edit cells were run (see the `image_edit` scenario)._

## MTP / NextN speculative decoding (on vs off)

Single-stream decode tok/s with MTP/NextN speculative decoding off vs on (TensorSharp only). Speedup `< 1.0×` means speculation cost more than it saved for that cell — expected when the fused full-model decode path is already the fast path.

_No MTP on/off pairs were run (use `--mtp off,on`)._

## Parallel-request scaling (concurrency)

`decode/req` is the mean per-request decode tok/s; `aggregate` is the system-wide decode throughput (total generated tokens / the wall window during which any sequence was decoding) when N identical requests are fired at one server at once.

_No parallel-request cells were run (use `--concurrency 1,4,8`)._

## Function-calling correctness

_No function-call cells were run._
