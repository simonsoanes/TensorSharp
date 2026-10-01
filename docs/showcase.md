# TensorSharp at work

[English](showcase.md) | [中文](showcase_zh-cn.md)

> Part of the [TensorSharp](../README.md) documentation. The README shows these screenshots in [See it in action](../README.md#see-it-in-action).

One engine, four ways to use it. Each screenshot is an unedited capture of a real run on an Apple M5 Pro (48 GB), taken on 2026-09-30.

## TensorSharp.Cli: models in your terminal

The console app runs a GGUF model from the command line: one-shot prompts, an interactive chat (`--chat`) that takes `/image`, `/audio`, `/video` and `/text` attachments, image and video generation, JSONL batches, and built-in benchmarks. Every chat reply ends with its prefill and decode timings. See the [Console Application guide](../USAGE.md#console-application).

<p align="center"><img src="../website/assets/screenshots/tensorsharp-cli.png" alt="TensorSharp.Cli in a terminal: an interactive chat with Gemma 4 E4B on Metal that reads the project README and answers two questions about it, each reply ending with its prefill and decode timings" width="700"></p>

<sub>Gemma 4 E4B (Q8_0) on `ggml_metal`, in the interactive chat. `/text` attaches the project README (40,224 characters). Both answers decode at about 40 tokens/s, and the second turn reuses the cached README, so its first token arrives in 140 ms.</sub>

## TensorSharp.Server.Host: Web UI chat and compatible APIs

The server hosts one model for a browser chat and for any Ollama or OpenAI client, on the same port. The Web UI streams Markdown answers, takes image, audio, video, PDF and text uploads, and shows the model's reasoning on request. Started with `--code-exec`, it lets the model write and run code in a sandbox and hands back the files it made. See the [Web Application guide](../USAGE.md#web-application).

<p align="center"><img src="../website/assets/screenshots/tensorsharp-webui.png" alt="The TensorSharp Web UI: Qwen3.8 27B compared two mortgages by writing and running a Python script, answered with a table and a recommendation, and offered the script and the amortization CSV as downloads" width="880"></p>

<sub>Qwen3.8 27B (UD-Q4_K_XL) on `ggml_metal`, started with `--code-exec`. The model wrote a Python script, ran it in the macOS sandbox, checked that both schedules end at a $0 balance, and returned the script and the 540-row CSV as downloads. The stats line counts every token the turn generated, reasoning and tool calls included.</sub>

## TensorAgent on iPhone: a private agent in your pocket

TensorAgent is a native iPhone and iPad app on the same engine. It downloads a model once, then runs chat, photo and file questions, dictation, and agent work in built-in Python and JavaScript runtimes on the device; by default nothing leaves the phone. See the [TensorAgent README](../TensorAgent/README.md).

<p align="center"><img src="../website/assets/screenshots/tensoragent-iphone.png" alt="TensorAgent on an iPhone: Gemma 4 E2B scaled a recipe from 4 to 10 people by running a Python script in the app's built-in Python, and answered with a table" width="300"></p>

<sub>Gemma 4 E2B (Q8_0) in the iPhone 17 Pro simulator. The simulator has no GPU, so the engine runs on `ggml_cpu` there; on an iPhone it uses Metal. The model wrote a short script, ran it in the app's built-in Python, and answered with the table.</sub>

## TensorAgent on the desktop: macOS and Windows

The same project builds a Mac app, and a Windows app that has not yet been built or run. On the Mac, the model's code runs as real `bash`, `python3` and `node` processes confined by the macOS sandbox, and the Playwright skill can drive a browser. See [TensorAgent on the desktop](../TensorAgent/README.md#on-the-desktop-macos-and-windows).

<p align="center"><img src="../website/assets/screenshots/tensoragent-mac.png" alt="TensorAgent on a Mac: Qwen3.5 9B wrote a Python file with a Roman-numeral converter and unit tests, ran them, fixed the function when the first run failed, and reran until all five tests passed" width="880"></p>

<sub>Qwen3.5 9B (IQ4_XS) on Metal. The model wrote a Python file with a Roman-numeral converter and its unit tests, and ran it with the Mac's own `python3` inside the macOS sandbox. The first run failed, so it fixed the function (one malformed patch was rejected on the way) and reran until all five tests passed.</sub>
