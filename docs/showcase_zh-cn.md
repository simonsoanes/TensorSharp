# TensorSharp 实际运行截图

[English](showcase.md) | [中文](showcase_zh-cn.md)

> [TensorSharp](../README_zh-cn.md) 文档的一部分。README 在[实际运行效果](../README_zh-cn.md#实际运行效果)中展示了这几张截图。

同一个引擎，四种用法：终端推理、浏览器聊天与代码任务、移动智能体和桌面图像编辑。CLI、Web UI 与 iPhone 模拟器截图于 2026-09-30 在 Apple M5 Pro（48 GB）上拍摄。桌面截图于 2026-10-02 展示当前 Mac 应用中已保存的 Qwen-Image-2.1 编辑结果，不是一次新的推理基准。

## TensorSharp.Cli：在终端里运行模型

控制台应用在命令行中运行 GGUF 模型：一次性提示词；交互式聊天（`--chat`），可用 `/image`、`/audio`、`/video`、`/text` 附加文件；图像与视频生成；JSONL 批处理；以及内置基准测试。每条聊天回复末尾都会给出预填充与解码耗时。详见[控制台应用指南](../USAGE_zh-cn.md#控制台应用)。

<p align="center"><img src="../website/assets/screenshots/tensorsharp-cli.png" alt="终端中的 TensorSharp.Cli：Gemma 4 E4B 在 Metal 上的交互式聊天，读取项目的 README 并回答两个问题，每条回复末尾给出预填充与解码耗时" width="700"></p>

<sub>Gemma 4 E4B（Q8_0）运行于 `ggml_metal` 的交互式聊天。`/text` 附加了项目的 README（40,224 个字符）。两次回答的解码速度都约为每秒 40 个 token；第二轮复用已缓存的 README，首个 token 在 140 毫秒内返回。</sub>

## TensorSharp.Server.Host：Web UI 聊天与兼容 API

服务器在同一端口上托管一个模型，同时供浏览器聊天和任意 Ollama / OpenAI 客户端使用。Web UI 以 Markdown 流式显示回答，支持上传图像、音频、视频、PDF 与文本，并可按需展开模型的推理过程。以 `--code-exec` 启动时，模型可以在沙箱中编写并运行代码，再把生成的文件交还给你。详见 [Web 应用指南](../USAGE_zh-cn.md#web-应用)。

<p align="center"><img src="../website/assets/screenshots/tensorsharp-webui.png" alt="TensorSharp Web UI：Qwen3.8 27B 编写并运行 Python 脚本比较两种房贷，用表格和一句建议作答，并提供脚本与还款计划 CSV 下载" width="880"></p>

<sub>Qwen3.8 27B（UD-Q4_K_XL）运行于 `ggml_metal`，以 `--code-exec` 启动。模型编写了 Python 脚本，在 macOS 沙箱中运行，核对两份还款计划都在余额 $0 处结束，并把脚本和 540 行的 CSV 作为下载返回。统计行计入本轮生成的全部 token，包括推理与工具调用。</sub>

## iPhone 上的 TensorAgent：装进口袋的私有智能体

TensorAgent 是基于同一引擎的原生 iPhone / iPad 应用。模型只需下载一次，之后聊天、照片与文件问答、语音输入，以及在内置 Python 和 JavaScript 运行时中完成的智能体任务都在设备上运行；默认情况下没有任何内容离开手机。详见 [TensorAgent README](../TensorAgent/README.md)。

<p align="center"><img src="../website/assets/screenshots/tensoragent-iphone.png" alt="iPhone 上的 TensorAgent：Gemma 4 E2B 在应用内置的 Python 中运行脚本，把食谱从 4 人份换算为 10 人份，并以表格作答" width="300"></p>

<sub>Gemma 4 E2B（Q8_0）运行在 iPhone 17 Pro 模拟器中。模拟器没有 GPU，因此引擎在那里使用 `ggml_cpu`；在 iPhone 真机上则使用 Metal。模型写了一个简短的脚本，在应用内置的 Python 中运行，并以表格作答。</sub>

## 桌面版 TensorAgent：macOS 与 Windows

TensorAgent 在同一会话界面中提供文本和照片/文件聊天、智能体代码与浏览器任务、图像生成/编辑，以及带音频的视频生成。Mac 模型目录包含 Qwen-Image-2.1、可选 LoRA 插件和局部区域编辑器，以及 MiniMax-H3；可用模型取决于内存容量。在 Mac 上，代码以真实的 `bash`、`python3`、`node` 进程运行，并受 macOS 沙箱约束；Playwright 技能还可以驱动浏览器。同一工程的 Windows 应用头已有有限的聊天与工具验证，图像/音频/视频生成尚未验证。详见[桌面版 TensorAgent](../TensorAgent/README.md#on-the-desktop-macos-and-windows)。

<p align="center"><img src="../website/assets/screenshots/tensoragent-mac.png" alt="Mac 上的 TensorAgent：已保存的 Qwen-Image-2.1 图像编辑把 TensorSharp 横幅背景改为繁星蓝色夜空，提供原图对比与再次编辑操作" width="880"></p>

<sub>当前 Mac Catalyst 应用展示已保存的 Qwen-Image-2.1（Q4_K_M）编辑结果，设备为 M5 Pro（48 GB），后端为 GGML Metal。提示词为“把背景改成带星星的深蓝色夜空，保持文字不变”。结果提供 Compare original（原图对比）与 Edit again（再次编辑）操作。本次截图展示已完成的会话，不测量新的编辑耗时，也不演示遮罩或 LoRA。</sub>
