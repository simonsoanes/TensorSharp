# 功能特性
[English](FEATURES.md) | [中文](FEATURES_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分。


- **多架构支持** —— Gemma 4、Gemma 3、DiffusionGemma、Qwen 3、Qwen 3.5/3.6-family、GPT OSS、Nemotron-H、Mistral 3，以及 Qwen-Image-Edit（图像编辑）
- **多模态推理** —— 图像、视频和音频输入（Gemma 4）；图像输入（Gemma 3 / Qwen 3.5-family / Mistral 3 / Nemotron-H Omni）
- **思维链 / 推理模式** —— 通过 `<think>` / `<|channel>thought` / `<|channel>analysis` 标签输出结构化的思维链推理（Qwen 3、Qwen 3.5/3.6-family、Gemma 4、GPT OSS、Nemotron-H）
- **工具调用 / 函数调用** —— 模型可调用用户定义的工具；所有三种 API 风格均支持多轮工具调用对话
- **量化模型支持** —— 加载 Q4_K_M、Q8_0、F16、MXFP4 等量化格式的 GGUF 文件；执行原生量化矩阵乘法（matmul），无需反量化到 FP32，并且纯 C# CPU 后端在加载大型 GGUF 时也会保持量化权重压缩状态
- **GPU 加速** —— 通过 GGML 支持 Apple Metal（macOS）、GGML CUDA（Windows/Linux + NVIDIA）和 GGML Vulkan（Windows/Linux + AMD/Intel/NVIDIA），并提供 Direct CUDA/cuBLAS 后端（含 PTX 内核与未覆盖算子的 CPU 回退），以及面向 Apple Silicon 的 MLX 后端（mlx-c / Metal）
- **优化后的纯 C# CPU 后端** —— 为 GEMM、RMSNorm、RoPE、softmax、融合激活等推理热点路径提供托管快速路径和 SIMD 内核
- **连续批处理 & 分页 KV 缓存** —— vLLM 风格的分页 KV 块池，跨请求的块级哈希前缀共享，迭代级调度器（可在批内动态加入/抢占序列），可选的 SSD 冷层用于超大 KV 工作集，原生融合分页注意力内核（`TSGgml_PagedAttentionForward`，在 Metal/CUDA/Vulkan 上驱动 `ggml_flash_attn_ext`）。`TensorSharp.Server` 默认启用，可用 `--no-continuous-batching` 关闭。详见 [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。
- **MTP / NextN 投机解码** —— 多 token 预测草稿头加速单序列（无并发）decode。Qwen 3.6 将 NextN 块内嵌在主干 GGUF 中；Gemma 4 通过 `--mtp-draft-model` 加载独立的 EAGLE 风格 `gemma4-assistant` 草稿 GGUF，其草稿层读取目标模型自身的 KV 缓存。草稿头每步最多提议 `--mtp-draft` 个 token（草稿置信度 ≥ `--mtp-pmin` 时保留），主干用一次批量前向完成验证；起草与验证均由该请求自己的采样器（含惩罚项）驱动，因此输出与标准 decode 完全一致。服务端通过 `--mtp-spec` 启用（默认关闭）；CLI 没有 MTP 参数，需设置 `TS_MTP_*` 环境变量。ggml 后端有融合的多 token 验证 / 草稿步内核，是明确收益；纯 C# `cuda` 后端运行完全驻留 GPU 的逐算子验证 / 草稿，同样有收益；CPU / MLX 保持标准 decode。环境变量：`TS_MTP_*`（通用）与 `TS_GMTP_*`（Gemma 4 调优）。
- **批处理 / 并行推理** —— 已为 Mistral 3、Gemma 4、GPT OSS、Qwen 3、Qwen 3.5/3.6-family、Nemotron-H 默认启用 `IBatchedPagedModel.ForwardBatch`，能在一次前向传播中打包 N 个序列，使用 `slotMapping` 进行分页 K/V 写入，并通过原生内核做按序列注意力。Gemma 4、Qwen 3.5/3.6、GPT OSS 与 Nemotron-H 提供各自的 `TS_<FAMILY>_BATCHED=0` 兜底开关；Qwen 3 与 Mistral 3 没有家族专属开关，请用全局 `TS_SCHED_DISABLE_BATCHED=1` 强制回到按序列 KV-swap 路径。
- **兼容 Ollama 与 OpenAI API** —— 可作为现有工具链的即插即用替代端点
- **可配置采样** —— temperature、top-k、top-p、min-p、重复/存在/频率惩罚、seed、停止序列
- **聊天模板** —— 从 GGUF 元数据自动加载（Jinja2），并为不同架构提供硬编码回退模板
- **推理引擎** —— `TensorSharp.Server` 中的新 `InferenceEngine`（工作线程调度器 + 分页块池）取代了旧的单请求 FIFO 队列。旧队列对象现在只是状态 / 事件形状的兼容 shim；引擎本身已经处理并发。
- **批处理** —— 控制台应用支持 JSONL 输入，并内置用于测量 prefill / decode 吞吐的推理基准
- **流式输出** —— 按 token 输出（Web 通过 SSE，控制台通过 stdout），并支持中断/停止正在生成的请求
- **文本扩散生成** —— DiffusionGemma 使用 EntropyBound 迭代去噪采样器，而不是自回归 `Forward()`。CLI 提供 `--diffusion-steps`、`--diffusion-seed` 与 `--diffusion-blocks`；Web UI 使用整条消息 `replace` 事件展示实时去噪预览，并通过 `DiffusionBatchScheduler` 批处理并发扩散请求。
- **图像编辑（Qwen-Image-Edit）** —— 提示词加输入图像生成编辑后的图像。所加载的 `qwen_image` GGUF 是 MMDiT 扩散 Transformer；TensorSharp 在其旁解析两个伴随 GGUF——Qwen-Image VAE（图像 ↔ 16 通道潜变量）与 Qwen2.5-VL-7B 文本编码器（提示词 → 3584 维条件，可选通过 `mmproj` 做视觉接地）。流水线对参考图做 VAE 编码、构建文本（及可选图像）条件、运行带参考潜变量拼接的 FlowMatch-Euler true-CFG 去噪循环，再 VAE 解码回像素。整个 60 块 DiT 前向被 CUDA 图捕获（`TSGgml_QwenImageForward`），flash 注意力默认开启，目标面积按设备 VRAM 预算自动钳制。可选的 Lightning 蒸馏 LoRA（`--qwen-image-lora` / `TS_QWEN_IMAGE_LORA`，`.safetensors`）会在加载时合并进 DiT 权重，将去噪步数缩减为该 LoRA 的步数（例如 4 或 8），并把 CFG 切换为 1.0（无负向分支）。可从 C# 通过 `QwenImageModel.EditImage(prompt, RgbImage, QwenImageParams)` 驱动，从 CLI 图像编辑模式（`--image`、`--prompt`、`--cfg`、`--diffusion-steps`、`--diffusion-seed`）驱动，以及从带实时去噪预览的 Web UI 驱动。→ [Qwen-Image-Edit 卡片](docs/models/qwenimage_zh-cn.md)
- **混合 SSM-Transformer** —— Nemotron-H 在单个模型中混合 Mamba2 SSM 层、纯注意力层和 MoE FFN 层；Mamba2 步现在同时提供单序列原生内核与批处理原生内核（`TSGgml_NemotronMamba2BatchedStepF32`，NEON SIMD + GCD 并行）。
- **混合注意力-递归网络** —— Qwen 3.5/3.6-family 在同一模型中混合全注意力层与 GatedDeltaNet 递归层；批处理路径下递归运行状态保存在每槽位的递归状态池中
- **专家混合（MoE）** —— 支持 Gemma 4 MoE 变体（例如 gemma-4-26B-A4B）、GPT OSS MoE（例如 gpt-oss-20b）、Qwen 3.5/3.6-family MoE（`qwen35moe` / `qwen3next` 变体，例如 Qwen3.5-35B-A3B）以及 Nemotron-H MoE FFN 层
- **批量 GPU MoE** —— Qwen 3.5/3.6-family 与 Nemotron-H 在 decode 时通过单次融合的 GGML 计算图调度处理所有被选中的专家（Qwen 3.5-family 还包括可选的 shared expert 与残差加法），消除每个专家的 CPU-GPU 往返
- **KV 缓存编解码器** —— 通过 `IKvBlockCodec` 接口插件化；内置 TurboQuant（2-bit 仿射 / Q4 / Q8）分页块压缩。CLI 的 `--paged-kv-quant-bits` 接受 `0|2|4|8`；服务端旧式独立分页参数接受 `0|4|8`，也可直接用 `TS_KV_PAGED_QUANT_BITS=2` 选择 2-bit 编解码器。2-bit 档位在 fp32 块上可达约 10 倍压缩，面向超长上下文。
- **消息编辑** —— 在 Web 聊天界面中编辑或删除历史消息，并从该位置重新生成回复
- **文本/图像/音频/视频/PDF 上传** —— Web 界面支持最大 500 MB 的文件上传并完整保留文本内容；原生数字 PDF 会完整提取文本层（可通过 `TS_PDF_MAX_PAGES` 显式限制页数）。最终提示词按模型的实际上下文窗口检查，而不是使用任意的上传预算
- **每轮可观测性** —— 结构化日志会完整保留用户输入与模型原始输出（包括 `<think>` 思维链和最终结果），并记录 KV 缓存命中率。同样的命中率指标通过所有 API 透出：Ollama 的 `prompt_cache_hit_tokens` / `prompt_cache_hit_ratio`、OpenAI 的 `usage.prompt_tokens_details.cached_tokens`，以及 Web UI SSE `done` 事件中的 `promptTokens` / `kvReusedTokens` / `kvReusePercent`


## 思维链 / 推理模式

支持思维链模式的模型（Qwen 3、Qwen 3.5/3.6-family、Gemma 4、GPT OSS、Nemotron-H）可以在生成最终答案之前产出结构化的思维链推理内容。思维内容与主要回复分开，客户端可选择显示或隐藏。

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H：** 使用 `<think>...</think>` 标签
- **Gemma 4：** 使用 `<|channel>thought\n...<channel|>` 标签
- **GPT OSS：** 使用 Harmony 格式，以 `<|channel|>analysis` 标记思维过程，以 `<|channel|>final` 标记最终回复

通过 `--think`（控制台）、`"think": true`（Ollama API）或 Web 界面中的思维链开关启用。

## MTP / NextN 投机解码

部分架构自带**多 token 预测（MTP / NextN）草稿头**，让 `TensorSharp.Server` 能为单序列（无并发）请求运行无损投机解码。草稿头廉价地提议若干未来 token，主干用一次批量前向验证全部 token，被接受的 token 一步提交。由于起草与验证都由该请求自己的采样器（temperature、top-k/p、重复/存在/频率惩罚）驱动，输出与标准 decode 完全一致——投机只改变产生这些 token 所需的前向次数。

投机解码**默认关闭**。在服务端通过 `--mtp-spec`（环境变量 `TS_MTP_SPEC=1`）启用：

```bash
# Qwen 3.6 —— 使用 -MTP- 仓库 GGUF，确保主干保留内嵌 NextN 块
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda \
    --mtp-spec --mtp-draft 8 --mtp-pmin 0.75

# Gemma 4 —— 加载与目标匹配的独立 gemma4-assistant 草稿 GGUF
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
    --mtp-spec --mtp-draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

**两种草稿头形态：**

- **Qwen 3.6（内嵌 NextN）** —— GGUF 在主干栈之后带有一个额外解码块（`{arch}.nextn_predict_layers`）以及 NextN 投影 / 归一化张量。无需独立文件，`--mtp-draft-model` 被忽略。主干的递归状态（GatedDeltaNet）会被快照，以便部分被拒的验证批次可以回滚。
- **Gemma 4（独立 `gemma4-assistant` GGUF）** —— 通过 `--mtp-draft-model` 加载的 EAGLE 风格递归草稿器。它自身不保存任何 K/V：每个草稿层都查询**目标模型**已有的逐层 KV 缓存（最后一个 local 层 + 最后一个 global 层），因此在给定 `(token, hidden)` 时草稿器是无状态的。草稿的隐藏维度必须与目标一致——12B 目标配 12B 草稿，而非 26B-A4B 草稿。草稿 GGUF 不匹配、缺失或不完整会在启动时**立即失败**并给出修复提示，而非静默关闭投机。

**何处有收益**（自动启用；否则引擎走标准 decode）：

| 后端 | Qwen 3.6 | Gemma 4 |
|---|---|---|
| GGML CUDA / GGML Metal | ✅ 融合多 token 验证 + 草稿步内核 | ✅ 融合多 token 验证 + 草稿步内核 |
| Direct CUDA（`cuda`，纯 C#） | ✅ 完全驻留 GPU 的逐算子验证 / 草稿 | ✅ 完全驻留 GPU 的逐算子验证 / 草稿 |
| CPU / GGML CPU / MLX | 标准 decode（验证跟不上） | 标准 decode |

调优：`--mtp-draft`（默认 `8`）限制每步起草的 token 数；`--mtp-pmin`（默认 `0.75`）是保留 token 所需的最低草稿置信度（遇到第一个低置信 token 即停止起草）。Gemma 4 草稿路径 A/B 开关为 `TS_GMTP_*` 环境变量（见 [Web 应用](USAGE_zh-cn.md#web-应用) 下的 **MTP / 投机解码调优变量** 表）。各架构具体机制见 [Qwen 3.5/3.6 卡片](docs/models/qwen35_zh-cn.md) 与 [Gemma 4 卡片](docs/models/gemma4_zh-cn.md)。

## 工具调用 / 函数调用

模型可以调用用户定义的工具并参与多轮工具调用对话。将工具定义为 JSON 格式，通过 `--tools`（控制台）或 API 中的 `tools` 参数传入。

各架构使用各自的工具调用格式：

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H：** `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Gemma 4：** `<|tool_call>call:function_name{args}<tool_call|>`
- **GPT OSS（Harmony）：** 工具以 TypeScript namespace 形式声明在 developer 消息中，调用通过 commentary channel 输出：`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`

输出解析器（`OutputParser.cs`）会自动从模型原始输出中提取工具调用，与架构无关。

## 多模态支持

### Gemma 4

Gemma 4 模型支持图像、视频和音频输入。上文 E4B 示例使用同仓库的 `mmproj-gemma-4-E4B-it-Q8_0.gguf`；请通过 `--mmproj` 显式传入（其他目标尺寸使用各自匹配的投影器）。

- **图像：** PNG、JPEG、HEIC/HEIF
- **视频：** MP4（使用 OpenCV 以 1 fps 基于时间抽帧；可通过 `VIDEO_SAMPLE_FPS` / `VIDEO_MAX_FRAMES` 调整）
- **音频：** WAV（16kHz 单声道）、MP3、OGG Vorbis

### Gemma 3

Gemma 3 支持 PNG、JPEG 与 HEIC/HEIF 图像输入。上文非 gated 示例使用 `mmproj-model-f16.gguf`；请通过 `--mmproj` 显式传入。

### Qwen 3.5 / 3.6 family

所有 Qwen 3.5/3.6-family 变体（`qwen35`、`qwen35moe` 与 `qwen3next`）共用同一个 `Qwen35Model` 实现。图像输入通过支持动态分辨率的 `Qwen35VisionEncoder` 处理；请显式传入所选仓库的投影器（上文 9B 与 Qwen 3.6 示例均为 `mmproj-F16.gguf`）。MoE 变体（例如 Qwen3.5-35B-A3B，以及使用同一架构标识的 Qwen3.6-35B-A3B GGUF）在 decode 时还会启用融合的 `MoEExpertsSwiGLUResidual` GGML 内核，将所有被选中的专家、可选的 shared expert 与残差加法合并到一次 GPU 计算图调度中执行。

### Mistral 3

Mistral 3 通过 Pixtral 视觉编码器支持图像输入。示例仓库使用 `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf`；请通过 `--mmproj` 显式传入。

- **图像：** PNG、JPEG、HEIC/HEIF

### Nemotron-H（Omni 发行版）

Nemotron Omni 发行版加入了 RADIO / v2_vl ViT 图像编码器。通过 `--mmproj` 传入对应的多模态投影器（例如 `nvidia_Nemotron-H-Omni-mmproj.gguf`）即可启用；语言模型 GGUF 不变。图像 token 在 `<image>` 占位符处插入，并由多模态注入器自动展开为 `<img>` + N 个 tile token + `</img>`。

- **图像：** PNG、JPEG、HEIC/HEIF
- **音频：** 聊天模板会为每个上传的音频文件发出一个 `<so_embedding>` token，CLI 仍会运行 Parakeet 风格 log-mel 预处理器以验证管线，但真正的音频推理需要尚未在公开 GGUF 中发布的 Parakeet 音频 mmproj。

