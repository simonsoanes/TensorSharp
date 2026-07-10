# 模型架构卡片

[English](model_cards.md) | [中文](model_cards_zh-cn.md)

> 本文件已被拆分。每个模型现在都在 [`docs/models/`](models/README_zh-cn.md)
> 下有独立卡片，每张英文卡片旁边都有对应的中文版本。

通过下表可以直接跳到具体架构，或阅读
[`docs/models/README_zh-cn.md`](models/README_zh-cn.md) 了解完整的实现矩阵
与特性对比。

每张当前卡片开头都列有已核对的 Hugging Face 下载入口、精确示例文件名，以及可复制的
`TensorSharp.Cli` / `TensorSharp.Server` 命令。已验证的快速上手路径（见仓库
[README](../README_zh-cn.md#快速开始)）是 TensorSharp 的 Gemma 4 E4B Q8_0
原生 GGML 家族 / 路径层级：使用推荐公开
[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)
中的 `gemma-4-E4B-it-Q8_0.gguf`，后端选择 `ggml_cuda`、`ggml_metal` 或
`ggml_vulkan`；详见 [Gemma 4 卡片](models/gemma4_zh-cn.md#已验证的-gemma-4-e4b-原生-ggml-快速路径)。
纯文本不需要 `mmproj`；图像、视频或音频输入需要匹配的 `mmproj`。这里不声称基准输入
对应某个公开文件的特定校验和。

| 架构 | 中文卡片 | English card |
|---|---|---|
| Gemma 3 | [models/gemma3_zh-cn.md](models/gemma3_zh-cn.md) | [models/gemma3.md](models/gemma3.md) |
| Gemma 4 | [models/gemma4_zh-cn.md](models/gemma4_zh-cn.md) | [models/gemma4.md](models/gemma4.md) |
| DiffusionGemma | [models/diffusiongemma_zh-cn.md](models/diffusiongemma_zh-cn.md) | [models/diffusiongemma.md](models/diffusiongemma.md) |
| Qwen 3 | [models/qwen3_zh-cn.md](models/qwen3_zh-cn.md) | [models/qwen3.md](models/qwen3.md) |
| Qwen 3.5 / 3.6 family | [models/qwen35_zh-cn.md](models/qwen35_zh-cn.md) | [models/qwen35.md](models/qwen35.md) |
| GPT OSS | [models/gptoss_zh-cn.md](models/gptoss_zh-cn.md) | [models/gptoss.md](models/gptoss.md) |
| Nemotron-H | [models/nemotron_zh-cn.md](models/nemotron_zh-cn.md) | [models/nemotron.md](models/nemotron.md) |
| Mistral 3 | [models/mistral3_zh-cn.md](models/mistral3_zh-cn.md) | [models/mistral3.md](models/mistral3.md) |
| Qwen-Image-Edit | [models/qwenimage_zh-cn.md](models/qwenimage_zh-cn.md) | [models/qwenimage.md](models/qwenimage.md) |

每张卡片会把工程师或研究员从“从未听说过这个模型”带到“可以解释它的前向计算图，
并能在 TensorSharp 中复现推理路径”，统一覆盖：

1. 已核对的下载入口与可运行 CLI / Server 命令
2. 来源与目标（提供方、GGUF 架构标识、模态、思维链、工具调用）
3. 模型架构（顶层模块图、层数、每层异构性）
4. 前向计算图（per-token decode 与多 token prefill 中算子的精确顺序）
5. 组件细节（attention、FFN/SSM、routing、normalization、RoPE 变体、视觉/音频编码器）
6. 参数与配置（GGUF 元数据 key、权重张量命名、dtype 要求）
7. TensorSharp 实现走读
8. Prefill 优化
9. Decode 优化
10. 内存与 KV cache 策略
11. 多模态管线（如适用）
12. 输出解析器与聊天模板
13. 优化机会

新增架构时，请按
[`docs/models/README_zh-cn.md`](models/README_zh-cn.md#新增模型架构) 中的清单操作。
