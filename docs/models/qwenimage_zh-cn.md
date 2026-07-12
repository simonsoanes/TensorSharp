# Qwen-Image-Edit

[← 返回模型索引](README_zh-cn.md)

## 状态快照

| 字段 | 状态 |
|---|---|
| GGUF 架构键 | `qwen_image`（MMDiT 扩散 Transformer） |
| 源类 | [`QwenImageModel`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs)（+ 内部 [`QwenImagePipeline`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs)） |
| 任务 | 图像编辑——提示词 + 输入图像（可多张）→ 编辑后的图像 |
| 模态 | 图像（多张）输入 + 文本输入 → 图像输出（多图合成遵循 `QwenImageEditPlusPipeline`："Picture 1"、"Picture 2"…） |
| 思考 / 工具 | 不适用 |
| 生成方式 | FlowMatch-Euler 扩散去噪（true-CFG），**非**自回归 token 解码 |
| CLI 支持 | `TensorSharp.Cli` 检测到 `QwenImageModel` 后进入图像编辑模式（`--image` 可重复以进行多图编辑、`--prompt`、`--output`、`--cfg`、`--diffusion-steps`、`--diffusion-seed`、`--width`/`--height` 强制输出尺寸、`--offload-cpu` 总是从内存流式上传 DiT 权重） |
| 服务器支持 | Web UI 图像编辑流程：`POST /api/image-edit` 与 `POST /api/image-edit/stream`（SSE，含实时去噪预览）；两者都接受多张图像（multipart 多个 `image` 部件 / JSON `imagePaths[]`） |
| 连续批处理 | 无——图像编辑串行执行（扩散网络非线程安全），并发请求逐个处理 |

## 下载

图像编辑需要一套**四组件**文件。DiT GGUF 即 `--model`；伴随文件从同一目录自动
解析，也可以用 `--qwen-image-vae` / `--qwen-image-vl` / `--qwen-image-mmproj` /
`--qwen-image-lora`（CLI 与服务器）显式指定——等价的环境变量为
`TS_QWEN_IMAGE_VAE` / `TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ` /
`TS_QWEN_IMAGE_LORA`：

| 组件 | HF 仓库 | 文件 |
|---|---|---|
| MMDiT DiT（即 `--model` GGUF） | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) | `qwen-image-edit-2511-Q4_K_M.gguf`（提供 Q2_K…Q8_0 量化阶梯；`Q2_K` 可放进 16 GB VRAM） |
| Qwen-Image VAE | [QuantStack/Qwen-Image-Edit-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) | `VAE/Qwen_Image-VAE.safetensors`（254 MB） |
| Qwen2.5-VL-7B 文本编码器 | [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) | `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`（低 VRAM 可用 `UD-IQ2_XXS`） |
| 视觉投影器（可选——图像接地条件） | 同一仓库 | `mmproj-BF16.gguf` |
| Lightning LoRA（可选——4/8 步编辑） | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` |

```bash
# needs: pip install -U huggingface_hub
hf download unsloth/Qwen-Image-Edit-2511-GGUF qwen-image-edit-2511-Q4_K_M.gguf --local-dir models
hf download QuantStack/Qwen-Image-Edit-GGUF VAE/Qwen_Image-VAE.safetensors --local-dir models
hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --local-dir models
hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF mmproj-BF16.gguf --local-dir models
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --local-dir models
```

`hf download` 会保留仓库中的 `VAE/` 子目录——请把 `Qwen_Image-VAE.safetensors`
移到 DiT GGUF 旁边（自动解析在 DiT 所在目录中查找这个确切文件名），或改用
`--qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors` 指定。

CLI 编辑（当 `--model` 是 `qwen_image` GGUF 时自动进入图像编辑模式；`--image`
必填，编辑指令写在 `--prompt` 中）：

```bash
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/qwen-image-edit-2511-Q4_K_M.gguf --backend ggml_cuda \
  --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors \
  --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --image input.png --prompt "Replace the sky with a golden sunset" \
  --output edited.png --cfg 2.5 --diffusion-steps 30
```

（不传 `--cfg` / `--diffusion-steps` 则走自动：30 步 / cfg 2.5；通过
`--qwen-image-lora` 加载 Lightning LoRA 时为其训练步数 / cfg 1.0。）

**多图编辑**——重复 `--image` 可合成多张输入；第一张图决定输出尺寸/比例，
每张图按传入顺序成为提示词可引用的 "Picture N"（图 N）参考：

```bash
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/qwen-image-edit-2511-Q4_K_M.gguf --backend ggml_cuda \
  --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors \
  --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --image model-photo.heic --image dress.png \
  --prompt "请为模特换上图中的衣服" --output edited.png
```

服务器（`http://localhost:5000/index.html` 的 Web UI 会把图像 + 提示词路由到
`POST /api/image-edit/stream`——附上一张图片并输入编辑指令即可）：

```bash
dotnet run --project TensorSharp.Server -c Release -- \
  --model models/qwen-image-edit-2511-Q4_K_M.gguf --backend ggml_cuda \
  --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors \
  --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```

## 1. 来源与意图

Qwen-Image-Edit 是一个指令驱动的图像编辑器。与自回归 LLM 不同，所加载的
`qwen_image` GGUF **仅**是 MMDiT（多模态扩散 Transformer）；图像编辑还需要两个
DiT GGUF 本身不含的网络：

- **Qwen-Image VAE**——图像 ↔ 16 通道潜变量（空间 8× 下采样），以及
- **Qwen2.5-VL-7B 文本编码器**——提示词 → 3584 维条件，外加可选的 `mmproj`
  视觉塔以实现图像接地（image-grounded）条件。

`ModelBase.Create()` 将 `general.architecture = qwen_image` 路由到
`QwenImageModel`。伴随 GGUF 从 DiT GGUF 所在目录解析，或通过
`TS_QWEN_IMAGE_VAE` / `TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ` 环境变量指定
（VAE 可为原始 `.safetensors`，直接加载并将 BF16→F32 上转）。`QwenImageModel`
不是 `IModelArchitecture` 文本生成器：`Forward()` 抛异常，编辑通过
[`EditImage()`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) 驱动。

## 2. 流水线（去噪循环）

[`QwenImagePipeline.Edit`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs)
编排完整的编辑过程：

```text
输入图像（可多张，第一张决定输出尺寸/比例）
  -> 预处理（保持纵横比；尺寸为 16 的倍数；面积按 VRAM 预算钳制，最高至原生 ~1 MP；
     其余参考图各自保持纵横比，面积至多 min(原生, 输出面积)）
  -> 逐张 VAE 编码 -> 归一化（diffusers latents 均值/标准差）-> pack 为 [refSeq_i, 64]
  -> 文本条件：Qwen2.5-VL 编码(提示词 [+ 每张图的 "Picture N" 视觉接地 token])
       （CfgScale > 1 时再编码负向提示词，用于 true-CFG；视觉嵌入在两趟间缓存复用）
  -> 释放文本 + 视觉编码器（为 DiT 回收 VRAM）
  -> 噪声潜变量（带种子高斯）pack 为 [genSeq, 64]
  -> 拼接 [生成 | 参考1 | 参考2 | ...] token（modulateIndex 标记所有参考 token；
     DiT RoPE 给每个流独立的 frame 索引 0,1,2,...，空间网格居中）
  -> FlowMatch-Euler 调度器（timestep == sigma），First-Block-Cache 复位
  -> 每步：
       DiT 速度预测（cond）[+ neg -> true-CFG 合并 + 逐行重归一化]
       scheduler.Step(latents, v, step)
       通过 OnStep 回调可选地输出解码的 RGB 预览（降采样）
  -> unpack -> 反归一化 -> VAE 解码 -> 输出图像
```

true-CFG 逐 token 行合并有条件与无条件速度（`comb = neg + scale·(cond − neg)`），
并将每行重归一化回有条件范数。当 `CfgScale <= 1` 时跳过负向分支（每步只跑一次
DiT 前向，而非两次）。

## 3. MMDiT 架构常量

`qwen_image` GGUF 不携带超参数；它们固定在
[`QwenImageModel`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) 中：

| 常量 | 值 |
|---|---|
| 隐藏维度 | 3072 |
| 层数（双流块） | 60 |
| 注意力头数 | 24（头维度 128） |
| DiT 输入通道 | 64（16 潜通道 × 2×2 patch） |
| 文本条件维度 | 3584（Qwen2.5-VL 隐藏维度） |
| VAE 潜通道 | 16 |
| VAE 缩放因子 | 8 |
| RoPE 轴维度 | {16, 56, 56}（和 = 头维度 128） |
| Eps | 1e-6 |

每个 MMDiT 块以多模态 RoPE 联合关注图像与文本 token；timestep/文本调制通过逐
token 的 `modulateIndex`（0 = 生成半区，1 = 参考半区）施加。参考潜变量被拼接进
token 序列，使编辑接地于原图。

## 4. TensorSharp 实现

| 组件 | 文件 |
|---|---|
| 模型入口 / 伴随文件解析 | [`QwenImageModel.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) |
| 去噪编排 | [`QwenImagePipeline.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs) |
| 生成参数 | [`QwenImageParams.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageParams.cs) |
| MMDiT（图/缓存/CFG-batch/native/整模型） | `QwenImageDiT*.cs` |
| FlowMatch-Euler 调度器 | [`QwenImageScheduler.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageScheduler.cs) |
| VAE（+ 参考数学） | [`QwenImageVae.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageVae.cs) |
| Qwen2.5-VL 文本编码器 | [`QwenImageTextEncoder.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageTextEncoder.cs) |
| 视觉塔（接地） | [`QwenImageVisionEncoder.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageVisionEncoder.cs) |
| 图像解码 / 编码（PNG） | [`ImageIO.cs`](../../TensorSharp.Models/Models/QwenImage/ImageIO.cs) |
| 融合 native 内核 | [`ggml_ops_qwen_image.cpp`](../../TensorSharp.GGML.Native/ggml_ops_qwen_image.cpp) |

## 5. 加速状态

- **CUDA 图捕获的整 DiT 前向**（`TSGgml_QwenImageForward`）：权重一次性常驻，整个
  60 块前向被捕获进单个可重放的图并配专用 `gallocr`。这把去噪从受启动开销限制
  （~40% GPU）变为受计算限制（~100% GPU）——单次前向成本下降约 2.9×，在被测
  CUDA 机器上 8 步去噪从 ~153 s 降到 ~63 s。用 `TS_QWEN_DIT_WHOLE_CAPTURE=0`
  关闭。
- **默认 flash 注意力**（ggml-cuda 路径；`TS_QWEN_DIT_FLASH=0` 强制显式分数路径）：
  注意力内存为 O(n)，因此在 VAE 解码的 im2col 暂存成为瓶颈前可容纳更高分辨率。
- **CFG-batching**：当合并的 token 块能装入 VRAM 时，两个引导分支在一次摊销启动
  的融合前向中运行（`TS_QWEN_DIT_CFG_BATCH_MAXTOK`）；更大的图像回退为每步两次
  前向。
- **Lightning 蒸馏 LoRA（4/8 步编辑）**——`--qwen-image-lora <lora.safetensors>`
  （CLI + 服务器）/ `TS_QWEN_IMAGE_LORA` 以**运行时旁路**在每个目标投影旁应用
  DiT LoRA：`y = W_quant·x + b + (alpha/rank)·up·(down·x)`（F32 计算），量化基础
  权重保持不变。（合并进权重——stable-diffusion.cpp 的反量化→加→重量化路径——
  只对 F16/Q8_0 存储成立：Lightning 增量约 1e-4 RMS，远低于 Q2_K 量化步长，
  合并会被重量化噪声吞掉。）LoRA 因子与权重一样常驻显存、可被 CUDA 图捕获，
  每次前向额外开销几个百分点 + ~1.6 GB 显存。配合 lightx2v 的
  [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
  检查点，采样默认值自动切换（从文件名解析）：其训练步数（4 或 8）、cfg 1.0
  （每步单次前向）、以及蒸馏所用的固定 timestep shift 3——把默认的 60 次 DiT
  前向减少到 4–8 次。显式 `steps`/`cfg` 仍然优先；`TS_QWEN_IMAGE_LORA_SCALE`
  调节 LoRA 乘数（默认 1.0）。旁路实现在整模型前向（CUDA 默认路径）**以及**
  融合逐块内核（CPU offload 走的流式路径）——因此 Lightning 在 offload 高分辨率
  下依然生效；只有 3 调用 / 托管回退路径不应用 LoRA（会打印警告 / 抛错）。
- **整步去噪缓存（EasyCache）**——**可选启用**，移植自 stable-diffusion.cpp 的
  `easycache.hpp`（sd.cpp 同样只通过 `--cache-mode` 显式启用）。它用测得的输入
  潜变量变化（乘以经验跟踪的输入→输出变换率）
  预测每步的输出变化；当累计预测保持在阈值以下时，**跳过两个 CFG 分支的整个
  DiT 前向**，各分支的速度重建为 `input + cached(output − input)`。最初 ~15% 和
  最后 ~5% 的步总是计算。默认阈值下通常跳过 40–55% 的步（30 步的运行只实际计算
  ~19 步），在编辑负载上会明显软化面部等细节——因此**默认关闭**：质量是默认值，
  提速需显式选择（`TS_QWEN_DIT_CACHE_MODE=easycache`）。
  开关：`TS_QWEN_DIT_CACHE_MODE`（默认 `off` /`easycache`/`fbc`/`both`）、
  `TS_QWEN_DIT_EASYCACHE_THRESHOLD`（默认 0.2——越低越接近无缓存）、
  `TS_QWEN_DIT_EASYCACHE_START`/`_END`（窗口比例，0.15/0.95）、
  `TS_QWEN_DIT_CACHE_DEBUG=1`（每步决策日志）。
- 跨去噪步的 **First-Block-Cache**（每次生成复位）；用
  `TS_QWEN_DIT_CACHE_MODE=fbc` 选择。它每分支总是计算 block 0，在低变化步跳过
  blocks 1..59——节省严格少于整步缓存，但其决策基于真实的 block-0 残差而非预测。
- **融合条件编码器主干**（`TSGgml_QwenTeTrunk`，默认开启）：Qwen2.5-VL 文本编码器
  LLM（28 层，GQA，因果）与其视觉塔（32 层，MHA，窗口/全注意力以加性掩码实现）
  各自把整个层栈作为**一个**设备图运行——逐算子路径每层要付约 10 次设备⇄主机
  往返（M-RoPE、bias 加法、SiLU、块掩码 softmax 都是主机循环）。RoPE cos/sin 表
  在主机预计算；权重按 GGUF mmap 指针常驻绑定。默认编辑下：视觉 4.7 s → **1.6 s**、
  LLM 预填充 2.8 s → **0.8 s**（文本条件阶段 11.2 s → 2.5 s）。已对 numpy oracle
  验证（`te-verify` cosine 0.9989、`vis-verify` 0.9994——均不低于逐算子路径自身）。
  后端无法运行时回退到逐算子路径；`TS_QWEN_TE_FUSED=0` 禁用两者。
- **VRAM 预算 + CPU offload**：条件生成后释放文本 + 视觉编码器，使 DiT（及其注意力
  暂存）独占工作集，并在最终 VAE 解码前归还持久复用 `gallocr`。自动输出尺寸
  **以质量优先**：目标是模型原生 ~1 MP 训练分辨率（与 diffusers
  `QwenImageEditPlusPipeline` / sd.cpp 一致——旧默认按步数/参考图数量缩小面积作为
  速度预算，导致 30 步双图编辑只渲染 ~0.4 MP、面部明显模糊）。当该分辨率无法与
  常驻 DiT 权重共存时，自动启用 **CPU offload**（等价于 sd.cpp 的
  `--offload-to-cpu`）：绕过整模型常驻图，把 60 个 block 作为若干**分块整模型图**
  运行（每块 `TS_QWEN_DIT_OFFLOAD_CHUNK` 个 block，默认 10）——权重放在共享复用
  `gallocr` 的按调用输入槽里（缓冲区逐块重新规划、复用，显存同时只驻留一个分块的
  权重），AdaLN 调制仍在图内计算（不再有逐块的主机端调制展开上传）。设备拷贝常驻
  预算只把「扣除激活后仍放得下」的部分留在显存（先是 flash 掩码、再是权重；显存越
  大 PCIe 流量越少，平滑退化到全流式）。实测 16 GB RTX 3080 Laptop 上 912×1136
  双图编辑（12.7k token）：每次前向 ~19 s / 每个 true-CFG 步 ~41 s——与 sd.cpp 的
  `--offload-to-cpu` 相当；这也是该卡上 ~0.4 MP 与原生 ~1 MP 双图编辑的差别。
  `--offload-cpu`（`TS_QWEN_IMAGE_OFFLOAD_CPU=1`）强制常开；
  `TS_QWEN_IMAGE_OFFLOAD_CPU=0` 恢复旧的「钳制分辨率」行为。`--width`/`--height` 仍受硬件显存上限约束
  （offload 会提高该上限）——超大请求（如 16 GB 卡上的 2048×2048）会被钳制到
  能放下的最大尺寸并给出警告，而不是 OOM 成噪声图。
- **融合整 VAE 图**（`TSGgml_QwenVaeRun`，默认开启）：整个 VAE 编码/解码作为**一个**
  设备常驻 ggml 图运行——C# 侧按已验证的 `VaeReferenceMath` 拓扑发出扁平算子列表，
  特征全程留在 GPU 上，权重从稳定缓冲常驻绑定（只上传一次），每个卷积在其临时
  F16 im2col 适合预算时用 im2col+GEMM（tensor core；`TS_QWEN_VAE_FUSED_IM2COL_BUDGET`，
  默认 2 GiB；gallocr 复用暂存，峰值只有一个卷积的 im2col），超出预算则用
  `ggml_conv_2d_direct`。取代了逐卷积路径的数 GB PCIe 往返 + CPU SiLU/norm 循环：
  928×688 下编码 19.5 s → **0.95 s**、解码 22.8 s → **1.35 s**，与 diffusers oracle
  位级等价（PSNR 99 dB，与旧路径相同）。后端无法运行时回退到逐卷积路径；
  `TS_QWEN_VAE_FUSED=0` 禁用。
- **分带切块的 VAE 卷积**（`VaeReferenceMath.TryGpuConv2dMaybeTiled`）：`ggml_conv_2d`
  会物化一个 `~IC·KH·KW·OH·OW·2` 字节的 F16 im2col，高分辨率下达数 GB，曾溢出到
  共享显存（VAE 慢约 3×）甚至 OOM——这正是把输出限制到 ~0.55 MP 的真正瓶颈。现按
  输出水平条带切分（预算 `TS_QWEN_VAE_CONV_TILE_BYTES`，默认 1 GiB），每条带在手动
  补零的输入切片上重跑同一卷积，结果**逐位一致**（无拼缝——只有临时 im2col 被限界），
  使面积钳制可达模型**原生 ~1 MP**，显著改善面部 / 细节。

对比测试：在本项目的 CUDA `image_edit` 基准场景上（可通过
[`benchmarks/engine_comparison`](../../benchmarks/engine_comparison) 复现；
544×1184 分辨率的 4 步 Lightning 编辑），TensorSharp 完成一次热编辑耗时
**40.44 s**，stable-diffusion.cpp 为 48.16 s（快约 1.19×）；冷启动首个请求为
54.11 s。

重要开关：

| 变量 | 效果 |
|---|---|
| `TS_QWEN_IMAGE_VAE` / `TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ` | 覆盖解析到的伴随 GGUF（CLI 暴露为 `--qwen-image-vae` / `--qwen-image-vl` / `--qwen-image-mmproj`） |
| `TS_QWEN_IMAGE_NO_VISION=1` | 跳过视觉接地（更快的纯文本无接地条件） |
| `TS_QWEN_IMAGE_REF_AREA` | `inputs[1..]` 的参考潜变量面积（像素）——DiT 从中复制面部/纹理细节的来源。默认（CUDA）：原生 ~1 MP、保持各参考图自身纵横比，**与输出尺寸无关**（diffusers Edit Plus 的 `VAE_IMAGE_SIZE`；旧规则会随较小输出一起缩小参考图，悄悄丢弃输入细节）。钳制在 [65536, 4 MP]；超过 ~1 MP 能保留更多输入细节但超出训练分布，且每个参考图消耗 area/256 个注意力 token |
| `TS_QWEN_IMAGE_VISION_MIN_PIXELS` / `TS_QWEN_IMAGE_VISION_MAX_PIXELS` | 视觉塔条件图像的像素区间（默认 384²、560²——Qwen-Image-Edit-2511 参考实现的尺寸规则）：区间内的图像保持**自身分辨率**（对齐到 /28），从原图单次 Lanczos 重采样。旧流程把所有输入压到恰好 384² 再放大回处理器下限——两次重采样，高分辨率输入只剩不到一半的像素 |
| `TS_QWEN_IMAGE_LORA` | 以运行时 F32 旁路应用的 DiT LoRA safetensors（CLI/服务器：`--qwen-image-lora`）；Lightning 检查点还会切换采样默认值（其步数、cfg 1.0、固定 shift 3） |
| `TS_QWEN_IMAGE_LORA_SCALE` | LoRA 乘数（默认 1.0；0 = 结构上启用但零效果） |
| `TS_QWEN_IMAGE_MAX_AREA` | 限制自动目标面积（Metal 速度钳制；CUDA 上可选的低于原生的质量/速度折衷） |
| `TS_QWEN_IMAGE_OFFLOAD_CPU` | CPU offload（CLI/服务器：`--offload-cpu`）：`1` 总是从内存流式上传 DiT 权重；`0` 从不（改为钳制分辨率）；未设置 = **自动**（目标分辨率放不下常驻权重时才启用） |
| `TS_QWEN_DIT_OFFLOAD_CHUNK` | offload 路径上每个分块整模型图包含的 block 数（默认 10；更小 = 权重槽占用显存更少、分块边界 PCIe 更多） |
| `TS_QWEN_DIT_WHOLE_CAPTURE=0` | 禁用 CUDA 图捕获的整 DiT 前向 |
| `TS_QWEN_DIT_FLASH=0` | 强制显式分数注意力路径（更紧的二次 VRAM 预算） |
| `TS_QWEN_DIT_CFG_BATCH_MAXTOK` | 保持 CFG-batching 启用的 token 预算 |
| `TS_QWEN_DIT_CACHE_MODE` | 去噪缓存：`off`（**默认**——质量优先，与 sd.cpp 一致）、`easycache`（整步跳过）、`fbc`（First-Block-Cache）、`both` |
| `TS_QWEN_DIT_EASYCACHE_THRESHOLD` | EasyCache 累计变化阈值（默认 0.2；越低越接近无缓存，越高跳过越多） |
| `TS_QWEN_DIT_CACHE=0` | 旧版总开关——禁用所有去噪缓存 |
| `TS_QIMG_DEBUG=1` | 每步速度 / 潜变量统计 |

## 6. 生成参数（`QwenImageParams`）

| 属性 | 默认 | 含义 |
|---|---:|---|
| `Steps` | 0（自动） | FlowMatch-Euler 去噪步数。自动 = 30，或已加载 Lightning LoRA 的训练步数（4/8） |
| `CfgScale` | 0（自动） | true-CFG 引导尺度；`<= 1` 关闭负向分支。自动 = 2.5（Qwen-Image-Edit-2511 推荐值——4.0 会过度引导：面部扭曲、颜色过饱和），加载 Lightning LoRA 时为 1.0。需要更强风格化可调到 3.5–4，但会牺牲面部保真度 |
| `NegativePrompt` | `" "` | CFG 分支的负向提示词（仅 `CfgScale > 1` 时使用） |
| `Seed` | 0 | 确定性初始噪声种子 |
| `TargetArea` | 1024×1024 | 输出面积（像素，纵横比随输入；尺寸对齐到 /16） |
| `Width` / `Height` | 0 | 显式输出尺寸（绕过 VRAM 面积钳制） |
| `OnStep` | null | 每步回调 `(step, totalSteps, preview)`，用于实时 UI 反馈 |
| `PreviewCount` | 0 | 整个循环中输出的解码 RGB 预览数量 |

## 7. 服务器行为

当 Web UI 托管一个 `qwen_image` DiT GGUF 时，上传 + 编辑端点接管：

- `POST /api/image-edit` 运行单次编辑并返回输出图像。
  Multipart：一个或多个 `image` 文件部件 + `prompt`/`steps`/`cfg`/`seed`/
  `targetArea` 字段。JSON：`{ imagePaths: [...], prompt, ... }`（或旧版单个
  `imagePath`），路径来自 `POST /api/upload`。
- `POST /api/image-edit/stream` 流式发送 SSE 帧：进度滴答加上节流的解码预览
  （`{ imageEdit: true, step, total, image, width, height }`），随后在 `done`
  之前发送最终的全分辨率图像。请求把步数留为自动（0）时预览同样可用——最多
  输出 8 帧均匀分布的预览；预览编码失败会降级为仅进度的滴答，而不会中止编辑。
- 编辑在共享锁后串行执行（扩散网络非线程安全），因此并发请求逐个运行。
- Ollama / OpenAI 聊天适配器是自回归的，不暴露图像编辑。

## 8. 待办

- 并发编辑被串行化；跨请求的批量 / 流水线编辑尚未实现。
- 视觉接地条件准确但开销大；最实用的全质量编辑在带整 DiT 捕获的 CUDA 路径上。
