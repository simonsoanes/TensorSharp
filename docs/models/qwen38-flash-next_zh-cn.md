# Qwen 3.8 Flash Next（`qwen4exp`）

[← 返回模型索引](README_zh-cn.md) | [English](qwen38-flash-next.md)

Qwen3.8-Flash-Next 是一个混合型 MoE：GatedDeltaNet 递归层与全注意力层交错
（其中一部分全注意力层挂在 Qwen Sparse Attention 的 indexer 后面），再加上一个
PLE n-gram 嵌入块、×4 hyper-connection 流以及 512 专家的 MoE。GGUF 架构 id 是
`qwen4exp`。权重：
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
（每个量化档一个子目录、均为多分片；`--model` 指向 `-00001-of-` 那一片；图像输入
需要 `mmproj-BF16.gguf`：CLI 在给出 `--image` 时会从模型旁边自动加载，服务端则需要显式传
`--mmproj`）。

## TensorSharp 如何运行它

在 GGML 后端上，整个 token（几乎）只跑一张图——嵌入、PLE（在图内）、全部 48 层、
最后的 mixer 以及 LM head——并配一个按形状索引的已捕获图缓存
（span 放弃时改走逐层融合 kernel，后者再逐算子回退）。视觉沿用
Qwen3.5-VL 塔，位置用 (T,H,W) IMRoPE；支持多图与多轮图像会话，并在轮次之间复用
KV（GDN 递归无法回退，因此只有当新 prompt **恰好扩展**已缓存前缀时才复用；见[保留前缀复用](#保留前缀复用)）。在 radix 前缀缓存中，这种复用止于会话中第一个图像或视频 span：该系列尚未声明可跨媒体 span 复用（它保存了一段 M-RoPE 缓存间隙，目前还没有参照位置测试覆盖）。由于该系列只能从长度与匹配长度完全一致的 holder 或检查点续接，之后的轮次最多复用附件之前已存储的检查点（通常是系统提示词），其余部分重新 prefill。

思考模式可以开启或关闭。关闭时，助手轮次以已发布模板输出的闭合空块 `<think>\n\n</think>` 开头，重放历史时也保留这一确切后缀，因此缓存前缀仍能匹配。

## 工具调用与 Agent 工作流

`qwen4exp` 通过 Qwen ChatML 输出解析器返回结构化工具调用。支持 `<tool_call>`
内的 JSON 与 `<function=...><parameter=...>` 格式，以及流式分片和思考模式。
调用方工具以带调用 ID 的 OpenAI `tool_calls` 返回，结束原因是 `tool_calls`；
配置启用后，内置技能与代码工具由服务端 Agent 循环执行。

XML 参数按工具声明的 schema 解析：字符串 `123`、`true` 与 JSON 源码不会变成
数字、布尔值或对象。解析器仅移除两侧各一个格式换行，保留源码缩进与额外空行；
参数或 JSON 字符串内的 `</tool_call>` 不会提前结束调用。不完整的参数或函数不会
成为可执行调用；EOS 时已有完整正文、仅缺外层结束标记的恢复行为保持不变。

使用 `--skills-dir` 启用技能目录，`--skills-allow-exec` 启用技能脚本，
`--code-exec` 启用工作区文件与 shell 工具；编辑工具为 `apply_patch`。
执行与沙箱配置见 [Agent Skills](../agent_skills.md)。

由于该系列会渲染工具声明并带有这个解析器，它在服务端也可以使用
[子智能体委派](../multi_agent.md)（在对话路径上默认开启，与 skills 和 `--code-exec` 无关）。
`--no-multi-agent`（或请求中的 `multi_agent: false`）可将其关闭；CLI 没有子智能体。
该系列没有发布任何委派相关的实测结果。

可复用验证脚本：`eng/validation/validate-qwen38-tool-calls.py` 检查通用 API 的
流式/非流式、思考开/关与工具结果回传；`validate-release-agent-workflows.py`
配合 `eng/validation/fixtures/skills` 检查技能发现、读取、脚本、shell、代码生成
与读取/编辑/运行。`verify-agent-code-artifacts.py` 对最终源码使用额外输入独立执行。
若验证服务器显式关闭沙箱，两个执行验证脚本需传 `--sandbox-off`，报告只证明功能
执行成功，不证明沙箱隔离。完整命令见[英文版](qwen38-flash-next.md#tool-calling-and-agent-workflows)。

2026-09-19 使用提供的 UD-IQ4_XS 模型，在三张 NVIDIA A40 上验证
（`ggml_cuda`，按 15/16/17 层切分，关闭 MTP）；构建使用未修改的 upstream ggml
`456172ec733a135778adcd32d00e576a58232e45`：

- 160 项托管回归测试通过，无失败或跳过。
- 12 项常规通用工具用例全部通过：天气、数字字符串/JSON 源码、多行 Python
  （含尾部换行），分别覆盖流式与思考开/关，并用实际调用 ID 完成工具结果回传。
- 12 项技能/shell/代码工作流所需工具均成功执行；4 份最终生成或编辑的程序用额外
  输入独立执行通过。分离工具调用前的说明文字后，严格最终回答检查为 9/12 通过；
  另外 3 项在正确值外添加反引号或说明，仍记为失败。
- 4 项包含工具标记字面量的压力用例均在调用完成前以 EOS 结束，端到端仍失败；
  完整 XML/JSON 字面量调用已通过解析器回归。日志未暴露最终采样 token 的 ID，
  工具标记 token 本身不属于 EOS。

VM 禁止 user namespace，执行验证显式关闭了沙箱。本轮仅覆盖一种量化与串行请求，
不验证沙箱隔离、MTP、其他设备或性能。完整请求、SSE、产物、构建信息及失败证据保存在
已忽略的 `docs/validation/qwen38-tool-calling/` 目录。

## 视频输入

视频以 OpenAI Chat Completions 的 `video_url` content part 送达模型，内容是 base64 的
MP4、WebM 或 MOV data URI（不会抓取远程 URL）：

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,...","fps":1,"max_frames":8}}
```

`fps`（0 < fps ≤ 60）与 `max_frames`（1–64）可选；默认值取 `VIDEO_SAMPLE_FPS` 与正的
`VIDEO_MAX_FRAMES`，否则为 1 fps 与 16 帧，更长的片段会在全长上均匀采样。服务端把片段解码成
有序的帧，每帧带其源时间（帧序号除以探测到的帧率，变帧率片段为近似值），之后由 Qwen-VL 的
视频布局接手：

- **帧对（temporal pair）。** 视觉塔的 patch embedding 有两个时间切片
  （`v.patch_embd.weight` 与 `.weight.1`），所以连续帧两两合并，与 Qwen-VL processor 堆叠帧
  的方式完全一致；奇数帧的片段会重复最后一帧补齐最后一对。processor 以 2 fps 采样，所以它合并的
  两帧相隔 0.5 秒；TensorSharp 只在两帧相距不超过 `QwenVideoFrames.MaxPairedFrameGapSeconds`
  （0.575 秒）时才把它们配成一对。更稀疏的帧——默认的 1 fps，或按 `max_frames` 分散到长片段上的
  帧——是不同的画面，因此各自独占一个时间 patch（像静态图一样重复该帧），并保留自己的时间标签。
  把这种帧配对会把它们混在一起：卡片 17、42、86 的三帧 1 fps 片段读成 `["12", "47", "86"]`，
  每帧一个 patch 时读成 `["17", "42", "86"]`。代价是每个采样帧一个 patch，而不是每两帧一个。每一对单独编码（参考实现的视觉塔
  只在一个时间 patch 内做注意力），得到与一张静态帧相同的合并 patch token 数。整段片段按同一个
  视频像素预算（`Qwen35ImageProcessor.VideoMinPixels` / `VideoMaxPixels`）整体缩放，因此同一片段的
  每一对共用一个网格；即使用最小网格也放不进该预算的帧数会被拒绝。
- **提示词布局。** 模板把视频 part 渲染成 `<|vision_start|><|video_pad|><|vision_end|>`，整个外层
  span 会被替换——与 Qwen3-VL processor（transformers v4.57.1 `processing_qwen3_vl.py`）一致，片段外
  不再多包一对分隔符——为每对一个 `<t seconds><|vision_start|><|video_pad|>…<|vision_end|>` 块，
  `t` 是该对的平均源时间（保留一位小数）。由于帧对的标签丢失了两帧各自的时间（且 `max_frames`
  上限可能选中不相邻的帧），这些块之前会有一行文本
  `Sampled video frame times in chronological order: 0, 1, 2 seconds.`，列出全部采样源时间；逐对的
  视觉 token 布局不变。一条消息里的两个 `video_url` part 渲染成两段片段。同一条消息里的静态图保留
  各自的 `<|image_pad|>` span，按附件顺序排列。
- **编码缓存。** 帧对的 embedding 以两帧路径加片段尺寸为键缓存，任一帧文件的大小或时间戳变化
  （而不只是较晚那一帧）都会使其失效。
- **位置。** 每一对都像一张静态图那样定位，其 (T, H, W) 坐标从该对所处的运行位置起算——这
  就是 Qwen3-VL `get_rope_index` 把视频网格拆成逐对条目的规则——因此相邻帧对拿到严格递增的
  时间轴 M-RoPE id，中间的时间标签文本推进位置流，片段之后的文本从最后一对的网格之后继续。
  QSA indexer 的位置历史、MTP 草稿追赶以及片段之后的旋转/cache gap 记录的都是同一套坐标，
  所以投机解码与保留前缀和目标模型一致。

它不是什么：帧是采样得到的，不是由时间编码器解码；帧时间是采样到的源时间，而非重新对齐到
2 fps 的流。通过 Web UI 上传的视频帧不带源时间，仍然按静态图处理，每帧一个 span，与以前一样。
完整 checkpoint 的检查是 `benchmarks/engine_comparison/validate_deepseek41_media.py` 的
`video_order` / `video_timestamp` 场景，对象是挂了 `mmproj-BF16.gguf` 的 Qwen3.8 服务。

## 思考预算

开启思考时，思考内容一旦达到 `TS_THINKING_BUDGET`（`max_tokens` 不小于 512 时默认为其 75%），服务端和交互式 CLI 都会闭合思考块，答案随后在原 `max_tokens` 内生成。`</think>` 是单个训练过的 token（248069），宿主会先写入 Qwen 官方发布的交接句（"Considering the limited time by the user, I have to give the solution based on the thinking directly now."）。2026-09-29 之前该系列没有闭合 token，思考达到预算的轮次会以**空答案**结束（`finish_reason` 为 `thinking_budget`）：在 4x A40（`--tp 4` 与 `--layer-split 4`）上以 `max_tokens` 2000 运行四个并发的三轮会话，通过 0/4，失败的都是空答案轮次。加入交接句后，同样的 `--tp 4` 运行通过 4/4：交接句闭合了四个轮次，每一轮都有答案，每个会话都复用了上一轮（第 3 轮复用 1591-3509 个提示 token 中的 1564-3482 个）。

## 连续批处理

并发请求通过**逐序列状态持有者**（per-sequence state holders）来服务：每个在飞请求
各自拥有自己的注意力 KV 与 QSA indexer 缓存、GDN 卷积与 delta-net 状态、PLE 卷积
历史与 n-gram 窗口，以及固定下来的 kernel 描述符。原生 kernel 用持有者的 host 种子
指针作为设备驻留递归状态的键，用描述符地址作为已缓存图的键，所以切换请求只是一次
引用交换——不需要状态下载 / 上传，也不需要重建图——每个序列都在自己那张已捕获的
单图融合 decode 上解码。引擎按步在各序列间轮询（`SupportsPerSequenceFusedForward`）；
融合的 N 路批量 decode 属于后续优化。

**并发时的贪心输出可能与单独运行不同，原因在于 prefill 的分块形状。** 调度器对单独一个请求用
一个大块 prefill（`TS_SCHED_SOLO_PREFILL_CHUNK` 与 `TS_SCHED_MAX_BATCHED_TOKENS` 中较小者），对并发
请求则按步预算分份，而本模型的 logits 依赖分块大小。在 UD-Q2_K_XL、三 GPU 按层切分上用
`benchmarks/ChunkParityProbe` 对一个 19,121 token 的 prompt 实测：重复同样的 4096 分块逐位一致
（max |Δlogit| 为 0）；1024 与 512 token 的分块让 logits 最多偏移 1.3，并在近似平局处翻转贪心解码——
最早在第 9 个输出 token，top-2 差值为 0.002（标题的第一个词）。保持分块形状相同就消除了这一效应：
一个 2,928 token 的 prompt，每个请求都一次 prefill 完（`TS_SCHED_PREFILL_CHUNK=4096`、
`TS_SCHED_MAX_BATCHED_TOKENS=16384`），4 路并发 × 3 轮的输出与单独运行逐字节一致（512 token，12/12），
所以逐序列 holder 之间没有状态泄漏，轮询 decode 也不依赖并发度。CUDA 上不承诺 prefill 形状的宽度不变性，
所以 fixture 的分块与整段关卡给差异设上界，而不是要求逐位一致；见 [保留前缀复用](#保留前缀复用)。

## 保留前缀复用

`Qwen4ExpModel.RetainedCache.cs` 为 `qwen4exp` 提供与 Qwen 3.5、DeepSeek V4 路径相同的保留 holder 复用：

- 结束的会话的整个逐序列 holder 会被**保留**，并为恰好扩展它的下一轮重新设键。什么都不移动：以该
  holder 为键的原生状态条目、它的已捕获图以及草稿头的私有 K/V 都留在原处。
- 所有聊天共享的 prompt 结尾处的状态会被**检查点**为以主机为准的深拷贝（注意力 K/V、QSA 原始 key
  与位置、GDN/PLE 递归状态、私有 MTP 状态），并**克隆**进每个新聊天。缺少权威原生状态时，克隆会
  拒绝执行，而不是拷贝陈旧的主机种子。
- 复用**仅限精确前缀**（`IExactFusedCacheReuse`）：新 prompt 没有逐 token 复现到最后一个的 holder
  不是它的延续，任何部分匹配都会重新 prefill。
- 保留的会话与检查点共用一个预算 `TS_Q4E_RETAINED_CACHE_MB`（受实测内存余量限制；
  `0` 或无法解析的值会拒绝所有保留）。未设置时，预算为实测余量的一半（与 Qwen 3.5 对空闲
  holder 采用的规则相同），只有在无法测得余量时才使用 4096 MB。此前固定的 4096 MB 默认值在
  4x A40 张量并行下只能容纳四个并发会话中的三个（1.6k token 时每个 holder 1318.6 MB，余量
  15 GB），其中一个会话每一轮都要重新 prefill。radix 前缀缓存负责保留与驱逐，这个预算只会拒绝放不下的
  holder（只报告一次）。
- 它需要完整的 GGML token-span 路径（每一份逐序列状态都驻留在设备上并以 holder 为键），以及原生
  条目可以精确拷贝的 GDN 状态布局。保留在按层切分下可用；检查点在按层切分下被接受，在张量并行下
  被拒绝。

证据（合成 fixture，不代表训练模型的验收或性能）：
`Qwen4ExpRetainedCacheTests` / `Qwen4ExpRetainedCachePolicyTests` 覆盖保留 A/B/A、检查点克隆、投机重绑定、
预算驱逐、缺失状态拒绝以及 QSA 首次/重置增长，并在 CUDA 上覆盖真实双 GPU 按层切分的检查点生命周期。
所有关卡在 CPU 上都逐位一致。在单卡 CUDA 上，一次 4 token 的目标验证与 4 次单 token 前向逐位相同
（`TeacherForcedTargetVerify_…`），以 2–4 为块提交的 32 个 teacher-forced token 在每一行上都与标量解码
逐位相同（`RepeatedTargetBlocks_…`）——见 [验证行使用单 token kernel](#验证行使用单-token-kernel)。
16 token 的 prefill 再接 4 个 token，在 CUDA 上与一次 20 token 的 prefill 并不逐位相同，因为 prefill 的
kernel 按批宽度选择：`SharedPrefixChunking_…` 在 CUDA 上把差异上界设为 1e-2（实测 logits 相差 1.7e-4 到
4.4e-4；它当初要抓的陈旧种子缺陷让 logits 偏移了 0.3155），并且只允许在 top-2 差值不超过实测差异两倍的
近似平局处改变贪心结果（2026-09-17 在 A40 上实测）。

## 共享 MTP 头的投机解码

图像请求也可使用学习得到的草稿头：调度器在每个投机 prefill 块之前排入对应的图像
embedding，并保留其 MRoPE 位置。prefill 后为重试保留的图像片段不再阻止投机 decode。
这仍要求单独请求从位置 0 开始 prefill；下述保留前缀与并发请求限制仍然适用。

2026-09-27 另以 UD-IQ1_S、两张 RTX PRO 4000 Blackwell、`--layer-split 2`、
上下文 1024、常驻共享 Q8_0 MTP 头及 BF16 视觉伴随文件验证。一次预热后，三轮实测的
文本/图像输出均与普通贪心逐 token 一致，MTP 与 ngram 都有实际起草。文本在 64 token
上限停止；图像回答在 204 个可见 token 后以 EOS 完成，数字与颜色描述正确。MTP 的逐轮
配对 decode 工作线程计算时间加速比中位数为文本 1.215 倍、图像 1.292 倍。图像请求计时
不包含同步图像准备与编码；这些短文本复制检查不代表通用质量或完整媒体请求延迟。
另外，普通/MTP HTTP 两种模式各通过 24/24 文本请求与 3/3 图像场景，包含附件顺序与
历史图像。该配置因余量不足拒绝保留缓存，因此后续轮次重新 prefill。本地证据：
`docs/validation/model-matrix-20260927/qwen38/SUMMARY.md`（不提交）。该架构现支持下文所述的本地张量并行；跨节点执行仍不支持。

`--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` 挂上逐 token 的 MTP 块（仅限 GGML 后端；该头必须是单个 GGUF 文件，并在模型加载时挂上）；它只为从位置 0 开始
prefill 的单独请求做投机（与其他序列共享的步，以及延续保留 holder 或共享前缀克隆的轮次，都按普通
方式解码——草稿头有自己的 K/V，无法跨越它从未重放过的位置起草）。在 UD-Q2_K_XL、三 GPU 按层切分
（A40）上以 `--spec-draft 3` 实测：

- **一致性。** 192 token 的代码复制流与普通贪心完全一致，在当前的验证行 kernel 下速度为普通 decode 的
  1.69 倍（2026-09-17 下文改动之前为 1.75-1.96 倍；141/141 草稿被接受，无回滚）；投机 prefill（`SpecForward`）在相同分块下与普通 prefill 逐位一致。2026-09-17 之前，4 行
  verify 的舍入与 1 行 decode 步不同，可能把普通贪心写出的裸 JSON 对象变成 ```` ```json ```` 围栏
  回答；现在验证行使用单 token kernel（见下文）。
- **散文不划算。** 18 个散文请求的接受率为 68-70%（每次 verify 3.0 个 token），但一次 verify 约 45 ms，
  部分接受还要加约 43-46 ms 的回滚（恢复递归状态、重新前向保留的行），而被调速器暂停的步仍然要跑
  带隐藏状态捕获的单行投机前向（约 24-26 ms，普通 decode 为 19 ms）：512 token 散文在 c1 下 decode 为
  44-46 tok/s，普通为 52；8k prompt 之后为 34-38，普通为 40。

### 验证行使用单 token kernel

一次 verify 以及对其接受前缀的重放，都会把 2–8 个 token 放进同一张 span 计算图（同样长度的 prefill 也是如此），
而在 CUDA 上 ggml 会按批宽度选择多个 kernel。在这些宽度下，一行的舍入方式与它对应的单 token decode 步并不
相同：F32 投影超过 3 列后离开 `mul_mat_vec_f`，改走 tensor-core / cuBLAS TF32 路径（router logits 偏移
2.6e-3）；BF16 的 QSA indexer 投影从 2 列起走半精度路径（5.9e-3）；路由专家换成多 token MoE kernel
（4.8e-7）；flash attention 换成多 query 启动（2.9e-5）。经过 48 层 MoE 与 QSA 路由，这已不是最后一位的
噪声：在三张 A40 上的 UD-Q2_K_XL 里，3,248 token prompt 之后 teacher-force 前 48 个贪心 token，2、3、4 行
verify 的每一行都与其 decode 步不同，logits 最多相差 2.5，48 行里有 4–6 行改变了贪心 token——并不只发生在
近似平局处。

因此在 CPU 与 CUDA 上，2–8 个 token 的 span 计算图用其单 token 图会运行的 kernel 构建每一行：浮点投影把 token 放在
广播轴上（CUDA 上一次 `mul_mat_vec_f` 启动）。只有实测过的 NVIDIA A40 上的 Q4_K、Q5_K、Q6_K、Q8_0 投影
按至多 4 行一块运行；其他设备和量化类型在广播轴上使用单列归约。Turing 与 GB10 在宽度 1 时的 MMVQ 归约
与 A40 不同，不能全局套用四行分组。路由专家与注意力
逐行展开，每个注意力行读取的 KV 窗口与 mask 行恰好就是它的 decode 步所读的那些。不超过 8 个 token 的图
（包括 decode）还会让两个是否融合取决于内存复用的 ggml-cuda 融合（MoE 加权归约；RMS norm + RoPE）的输入
保持分配，于是这两个融合在任何宽度下都会发生。单 token kernel 不变。CUDA 超过 8 token 的 prefill
会在顺序求和前物化专家加权输出，避免依赖内存分配的 FMA 融合使层切分与张量切分产生不同舍入。
CPU 与 Metal 保留现有的 prefill 图构建方式。

CPU 与 CUDA 将每轮草稿限制为 7 个 token；验证还包含待提交的 anchor，合计最多 8 行。
该硬上限同样约束显式 `--spec-draft` 和自定义 drafter，默认首选窗口仍为 3 个草稿 token。

补充测试现覆盖宽度 1–8 的每一行。macOS ARM CPU 也需要此构建方式：原路径在宽度 2、4 时，虽然保存的 GDN、PLE、
KV 状态相同，logits 仍与逐 token decode 不同。启用 CPU 路径后严格的 fixture 测试通过；测试耗时不视为性能基准。
下方耗时仅来自原 A40 测量，不能作为其他 GPU 架构或广播回退路径的性能结论。测试 hook 构建可在启动时设置
`TS_Q4E_TEST_MMVQ_CHANNELS=1`，在 A40 上验证广播回退路径。

在同一环境下交替重复三次实测：宽度 2、3、4 的每个验证行现在都与其 decode 步逐位相同（48 行中 0 行不同，
没有贪心翻转）。4 行 verify 耗时 32.4–33.1 ms，此前为 29.3–29.6 ms（+11%）；3 行 28.3–29.9 对 26.8–27.2（+8%）；
2 行 24.5–24.8 对 24.1–24.3（+2%）；decode 步（20.6–20.7 ms 对 20.6–21.1）与 3,248 token 的 prefill
（2,335–2,338 ms 对 2,324–2,359）不变。在 192 token 的代码复制流上端到端测量（每种 kernel 各 6 轮，所有输出
都与普通贪心一致），MTP 投机为 83.2 tok/s，此前为 86.5（相对普通 decode 从 1.84 倍变为 1.69 倍）；n-gram 投机为
73.8，此前为 79.5；普通 decode（49.1 对 47.0）与 prefill（830 对 804 tok/s）没有退化：精确性的代价由投机承担。

## 多 GPU

`ggml_cuda` 的 `--tp N` 切分每个路由专家及共享专家：gate/up 按中间通道切分，汇集激活后，
down 按输出行切分，再汇集输出供 hyper-connection 写回。每个 rank 保留全部专家 ID。
down 点积保持原始完整宽度，避免求和顺序的微小差异被后续激活量化放大。
注意力、GDN、QSA、PLE 使用复制的权重及各 rank 独立状态，
输出头仅在 rank 0 执行。图像与工具调用沿用同一目标图；投机回滚会恢复每个 rank
的 GDN/PLE 状态。TP 下仍不支持共享前缀检查点，也不支持跨节点执行。
两次汇集使用 TensorSharp CUDA FP32 collective；回退路径以零复制切片避开
上游 CUDA 自动 BF16 阈值，设备 collective 不可用时使用 FP32 主机归约。
两次通信增加数据量以保持数值精度，无需修改 ggml。

FFN 中间宽度和输出宽度必须能被并行度整除；投影按完整输出行切片，保留完整输入量化块。
量化 prefill 保留原始 MMQ tile 和归约几何，gate/up 切片保留重叠的 128 行边界 tile，再裁剪输出。
在中间宽度 640 的检查点上，TP2 每个 rank 为逻辑 320 行保存 384 行，TP4 为逻辑 160 行保存 256 行。
不支持的配置会在批量加载权重前依据 GGUF 元数据拒绝。当前模型路径要求 CUDA MMQ stream-K，
FFN 类型限于 Q2_K、Q3_K、Q4_K、Q6_K、IQ3_S、IQ4_XS、IQ4_NL 和 Q8_0；完整输出行数及 down 输出切片须为 128 的倍数。
其他 FFN 类型（包括 F32/F16/BF16）、设备或无法保持归约顺序的布局会明确拒绝；物理 GPU 验证使用 NVIDIA A40。
专家切片目前另占总路由专家字节数及重叠行的
主机缓冲区，并保留至模型释放。加载时不再预读整个稀疏 PLE 表，所需行按需读取。

`--layer-split N` 仍表示按完整层连续分配至多个 GPU，仅适用于 `ggml_cuda` 和
`ggml_vulkan`。不得同时使用 `--tp` 与 `--layer-split`。
`eng/tests/qwen4exp-tensor-parallel.py` 验证两层 prefill、重放、QSA、多轴 RoPE、
全部 logits 和多 rank 状态回滚。量化模式
（`--quantized-ffn --tokens 1,2,3,4,5,6,7,8 --rollback-width 8`）在 CUDA TP2 与 TP4 上均通过全部 102 项检查，
包含验证宽度 8 的循环状态、QSA 和 PLE 快照恢复，hidden 与 logits 误差均为零。
CPU TP4 loopback 另行通过全部 42 项 F32 检查，仅用于正确性验证。
合成 F32 CUDA TP4 的宽度 17 重放超过原有误差门限（hidden 最大误差 3.49e-5）；
公共模型入口明确拒绝 F32 FFN，此场景不计为通过。
`eng/tests/qwen4exp-tp-quantized-ffn.py --hidden 2560` 另行验证真实 640 通道宽度、
IQ3_S/IQ4_NL 与 IQ4_XS/Q8_0 gate/down、Q8_0 共享专家及宽度 1 至 8、17、31、128；
CUDA TP2 在 `--require-bitwise` 下逐位一致。
`eng/ForcedLogitProbe` 在固定相同 token 历史下比较完整模型 logits，避免早期贪心
分歧掩盖后续 decode 的数值误差。

UD-IQ4_XS 检查点在 NVIDIA A40 上，TP2、TP4 各有 120 行完整词表 logits 与稳定舍入后的普通 layer2 执行逐字节一致：
三个文本 prompt、一个单 token 合成 prompt、一个 128 token 合成 prefill，每例固定历史运行 24 步。
TP2 两种路径均使用 native `da25f156`；TP4 使用 native `1ba6d7a4`，与保存的 `da25f156` 普通执行参照比较。
下述最终 HTTP 与投机检查使用 native `473ee64d`。
所用 ggml 为未修改的 `353b63b439f27ab2cc19dac97ab1681ba6d2d084`。
CUDA prefill 舍入稳定化可能改变旧二进制的 logits 或低 margin 贪心选择；旧参照向量单独保留，不宣称与旧版逐位兼容。
最终 HTTP 对比中，与旧二进制的首个 logits 向量相对 L2 误差为 0.0416；16 个文本结果有 14 个完全相同，
另两个仅有标点差异。该历史数值对比不通过严格一致性门限。

最终 CUDA TP2 HTTP 测试与同版本 layer2 执行在全部 16 个文本 prompt、4 个工具调用往返和 4 个图像回答轮次上一致，
首个真实 prefill 的 248,320 个 logits 逐字节相同。学习型 MTP 和 n-gram 投机在文本与图像测试中
均保持全部 96 个普通贪心 token 一致。压力配置为 `TS_SPEC_DRAFT=7 TS_SPEC_PMIN=0`；
MTP 在两个场景中均实际达到验证宽度 8，并覆盖拒绝回滚。所有进程正常退出，运行前后检查确认二进制未改变。

另一次六进程启动基准使用 2× NVIDIA A40、UD-IQ4_XS、上下文 4096、F16 KV、128 token 内核预热及两个 CPU 线程。
下表按执行顺序保留每种模式的两次启动。每个进程对固定输入的 pp512/tg128 计时五轮，
另以不计时的完整 128 token 贪心序列检查正确性。吞吐单元格依次为
**首轮 / 第 2–5 轮中位数 / 全部五轮范围**，单位 token/s；加载时间不含内核预热。

| 模式 / 启动序号 | 加载（秒） | 预热（秒） | pp512：首轮 / 中位数 / 范围 | tg128：首轮 / 中位数 / 范围 |
|---|---:|---:|---|---|
| 旧版 layer2 / 1 | 64.85 | 20.67 | 259.2 / 406.60 / 259.2–439.7 | 24.9 / 29.45 / 24.9–36.6 |
| 当前 layer2 / 1 | 48.98 | 21.51 | 262.7 / 427.75 / 262.7–446.5 | 30.3 / 37.90 / 29.2–38.0 |
| 当前 TP2 / 1 | 85.68 | 10.78 | 229.0 / 345.75 / 229.0–358.9 | 22.0 / 24.10 / 18.2–25.9 |
| 旧版 layer2 / 2 | 68.74 | 22.14 | 255.9 / 419.25 / 234.0–454.1 | 29.2 / 30.40 / 29.2–35.7 |
| 当前 TP2 / 2 | 80.22 | 12.56 | 230.3 / 349.15 / 213.9–356.7 | 18.6 / 27.30 / 14.2–34.3 |
| 当前 layer2 / 2 | 25.66 | 18.13 | 258.3 / 414.65 / 234.9–422.1 | 30.3 / 33.85 / 30.2–37.8 |

六个进程均正常退出，并通过运行时文件身份检查。当前 layer2 与 TP2 的两次启动均生成完全相同的完整贪心序列。
汇总稳态吞吐为 layer2 **421.20 / 35.875**，TP2 **347.45 / 25.70** token/s：
此机器上 TP 的 **prefill 慢 17.5%，decode 慢 28.4%**。Attention 与循环状态在各 rank 复制，
每层两次精确 F32 FFN 集合通信增加了这些 PCIe GPU 的通信开销（`NCCL_P2P_DISABLE=1`）。
本次实测按层切分更快。

旧二进制的合成贪心序列从 decode 下标 50（从零计数）起与全部当前运行产生分歧，
严格兼容性对比仍然失败；当前与旧版的吞吐比值未通过 token 一致性资格检查。
这些加载是未清除页缓存的混合/热缓存测量，GPU 时钟仅记录、未锁定。
明显的计时和加载波动不支持冷存储或普遍加速的结论。

另有一次相同二进制和设置的 TP2 诊断，仅改为
`GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_BF16_THRESHOLD=0`。
完整贪心序列及运行时检查通过，但性能取舍不一致：pp512
**216.3 / 309.15 / 202.7–321.4**，tg128 **16.5 / 30.30 / 16.5–40.1**
（首轮 / 稳态中位数 / 全部五轮范围），加载 100.69 秒、预热 10.05 秒。
这一次启动的 prefill 更慢，不足以支持更改默认 NCCL 传输。

下列历史性能数据仅对应按层切分。

实测：2× A100-80GB，Qwen3.8-Flash-Next-UD-Q2_K_XL（73.4 GiB）：

- 1 卡与 2 卡运行的贪心输出**逐字节一致**（SHA-256 相同）。
- 显存 24.2 GB + 26.2 GB——大约每张卡各放半个模型，而不是一张卡放下全部。
- 吞吐不变：两种情况下 prefill 都在 ~1520–1550 t/s，decode 都在 ~56 t/s。
  作为参照，同一台机器上的 llama.cpp：1 张 GPU pp1536 1094 / tg128 61.2；
  2 张 GPU `-sm layer` 1200 / 61.5——也就是说 llama.cpp 从第二张卡上同样只拿到
  约 10% 的 prefill 提升、decode 基本为 0。

启动时会打印实际走的是哪种模式，以及每张 GPU 分到的层数 / 字节数。
`TS_Q4E_LAYER_SPLIT=20,28` 可以用显式的每卡层数覆盖自动均衡（精神上等同于
llama.cpp 的 `--tensor-split`），并且在无法满足给定值时直接抛异常，而不是悄悄忽略
——这很有用，因为自动均衡只按权重计价，看不见视觉塔，而视觉塔加载得更晚、会落在
GPU 0 上。

## 基准矩阵

[`benchmark_config_glm53_qwen38.json`](../../benchmarks/engine_comparison/benchmark_config_glm53_qwen38.json)
以 `qwen38-flash-next` 的名字把本模型注册到固定的 Hugging Face revision 上，并挂上
它的 `mmproj-BF16.gguf`，好让 `image` 场景能跑。其中两条事实值得在这里重复。

一是已发布的 Q8_0 分片里**完全没有** `nextn` / `mtp` 张量，因此 `mtp_supported`
为 false，`--mtp on` 的格子会带着理由被跳过，而不是悄悄按普通解码跑掉。

二是**本模型只能跑在会传 `--layer-split N` 的那一列上**。原因就在上一节：切分度来自 `--layer-split`，
所以在不传 `--layer-split` 的后端列上，TensorSharp 只会建单设备上下文，175.3 GiB 会全部压到
一张卡上。因此配置里给了它 `min_tp`（4，仅按权重算出的下限——8 才是这台 8×A40 机器
应当使用的度数），在不传 `--layer-split` 的那一列上，这些格子会被记为
`needs --tp 4 (does not fit 1 GPU(s))` 的跳过，而不是留给它去 OOM。这里的 `--tp` 是
基准工具保留的 GPU 数量选择参数；所选后端列向 TensorSharp 传入的是 `--layer-split`。跑法：

```
python run_matrix.py --config benchmark_config_glm53_qwen38.json \
    --models qwen38-flash-next --backends ggml_cuda_split
```

那一列会让 llama.cpp 用 `--split-mode layer` 切在同样这些 GPU 上，于是参照列两边是
同一种整层放置方式。这里的历史结果只验证层切分，不代表上文新增张量并行的性能。
