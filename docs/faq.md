# 常见问题

本文整理开发与使用 Kuavo Data Challenge 项目时常见的问题与解决办法，按**环境与路径**、**训练**、**部署与推理**、**模型与权重**、**PI0/PI05 专项**等分类。

!!! abstract "快速跳转"
    下表为按类别整理的问答索引，便于快速定位到对应小节。

| 类别 | 编号 | 问题摘要 |
|------|------|----------|
| 环境与路径 | Q1.1 | 配置文件或 checkpoint 路径报错 |
| 环境与路径 | Q1.2 | HuggingFace repo id 与本地路径混淆 |
| 环境与路径 | Q1.3 | 缺少 gcc / ccache |
| 训练 | Q2.1 | PI0 配置字段错误（freeze_vision_encoder 等） |
| 训练 | Q2.2 | policy.repo_id 缺失 |
| 训练 | Q2.3 | CUDA 显存不足 |
| 训练 | Q2.4 | 无法连接 HuggingFace |
| 部署与推理 | Q3.1 | 仿真脚本卡在等待 init |
| 部署与推理 | Q3.2 | No task found in complementary data |
| 部署与推理 | Q3.3 | 训练与部署模态一致性 |
| 模型与权重 | Q4.1 | Missing key(s): embed_tokens.weight |
| 模型与权重 | Q4.2 | 3 channels / 1 channel 不匹配 |
| 模型与权重 | Q4.3 | PaliGemma tokenizer 路径变更 |
| PI0/PI05 | Q5.1 ~ Q5.6 | 配置、past_key_values、denoise_step、无语言/仅 RGB 部署等 |
| 其他 | Q6.1 / Q6.2 | SDK 版本、仅 RGB 与 RGB+Depth 确认 |

---

## 1. 环境与路径

### Q1.1 配置文件或 checkpoint 路径报错 `FileNotFoundError`

**现象**：运行脚本时提示 `No such file or directory: '.../kuavo_env.yaml'` 或 `Run directory does not exist`、`policy_preprocessor.json not found`。

**原因**：当前工作目录与配置里写的路径不一致，或训练输出目录/epoch 子目录不存在。

**处理**：

- 在**项目根目录**下执行命令（如 `cd /path/to/kuavo_data_challenge`），再用相对路径：`--config configs/deploy/kuavo_env.yaml`。
- 部署时确认 `inference.task`、`inference.method`、`inference.timestamp`、`inference.epoch` 与真实训练输出一致，或正确填写 `inference.checkpoint_run_dir`；run 目录下需有 `policy_preprocessor.json` 和 `epochN/`（或 `epochbest/`）。

### Q1.2 提示 `FileNotFoundError: Not a valid HuggingFace repo id ... outputs/train/.../epoch3`

**现象**：加载 preprocessor 或策略时，把本地路径 `outputs/train/...` 当成了 HuggingFace 的 repo id，导致校验失败。

**原因**：部分接口（如 `PolicyProcessorPipeline.from_pretrained`）要求传入的是合法 Hub ID 或**本地且含 config 的目录**；run 根目录需有 `config.json`、`policy_preprocessor.json` 等，而不是只传 epoch 子目录。

**处理**：用 run 目录（如 `outputs/train/<task>/<method>/<timestamp>/`）作为 preprocessor 的加载路径；模型权重路径为 run 下的 `epochN/`。部署脚本中已按此约定解析，检查 `checkpoint_run_dir` 或 task/method/timestamp 是否指到正确的 run 目录。

### Q1.3 没有 gcc / ccache 导致报错 `FileNotFoundError: 'ccache gcc'`

**现象**：开启 `torch.compile` 或某些依赖 Inductor 的代码时，提示找不到 `ccache gcc`。

**原因**：PyTorch Inductor 会调用系统 C/C++ 编译器，环境中未安装或未在 PATH 中。

**处理**：安装 gcc 与 ccache（如 `apt install gcc ccache`），或部署/推理时在配置中关闭编译：对 PI0/PI05 设置 `compile_model=false`（部署脚本中已默认关闭）。

---

## 2. 训练

### Q2.1 `DecodingError: The fields freeze_vision_encoder, train_expert_only are not valid for PI0Config`

**现象**：用 PI0 训练时，命令行或配置里传了 `--policy.freeze_vision_encoder`、`--policy.train_expert_only`，报错字段不存在。

**原因**：这两个参数属于 **PI05** 的配置项，**PI0** 的 `PI0Config` 中没有定义。

**处理**：训练 PI0 时不要加 `--policy.freeze_vision_encoder` 和 `--policy.train_expert_only`；仅在使用 PI05 时再按需加上。

### Q2.2 报错 `ValueError: 'policy.repo_id' argument missing`

**现象**：训练 PI0/PI05 时提示必须提供 `policy.repo_id`。

**原因**：默认配置或脚本假定会 push 到 HuggingFace Hub，因此要求填写 `repo_id`。

**处理**：若只做本地训练、不推送 Hub，在命令行加上 `--policy.push_to_hub=false`（或配置里 `policy.push_to_hub: false`），则无需提供 `repo_id`。

### Q2.3 训练时 CUDA 显存不足（Out of Memory）

**现象**：`torch.OutOfMemoryError: CUDA out of memory`。

**处理**：

- 减小 `batch_size`（如从 32 降到 16 或 8）；
- 使用 `CUDA_VISIBLE_DEVICES` 指定单卡，避免多卡误占显存；
- PI0/PI05 可开启 `gradient_checkpointing: true` 以降低显存；
- 使用更大显存 GPU 或减小模型/分辨率。

### Q2.4 训练时无法连接 HuggingFace（OSError / LocalEntryNotFoundError）

**现象**：下载 tokenizer 或模型时提示无法连接 `huggingface.co`，或本地缓存找不到。

**处理**：

- **离线**：设置 `HF_HUB_OFFLINE=1`，并保证 tokenizer/模型已提前下载到本地；在配置或环境变量中把 tokenizer 路径指到该本地目录（见 [PI0/PI05 策略说明](policy/pi0.md) 中的 HuggingFace 下载与 `PALIGEMMA_TOKENIZER_PATH`）。
- **网络**：如需在线下载，检查代理与 `huggingface-cli login`，并确认已接受 PaliGemma 等模型的使用条款。

---

## 3. 部署与推理

### Q3.1 仿真已启动，但脚本一直卡在 “Waiting for first env init”

**现象**：运行仿真自动测试时，脚本在等 init 信号，迟迟不往下执行。

**原因**：仿真器与部署脚本未使用同一 `ROS_MASTER_URI`，或仿真器没有在就绪时调用 `/simulator/init`。

**处理**：

- 确认两边在同一 ROS master（同一台机或正确设置 `ROS_MASTER_URI`）；
- 确认仿真器在就绪时会调用 `rosservice call /simulator/init`；
- 用 `rosservice list | grep simulator` 检查是否有 `/simulator/init`、`/simulator/reset` 等服务。

### Q3.2 部署时报错 `ValueError: No task found in complementary data`

**现象**：PI0/PI05 推理时，preprocessor 报“在 complementary data 中找不到 task”。

**原因**：tokenizer 等处理器要求观测里存在 `task` 键（语言条件），而当前观测没有提供。

**处理**：

- **无语言条件**：在送入 preprocessor 前，给观测加上占位：`observation["task"] = [""]`（部署脚本中已对 PI0/PI05 做此处理）；
- **有语言条件**：保证训练与部署都传入一致的 `task` 文本，且键名与训练时一致。

### Q3.3 训练是无语言 / 仅 RGB，部署时是否也要一致？

**答**：要。训练时若是**无语言、仅 RGB**，部署时也应**不注入语义 task（用空字符串占位）、只送 RGB**；否则输入分布不一致，容易导致效果差或报错。反之，若训练用了语言或 RGB+Depth，部署时也要提供对应模态。

---

## 4. 模型与权重

### Q4.1 加载权重时提示 `Missing key(s): embed_tokens.weight` 等

**现象**：`Remapped ... state dict keys`，并提示缺少 `language_model.embed_tokens.weight` 等。

**原因**：当前 checkpoint 是**微调后的策略**（只保存了专家头等部分），没有包含底座 VLM 的 `embed_tokens`；或底座与当前代码结构不一致。

**处理**：若仅做推理，且确认训练时就是部分加载，可忽略与底座 embedding 相关的 missing key，只要实际用到的参数已加载即可；若希望完整保存/加载底座，需用支持完整保存的脚本或配置保存完整 state_dict。该 missing key 一般**不表示你缺数据**，而是预训练底座里本来就不包含在本次保存的权重中。

### Q4.2 报错 `RuntimeError: expected input to have 3 channels, but got 1 channels`

**现象**：推理时出现 “weight of size [..., 3, ...], expected input ... to have 3 channels, but got 1”。

**原因**：视觉 backbone 按 RGB 三通道设计，但当前输入里把**深度图（1 通道）**当成了图像特征送入；或部署时仍传入 depth 而策略是 RGB-only。

**处理**：

- 部署为 **RGB-only** 时：在推理前从观测中移除 `observation.depth_h`、`observation.depth_l`、`observation.depth_r`（部署脚本中 PI0/PI05 已做）；
- 训练与部署的输入模态要一致（仅 RGB vs RGB+Depth）。

### Q4.3 PaliGemma tokenizer 路径变了怎么办？

**现象**：换机器或路径后，checkpoint 里保存的 tokenizer 路径失效，或报权限错误（如 `Permission denied: '/root/.../paligemma-3b-pt-224'`）。

**处理**：

- 设置环境变量：`export PALIGEMMA_TOKENIZER_PATH=/你的新路径/paligemma-3b-pt-224`，部署脚本会优先使用该路径；
- 或在代码中统一改默认常量（如 `sim_auto_test.py`、`real_single_test.py` 里的 `DEFAULT_PALIGEMMA_TOKENIZER_PATH`），保证只有一处写死路径；
- 确保该目录下有 tokenizer 所需文件（如 `tokenizer.json`、`special_tokens_map.json` 等）。

---

## 5. PI0 / PI05 专项

### Q5.1 `AttributeError: 'PI05Config' object has no attribute 'use_depth'`

**现象**：加载 PI05（或 PI0）配置时访问 `config.use_depth`、`config.depth_features` 报错。

**原因**：当前使用的 LeRobot 默认 `PI05Config`/`PI0Config` 可能未包含这些扩展字段；或 checkpoint 来自未带 depth 扩展的版本。

**处理**：在封装层（如 `PI05PolicyWrapper`、`PI0PolicyWrapper`）中，用 `getattr(config, "use_depth", False)`、`getattr(config, "depth_features", [])` 等方式安全读取；若确实不做 depth，可统一当 `use_depth=False`、`depth_features=[]` 处理，避免直接访问不存在的属性。

### Q5.2 `AttributeError: 'tuple' object has no attribute 'get_seq_length'`（past_key_values）

**现象**：PI0/PI05 推理时，某处期望 `past_key_values` 是带 `get_seq_length` 的对象，实际收到的是 `tuple`。

**原因**：不同 `transformers` 版本或 `torch.compile` 下，`past_key_values` 有时为 tuple 而非 `DynamicCache`。

**处理**：在模型前向中，若需调用 `get_seq_length()`，先判断类型；若是 tuple，可改为用序列长度推导或兼容两种形式的写法；或关闭 `compile_model` 使用 eager 模式推理。

### Q5.3 `TypeError: GemmaRMSNorm.forward() got an unexpected keyword argument 'cond'`

**现象**：PI05 前向中调用 `input_layernorm(..., cond=...)` 报错。

**原因**：官方 `GemmaRMSNorm` 的 `forward` 没有 `cond` 参数，是项目内自定义用法与当前 transformers 版本不一致。

**处理**：在调用处去掉 `cond` 参数，或改为仅传入该 LayerNorm 支持的参数（如 `hidden_states`）。

### Q5.4 `AttributeError: module 'transformers.models.gemma.modeling_gemma' has no attribute '_gated_residual'`

**现象**：在 PI05 相关代码里使用 `modeling_gemma._gated_residual` 报错。

**原因**：`_gated_residual` 在新版 transformers 中可能改名或移动位置。

**处理**：改为从 `transformers.models.gemma.modeling_gemma` 显式导入当前版本提供的名字（如 `_gated_residual` 或新名称），或使用该版本文档中推荐的 API。

### Q5.5 `TypeError: denoise_step() takes 5 positional arguments but 7 were given`

**现象**：调用 PI05 的 `denoise_step` 时参数个数不匹配。

**原因**：LeRobot 或本项目对 `denoise_step` 的签名做过调整，调用处仍按旧参数列表传参。

**处理**：对照当前 `modeling_pi05.py`（或 wrapper）里 `denoise_step` 的定义，修改调用处参数个数与顺序，只传定义中的参数。

### Q5.6 训练时加了“无语言”或“仅 RGB”，部署要注意什么？

**答**：

- **无语言**：部署时在观测里加 `observation["task"] = [""]`，满足 processor 对 `task` 键的依赖即可，不要注入真实任务描述。
- **仅 RGB**：部署时从观测中删除所有 depth 键，并在加载策略时从 `config.input_features` 中去掉 depth 相关键（当前部署脚本已对 PI0/PI05 做这两步）。

---

## 6. 其他

### Q6.1 kuavo_humanoid_sdk 版本不匹配导致机械臂初始化失败

**现象**：真机部署时报“机械臂初始化失败”或通信异常。

**处理**：下位机与上位机使用**同一版本**的 SDK。在下位机执行 `cd ~/kuavo-ros-opensource && git describe --tag` 查看版本，然后在本机执行 `pip install kuavo-humanoid-sdk==<该版本>`（如 `1.2.2`、`1.3.1b98`）。详见 [真机评测](deployment/real_eval.md) 与 `kuavo_deploy/readme` 相关说明。

### Q6.2 如何确认当前是“仅 RGB”还是“RGB+Depth”？

**答**：看两处：（1）**训练**：若在 `lerobot_train.py` 或数据管线里删除了 `observation.depth_*`，且 config 里未配 `depth_features`，即为仅 RGB；（2）**部署**：若在 `sim_auto_test`/`real_single_test` 里删除 depth 键并裁剪 `input_features` 中的 depth，即为仅 RGB 部署。两者需一致。

---

## 7. 相关文档

| 文档 | 说明 |
|------|------|
| [快速开始](getting_started/quick_start.md) | 环境、数据、训练与部署流程 |
| [训练流水线](training/pipeline.md) | 统一入口与各策略训练方式 |
| [仿真自动测试](deployment/sim_auto_test.md) | 仿真部署与配置 |
| [真机评测](deployment/real_eval.md) | 真机部署与 SDK 版本 |
| [PI0 策略说明](policy/pi0.md) / [PI05 策略说明](policy/pi05.md) | 预训练模型下载与配置 |

更多资讯与技术经验可前往 [OpenLET 资讯](https://openlet.openatom.tech/explore/journalism) 浏览。
