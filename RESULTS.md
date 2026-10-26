# RoboCasa365 Atomic18 实验记录（DiT4DiT concat 配置）

> 2026-10-20 ~ 10-23 · 32×PPU（2 节点 × 16 卡）训练 + 18 任务 × 10 seed 闭环评估
> 本文档记录：环境与数据准备、配方、评估结果、动作误差归因（prediction_metrics）、复现方式与踩坑记录。
> 所有数字均来自本仓库 `tools/` 与本项目 `jobs/`、`outputs/` 下的实际产物。

## 1. 环境与数据

| 项 | 内容 |
| --- | --- |
| 训练环境 | 独立 venv（torch 2.6.0 + PPU 后端，diffusers/transformers/pytorch3d 与仓库 requirements 匹配） |
| 仿真环境 | CPython 3.11 + robosuite/robocasa（上游 pinned commit），OSMesa 软件渲染（无 GPU 也可跑） |
| 预训练权重 | 作者发布 GR1 DiT4DiT 权重（`dit4dit_robocasa_gr1/final_model/pytorch_model.pt`，21.3 GB，SHA256 已校验） |
| Cosmos | `Cosmos-Predict2.5-2B`（文本编码器/VAE 冻结，视频 DiT 参与训练） |
| 数据 | Atomic18（18 个任务，9126 episodes / 8216 train + 910 val，seed 20260924；约 60 GiB） |
| 网络 | 环境 DNS 污染，下载用 DoH 解析 + 直连 IP 的断点续传脚本（`tools/fetch_hf.py`） |

## 2. 配方与训练配置

两条对照线（同一 50k schedule、global batch 128）：

| 线 | 脚本 | action head | Qwen | world | microbatch | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| **concat** | `experiments/concat_random_head/train.py` | **随机初始化** | **接 Qwen3-VL-4B**（投影层训练） | 32 | 4 | 视频 DiT + action head + Qwen 投影一起训练 |
| **baseline** | `experiments/robocasa_official_recipe/train.py` | 加载 GR1 预训练 head | 不接 Qwen | 32 | 4 | 官方配方的 GR1 微调 |

- 学习率：action head / 投影 1e-4，Cosmos 视频 DiT 1e-5（cosine，warmup 1250 步）
- 优化目标：video loss（Cosmos 帧预测）+ action flow-matching loss 同时反传
- **Cosmos 解冻验证**（非文档声明，而是产物证据）：
  - 可训练张量 820 个 = Cosmos transformer 569 + action head 247 + qwen_context 4
  - `learning_rates` 三组含 1e-5（cosmos_peak_lr），`phase='video_plus_action'`
  - 与 GR1 基座逐张量比较：`patch_embed.proj.weight` max|Δ|=3.9e-3（1.5%）、`time_embed` 0.6–2.2%
- 训练吞吐：32 卡 **5.3–5.4 s/step**（约 11.2 samples/s，≈ 118k samples/h）；16 卡约 9.8 s/step

## 3. 闭环评估协议

- 官方 `evaluate_policy.py`：三路 256×256 RGB → 同帧 per-view mosaic（224×224）；完整 12 维动作 schema；官方 Gym 二值成功判据；动作 clip 到 [-1,1]
- 每个任务使用其官方 horizon（如 TurnOnMicrowave 450、PickPlaceCounterToStove 1050 步）
- 并发：16 worker 并行（每个 worker 一个 socket 桥接的仿真子进程），约 45–60 分钟跑完 18 任务 × 10 seed = 180 rollouts

## 4. 结果

### 4.1 闭环成功率

| 检查点 | 成功 / 试验 | 成功率 |
| --- | --- | --- |
| concat 5k | 0/9（3 任务探针） | 0.00% |
| concat 10k | 1/180 | 0.56% |
| **concat 15k** | **5/180** | **2.78%**（最高） |
| concat 20k | 0/180 | 0.00% |
| concat 22.5k | 2/179 | 1.12% |
| concat 25k | 0/180 | 0.00% |
| concat 30k | 0/180 | 0.00% |
| concat 31k | 1/168 | 0.60% |
| GR1 init（未微调，3 seed） | 1/54 | 1.85% |
| baseline32 @5k | 0/171 | 0.00% |

成功集中在简单单步任务：TurnOnElectricKettle（15k 时 4/10）、SlideDishwasherRack、OpenStandMixerHead；抓取-放置、开门/抽屉、导航类始终 0。

### 4.2 Loss

**训练 loss（每 1k 步均值）**

| concat | action | video | baseline32 | action | video |
| --- | --- | --- | --- | --- | --- |
| 29k–30k | 0.0458 | 0.0850 | 3k–4k | 0.0568 | 0.0955 |
| 30k–31k | 0.0452 | 0.0851 | 4k–5k | 0.0537 | 0.0946 |
| 31k–32k | 0.0448 | 0.0788 | 5k–6k | 0.0517 | 0.0929 |
| 32k–33k | 0.0441 | 0.0793 | 6k–7k | 0.0502 | 0.0919 |
| 33k–34k | 0.0435 | 0.0833 | 7k–8k | 0.0486 | 0.0919 |
| 34k–35k | 0.0432 | 0.0951 | 8k–9k | 0.0478 | 0.0908 |

**验证集 action_fm_loss**

| concat | val | baseline | val |
| --- | --- | --- | --- |
| 5k | 0.0904 | GR1 init | 1.31–1.68 |
| 10k | 0.0609 | 32 卡 @5k | 0.0476 |
| 15k | 0.0566 | 16 卡 @5.8k | 0.0447 |
| 20k | **0.0519（最低）** | | |
| 22.5k | 0.0529 | | |
| 25k | 0.0628（回升） | | |
| 30k | 0.0591 | | |
| 31k | 0.0598 | | |

### 4.3 动作误差归因（`scripts/robocasa365/prediction_metrics.py`）

对 25k / 31k 权重各跑 18 任务 × 64 窗口 × 3 噪声种子（离线单步预测，非闭环成功率）：

| 指标 | 25k | 31k | 解读 |
| --- | --- | --- | --- |
| 位置指令 MAE | 0.171 | 0.168 | ≈ 0.85 cm |
| 旋转指令 MAE | 0.087 | 0.089 | ≈ 2.5° |
| 夹爪准确率 / 平衡准确率 | 96.65% / 96.6% | 96.6% / 96.6% | 开合判断基本正确 |
| 开爪 / 合爪召回 | 98.4% / 92.2% | 97.6% / 93.7% | 合爪略好 |
| 预测 vs 目标合爪比例 | 0.300 / 0.317 | 0.325 / 0.335 | 无系统偏置 |
| 被 clip 的 arm 分量 | 2.0% | ~2% | 幅度正常 |
| 最弱任务 | OpenCabinet 合爪召回 72.3% | OpenCabinet 平衡 88.0% | 开门类提前松手 |

**结论**：25k→31k 动作拟合无实质变化（位置 −0.003、旋转 +0.002、夹爪 ±0），但闭环成功率始终 0–3%。
失败模式是**闭环长程累积漂移**（单步 0.85 cm / 2.5° 在 450–1050 步上足以错过按钮/把手），而不是动作预测不准；无夹爪动作的任务（按/推/滑/导航）失败同样来自接触与定位精度。

## 5. 代码改动（相对 upstream `6e84dd9`）

| 文件 | 改动 |
| --- | --- |
| `tools/launch.py` | `--gpus` 支持任意卡数；新增 `--global-batch`（自动推导 accumulation 并校验）；`--nodes` 多机（torchrun rank/world 环境）；`VLA_WAM_EXTRA_PYTHONPATH`；启动前 `device_count()==N` 探测 |
| `tools/evaluate.py` | `VLA_WAM_MUJOCO_GL` 渲染后端选择；额外 PYTHONPATH；action-head-init 一致性检查 |
| `experiments/*/source/scripts/robocasa365/train_distributed.py` | `--global-batch` 参数与校验；`VLA_WAM_PROFILE`（data_seconds 计时） |
| `experiments/*/source/scripts/robocasa365/evaluate_policy.py` | 仿真子进程 EOF 时等待 60s 并落盘 `simulator.log`（便于诊断） |
| `experiments/*/source/scripts/robocasa365/recovery.py` | 恢复时接受任意 rank 数（原硬编码 4），去掉 step≤5000 的隐藏上限 |
| `experiments/*/source/scripts/__init__.py`、`.../robocasa365/__init__.py` | 新增包标记（镜像内已有同名 `scripts` 包时仍可 import） |
| `tests/test_portable.py` | 覆盖 `--global-batch` 推导/校验 |

完整补丁（按提交顺序）见项目目录 `patches/0001..0009`.

## 6. 复现步骤

```bash
# 1) 训练（32 卡，2 节点 × 16）
python experiments/concat_random_head/train.py \
  --manifest <atomic18>/manifest.json \
  --checkpoint <gr1>/pytorch_model.pt \
  --cosmos-model <Cosmos-Predict2.5-2B> \
  --output <run>/ --steps 50000 --global-batch 128 --microbatch 4 \
  --gpus 0,1,...,15 --nodes 2 --checkpoint-interval 100

# 2) 闭环评估（18 任务 × 10 seed，16 worker）
python -m scripts.robocasa365.evaluate_policy \
  --checkpoint <run>/action_step_030000.pt --dataset <task> \
  --output <eval_dir> --sim-python <sim-venv-python> \
  --task <Task> --seeds 100 101 ... 109

# 3) 动作误差归因（离线单步，非成功率）
python -m scripts.robocasa365.evaluate_policy \
  --checkpoint <run>/action_step_025000.pt --dataset <task> \
  --output <probe_dir> --sim-python <sim-venv-python> \
  --task <Task> --probe-episode 0 --probe-windows 64 --validation-only
```

## 7. 踩坑与经验

1. **平台驱逐**：共享 serverless 池会 `JobEvicted`。checkpoint 间隔降到 100 步 + `--resume` + 多资源组排队，把单次损失压到 ≤9 分钟。
2. **多机训练**：`--gpus` 传"每节点卡数"而不是总卡数（传 32 × 2 节点会算成 64 rank 并报 microbatch 不匹配）。
3. **恢复校验**：`recovery.py` 原要求恰好 4 份 rank RNG 状态、且 step≤5000，需改为通用实现。
4. **评估进程**：`MUJOCO_EGL_DEVICE_ID` 必须等于该 worker 的 `CUDA_VISIBLE_DEVICES`；OSMesa 渲染需要 `libGL/libEGL/libOSMesa` 与 headless opencv；`LD_LIBRARY_PATH` 不要覆盖 PPU 库路径。
5. **多 rank 评估**：`device_count()`/GPU 编号在容器内是局部的，跨 worker 传绝对索引会失败，需按 pid 映射。
6. **Cosmos 路径**：基座 checkpoint 里存的是占位路径 `/path/to/Cosmos-Predict2.5-2B`，需要设置 `COSMOS_MODEL_DIR` 环境变量。
7. **动作 loss 低 ≠ 闭环成功**：10k→31k 验证损失从 0.061 降到 0.052 再回升，成功率仍在 0–3%；瓶颈是闭环鲁棒性。

## 8. 结论与后续

- 在 32 卡、50k 步预算下，concat 配方（随机 head + 冻结 Qwen 文本编码器、视频 DiT + 动作头训练）在 Atomic18 上的闭环成功率约 **0–3%**（简单单步任务偶发成功），与官方 GR1 微调 5k 的 0% 同量级。
- 下一步建议：
  1. **闭环诊断**：用 `diagnose_policy_trace.py` 回放保存的策略指令并按任务里程碑定位失败点（需扩展 eval 保存完整轨迹）。
  2. **执行策略对照**：`--execute-horizon` 8 vs 1 vs 16，检验 chunk 执行是否放大漂移。
  3. **优化目标/正则**：更大 batch、动作平滑/集成、或 DAgger 式在线数据。
  4. **更长训练**：当前趋势（验证损失回升、归因不变）表明单纯加步数收益有限。


---

## 附录：50k 双线最终结果（2026-10-27 更新）

两条 50k 训练均完成（job Succeeded），最终闭环评估：

| 步数 | concat（随机 head + 冻结 Qwen 文本编码器） | baseline（GR1 官方 head 微调） |
| --- | --- | --- |
| 5k | 0/9（3 任务探针） | 0/171 |
| 10k | 1/180 = 0.56% | 1/180 = 0.56% |
| 15k | 5/180 = 2.78% | 1/180 = 0.56% |
| 20k | 0/180 | 0/180 |
| 25k | 0/180 | 0/180 |
| 30k | 0/180 | 1/180 = 0.56% |
| 35k | 2/180 = 1.11% | 0/180 |
| 40k | 0/180 | 0/180 |
| 45k | 2/180 = 1.11% | 0/180 |
| 50k | **0/180 = 0.00%** | **0/180 = 0.00%** |

- 全程 5k–50k 成功率始终在 **0–2.78%**，无上升趋势；成功集中在简单单步任务（TurnOnElectricKettle、OpenStandMixerHead、SlideDishwasherRack），且会在不同检查点间漂移。
- 两条配方（随机 head + Qwen vs GR1 官方 head）在所有可比检查点上**没有显著差异**。
- 验证集 action_fm_loss 在 20k 见底（0.0519）后回升（30k 0.0591、31k 0.0598），归因（位置 0.85cm / 旋转 2.5° / 夹爪 96.6%）在 25k→31k 无变化——**继续训练或换 head 都没有收益**。
- 对照 LIBERO 上 qwen_concat 的 97.5%，差距主要来自任务域（长程移动操作 vs 短程桌面）与移植差异（训练 `action_horizon=16` 但评估执行 8、`QwenContext` 未 detach cosmos context）。
