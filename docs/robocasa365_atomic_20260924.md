# RoboCasa365 原子任务诊断

用户要求先用简单原子任务验证 DiT4DiT 的学习能力，再推进 Composite。此前没有完成原子任务训练或策略成功率评估；旧的 0/10 属于 StirVegetables Composite。

**最新状态：**用户随后要求扩大数据并使用四卡。此单卡任务 `t-20260924173006-4fr79` 已在完成 **2,000 updates** 后停止，平台状态 Killed；`action_step_002000.pt` 已读回确认，247 个动作参数张量及 optimizer 均保留。开发评估启动后为切换资源而中断，完成 trial 数为 0；没有这轮策略成功率。下面的 10k 及 100 次评估是原计划，未完成。新主线见 [18 任务四卡实验](robocasa365_atomic18_20260924.md)，从官方 GR1 重新初始化，不能累计本轮 2,000 步，也不能与旧 StirVegetables step2000 混淆。

## 已固定的实验

- 官方 `atomic_seen` 中的 **PickPlaceCounterToCabinet**：将台面物体放进已经打开的橱柜。官方 horizon 为 750，成功必须满足物体在橱柜内且夹爪已离开。
- 使用官方 target human 数据 `20250811`，共 **502 条演示、131,904 帧**；按完整 episode 固定划分 **452 train / 50 validation**。
- 每条训练演示均可被采样；所有验证演示均排除在训练和归一化之外。连续动作统计只使用训练集。离线 loss 使用固定的 8 个验证窗口，是验证代理指标。
- 保留完整 state16/action12、底盘和 torso 真实命令。全数据动作往返转换最大误差为 `1.6484941756100824e-7`。RGB、state、action、padding 和 batch 的实际 CPU 验收已通过。
- 从已恢复、SHA256 验证通过的**官方 RoboCasa-GR1**初始化。官方文件为 21,283,661,981 bytes，SHA256 `fc65e78ab7c3de040d2df9f41416640e49befec58c784f550ecdaf41ee167098`，revision `46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44`。这次是新实验，不能称为旧 step2000 的续训。
- Cosmos 构造所需的模型张量从同一份恢复的 GR1 权重导出；读取旧 EFS 的仅为架构/tokenizer 元数据和 Python 运行环境。完整策略加载保持 `strict=True`。
- **1 × A800，10,000 optimizer updates，microbatch1 × accumulation4，500-step warmup**。训练 State Encoder、Action Encoder/Decoder 和 Action DiT；Video DiT、VAE、Qwen 文本编码器冻结。
- 先过 10-update 的有限 loss、非零梯度、实际参数更新、checkpoint 写回和恢复关卡。2,000 和 5,000 步分别在固定开发种子 100–119 上评估 20 次，再按相同优化器和学习率计划恢复训练。
- 固定最终 step10000，在**独立种子 1000–1099 上评估 100 次**，完整官方 horizon、官方成功判定。开发集、GT replay、demo 初始场景结果不混入最终成功率。
- 权重、配置和日志写入当前可访问的 vePFS `artifacts/` 与 `runs/`。云任务仅额外挂载只读 EFS 运行环境，不挂载失效的旧 NAS。

## 解释边界

这是 Atomic seen 中的 **1 个任务**，不是全部 18 个 Atomic seen 的成绩。成功能证明当前训练和接口至少能学习这个基础任务，不能单独证明 Composite 失败只由长序列导致；失败则继续定位模型控制和训练设置，不能把原因直接归结为 Composite 太难。

GT 在回放前固定选训练 episode 0、246、501，采用与策略一致的 Gym、三相机和渲染分辨率，仅恢复初始状态并重放原始动作。具体结果见 `runs/robocasa365_atomic_20260924/gt_replay_summary.json`。它们只验证演示动作链路。

## 后续 Composite 主线

全 16 个 Composite seen 的 7,271 条训练/806 条验证数据保持不变。最终评估已按用户要求改为**每任务 100 次，共 1,600 次**；新协议在 `runs/robocasa365_recovered_20260924/prepared/manifest.json`，原 320 次协议保留为历史版本。

新的大规模冻结视频基线配置为 50,000 updates，目前优先等待原子任务诊断。原基于旧 step2000 的 Video DiT A/B 尚未完成，旧权重仍无法访问；不会伪造该对照。部分解冻对照应明确使用同一个可访问的新 checkpoint，并与冻结视频分支比较，不能悄悄冒充旧实验。

## 产物

- `runs/robocasa365_atomic_20260924/data/datasets.json`：官方目录对照、下载与完整性记录。
- `.../prepared/manifest.json`、`normalization.json`：固定 split、归一化及开发/最终种子。
- `.../adapter_verification/acceptance.json`：实际全量动作和 RGB/batch 验收。
- `.../gt_replay_plan.json`、`gt_replay_summary.json`：预先指定的 GT 诊断及结果。
- `.../source/`、`source_hashes.json`：云端运行的源码快照。
- `scripts/volc/robocasa365_atomic_1gpu.yaml`：单卡云任务配置。
- `.../submission.json`、`stage.txt`、`train_live/`：实际提交和实时训练进展；是否存在以文件为准。
- `.../evaluation_100/report.json`：最终真实策略结果，尚未执行时不填成功率。
