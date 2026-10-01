# Composite16：Atomic50k 初始化的训练版本

2026-09-27，`runs/robocasa365_composite16_atomicinit_4gpu_20260927/` 的新版本已完成软件/数据排队前验收。149 个源码文件已冻结。尚未提交 GPU 作业，Composite 训练更新和策略评估次数均为 0；等待当前 Atomic18 最终评估、独立复核和四卡资源释放。

保留完整 Composite seen 16 任务：7,271 条训练演示、806 条验证演示，按完整演示隔离；训练 5,403,990 帧，验证 598,275 帧。任务集合、数据、归一化和完整 12D 动作合同未改变。任务并非全部固定底盘，保留全部 base 维度。

初始化使用发布 GR1 的 Video DiT、VAE、文本编码器，加上已审计的 Atomic18 step050000 动作分支。训练全部 State Encoder、Action Encoder/Decoder 和 Action DiT；其他部分冻结。只迁移动作权重，不继承 Atomic 的 optimizer、scheduler、sampler，Composite 更新步数从 0 开始。

| 设置 | 值 |
| --- | --- |
| 资源 | 4×A800，最大并发不超过 4 张 |
| microbatch / accumulation / global batch | 4 / 4 / 64 |
| Composite 新增训练步数 | 50,000 |
| 动作学习率 | 峰值 1e-4，结束 1e-5，warmup 1,000 步 |
| 开发评估 | 10k、25k 各 16×20 次，seeds100–119 |
| 最终评估 | 50k，16×100 次，seeds1000–1099 |
| 场景协议 | `stable_counter_v1`，核对实际场景身份 |

本轮从新快照实际执行了完整模型 CPU 前向/反向，batch 包含 DeliverStraw、GetToastedBread、KettleBoiling、LoadDishwasher。loss 为 0.1435546875，247 个动作参数张量全部有有限梯度；State Encoder、Action Encoder、Action Decoder 梯度范数分别约 0.05816、0.29866、1.15071。该检查没有 optimizer 更新，不是策略成功率或 CUDA 验收。正式 trainer 的四进程 CPU 导入和 Gloo 通信也重新执行并退出 0。

数据、adapter、场景初始化、相机和通用 DDP 数值/恢复检查沿用经哈希确认源码完全相同的旧证据，保留原时间戳，没有声称重新执行。Atomic 权重加载的 6 项合同测试和真实 247 张量逐值加载检查也按源码身份复用；新版本对实际 warm-start 完整模型前向/反向和 trainer 导入另做了检查。证据及复用范围记录在 `prequeue_acceptance.json`、`warm_start_evidence_reuse.json`、`fresh_checks.json`。

云入口仅首次训练使用 `--warm-start`。后续每次都使用本轮 `--resume`。取得四卡后必须依次通过：全部 16 任务真实三相机渲染、10 步四卡训练、独立进程恢复到 12 步、四卡策略/仿真接口短测。12 步时额外检查 optimizer 和 scheduler 均为本轮 12 步，来源记录仍为 Atomic50k，四 rank 参数一致；通过后才长训。排队前检查不能保证尚未分配的 CUDA、NCCL 和显存一定正常。

Atomic 初始化的依据是 32 条演示、64 个观测窗口、256 次配对离线预测：16 个任务的所抽验证窗口 arm 命令 MAE 均降低。它支持作为后续训练候选，不证明其闭环成功率更高或方案已最优。原 GR1 初始化版本 `runs/robocasa365_composite16_4gpu_20260927/` 继续保留。详细离线范围见 `docs/robocasa365_initialization_diagnosis_20260927.md`。

当前工作集中于现有 50k 权重、剩余 Atomic 评估和新 Composite 训练。旧 16-demo step2000 只用于历史起点复现，缺失不影响此版本。2026-09-27 09:28 的 0 GPU 恢复提交因平台找不到原 NAS 资源 ID 而失败，按任务名查询也没有创建任务；不继续把旧 NAS 恢复放在当前推进路径上。精确历史起点的 A/B 仍未完成，不以新实验冒充其结果。
