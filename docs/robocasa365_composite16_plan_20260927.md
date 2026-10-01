# Composite16 新场景协议版本与排队前证据

记录时间：2026-09-27T07:09:05.642558+00:00。新实验目录为 `runs/robocasa365_composite16_4gpu_20260927/`。软件/数据 prequeue 已通过，尚未提交云端，没有新训练更新或策略成功率。

保留官方 Composite seen 全部 16 任务：7,271 条训练演示、806 条验证演示，合计 6,002,265 帧；按完整演示划分。相对旧 Composite 方案，训练数据、拆分、归一化、模型及训练源码均不变。完整 12D action 保留，不能把所有任务当作固定底盘任务。

| 任务 | 训练演示 | 验证演示 | 含非零 base 命令的演示 / 全部演示 |
| --- | ---: | ---: | ---: |
| DeliverStraw | 454 | 50 | 504/504 |
| GetToastedBread | 455 | 51 | 506/506 |
| KettleBoiling | 451 | 50 | 28/501 |
| LoadDishwasher | 451 | 50 | 297/501 |
| PackIdenticalLunches | 451 | 50 | 501/501 |
| PreSoakPan | 451 | 50 | 67/501 |
| PrepareCoffee | 463 | 51 | 0/514 |
| RinseSinkBasin | 458 | 51 | 2/509 |
| ScrubCuttingBoard | 454 | 50 | 41/504 |
| SearingMeat | 451 | 50 | 501/501 |
| SetUpCuttingStation | 454 | 50 | 475/504 |
| StackBowlsCabinet | 463 | 52 | 29/515 |
| SteamInMicrowave | 460 | 51 | 511/511 |
| StirVegetables | 451 | 50 | 388/501 |
| StoreLeftoversInBowl | 453 | 50 | 503/503 |
| WashLettuce | 451 | 50 | 21/501 |

上表是原始数据的命令统计。仅 PrepareCoffee 的全部已记录演示均为零 base 命令；这不能证明所有新场景都不需要移动底盘。mode 不等同底盘开关。原小样本 StirVegetables 的固定底盘属性也不能外推到全部 501 条演示。证据：`base_command_applicability.json`。

本轮已实际完成：38 项协议/评估检查、10 项四进程梯度与恢复检查；16 个真实 CPU 环境各 reset 并执行 8 步完整 12D 动作，共 128 步；重新读取并核对全部 8,077 个 state/action parquet 文件哈希。三相机在 CPU 检查中是黑图占位，未作为渲染通过的依据。视频文件没有重新全量计算哈希。

模型及其训练依赖源码、数据和归一化未变，旧版真实 CPU 全模型 forward/backward 与训练入口导入证据经哈希核对后复用；没有宣称这两项本轮重新执行。旧录制 demo 的相机对齐证据只支持相机/图像约定，不代表新场景协议的 GT replay 验收。所有证据及复用边界见 `prequeue_acceptance.json`、`unchanged_evidence_reuse.json`。

新协议 `stable_counter_v1` 仅在新仿真进程内保序去重 Counter 候选区域；共享仿真库、当前 Atomic18 云端快照和旧 Composite 快照不变。开发、最终和四卡接口评估显式使用同一协议，禁止混入原版结果。

取得四卡后，入口先检查完整 16 任务的真实三相机渲染，再做 10 步四卡训练、独立进程恢复到 12 步和四卡策略/仿真接口检查；全通过后才进入长训练。任何硬件检查失败即停止该入口，不能据 CPU prequeue 保证云端硬件不报错。

当前准备的训练方案仍为官方 GR1 初始化、global batch64、50,000 updates，训练 State Encoder、Action Encoder/Decoder 和 Action DiT；Video DiT、VAE、Qwen 冻结。10k/25k 各 16×20 开发试验，50k 使用 16×100 个与开发分离的最终场景。提交前继续复核完整 Atomic18 结果和失败诊断；若据此调整训练方案，需要新版本验收，当前版本不被称为已证明最优方案。

提交条件仍是当前四卡任务已终态并释放资源、已有结果完成复核；总并发不超过 4×A800。原 16-demo step2000 的三个路径在 06:51 UTC 复查仍不可访问，精确历史起点的冻结/部分解冻 A/B 尚未完成。新实验不能替代该要求。
