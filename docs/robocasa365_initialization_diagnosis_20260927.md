# Composite 初始化与 replay 补充诊断

截至 2026-09-27 09:07 UTC，Atomic18 的 50,000 步权重已保存并独立校验；四卡仍在完成原计划的 1,800 次最终评估。Composite16 尚未开始训练。以下是用于选择后续训练方案的 CPU 诊断，不增加闭环策略成功率。

## Atomic 权重能否用于 Composite

生产 adapter 对连续 action 使用 `min_max`。两轮 normalization 文件的分位数不同，但实际使用的 action min/max 相同：最小值为 `[-1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1]`，最大值为 `[1,1,1,0,1,1,1,1,1,1,1,1]`。mode/gripper 保留原始值，state 是原始 16D 补零到 64D。不能据未使用的 q01/q99 不同判断权重不兼容。

已在全部 16 个 Composite 任务、每任务训练/验证各一条演示的初始和中段窗口上，实际比较两种统计配置生成的 state、action 和 mask，并验证合成预测通过正式 decoder 后逐值一致。共 32 条不同演示、64 个观测窗口；重复噪声不算新演示。

新增显式 `train_distributed.py --warm-start ... --warm-start-run ...`，仅载入经审计的 Atomic 动作分支。它检查源权重 SHA、原 GR1 起点、实际 adapter 源码、输入合同、有效归一化及完整动作参数集合。optimizer、scheduler、sampler 重新建立，Composite 更新步数从零计算；不能和 `--resume` 同时使用。后续恢复读取 Composite 自己的 checkpoint，并保留 Atomic 来源记录。

从独立的 150 文件源码快照实际完成：

- 6 项 warm-start 合同和拒绝错误输入的测试；
- 真实完整 Action DiT 加载：247 个张量、163,276,320 参数与 Atomic50k 逐值一致，非动作哨兵参数及 CPU RNG 不变，新 optimizer state 为空；进程退出码为 0；
- 正式 trainer 的四进程 CPU 导入和 Gloo 通信；进程退出码为 0。

证据：`runs/robocasa365_atomic_warmstart_controls_20260927/cpu_acceptance_summary.json`。这些检查没有执行新的训练更新，也没有验证 CUDA 显存、NCCL 或 GPU checkpoint 恢复。

两种初始化的离线预测比较已完成，进程退出码为 0：发布版 GR1 动作分支与 Atomic50k 动作分支共享相同冻结 Cosmos 特征，使用配对噪声和 4 次积分。全部 256 次 action chunk 预测已保存并经独立 NPZ 复算，只统计实际执行的前 8 步。此结果不能证明闭环成功率提高。

决定把 Atomic50k 动作权重初始化作为优先验收的 Composite 候选，重新建立 optimizer，Composite 步数从 0 计算。现有 `runs/robocasa365_composite16_4gpu_20260927/` 保持 GR1 初始化方案作为备选。新的初始化需要独立实验目录、冻结源码和排队前验收，当前尚未完成这些新验收或提交云端。当前云端 Atomic 源码和已有 Composite 快照均未修改。


### 已完成的配对离线结果

下表是控制命令尺度的平均绝对误差，不是 EEF 实际位姿误差。每个任务的训练和验证各只有一条演示、初始/中段两个窗口、两次配对噪声；不能据此计算可靠的任务级置信区间或外推总体成功率。

| 任务 | 训练 arm MAE：GR1 → Atomic50k | 验证 arm MAE：GR1 → Atomic50k |
| --- | ---: | ---: |
| DeliverStraw | 0.2724 → 0.1461 | 0.2087 → 0.1509 |
| GetToastedBread | 0.1813 → 0.0954 | 0.2293 → 0.0710 |
| KettleBoiling | 0.3723 → 0.2090 | 0.2326 → 0.1133 |
| LoadDishwasher | 0.2200 → 0.1827 | 0.2482 → 0.1419 |
| PackIdenticalLunches | 0.1983 → 0.0858 | 0.3022 → 0.1737 |
| PreSoakPan | 0.2818 → 0.1240 | 0.2311 → 0.1194 |
| PrepareCoffee | 0.2348 → 0.1738 | 0.2325 → 0.1468 |
| RinseSinkBasin | 0.4473 → 0.2728 | 0.2420 → 0.1156 |
| ScrubCuttingBoard | 0.2977 → 0.2225 | 0.2686 → 0.2000 |
| SearingMeat | 0.4052 → 0.2538 | 0.3446 → 0.1238 |
| SetUpCuttingStation | 0.3311 → 0.1803 | 0.2831 → 0.0981 |
| StackBowlsCabinet | 0.3107 → 0.1155 | 0.2836 → 0.2320 |
| SteamInMicrowave | 0.2510 → 0.1658 | 0.1860 → 0.1193 |
| StirVegetables | 0.2830 → 0.1483 | 0.1777 → 0.0748 |
| StoreLeftoversInBowl | 0.2370 → 0.1332 | 0.3004 → 0.1208 |
| WashLettuce | 0.3797 → 0.2252 | 0.4060 → 0.2284 |

验证集合并 arm MAE 从 0.26103 降至 0.13936；平移命令 MAE 从 0.24877 降至 0.21942，旋转命令 MAE 从 0.27329 降至 0.05930。改善主要来自旋转部分，不能把整体降幅都解释成抓放能力提高。夹爪命令准确率从 76.17% 升至 98.05%，mode 正类召回从 0/72 升至 26/72；这些分母是相关命令位置，并非独立场景。mode 召回仍弱，后续需要闭环验证。

证据：`runs/robocasa365_composite_initialization_20260927/independent_review.json`，包含全部任务和训练/验证的独立复算；决策记录为同目录 `initialization_decision.json`。

## replay 物理恢复诊断的边界

实际比较了 fresh native、原数据 XML 的 episode0 和 episode3 三种场景：相同完整 `mjSTATE_INTEGRATION` 和固定 motor ctrl 下，原模型重复执行 250 个物理 tick 的误差均为 0；但将模型通过 `get_xml()` 再导出并重新编译后，从第一 tick 开始分叉，且质量、惯量等模型参数发生变化。

| 场景 | 250 tick 的 qpos 最大差 | qvel 最大差 |
| --- | ---: | ---: |
| fresh native | 0.00108959 | 0.261299 |
| recorded XML episode0 | 0.00244527 | 1.770966 |
| recorded XML episode3 | 0.01567979 | 0.301806 |

qpos 混合多种自由度和单位，表中数值不能直接解释成毫米位置误差。官方录制代码也明确避免用这种再导出 XML 进行物理恢复，改为使用原始环境 XML。

当前 GT reset 直接加载数据集记录的原 XML，没有上述再次导出步骤，因此这项结果不能直接解释现有 GT 接触漂移。原 episode0/3 漂移仍未证明完全解决；原数据 action 为 float64，也不支持此前猜测的 float32 压缩解释。没有修改共享仿真库或运行中的评估。证据：`runs/robocasa365_replay_model_precision_20260927/audit/report.json`。

## 原 step2000 的存储状态

已成功导出原云任务 `t-20260923152847-l2xf9` 的配置，确认旧 NAS 的 ID 与挂载地址。当前进程没有该 NAS 挂载，域名解析失败，未实际建立 NFS 连接；当前账户的开发实例列表为空。这些结果只说明当前不能访问，不能证明旧文件已删除。

原 16-demo StirVegetables step2000 仍未恢复。冻结/部分解冻 Video DiT 的精确历史起点 A/B 保持未完成，不能用 Atomic50k 或同名的其他 step2000 替代。证据：`runs/robocasa365_legacy_step2000_20260924/storage_recovery_20260927/`。
