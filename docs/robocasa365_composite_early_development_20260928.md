**Composite seen16：早期开发评估完整结果**

证据核验时间：2026-09-28T01:31:01.557135+00:00。Atomic50k 初始化后，新增 12 步与新增 2,000 步的两轮开发评估均为 **0/320（0%）**。每轮覆盖全部 16 个 Composite seen 任务，各 20 个新 reset 场景，种子 100–119。两轮按权重分别报告，不将 640 次合并为一份模型的成功率。

| 任务 | 新增12步 | 新增2000步 | 每轮成功率 |
| --- | --- | --- | --- |
| DeliverStraw | 0/20 | 0/20 | 0% |
| GetToastedBread | 0/20 | 0/20 | 0% |
| KettleBoiling | 0/20 | 0/20 | 0% |
| LoadDishwasher | 0/20 | 0/20 | 0% |
| PackIdenticalLunches | 0/20 | 0/20 | 0% |
| PreSoakPan | 0/20 | 0/20 | 0% |
| PrepareCoffee | 0/20 | 0/20 | 0% |
| RinseSinkBasin | 0/20 | 0/20 | 0% |
| ScrubCuttingBoard | 0/20 | 0/20 | 0% |
| SearingMeat | 0/20 | 0/20 | 0% |
| SetUpCuttingStation | 0/20 | 0/20 | 0% |
| StackBowlsCabinet | 0/20 | 0/20 | 0% |
| SteamInMicrowave | 0/20 | 0/20 | 0% |
| StirVegetables | 0/20 | 0/20 | 0% |
| StoreLeftoversInBowl | 0/20 | 0/20 | 0% |
| WashLettuce | 0/20 | 0/20 | 0% |

每轮合并成功率与任务宏平均均为 0%；试验层面的 Wilson 95% 区间为 0%–1.186%。每任务 0/20 的区间约为 0%–16.11%，因此不能据此精确估计单任务成功率，也不能断言真实成功率严格为零。

每轮均使用 stable_counter_v1、fresh target reset、官方完整 horizon 和官方成功判定，预测动作每次执行8步后重新观测。320/320 配对场景的初始 XML 文本及完整物理状态一致。两轮均有 741,000 个控制步通过保存动作链独立审计；每轮320次策略失败均耗尽官方 horizon，基础设施失败 attempt 为0。

配对结果 gained=0、lost=0，观测差值为0。这组全零数据不能证明两份策略等价，退化的配对 bootstrap 区间不能用于等价性结论。

| 权重 | 提交到独立审计完成（4卡） | 单场景中位时长 | 单场景均值 |
| --- | --- | --- | --- |
| 新增12步 | 5.26小时 | 204.7秒 | 219.7秒 |
| 新增2000步 | 5.25小时 | 204.1秒 | 217.5秒 |

以上时长包含实际提交后的启动、评估和独立审计，不是纯仿真耗时；新任务排队及机器条件会改变耗时。

新增12步是启动后的基线，不是未经 Composite 更新的 Atomic50k。新增2000步与历史16演示 StirVegetables step2000无关，也不是 Video DiT 解冻对照。两轮早期结果不代表后续10k/25k/50k；正式50k仍计划16×100=1,600次独立最终评估。

动作链通过不足以判定视觉/状态接口全部正确；全零也不能独自证明训练步数不足、任务太难或需要解冻某个模块。保存策略轨迹的语义诊断另行核验，历史GT接触漂移仍未证明全部解决。

完整机器可读汇总：[summary.json](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_early_development_report_20260928/summary.json)。
全部配对场景证据：[comparison_start12_vs_composite2000.json](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite16_step002000_dev_20260927/comparison_start12_vs_composite2000.json)。
