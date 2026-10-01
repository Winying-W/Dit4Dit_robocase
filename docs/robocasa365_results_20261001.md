**截至 2026-10-01 15:11 UTC，四卡恢复任务正在运行。Composite 已保存到新增25,004步；25k开发评估最新独立审计为203/320次完成、1次成功。当前效果仍弱，完整16任务成功率尚未形成。**

Atomic18的50k最终结果仍为279/1,800（15.5%），其中五类搬运21/500（4.2%）。Composite10k完整开发结果为0/320。

25k进度采用2026-10-01 15:11 UTC的逐次审计快照：原171次完整记录保留，恢复后新增32次完整评估、0次成功，合计203次。全部203次保存的预测、反归一化、12维动作顺序和仿真实际接收动作通过独立核对。运行中的总表重启后按任务组重新发布，目前部分表只含20次；旧记录没有丢失。本报告按任务/种子独立核算完整轨迹，不能将部分总表计数下降当作结果丢失。

| 任务 | 已完成 / 计划 | 成功 |
| --- | ---: | ---: |
| DeliverStraw | 20/20 | 0 |
| GetToastedBread | 19/20 | 0 |
| KettleBoiling | 20/20 | 0 |
| LoadDishwasher | 20/20 | 0 |
| PackIdenticalLunches | 10/20 | 0 |
| PreSoakPan | 0/20 | 0 |
| PrepareCoffee | 20/20 | 0 |
| RinseSinkBasin | 20/20 | 0 |
| ScrubCuttingBoard | 20/20 | 0 |
| SearingMeat | 12/20 | 0 |
| SetUpCuttingStation | 0/20 | 0 |
| StackBowlsCabinet | 0/20 | 0 |
| SteamInMicrowave | 20/20 | 0 |
| StirVegetables | 0/20 | 0 |
| StoreLeftoversInBowl | 2/20 | 0 |
| WashLettuce | 20/20 | 1 |

唯一成功为WashLettuce seed113，503个控制步；该任务已完成20次，成功率1/20（5%）。各任务当前样本数不均衡，整体仍缺117次，因此不报告完整16任务宏平均或合并成功率。这是开发评估；最终16×100个独立新场景尚未开始。

主线任务为`t-20261001213401-8w54b`，最新云状态Running，阶段`development_25000_remaining_trials`。四路真实CUDA策略与仿真各8步均通过，短测试不计成功率。完成320次开发评估及审计后，将从25004保留Adam、学习率计划及四个rank随机数状态续训到50000，再执行最终1600次评估。权重没有丢失，最新保存和读回验证通过，独立文件系统备份观察器正在运行。

原训练任务因存储配额故障停止，新输出已迁至EFS。第一次恢复任务`t-20261001210533-48zqr`完成25000→25002、独立恢复25002→25004后，因过长TMPDIR导致`AF_UNIX path too long`。独立修复入口改用短`/tmp`通信路径，日志、视频及权重保存在EFS；本次已通过分配节点上的实际四卡仿真验证。原冻结源码、失败日志和权重均保留。

视觉对照已完成预排队验收：从同一Composite25k开始，两路各新增2000步，全16任务、7271训练/806验证演示。一条只训练动作分支；另一条额外训练Video DiT第16/17块并使用真实未来视频损失。两路及推理精度对齐，VAE和Qwen保持冻结。真实CPU两路更新、独立进程精确恢复、全部检查点审计及16项回归通过；CPU有效batch1不代替GPU batch64/显存验证。

两路候选均通过提交前检查；空闲卡观察器在出现资源后已提交冻结路`t-20261001231446-k5fwd`。截至2026-10-01 15:24 UTC，平台Running，分配节点的A800/64MiB存储/IPC检查通过，已完成独立恢复到第4步、权重读回通过，目前进入实际策略/仿真测试；完整GPU恢复、显存及策略/仿真门槛仍需验收后才继续2000步。联合路尚未提交，继续等待空闲卡。**目前没有视觉对照成功率结论。**这是新的Composite25k对照，不是原16条演示step2000实验。

此前普通采样/起始切换加权采样从同一Composite10k各新增2000步，均为0/60，60对初态及动作审计通过；不扩大该方案。现有低成功率尚不能唯一归因于训练步数或冻结视觉。动作链通过不等于视觉条件、物体定位、抓取和流程组合都正确；专家GT接触漂移仍未彻底解决。

[最新逐次审计](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_recovery_20261001/ipc_fix/observations/live_trials_20261001T151152Z.json) · [最新云端状态](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_recovery_20261001/ipc_fix/latest_observation.json) · [恢复前171次审计](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_recovery_20261001/recovered_25000_review.json) · [视觉对照证据](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/docs/robocasa365_visual_precision_20261001.md)

2026-10-01 15:13 UTC补充：取吸管25k seed100的2550步语义重放及独立比较完成，仍未开抽屉或抓取，全部记录状态与原评估完全一致。取面包重放继续进行。额外视觉对照的空闲卡观察器已启动，9项决策控制检查通过；资源可用时顺序提交，当前尚无新提交。[诊断详情](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/docs/robocasa365_composite25k_diagnosis_20261001.md)。

2026-10-01 15:24 UTC补充：取面包25k seed100的完整3000步重放和独立复算已完成，375次查询状态及初/终态均与原评估零误差一致，仍无拨杆启动、抓取或入盘。取吸管/取面包两例合计5550控制步已完成诊断与独立归档，不能增加评估分母。
