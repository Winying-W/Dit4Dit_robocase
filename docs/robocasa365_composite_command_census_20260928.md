**Composite16 全量动作统计：应检查切换时机与控制模式**

统计与独立复算完成于 2026-09-28 03:25 UTC。覆盖当前实际训练 manifest 的全部 8,077 条演示：训练 7,271 条、5,403,990 帧，验证 806 条、598,275 帧。两次读取均核对全部原始 parquet 的 SHA256 与 manifest 一致。独立复算使用另一种切换窗口计数方法，逐演示和 32 个 task/split 分组均无不一致。

当前冻结训练代码先均匀选任务，再均匀选该任务的训练帧。下表为各任务帧比例的等权平均，因此训练列描述这一采样规则下的概率，不是重建的实际抽样次数。验证列仅为同口径的数据描述，不能当成现有固定验证窗口的实际分布。

| 原始命令统计 | 训练 | 验证 |
| --- | ---: | ---: |
| 底盘模式命令 | 10.54% | 10.80% |
| 夹爪关闭命令 | 44.44% | 44.67% |
| 未来前 8 步包含模式切换的起点 | 1.59% | 1.68% |
| 未来前 8 步包含夹爪切换的起点 | 3.92% | 3.89% |
| 未来前 8 步包含上述任一切换的起点 | 5.51% | 5.57% |
| 未来前 16 步包含上述任一切换的起点 | 11.78% | 11.92% |

“包含切换”要求窗口中同时出现切换前后的录制标签；尾部仅计真实有效帧。所有原始 mode/gripper 标签均为 -1 或 +1。命令切换不是物理抓取、接触或设备启动标签；以上比例也不是成功率或监督是否充分的判定。

夹爪关闭命令本身不稀少。即使某条策略轨迹从未关闭夹爪，也不能据此推断训练数据缺少关闭样本。值得优先比较的是切换附近的预测、时机以及实际控制模式，而不是直接提高全部关闭命令的损失权重。切换窗口密度较低只是提出诊断方向，还不能证明它导致了失败。

**当前数据不能称为全固定底盘。** 这里采用严格的命令判定：整条演示的 base 四维均为零，且 mode 始终为 -1。它不是官方任务类别的重新命名，也不等同于对机器人实际底盘位姿的测量。

| 任务 | 训练演示 | 符合严格固定底盘命令条件 |
| --- | ---: | ---: |
| DeliverStraw | 454 | 0 |
| GetToastedBread | 455 | 0 |
| KettleBoiling | 451 | 428 |
| LoadDishwasher | 451 | 184 |
| PackIdenticalLunches | 451 | 0 |
| PreSoakPan | 451 | 391 |
| PrepareCoffee | 463 | 463 |
| RinseSinkBasin | 458 | 456 |
| ScrubCuttingBoard | 454 | 419 |
| SearingMeat | 451 | 0 |
| SetUpCuttingStation | 454 | 27 |
| StackBowlsCabinet | 463 | 438 |
| SteamInMicrowave | 460 | 0 |
| StirVegetables | 451 | 104 |
| StoreLeftoversInBowl | 453 | 0 |
| WashLettuce | 451 | 433 |
| 合计 | 7,271 | 3,343 |

验证演示中 359/806 条符合相同条件。接口一直保留完整 12D action，包括 base/mode；这项发现不表示发生了删维错误，也不允许在评估时擅自把底盘动作清零。GetToastedBread 全局含底盘模式，不意味着它在起始阶段就需要底盘命令。

当前 120 条演示的命令拟合对照将分别比较机械臂模式与底盘模式下的连续动作误差，以及 mode/gripper 的正负类召回、错误率、切换窗口和零动作基线。等待模型预测完成后，才决定是否开展改变采样的训练对照。

证据：[全量统计](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_command_census_20260928/report.json)、[独立原始数据复算](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_command_census_20260928/independent_review.json)、[运行中的训练采样源码](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite16_atomicinit_4gpu_20260927/source/scripts/robocasa365/multitask_data.py:20)。本统计没有修改训练、进行优化器更新或新增策略试验。
