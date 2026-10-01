# Composite16 独立结果审计

2026-09-27 10:54 UTC，新 CPU observer PID 3491393 已启动并通过 ps 核对存活。进程每30秒读取 Composite 的训练记录、10k/25k 开发评估和50k最终1,600次评估；最长运行240小时。它不能提交、停止或重启 GPU 作业。

独立六文件源码快照位于 `runs/robocasa365_composite_observer_20260927/source/`，对应 `source_hashes.json`。当前 Atomic131文件、Composite149文件训练快照均逐文件验证未改；原 Atomic 审计和自动接续进程仍在运行。

新版本从实际训练 `split.json` 读取离线验证窗口数，核对清单哈希、任务顺序、完整演示划分和窗口列表。训练尚未开始时报告 null，避免把 CPU 准备记录当成正式训练。当前 Atomic 实际训练为144窗口；当前 Composite 完整模型 CPU 检查为128窗口。这些都是离线loss采样窗口，与闭环试验次数和全部验证演示数量不同。

17项测试通过，包括缺失训练数据、重复窗口、篡改划分、错误清单、场景协议混用、假分母和不完整结果。真实检查读取了上述两份划分，以及一条已有Atomic开发轨迹的900步保存动作。首次误用了早期复用adapter的清单，严格哈希检查拒绝；随后使用当前manifest的完整模型CPU检查，未放宽任何校验。具体记录保存在 `real_checks/first_attempt_rejected.json`。

运行计划要求 stable_counter_v1，审计对不符的协议报错。不完整的16任务结果不发布总体成功率；动作审计通过也不能证明全部视觉接口或GT接触漂移正确。

验收：`runs/robocasa365_composite_observer_20260927/acceptance.json`。
进程记录：`runs/robocasa365_composite_observer_20260927/process.json`。
活动状态：`runs/robocasa365_composite_observer_20260927/live/status.json`。
实际存活核查：`runs/robocasa365_composite_observer_20260927/live_verified.json`。

本轮没有新增GPU作业、训练更新或策略试验。Composite当前仍等待Atomic完整结果复核和四卡释放；自动接续进程独立负责提交，不能据此手工重复排队。
