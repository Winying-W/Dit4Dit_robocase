# RoboCasa365 训练与开发评估进度

2026-09-27 14:44 UTC 更新：利用空闲资源再增加四张 A800，当前三个任务共 12 卡：四卡 Composite 主训练、四卡第 12 步基线、四卡 Composite 第 2,000 步开发评估。主训练最新核查 2,111/50,000 步，2,000 步权重及独立 EFS 备份均通过 CPU 审计。新增任务 `t-20260927223814-srfgj` 已通过这份权重的四卡 CUDA 策略/仿真短测，进入 16×20 次开发评估。起点评估完成 73/320 次；已跑满的 DeliverStraw、GetToastedBread 分别 0/20，合计 111,000 控制步独立动作审计通过，不能据此报告全 16 任务或第 2,000 步的成功率。详情见 [早期开发评估](robocasa365_composite_early_development_20260927.md)。

以下为此前时间点的历史记录。

2026-09-27 13:14 UTC 更新：Composite 新增 1,022 步，第 1,000 步 checkpoint 已发布。另将这份权重和配套配置复制到独立 EFS，SHA256 回读及全部 247 个动作张量、Adam 矩、优化器/调度器步数、四 rank RNG 和归一化/划分检查通过；本项为 CPU 完整性检查，不是新的 CUDA 恢复试验。SHA256 为 `428793067427918d168bb4f1ff744cdc6a9ddabbc877a007296f418b2c70cee7`，回执为 `runs/robocasa365_composite16_atomicinit_4gpu_20260927/first_1000_checkpoint/backup_receipt.json`。10k/25k/50k 备份进程保持运行。

同时四卡基线已完成 6/320 次。先完成的 DeliverStraw、GetToastedBread 各一条完整试验共 5,550 控制步，已独立复算保存动作链并通过；两条均耗尽官方时域且未成功。这里只验证首批落盘试验的接口和记录，尚无完整 Composite 成功率。两个具体云任务均重新查询为 Running，当前证据见 `first_1000_checkpoint/progress.json`。

2026-09-27 13:05 UTC 更新：Atomic 四卡任务已成功结束，最终评估为 279/1,800（15.50%），完整结果见 [最终报告](robocasa365_atomic18_final_20260927.md)。Composite 主训练 `t-20260927193400-hbdcg` 继续运行，核查时新增 907 步。Atomic 释放的四卡已由唯一一次自动提交的基线任务 `t-20260927210100-n76gv` 接续使用；云端 Running，CUDA 启动检查通过。当前共八卡，用于 Composite 训练和开发基线评估。13:05 时基线尚未发布完成试验。

12:57 UTC 重新查询队列：A800 总配额 24、已分配 10。下面保留首次并行启动时的验收记录；其时间和步数为历史值。

2026-09-27 11:53 UTC：Composite 已通过全部 GPU 启动检查，进入 `training_to_10000_of_50000`。实际完成并保存本轮第 12 步，50k 长训和后续成功率评估仍在进行。两个云任务均为 Running，共使用八张不同的 A800。

- Atomic18 四卡任务：`t-20260924183604-77jsj`，继续完成 50k 权重的 1,800 次最终评估。
- Composite16 新增四卡任务：`t-20260927193400-hbdcg`。从 Atomic50k 动作权重初始化，优化器、调度器、采样器从本轮第 0 步开始，新增 50,000 步。
- 数据：7,271 条训练演示、806 条验证演示；约 540 万训练帧。
- 训练 State Encoder、Action Encoder/Decoder、Action DiT；视觉及语言基座维持已验收方案。
- 10k、25k 各评估 16×20 次；50k 评估 16×100 次。两个任务集分别报告成功率。

用户明确允许在有空闲卡时增加申请。提交前官方 `GetResourceQueue` 返回 A800 总量 24、已分配 6，CPU 和内存余量也满足新增四卡。提交后已分配增加到 10，其中当前两个 DiT4DiT 任务共 8 卡。容量响应属于队列配额证据，实际分配另由云任务状态及容器中四张 A800 的输出确认。

旧的 CPU 串行提交进程已核对身份后停止；Atomic GPU 任务、独立评估审计和 checkpoint 备份进程继续运行。仅提交了一次 Composite。提交意图和云回执都已持久化；后续不得重复提交同一候选。

提交器修复了 CLI 无匹配结果时 `没有匹配条件的任务` 提示行导致的 JSON 解析错误。15 项相关测试通过；容量不足、错误队列、停止队列和 API 错误均拒绝提交。修改没有进入已验收训练快照。

启动检查全部通过：16 任务三相机渲染、四卡 10 步训练、独立恢复至 12 步、四卡策略与仿真接口。四个策略检查任务为 StirVegetables、PrepareCoffee、DeliverStraw、LoadDishwasher，各执行 8 步并检查预测、反归一化和仿真指令一致性。短测 rollout 不计入成功率。

真实训练证据确认 Atomic50k 的 247 个动作张量完成初始化，约 1.63 亿参数参与训练；新优化器累计 12 步。第 1 步 State Encoder、Action Encoder/Decoder 梯度均非零，第 10 步四个 rank 的权重 SHA256 相同。训练峰值显存约每卡 24.44 GiB，已记录的 loss 与梯度均为有限值。全局 batch 为 64。

11:44 UTC 更新：云日志已逐一记录全部 16 任务通过，入口进入 `four_gpu_preflight_10_updates`。渲染报告的临时目录由云容器以 root 创建为 0700，本地用户目前不能直接读取该目录；调整该目录读取权限的远程命令因 IAM 权限不足未执行。此阶段依据容器内完整检查、成功退出标记和任务日志验收，未声称本地已独立读取详细渲染报告。训练输出和后续审计目录不依赖该临时目录。

10k、25k、50k 权重由独立 CPU 进程备份到 EFS，经 SHA256 和 CPU 加载检查后才登记完成。当前已有恢复资料备份；新权重须等实际训练产生，不能提前称为已备份。

证据入口：

- `runs/robocasa365_extra_capacity_20260927/parallel_authorization.json`
- `runs/robocasa365_extra_capacity_20260927/queue_at_submit.json`
- `runs/robocasa365_extra_capacity_20260927/queue_after_allocation.json`
- `runs/robocasa365_extra_capacity_20260927/submission_status.json`
- `runs/robocasa365_extra_capacity_20260927/gpu_startup_acceptance.json`
- `runs/robocasa365_extra_capacity_20260927/first_training_evidence.json`
- `runs/robocasa365_composite16_atomicinit_4gpu_20260927/submission.json`
- `runs/robocasa365_composite16_atomicinit_4gpu_20260927/stage.txt`
- `runs/robocasa365_composite_observer_20260927/live/status.json`
- `runs/robocasa365_composite_backup_20260927/live/status.json`
