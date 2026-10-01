# Composite16 启动权重开发基线

2026-09-27 13:14 UTC 更新：四个评估进程已真实执行闭环策略，完成 6/320 次。先完成的 DeliverStraw、GetToastedBread 各一条、共 5,550 步已经通过独立保存动作审计，完整报告为 `early_trial_audit/review_20260927T131129Z.json`。两条均到达官方 horizon 且未成功；它们不是完整基线结果。主训练同时新增至 1,022 步，八卡并行继续。

2026-09-27 13:05 UTC 更新：Atomic 已成功结束且完整 1,800 次最终评估通过独立审计，等待进程于 13:01 UTC 自动提交一次基线任务 `t-20260927210100-n76gv`，之后退出。已独立查询该任务为 Running、四张 A800，入口已通过 CUDA 候选检查并进入 `development_start12_320_trials`。Composite 四卡主训练同时继续运行，共占八卡。13:05 时尚未发布完成试验，不能报告基线成功率。证据为 `handoff/started_20260927.json` 和 `submission.json`；不要重启提交等待进程或重复提交。

为判断 Composite 再训练带来的变化，增加一份使用本轮第 12 步固定权重的开发评估。该权重已经继承 Atomic50k，并执行过 12 次 Composite 优化器更新；报告将明确标记为第 12 步，不称为零步模型或最终结果。

基线覆盖全部 16 个 Composite seen 任务，每任务使用开发种子 100–119，共 320 次。采用官方完整 horizon、fresh target reset、动作执行时域 8 和 `stable_counter_v1` 场景协议。与后续 10k/25k 使用同一份任务清单和开发种子；正式比较时还需检查逐场景 XML 和初始状态的一致性。最终种子 1000–1099 不参与基线，也不会将这些开发试验并入 50k 的 1,600 次最终评估。

已完成第 12 步 checkpoint 的独立 CPU 全量检查：247 个动作张量、全部 Adam 矩均为有限值，优化器和调度器步数均为 12，四个采样器状态不同，数据划分和归一化一致。checkpoint SHA256 为 `4d2e3808a2d34a5315cfe5b8b3f9bfdbf180d9b7ed94ba46b9c186a77d085600`。这与之前四卡策略接口短测使用的权重完全相同。

12:09 UTC：权重及恢复资料已额外备份到独立 EFS，副本哈希及全部参数、Adam 矩的 CPU 加载检查通过；候选评估的 11 个配置与验收文件也已复制并校验。备份路径为 `/file_system/efs/checkpoint/intern/haozhe.jia/projects/DiT4DiT/artifacts/checkpoint_backup/robocasa365_composite16_start12_dev_20260927`，回执见候选目录下 `startup_backup/`。原里程碑备份进程继续运行。

评估直接使用当前 Composite 冻结快照的 149 个源文件，独立输出到 `runs/robocasa365_composite16_start12_dev_20260927/evaluation`。启动配置不调用 trainer，不改变训练进程或权重。入口和 CLI 导入检查通过，8 项提交保护测试覆盖运行中依赖、失败依赖、缺失审计、重复任务、额外资源占用及提交结果不明确等情况。

提交条件为 Atomic 云任务 `t-20260924183604-77jsj` 成功结束、全部最终评估通过独立审计，并重新检查资源。随后使用释放的四卡，继续与 Composite 长训四卡并行，项目最多八卡。提交前核对 `submission_status.json`、`submission_intent.json` 和 `submission.json`；存在意图或回执时不得再次提交。

12:34 UTC 已启动 CPU 等待进程，PID 3694914；12:35 已核对进程命令和源码哈希，确认正在观察同一个 Atomic 云任务。它每 30 秒调用已验收的 `submit_when_released.py --submit`，最多等待 6 小时。只在 Atomic 成功结束、完整审计通过且资源满足时提交一次；任何提交结果不明确、辅助进程超时或异常输出都会停止自动重试。

12:34 UTC 启动等待时基线任务尚未提交。运行记录在 `handoff/process.json`、`handoff/status.json` 和 `handoff/watcher.log`；最终提交状态见本文开头。等待进程不能取消已有 GPU 任务。后续监控已有基线任务；基线完成后会执行独立动作审计。

证据目录：`runs/robocasa365_composite16_start12_dev_20260927/`，包含 `plan.json`、`prequeue_acceptance.json`、`candidate_verification.json`、`checkpoint_cpu_audit.json`、`submission_tests.log` 和 `submission_status.json`。

这项基线不替代 Video DiT 冻结/解冻受控实验，也不补足历史 step2000 对照。Composite 50k 主训练、里程碑备份与最终评估计划保持不变。
