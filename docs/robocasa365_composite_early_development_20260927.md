# Composite 第 2,000 步开发评估

2026-09-27 14:44 UTC：新增四卡任务 `t-20260927223814-srfgj` 为 Running。第 2,000 步权重在 StirVegetables、PrepareCoffee、DeliverStraw、LoadDishwasher 上的四卡 CUDA 策略/仿真短测全部通过，已进入 320 次完整开发评估；尚未发布完整试验结果。主训练核查到新增 2,111 步，起点评估完成 73/320 次。

用户允许空闲时增加卡。提交前官方资源接口确认还有 14 张 A800、220 CPU、4,024 GiB 内存未分配。本次增加四卡，项目共用 12 卡：四卡 Composite 主训练、四卡第 12 步起点评估、四卡第 2,000 步开发评估。

这份权重来自 Atomic18 50k 初始化后的 **Composite 新增 2,000 步**，与早期 StirVegetables 的历史 step2000 无关。主训练仍按原计划新增 50k，10k/25k 开发评估及 50k 的 1,600 次最终评估保持原方案。

本次评估完整 16 个 Composite seen 任务，每任务开发 seeds 100–119，共 320 次，使用官方完整 horizon、fresh target reset、stable_counter_v1、执行窗口 8。最终 seeds 1000–1099 不参与。CUDA 短测的四条 8 步轨迹单独保存，不计入成功率。

第 2,000 步由主训练发布的 `action_latest.pt` 固定复制，247 个训练张量、全部 Adam 矩、优化器/调度器步数、四 rank RNG、数据划分与归一化已通过 CPU 检查。另已复制到独立 EFS 并重新审计，SHA256 为 `a61d562fc5492ec749e904bec81f0b546825ec37c26d8037a86c913aa533cdb6`。完整性检查为 CPU 证据；另已实际执行并独立核查四卡 GPU 短测，见 `gpu_startup_acceptance.json`。

起点评估和本次评估全部完成后，逐项核对 320 对场景的初始 XML 和完整物理状态。只有全部匹配，才报告配对成功率变化；不因零成功或某个离线 loss 单独判断是否应该解冻 Video DiT。

证据：

- 候选、唯一提交回执及进度：`runs/robocasa365_composite16_step002000_dev_20260927/`
- `prequeue_acceptance.json`：CPU 检查、既有第 12 步架构接口证据、11 项提交测试及必须先通过的新 CUDA 短测。
- `checkpoint_backup_receipt.json`：固定副本及独立 EFS 备份回读。
- `evaluation_smoke/acceptance.json`：仅在四个新 GPU 短测全部通过后生成。
- `evaluation/report.json`、`independent_review.json`：全量开发评估及独立动作审计。
- `comparison_plan.json`：与第 12 步基线比较的明确命令。

本项尚不能给出第 2,000 步的完整成功率，也不替代最终 1,600 次评估。

2026-09-27 14:56 UTC：第 2,000 步已产生完整开发试验。首两条 DeliverStraw/GetToastedBread（seed100）共 5,550 个控制步的独立动作审计通过，且两条初始 XML/完整物理状态与第 12 步基线精确匹配（状态最大误差 0）；两条均未成功。仅据此不能给出全 320 次的成功率或训练效果结论。

自动开发对比进程已启动，PID 和实际命令行、心跳已核验。9 项边界测试通过，真实未完成评估也正确保持等待。它负责 12→2,000、12→10,000、12→25,000、10,000→25,000 四组完整 320 对场景的对比；结果含场景不匹配时保留全部失败证据并抑制整体配对提升估计。进程记录、状态及验收位于 `runs/robocasa365_composite_comparison_observer_20260927/`。此 CPU 进程不提交或修改云任务。
