# 原 step2000 冻结 / 部分解冻对照：控制变量验收

当前处置：本文件保留早期方案和验收记录。旧 step2000 恢复已停止，不再作为当前训练或评估的依赖；下文的历史 GPU 对照未执行。现有 Atomic50k 的新对照控制见 [Atomic50k CPU 验收](robocasa365_atomic50k_ab_controls_20260927.md)。当前优先完成 Atomic18 评估并推进 Composite16。

2026-09-27：对照机制已接入单卡 trainer，并通过真实发布版 GR1 模型的 CPU 前向、反向验证。原 StirVegetables 16-demo step2000 仍不可访问，历史 A/B 尚未开始；本轮没有 optimizer 更新或新增策略评估。

## 为什么补充这些控制

在相同权重、观测和全局随机种子下，辅助 video FM 会先消耗 Torch 随机数，改变后续 Action DiT 的噪声和时间采样。实际检查中动作特征相同，但未控制的两分支 action loss 分别为 1.3828125 和 0.6171875。这不是模型学得更好的证据。

原 trainer 对所有可训练参数统一裁剪梯度；加入 Video DiT 参数后，video 梯度可能改变 action 梯度的缩放。两个分支因此需要独立的动作随机数流及相同的动作裁剪规则。

## 新协议及验收

`train_single_gpu.py --paired-ab` 显式开启 `paired_video_action_rng_v1`：按 seed、实际本地更新编号和 microbatch 编号派生 backbone/action 随机数流；退出上下文时恢复 RNG 及 action forward，异常路径也恢复。各 optimizer group 独立裁剪。协议、seed 和裁剪规则写入 checkpoint 的 training_schedule；精确恢复和 LR 延长都拒绝更换这些设置。未开启参数时保留原训练随机数及统一裁剪规则。

新冻结目录：`runs/robocasa365_legacy_ab_20260927/source/`，151 个文件。真实模型检查经新 trainer 的实际 forward helper 执行，使用相同训练观测，CPU bfloat16；没有 optimizer.step。

| 验证 | 结果 |
| --- | --- |
| 冻结 / 部分解冻的 action features、target、mask、state | 逐项相同，最大误差 0 |
| ActionEncoder 实收 noisy actions、timesteps | 逐项相同，最大误差 0 |
| 247 个 action 参数的原始 / 分组裁剪后梯度 | 两组哈希完全一致 |
| 部分解冻 video blocks 16/17 | 40 个梯度张量有效 |
| action loss 到 video 的梯度 | 维持官方 stop-gradient |
| 只将未来标签视频置零 | video loss 改变，动作输入、噪声及 action loss 不变 |
| 接入 trainer 前后的相同检查 | 所有捕获张量与梯度哈希一致 |
| 随机数、异常恢复、resume 与结果选择测试 | 10 项通过 |

CPU 检查使用发布版 GR1，只验证实验机制。没有证明原 step2000 历史 A/B 或 CUDA 训练通过，也不能据此声称部分解冻提高成功率。

## 实际运行仍需完成

两分支将从同一个原 step2000 warm-start，各追加 1,000 步，动作 LR 1e-4；部分解冻分支另外训练 video blocks 16/17，LR 1e-5，保留官方梯度分离。单卡依次执行，先完成 10 步、独立恢复到 12 步及短闭环硬件检查，再长训。开发评估各 10 个 seed100–109。

新入口显式采用 stable_counter_v1。选择脚本要求训练控制一致、仿真实际协议 receipt 合法、每对场景完整初始物理状态和 XML 相同；相同 seed 本身不足以视为配对。10 次开发试验用于候选方案选择，不作为统计优越性结论。

缺少原权重时入口已实际拒绝启动。完整 prequeue 尚未完成，任务未提交。当前 Atomic18 四卡作业和已验收 Composite16 快照未修改；本轮复核两者冻结源码哈希均一致。仍遵守最多同时四卡。

证据：

- `runs/robocasa365_legacy_ab_20260927/controls_cpu/acceptance.json`
- `runs/robocasa365_legacy_ab_20260927/draft_validation.json`
- `runs/robocasa365_legacy_ab_20260927/tests.log`
- `runs/robocasa365_legacy_ab_20260927/plan.json`

旧存储调查：08:06 UTC 三个已知原权重路径仍为 FileNotFound；当前进程挂载列表中没有 `/file_system/nas` 挂载。旧作业配置明确使用 NAS，但普通 volc get 的格式字段不提供 Storages，本次查询因此失败并留档。这不能证明 NAS 内容已删除，也未通过远程挂载读到原文件；未提交恢复作业。记录位于 `runs/robocasa365_legacy_step2000_20260924/path_checks/storage_task_20260927T080617Z.json`。
