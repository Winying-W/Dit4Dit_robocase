# Atomic50k 新视频解冻对照：CPU 控制验收

2026-09-27：现有 Atomic18 step050000 已作为共同动作初始化，完成真实 CPU 模型前向、反向和原训练数据合同检查。这是新对照的准备工作，尚未训练、提交 GPU 作业或产生策略成功率。

当前执行顺序为：完成 Atomic18 的 1,800 次最终评估及独立审计，待四卡释放后推进已验收的 Composite16 训练。视频解冻对照单独准备，不延误 Composite。早期 StirVegetables 16-demo step2000 的恢复已停止，不再是训练或评估的前置条件；原历史对照保留为未执行记录。

## 实际完成的检查

- 正式单卡 trainer 支持显式 `--manifest --warm-start --warm-start-run --paired-ab`。加载 Atomic50k 的 247 个动作参数张量；Video DiT、VAE 和文本编码器从 GR1 基座加载，优化器、调度器、采样器及本轮步数重新建立。精确恢复使用 `--resume`，恢复初始化来源和随机数控制协议。
- 正式多任务数据准备重新读取完整 Atomic18 划分：8,216 条训练演示、910 条验证演示、2,007,192 个训练帧、144 个固定验证窗口。划分完全相同，归一化文件逐字节相同，输入配置兼容。
- 在一个真实 PickPlaceCounterToCabinet 训练观测上执行五次完整 CPU 前向。冻结与部分解冻分支的动作输入、噪声、时间采样及 247 个动作梯度张量完全一致；部分解冻的 video blocks 16/17 有 40 个有效梯度张量。改变未来视频标签只改变辅助视频监督。保留官方梯度分离。
- 六项初始化合同测试通过，覆盖错误起点、数据/归一化/源码不匹配、非有限或额外参数，以及动作权重加载的隔离性。冻结快照的正式单卡 CLI 导入成功，155 个源文件哈希一致。

以上没有执行 `optimizer.step()`，不证明 CUDA 显存、恢复、长期训练或闭环收益。新对照的 GPU 预算、入口及硬件检查仍待完成。

## 证据

- 汇总：`runs/robocasa365_atomic50k_ab_controls_20260927/cpu_acceptance_summary.json`
- 真实模型控制：`runs/robocasa365_atomic50k_ab_controls_20260927/cpu/acceptance.json`
- 原数据合同：`runs/robocasa365_atomic50k_ab_controls_20260927/same_data_contract.json`
- 测试及 CLI：同目录下 `unit_tests.log`、`trainer_help.log`
- Atomic50k SHA256：`2cc6fa048869f55d4a5f2e1f7df4419d090dbde2c7ba244e5b0d0110996ae49c`

这份验收与原 step2000 历史对照、Composite149文件快照的提交前验收分别记录，不能相互替代。
