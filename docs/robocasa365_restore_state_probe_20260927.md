# GT replay：未记录的求解器状态对照

2026-09-27，在 StirVegetables episode0、3、11 的场景中，分别取启动完成后和执行 32 个 GT 命令后生成的本地快照。每个快照执行五种恢复方式、各 250 个物理 tick，共六个快照、30 条物理轨迹、7,500 个 tick。所有分支共享同一个编译后的 MuJoCo 模型和相同固定电机命令，没有 XML 再导出、渲染或策略推理。

| 恢复方式 | 六个快照的结果 |
| --- | --- |
| 完整 `mjSTATE_INTEGRATION`，重复两次 | 逐值完全相同 |
| 完整状态，仅将 `qacc_warmstart` 清零 | 从第一个物理 tick 开始不同 |
| 只恢复录制状态格式中的 time/qpos/qvel/act，另给定相同 motor ctrl | 与仅清零 warmstart 的分支完全相同 |
| 录制状态格式，再补回本地快照的 warmstart | 与完整状态分支逐值完全相同 |

这证明在本次固定电机命令的对照中，省略 warmstart 足以改变轨迹。0.5 秒内自由关节物体的位置差异最大约 0.367 mm；它不是整段 GT replay 的最终误差。保存的全部轨迹已独立重新比较，正、负对照均成立。

官方 `get_episode_states()` 读取 `extras/episode_*/states.npz` 的 `states` 数组。本实验验证该录制状态格式对应 time/qpos/qvel/act，未包含完整积分状态中的 warmstart。这里使用的是当前环境新生成快照的 warmstart，没有恢复原始演示录制时遗漏的值。因此不能把这个结果当作旧 episode0/3 漂移根因已确定、GT replay 已修复，或策略低成功率已得到解释。

后续需要把完整积分状态作为新录制数据的回放信息，并验证实际控制器动作回放；现有官方演示的严格重放仍需进一步定位原始状态恢复和采集流程。当前官方仿真参数、正在运行的 Atomic 评估和已验收的 Composite 快照保持原版本。

证据：

- `runs/robocasa365_restore_state_probe_20260927/audit/report.json`
- `runs/robocasa365_restore_state_probe_20260927/independent_review.json`
- 同目录 `source/` 的两文件冻结快照及 `source_hashes.json`
- 各案例的 `traces.npz` 保存五个分支的完整位置/速度轨迹和输入状态

实验进程退出 0。没有新增模型训练或策略成功率试验。
