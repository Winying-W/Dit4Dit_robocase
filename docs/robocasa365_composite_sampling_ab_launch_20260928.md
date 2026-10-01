**关键阶段采样对照已提交，两路单卡，原四卡主线继续。**

截至 2026-09-28 05:44 UTC，原采样任务 `t-20260928132617-xhvm4`、关键阶段采样任务 `t-20260928132748-gjdmd` 均由云平台确认 Running，两路已经通过 CUDA 更新、独立进程恢复和模型接入仿真的短检查，分别新增训练至 24、19 步，正在继续各自的 2,000 步预算。尚未完成新一轮开发评估，没有新增对照成功率。

两路均继承同一份 Composite 新增 10k 动作权重，SHA256 为 `32248f3b89e29a805c141bfa618bf54a5b4441bd903cf8f109f2d8877461fe4c`。每路重新初始化优化器和学习率计划，新增训练 2,000 步。该实验与早期原 16 演示 StirVegetables step2000 的指定对照不同，不替代它。

| 项目 | 两路共同设置 |
| --- | --- |
| 数据 | 全部 16 个 Composite seen 任务，7,271 条训练演示，806 条验证演示隔离 |
| 数据量 | 5,403,990 个训练帧，保留完整 base/mode 和 12D action |
| 训练模块 | State Encoder、Action Encoder/Decoder、Action DiT，共 247 个参数张量、163,276,320 个参数 |
| 冻结模块 | Video DiT、VAE、Qwen |
| 资源 | 每路 1 张 A800；原四卡主线保持，共申请使用 6 张卡 |
| 有效 batch | microbatch 4 × 累积 16 = 64 |
| 学习率 | 峰值 1e-4，200 步 warmup，余弦下降至峰值的 10% |
| 随机控制 | 相同任务抽样序列；配对的骨干和动作噪声流；分别保存采样与模型 RNG |
| 开发评估 | 取面包、取吸管、搅拌，各 20 个相同开发场景，每路 60 次；完整官方 horizon |

唯一计划中的训练分布差异：对照路均匀抽训练帧；候选路按 50% 普通帧、25% 各演示前 16 帧、25% 有效前 8 步含 mode/gripper 切换的帧抽样。切换是录制命令标签，不能当成抓取或接触成功标签。

**排队前已验证真实训练更新与断点恢复。** 实际生产训练入口在 CPU 上加载完整 GR1 视觉/语言基座和 Composite10k 动作权重，用真实数据分别运行两路各两步；再从候选路第一步保存点在独立进程重跑第二步。共执行五次 CPU 检查更新，整份 checkpoint 的权重、Adam 状态、scheduler、采样 RNG 和 Torch RNG 逐值一致，恢复后的 loss、梯度及抽样也一致。State Encoder、Action Encoder/Decoder、Action DiT 都有非零更新，全部 247 个动作参数张量的梯度有限。CPU 检查权重通过保存协议与 GPU 训练隔离，不用于新云任务的初始化。

51 项相关检查通过。新增开发任务筛选保留原完整训练 manifest，显式标注三任务子集，禁止把这种筛选用于正式最终评估。正式 16×100 新场景测试仍属于原四卡主线的后续计划。

**GPU 启动检查已通过。** 每路先执行 10 次真实 CUDA 更新，再在独立进程恢复到第 12 步，权重/优化器/采样状态均通过独立核验，State Encoder、Action Encoder/Decoder、Action DiT 在恢复后都继续更新，峰值显存约 23.83 GiB。两路当前模型均在 GetToastedBread 新场景完成 8 控制步，保存动作审计通过，已自动继续新增 2,000 步，随后各评估 60 次。这段 8 步检查不计入成功率。进一步复算前 12 步的 768 次 GPU 实际抽样，任务序列与学习率一致、输入帧序列不同，checkpoint 中的采样/CPU/CUDA RNG 终点相同。

恢复配置、完整训练/评估源码、数据划分及采样池已独立备份至 EFS；两路第 12 步权重已复制、核验 SHA256，并读回全部动作参数和 Adam 状态。备份过程发现云端复制采样池时保留了 `0600` 权限，使本地进程无法读取两份 root 所有的元数据副本；已改用 checkpoint 合同锁定的原始采样池字节完成备份。权重本身一直可读，训练任务和冻结源码未被重启或修改。修复后的 CPU 观察器继续备份第 1,000、2,000 步。原 Composite10k 权重已有单独 EFS 备份。备份不是 CUDA 恢复通过的替代证据。

提交前两次检查均确认队列有空闲资源；第二路提交时已计入第一路的 Waiting/Queue 状态。没有停止或重启原四卡任务。截至 05:46 UTC，原 Composite10k 开发评估为 245/320 次、成功 0 次，仍是局部结果。

专家 GT 接触漂移仍未彻底解决。本轮没有声称它已修复，也没有用 CPU loss、命令预测或保存动作审计替代闭环成功率。

证据：[CPU 真实更新与精确恢复](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_sampling_ab_20260928/cpu_preflight/acceptance.json)、[排队前检查](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_sampling_ab_20260928/prequeue_acceptance.json)、[CUDA 验收与抽样配对复核](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_sampling_ab_20260928/cuda_gate_evidence_20260928.json)、[两路云状态](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_sampling_ab_20260928/latest_observation.json)、[独立恢复配置备份](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_sampling_ab_20260928/backups/recovery_metadata.json)。
