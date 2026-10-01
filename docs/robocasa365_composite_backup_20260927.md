# Composite16 自动备份

2026-09-27 11:06 UTC，CPU进程 PID 3508205 已从独立三文件快照启动，最长240小时，每30秒检查10k、25k、50k不可变checkpoint。已通过ps核对存活。当前Composite尚未训练，三个权重备份都处于等待状态。

目标EFS目录：

`/file_system/efs/checkpoint/intern/haozhe.jia/projects/DiT4DiT/artifacts/checkpoint_backup/robocasa365_composite16_atomicinit_4gpu_20260927`

恢复配置、数据清单、归一化和149个冻结源码文件，共154文件、3,187,891字节，已经复制并逐文件SHA256回读一致。源和目标挂载设备不同。完整GR1已在现有Atomic备份中，本次重新计算其21GB文件SHA256并保存明确引用；没有再创建一份重复基座。

后续每个checkpoint须等trainer完成命名快照并发布latest记录后才读取。复制期间检查源身份、源/目标SHA256；目标不同则拒绝覆盖。副本实际CPU加载后核对247个动作张量、全部Adam矩、更新步数、scheduler、四rank随机状态、数据划分和归一化。Composite还需严格匹配已批准的Atomic50k初始化及优化器重置来源。完成后才写入checkpoint备份收据；检查失败会保留证据并退出，不能据一个权重文件存在就声称备份完成。

10项合成对抗测试通过，覆盖并发修改、错误哈希、目标冲突、未发布快照、NaN及错误初始化。实际用已备份Atomic50k完成新审计器回归检查，247张量及优化器状态均有效，进程退出0。合成checkpoint仅用于验证程序，不是Composite训练。

CPU副本读取不能代替GPU恢复测试。原始视频/parquet与安装环境未复制；checkpoint保留原绝对路径，恢复时须保持布局或验证迁移。此进程不提交或停止GPU任务，不改变训练源码。

证据：`runs/robocasa365_composite_backup_20260927/acceptance.json`、`atomic50k_real_readback.json`、`process.json`、`live/status.json`。
