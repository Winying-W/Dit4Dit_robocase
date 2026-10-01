# Atomic18 当前权重的 EFS 备份

2026-09-27，当前训练的 10k、25k、50k 权重、发布版 GR1 基座、模型配置、数据划分、归一化及 131 个冻结源码文件已复制到 EFS。共 155 个文件、27,169,521,489 字节。源 vePFS 与目标 EFS 的挂载设备编号不同。

备份根目录：

`/file_system/efs/checkpoint/intern/haozhe.jia/projects/DiT4DiT/artifacts/checkpoint_backup/robocasa365_atomic18_20260924/20260927_s50k`

其下保留项目内原相对目录。完整文件清单和每个文件的 SHA256 位于 `BACKUP_MANIFEST.json`。复制时计算源数据摘要，写入后重新读取目标文件计算摘要；155 个文件全部一致。已知 Atomic50k 和 GR1 摘要也与之前的独立验收相符。

| 权重 | SHA256 |
| --- | --- |
| Atomic18 10k | `7fe2316795ec86b529b63ca48337238b6199f488c25c8cd42a5cfbf85f17af77` |
| Atomic18 25k | `00fc31054fa743bfc304ee853fd2ee7146b9f0a7679bf05a1e5cdfe14f185de5` |
| Atomic18 50k | `2cc6fa048869f55d4a5f2e1f7df4419d090dbde2c7ba244e5b0d0110996ae49c` |
| GR1 基座 | `fc65e78ab7c3de040d2df9f41416640e49befec58c784f550ecdaf41ee167098` |

实际从 EFS 副本执行了 CPU `torch.load`。三份训练权重的 global/phase step、247 个动作张量、163,276,320 个动作参数、247 组优化器状态、调度器步数和四 rank RNG 状态结构均符合记录。没有执行 GPU 恢复或新增优化器更新。

恢复时需核对清单及配置。checkpoint 内保留原绝对路径，应恢复原目录布局，或显式验证搬迁后的路径映射。原始数据视频、parquet 和安装环境未纳入此副本；数据清单和训练划分已保存。当前训练和评估仍使用 vePFS 原文件。

本地执行证据：`runs/robocasa365_checkpoint_backup_20260927/acceptance.json`、`checkpoint_readback.json`、`backup.log`。备份进程与读取进程均退出 0。
