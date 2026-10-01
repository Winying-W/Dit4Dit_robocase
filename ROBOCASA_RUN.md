# DiT4DiT RoboCasa-GR1 运行记录

2026-09-20：已使用作者公开 GR1 权重跑通本机闭环，完成提交前检查，并提交正式云端评测。

## 当前实测结果

本机流程验证（不是论文规模评测）：**1/2 成功，50%**。

| 任务 | 成功/轮数 | 环境步数 |
|---|---:|---:|
| FromCuttingboardToPan | 1/1 | 720 |
| CanToDrawerClose | 0/1 | 720 |

策略在 NVIDIA L20 上使用 bf16，动作块为 12。两轮均完整运行，动作输出检查通过；按原作者评测代码计算也是 1/2。

- [小测汇总](results/gr1_smoke_bf16/summary.json)
- [成功任务视频](results/gr1_smoke_bf16/02/episode_000.mp4)
- [失败任务视频](results/gr1_smoke_bf16/23/episode_000.mp4)
- [24/24 场景预检](results/gr1_preflight_24/preflight_passed.json)
- [权重校验、源码版本和运行清单](results/gr1_setup_2gpu/manifest.json)

## 正式云端评测

- 任务 ID：`t-20260921103813-66sng`
- 当前提交状态：Running，2026-09-21 02:38 UTC。
- 资源：单机 2×A800-80GB，16 CPU，128 GiB 内存，最长 24 小时。
- 规模：24 个任务 × 50 轮 = 1,200 轮，每轮最多 720 环境步，动作块 12。
- [提交 YAML](scripts/volc/robocasa_gr1_eval_2gpu.yaml)
- [入口脚本](scripts/volc/entrypoint_gr1_2gpu.sh)
- 输出目录：`results/gr1_volc_20260921_2gpu/`
- 最终结果：`summary.json`、`summary.md`，各任务子目录中的 `episodes.jsonl` 和首轮视频。

```bash
volc ml_task get --id t-20260921103813-66sng --output json
volc ml_task logs --task t-20260921103813-66sng --instance worker_0 -f
```

## 与作者设置的对应关系

[论文 v1](https://arxiv.org/html/2603.10448v1) 第 5.1 节规定 24 任务、每任务 50 轮、720 步，论文报告平均成功率 **50.8%**。当前仓库 Model Zoo 为公开权重报告 **56.7%**。本次正式任务使用公开权重和作者批量脚本的评测规模；本机两轮结果不能替代这两个参考指标。

策略仓库：`mondo-robotics/dit4dit-model`，revision `46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44`。21,283,661,981 字节权重的 SHA256 已验证为 `fc65e78ab7c3de040d2df9f41416640e49befec58c784f550ecdaf41ee167098`。

独立 `.sim-gr1` 仿真环境使用 robosuite 1.5.1、MuJoCo 3.2.6、Gymnasium 0.29.1、NumPy 1.26.4；策略复用已有 `.conda` 环境。两者实际存储在项目已有 NAS 目录，通过 YAML 挂载使用，不在排队任务中临时安装或下载。

评测沿用官方图像处理、状态编码和动作反归一化。新增逐轮 JSON 与可校验的汇总，避免向量环境自动重置丢失最终成功标志；同时保留原作者统计口径用于对照。官方固定版本的 Sketchfab 资源不含可选 book 干扰物，仿真器会警告并跳过；未替换任何场景资产。详细设置见 [评测协议](scripts/robocasa/PROTOCOL.md)。

后台监控已启动：每 60 秒更新 [云端状态](results/gr1_setup_2gpu/cloud_status.json)，任务结束后自动在本文件追加正式成功率或失败状态。监控脚本：`scripts/volc/watch_gr1.py`；不会自动重提任务。

## 2026-09-21 减少用卡

按用户要求，已取消尚未启动的旧 8 卡任务 `t-20260920234454-5xcv8`（已确认 Killed），改为 2 卡任务 `t-20260921103813-66sng`。24 个任务按 GPU 交替分配，每卡 12 个任务；总评测规模仍为 1,200 轮。沿用通过检查的策略与仿真环境，重新检查 YAML、入口 GPU 编号、输出隔离和原有本机预检记录。

新任务已进入 **Running**；云端场景自检通过，GPU 0、1 的策略服务均已成功加载并开始接受闭环推理请求。[云端启动验证](results/gr1_setup_2gpu/cloud_startup_verified.json)。

## 云端最终状态（监控自动记录）

任务 `t-20260921103813-66sng` 状态：**Success**。

正式评测 **681/1200，成功率 56.75%**；原作者统计口径 56.25%。

[完整汇总](results/gr1_volc_20260921_2gpu/summary.json) · [逐任务表格](results/gr1_volc_20260921_2gpu/summary.md)
