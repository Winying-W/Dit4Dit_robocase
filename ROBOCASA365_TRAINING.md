# RoboCasa365 → DiT4DiT 单卡训练结果

最新进度（2026-09-23）：冻结骨干的Action DiT已从此前step300续训到**step2000**，训练任务`t-20260923152847-l2xf9`为Success。独立进程恢复后复现同一验证loss **0.1621078234**；数据划分与训练归一化不变，峰值显存22.75GiB。该分支没有接续首轮部分解冻的step400。

真实checkpoint已通过完整15,980帧动作往返检查和32步闭环接口验收，最大动作还原误差1.02e-7。完整任务评估`t-20260923163223-f4hcf`已Success：step2000新场景0/10、留出非移动初始场景0/4，step300配对对照0/3。17条均跑满2400步；40,800个实际动作的完整链路检查通过。接口与训练已跑通，但本轮未展示任务成功能力。详细状态与协议见[双线运行记录](docs/robocasa365_dual_run_20260923.md)。下文保留首轮400步实验记录。

训练的是DiT4DiT的Action DiT（含状态编码器）；Cosmos内部的Qwen2.5-VL文本编码器和VAE均冻结。不能把“没有训练Qwen”表述成“模型中没有Qwen组件”。

## 数据和接口

- 官方 StirVegetables / Composite seen / target human 数据已在 NAS 可用，501 episodes、409202 frames、20 FPS；云端逐条核验 parquet、三路视频及 replay states。
- 原始三路 RGB 各256×256；训练缩放各128×128，同一时刻拼成128×384。state16维补齐64，action12维补齐32，动作 chunk=16，保留完整 action schema 和 mask。
- 完整数据含移动底盘演示；按真实命令筛出113条固定底盘演示，本轮仅选16条训练、4条验证。归一化仅用训练轨迹重算。
- 真实模型 forward / 双 loss / 动作反归一化通过；预测形状[1,16,32]，解码动作[1,16,12]。

## 首轮400步训练结果

按用户提供的火山云 CLI 指南排队，资源为1× A800-SXM4-80GB、8 CPU、128 GiB主机内存；队列 `q-20260629215336-k8k4r`。

| 阶段 | 更新次数 | 可训练模块 | 固定验证 action FM loss |
| --- | ---: | --- | ---: |
| GR1权重初始化 | 0 | — | 2.065245 |
| 冻结Cosmos | 300 | Action DiT（含state encoder） | 0.181549 |
| 部分解冻 | 100 | Action DiT + 视频blocks16/17（零起始） | 0.176006 |

microbatch=1，梯度累积4；动作LR=1e-4，视频LR=1e-5。text encoder和VAE始终冻结，保留原版stop-gradient：动作loss更新动作模型，视频loss更新解冻的视频层。冻结阶段峰值显存22.75 GiB，部分解冻阶段24.09 GiB；已检查有限非零梯度和实际权重变化。

验证为4条未参与训练的轨迹、每条2个固定窗口，输入当前图像并固定噪声。它是8个窗口上的离线指标；部分解冻的小幅改善不能证明统计显著收益或闭环能力。GT replay剩余物理漂移的失效环节已定位，但尚未消除；本次闭环结果与GT诊断分开报告。

## Checkpoint和恢复

NAS输出目录：`results/robocasa365_train_20260922/train/`。

- 冻结阶段：`action_latest.pt`，global_step=300。
- 部分解冻阶段：`partial_joint_latest.pt`，global_step=400。
- 第10步实际退出进程，再恢复optimizer、scheduler和RNG继续训练。
- 最终checkpoint保存并读回通过；另起云任务加载第400步checkpoint，零次额外更新，重新验证loss=0.1760062855，与训练结束一致。

这些是依赖原始GR1权重的增量训练checkpoint，需结合该目录的归一化和配置文件使用，不能直接传给官方 `from_pretrained`。当前开发机未挂载NAS，权重应在挂载NAS的云任务中读取。恢复命令见[详细方案](docs/robocasa365_training_plan.md)。

主训练任务 `t-20260923001913-njx5v` 在完成400步并保存checkpoint后，写汇总时触发 `NameError: start`，因此云状态为Failed。已改为 `start_time`；恢复任务 `t-20260923004328-96k7q` 成功收尾，没有重复训练。

## 可复查记录

- [汇总报告](runs/robocasa365_train_setup/training_report.json)和[日志指标CSV](runs/robocasa365_train_setup/metrics_logged.csv)：日志按间隔输出，CSV不是全部400步；完整逐步指标在NAS输出目录的metrics.jsonl。
- [最终恢复任务状态](runs/robocasa365_train_setup/recovery_status.json)和[恢复日志](runs/robocasa365_train_setup/recovery_final.log)。
- [最新checkpoint记录](runs/robocasa365_train_setup/live/latest.json)、[训练配置](runs/robocasa365_train_setup/live/run_config.json)、[轨迹划分](runs/robocasa365_train_setup/live/split.json)。
- [文献依据和解冻方案](docs/robocasa365_training_plan.md)：部分解冻是工程实验；DiT4DiT论文的正式配方更新视频和动作两个DiT，不能等同本轮设置。

最新闭环对比为冻结骨干的step300与step2000，历史部分解冻step400未纳入本轮。下一步优先单演示过拟合，检查手臂/夹爪推理误差及训练场景闭环，再扩大数据或解冻范围。视频解冻仍需独立对照并重新测量显存和成功率。

本地收尾检查：Python编译、入口shell语法、git diff空白检查及报告一致性通过。额外Ruff检查因工具下载未完成而中止，未将其记为通过。

2026-09-23 GT回放排查更新：已确认采集时漏记的零动作初始化导致0.05秒时间错位，回放入口已修复并改为每条新建环境；剩余开环偏离未全部消除。详情见[定位报告](ROBOCASA365_REPLAY_DIAGNOSIS.md)。
