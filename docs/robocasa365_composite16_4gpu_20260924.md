# 完整 Composite seen 四卡训练与评估准备

这条实验保留完整16个Composite seen任务的原目标，与目前运行的Atomic18独立记录。训练与仿真公共代码沿用已通过实际四A800更新、恢复及动作审计的实现；本任务仍必须取得自己的排队前验收和到卡短验收。当前没有提交Composite云任务，不同时申请额外四卡。

## 固定数据与初始化

使用全部官方target human数据：8,077条演示、6,002,265帧，7,271 train / 806 validation，对应5,403,990 / 598,275帧。episode划分、数据路径及共享训练归一化与原16任务完整审计逐项一致，扩大的是最终评估种子数量。原审计的全部parquet内容哈希重新核对，报告 `data_content_identity.json`；不会把新路径上的文件直接当成旧数据验收通过。

原始state16/action12完整保留，包括base、torso和mode。任务均匀采样后在训练帧内均匀采样；验证演示不参与训练及归一化。固定离线验证代理为每任务4条验证演示的8个窗口，共128个窗口。该loss不等同于全验证集或策略成功率。

从恢复的官方RoboCasa-GR1重新初始化；不从Atomic18模型或旧StirVegetables step2000热启动。训练机器人与动作仍是RoboCasa365 PandaOmron。全部State Encoder、Action Encoder/Decoder及Action DiT共247个张量参与优化，Cosmos Video DiT、VAE、Qwen文本编码器冻结。

## 训练和正式评估

- 4×A800，microbatch4×accumulation4×world4=global batch64，50,000 updates。
- warmup1000，AdamW峰值1e-4，cosine到1e-5；全局有效action元素加权梯度，与Atomic18相同。
- step10短训练、独立torchrun恢复至step12、四卡策略/仿真短检查；全部通过才放行长训练。
- step10k、25k各16×20=320次开发评估，seeds100–119。
- 固定step50k，完整官方horizon、fresh Gym target，seeds1000–1099，16×100=1,600次最终评估；四个GPU策略worker，每卡一个。
- 分别报告逐任务成功数/分母/Wilson95%区间、宏平均及合并成功率。基础设施失败不伪装成策略失败或从分母静默删除；未完成不输出完整benchmark分数。
- 云任务最长7天是超时上限，实际训练/评估结束立即退出。只能在Atomic训练与评估结果复核、其资源已释放后提交，总占用不超过用户授权的四卡。

## 验收与解释边界

真实Composite batch4已完成完整模型CPU forward/backward，无缺失梯度或非有限值。16个任务分别执行Gym reset、完整12D随机step和三相机录像检查。实际torchrun导入测试覆盖上轮重复初始化问题；独立评估进程清理训练rank及通信端口变量。

另在固定seed100、与策略相同的Gym/render协议重放StirVegetables episode11，984步、首次成功968步、最终success_time5。只恢复初始状态并补采集漏记的一次zero step，随后执行原始动作，未中途注入状态。这条GT成功不解决旧episode0/3全部物理漂移，也不算策略成功。其初始三相机RGB另与数据集第一帧逐项比较。

这份准备不会完成原step2000的冻结/部分解冻Video DiT对照。原权重在旧NAS及已知备份路径仍不存在可读文件；该项继续单独标为未完成，不能用新模型的结果替代。

运行目录为 `runs/robocasa365_composite16_4gpu_20260924/`：`plan.json`固定方案，`prequeue_acceptance.json`汇总排队前实际证据，`source/`与`source_hashes.json`固定执行版本，`status.json`区分准备与提交。没有`submission.json`时，不表示云任务已开始。

排队前验收已于 **2026-09-24 11:03 UTC** 完成：全部8,077个parquet哈希一致，16个环境全部通过，真实batch4前后向通过，10项CPU数值/协议测试及四进程真实trainer导入通过。episode11三相机首帧RGB的MAE为1.30–2.04/255，正确方向明显优于翻转图像。135个文件组成固定源码快照。云端查询确认尚未提交Composite任务；状态为等待Atomic结果与资源，到卡后的CUDA/NCCL及策略仿真短验收仍须执行。
