**截至 2026-10-01 15:24 UTC，冻结视觉对照已在单张A800上运行，已独立恢复到第4步并保存，正在进行实际策略/仿真测试；局部联合路仍等待空闲GPU。尚无视觉对照闭环成功率。**

从同一Composite25k动作权重出发，使用全16任务（7271训练、806验证演示，5,403,990训练帧），重置两路优化器，各新增2000步。冻结路更新State Encoder、Action Encoder/Decoder和Action DiT；部分联合路额外更新Video DiT第16/17块，使用真实未来视频FM损失。VAE和Qwen始终冻结。

两路都让指定视频块的参数使用FP32，前向使用BF16 autocast；冻结路不更新这些参数。评估在加载检查点前恢复相同指定层精度。两路共享训练样本、动作噪声和动作学习率，并按参数组独立裁剪梯度。计划单卡microbatch1×accumulation64，有效batch64，warmup200步，动作学习率1e-4，视频学习率1e-5。

完成的验收：

- 真完整模型的features、targets、state、mask、动作噪声、时间步、原始及裁剪后动作梯度相同；改变未来视频标签仅改变辅助视频损失。
- 生产trainer两路各完成2次CPU更新，独立进程从部分联合路第1步恢复到第2步，共5次实际CPU优化器更新。两路首步动作更新完全相同。
- 独立恢复后整个检查点1,149个张量、905,073,471个元素逐值一致，包括权重、Adam、学习率与随机数状态。三个检查点审计通过；可训练参数及Adam为有限FP32，采样随机数独立重现，两个视频块40个张量均有实际更新。
- 精度恢复、随机数和初始化回归16项通过。16任务数据验收复用主线，manifest、normalization、adapter及多任务数据代码逐项一致。
- EFS训练输出与独立vePFS备份各64MiB实际写入、fsync、rename和读回通过；备份观察器PID 3781306存活，配置、冻结源码及CPU证据已归档。

CPU有效batch1，只用于接口、数值与恢复验收；不宣称GPU显存已经验证，也不使用CPU产生的权重初始化正式实验。候选215文件源码与预排队验收已冻结；主线已提交源码和训练协议不变。

两路提交前检查均完成；14:53 UTC队列24张A800全部占用，均记录`waiting_for_idle_capacity`。分配后分别先做2次真实CUDA更新、独立进程恢复到4步、检查点审计以及8步实际策略/仿真测试。通过才继续至2000步；checkpoint第4、1000、2000步由观察器复制至独立文件系统并审计。

首轮开发筛查为GetToastedBread、DeliverStraw、StirVegetables各20次，共60次；扩大前需完整16任务开发评估。主线最终1600次独立场景不用于选择该候选。新实验起点是Composite25k，不能冒充历史16演示step2000对照。

前两版候选的证据仍保留。第一版BF16可训练视频参数通过CPU恢复，但存在小更新舍入风险：对32个真实矩阵各前65,536个值施加人为恒定单位梯度和200步warmup，BF16采样元素中13.6%–79.6%未变，FP32全部变化。这是合成梯度数值实验，不是真实训练丢失更新的比例，也不能解释视觉一直冻结的旧主线低成功率。中间版只将联合路转FP32，导致初始特征/动作梯度不同，被标记`controls_not_proven`，未进行优化器更新或GPU提交。当前版本已将两路及推理精度对齐并通过全部上述CPU验收。

[当前计划](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_visual_precision_matched_20261001/plan.json) · [CPU精确恢复](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_visual_precision_matched_20261001/runtime/cpu_preflight/acceptance.json) · [预排队验收](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_visual_precision_matched_20261001/prequeue_acceptance.json) · [备份状态](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_visual_precision_matched_20261001/backups/status.json) · [冻结路提交检查](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_visual_precision_matched_20261001/frozen/submission_status.json) · [联合路提交检查](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_visual_precision_matched_20261001/partial_joint/submission_status.json)

2026-10-01 15:13 UTC：独立空闲卡观察器已启动（PID 3813208），每60秒检查资源。首次提交后等待实际分配，再考虑第二路；有已有提交凭据、模糊意图、失败任务或主线未分配时不重复申请。9项控制检查和真实无提交扫描通过。实际提交仍由原冻结助手重新完成全部候选、备份和资源检查。当前无空闲卡、无新视觉GPU任务。

2026-10-01 15:24 UTC实际更新：冻结路`t-20261001231446-k5fwd`为Running，观测CUDA优化器步数[1, 2, 3, 4]，已发布第4步可读回权重。已观测动作loss/梯度有限，有效batch64，峰值分配显存约23.01GiB。整体CUDA/实际仿真门槛仍待最终验收；这些检查不是成功率。联合路等待额外空闲卡，既有观察器负责顺序提交。
