**Video DiT 后续适配的依据与边界**

本次复查已有论文归档、官方 RoboCasa-GR1 配置和实际梯度检查记录；没有修改或提交新的训练任务。

DiT4DiT 论文 §4.3 使用视频与动作的双 flow matching 目标，冻结文本编码器和 VAE，更新 Video DiT 与 Action DiT。附录给出视频学习率 1e-5、动作学习率 1e-4。官方 RoboCasa-GR1 YAML 的模式是 `joint`，公开启动脚本的冻结清单仅包含 text encoder、VAE；代码提取层为零起始的 17，对应论文表中的第 18 层。

当前 RoboCasa365 的 Atomic18 与 Composite16 主线是另一项已明确约定的基线：保留 GR1 的视频／语言基座，只更新整个动作分支。因此它的低成功率还不能证明 DiT4DiT 的联合适配方案在 RoboCasa365 上无效，也不能反过来保证联合适配会有效。

真正实施视觉适配时，不能只把视频参数设置成可训练。当前公开实现对供动作分支使用的视觉 hidden 执行 detach；已有真实 CPU backward 证明：action loss 更新动作参数，视频 flow matching loss 更新所选视频参数。视频更新需要真实未来视频监督。此前控制检查还确认，改变未来视频标签只改变辅助视频损失，动作输入、动作噪声和初始动作梯度保持相同；比较时需保留独立随机数流和分组梯度裁剪。

已有的有限显存候选选择视频 blocks 16、17（零起始），其余视频层、文本编码器和 VAE 冻结；这两个块覆盖当前提取位置。它是工程对照，不是论文证明最优的解冻范围。旧 CPU 控制检查覆盖 247 个动作梯度张量和 40 个视频梯度张量，没有证明当前 Composite 新检查点的 CUDA 更新、显存或闭环收益。

如果当前 120 条演示的命令拟合诊断支持继续检查视觉适配，下一项新对照应满足：

- 两路从同一份可访问的 Composite 检查点开始，数据、划分、统计、动作目标、采样次序、有效 batch 和评估场景相同，分别建立优化器与学习率计划。
- 冻结路只更新动作分支；部分联合路额外更新指定视频块并使用视频监督。两路动作输入和噪声对齐，加入视频参数不能改变动作梯度的裁剪尺度。
- 在申请训练卡前完成新的初始化合同、真实数据输入与 CPU 梯度检查；进入分配到的 GPU 后，必须先通过有限非零梯度、真实参数更新、显存和保存／恢复检查，再进入完整训练。
- 使用单独实验目录和开发评估，记录抓取、控制模式等阶段指标及真实闭环成功率。当前四卡 50k 主线与最终 1,600 次评估保持独立。

现有单卡入口的多任务 warm-start 显式面向 Atomic18 50k；不能直接把 Composite10k 路径替换进去就当作已经支持。若采用新的 Composite 起点，仍需要独立实现并验收初始化合同。新对照也不能冒充历史 16 演示 step2000 的指定起点试验。

OpenVLA 的参数高效微调研究可以启发受限资源下的方案选择，但它不是 DiT4DiT／RoboCasa365 上 LoRA 或某个解冻顺序有效的直接证据。当前尚未确定需要扩大解冻，更没有“解冻后成功率必然提升”的结论。

参考：[DiT4DiT 论文](https://arxiv.org/html/2603.10448)、[官方 RoboCasa 启动脚本](https://github.com/Mondo-Robotics/DiT4DiT/blob/main/examples/Robocasa_tabletop/train_files/run_robocasa.sh)、[已有 CPU 控制检查](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_atomic50k_ab_controls_20260927/cpu_acceptance_summary.json)、[命令拟合诊断协议](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/docs/robocasa365_composite_command_diagnosis_20260928.md)。

2026-10-01 更新：已建立独立的 Composite25k 新对照，沿用 16 任务、7,271/806 条演示划分。现有入口已补充并验收显式 Composite 初始化合同，不能用此完成情况代替原16演示step2000实验。新的真实CPU控制检查已经通过：同一25k起点、真实StirVegetables训练帧上，两路输入、动作噪声、原始及裁剪后动作梯度一致；改变未来视频标签只改变辅助视频损失。

生产入口的CPU两路各2步及部分联合路独立恢复正在执行，CPU有效batch为1；拟议GPU有效batch为64。尚未提交新视觉GPU任务、验证显存或得到闭环收益。证据位于 `runs/robocasa365_composite25k_visual_ab_20261001/runtime/cpu_controls/acceptance.json`。
