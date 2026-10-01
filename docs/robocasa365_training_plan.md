# RoboCasa365 单卡训练与解冻方案（2026-09-22）

本次只使用官方 Composite seen 的 StirVegetables target human 数据。复用 NAS 中已下载的单任务数据（501 episodes）；云入口逐条检查 parquet、三相机视频和 replay states，缺失整个数据集时才下载该任务。不下载其余 364 个任务。

## 文献与代码依据

1. DiT4DiT, arXiv:2603.10448，§4.3 / Algorithm 1：冻结文本编码器和视觉 VAE，仅更新视频 DiT 与动作 DiT。附录训练表：视频 LR=1e-5，动作 LR=1e-4，AdamW betas=(0.9,0.95)，梯度裁剪=1；正式实验 32 GPUs、100000 steps，不能与本次短实验等同。
   - https://arxiv.org/html/2603.10448
   - https://github.com/Mondo-Robotics/DiT4DiT/blob/main/examples/Robocasa_tabletop/train_files/run_robocasa.sh
   - 仓库 `Cosmos25.py` 的 hook 和最终 hidden 使用 detach。本次保留公开实现：动作 loss 更新 Action DiT，视频 flow-matching loss 更新解冻的视频块。不声称 action gradient 穿过视频模型。
2. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success, arXiv:2502.19645：研究参数高效适配和连续动作头，附录列出 LoRA rank=32。这支持“有限算力先减少更新参数”作为可选路线，但该论文针对 OpenVLA，不能作为 DiT4DiT 的 LoRA 或逐层解冻收益证据。
   - https://arxiv.org/html/2502.19645

本地保存上述论文 HTML 与文本供复查。没有找到足以证明 RoboCasa365 上最佳解冻顺序的直接实验，因此以下逐步解冻是有检查关卡的工程试验，不是论文原配方。

## 一张卡上的有界试验

申请同一个已验证资源队列的一张 A800 80GB（8 CPU、128 GiB host memory），最长 4 小时，禁止自动重试或提交重复任务。

1. 挂载原 NAS，核验/准备单任务数据和已有模型环境。
2. 重新运行真实数据 forward、双 loss、prediction、动作反归一化验证。
3. 按真实动作筛选固定底盘且 mode=-1 的演示；固定 seed=42，16 条用于训练、4 条用于验证。按 episode 隔离，归一化只使用训练轨迹重算。
4. **Action 阶段：300 次 optimizer 更新**。冻结完整 Cosmos，训练完整 Action DiT（含 state encoder）。LR=1e-4，microbatch=1，累积4，warmup20；当前帧输入。未来动作 chunk=16，32 action / 64 state 容器保留，padding loss 使用 mask。
5. **部分联合阶段：100 次 optimizer 更新**。继续训练 Action DiT，同时解冻视频 Transformer blocks 16、17（零起始，与 extraction layer=17 对齐）。其余视频层、text encoder、VAE 仍冻结。视频 LR=1e-5，动作 LR=1e-4；当前帧+8未来视频帧，loss=action+video；启用视频梯度检查点。
6. 每个阶段首步检查有限非零梯度与真实参数变化。第1、10步以及每50步保存 checkpoint 并读回核对；第10步退出进程再从同一 scheduler/RNG 状态继续，验证实际断点续训。阶段结束保存验证记录。验证使用固定噪声、未参与训练的轨迹、当前图像；4条验证轨迹各抽取2个窗口，共8个窗口。指标为小规模 action flow-matching proxy，非全验证集平均，更不是成功率。

每路图像128×128，拼图128×384。冻结 text encoder 为 BF16，其余权重保持 FP32，沿用模型自身 autocast。单卡可行性以实际 backward 与 optimizer 显存为准。若有限梯度/显存检查失败，立即退出并修复，不能跳过或静默退化成只 forward。

## Checkpoint 与恢复

checkpoint 保存所有累计训练过的参数（第二阶段同时包含 action 与视频16/17层）、AdamW 状态、scheduler、torch/CUDA/sampler RNG、split 与原始GR1权重路径。它是 **依赖原始 checkpoint 的训练增量文件**，不能直接传给官方 `from_pretrained`。

```bash
source scripts/robocasa365/env.sh
"$MODEL365_PYTHON" scripts/robocasa365/train_single_gpu.py \
  --dataset "$DATASET365" \
  --checkpoint checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt \
  --output results/robocasa365_train_20260922/train \
  --resume results/robocasa365_train_20260922/train/action_latest.pt
```

恢复必须在已分配的 GPU 任务中运行。本机不启动开发机 GPU 训练。

## 后续扩大解冻

本轮确认 partial_joint 的视频梯度和 checkpoint 后，再考虑解冻0–17层，最后整段视频 DiT；各步保留 text/VAE 冻结并重新评估显存。完整单任务训练再扩大至训练集全部演示、episode holdout、多次验证和闭环。GT replay 有部分轨迹漂移未完全定位，因此不能以本轮 loss 宣称闭环控制能力。

## 本轮已执行结论

300次冻结更新和100次部分解冻更新均已完成，固定验证FM loss依次为2.065245→0.181549→0.176006；峰值显存24.09 GiB。第400步checkpoint已在独立云进程恢复并复现最终验证结果，恢复任务Success。完整证据见[训练报告](../ROBOCASA365_TRAINING.md)。

上面的action_latest恢复命令从第300步开始重跑部分解冻阶段。只验证最终checkpoint、不追加训练时，将 `--resume` 改为 `results/robocasa365_train_20260922/train/partial_joint_latest.pt`，并保持 `--action-steps 300 --joint-steps 100 --accumulation 4`。报告中的恢复耗时仅对应恢复进程，不代表400步总训练时间。

## 2000步冻结骨干试验之后的训练关卡

2026-09-23：冻结Action DiT分支已从step300续训到step2000；该分支没有使用部分解冻step400的权重。续训学习率以`runs/robocasa365_dual_20260923/live/continuation_schedule.json`为准，不能把历史配置中的1e-4误记为续训LR。16条训练轨迹有12,786帧，每步随机抽4个窗口，2000步累计8000次有放回窗口抽样，相当于起始帧数量的约0.626倍；这不是完成一次无重复遍历，每个窗口包含16个未来动作。

在完整闭环结果出来后，下一轮建议先做有界诊断：

1. 从训练集选一条已验证可GT action replay成功的固定底盘演示（例如episode5），保留现有20条划分及归一化文件作为独立基线。单演示实验另建输出目录，显式记录它使用的训练统计，绝不覆盖现有2000步checkpoint。
2. 只训练Action DiT，检查真实推理的动作chunk能否拟合该演示。除FM loss外，计算反归一化后的EEF delta误差、旋转delta误差和二值夹爪正确率，分别检查chunk的第1、8、16步；只看加噪训练loss不能证明采样出的动作正确。当前非移动子集的4个base/torso分量与1个mode分量为常数，它们仍保留在完整12维schema和训练目标中；因此还需单独报告6个手臂分量与夹爪的推理误差，避免整体平均值掩盖操作维度的误差。
3. 从这条训练演示的初始场景做模型闭环，模型独立输出全部动作。这是训练场景过拟合验收，明确不算留出集成功率。若仍失败，检查视频/状态与目标动作的时间对齐、噪声采样与推理步数、执行chunk长度，再决定是否增加更新次数。
4. 若单演示闭环通过，再扩大固定底盘数据。固定留出的4条不能进入训练，其他非移动候选需独立构建新split；归一化仅重新拟合新训练集，与新checkpoint绑定。先比较冻结骨干效果，再用相同数据和相同评估协议做部分视频解冻对照。
5. 视频解冻保留论文/公开实现的梯度路线：action loss更新Action DiT，video loss更新解冻的视频层；Qwen文本编码器和VAE始终冻结。扩大解冻范围前重新检查真实梯度、权重变化和单卡显存，不能把开关设成train就当作视频层已学习。

这些是后续实验关卡，尚未自动启动新训练任务，也不承诺延长训练或解冻即可获得非零成功率。
