# 本次交付验证（2026-10-19）

以下是本次执行结果，不是从旧实验 README 推断出的状态。

| 检查 | 结果 |
|---|---|
| 新入口、两份源码一致性、固定划分、Qwen 梯度/当前帧/attention mask | 8 passed，6 subtests passed |
| 基线 optimizer/recovery/distributed/data/prediction/replay 测试 | 23 passed，4 subtests passed |
| concat 同组测试 | 23 passed，4 subtests passed |
| RoboCasa 仿真 Python 中的动作映射/协议测试 | 5 passed |
| 全部 Atomic18 数据重定位检查 | 18 个任务；9126 个 episode 的 parquet SHA256 校验通过；三路视频存在且非空 |
| 官方学习率调度 | 与 HF 的最大因子差 1.1102230246251565e-16 |
| 真实训练数据 | 9 帧，每帧 3×224×672；action 16×32、state 1×64；12 个有效 action 维度 |
| 真实 GR1 action head 重置 | 247 个张量、163276320 个参数；同 seed 可复现，不同 seed 不同；原 head 无张量被保留；不推进全局 RNG |
| 基线实际入口四进程 CPU/Gloo | 通过，collective sum=10，重复初始化安全 |
| concat 实际入口四进程 CPU/Gloo | 通过，collective sum=10 |
| 两套 portable CLI 使用实际数据/权重路径 dry-run | 通过，基线 gr1/no-Qwen，concat random/Qwen；global batch=128 |
| 环境导入 | Cosmos2_5_PredictBasePipeline、Qwen3VLForConditionalGeneration、实际训练器、数据适配器均通过 |
| Python AST / shell 语法 | 通过 |
| 真实 GR1 基线 CPU 动作路径 forward/backward，repeat=4 | loss=0.55859375；247 个训练张量梯度均有限；约 220 秒 |
| 真实 GR1 Cosmos + 随机 head + Qwen CPU 动作路径 forward/backward，repeat=4 | loss=1.265625；251 个 head/projector 训练张量梯度均有限；约 209 秒 |

测试环境：Python 3.10.20、torch 2.7.0+cu128、torchvision 0.22.0+cu128、diffusers 0.37.0.dev0（固定 revision）、transformers 4.57.6、PyTorch3D 0.7.9、eva-decord 0.6.1、pydantic 2.10.6。仿真动作协议在独立 RoboCasa Python 3.11 环境运行。

第一次把仿真协议测试放在训练环境执行时，2 项因训练环境未安装 robocasa 而失败；随后使用实际仿真环境运行全部 5 项并通过。交付的 `tools/test_cpu.sh` 把训练测试与仿真测试分开，训练无需安装模拟器。

完整 GR1 权重严格加载通过；两组测试中 Cosmos 冻结，concat 的 Qwen encoder 冻结。完整 CPU 动作路径检查使用一条真实样本、当前帧，不包含 video-loss 或 optimizer 更新。

部分机器可读结果保存在 `validation/`。源代码和交付文件可运行 `python tools/verify_source.py` 校验。

## 尚未验证

- 全新环境从零执行 `tools/install.sh` 的安装过程。
- 目标服务器的完整模型 CUDA forward/backward、Cosmos video-loss 更新、NCCL、优化器显存和性能。
- 目标服务器的完整模型两步/100 步训练及 GPU 精确恢复。
- 仿真完整资产、EGL reset/step/render 和闭环成功率。

当前机器 GPU 都有占用，未启动 GPU 实验。上述未验证项必须在目标服务器完成，不能保证任意硬件/驱动/依赖组合“绝不报错”。README 提供 2 步 → 100 步 → 完整训练的验收流程。
