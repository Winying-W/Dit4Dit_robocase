# RoboCasa365 Composite seen → DiT4DiT 接口验收

2026-09-21：**真实数据到原版 DiT4DiT 的 forward、双 loss、动作预测已经通过。**
没有 backward、optimizer、参数更新或正式训练，也没有声称模型已具备该任务的策略能力。

## 本次范围与结果

- 官方 `TARGET_TASKS['composite_seen']` 中的 `StirVegetables`，target human，登记日期 `20250814`，horizon=2400。
- 只下载这一个任务：501 条轨迹、409202 帧、20 FPS、三路 256×256 RGB。
- 数据源：ModelScope `nv-community/robocasa365-datasets` 的 `target/composite/StirVegetables/20250814/lerobot.tar`，保留原始 `extras/`。镜像来源并非独立逐字节校验过的 Box 副本。
- 此任务并非所有演示都固定底盘。验收模型使用 episode 0 的 timestep 100；episode 0、3、5、10、11 均经数据检查确认底盘四维全为零、control_mode=-1。没有将移动轨迹的底盘命令擅自清零。
- 随机环境检查按官方例子将 base command 置零，完成 reset、120 次 step 和 MP4。

| 验收项 | 实测结果 |
|---|---|
| LeRobot loader | 501 条轨迹均被读取，无跳过 |
| 连续 batch | 3 个 batch，每批 2 个样本，包含 episode 尾部 |
| 标准 dataloader 入口 | `build_dataloader(..., dataset_py='robocasa365_datasets')` 通过 |
| 模型 batch | 1 个真实样本，当前帧＋8 个未来视频监督帧 |
| 视频输入 | `[1,9,3,128,384]`，每个时间点横向拼接左/右/手腕 |
| state / action | `[1,1,64]` / `[1,16,32]`，有效维度 16 / 12 |
| Cosmos hidden | `[1,576,2048]` |
| action loss | 1.3765853643417358，有限 |
| future-video loss | 0.9006444811820984，有限 |
| 原始预测 | `[1,16,32]`，全部有限 |
| 反归一化＋环境排列 | `[1,16,12]`，全部有限 |
| GT 动作归一化往返 | 最大误差 `4.43e-8` |
| 显存 | 峰值 allocated 35.40 GiB / reserved 35.46 GiB，单张 L20 |
| 梯度与更新 | `torch.no_grad()`，backward=0，optimizer step=0 |

完整模型由发布的 `dit4dit_robocasa_gr1/final_model/pytorch_model.pt` 严格加载。
仅冻结文本编码器加载为 BF16，视频和动作模型权重保留 FP32；模型自己的 autocast 逻辑不变。
未修改 `DiT4DiT/model/`。GR1 权重仅用于接口验收，365 归一化来自本任务数据，未使用 GR1 动作统计。

224×224 单相机分辨率的 loader 也通过（拼图 224×672），但 9 帧完整模型 forward 在 44.53 GiB L20 上 OOM，故单卡验收使用 128×128。不能据此声称 224 分辨率训练显存已验证。

## GR1 与 365 schema 对照

GR1 列来自仓库官方配置和 transform 源码；本次没有另下载 GR1 数据实测 batch。
365 列来自下载数据和实际运行。

| 项目 | GR1 官方配置 | 本次 RoboCasa365 |
|---|---|---|
| 相机 | ego_view | agentview_left / agentview_right / eye_in_hand |
| 原始 state | 29 个关节值 | 16：base pos(3)、base quat(4)、base-relative EEF pos(3)、EEF quat(4)、gripper qpos(2) |
| state 变换 | 各部分 sin/cos → 58 → pad64 | 保持原位置/四元数/夹爪顺序 → pad64 |
| 原始 action | 左臂7、右臂7、左手6、右手6、腰3 | base4、mode1、EEF position3、EEF rotation3、gripper1 |
| action 容器 | 29 → pad32 | 12 → pad32 |
| 动作语义 | joint position | OSC_POSE delta，input_ref_frame=base；rotation 为控制器旋转增量 |
| action chunk | 16 | 16，episode 边界外 label 使用 mask 排除 |
| 图像时间序列 | 0,2,…,16 共9帧 | 相同；相机在同一时间点拼接，不当作未来帧 |

365 连续动作按数据集 min/max 归一化；mode/gripper 保持官方值。state 保留原值，没有对四元数施加 sin/cos。
模型的预测只接收当前拼图、state 和 instruction，未来帧仅用于 forward 的视频监督。
环境排列：`dataset[..., [5,6,7,8,9,10,11,0,1,2,3,4]]`。
预测没有送入环境执行，本次停止于接口验收。

## GT replay 的证据与限制

每条只在开头恢复原始 XML、episode metadata 和初始 MuJoCo state，后续只执行 GT action；没有逐帧设置录制状态冒充动作重放。

| episode | steps | 初始 state 最大误差 | GT action replay 任务完成 |
|---|---:|---:|---|
| 0 | 903 | 0 | 否 |
| 3 | 1014 | 0 | 否 |
| 5 | 714 | 5.96e-8 | 是 |
| 10 | 1223 | 0 | 是 |
| 11 | 984 | 0 | 否 |

三路初始画面对齐（uint8 RGB MAE 约 1.2–2.1，倒置图像误差明显更大）。官方动作重排与适配器映射一致。
两条固定底盘 GT 轨迹成功复现任务过程；五条中有三条未完成，且所有长程动作回放都有非零状态偏移。
**这证明接口与至少两条任务回放可用，不代表所有演示可以确定性重现。** 原始失败证据完整保留，偏移来源尚未完全定位；后续正式训练/闭环评测前仍应排查重放稳定性。

## 复现模型验收

```bash
cd /file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
GPU_ID=5 bash scripts/robocasa365/run_model_interface.sh
```

首次环境依赖使用本机已有 RoboCasa365 venv（Python 3.11、MuJoCo 3.3.1）和本项目模型 conda 环境。
`env.sh` 中的 RoboCasa365 路径可由 `ROBOCASA365_ROOT` 覆盖。上述命令使用已下载数据和权重，不启动训练。

```bash
source scripts/robocasa365/env.sh
CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 "$SIM365_PYTHON" scripts/robocasa365/env_smoke.py \
  --output results/robocasa365_interface
"$SIM365_PYTHON" scripts/robocasa365/inspect_data.py \
  --dataset "$DATASET365" --output results/robocasa365_interface
CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 "$SIM365_PYTHON" scripts/robocasa365/replay.py \
  --dataset "$DATASET365" --episodes 5 10 --output results/robocasa365_interface/replay_confirm
```

## 文件与证据

- `DiT4DiT/dataloader/robocasa365_datasets.py`：robot config、LeRobot adapter、时间对齐拼图、反归一化。
- `DiT4DiT/dataloader/__init__.py`：新增独立 dataloader 分支。
- `DiT4DiT/config/robocasa/dit4dit_robocasa365_interface.yaml`：单任务接口配置。
- `scripts/robocasa365/`：环境、数据、replay、模型与语义检查脚本。
- [模型实测报告](results/robocasa365_interface/model_128/model_forward.json)
- [适配器语义验证](results/robocasa365_interface/model_128/adapter_verified.json)
- [完整数据 schema](results/robocasa365_interface/dataset_schema.json)
- [环境报告](results/robocasa365_interface/env.json)
- [原始三条 replay](results/robocasa365_interface/replay.json) / [补充两条 replay](results/robocasa365_interface/replay_additional/replay.json)
- [成功 replay：episode 5](results/robocasa365_interface/replay_000005.mp4)
- [成功 replay：episode 10](results/robocasa365_interface/replay_additional/replay_000010.mp4)
- [真实模型输入拼图](results/robocasa365_interface/model_128/input_mosaic.png)

2026-09-23 GT回放排查更新：已确认采集时漏记的零动作初始化导致0.05秒时间错位，回放入口已修复并改为每条新建环境；剩余开环偏离未全部消除。详情见[定位报告](ROBOCASA365_REPLAY_DIAGNOSIS.md)。
