# DiT4DiT 本机运行记录

2026-09-18 已在本机跑通官方预训练策略的 LIBERO 闭环评测。

## 已验证结果

- LIBERO-Spatial：10 个任务，每任务 1 个 episode，**10/10 成功（100%）**；评测循环约 95 秒。
- 这是小规模流程验证；官方完整设置为 4 个 suite、每任务 50 个 episode，不能用此次结果替代论文完整指标。
- NVIDIA L20 GPU 2；PyTorch 2.7.0+cu128；Python 3.10.20。
- GPU 动作推理检查：输出 `(1, 8, 8)`，全部数值有限，首次调用约 2.64 秒。
- LIBERO 双相机 EGL 渲染和环境步进检查通过。
- `uv pip check`：184 个包，全部依赖兼容。

结果：

- [每任务结果 JSON](results/libero_smoke_20260918/libero_spatial/summary.json)
- [评测日志](results/libero_smoke_20260918/libero_spatial.log)
- [10 个任务视频](results/libero_smoke_20260918/libero_spatial/)
- [策略服务日志](results/libero_smoke_20260918/server.log)
- [仿真检查](results/preflight/simulation.json) / [GPU 推理检查](results/preflight/policy.json)

## Conda 环境

使用服务器现有 Miniforge，与 `fastw`、`RoboDojo` 同级注册 `dit4dit`：

```bash
source /file_system/vepfs/intern/haozhe.jia/miniforge3/etc/profile.d/conda.sh
conda activate dit4dit
cd /file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
source scripts/local/env.sh
```

`env.sh` 会激活 `dit4dit`，同时配置项目路径、EGL、LIBERO 路径及缓存。仅激活 Conda 后，运行仿真前仍应加载这些项目配置。

由于 VEPFS 写入触发用户配额，环境、大文件、日志和结果实际存放在：

```text
/file_system/nas/intern/haozhe.jia/projects/DiT4DiT/
```

代码仍位于用户要求的 `projects/DiT4DiT`。项目 `.conda` 和 Miniforge 的 `envs/dit4dit` 都链接到 NAS 的 `conda-env`；`checkpoints`、`results`、`logs`、`third_party`、`.cache` 也通过项目链接访问。

## 一键运行

默认执行 Spatial 的全部 10 个任务，每任务 1 轮。运行前自行选择有约 30 GB 可用显存的 GPU。

```bash
cd /file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
GPU_ID=2 bash scripts/local/run_libero.sh
```

只跑任务 0：

```bash
GPU_ID=2 TASK_IDS=0 TRIALS=1 bash scripts/local/run_libero.sh
```

完整官方评测规模：

```bash
GPU_ID=2 TRIALS=50 \
SUITES="libero_spatial libero_object libero_goal libero_10" \
bash scripts/local/run_libero.sh
```

脚本自动检查端口、启动服务、等待 WebSocket 就绪、执行评测并在退出时关闭服务。可以通过 `PORT`（默认 15694）、`RUN_DIR`、`CKPT` 覆盖对应参数。输出包含逐任务视频、日志和 `summary.json`。

## 固定版本与本机适配

- DiT4DiT Git：`66a6f3a12e2c8740157c1e478795952c040d31dd`。
- LIBERO Git：`8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- 策略：`mondo-robotics/dit4dit-model`，revision `46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44`。
- Cosmos 配置/tokenizer：`nvidia/Cosmos-Predict2.5-2B`，revision `0d37c7498f54cee3c599d438d895a0a4a8608064`。
- 完整策略检查点已包含文本编码器、视频 Transformer、VAE 和动作模型。本次从它导出了 `Cosmos-Predict2.5-2B-from-policy` 初始化文件，随后由上游 `load_state_dict(..., strict=True)` 加载同一个完整策略。最终策略参数来源仍是官方检查点。导出记录在该目录的 `POLICY_DERIVATION.json`。
- 解决上游 requirements 中 PyTorch 2.7 与 CUDA 12.4/Triton/SymPy 的版本冲突，使用 CUDA 12.8、Triton 3.3、SymPy 1.14。
- 保留 `eva-decord`，去掉重复且 wheel 平台元数据异常的 `decord`。
- 使用官方 PyTorch3D 0.7.9 的 Python transforms 替换平台元数据异常的 `pipablepytorch3d`。当前项目使用的坐标变换已验证；没有编译额外 C++/CUDA ops。
- `configure_libero.py` 关闭 robosuite 的共享 `/tmp/robosuite.log` 文件日志，并为官方仓库自带的可信 NumPy 初始状态显式设置 `torch.load(..., weights_only=False)`，兼容 PyTorch 2.7。
- EGL 分发库复用服务器原有 OpenPI 环境中的库，复制到本项目 `.cache/egl`。
- 对上游评测脚本仅增加可选任务列表、参数验证、环境关闭和结果 JSON 输出，保留任务时限和动作处理逻辑。

## 环境重建与检查

环境记录位于：

- `scripts/local/environment.yml`：基础 Conda 定义。
- `scripts/local/conda-explicit.txt`：本机 Conda 包精确版本。
- `scripts/local/requirements-cu128.txt`、`requirements-libero.txt`：兼容依赖清单。
- `scripts/local/pip-freeze.txt`：最终 Python 包记录。

已有 Conda 环境下，可执行 `bash scripts/local/setup_environment.sh` 安装依赖。网络需要代理时可设置本机已有的 `HTTPS_PROXY=http://127.0.0.1:17890` 和 `HTTP_PROXY`。不需要修改其他项目环境。

```bash
source scripts/local/env.sh
python scripts/local/configure_libero.py
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 python scripts/local/check_sim.py
```

重新下载模型使用 `python scripts/local/download_models.py`，默认复用完整策略中的基座权重。`--full-backbone` 可单独下载 NVIDIA 初始化权重。Hugging Face 凭据保存在项目专用缓存，权限为 0600，不纳入 Git。
