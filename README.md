# DiT4DiT / Qwen concat on RoboCasa365 Atomic18

两套可独立启动的实验代码，用于迁移到普通 Linux GPU 服务器。

| 目录 | Cosmos 初始化 | Action head 初始化 | Qwen |
|---|---|---|---|
| `experiments/robocasa_official_recipe` | 官方 GR1 | 官方 GR1 | 不接入 |
| `experiments/concat_random_head` | 官方 GR1 | 随机，seed 42 | 冻结 Qwen3-VL-4B-Instruct，LN + Linear 后拼接 |

两个目录各自带有完整 `source/`，源码保持一致，入口严格区分配置。这里的 concat 指 C 组；不包含旧的 128/repeat-1 A/B 实验和云平台调度代码。
**基线 vs concat 同时改变 head 初始化和 Qwen，不能把差异全部归因于 Qwen。**

共同配方：每视角 224×224，三视角横拼 224×672；9 个视频帧；action/state/horizon=32/64/16，真实 action 12 维；repeat=4；4 GPU，global batch=128；Cosmos video DiT 用 video loss 更新，LR=1e-5；action head/projector LR=1e-4；action 梯度不进入 Cosmos；VAE/text/Qwen 冻结；HF cosine_with_min_lr，warmup 为总步数 2.5%，末端比例 1/60；官方每 microbatch masked mean loss；评估每次执行 8 步。

## 1. 安装训练环境

建议 Python 3.10，先创建隔离环境，再安装。当前验证环境为 torch 2.7.0+cu128 / torchvision 0.22.0+cu128。目标机器驱动需支持所选 CUDA wheel。

```bash
conda create -n dit4dit python=3.10 -y
conda activate dit4dit
bash tools/install.sh
bash tools/test_cpu.sh
```

`tools/install.sh` 固定 Cosmos 所需的 diffusers revision 和 Qwen3 支持版本。需要编译器构建 PyTorch3D 的 Python 包，但本实验仅使用 transforms，不需要它的 CUDA 扩展。训练无需安装 RoboCasa 仿真器；闭环评估使用另一个 Python 环境。

## 2. 必须单独迁移的权重和数据

Git 保存代码、固定的数据划分和归一化统计，**不包含大模型权重、视频/轨迹数据和仿真资产**。请从原服务器复制以下完整目录，复制符号链接时保留其实际内容（例如 `rsync -aL`）。具体原始路径见 [TRANSFER.md](TRANSFER.md)。

```text
/models/
├── dit4dit_robocasa_gr1/
│   ├── config.yaml
│   ├── dataset_statistics.json
│   └── final_model/pytorch_model.pt
├── Cosmos-Predict2.5-2B-from-GR1/
│   ├── model_index.json
│   ├── transformer/     # config + safetensors
│   ├── vae/             # config + safetensors
│   ├── text_encoder/    # config + safetensors
│   ├── tokenizer/
│   └── scheduler/
└── Qwen3-VL-4B-Instruct/ # 全部配置、processor/tokenizer 和全部权重分片
/data/atomic/
├── CloseBlenderLid/20250822/  # meta/, data/, videos/, extras/
└── ...                      # 共 18 个任务，保持相对层级
```

`--cosmos-model` 覆盖 GR1 config 内旧服务器的 `base_model` 路径，无需手改 checkpoint 配置。
**不能只复制 GR1 的 .pt 文件**：构建模型仍需要 config、统计以及 Cosmos pipeline 的配置/tokenizer/组件文件。

固定数据划分：8216 个训练 episode、910 个验证 episode，seed 20260924。运行以下命令校验全部 parquet SHA256、视频存在性，并只替换数据根路径：

```bash
python tools/prepare_data.py --data-root /data/atomic --output prepared/atomic18
export ROBOCASA_MANIFEST="$PWD/prepared/atomic18/manifest.json"
export GR1_CHECKPOINT=/models/dit4dit_robocasa_gr1/final_model/pytorch_model.pt
export COSMOS_MODEL_DIR=/models/Cosmos-Predict2.5-2B-from-GR1
export QWEN_MODEL_DIR=/models/Qwen3-VL-4B-Instruct
bash tools/test_cpu.sh  # 设置了以上变量后也运行真实数据/head 初始化检查
```

## 3. 启动两套实验

每组需要四张 GPU；八卡机器可同时跑，两组使用不同的 GPU 和输出目录。原实验以 A800 80GB 为目标，但 **224/repeat-4 的实际显存与速度仍需在目标机器测量**。默认 microbatch=1、accumulation=32；支持 1/2/4/8/16/32，global batch 始终为 128。

下面 `TOTAL_STEPS=20000` 只是命令示例，不是已经确定的实验预算。先按目标机器速度确定预算；两组使用相同预算。开始 smoke 时就传入最终计划步数，续训不能改变步数、microbatch 或其他 schedule 参数。

```bash
TOTAL_STEPS=20000

python experiments/robocasa_official_recipe/train.py \
  --manifest "$ROBOCASA_MANIFEST" --checkpoint "$GR1_CHECKPOINT" \
  --cosmos-model "$COSMOS_MODEL_DIR" --output outputs/baseline \
  --steps "$TOTAL_STEPS" --gpus 0,1,2,3 --microbatch 1 --stop-after 2

python experiments/concat_random_head/train.py \
  --manifest "$ROBOCASA_MANIFEST" --checkpoint "$GR1_CHECKPOINT" \
  --cosmos-model "$COSMOS_MODEL_DIR" --qwen-model "$QWEN_MODEL_DIR" \
  --output outputs/concat --steps "$TOTAL_STEPS" --gpus 4,5,6,7 \
  --microbatch 1 --stop-after 2
```

`--stop-after 2` 执行真实完整 forward/backward、优化器、DDP 和保存。HF 调度第 1 步 LR=0，第 2 步才发生非零更新；所以只跑一步不足以验证更新。初始 validation 会先遍历验证窗口，可能需要较长时间。

确认 `metrics.jsonl` 中 loss/gradient 有限，`distributed_preflight.json` 中设备正确，再使用原命令 **加 `--resume`，把 `--stop-after 2` 改为 `--stop-after 100`**。100 步后检查速度、显存和 `rank_weights_step_000100.json`；最后移除 `--stop-after`，保留 `--resume` 完成训练。每组的训练器会检查梯度、checkpoint 读回和各 rank 权重一致性。

- `action_latest.pt`：完整 optimizer/scheduler/RNG 恢复文件。
- `action_step_*.pt`：策略快照，不能用于恢复优化器。
- `--dry-run`：校验路径并打印启动命令，不启动训练。
- 启动器使用输出目录锁，并在恢复前修整未提交日志。
- 迁移后启动的新实验可以正常恢复；历史 checkpoint 内含旧的绝对路径，不能直接当作新服务器的精确续训文件。

可先进行四 rank 的 CPU 导入/通信检查（不读取模型和数据）：

```bash
python experiments/robocasa_official_recipe/train.py \
  --manifest /tmp/unused.json --checkpoint /tmp/unused.pt --cosmos-model /tmp/unused \
  --output outputs/cpu_preflight --steps 20000 --cpu-runtime-preflight
```

## 4. 闭环评估

另建 Python 3.11 仿真环境，安装 RoboCasa365 及对应资产，版本见 [TRANSFER.md](TRANSFER.md)。训练环境的 Python 驱动模型，`--sim-python` 指向仿真环境 Python。需要 EGL/OpenGL 驱动；不要复制旧服务器的硬编码系统动态库路径。

训练完成后，在训练环境执行（concat 对应替换 experiment 和 checkpoint）：

```bash
python tools/evaluate.py --experiment robocasa_official_recipe \
  --checkpoint outputs/baseline/action_step_020000.pt \
  --manifest "$ROBOCASA_MANIFEST" --cosmos-model "$COSMOS_MODEL_DIR" \
  --sim-python /path/to/simulator/.venv/bin/python \
  --output outputs/baseline_dev_final --gpu 0
```

默认每任务 seeds 100–109，共 18×10 次开发评估，执行 horizon=8。只有最终余弦降到底的快照用于正式比较；5k 快照只作诊断。开发 seeds 不等于 manifest 中 1000–1099 的保留最终测试 seeds。

## 5. 已验证范围

详见 [VALIDATION.md](VALIDATION.md)。CPU 逻辑、真实数据、真实 GR1 head 初始化、两组真实模型动作路径前向/反向、实际四进程入口及仿真动作协议已验证。**未在目标服务器执行完整模型 GPU 训练、显存验收或闭环成功率评估，也未在全新环境实际重装全部依赖。** 请按上面的 2 步 → 100 步流程验收后长跑，不能把 CPU 通过等同于 GPU 全链路通过。

上游 DiT4DiT 使用 MIT License，保留 `LICENSE` 和源码版权声明。迁移修改及原始源码摘要见 [PORTABILITY.md](PORTABILITY.md) 和 `UPSTREAM_SOURCE_HASHES.json`。

## 可选：真实模型 CPU 动作路径检查

当前机器无空闲 GPU 时，可加载真实 GR1 + Qwen，对两套模型各做一次动作路径 forward/backward：

```bash
CUDA_VISIBLE_DEVICES= NO_ALBUMENTATIONS_UPDATE=1 python tools/check_model_cpu.py \
  --manifest "$ROBOCASA_MANIFEST" --checkpoint "$GR1_CHECKPOINT" \
  --cosmos-model "$COSMOS_MODEL_DIR" --qwen-model "$QWEN_MODEL_DIR" \
  --output outputs/cpu_model_check.json
```

需要数十 GB 内存和数分钟计算；测试使用当前帧、repeat=4，不训练视频分支，不执行 optimizer。它验证严格 GR1 加载、真实图像/指令到动作 loss、head/projector 的有限梯度及冻结分支隔离，不能替代 GPU smoke。
