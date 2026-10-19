# 迁移清单

以下路径是**原服务器上的复制来源**，不是新服务器的运行配置。新服务器可自由选择目标路径，通过参数传入。

| 资源 | 原服务器来源 |
|---|---|
| GR1 checkpoint 整目录 | `/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/artifacts/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/` |
| Cosmos construction assets 整目录 | `/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/artifacts/checkpoints/Cosmos-Predict2.5-2B-from-GR1/` |
| 冻结 Qwen3-VL-4B | `/file_system/efs/checkpoint/intern/haozhe.jia/projects/Owen_dit4dit/models/Qwen3-VL-4B-Instruct/` |
| Atomic18 数据根目录 | `/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/data/robocasa365/v1.0/target/atomic/` |

示例（在原服务器运行，将 `TARGET_SERVER` 改成实际 SSH 目标）：

```bash
TARGET_SERVER=user@server
rsync -aL --info=progress2 /file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/artifacts/checkpoints/dit4dit-model/dit4dit_robocasa_gr1 "$TARGET_SERVER:/models/"
rsync -aL --info=progress2 /file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/artifacts/checkpoints/Cosmos-Predict2.5-2B-from-GR1 "$TARGET_SERVER:/models/"
rsync -aL --info=progress2 /file_system/efs/checkpoint/intern/haozhe.jia/projects/Owen_dit4dit/models/Qwen3-VL-4B-Instruct "$TARGET_SERVER:/models/"
rsync -aL --info=progress2 /file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/data/robocasa365/v1.0/target/atomic "$TARGET_SERVER:/data/"
```

应先在目标服务器创建 `/models` 和 `/data`，或自行替换成有写权限的目录。不要复制 conda 环境的绝对路径软链接，按 README 重建环境。

`data/model_assets.json` 记录 GR1 SHA256、上游 revision 和 Cosmos 各组件 SHA256。可用 `tools/verify_model_assets.py` 验证复制完整性。Qwen 使用完整原目录复制，并检查所有分片；这不等于验证其训练来源。

```bash
python tools/verify_model_assets.py \
  --checkpoint /models/dit4dit_robocasa_gr1/final_model/pytorch_model.pt \
  --cosmos-model /models/Cosmos-Predict2.5-2B-from-GR1 \
  --qwen-model /models/Qwen3-VL-4B-Instruct
```

## 仿真环境（仅闭环评估需要）

已用于本次动作协议测试的版本：

- Python 3.11 独立环境。
- RoboCasa365：`https://github.com/robocasa/robocasa`，commit `921c9a5736a8d0ea5589657898aadcfa55a6a195`。
- robosuite：`https://github.com/ARISE-Initiative/robosuite`，commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`（版本标识 1.5.2）。
- MuJoCo 3.3.1，gymnasium 0.29.1。

在仿真环境安装上述指定版本与其依赖，按 RoboCasa 官方安装说明下载 kitchen/object 资产。可参考原服务器 `/file_system/vepfs/intern/haozhe.jia/projects/openpi/robocasa365/` 的资产布局，但不要直接复制 `.venv` 当作可迁移环境。无头渲染需要目标机的 EGL 库和 NVIDIA 驱动。

```bash
# 在仿真 Python 环境，repo 路径替换为本仓库的绝对路径
export PYTHONPATH=/path/to/Dit4Dit_robocase/experiments/robocasa_official_recipe/source
MUJOCO_GL=egl python "$PYTHONPATH/scripts/robocasa365/tests/test_eval_protocol.py"
```

协议测试不创建完整环境，也不验证所有仿真资产和渲染。目标机应再运行 `source/scripts/robocasa365/env_smoke.py --help` 查看环境冒烟入口，然后做实际 reset/step/render 验证。
