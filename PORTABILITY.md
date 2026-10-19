# 相对当前实验快照的迁移修改

原始来源：`Owen_dit4dit/experiments/robocasa_official_recipe` 和 `concat_random_head`，两份 source 原本逐字节相同。原始逐文件 SHA256 位于 `UPSTREAM_SOURCE_HASHES.json`。

- 每个实验新增 `train.py`，共享 `tools/launch.py`：普通单机四卡 torchrun；路径、训练步数、GPU、microbatch、resume 都由 CLI 明确传入；基线拒绝 Qwen 参数，concat 强制 Qwen 与随机 head。
- `base_framework.py` 支持 `COSMOS_MODEL_DIR` 覆盖 checkpoint 配置内的旧路径，训练/评估都生效；仍严格加载完整 GR1 state_dict。
- `train_distributed.py` 增加已完成训练 resume 的提前返回，避免空训练循环后引用未定义 metrics。并在首两步记录参数更新，覆盖官方 warmup 第 1 步 LR=0、第 2 步非零更新的情况。训练目标、优化器、梯度路径、初始化和学习率公式未改。
- 真实资源测试通过 `GR1_CHECKPOINT`、`ROBOCASA_MANIFEST` 取路径。
- 保留 Atomic18 的原始 episode 划分、parquet SHA256 和训练集归一化。`tools/prepare_data.py` 验证数据，只重定位路径。
- 省略原云平台 `cloud/`、`scripts/volc/`、shell 入口及历史专用 `prepare_recovered_runtime.py`、`verify_prequeue.py`；使用仓库根 tools 提供可移植替代。保留模型、数据、训练、恢复、评估及其它诊断 Python 源码。
- 新增环境检查、模型文件校验、实验入口契约和 Qwen mask/梯度隔离测试。

两个实验的最终 source 继续保持逐字节一致，由 `tests/test_portable.py` 校验。运行输出不进入 Git。
