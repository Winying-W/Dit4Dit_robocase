# Atomic18 实验产物索引（失败实验的完整证据）

`atomic18_results.tar.gz`（6.1 MB）包含以下真实产物（从训练/评估输出目录打包，非重算）：

```
results_atomic18/
├── runs/
│   ├── concat_10k_2n/          # concat（随机 head + Qwen），32 卡 world 32，10k→35k 步
│   ├── baseline_10k_32card/    # baseline（GR1 官方 head 微调），32 卡 world 32，0→10k 步
│   ├── baseline_10k/           # baseline，16 卡 world 16，0→8.6k 步（早期版本）
│   └── baseline_gr1/           # GR1 init 参照（官方配方 2 步 run，≈未微调权重）
│       ├── metrics.jsonl            # 每步训练指标（action_loss / video_loss / lr / grad / 显存）
│       ├── run_config.json          # 完整训练配置（模型、数据、schedule hash、base checkpoint）
│       ├── status.json / latest.json
│       ├── action_trainable.json    # 可训练张量清单与形状
│       ├── split.json / normalization.json / dataset_manifest.json
│       └── snapshots.txt            # 该 run 已有的 action_step_*.pt 列表
├── evals/                      # 每个评估目录：
│   ├── success_rate_summary.json    # 18 任务 × 10 seed 汇总（successes / trials / rate）
│   └── <Task>/seed_XXX/
│       ├── evaluation.json          # 单个 rollout 的最终评估记录（成功、步数、动作诊断统计）
│       ├── checkpoint_verification.json  # 该次评估所用 checkpoint 的校验记录（sha256、global_step、protocol、验证损失等）
│       └── inference_summary.json   # 推理摘要（checkpoint、normalization hash、执行 horizon、queries 等）
└── probes/                     # 离线单步动作归因（prediction_metrics）
    ├── probe25k/<Task>/offline_prediction_metrics.json
    └── probe31k/<Task>/offline_prediction_metrics.json
```

评估目录与检查点的对应：

| 评估目录 | 检查点 |
| --- | --- |
| `evals/eval_10k` | `concat_10k_2n/action_step_010000.pt` |
| `evals/eval_15k` | `concat_10k_2n/action_step_015000.pt` |
| `evals/eval_20k` | `concat_10k_2n/action_step_020000.pt` |
| `evals/eval_concat22k` | `action_latest.pt` @ step 22500 |
| `evals/eval_25k` | `concat_10k_2n/action_step_025000.pt` |
| `evals/eval_30k` | `concat_10k_2n/action_step_030000.pt` |
| `evals/eval_31k` | 31k 权重副本 @ step 31250 |
| `evals/eval_5k_probe` | `concat_10k_2n/action_step_005000.pt`（3 任务探针） |
| `evals/eval_baseline18` | `baseline_gr1/action_latest.pt`（GR1 init，3 seed） |
| `evals/eval_baseline` | `baseline_gr1/action_latest.pt`（3 任务，3 seed） |
| `evals/eval_baseline32_5k` | `baseline_10k_32card/action_step_005000.pt` |
| `evals/eval_baseline32` | `baseline_10k_32card/action_step_010000.pt` |
| `evals/eval_baseline5k` | `baseline_10k/action_latest.pt` @ step 5500（3 任务） |

> 大文件（`action_step_*.pt`、`action_latest.pt`、rollout 视频/轨迹）未打包；它们在原始主机上的路径：
> `/mnt/home/jiahaozhe/deepseek-workspace/vla_wam/outputs/{concat_10k_2n,baseline_10k_32card,baseline_10k,baseline_gr1}/`
