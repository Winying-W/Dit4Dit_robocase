# RoboCasa-GR1 evaluation protocol

Sources read on 2026-09-20:
- Paper: https://arxiv.org/html/2603.10448v1 (section 5.1: 24 tasks, 50 rollouts/task, 720 environment steps; reported average 50.8%).
- Released checkpoint and README: https://huggingface.co/mondo-robotics/dit4dit-model/tree/46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44/dit4dit_robocasa_gr1 (README reference 56.7%; five reported runs 56.3–57.4%).
- Upstream batch runner: `examples/Robocasa_tabletop/eval_files/batch_eval_args.sh` (24 task IDs, n_envs=1, action chunk=12, max steps=720, episodes=50).
- Simulator: https://github.com/robocasa/robocasa-gr1-tabletop-tasks at 4840e67 (robosuite 1.5.1, MuJoCo 3.2.6, Gymnasium 0.29.1, NumPy 1.26.4).

No training dataset is needed for inference. Checkpoint contains all learned model parameters, and is loaded using the original strict state-dict loader. The existing local Cosmos scaffold initializes the model before **all** parameters are overwritten from the GR1 checkpoint. Only `framework.cosmos25.base_model` in the downloaded configuration is changed to a local path; original config is retained as `config.upstream.yaml`.

`evaluate.py` reuses the original policy adapter, image crop/resize, sine/cosine state encoding, normalization statistics and MultiStepWrapper. The simulator RNG is initialized with seed 7 per task; reset calls use NumPy seeds 7..56. Policy diffusion sampling remains stochastic, as in the original server. Each task runs in an independent process with one environment. First episode video is retained; all episode outcomes and step counts are written to JSONL. Environment errors abort the run rather than counting as policy failures.

The original evaluator reads only `env_infos['success'][env_idx][0]`, and Gymnasium 0.29's vector autoreset replaces terminal info with next-reset info. Our evaluator captures success from the original wrapper's max-aggregated reward before an explicit reset. We record both `success` (any successful environment step) and `author_success` (the original sampling and terminal-info behavior) so any difference is visible. The task horizon and action processing are unchanged. Preflight reports are explicitly excluded from policy success summaries.

Commands:

```bash
# Validate all 24 scenes: actual EGL reset + 12 hold-position steps each.
GPU_LIST=4 PREFLIGHT_ONLY=1 RUN_DIR="$PWD/results/gr1_preflight_24" bash scripts/robocasa/run.sh
# Local closed-loop check of two tasks, one full episode each.
GPU_LIST=6 TASK_INDICES='2 23' EPISODES=1 USE_BF16=1 RUN_DIR="$PWD/results/gr1_smoke_bf16" bash scripts/robocasa/run.sh
# Submit only after local checks pass.
bash scripts/volc/submit_gr1.sh
```

Cloud output: `results/gr1_volc_20260920/summary.json` and `summary.md`, with per-task directories and logs. YAML mounts the user's vePFS directory and existing NAS, runs the prepared shared Python environments, and requires no download or installation during the queued job.

The pinned official Sketchfab archive has no `book` category although the simulator lists it as an optional distractor. The original simulator emits warnings and skips those optional distractors. The archive was downloaded unchanged; no replacement assets were introduced.
