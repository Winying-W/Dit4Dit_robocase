**RoboCasa365 Atomic18：DiT4DiT 50k 最终闭环评估**

核验时间：2026-09-27T13:02:19.916734+00:00。全部 18 个任务各 100 次，共 1,800 次；成功 279 次，失败 1521 次。合并成功率和任务宏平均均为 **15.50%**，Wilson 95% 区间为 13.90%–17.25%。

使用固定 Atomic50k 权重、fresh target reset、最终种子 1000–1099、官方完整 horizon 和官方成功判定，动作执行时域为 8。云任务成功退出，完整评估及保存动作链均通过独立审计。

| 任务 | 成功 / 次数 | 成功率 | Wilson 95% 区间 |
| --- | --- | --- | --- |
| CloseBlenderLid | 0/100 | 0% | 0.00%–3.70% |
| CloseFridge | 42/100 | 42% | 32.80%–51.79% |
| CloseToasterOvenDoor | 12/100 | 12% | 7.00%–19.81% |
| CoffeeSetupMug | 5/100 | 5% | 2.15%–11.18% |
| NavigateKitchen | 1/100 | 1% | 0.18%–5.45% |
| OpenCabinet | 4/100 | 4% | 1.57%–9.84% |
| OpenDrawer | 10/100 | 10% | 5.52%–17.44% |
| OpenStandMixerHead | 75/100 | 75% | 65.70%–82.45% |
| PickPlaceCounterToCabinet | 7/100 | 7% | 3.43%–13.75% |
| PickPlaceCounterToStove | 2/100 | 2% | 0.55%–7.00% |
| PickPlaceDrawerToCounter | 1/100 | 1% | 0.18%–5.45% |
| PickPlaceSinkToCounter | 3/100 | 3% | 1.03%–8.45% |
| PickPlaceToasterToCounter | 8/100 | 8% | 4.11%–15.00% |
| SlideDishwasherRack | 15/100 | 15% | 9.31%–23.28% |
| TurnOffStove | 2/100 | 2% | 0.55%–7.00% |
| TurnOnElectricKettle | 38/100 | 38% | 29.10%–47.79% |
| TurnOnMicrowave | 44/100 | 44% | 34.67%–53.77% |
| TurnOnSinkFaucet | 10/100 | 10% | 5.52%–17.44% |

基础设施未完成/失败 attempt 记录：0。策略失败结束类型：{'official_horizon_exhausted': 1521}。已独立核查 1,106,180 个控制步，包含预测、反归一化、裁剪/阈值、12D 排列和仿真实收指令。

18 个任务是官方 Atomic seen 组，不是所有原子任务或全部 365 个任务。单任务的 42% 或 75% 不能代表总体率。整体表现仍偏弱，动作链核验不能独自证明所有观测接口无误，也不能把失败统一归为训练步数不足。旧 GT demo 接触漂移仍未证明全部解决。

本轮 Atomic 使用 official_fresh_v1；后续 Composite 使用 stable_counter_v1，分别报告。置信区间采用试验层面的二项近似，不代表未评估任务的泛化区间。后续调参使用开发场景；若使用这些最终结果进行新版本选择，需另设新的最终场景。

固定权重：`/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/artifacts/training/robocasa365_atomic18_20260924/action_step_050000.pt`。SHA256：`2cc6fa048869f55d4a5f2e1f7df4419d090dbde2c7ba244e5b0d0110996ae49c`。10k、25k、50k 与基座的独立 EFS 备份记录见 [备份验收](robocasa365_checkpoint_backup_20260927.md)。

Composite 主线继续按已批准方案执行：从 Atomic50k 初始化，四卡新增 50k 步；10k/25k 各 320 次开发评估，50k 做 1,600 次最终评估。此次 Atomic 最终结果不计入 Composite 成功率。

完整分母与逐任务证据：`/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_atomic18_20260924/final_result_20260927/summary.json`。独立审计：`/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_continuation_20260924/observer/evaluation_1800/review_7080de6df002915b79c0eec86bc2d82d5a23c30b365acc8831ac29b254b3c8ac.json`。
