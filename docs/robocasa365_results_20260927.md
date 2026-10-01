# RoboCasa365：当前结果入口与历史记录

2026-09-27 13:05 UTC 更新：Atomic18 的 1,800 次最终评估已全部完成，279 次成功，合并成功率和任务宏平均均为 **15.50%**（Wilson 95%：13.90%–17.25%）。云任务成功退出，18×100 次及 1,106,180 个控制步已独立审计；没有基础设施失败记录。完整逐任务表、失败分母和范围说明见 [Atomic18 最终报告](robocasa365_atomic18_final_20260927.md)。

Composite 主训练已在四卡运行，核查时新增 907 步；另外四卡已自动接续启动 Composite 第 12 步权重的 320 次开发基线，任务为 `t-20260927210100-n76gv`。基线 CUDA 启动检查通过，尚无完整 Composite 成功率。见 [并行训练](robocasa365_parallel_training_20260927.md)和[基线记录](robocasa365_composite_start12_baseline_20260927.md)。

以下保留截至 11:06 UTC 及更早时点的历史结果与诊断；其中的运行状态和局部试验数不代表当前进度。

## 历史记录

当前结果更新至 2026-09-27 11:06 UTC。Atomic18 已完成 50,000 步训练，当前四卡任务仍为 Running。最终评估已有 1,392/1,800 条单条记录；12 个完整任务共 1,200 次已独立审计，成功 94 次。这是已完成任务集合的结果，全部18任务总体成功率仍待完成。Composite16 尚未提交或训练。

旧 StirVegetables 16-demo step2000 恢复已停止，已从当前执行依赖中移除。现有 10k、25k、50k 权重均存在；先完成当前评估，待四卡释放后推进 Composite。现有 50k 的新视频解冻对照已通过 CPU 控制验收，尚未训练。详见 [新对照验收](robocasa365_atomic50k_ab_controls_20260927.md)。后文保留带时点的历史诊断记录。

Composite 自动衔接的本地 CPU 进程已实际启动并核对存活：原任务结束、完整评估复核和资源释放后才提交，提交不明确时禁止自动重试。11 项测试与真实云端只读检查通过，当前没有新增 GPU 作业。见 [自动衔接记录](robocasa365_composite_handoff_20260927.md)。

当前 10k、25k、50k 权重、GR1 基座和恢复配置已在 EFS 完成 155 文件、约 27 GB 的备份，全部 SHA256 回读一致。三份训练权重已从副本实际 CPU 加载验证。见 [权重备份记录](robocasa365_checkpoint_backup_20260927.md)。

评测范围：当前安装版本的官方数据注册表有 65 个 Atomic 数据集任务，本轮完整覆盖其中官方 `Atomic seen` 目标组的 18 个，另 47 个未训练或评估。`full_official_task_set` 在本轮指完整覆盖所选命名组。后续 Composite 也是官方 `Composite seen` 16 任务组。目录与清单逐项核对见 `runs/robocasa365_task_scope_20260927/coverage.json`。

## 50k 已完成的单任务最终结果

以下各任务完成100个新场景。

| 任务 | 成功 / 试验 | 成功率 | Wilson 95% 区间 | 独立审计控制步 |
| --- | --- | --- | --- | --- |
| CloseBlenderLid | 0/100 | 0% | 0.00%–3.70% | 90,000 |
| CloseFridge | 42/100 | 42% | 32.80%–51.79% | 70,419 |
| CoffeeSetupMug | 5/100 | 5% | 2.15%–11.18% | 58,748 |
| OpenCabinet | 4/100 | 4% | 1.57%–9.84% | 102,876 |
| OpenDrawer | 10/100 | 10% | 5.52%–17.44% | 71,263 |
| PickPlaceCounterToCabinet | 7/100 | 7% | 3.43%–13.75% | 72,016 |
| PickPlaceCounterToStove | 2/100 | 2% | 0.55%–7.00% | 59,444 |
| PickPlaceDrawerToCounter | 1/100 | 1% | 0.18%–5.45% | 74,675 |
| PickPlaceSinkToCounter | 3/100 | 3% | 1.03%–8.45% | 88,762 |
| PickPlaceToasterToCounter | 8/100 | 8% | 4.11%–15.00% | 58,695 |
| TurnOffStove | 2/100 | 2% | 0.55%–7.00% | 74,014 |
| TurnOnSinkFaucet | 10/100 | 10% | 5.52%–17.44% | 57,558 |

当前完整任务集合共 94/1200 成功；这不是完整18任务的总体成功率。不同更新时间纳入的任务集合不同，不能把这些局部合并值当作模型变好或变差的趋势。

证据：`runs/robocasa365_continuation_20260924/observer/evaluation_1800/review_89d6fbc66216f80ce675e4196f83814611dc893ff29a53f1a76fdb2e6b991c27.json`；运行中单条进度：`runs/robocasa365_atomic18_20260924/observations/partial_final_20260927T105402Z.json`。

## 已完成的开发评估

| checkpoint | 成功 / 试验数 | 合并成功率 | 宏平均 | Wilson 95% 区间 |
| --- | --- | --- | --- | --- |
| 10,000 | 11/360 | 3.06% | 3.06% | 1.71%–5.39% |
| 25,000 | 25/360 | 6.94% | 6.94% | 4.75%–10.05% |

每轮 18 个 Atomic seen 任务各 20 个 fresh target 场景，官方完整 horizon 与 success checker。720 次开发结果均通过独立保存动作审计；没有基础设施失败记录。两轮使用相同开发 seed 列表，不是 720 个独立新场景，也不是最终评估。

| 任务 | 10k 成功 / 20 | 25k 成功 / 20 | 25k 成功率 |
| --- | --- | --- | --- |
| CloseBlenderLid | 0 | 0 | 0% |
| CloseFridge | 4 | 3 | 15% |
| CloseToasterOvenDoor | 0 | 0 | 0% |
| CoffeeSetupMug | 0 | 0 | 0% |
| NavigateKitchen | 0 | 0 | 0% |
| OpenCabinet | 0 | 0 | 0% |
| OpenDrawer | 0 | 0 | 0% |
| OpenStandMixerHead | 3 | 14 | 70% |
| PickPlaceCounterToCabinet | 0 | 0 | 0% |
| PickPlaceCounterToStove | 0 | 0 | 0% |
| PickPlaceDrawerToCounter | 0 | 0 | 0% |
| PickPlaceSinkToCounter | 0 | 0 | 0% |
| PickPlaceToasterToCounter | 0 | 0 | 0% |
| SlideDishwasherRack | 2 | 5 | 25% |
| TurnOffStove | 0 | 0 | 0% |
| TurnOnElectricKettle | 2 | 0 | 0% |
| TurnOnMicrowave | 0 | 1 | 5% |
| TurnOnSinkFaucet | 0 | 2 | 10% |

25k 的 13 个任务为零成功，全部 5 个 PickPlace 任务为 0/20。此结果说明整体效果仍弱，不支持直接声称模型已经适配良好。

## 诊断证据与限制

1. 已保存预测 → 训练统计反归一化 → 裁剪/官方二值阈值 → 12D 重排 → simulator 实收命令，在全部 720 次开发试验中通过复核；控制器为 base 参考系的 OSC_POSE delta。这个范围内没有发现动作转换错误，但它不能证明所有观测/视觉适配问题都不存在。
2. 对 360 对相同 seed 额外核对初始场景及完整物理状态，只有 332 对匹配，28 对不匹配。分别的成功率仍是各自 fresh 场景的观测结果，不能把全部 360 对声称为完全相同场景的受控比较。原因待定位；没有用放宽物理状态标准消掉差异。
3. 25k Cabinet seed100 的 750 步保存动作完成诊断重放：初始/最终完整物理状态及 94 个查询 state 均误差 0，XML 完全一致。模型第 132 步触发目标抓取谓词，最长连续 612 步、最多抬起约 45.7cm。第 477–480 步对象满足入柜谓词，但这四步夹爪均未远离对象，最终仍持物且对象已不在柜内。该例失败在完成放置/松手离开阶段，不能说它完全不会抓取；也不能把单例解释外推到全部失败。
4. NavigateKitchen 训练演示 450 条、64,913 帧，其中 91.46% 为底盘模式；25k 的 20 条导航开发轨迹仅 2.14% 控制步启用底盘模式，全部失败。这是不同访问状态分布的描述性比较，提示优先检查模式预测与导航行为，不能独自证明归一化错误或某一网络模块为根因。
5. 四条开发失败的诊断现已全部完成，共 2,700 控制步、339 个查询 state，初始/最终完整物理状态及查询 state 均误差 0。Cabinet seed101/102 均未触发目标抓取谓词；Navigation seed100 仅 30/450 步启用底盘模式，离目标最近仍约 2.29m。连同 seed100 的放置/松手失败，说明失败阶段不同，不能统一归因为完全不会抓取或单一动作转换错误。这 4 条均为原保存动作重放，没有新增策略成功率。

## 权重与正在执行的评估

固定 10k、25k、50k 权重均存在。50k 的 247 个动作参数张量、163,276,320 参数以及优化器/调度器、split、归一化、四 rank CPU RNG 已独立读回；CUDA RNG 仅检查结构。本轮未新增 GPU 恢复测试。
50k 文件：`artifacts/training/robocasa365_atomic18_20260924/action_step_050000.pt`，SHA256 `2cc6fa048869f55d4a5f2e1f7df4419d090dbde2c7ba244e5b0d0110996ae49c`。

截至 09:08 UTC，1,800 次最终评估已有 929 个单条试验记录完成。其中整组完成的 8 个任务共 800 次已独立审计，其余运行中记录不用于发布最终总体成功率。逐任务进度见 `runs/robocasa365_atomic18_20260924/observations/partial_final_20260927T090831Z.json`。

当前四卡全部用于最终评估，没有新排队任务。下一轮根据开发失败诊断决定继续动作训练、数据/采样调整或部分 Video DiT 对照；Composite16 仍待训练和正式评估。原 StirVegetables 16-demo step2000 的三个已知路径复查仍不可访问，原指定起点 A/B 未完成。

证据目录：`runs/robocasa365_atomic18_20260924/diagnosis_20260927/`；开发报告分别为 `development_10000/report.json`、`development_25000/report.json`；独立复核位于 `runs/robocasa365_continuation_20260924/observer/`。

## 新增 CPU 诊断：预测拟合与闭环行为

`cpu_prediction_fit/report.json` 已完成：NavigateKitchen、PickPlaceCounterToCabinet 各取训练/验证两条演示，每条取 10%/50% 位置，共 8 条演示、16 个不同观测窗口。比较 25k/50k、4/16 次积分、两个固定噪声种子，共 128 个动作 chunk。重复噪声和 chunk 内动作不能当独立演示；每个任务/划分实际只有 2 条演示、4 个窗口。冻结 backbone 特征在各对照之间复用；CPU 精度及随机数不等同 CUDA 闭环推理。

以下均为 4 次积分下的小样本离线诊断，不能替代成功率：

| 指标（验证窗口） | 25k | 50k |
| --- | --- | --- |
| 导航底盘模式正类召回 | 92.16% | 96.08% |
| 导航底盘命令前 8 步 MAE | 0.18445 | 0.12451 |
| Cabinet EEF 平移命令 MAE | 0.18389 | 0.15890 |

底盘及 EEF 命令误差使用控制命令单位，不是实际机器人终点误差。这些导航演示观测没有呈现全局底盘模式预测崩溃；Cabinet 所抽窗口的开合标签也能预测，但没有专门覆盖最后松手阶段。增加到 16 次积分没有一致收益。目前证据不足以把后续方案简单定为增加推理步数或调整二值阈值。

进一步读取导航全部 450 条训练演示：前 16 步底盘模式比例 38.81%，全轨迹比例 91.46%；25k 开发评估前 16 步为 3.44%。20 个开发初始状态的每个分量均在训练全集范围内（容差 1e-6）。这排除了这些分量明显超范围的解释，但不能证明观测语义、图像分布或闭环状态分布完全一致。接下来应在开发集上覆盖开局、模式切换和最终松手阶段，分别检查预测与实际行为。证据：`cpu_prediction_fit_review.json`、`navigation_initial_state_comparison.json`。

## 新增 CPU 诊断：相同 seed 场景不一致

发现 `Counter.get_reset_regions()` 用 `list(set(valid_geoms))` 对 XML Element 对象去重；其顺序依赖对象身份，固定 `PYTHONHASHSEED=0` 仍不能保证顺序。候选修正为 `list(dict.fromkeys(valid_geoms))`，保留首次出现顺序和原候选集合。

已执行 5 个代表 task/seed × 原实现/候选修正 × 3 个独立进程，共 30 次初始化。两组均关闭渲染，比较完整物理状态、策略 state、XML 和 episode metadata。原实现的 NavigateKitchen:110、OpenStandMixerHead:107、PickPlaceSinkToCounter:101、TurnOnSinkFaucet:101 均出现跨进程差异；TurnOnElectricKettle:106 此次三次未复现差异。候选修正的五组各三次均逐项一致，状态最大误差为 0。

补丁只存在于独立诊断进程，已核对共享 counter.py 的 SHA256 始终为 `77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a`。当前正式评估及原快照没有改动。第一次开启渲染的 smoke 因 EGL 对空 CUDA_VISIBLE_DEVICES 的解析失败，记录保留于 `reset_probe_smoke/`；随后完全关闭渲染的 v2 smoke 和 30 次对照均成功。候选源码及证据分别在 `reset_probe_source_v2/`、`reset_determinism_v2/report.json`。

这说明候选修正消除了上述测试中的初始化不一致，尚未证明所有 seed 或带渲染场景均可复现，也没有解决旧 GT demo 的接触物理漂移。今后的受控 A/B 仍需核对实际场景身份；若把修正用于新实验，须创建新快照并重新验收，不能沿用当前 Composite 旧版本的 prequeue 结论。

## 分阶段诊断与 Composite 覆盖补充（06:11 UTC）

按 seed78 抽取导航和入柜各训练/验证 4 条演示，共 16 条；固定选择初始帧以及持续至少 4 步的模式切换/最后松手命令。3 条导航演示没有符合该定义的切换，明确保留缺失记录，没有补抽有利样本。共 29 个观测窗口、25k/50k 两个权重、两次噪声重复，得到 116 个预测 chunk。6 项指标和选窗测试通过，所有预测均实际在 CPU 完成。

独立复核只看当前策略实际执行的前 8 步，发现：

| 验证集关键指标 | 25k | 50k | 范围 |
| --- | --- | --- | --- |
| Cabinet 最后松手窗口，open 命令召回 | 8/32（25%） | 11/32（34.38%） | 4 条演示、两次噪声；32 个相关命令位置，不能当 32 条独立演示 |
| Cabinet 初始窗口，6D arm 命令 MAE | 0.25366 | 0.20063 | 4 条演示 |
| 导航初始窗口，6D arm 命令 MAE | 0.13230 | 0.14649 | 4 条演示 |

导航初始窗口的 mode 正类仅来自 1 条验证演示，其 16 个重复/相关位置在两权重下均仅预测对 1 个，不能外推成总体召回率。以上说明中段小窗口的高开合准确率不能代表开局或最终松手能力；更多训练有局部改善，但没有消除全部关键阶段错误。证据：`stage_prediction_fit/report.json`、`stage_prediction_fit_review.json`；源码固定于 `stage_probe_source/`。

对导航全部 450 条训练演示另查：312 条先有手臂准备动作，首次持续正 mode 的中位 step 为 15；这 312 条在切换前的 EEF 位移中位数 0.14875 m，不能简单当作空等待删除。训练轨迹的底盘平移命令平均范数 0.66264，25k 开发轨迹为 0.02054；但这是不同访问状态分布，不是单一根因证明。`HybridMobileBase.set_goal()` 的 mode 主要选择手臂 goal_update_mode 为 desired/achieved，所有 part controller 在两种模式下都接收各自命令；不能把正 mode 比例直接解释为底盘是否能够运动。证据：`navigation_start_transition_statistics.json`、`navigation_continuous_commands.json`、`navigation_controller_mode_semantics.json`。

初始化排序候选修正现已扩展到完整 Composite16，每任务开发 seed107、原实现/修正各 3 个独立无渲染进程，共 96 次。原实现的 KettleBoiling、PreSoakPan、ScrubCuttingBoard、StirVegetables 出现不一致；修正后 16 组均连续 3 次完整物理状态、策略 state、XML 与 metadata 一致，物理状态误差 0。证据：`runs/robocasa365_composite16_4gpu_20260924/reset_determinism_20260927/report.json`。这扩展了任务覆盖，仍只覆盖一个 seed、无渲染初始化；不等同新协议已通过完整 prequeue 或策略成功率。

当前决定：继续完成原版 1,800 次最终评估；未来评估显式采用新版本场景协议并重新验收。后续训练对照优先覆盖开局准备和松手阶段，保留原始二值阈值及完整演示，先用开发场景验证调整收益。Composite16 训练目标保持，等待四卡释放；原16-demo step2000 对照仍不能用 Atomic18 权重代替。


## 场景协议正式接入与 CPU 验收（本轮新增）

评估入口现显式区分原版 `official_fresh_v1` 和保序修正 `stable_counter_v1`。新协议要求进程启动时设置 `PYTHONHASHSEED=0`，核验共享源文件及实际函数哈希，并在 benchmark、模型 worker 和 simulator 之间验证一致性；不同协议禁止混合汇总或在同一目录续跑。修正只在新评估进程内生效，共享仿真库和当前云端 Atomic18 快照未改。

新独立快照已完成 33 项相关测试、4 个生产 CLI 导入检查、16 次真实无渲染 reset；修正后的 8 次场景与此前候选实现的完整物理状态、策略 state、XML 和 metadata 严格一致。另通过真实 `eval_simulator` Unix socket 入口，在 StirVegetables seed107/108 各执行 8 步合成 12D 动作，验证模式/夹爪映射和跨 trial 协议一致。该项使用黑色图像占位，不加载 DiT，不新增策略试验。

验收汇总：`runs/robocasa365_scene_protocol_20260927/cpu_acceptance_summary.json`；真实入口记录：`cpu_ipc_stable/acceptance.json`。新协议仍需带渲染 GPU gate 和新的 Composite prequeue；旧版 prequeue 不能代替新版本验收，旧 GT demo 接触漂移也未证明全部解决。


## Composite16 新版本已通过软件/数据 prequeue

2026-09-27T07:09:05.642558+00:00：新目录 `runs/robocasa365_composite16_4gpu_20260927/` 完成 16 任务共 128 步真实 CPU 环境检查、38 项协议测试、10 项分布式/恢复测试，以及全部 8,077 个演示 parquet 内容复核。新入口把完整 16 任务的真实渲染列为取得四卡后的首项必过检查，随后通过训练/恢复和策略到仿真的硬件检查，才允许进入长训练。仍未提交，训练和评估数均为 0。

Atomic18 当前单条完成量为 548/1,800，已独立审计的完整任务仍为四组共 400 次、49 次成功；不发布不完整的总体成功率。进度证据：`runs/robocasa365_atomic18_20260924/observations/partial_final_20260927T070905Z.json`。新版本的训练计划、证据复用边界、固定底盘适用性和待完成硬件检查见 `docs/robocasa365_composite16_plan_20260927.md`。原 16-demo step2000 的三个已知路径仍不可访问，历史 A/B 保持未完成。


## 输入路径与实际训练覆盖补充

2026-09-27T07:37:04.938594+00:00：34 任务、68 条演示、204 个窗口的训练/评估 state、拼图、语言和 Cosmos 输入一致；这项使用相同录制 RGB，不证明全部 fresh 场景的视觉标定。重建 320 万个实际训练窗口后，四 rank RNG 与 50k checkpoint 精确匹配；全部 8,216 条训练演示被用到，77.04% 的观测起点、99.89% 的不同动作目标帧至少出现一次。详细范围、松手阶段训练/验证差距及 Composite 采样覆盖的解析预估见 `docs/robocasa365_training_diagnosis_20260927.md`。

当前最终评估已有 633/1,800 个单条完成记录，已完整审计的仍为四任务共 400 次、49 次成功；总体成功率留空。证据：`runs/robocasa365_atomic18_20260924/observations/partial_final_20260927T073704Z.json`。当前云作业和 CPU 独立审计进程均已核对仍在运行；无新 GPU 提交，Composite 冻结快照未改。


## 冻结 / 部分解冻对照控制已接入（08:08 UTC）

新单卡 trainer 的 opt-in 随机数/分组裁剪协议、恢复一致性检查及 stable 场景选择检查已完成；真实发布版 GR1 的 CPU 前向/反向控制验证与 10 项测试通过。五个整任务已独立审计共 500 次、59 次成功，最终评估单条进度为 731/1,800。原 16-demo step2000 仍不可访问，指定共同起点的 A/B 未执行，也未宣称解冻能提高成功率。当前 Atomic18 与 Composite16 冻结快照哈希均复核未变。详情见 `docs/robocasa365_paired_ab_20260927.md`。


## 初始化兼容性与 replay 补充诊断（09:00 UTC）

已核对 Atomic/Composite adapter 实际使用的 min/max 兼容，并完成真实 Action DiT 权重迁移及四进程 CPU trainer 导入。两种初始化的 256 次配对离线预测及独立复算已完成，16 个任务所抽验证窗口的 arm MAE 均下降；优先验收 Atomic50k 初始化候选，新的 Composite 作业尚未提交。另确认 XML 再导出会改变物理，但当前 GT reset 不包含该再导出步骤，旧接触漂移仍未全部解决。原 NAS 域名当前无法解析，原 step2000 仍未恢复。详细证据和边界见 `docs/robocasa365_initialization_diagnosis_20260927.md`。


## Atomic 初始化的 Composite16 验收完成

新149文件快照实际通过了完整CPU模型batch前向/反向和四进程trainer导入；247个动作张量均有有限梯度。新初始化的独立验收报告位于 `runs/robocasa365_composite16_atomicinit_4gpu_20260927/prequeue_acceptance.json`，训练和策略试验均仍为0，等待现有四卡评估结束及硬件门槛。计划见 `docs/robocasa365_composite16_atomicinit_plan_20260927.md`。旧NAS的0GPU恢复提交被平台以资源ID不存在拒绝，没有创建任务；不再继续追查旧存储，现有10k/25k/50k权重均已重新确认存在。

## 同模型恢复状态诊断

六个本地快照的30个固定电机命令物理分支已完成并独立复核。缺失warmstart会引入差异，补回后与完整状态精确一致；没有恢复原demo的隐含状态，也没有证明旧GT漂移已修复。详见 [恢复状态对照](robocasa365_restore_state_probe_20260927.md)。当前执行顺序仍为先完成Atomic18评估，再训练及评估Composite16。

## Composite 独立审计进程已就绪

2026-09-27 10:54 UTC：新 CPU observer 已从独立六文件快照启动，并通过进程句柄核对存活。17 项测试、实际 Atomic 144 个离线验证窗口、当前 Composite CPU prequeue 128 个窗口及一条900步保存动作核查均通过。Composite 尚未训练时窗口数报告为 null；不把 prequeue 当正式训练。新增检查拒绝与运行计划不符的场景协议。现有训练快照131/149文件均逐一核对未改。详见 [Composite 审计记录](robocasa365_composite_observer_20260927.md)。

## Composite 训练确认与自动备份

2026-09-27 11:06 UTC，用户明确确认 Atomic50k → Composite16 新增50k四卡训练及最终1,600次评估，沿用当前 active goal。自动接续、独立审计和 checkpoint 备份进程均已实际核对存活。Composite恢复资料154文件已复制到EFS并回读校验；10k/25k/50k权重在产生后自动复制与CPU加载审计，当前尚无Composite权重，不能声称已完成这些备份。见 [自动备份说明](robocasa365_composite_backup_20260927.md)。
