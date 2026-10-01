# 原 StirVegetables step2000：诊断与恢复准备

原模型使用16条StirVegetables训练演示，历史fresh目标场景结果为0/10。旧NAS上的训练权重尚不可读；当前Cabinet的step2000和后续单demo过拟合权重均不能替代它。新的Atomic18四卡训练从已恢复的官方GR1初始化，独立推进。

## seed100 首次失败定位

使用原策略保存的2400步动作，在原生Gym fresh target seed100中重放，不注入demo状态、不补demo启动零动作。初始XML完全相同，初始/最终完整MuJoCo state及300次策略查询state最大误差全部为0。这说明下列谓词对应原来的失败轨迹；没有新增策略评估样本。

- 2400步内，两个蔬菜和锅铲的官方抓取判定从未成立。
- 两个蔬菜均未入锅；锅从第1步起就在目标灶台，结束时仍在。
- 夹爪首次闭合命令在step321，闭合命令占21.83%。底盘命令绝对值最大为0，base mode命令占比为0。

该案例的首个未完成阶段是获得物体/抓取，尚未推进到入锅和搅拌。这与后续单demo过拟合模型在episode5场景中短暂抓住蔬菜、随后丢失的诊断不同。现有证据不确定唯一网络模块根因，不能声称解冻视频必然解决，也不代表其他9个fresh场景。

证据：[完整报告](../runs/robocasa365_legacy_step2000_20260924/first_failure_report.json)、[逐步谓词](../runs/robocasa365_legacy_step2000_20260924/first_failure/predicates.json)、[查询状态误差](../runs/robocasa365_legacy_step2000_20260924/first_failure/query_errors.json)。

## 恢复前可完成的检查

已从当前可读的官方数据重新生成原16条train/4条validation划分及训练集归一化。划分与原记录逐值一致，train共12,786帧；normalization SHA256为`77d32db8bb1b30e24ec7680585858118e4bda30a6f3102eb2392ef7d1011408a`，与原文件一致。证据：[metadata reconstruction](../runs/robocasa365_legacy_step2000_20260924/metadata_reconstruction/acceptance.json)。这恢复的是数据配置，不是训练权重。

[权重恢复检查器](../scripts/robocasa365/stage_legacy_ab_checkpoint.py)要求原action phase step2000、原split/lineage、原GR1引用、247个动作张量名字/形状及原normalization全部一致。真实Cabinet step2000已作为负例执行，正确因split/lineage不同拒绝，见[负例结果](../runs/robocasa365_legacy_step2000_20260924/negative_identity_check.json)。

原文件恢复后，检查器会核验当前官方GR1的完整SHA256，只重定位增量checkpoint的base路径，并对训练张量生成完整digest、写盘读回核验。真实原文件尚缺，因此正例staging与原模型A/B尚未执行，不能标记为通过。旧NAS版A/B入口仍不能直接提交。

已新增使用可用vePFS/EFS挂载的[单卡入口草案](../scripts/volc/entrypoint_robocasa365_legacy_ab.sh)与[YAML草案](../scripts/volc/robocasa365_legacy_ab_1gpu.yaml)，独立输出目录为`runs/robocasa365_legacy_ab_20260924`。没有提交云任务，也没有生成complete的prequeue报告或冒充恢复原权重。实际通过bash语法、镜像/挂载比对及缺少原输入的执行拒绝检查，见[草案检查结果](../runs/robocasa365_legacy_ab_20260924/draft_validation.json)。

草案要求两臂各先做10次真实更新、独立进程恢复至12次，以及独立8步策略到仿真动作检查；全部通过后各追加至本轮1000 updates，再按原配对seeds100–109评估。短检查种子99999不计入对照成功率。已完成评估经验证后可复用，未完成评估目录不得自动覆盖。尚须恢复原checkpoint、固定源码快照、完成全部排队前验收，并确认现有四卡资源已经释放，才能提交。具体待办保存在[计划](../runs/robocasa365_legacy_ab_20260924/plan.json)。

后续受控对照保持同一原step2000起点、同一16-demo归一化、每臂追加1000 updates、accumulation4、paired fresh seeds100–109。A冻结视频，B只解冻Video DiT blocks16/17并使用真实video FM loss，保留官方stop-gradient。不得在当前四卡Atomic18占用资源时另行提交GPU任务。

## 部分解冻方案的实际 CPU 梯度验收

已用恢复的官方GR1初始化、原16-demo训练归一化及真实episode5第100步的9帧视频，执行完整DiT4DiT forward，并分别对action loss和future-video loss反向传播。使用当前单卡trainer的`configure(model, 'partial_joint')`，没有执行optimizer update，也没有使用缺失的旧step2000。

| 梯度来源 | 有梯度的模块 | 张量数 | 参数量 | 梯度范数 |
|---|---|---:|---:|---:|
| action loss | 全部action_model | 247 | 163,276,320 | 7.50788 |
| future-video loss | Video DiT block16 | 20 | 69,206,528 | 0.12926 |
| future-video loss | Video DiT block17 | 20 | 69,206,528 | 0.14063 |

上述张量均无缺失或非有限梯度。action loss没有给视频参数产生梯度；future-video loss没有给动作参数产生梯度。其余Video DiT、文本编码器及VAE参数保持冻结。这是对公开实现stop-gradient行为的实际检查，不能把论文的联合训练描述解读为当前代码的action loss直接更新视频。

CPU BF16下本batch的action loss为0.59765625，future-video loss为0.156570822；这些数值只证明损失有限，不是成功率。该检查只覆盖架构、真实数据、训练参数范围及梯度路径；旧checkpoint身份、CUDA显存、实际更新/恢复和A/B闭环结果仍需各自验收。正在运行的Atomic18源码快照与训练配置没有修改。

证据：[CPU完整报告](../runs/robocasa365_legacy_step2000_20260924/partial_video_cpu/acceptance.json)、[执行日志](../runs/robocasa365_legacy_step2000_20260924/partial_video_cpu.log)、[检查脚本](../scripts/robocasa365/verify_partial_video_cpu.py)。

原GT episode0/3的开环接触漂移仍是单独未完成项；这次保存策略动作的精确重放没有证明所有GT演示都能成功回放。

已扩展到其余 seeds101–109 的历史动作诊断，仅使用本地 CPU，不重新运行策略模型，不增加成功率分母。执行器为 `runs/robocasa365_legacy_step2000_20260924/diagnose_remaining_trials.py`，实时报告为 `remaining_failures/report.json`。首先完整验收 seed101，再以两个 CPU 进程继续其余场景；每条轨迹必须核对全部 2400 步长度、300 次查询状态和初始/最终完整物理状态，才能解释其谓词。

新增诊断曾因 CPU 渲染器不接受空的 `CUDA_VISIBLE_DEVICES` 在环境创建时失败，随后又被过严的 XML 文本相等检查拦下；相关日志和当时源码保留在 `remaining_failures_attempts/`。seed101 的 XML 差异最终定位为 76 个 OBJ 网格是否显式写出默认 `content_type="model/obj"`，没有其他属性或树结构差异。两份 XML 在 MuJoCo 3.3.1 下编译后，522 个模型数值数组/标量、字符串及 opt/stat 字段完全一致。新比较器仅忽略这个可由 `.obj` 文件后缀推断的默认标注，仍拒绝网格文件/缩放、物体位置、仿真时间步长和树结构变化；6 项针对性测试通过。证据为 `scene_identity_acceptance.json`。此修正只用于本地诊断，没有修改运行中的训练或评估源码快照。

seed101 随后已从原 fresh Gym seed 完整重放 2400 步，未注入任何保存状态或演示状态；初始完整 state、全部 300 次查询 state 和最终完整 state 的最大误差均为 0。两个蔬菜和锅铲的抓取判定全程未成立，两蔬菜均未入锅，锅自第 1 步就在目标灶台。它与已诊断的 seed100 一致，首个未完成阶段仍是获得物体/抓取；这个单例本身不能代表其余场景或确定唯一网络根因。

2026-09-24 13:56 UTC，**原 seeds100–109 全部 10 条失败轨迹的诊断汇总完成**，见 [逐场景完整报告](../runs/robocasa365_legacy_step2000_20260924/historical_failure_summary.json)。seed100 复用先前证据，新增完成其余 9 条保存动作重放；共 24,000 个控制步，20 个初始/最终完整物理状态以及 3,000 个查询时刻的观测 state 均与原记录逐值一致。未注入演示状态，未恢复旧策略权重，未新增策略推理或试验分母；原历史成功率仍为 **0/10**。

| 原场景 seed | 蔬菜1最长连续抓取步数 | 蔬菜2最长连续抓取步数 | 锅铲最长连续抓取步数 | 任一蔬菜曾入锅 |
|---|---:|---:|---:|---|
| 100 | 0 | 0 | 0 | 否 |
| 101 | 0 | 0 | 0 | 否 |
| 102 | 1 | 0 | 0 | 否 |
| 103 | 0 | 0 | 0 | 否 |
| 104 | 0 | 0 | 0 | 否 |
| 105 | 6 | 0 | 0 | 否 |
| 106 | 0 | 0 | 0 | 否 |
| 107 | 0 | 0 | 689 | 否 |
| 108 | 0 | 0 | 0 | 否 |
| 109 | 0 | 0 | 0 | 否 |

8 个场景从未触发蔬菜抓取判定；另外 2 个只短暂触发，seed102 共 1 步、seed105 累计 14 步且最长连续 6 步。10 个场景均未完成任何蔬菜入锅。seed107 能持续抓住锅铲，因此不能概括为“该模型完全不会抓物体”；这批失败主要停在蔬菜抓取及入锅阶段，尚未验证后续搅拌能力。谓词只能定位行为失败，不能确定 State Encoder、Action DiT 或 Video DiT 中哪一个模块是唯一根因。

seed107 的第一次额外诊断出现初始 state 最大差异 1.177021233359726，检查拒绝继续，未当成策略结果；单独初始状态复查与随后串行完整重放均精确匹配。失败目录 `remaining_failures/trial_007/`、初始状态复查 `seed107_initial_probe/` 和成功重放 `remaining_failures_retries/seed107_attempt_1/` 全部保留。首次不一致的原因尚未确定，不能声称 fresh reset 在所有执行条件下都绝对可复现；这也没有解决原 GT episode0/3 的接触漂移。

已在 `diagnosis_source_snapshot/` 留存 7 个对应的诊断入口、辅助代码、汇总器和针对性测试文件，逐文件 SHA256 与生成上述结果时的记录一致。清单为 [源码留存 manifest](../runs/robocasa365_legacy_step2000_20260924/diagnosis_source_snapshot/manifest.json)。这是供复核的源码快照，运行环境、仿真资产和输入轨迹仍引用各自记录的位置；没有改变云端运行中的源码快照。
