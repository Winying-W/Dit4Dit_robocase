# DiT4DiT：18 个 Atomic seen 任务四卡实验

用户要求扩大数据、使用四张卡，并在排队前完成可执行的软件验收。本轮使用全部 18 个 Atomic seen 的官方 target human 演示，从官方 RoboCasa-GR1 DiT4DiT 权重重新初始化。机器人及动作 schema 为 RoboCasa365 PandaOmron；没有将 GR1 数据冒充 RoboCasa365，也没有使用 LIBERO 权重。

## 数据和训练

- 9,126 条完整演示、2,231,347 帧；固定 episode 划分为 **8,216 train / 910 validation**，分别为 2,007,192 / 224,155 帧。归一化仅拟合训练集。这里不是 10,000 条单任务 MimicGen 数据。
- 任务均匀采样，再在任务内均匀采样训练帧。所有训练演示可被采样，验证演示排除在训练及统计之外。离线验证 loss 使用每任务 4 条验证演示的 8 个固定窗口，共 144 个窗口；它不是全部验证集或成功率。
- 保留 state16/action12，包含真实 base、torso、mode 命令；不因任务名称而删除底盘维度。控制为 OSC_POSE delta，旋转为轴角增量。全数据归一化往返最大误差为 `8.855547228847627e-08`。
- **4 × A800 80GB，单机四进程 DDP，microbatch4 × accumulation4 × world4 = global batch64，50,000 optimizer updates**。1000-step warmup，AdamW 峰值 1e-4，cosine 降至 1e-5。
- 训练全部 `action_model.*`：State Encoder、Action Encoder/Decoder、Action DiT，共 247 个参数张量、163,276,320 个参数。Cosmos Video DiT、VAE、Qwen 文本编码器冻结。后续解冻视频的实验需单独记录。
- 各 rank 采用独立采样/RNG 流。累积梯度按全部 rank、全部 microbatch 的有效 action 元素总数归一化，避免各窗口 padding 不同导致加权错误。保存 optimizer、scheduler、各 rank RNG/sampler；恢复时严格检查 split、统计及学习率/batch/world size 配置。
- 共享归一化发生变化，不能从单任务 Cabinet 权重直接续训。本轮从已核验 SHA256 的官方 GR1 新初始化；不是旧 StirVegetables step2000 的延续。

## 排队前和到卡后的验收

已有真实 CPU batch4 完整 Cosmos/Action forward 与 backward：247 个动作分支张量均有有限梯度，State Encoder、Action Encoder、Action Decoder 梯度均非零；18 个环境均通过 Gym reset、8 次完整 12D step、controller 和三相机录像检查。四进程 Gloo 数值测试证明 masked accumulation 与拼接全 batch 的梯度一致，并验证 AdamW/RNG 恢复。9 项 CPU 测试和真实运行环境的命令入口均须通过。

证据、源码快照 hash、数据/环境报告、模型报告、挂载及启动配置汇总到 `runs/robocasa365_atomic18_20260924/prequeue_acceptance.json`。必须先生成 complete 验收，再提交四卡。只读 EFS 运行环境、镜像及挂载沿用已经实际运行的单卡任务；不挂载失效旧 NAS。

实际 CUDA/NCCL 和显存不能由 CPU 测试证明。到卡先运行 10 updates，验证有限梯度和真实参数更新、四 rank 权重完全一致、checkpoint 写回；另起 torchrun 恢复至 step12。随后四个 GPU 策略/仿真进程各运行最多 8 步，独立审计预测→反归一化→实际 simulator action。任一检查失败即退出，不进入长训练；各短检查设 1200 秒超时。短轨迹独立存放，不混入成功率。

## 评估协议

- step10k、25k：每任务 20 个开发场景，seeds100–119，每次共 360 trials。
- 固定 step50k：每任务 100 个独立最终场景，seeds1000–1099，共 **1,800 trials**。
- 使用 fresh Gym target reset、完整官方 horizon、官方 success checker。四 GPU 独立评估，每张卡同时一个策略进程。真实动作保存后独立审计。
- 报告逐任务分母、成功率及 Wilson 95% 区间，另报宏平均/合并成功率。基础设施失败单列，未完成不能声称完整成功率。GT、训练场景和截短接口测试均排除。

## 任务与后续目标

CloseBlenderLid、CloseFridge、CloseToasterOvenDoor、CoffeeSetupMug、NavigateKitchen、OpenCabinet、OpenDrawer、OpenStandMixerHead、PickPlaceCounterToCabinet、PickPlaceCounterToStove、PickPlaceDrawerToCounter、PickPlaceSinkToCounter、PickPlaceToasterToCounter、SlideDishwasherRack、TurnOffStove、TurnOnElectricKettle、TurnOnMicrowave、TurnOnSinkFaucet。

原单任务 Cabinet 实验在切换四卡前保存已完成 checkpoint 并停下，实际停止步数见其 `transition_to_atomic18.json`。Atomic18 是新实验，不能将前者步数累计到本轮。

后续仍需全 16 个 Composite seen：7,271 train / 806 validation，最终 16×100=1,600 trials；Atomic 成功率不能替代 Composite。旧 StirVegetables step2000 仍不可访问，原冻结/部分解冻对照尚未完成；旧部分 GT replay 漂移也不能因为 Cabinet 的三条 GT 成功而宣称全部解决。

运行状态以本轮 `submission.json`、云端任务状态、`stage.txt`、`train_live/` 为准。训练、权重和评估结果写入 vePFS，可按 checkpoint 恢复。

## 实际提交

排队前汇总验收于 2026-09-24 10:22 UTC 完成；冻结快照 131 个文件，9 项测试通过。实际云任务为 **`t-20260924182441-p2qs6`**；平台回读确认 4×NVIDIA A800-SXM4-80GB、32 vCPU、512 GiB、单 worker。提交后首先观察到 Staging；这不表示硬件短验收或 50k 训练已经完成。旧单卡任务先确认 Killed，随后才提交四卡，没有同时占用五张卡。

该首次四卡任务随后在短验收入口失败，完成 updates 为 **0**。根因是 `baseframework` 导入日志器时调用 `Accelerate.PartialState()`，在 torchrun 环境中已经建立进程组；新 trainer 又调用 `init_process_group`，产生重复初始化错误。此前独立 Gloo 数值测试及普通 `--help` 没覆盖这条实际导入链，属于排队前检查遗漏。失败日志、配置和源码完整保留在 `attempts/01_import_group_failure/`，没有删除或记为硬件通过。

修复采用复用并核验已存在进程组，同时添加 **真实训练入口的四进程 torchrun CPU 验收**，验证四 rank 导入初始化、重复调用安全及 collective 通信。此报告成为 prequeue 必过项；修正版任务后缀为 `-r2`，实际提交号以当前 `submission.json` 为准。未看到实际 GPU gate 通过之前，不能声称四卡训练已跑通。

修正版还明确清理独立评估 worker 继承的 `RANK/WORLD_SIZE/MASTER_PORT/TORCHELASTIC_*` 环境变量，避免四个独立策略意外创建同一进程组。新增该项测试后共 **10 项 CPU 测试通过**，实际 torchrun 导入验收及 prequeue 重新通过。第二次提交任务号为 **`t-20260924183604-77jsj`**，源码仍为固定快照，第一份失败证据保持归档。

修正版的实际 **四卡短验收全部通过**：10 次真实更新后保存，独立 torchrun 恢复至 step12；step1、10、11 的四 rank 权重 SHA 完全一致，全部动作张量有梯度，State/Action Encoder 和 Action Decoder 梯度均非零，checkpoint 读回成功。峰值已分配显存约 24.44 GiB/卡，短测更新约 4.4 秒/步。随后四个 GPU 策略分别在 Cabinet、NavigateKitchen、OpenDrawer、TurnOnSinkFaucet 中完成 8 步真实闭环，保存预测与实际 simulator 命令的独立审计全部通过。32 个短测 step 排除在成功率之外。

汇总证据为 `runs/robocasa365_atomic18_20260924/cloud_short_acceptance.json`；训练/恢复详见 `four_gpu_training_acceptance.json`，仿真详见 `evaluation_smoke/acceptance.json`。任务已自动进入 `training_to_10000_of_50000` 阶段；后续进度以 `train_live/status.json` 和 `latest.json` 为准。按当前短测速度，纯训练约 61 小时，仅作初步估计，不含正式评估和后续速度波动。

另增加只读CPU保存检查，逐张量读取全部247个训练权重和Adam一、二阶矩，检查优化器与scheduler步数、原split/统计、四个rank的随机状态，并记录checkpoint完整SHA256。step13的真实checkpoint已经通过，证据位于`checkpoint_audits/step_000013.json`；这不表示实时训练步数已经全部落盘。只读观察进程等待下一次常规保存（step1000），通过后将在同目录写对应step报告。观察进程不会提交、停止或重启GPU任务；CUDA恢复能力仍以此前独立torchrun实测为准。

2026-09-24 11:59 UTC，**step1000常规checkpoint已实际保存并通过全部张量/优化器/随机状态的独立CPU读回**，证据为`checkpoint_audits/step_001000.json`。文件为1,959,697,139 bytes，SHA256为`48761780fd42d055a56896db9c73ec6bb9500753b48d96ac61956aa163719645`。观察进程已正常退出，四卡训练继续。这里核验的是当时的`action_latest.pt`；后续常规保存会滚动替换该路径，报告记录的是被观察版本，不能用其旧hash证明未来版本。

同一固定144个验证窗口的action FM loss从初始化step0的2.39194890下降到step1000的0.10061010。它覆盖声明的离线窗口，不是910条验证演示的全帧评估，更不是闭环成功率。正式开发评估仍为step10k/25k各360次，最终评估为step50k共1800次。

可用`scripts/robocasa365/observe_cloud_run.py --task-id t-20260924183604-77jsj --run runs/robocasa365_atomic18_20260924 --training-output artifacts/training/robocasa365_atomic18_20260924 --update-progress`继续观察。它每次先查询真实平台任务，再读取当前阶段、实时/落盘步数以及各开发/最终评估摘要，并保存带时间戳的证据。已在实际Running任务上执行验证。训练阶段使用`global_step`，训练完成状态使用`global_steps`；交接/完成状态没有loss字段时记录为该阶段未提供，不推断NaN。该脚本只写本地观察/进度文件，不提交、停止、重启云任务，不改变Goal生命周期状态，也不把开发样本与最终样本合并为成功率。

2026-09-24 13:13 UTC，**本轮 Atomic18 的 step2000 checkpoint 已通过独立 CPU 读回**：全部 247 个训练张量、163,276,320 个参数及 Adam 一、二阶矩均有限，optimizer/scheduler 步数均为 2000，split 和归一化一致；四个 rank 的 CPU 随机状态可恢复、sampler 状态独立，CUDA RNG 缓冲只检查结构。证据为 `checkpoint_audits/step_002000.json`，汇总为 `checkpoint_audits/milestone_002000.json`。文件为 1,959,697,139 bytes，SHA256 为 `5a6bab071c4b2e60706006d6ff8c828d4dbeb790914182aaa0db90ba92271675`；这是当时滚动文件 `action_latest.pt` 的版本，未新增不可变的 step2000 权重副本。

同一固定 144 个验证窗口的 action FM loss 为 **0.08790791**，step1000 为 0.10061010；前 2000 条训练指标记录连续，loss、梯度范数和用时均有限。该检查没有进行额外 GPU 恢复或策略评估，也不是原 StirVegetables 16-demo step2000 的恢复。13:14 UTC 平台仍为 Running，已到 2042 updates；开发/最终策略试验尚未开始，完整目标保持进行中。

随后已增加 step2000 固定版本留存：`artifacts/checkpoint_archive/robocasa365_atomic18_20260924/step_002000/action_step_002000.pt`，并保留归一化、data config、run config、split、manifest、参数清单及源码身份等配套文件。独立复制后的权重 SHA256 与上述已审计版本完全一致，全部 11 个文件的大小和校验值见 `checkpoint_audits/retained_step_002000.json`。后续滚动保存不会覆盖该目录；留存位于同一 vePFS，不能称为独立存储的容灾备份，也不替代缺失的原 StirVegetables 权重。13:20 UTC 核实云任务仍在运行，已到 2122 updates；本次留存未修改训练源码或占用额外 GPU。

2026-09-24 14:26 UTC，**step3000 常规保存已通过独立 CPU 读回**：全部 247 个训练张量及 Adam 动量有限，优化器/调度器步数、split、归一化和四 rank 随机状态检查通过；没有另做 GPU 恢复或策略试验。证据为 `checkpoint_audits/step_003000.json` 和 `checkpoint_audits/milestone_003000.json`。此次滚动文件为 1,959,697,139 bytes，SHA256 为 `9cbbebe3225f4f6215fee03e93d6e013ab95fdbff7ef41b07f5014fb3928dca5`；未新增固定 step3000 权重副本，已有 step2000 留存保持不变。

同一固定 144 个验证窗口的 loss 为 step0 **2.39194890**、step1000 **0.10061010**、step2000 **0.08790791**、step3000 **0.07733860**；这仍不是完整验证集或闭环成功率。前 3000 条训练记录连续且数值有限。14:27 UTC 平台核实仍为 Running，实时 3039 updates、已保存 3000；step10k 的 360 次开发评估尚未开始，完整目标保持进行中。

2026-09-24 16:14 UTC，**step4000 常规保存通过独立 CPU 读回**：全部 247 个训练张量及 Adam 动量有限，optimizer/scheduler 步数、split、归一化和四 rank CPU 随机状态检查通过；CUDA RNG 缓冲仅检查结构。证据为 `checkpoint_audits/step_004000.json` 和 `checkpoint_audits/milestone_004000.json`。此次滚动权重为 1,959,697,139 bytes，SHA256 为 `d580f183a3d5fbcf75bf9023d150abe9f4187961519cab42c857aca6087542c8`；未增加固定 step4000 权重副本。

step4000 的固定 144 窗口验证 loss 为 **0.07334490**，前 4000 条训练指标连续且数值有限。16:16 UTC 平台仍为 Running，实时 4517 updates，保存点为 4000；开发及最终闭环评估尚未开始。16:14 UTC 读取的最近 500 步平均用时为 4.3827 秒/update，纯训练预计约 6.7 小时后到 10k；此估计不包括后续验证、保存、切换及完整 360 次开发评估的耗时。当前没有本轮正式策略成功率，离线 loss 不能代替它。

16:40 UTC，按用户要求已将 step4000 另行固定留存至 `artifacts/checkpoint_archive/robocasa365_atomic18_20260924/step_004000/action_step_004000.pt`，配套保存配置、归一化、split、manifest、参数清单、trainer 源码及源码身份等共 12 个文件。复制后的 SHA256 与原审计版本相同，副本已 CPU 读回，优化器/调度器步数及四 rank RNG 检查通过。证据为 `checkpoint_audits/retained_step_004000.json`；保存于同一 vePFS，现有 step2000 留存保留。

16:55 UTC，云任务仍为 Running，实时 5058 updates，常规 step5000 已保存。用户新增的持续训练/评估/诊断要求见 `docs/robocasa365_continuation_20260924.md`。新增 CPU 观察/复核脚本通过 9 项协议测试、真实短轨迹拒绝检查及固定源码单次运行检查，观察进程已启动并核实新鲜心跳；它等待 10k/25k/50k 的真实报告，不产生策略试验、不占额外 GPU、不更改运行中的训练快照。结果和进程证据位于 `runs/robocasa365_continuation_20260924/`。

16:56 UTC，step5000 的全部动作参数、Adam 动量、优化器/调度器步数、split/归一化和四 rank CPU 随机状态已通过独立 CPU 审计；滚动权重 SHA256 为 `746f87dd4a91a2f27dcecaf350e50f329893da9659eddb169c5a57bf9e5c3986`，见 `checkpoint_audits/step_005000.json` 和 `checkpoint_audits/milestone_005000.json`。固定 144 窗口验证 loss 为 **0.07554504**，比 step4000 的 0.07334490 小幅回升；这一次测量不能确定过拟合，也不能解释尚未测得的闭环成功率。没有新增固定 step5000 副本或策略评估。
