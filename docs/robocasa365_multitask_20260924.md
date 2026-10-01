# RoboCasa365 Composite seen：多任务训练与评估记录

当前状态：**全18 Atomic seen 的四卡 DiT4DiT 训练正在云端运行**，任务 `t-20260924183604-77jsj`，已通过真实四卡更新、独立进程恢复、四策略到仿真动作审计。使用8,216条训练/910条验证，计划50k updates与1,800次最终评估；详见 [Atomic18记录](robocasa365_atomic18_20260924.md)。完整16 Composite已通过四卡方案的排队前软件、数据、环境和完整CPU模型验收，主训练尚未提交，须等当前Atomic结果复核且资源释放；详见 [Composite16记录](robocasa365_composite16_4gpu_20260924.md)。Composite最终评估为每任务100次、共1,600次，旧320次方案仅作历史记录。旧训练step2000仍不可访问，历史A/B未完成，Goal为active。

## 已完成的实际工作

- 官方 `TARGET_TASKS['composite_seen']` 全部 16 个任务；逐个核对下载镜像日期与官方 registry。
- 数据位于项目 `data/robocasa365/v1.0/target/composite/`，使用可用的 vePFS，未依赖失效 NAS。
- 共 **8,077 个 episode、6,002,265 帧**；确定性地按任务划分 **7,271 train / 806 validation**。完整 episode 分割，无训练/验证交叉。
- 检查全部 parquet 的 state16/action12、有限值、原始动作范围、mode/gripper 二值、相机文件和 replay extras。
- 全部 16 个 adapter 读取实际 RGB，并逐行审计全部 train/validation action 的 transform → inverse → robosuite reorder。最大误差 **8.855547228847627e-08**。
- 每个样本：video `[9,3,128,384]`、state `[1,64]`、action/mask `[16,32]`。camera 在图像宽度方向拼接，时间维独立；最后一帧仅 12 个 action 元素有效。
- 归一化只使用 7,271 条训练演示，所有任务共享连续动作 min/max。底盘/躯干真实动作参与拟合；控制模式和 gripper 保留原始 -1/+1。state 不归一化。
- 跨任务 batch、16,000 次任务均衡采样检查通过。8 个针对 split/统计分母/区间/动作转换/控制器的测试通过。

| Task | Train | Validation | Frames | Base-mode frames |
|---|---:|---:|---:|---:|
| DeliverStraw | 454 | 50 | 433307 | 126544 |
| GetToastedBread | 455 | 51 | 654840 | 95504 |
| KettleBoiling | 451 | 50 | 228349 | 667 |
| LoadDishwasher | 451 | 50 | 369430 | 18494 |
| PackIdenticalLunches | 451 | 50 | 719964 | 249021 |
| PreSoakPan | 451 | 50 | 395501 | 4565 |
| PrepareCoffee | 463 | 51 | 279534 | 0 |
| RinseSinkBasin | 458 | 51 | 211036 | 91 |
| ScrubCuttingBoard | 454 | 50 | 228864 | 1832 |
| SearingMeat | 451 | 50 | 439466 | 56957 |
| SetUpCuttingStation | 454 | 50 | 336025 | 63205 |
| StackBowlsCabinet | 463 | 52 | 175620 | 4146 |
| SteamInMicrowave | 460 | 51 | 487012 | 78776 |
| StirVegetables | 451 | 50 | 409202 | 22330 |
| StoreLeftoversInBowl | 453 | 50 | 390829 | 106445 |
| WashLettuce | 451 | 50 | 243286 | 1029 |

Base-mode frames 指原始 action[4] == +1 的帧数，包含 train 和 validation；不是成功数，也不等于位移帧数。

## 原 step2000 与后续过拟合模型的失败诊断

原16-demo StirVegetables step2000 的 fresh seed100 保存动作已完整重放2400步。原始初始XML一致，初始/最终完整MuJoCo state及全部300次策略查询state的最大误差均为0。因此这次诊断对应原失败轨迹，不是新运行的策略试验。

在全部2400步中，官方两个蔬菜及锅铲的抓取判定均未成立，两个蔬菜也均未入锅；锅从第1步已在目标灶台。该案例的首个未完成阶段是**获得物体/抓取**。夹爪首次闭合命令为step321，闭合命令占21.83%；底盘命令始终为0。这不证明唯一网络根因，也不能推广为其余9个fresh场景的诊断。证据及原权重恢复的身份检查见 [原step2000记录](robocasa365_legacy_step2000_20260924.md)。

下面的“先抓到后丢失”属于**后续单demo过拟合模型**，不是原step2000；两者必须分别解读。

对既有 episode5 GT 与 overfit H8 trial000 的保存动作重新 step，同一 Gym、同一 seed100、同一启动修正、同一三相机分辨率。没有中途注入状态，没有修改物理或成功检查器。

两者初始和最终 MuJoCo state 与原轨迹完全一致。策略全部 300 次查询的 state 最大误差也是 0。因此下列诊断对应原来的失败轨迹，而非新产生的物理漂移。

| 事件 | GT | 策略 H8 |
|---|---:|---:|
| 首次 grasp veg1 | 105 | 74 |
| 首次 veg1 入锅 | 118 | 未发生 |
| 首次 grasp veg2 | 189 | 162 |
| 首次 veg2 入锅 | 252 | 未发生 |
| 首次 grasp spatula | 345 | 未发生 |
| 官方首次成功 | 558 | 2400 步内未成功 |

策略 veg1 的 grasp predicate 只在 74–79 步成立，veg2 只在 162–165 步成立；丢失抓取前物体距锅中心水平距离约 37.1 cm / 35.2 cm。丢失抓取附近 gripper 预测仍高于官方闭合阈值 0.5，不能归咎于这些时刻的错误开夹命令。这支持优先检查抓取几何、抬升/搬运的精确控制与反馈。**还不能据此证明具体某个网络模块是根因，也不能说解冻视频必然解决。**

这不消除此前其它 GT episode 的物理漂移；本次只验证上述两条保存轨迹。

## 历史单卡方案（其中原 step2000 A/B 尚待恢复权重）

下列15k/320次及单卡主训练安排已经被用户后续扩数据、四卡、50k/1600次的要求替代。第1–2项原step2000对照仍是未完成事项；不能用新初始化训练冒充该对照。

1. 单卡 A/B：原 16-demo step2000 为共同起点，A 冻结 Video DiT，B 只解冻 Video DiT block16/17；各追加 1000 次更新，accumulation4。B 使用真实 future-video FM loss；保留官方 stop-gradient，不声称 action loss 能穿过 detach 更新视频。
2. 两臂均在 StirVegetables fresh seed100–109 上完整评估。逐 seed 核对初始状态/XML 和实际 simulator 动作。优先成功次数更多的 recipe；次数相同则 B 必须使固定离线验证 FM loss 至少改善超过 5%，否则选冻结。这只是小规模开发集选择。
3. 多任务归一化已变化，主训练从发布的 RoboCasa-GR1 初始化重新适配，**不把固定零底盘统计或单 demo overfit 权重作为多任务续训**。任务均匀采样，再在该任务训练帧内均匀采样。State Encoder、Action Encoder/Decoder、Action DiT 全部训练。
4. 先进行 10-update 数值/梯度/checkpoint gate，然后恢复到总共 **15000 updates、accumulation4**。选 frozen 则全部 action-only；选 partial_joint 则先12000 action-only，再3000 partial-joint。Qwen text encoder 和 VAE 全程冻结。本轮是扩大数据的单卡基线，不声称充分收敛。
5. 固定最终 step15000，**16 个任务 × fresh seed1000–1019 = 320 次**。每任务官方 horizon，官方 success checker，完整12D动作。单卡串行，不使用开发机训练 GPU。
6. 逐任务成功数/分母/Wilson95%区间，宏平均及合并成功率；相同每任务样本量使两个点估计相同。保留基础设施错误，每任务最多一次失败重试；缺任务/缺 seed 时不生成完整 benchmark 成功率。GT、demo-init、开发集 seed100–109 都不计入。

## 历史 NAS 阻塞及初次下载失败

以下是恢复公开GR1之前的记录。现在公开GR1已恢复，新多任务训练能通过vePFS/EFS运行；仍然缺少的是自己训练的旧step2000。

两次提交均在创建训练任务前失败，没有新 task ID、没有占用队列 GPU：

- 原 NAS id `enas-cnbja0e9f52c6189a9`：平台返回 `not found nas`。
- 同一 NAS 地址：平台返回 `not find NasAddr`。

当前开发环境的 `/file_system/nas` 也不存在。项目 `.conda`、`checkpoints`、`results` 指向该盘。

已找到可读备份 `/file_system/efs/checkpoint/intern/haozhe.jia/projects/DiT4DiT/`，其 Python 环境已实际用于本轮 CPU adapter 验收；但该备份只有 LIBERO 初始化，缺少 RoboCasa-GR1 初始化和此前训练的 step2000。对自己备份范围的搜索未找到这些权重。

恢复公开 GR1 权重的尝试也未成功：直连 Hugging Face 网络不可达，文档中的本机 `127.0.0.1:17890` 代理拒绝连接。未用 LIBERO 或随机权重冒充原 GR1 / step2000，也未把重新初始化说成续训。

需要恢复原 NAS 登记/挂载，或提供 **原 GR1 + 原16-demo step2000 及相邻 normalization/config 文件**的可访问位置。已向用户询问存储是否迁移。在此之前不能执行既定 A/B 和正式320次评估。

## 关键产物

所有路径相对于项目根目录。

- `runs/robocasa365_multitask_20260924/progress.json`：机器可读进展与阻塞。
- `.../data_vepfs/datasets.json`：16任务下载/完整性/metadata hashes。
- `.../prepared/manifest.json`、`normalization.json`：不可变 split、训练统计、固定评估 seeds。
- `.../adapter_verification/acceptance.json`：实际6,002,265帧动作链路和 RGB/batch 验收。
- `.../diagnosis/comparison.json`：GT 与策略的逐阶段、grasp interval 对照。
- `.../source/`：A/B 源码快照；`.../source_multitask/`：多任务源码快照；旁边有 SHA256 清单。
- `scripts/volc/robocasa365_ab_data_1gpu.yaml`、`robocasa365_multitask_1gpu.yaml`：尚未成功提交的单卡配置；必须修复存储配置后再提交。

历史原 step2000 的 **0/10 fresh StirVegetables** 和单 demo overfit 的 **0/3 H8 训练场景诊断**，仍不是16任务成功率。本轮新训练更新数0、正式评估次数0，不填造成功率。

## 后续复核：评估恢复与当前阻塞

首次自动续跑再次通过真实任务提交核查存储。平台仍返回 `not find NasAddr`（RequestID `20260924013325B75EB33B91539C4B8A0B`），没有创建新任务。该阻塞已连续出现于两个 goal turn，goal 仍 active。

评估现在固定增量权重、基座权重、两份配置及两份统计文件的 SHA256；同路径同大小的权重被替换也会使恢复校验失败。逐任务评估结果必须与总体 benchmark 的指纹一致。评估持有内核文件锁，并把锁传给模型子进程，父进程退出时仍存活的模型 worker 会继续阻止重复启动。

5 项针对 split、未完成分母、零成功区间、同路径内容替换、实际子进程继承锁的测试通过。最后两项是本次新增检查。源码快照与 hashes 已同步。没有执行模型训练或320次正式评估，训练/策略结果计数仍为0。

第二次自动续跑复核：原 NAS、GR1 初始化与原 step2000 在已知位置和可读备份均不可访问；A/B 与多任务任务在全部云状态下均无匹配任务。保存证据 `final_storage_check.json`。同一外部阻塞连续三个 goal turn，goal 已正式设为 **blocked**。最终320次策略评估仍未执行，需要恢复原存储或提供准确的可访问权重位置后继续。

2026-09-24 恢复后的三轮复核仍受同一存储访问阻塞，goal 再次设为 blocked。云任务提交错误仍为 not find NasAddr，原 NAS id 为 enas-cnbja0e9f52c6189a9；当前证据不证明盘内数据被删除。普通 volc/mlp CLI 未提供 NAS 列表入口，未调用管理员 API。需要存储当前状态/新挂载地址，或原权重的新路径。证据为 resumed_storage_check_3.json。

## 官方初始权重恢复与原子任务优先

通过 Hugging Face 镜像恢复固定 revision 的官方 RoboCasa-GR1 权重，完整 21,283,661,981 bytes 的 SHA256 与原记录一致。所有新模型张量与输出置于 vePFS；Python 环境通过只读 EFS 挂载。原 NAS 的错误没有被当成数据删除证据。

用户要求先检查简单 Atomic task。PickPlaceCounterToCabinet 的 502 条演示已按 452/50 分割并通过 131,904 帧审计；预先选定的 GT episode 0/246/501 均按策略同一 Gym/render 协议成功。云任务 `t-20260924173006-4fr79` 实际为 Running，1×A800，计划 10000 updates 和 100 次独立最终策略评估。详情见 [原子任务记录](robocasa365_atomic_20260924.md)。原子任务结果不能代替 Composite 成功率。

新的完整 Composite 协议位于 `runs/robocasa365_recovered_20260924/prepared/manifest.json`，保留原始训练 split/statistics，最终种子为 1000–1099，共 1600 次。50k 基线配置已准备，等待原子任务诊断后推进。仍未执行原 step2000 的冻结/部分解冻 A/B，不声称旧权重已经找回。

## 用户要求扩数据与四卡后的更新

主线已扩大为全部 18 个 Atomic seen：8,216 条训练 / 910 条验证、2,231,347 帧，四 A800 DDP、global batch64、50k updates，最终 1,800 次独立场景评估。单任务 Cabinet 仅保留为小基线，在四卡提交前保存并停止；不能把它的步数加到新实验。按用户要求先完成排队前验收，到卡还需短 CUDA/NCCL、恢复及四策略仿真检查。详细配置及解释边界见 [四卡 Atomic18 记录](robocasa365_atomic18_20260924.md)。完整 Composite 和原 step2000 对照的未完成事项仍保留。
