# RoboCasa365 双线运行最终报告（2026-09-23）

单卡DiT4DiT训练与全部闭环评估已完成，训练和评估云任务均为Success。数据/模型/仿真动作链路通过核验，但本轮没有得到成功的策略rollout；不能把接口跑通或loss下降称为学会任务。GT replay已修复确认的时间错位，剩余物理漂移仍存在。

## 完整闭环结果

| checkpoint | 初始场景 | 样本 | 成功 | 说明 |
| --- | --- | ---: | ---: | --- |
| step2000 | 官方fresh Gym target，seed100～109 | 10 | 0/10 | 新场景未按固定底盘可完成性筛选 |
| step300 | 官方fresh Gym target，seed100～102 | 3 | 0/3 | 与step2000的前三条配对 |
| step2000 | 留出非移动demo初始场景157/386/480/489 | 4 | 0/4 | 只恢复初始场景，随后模型独立闭环 |

任务为StirVegetables / composite_seen / target human。17条试验均跑满官方horizon2400，20Hz；每次预测16步、执行8步，再获取新观测。使用官方成功判定。32步smoke及GT replay不计入上表，不合并两种初始场景分布来报告一个总体成功率。

三个配对seed的初始完整仿真state和XML完全一致。seed100首个策略查询的RGB、state、指令也完全一致，两份checkpoint的预测不同。step2000的smoke与正式seed100 trial前32步动作、前4个预测chunk和query state完全一致；该复现结论仅覆盖这32步。

这些是小样本诊断，不是365任务benchmark。0/10的Wilson 95%区间约为0～27.8%；0/3约0～56.1%；0/4约0～49.0%，也不把人为选定初始场景视为整体任务分布的随机样本。两份checkpoint在共同三个seed上均0/3，未观察到闭环成功率改善。

## 训练证据

- 训练任务`t-20260923152847-l2xf9`：Success，3730秒，1×A800 80GB，峰值分配显存22.75GiB。
- 从冻结骨干分支step300继续到step2000，未接续历史部分解冻step400。训练16条、验证4条，原有split和训练归一化保持完全一致。
- 247个可训练tensor全部为`action_model.*`，共163,276,320参数，包含状态编码器。Cosmos、VAE与内部Qwen2.5-VL文本编码器冻结。
- 保留AdamW和RNG；续训LR从1e-5起，20步升至3e-5，再余弦降至3e-6。每次优化更新累积4个microbatch。
- 8个固定验证窗口的action FM loss由step300的0.1815487165降至step2000的0.1621078234；独立进程恢复逐值复现最终loss。此离线指标不是闭环成功率。

最新checkpoint：`results/robocasa365_dual_20260923/train/action_step_002000.pt`，另有`action_latest.pt`及500/1000步快照。这是增量checkpoint：需加载原始GR1基座，再加载`trained_state`，并使用邻接的`normalization.json`与`data_config.yaml`。NAS权重需要在有对应挂载的环境中读取。

## Action类型与整条链路

原始12维顺序为base/torso4、mode1、EEF position3、EEF rotation3、gripper1；完整保留。控制器为OSC_POSE、delta输入、base参考系，旋转为轴角旋转向量增量。训练连续动作做训练集min/max归一化；推理反归一化回原始命令，再裁剪至[-1,1]，最后由官方Gym二值化mode/gripper并重排到robosuite顺序。OSC内部只做一次物理缩放：位置每轴±0.05m、旋转分量±0.5rad。

20条训练/验证episode共15,980帧，通过实际adapter的raw→normalize→denormalize检查，最大误差1.0217939094836481e-7。所有17条完整策略试验共40,800步，运行时截获真正进入底层env.step的动作逐步验证；另外从保存的模型预测独立复算，不调用adapter解码函数，prediction→simulator最大误差6.550286180129206e-8。未发现delta/absolute混用、重复缩放、维度错序或chunk时间错位。

错误注入验证能够拒绝漏掉旋转反归一化、重复物理缩放、错序、错时序和absolute控制器。视频首尾可解码、尺寸256×768，每条601帧。原始state保持训练时的米制坐标；官方Gym把state空间统一声明为[-1,1]产生的观测警告不通过裁剪state来掩盖。详见[动作接口报告](robocasa365_action_contract.md)。

## GT replay剩余问题

采集时漏记的一次初始化零动作造成0.05秒错位，已修复；每条独立创建环境并固定渲染协议。episode0/3剩余失败定位到接触与抓取：episode0锅铲与veg1接触减少，有效搅拌计数1；episode3未形成锅铲抓取条件，计数0。原始录制状态对应计数均6，官方阈值5。

这定位了失败环节，尚未完全解释或消除最初微小数值差。没有改变物理参数或成功标准，也没有给策略注入GT动作/后续状态。诊断用状态注入独立标注，不计入策略成功。见[GT replay诊断](../ROBOCASA365_REPLAY_DIAGNOSIS.md)。

## 文件与后续训练

- [最终机器报告](../runs/robocasa365_dual_20260923/final/report.json)：训练恢复、split、归一化、三组评估、逐步动作、视频和配对检查。
- [新场景结果](../runs/robocasa365_policy_eval_20260923/step2000_target/evaluation.json)、[step300对照](../runs/robocasa365_policy_eval_20260923/step300_target/evaluation.json)、[非移动留出结果](../runs/robocasa365_policy_eval_20260923/step2000_heldout_nonmobile/evaluation.json)。各目录保留checkpoint核查、独立动作复算、每条trajectory.npz和rollout.mp4。
- [新场景seed100视频](../runs/robocasa365_policy_eval_20260923/step2000_target/trial_000/rollout.mp4)、[留出episode157视频](../runs/robocasa365_policy_eval_20260923/step2000_heldout_nonmobile/trial_000/rollout.mp4)。
- 评估任务`t-20260923163223-f4hcf`：Success，4372秒，退出码0；与训练任务串行使用一张GPU。入口为`scripts/volc/entrypoint_robocasa365_policy_eval.sh`，配置为`scripts/volc/robocasa365_policy_eval_1gpu.yaml`。

下一轮优先做单演示过拟合，并报告手臂/夹爪的真实推理误差及训练初始场景闭环结果，再扩大固定底盘数据或做视频解冻对照。具体关卡见[训练方案](robocasa365_training_plan.md)。本轮未启动额外训练，不把延长训练或解冻当作保证成功的措施。
