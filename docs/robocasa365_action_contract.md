# RoboCasa365 → DiT4DiT 动作接口核查

源码与本任务metadata均确认：手臂为OSC_POSE、delta输入、base参考坐标系。不是absolute末端位姿。旋转为轴角增量，不把3个旋转分量当四元数或绝对欧拉角。

| LeRobot动作切片 | 含义 | 学习时处理 | 发给仿真时处理 |
| --- | --- | --- | --- |
| 0:4 | 底盘3维+躯干1维 | 训练集min/max；选中非移动episode全部0 | 反归一化回完整4维0，不删除 |
| 4:5 | 控制模式 | 保留原始-1/+1，训练子集均-1 | 官方Gym阈值0.5，低于阈值为-1（arm mode） |
| 5:8 | EEF位置delta命令 | 训练集min/max | 先反归一化到原始命令，OSC内部再缩放到每轴±0.05 m |
| 8:11 | EEF旋转delta命令 | 训练集min/max，不改旋转表示 | 先反归一化，OSC内部再缩放到轴角分量±0.5 rad |
| 11:12 | 夹爪命令 | 保留原始-1/+1 | 官方Gym阈值0.5；-1张开，+1闭合 |

连续维的训练归一化：`z=2*(a-min)/(max-min)-1`；反变换`a=(z+1)/2*(max-min)+min`。常数base维训练编码0，解码还原0。mode/gripper不走连续归一化。原始动作虽然在[-1,1]内，旋转各轴的训练实际min/max不全是±1，因此推理必须反归一化，不能直接把模型输出当原始控制器命令。

模型action为16×32，仅前12维有效；其余20维为padding并被训练mask排除。推理裁剪的是反归一化后的原始命令，范围[-1,1]；不先把验证集归一化值裁剪到训练统计范围。手臂的米/弧度缩放只在OSC内执行一次。

直接robosuite向量顺序为EEF position3、rotation3、gripper1、base3、torso1、mode1，对应LeRobot重排`[5,6,7,8,9,10,11,0,1,2,3,4]`。Gym路径按命名action字典传入，由官方wrapper做同一重排及二值化，不再额外重排第二遍。

## 源码、真实控制器和云端验证

已验证：录制metadata与本地PandaOmron默认配置的arm/base/torso参数逐项一致；录制环境实际控制器快照也是delta/base；官方LeRobot `get_episode_actions`默认读取原始action，当前版本对`abs_actions=True`直接抛NotImplementedError。已有GT官方与adapter完整轨迹一致证据，说明那条回放路径没有额外absolute转换。

本地12项检查通过，包含动作映射、100个合成完整12维命令与官方Gym转换的一致性，以及绝对动作模式/错误缩放的拒绝测试。

真实环境补充检查已通过：`verify_action_runtime.py`在开发机CPU上按录制metadata创建了无渲染RoboCasa365环境，验证实际controller、维数、切片、20Hz以及OSC真实缩放函数。测试命令`[1,-1,0.5,0.4,-0.3,0.2]`实际映射为`[0.05,-0.05,0.025,0.2,-0.15,0.1]`，随后通过官方key converter构建完整命令并成功执行一步。控制器仍为delta，arm模式的goal_update_mode为achieved。报告：`runs/robocasa365_dual_20260923/action_contract/controller_runtime.json`。这不是训练策略评估。

云端评估已执行以下强制检查：

1. 使用checkpoint旁的训练归一化，通过实际adapter transform遍历16个训练与4个验证episode的全部action，检查raw→normalize→denormalize误差≤1e-6，检查维数、padding截取和重排。
2. 每个新建Gym环境检查实际控制器input_type、参考系、Hz、input/output limits以及action切片；absolute配置会中止评估。
3. 每一步截获真正进入底层env.step的12维命令，与该步解码动作经官方规则转换的独立预期比较；保存实际收到的仿真动作，避免只检查配置却漏掉运行时错误。

实现：`scripts/robocasa365/action_audit.py`、`eval_protocol.py`、`eval_simulator.py`。结果记录在评估目录的`action_chain_audit.json`、每个trial的`controller_contract.json`及`trajectory.npz`。这些检查不改变当前训练的动作语义，也不声称已经解决剩余开环物理漂移。


## 首次云端真实checkpoint验证已通过

任务`t-20260923163223-f4hcf`的32步smoke已完成。20个训练/验证episode合计15,980帧的实际adapter归一化→反归一化全部通过，最大绝对误差1.0217939094836481e-7；运行时截获的32个底层完整12维动作与解码/官方映射预期一致，实际控制器delta/base、底盘命令保持0，未出现正模式切换。模型预测shape为1×16×32，解码16×12，查询4次。该32步验收不构成完整任务成功率。

证据：`runs/robocasa365_policy_eval_20260923/smoke/action_chain_audit.json`、`smoke/evaluation.json`、`smoke/trial_000/controller_contract.json`、`smoke/trial_000/trajectory.npz`。

运行时模型类为DiT4DiT / FlowmatchingActionHead / CosmosTransformer3DModel。Cosmos内部文本编码器类为Qwen2_5_VLForConditionalGeneration，属于冻结模块；247个训练参数tensor全部位于`action_model.*`。因此“不是在训练Qwen”成立，但不应表述为模型中完全没有Qwen组件。


完整target评估中的Gym观测边界警告已核对：官方wrapper把所有state字段声明为[-1,1]，但base_position是米制世界坐标。seed100首帧base_y=-1.4971114，超出声明范围，原始state均finite，RGB为3×256×256×3 uint8。保持与训练一致的原始state，不把米制坐标裁剪到[-1,1]来消除警告。此警告与动作范围或absolute/delta无关。


## 保存预测的独立复算

`scripts/robocasa365/audit_saved_actions.py`不调用adapter的解码函数，也不调用评估路径的动作转换函数，直接从训练`normalization.json`和`trajectory.npz`中的模型预测重算完整链路：截取前12维、连续维min/max逆变换、执行每个chunk前8步、原始命令裁剪、mode/gripper阈值以及12维重排，再逐步对比底层仿真实际收到的动作。它同时验证query时间索引，避免chunk执行位置错位。

首次覆盖已完成的6条完整target trial，共14,400步；全部通过，prediction→simulator最大误差6.212649239500934e-8。最终17条完整trial已全部通过同一复算，共40,800步，prediction→simulator最大误差6.550286180129206e-8。报告为各阶段的`saved_prediction_audit.json`。这补充了15,980帧GT action归一化往返检查，分别覆盖真实训练数据与真实模型预测。

注意日志中的`clipped_components_in_predicted_chunks`包含mode/gripper及每个16步chunk中未执行的后8步。它不能直接解释为手臂饱和率。独立报告另外计算实际执行的6个手臂分量裁剪比例；初次6条检查各为约0.007%～0.125%。


## 最终验收

评估云任务`t-20260923163223-f4hcf`已Success，退出码0。step2000 fresh target为0/10，step300配对对照0/3，step2000留出非移动初始场景0/4。各组均完成官方2400步horizon，所有实际动作与保存预测的独立解码一致；任务失败不能用接口验收通过来掩盖。三个配对seed的初始state/XML逐元素相同。完整机器证据：`runs/robocasa365_dual_20260923/final/report.json`；[最终说明](robocasa365_dual_run_20260923.md)。
