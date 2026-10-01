# RoboCasa365 GT action replay 定位与修复

本轮确认了漏掉采集初始化步骤的时间错位，并把回放改为每条演示新建固定seed环境。剩余开环状态偏离仍存在，不能声称所有GT演示精确重现；DiT4DiT模型未参与本轮实验。

## 确认的问题

采集代码 `robocasa/scripts/collect_demos.py` 在 DataCollectionWrapper 保存初始状态之后，调用一次 `env.env.step(zero_action)`，明确不记录该动作。该零动作包含control_mode=0。随后才记录真实演示动作。

本任务的原始MuJoCo时间是：

```text
states[0] = 0.50 s
states[1] = 0.60 s
states[2] = 0.65 s
states[3] = 0.70 s
```

控制频率20Hz，每步0.05秒。直接从states[0]执行actions[0]只会走到0.55秒，相对states[1]整整少了一步。原版官方playback和我们的旧脚本均存在此错位。

修复：完整恢复初始XML、metadata和state后，检查仿真时间戳。只有首间隔=2个控制周期、后续均=1个周期时，补一次采集时的全零动作；正常时间序列不补，未知间隔报错。之后只执行GT动作，绝不逐帧注入录制状态。LeRobot帧时间戳不能代替extras里的MuJoCo时间戳。

实现：[replay_utils.py](scripts/robocasa365/replay_utils.py)、[replay.py](scripts/robocasa365/replay.py)。`--startup none`保留旧流程用于对照。

另外，原脚本跨episode复用环境；本轮发现复用同一个环境并再次恢复同一episode，也能产生极小的首步数值差，长程接触后明显放大。最终脚本每条演示使用fresh environment和显式seed，避免运行顺序影响。没有改变物理参数或成功阈值。

## 已完成的对照

环境：MuJoCo3.3.1、robosuite1.5.2、NumPy2.2.5；前两者与数据metadata一致。源代码提交：RoboCasa `921c9a5`，robosuite `5ce6643`。GT action与extras state均float64，官方动作重排和adapter映射完全相同。

在episode0上，官方回放与旧adapter路径的完整903步状态数组逐元素完全相同，排除了这两条调用路径的差异。

| episode | fresh环境的官方原流程 | 补回采集初始化 | 补步后首个state L2误差 |
| --- | --- | --- | ---: |
| 0 | 失败 | 失败 | 0.000506 |
| 3 | 失败 | 失败 | 1.60e-14 |
| 5 | 成功 | 成功 | 0.535187 |
| 10 | 成功 | 成功 | 3.40e-9 |
| 11 | 成功 | 成功 | 6.194e-01 |

两组均3/5；这是5条有意选取的GT回放诊断样本，**不是策略成功率，也不是任务的总体回放成功率估计**。此前环境复用的旧结果是2/5，不能用它证明补步提升了成功率。

补步后5条演示的时间偏差均消除。episode0首步L2误差由0.433091降为0.000506，最早剩余偏差主要在oil bottle等接触物体；到后期蔬菜、锅和锅铲出现厘米级位置差。L2混合位置、旋转和速度，本身没有“厘米”的含义；诊断脚本另存了每个free joint的米制位置误差。

进一步前15步实验：模拟采集设备的动作转换调用结果与补步基线相同；关闭Numba JIT或额外调用sim.forward均未消除偏差。没有将这些无明确收益的改动合入回放。

## 怎样解释剩余偏差

episode0/3/11的完整903/1014/984步fresh环境回放，在两个独立云进程间逐元素完全一致；同一env实例反复reset的首步误差分别为3.25e-13和7.44e-14，长程差异可达13.47和4.75（混合状态L∞，不是米）。episode11复用环境后的首步最大速度分量差达到0.617，主要来自锅铲，任务结果由成功变为失败。这是使用fresh环境的实测依据，不能把相同seed的reset误认为恢复了所有隐式状态。

上游已有同类问题。维护者Abhiram824在[issue209的回复](https://github.com/robocasa/robocasa/issues/209#issuecomment-4896068511)明确表示：

> open-loop action replay is very fragile [and] is not expected to always result in a success.

这支持“GT开环不保证100%”这一限制，不足以单独证明我们两条失败轨迹的全部微观原因。对0、3、11独立进行的完整录制状态回放，均通过当前官方成功判定；success_time分别为6、6、5。新建环境的跨进程完整轨迹一致性也已通过。状态回放只用于诊断成功判定，不计入GT动作回放的成功数。

不能通过降低成功阈值、改质量/摩擦或逐帧纠正状态，来把GT回放变成成功。也不应仅凭开环回放失败就删除这些训练演示。

## 复现与证据

在已分配且挂载NAS的云任务内运行：

```bash
source scripts/robocasa365/env.sh
"$SIM365_PYTHON" scripts/robocasa365/replay.py \
  --dataset "$DATASET365" --episodes 0 3 5 10 11 \
  --output runs/my_gt_replay --no-video
```

需要视频时移除`--no-video`。正式策略评估仍使用官方环境的正常reset/step；这个补步逻辑专用于恢复录制演示，不盲目添加到Gym reset后。

- [官方与adapter对照](runs/robocasa365_replay_diagnose_20260923/summary.json)
- [补步与官方对照](runs/robocasa365_replay_warmup_20260923/summary.json)
- [前15步诊断](runs/robocasa365_replay_probe_20260923/normal/summary.json)
- [关闭JIT对照](runs/robocasa365_replay_probe_20260923/no_jit/summary.json)
- [环境复用及成功判定检查](runs/robocasa365_replay_contract_20260923/contract.json)
- [单元回归检查](scripts/robocasa365/tests/test_replay_timing.py)：7项通过，覆盖漏步、正常记录、异常间隔和只执行一次零动作。

诊断任务 `t-20260923095406-k5tm4` 在得到episode0的官方/adapter完整一致结果后主动取消，停止无必要的float64重复组。补步对照任务 `t-20260923095816-zkcl4`、短诊断任务 `t-20260923101111-gs6f5` 均Success。

原始diagnose JSON有一个命名限制：warmup组的`initial_full_state_max_error`是在执行初始化零动作之后与states[0]比较，包含了预期的初始化运动，不代表restore错误。真实restore误差见每组`initial.json`里的`restored_state_max_error=0`。最终replay入口已在初始化之前记录`initial_full_state_max_error`，消除了该歧义。

环境复用/成功判定任务 `t-20260923101757-hb8dt` 已Success。记录：[跨进程完整轨迹对比](runs/robocasa365_replay_contract_20260923/fresh_repeat_comparison.json)。

## 修复后的验收边界

已消除的是已确认的时间错位，以及回放入口对跨episode环境复用的依赖。基准对照中的5条轨迹全部时间对齐；失败轨迹0/3仍如实保留。已检查的0/3/11原始录制状态均可通过官方成功判定，同一seed的新建环境完整回放在独立进程间完全一致。

尚不能声称已解释或消除了跨录制环境的所有微观数值差异。现有证据不支持更改物理参数、动作语义或任务成功标准。后续DiT4DiT闭环评估应固定环境创建协议、seed、horizon和action chunk执行方式，单独统计模型成功率，并保留GT回放诊断作为环境说明。

## 渲染协议也必须固定

最终入口在headless模式复测episode0/11，时间误差均为0，末态误差与之前的fresh环境诊断一致。开启三相机离屏渲染后，episode11仍成功，但末态相对录制状态的L2从1.302409变为3.137417，success_time从5变为7。因此不能把headless和rendered回放要求为同一条数值轨迹。

源码中离屏渲染上下文创建会额外调用`sim.forward()`（robosuite `binding_utils.py` 的 `MjRenderContext.__init__`），渲染开关确实改变初始化调用路径。这里不把它等同于全部微观误差的因果证明。

任务`t-20260923102744-dmkfw`完成了两条headless回放和一条视频回放，但在错误地要求跨渲染模式轨迹一致时断言失败。保留原结果，追加相同渲染协议下的独立重复验证，而非删除失败或放宽物理/成功标准。策略评估必须固定相机设置、渲染开关和观测更新时机。

最终同协议渲染复核任务 `t-20260923103347-htq89` 已Success。episode11两次独立视频回放的逐步误差记录完全一致，成功判定均通过（success_time=7），视频492帧、256×768。初始RGB与数据集图像MAE约1.2–2.0；视频像素本身未要求逐位一致。

- [最终机器可读报告](runs/robocasa365_replay_fixed_20260923/diagnosis_report.json)
- [入口验证结果](runs/robocasa365_replay_fixed_20260923/verification.json)
- [最终云任务状态](runs/robocasa365_replay_fixed_20260923/render_status.json)
- [成功GT视频：episode11](runs/robocasa365_replay_fixed_20260923/video_repeat/replay_000011.mp4)

本轮完成了已确认接口问题的修复和复现协议验证，**并未消除episode0/3相对录制轨迹的全部开环偏差**。这些限制保留在报告和JSON中，不能将本轮结论写成“GT replay全部通过”。


## 双线推进：两条失败轨迹的具体失效环节（2026-09-23）

本轮在单卡训练任务的 CPU 进程中，对此前保存的完整 GT action 轨迹做逐步位置误差分析，并在独立新环境中检查录制状态与回放状态快照对应的官方任务条件。该状态注入仅用于诊断，**不是新增的 GT action replay，也不是策略评估**。原始动作、物理参数和成功判定均未改动。

| episode | 首个任务物体位置误差 >1 mm | >1 cm | 具体失败环节 |
| --- | --- | --- | --- |
| 0 | 第104步，veg1 | 第180步，veg1 | 锅铲抓取仍成立，但后续锅铲与veg1接触显著减少，有效搅拌只计1次，未达5次 |
| 3 | 第98步，veg2 | 第198步，veg2 | 锅铲第539步偏移超过1 mm，第544步超过1 cm；回放轨迹未形成官方锅铲抓取条件，搅拌计数0 |

在episode0中，录制状态的锅铲抓取条件成立273帧，GT回放快照也是273帧；锅铲与veg1接触从23帧变成4帧。录制轨迹6次有效搅拌中的第794～797步，回放里锅铲均未接触veg1。episode3中，录制轨迹有362帧锅铲抓取成立，回放快照为0帧。独立快照检查的累计搅拌计数分别为1、0，与原始动作回放一致；录制状态均为6。

这些证据把失败定位到接触与抓取路径，**仍不足以确定最初微小数值差的全部来源**，也没有消除真实GT开环漂移。应保留这两条失败结果和原始训练演示，不调低成功阈值、不插入GT状态纠正动作回放。

证据：[双线诊断报告](runs/robocasa365_dual_20260923/replay/report.json)，诊断脚本：[analyze_replay_failure.py](scripts/robocasa365/analyze_replay_failure.py)。训练与后续模型闭环评估可继续，正式策略采用固定的fresh Gym reset与RGB观测协议，单独统计成功率。
