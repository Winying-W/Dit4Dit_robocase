**Composite 早期失败阶段诊断进度**

核验时间：2026-09-28T01:54:37.258099+00:00。以下是当前多任务权重在固定开发种子100的保存策略轨迹诊断，不新增策略评估次数。新增12步与新增2000步均以Atomic50k初始化。

| 任务 | Composite新增步数 | 轨迹长度 | 状态查询核对 | 已观测到的失败阶段 |
| --- | ---: | ---: | ---: | --- |
| DeliverStraw | 2000 | 2550 | 319 | 抽屉开度约为0，未抓到吸管、未入杯 |
| GetToastedBread | 2000 | 3000 | 375 | 未启动带面包的吐司机，未抓到面包、未放入盘子 |
| DeliverStraw | 12 | 2550 | 319 | 抽屉开度约为0，未抓到吸管、未入杯 |

这三条轨迹的初始和最终完整物理状态，以及全部保存的查询状态，均与原策略评估零误差一致；实际传入仿真的动作在诊断中逐步核对。它们支持对这些具体轨迹的阶段描述，不能外推全部失败，也没有证明训练步数、视觉特征或任务难度中的某一项是唯一根因。

取烤面包的新增12步案例仍由原诊断进程执行。另已单独声明并启动当前两份多任务权重的StirVegetables案例：该任务先要求两种蔬菜入锅，再抓锅铲搅拌，官方成功时间计数至少为5。诊断只读取官方维护的计数和物体条件，不额外调用会修改历史位置的_detect_stirring。当前StirVegetables结果尚未完成，不能当作已定位其失败。

CPU诊断使用项目内隔离的Mesa llvmpipe。取烤面包reset遗留的GL_INVALID_VALUE被单独记录，renderer查询后无新GL错误；本诊断没有比较渲染像素，因此不声明图像一致。

此前204个真实录制窗口的训练/评估输入构造一致性已经完成，本轮复核了同一相机顺序、拼图和state路径；未重复运行该试验，也未把录制帧的构造一致性当作新场景相机标定证明。已有证据见docs/robocasa365_training_diagnosis_20260927.md。

三例独立复核：[evidence.json](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_failure_evidence_20260928/evidence_20260928T015158Z.json)。
当前StirVegetables计划：[plan.json](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_stir_failure_diagnosis_20260928/plan.json)。

这些当前多任务案例与历史16演示StirVegetables step2000不同；历史指定起点的冻结/部分解冻Video DiT对照、旧GT接触漂移仍未证明完成。
