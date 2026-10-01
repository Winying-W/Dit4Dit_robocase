**Composite12/2000：四条早期失败轨迹的阶段诊断**

核验时间：2026-09-28T02:01:56.828425+00:00。固定开发种子100，DeliverStraw与GetToastedBread各在新增12步和新增2000步诊断一次。共11,100控制步、1,388次保存的状态查询；四条轨迹的初始和最终完整物理状态及所有查询状态误差均为0。四例均为原评估失败轨迹的复现，不增加策略评估分母。

| 任务 | Composite新增步数 | 已观测结果 |
| --- | ---: | --- |
| DeliverStraw | 12 | 抽屉几乎保持关闭；没有抓到吸管，也没有入杯。 |
| DeliverStraw | 2000 | 抽屉几乎保持关闭；没有抓到吸管，也没有入杯。 |
| GetToastedBread | 12 | 没有启动带面包的吐司机；没有抓到面包，也没有放入盘子。 |
| GetToastedBread | 2000 | 没有启动带面包的吐司机；没有抓到面包，也没有放入盘子。 |

对Composite2000取烤面包案例，两个拨杆全程读数为0；官方启动阈值为0.9。机械臂末端相对初始位置最大偏移约0.781米，但手到面包中心距离由约0.380米增至最终约1.123米，面包位置只有仿真静置尺度的微小变化。直接失败条件是没有压动拨杆；为何策略没有学出有效动作仍未确定。

这四例支持“必要的前期步骤未完成”，不能说明两个权重等价，不能将所有Composite失败归为同一根因，也不能判断增加训练步数或解冻Video DiT必然有效。后续10k同场景比较仍需独立完成。

诊断从fresh target reset开始，以stable_counter_v1固定场景；逐步核对仿真实收动作。使用隔离Mesa llvmpipe CPU渲染。取烤面包reset遗留GL_INVALID_VALUE已记录，元数据查询不产生新错误；本诊断不比较图像像素，也不证明新场景相机标定一致。

原软件诊断第一次运行的进度文件保留失败状态，因为其取烤面包查询失败；其中两条DeliverStraw完整轨迹已经通过。随后独立toaster目录的两条取烤面包轨迹补齐。最终四例完成状态以本独立核验为准，失败尝试没有被覆盖。

这是当前Atomic50k初始化的多任务权重，不能替代历史16演示StirVegetables step2000的指定起点A/B，也不能证明旧GT接触漂移已经解决。StirVegetables当前多任务诊断另外报告。

四例独立证据：[evidence.json](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_failure_evidence_20260928/evidence_20260928T020156Z.json)。
拨杆及运动量化：[toaster2000_direct_failure.json](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_failure_evidence_20260928/toaster2000_direct_failure.json)。
