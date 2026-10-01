**Composite25k取吸管的同场景诊断已完成：仍未打开抽屉、未抓住吸管。取面包也完成完整重放，仍未启动吐司机或抓住面包。**

取吸管使用预先固定的开发seed100，与Composite新增12、2000、10000步的同一场景比较。四份场景XML及完整初态一致；25k重放2550个控制步，初态、终态和319次保存查询状态均与原评估零误差一致。保存动作的反归一化、维度顺序及仿真实收动作独立审计通过。这里的25k是Atomic50k之后新增的Composite训练步数。

| Composite新增步数 | 抓住吸管步数 | 抽屉开度最大值 | 末端到吸管中心最小距离 | 关闭夹爪命令占比 |
| --- | ---: | ---: | ---: | ---: |
| 12 | 0 | 1.03e-07 | 0.571 m | 0.04% |
| 2,000 | 0 | 1.03e-07 | 0.587 m | 46.43% |
| 10,000 | 0 | 1.03e-07 | 0.307 m | 78.90% |
| 25,000 | 0 | 1.03e-07 | 0.275 m | 17.80% |

四次抽屉开度都仅约1e-7，吸管均未入杯。25k轨迹末端曾到距吸管中心约0.275米的位置，但未完成开抽屉或抓取。该案例卡在前期操作，不能仅用后续长流程组合解释。夹爪命令占比随检查点变化，也不能当成抓取成功或训练改善的证据。距离是物体中心距离；抓取和入杯使用官方仿真物理判断。

此结论仅针对一个配对开发场景。25k取吸管的完整20次开发评估为0/20，但尚未逐条做相同语义诊断，不能据此断言20条都具有同一失败原因。原专家GT接触漂移尚未彻底解决；策略保存动作的精确重放不能替代该问题修复。

取面包采用相同seed100及匹配的早期/25k初态，完整3000步重放及375次保存查询状态全部零误差复现。独立复算确认两个拨杆最大读数均为0，没有启动带面包的吐司机，没有抓住面包、没有放入盘子；25k的末端到面包中心最小距离约0.284米。Composite12、2k、10k、25k四条同场景轨迹都没有完成这些必要步骤。

两例共新增5550步诊断重放、694次查询状态核验，初态/终态/查询状态均与原评估完全一致。它们都是已保存动作重放，不增加策略成功率分母，不调用本地GPU，也不能单独确定视觉条件、动作拟合或训练预算中的哪一项是根因。

截至2026-10-01 15:11 UTC，25k开发评估已独立审计203/320次，成功1次；全部203次动作转换通过。GPU运行总表按任务组完成重新发布，当前部分总表仅含20次，旧171次记录没有丢失。逐次审计按任务/种子核对所有完整记录，完整16任务宏平均、合并率和总体区间仍待320次齐全后报告。

两路Composite25k视觉候选预排队验收及备份已完成；空闲卡观察器每60秒查询一次，仅在资源可用时调用冻结提交助手。第一路实际分配后才考虑第二路，总项目GPU不超过6；提交意图不明或任务失败则停止自动提交并要求复核。9项独立控制检查及真实无提交扫描通过。截至2026-10-01 15:24 UTC，空闲卡观察器已提交冻结路`t-20261001231446-k5fwd`，平台Running。已完成独立恢复到第4步并保存权重，正在执行实际策略/仿真门槛。联合路仍等待空闲卡。

[取吸管独立复算](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_failure_diagnosis_20261001/DeliverStraw_independent_comparison.json) · [本轮逐次评估审计](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_recovery_20261001/ipc_fix/observations/live_trials_20261001T151152Z.json) · [诊断进程计划](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_failure_diagnosis_20261001/plan.json) · [空闲卡观察状态](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_visual_precision_matched_20261001/capacity_observer/status.json)

两例完成记录、源代码、独立复算及日志共34文件已复制到独立EFS并逐文件读回校验。初次归档因EFS不支持硬链接发布而失败，失败证据保留；更换为独占新目录内的临时文件校验后rename发布完成。该问题仅影响这份诊断归档，模型权重备份目标为vePFS。

[取面包独立复算](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_failure_diagnosis_20261001/GetToastedBread_independent_comparison.json) · [完整诊断归档](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_failure_diagnosis_20261001/complete_archive_receipt.json) · [视觉任务实际云状态](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite25k_visual_precision_matched_20261001/observations/cloud_20261001T152459Z.json)
