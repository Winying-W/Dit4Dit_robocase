**Composite16：新增10k步的完整开发评估为0/320，任务宏平均和合并成功率均为0%。**

结果于2026年9月28日06:43 UTC完成独立审计。固定16个Composite seen任务，每任务20个开发场景，种子100–119；全部使用fresh target reset、stable_counter_v1场景协议、官方完整horizon和官方成功判定。动作预测16步、每次执行前8步。此为开发评估，最终16×100次新场景评估尚未执行。

| 任务 | 成功／次数 | 成功率 | Wilson 95%区间 |
| --- | ---: | ---: | ---: |
| DeliverStraw | 0/20 | 0% | 0.00%–16.11% |
| GetToastedBread | 0/20 | 0% | 0.00%–16.11% |
| KettleBoiling | 0/20 | 0% | 0.00%–16.11% |
| LoadDishwasher | 0/20 | 0% | 0.00%–16.11% |
| PackIdenticalLunches | 0/20 | 0% | 0.00%–16.11% |
| PreSoakPan | 0/20 | 0% | 0.00%–16.11% |
| PrepareCoffee | 0/20 | 0% | 0.00%–16.11% |
| RinseSinkBasin | 0/20 | 0% | 0.00%–16.11% |
| ScrubCuttingBoard | 0/20 | 0% | 0.00%–16.11% |
| SearingMeat | 0/20 | 0% | 0.00%–16.11% |
| SetUpCuttingStation | 0/20 | 0% | 0.00%–16.11% |
| StackBowlsCabinet | 0/20 | 0% | 0.00%–16.11% |
| SteamInMicrowave | 0/20 | 0% | 0.00%–16.11% |
| StirVegetables | 0/20 | 0% | 0.00%–16.11% |
| StoreLeftoversInBowl | 0/20 | 0% | 0.00%–16.11% |
| WashLettuce | 0/20 | 0% | 0.00%–16.11% |

合并Wilson 95%区间为0%–1.19%。区间是固定任务下试验层面的二项近似，不代表未见任务的泛化区间。所有320次失败均跑到官方horizon，记录的基础设施失败attempt为0。保存的预测、反归一化、裁剪/阈值、12D排列与仿真实收动作共741,000步通过独立核验。

训练继承GR1视觉/语言基座及Atomic18的50k动作分支，然后在7,271条Composite训练演示、5,403,990训练帧上新增10k步；806条完整验证演示隔离。四卡、有效batch64，训练State Encoder、Action Encoder/Decoder和Action DiT；Video DiT、VAE、Qwen冻结。10k权重已独立EFS备份。

此前Composite新增12步、2k也各为0/320。12步与10k的全部320对初始XML和完整物理状态匹配，配对结果无新增成功、无丢失成功。全零样本下bootstrap差值区间退化为[0,0]，不能据此断言两个策略等效或继续训练没有任何行为变化。三份权重评的是相同开发场景，也不能合并成960个独立新场景。

三个seed100案例已经定位到必要的前期操作未完成：取面包未启动吐司机，取吸管未打开抽屉，搅拌未抓住食材或让食材入锅。搅拌10k虽有约77%的控制步关闭夹爪，末端离两种食材中心最近仍约0.75/0.86米。此类阶段证据支持继续检查接近目标、控制模式和夹爪时机，不能外推为全部失败的唯一原因。

本轮新增检查：三个案例的2k/10k实际首帧RGB、state和语言完全相同；全部8,077条演示的静态相机安装元数据一致，并与三个新场景的安装位置、方向和左右相机视角匹配。腕相机视角未写入这份元数据，不能据此宣称全数据腕相机标定已核完。这些检查也不能排除视觉特征不足或闭环误差积累。额外CPU分割可见性检查的失败尝试保留，尚未据此更改训练或评估。

主线已自动继续向新增25k步训练，之后按计划评估25k/50k。两路单卡采样对照继续从同一Composite10k权重各新增2k步，再比较原定三任务的60对开发场景；它们仍训练全部16任务。下一步依据开发成功率与物理阶段的改善决定是否扩大候选或开展局部视觉适配。原专家GT接触漂移及历史16演示step2000的指定视觉解冻对照仍未完成，当前实验不替代它们。

证据：[逐任务CSV](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite10k_result_20260928/per_task.csv)、[完整结果](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite10k_result_20260928/summary.json)、[独立动作审计](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_composite_observer_20260927/live/development_10000/review_e9df62d53e1ab8b03a43e7d43ab494d24288e178515a8bcdc267c1ca31b5209a.json)、[320对场景比较](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_development_comparison_20260927/composite_000012_010000.json)。
