**相机检查已核验静态安装和三例初始模型输入；没有证明目标始终可见。**

全部8,077条Composite演示（7,271训练、806验证）的相机安装元数据只有一个配置；三个开发新场景的位置、方向和左右60度FOV与之匹配。元数据未记录腕相机FOV，因此不能声称全数据腕相机视角已核完。三任务2k与10k保存的首帧RGB、state和语言完全一致。

StirVegetables seed100的几何投影显示，两种食材中心位于左相机左侧边缘，锅铲中心在画外；右相机与腕相机中的三个目标中心均在画外。这只描述物体中心，不能推出物体完全不可见、必须移动底盘或所有失败都源于视觉。

CPU分割渲染触发GL_INVALID_FRAMEBUFFER_OPERATION(1286)，所有失败尝试和日志均保留，分割mask像素计数被弃用。几何中心投影独立保留；标注图使用原GPU RGB，没有使用失败分割图。专家GT接触漂移仍是另外尚未彻底解决的问题。

[相机元数据报告](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_live_camera_audit_20260928/camera_metadata_report.json) · [几何投影与边界](/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT/runs/robocasa365_live_camera_audit_20260928/geometry_review.json)
