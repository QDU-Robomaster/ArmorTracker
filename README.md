# ArmorTracker

`ArmorTracker` 是 Webots/Linux 自瞄链路里的目标级跟踪模块。输入来自
`ArmorDetector` 发布的检测结果和 `CameraFrameSync` 同步帧，输出
`tracker/target_frame` 同帧目标包。
云台角、发送包和开火判定由后级 Aimer 负责。

## 文件结构

- `ArmorTracker.hpp`：模块入口、配置、`target_frame` payload 和运行态成员。
- `ArmorTrackerPipeline.hpp`：detector topic 回调、target_frame 发布和内置 preview 绘制。
- `ArmorTrackerCore.hpp`：detector 输入到 tracker 输出的门面适配。
- `ArmorTrackerModel.hpp`：PnP、目标状态、整车 EKF 和跟踪状态机。
- `ArmorTrackerMath.hpp`：角度/坐标转换和 EKF 基础工具。
- `ArmorTrackerTarget.hpp`：`tracker/target_frame` 内携带的目标状态消息。
- `tools/tracker_replay/armor_tracker_replay.cpp`：离线重放一致性检查工具。

`ArmorTracker` 主体是模板头文件实现，CMake 只暴露 include 目录；模块本身不再编译
额外 `.cpp` 源文件。

## 配置

运行配置只保留当前链路实际使用的三组字段：

- `cfg.tracker`：目标过滤、进入跟踪与丢失阈值、输出坐标帧。
- `cfg.extrinsic.camera_to_body`：手眼外参，表达从 OpenCV 相机系 `C` 到公开本体系
  `B` 的变换 `p_B = R_BC * p_C + t_BC`。`rotation` 为 `wxyz` 四元数，
  `translation` 单位为 m。`B` 为右手系，`x` 向右，`y` 向前，`z` 向上。
- `cfg.preview`：`VisionPreview::RuntimeParam`，控制本模块自带预览线程。

默认 detector topic 为 `armor_detector/armors_frame`。tracker 对外只发布
`tracker/target_frame`。

默认 `cfg.tracker.output_frame: 1` 输出右手系：`x` 向右，`y` 向前，`z` 向上；
yaw 以前向为 0，左转为正。`output_frame: 0` 保留 tracker 内部 world frame。

## Preview

内置 preview 只在 `cfg.preview.enabled: true` 时启动，不订阅 topic、不录像、不反压主链路。
它绘制 detector 原始四边形、tracker 整车中心、四个装甲面中心、相邻装甲面连线，以及带固定
pitch 的装甲板物理框。多车跟踪时，preview 会绘制所有 active 车辆，并在车体中心标注编号和
当前选择评分；被选中的车辆用更醒目的中心和连线显示。detector preview 不在这里处理。

## Target Selection

tracker 内部按装甲板编号维护多套车辆 EKF 状态，同一 slot 丢失后不会清空 EKF。
同一帧里出现多个编号时，各编号状态独立更新；
`tracker/target_frame` 中只携带一个当前选择目标。当前选择分数使用装甲板观测数量的低通值、距离、
可打击面积、自旋速度和目标相对当前云台视轴的角度差，并用滞回 margin 避免输出目标抖动。

## 验证

单车过滤回归使用 `tools/tracker_replay/armor_tracker_replay.cpp` 对固定数据集重放；多车目标选择
需要使用不按编号过滤的 replay，确认同帧多编号输入会独立更新各 slot 并只输出当前选中的目标。
BSP 验证需要生成当前 `cfg.tracker` / `cfg.preview` 结构对应的 `User/xrobot_main.hpp` 后再构建
Linux 和 Webots 目标。
