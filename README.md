# ArmorTracker

`ArmorTracker` 是 Webots/Linux 自瞄链路里的目标级跟踪模块。输入来自
`ArmorDetector` 发布的检测结果和 `CameraFrameSync` 同步帧，输出
`tracker/target`、`tracker/ekf_points`、`tracker/info` 和轻量候选调试信息。
云台角、发送包和开火判定由后级 Aimer 负责。

## 文件结构

- `ArmorTracker.hpp`：模块入口、配置、topic payload 和运行态成员。
- `ArmorTrackerPipeline.hpp`：detector topic 回调、消息发布和内置 preview 绘制。
- `ArmorTrackerCore.hpp`：detector 输入到 tracker 输出的门面适配。
- `ArmorTrackerModel.hpp`：PnP、目标状态、整车 EKF 和跟踪状态机。
- `ArmorTrackerMath.hpp`：角度/坐标转换和 EKF 基础工具。
- `ArmorTrackerTarget.hpp`：`tracker/target` 发布的目标状态消息。
- `tools/tracker_replay/armor_tracker_replay.cpp`：离线重放一致性检查工具。

`ArmorTracker` 主体是模板头文件实现，CMake 只暴露 include 目录；模块本身不再编译
额外 `.cpp` 源文件。

## 配置

运行配置只保留当前链路实际使用的三组字段：

- `cfg.tracker`：颜色/目标过滤、进入跟踪与丢失阈值、输出坐标帧。
- `cfg.overlay`：内置 preview 的装甲板物理尺寸和固定安装 pitch。
- `cfg.preview`：`VisionPreview::RuntimeParam`，控制本模块自带预览线程。

默认 detector topic 为 `armor_detector/armors_frame`。tracker 输出 topic 位于
`tracker` 域。

## Preview

内置 preview 只在 `cfg.preview.enabled: true` 时启动，不订阅 topic、不录像、不反压主链路。
它绘制 detector 原始四边形、tracker 整车中心、四个装甲面中心、相邻装甲面连线，以及带固定
pitch 的装甲板物理框。detector preview 不在这里处理。

## 验证

行为一致性使用 `tools/tracker_replay/armor_tracker_replay.cpp` 对固定数据集重放，并和锁定参考
TSV 做逐字段比较。BSP 验证需要生成当前 `cfg.tracker` / `cfg.overlay` 结构对应的
`User/xrobot_main.hpp` 后再构建 Linux 和 Webots 目标。
