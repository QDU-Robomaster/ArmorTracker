# ArmorTracker

`ArmorTracker` 是 Webots/Linux 自瞄链路里的目标级跟踪模块。输入来自
`ArmorDetector` 发布的检测结果和 `CameraFrameSync` 同步帧，输出 tracker
域下的目标状态和调试信息。云台角、发送包和开火判定由后级 Aimer 负责。

## 文件结构

- `ArmorTracker.hpp`：模块入口、配置、topic、EKF 接线和主状态。
- `ArmorTrackerPipeline.hpp`：每帧检测结果进入后的主处理流程。
- `ArmorTrackerCommon.hpp`：yaw 展开、时间戳差值、图像面积等通用小函数。
- `ArmorTrackerRuntimeSupport.hpp`：运行时开关、相机位姿转换和投影辅助逻辑。
- `ArmorTrackerFaceSelector.hpp`：装甲面候选评分与选面策略。
- `ArmorTrackerSelectionSupport.hpp`：选面后的 face 与 image-track 绑定维护。
- `ArmorTrackerObserver.hpp`、`ArmorTrackerObserverRuntimeSupport.hpp`：整车几何观测模型和运行时状态映射。
- `ArmorTrackerImageTracker.hpp`：图像域短时身份跟踪，只用于辅助同一装甲板判断。
- `ArmorTrackerDebugSupport.hpp`、`ArmorTrackerStateAuditSupport.hpp`：预览绘制和状态审计输出。
- `SolveTrajectory.*`、`TrajectoryCompensationTable.hpp`、`table.bin`：保留 `Target` 消息定义和旧弹道工具，当前 tracker 主流程不再发布弹道命令。
- `extended_kalman_filter.*`：通用 EKF 实现。

## 构建边界

运行时只编译：

- `SolveTrajectory.cpp`
- `extended_kalman_filter.cpp`

`ArmorTracker` 主体是模板头文件实现。`TableGenerator.cpp` 是离线弹道表生成工具，不应链接进运行时目标。

tracker 日志会输出 `double` 观测量和 `uint64_t` 图像时间戳，模块 CMake 会显式打开 libxr 的
`LIBXR_PRINT_FLOAT_ENABLE_DOUBLE` 和 `LIBXR_PRINT_INTEGER_ENABLE_64BIT`。

## 弹道表

`table.bin` 是运行时弹道补偿表。如果需要重生成，单独编译运行 `TableGenerator.cpp`，确认结果后再替换
`table.bin`。

## 验证备注

Webots 验证时如果 tracker 一直停在 `LOST`，先检查配置里的空间过滤阈值。当前验证世界中目标位姿会超过
`max_z_position: 1.0` 的默认烟测阈值，放宽到 `30.0` 后 tracker 能稳定进入 `TRACKING`，各输出 topic
按图像频率发布。接入 Aimer 后，`tracker/send` 和 `tracker/target_eulr` 应只由 Aimer 发布，避免命令双写。
