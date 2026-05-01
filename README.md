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
- `ArmorTrackerDebugSupport.hpp`、`ArmorTrackerStateAuditSupport.hpp`：调试 topic 组包和状态审计输出。
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
`max_z_position: 1.0` 的默认验证阈值，放宽到 `30.0` 后 tracker 能稳定进入 `TRACKING`，各输出 topic
按图像频率发布。接入 Aimer 后，`tracker/send` 和 `tracker/target_eulr` 应只由 Aimer 发布，避免命令双写。

tracker 以同步图像的传感器时间戳计算 `dt`。如果进程启动、调试录像、Webots 暂停等情况造成相邻图像时间戳
出现大跳变，模块会丢弃旧 EKF / 装甲面绑定 / 图像域短时跟踪状态，并从当前帧重新进入 `DETECTING`。这种帧只
发布调试数据，不发布有效 `tracking` 目标，避免启动瞬态或长时间阻塞后的旧状态污染后级模块。

## 当前 SP 默认参数

固定云台 1000 帧 Webots 重复验证后，SP 模型保留两个默认值：

- `XR_TRACKER_SP_Q_XYZ=300`：位置过程噪声，允许同步稳定后更快吃进传感器侧更新。
- `XR_TRACKER_SP_PAIR_DZ_ALPHA=0.40`：双装甲高低差软融合权重，只修正 `Z_ARMOR / DELTA_Z`，不恢复双装甲连续全状态更新。

`cfg.sp` 用于场景级选择 SP 观测策略：

- `enable_pair_dz`：允许双装甲高低差软融合。Webots 四面目标可以打开，用于约束高低面；默认关闭，避免实拍/录制数据里错误双面观测污染状态。
- `measurement_recenter_alpha`：单装甲测量重定位权重。默认 `1.0` 保留当前主线行为；Webots 可按场景降低，避免单帧 PnP 尾巴强行拉动整车中心。
- `quality_recenter`：按匹配质量动态调节重定位权重。Webots 可打开，用 `score / yaw / xyz` 门控削弱低质量测量。
- `enable_multi_armor_fuse`：同时看见两片以上同编号装甲板时，才用多面几何观测在线修正中心和半径。单面观测不应被当成整车尺寸真值。

这些配置仍可被同名环境变量覆盖，用于现场继续做数据驱动调参；默认不启用 canonical 初始化、固定输出外推、强制测量锚定或 direct XYZ 更新。
