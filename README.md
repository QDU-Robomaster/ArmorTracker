# ArmorTracker

`ArmorTracker` 是 Webots/Linux 自瞄链路里的目标级跟踪模块。输入来自
`ArmorDetector` 发布的检测结果和 `CameraFrameSync` 同步帧，输出 tracker
域下的目标状态和调试信息。云台角、发送包和开火判定由后级 Aimer 负责。

## 文件结构

- `ArmorTracker.hpp`：模块入口、配置、topic、EKF 接线和主状态。
- `ArmorTrackerTarget.hpp`：`tracker/target` 发布的目标状态消息。
- `ArmorTrackerPipeline.hpp`：每帧检测结果进入后的主处理流程。
- `ArmorTrackerCommon.hpp`：yaw 展开、时间戳差值、图像面积等通用小函数。
- `ArmorTrackerRuntimeConfig.hpp`：运行时开关、相机位姿转换和投影辅助逻辑。
- `ArmorTrackerVehicleModel.hpp`：整车状态预测、装甲面匹配和 EKF 更新核心。
- `ArmorTrackerFaceSelector.hpp`：装甲面候选评分与选面策略。
- `ArmorTrackerObserver.hpp`、`ArmorTrackerRuntimeAdapter.hpp`：整车几何观测模型和运行时状态映射。
- `ArmorTrackerImageTracker.hpp`：图像域短时身份跟踪，只用于辅助同一装甲板判断。
- `extended_kalman_filter.*`：通用 EKF 实现。

## 构建边界

运行时只编译：

- `extended_kalman_filter.cpp`

`ArmorTracker` 主体是模板头文件实现。tracker 不包含弹道解算、开火判定或云台命令发布逻辑。

tracker 日志会输出 `double` 观测量和 `uint64_t` 图像时间戳。BSP 或 CI 的顶层 CMake 必须在
`add_subdirectory(libxr)` 前显式打开 `LIBXR_PRINT_FLOAT_ENABLE_DOUBLE` 和
`LIBXR_PRINT_INTEGER_ENABLE_64BIT`。

## 实时预览

tracker 可选组合 `VisionPreview`，配置入口是 `cfg.preview`。预览只在
`preview.enabled: true` 时启动 OpenCV 窗口；它不订阅 topic、不录像、不写调试文件。
主链路在发布 `tracker/target` 和 `tracker/ekf_points` 后提交当前图像与输出快照，
窗口线程负责绘制 detector 四边形、EKF 中心和四个装甲点。

## 验证备注

Webots 验证时如果 tracker 一直停在 `LOST`，先检查配置里的空间过滤阈值。当前验证世界中目标位姿会超过
`max_z_position: 1.0` 的默认验证阈值，放宽到 `30.0` 后 tracker 能稳定进入 `TRACKING`，各输出 topic
按图像频率发布。

tracker 以同步图像的传感器时间戳计算 `dt`。如果进程启动、调试录像、Webots 暂停等情况造成相邻图像时间戳
出现大跳变，模块会丢弃旧 EKF / 装甲面绑定 / 图像域短时跟踪状态，并从当前帧重新进入 `DETECTING`。这种帧只
发布调试数据，不发布有效 `tracking` 目标，避免启动瞬态或长时间阻塞后的旧状态污染后级模块。

## 当前整车模型默认参数

固定云台 1000 帧 Webots 和内录 replay 对比后，整车模型保留以下默认策略：

- `XR_TRACKER_MODEL_Q_XYZ=300`：位置过程噪声，允许同步稳定后更快吃进传感器侧更新。
- 双装甲高低差软融合默认开启；固定俯仰装甲 yaw 重估保留为实验开关，默认关闭。
- `ekf_points` 的整车输出测量锚定默认关闭，避免调试/预览里的整车多面输出跟随单帧 PnP 抖动。
- Aimer 使用当前可见测量面作为瞄准锚点默认开启，和 `ekf_points` 整车输出锚定解耦。

`cfg.model` 用于场景级选择整车观测策略：

- `enable_pair_dz`：允许双装甲高低差软融合，用于约束四面目标高低面；当前候选默认开启，配合单面观测冻结 `dz`，由双板观测负责恢复交错装甲板高低差。
- `measurement_recenter_alpha`：单装甲测量重定位权重。当前候选默认 `0.25`，避免单帧 PnP 尾巴强行拉动整车中心。
- `quality_recenter`：按匹配质量动态调节重定位权重。当前候选默认开启，用 `score / yaw / xyz` 门控削弱低质量测量。
- `enable_pair_geometry`：双装甲显式估计整车中心和半径。当前候选默认关闭；内录二分显示它会污染中心/半径，交错高低差由 `pair_dz` 单独恢复。
- `enable_output_meas_anchor`：默认关闭。开启后只在输出层用当前可见面测量锚定输出，内部车体 EKF 仍保持连续几何状态；输出给 `ekf_points` 时使用刚性平移整车输出，避免单面替换造成视觉形变。
- `enable_aimer_meas_anchor`：默认开启。允许后级 Aimer 使用当前可见测量面作为瞄准锚点，不要求 `ekf_points` 整车输出同步锚定。
- `enable_fixed_pose_yaw_opt`：默认关闭。对固定俯仰小装甲按 2D 重投影重估 yaw。内录验证中该搜索会把部分同面候选 yaw diff 推到接近 `pi` 并触发面选择拒绝；如需复现实验可设置 `XR_TRACKER_MODEL_ENABLE_FIXED_POSE_YAW_OPT=1`。
- 单装甲板默认不改变 `DELTA_Z`，只允许双板观测写入高度差；如需复现实验可设置 `XR_TRACKER_MODEL_DISABLE_FREEZE_SINGLE_DZ=1`。

这些配置仍可被同名环境变量覆盖，用于现场继续做数据驱动调参；默认不启用 canonical 初始化、固定输出外推、多装甲全车融合或 direct XYZ 更新。

观测质量门控默认开启，但 PnP yaw 的 `±pi` 折叠默认关闭；内录 1000 帧 A/B 中该折叠会降低匹配连续性。
如需复现实验 profile，可显式设置 `XR_TRACKER_ENABLE_PNP_PI_YAW_FOLD=1`。
