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
- `ArmorTrackerAaest.hpp`：可选的整车观测器 aaest（见下文 aaest 一节），从 aasim 仓库原样同步。
- `tools/coordinate_semantics_check.cpp`：公开坐标系姿态语义回归检查，防止
  `host/gimbal_quat` 被额外固定旋转翻转 roll/pitch。
- `tools/tracker_replay/armor_tracker_replay.cpp`：离线重放一致性检查工具。

`ArmorTracker` 主体是模板头文件实现，CMake 只暴露 include 目录；模块本身不再编译
额外 `.cpp` 源文件。

## 配置

运行配置只保留当前链路实际使用的三组字段：

- `cfg.tracker`：目标过滤、进入跟踪与丢失阈值；`use_aaest`、`aaest_latency_s`、`aaest_bullet_speed_m_s`
  控制可选的 aaest 观测器。
- `cfg.extrinsic.camera_mount_to_body`：手眼外参，只表达相机安装坐标系 `M` 到公开
  本体系 `B` 的真实安装偏差。`M` 与 OpenCV 相机系 `C` 同原点，并与 `B` 使用
  同一轴约定：右手系，`x` 向右，`y` 向前，`z` 向上。`C` 到 `M` 的固定轴变换
  由代码内部处理，不需要写进 YAML。`rotation` 为 `wxyz` 四元数，`translation`
  单位为 m。
- `cfg.preview`：`VisionPreview::RuntimeParam`，控制本模块自带预览线程。

默认 detector topic 为 `armor_detector/armors_frame`。tracker 对外只发布
`tracker/target_frame`。

Tracker 在构造期从 `CameraFrameSync::Calibration()` 复制一份原生相机标定。Detector
发布的四角点保持原生传感器坐标，Tracker 独立使用 230 mm 大装甲板模型和原生相机
K/D 执行 PnP、重投影。支持当前直接 PnP 路径的无畸变、5 项和 8 项畸变系数；需要
预先去畸变的模型或无效标定会记录错误并禁用观测。PnP 失败或结果非有限时不把该观测
送入跟踪更新。本阶段仍不复用 Detector pose，两者的尺寸和 yaw 处理语义保持独立。
异步 pending frame 只复制 `SharedFrame` 所有权、IMU 和检测结果，不复制图像字节。
worker 将所有权移动到栈上 `TrackedFrame`，同步发布 `const TrackedFrame*`；逐帧
`FrameGeometry` 始终只从 `SharedFrame.Get()->geometry` 读取。

输出统一使用与公开本体系 `B` 同向的惯性解算轴 `O`：右手系，`x` 向右，
`y` 向前，`z` 向上；yaw 以前向为 0，左转为正。`O` 的轴向不随当前云台 yaw
转动，因此后级 Aimer 从 `tracker/target_frame` 解出的 `host/target_euler.yaw`
是下位机可直接消费的绝对云台目标角。preview 使用 `output_to_camera` 把 `O`
中的目标几何投回同帧相机图像。历史 tracker `x` 前、`y` 左坐标基已经不再作为
配置或输出选项暴露。

## Preview

内置 preview 只在 `cfg.preview.enabled: true` 时启动，不订阅 topic、不录像、不反压主链路。
它把 detector 原生角点和 tracker 原生重投影逆映射到当前帧后，绘制 detector 四边形、tracker 整车中心、四个装甲面中心、相邻装甲面连线，以及带固定
倾角的装甲板物理框。多车跟踪时，preview 会绘制所有 active 车辆，并在车体中心标注编号和
当前选择评分；被选中的车辆用更醒目的中心和连线显示。detector preview 不在这里处理。

## Target Selection

tracker 内部按装甲板编号维护多套车辆 EKF 状态，同一 slot 丢失后不会清空 EKF。
同一帧里出现多个编号时，各编号状态独立更新；
`tracker/target_frame` 中只携带一个当前选择目标。当前选择分数使用装甲板观测数量的低通值、距离、
可打击面积、自旋速度和目标相对当前云台视轴的角度差，并用滞回 margin 避免输出目标抖动。
可打击面积在 `NativeToFrame` 后计算，因此 2x wide 模式不会产生 4 倍面积偏置。

候选的图像中心排序现在使用原生标定主点，而不是历史硬编码 `(720, 540)`。这是独立的
legacy bug 修复，可能只在同优先级候选排序接近边界时造成预期差异。

## aaest 观测器（可选）

`cfg.tracker.use_aaest: true` 时，四块装甲板的车辆改由 aaest 输出整车状态；目标的建立、丢失和选择
仍由本模块的 tracker 负责。前哨站（5）、基地（7）和大装甲的平衡步兵（2–4 号）保持原输出。

aaest 是角点级的平稳 EKF，加上对操作手换档（平移和自旋加速度阶跃）的 GLR 检测、周期变速陀螺的
谐振子模型，以及按角点似然选择的四个滤波器；首次观测用 IPPE 位姿和一组转速假设起始。算法规格和
仿真、实录结果在 aasim 仓库的 `docs/glr_estimator.md`、`docs/input_prediction.md`。

输出字段的含义与原来相同，只有两处差别：

- `velocity` 和 `v_yaw` 是未来 h 秒内的平均值，含按换档推断的前推；h 取
  `aaest_latency_s + 目标水平距离 / aaest_bullet_speed_m_s`（默认 0.07 s 与 23 m/s）。
- `tracked_face_index` 是当前最正对射手的板。

同步帧的 IMU 带陀螺仪角速度时（CameraFrameSync 的 TRIGGER 模式），aaest 同时估计图像相对 IMU 的
时间偏差，投影改用图像内容时刻的姿态；否则相机与 IMU 1–2 ms 的同步误差会把云台转动读成目标横向
速度，和 Aimer 的超前、云台一起形成约 7.5 Hz 的极限环。角速度为零（LATEST_IMU 模式）时行为不变。

相机标定按 OpenCV 五参数畸变使用；第 6 项以后有非零系数时 aaest 不启用，输出保持原样。
相机安装外参与 PnP 使用同一份 `cfg.extrinsic.camera_mount_to_body`。

`tools/tracker_replay/armor_tracker_replay.cpp` 读取环境变量 `TRACKER_USE_AAEST=1` 打开 aaest，
`TRACKER_AAEST_HORIZON` 把 h 固定为给定秒数（用于和 aasim 的独立回放 `aaest_replay` 对比）。

## 验证

标定尺寸与内参的一致性由 CameraBase 校验；内部 PnP 求解器只检查数值、模型支持和
求解结果，不要求离线角点回放额外提供图像宽高。输入时间戳回退时，TrackerCore
清除旧目标与滤波状态、重建时间基线，并使用不变的配置和标定重新捕获目标。

单车过滤回归使用 `tools/tracker_replay/armor_tracker_replay.cpp` 对固定数据集重放；多车目标选择
需要使用不按编号过滤的 replay，确认同帧多编号输入会独立更新各 slot 并只输出当前选中的目标。
BSP 验证需要生成当前 `cfg.tracker` / `cfg.preview` 结构对应的 `User/xrobot_main.hpp` 后再构建
Linux 和 Webots 目标。

坐标语义回归需要至少覆盖 `tools/coordinate_semantics_check.cpp`：
`host/gimbal_quat` 已经是公开本体系 `B` 的姿态，Tracker 只能归一化后直接转矩阵；
任何额外的固定 basis 旋转都会让 roll/pitch 反号，并污染输出目标高度。
