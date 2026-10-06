# ArmorTracker

装甲板跟踪：四块板的整车用整车估计器，其余用兜底 EKF，多目标中选出要打的一个，发布跟踪帧 / Armor tracking: the vehicle estimator for four-plate vehicles and a fallback EKF for the rest; picks one of several targets and publishes tracked frames

## 1. 模块作用 / Purpose

ArmorTracker 订阅 `<相机名>_detected`，按编号把装甲板分配到各自的跟踪槽，估计每辆车的中心、速度、朝向、转速和半径，从中选出要打的目标，发布 `TrackedFrame{detected, target, 世界到相机的变换}` 到 `<相机名>_tracked`。目标的字段与语义沿用 `ArmorTrackerTarget`，Aimer 直接读取。

ArmorTracker subscribes to `<camera>_detected`, assigns armors to one track slot per number, estimates each vehicle's centre, velocity, heading, spin and radii, picks the target to engage and publishes `TrackedFrame{detected, target, world-to-camera transform}` on `<camera>_tracked`. The target keeps the fields and meaning of `ArmorTrackerTarget`, which Aimer reads directly.

## 2. 整车估计器 / Vehicle Estimator

四块板的车（步兵与平衡步兵、英雄、哨兵、工程）由 `Vehicle::VehicleEstimator` 估计。它按 aasim `docs/glr_estimator.md` 重新实现了原型 aaest：

Four-plate vehicles (infantry including balance infantry, hero, sentry, engineer) are estimated by `Vehicle::VehicleEstimator`, a re-implementation of the prototype aaest following aasim `docs/glr_estimator.md`:

| 文件 / File | 内容 / Content |
| --- | --- |
| `VehicleGeometry.hpp` | 16 维状态（含图像时间偏差）、相机投影、装甲板几何 / 16-state layout (with the image delay), projection, plate geometry |
| `CornerEkf.hpp` | 以四个角点为观测的 EKF，精确离散化，谐振子自旋，预测速率 / Corner-level EKF with exact discretisation, harmonic spin and predicted rates |
| `ManeuverDetector.hpp` | GLR 换档检测：中心加速度与角加速度的阶跃 / GLR step detection of the centre and spin accelerations |
| `PoseBootstrap.hpp` | IPPE 位姿加 7 个转速假设的起步 / IPPE pose with seven spin-rate hypotheses |
| `ModelSelector.hpp` | 四个滤波器（STEADY、VARY、AGILE、HARM）按角点似然选择 / Choice among four filters by corner likelihood |
| `VehicleEstimator.hpp` | 组合以上部件 / Combines the parts |

输出的速度与转速是未来 h 秒内的平均值，h = `latency_s` + 距离 / `bullet_speed`，即从这一帧到弹丸命中的预期时间。关键点取 AutoAimTypes 的 lightbar4 尺寸，与 v4 检测器的角点定义一致。

The reported velocity and spin are means over the next h seconds, h = `latency_s` + distance / `bullet_speed`, the expected frame-to-impact time. The keypoints use the lightbar4 sizes from AutoAimTypes, matching the v4 detector's corners.

相机曝光与 IMU 采样不严格同步时，图像内容比配对的姿态早 δ，云台转动会被读成目标的横向运动，并与 Aimer 的超前形成约 7.5 Hz 的振荡。估计器把 δ 作为第 16 个状态（规格 §13）：模块把同步帧 IMU 的本体系角速度 `angular_velocity_xyz` 传入，投影改用 t − δ 时刻的姿态。δ 是传感器的量，目标丢失后沿用上次的估计；角速度为零时与不估计 δ 逐位相同。

When camera exposure and IMU sampling are not exactly synchronised, the image content is δ older than its attitude; gimbal rotation then reads as lateral target motion and closes a ~7.5 Hz oscillation with the Aimer lead. The estimator carries δ as state 16 (spec §13): the Module passes the body rate `angular_velocity_xyz` of the synced IMU sample, and the projection uses the attitude at t − δ. δ is a sensor property and is kept over target loss; a zero rate is bit-identical to no δ state.

`tools/vehicle_replay` 读回放数据包的检测 TSV 与 IMU CSV，命令行与输出格式与 aasim 的 `aaest_replay` 相同，角速度按零处理；`--trackset 1` 走模块的路径（角点顺序转换、相对时间、按距离的时域）。与 aasim `e7abbf6` 的 aaest 对照：同一编译参数下，含角速度与图像时间偏差的合成场景逐位相同；回放工具用模块的编译参数构建，在 new_v7_0507（2、6 号）与 5p9_0509（1、2 号）上跟踪标志、输出滤波器与正对板逐帧相同，位置与速度相差不超过 4e-9。

`tools/vehicle_replay` reads the replay package's detection TSV and IMU CSV with the same command line and output as aasim's `aaest_replay`, with a zero body rate; `--trackset 1` replays the Module's path (corner order, relative time, distance-based horizon). Against aaest at aasim `e7abbf6`: with the same build flags, synthetic scenes with a body rate and an image delay match bit for bit; the replay tool, built with the Module's flags, matches the tracking flag, reporting filter and facing plate of every frame on new_v7_0507 (numbers 2 and 6) and 5p9_0509 (1 and 2), with position and velocity within 4e-9.

## 3. 兜底 EKF / Fallback EKF

前哨站（三块板，三档高度）和基地（三块板）不是四块板的车，由 `FallbackTarget.hpp` 跟踪：逐块板 IPPE 求位姿，朝向在机体前向 ±70° 内按 1° 搜索重投影误差最小者，再用 11 维 EKF 跟踪中心、速度、朝向、转速和半径。前哨站按物理约定建模：板朝外，朝向与车辆同一约定，中心是转轴并由 EKF 逐帧修正；高度相位由换面时的高度跳变确定，丢失后用上一次的中心重新起始。

The outpost (three plates at three heights) and the base (three plates) are not four-plate vehicles and are tracked by `FallbackTarget.hpp`: IPPE per plate, the heading searched in 1° steps within ±70° of the body forward for the smallest reprojection error, then an 11-state EKF of centre, velocity, heading, spin and radius. The outpost follows the physical convention: plates face outward, the heading convention is the vehicles', and the centre is the spin axis, corrected by the EKF every frame; the height phase comes from the height jump at a face change, and after a loss it restarts from the previous centre.

## 4. 目标管理 / Target Management

`TrackSet.hpp` 为编号 1–5、前哨站、哨兵、基地各维护一个槽。板的大小由编号决定：1 号和基地为大板，其余为小板，不看检测器的大小输出。

`TrackSet.hpp` keeps one slot each for numbers 1–5, the outpost, the sentry and the base. The plate size follows the number: number 1 and the base are large, the rest small, regardless of the detector's size output.

槽的状态：`LOST` → 看到即 `DETECTING` → 连续看到 `min_detect_count` 帧为 `TRACKING` → 没看到为 `TEMP_LOST` → 连续 `max_temp_lost` 帧（前哨站 `outpost_max_temp_lost`）没看到回到 `LOST`。状态只决定能否被选中；整车估计器连续 2 s 没看到才丢弃，兜底目标在发散或 NIS 连续超限时重置。

Slot states: `LOST` → `DETECTING` when seen → `TRACKING` after `min_detect_count` frames → `TEMP_LOST` when unseen → back to `LOST` after `max_temp_lost` frames (`outpost_max_temp_lost` for the outpost). The state only decides whether a slot can be chosen; a vehicle estimator is dropped after 2 s unseen, and a fallback target is reset when it diverges or keeps failing NIS.

每个可选的槽打分：观测数、距离、可打面积（原生像素）、转速、偏离光轴的角度，各项归一化后加权，`DETECTING` 与 `TEMP_LOST` 打折扣。得分最高者为目标，换目标要领先 `switch_margin`。设置 `target_number` 后只打该编号。

Each selectable slot is scored from the observation count, distance, hittable area (native pixels), spin and angle off the optical axis, normalised and weighted, with `DETECTING` and `TEMP_LOST` discounted. The best score is the target, and switching needs a lead of `switch_margin`. With `target_number` set only that number is engaged.

## 5. 线程与 Topic / Threads and Topics

检测帧进一个容量为 2 的队列，满了在检测器的发布线程里等待，所以每一帧都被跟踪；工作线程按顺序处理并发布，每收一帧发一帧。

Detected frames enter a queue of two; when it is full the detector's publishing thread waits, so every frame is tracked. The worker thread processes and publishes in order, one frame out per frame in.

| Topic | 载荷 / Payload | 方向 / Direction |
| --- | --- | --- |
| `<相机名>_detected` | `const AutoAim::DetectedFrame*` | 订阅 / Subscribed |
| `<相机名>_tracked` | `const AutoAim::TrackedFrame*` | 发布 / Published |

`TrackedFrame` 的 `output_to_camera_rotation/translation` 是这一帧世界系到相机光学系的变换，预览用它把目标画回图像。

`TrackedFrame`'s `output_to_camera_rotation/translation` is this frame's world-to-optical transform, which the preview uses to draw the target on the image.

## 6. 配置示例 / Configuration Example

```yaml
modules:
  - module: QDU-Robomaster/ArmorTracker
    id: tracker
    args:
      - settings:
          camera_name: "gimbal"
          mount_rotation_wxyz: [1.0, 0.0, 0.0, 0.0]
          mount_translation: [0.0, 0.0, 0.0]
          target_number: -1
          min_detect_count: 2
          max_temp_lost: 15
          outpost_max_temp_lost: 75
          latency_s: 0.07
          bullet_speed: 23.0
          select: {}
```

`mount_rotation_wxyz`、`mount_translation` 是相机安装到云台本体的旋转与平移（本体系 x 右、y 前、z 上）。相机内参与畸变取自图像帧携带的标定。

`mount_rotation_wxyz` and `mount_translation` are the camera mounting on the gimbal body (body x right, y forward, z up). The intrinsics and distortion come from the calibration carried by the image frame.

## 7. 测试 / Tests

`tests/tracker_test.cpp` 用合成的车辆检测检查：整车估计器在转动目标上收敛（中心 < 2 cm、转速 < 0.3 rad/s）、图像比姿态旧 2 ms 且云台 7.5 Hz 摆动时给角速度后偏差估计误差 < 0.5 ms、速度误差减半以上、两个目标中选近的并在其消失后换到另一个、检测器把 3 号报成大板时仍按小板整车跟踪、基地按三块板兜底、板朝外的前哨站中心落在转轴上（< 5 cm）、模块每收一帧发一帧。

`tests/tracker_test.cpp` checks with synthetic vehicle detections that the vehicle estimator converges on a spinning target (centre < 2 cm, spin < 0.3 rad/s), with the image 2 ms older than its attitude under a 7.5 Hz gimbal sway the body rate brings the delay estimate within 0.5 ms and at least halves the velocity error, the nearer of two targets is chosen and the other takes over when it disappears, number 3 reported as large is still tracked as a small-plate vehicle, the base falls back to a three-plate target, an outward-facing outpost keeps its centre on the spin axis (< 5 cm), and the Module publishes one frame per frame.

## 8. 依赖 / Dependencies

CameraBase、AutoAimTypes、LibXR、Eigen 3（含 `unsupported/Eigen/MatrixFunctions`）、OpenCV（core、calib3d、imgproc）。

CameraBase, AutoAimTypes, LibXR, Eigen 3 (with `unsupported/Eigen/MatrixFunctions`), OpenCV (core, calib3d, imgproc).
