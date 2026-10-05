# ArmorTracker

装甲板跟踪：四块板的整车用整车估计器，其余用兜底 EKF，多目标中选出要打的一个，发布跟踪帧 / Armor tracking: the vehicle estimator for four-plate vehicles and a fallback EKF for the rest; picks one of several targets and publishes tracked frames

## 1. 模块作用 / Purpose

ArmorTracker 订阅 `<相机名>_detected`，按编号把装甲板分配到各自的跟踪槽，估计每辆车的中心、速度、朝向、转速和半径，从中选出要打的目标，发布 `TrackedFrame{detected, target, 世界到相机的变换}` 到 `<相机名>_tracked`。目标的字段与语义沿用 `ArmorTrackerTarget`，Aimer 直接读取。

ArmorTracker subscribes to `<camera>_detected`, assigns armors to one track slot per number, estimates each vehicle's centre, velocity, heading, spin and radii, picks the target to engage and publishes `TrackedFrame{detected, target, world-to-camera transform}` on `<camera>_tracked`. The target keeps the fields and meaning of `ArmorTrackerTarget`, which Aimer reads directly.

## 2. 整车估计器 / Vehicle Estimator

四块板的车（步兵、英雄、哨兵、工程）由 `Vehicle::VehicleEstimator` 估计。它按 aasim `docs/glr_estimator.md` 重新实现了原型 aaest：

Four-plate vehicles (infantry, hero, sentry, engineer) are estimated by `Vehicle::VehicleEstimator`, a re-implementation of the prototype aaest following aasim `docs/glr_estimator.md`:

| 文件 / File | 内容 / Content |
| --- | --- |
| `VehicleGeometry.hpp` | 15 维状态、相机投影、装甲板几何 / 15-state layout, projection, plate geometry |
| `CornerEkf.hpp` | 以四个角点为观测的 EKF，精确离散化，谐振子自旋，预测速率 / Corner-level EKF with exact discretisation, harmonic spin and predicted rates |
| `ManeuverDetector.hpp` | GLR 换档检测：中心加速度与角加速度的阶跃 / GLR step detection of the centre and spin accelerations |
| `PoseBootstrap.hpp` | IPPE 位姿加 7 个转速假设的起步 / IPPE pose with seven spin-rate hypotheses |
| `ModelSelector.hpp` | 四个滤波器（STEADY、VARY、AGILE、HARM）按角点似然选择 / Choice among four filters by corner likelihood |
| `VehicleEstimator.hpp` | 组合以上部件 / Combines the parts |

输出的速度与转速是未来 h 秒内的平均值，h = `latency_s` + 距离 / `bullet_speed`，即从这一帧到弹丸命中的预期时间。关键点取 AutoAimTypes 的 lightbar4 尺寸，与 v4 检测器的角点定义一致。

The reported velocity and spin are means over the next h seconds, h = `latency_s` + distance / `bullet_speed`, the expected frame-to-impact time. The keypoints use the lightbar4 sizes from AutoAimTypes, matching the v4 detector's corners.

`tools/vehicle_replay` 读回放数据包的检测 TSV 与 IMU CSV，命令行与输出格式与 aasim 的 `aaest_replay` 相同。用原型的关键点尺寸并以相同编译参数构建时，new_v7_0507（2、6 号）与 5p9_0509（1、2 号）上每帧每个字段都与 `aaest_replay` 相同；`--trackset 1` 走模块的路径（角点顺序转换、相对时间、按距离的时域）。

`tools/vehicle_replay` reads the replay package's detection TSV and IMU CSV with the same command line and output as aasim's `aaest_replay`. With the prototype's keypoint sizes and the same build flags, every field of every frame matches `aaest_replay` on new_v7_0507 (numbers 2 and 6) and 5p9_0509 (1 and 2); `--trackset 1` replays the Module's path (corner order, relative time, distance-based horizon).

## 3. 兜底 EKF / Fallback EKF

前哨站（三块板，三档高度）、基地（三块板）和平衡步兵（3、4、5 号的大板，两块板）不是四块板的车，由 `FallbackTarget.hpp` 跟踪：逐块板 IPPE 求位姿，朝向在机体前向 ±70° 内按 1° 搜索重投影误差最小者，再用 11 维 EKF 跟踪中心、速度、朝向、转速和半径。前哨站按换面时的高度跳变确定高度相位，丢失后用上一次的中心重新起始。

The outpost (three plates at three heights), the base (three plates) and balance infantry (large plates on numbers 3, 4, 5; two plates) are not four-plate vehicles and are tracked by `FallbackTarget.hpp`: IPPE per plate, the heading searched in 1° steps within ±70° of the body forward for the smallest reprojection error, then an 11-state EKF of centre, velocity, heading, spin and radius. The outpost's height phase comes from the height jump at a face change, and after a loss it restarts from the previous centre.

## 4. 目标管理 / Target Management

`TrackSet.hpp` 为编号 1–5、前哨站、哨兵、基地各维护一个槽。1 号和基地一律按大板处理；槽的大小板在起始时确定，之后大小不同的检测不进入该槽。

`TrackSet.hpp` keeps one slot each for numbers 1–5, the outpost, the sentry and the base. Number 1 and the base always count as large; a slot's plate size is fixed when it starts and detections of the other size are not fed to it.

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

`tests/tracker_test.cpp` 用合成的车辆检测检查：整车估计器在转动目标上收敛（中心 < 2 cm、转速 < 0.3 rad/s）、两个目标中选近的并在其消失后换到另一个、大板 3 号按平衡步兵兜底、模块每收一帧发一帧。

`tests/tracker_test.cpp` checks with synthetic vehicle detections that the vehicle estimator converges on a spinning target (centre < 2 cm, spin < 0.3 rad/s), the nearer of two targets is chosen and the other takes over when it disappears, a large number 3 falls back to balance infantry, and the Module publishes one frame per frame.

## 8. 依赖 / Dependencies

CameraBase、AutoAimTypes、LibXR、Eigen 3（含 `unsupported/Eigen/MatrixFunctions`）、OpenCV（core、calib3d、imgproc）。

CameraBase, AutoAimTypes, LibXR, Eigen 3 (with `unsupported/Eigen/MatrixFunctions`), OpenCV (core, calib3d, imgproc).
