# ArmorTracker

装甲板目标级跟踪：由检测结果和同帧 IMU 姿态维护整车 EKF 并发布同帧目标 / Armor target tracking Module that maintains per-vehicle EKF states from detections and same-frame IMU attitude and publishes the same-frame target

## 1. 模块作用 / Purpose

ArmorTracker 订阅 `ArmorDetector` 发布的 `armors_frame`（携带 `CameraFrameSync` 同步帧的图像、IMU 与检测结果），按装甲板编号为每辆车维护整车 EKF 状态，选出一个当前目标，在 `tracker` 域发布 `target_frame`。Topic 时间戳为 `SyncedFrame::imu.timestamp_us`。构造时等待 `armors_frame` 出现，因此 ArmorDetector 实例在 `modules:` 中位于本实例之前。

Topic 回调校验帧几何后，把共享图像所有权、IMU 和检测结果复制到 16 槽固定队列，图像字节保持共享。队列已满时回调等待空槽。单个 worker 线程依次取出一帧，把图像所有权移动到栈上的 `TrackedFrame`，运行 tracker 后同步发布 `const TrackedFrame*`，指针在发布回调期间有效。几何非法的帧在回调中被拒绝，逐帧 `FrameGeometry` 取自 `SharedFrame` 内的 `geometry`。

`TrackedFrame` 包含 `sequence`、`image`、`imu`、`target`（`ArmorTrackerTarget`）、`output_to_camera_rotation`（row-major 3x3）与 `output_to_camera_translation`（单位 m），后两者为输出系 `O` 到 OpenCV 相机系的变换。`ArmorTrackerTarget` 包含 `tracking`、目标编号 `id`、装甲面数量 `armors_num`、整车中心位置与速度（m、m/s）、`yaw` 与 `v_yaw`（rad、rad/s）、偶数面与奇数面的装甲半径、奇偶面高度差 `dz`（m）、当前绑定的装甲面索引、前哨站高度相位和换面标志。

同一帧出现多个编号时，各编号的 EKF 状态独立更新，`target_frame` 携带其中一个当前目标。选择分数由近期可见装甲数量、距离、可打击面积、自旋速度和目标相对当前视轴的角度差加权得到，并按状态（`detecting`、`temp_lost`）缩放。当前目标只在其他目标的分数高出 `switch_margin` 时被替换。`require_target_tag` 为 `true` 时只选择 `target_tag_id` 对应的编号。输入时间戳回退时，tracker 清除目标与滤波状态并重新建立时间基线，配置与标定保持不变。

`cfg.preview.enabled` 为 `true` 时启动内置预览，把 detector 四边形、整车中心、各装甲面中心与相邻面连线、装甲板物理框绘制到当前帧图像，并为所有 active 车辆标注编号和选择分数，当前目标以更醒目的方式显示。预览通过 `VisionPreview` 输出。

模块在 RamFS 中创建命令文件 `armor_tracker`：

```text
armor_tracker show                       # 打印 tracker 与外参配置
armor_tracker target_tag_id <value>      # 设置 target_tag_id
armor_tracker require_target_tag <0|1>   # 设置 require_target_tag
```

修改在 worker 处理下一帧之前生效，生效时重新配置 tracker 并重启预览。`OnMonitor()` 输出自上次调用以来的入队帧数和处理帧数、队列就绪数、占用数与高水位、队列满等待次数、平均生产者等待时间（ms），以及 worker 单帧服务耗时的计数、平均、最小与最大值（us）。

ArmorTracker subscribes to `armors_frame` published by `ArmorDetector` (carrying the image, IMU and detections of a `CameraFrameSync` synchronized frame), maintains an EKF state per vehicle by armor number, selects one current target and publishes `target_frame` in the `tracker` domain. The Topic timestamp is `SyncedFrame::imu.timestamp_us`. The constructor waits for `armors_frame` to appear, so the ArmorDetector instance is listed before this instance in `modules:`.

After validating the frame geometry, the Topic callback copies the shared image ownership, the IMU and the detections into a fixed 16-slot queue; the image bytes stay shared. When the queue is full the callback waits for a free slot. A single worker thread takes one frame at a time, moves the image ownership into a `TrackedFrame` on its stack, runs the tracker and publishes `const TrackedFrame*` synchronously; the pointer is valid during the publish callback. Frames with invalid geometry are rejected in the callback, and the per-frame `FrameGeometry` is read from the `geometry` inside `SharedFrame`.

`TrackedFrame` contains `sequence`, `image`, `imu`, `target` (`ArmorTrackerTarget`), `output_to_camera_rotation` (row-major 3x3) and `output_to_camera_translation` (m); the last two are the transform from the output frame `O` to the OpenCV camera frame. `ArmorTrackerTarget` contains `tracking`, the target number `id`, the armor face count `armors_num`, the vehicle center position and velocity (m, m/s), `yaw` and `v_yaw` (rad, rad/s), the armor radii of the even and odd faces, the height difference `dz` between even and odd faces (m), the currently bound armor face index, the outpost height phase and a face-switch flag.

When one frame contains several armor numbers, the EKF state of each number is updated independently and `target_frame` carries one current target. The selection score is a weighted sum of the recent visible armor count, distance, hittable area, spin speed and the angle between the target and the current optical axis, scaled by state (`detecting`, `temp_lost`). The current target is replaced only when another target scores higher by more than `switch_margin`. With `require_target_tag` set to `true`, only the number given by `target_tag_id` is selected. When the input timestamp goes backward, the tracker clears the target and filter state and re-establishes the time base; the configuration and calibration are kept.

With `cfg.preview.enabled` set to `true`, the built-in preview starts. It draws the detector quadrilateral, the vehicle center, the armor face centers with the lines between adjacent faces, and the physical armor frames onto the current frame image, and labels every active vehicle with its number and selection score; the current target is drawn more prominently. The preview is output through `VisionPreview`.

The Module creates the RamFS command file `armor_tracker`:

```text
armor_tracker show                       # print the tracker and extrinsic configuration
armor_tracker target_tag_id <value>      # set target_tag_id
armor_tracker require_target_tag <0|1>   # set require_target_tag
```

A change takes effect before the worker processes the next frame, at which point the tracker is reconfigured and the preview restarted. `OnMonitor()` prints the enqueued and processed frame counts since the previous call, the queue ready count, occupancy and high-water mark, the queue-full wait count, the average producer wait time (ms), and the count, average, minimum and maximum of the worker per-frame service time (us).

## 2. 坐标与 PnP / Coordinates and PnP

`cfg.extrinsic.camera_mount_to_body` 是手眼外参，表示相机安装坐标系 `M` 到公开本体系 `B` 的安装偏差。`M` 与 OpenCV 相机系 `C`（`x` 向右、`y` 向下、`z` 向前）同原点，并与 `B` 使用同一轴约定：右手系，`x` 向右，`y` 向前，`z` 向上。`C` 到 `M` 的固定轴变换在模块内部完成。`rotation` 为 `wxyz` 四元数，`translation` 单位为 m。

构造时，tracker 从 `CameraFrameSync::Calibration()` 复制原生相机标定。detector 发布的四角点为原生传感器坐标，tracker 按装甲板类型使用 230 mm（大）或 135 mm（小）宽、56 mm 灯条长的模型和原生相机 K/D 独立执行 PnP 与重投影。标定支持无畸变、5 项和 8 项畸变系数。标定无效或畸变模型需要预先去畸变时，模块记录错误并禁用观测。PnP 失败或结果非有限的观测不进入跟踪更新。

同步帧 IMU 四元数（与 `host` 域 `gimbal_quat` 相同，已是公开本体系 `B` 的姿态）作为本体到世界的旋转，归一化后直接转为矩阵。

输出使用与公开本体系 `B` 同向的惯性解算轴 `O`：右手系，`x` 向右，`y` 向前，`z` 向上，yaw 以前向为 0、左转为正。`O` 的轴向不随当前云台 yaw 转动，后级 Aimer 由此解出的 yaw 是下位机可直接使用的绝对云台目标角。预览通过 `output_to_camera` 把 `O` 中的目标几何投回同帧相机图像。

`cfg.extrinsic.camera_mount_to_body` is the hand-eye extrinsic, the mounting offset from the camera mount frame `M` to the public body frame `B`. `M` shares its origin with the OpenCV camera frame `C` (`x` right, `y` down, `z` forward) and uses the same axis convention as `B`: right-handed, `x` right, `y` forward, `z` up. The fixed axis conversion from `C` to `M` is done inside the Module. `rotation` is a `wxyz` quaternion and `translation` is in m.

At construction, the tracker copies the native camera calibration from `CameraFrameSync::Calibration()`. The four corner points published by the detector are in native sensor coordinates. The tracker runs PnP and reprojection independently with the native camera K/D and a model of 230 mm (large) or 135 mm (small) armor width and 56 mm light bar length, selected by armor type. The calibration supports no distortion, 5 coefficients and 8 coefficients. When the calibration is invalid or the distortion model requires undistortion first, the Module logs an error and disables observations. Observations whose PnP fails or yields non-finite results do not enter the tracking update.

The IMU quaternion of the synchronized frame (the same as `gimbal_quat` in the `host` domain, already the attitude of the public body frame `B`) is the body-to-world rotation, normalized and converted to a matrix directly.

The output uses the inertial solution axes `O`, oriented like the public body frame `B`: right-handed, `x` right, `y` forward, `z` up, with yaw 0 forward and positive to the left. The axes of `O` do not rotate with the current gimbal yaw, so the yaw solved from it by the downstream Aimer is an absolute gimbal target angle that the lower controller can use directly. The preview projects the target geometry in `O` back onto the same-frame camera image through `output_to_camera`.

## 3. 构造接口 / Constructor

```cpp
template <CameraTypes::FrameLayout FrameLayoutV>
class ArmorTracker;

explicit ArmorTracker(LibXR::RamFS& ramfs, FrameSync& sync, Config cfg = DefaultConfig());
```

模板参数：

- `FrameLayoutV`：帧布局，与上游相机、CameraFrameSync 和 ArmorDetector 使用的帧布局相同。

依赖：

- `ramfs`：`LibXR::RamFS`，用于注册命令文件 `armor_tracker`。
- `sync`：`CameraFrameSync<FrameLayoutV>&`，构造时从中复制原生相机标定。

配置参数（`Config`，`DefaultConfig()` 为全部默认值）：

- `tracker.require_target_tag`：只选择 `target_tag_id` 对应的编号，默认 `false`。
- `tracker.target_tag_id`：指定的目标编号，默认 `-1`。
- `tracker.min_detect_count`：由检测态进入跟踪态所需的检测次数，默认 `2`。
- `tracker.max_temp_lost_count`：暂时丢失的帧数上限，默认 `15`。
- `tracker.outpost_max_temp_lost_count`：前哨站暂时丢失的帧数上限，默认 `75`。
- `tracker.target_select.observed_count_weight`：近期可见装甲数量评分的权重，默认 `1.6`。
- `tracker.target_select.distance_weight`：距离评分的权重，默认 `2.0`。
- `tracker.target_select.area_weight`：可打击面积评分的权重，默认 `1.2`。
- `tracker.target_select.spin_weight`：自旋评分的权重，默认 `0.8`。
- `tracker.target_select.angle_weight`：视轴角差评分的权重，默认 `2.0`。
- `tracker.target_select.max_distance_m`：距离评分取满分对应的距离，单位 m，默认 `8.0`。
- `tracker.target_select.distance_span_m`：距离评分由满分降到 0 的距离跨度，单位 m，默认 `7.5`。
- `tracker.target_select.area_norm_px`：面积评分的归一化面积，单位 px，默认 `6000.0`。
- `tracker.target_select.observed_count_norm`：数量评分的归一化数量，默认 `4.0`。
- `tracker.target_select.max_spin_rad_s`：自旋评分的归一化角速度，单位 rad/s，默认 `8.0`。
- `tracker.target_select.max_angle_norm`：视轴角差评分的归一化角差，默认 `0.5`。
- `tracker.target_select.detecting_scale`：`detecting` 状态的分数倍率，默认 `0.55`。
- `tracker.target_select.temp_lost_scale`：`temp_lost` 状态的分数倍率，默认 `0.35`。
- `tracker.target_select.switch_margin`：替换当前目标所需的最小分差，默认 `0.25`。
- `extrinsic.camera_mount_to_body.rotation`：`wxyz` 四元数，默认 `[1, 0, 0, 0]`。
- `extrinsic.camera_mount_to_body.translation`：平移，单位 m，默认 `[0, 0, 0]`。
- `preview`：`VisionPreview::RuntimeParam`，默认关闭，字段见 VisionPreview。

Template parameter:

- `FrameLayoutV`: the frame layout, identical to that of the upstream camera, CameraFrameSync and ArmorDetector.

Dependencies:

- `ramfs`: `LibXR::RamFS`, used to register the command file `armor_tracker`.
- `sync`: `CameraFrameSync<FrameLayoutV>&`, from which the native camera calibration is copied at construction.

Configuration parameters (`Config`; `DefaultConfig()` holds all defaults):

- `tracker.require_target_tag`: select only the number given by `target_tag_id`, default `false`.
- `tracker.target_tag_id`: the designated target number, default `-1`.
- `tracker.min_detect_count`: detections required to move from the detecting state to the tracking state, default `2`.
- `tracker.max_temp_lost_count`: upper limit of temporarily lost frames, default `15`.
- `tracker.outpost_max_temp_lost_count`: upper limit of temporarily lost frames for the outpost, default `75`.
- `tracker.target_select.observed_count_weight`: weight of the recent visible armor count score, default `1.6`.
- `tracker.target_select.distance_weight`: weight of the distance score, default `2.0`.
- `tracker.target_select.area_weight`: weight of the hittable area score, default `1.2`.
- `tracker.target_select.spin_weight`: weight of the spin score, default `0.8`.
- `tracker.target_select.angle_weight`: weight of the optical-axis angle score, default `2.0`.
- `tracker.target_select.max_distance_m`: distance at which the distance score is full, in m, default `8.0`.
- `tracker.target_select.distance_span_m`: distance span over which the distance score falls from full to 0, in m, default `7.5`.
- `tracker.target_select.area_norm_px`: normalization area of the area score, in px, default `6000.0`.
- `tracker.target_select.observed_count_norm`: normalization count of the count score, default `4.0`.
- `tracker.target_select.max_spin_rad_s`: normalization angular velocity of the spin score, in rad/s, default `8.0`.
- `tracker.target_select.max_angle_norm`: normalization angle of the optical-axis angle score, default `0.5`.
- `tracker.target_select.detecting_scale`: score scale in the `detecting` state, default `0.55`.
- `tracker.target_select.temp_lost_scale`: score scale in the `temp_lost` state, default `0.35`.
- `tracker.target_select.switch_margin`: minimum score difference required to replace the current target, default `0.25`.
- `extrinsic.camera_mount_to_body.rotation`: `wxyz` quaternion, default `[1, 0, 0, 0]`.
- `extrinsic.camera_mount_to_body.translation`: translation in m, default `[0, 0, 0]`.
- `preview`: `VisionPreview::RuntimeParam`, disabled by default; see VisionPreview for the fields.

## 4. Topic

| Topic | 方向 | 类型 | 说明 |
| --- | --- | --- | --- |
| `armor_detector` 域的 `armors_frame` | 订阅 | `const DetectedFrame<FrameLayoutV>*` | ArmorDetector 发布的检测结果，携带图像、IMU 与检测结果 |
| `tracker` 域的 `target_frame` | 发布 | `const TrackedFrame<FrameLayoutV>*` | 同帧目标包，时间戳为 `SyncedFrame::imu.timestamp_us`，指针在同步回调期间有效 |

| Topic | Direction | Type | Meaning |
| --- | --- | --- | --- |
| `armors_frame` in the `armor_detector` domain | Subscribe | `const DetectedFrame<FrameLayoutV>*` | Detection result published by ArmorDetector, carrying the image, IMU and detections |
| `target_frame` in the `tracker` domain | Publish | `const TrackedFrame<FrameLayoutV>*` | Same-frame target packet, timestamp `SyncedFrame::imu.timestamp_us`, the pointer is valid during the synchronous callback |

## 5. 配置示例 / Configuration Example

`xrobot instance add QDU-Robomaster/ArmorTracker` 写入的实例：模板实参填为帧布局 constexpr，依赖填为 RamFS 的硬件注册名和 CameraFrameSync 实例的 id，`cfg` 展开为字段映射并填入外参与预览设置。帧布局 `AutoAimRunConfig::MainFrameLayout` 在配置的 `constexprs:` 中定义，须与相机输出一致。RamFS 对象 `ramfs` 由 BSP 的 `XR_REGISTER`（硬件注册）提供。

An instance written by `xrobot instance add QDU-Robomaster/ArmorTracker`: the template argument is set to the frame layout constexpr, the dependencies are set to the Registration name of the RamFS and the id of the CameraFrameSync instance, and `cfg` is expanded into a field mapping with the extrinsic and preview settings filled in. The frame layout `AutoAimRunConfig::MainFrameLayout` is defined under `constexprs:` of the Configuration and matches the camera output. The RamFS object `ramfs` is provided by the BSP's `XR_REGISTER` (Registration).

```yaml
constexpr_namespace: AutoAimRunConfig
constexpr_includes:
  - CameraBase.hpp
constexprs:
  MainFrameLayout:
    type: CameraTypes::FrameLayout
    value: '{.width = 800, .height = 600, .step = 2400, .encoding = CameraTypes::Encoding::BGR8}'
modules:
  - module: QDU-Robomaster/ArmorTracker
    id: ArmorTracker_0
    template_args:
      - AutoAimRunConfig::MainFrameLayout
    args:
      - ramfs: ramfs
      - sync: CameraFrameSync_0
      - cfg:
          tracker:
            require_target_tag: false
            target_tag_id: -1
            min_detect_count: 2
            max_temp_lost_count: 15
            outpost_max_temp_lost_count: 75
            target_select:
              observed_count_weight: 1.6
              distance_weight: 2.0
              area_weight: 1.2
              spin_weight: 0.8
              angle_weight: 2.0
              max_distance_m: 8.0
              distance_span_m: 7.5
              area_norm_px: 6000.0
              observed_count_norm: 4.0
              max_spin_rad_s: 8.0
              max_angle_norm: 0.5
              detecting_scale: 0.55
              temp_lost_scale: 0.35
              switch_margin: 0.25
          extrinsic:
            camera_mount_to_body:
              rotation: [1.0, 0.0, 0.0, 0.0]
              translation: [0.0, 0.0, 0.0]
          preview:
            enabled: true
            preview_window_name: "armor_tracker_preview"
            preview_scale: 0.5
            preview_wait_key_ms: 1
            queue_capacity: 1
            output_mode: "web"
            web_bind_address: "0.0.0.0"
            web_port: 8080
            web_stream_name: "armor_tracker"
            max_fps: 30.0
```

`CameraFrameSync_0` 与 ArmorDetector 实例在 `modules:` 中位于本实例之前，并使用相同的 `template_args`。Aimer 订阅本模块的 `target_frame`。

`CameraFrameSync_0` and the ArmorDetector instance are listed before this instance in `modules:` and use the same `template_args`. Aimer subscribes to the `target_frame` of this Module.

## 6. 依赖与硬件 / Dependencies and Hardware

依赖：

- `QDU-Robomaster/ArmorDetector`：检测结果类型与 `armors_frame` 输入。
- `QDU-Robomaster/CameraFrameSync`：原生标定来源与同步帧类型。
- `QDU-Robomaster/CameraBase`：帧布局、几何与共享图像类型。
- `QDU-Robomaster/VisionPreview`：跟踪结果预览。
- `xrobot-org/DurationStatistics`：worker 耗时统计。
- LibXR。
- OpenCV 4（`core`、`calib3d`、`imgproc`）与 Eigen。

硬件：由 CameraFrameSync 提供同步帧的相机与带姿态输出的 IMU，标定与帧布局须与相机输出一致。

Dependencies:

- `QDU-Robomaster/ArmorDetector`: detection result type and the `armors_frame` input.
- `QDU-Robomaster/CameraFrameSync`: source of the native calibration and the synchronized frame type.
- `QDU-Robomaster/CameraBase`: frame layout, geometry and shared image types.
- `QDU-Robomaster/VisionPreview`: preview of the tracking result.
- `xrobot-org/DurationStatistics`: worker service time statistics.
- LibXR.
- OpenCV 4 (`core`, `calib3d`, `imgproc`) and Eigen.

Hardware: a camera and an IMU with attitude output whose synchronized frames are provided by CameraFrameSync; the calibration and the frame layout match the camera output.
