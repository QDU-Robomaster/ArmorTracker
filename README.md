# ArmorTracker

`ArmorTracker` 是 Webots/Linux 自瞄链路里的目标级跟踪模块。输入来自 `ArmorDetector`
发布的检测结果（其中携带 `CameraFrameSync` 同步帧的图像和 IMU），输出 `tracker` 域的
`target_frame` 同帧目标包。云台角、发送包和开火判定由后级 Aimer 负责。

## 文件结构

- `ArmorTracker.hpp`：模块入口、配置、`target_frame` payload 和运行态成员。
- `ArmorTrackerPipeline.hpp`：detector topic 回调、worker、target_frame 发布、RamFS 命令和内置
  preview 绘制。
- `ArmorTrackerQueue.hpp`：回调与 worker 之间的固定槽队列。
- `ArmorTrackerFrameAdapter.hpp`：detector 结果到 tracker 输入的逐帧几何适配。
- `ArmorTrackerCore.hpp`：detector 输入到 tracker 输出的门面适配。
- `ArmorTrackerModel.hpp`：PnP、目标状态、整车 EKF 和跟踪状态机。
- `ArmorTrackerMath.hpp`：角度/坐标转换和 EKF 基础工具。
- `ArmorTrackerTarget.hpp`：`target_frame` 内携带的目标状态消息。
- `tools/coordinate_semantics_check.cpp`：公开坐标系姿态语义回归检查，防止 IMU 姿态被额外
  固定旋转翻转 roll/pitch。
- `tools/tracker_replay/armor_tracker_replay.cpp`：基于 TSV 的离线确定性重放工具。

`ArmorTracker` 主体是模板头文件实现，CMake 只暴露 include 目录，不编译额外 `.cpp` 源文件。

## 输入输出与运行时

- 输入：`armor_detector` 域的 `armors_frame`（`const DetectedFrame<FrameLayoutV>*`）。构造时
  等待该 topic 出现，因此 ArmorDetector 实例须先构造。
- 输出：`tracker` 域的 `target_frame`（`const TrackedFrame<FrameLayoutV>*`），Topic 时间戳为
  `SyncedFrame::imu.timestamp_us`。指针只在同步回调期间有效。

回调只复制 `SharedFrame` 所有权、IMU 和检测结果到 16 槽固定队列，不复制图像字节；队列满时
回调等待空槽（`OnMonitor()` 中的 `full_wait` 计数）。单个 worker 线程取出一帧，将所有权
移动到栈上 `TrackedFrame`，运行 tracker 后同步发布 `const TrackedFrame*`；逐帧
`FrameGeometry` 始终只从 `SharedFrame.Get()->geometry` 读取。几何非法的检测帧会被拒绝。

`TrackedFrame` 包含 `sequence`、`image`、`imu`、`target`（`ArmorTrackerTarget`）以及
`output_to_camera_rotation` / `output_to_camera_translation`（输出系 `O` 到 OpenCV 相机系的
变换，row-major，单位 m）。`ArmorTrackerTarget` 包含 `tracking`、目标编号 `id`、装甲面数量、
整车中心位置/速度（m、m/s）、`yaw` / `v_yaw`（rad、rad/s）、两组装甲半径、奇偶面高度差
`dz`、当前绑定的装甲面索引、前哨站高度相位和换面标志。

## 坐标与 PnP

`cfg.extrinsic.camera_mount_to_body` 是手眼外参，只表达相机安装坐标系 `M` 到公开本体系 `B`
的真实安装偏差。`M` 与 OpenCV 相机系 `C` 同原点，并与 `B` 使用同一轴约定：右手系，`x` 向右，
`y` 向前，`z` 向上。`C` 到 `M` 的固定轴变换由代码内部处理，不需要写进配置。`rotation` 为
`wxyz` 四元数，`translation` 单位为 m。

Tracker 在构造期从 `CameraFrameSync::Calibration()` 复制一份原生相机标定。Detector 发布的
四角点保持原生传感器坐标，Tracker 按装甲板类型使用 230 mm（大）或 135 mm（小）宽、56 mm
灯条长的模型和原生相机 K/D 独立执行 PnP、重投影，不复用 Detector 的 pose。支持无畸变、5 项
和 8 项畸变系数；需要预先去畸变的模型或无效标定会记录错误并禁用观测。PnP 失败或结果非有限时
不把该观测送入跟踪更新。

同步帧 IMU 四元数（与 `host` 域 `gimbal_quat` 相同，已是公开本体系 `B` 的姿态）作为本体到
世界的旋转，只归一化后直接转矩阵；任何额外的固定 basis 旋转都会让 roll/pitch 反号，并污染
输出目标高度。

输出统一使用与公开本体系 `B` 同向的惯性解算轴 `O`：右手系，`x` 向右，`y` 向前，`z` 向上；
yaw 以前向为 0，左转为正。`O` 的轴向不随当前云台 yaw 转动，因此后级 Aimer 由此解出的 yaw
是下位机可直接消费的绝对云台目标角。preview 使用 `output_to_camera` 把 `O` 中的目标几何投回
同帧相机图像。

输入时间戳回退时，TrackerCore 清除旧目标与滤波状态、重建时间基线，并使用不变的配置和标定
重新捕获目标。

## Target Selection

tracker 内部按装甲板编号维护多套车辆 EKF 状态，同一 slot 丢失后不会清空 EKF。同一帧里出现
多个编号时，各编号状态独立更新；`target_frame` 中只携带一个当前选择目标。当前选择分数使用
装甲板观测数量的低通值、距离、可打击面积、自旋速度和目标相对当前云台视轴的角度差，并用滞回
margin 避免输出目标抖动。可打击面积在 `NativeToFrame` 后计算，因此 2x wide 模式不会产生
4 倍面积偏置。候选的图像中心排序使用原生标定主点。

## Preview

内置 preview 只在 `cfg.preview.enabled: true` 时启动，不订阅 topic、不录像、不反压主链路。
它把 detector 原生角点和 tracker 原生重投影逆映射到当前帧后，绘制 detector 四边形、tracker
整车中心、四个装甲面中心、相邻装甲面连线，以及带固定倾角的装甲板物理框。多车跟踪时，preview
会绘制所有 active 车辆，并在车体中心标注编号和当前选择评分；被选中的车辆用更醒目的中心和
连线显示。detector preview 不在这里处理。

## RamFS 命令与监控

模块创建名为 `armor_tracker` 的 RamFS 命令文件：

```text
armor_tracker show                       # 打印当前 tracker / extrinsic 配置
armor_tracker target_tag_id <value>      # 只跟踪指定编号
armor_tracker require_target_tag <0|1>   # 是否要求目标编号匹配
```

修改在 worker 处理下一帧前生效（重新配置 tracker 并重启 preview）。

`OnMonitor()` 打印自上次调用以来的入队、覆盖、处理帧数，队列就绪/占用/高水位、`full_wait`
次数、平均生产者等待时间，以及 worker 单帧服务耗时统计。

## 依赖

- `QDU-Robomaster/ArmorDetector`：检测结果类型和 `armors_frame` 输入。
- `QDU-Robomaster/CameraFrameSync`：原生标定来源和同步帧类型。
- `QDU-Robomaster/VisionPreview`：跟踪结果预览。
- `xrobot-org/DurationStatistics`：worker 耗时统计。
- `QDU-Robomaster/CameraBase`：帧布局、geometry 与共享图像类型。
- 外部：OpenCV 4（`core`、`calib3d`、`imgproc`），Eigen。

## 构造接口

```cpp
template <CameraTypes::FrameLayout FrameLayoutV>
class ArmorTracker;

explicit ArmorTracker(
    LibXR::RamFS& ramfs,
    FrameSync& sync,
    Config cfg = DefaultConfig());
```

模板参数：

- `FrameLayoutV`：帧布局，必须与上游相机、CameraFrameSync 和 ArmorDetector 相同。

依赖：

- `ramfs`：`LibXR::RamFS`，注册 `armor_tracker` 命令文件。
- `sync`：`CameraFrameSync<FrameLayoutV>&`，只用于在构造时复制原生标定。

配置 `cfg`（`Config`，`DefaultConfig()` 即全部默认值）：

- `tracker.require_target_tag`：是否只跟踪 `target_tag_id`，默认 `false`。
- `tracker.target_tag_id`：指定目标编号，默认 `-1`。
- `tracker.min_detect_count`：从检测态进入跟踪所需的检测次数，默认 `2`。
- `tracker.max_temp_lost_count`：暂时丢失帧数上限，默认 `15`。
- `tracker.outpost_max_temp_lost_count`：前哨站暂时丢失帧数上限，默认 `75`。
- `tracker.target_select`：多车选择评分，默认 `observed_count_weight = 1.6`、
  `distance_weight = 2.0`、`area_weight = 1.2`、`spin_weight = 0.8`、`angle_weight = 2.0`、
  `max_distance_m = 8.0`、`distance_span_m = 7.5`、`area_norm_px = 6000.0`、
  `observed_count_norm = 4.0`、`max_spin_rad_s = 8.0`、`max_angle_norm = 0.5`、
  `detecting_scale = 0.55`、`temp_lost_scale = 0.35`、`switch_margin = 0.25`。
- `extrinsic.camera_mount_to_body.rotation`：`wxyz` 四元数，默认 `[1, 0, 0, 0]`。
- `extrinsic.camera_mount_to_body.translation`：单位 m，默认 `[0, 0, 0]`。
- `preview`：`VisionPreview::RuntimeParam`，默认关闭，字段见 VisionPreview。

## 使用

```sh
xrobot module add QDU-Robomaster/ArmorTracker
xrobot setup
xrobot instance add QDU-Robomaster/ArmorTracker
```

`xrobot instance add` 在 `User/xrobot.yaml` 中写入一个实例，依赖项留空，默认值按源码写出；
把 `ramfs` 填为 BSP 中用 `XR_REGISTER` 注册的 RamFS 对象名，`sync` 填为前面 CameraFrameSync
实例的 id。帧布局用 constexpr 定义，必须与相机输出一致：

```yaml
constexpr_includes:
  - CameraBase.hpp
constexprs:
  FrameLayout:
    type: CameraTypes::FrameLayout
    value: '{.width = 640, .height = 480, .step = 1920, .encoding = CameraTypes::Encoding::BGR8}'
modules:
  - module: QDU-Robomaster/ArmorTracker
    id: armortracker_0
    template_args:
      - ProjectConstexpr::FrameLayout
    args:
      - ramfs: ramfs
      - sync: cameraframesync_0
      - cfg: ArmorTracker<ProjectConstexpr::FrameLayout>::DefaultConfig()
```

BSP 侧：

```cpp
XR_REGISTER(ramfs, LibXR::RamFS);
```

`cameraframesync_0` 是 CameraFrameSync 实例的 id；它和 ArmorDetector 实例都必须在 `modules:`
中列在本实例之前，并使用同一个 `template_args`。Aimer 订阅本模块的 `target_frame`。

`cfg` 也可以写成 YAML map（字段名同上，字符串写成 C++ 字符串字面量），例如填写外参并打开
Web 预览：

```yaml
cfg:
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
    preview_window_name: '"armor_tracker_preview"'
    preview_scale: 0.5
    preview_wait_key_ms: 1
    queue_capacity: 1
    output_mode: '"web"'
    web_bind_address: '"0.0.0.0"'
    web_port: 8080
    web_stream_name: '"armor_tracker"'
    max_fps: 30.0
```

填好后再次运行 `xrobot setup`，生成 `User/xrobot_main.hpp`。

`xrobot module show .`（在本仓库中）或 `xrobot module show Modules/QDU-Robomaster/ArmorTracker`
（在 BSP 中）打印当前的构造函数。

## 验证

在打开 `BUILD_TESTING` 的 BSP 构建中，本模块加入 `armor_tracker_frame_geometry_test`、
`armor_tracker_distortion_projection_test`、`armor_tracker_queue_contract_test` 和
`armor_tracker_stage_frame_contract_test`，用 `ctest` 运行。

标定尺寸与内参的一致性由 CameraBase 校验；内部 PnP 求解器只检查数值、模型支持和求解结果，
不要求离线角点回放额外提供图像宽高。

单车过滤回归使用 `tools/tracker_replay/armor_tracker_replay.cpp` 对固定数据集重放；多车目标
选择需要使用不按编号过滤的 replay，确认同帧多编号输入会独立更新各 slot 并只输出当前选中的
目标。坐标语义回归至少需要覆盖 `tools/coordinate_semantics_check.cpp`。
