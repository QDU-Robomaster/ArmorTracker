#pragma once

/**
 * @file ArmorTracker.hpp
 * @brief ArmorTracker 模块接口、配置与 Topic 载荷。
 *        ArmorTracker Module interface, configuration and Topic payloads.
 */

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 装甲板目标级跟踪：由检测结果和同帧 IMU 姿态维护整车 EKF 并发布同帧目标 / Armor target tracking Module that maintains per-vehicle EKF states from detections and same-frame IMU attitude and publishes the same-frame target
depends:
- id: QDU-Robomaster/ArmorDetector
  ref: same-or-dev
- id: QDU-Robomaster/CameraFrameSync
  ref: same-or-dev
- id: QDU-Robomaster/VisionPreview
  ref: same-or-dev
- id: xrobot-org/DurationStatistics
  ref: same-or-dev
- id: QDU-Robomaster/CameraBase
  ref: same-or-dev
=== END MANIFEST === */
// clang-format on

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ArmorDetectorTypes.hpp"
#include "ArmorTrackerCore.hpp"
#include "ArmorTrackerFrameAdapter.hpp"
#include "ArmorTrackerQueue.hpp"
#include "ArmorTrackerTarget.hpp"
#include "CameraFrameSync.hpp"
#include "DurationStatistics.hpp"
#include "VisionPreview.hpp"
#include "libxr_def.hpp"
#include "libxr_time.hpp"
#include "logger.hpp"
#include "message.hpp"
#include "ramfs.hpp"
#include "timebase.hpp"

#if defined(__has_include)
#if __has_include("print/print_api.hpp")
#define TRACKER_STDIO_HAS_COMPILED_PRINTF 1
#endif
#endif

#ifndef TRACKER_STDIO_HAS_COMPILED_PRINTF
#define TRACKER_STDIO_HAS_COMPILED_PRINTF 0
#endif

#if TRACKER_STDIO_HAS_COMPILED_PRINTF
#define TRACKER_STDIO_PRINT(format_literal) LibXR::STDIO::Printf<format_literal>()
#define TRACKER_STDIO_PRINTF(format_literal, ...) \
  LibXR::STDIO::Printf<format_literal>(__VA_ARGS__)
#else
#define TRACKER_STDIO_PRINT(format_literal) LibXR::STDIO::Printf(format_literal)
#define TRACKER_STDIO_PRINTF(format_literal, ...) \
  LibXR::STDIO::Printf(format_literal, __VA_ARGS__)
#endif

/**
 * @brief 装甲板目标级跟踪模块。
 *        Armor target tracking Module.
 *
 * 订阅 ArmorDetector 发布的检测帧，按装甲板编号维护整车 EKF 状态，在 `tracker` 域
 * 发布同帧目标包 `target_frame`，并按配置输出内置预览。
 * Subscribes to the detection frames published by ArmorDetector, maintains an EKF state
 * per vehicle by armor number, publishes the same-frame target packet `target_frame` in
 * the `tracker` domain, and outputs the built-in preview when enabled.
 *
 * @tparam FrameLayoutV 帧布局，与上游相机、CameraFrameSync 和 ArmorDetector 相同。
 *                      Frame layout, identical to the upstream camera, CameraFrameSync
 *                      and ArmorDetector.
 */
template <CameraTypes::FrameLayout FrameLayoutV>
class ArmorTracker
{
 public:
  using FrameSync = CameraFrameSync<FrameLayoutV>;
  using Base = typename FrameSync::Base;
  using CameraCalibration = CameraTypes::CameraCalibration;
  using FrameGeometry = CameraTypes::FrameGeometry;
  using ImageFrame = typename FrameSync::ImageFrame;
  using SharedFrame = typename FrameSync::SharedFrame;
  using ImuStamped = typename FrameSync::ImuStamped;
  using DetectionFrame = DetectedFrame<FrameLayoutV>;
  using DetectionMessage = DetectedFrameMessage<FrameLayoutV>;
  using TargetFrame = TrackedFrame<FrameLayoutV>;
  using TargetFrameMessage = TrackedFrameMessage<FrameLayoutV>;

  static inline constexpr auto frame_layout = Base::frame_layout;
  static inline constexpr std::size_t pending_frame_capacity = 16U;

  /**
   * @brief 模块配置。
   *        Module configuration.
   */
  struct Config
  {
    /**
     * @brief 目标选择与状态机参数。
     *        Target selection and state-machine parameters.
     */
    struct TrackerParams
    {
      bool require_target_tag = false;       ///< 只选择 target_tag_id 对应的编号
                                             ///< Select only target_tag_id
      int target_tag_id = -1;                ///< 指定的目标编号
                                             ///< Designated target number
      int min_detect_count = 2;              ///< 进入跟踪态所需的检测次数
                                             ///< Detections needed to start tracking
      int max_temp_lost_count = 15;          ///< 暂时丢失的帧数上限
                                             ///< Max temporarily lost frames
      int outpost_max_temp_lost_count = 75;  ///< 前哨站暂时丢失的帧数上限
                                             ///< Max temporarily lost frames (outpost)

      /**
       * @brief 多车目标选择评分参数。
       *        Target selection score parameters for multiple vehicles.
       */
      struct TargetSelectParams
      {
        double observed_count_weight = 1.6;  ///< 近期可见装甲数量评分的权重
                                             ///< Weight of visible armor count score
        double distance_weight = 2.0;        ///< 距离评分的权重
                                             ///< Weight of distance score
        double area_weight = 1.2;            ///< 可打击面积评分的权重
                                             ///< Weight of hittable area score
        double spin_weight = 0.8;            ///< 自旋评分的权重
                                             ///< Weight of spin score
        double angle_weight = 2.0;           ///< 视轴角差评分的权重
                                             ///< Weight of optical-axis angle score
        double max_distance_m = 8.0;         ///< 距离评分满分对应的距离 (m)
                                             ///< Distance of full score (m)
        double distance_span_m = 7.5;        ///< 距离评分降到 0 的跨度 (m)
                                             ///< Span down to score 0 (m)
        double area_norm_px = 6000.0;        ///< 面积评分的归一化面积 (px)
                                             ///< Normalization area (px)
        double observed_count_norm = 4.0;    ///< 数量评分的归一化数量
                                             ///< Normalization count
        double max_spin_rad_s = 8.0;         ///< 自旋评分的归一化角速度 (rad/s)
                                             ///< Normalization rate (rad/s)
        double max_angle_norm = 0.5;         ///< 视轴角差评分的归一化角差
                                             ///< Normalization axis angle
        double detecting_scale = 0.55;       ///< detecting 状态的分数倍率
                                             ///< Scale in detecting state
        double temp_lost_scale = 0.35;       ///< temp_lost 状态的分数倍率
                                             ///< Scale in temp_lost state
        double switch_margin = 0.25;         ///< 替换当前目标所需的最小分差
                                             ///< Min gap to replace target
      } target_select;                       ///< 多车目标选择评分参数
                                             ///< Selection score parameters
      bool use_aaest = false;                ///< 四块装甲板的车辆改由 aaest 输出
                                             ///< Report four-plate vehicles from aaest
      double aaest_latency_s = 0.07;         ///< aaest 速率平均窗口的固定部分 (s)
                                             ///< Fixed part of the aaest rate window (s)
      double aaest_bullet_speed_m_s = 23.0;  ///< aaest 速率窗口的飞行时间所用弹速 (m/s)
                                             ///< Bullet speed for the flight time (m/s)
    } tracker;                               ///< 目标选择与状态机参数
                                             ///< Selection and state machine

    /**
     * @brief 相机安装外参，表达在公开本体系 B 中。
     *        Camera mounting extrinsic expressed in the public body frame B.
     */
    struct ExtrinsicParams
    {
      /**
       * @brief 相机安装系 M 到本体系 B 的变换。
       *        Transform from the camera mount frame M to the body frame B.
       *
       * M 与 OpenCV 相机系 C 同原点，并与 B 使用同一轴约定：x 向右，y 向前，z 向上。
       * C 到 M 的固定轴变换由模块内部完成。
       * M shares its origin with the OpenCV camera frame C and uses the same axis
       * convention as B: x right, y forward, z up. The fixed axis conversion from C to
       * M is done inside the Module.
       */
      struct CameraMountToBody
      {
        /// 单位四元数，wxyz 顺序
        /// Unit quaternion in wxyz order
        std::array<double, 4> rotation = {1.0, 0.0, 0.0, 0.0};
        /// 平移 (m)
        /// Translation (m)
        std::array<double, 3> translation = {0.0, 0.0, 0.0};
      } camera_mount_to_body;  ///< 安装系 M 到本体系 B 的变换
                               ///< Transform from M to B
    } extrinsic;               ///< 相机安装外参
                               ///< Camera mounting extrinsic

    /// 内置预览参数，默认关闭
    /// Built-in preview parameters, disabled by default
    VisionPreview::RuntimeParam preview{.preview_window_name = "armor_tracker_preview",
                                        .preview_scale = 0.5,
                                        .web_stream_name = "armor_tracker"};
  };

  /**
   * @brief 流水线计数与队列状态。
   *        Pipeline counters and queue state.
   */
  struct PipelineMetrics
  {
    uint64_t enqueued{0};                ///< 累计入队帧数
                                         ///< Frames enqueued in total
    uint64_t overwritten{0};             ///< 累计覆盖帧数
                                         ///< Frames overwritten in total
    uint64_t processed{0};               ///< 累计处理帧数
                                         ///< Frames processed in total
    std::size_t queue_ready{0};          ///< 等待 worker 处理的槽数
                                         ///< Slots waiting for the worker
    std::size_t queue_occupied{0};       ///< 被占用的槽数
                                         ///< Occupied slots
    std::size_t queue_high_water{0};     ///< 占用槽数的历史最大值
                                         ///< Highest occupied slot count
    std::size_t image_storage_bytes{0};  ///< 队列自有的图像存储字节数
                                         ///< Bytes of image storage owned by the queue
    std::size_t slot_storage_bytes{0};   ///< 全部槽占用的字节数
                                         ///< Bytes occupied by all slots
    uint64_t queue_full_waits{0};        ///< 回调因队列已满而等待的次数
                                         ///< Times the callback waited on a full queue
    uint64_t producer_wait_us{0};        ///< 回调等待空槽的累计时间 (us)
                                         ///< Callback wait for a slot, total (us)
    uint64_t worker_service_us{0};       ///< worker 处理帧的累计耗时 (us)
                                         ///< Worker processing time, total (us)
    bool producer_active{false};         ///< 回调正在写入一个槽
                                         ///< The callback is writing a slot
    bool worker_active{false};           ///< worker 正在处理一个槽
                                         ///< The worker is processing a slot
  };

  /**
   * @brief 保存在有界 FIFO 中的 worker 输入。
   *        Worker input retained in the bounded FIFO.
   *
   * 回调复制 SharedFrame 所有权、IMU 和检测结果，像素数据保持共享。
   * The callback copies the SharedFrame ownership, the IMU and the detections; the pixel
   * data stays shared.
   */
  struct PendingDetectionFrame
  {
    uint64_t sequence{0};
    SharedFrame image{};
    ImuStamped imu{};
    ArmorDetectorResults detections{};

    PendingDetectionFrame() = default;
    PendingDetectionFrame(const PendingDetectionFrame&) = delete;
    PendingDetectionFrame& operator=(const PendingDetectionFrame&) = delete;
    PendingDetectionFrame(PendingDetectionFrame&&) = delete;
    PendingDetectionFrame& operator=(PendingDetectionFrame&&) = delete;
  };

  /**
   * @brief 返回全部默认值的配置。
   *        Return the configuration holding all defaults.
   *
   * 预览默认关闭，窗口名为 `armor_tracker_preview`，缩放为 0.5，Web 流名为
   * `armor_tracker`。
   * The preview is disabled by default, with window name `armor_tracker_preview`,
   * scale 0.5 and web stream name `armor_tracker`.
   *
   * @return 默认配置。
   *         Default configuration.
   */
  static Config DefaultConfig() { return {}; }

  /**
   * @brief 构造 ArmorTracker，注册命令文件，启动 worker 线程并订阅检测帧 Topic。
   *        Construct ArmorTracker, register the command file, start the worker thread
   *        and subscribe to the detection frame Topic.
   *
   * 构造时阻塞等待 `armor_detector` 域的 `armors_frame` 出现。
   * The constructor blocks until `armors_frame` appears in the `armor_detector` domain.
   *
   * @param ramfs 用于注册命令文件 `armor_tracker` 的 RamFS。
   *              RamFS on which the command file `armor_tracker` is registered.
   * @param sync CameraFrameSync 实例，构造时从中复制原生相机标定。
   *             CameraFrameSync instance from which the native camera calibration is
   *             copied at construction.
   * @param cfg 模块配置。
   *            Module configuration.
   */
  explicit ArmorTracker(LibXR::RamFS& ramfs, FrameSync& sync,
                        Config cfg = DefaultConfig());

  /**
   * @brief RamFS 命令入口，用于显示配置或修改 target_tag_id 与 require_target_tag。
   *        RamFS command entry that shows the configuration or sets target_tag_id and
   *        require_target_tag.
   *
   * @param self ArmorTracker 实例。
   *             ArmorTracker instance.
   * @param argc 参数个数。
   *             Argument count.
   * @param argv 参数列表。
   *             Argument list.
   * @return 成功为 0，命令无法识别为 -1。
   *         0 on success, -1 when the command is not recognized.
   */
  static int CommandFun(ArmorTracker* self, int argc, char** argv);

  /**
   * @brief 获取当前配置。
   *        Get the current configuration.
   *
   * @return 当前配置的引用。
   *         Reference to the current configuration.
   */
  const Config& GetConfig() const { return cfg_; }

  /**
   * @brief 替换配置，重新配置 tracker 并重启预览。
   *        Replace the configuration, reconfigure the tracker and restart the preview.
   *
   * @param cfg 新配置。
   *            New configuration.
   */
  void SetConfig(const Config& cfg);

  /**
   * @brief 判断所有入队帧是否均已处理且 worker 空闲。
   *        Check whether every enqueued frame has been processed and the worker is idle.
   *
   * @return 流水线已排空为 true。
   *         True when the pipeline is drained.
   */
  [[nodiscard]] bool PipelineDrained() const noexcept;

  /**
   * @brief 获取当前流水线计数与队列状态。
   *        Get the current pipeline counters and queue state.
   *
   * @return 流水线计数与队列状态。
   *         Pipeline counters and queue state.
   */
  [[nodiscard]] PipelineMetrics GetPipelineMetrics() const;

  /**
   * @brief RamFS 命令回调的 C 风格适配函数。
   *        C-style adapter for the RamFS command callback.
   *
   * @param instance ArmorTracker 实例指针。
   *                 Pointer to the ArmorTracker instance.
   * @param argc 参数个数。
   *             Argument count.
   * @param argv 参数列表。
   *             Argument list.
   * @return CommandFun 的返回值。
   *         Return value of CommandFun.
   */
  static int CommandAdapter(void* instance, int argc, char** argv)
  {
    return CommandFun(static_cast<ArmorTracker*>(instance), argc, argv);
  }

  /**
   * @brief 输出自上次调用以来的队列状态和 worker 单帧服务耗时统计。
   *        Print the queue state since the previous call and the worker per-frame
   *        service time statistics.
   */
  void OnMonitor();

 private:
  static constexpr const char* kDetectorTopicName = "armors_frame";

  /**
   * @brief 把模块配置转换为 tracker 内核配置。
   *        Convert the Module configuration into the tracker core configuration.
   */
  armor_tracker_detail::Config BuildTrackerConfig() const;

  /**
   * @brief 校验一帧检测结果并复制到队列。
   *        Validate one detection frame and copy it into the queue.
   */
  void ArmorsCallback(DetectionMessage message);

  /**
   * @brief worker 线程入口，依次处理队列中的检测帧。
   *        Worker thread entry that processes the queued detection frames in order.
   */
  static void TrackerWorkerThreadFun(ArmorTracker* self);

  /**
   * @brief 在 worker 中运行 tracker 并发布目标帧。
   *        Run the tracker in the worker and publish the target frame.
   */
  void ProcessPendingDetectionFrame(PendingDetectionFrame& frame);

  /**
   * @brief 等待并订阅检测帧 Topic。
   *        Wait for and subscribe to the detection frame Topic.
   */
  void SubscribeDetectorTopic();

  /**
   * @brief 预览启用时提交一帧叠加绘制任务。
   *        Submit one overlay drawing job when the preview is enabled.
   */
  void SubmitPreview(const ImageFrame& image_frame,
                     const ArmorDetectorResults& detector_armors,
                     const ArmorTrackerTarget& target_msg,
                     const armor_tracker_detail::Output& output);

  static CameraCalibration CopyCalibration(FrameSync& sync) { return sync.Calibration(); }

  Config cfg_;
  const CameraCalibration calibration_;
  armor_tracker_detail::TrackerCore tracker_{};
  VisionPreview preview_{};

  std::optional<LibXR::Topic::Domain> armor_detector_domain_{};
  std::optional<LibXR::Topic::Domain> tracker_domain_{};
  LibXR::Topic armors_topic_ = LibXR::Topic();
  LibXR::Topic target_frame_topic_ = LibXR::Topic();

  const char* name_ = "armor_tracker";
  std::optional<LibXR::RamFS::File> cmd_file_{};
  std::atomic<bool> params_is_changed_{false};
  armor_tracker_pipeline::FixedSlotQueue<PendingDetectionFrame, pending_frame_capacity>
      pending_frames_;
  std::atomic<uint64_t> enqueued_frame_count_{0};
  std::atomic<uint64_t> overwritten_frame_count_{0};
  std::atomic<uint64_t> processed_frame_count_{0};
  std::atomic<uint64_t> process_time_us_accum_{0};
  XRobot::DurationStatistics worker_service_duration_{};
  uint64_t last_monitor_enqueued_{0};
  uint64_t last_monitor_overwritten_{0};
  uint64_t last_monitor_processed_{0};
  uint64_t last_monitor_full_wait_count_{0};
  uint64_t last_monitor_producer_wait_ns_{0};
};

#include "ArmorTrackerPipeline.hpp"
