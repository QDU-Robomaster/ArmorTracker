#pragma once

/**
 * @file ArmorTracker.hpp
 * @brief ArmorTracker module interface, configuration, and topic payloads.
 */

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: Armor tracker
constructor_args:
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
      enabled: false
      preview_window_name: "armor_tracker_preview"
      preview_scale: 0.5
      preview_wait_key_ms: 1
      queue_capacity: 1
      output_mode: "window"
      web_bind_address: "0.0.0.0"
      web_port: 8080
      web_stream_name: "armor_tracker"
      max_fps: 30.0
  sync: '@camera_frame_sync'
template_args:
  - Layout:
      width: 1280
      height: 720
      step: 3840
      encoding: CameraTypes::Encoding::BGR8
required_hardware: []
depends:
  - qdu-future/ArmorDetector
  - qdu-future/CameraFrameSync
  - qdu-future/VisionPreview
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
#include "AutoAimReplayBenchmark.hpp"
#include "CameraFrameSync.hpp"
#include "VisionPreview.hpp"
#include "app_framework.hpp"
#include "libxr_time.hpp"
#include "logger.hpp"
#include "message.hpp"
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
 * @brief Application module that tracks RoboMaster armor targets.
 *
 * The module subscribes to detector frame packets, runs the header-only tracker
 * core, publishes the same-frame target packet, and optionally submits a built-in
 * preview overlay.
 */
template <CameraTypes::FrameLayout FrameLayoutV>
class ArmorTracker : public LibXR::Application
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
   * @brief Runtime configuration loaded from the module YAML config.
   */
  struct Config
  {
    /**
     * @brief Core target selection and state-machine parameters.
     */
    struct TrackerParams
    {
      bool require_target_tag = false;
      int target_tag_id = -1;
      int min_detect_count = 2;
      int max_temp_lost_count = 15;
      int outpost_max_temp_lost_count = 75;
      struct TargetSelectParams
      {
        double observed_count_weight = 1.6;
        double distance_weight = 2.0;
        double area_weight = 1.2;
        double spin_weight = 0.8;
        double angle_weight = 2.0;
        double max_distance_m = 8.0;
        double distance_span_m = 7.5;
        double area_norm_px = 6000.0;
        double observed_count_norm = 4.0;
        double max_spin_rad_s = 8.0;
        double max_angle_norm = 0.5;
        double detecting_scale = 0.55;
        double temp_lost_scale = 0.35;
        double switch_margin = 0.25;
      } target_select;
    } tracker;

    /**
     * @brief Camera mounting extrinsic expressed in public body frame B.
     */
    struct ExtrinsicParams
    {
      /**
       * @brief Transform from camera mount frame M to body frame B.
       *
       * Rotation is a unit quaternion in wxyz order and translation is in
       * meters. M shares the OpenCV camera origin and uses the same axis
       * convention as B: x right, y forward, z up. The fixed OpenCV camera
       * frame C to mount frame M axis conversion is handled inside
       * ArmorTracker and is not part of this user config.
       */
      struct CameraMountToBody
      {
        std::array<double, 4> rotation = {1.0, 0.0, 0.0, 0.0};
        std::array<double, 3> translation = {0.0, 0.0, 0.0};
      } camera_mount_to_body;
    } extrinsic;

    VisionPreview::RuntimeParam preview{};
  };

  struct PipelineMetrics
  {
    uint64_t enqueued{0};
    uint64_t overwritten{0};
    uint64_t processed{0};
    std::size_t queue_ready{0};
    std::size_t queue_occupied{0};
    std::size_t queue_high_water{0};
    std::size_t image_storage_bytes{0};
    std::size_t slot_storage_bytes{0};
    uint64_t queue_full_waits{0};
    uint64_t producer_wait_us{0};
    uint64_t worker_service_us{0};
    bool producer_active{false};
    bool worker_active{false};
  };

  /**
   * @brief Tracker worker input retained in the bounded FIFO.
   *
   * The callback copies SharedFrame ownership, IMU, and detections. Pixel storage is
   * never copied.
   */
  struct PendingDetectionFrame
  {
    uint64_t sequence{0};
    uint64_t admission_sequence{0};
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
   * @brief Construct the module and subscribe to the detector topic.
   */
  explicit ArmorTracker(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app,
                        Config cfg, FrameSync* sync);

  explicit ArmorTracker(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app,
                        Config cfg, FrameSync& sync);

  /**
   * @brief RamFS command entry used to show or update selected tracker params.
   */
  static int CommandFun(ArmorTracker* self, int argc, char** argv);

  /**
   * @brief Get the current runtime configuration.
   */
  const Config& GetConfig() const { return cfg_; }

  /**
   * @brief Replace runtime configuration and restart dependent runtime services.
   */
  void SetConfig(const Config& cfg);

  /** Return true after every enqueued frame is processed and the worker is idle. */
  [[nodiscard]] bool PipelineDrained() const noexcept;

  /** Return the current bounded Tracker pipeline counters and queue state. */
  [[nodiscard]] PipelineMetrics GetPipelineMetrics() const;

  /**
   * @brief C-style adapter for the RamFS command callback.
   */
  static int CommandAdapter(void* instance, int argc, char** argv)
  {
    return CommandFun(static_cast<ArmorTracker*>(instance), argc, argv);
  }

  /**
   * @brief Application monitor hook; tracker has no periodic monitor work.
   */
  void OnMonitor() override;

 private:
  static constexpr const char* kDetectorTopicName = "armors_frame";

  /**
   * @brief Convert module config into the internal tracker core config.
   */
  armor_tracker_detail::Config BuildTrackerConfig() const;

  /**
   * @brief Process one detector frame packet and publish tracker outputs.
   */
  void ArmorsCallback(DetectionMessage message);

  /**
   * @brief 异步 worker 入口：等待 callback 拷贝完成的检测帧。
   */
  static void TrackerWorkerThreadFun(ArmorTracker* self);

  /**
   * @brief 在异步 worker 中处理一帧自有检测数据。
   */
  void ProcessPendingDetectionFrame(PendingDetectionFrame& frame);

  /**
   * @brief Resolve and subscribe to the detector frame topic.
   */
  void SubscribeDetectorTopic();

  /**
   * @brief Submit a preview overlay job when the preview runtime is enabled.
   */
  void SubmitPreview(const ImageFrame& image_frame,
                     const ArmorDetectorResults& detector_armors,
                     const ArmorTrackerTarget& target_msg,
                     const armor_tracker_detail::Output& output);

  static CameraCalibration CopyCalibration(FrameSync* sync)
  {
    ASSERT(sync != nullptr);
    return sync->Calibration();
  }

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
  std::atomic<uint64_t> admission_sequence_count_{0};
  std::atomic<uint64_t> worker_sequence_count_{0};
  std::atomic<uint64_t> enqueued_frame_count_{0};
  std::atomic<uint64_t> overwritten_frame_count_{0};
  std::atomic<uint64_t> processed_frame_count_{0};
  std::atomic<uint64_t> process_time_us_accum_{0};
  uint64_t last_monitor_enqueued_{0};
  uint64_t last_monitor_overwritten_{0};
  uint64_t last_monitor_processed_{0};
  uint64_t last_monitor_process_time_us_{0};
  uint64_t last_monitor_full_wait_count_{0};
  uint64_t last_monitor_producer_wait_ns_{0};
};

#include "ArmorTrackerPipeline.hpp"
