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
      output_frame: 1

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
  - Info:
      width: 1280
      height: 720
      step: 3840
      encoding: CameraTypes::Encoding::BGR8
      camera_matrix: [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
      distortion_model: CameraTypes::DistortionModel::PLUMB_BOB
      distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
      rectification_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      projection_matrix: [800.0, 0.0, 640.0, 0.0, 0.0, 800.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0]
required_hardware: []
depends:
  - qdu-future/ArmorDetector
  - qdu-future/CameraFrameSync
  - qdu-future/VisionPreview
=== END MANIFEST === */
// clang-format on

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "ArmorDetectorTypes.hpp"
#include "ArmorTrackerCore.hpp"
#include "ArmorTrackerTarget.hpp"
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
 * core, publishes target/debug topics, and optionally submits a built-in preview
 * overlay. Topic payloads stay plain aggregate types for SharedTopic recorders.
 */
template <CameraTypes::CameraInfo CameraInfoV>
class ArmorTracker : public LibXR::Application
{
 public:
  using FrameSync = CameraFrameSync<CameraInfoV>;
  using Base = typename FrameSync::Base;
  using CameraInfo = typename Base::CameraInfo;
  using ImageFrame = typename FrameSync::ImageFrame;
  using DetectionPacket = ArmorDetectionsFramePacket<CameraInfoV>;
  using DetectionMessage = ArmorDetectionsFrameMessage<CameraInfoV>;
  using DetectionMessageArg = typename std::conditional<
      std::is_pointer<DetectionMessage>::value, DetectionMessage,
      const DetectionMessage&>::type;

  static inline constexpr CameraInfo kCameraInfo = CameraInfoV;

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
      int output_frame = 1;
    } tracker;

    VisionPreview::RuntimeParam preview{};
  };

  /**
   * @brief Compact topic payload with the selected face position and yaw.
   */
  struct TrackerInfo
  {
    double position_diff{};
    double yaw_diff{};
    LibXR::Position<double> position{};
    double yaw{};
  };

  /**
   * @brief Projected target center and armor-face anchors for preview/debug.
   */
  struct EkfPointsMsg
  {
    uint64_t image_timestamp_us{};
    uint8_t count{};
    LibXR::Position<double> center_cam{};
    LibXR::Position<double> armors_cam[4]{};
    bool valid[5]{};
  };

  /**
   * @brief Per-detection candidate metadata kept for recorder compatibility.
   */
  struct CandidateDebugItem
  {
    uint8_t armor_index{};
    uint8_t face_index{};
    uint8_t same_number{};
    uint8_t reserved0{};
    int16_t image_track_id{-1};
    uint8_t image_track_confirmed{};
    uint8_t same_persistent_track{};
    ArmorNumber number{ArmorNumber::INVALID};
    ArmorType type{ArmorType::INVALID};
    uint8_t reserved1{};
    uint8_t reserved2{};
    float score{};
    float position_diff{};
    float yaw_diff{};
    float view_bonus{};
    float area_score{};
    float frontality{};
    float observation_quality_penalty{};
    float center_x{};
    float center_y{};
    float predicted_yaw{};
    float measured_yaw{};
  };

  /**
   * @brief Candidate/debug topic payload consumed by preview and record tools.
   *
   * Several fields are zero-filled by the simplified tracker. They remain in the
   * payload because current BSP previews and offline recorders use this topic
   * contract by structure size and field offsets.
   */
  struct CandidateDebugMsg
  {
    static constexpr uint8_t kMaxItems = 24;
    static constexpr uint8_t kMaxDetections = 8;

    uint64_t image_timestamp_us{};
    uint8_t count{};
    uint8_t selected_index{255};
    uint8_t matched{};
    uint8_t accepted_mode{};
    uint8_t detection_count{};
    int8_t preferred_adjacent_face{-1};
    uint8_t tracked_armors_num{};
    uint8_t has_same_number_candidate{};
    uint8_t face_switch_enabled{};
    uint8_t relaxed_face_switch_enabled{};
    uint8_t odd_face_switch_enabled{};
    uint8_t view_priority_enabled{};
    uint8_t directional_face_switch_enabled{};
    uint8_t tracked_face_track_id_valid{};
    int16_t tracked_face_track_id{-1};
    float predicted_vyaw{};
    float max_match_distance{};
    float max_match_yaw_diff{};
    float relaxed_same_face_distance{};
    float relaxed_face_switch_distance{};
    float relaxed_face_switch_yaw_diff{};
    float face_switch_score_deadzone{};
    float face_switch_position_deadzone{};
    float face_switch_yaw_deadzone{};
    float face_switch_timeout_sec{};
    float face_switch_cooldown_remaining{};
    float best_same_face_score{};
    float best_switch_face_score{};
    uint8_t same_face_matched{};
    uint8_t switch_face_matched{};
    uint8_t switch_blocked_by_timeout{};
    uint8_t switch_allowed{};
    uint8_t ekf_update_valid{};
    uint8_t ekf_update_mode{};
    int8_t ekf_update_face{-1};
    uint8_t ekf_freeze_delta_z{};
    uint8_t ekf_range_clamped{};
    float ekf_raw_range_m{};
    float ekf_range_m{};
    float ekf_mahalanobis{};
    float ekf_pre_res_x{};
    float ekf_pre_res_y{};
    float ekf_pre_res_z{};
    float ekf_pre_res_norm{};
    float ekf_post_res_x{};
    float ekf_post_res_y{};
    float ekf_post_res_z{};
    float ekf_post_res_norm{};
    float ekf_innov_0{};
    float ekf_innov_1{};
    float ekf_innov_2{};
    float ekf_innov_3{};
    float ekf_r_0{};
    float ekf_r_1{};
    float ekf_r_2{};
    float ekf_r_3{};
    std::array<int16_t, kMaxDetections> detection_track_ids{};
    std::array<uint8_t, kMaxDetections> detection_track_confirmed{};
    CandidateDebugItem items[kMaxItems]{};
  };

  /**
   * @brief Construct the module and subscribe to the detector topic.
   */
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
  void ArmorsCallback(DetectionMessageArg message);

  /**
   * @brief Resolve and subscribe to the detector result topic.
   */
  void SubscribeDetectorTopic();

  /**
   * @brief Submit a preview overlay job when the preview runtime is enabled.
   */
  void SubmitPreview(const ImageFrame& image_frame,
                     const ArmorDetectorResults& detector_armors,
                     const ArmorTrackerTarget& target_msg,
                     const CandidateDebugMsg& candidate_debug_msg,
                     const armor_tracker_detail::Output& output);

  Config cfg_;
  armor_tracker_detail::TrackerCore tracker_{};
  VisionPreview preview_{};

  LibXR::Topic::Domain armor_detector_domain_ =
      LibXR::Topic::Domain("armor_detector");
  LibXR::Topic::Domain tracker_domain_ = LibXR::Topic::Domain("tracker");
  LibXR::Topic armors_topic_ = LibXR::Topic();
  LibXR::Topic info_topic_ =
      LibXR::Topic("info", sizeof(TrackerInfo), &tracker_domain_);
  LibXR::Topic target_topic_ =
      LibXR::Topic("target", sizeof(ArmorTrackerTarget), &tracker_domain_);
  LibXR::Topic ekf_points_topic_ =
      LibXR::Topic("ekf_points", sizeof(EkfPointsMsg), &tracker_domain_);
  LibXR::Topic candidate_debug_topic_ =
      LibXR::Topic("candidate_debug", sizeof(CandidateDebugMsg), &tracker_domain_);

  const char* name_ = "armor_tracker";
  LibXR::RamFS::File cmd_file_;
  std::atomic<bool> params_is_changed_{false};
  EkfPointsMsg ekf_msg_{};
  CandidateDebugMsg candidate_debug_msg_{};
  FrameSync& sync_;
};

#include "ArmorTrackerPipeline.hpp"
