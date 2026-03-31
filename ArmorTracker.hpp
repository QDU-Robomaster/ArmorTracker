#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: sp_vision style armor tracker
constructor_args:
  cfg:
    limits:
      max_armor_distance: 10.0
      max_z_position: 1.0
    thresholds:
      min_detect_count: 5
      max_temp_lost_count: 15
      outpost_max_temp_lost_count: 75
    frames:
      rotation: [1.0, 0.0, 0.0, 0.0]
      translation: [0.0, 0.0, 0.0]
    debug:
      preview: false
      wait_key_ms: 1
      overlay_scale: 0.75
      draw_candidates: true
template_args: []
required_hardware: []
depends:
  - qdu-future/ArmorDetector@feat/sp_vision
  - qdu-future/CameraBase
=== END MANIFEST === */
// clang-format on

#include <Eigen/Dense>

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "CameraBase.hpp"
#include "Target.hpp"
#include "TrackerTypes.hpp"
#include "app_framework.hpp"
#include "armor.hpp"
#include "libxr.hpp"
#include "mutex.hpp"
#include "transform.hpp"

class ArmorTracker : public LibXR::Application
{
 public:
  struct Config
  {
    struct Limits
    {
      double max_armor_distance{10.0};
      double max_z_position{1.0};
    } limits;

    struct Thresholds
    {
      int min_detect_count{5};
      int max_temp_lost_count{15};
      int outpost_max_temp_lost_count{75};
    } thresholds;

    struct Frames
    {
      // Pose of the camera frame expressed in gimbal coordinates.
      LibXR::Transform<double> camera_to_gimbal_transform{};

      Frames(std::array<double, 4> rotation = {1.0, 0.0, 0.0, 0.0},
             std::array<double, 3> translation = {0.0, 0.0, 0.0})
          : camera_to_gimbal_transform(
                LibXR::Quaternion<double>(rotation[0], rotation[1], rotation[2],
                                          rotation[3]),
                LibXR::Position<double>(translation[0], translation[1], translation[2]))
      {
      }
    } frames;

    struct Debug
    {
      bool preview{false};
      int wait_key_ms{1};
      double overlay_scale{0.75};
      bool draw_candidates{true};
    } debug;
  };

  enum class State : std::uint8_t
  {
    LOST = 0,
    DETECTING = 1,
    TRACKING = 2,
    TEMP_LOST = 3,
  };

  struct TrackerInfo
  {
    double position_diff{};
    double yaw_diff{};
    LibXR::Position<double> position{};
    double yaw{};
  };

  struct TrackerMetrics
  {
    uint64_t frame_index{0};
    uint32_t input_armor_count{0};
    uint32_t matched_armor_count{0};
    uint32_t projection_sample_count{0};
    uint32_t state{0};
    uint32_t reset_count{0};
    bool tracking{false};
    ArmorNumber tracked_id{ArmorNumber::INVALID};
    double tracker_latency_ms{0.0};
    double last_nis{0.0};
    double recent_nis_failure_rate{0.0};
    double prediction_center_error_px{0.0};
    double prediction_corner_error_px{0.0};
  };

  ArmorTracker(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app, Config cfg);

  void OnMonitor() override {}

 private:
  void CameraInfoCallback(CameraBase::CameraInfo* camera_info);
  void ImageCallback(cv::Mat* img_msg);
  void ArmorsCallback(ArmorDetectionsMessage& armors_msg);
  void GimbalRotationCallback(LibXR::Quaternion<float>* rotation_msg);
  cv::Mat ConvertToBgr(const cv::Mat& input) const;
  void ShowDebugPreview(const ArmorDetectionsMessage& armors_msg,
                        const std::vector<TrackedArmorObservation>& observations,
                        std::size_t matched_armor_count,
                        const LibXR::Quaternion<double>& frame_gimbal_rotation);
  bool ShouldShowPreview();

  std::vector<TrackedArmorObservation> BuildObservations(
      const ArmorDetectionsMessage& armors_msg,
      const LibXR::Quaternion<double>& frame_gimbal_rotation) const;
  void SortObservations(std::vector<TrackedArmorObservation>& armors) const;
  bool SetTarget(std::vector<TrackedArmorObservation>& armors,
                 LibXR::MicrosecondTimestamp timestamp);
  bool UpdateTarget(std::vector<TrackedArmorObservation>& armors,
                    LibXR::MicrosecondTimestamp timestamp);
  void StateMachine(bool found);
  void PublishOutputs(std::size_t input_armor_count, std::size_t matched_armor_count,
                      double tracker_latency_ms, uint32_t projection_sample_count,
                      double prediction_center_error_px,
                      double prediction_corner_error_px);
  void EvaluateProjectionMetrics(
      const std::vector<TrackedArmorObservation>& observations,
      const LibXR::Quaternion<double>& frame_gimbal_rotation,
      uint32_t& projection_sample_count, double& prediction_center_error_px,
      double& prediction_corner_error_px) const;
  void PushGimbalRotationSample(LibXR::MicrosecondTimestamp timestamp,
                                const LibXR::Quaternion<double>& rotation);
  bool QueryGimbalRotationAt(LibXR::MicrosecondTimestamp timestamp,
                             LibXR::Quaternion<double>& rotation) const;
  double GetArmorWorldYaw(const LibXR::Quaternion<double>& rotation) const;

  struct TimedGimbalRotation
  {
    LibXR::MicrosecondTimestamp timestamp{};
    LibXR::Quaternion<double> rotation{};
  };

 private:
  Config cfg_{};
  State state_{State::LOST};
  int detect_count_{0};
  int temp_lost_count_{0};
  ArmorNumber tracked_id_{ArmorNumber::INVALID};
  Target target_{};
  LibXR::MicrosecondTimestamp last_timestamp_{};
  uint64_t frame_index_{0};
  uint32_t reset_count_{0};

  std::shared_ptr<CameraBase::CameraInfo> camera_info_{};
  cv::Mat latest_frame_{};
  bool preview_available_{true};
  bool preview_warned_{false};

  LibXR::Transform<double> camera_to_gimbal_transform_static_{};
  std::deque<TimedGimbalRotation> gimbal_rotation_history_{};
  LibXR::Quaternion<double> gimbal_rotation_{1.0, 0.0, 0.0, 0.0};
  bool has_gimbal_rotation_{false};
  mutable LibXR::Mutex gimbal_rotation_lock_{};
  mutable LibXR::Mutex preview_frame_lock_{};

  TrackerInfo info_msg_{};
  TrackerMetrics metrics_msg_{};
  TrackerTarget target_msg_{};

  LibXR::Topic::Domain tracker_domain_ = LibXR::Topic::Domain("tracker");
  LibXR::Topic info_topic_ = LibXR::Topic("info", sizeof(TrackerInfo), &tracker_domain_);
  LibXR::Topic metrics_topic_ =
      LibXR::Topic("metrics", sizeof(TrackerMetrics), &tracker_domain_);
  LibXR::Topic target_topic_ =
      LibXR::Topic("target", sizeof(TrackerTarget), &tracker_domain_);
};
