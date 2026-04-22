#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: Armor tracker
constructor_args:
  cfg:
    limits:
      max_armor_distance: 10.0
      max_z_position: 1.0

    match:
      max_match_distance: 0.15
      max_match_yaw_diff: 1.0

    thresholds:
      tracking_thres: 5
      lost_time_thres: 0.3

    solver:
      k: 0.092
      bias_time: 100
      s_bias: 0.19133
      z_bias: 0.21265
      calculate_mode: SolveTrajectory::CalculateMode::NORMAL
      table_config:
        max_x: 13.0
        min_x: 0.0
        max_y: 1.0
        min_y: -1.0
        precision: 0.01
        filename: "table.bin"

    ekf:
      sigma2_q_xyz: 20.0
      sigma2_q_yaw: 100.0
      sigma2_q_r: 800

    noise:
      r_xyz_factor: 0.05
      r_yaw: 0.02

    frames:
      rotation: [0.0, 0.0, 0.0, 0.0]
      translation: [0.0, 0.0, 0.0]
  sync: '@camera_frame_sync'
template_args:
  - Info:
      width: 1280
      height: 720
      step: 3840
      encoding: CameraTypes::Encoding::BGR8
      camera_matrix: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
      distortion_model: CameraTypes::DistortionModel::PLUMB_BOB
      distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
      rectification_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      projection_matrix: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
required_hardware: []
depends:
  - qdu-future/ArmorDetector
  - qdu-future/CameraFrameSync
=== END MANIFEST === */
// clang-format on

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// 框架与外部依赖头
#include "ArmorTrackerFaceSelector.hpp"
#include "ArmorTrackerImageTracker.hpp"
#include "ArmorTrackerObserver.hpp"
#include "CameraFrameSync.hpp"
#include "SolveTrajectory.hpp"
#include "app_framework.hpp"
#include "armor.hpp"
#include "cycle_value.hpp"
#include "extended_kalman_filter.hpp"
#include "libxr_time.hpp"
#include "logger.hpp"
#include "message.hpp"
#include "mutex.hpp"
#include "thread.hpp"
#include "timebase.hpp"
#include "transform.hpp"

namespace armor_tracker
{
struct FaceMatchCandidate;
struct FaceSelectionPolicy;
struct ObserverPolicy;
struct ObserverRuntime;
struct FaceSelectionResult;
struct FaceSelectionTrackedState;

// 只保留轻量状态绑定逻辑，避免再拆一个过碎的独立头文件。
struct FaceBindingRuntime
{
  int tracked_armors_num = 4;
  bool tracked_face_track_id_valid = false;
  uint16_t tracked_face_track_id = 0;
  std::array<bool, 4> face_track_id_valid{};
  std::array<uint16_t, 4> face_track_id{};
};

inline void ApplySelectedFaceBinding(FaceBindingRuntime& runtime,
                                     const FaceMatchCandidate& selected_candidate,
                                     bool did_face_switch)
{
  if (did_face_switch)
  {
    const int face_count = std::max(1, std::min(4, runtime.tracked_armors_num));
    std::array<bool, 4> rotated_valid{};
    std::array<uint16_t, 4> rotated_ids{};
    for (int face_slot = 0; face_slot < face_count; ++face_slot)
    {
      const int old_slot = (face_slot + selected_candidate.face_index) % face_count;
      rotated_valid[face_slot] = runtime.face_track_id_valid[old_slot];
      rotated_ids[face_slot] = runtime.face_track_id[old_slot];
    }
    runtime.face_track_id_valid = rotated_valid;
    runtime.face_track_id = rotated_ids;
  }

  if (runtime.face_track_id_valid[0])
  {
    runtime.tracked_face_track_id_valid = true;
    runtime.tracked_face_track_id = runtime.face_track_id[0];
  }
  else
  {
    runtime.tracked_face_track_id_valid = false;
  }

  if (selected_candidate.image_track_id >= 0 &&
      selected_candidate.confirmed_image_track)
  {
    runtime.tracked_face_track_id_valid = true;
    runtime.tracked_face_track_id =
        static_cast<uint16_t>(selected_candidate.image_track_id);
    runtime.face_track_id_valid[0] = true;
    runtime.face_track_id[0] = runtime.tracked_face_track_id;
  }
}

// 选面日志上下文继续保留，但并回主头，避免无意义碎文件。
struct FaceSelectionLogContext
{
  ArmorNumber tracked_id = ArmorNumber::INVALID;
  double face_switch_timeout_sec = 0.0;
  double face_switch_score_deadzone = 0.0;
  double face_switch_position_deadzone = 0.0;
  double face_switch_yaw_deadzone = 0.0;
  double face_switch_cooldown_remaining = 0.0;
};

inline void LogRejectedSelection(const FaceSelectionResult& selection,
                                 const FaceSelectionLogContext& context)
{
  const auto& best_candidate = selection.best_candidate;
  const auto& best_same_face_candidate = selection.best_same_face_candidate;
  const auto& best_switch_candidate = selection.best_switch_candidate;
  XR_LOG_DEBUG(
      "No matched armor found! same_number=%d best_face=%d score=%.3f pos_diff=%.3f yaw_diff=%.3f same_score=%.3f switch_score=%.3f cooldown=%.3f image_track=%d confirmed=%d persistent=%d img_diff=%.1f area_log=%.3f",
      selection.has_same_number_candidate ? 1 : 0, best_candidate.face_index,
      best_candidate.score, best_candidate.position_diff, best_candidate.yaw_diff,
      best_same_face_candidate.score, best_switch_candidate.score,
      context.face_switch_cooldown_remaining, best_candidate.image_track_id,
      best_candidate.confirmed_image_track ? 1 : 0,
      best_candidate.same_persistent_track ? 1 : 0,
      best_candidate.image_center_diff, best_candidate.area_ratio_log);
}

inline void LogAcceptedSelection(const FaceSelectionResult& selection,
                                 const FaceSelectionLogContext& context)
{
  const auto& selected_candidate = selection.selected_candidate;
  const auto& best_same_face_candidate = selection.best_same_face_candidate;
  const auto& best_switch_candidate = selection.best_switch_candidate;
  const bool did_face_switch = selected_candidate.face_index != 0;

  XR_LOG_DEBUG(
      "Tracker pick: armor=%zu num=%d face=%d same=%d score=%.3f pos_diff=%.3f yaw_diff=%.3f view_bonus=%.3f area=%.3f frontality=%.3f cooldown=%.3f",
      selected_candidate.armor_index,
      static_cast<int>(selected_candidate.armor.number),
      selected_candidate.face_index, selected_candidate.same_number ? 1 : 0,
      selected_candidate.score, selected_candidate.position_diff,
      selected_candidate.yaw_diff, selected_candidate.view_bonus,
      selected_candidate.area_score, selected_candidate.frontality,
      context.face_switch_cooldown_remaining);

  if (did_face_switch)
  {
    if (!selection.strict_face_switch_match &&
        !selection.relaxed_face_switch_match &&
        selection.id_assisted_face_rebind_match)
    {
      XR_LOG_DEBUG(
          "Tracker id-assisted face rebind: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f observed_persistent=%d",
          selected_candidate.face_index, selected_candidate.position_diff,
          selected_candidate.yaw_diff,
          static_cast<int>(selected_candidate.armor.number),
          context.face_switch_timeout_sec,
          selection.observed_persistent_track_this_frame ? 1 : 0);
      return;
    }
    if (!selection.strict_face_switch_match &&
        !selection.relaxed_face_switch_match &&
        selection.id_assisted_face_handover_match)
    {
      XR_LOG_DEBUG(
          "Tracker id-assisted face handover: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f same_pos=%.3f",
          selected_candidate.face_index, selected_candidate.position_diff,
          selected_candidate.yaw_diff,
          static_cast<int>(selected_candidate.armor.number),
          context.face_switch_timeout_sec,
          best_same_face_candidate.position_diff);
      return;
    }
    if (!selection.strict_face_switch_match &&
        selection.relaxed_face_switch_match)
    {
      XR_LOG_DEBUG(
          "Tracker relaxed face switch: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f",
          selected_candidate.face_index, selected_candidate.position_diff,
          selected_candidate.yaw_diff,
          static_cast<int>(selected_candidate.armor.number),
          context.face_switch_timeout_sec);
      return;
    }

    XR_LOG_DEBUG(
        "Tracker face switch: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f",
        selected_candidate.face_index, selected_candidate.position_diff,
        selected_candidate.yaw_diff,
        static_cast<int>(selected_candidate.armor.number),
        context.face_switch_timeout_sec);
    return;
  }

  if (!selection.strict_same_face_match && selection.relaxed_same_face_match)
  {
    XR_LOG_DEBUG("Tracker relaxed same-face match: pos_diff=%.3f yaw_diff=%.3f",
                 selected_candidate.position_diff, selected_candidate.yaw_diff);
    return;
  }
  if (selection.id_assisted_same_face_hold)
  {
    XR_LOG_DEBUG(
        "Tracker hold same-face by persistent image id: pos_diff=%.3f yaw_diff=%.3f img_diff=%.1f area_log=%.3f",
        selected_candidate.position_diff, selected_candidate.yaw_diff,
        selected_candidate.image_center_diff, selected_candidate.area_ratio_log);
    return;
  }
  if (selection.switch_blocked_by_timeout && selection.matched_switch_face)
  {
    XR_LOG_DEBUG(
        "Tracker hold same-face by timeout: cooldown=%.3f same_score=%.3f switch_score=%.3f",
        context.face_switch_cooldown_remaining, best_same_face_candidate.score,
        best_switch_candidate.score);
    return;
  }
  if (selection.switch_blocked_by_id_mismatch)
  {
    XR_LOG_DEBUG(
        "Tracker hold same-face by id mismatch: tracked_num=%d switch_num=%d same_score=%.3f switch_score=%.3f",
        static_cast<int>(context.tracked_id),
        static_cast<int>(best_switch_candidate.armor.number),
        best_same_face_candidate.score, best_switch_candidate.score);
    return;
  }
  if (!selection.allow_face_switch && selection.matched_switch_face &&
      selection.matched_same_face)
  {
    XR_LOG_DEBUG(
        "Tracker hold same-face by deadzone: same_score=%.3f switch_score=%.3f score_dz=%.3f pos_dz=%.3f yaw_dz=%.3f",
        best_same_face_candidate.score, best_switch_candidate.score,
        context.face_switch_score_deadzone,
        context.face_switch_position_deadzone,
        context.face_switch_yaw_deadzone);
  }
}
}  // namespace armor_tracker

struct ArmorTrackerSend
{
  bool is_fire{};
  LibXR::Position<double> position{};
  double v_yaw{};
  double pitch{};
  double yaw{};
  Eigen::Matrix<double, 3, 1> cmd_vel_linear = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> cmd_vel_angular = Eigen::Matrix<double, 3, 1>::Zero();
};

template <CameraTypes::CameraInfo CameraInfoV>
class ArmorTracker : public LibXR::Application
{
 public:
  using FrameSync = CameraFrameSync<CameraInfoV>;
  using Base = typename FrameSync::Base;
  using CameraInfo = typename Base::CameraInfo;
  using SyncedFrame = typename FrameSync::SyncedFrame;

  static inline constexpr CameraInfo kCameraInfo = CameraInfoV;

  // ====================== 配置参数（构造入参聚合） ======================
  struct Config
  {
    struct Limits
    {
      double max_armor_distance = 10.0;  // 过滤距离阈值（XOY）
      double max_z_position = 1.0;
    } limits;

    struct Match
    {
      double max_match_distance = 0.15;  // 匹配位置阈值（m）
      double max_match_yaw_diff = 1.0;   // 匹配 yaw 阈值（rad）
    } match;

    struct Thresholds
    {
      int tracking_thres = 5;        // 进入 TRACKING 需要的连续匹配帧数
      double lost_time_thres = 0.3;  // 进入 LOST 的时间阈值（秒）
    } thresholds;

    struct Solver
    {
      double k = 0.092;  // 弹道解算参数
      int bias_time = 100;
      double s_bias = 0.19133;
      double z_bias = 0.21265;
      SolveTrajectory::CalculateMode calculate_mode = SolveTrajectory::NORMAL;
      TrajectoryTable::TableConfig table_config;
    } solver;

    struct Ekf
    {
      double sigma2_q_xyz = 20.0;   // 过程噪声（位置/速度）
      double sigma2_q_yaw = 100.0;  // 过程噪声（yaw/wyaw）
      double sigma2_q_r = 800;      // 过程噪声（半径）
    } ekf;

    struct Noise
    {
      double r_xyz_factor = 0.05;  // 观测噪声（随距离缩放）
      double r_yaw = 0.02;         // 观测噪声（yaw）
    } noise;

    struct Frames
    {
      LibXR::Transform<double> base_transform_static = {};
      Frames(std::array<double, 4> rotation, std::array<double, 3> translation)
          : base_transform_static{
                LibXR::Quaternion<double>(rotation[0], rotation[1], rotation[2],
                                          rotation[3]),
                LibXR::Position<double>(translation[0], translation[1], translation[2])}
      {
      }
    } frames;
  };

  // ====================== 公共类型 ======================
  enum class ArmorsNum : std::uint8_t
  {
    NORMAL_4 = 4,
    OUTPOST_3 = 3
  };

  enum class State : std::uint8_t
  {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
  };

  struct TrackerInfo
  {
    double position_diff{};
    double yaw_diff{};
    LibXR::Position<double> position{};
    double yaw{};
  };

  using Send = ArmorTrackerSend;

  struct EkfPointsMsg
  {
    uint8_t count;                          // 实际装甲块数量（3或4）
    LibXR::Position<double> center_cam;     // 相机←中心 3D
    LibXR::Position<double> armors_cam[4];  // 相机←装甲 3D（最多4块）
    bool valid[5];
  };

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
    float center_x{};
    float center_y{};
    float predicted_yaw{};
    float measured_yaw{};
  };

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
    std::array<int16_t, kMaxDetections> detection_track_ids{};
    std::array<uint8_t, kMaxDetections> detection_track_confirmed{};
    CandidateDebugItem items[kMaxItems]{};
  };

 private:
  struct TimedCameraPose
  {
    uint64_t timestamp_us{};
    LibXR::Transform<double> pose{};
  };

 public:
  // ====================== 构造与监控 ======================
  explicit ArmorTracker(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app,
                        Config cfg, FrameSync& sync);

  static int CommandFun(ArmorTracker* self, int argc, char** argv);
  const Config& GetConfig() const { return cfg_; }
  void SetConfig(const Config& cfg);
  static int CommandAdapter(void* instance, int argc, char** argv)
  {
    return CommandFun(static_cast<ArmorTracker*>(instance), argc, argv);
  }

  void OnMonitor() override;

 private:
  // ====================== 内部算法接口（原 Tracker 逻辑）
  // ======================
  void Init(const ArmorDetectorResults& armors_msg);
  void Update(const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us);
  void UpdateImageIdTracks(const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us);
  int FindDetectionTrackId(std::size_t armor_index) const;
  bool IsDetectionTrackConfirmed(std::size_t armor_index) const;
  void PushCameraPose(uint64_t timestamp_us,
                      const LibXR::Quaternion<double>& camera_rotation);
  bool LookupCameraPose(uint64_t image_timestamp_us, LibXR::Transform<double>& pose_out);
  void FillCandidateDebugFromSelection(
      const armor_tracker::FaceSelectionResult& selection,
      CandidateDebugMsg& candidate_debug);
  void FillCandidateDebugPolicy(
      CandidateDebugMsg& candidate_debug, const Eigen::VectorXd& ekf_prediction,
      const armor_tracker::FaceSelectionPolicy& face_policy) const;
  armor_tracker::FaceSelectionPolicy BuildFaceSelectionPolicy() const;
  armor_tracker::FaceSelectionTrackedState BuildFaceSelectionTrackedState() const;
  Eigen::Vector3d GetCameraWorldPosition();
  void AdvanceTrackerState(bool matched);
  bool ApplyFaceSelection(const armor_tracker::FaceSelectionResult& selection,
                          const ArmorDetectorResults& armors_msg,
                          const Eigen::VectorXd& ekf_prediction,
                          CandidateDebugMsg& candidate_debug);
  armor_tracker::ObserverPolicy BuildObserverPolicy() const;
  armor_tracker::ObserverRuntime BuildObserverRuntime() const;
  void ApplyObserverRuntime(const armor_tracker::ObserverRuntime& runtime);
  armor_tracker::FaceBindingRuntime BuildFaceBindingRuntime() const;
  void ApplyFaceBindingRuntime(const armor_tracker::FaceBindingRuntime& runtime);
  void ApplySelectedIdentity(const armor_tracker::FaceMatchCandidate& selected_candidate);
  void ApplySelectedFaceBinding(const armor_tracker::FaceMatchCandidate& selected_candidate,
                                bool did_face_switch);
  void ApplySelectedMeasurementUpdate(
      const armor_tracker::FaceMatchCandidate& selected_candidate,
      const ArmorDetectorResults& armors_msg, const Eigen::VectorXd& ekf_prediction,
      bool did_face_switch);

  // ====================== IO 与回调（原 Node 逻辑） ======================
  void VelocityCallback(double velocity_msg);
  void ArmorsCallback(ArmorDetectorResults armors_msg, uint64_t image_timestamp_us);

  // ====================== 辅助函数 ======================
  void InitEKF(const ArmorDetectorResult& a);
  void UpdateArmorsNum(const ArmorDetectorResult&);
  void UpdateDzReference(const ArmorDetectorResults& armors_msg,
                         const ArmorDetectorResult& anchor);
  void FuseMultiArmorObservation(const ArmorDetectorResults& armors_msg);
  void HandleArmorJump(const ArmorDetectorResult& current_armor, double measured_yaw);
  void SwitchTrackedFace(int face_index, const ArmorDetectorResult& current_armor,
                         double measured_yaw);
  double OrientationToYaw(const LibXR::Quaternion<double>& q);
  double GetArmorYawFromState(const Eigen::VectorXd& x, int face_index = 0);
  Eigen::Vector3d GetArmorPositionFromState(const Eigen::VectorXd& x, int face_index = 0);
  static double ArmorImageArea(const ArmorDetectorResult& armor);
  static double TimestampDeltaSeconds(uint64_t newer, uint64_t older);
  static void SyncFramePoseThreadFun(ArmorTracker<CameraInfoV>* self);
#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
  static void PreviewImageThreadFun(ArmorTracker<CameraInfoV>* self);
  static void RenderPreviewFrame(ArmorTracker<CameraInfoV>* self, cv::Mat frame);
#endif

  // ====================== 内部聚合成员（类内聚合） ======================
  struct EKFBlock
  {
    ExtendedKalmanFilter ekf;
    Eigen::VectorXd measurement = Eigen::VectorXd::Zero(4);  // z = [xa,ya,za,yaw]
    Eigen::VectorXd state =
        Eigen::VectorXd::Zero(9);  // x = [xc,vxc,yc,vyc,za,vza,yaw,vyaw,r]
  } ekf_;

  armor_tracker::ImageTrackManager image_tracker_{};

  struct TrackRuntime
  {
    State state = State::LOST;
    int detect_count = 0;
    int lost_count = 0;
    int tracking_thres = 5;
    int lost_thres = 0;  // 帧数阈值（由时间阈值换算）
    double last_yaw = 0.0;
    double info_position_diff = 0.0;
    double info_yaw_diff = 0.0;
    double face_switch_cooldown_remaining = 0.0;

    ArmorNumber tracked_id = ArmorNumber::INVALID;
    ArmorDetectorResult tracked_armor{};
    ArmorsNum tracked_armors_num = ArmorsNum::NORMAL_4;
    bool tracked_face_track_id_valid = false;
    uint16_t tracked_face_track_id = 0;
    std::array<bool, 4> face_track_id_valid{};
    std::array<uint16_t, 4> face_track_id{};

    // 另一片装甲板信息
    double dz = 0.0;
    double dz_abs_ref = 0.0;
    double another_r = 0.0;

  } rt_;

  struct TimeBlock
  {
    LibXR::MicrosecondTimestamp last_time = LibXR::Timebase::GetMicroseconds();
    uint64_t last_image_timestamp_us = 0;
    double dt = 1.0 / 100.0;  // 初始假定 100Hz
  } time_;

  struct IOBlock
  {
    // 坐标变换
    static constexpr std::size_t kCameraPoseHistorySize = 32;
    LibXR::Transform<double> gimbal_to_camera_transform_static{};
    LibXR::Quaternion<double> gimbal_rotation{};
    LibXR::Transform<double> latest_camera_pose{};
    bool latest_camera_pose_valid = false;
    std::array<TimedCameraPose, kCameraPoseHistorySize> camera_pose_history{};
    std::size_t camera_pose_history_head = 0;
    std::size_t camera_pose_history_count = 0;
    LibXR::Mutex gimbal_rotation_lock;

    // 发布者
    LibXR::Topic::Domain tracker_domain = LibXR::Topic::Domain("tracker");
    LibXR::Topic info_topic = LibXR::Topic("info", sizeof(TrackerInfo), &tracker_domain);
    LibXR::Topic target_topic =
        LibXR::Topic("target", sizeof(SolveTrajectory::Target), &tracker_domain);
    LibXR::Topic target_eulr_topic =
        LibXR::Topic("target_eulr", sizeof(LibXR::EulerAngle<float>), &tracker_domain);
    LibXR::Topic fire_notify_topic =
        LibXR::Topic("fire_notify", sizeof(uint8_t), &tracker_domain);
    LibXR::Topic send_topic = LibXR::Topic("send", sizeof(Send), &tracker_domain);
    LibXR::Topic ekf_points_topic =
        LibXR::Topic("ekf_points", sizeof(EkfPointsMsg), &tracker_domain);
    LibXR::Topic candidate_debug_topic =
        LibXR::Topic("candidate_debug", sizeof(CandidateDebugMsg), &tracker_domain);

    // 轨迹解算
    std::unique_ptr<SolveTrajectory> solver;
  } io_;

  // 保存配置（类内聚合）
  Config cfg_;
  Config::Solver solver_cfg_;

  const char* name_ = "armor_tracker";
  LibXR::RamFS::File cmd_file_;
  std::atomic<bool> params_is_changed_{false};
  LibXR::Thread sync_frame_pose_thread_{};
#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
  LibXR::Thread preview_image_thread_{};
#endif

  EkfPointsMsg ekf_msg_;
  CandidateDebugMsg candidate_debug_msg_{};
  FrameSync& sync_;
};


namespace armor_tracker_detail
{
inline double UnwrapYawNear(double yaw, double reference_yaw)
{
  const double delta =
      LibXR::CycleValue<double>(yaw) - LibXR::CycleValue<double>(reference_yaw);
  return reference_yaw + delta;
}

inline double QuaternionToYaw(const LibXR::Quaternion<double>& q)
{
  LibXR::EulerAngle<double> eulr =
      LibXR::RotationMatrix<double>(q.ToRotationMatrix()).ToEulerAngle();
  return eulr.Yaw();
}

inline double OrientationToYawNear(const LibXR::Quaternion<double>& q, double reference_yaw)
{
  return UnwrapYawNear(QuaternionToYaw(q), reference_yaw);
}

inline double AngularDiffAbs(double lhs, double rhs)
{
  return std::abs(LibXR::CycleValue<double>(lhs) - LibXR::CycleValue<double>(rhs));
}

inline void LogImpossibleYawDiff(const char* tag, std::size_t armor_index, int face_index,
                          double measured_yaw, double predicted_yaw, double yaw_diff)
{
  if (!(std::isfinite(yaw_diff)) || yaw_diff <= M_PI + 1e-3)
  {
    return;
  }
  const double wrapped_measured = LibXR::CycleValue<double>(measured_yaw);
  const double wrapped_predicted = LibXR::CycleValue<double>(predicted_yaw);
  XR_LOG_ERROR(
      "Impossible yaw diff[%s]: armor=%zu face=%d measured=%.6f predicted=%.6f wrapped_measured=%.6f wrapped_predicted=%.6f yaw_diff=%.6f direct_cycle_sub=%.6f raw_sub=%.6f",
      tag, armor_index, face_index, measured_yaw, predicted_yaw, wrapped_measured,
      wrapped_predicted, yaw_diff,
      std::abs(LibXR::CycleValue<double>(measured_yaw) -
               LibXR::CycleValue<double>(predicted_yaw)),
      std::abs(measured_yaw - predicted_yaw));
}

inline LibXR::Quaternion<double> PackedCameraRotation(
    const std::array<float, 4>& rotation_wxyz)
{
  return LibXR::Quaternion<double>(rotation_wxyz[0], rotation_wxyz[1],
                                   rotation_wxyz[2], rotation_wxyz[3]);
}

constexpr uint32_t kArmorTrackerSyncFrameWaitTimeoutMs = 100;

inline int ArmorTrackerCvTypeFromEncoding(CameraTypes::Encoding encoding)
{
  switch (encoding)
  {
    case CameraTypes::Encoding::RGB8:
    case CameraTypes::Encoding::BGR8:
      return CV_8UC3;
    case CameraTypes::Encoding::RGBA8:
    case CameraTypes::Encoding::BGRA8:
      return CV_8UC4;
    case CameraTypes::Encoding::MONO8:
      return CV_8UC1;
    default:
      return -1;
  }
}

inline cv::Mat ArmorTrackerConvertToBgrWithEncoding(const cv::Mat& input, CameraTypes::Encoding encoding)
{
  switch (encoding)
  {
    case CameraTypes::Encoding::RGB8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
      return output;
    }
    case CameraTypes::Encoding::BGRA8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_BGRA2BGR);
      return output;
    }
    case CameraTypes::Encoding::RGBA8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_RGBA2BGR);
      return output;
    }
    case CameraTypes::Encoding::MONO8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);
      return output;
    }
    default:
      // Tracker preview will draw overlays in-place. Even when the source is
      // already BGR, detach it from the shared sync buffer before rendering.
      return input.clone();
  }
}

inline LibXR::Transform<double> ArmorTrackerCameraRotationToTrackerWorldPose(
    const LibXR::Quaternion<double>& camera_rotation,
    const LibXR::Transform<double>& gimbal_to_camera_transform_static)
{
  return LibXR::Transform<double>(camera_rotation, {0.0, 0.0, 0.0}) +
         gimbal_to_camera_transform_static;
}

inline uint64_t TimestampAbsDiff(uint64_t lhs, uint64_t rhs)
{
  return lhs >= rhs ? (lhs - rhs) : (rhs - lhs);
}
inline bool SingleArmorModeEnabled()
{
  const char* env = std::getenv("XR_TRACKER_SINGLE_ARMOR_MODE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool MultiArmorFuseEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_MULTI_FUSE");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline bool SymmetricGeometryEnabled()
{
  const char* env = std::getenv("XR_TRACKER_FORCE_SYMMETRIC_GEOMETRY");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool FaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline bool RelaxedFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_RELAXED_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline bool FaceSwitchRecenterEnabled()
{
  const char* env = std::getenv("XR_TRACKER_DISABLE_FACE_SWITCH_RECENTER");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline bool OddFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_ODD_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline bool ViewPriorityEnabled()
{
  const char* env = std::getenv("XR_TRACKER_ENABLE_VIEW_PRIORITY");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool ArmorTrackerPreviewUiAvailable()
{
  const char* display = std::getenv("DISPLAY");
  const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
  if ((display != nullptr && display[0] != '\0') ||
      (wayland_display != nullptr && wayland_display[0] != '\0'))
  {
    return true;
  }

  const char* qt_platform = std::getenv("QT_QPA_PLATFORM");
  if (qt_platform == nullptr || qt_platform[0] == '\0')
  {
    return false;
  }

  return std::strcmp(qt_platform, "offscreen") == 0 ||
         std::strcmp(qt_platform, "minimal") == 0;
}

inline const char* ArmorTrackerArmorsTopicName()
{
  const char* env = std::getenv("XR_ARMORS_TOPIC_NAME");
  return (env != nullptr && env[0] != '\0') ? env : "armors_result";
}


inline bool DirectionalFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_ENABLE_DIRECTIONAL_FACE_SWITCH");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline double ParseEnvDouble(const char* name, double default_value)
{
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0')
  {
    return default_value;
  }
  char* end = nullptr;
  const double parsed = std::strtod(env, &end);
  if (end == env || !std::isfinite(parsed))
  {
    return default_value;
  }
  return parsed;
}

inline std::uint32_t ParseEnvUint(const char* name, std::uint32_t default_value)
{
  return static_cast<std::uint32_t>(std::max(
      1.0, std::round(ParseEnvDouble(name, static_cast<double>(default_value)))));
}

inline double SingleArmorImageCenterGatePx()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SINGLE_ARMOR_IMAGE_GATE_PX", 180.0));
}

inline double SingleArmorAreaLogGate()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SINGLE_ARMOR_AREA_LOG_GATE", 0.80));
}

inline double FaceSwitchScoreDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_SCORE_DEADZONE", 0.15));
}

inline double FaceSwitchPositionDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_POSITION_DEADZONE", 0.05));
}

inline double FaceSwitchYawDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_YAW_DEADZONE", 0.35));
}

inline double FaceSwitchTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_TIMEOUT_SEC", 0.08));
}

inline bool IdAssistEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_IMAGE_ID_ASSIST");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline double IdAssistSameFaceCenterGatePx()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_ASSIST_CENTER_GATE_PX", 85.0));
}

inline double IdAssistSameFaceAreaLogGate()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_ASSIST_AREA_LOG_GATE", 0.45));
}

inline std::uint32_t IdTrackAppearHits()
{
  return ParseEnvUint("XR_TRACKER_ID_APPEAR_HITS", 2U);
}

inline double IdTrackAppearTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_APPEAR_TIMEOUT_SEC", 0.01));
}

inline std::uint32_t IdTrackTentativeMisses()
{
  return ParseEnvUint("XR_TRACKER_ID_TENTATIVE_MISSES", 2U);
}

inline double IdTrackTentativeTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_TENTATIVE_TIMEOUT_SEC", 0.03));
}

inline std::uint32_t IdTrackDisappearMisses()
{
  return ParseEnvUint("XR_TRACKER_ID_DISAPPEAR_MISSES", 3U);
}

inline double IdTrackDisappearTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_DISAPPEAR_TIMEOUT_SEC", 0.06));
}

}  // namespace armor_tracker_detail

using armor_tracker_detail::AngularDiffAbs;
using armor_tracker_detail::ArmorTrackerArmorsTopicName;
using armor_tracker_detail::ArmorTrackerCameraRotationToTrackerWorldPose;
using armor_tracker_detail::ArmorTrackerConvertToBgrWithEncoding;
using armor_tracker_detail::ArmorTrackerCvTypeFromEncoding;
using armor_tracker_detail::DirectionalFaceSwitchEnabled;
using armor_tracker_detail::FaceSwitchEnabled;
using armor_tracker_detail::FaceSwitchPositionDeadzone;
using armor_tracker_detail::FaceSwitchRecenterEnabled;
using armor_tracker_detail::FaceSwitchScoreDeadzone;
using armor_tracker_detail::FaceSwitchTimeoutSec;
using armor_tracker_detail::FaceSwitchYawDeadzone;
using armor_tracker_detail::IdAssistEnabled;
using armor_tracker_detail::IdAssistSameFaceAreaLogGate;
using armor_tracker_detail::IdAssistSameFaceCenterGatePx;
using armor_tracker_detail::IdTrackAppearHits;
using armor_tracker_detail::IdTrackAppearTimeoutSec;
using armor_tracker_detail::IdTrackDisappearMisses;
using armor_tracker_detail::IdTrackDisappearTimeoutSec;
using armor_tracker_detail::IdTrackTentativeMisses;
using armor_tracker_detail::IdTrackTentativeTimeoutSec;
using armor_tracker_detail::ArmorTrackerPreviewUiAvailable;
using armor_tracker_detail::LogImpossibleYawDiff;
using armor_tracker_detail::MultiArmorFuseEnabled;
using armor_tracker_detail::OddFaceSwitchEnabled;
using armor_tracker_detail::OrientationToYawNear;
using armor_tracker_detail::QuaternionToYaw;
using armor_tracker_detail::RelaxedFaceSwitchEnabled;
using armor_tracker_detail::SingleArmorAreaLogGate;
using armor_tracker_detail::SingleArmorImageCenterGatePx;
using armor_tracker_detail::SingleArmorModeEnabled;
using armor_tracker_detail::SymmetricGeometryEnabled;
using armor_tracker_detail::TimestampAbsDiff;
using armor_tracker_detail::UnwrapYawNear;
using armor_tracker_detail::ViewPriorityEnabled;
using armor_tracker_detail::kArmorTrackerSyncFrameWaitTimeoutMs;

template <CameraTypes::CameraInfo CameraInfoV>
ArmorTracker<CameraInfoV>::ArmorTracker(LibXR::HardwareContainer& hw,
                                        LibXR::ApplicationManager&,
                                        Config cfg, FrameSync& sync)
    : cfg_(std::move(cfg)),
      solver_cfg_(cfg_.solver),
      cmd_file_(LibXR::RamFS::CreateFile(name_, CommandFun, this)),
      sync_(sync)
{
  XR_LOG_INFO("Starting ArmorTracker!");

  hw.template FindOrExit<LibXR::RamFS>({"ramfs"})->Add(cmd_file_);

  // 轨迹解算器
  io_.solver = std::make_unique<SolveTrajectory>(
      solver_cfg_.k, solver_cfg_.bias_time, solver_cfg_.s_bias, solver_cfg_.z_bias,
      solver_cfg_.calculate_mode, solver_cfg_.table_config);

  // 初值（和老逻辑一致）
  rt_.tracking_thres = cfg_.thresholds.tracking_thres;
  io_.gimbal_to_camera_transform_static = cfg_.frames.base_transform_static;

  // ---------------- EKF 设置 ----------------
  // 状态 x = [xc, vxc, yc, vyc, za, vza, yaw, vyaw, r]
  // 观测 z = [xa, ya, za, yaw]
  auto f = [this](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd x_new = x;
    x_new(ExtendedKalmanFilter::X_CENTER) +=
        x(ExtendedKalmanFilter::V_X_CENTER) * time_.dt;
    x_new(ExtendedKalmanFilter::Y_CENTER) +=
        x(ExtendedKalmanFilter::V_Y_CENTER) * time_.dt;
    x_new(ExtendedKalmanFilter::Z_ARMOR) += x(ExtendedKalmanFilter::V_Z_ARMOR) * time_.dt;
    x_new(ExtendedKalmanFilter::YAW) += x(ExtendedKalmanFilter::V_YAW) * time_.dt;
    return x_new;
  };
  auto j_f = [this](const Eigen::VectorXd&)
  {
    Eigen::MatrixXd f(9, 9);
    double d = time_.dt;
    // clang-format off
    f << 1, d, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, d, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, d, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, d, 0,
         0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1;
    // clang-format on
    return f;
  };
  auto h = [](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd z(4);
    double xc = x(ExtendedKalmanFilter::X_CENTER), yc = x(ExtendedKalmanFilter::Y_CENTER),
           yaw = x(ExtendedKalmanFilter::YAW), r = x(ExtendedKalmanFilter::ROBOT_R);
    z(0) = xc - r * std::cos(yaw);            // xa
    z(1) = yc - r * std::sin(yaw);            // ya
    z(2) = x(ExtendedKalmanFilter::Z_ARMOR);  // za
    z(3) = x(ExtendedKalmanFilter::YAW);      // yaw
    return z;
  };
  auto j_h = [](const Eigen::VectorXd& x)
  {
    Eigen::MatrixXd h(4, 9);
    double yaw = x(6), r = x(8);
    //                 xc vxc yc vyc za vza yaw               vyaw r
    h << /*xa*/ 1, 0, 0, 0, 0, 0, r * std::sin(yaw), 0, -std::cos(yaw),
        /*ya */ 0, 0, 1, 0, 0, 0, -r * std::cos(yaw), 0, -std::sin(yaw),
        /*za */ 0, 0, 0, 0, 1, 0, 0, 0, 0,
        /*yaw*/ 0, 0, 0, 0, 0, 0, 1, 0, 0;
    return h;
  };
  auto u_q = [this]()
  {
    Eigen::MatrixXd q(9, 9);
    double t = time_.dt, x = cfg_.ekf.sigma2_q_xyz, y = cfg_.ekf.sigma2_q_yaw,
           r = cfg_.ekf.sigma2_q_r;
    double q_x_x = std::pow(t, 4) / 4 * x, q_x_vx = std::pow(t, 3) / 2 * x,
           q_vx_vx = std::pow(t, 2) * x;
    double q_y_y = std::pow(t, 4) / 4 * y, q_y_vy = std::pow(t, 3) / 2 * y,
           q_vy_vy = std::pow(t, 2) * y;
    double q_r = std::pow(t, 4) / 4 * r;
    // clang-format off
    q.setZero();
    q(0,0)=q_x_x;  q(0,1)=q_x_vx; q(1,0)=q_x_vx; q(1,1)=q_vx_vx;
    q(2,2)=q_x_x;  q(2,3)=q_x_vx; q(3,2)=q_x_vx; q(3,3)=q_vx_vx;
    q(4,4)=q_x_x;  q(4,5)=q_x_vx; q(5,4)=q_x_vx; q(5,5)=q_vx_vx;
    q(6,6)=q_y_y;  q(6,7)=q_y_vy; q(7,6)=q_y_vy; q(7,7)=q_vy_vy;
    q(8,8)=q_r;
    // clang-format on
    return q;
  };
  auto u_r = [this](const Eigen::VectorXd& z)
  {
    Eigen::DiagonalMatrix<double, 4> r;
    double x = cfg_.noise.r_xyz_factor;
    r.diagonal() << std::abs(x * z[0]), std::abs(x * z[1]), std::abs(x * z[2]),
        cfg_.noise.r_yaw;
    return r;
  };
  Eigen::DiagonalMatrix<double, 9> p0;
  p0.setIdentity();
  ekf_.ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};

  // ---------------- Topics & 回调 ----------------
  // 装甲板识别结果订阅
  LibXR::Topic::Domain armor_detector_domain = LibXR::Topic::Domain("armor_detector");
  LibXR::Topic armors_topic = LibXR::Topic(
      LibXR::Topic::WaitTopic(ArmorTrackerArmorsTopicName(), UINT32_MAX, &armor_detector_domain));
  auto armors_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto* armors_msg = reinterpret_cast<ArmorDetectionsMessage*>(data.addr_);
        if (self->params_is_changed_ == true)
        {
          self->SetConfig(self->cfg_);
          self->params_is_changed_ = false;
        }
        self->ArmorsCallback(armors_msg->results, armors_msg->image_timestamp_us);
      },
      this);
  armors_topic.RegisterCallback(armors_cb);

  // 弹丸速度订阅（用于弹道解算初始化）
  LibXR::Topic::Domain referee_domain = LibXR::Topic::Domain("referee");
  LibXR::Topic bullet_speed_tp =
      LibXR::Topic::FindOrCreate<float>("bullet_speed", &referee_domain);
  auto velocity_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto velocity_msg = reinterpret_cast<float*>(data.addr_);
        self->VelocityCallback(*velocity_msg);
      },
      this);
  bullet_speed_tp.RegisterCallback(velocity_cb);

  // 云台姿态订阅
  LibXR::Topic::Domain gimbal_domain = LibXR::Topic::Domain("gimbal");
  LibXR::Topic gimbal_rotation_topic =
      LibXR::Topic::FindOrCreate<LibXR::Quaternion<float>>("rotation", &gimbal_domain);
  auto base_rotation_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        LibXR::Mutex::LockGuard lock(self->io_.gimbal_rotation_lock);
        auto base_rotation_msg = reinterpret_cast<LibXR::Quaternion<float>*>(data.addr_);
        self->io_.gimbal_rotation =
            LibXR::Quaternion<double>(base_rotation_msg->w(), base_rotation_msg->x(),
                                      base_rotation_msg->y(), base_rotation_msg->z());
      },
      this);
  gimbal_rotation_topic.RegisterCallback(base_rotation_cb);

  sync_frame_pose_thread_.Create(this, SyncFramePoseThreadFun, "TrackPoseSync",
                                 static_cast<size_t>(1024 * 64),
                                 LibXR::Thread::Priority::LOW);

  io_.solver->SetFireCallback(
      [&](bool is_fire)
      {
        XR_LOG_INFO("is_fire: {}", is_fire);
        // uint8_t fire_notify = is_fire ? 1 : 0;
        uint8_t fire_notify = 0;
        io_.fire_notify_topic.Publish(fire_notify);
      });

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
  preview_image_thread_.Create(this, PreviewImageThreadFun, "TrackPreviewImg",
                               static_cast<size_t>(1024 * 128),
                               LibXR::Thread::Priority::LOW);
#endif
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SyncFramePoseThreadFun(ArmorTracker<CameraInfoV>* self)
{
  XR_LOG_PASS("ArmorTracker pose sync uses image=%s imu=%s", self->sync_.ImageTopicName(),
              self->sync_.ImuTopicName());

  while (true)
  {
    typename FrameSync::Subscriber subscriber(self->sync_);
    if (!subscriber.Valid())
    {
      LibXR::Thread::Sleep(200);
      continue;
    }

    SyncedFrame synced_frame;
    while (true)
    {
      const auto wait_ans =
          subscriber.Wait(synced_frame, kArmorTrackerSyncFrameWaitTimeoutMs);
      if (wait_ans == LibXR::ErrorCode::TIMEOUT)
      {
        continue;
      }
      if (wait_ans != LibXR::ErrorCode::OK)
      {
        break;
      }

      const auto* image_frame = synced_frame.GetImageFrame();
      if (image_frame != nullptr)
      {
        self->PushCameraPose(image_frame->timestamp_us,
                             armor_tracker_detail::PackedCameraRotation(
                                 synced_frame.imu.rotation_wxyz));
      }
    }
  }
}

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::RenderPreviewFrame(ArmorTracker<CameraInfoV>* self, cv::Mat frame)
{
  if (frame.empty())
  {
    return;
  }

  EkfPointsMsg& ekf = self->ekf_msg_;
  const CameraInfo& cam = ArmorTracker<CameraInfoV>::kCameraInfo;
  const bool has_distortion =
      (cam.distortion_model == CameraTypes::DistortionModel::PLUMB_BOB);

  const auto& k_arr = cam.camera_matrix;
  cv::Mat k = (cv::Mat_<double>(3, 3) << k_arr[0], k_arr[1], k_arr[2], k_arr[3],
               k_arr[4], k_arr[5], k_arr[6], k_arr[7], k_arr[8]);

  cv::Mat d;
  if (has_distortion)
  {
    std::vector<double> dvec = {cam.distortion_coefficients[0],
                                cam.distortion_coefficients[1],
                                cam.distortion_coefficients[2],
                                cam.distortion_coefficients[3],
                                cam.distortion_coefficients[4]};
    d = cv::Mat(dvec).clone().reshape(1, 1);
  }

  const double sx = static_cast<double>(frame.cols) / static_cast<double>(cam.width);
  const double sy = static_cast<double>(frame.rows) / static_cast<double>(cam.height);
  cv::Mat k_scaled = k.clone();
  k_scaled.at<double>(0, 0) *= sx;
  k_scaled.at<double>(1, 1) *= sy;
  k_scaled.at<double>(0, 2) *= sx;
  k_scaled.at<double>(1, 2) *= sy;

  auto project = [&](const Eigen::Vector3d& pc, cv::Point2d& uv) -> bool
  {
    if (!(pc.z() > 1e-6) || !std::isfinite(pc.x()) || !std::isfinite(pc.y()) ||
        !std::isfinite(pc.z()))
    {
      return false;
    }

    std::vector<cv::Point3d> obj{cv::Point3d(pc.x(), pc.y(), pc.z())};
    static cv::Mat rvec = cv::Mat::zeros(1, 3, CV_64F);
    static cv::Mat tvec = cv::Mat::zeros(1, 3, CV_64F);
    std::vector<cv::Point2d> imgpts;
    cv::projectPoints(obj, rvec, tvec, k_scaled, d, imgpts);
    uv = imgpts[0];
    return (0 <= uv.x && uv.x < frame.cols && 0 <= uv.y && uv.y < frame.rows);
  };

  if (ekf.valid[0])
  {
    cv::Point2d uv;
    Eigen::Vector3d pc(ekf.center_cam.x(), ekf.center_cam.y(), ekf.center_cam.z());
    if (project(pc, uv))
    {
      cv::circle(frame, uv, 5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
      cv::putText(frame, "C", uv + cv::Point2d(6, -6), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
  }

  for (int i = 0; i < std::min<int>(ekf.count, 4); ++i)
  {
    if (!ekf.valid[i + 1])
    {
      continue;
    }
    cv::Point2d uv;
    Eigen::Vector3d pc(ekf.armors_cam[i].x(), ekf.armors_cam[i].y(),
                       ekf.armors_cam[i].z());
    if (project(pc, uv))
    {
      cv::circle(frame, uv, 4, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
      char buf[16];
      (void)std::snprintf(buf, sizeof(buf), "A%d", i);
      cv::putText(frame, buf, uv + cv::Point2d(6, -6), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
  }

  for (int i = 0; i < std::min<int>(ekf.count, 4); ++i)
  {
    if (!ekf.valid[0] || !ekf.valid[i + 1])
    {
      continue;
    }
    cv::Point2d uc, ua;
    Eigen::Vector3d pc_c(ekf.center_cam.x(), ekf.center_cam.y(), ekf.center_cam.z());
    Eigen::Vector3d pc_a(ekf.armors_cam[i].x(), ekf.armors_cam[i].y(),
                         ekf.armors_cam[i].z());
    if (project(pc_c, uc) && project(pc_a, ua))
    {
      cv::line(frame, uc, ua, cv::Scalar(80, 180, 255), 1, cv::LINE_AA);
    }
  }

  cv::imshow("ekf_overlay", frame);
  cv::waitKey(1);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::PreviewImageThreadFun(ArmorTracker<CameraInfoV>* self)
{
  if (!ArmorTrackerPreviewUiAvailable())
  {
    XR_LOG_WARN("ArmorTracker preview disabled because no display backend is available");
    return;
  }

  XR_LOG_PASS("ArmorTracker preview uses sync frame topic");

  while (true)
  {
    typename FrameSync::Subscriber subscriber(self->sync_);
    if (!subscriber.Valid())
    {
      LibXR::Thread::Sleep(200);
      continue;
    }

    SyncedFrame synced_frame;
    while (true)
    {
      const auto wait_ans =
          subscriber.Wait(synced_frame, kArmorTrackerSyncFrameWaitTimeoutMs);
      if (wait_ans == LibXR::ErrorCode::TIMEOUT)
      {
        continue;
      }
      if (wait_ans != LibXR::ErrorCode::OK)
      {
        break;
      }

      const auto* image_frame = synced_frame.GetImageFrame();
      if (image_frame != nullptr)
      {
        const int cv_type = ArmorTrackerCvTypeFromEncoding(kCameraInfo.encoding);
        if (cv_type >= 0)
        {
          cv::Mat input(static_cast<int>(kCameraInfo.height),
                        static_cast<int>(kCameraInfo.width), cv_type,
                        const_cast<uint8_t*>(image_frame->data.data()),
                        static_cast<size_t>(kCameraInfo.step));
          cv::Mat frame = ArmorTrackerConvertToBgrWithEncoding(input, kCameraInfo.encoding);
          if (!frame.empty())
          {
            RenderPreviewFrame(self, frame);
          }
        }
      }
    }
  }
}

#endif

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::OnMonitor() {}

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::ArmorImageArea(const ArmorDetectorResult& armor)
{
  return std::abs(cv::contourArea(
      std::vector<cv::Point2f>(armor.points.begin(), armor.points.end())));
}

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::TimestampDeltaSeconds(uint64_t newer, uint64_t older)
{
  if (newer > older)
  {
    return static_cast<double>(newer - older) / 1000000.0;
  }
  return 0.0;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::PushCameraPose(uint64_t timestamp_us,
                                  const LibXR::Quaternion<double>& camera_rotation)
{
  LibXR::Mutex::LockGuard lock(io_.gimbal_rotation_lock);
  io_.gimbal_rotation = camera_rotation;
  io_.latest_camera_pose =
      ArmorTrackerCameraRotationToTrackerWorldPose(io_.gimbal_rotation,
                                       io_.gimbal_to_camera_transform_static);
  io_.latest_camera_pose_valid = true;
  io_.camera_pose_history[io_.camera_pose_history_head] =
      TimedCameraPose{timestamp_us, io_.latest_camera_pose};
  io_.camera_pose_history_head =
      (io_.camera_pose_history_head + 1) % IOBlock::kCameraPoseHistorySize;
  io_.camera_pose_history_count =
      std::min(io_.camera_pose_history_count + 1, IOBlock::kCameraPoseHistorySize);
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::LookupCameraPose(uint64_t image_timestamp_us,
                                    LibXR::Transform<double>& pose_out)
{
  LibXR::Mutex::LockGuard lock(io_.gimbal_rotation_lock);
  if (image_timestamp_us > 0 && io_.camera_pose_history_count > 0)
  {
    uint64_t best_diff = UINT64_MAX;
    std::size_t best_index = 0;
    bool found = false;
    for (std::size_t i = 0; i < io_.camera_pose_history_count; ++i)
    {
      const std::size_t index =
          (io_.camera_pose_history_head + IOBlock::kCameraPoseHistorySize - 1 - i) %
          IOBlock::kCameraPoseHistorySize;
      const auto& msg = io_.camera_pose_history[index];
      const uint64_t ts = msg.timestamp_us;
      if (ts == 0)
      {
        continue;
      }
      const uint64_t diff = TimestampAbsDiff(ts, image_timestamp_us);
      if (!found || diff < best_diff)
      {
        found = true;
        best_diff = diff;
        best_index = index;
        if (diff == 0)
        {
          break;
        }
      }
    }
    if (found && best_diff <= 20000)
    {
      pose_out = io_.camera_pose_history[best_index].pose;
      io_.latest_camera_pose = pose_out;
      io_.latest_camera_pose_valid = true;
      return true;
    }
  }

  if (io_.latest_camera_pose_valid)
  {
    pose_out = io_.latest_camera_pose;
    return false;
  }

  pose_out = LibXR::Transform<double>(io_.gimbal_rotation, {0.0, 0.0, 0.0}) +
             io_.gimbal_to_camera_transform_static;
  io_.latest_camera_pose = pose_out;
  io_.latest_camera_pose_valid = true;
  return false;
}

template <CameraTypes::CameraInfo CameraInfoV>
int ArmorTracker<CameraInfoV>::FindDetectionTrackId(std::size_t armor_index) const
{
  return image_tracker_.FindDetectionTrackId(armor_index);
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::IsDetectionTrackConfirmed(std::size_t armor_index) const
{
  return image_tracker_.IsDetectionTrackConfirmed(armor_index);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::UpdateImageIdTracks(const ArmorDetectorResults& armors_msg,
                                       uint64_t image_timestamp_us)
{
  image_tracker_.Update(
      armors_msg, image_timestamp_us,
      armor_tracker::ImageTrackConfig{
          .appear_hits = IdTrackAppearHits(),
          .appear_timeout_sec = IdTrackAppearTimeoutSec(),
          .tentative_misses = IdTrackTentativeMisses(),
          .tentative_timeout_sec = IdTrackTentativeTimeoutSec(),
          .disappear_misses = IdTrackDisappearMisses(),
          .disappear_timeout_sec = IdTrackDisappearTimeoutSec(),
      });

  if (rt_.tracked_face_track_id_valid)
  {
    bool found_bound_track = false;
    for (const auto& track : image_tracker_.Tracks())
    {
      if (track.active && track.track_id == rt_.tracked_face_track_id)
      {
        found_bound_track = true;
        break;
      }
    }
    if (!found_bound_track)
    {
      XR_LOG_DEBUG("Tracker face bind cleared: tracked_id=%d",
                   static_cast<int>(rt_.tracked_face_track_id));
      rt_.tracked_face_track_id_valid = false;
    }
  }

  const int bound_face_count = std::max(1, static_cast<int>(rt_.tracked_armors_num));
  for (int face_index = 0; face_index < 4; ++face_index)
  {
    if (!rt_.face_track_id_valid[face_index])
    {
      continue;
    }
    bool found_bound_track = false;
    for (const auto& track : image_tracker_.Tracks())
    {
      if (track.active && track.track_id == rt_.face_track_id[face_index])
      {
        found_bound_track = true;
        break;
      }
    }
    if (!found_bound_track)
    {
      rt_.face_track_id_valid[face_index] = false;
    }
  }
  if (rt_.tracked_face_track_id_valid)
  {
    rt_.face_track_id_valid[0] = true;
    rt_.face_track_id[0] = rt_.tracked_face_track_id;
  }
  else
  {
    rt_.face_track_id_valid[0] = false;
  }
  for (int face_index = bound_face_count; face_index < 4; ++face_index)
  {
    rt_.face_track_id_valid[face_index] = false;
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::Init(const ArmorDetectorResults& armors_msg)
{
  if (armors_msg.empty())
  {
    return;
  }

  double min_distance = DBL_MAX;
  std::size_t tracked_index = 0;
  rt_.tracked_armor = armors_msg[0];
  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    if (armor.distance_to_image_center < min_distance)
    {
      min_distance = armor.distance_to_image_center;
      tracked_index = armor_index;
      rt_.tracked_armor = armor;
    }
  }

  const int detection_track_id = FindDetectionTrackId(tracked_index);
  const bool detection_track_confirmed = IsDetectionTrackConfirmed(tracked_index);
  rt_.tracked_face_track_id_valid = detection_track_id >= 0 && detection_track_confirmed;
  rt_.tracked_face_track_id = detection_track_id >= 0 ?
      static_cast<uint16_t>(detection_track_id) : 0;
  rt_.face_track_id_valid.fill(false);
  rt_.face_track_id.fill(0);
  if (rt_.tracked_face_track_id_valid)
  {
    rt_.face_track_id_valid[0] = true;
    rt_.face_track_id[0] = rt_.tracked_face_track_id;
  }

  InitEKF(rt_.tracked_armor);
  XR_LOG_DEBUG("Init EKF!");

  rt_.tracked_id = rt_.tracked_armor.number;
  rt_.state = State::DETECTING;
  UpdateArmorsNum(rt_.tracked_armor);
  candidate_debug_msg_ = CandidateDebugMsg{};
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::Update(const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us)
{
  Eigen::VectorXd ekf_prediction = ekf_.ekf.Predict();  // 预测
  XR_LOG_DEBUG("EKF predict");
  (void)image_timestamp_us;
  bool matched = false;
  ekf_.state = ekf_prediction;
  rt_.face_switch_cooldown_remaining =
      std::max(0.0, rt_.face_switch_cooldown_remaining - time_.dt);
  const bool symmetric_geometry_enabled = SymmetricGeometryEnabled();
  if (symmetric_geometry_enabled)
  {
    rt_.another_r = ekf_.state(8);
    rt_.dz = 0.0;
    rt_.dz_abs_ref = 0.0;
  }

  ArmorTracker<CameraInfoV>::CandidateDebugMsg candidate_debug{};
  std::fill(candidate_debug.detection_track_ids.begin(),
            candidate_debug.detection_track_ids.end(), static_cast<int16_t>(-1));
  std::fill(candidate_debug.detection_track_confirmed.begin(),
            candidate_debug.detection_track_confirmed.end(), static_cast<uint8_t>(0));
  const auto face_policy = BuildFaceSelectionPolicy();
  FillCandidateDebugPolicy(candidate_debug, ekf_prediction, face_policy);
  rt_.info_position_diff = DBL_MAX;
  rt_.info_yaw_diff = DBL_MAX;

  if (!armors_msg.empty())
  {
    const auto face_selection = armor_tracker::SelectFaceMatch(
        armors_msg, BuildFaceSelectionTrackedState(), face_policy,
        GetCameraWorldPosition(),
        ekf_prediction(ExtendedKalmanFilter::V_YAW),
        [this](std::size_t armor_index) { return FindDetectionTrackId(armor_index); },
        [this](std::size_t armor_index)
        { return IsDetectionTrackConfirmed(armor_index); },
        [this, &ekf_prediction](int face_index)
        { return GetArmorPositionFromState(ekf_prediction, face_index); },
        [this, &ekf_prediction](int face_index)
        { return GetArmorYawFromState(ekf_prediction, face_index); });
    FillCandidateDebugFromSelection(face_selection, candidate_debug);
    matched = ApplyFaceSelection(face_selection, armors_msg, ekf_prediction,
                                 candidate_debug);
  }

  if (symmetric_geometry_enabled)
  {
    rt_.another_r = ekf_.state(8);
    rt_.dz = 0.0;
  }

  // 防止半径发散
  if (ekf_.state(8) < 0.12)
  {
    ekf_.state(8) = 0.12;
    ekf_.ekf.SetState(ekf_.state);
  }
  else if (ekf_.state(8) > 0.4)
  {
    ekf_.state(8) = 0.4;
    ekf_.ekf.SetState(ekf_.state);
  }

  AdvanceTrackerState(matched);

  candidate_debug_msg_ = candidate_debug;
  candidate_debug_msg_.tracked_face_track_id_valid =
      rt_.tracked_face_track_id_valid ? 1 : 0;
  candidate_debug_msg_.tracked_face_track_id =
      rt_.tracked_face_track_id_valid ? static_cast<int16_t>(rt_.tracked_face_track_id)
                                      : static_cast<int16_t>(-1);
}

template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker::ObserverPolicy ArmorTracker<CameraInfoV>::BuildObserverPolicy() const
{
  armor_tracker::ObserverPolicy policy{};
  policy.single_armor_mode = SingleArmorModeEnabled();
  policy.symmetric_geometry_enabled = SymmetricGeometryEnabled();
  policy.face_switch_recenter_enabled = FaceSwitchRecenterEnabled();
  policy.max_match_distance = cfg_.match.max_match_distance;
  policy.max_match_yaw_diff = cfg_.match.max_match_yaw_diff;
  policy.initial_radius = 0.26;
  return policy;
}

template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker::ObserverRuntime ArmorTracker<CameraInfoV>::BuildObserverRuntime() const
{
  armor_tracker::ObserverRuntime runtime{};
  runtime.tracked_id = rt_.tracked_id;
  runtime.tracked_armor_type = rt_.tracked_armor.type;
  runtime.tracked_armors_num = static_cast<int>(rt_.tracked_armors_num);
  runtime.tracked_face_track_id_valid = rt_.tracked_face_track_id_valid;
  runtime.tracked_face_track_id = rt_.tracked_face_track_id;
  runtime.face_track_id_valid = rt_.face_track_id_valid;
  runtime.face_track_id = rt_.face_track_id;
  runtime.last_yaw = rt_.last_yaw;
  runtime.face_switch_cooldown_remaining = rt_.face_switch_cooldown_remaining;
  runtime.dz = rt_.dz;
  runtime.dz_abs_ref = rt_.dz_abs_ref;
  runtime.another_r = rt_.another_r;
  return runtime;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ApplyObserverRuntime(
    const armor_tracker::ObserverRuntime& runtime)
{
  rt_.tracked_id = runtime.tracked_id;
  rt_.tracked_armors_num =
      static_cast<ArmorsNum>(runtime.tracked_armors_num);
  rt_.tracked_face_track_id_valid = runtime.tracked_face_track_id_valid;
  rt_.tracked_face_track_id = runtime.tracked_face_track_id;
  rt_.face_track_id_valid = runtime.face_track_id_valid;
  rt_.face_track_id = runtime.face_track_id;
  rt_.last_yaw = runtime.last_yaw;
  rt_.face_switch_cooldown_remaining = runtime.face_switch_cooldown_remaining;
  rt_.dz = runtime.dz;
  rt_.dz_abs_ref = runtime.dz_abs_ref;
  rt_.another_r = runtime.another_r;
}

template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker::FaceBindingRuntime ArmorTracker<CameraInfoV>::BuildFaceBindingRuntime() const
{
  armor_tracker::FaceBindingRuntime runtime{};
  runtime.tracked_armors_num = static_cast<int>(rt_.tracked_armors_num);
  runtime.tracked_face_track_id_valid = rt_.tracked_face_track_id_valid;
  runtime.tracked_face_track_id = rt_.tracked_face_track_id;
  runtime.face_track_id_valid = rt_.face_track_id_valid;
  runtime.face_track_id = rt_.face_track_id;
  return runtime;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ApplyFaceBindingRuntime(
    const armor_tracker::FaceBindingRuntime& runtime)
{
  rt_.tracked_face_track_id_valid = runtime.tracked_face_track_id_valid;
  rt_.tracked_face_track_id = runtime.tracked_face_track_id;
  rt_.face_track_id_valid = runtime.face_track_id_valid;
  rt_.face_track_id = runtime.face_track_id;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ApplySelectedIdentity(
    const armor_tracker::FaceMatchCandidate& selected_candidate)
{
  rt_.tracked_armor = selected_candidate.armor;
  if (rt_.tracked_id == ArmorNumber::INVALID || selected_candidate.same_number)
  {
    rt_.tracked_id = selected_candidate.armor.number;
  }
  else
  {
    XR_LOG_DEBUG(
        "Tracker keep tracked id: tracked_num=%d selected_num=%d face=%d image_track=%d confirmed=%d",
        static_cast<int>(rt_.tracked_id),
        static_cast<int>(selected_candidate.armor.number),
        selected_candidate.face_index, selected_candidate.image_track_id,
        selected_candidate.confirmed_image_track ? 1 : 0);
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ApplySelectedFaceBinding(
    const armor_tracker::FaceMatchCandidate& selected_candidate, bool did_face_switch)
{
  const int tracked_face_track_before =
      rt_.tracked_face_track_id_valid ? static_cast<int>(rt_.tracked_face_track_id)
                                      : -1;
  auto binding_runtime = BuildFaceBindingRuntime();
  armor_tracker::ApplySelectedFaceBinding(binding_runtime, selected_candidate,
                                          did_face_switch);
  ApplyFaceBindingRuntime(binding_runtime);

  if (did_face_switch || (selected_candidate.image_track_id >= 0 &&
                          selected_candidate.confirmed_image_track))
  {
    XR_LOG_DEBUG(
        "Tracker face bind: switch=%d sel_face=%d sel_image=%d confirmed=%d tracked_before=%d tracked_after=%d slots=[%d,%d,%d,%d] valid=[%d,%d,%d,%d]",
        did_face_switch ? 1 : 0, selected_candidate.face_index,
        selected_candidate.image_track_id,
        selected_candidate.confirmed_image_track ? 1 : 0,
        tracked_face_track_before,
        rt_.tracked_face_track_id_valid ? static_cast<int>(rt_.tracked_face_track_id)
                                        : -1,
        static_cast<int>(rt_.face_track_id[0]),
        static_cast<int>(rt_.face_track_id[1]),
        static_cast<int>(rt_.face_track_id[2]),
        static_cast<int>(rt_.face_track_id[3]),
        rt_.face_track_id_valid[0] ? 1 : 0,
        rt_.face_track_id_valid[1] ? 1 : 0,
        rt_.face_track_id_valid[2] ? 1 : 0,
        rt_.face_track_id_valid[3] ? 1 : 0);
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ApplySelectedMeasurementUpdate(
    const armor_tracker::FaceMatchCandidate& selected_candidate,
    const ArmorDetectorResults& armors_msg, const Eigen::VectorXd& ekf_prediction,
    bool did_face_switch)
{
  const double preserved_v_yaw = ekf_prediction(ExtendedKalmanFilter::V_YAW);
  rt_.last_yaw = selected_candidate.measured_yaw;

  const auto p = selected_candidate.armor.pose.translation;
  ekf_.measurement =
      Eigen::Vector4d(p.x(), p.y(), p.z(), selected_candidate.measured_yaw);
  ekf_.state = ekf_.ekf.Update(ekf_.measurement);
  if (did_face_switch)
  {
    // Face relabeling is discrete geometry bookkeeping, not physical angular acceleration.
    ekf_.state(ExtendedKalmanFilter::V_YAW) = preserved_v_yaw;
    ekf_.ekf.SetState(ekf_.state);
  }

  UpdateDzReference(armors_msg, selected_candidate.armor);
  if (MultiArmorFuseEnabled())
  {
    FuseMultiArmorObservation(armors_msg);
  }
  XR_LOG_DEBUG("EKF update");
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::FillCandidateDebugFromSelection(
    const armor_tracker::FaceSelectionResult& selection,
    CandidateDebugMsg& candidate_debug)
{
  candidate_debug.count = selection.debug.count;
  candidate_debug.selected_index = selection.debug.selected_index;
  candidate_debug.detection_count = selection.debug.detection_count;
  candidate_debug.preferred_adjacent_face = selection.debug.preferred_adjacent_face;
  candidate_debug.has_same_number_candidate =
      selection.debug.has_same_number_candidate;
  candidate_debug.relaxed_same_face_distance =
      selection.debug.relaxed_same_face_distance;
  candidate_debug.relaxed_face_switch_distance =
      selection.debug.relaxed_face_switch_distance;
  candidate_debug.relaxed_face_switch_yaw_diff =
      selection.debug.relaxed_face_switch_yaw_diff;
  candidate_debug.best_same_face_score = selection.debug.best_same_face_score;
  candidate_debug.best_switch_face_score = selection.debug.best_switch_face_score;
  candidate_debug.same_face_matched = selection.debug.same_face_matched;
  candidate_debug.switch_face_matched = selection.debug.switch_face_matched;
  candidate_debug.switch_blocked_by_timeout =
      selection.debug.switch_blocked_by_timeout;
  candidate_debug.switch_allowed = selection.debug.switch_allowed;
  candidate_debug.detection_track_ids = selection.debug.detection_track_ids;
  candidate_debug.detection_track_confirmed =
      selection.debug.detection_track_confirmed;

  for (std::size_t item_index = 0; item_index < selection.debug.count; ++item_index)
  {
    const auto& src = selection.debug.items[item_index];
    auto& dst = candidate_debug.items[item_index];
    dst.armor_index = src.armor_index;
    dst.face_index = src.face_index;
    dst.same_number = src.same_number;
    dst.image_track_id = src.image_track_id;
    dst.image_track_confirmed = src.image_track_confirmed;
    dst.same_persistent_track = src.same_persistent_track;
    dst.number = src.number;
    dst.type = src.type;
    dst.score = src.score;
    dst.position_diff = src.position_diff;
    dst.yaw_diff = src.yaw_diff;
    dst.view_bonus = src.view_bonus;
    dst.area_score = src.area_score;
    dst.frontality = src.frontality;
    dst.center_x = src.center_x;
    dst.center_y = src.center_y;
    dst.predicted_yaw = src.predicted_yaw;
    dst.measured_yaw = src.measured_yaw;
  }
  candidate_debug.face_switch_cooldown_remaining =
      static_cast<float>(rt_.face_switch_cooldown_remaining);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::FillCandidateDebugPolicy(
    CandidateDebugMsg& candidate_debug, const Eigen::VectorXd& ekf_prediction,
    const armor_tracker::FaceSelectionPolicy& face_policy) const
{
  candidate_debug.face_switch_enabled = face_policy.face_switch_enabled ? 1 : 0;
  candidate_debug.relaxed_face_switch_enabled =
      face_policy.relaxed_face_switch_enabled ? 1 : 0;
  candidate_debug.odd_face_switch_enabled =
      face_policy.odd_face_switch_enabled ? 1 : 0;
  candidate_debug.view_priority_enabled =
      face_policy.view_priority_enabled ? 1 : 0;
  candidate_debug.directional_face_switch_enabled =
      face_policy.directional_face_switch_enabled ? 1 : 0;
  candidate_debug.tracked_face_track_id_valid =
      rt_.tracked_face_track_id_valid ? 1 : 0;
  candidate_debug.tracked_face_track_id =
      rt_.tracked_face_track_id_valid
          ? static_cast<int16_t>(rt_.tracked_face_track_id)
          : static_cast<int16_t>(-1);
  candidate_debug.tracked_armors_num = static_cast<uint8_t>(rt_.tracked_armors_num);
  candidate_debug.predicted_vyaw =
      static_cast<float>(ekf_prediction(ExtendedKalmanFilter::V_YAW));
  candidate_debug.max_match_distance =
      static_cast<float>(cfg_.match.max_match_distance);
  candidate_debug.max_match_yaw_diff =
      static_cast<float>(cfg_.match.max_match_yaw_diff);
  candidate_debug.face_switch_score_deadzone =
      static_cast<float>(face_policy.face_switch_score_deadzone);
  candidate_debug.face_switch_position_deadzone =
      static_cast<float>(face_policy.face_switch_position_deadzone);
  candidate_debug.face_switch_yaw_deadzone =
      static_cast<float>(face_policy.face_switch_yaw_deadzone);
  candidate_debug.face_switch_timeout_sec =
      static_cast<float>(face_policy.face_switch_timeout_sec);
  candidate_debug.face_switch_cooldown_remaining =
      static_cast<float>(rt_.face_switch_cooldown_remaining);
}

template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker::FaceSelectionPolicy ArmorTracker<CameraInfoV>::BuildFaceSelectionPolicy() const
{
  armor_tracker::FaceSelectionPolicy face_policy{};
  face_policy.single_armor_mode = SingleArmorModeEnabled();
  face_policy.id_assist_enabled = IdAssistEnabled();
  face_policy.face_switch_enabled = FaceSwitchEnabled();
  face_policy.relaxed_face_switch_enabled = RelaxedFaceSwitchEnabled();
  face_policy.odd_face_switch_enabled = OddFaceSwitchEnabled();
  face_policy.view_priority_enabled = ViewPriorityEnabled();
  face_policy.directional_face_switch_enabled = DirectionalFaceSwitchEnabled();
  face_policy.symmetric_geometry_enabled = SymmetricGeometryEnabled();
  face_policy.max_match_distance = cfg_.match.max_match_distance;
  face_policy.max_match_yaw_diff = cfg_.match.max_match_yaw_diff;
  face_policy.single_armor_image_center_gate_px = SingleArmorImageCenterGatePx();
  face_policy.single_armor_area_log_gate = SingleArmorAreaLogGate();
  face_policy.face_switch_score_deadzone = FaceSwitchScoreDeadzone();
  face_policy.face_switch_position_deadzone = FaceSwitchPositionDeadzone();
  face_policy.face_switch_yaw_deadzone = FaceSwitchYawDeadzone();
  face_policy.face_switch_timeout_sec = FaceSwitchTimeoutSec();
  face_policy.id_assist_same_face_center_gate_px = IdAssistSameFaceCenterGatePx();
  face_policy.id_assist_same_face_area_log_gate = IdAssistSameFaceAreaLogGate();
  return face_policy;
}

template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker::FaceSelectionTrackedState
ArmorTracker<CameraInfoV>::BuildFaceSelectionTrackedState() const
{
  armor_tracker::FaceSelectionTrackedState tracked_state{};
  tracked_state.tracked_armor = rt_.tracked_armor;
  tracked_state.tracked_id = rt_.tracked_id;
  tracked_state.tracked_armors_num = static_cast<int>(rt_.tracked_armors_num);
  tracked_state.tracked_face_track_id_valid = rt_.tracked_face_track_id_valid;
  tracked_state.tracked_face_track_id = rt_.tracked_face_track_id;
  tracked_state.face_switch_cooldown_remaining =
      rt_.face_switch_cooldown_remaining;
  tracked_state.dz_abs_ref = rt_.dz_abs_ref;
  return tracked_state;
}

template <CameraTypes::CameraInfo CameraInfoV>
Eigen::Vector3d ArmorTracker<CameraInfoV>::GetCameraWorldPosition()
{
  LibXR::Mutex::LockGuard lock(io_.gimbal_rotation_lock);
  const LibXR::Transform<double> t_wc =
      io_.latest_camera_pose_valid
          ? io_.latest_camera_pose
          : (LibXR::Transform<double>(io_.gimbal_rotation, {0.0, 0.0, 0.0}) +
             io_.gimbal_to_camera_transform_static);
  return Eigen::Vector3d(t_wc.translation.x(), t_wc.translation.y(),
                         t_wc.translation.z());
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::AdvanceTrackerState(bool matched)
{
  if (rt_.state == State::DETECTING)
  {
    if (matched)
    {
      rt_.detect_count++;
      if (rt_.detect_count > rt_.tracking_thres)
      {
        rt_.detect_count = 0;
        rt_.state = State::TRACKING;
      }
      return;
    }

    rt_.detect_count = 0;
    rt_.state = State::LOST;
    return;
  }

  if (rt_.state == State::TRACKING)
  {
    if (!matched)
    {
      rt_.state = State::TEMP_LOST;
      rt_.lost_count++;
    }
    return;
  }

  if (rt_.state == State::TEMP_LOST)
  {
    if (!matched)
    {
      rt_.lost_count++;
      if (rt_.lost_count > rt_.lost_thres)
      {
        rt_.lost_count = 0;
        rt_.state = State::LOST;
      }
      return;
    }

    rt_.state = State::TRACKING;
    rt_.lost_count = 0;
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::ApplyFaceSelection(
    const armor_tracker::FaceSelectionResult& selection,
    const ArmorDetectorResults& armors_msg, const Eigen::VectorXd& ekf_prediction,
    CandidateDebugMsg& candidate_debug)
{
  rt_.info_position_diff = selection.info_position_diff;
  rt_.info_yaw_diff = selection.info_yaw_diff;
  const armor_tracker::FaceSelectionLogContext log_context{
      .tracked_id = rt_.tracked_id,
      .face_switch_timeout_sec = candidate_debug.face_switch_timeout_sec,
      .face_switch_score_deadzone = candidate_debug.face_switch_score_deadzone,
      .face_switch_position_deadzone =
          candidate_debug.face_switch_position_deadzone,
      .face_switch_yaw_deadzone = candidate_debug.face_switch_yaw_deadzone,
      .face_switch_cooldown_remaining = rt_.face_switch_cooldown_remaining,
  };

  if (!selection.has_selected_candidate)
  {
    armor_tracker::LogRejectedSelection(selection, log_context);
    return false;
  }

  const auto& selected_candidate = selection.selected_candidate;
  const bool did_face_switch = selected_candidate.face_index != 0;
  candidate_debug.matched = 1;
  candidate_debug.accepted_mode = selection.accepted_mode;
  armor_tracker::LogAcceptedSelection(selection, log_context);

  if (did_face_switch)
  {
    SwitchTrackedFace(selected_candidate.face_index, selected_candidate.armor,
                      selected_candidate.measured_yaw);
    rt_.face_switch_cooldown_remaining = candidate_debug.face_switch_timeout_sec;
    candidate_debug.face_switch_cooldown_remaining =
        static_cast<float>(rt_.face_switch_cooldown_remaining);
  }

  ApplySelectedIdentity(selected_candidate);
  ApplySelectedFaceBinding(selected_candidate, did_face_switch);
  ApplySelectedMeasurementUpdate(selected_candidate, armors_msg, ekf_prediction,
                                 did_face_switch);
  return true;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::VelocityCallback(double velocity_msg)
{
  io_.solver->Init(velocity_msg);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ArmorsCallback(ArmorDetectorResults armors_msg,
                                  uint64_t image_timestamp_us)
{
  // 图像坐标 -> tracker 世界坐标。
  // 当前 tracker 世界系原点仍定义在云台，不是 Webots 全局世界；
  // 因此这里只使用按帧对齐的相机旋转，平移继续沿用静态 gimbal->camera 外参。
  LibXR::Transform<double> camera_pose_world{};
  LookupCameraPose(image_timestamp_us, camera_pose_world);
  for (auto& armor : armors_msg)
  {
    LibXR::Transform<double> tf = armor.pose;
    armor.pose = camera_pose_world + tf;
  }

  // 过滤异常装甲
  armors_msg.erase(
      std::remove_if(
          armors_msg.begin(), armors_msg.end(),
          [this](const ArmorDetectorResult& armor)
          {
            return std::abs(armor.pose.translation.z()) > cfg_.limits.max_z_position ||
                   Eigen::Vector2d(armor.pose.translation.x(), armor.pose.translation.y())
                           .norm() > cfg_.limits.max_armor_distance;
          }),
      armors_msg.end());

  UpdateImageIdTracks(armors_msg, image_timestamp_us);

  // 构造消息
  TrackerInfo info_msg{};
  SolveTrajectory::Target target_msg{};
  LibXR::EulerAngle<float> target_eulr;
  bool publish_target_eulr = false;
  Send send_msg{};
  ekf_msg_ = {};
  target_msg.id = ArmorNumber::INVALID;

  auto time = LibXR::Timebase::GetMicroseconds();

  // 跟踪更新
  if (rt_.state == State::LOST)
  {
    Init(armors_msg);
    target_msg.tracking = false;
  }
  else
  {
    // 优先使用图像时间戳，避免 Webots 低流速时 wall clock 与 sim time 脱钩。
    if (image_timestamp_us > 0 && time_.last_image_timestamp_us > 0 &&
        image_timestamp_us > time_.last_image_timestamp_us)
    {
      time_.dt = static_cast<double>(image_timestamp_us - time_.last_image_timestamp_us) /
                 1000000.0;
    }
    else
    {
      time_.dt = (time - time_.last_time).ToSecond();
    }
    if (time_.dt <= 0)
    {
      time_.dt = 1.0 / 100.0;
    }
    const double max_dt_before_reset =
        std::max(cfg_.thresholds.lost_time_thres, 0.15);
    if (time_.dt > max_dt_before_reset)
    {
      XR_LOG_WARN("ArmorTracker large dt %.3f s, clamp to default frame step",
                  time_.dt);
      time_.dt = 1.0 / 100.0;
    }
    rt_.lost_thres = static_cast<int>(cfg_.thresholds.lost_time_thres / time_.dt);
    if (rt_.lost_thres < 1)
    {
      rt_.lost_thres = 1;
    }

    Update(armors_msg, image_timestamp_us);

    // 发布 Info
    info_msg.position_diff = rt_.info_position_diff;
    info_msg.yaw_diff = rt_.info_yaw_diff;
    info_msg.position.x() = ekf_.measurement(0);
    info_msg.position.y() = ekf_.measurement(1);
    info_msg.position.z() = ekf_.measurement(2);
    info_msg.yaw = ekf_.measurement(3);
    io_.info_topic.Publish(info_msg);

    if (rt_.state == State::DETECTING)
    {
      target_msg.tracking = false;
    }
    else if (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST)
    {
      target_msg.tracking = true;
      const auto& state = ekf_.state;
      target_msg.id = rt_.tracked_id;
      target_msg.armors_num = static_cast<int>(rt_.tracked_armors_num);
      target_msg.position.x() = state(0);
      target_msg.velocity.x() = state(1);
      target_msg.position.y() = state(2);
      target_msg.velocity.y() = state(3);
      target_msg.position.z() = state(4);
      target_msg.velocity.z() = state(5);
      target_msg.yaw = state(6);
      target_msg.v_yaw = state(7);
      target_msg.radius_1 = state(8);
      target_msg.radius_2 = rt_.another_r;
      target_msg.dz = rt_.dz;

      XR_LOG_DEBUG(
          "Target position: (%.3f, %.3f, %.3f) velocity: (%.3f, %.3f, "
          "%.3f) yaw: %.3f "
          "v_yaw: %.3f radius_1: %.3f radius_2: %.3f dz: %.3f",
          target_msg.position.x(), target_msg.position.y(), target_msg.position.z(),
          target_msg.velocity.x(), target_msg.velocity.y(), target_msg.velocity.z(),
          target_msg.yaw, target_msg.v_yaw, target_msg.radius_1, target_msg.radius_2,
          target_msg.dz);

      float pitch = 0, yaw = 0, aim_x = 0, aim_y = 0, aim_z = 0;
      io_.solver->AutoSolveTrajectory(pitch, yaw, aim_x, aim_y, aim_z, &target_msg);

      XR_LOG_DEBUG(
          "AutoSolveTrajectory: pitch: %.3f yaw: %.3f aim_x: %.3f "
          "aim_y: %.3f aim_z: "
          "%.3f",
          pitch, yaw, aim_x, aim_y, aim_z);

      if (std::isfinite(pitch) && std::isfinite(yaw))
      {
        target_eulr.Pitch() = pitch;
        target_eulr.Yaw() = yaw;
        publish_target_eulr = true;
      }
      else
      {
        XR_LOG_WARN("ArmorTracker skipped non-finite target_eulr pitch=%f yaw=%f",
                    static_cast<double>(pitch), static_cast<double>(yaw));
      }

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
      Eigen::Vector3d pw_center, pw_armors[4];
      {
        const auto& st = ekf_.state;  // [xc,vxc,yc,vyc,za,vza,yaw,vyaw,r1]
        const double XC = st(0), YC = st(2), ZA = st(4);
        double center_z = ZA;
        if (rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
        {
          center_z += rt_.dz * 0.5;
        }

        pw_center = {XC, YC, center_z};
        for (int i = 0; i < 4; ++i)
        {
          pw_armors[i] = GetArmorPositionFromState(st, i);
        }
      }

      // === 计算 相机←世界 外参：T_CW = T_WC^-1 ===
      LibXR::Transform<double> t_wc;
      {
        LibXR::Mutex::LockGuard lock(io_.gimbal_rotation_lock);
        t_wc = io_.latest_camera_pose_valid
                   ? io_.latest_camera_pose
                   : (LibXR::Transform<double>(io_.gimbal_rotation, {0.0, 0.0, 0.0}) +
                      io_.gimbal_to_camera_transform_static);
      }
      auto r_wc = t_wc.rotation.ToRotationMatrix();
      Eigen::Matrix3d r_cw = r_wc.transpose();  // 相机←世界 旋转
      Eigen::Vector3d twc(t_wc.translation.x(), t_wc.translation.y(),
                          t_wc.translation.z());

      // === 变到相机系并发布 ===
      ekf_msg_.count = static_cast<uint8_t>(rt_.tracked_armors_num);

      auto to_cam = [&](const Eigen::Vector3d& pw,
                        LibXR::Position<double>& out_pt) -> bool
      {
        Eigen::Vector3d pc = r_cw * (pw - twc);
        out_pt = LibXR::Position<double>{pc.x(), pc.y(), pc.z()};
        return pc.z() > 1e-6;  // 在相机前方才算可见
      };

      // center
      ekf_msg_.valid[0] = to_cam(pw_center, ekf_msg_.center_cam);

      // armors
      for (int i = 0; i < 4; ++i)
      {
        ekf_msg_.valid[i + 1] = to_cam(pw_armors[i], ekf_msg_.armors_cam[i]);
      }
#endif
      send_msg.position.x() = aim_x;
      send_msg.position.y() = aim_y;
      send_msg.position.z() = aim_z;
      send_msg.v_yaw = target_msg.v_yaw;
      send_msg.pitch = pitch;
      send_msg.yaw = yaw;
    }
  }

  time_.last_time = time;
  time_.last_image_timestamp_us = image_timestamp_us;

  candidate_debug_msg_.image_timestamp_us = image_timestamp_us;
  io_.candidate_debug_topic.Publish(candidate_debug_msg_);
  if (publish_target_eulr)
  {
    io_.target_eulr_topic.Publish(target_eulr);
  }
  io_.send_topic.Publish(send_msg);
#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
  io_.ekf_points_topic.Publish(ekf_msg_);
#endif
  io_.target_topic.Publish(target_msg);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::InitEKF(const ArmorDetectorResult& a)
{
  auto runtime = BuildObserverRuntime();
  armor_tracker::InitEkfState(ekf_.state, runtime, BuildObserverPolicy(), a);
  ApplyObserverRuntime(runtime);
  ekf_.ekf.SetState(ekf_.state);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::UpdateArmorsNum(const ArmorDetectorResult&)
{
  auto runtime = BuildObserverRuntime();
  armor_tracker::UpdateArmorsNum(runtime, BuildObserverPolicy());
  ApplyObserverRuntime(runtime);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::UpdateDzReference(const ArmorDetectorResults& armors_msg,
                                     const ArmorDetectorResult& anchor)
{
  auto runtime = BuildObserverRuntime();
  armor_tracker::UpdateDzReference(runtime, BuildObserverPolicy(), armors_msg,
                                   anchor);
  ApplyObserverRuntime(runtime);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::FuseMultiArmorObservation(const ArmorDetectorResults& armors_msg)
{
  auto runtime = BuildObserverRuntime();
  const bool fused = armor_tracker::FuseMultiArmorObservation(
      runtime, ekf_.state, BuildObserverPolicy(), armors_msg,
      [this](std::size_t armor_index) { return FindDetectionTrackId(armor_index); },
      [this](std::size_t armor_index)
      { return IsDetectionTrackConfirmed(armor_index); });
  ApplyObserverRuntime(runtime);
  if (fused)
  {
    ekf_.ekf.SetState(ekf_.state);
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SwitchTrackedFace(int face_index,
                                     const ArmorDetectorResult& current_armor,
                                     double measured_yaw)
{
  auto runtime = BuildObserverRuntime();
  armor_tracker::SwitchTrackedFace(runtime, ekf_.state, BuildObserverPolicy(),
                                   face_index, current_armor, measured_yaw);
  ApplyObserverRuntime(runtime);
  ekf_.ekf.SetState(ekf_.state);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::HandleArmorJump(const ArmorDetectorResult& current_armor,
                                   double measured_yaw)
{
  auto runtime = BuildObserverRuntime();
  armor_tracker::HandleArmorJump(runtime, ekf_.state, BuildObserverPolicy(),
                                 current_armor, measured_yaw);
  ApplyObserverRuntime(runtime);
  ekf_.ekf.SetState(ekf_.state);
}

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::OrientationToYaw(const LibXR::Quaternion<double>& q)
{
  auto runtime = BuildObserverRuntime();
  const double yaw = armor_tracker::OrientationToYaw(q, runtime);
  ApplyObserverRuntime(runtime);
  return yaw;
}

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::GetArmorYawFromState(const Eigen::VectorXd& x, int face_index)
{
  return armor_tracker::GetArmorYawFromState(x, BuildObserverRuntime(), face_index);
}

template <CameraTypes::CameraInfo CameraInfoV>
Eigen::Vector3d ArmorTracker<CameraInfoV>::GetArmorPositionFromState(const Eigen::VectorXd& x,
                                                        int face_index)
{
  return armor_tracker::GetArmorPositionFromState(x, BuildObserverRuntime(),
                                                  BuildObserverPolicy(),
                                                  face_index);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SetConfig(const Config& cfg)
{
  if (cfg.thresholds.tracking_thres != rt_.tracking_thres)
  {
    rt_.tracking_thres = cfg.thresholds.tracking_thres;
  }
  cfg_ = cfg;
  if (cfg.solver.bias_time != solver_cfg_.bias_time ||
      cfg.solver.s_bias != solver_cfg_.s_bias || cfg.solver.z_bias != solver_cfg_.z_bias)
  {
    solver_cfg_ = cfg_.solver;
    io_.solver->SetBiasTime(solver_cfg_.bias_time);
    io_.solver->SetSBias(static_cast<float>(solver_cfg_.s_bias));
    io_.solver->SetZBias(static_cast<float>(solver_cfg_.z_bias));
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
int ArmorTracker<CameraInfoV>::CommandFun(ArmorTracker<CameraInfoV>* self, int argc, char** argv)
{
  if (argc == 1)
  {
    LibXR::STDIO::Printf("ArmorTracker\n\n");
    LibXR::STDIO::Printf("Usage\r\n");
    LibXR::STDIO::Printf("  show\r\n");
    LibXR::STDIO::Printf("  max_armor_distance <value>\r\n");
    LibXR::STDIO::Printf("  max_z_position <value>\r\n");
    LibXR::STDIO::Printf("  max_match_distance <value>\r\n");
    LibXR::STDIO::Printf("  max_match_yaw_diff <value>\r\n");
    LibXR::STDIO::Printf("  tracking_thres <value>\r\n");
    LibXR::STDIO::Printf("  bias_time <value>\r\n");
    LibXR::STDIO::Printf("  s_bias <value>\r\n");
    LibXR::STDIO::Printf("  z_bias <value>\r\n");
    LibXR::STDIO::Printf("  sigma2_q_xyz <value>\r\n");
    LibXR::STDIO::Printf("  sigma2_q_yaw <value>\r\n");
    LibXR::STDIO::Printf("  sigma2_q_r <value>\r\n");
    LibXR::STDIO::Printf("  r_xyz_factor <value>\r\n");
    LibXR::STDIO::Printf("  r_yaw <value>\r\n");
    return 0;
  }
  else if (argc == 2)
  {
    std::string cmd = argv[1];
    if (cmd == "show")
    {
      // clang-format off
      LibXR::STDIO::Printf("name: ArmorTracker\r\n");
      LibXR::STDIO::Printf("cfg:\r\n");
      LibXR::STDIO::Printf("  limits:\r\n");
      LibXR::STDIO::Printf("    max_armor_distance: %f\r\n", self->cfg_.limits.max_armor_distance);
      LibXR::STDIO::Printf("    max_z_position: %f\r\n", self->cfg_.limits.max_z_position);
      LibXR::STDIO::Printf("  match:\r\n");
      LibXR::STDIO::Printf("    max_match_distance: %f\r\n", self->cfg_.match.max_match_distance);
      LibXR::STDIO::Printf("    max_match_yaw_diff: %f\r\n", self->cfg_.match.max_match_yaw_diff);
      LibXR::STDIO::Printf("  thresholds:\r\n");
      LibXR::STDIO::Printf("    tracking_thres: %d\r\n", self->cfg_.thresholds.tracking_thres);
      LibXR::STDIO::Printf("    lost_time_thres: %f\r\n", self->cfg_.thresholds.lost_time_thres);
      LibXR::STDIO::Printf("  solver:\r\n");
      LibXR::STDIO::Printf("    k: %f\r\n", self->cfg_.solver.k);
      LibXR::STDIO::Printf("    bias_time: %d\r\n", self->cfg_.solver.bias_time);
      LibXR::STDIO::Printf("    s_bias: %f\r\n", self->cfg_.solver.s_bias);
      LibXR::STDIO::Printf("    z_bias: %f\r\n", self->cfg_.solver.z_bias);
      LibXR::STDIO::Printf("    calculate_mode: %d\r\n", static_cast<int>(self->cfg_.solver.calculate_mode));
      LibXR::STDIO::Printf("    table_config:\r\n");
      LibXR::STDIO::Printf("      max_x: %f\r\n", self->cfg_.solver.table_config.max_x);
      LibXR::STDIO::Printf("      min_x: %f\r\n", self->cfg_.solver.table_config.min_x);
      LibXR::STDIO::Printf("      max_y: %f\r\n", self->cfg_.solver.table_config.max_y);
      LibXR::STDIO::Printf("      min_y: %f\r\n", self->cfg_.solver.table_config.min_y);
      LibXR::STDIO::Printf("      resolution: %f\r\n", self->cfg_.solver.table_config.resolution);  
      LibXR::STDIO::Printf("      filename: %s\r\n", self->cfg_.solver.table_config.filename.c_str());
      LibXR::STDIO::Printf("  ekf:\r\n");
      LibXR::STDIO::Printf("    sigma2_q_xyz: %f\r\n", self->cfg_.ekf.sigma2_q_xyz);
      LibXR::STDIO::Printf("    sigma2_q_yaw: %f\r\n", self->cfg_.ekf.sigma2_q_yaw);
      LibXR::STDIO::Printf("    sigma2_q_r: %f\r\n", self->cfg_.ekf.sigma2_q_r);
      LibXR::STDIO::Printf("  noise:\r\n");
      LibXR::STDIO::Printf("    r_xyz_factor: %f\r\n", self->cfg_.noise.r_xyz_factor);
      LibXR::STDIO::Printf("    r_yaw: %f\r\n", self->cfg_.noise.r_yaw);
      LibXR::STDIO::Printf("  frames:\r\n");
      LibXR::STDIO::Printf("    rotation:\r\n");
      LibXR::STDIO::Printf("      - %f\r\n", self->cfg_.frames.base_transform_static.rotation(0));
      LibXR::STDIO::Printf("      - %f\r\n", self->cfg_.frames.base_transform_static.rotation(1));
      LibXR::STDIO::Printf("      - %f\r\n", self->cfg_.frames.base_transform_static.rotation(2));
      LibXR::STDIO::Printf("      - %f\r\n", self->cfg_.frames.base_transform_static.rotation(3));
      LibXR::STDIO::Printf("    translation:\r\n");
      LibXR::STDIO::Printf("      - %f\r\n", self->cfg_.frames.base_transform_static.translation(0));
      LibXR::STDIO::Printf("      - %f\r\n", self->cfg_.frames.base_transform_static.translation(1));
      LibXR::STDIO::Printf("      - %f\r\n", self->cfg_.frames.base_transform_static.translation(2));
      // clang-format on
    }
    return 0;
  }
  else if (argc == 3)
  {
    std::string cmd = argv[1];
    if (cmd == "max_armor_distance")
    {
      self->cfg_.limits.max_armor_distance = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_z_position")
    {
      self->cfg_.limits.max_z_position = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_match_distance")
    {
      self->cfg_.match.max_match_distance = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_match_yaw_diff")
    {
      self->cfg_.match.max_match_yaw_diff = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "tracking_thres")
    {
      self->cfg_.thresholds.tracking_thres = std::stoi(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "bias_time")
    {
      self->cfg_.solver.bias_time = std::stoi(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "s_bias")
    {
      self->cfg_.solver.s_bias = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "z_bias")
    {
      self->cfg_.solver.z_bias = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "sigma2_q_xyz")
    {
      self->cfg_.ekf.sigma2_q_xyz = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "sigma2_q_yaw")
    {
      self->cfg_.ekf.sigma2_q_yaw = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "sigma2_q_r")
    {
      self->cfg_.ekf.sigma2_q_r = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "r_xyz_factor")
    {
      self->cfg_.noise.r_xyz_factor = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "r_yaw")
    {
      self->cfg_.noise.r_yaw = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else
    {
      LibXR::STDIO::Printf("Unknown command: %s\n", argv[1]);
      return -1;
    }
    return 0;
  }
  LibXR::STDIO::Printf("Unknown command: %s\n", argv[1]);
  return -1;
}
