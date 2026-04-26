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

    geometry:
      initial_radius: 0.26
      min_radius: 0.12
      max_radius: 0.4

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
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Eigen>

// 框架与外部依赖头
#include "ArmorTrackerCommon.hpp"
#include "ArmorTrackerFaceSelector.hpp"
#include "ArmorTrackerImageTracker.hpp"
#include "ArmorTrackerObserver.hpp"
#include "ArmorTrackerRuntimeSupport.hpp"
#include "ArmorTrackerSelectionSupport.hpp"
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

namespace cv
{
class Mat;
}

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
  using ImageFrame = typename FrameSync::ImageFrame;
  using ImuStamped = typename FrameSync::ImuStamped;
  using SyncedFrame = typename FrameSync::SyncedFrame;
  using DetectionMessage = ArmorDetectionsFrameMessage<CameraInfoV>;

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

    struct Geometry
    {
      double initial_radius = 0.26;  // EKF 初始化半径先验
      double min_radius = 0.12;      // 几何状态下界
      double max_radius = 0.4;       // 几何状态上界
    } geometry;

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
    uint64_t image_timestamp_us{};
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
  void UpdateSingleArmorMode(const ArmorDetectorResults& armors_msg,
                             uint64_t image_timestamp_us);
  std::optional<ArmorDetectorResult> SelectSingleArmorObservation(
      const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us,
      std::size_t& selected_index, int& detection_track_id,
      bool& confirmed_track, float& selected_center_diff, float& selected_area_log,
      float& selected_score);
  void UpdateImageIdTracks(const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us);
  int FindDetectionTrackId(std::size_t armor_index) const;
  bool IsDetectionTrackConfirmed(std::size_t armor_index) const;
  void FillCandidateDebugFromSelection(
      const armor_tracker::FaceSelectionResult& selection,
      CandidateDebugMsg& candidate_debug);
  void FillCandidateDebugPolicy(
      CandidateDebugMsg& candidate_debug, const Eigen::VectorXd& ekf_prediction,
      const armor_tracker::FaceSelectionPolicy& face_policy) const;
  void WriteStateAuditRow(
      uint64_t image_timestamp_us, const Eigen::VectorXd& ekf_prediction,
      const armor_tracker::FaceSelectionResult* selection, bool matched);
  armor_tracker::FaceSelectionPolicy BuildFaceSelectionPolicy() const;
  armor_tracker::FaceSelectionTrackedState BuildFaceSelectionTrackedState() const;
  Eigen::Vector3d GetCameraWorldPosition();
  void AdvanceTrackerState(bool matched);
  bool ApplyFaceSelection(const armor_tracker::FaceSelectionResult& selection,
                          const ArmorDetectorResults& armors_msg,
                          CandidateDebugMsg& candidate_debug);
  bool TryRecoverTempLost(const ArmorDetectorResults& armors_msg,
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
      const ArmorDetectorResults& armors_msg, int observed_face_index,
      bool recenter_before_update);
  void FillSingleArmorDebug(std::size_t selected_index, int detection_track_id,
                            bool confirmed_track, float score, float center_diff,
                            float area_log);

  // ====================== IO 与回调（原 Node 逻辑） ======================
  void VelocityCallback(double velocity_msg);
  void ArmorsCallback(const DetectionMessage& message);

  // ====================== 辅助函数 ======================
  void InitEKF(const ArmorDetectorResult& a);
  void UpdateArmorsNum();
  void SyncDzReferenceFromState();
  void FuseMultiArmorObservation(const ArmorDetectorResults& armors_msg);
  void SwitchTrackedFace(int face_index, double measured_yaw);
  void RecenterTrackedStateToMeasurement(const ArmorDetectorResult& armor,
                                         int observed_face_index,
                                         double measured_yaw);
  double OrientationToYaw(const LibXR::Quaternion<double>& q);
  int LocalFaceToCanonicalFace(int local_face_index) const;
  void SyncGeometryRuntimeFromState();
  void ClampGeometryState();
  double GetArmorYawFromState(const Eigen::VectorXd& x, int face_index = 0) const;
  Eigen::Vector3d GetArmorPositionFromState(const Eigen::VectorXd& x,
                                            int face_index = 0) const;

  struct SpArmorMatch
  {
    int id = 0;
    double score = std::numeric_limits<double>::infinity();
    double yaw_error = 0.0;
    double pitch_error = 0.0;
    double distance_error = 0.0;
    double angle_error = 0.0;
    double xyz_error = 0.0;
    double measured_yaw = 0.0;
  };

  struct SpPairObservation
  {
    std::size_t armor_index = 0;
    ArmorDetectorResult armor{};
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  };

  struct SpPairMatch
  {
    bool valid = false;
    SpPairObservation left{};
    SpPairObservation right{};
    int left_face = 0;
    int right_face = 0;
    SpArmorMatch left_match{};
    SpArmorMatch right_match{};
    double score = std::numeric_limits<double>::infinity();
    double yaw = 0.0;
    double dz_observed = 0.0;
    double low_z = 0.0;
    double high_z = 0.0;
    bool left_is_high = false;
    int tracked_face = 0;
    std::size_t tracked_armor_index = 0;
    ArmorDetectorResult tracked_armor{};
    SpArmorMatch tracked_match{};
  };

  static double SpLimitRad(double angle);
  static double SpDetectorYawNear(const LibXR::Quaternion<double>& q,
                                  double reference_yaw);
  static Eigen::Vector3d SpXyzToYpd(const Eigen::Vector3d& xyz);
  static Eigen::MatrixXd SpXyzToYpdJacobian(const Eigen::Vector3d& xyz);
  static bool SpIsBalanceArmor(const ArmorDetectorResult& armor);
  static int SpArmorCountFor(const ArmorDetectorResult& armor);
  static double SpInitialRadiusFor(const ArmorDetectorResult& armor);
  static Eigen::VectorXd SpInitialP0DiagFor(const ArmorDetectorResult& armor);
  Eigen::Vector3d SpArmorPosition(const Eigen::VectorXd& state, int id) const;
  Eigen::MatrixXd SpObservationJacobian(const Eigen::VectorXd& state, int id) const;
  SpArmorMatch SpMatchArmorToFace(const ArmorDetectorResult& armor,
                                  const Eigen::VectorXd& state,
                                  int face_index) const;
  SpArmorMatch SpMatchArmor(const ArmorDetectorResult& armor,
                            const Eigen::VectorXd& state) const;
  bool SpTryCanonicalizeInitialState(const ArmorDetectorResults& armors_msg,
                                     bool force);
  bool SpResolvePairMatch(const ArmorDetectorResults& armors_msg,
                          const Eigen::VectorXd& state,
                          SpPairMatch& pair_match) const;
  void SpPredict();
  void SpUpdatePair(const SpPairMatch& pair_match);
  void SpUpdate(const ArmorDetectorResult& armor, const SpArmorMatch& match,
                bool freeze_delta_z);
  bool SpStateDiverged() const;
#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
  static void PreviewImageThreadFun(ArmorTracker<CameraInfoV>* self);
  static void RenderPreviewFrame(ArmorTracker<CameraInfoV>* self, cv::Mat frame);
#endif

  // ====================== 内部聚合成员（类内聚合） ======================
  struct EKFBlock
  {
    enum class MeasurementGeometryMode : std::uint8_t
    {
      FULL_BODY = 0,
      VISIBLE_FACE_ONLY = 1,
    };

    ExtendedKalmanFilter ekf;
    Eigen::VectorXd measurement = Eigen::VectorXd::Zero(4);  // z = [xa,ya,za,yaw]
    Eigen::VectorXd state =
        Eigen::VectorXd::Zero(11);  // x = [xc,vxc,yc,vyc,za,vza,yaw,vyaw,r1,dr,dz]
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(11, 11);
    int measurement_face_index = 0;
    MeasurementGeometryMode measurement_geometry_mode =
        MeasurementGeometryMode::FULL_BODY;
  } ekf_;

  armor_tracker::ImageTrackManager image_tracker_{};

  struct TrackRuntime
  {
    State state = State::LOST;
    int detect_count = 0;
    int lost_count = 0;
    int tracking_thres = 5;
    int lost_thres = 0;  // 帧数阈值（由时间阈值换算）
    uint8_t recovery_count = 0;
    double last_yaw = 0.0;
    double info_position_diff = 0.0;
    double info_yaw_diff = 0.0;
    double face_switch_cooldown_remaining = 0.0;
    int update_count = 0;
    int switch_count = 0;

    ArmorNumber tracked_id = ArmorNumber::INVALID;
    ArmorDetectorResult tracked_armor{};
    ArmorsNum tracked_armors_num = ArmorsNum::NORMAL_4;
    int tracked_face_index = 0;
    bool tracked_face_track_id_valid = false;
    uint16_t tracked_face_track_id = 0;
    std::array<bool, 4> face_track_id_valid{};
    std::array<uint16_t, 4> face_track_id{};
    bool sp_initial_phase_resolved = false;
    bool sp_pair_delta_z_valid = false;
    bool measurement_valid_current_frame = false;

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
    LibXR::Transform<double> gimbal_to_camera_transform_static{};
    LibXR::Transform<double> current_camera_pose{};
    bool current_camera_pose_valid = false;

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
#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
  LibXR::Thread preview_image_thread_{};
#endif

  EkfPointsMsg ekf_msg_;
  CandidateDebugMsg candidate_debug_msg_{};
  struct StateAuditBlock
  {
    std::string path{};
    std::ofstream file{};
    bool open_failed = false;
  } state_audit_;
  FrameSync& sync_;
};

using armor_tracker_detail::ArmorTrackerArmorsTopicName;
using armor_tracker_detail::ArmorTrackerCameraRotationToTrackerWorldPose;
using armor_tracker_detail::ArmorTrackerConvertToBgrWithEncoding;
using armor_tracker_detail::ArmorTrackerCvTypeFromEncoding;
using armor_tracker_detail::DirectionalFaceSwitchEnabled;
using armor_tracker_detail::FaceSwitchEnabled;
using armor_tracker_detail::FaceSwitchPositionDeadzone;
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
using armor_tracker_detail::MultiArmorFuseEnabled;
using armor_tracker_detail::OddFaceSwitchEnabled;
using armor_tracker_detail::RelaxedFaceSwitchEnabled;
using armor_tracker_detail::SingleArmorAreaLogGate;
using armor_tracker_detail::SingleArmorImageCenterGatePx;
using armor_tracker_detail::SingleArmorModeEnabled;
using armor_tracker_detail::SymmetricGeometryEnabled;
using armor_tracker_detail::SpDeltaZInitialVariance;
using armor_tracker_detail::SpDeltaZProcessVariance;
using armor_tracker_detail::SpDirectDeltaZAlpha;
using armor_tracker_detail::SpDirectDeltaZEnabled;
using armor_tracker_detail::SpDirectDeltaZMaxAbs;
using armor_tracker_detail::SpCanonicalInitEnabled;
using armor_tracker_detail::SpCanonicalInitMaxAbsDz;
using armor_tracker_detail::SpCanonicalInitMaxUpdates;
using armor_tracker_detail::SpCanonicalInitMaxScore;
using armor_tracker_detail::SpCanonicalInitMinHeight;
using armor_tracker_detail::SpCanonicalInitPreferPositiveDz;
using armor_tracker_detail::SpPitchVarianceScale;
using armor_tracker_detail::SpPairDeltaZAlpha;
using armor_tracker_detail::SpPairDeltaZEnabled;
using armor_tracker_detail::SpPairDeltaZMaxAbs;
using armor_tracker_detail::SpPairDeltaZMinHeight;
using armor_tracker_detail::SpPairDeltaZVariance;
using armor_tracker_detail::SpMeasurementAnchoredOutputEnabled;
using armor_tracker_detail::SpStaticDeltaZ;
using armor_tracker_detail::SpStaticDeltaZEnabled;
using armor_tracker_detail::TempLostRecoveryEnabled;
using armor_tracker_detail::ViewPriorityEnabled;
using armor_tracker_detail::kArmorTrackerSyncFrameWaitTimeoutMs;
using armor_tracker::AngularDiffAbs;
using armor_tracker::LogImpossibleYawDiff;
using armor_tracker::OrientationToYawNear;
using armor_tracker::QuaternionToYaw;
using armor_tracker::TimestampAbsDiff;
using armor_tracker::UnwrapYawNear;

// 分离出的实现块：让主头文件只保留 tracker 主流程。
#include "ArmorTrackerDebugSupport.hpp"
#include "ArmorTrackerSpCore.hpp"
#include "ArmorTrackerObserverRuntimeSupport.hpp"
#include "ArmorTrackerPipeline.hpp"
#include "ArmorTrackerSingleArmorSupport.hpp"
#include "ArmorTrackerStateAuditSupport.hpp"

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
  // 状态 x = [xc, vxc, yc, vyc, za, vza, yaw0, vyaw, r1, dr, dz]
  // 其中 yaw0/r1 对应 canonical face 0，face 1/3 使用 r1 + dr 与 za + dz。
  // 观测 z = [xa, ya, za, yaw(face_i)]，由 measurement_face_index 指定当前观测面。
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
    Eigen::MatrixXd f = Eigen::MatrixXd::Identity(11, 11);
    double d = time_.dt;
    f(ExtendedKalmanFilter::X_CENTER, ExtendedKalmanFilter::V_X_CENTER) = d;
    f(ExtendedKalmanFilter::Y_CENTER, ExtendedKalmanFilter::V_Y_CENTER) = d;
    f(ExtendedKalmanFilter::Z_ARMOR, ExtendedKalmanFilter::V_Z_ARMOR) = d;
    f(ExtendedKalmanFilter::YAW, ExtendedKalmanFilter::V_YAW) = d;
    return f;
  };
  auto h = [this](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd z(4);
    const int face_index = std::max(0, ekf_.measurement_face_index);
    const Eigen::Vector3d armor_position =
        GetArmorPositionFromState(x, face_index);
    z(0) = armor_position.x();
    z(1) = armor_position.y();
    z(2) = armor_position.z();
    z(3) = GetArmorYawFromState(x, face_index);
    return z;
  };
  auto j_h = [this](const Eigen::VectorXd& x)
  {
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(4, 11);
    const auto runtime = BuildObserverRuntime();
    const auto policy = BuildObserverPolicy();
    const int face_index = std::max(0, ekf_.measurement_face_index);
    const double yaw =
        armor_tracker::GetArmorYawFromState(x, runtime, face_index);
    const bool odd_face =
        !policy.symmetric_geometry_enabled && runtime.tracked_armors_num == 4 &&
        (face_index % 2 == 1);
    const bool visible_face_only_geometry =
        ekf_.measurement_geometry_mode ==
        EKFBlock::MeasurementGeometryMode::VISIBLE_FACE_ONLY;
    const double radius =
        odd_face ? (x(ExtendedKalmanFilter::ROBOT_R) + x(ExtendedKalmanFilter::DELTA_R))
                 : x(ExtendedKalmanFilter::ROBOT_R);

    h(0, ExtendedKalmanFilter::X_CENTER) = 1.0;
    h(0, ExtendedKalmanFilter::YAW) = radius * std::sin(yaw);
    // 单面连续跟踪时，只让观测收敛当前可见装甲板。
    // 否则 x/y 残差会被错误吸收到整车半径里，逐帧把 r1/r2 推到 clamp。
    if (!visible_face_only_geometry)
    {
      h(0, ExtendedKalmanFilter::ROBOT_R) = -std::cos(yaw);
    }

    h(1, ExtendedKalmanFilter::Y_CENTER) = 1.0;
    h(1, ExtendedKalmanFilter::YAW) = -radius * std::cos(yaw);
    if (!visible_face_only_geometry)
    {
      h(1, ExtendedKalmanFilter::ROBOT_R) = -std::sin(yaw);
    }

    h(2, ExtendedKalmanFilter::Z_ARMOR) = 1.0;
    h(3, ExtendedKalmanFilter::YAW) = 1.0;

    if (odd_face)
    {
      if (!visible_face_only_geometry)
      {
        h(0, ExtendedKalmanFilter::DELTA_R) = -std::cos(yaw);
        h(1, ExtendedKalmanFilter::DELTA_R) = -std::sin(yaw);
      }
      h(2, ExtendedKalmanFilter::DELTA_Z) = 1.0;
    }
    return h;
  };
  auto u_q = [this]()
  {
    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(11, 11);
    double t = time_.dt, x = cfg_.ekf.sigma2_q_xyz, y = cfg_.ekf.sigma2_q_yaw,
           r = cfg_.ekf.sigma2_q_r;
    double q_x_x = std::pow(t, 4) / 4 * x, q_x_vx = std::pow(t, 3) / 2 * x,
           q_vx_vx = std::pow(t, 2) * x;
    double q_y_y = std::pow(t, 4) / 4 * y, q_y_vy = std::pow(t, 3) / 2 * y,
           q_vy_vy = std::pow(t, 2) * y;
    double q_r = std::pow(t, 4) / 4 * r;
    const double min_radius =
        std::min(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    const double max_radius =
        std::max(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    const double geometry_span = std::max(0.01, max_radius - min_radius);
    const double q_delta_r = std::max(q_r, std::pow(geometry_span * 0.05, 2));
    const double q_delta_z = std::max(q_x_x, std::pow(geometry_span * 0.06, 2));
    q(0,0)=q_x_x;  q(0,1)=q_x_vx; q(1,0)=q_x_vx; q(1,1)=q_vx_vx;
    q(2,2)=q_x_x;  q(2,3)=q_x_vx; q(3,2)=q_x_vx; q(3,3)=q_vx_vx;
    q(4,4)=q_x_x;  q(4,5)=q_x_vx; q(5,4)=q_x_vx; q(5,5)=q_vx_vx;
    q(6,6)=q_y_y;  q(6,7)=q_y_vy; q(7,6)=q_y_vy; q(7,7)=q_vy_vy;
    q(8,8)=std::max(q_r, q_delta_r);
    q(9,9)=q_delta_r;
    q(10,10)=q_delta_z;
    return q;
  };
  auto u_r = [this](const Eigen::VectorXd& z)
  {
    Eigen::DiagonalMatrix<double, 4> r;
    const double range =
        std::max(1e-6, Eigen::Vector3d(z[0], z[1], z[2]).norm());
    const double position_sigma =
        std::max(0.01, cfg_.noise.r_xyz_factor * range);
    const double position_variance = position_sigma * position_sigma;
    r.diagonal() << position_variance, position_variance, position_variance,
        cfg_.noise.r_yaw;
    return r;
  };
  Eigen::DiagonalMatrix<double, 11> p0;
  p0.setIdentity();
  {
    const double min_radius =
        std::min(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    const double max_radius =
        std::max(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    const double geometry_span = std::max(0.01, max_radius - min_radius);
    const double delta_r_prior_sigma = std::max(0.03, geometry_span * 0.15);
    const double delta_z_prior_sigma = std::max(0.03, geometry_span * 0.20);
    p0.diagonal()(ExtendedKalmanFilter::DELTA_R) =
        delta_r_prior_sigma * delta_r_prior_sigma;
    p0.diagonal()(ExtendedKalmanFilter::DELTA_Z) =
        delta_z_prior_sigma * delta_z_prior_sigma;
  }
  ekf_.ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};

  // ---------------- Topics & 回调 ----------------
  // 装甲板识别结果订阅
  LibXR::Topic::Domain armor_detector_domain = LibXR::Topic::Domain("armor_detector");
  LibXR::Topic armors_topic = LibXR::Topic(
      LibXR::Topic::WaitTopic(ArmorTrackerArmorsTopicName(), UINT32_MAX, &armor_detector_domain));
  auto armors_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto* armors_msg = reinterpret_cast<DetectionMessage*>(data.addr_);
        if (self->params_is_changed_ == true)
        {
          self->SetConfig(self->cfg_);
          self->params_is_changed_ = false;
        }
        self->ArmorsCallback(*armors_msg);
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

  io_.solver->SetFireCallback(
      [&](bool is_fire)
      {
        XR_LOG_INFO("is_fire: {}", is_fire);
        // uint8_t fire_notify = is_fire ? 1 : 0;
        uint8_t fire_notify = 0;
        io_.fire_notify_topic.Publish(fire_notify);
      });

  if (const char* audit_env = std::getenv("XR_TRACKER_STATE_AUDIT_PATH"))
  {
    if (audit_env[0] != '\0')
    {
      state_audit_.path = audit_env;
    }
  }

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
  preview_image_thread_.Create(this, PreviewImageThreadFun, "TrackPreviewImg",
                               static_cast<size_t>(1024 * 128),
                               LibXR::Thread::Priority::LOW);
#endif
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::OnMonitor() {}

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
void ArmorTracker<CameraInfoV>::FillSingleArmorDebug(std::size_t selected_index,
                                                     int detection_track_id,
                                                     bool confirmed_track,
                                                     float score,
                                                     float center_diff,
                                                     float area_log)
{
  CandidateDebugMsg debug{};
  candidate_debug_msg_.matched =
      (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST) ? 1 : 0;
  debug.matched =
      (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST) ? 1 : 0;
  debug.count = 1;
  debug.selected_index = 0;
  debug.detection_count = 1;
  debug.tracked_armors_num = 1;
  debug.tracked_face_track_id_valid = confirmed_track ? 1 : 0;
  debug.tracked_face_track_id =
      confirmed_track && detection_track_id >= 0
          ? static_cast<int16_t>(detection_track_id)
          : static_cast<int16_t>(-1);
  debug.best_same_face_score = score;
  debug.same_face_matched = debug.matched;
  debug.switch_face_matched = 0;
  debug.switch_allowed = 0;
  debug.detection_track_ids.fill(static_cast<int16_t>(-1));
  debug.detection_track_confirmed.fill(static_cast<uint8_t>(0));
  if (selected_index < CandidateDebugMsg::kMaxDetections)
  {
    debug.detection_track_ids[selected_index] =
        static_cast<int16_t>(detection_track_id);
    debug.detection_track_confirmed[selected_index] =
        confirmed_track ? 1 : 0;
  }

  auto& item = debug.items[0];
  item.armor_index = static_cast<uint8_t>(selected_index);
  item.face_index = 0;
  item.same_number = 1;
  item.image_track_id = static_cast<int16_t>(detection_track_id);
  item.image_track_confirmed = confirmed_track ? 1 : 0;
  item.same_persistent_track = confirmed_track ? 1 : 0;
  item.number = rt_.tracked_armor.number;
  item.type = rt_.tracked_armor.type;
  item.score = score;
  item.position_diff = 0.0f;
  item.yaw_diff = 0.0f;
  item.view_bonus = 0.0f;
  item.area_score = static_cast<float>(armor_tracker::ArmorImageArea(rt_.tracked_armor));
  item.frontality = 0.0f;
  item.center_x = rt_.tracked_armor.center.x;
  item.center_y = rt_.tracked_armor.center.y;
  item.predicted_yaw = static_cast<float>(rt_.last_yaw);
  item.measured_yaw = static_cast<float>(rt_.last_yaw);
  debug.relaxed_same_face_distance = center_diff;
  debug.relaxed_face_switch_distance = area_log;
  candidate_debug_msg_ = debug;
}

template <CameraTypes::CameraInfo CameraInfoV>
std::optional<ArmorDetectorResult> ArmorTracker<CameraInfoV>::SelectSingleArmorObservation(
    const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us,
    std::size_t& selected_index, int& detection_track_id,
    bool& confirmed_track, float& selected_center_diff, float& selected_area_log,
    float& selected_score)
{
  (void)image_timestamp_us;
  selected_index = 0;
  detection_track_id = -1;
  confirmed_track = false;
  selected_center_diff = 0.0f;
  selected_area_log = 0.0f;
  selected_score = 0.0f;
  struct Candidate
  {
    std::size_t armor_index = 0;
    int detection_track_id = -1;
    bool confirmed_track = false;
    float score = 0.0f;
    float center_diff = 0.0f;
    float area_log = 0.0f;
    ArmorDetectorResult armor{};
  };

  bool has_candidate = false;
  Candidate best{};
  const double tracked_area =
      std::max(1.0, armor_tracker::ArmorImageArea(rt_.tracked_armor));
  const bool have_tracked_armor = rt_.tracked_id != ArmorNumber::INVALID;
  const bool have_tracked_pose =
      have_tracked_armor && (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST);
  const double prev_range =
      have_tracked_pose
          ? std::sqrt(std::pow(rt_.tracked_armor.pose.translation.x(), 2.0) +
                      std::pow(rt_.tracked_armor.pose.translation.y(), 2.0) +
                      std::pow(rt_.tracked_armor.pose.translation.z(), 2.0))
          : 0.0;
  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    const int detection_track_id = FindDetectionTrackId(armor_index);
    const bool confirmed_track = IsDetectionTrackConfirmed(armor_index);

    float score = 0.0f;
    const bool same_confirmed_track =
        rt_.tracked_face_track_id_valid && confirmed_track &&
        detection_track_id >= 0 &&
        static_cast<uint16_t>(detection_track_id) == rt_.tracked_face_track_id;
    if (same_confirmed_track)
    {
      score -= 8.0f;
    }
    else if (have_tracked_armor && armor.number == rt_.tracked_id &&
             armor.number != ArmorNumber::INVALID)
    {
      score -= 4.0f;
    }

    if (armor.type == rt_.tracked_armor.type &&
        armor.type != ArmorType::INVALID)
    {
      score -= 1.0f;
    }

    const float center_diff = have_tracked_armor
                                  ? static_cast<float>(std::hypot(
                                        armor.center.x - rt_.tracked_armor.center.x,
                                        armor.center.y - rt_.tracked_armor.center.y))
                                  : static_cast<float>(armor.distance_to_image_center);
    const float area_log = have_tracked_armor
                               ? static_cast<float>(std::abs(std::log(
                                     std::max(1.0, armor_tracker::ArmorImageArea(armor)) /
                                     tracked_area)))
                               : 0.0f;
    const double range =
        std::sqrt(std::pow(armor.pose.translation.x(), 2.0) +
                  std::pow(armor.pose.translation.y(), 2.0) +
                  std::pow(armor.pose.translation.z(), 2.0));
    const float range_diff =
        have_tracked_pose ? static_cast<float>(std::abs(range - prev_range)) : 0.0f;

    score += 0.012f * center_diff;
    score += 0.35f * area_log;
    score += 1.40f * range_diff;
    score -= 0.45f * armor.confidence;
    score += 0.0015f *
             static_cast<float>(armor.distance_to_image_center);

    if (rt_.state == State::TEMP_LOST)
    {
      const bool same_label =
          have_tracked_armor && armor.number == rt_.tracked_id &&
          armor.type == rt_.tracked_armor.type &&
          armor.number != ArmorNumber::INVALID;
      if (!(same_confirmed_track ||
            (same_label && center_diff <= 120.0f && area_log <= 0.50f)))
      {
        continue;
      }
    }
    if (have_tracked_armor && same_confirmed_track &&
        (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST))
    {
      // 同一条已确认的 image-track 在 Webots 中会出现连续的小像素位移配合较大的 PnP 深度抖动。
      // 这里不能再用“绝对 3D jump”直接拒绝，否则会把真实连续观测整段打掉。
      const double yaw_now =
          armor_tracker::OrientationToYawNear(armor.pose.rotation, rt_.last_yaw);
      const double yaw_jump =
          armor_tracker::AngularDiffAbs(yaw_now, rt_.last_yaw);
      const bool suspicious_pose_jump =
          center_diff < 6.0f &&
          yaw_jump > 1.20 &&
          std::abs(armor.pose.translation.z() - rt_.tracked_armor.pose.translation.z()) > 0.12;
      if (suspicious_pose_jump)
      {
        XR_LOG_DEBUG(
            "SingleArmor reject pose jump: idx=%zu track=%d yaw_jump=%.3f center_diff=%.1f area_log=%.3f prev=(%.3f,%.3f,%.3f) now=(%.3f,%.3f,%.3f)",
            armor_index, detection_track_id, yaw_jump,
            static_cast<double>(center_diff), static_cast<double>(area_log), rt_.tracked_armor.pose.translation.x(),
            rt_.tracked_armor.pose.translation.y(), rt_.tracked_armor.pose.translation.z(),
            armor.pose.translation.x(), armor.pose.translation.y(),
            armor.pose.translation.z());
        continue;
      }
    }

    if (!has_candidate || score < best.score)
    {
      has_candidate = true;
      best.armor_index = armor_index;
      best.detection_track_id = detection_track_id;
      best.confirmed_track = confirmed_track;
      best.score = score;
      best.center_diff = center_diff;
      best.area_log = area_log;
      best.armor = armor;
    }
  }
  if (!has_candidate)
  {
    return std::nullopt;
  }

  selected_index = best.armor_index;
  detection_track_id = best.detection_track_id;
  confirmed_track = best.confirmed_track;
  selected_center_diff = best.center_diff;
  selected_area_log = best.area_log;
  selected_score = best.score;
  return best.armor;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::InitEKF(const ArmorDetectorResult& a)
{
  const Eigen::Vector3d xyz(a.pose.translation.x(), a.pose.translation.y(),
                            a.pose.translation.z());
  rt_.last_yaw = 0.0;
  const double yaw = SpDetectorYawNear(a.pose.rotation, 0.0);
  const double radius = SpInitialRadiusFor(a);
  const double center_x = xyz.x() - radius * std::cos(yaw);
  const double center_y = xyz.y() - radius * std::sin(yaw);
  const double center_z = xyz.z();

  ekf_.state = Eigen::VectorXd::Zero(11);
  ekf_.state << center_x, 0.0, center_y, 0.0, center_z, 0.0, yaw, 0.0,
      radius, 0.0, 0.0;
  if (SpArmorCountFor(a) == 4 && SpStaticDeltaZEnabled())
  {
    ekf_.state(ExtendedKalmanFilter::DELTA_Z) = SpStaticDeltaZ();
  }
  ekf_.covariance = SpInitialP0DiagFor(a).asDiagonal();
  ekf_.measurement_face_index = 0;
  ekf_.measurement =
      Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);
  rt_.last_yaw = yaw;
  rt_.tracked_armors_num = static_cast<ArmorsNum>(SpArmorCountFor(a));
  rt_.tracked_face_index = 0;
  rt_.another_r = radius;
  rt_.dz = ekf_.state(ExtendedKalmanFilter::DELTA_Z);
  rt_.dz_abs_ref = std::abs(rt_.dz);
  rt_.face_switch_cooldown_remaining = 0.0;
  rt_.sp_pair_delta_z_valid = false;
  ekf_.ekf.SetState(ekf_.state);
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
      LibXR::STDIO::Printf("  geometry:\r\n");
      LibXR::STDIO::Printf("    initial_radius: %f\r\n", self->cfg_.geometry.initial_radius);
      LibXR::STDIO::Printf("    min_radius: %f\r\n", self->cfg_.geometry.min_radius);
      LibXR::STDIO::Printf("    max_radius: %f\r\n", self->cfg_.geometry.max_radius);
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
