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
template_args: []
required_hardware: []
depends:
  - qdu-future/ArmorDetector
=== END MANIFEST === */
// clang-format on

#include <Eigen/Eigen>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <opencv2/core/types.hpp>

// 框架与外部依赖头
#include "CameraBase.hpp"
#include "SolveTrajectory.hpp"
#include "app_framework.hpp"
#include "armor.hpp"
#include "extended_kalman_filter.hpp"
#include "libxr_time.hpp"
#include "message.hpp"
#include "mutex.hpp"
#include "timebase.hpp"
#include "transform.hpp"

class ArmorTracker : public LibXR::Application
{
 public:
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

  struct Send
  {
    bool is_fire{};
    LibXR::Position<double> position{};
    double v_yaw{};
    double pitch{};
    double yaw{};
    Eigen::Matrix<double, 3, 1> cmd_vel_linear = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> cmd_vel_angular = Eigen::Matrix<double, 3, 1>::Zero();
  };

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

 public:
  // ====================== 构造与监控 ======================
  explicit ArmorTracker(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app,
                        Config cfg);
                        
  static int CommandFun(ArmorTracker* self, int argc, char** argv);
  const Config& GetConfig() const { return cfg_; }
  void SetConfig(const Config& cfg);
  static int CommandAdapter(void* instance, int argc, char** argv)
  {
    return CommandFun(static_cast<ArmorTracker*>(instance), argc, argv);
  }

  void OnMonitor() override;

 private:
  struct ImageIdTrack;

  // ====================== 内部算法接口（原 Tracker 逻辑）
  // ======================
  void Init(const ArmorDetectorResults& armors_msg);
  void Update(const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us);
  void UpdateImageIdTracks(const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us);
  int FindDetectionTrackId(std::size_t armor_index) const;
  bool IsDetectionTrackConfirmed(std::size_t armor_index) const;
  void PushCameraPose(const CameraBase::PoseStamped& pose_msg);
  bool LookupCameraPose(uint64_t image_timestamp_us, LibXR::Transform<double>& pose_out);

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
  static bool CompatibleImageTrackLabel(const ImageIdTrack& track,
                                        const ArmorDetectorResult& armor);

  // ====================== 内部聚合成员（类内聚合） ======================
  struct EKFBlock
  {
    ExtendedKalmanFilter ekf;
    Eigen::VectorXd measurement = Eigen::VectorXd::Zero(4);  // z = [xa,ya,za,yaw]
    Eigen::VectorXd state =
        Eigen::VectorXd::Zero(9);  // x = [xc,vxc,yc,vyc,za,vza,yaw,vyaw,r]
  } ekf_;

  struct ImageIdTrack
  {
    bool active = false;
    bool confirmed = false;
    uint16_t track_id = 0;
    ArmorColor color = ArmorColor::UNKNOWN;
    ArmorNumber number = ArmorNumber::INVALID;
    ArmorType type = ArmorType::INVALID;
    float confidence = 0.0f;
    cv::Point2f image_center{};
    cv::Point2f image_velocity{};
    double area = 0.0;
    double area_rate = 0.0;
    uint64_t first_timestamp_us = 0;
    uint64_t last_timestamp_us = 0;
    uint64_t last_seen_timestamp_us = 0;
    uint32_t age = 0;
    uint32_t hit_count = 0;
    uint32_t miss_count = 0;
    bool matched_this_frame = false;
    uint8_t matched_armor_index = 255;
  };

  struct IdTrackerRuntime
  {
    static constexpr std::size_t kMaxTracks = 8;
    std::array<ImageIdTrack, kMaxTracks> tracks{};
    std::vector<int> detection_track_ids{};
    std::vector<uint8_t> detection_track_confirmed{};
    uint16_t next_track_id = 0;
  } id_tracker_;

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
    std::array<CameraBase::PoseStamped, kCameraPoseHistorySize> camera_pose_history{};
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

  EkfPointsMsg ekf_msg_;
  CandidateDebugMsg candidate_debug_msg_{};
  std::shared_ptr<CameraBase::CameraInfo> cam_info_{};  ///< 相机内参/畸变
};
