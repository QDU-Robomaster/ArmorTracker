#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: Armor tracker
constructor_args:
  cfg:
    limits:
      max_armor_distance: 10.0

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
#include <memory>
#include <utility>
#include <vector>

// 框架与外部依赖头
#include "SolveTrajectory.hpp"
#include "app_framework.hpp"
#include "armor.hpp"
#include "cycle_value.hpp"
#include "extended_kalman_filter.hpp"
#include "libxr.hpp"
#include "libxr_time.hpp"
#include "logger.hpp"
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
  enum class ArmorsNum
  {
    NORMAL_4 = 4,
    OUTPOST_3 = 3
  };

  enum State
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

 public:
  // ====================== 构造与监控 ======================
  explicit ArmorTracker(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app,
                        Config cfg);

  void OnMonitor() override;

 private:
  // ====================== 内部算法接口（原 Tracker 逻辑） ======================
  void Init(const ArmorDetectorResults& armors_msg);
  void Update(const ArmorDetectorResults& armors_msg);

  // ====================== IO 与回调（原 Node 逻辑） ======================
  void VelocityCallback(double velocity_msg);
  void ArmorsCallback(ArmorDetectorResults& armors_msg);

  // ====================== 辅助函数 ======================
  void InitEKF(const ArmorDetectorResult& a);
  void UpdateArmorsNum(const ArmorDetectorResult&);
  void HandleArmorJump(const ArmorDetectorResult& current_armor);
  double OrientationToYaw(const LibXR::Quaternion<double>& q);
  Eigen::Vector3d GetArmorPositionFromState(const Eigen::VectorXd& x);

  // ====================== 内部聚合成员（类内聚合） ======================
  struct EKFBlock
  {
    ExtendedKalmanFilter ekf;
    Eigen::VectorXd measurement = Eigen::VectorXd::Zero(4);  // z = [xa,ya,za,yaw]
    Eigen::VectorXd state =
        Eigen::VectorXd::Zero(9);  // x = [xc,vxc,yc,vyc,za,vza,yaw,vyaw,r]
  } ekf_;

  struct TrackRuntime
  {
    State state = LOST;
    int detect_count = 0;
    int lost_count = 0;
    int tracking_thres = 5;
    int lost_thres = 0;  // 帧数阈值（由时间阈值换算）
    double last_yaw = 0.0;
    double info_position_diff = 0.0;
    double info_yaw_diff = 0.0;

    ArmorNumber tracked_id = ArmorNumber::INVALID;
    ArmorDetectorResult tracked_armor{};
    ArmorsNum tracked_armors_num = ArmorsNum::NORMAL_4;

    // 另一片装甲板信息
    double dz = 0.0;
    double another_r = 0.0;
  } rt_;

  struct TimeBlock
  {
    LibXR::MicrosecondTimestamp last_time = LibXR::Timebase::GetMicroseconds();
    double dt = 1.0 / 100.0;  // 初始假定 100Hz
  } time_;

  struct IOBlock
  {
    // 坐标变换
    LibXR::Transform<double> base_transform_static{};
    LibXR::Quaternion<double> base_rotation{};
    LibXR::Mutex base_rotation_lock;
    LibXR::Topic tf_topic = LibXR::Topic("/tf", sizeof(LibXR::Quaternion<double>));

    // 发布者
    LibXR::Topic info_topic = LibXR::Topic("/tracker/info", sizeof(TrackerInfo));
    LibXR::Topic target_topic =
        LibXR::Topic("/tracker/target", sizeof(SolveTrajectory::Target));
    LibXR::Topic send_topic = LibXR::Topic("/tracker/send", sizeof(Send));

    // 轨迹解算
    std::unique_ptr<SolveTrajectory> solver;
  } io_;

  // 保存配置（类内聚合）
  Config cfg_;
};
