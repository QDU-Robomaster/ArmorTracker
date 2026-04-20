#include "ArmorTracker.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

#include "cycle_value.hpp"
#include "logger.hpp"
#include "message.hpp"
#include "transform.hpp"

namespace
{
double UnwrapYawNear(double yaw, double reference_yaw)
{
  const double delta =
      LibXR::CycleValue<double>(yaw) - LibXR::CycleValue<double>(reference_yaw);
  return reference_yaw + delta;
}

double QuaternionToYaw(const LibXR::Quaternion<double>& q)
{
  LibXR::EulerAngle<double> eulr =
      LibXR::RotationMatrix<double>(q.ToRotationMatrix()).ToEulerAngle();
  return eulr.Yaw();
}

double OrientationToYawNear(const LibXR::Quaternion<double>& q, double reference_yaw)
{
  return UnwrapYawNear(QuaternionToYaw(q), reference_yaw);
}

double AngularDiffAbs(double lhs, double rhs)
{
  return std::abs(LibXR::CycleValue<double>(lhs) - LibXR::CycleValue<double>(rhs));
}

void LogImpossibleYawDiff(const char* tag, std::size_t armor_index, int face_index,
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


LibXR::Quaternion<double> PoseStampedToQuaternion(const PoseStamped& pose_msg)
{
  return LibXR::Quaternion<double>(pose_msg.rotation.w(), pose_msg.rotation.x(),
                                   pose_msg.rotation.y(), pose_msg.rotation.z());
}

LibXR::Transform<double> CameraRotationToTrackerWorldPose(
    const LibXR::Quaternion<double>& camera_rotation,
    const LibXR::Transform<double>& gimbal_to_camera_transform_static)
{
  return LibXR::Transform<double>(camera_rotation, {0.0, 0.0, 0.0}) +
         gimbal_to_camera_transform_static;
}

uint64_t TimestampAbsDiff(uint64_t lhs, uint64_t rhs)
{
  return lhs >= rhs ? (lhs - rhs) : (rhs - lhs);
}
double FaceSwitchPenalty(int face_index)
{
  if (face_index == 0)
  {
    return 0.0;
  }
  return face_index == 2 ? 0.45 : 0.20;
}

bool SingleArmorModeEnabled()
{
  const char* env = std::getenv("XR_TRACKER_SINGLE_ARMOR_MODE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool MultiArmorFuseEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_MULTI_FUSE");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

bool SymmetricGeometryEnabled()
{
  const char* env = std::getenv("XR_TRACKER_FORCE_SYMMETRIC_GEOMETRY");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool FaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

bool RelaxedFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_RELAXED_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

bool FaceSwitchRecenterEnabled()
{
  const char* env = std::getenv("XR_TRACKER_DISABLE_FACE_SWITCH_RECENTER");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

bool OddFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_ODD_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

bool ViewPriorityEnabled()
{
  const char* env = std::getenv("XR_TRACKER_ENABLE_VIEW_PRIORITY");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

const char* ArmorsTopicName()
{
  const char* env = std::getenv("XR_ARMORS_TOPIC_NAME");
  return (env != nullptr && env[0] != '\0') ? env : "armors_result";
}


bool DirectionalFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_ENABLE_DIRECTIONAL_FACE_SWITCH");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

double ParseEnvDouble(const char* name, double default_value)
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

std::uint32_t ParseEnvUint(const char* name, std::uint32_t default_value)
{
  return static_cast<std::uint32_t>(std::max(
      1.0, std::round(ParseEnvDouble(name, static_cast<double>(default_value)))));
}

double SingleArmorImageCenterGatePx()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SINGLE_ARMOR_IMAGE_GATE_PX", 180.0));
}

double SingleArmorAreaLogGate()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SINGLE_ARMOR_AREA_LOG_GATE", 0.80));
}

double FaceSwitchScoreDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_SCORE_DEADZONE", 0.15));
}

double FaceSwitchPositionDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_POSITION_DEADZONE", 0.05));
}

double FaceSwitchYawDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_YAW_DEADZONE", 0.35));
}

double FaceSwitchTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_TIMEOUT_SEC", 0.08));
}

bool IdAssistEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_IMAGE_ID_ASSIST");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

double IdAssistSameFaceCenterGatePx()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_ASSIST_CENTER_GATE_PX", 85.0));
}

double IdAssistSameFaceAreaLogGate()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_ASSIST_AREA_LOG_GATE", 0.45));
}

std::uint32_t IdTrackAppearHits()
{
  return ParseEnvUint("XR_TRACKER_ID_APPEAR_HITS", 2U);
}

double IdTrackAppearTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_APPEAR_TIMEOUT_SEC", 0.01));
}

std::uint32_t IdTrackTentativeMisses()
{
  return ParseEnvUint("XR_TRACKER_ID_TENTATIVE_MISSES", 2U);
}

double IdTrackTentativeTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_TENTATIVE_TIMEOUT_SEC", 0.03));
}

std::uint32_t IdTrackDisappearMisses()
{
  return ParseEnvUint("XR_TRACKER_ID_DISAPPEAR_MISSES", 3U);
}

double IdTrackDisappearTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_DISAPPEAR_TIMEOUT_SEC", 0.06));
}

struct ArmorMatchCandidate
{
  ArmorDetectorResult armor{};
  std::size_t armor_index = 0;
  uint8_t debug_index = ArmorTracker::CandidateDebugMsg::kMaxItems;
  int face_index = -1;
  bool same_number = false;
  int image_track_id = -1;
  bool confirmed_image_track = false;
  bool same_persistent_track = false;
  double measured_yaw = 0.0;
  double position_diff = DBL_MAX;
  double yaw_diff = DBL_MAX;
  double view_bonus = 0.0;
  double area_score = 0.0;
  double frontality = 0.0;
  double image_center_diff = DBL_MAX;
  double area_ratio_log = DBL_MAX;
  double score = DBL_MAX;
};

bool IsBetterMatchCandidate(const ArmorMatchCandidate& candidate,
                            const ArmorMatchCandidate& best)
{
  if (candidate.score < best.score - 1e-6)
  {
    return true;
  }
  if (std::abs(candidate.score - best.score) < 0.10 &&
      candidate.position_diff < best.position_diff - 1e-6)
  {
    return true;
  }
  if (std::abs(candidate.position_diff - best.position_diff) < 0.02 &&
      candidate.yaw_diff < best.yaw_diff)
  {
    return true;
  }
  return false;
}
}  // namespace

ArmorTracker::ArmorTracker(LibXR::HardwareContainer& hw, LibXR::ApplicationManager&,
                           Config cfg, CameraBase::CameraInfo camera_info)
    : cfg_(std::move(cfg)),
      solver_cfg_(cfg_.solver),
      cmd_file_(LibXR::RamFS::CreateFile(name_, CommandFun, this)),
      cam_info_(std::move(camera_info))
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
      LibXR::Topic::WaitTopic(ArmorsTopicName(), UINT32_MAX, &armor_detector_domain));
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

  auto camera_pose_topic = LibXR::Topic(LibXR::Topic::Find("camera_pose"));
  auto camera_pose_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto* pose_msg = reinterpret_cast<PoseStamped*>(data.addr_);
        if (pose_msg != nullptr)
        {
          self->PushCameraPose(*pose_msg);
        }
      },
      this);
  camera_pose_topic.RegisterCallback(camera_pose_cb);

  io_.solver->SetFireCallback(
      [&](bool is_fire)
      {
        XR_LOG_INFO("is_fire: {}", is_fire);
        // uint8_t fire_notify = is_fire ? 1 : 0;
        uint8_t fire_notify = 0;
        io_.fire_notify_topic.Publish(fire_notify);
      });

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE

  XR_LOG_PASS("ArmorTracker preview uses constructor camera info");

  auto img_topic = LibXR::Topic(LibXR::Topic::Find("image_raw"));
  auto img_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto* img_msg = reinterpret_cast<cv::Mat*>(data.addr_);
        cv::Mat frame = img_msg->clone();

        EkfPointsMsg& ekf = self->ekf_msg_;

        // —— 用构造注入的相机内参/畸变直接做投影 ——
        const CameraBase::CameraInfo& cam = self->cam_info_;

        // 只考虑 PLUMB_BOB；否则当作无畸变
        bool has_distortion =
            (cam.distortion_model == CameraBase::DistortionModel::PLUMB_BOB);

        // --- 构造 K(3x3) ---
        const auto& k_arr = cam.camera_matrix;  // 行优先 3x3
        cv::Mat k = (cv::Mat_<double>(3, 3) << k_arr[0], k_arr[1], k_arr[2], k_arr[3],
                     k_arr[4], k_arr[5], k_arr[6], k_arr[7], k_arr[8]);

        // --- 构造 D（PLUMB_BOB: k1,k2,p1,p2,k3）---
        cv::Mat d;
        if (has_distortion)
        {
          std::vector<double> dvec = {cam.distortion_coefficients[0],
                                      cam.distortion_coefficients[1],
                                      cam.distortion_coefficients[2],
                                      cam.distortion_coefficients[3],
                                      cam.distortion_coefficients[4]};
          d = cv::Mat(dvec).clone().reshape(1, 1);  // 1x5
        }
        else
        {
          d = cv::Mat();  // 空 -> 无畸变
        }

        // 若当前帧分辨率与标定分辨率不同，缩放 K；D 不缩放
        const double SX =
            static_cast<double>(frame.cols) / static_cast<double>(cam.width);
        const double SY =
            static_cast<double>(frame.rows) / static_cast<double>(cam.height);
        cv::Mat k_scaled = k.clone();
        k_scaled.at<double>(0, 0) *= SX;  // fx
        k_scaled.at<double>(1, 1) *= SY;  // fy
        k_scaled.at<double>(0, 2) *= SX;  // cx
        k_scaled.at<double>(1, 2) *= SY;  // cy

        auto project = [&](const Eigen::Vector3d& Pc, cv::Point2d& uv) -> bool
        {
          if (!(Pc.z() > 1e-6) || !std::isfinite(Pc.x()) || !std::isfinite(Pc.y()) ||
              !std::isfinite(Pc.z()))
          {
            return false;
          }

          std::vector<cv::Point3d> obj{cv::Point3d(Pc.x(), Pc.y(), Pc.z())};
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
          Eigen::Vector3d pc_c(ekf.center_cam.x(), ekf.center_cam.y(),
                               ekf.center_cam.z());
          Eigen::Vector3d pc_a(ekf.armors_cam[i].x(), ekf.armors_cam[i].y(),
                               ekf.armors_cam[i].z());
          if (project(pc_c, uc) && project(pc_a, ua))
          {
            cv::line(frame, uc, ua, cv::Scalar(80, 180, 255), 1, cv::LINE_AA);
          }
        }

        cv::imshow("ekf_overlay", frame);
        cv::waitKey(1);
      },
      this);

  img_topic.RegisterCallback(img_cb);
#endif
}

void ArmorTracker::OnMonitor() {}

double ArmorTracker::ArmorImageArea(const ArmorDetectorResult& armor)
{
  return std::abs(cv::contourArea(
      std::vector<cv::Point2f>(armor.points.begin(), armor.points.end())));
}

double ArmorTracker::TimestampDeltaSeconds(uint64_t newer, uint64_t older)
{
  if (newer > older)
  {
    return static_cast<double>(newer - older) / 1000000.0;
  }
  return 0.0;
}

void ArmorTracker::PushCameraPose(const PoseStamped& pose_msg)
{
  LibXR::Mutex::LockGuard lock(io_.gimbal_rotation_lock);
  io_.gimbal_rotation = PoseStampedToQuaternion(pose_msg);
  io_.latest_camera_pose =
      CameraRotationToTrackerWorldPose(io_.gimbal_rotation,
                                       io_.gimbal_to_camera_transform_static);
  io_.latest_camera_pose_valid = true;
  io_.camera_pose_history[io_.camera_pose_history_head] = pose_msg;
  io_.camera_pose_history_head =
      (io_.camera_pose_history_head + 1) % IOBlock::kCameraPoseHistorySize;
  io_.camera_pose_history_count =
      std::min(io_.camera_pose_history_count + 1, IOBlock::kCameraPoseHistorySize);
}

bool ArmorTracker::LookupCameraPose(uint64_t image_timestamp_us,
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
      const uint64_t ts = static_cast<uint64_t>(msg.timestamp);
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
      pose_out = CameraRotationToTrackerWorldPose(
          PoseStampedToQuaternion(io_.camera_pose_history[best_index]),
          io_.gimbal_to_camera_transform_static);
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

bool ArmorTracker::CompatibleImageTrackLabel(const ImageIdTrack& track,
                                             const ArmorDetectorResult& armor)
{
  if (track.type != ArmorType::INVALID && armor.type != ArmorType::INVALID &&
      track.type != armor.type)
  {
    return false;
  }
  if (track.color != ArmorColor::UNKNOWN && armor.color != ArmorColor::UNKNOWN &&
      track.color != armor.color)
  {
    return false;
  }
  if (track.number != ArmorNumber::INVALID && armor.number != ArmorNumber::INVALID &&
      track.number != armor.number)
  {
    return false;
  }
  return true;
}

int ArmorTracker::FindDetectionTrackId(std::size_t armor_index) const
{
  if (armor_index >= id_tracker_.detection_track_ids.size())
  {
    return -1;
  }
  return id_tracker_.detection_track_ids[armor_index];
}

bool ArmorTracker::IsDetectionTrackConfirmed(std::size_t armor_index) const
{
  if (armor_index >= id_tracker_.detection_track_confirmed.size())
  {
    return false;
  }
  return id_tracker_.detection_track_confirmed[armor_index] != 0;
}

void ArmorTracker::UpdateImageIdTracks(const ArmorDetectorResults& armors_msg,
                                       uint64_t image_timestamp_us)
{
  id_tracker_.detection_track_ids.assign(armors_msg.size(), -1);
  id_tracker_.detection_track_confirmed.assign(armors_msg.size(), 0);

  const std::uint32_t appear_hits = IdTrackAppearHits();
  const double appear_timeout_sec = IdTrackAppearTimeoutSec();
  const std::uint32_t tentative_misses = IdTrackTentativeMisses();
  const double tentative_timeout_sec = IdTrackTentativeTimeoutSec();
  const std::uint32_t disappear_misses = IdTrackDisappearMisses();
  const double disappear_timeout_sec = IdTrackDisappearTimeoutSec();

  auto timeout_satisfied = [](uint64_t now_us, uint64_t since_us, double timeout_sec)
  {
    if (timeout_sec <= 1e-9 || now_us == 0 || since_us == 0)
    {
      return true;
    }
    return TimestampDeltaSeconds(now_us, since_us) + 1e-9 >= timeout_sec;
  };

  auto compatible_track_pair = [](const ImageIdTrack& lhs, const ImageIdTrack& rhs)
  {
    if (lhs.type != ArmorType::INVALID && rhs.type != ArmorType::INVALID &&
        lhs.type != rhs.type)
    {
      return false;
    }
    if (lhs.color != ArmorColor::UNKNOWN && rhs.color != ArmorColor::UNKNOWN &&
        lhs.color != rhs.color)
    {
      return false;
    }
    if (lhs.number != ArmorNumber::INVALID && rhs.number != ArmorNumber::INVALID &&
        lhs.number != rhs.number)
    {
      return false;
    }
    return true;
  };

  auto reset_track = [](ImageIdTrack& track)
  {
    track = ImageIdTrack{};
  };

  auto assign_track = [&](ImageIdTrack& track, const ArmorDetectorResult& armor,
                          uint8_t armor_index, double score)
  {
    const double dt_raw = TimestampDeltaSeconds(image_timestamp_us, track.last_timestamp_us);
    const double dt = dt_raw > 1e-4 ? dt_raw : 1.0 / 100.0;
    const double measured_area = std::max(1.0, ArmorImageArea(armor));
    if (track.age > 0)
    {
      const cv::Point2f measured_image_velocity(
          static_cast<float>((armor.center.x - track.image_center.x) / dt),
          static_cast<float>((armor.center.y - track.image_center.y) / dt));
      track.image_velocity.x = 0.65f * track.image_velocity.x +
                               0.35f * measured_image_velocity.x;
      track.image_velocity.y = 0.65f * track.image_velocity.y +
                               0.35f * measured_image_velocity.y;
      const double measured_area_rate = (measured_area - track.area) / dt;
      track.area_rate = 0.65 * track.area_rate + 0.35 * measured_area_rate;
    }
    else
    {
      track.image_velocity = cv::Point2f(0.0f, 0.0f);
      track.area_rate = 0.0;
    }

    if (track.first_timestamp_us == 0)
    {
      track.first_timestamp_us = image_timestamp_us;
    }

    track.image_center = armor.center;
    track.area = measured_area;
    track.confidence = armor.confidence;
    if (armor.color != ArmorColor::UNKNOWN || track.color == ArmorColor::UNKNOWN)
    {
      track.color = armor.color;
    }
    if (armor.number != ArmorNumber::INVALID || track.number == ArmorNumber::INVALID)
    {
      track.number = armor.number;
    }
    if (armor.type != ArmorType::INVALID || track.type == ArmorType::INVALID)
    {
      track.type = armor.type;
    }
    track.last_timestamp_us = image_timestamp_us;
    track.last_seen_timestamp_us = image_timestamp_us;
    track.age++;
    track.hit_count++;
    track.miss_count = 0;
    if (!track.confirmed && track.hit_count >= appear_hits &&
        timeout_satisfied(image_timestamp_us, track.first_timestamp_us,
                          appear_timeout_sec))
    {
      track.confirmed = true;
    }
    track.matched_this_frame = true;
    track.matched_armor_index = armor_index;
    (void)score;
    track.active = true;
  };

  auto create_track = [&](const ArmorDetectorResult& armor, uint8_t armor_index)
  {
    for (auto& track : id_tracker_.tracks)
    {
      if (track.active)
      {
        continue;
      }
      track = ImageIdTrack{};
      track.active = true;
      track.confirmed = false;
      track.track_id = id_tracker_.next_track_id++;
      assign_track(track, armor, armor_index, 0.0);
      return;
    }
  };

  auto suppress_spawn = [&](const ArmorDetectorResult& armor)
  {
    for (const auto& track : id_tracker_.tracks)
    {
      if (!track.active || track.miss_count > 6U)
      {
        continue;
      }
      if (!CompatibleImageTrackLabel(track, armor))
      {
        continue;
      }
      const double dt = std::max(TimestampDeltaSeconds(image_timestamp_us, track.last_timestamp_us),
                                 1.0 / 100.0);
      const cv::Point2f predicted_center(
          track.image_center.x + track.image_velocity.x * static_cast<float>(dt),
          track.image_center.y + track.image_velocity.y * static_cast<float>(dt));
      const double center_diff = std::hypot(
          static_cast<double>(armor.center.x - predicted_center.x),
          static_cast<double>(armor.center.y - predicted_center.y));
      const double predicted_area = std::max(1.0, track.area + track.area_rate * dt);
      const double area = std::max(1.0, ArmorImageArea(armor));
      const double area_log = std::abs(std::log(area / predicted_area));
      const bool strong_same_track =
          (center_diff < 55.0 && area_log < 0.45) ||
          (center_diff < 28.0 && area_log < 0.90);
      const bool relaxed_confirmed_same_track =
          track.confirmed && ((center_diff < 72.0 && area_log < 0.28) ||
                              (center_diff < 42.0 && area_log < 0.45));
      if (strong_same_track || relaxed_confirmed_same_track)
      {
        return true;
      }
    }
    return false;
  };

  for (auto& track : id_tracker_.tracks)
  {
    if (!track.active)
    {
      continue;
    }
    track.matched_this_frame = false;
    track.matched_armor_index = 255;
  }

  std::array<cv::Point2f, IdTrackerRuntime::kMaxTracks> predicted_centers{};
  std::array<bool, IdTrackerRuntime::kMaxTracks> predicted_center_valid{};
  for (std::size_t track_slot = 0; track_slot < id_tracker_.tracks.size(); ++track_slot)
  {
    const auto& track = id_tracker_.tracks[track_slot];
    if (!track.active)
    {
      continue;
    }
    const double dt = std::max(TimestampDeltaSeconds(image_timestamp_us, track.last_timestamp_us),
                               1.0 / 100.0);
    predicted_centers[track_slot] = cv::Point2f(
        track.image_center.x + track.image_velocity.x * static_cast<float>(dt),
        track.image_center.y + track.image_velocity.y * static_cast<float>(dt));
    predicted_center_valid[track_slot] = true;
  }

  std::array<std::vector<double>, IdTrackerRuntime::kMaxTracks> order_bias_by_track;
  for (auto& bias_vec : order_bias_by_track)
  {
    bias_vec.assign(armors_msg.size(), 0.0);
  }

  auto apply_dual_order_bias = [&](std::size_t lhs_slot, std::size_t rhs_slot)
  {
    const auto& lhs_track = id_tracker_.tracks[lhs_slot];
    const auto& rhs_track = id_tracker_.tracks[rhs_slot];
    if (!lhs_track.active || !rhs_track.active || !lhs_track.confirmed || !rhs_track.confirmed)
    {
      return;
    }
    if (!predicted_center_valid[lhs_slot] || !predicted_center_valid[rhs_slot])
    {
      return;
    }
    if (!compatible_track_pair(lhs_track, rhs_track))
    {
      return;
    }

    std::size_t compatible_track_count = 0;
    for (const auto& track : id_tracker_.tracks)
    {
      if (!track.active || !track.confirmed)
      {
        continue;
      }
      if (track.type == lhs_track.type && track.number == lhs_track.number &&
          track.color == lhs_track.color)
      {
        compatible_track_count++;
      }
    }
    if (compatible_track_count != 2U)
    {
      return;
    }

    std::vector<std::size_t> compatible_detections;
    for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
    {
      if (CompatibleImageTrackLabel(lhs_track, armors_msg[armor_index]) &&
          CompatibleImageTrackLabel(rhs_track, armors_msg[armor_index]))
      {
        compatible_detections.push_back(armor_index);
      }
    }
    if (compatible_detections.size() != 2U)
    {
      return;
    }

    cv::Point2f axis(predicted_centers[rhs_slot].x - predicted_centers[lhs_slot].x,
                     predicted_centers[rhs_slot].y - predicted_centers[lhs_slot].y);
    double axis_norm = std::hypot(static_cast<double>(axis.x), static_cast<double>(axis.y));
    if (axis_norm < 20.0)
    {
      axis = std::abs(axis.x) >= std::abs(axis.y) ? cv::Point2f(1.0f, 0.0f)
                                                  : cv::Point2f(0.0f, 1.0f);
      axis_norm = 1.0;
    }
    axis.x = static_cast<float>(axis.x / axis_norm);
    axis.y = static_cast<float>(axis.y / axis_norm);

    auto project = [&](const cv::Point2f& p)
    {
      return static_cast<double>(p.x) * static_cast<double>(axis.x) +
             static_cast<double>(p.y) * static_cast<double>(axis.y);
    };

    std::array<std::size_t, 2> ordered_track_slots = {lhs_slot, rhs_slot};
    if (project(predicted_centers[ordered_track_slots[0]]) >
        project(predicted_centers[ordered_track_slots[1]]))
    {
      std::swap(ordered_track_slots[0], ordered_track_slots[1]);
    }

    std::array<std::size_t, 2> ordered_detection_indices = {compatible_detections[0],
                                                             compatible_detections[1]};
    if (project(armors_msg[ordered_detection_indices[0]].center) >
        project(armors_msg[ordered_detection_indices[1]].center))
    {
      std::swap(ordered_detection_indices[0], ordered_detection_indices[1]);
    }

    const double detection_sep = std::abs(project(armors_msg[ordered_detection_indices[1]].center) -
                                          project(armors_msg[ordered_detection_indices[0]].center));
    if (detection_sep < 18.0)
    {
      return;
    }

    const double predicted_sep = std::abs(project(predicted_centers[ordered_track_slots[1]]) -
                                          project(predicted_centers[ordered_track_slots[0]]));
    const double order_bias =
        predicted_sep > 90.0 && detection_sep > 60.0 ? 0.30 : 0.20;

    order_bias_by_track[ordered_track_slots[0]][ordered_detection_indices[0]] -= order_bias;
    order_bias_by_track[ordered_track_slots[0]][ordered_detection_indices[1]] += order_bias;
    order_bias_by_track[ordered_track_slots[1]][ordered_detection_indices[0]] += order_bias;
    order_bias_by_track[ordered_track_slots[1]][ordered_detection_indices[1]] -= order_bias;
  };

  for (std::size_t lhs_slot = 0; lhs_slot < id_tracker_.tracks.size(); ++lhs_slot)
  {
    for (std::size_t rhs_slot = lhs_slot + 1; rhs_slot < id_tracker_.tracks.size(); ++rhs_slot)
    {
      apply_dual_order_bias(lhs_slot, rhs_slot);
    }
  }

  struct MatchCandidate
  {
    std::size_t track_slot = 0;
    std::size_t armor_index = 0;
    double score = 0.0;
    double center_diff = 0.0;
    double area_log = 0.0;
  };
  std::vector<MatchCandidate> candidates;

  for (std::size_t track_slot = 0; track_slot < id_tracker_.tracks.size(); ++track_slot)
  {
    const auto& track = id_tracker_.tracks[track_slot];
    if (!track.active)
    {
      continue;
    }
    const double dt = std::max(TimestampDeltaSeconds(image_timestamp_us, track.last_timestamp_us),
                               1.0 / 100.0);
    const double miss_scale = static_cast<double>(std::min<uint32_t>(track.miss_count, 6U));
    const cv::Point2f predicted_center = predicted_centers[track_slot];
    const double predicted_area = std::max(1.0, track.area + track.area_rate * dt);
    const double center_score_gate = 80.0 + 15.0 * miss_scale;
    const double center_gate = 140.0 + 20.0 * miss_scale;
    const double area_gate = 0.55 + 0.08 * miss_scale;

    for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
    {
      const auto& armor = armors_msg[armor_index];
      if (!CompatibleImageTrackLabel(track, armor))
      {
        continue;
      }
      const double center_diff = std::hypot(
          static_cast<double>(armor.center.x - predicted_center.x),
          static_cast<double>(armor.center.y - predicted_center.y));
      const double area = std::max(1.0, ArmorImageArea(armor));
      const double area_log = std::abs(std::log(area / predicted_area));
      if (center_diff > center_gate || area_log > area_gate)
      {
        continue;
      }
      double score = 0.78 * center_diff / center_score_gate +
                     0.22 * area_log / area_gate -
                     0.05 * static_cast<double>(armor.confidence) +
                     order_bias_by_track[track_slot][armor_index];
      if (track.confirmed)
      {
        score -= 0.18;
      }
      if (track.miss_count == 0U)
      {
        score -= 0.08;
      }
      if (track.confirmed && track.miss_count == 0U && center_diff < 18.0 &&
          area_log < 0.12)
      {
        score -= 0.16;
      }
      candidates.push_back({track_slot, armor_index, score, center_diff, area_log});
    }
  }

  auto candidate_less = [&](const MatchCandidate& lhs, const MatchCandidate& rhs)
  {
    const auto& lhs_track = id_tracker_.tracks[lhs.track_slot];
    const auto& rhs_track = id_tracker_.tracks[rhs.track_slot];
    if (lhs_track.confirmed != rhs_track.confirmed &&
        std::abs(lhs.score - rhs.score) < 0.25)
    {
      return lhs_track.confirmed > rhs_track.confirmed;
    }
    if (lhs_track.miss_count != rhs_track.miss_count &&
        std::abs(lhs.score - rhs.score) < 0.20)
    {
      return lhs_track.miss_count < rhs_track.miss_count;
    }
    if (std::abs(lhs.score - rhs.score) > 1e-6)
    {
      return lhs.score < rhs.score;
    }
    return lhs_track.track_id < rhs_track.track_id;
  };

  std::vector<std::size_t> ordered_candidate_indices;
  ordered_candidate_indices.reserve(candidates.size());
  for (std::size_t candidate_index = 0; candidate_index < candidates.size();
       ++candidate_index)
  {
    ordered_candidate_indices.push_back(candidate_index);
  }
  std::sort(ordered_candidate_indices.begin(), ordered_candidate_indices.end(),
            [&](std::size_t lhs_index, std::size_t rhs_index)
            {
              return candidate_less(candidates[lhs_index], candidates[rhs_index]);
            });

  std::array<bool, IdTrackerRuntime::kMaxTracks> track_used{};
  std::vector<bool> detection_used(armors_msg.size(), false);
  std::vector<bool> candidate_selected(candidates.size(), false);
  std::array<std::vector<std::size_t>, IdTrackerRuntime::kMaxTracks> track_candidate_indices;
  std::vector<int> detection_bit_indices(armors_msg.size(), -1);
  std::vector<uint8_t> candidate_detection_bits(candidates.size(), 0);
  std::vector<std::size_t> unique_detection_indices;
  unique_detection_indices.reserve(armors_msg.size());
  for (std::size_t candidate_index = 0; candidate_index < candidates.size();
       ++candidate_index)
  {
    const auto& candidate = candidates[candidate_index];
    track_candidate_indices[candidate.track_slot].push_back(candidate_index);
    if (detection_bit_indices[candidate.armor_index] < 0)
    {
      detection_bit_indices[candidate.armor_index] =
          static_cast<int>(unique_detection_indices.size());
      unique_detection_indices.push_back(candidate.armor_index);
    }
    candidate_detection_bits[candidate_index] = static_cast<uint8_t>(
        detection_bit_indices[candidate.armor_index]);
  }

  for (auto& candidate_indices : track_candidate_indices)
  {
    std::sort(candidate_indices.begin(), candidate_indices.end(),
              [&](std::size_t lhs_index, std::size_t rhs_index)
              {
                return candidate_less(candidates[lhs_index], candidates[rhs_index]);
              });
  }

  bool used_global_assignment = false;
  std::vector<std::size_t> assignment_track_slots;
  assignment_track_slots.reserve(id_tracker_.tracks.size());
  for (std::size_t track_slot = 0; track_slot < id_tracker_.tracks.size(); ++track_slot)
  {
    if (!track_candidate_indices[track_slot].empty())
    {
      assignment_track_slots.push_back(track_slot);
    }
  }
  std::sort(assignment_track_slots.begin(), assignment_track_slots.end(),
            [&](std::size_t lhs_slot, std::size_t rhs_slot)
            {
              const auto& lhs_track = id_tracker_.tracks[lhs_slot];
              const auto& rhs_track = id_tracker_.tracks[rhs_slot];
              if (lhs_track.confirmed != rhs_track.confirmed)
              {
                return lhs_track.confirmed > rhs_track.confirmed;
              }
              if (lhs_track.miss_count != rhs_track.miss_count)
              {
                return lhs_track.miss_count < rhs_track.miss_count;
              }
              if (track_candidate_indices[lhs_slot].size() !=
                  track_candidate_indices[rhs_slot].size())
              {
                return track_candidate_indices[lhs_slot].size() <
                       track_candidate_indices[rhs_slot].size();
              }
              return lhs_track.track_id < rhs_track.track_id;
            });

  const std::size_t unique_detection_count = unique_detection_indices.size();
  if (!assignment_track_slots.empty() && unique_detection_count <= 16U)
  {
    struct AssignmentState
    {
      bool valid{false};
      uint16_t matched_count{0};
      uint16_t confirmed_matched_count{0};
      double total_score{0.0};
    };

    const std::size_t state_count = std::size_t{1} << unique_detection_count;
    const std::size_t memo_size = (assignment_track_slots.size() + 1U) * state_count;
    std::vector<uint8_t> memo_seen(memo_size, 0);
    std::vector<AssignmentState> memo(memo_size);
    std::vector<int32_t> memo_choice(memo_size, -2);

    auto assignment_better = [&](const AssignmentState& lhs, const AssignmentState& rhs)
    {
      if (!lhs.valid)
      {
        return false;
      }
      if (!rhs.valid)
      {
        return true;
      }
      if (lhs.matched_count != rhs.matched_count)
      {
        return lhs.matched_count > rhs.matched_count;
      }
      if (lhs.confirmed_matched_count != rhs.confirmed_matched_count)
      {
        return lhs.confirmed_matched_count > rhs.confirmed_matched_count;
      }
      if (std::abs(lhs.total_score - rhs.total_score) > 1e-9)
      {
        return lhs.total_score < rhs.total_score;
      }
      return false;
    };

    auto memo_index_of = [&](std::size_t track_order_index, std::size_t used_mask)
    {
      return track_order_index * state_count + used_mask;
    };

    auto solve_assignment =
        [&](auto&& self, std::size_t track_order_index,
            std::size_t used_mask) -> AssignmentState
    {
      const std::size_t memo_index = memo_index_of(track_order_index, used_mask);
      if (memo_seen[memo_index] != 0U)
      {
        return memo[memo_index];
      }
      memo_seen[memo_index] = 1U;

      AssignmentState best;
      best.valid = true;
      memo_choice[memo_index] = -1;
      if (track_order_index >= assignment_track_slots.size())
      {
        memo[memo_index] = best;
        return best;
      }

      best = self(self, track_order_index + 1U, used_mask);

      const std::size_t track_slot = assignment_track_slots[track_order_index];
      const auto& track = id_tracker_.tracks[track_slot];
      for (std::size_t candidate_index : track_candidate_indices[track_slot])
      {
        const std::size_t detection_bit = candidate_detection_bits[candidate_index];
        const std::size_t detection_mask = std::size_t{1} << detection_bit;
        if ((used_mask & detection_mask) != 0U)
        {
          continue;
        }

        AssignmentState next =
            self(self, track_order_index + 1U, used_mask | detection_mask);
        if (!next.valid)
        {
          continue;
        }
        next.matched_count++;
        if (track.confirmed)
        {
          next.confirmed_matched_count++;
        }
        next.total_score += candidates[candidate_index].score;
        if (assignment_better(next, best))
        {
          best = next;
          memo_choice[memo_index] = static_cast<int32_t>(candidate_index);
        }
      }

      memo[memo_index] = best;
      return best;
    };

    const AssignmentState best_assignment = solve_assignment(solve_assignment, 0U, 0U);
    if (best_assignment.valid)
    {
      used_global_assignment = true;
      std::size_t used_mask = 0U;
      for (std::size_t track_order_index = 0;
           track_order_index < assignment_track_slots.size(); ++track_order_index)
      {
        const std::size_t memo_index = memo_index_of(track_order_index, used_mask);
        const int32_t choice = memo_choice[memo_index];
        if (choice < 0)
        {
          continue;
        }
        const std::size_t candidate_index = static_cast<std::size_t>(choice);
        candidate_selected[candidate_index] = true;
        used_mask |= std::size_t{1} << candidate_detection_bits[candidate_index];
      }
    }
  }

  if (!used_global_assignment)
  {
    for (std::size_t candidate_index : ordered_candidate_indices)
    {
      const auto& candidate = candidates[candidate_index];
      if (track_used[candidate.track_slot] || detection_used[candidate.armor_index])
      {
        continue;
      }
      candidate_selected[candidate_index] = true;
      track_used[candidate.track_slot] = true;
      detection_used[candidate.armor_index] = true;
    }
    track_used.fill(false);
    std::fill(detection_used.begin(), detection_used.end(), false);
  }

  for (std::size_t candidate_index : ordered_candidate_indices)
  {
    if (!candidate_selected[candidate_index])
    {
      continue;
    }
    const auto& candidate = candidates[candidate_index];
    auto& track = id_tracker_.tracks[candidate.track_slot];
    assign_track(track, armors_msg[candidate.armor_index],
                 static_cast<uint8_t>(candidate.armor_index), candidate.score);
    track_used[candidate.track_slot] = true;
    detection_used[candidate.armor_index] = true;
  }

  for (std::size_t track_slot = 0; track_slot < id_tracker_.tracks.size(); ++track_slot)
  {
    auto& track = id_tracker_.tracks[track_slot];
    if (!track.active || track_used[track_slot])
    {
      continue;
    }

    const double dt = TimestampDeltaSeconds(image_timestamp_us, track.last_timestamp_us);
    if (dt > 1e-4)
    {
      track.image_center.x += track.image_velocity.x * static_cast<float>(dt);
      track.image_center.y += track.image_velocity.y * static_cast<float>(dt);
      track.area = std::max(1.0, track.area + track.area_rate * dt);
      track.last_timestamp_us = image_timestamp_us;
    }
    track.miss_count++;
    track.hit_count = 0;
    track.matched_this_frame = false;
    track.matched_armor_index = 255;

    const bool drop_tentative =
        !track.confirmed && track.miss_count >= tentative_misses &&
        timeout_satisfied(image_timestamp_us, track.last_seen_timestamp_us,
                          tentative_timeout_sec);
    const bool drop_confirmed =
        track.confirmed && track.miss_count >= disappear_misses &&
        timeout_satisfied(image_timestamp_us, track.last_seen_timestamp_us,
                          disappear_timeout_sec);
    if (drop_tentative || drop_confirmed)
    {
      reset_track(track);
    }
  }

  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    if (detection_used[armor_index] || suppress_spawn(armors_msg[armor_index]))
    {
      continue;
    }
    create_track(armors_msg[armor_index], static_cast<uint8_t>(armor_index));
  }

  for (const auto& track : id_tracker_.tracks)
  {
    if (!track.active || !track.matched_this_frame || track.matched_armor_index == 255)
    {
      continue;
    }
    if (track.matched_armor_index < id_tracker_.detection_track_ids.size())
    {
      id_tracker_.detection_track_ids[track.matched_armor_index] = track.track_id;
      id_tracker_.detection_track_confirmed[track.matched_armor_index] =
          track.confirmed ? 1 : 0;
    }
  }

  if (rt_.tracked_face_track_id_valid)
  {
    bool found_bound_track = false;
    for (const auto& track : id_tracker_.tracks)
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
    for (const auto& track : id_tracker_.tracks)
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

void ArmorTracker::Init(const ArmorDetectorResults& armors_msg)
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

void ArmorTracker::Update(const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us)
{
  Eigen::VectorXd ekf_prediction = ekf_.ekf.Predict();  // 预测
  XR_LOG_DEBUG("EKF predict");
  const bool single_armor_mode = SingleArmorModeEnabled();
  const bool id_assist_enabled = IdAssistEnabled();
  (void)image_timestamp_us;
  bool matched = false;
  ekf_.state = ekf_prediction;
  rt_.face_switch_cooldown_remaining =
      std::max(0.0, rt_.face_switch_cooldown_remaining - time_.dt);
  if (SymmetricGeometryEnabled())
  {
    rt_.another_r = ekf_.state(8);
    rt_.dz = 0.0;
    rt_.dz_abs_ref = 0.0;
  }

  ArmorTracker::CandidateDebugMsg candidate_debug{};
  std::fill(candidate_debug.detection_track_ids.begin(),
            candidate_debug.detection_track_ids.end(), static_cast<int16_t>(-1));
  std::fill(candidate_debug.detection_track_confirmed.begin(),
            candidate_debug.detection_track_confirmed.end(), static_cast<uint8_t>(0));
  candidate_debug.face_switch_enabled = FaceSwitchEnabled() ? 1 : 0;
  candidate_debug.relaxed_face_switch_enabled = RelaxedFaceSwitchEnabled() ? 1 : 0;
  candidate_debug.odd_face_switch_enabled = OddFaceSwitchEnabled() ? 1 : 0;
  candidate_debug.view_priority_enabled = ViewPriorityEnabled() ? 1 : 0;
  candidate_debug.directional_face_switch_enabled = DirectionalFaceSwitchEnabled() ? 1 : 0;
  candidate_debug.tracked_face_track_id_valid = rt_.tracked_face_track_id_valid ? 1 : 0;
  candidate_debug.tracked_face_track_id =
      rt_.tracked_face_track_id_valid ? static_cast<int16_t>(rt_.tracked_face_track_id)
                                      : static_cast<int16_t>(-1);
  candidate_debug.tracked_armors_num = static_cast<uint8_t>(rt_.tracked_armors_num);
  candidate_debug.predicted_vyaw =
      static_cast<float>(ekf_prediction(ExtendedKalmanFilter::V_YAW));
  candidate_debug.max_match_distance = static_cast<float>(cfg_.match.max_match_distance);
  candidate_debug.max_match_yaw_diff = static_cast<float>(cfg_.match.max_match_yaw_diff);
  candidate_debug.face_switch_score_deadzone =
      static_cast<float>(FaceSwitchScoreDeadzone());
  candidate_debug.face_switch_position_deadzone =
      static_cast<float>(FaceSwitchPositionDeadzone());
  candidate_debug.face_switch_yaw_deadzone =
      static_cast<float>(FaceSwitchYawDeadzone());
  candidate_debug.face_switch_timeout_sec = static_cast<float>(FaceSwitchTimeoutSec());
  candidate_debug.face_switch_cooldown_remaining =
      static_cast<float>(rt_.face_switch_cooldown_remaining);

  if (!armors_msg.empty())
  {
    ArmorMatchCandidate best_candidate;
    ArmorMatchCandidate best_same_face_candidate;
    ArmorMatchCandidate best_switch_candidate;
    const int armor_count =
        single_armor_mode
            ? 1
            : (FaceSwitchEnabled()
                   ? std::max(1, static_cast<int>(rt_.tracked_armors_num))
                   : 1);
    Eigen::Vector3d camera_world = Eigen::Vector3d::Zero();
    int preferred_adjacent_face = -1;
    if (DirectionalFaceSwitchEnabled() && rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
    {
      const double vyaw = ekf_prediction(ExtendedKalmanFilter::V_YAW);
      if (vyaw > 0.05)
      {
        preferred_adjacent_face = 1;
      }
      else if (vyaw < -0.05)
      {
        preferred_adjacent_face = 3;
      }
    }
    candidate_debug.preferred_adjacent_face = static_cast<int8_t>(preferred_adjacent_face);
    candidate_debug.detection_count = static_cast<uint8_t>(std::min<std::size_t>(
        armors_msg.size(), ArmorTracker::CandidateDebugMsg::kMaxDetections));
    for (std::size_t armor_index = 0; armor_index < candidate_debug.detection_count;
         ++armor_index)
    {
      const int detection_track_id = FindDetectionTrackId(armor_index);
      candidate_debug.detection_track_ids[armor_index] =
          static_cast<int16_t>(detection_track_id);
      candidate_debug.detection_track_confirmed[armor_index] =
          IsDetectionTrackConfirmed(armor_index) ? 1 : 0;
    }
    {
      LibXR::Mutex::LockGuard lock(io_.gimbal_rotation_lock);
      const LibXR::Transform<double> t_wc =
          io_.latest_camera_pose_valid
              ? io_.latest_camera_pose
              : (LibXR::Transform<double>(io_.gimbal_rotation, {0.0, 0.0, 0.0}) +
                 io_.gimbal_to_camera_transform_static);
      camera_world = Eigen::Vector3d(t_wc.translation.x(), t_wc.translation.y(),
                                     t_wc.translation.z());
    }

    bool has_same_number_candidate = false;
    if (rt_.tracked_id != ArmorNumber::INVALID)
    {
      for (const auto& armor : armors_msg)
      {
        if (rt_.tracked_armor.type != ArmorType::INVALID &&
            armor.type != rt_.tracked_armor.type)
        {
          continue;
        }
        if (armor.number == rt_.tracked_id)
        {
          has_same_number_candidate = true;
          break;
        }
      }
    }

    const double tracked_image_area = std::max(
        1.0, std::abs(cv::contourArea(std::vector<cv::Point2f>(
                 rt_.tracked_armor.points.begin(), rt_.tracked_armor.points.end()))));
    const cv::Point2f tracked_image_center = rt_.tracked_armor.center;
    bool observed_persistent_track_this_frame = false;

    for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
    {
      const auto& armor = armors_msg[armor_index];
      if (rt_.tracked_armor.type != ArmorType::INVALID &&
          armor.type != rt_.tracked_armor.type)
      {
        continue;
      }

      auto p = armor.pose.translation;
      Eigen::Vector3d position_vec(p.x(), p.y(), p.z());
      const double image_area = ArmorImageArea(armor);
      const double area_score = std::min(image_area / 2500.0, 1.0);
      const Eigen::Vector3d armor_front =
          armor.pose.rotation.ToRotationMatrix() * Eigen::Vector3d::UnitX();
      const Eigen::Vector3d armor_to_camera = camera_world - position_vec;
      double frontality = 0.0;
      if (armor_to_camera.norm() > 1e-6)
      {
        frontality =
            std::max(0.0, armor_front.normalized().dot(armor_to_camera.normalized()));
      }
      const double view_bonus =
          ViewPriorityEnabled() ? (0.35 * area_score + 0.35 * frontality) : 0.0;
      const double image_center_diff =
          std::hypot(static_cast<double>(armor.center.x - tracked_image_center.x),
                     static_cast<double>(armor.center.y - tracked_image_center.y));
      const double area_ratio_log =
          std::abs(std::log(std::max(image_area, 1.0) / tracked_image_area));
      const int image_track_id = FindDetectionTrackId(armor_index);
      const bool confirmed_image_track = IsDetectionTrackConfirmed(armor_index);
      const bool same_persistent_track =
          id_assist_enabled && rt_.tracked_face_track_id_valid && image_track_id >= 0 &&
          static_cast<uint16_t>(image_track_id) == rt_.tracked_face_track_id;
      observed_persistent_track_this_frame =
          observed_persistent_track_this_frame ||
          (confirmed_image_track && same_persistent_track);
      const bool same_number =
          rt_.tracked_id == ArmorNumber::INVALID || armor.number == rt_.tracked_id;
      if (has_same_number_candidate && !same_number)
      {
        continue;
      }
      if (single_armor_mode && rt_.tracked_id != ArmorNumber::INVALID &&
          (image_center_diff > SingleArmorImageCenterGatePx() ||
           area_ratio_log > SingleArmorAreaLogGate()))
      {
        XR_LOG_DEBUG(
            "Tracker single-armor reject: armor=%zu num=%d img_diff=%.1f area_log=%.3f",
            armor_index, static_cast<int>(armor.number), image_center_diff,
            area_ratio_log);
        continue;
      }

      for (int face_index = 0; face_index < armor_count; ++face_index)
      {
        if (DirectionalFaceSwitchEnabled() && rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
            face_index > 0)
        {
          if (face_index == 2 || face_index != preferred_adjacent_face)
          {
            continue;
          }
        }
        if (!OddFaceSwitchEnabled() && face_index > 0 && (face_index % 2 == 1))
        {
          continue;
        }
        if (id_assist_enabled && rt_.tracked_face_track_id_valid)
        {
          if (face_index == 0 && !same_persistent_track)
          {
            continue;
          }
          if (face_index > 0 && same_persistent_track)
          {
            continue;
          }
          if (face_index > 0 && !confirmed_image_track)
          {
            continue;
          }
        }
        Eigen::Vector3d predicted_position =
            GetArmorPositionFromState(ekf_prediction, face_index);
        const double predicted_yaw = GetArmorYawFromState(ekf_prediction, face_index);
        const double measured_yaw =
            OrientationToYawNear(armor.pose.rotation, predicted_yaw);

        const double position_diff = (predicted_position - position_vec).norm();
        const double current_yaw_diff = AngularDiffAbs(measured_yaw, predicted_yaw);
        LogImpossibleYawDiff("match", armor_index, face_index, measured_yaw, predicted_yaw, current_yaw_diff);
        if (!SymmetricGeometryEnabled() && rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
            face_index % 2 == 1 && rt_.dz_abs_ref > 0.02)
        {
          const double measured_dz_abs = std::abs(ekf_prediction(4) - position_vec.z());
          constexpr double kDzConsistencyTol = 0.03;
          if (std::abs(measured_dz_abs - rt_.dz_abs_ref) >= kDzConsistencyTol)
          {
            XR_LOG_DEBUG(
                "Tracker reject odd face by dz: armor=%zu face=%d measured=%.3f ref=%.3f",
                armor_index, face_index, measured_dz_abs, rt_.dz_abs_ref);
            continue;
          }
        }
        const double position_score =
            position_diff / std::max(cfg_.match.max_match_distance, 1e-6);
        const double yaw_score =
            current_yaw_diff / std::max(cfg_.match.max_match_yaw_diff, 1e-6);
        const double image_score =
            single_armor_mode
                ? image_center_diff / std::max(SingleArmorImageCenterGatePx(), 1.0)
                : 0.0;
        const double area_ratio_score =
            single_armor_mode
                ? area_ratio_log / std::max(SingleArmorAreaLogGate(), 1e-6)
                : 0.0;
        const double number_penalty = same_number ? 0.0 : 1.5;
        const double persistent_track_bonus =
            (same_persistent_track && face_index == 0) ? 0.45 : 0.0;
        const double confirmed_switch_bonus =
            (id_assist_enabled && face_index > 0 && confirmed_image_track) ? 0.08 : 0.0;
        const double score =
            position_score + 0.40 * yaw_score + FaceSwitchPenalty(face_index) +
            number_penalty - view_bonus + 0.35 * image_score + 0.20 * area_ratio_score -
            persistent_track_bonus - confirmed_switch_bonus;
        uint8_t debug_index = ArmorTracker::CandidateDebugMsg::kMaxItems;
        if (candidate_debug.count < ArmorTracker::CandidateDebugMsg::kMaxItems)
        {
          debug_index = candidate_debug.count;
          auto& item = candidate_debug.items[candidate_debug.count++];
          item.armor_index = static_cast<uint8_t>(std::min<std::size_t>(armor_index, 255));
          item.face_index = static_cast<uint8_t>(face_index);
          item.same_number = same_number ? 1 : 0;
          item.image_track_id = static_cast<int16_t>(image_track_id);
          item.image_track_confirmed = confirmed_image_track ? 1 : 0;
          item.same_persistent_track = same_persistent_track ? 1 : 0;
          item.number = armor.number;
          item.type = armor.type;
          item.score = static_cast<float>(score);
          item.position_diff = static_cast<float>(position_diff);
          item.yaw_diff = static_cast<float>(current_yaw_diff);
          item.view_bonus = static_cast<float>(view_bonus);
          item.area_score = static_cast<float>(area_score);
          item.frontality = static_cast<float>(frontality);
          item.center_x = armor.center.x;
          item.center_y = armor.center.y;
          item.predicted_yaw = static_cast<float>(predicted_yaw);
          item.measured_yaw = static_cast<float>(measured_yaw);
        }

        XR_LOG_DEBUG(
            "Tracker cand: armor=%zu num=%d face=%d same=%d score=%.3f pos_diff=%.3f yaw_diff=%.3f img_diff=%.1f area_log=%.3f view_bonus=%.3f area=%.3f frontality=%.3f",
            armor_index, static_cast<int>(armor.number), face_index, same_number ? 1 : 0,
            score, position_diff, current_yaw_diff, image_center_diff, area_ratio_log,
            view_bonus, area_score, frontality);

        ArmorMatchCandidate candidate{};
        candidate.armor = armor;
        candidate.armor_index = armor_index;
        candidate.debug_index = debug_index;
        candidate.face_index = face_index;
        candidate.same_number = same_number;
        candidate.image_track_id = image_track_id;
        candidate.confirmed_image_track = confirmed_image_track;
        candidate.same_persistent_track = same_persistent_track;
        candidate.measured_yaw = measured_yaw;
        candidate.position_diff = position_diff;
        candidate.yaw_diff = current_yaw_diff;
        candidate.view_bonus = view_bonus;
        candidate.area_score = area_score;
        candidate.frontality = frontality;
        candidate.image_center_diff = image_center_diff;
        candidate.area_ratio_log = area_ratio_log;
        candidate.score = score;

        if (IsBetterMatchCandidate(candidate, best_candidate))
        {
          best_candidate = candidate;
        }
        if (face_index == 0)
        {
          if (IsBetterMatchCandidate(candidate, best_same_face_candidate))
          {
            best_same_face_candidate = candidate;
          }
        }
        else if (IsBetterMatchCandidate(candidate, best_switch_candidate))
        {
          best_switch_candidate = candidate;
        }
      }
    }

    candidate_debug.has_same_number_candidate = has_same_number_candidate ? 1 : 0;

    const double relaxed_same_face_distance = cfg_.match.max_match_distance * 1.25;
    const double relaxed_face_switch_distance = cfg_.match.max_match_distance * 1.25;
    const double relaxed_face_switch_yaw_diff =
        std::max(cfg_.match.max_match_yaw_diff * 1.2,
                 cfg_.match.max_match_yaw_diff + 0.1);
    const double id_assisted_rebind_distance =
        std::min(relaxed_face_switch_distance, cfg_.match.max_match_distance * 1.30);
    const double id_assisted_rebind_yaw_diff =
        std::min(relaxed_face_switch_yaw_diff, cfg_.match.max_match_yaw_diff * 1.10);
    const double face_switch_score_deadzone = FaceSwitchScoreDeadzone();
    const double face_switch_position_deadzone = FaceSwitchPositionDeadzone();
    const double face_switch_position_tie_margin = 0.01;
    const double face_switch_yaw_deadzone = FaceSwitchYawDeadzone();
    const double face_switch_timeout_sec = FaceSwitchTimeoutSec();

    candidate_debug.relaxed_same_face_distance =
        static_cast<float>(relaxed_same_face_distance);
    candidate_debug.relaxed_face_switch_distance =
        static_cast<float>(relaxed_face_switch_distance);
    candidate_debug.relaxed_face_switch_yaw_diff =
        static_cast<float>(relaxed_face_switch_yaw_diff);
    candidate_debug.face_switch_cooldown_remaining =
        static_cast<float>(rt_.face_switch_cooldown_remaining);
    candidate_debug.best_same_face_score =
        best_same_face_candidate.face_index == 0
            ? static_cast<float>(best_same_face_candidate.score)
            : -1.0f;
    candidate_debug.best_switch_face_score =
        best_switch_candidate.face_index > 0
            ? static_cast<float>(best_switch_candidate.score)
            : -1.0f;

    const bool strict_same_face_match =
        best_same_face_candidate.face_index == 0 &&
        best_same_face_candidate.position_diff < cfg_.match.max_match_distance &&
        best_same_face_candidate.yaw_diff < cfg_.match.max_match_yaw_diff;
    const bool relaxed_same_face_match =
        best_same_face_candidate.face_index == 0 &&
        best_same_face_candidate.position_diff < relaxed_same_face_distance &&
        best_same_face_candidate.yaw_diff < cfg_.match.max_match_yaw_diff;
    const double id_assisted_same_face_distance =
        std::min(relaxed_same_face_distance, cfg_.match.max_match_distance * 1.10);
    const double id_assisted_same_face_yaw_diff =
        std::min(relaxed_face_switch_yaw_diff, cfg_.match.max_match_yaw_diff * 0.75);
    const bool id_assisted_same_face_match =
        id_assist_enabled && best_same_face_candidate.face_index == 0 &&
        best_same_face_candidate.same_persistent_track &&
        best_same_face_candidate.confirmed_image_track &&
        best_same_face_candidate.image_center_diff < IdAssistSameFaceCenterGatePx() &&
        best_same_face_candidate.area_ratio_log < IdAssistSameFaceAreaLogGate() &&
        best_same_face_candidate.position_diff < id_assisted_same_face_distance &&
        best_same_face_candidate.yaw_diff < id_assisted_same_face_yaw_diff;
    const bool matched_same_face =
        strict_same_face_match || relaxed_same_face_match || id_assisted_same_face_match;

    const bool strict_face_switch_match =
        best_switch_candidate.face_index > 0 &&
        best_switch_candidate.position_diff < cfg_.match.max_match_distance &&
        best_switch_candidate.yaw_diff < cfg_.match.max_match_yaw_diff;
    const bool relaxed_face_switch_match =
        RelaxedFaceSwitchEnabled() && best_switch_candidate.face_index > 0 &&
        best_switch_candidate.position_diff < relaxed_face_switch_distance &&
        best_switch_candidate.yaw_diff < relaxed_face_switch_yaw_diff;
    const bool persistent_track_missing_this_frame =
        id_assist_enabled && rt_.tracked_face_track_id_valid &&
        !observed_persistent_track_this_frame;
    const bool id_assisted_face_rebind_match =
        persistent_track_missing_this_frame && best_switch_candidate.face_index > 0 &&
        best_switch_candidate.confirmed_image_track &&
        best_switch_candidate.position_diff < id_assisted_rebind_distance &&
        best_switch_candidate.yaw_diff < id_assisted_rebind_yaw_diff;
    const bool id_assisted_face_handover_match =
        best_switch_candidate.face_index > 0 && best_switch_candidate.confirmed_image_track &&
        best_same_face_candidate.face_index == 0 &&
        best_same_face_candidate.same_persistent_track &&
        best_same_face_candidate.position_diff > relaxed_same_face_distance * 1.5 &&
        best_switch_candidate.position_diff < id_assisted_rebind_distance &&
        best_switch_candidate.yaw_diff < id_assisted_rebind_yaw_diff &&
        best_switch_candidate.position_diff + face_switch_position_deadzone <
            best_same_face_candidate.position_diff;
    const bool matched_switch_face =
        strict_face_switch_match || relaxed_face_switch_match ||
        id_assisted_face_rebind_match || id_assisted_face_handover_match;

    const bool switch_has_clear_score_advantage =
        matched_switch_face && best_same_face_candidate.face_index == 0 &&
        best_switch_candidate.score + face_switch_score_deadzone <
            best_same_face_candidate.score;
    const bool switch_has_clear_position_advantage =
        matched_switch_face && best_same_face_candidate.face_index == 0 &&
        best_switch_candidate.position_diff + face_switch_position_deadzone <
            best_same_face_candidate.position_diff;
    const bool switch_has_clear_yaw_advantage =
        matched_switch_face && best_same_face_candidate.face_index == 0 &&
        best_switch_candidate.position_diff <
            best_same_face_candidate.position_diff + face_switch_position_tie_margin &&
        best_switch_candidate.yaw_diff + face_switch_yaw_deadzone <
            best_same_face_candidate.yaw_diff;

    const bool switch_blocked_by_timeout =
        rt_.face_switch_cooldown_remaining > 1e-6 && matched_same_face;
    const bool allow_face_switch =
        matched_switch_face && !switch_blocked_by_timeout &&
        (!matched_same_face || switch_has_clear_score_advantage ||
         switch_has_clear_position_advantage || switch_has_clear_yaw_advantage);

    candidate_debug.same_face_matched = matched_same_face ? 1 : 0;
    candidate_debug.switch_face_matched = matched_switch_face ? 1 : 0;
    candidate_debug.switch_blocked_by_timeout = switch_blocked_by_timeout ? 1 : 0;
    candidate_debug.switch_allowed = allow_face_switch ? 1 : 0;

    const ArmorMatchCandidate* debug_candidate = nullptr;
    const ArmorMatchCandidate* selected_candidate = nullptr;
    uint8_t accepted_mode = 0;

    if (allow_face_switch)
    {
      selected_candidate = &best_switch_candidate;
      accepted_mode =
          strict_face_switch_match ? 1
                                   : (id_assisted_face_rebind_match ? 5
                                                                     : (id_assisted_face_handover_match ? 6 : 3));
    }
    else if (matched_same_face)
    {
      selected_candidate = &best_same_face_candidate;
      accepted_mode = strict_same_face_match ? 1 : (id_assisted_same_face_match ? 4 : 2);
    }

    if (selected_candidate != nullptr)
    {
      debug_candidate = selected_candidate;
    }
    else if (best_candidate.face_index >= 0)
    {
      debug_candidate = &best_candidate;
    }
    if (debug_candidate != nullptr)
    {
      candidate_debug.selected_index = debug_candidate->debug_index;
      rt_.info_position_diff = debug_candidate->position_diff;
      rt_.info_yaw_diff = debug_candidate->yaw_diff;
    }
    else
    {
      rt_.info_position_diff = DBL_MAX;
      rt_.info_yaw_diff = DBL_MAX;
    }

    if (selected_candidate != nullptr)
    {
      const bool did_face_switch = selected_candidate->face_index != 0;
      const double preserved_v_yaw =
          ekf_prediction(ExtendedKalmanFilter::V_YAW);
      candidate_debug.matched = 1;
      candidate_debug.accepted_mode = accepted_mode;
      XR_LOG_DEBUG(
          "Tracker pick: armor=%zu num=%d face=%d same=%d score=%.3f pos_diff=%.3f yaw_diff=%.3f view_bonus=%.3f area=%.3f frontality=%.3f cooldown=%.3f",
          selected_candidate->armor_index,
          static_cast<int>(selected_candidate->armor.number),
          selected_candidate->face_index, selected_candidate->same_number ? 1 : 0,
          selected_candidate->score, selected_candidate->position_diff,
          selected_candidate->yaw_diff, selected_candidate->view_bonus,
          selected_candidate->area_score, selected_candidate->frontality,
          rt_.face_switch_cooldown_remaining);

      if (did_face_switch)
      {
        if (!strict_face_switch_match && !relaxed_face_switch_match &&
            id_assisted_face_rebind_match)
        {
          XR_LOG_DEBUG(
              "Tracker id-assisted face rebind: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f observed_persistent=%d",
              selected_candidate->face_index, selected_candidate->position_diff,
              selected_candidate->yaw_diff,
              static_cast<int>(selected_candidate->armor.number),
              face_switch_timeout_sec,
              observed_persistent_track_this_frame ? 1 : 0);
        }
        else if (!strict_face_switch_match && !relaxed_face_switch_match &&
                 id_assisted_face_handover_match)
        {
          XR_LOG_DEBUG(
              "Tracker id-assisted face handover: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f same_pos=%.3f",
              selected_candidate->face_index, selected_candidate->position_diff,
              selected_candidate->yaw_diff,
              static_cast<int>(selected_candidate->armor.number),
              face_switch_timeout_sec,
              best_same_face_candidate.position_diff);
        }
        else if (!strict_face_switch_match && relaxed_face_switch_match)
        {
          XR_LOG_DEBUG(
              "Tracker relaxed face switch: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f",
              selected_candidate->face_index, selected_candidate->position_diff,
              selected_candidate->yaw_diff,
              static_cast<int>(selected_candidate->armor.number),
              face_switch_timeout_sec);
        }
        else
        {
          XR_LOG_DEBUG(
              "Tracker face switch: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f",
              selected_candidate->face_index, selected_candidate->position_diff,
              selected_candidate->yaw_diff,
              static_cast<int>(selected_candidate->armor.number),
              face_switch_timeout_sec);
        }
        SwitchTrackedFace(selected_candidate->face_index, selected_candidate->armor,
                          selected_candidate->measured_yaw);
        rt_.face_switch_cooldown_remaining = face_switch_timeout_sec;
        candidate_debug.face_switch_cooldown_remaining =
            static_cast<float>(rt_.face_switch_cooldown_remaining);
      }
      else if (!strict_same_face_match && relaxed_same_face_match)
      {
        XR_LOG_DEBUG("Tracker relaxed same-face match: pos_diff=%.3f yaw_diff=%.3f",
                     selected_candidate->position_diff, selected_candidate->yaw_diff);
      }
      else if (switch_blocked_by_timeout && matched_switch_face)
      {
        XR_LOG_DEBUG(
            "Tracker hold same-face by timeout: cooldown=%.3f same_score=%.3f switch_score=%.3f",
            rt_.face_switch_cooldown_remaining, best_same_face_candidate.score,
            best_switch_candidate.score);
      }
      else if (!allow_face_switch && matched_switch_face && matched_same_face)
      {
        XR_LOG_DEBUG(
            "Tracker hold same-face by deadzone: same_score=%.3f switch_score=%.3f score_dz=%.3f pos_dz=%.3f yaw_dz=%.3f",
            best_same_face_candidate.score, best_switch_candidate.score,
            face_switch_score_deadzone, face_switch_position_deadzone,
            face_switch_yaw_deadzone);
      }

      matched = true;
      rt_.tracked_armor = selected_candidate->armor;
      rt_.tracked_id = selected_candidate->armor.number;
      const int tracked_face_track_before =
          rt_.tracked_face_track_id_valid ? static_cast<int>(rt_.tracked_face_track_id) : -1;
      if (did_face_switch)
      {
        const int face_count = std::max(1, std::min(4, static_cast<int>(rt_.tracked_armors_num)));
        std::array<bool, 4> rotated_valid{};
        std::array<uint16_t, 4> rotated_ids{};
        for (int face_slot = 0; face_slot < face_count; ++face_slot)
        {
          const int old_slot = (face_slot + selected_candidate->face_index) % face_count;
          rotated_valid[face_slot] = rt_.face_track_id_valid[old_slot];
          rotated_ids[face_slot] = rt_.face_track_id[old_slot];
        }
        rt_.face_track_id_valid = rotated_valid;
        rt_.face_track_id = rotated_ids;
      }
      if (rt_.face_track_id_valid[0])
      {
        rt_.tracked_face_track_id_valid = true;
        rt_.tracked_face_track_id = rt_.face_track_id[0];
      }
      else
      {
        rt_.tracked_face_track_id_valid = false;
      }
      if (selected_candidate->image_track_id >= 0 &&
          selected_candidate->confirmed_image_track)
      {
        rt_.tracked_face_track_id_valid = true;
        rt_.tracked_face_track_id =
            static_cast<uint16_t>(selected_candidate->image_track_id);
        rt_.face_track_id_valid[0] = true;
        rt_.face_track_id[0] = rt_.tracked_face_track_id;
      }
      if (did_face_switch || (selected_candidate->image_track_id >= 0 &&
                              selected_candidate->confirmed_image_track))
      {
        XR_LOG_DEBUG(
            "Tracker face bind: switch=%d sel_face=%d sel_image=%d confirmed=%d tracked_before=%d tracked_after=%d slots=[%d,%d,%d,%d] valid=[%d,%d,%d,%d]",
            did_face_switch ? 1 : 0, selected_candidate->face_index,
            selected_candidate->image_track_id,
            selected_candidate->confirmed_image_track ? 1 : 0,
            tracked_face_track_before,
            rt_.tracked_face_track_id_valid ? static_cast<int>(rt_.tracked_face_track_id) : -1,
            static_cast<int>(rt_.face_track_id[0]), static_cast<int>(rt_.face_track_id[1]),
            static_cast<int>(rt_.face_track_id[2]), static_cast<int>(rt_.face_track_id[3]),
            rt_.face_track_id_valid[0] ? 1 : 0, rt_.face_track_id_valid[1] ? 1 : 0,
            rt_.face_track_id_valid[2] ? 1 : 0, rt_.face_track_id_valid[3] ? 1 : 0);
      }
      rt_.last_yaw = selected_candidate->measured_yaw;
      auto p = selected_candidate->armor.pose.translation;
      ekf_.measurement =
          Eigen::Vector4d(p.x(), p.y(), p.z(), selected_candidate->measured_yaw);
      ekf_.state = ekf_.ekf.Update(ekf_.measurement);
      if (did_face_switch)
      {
        // Face relabeling is discrete geometry bookkeeping, not physical angular acceleration.
        ekf_.state(ExtendedKalmanFilter::V_YAW) = preserved_v_yaw;
        ekf_.ekf.SetState(ekf_.state);
      }
      UpdateDzReference(armors_msg, selected_candidate->armor);
      if (MultiArmorFuseEnabled())
      {
        FuseMultiArmorObservation(armors_msg);
      }
      XR_LOG_DEBUG("EKF update");
    }
    else
    {
      XR_LOG_DEBUG(
          "No matched armor found! same_number=%d best_face=%d score=%.3f pos_diff=%.3f yaw_diff=%.3f same_score=%.3f switch_score=%.3f cooldown=%.3f image_track=%d confirmed=%d persistent=%d img_diff=%.1f area_log=%.3f",
          has_same_number_candidate ? 1 : 0, best_candidate.face_index, best_candidate.score,
          best_candidate.position_diff, best_candidate.yaw_diff,
          best_same_face_candidate.score, best_switch_candidate.score,
          rt_.face_switch_cooldown_remaining, best_candidate.image_track_id,
          best_candidate.confirmed_image_track ? 1 : 0,
          best_candidate.same_persistent_track ? 1 : 0,
          best_candidate.image_center_diff, best_candidate.area_ratio_log);
    }
  }

  if (SymmetricGeometryEnabled())
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

  // 状态机
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
    }
    else
    {
      rt_.detect_count = 0;
      rt_.state = State::LOST;
    }
  }
  else if (rt_.state == State::TRACKING)
  {
    if (!matched)
    {
      rt_.state = State::TEMP_LOST;
      rt_.lost_count++;
    }
  }
  else if (rt_.state == State::TEMP_LOST)
  {
    if (!matched)
    {
      rt_.lost_count++;
      if (rt_.lost_count > rt_.lost_thres)
      {
        rt_.lost_count = 0;
        rt_.state = State::LOST;
      }
    }
    else
    {
      rt_.state = State::TRACKING;
      rt_.lost_count = 0;
    }
  }

  candidate_debug_msg_ = candidate_debug;
  candidate_debug_msg_.tracked_face_track_id_valid =
      rt_.tracked_face_track_id_valid ? 1 : 0;
  candidate_debug_msg_.tracked_face_track_id =
      rt_.tracked_face_track_id_valid ? static_cast<int16_t>(rt_.tracked_face_track_id)
                                      : static_cast<int16_t>(-1);
}

void ArmorTracker::VelocityCallback(double velocity_msg)
{
  io_.solver->Init(velocity_msg);
}

void ArmorTracker::ArmorsCallback(ArmorDetectorResults armors_msg,
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

void ArmorTracker::InitEKF(const ArmorDetectorResult& a)
{
  double xa = a.pose.translation.x();
  double ya = a.pose.translation.y();
  double za = a.pose.translation.z();
  rt_.last_yaw = 0;
  double yaw = OrientationToYaw(a.pose.rotation);

  // 初始在目标后方 r=0.26 m
  ekf_.state = Eigen::VectorXd::Zero(9);
  double r = 0.26;
  double xc = xa + r * std::cos(yaw);
  double yc = ya + r * std::sin(yaw);
  rt_.dz = 0;
  rt_.dz_abs_ref = 0.0;
  rt_.another_r = r;
  rt_.face_switch_cooldown_remaining = 0.0;
  ekf_.state << xc, 0, yc, 0, za, 0, yaw, 0, r;

  ekf_.ekf.SetState(ekf_.state);
}

void ArmorTracker::UpdateArmorsNum(const ArmorDetectorResult&)
{
  if (SingleArmorModeEnabled())
  {
    rt_.tracked_armors_num = static_cast<ArmorsNum>(1);
    return;
  }
  if (rt_.tracked_id == ArmorNumber::OUTPOST)
  {
    rt_.tracked_armors_num = ArmorsNum::OUTPOST_3;
  }
  else
  {
    rt_.tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}

void ArmorTracker::UpdateDzReference(const ArmorDetectorResults& armors_msg,
                                     const ArmorDetectorResult& anchor)
{
  if (SymmetricGeometryEnabled() || rt_.tracked_armors_num != ArmorsNum::NORMAL_4)
  {
    return;
  }

  double min_z = DBL_MAX;
  double max_z = -DBL_MAX;
  int count = 0;
  for (const auto& armor : armors_msg)
  {
    if (anchor.number != ArmorNumber::INVALID && armor.number != anchor.number)
    {
      continue;
    }
    if (anchor.type != ArmorType::INVALID && armor.type != anchor.type)
    {
      continue;
    }
    const double z = armor.pose.translation.z();
    if (!std::isfinite(z))
    {
      continue;
    }
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
    ++count;
  }

  if (count < 2)
  {
    return;
  }

  const double measured_dz_abs = max_z - min_z;
  if (!(measured_dz_abs > 0.02) || !(measured_dz_abs < 0.20))
  {
    return;
  }

  if (rt_.dz_abs_ref <= 1e-6)
  {
    rt_.dz_abs_ref = measured_dz_abs;
  }
  else
  {
    rt_.dz_abs_ref = 0.8 * rt_.dz_abs_ref + 0.2 * measured_dz_abs;
  }

  const double z_mid = 0.5 * (min_z + max_z);
  const double anchor_z = anchor.pose.translation.z();
  constexpr double kMidTol = 0.005;
  if (anchor_z > z_mid + kMidTol)
  {
    rt_.dz = -rt_.dz_abs_ref;
  }
  else if (anchor_z < z_mid - kMidTol)
  {
    rt_.dz = rt_.dz_abs_ref;
  }
  else if (std::abs(rt_.dz) > 1e-6)
  {
    rt_.dz = std::copysign(rt_.dz_abs_ref, rt_.dz);
  }
}

void ArmorTracker::FuseMultiArmorObservation(const ArmorDetectorResults& armors_msg)
{
  if (rt_.tracked_armors_num != ArmorsNum::NORMAL_4 || armors_msg.size() < 2 ||
      rt_.tracked_id == ArmorNumber::INVALID)
  {
    return;
  }

  struct FaceObservation
  {
    bool valid = false;
    ArmorDetectorResult armor{};
    double measured_yaw = 0.0;
    double position_diff = DBL_MAX;
    int image_track_id = -1;
    bool confirmed_image_track = false;
  };

  struct FuseCandidate
  {
    std::size_t armor_index = 0;
    int face_index = -1;
    ArmorDetectorResult armor{};
    double measured_yaw = 0.0;
    double position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;
    int image_track_id = -1;
    bool confirmed_image_track = false;
    bool same_persistent_track = false;
  };

  std::array<FaceObservation, 4> faces{};
  bool have_even_face = false;
  bool have_odd_face = false;
  const double max_position_diff = cfg_.match.max_match_distance * 2.5;
  const double max_yaw_diff = std::max(cfg_.match.max_match_yaw_diff, 1.2);
  std::vector<FuseCandidate> candidates;

  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    if (armor.number != rt_.tracked_id)
    {
      continue;
    }
    if (rt_.tracked_armor.type != ArmorType::INVALID && armor.type != rt_.tracked_armor.type)
    {
      continue;
    }

    const auto p = armor.pose.translation;
    const Eigen::Vector3d position_vec(p.x(), p.y(), p.z());
    const int image_track_id = FindDetectionTrackId(armor_index);
    const bool confirmed_image_track = IsDetectionTrackConfirmed(armor_index);
    const bool same_persistent_track =
        rt_.tracked_face_track_id_valid && confirmed_image_track && image_track_id >= 0 &&
        static_cast<uint16_t>(image_track_id) == rt_.tracked_face_track_id;
    int bound_face_index = -1;
    if (confirmed_image_track && image_track_id >= 0)
    {
      for (int face_slot = 0; face_slot < 4; ++face_slot)
      {
        if (rt_.face_track_id_valid[face_slot] &&
            rt_.face_track_id[face_slot] == static_cast<uint16_t>(image_track_id))
        {
          bound_face_index = face_slot;
          break;
        }
      }
    }
    for (int face_index = 0; face_index < 4; ++face_index)
    {
      if (confirmed_image_track)
      {
        if (bound_face_index >= 0 && face_index != bound_face_index)
        {
          continue;
        }
        if (bound_face_index < 0 && rt_.face_track_id_valid[face_index])
        {
          continue;
        }
      }
      const Eigen::Vector3d predicted_position =
          GetArmorPositionFromState(ekf_.state, face_index);
      const double predicted_yaw = GetArmorYawFromState(ekf_.state, face_index);
      const double measured_yaw = OrientationToYawNear(armor.pose.rotation, predicted_yaw);
      const double position_diff = (predicted_position - position_vec).norm();
      const double yaw_diff = AngularDiffAbs(measured_yaw, predicted_yaw);
      LogImpossibleYawDiff("fuse", armor_index, face_index, measured_yaw, predicted_yaw, yaw_diff);
      if (position_diff >= max_position_diff || yaw_diff >= max_yaw_diff)
      {
        continue;
      }
      if (!SymmetricGeometryEnabled() && face_index % 2 == 1 && rt_.dz_abs_ref > 0.02)
      {
        const double measured_dz_abs = std::abs(ekf_.state(4) - position_vec.z());
        constexpr double kDzConsistencyTol = 0.03;
        if (std::abs(measured_dz_abs - rt_.dz_abs_ref) >= kDzConsistencyTol)
        {
          continue;
        }
      }
      candidates.push_back({armor_index, face_index, armor, measured_yaw, position_diff,
                            yaw_diff, image_track_id, confirmed_image_track,
                            same_persistent_track});
    }
  }

  if (candidates.size() < 2)
  {
    return;
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const FuseCandidate& lhs, const FuseCandidate& rhs)
            {
              if (lhs.same_persistent_track != rhs.same_persistent_track)
              {
                return lhs.same_persistent_track > rhs.same_persistent_track;
              }
              if (lhs.confirmed_image_track != rhs.confirmed_image_track &&
                  std::abs(lhs.position_diff - rhs.position_diff) < 0.03)
              {
                return lhs.confirmed_image_track > rhs.confirmed_image_track;
              }
              if (std::abs(lhs.position_diff - rhs.position_diff) > 1e-6)
              {
                return lhs.position_diff < rhs.position_diff;
              }
              return lhs.yaw_diff < rhs.yaw_diff;
            });

  std::vector<bool> armor_used(armors_msg.size(), false);
  std::array<bool, 4> face_used{};
  std::vector<int> used_confirmed_image_track_ids;
  int valid_face_count = 0;
  for (const auto& candidate : candidates)
  {
    if (candidate.face_index < 0 || candidate.face_index >= 4)
    {
      continue;
    }
    if (armor_used[candidate.armor_index] || face_used[candidate.face_index])
    {
      continue;
    }
    if (candidate.confirmed_image_track && candidate.image_track_id >= 0 &&
        std::find(used_confirmed_image_track_ids.begin(),
                  used_confirmed_image_track_ids.end(),
                  candidate.image_track_id) != used_confirmed_image_track_ids.end())
    {
      continue;
    }
    armor_used[candidate.armor_index] = true;
    face_used[candidate.face_index] = true;
    if (candidate.confirmed_image_track && candidate.image_track_id >= 0)
    {
      used_confirmed_image_track_ids.push_back(candidate.image_track_id);
    }
    faces[candidate.face_index].valid = true;
    faces[candidate.face_index].armor = candidate.armor;
    faces[candidate.face_index].measured_yaw = candidate.measured_yaw;
    faces[candidate.face_index].position_diff = candidate.position_diff;
    faces[candidate.face_index].image_track_id = candidate.image_track_id;
    faces[candidate.face_index].confirmed_image_track = candidate.confirmed_image_track;
    ++valid_face_count;
  }

  if (valid_face_count < 2)
  {
    return;
  }

  for (int face_index = 0; face_index < 4; ++face_index)
  {
    if (!faces[face_index].valid || !faces[face_index].confirmed_image_track ||
        faces[face_index].image_track_id < 0)
    {
      continue;
    }
    rt_.face_track_id_valid[face_index] = true;
    rt_.face_track_id[face_index] =
        static_cast<uint16_t>(faces[face_index].image_track_id);
  }
  if (rt_.face_track_id_valid[0])
  {
    rt_.tracked_face_track_id_valid = true;
    rt_.tracked_face_track_id = rt_.face_track_id[0];
  }

  int even_col = -1;
  int odd_col = -1;
  int cols = 2;
  for (int face_index = 0; face_index < 4; ++face_index)
  {
    if (!faces[face_index].valid)
    {
      continue;
    }
    if (face_index % 2 == 0)
    {
      have_even_face = true;
    }
    else
    {
      have_odd_face = true;
    }
  }
  if (have_even_face)
  {
    even_col = cols++;
  }
  if (have_odd_face)
  {
    odd_col = cols++;
  }

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(valid_face_count * 2, cols);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(valid_face_count * 2);
  int row = 0;
  for (int face_index = 0; face_index < 4; ++face_index)
  {
    if (!faces[face_index].valid)
    {
      continue;
    }
    const int radius_col = (face_index % 2 == 0) ? even_col : odd_col;
    if (radius_col < 0)
    {
      continue;
    }

    const auto p = faces[face_index].armor.pose.translation;
    const double yaw = faces[face_index].measured_yaw;
    A(row, 0) = 1.0;
    A(row, radius_col) = -std::cos(yaw);
    b(row) = p.x();
    ++row;
    A(row, 1) = 1.0;
    A(row, radius_col) = -std::sin(yaw);
    b(row) = p.y();
    ++row;
  }
  if (row < 4)
  {
    return;
  }
  const int fit_rows = row;
  if (valid_face_count == 2)
  {
    constexpr double kCenterPriorWeight = 0.35;
    constexpr double kRadiusPriorWeight = 0.50;
    const int prior_rows =
        2 + (have_even_face ? 1 : 0) + (have_odd_face ? 1 : 0);
    A.conservativeResize(row + prior_rows, cols);
    b.conservativeResize(row + prior_rows);

    A.row(row).setZero();
    A(row, 0) = kCenterPriorWeight;
    b(row) = kCenterPriorWeight * ekf_.state(0);
    ++row;

    A.row(row).setZero();
    A(row, 1) = kCenterPriorWeight;
    b(row) = kCenterPriorWeight * ekf_.state(2);
    ++row;

    if (have_even_face)
    {
      A.row(row).setZero();
      A(row, even_col) = kRadiusPriorWeight;
      b(row) = kRadiusPriorWeight * ekf_.state(8);
      ++row;
    }
    if (have_odd_face)
    {
      A.row(row).setZero();
      A(row, odd_col) = kRadiusPriorWeight;
      b(row) = kRadiusPriorWeight * rt_.another_r;
      ++row;
    }
  }
  else
  {
    A.conservativeResize(row, cols);
    b.conservativeResize(row);
  }

  const Eigen::VectorXd sol =
      Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeThinU | Eigen::ComputeThinV)
          .solve(b);
  const Eigen::VectorXd residual =
      A.topRows(fit_rows) * sol - b.head(fit_rows);
  const double rmse = std::sqrt(residual.squaredNorm() / std::max(1, fit_rows));
  if (!std::isfinite(rmse) || rmse > 0.05)
  {
    return;
  }

  const double fused_x = sol(0);
  const double fused_y = sol(1);
  const double fused_r_even = have_even_face ? sol(even_col) : ekf_.state(8);
  const double fused_r_odd = have_odd_face ? sol(odd_col) : rt_.another_r;
  if (!std::isfinite(fused_x) || !std::isfinite(fused_y) ||
      !std::isfinite(fused_r_even) || !std::isfinite(fused_r_odd) ||
      fused_r_even < 0.05 || fused_r_even > 0.45 || fused_r_odd < 0.05 ||
      fused_r_odd > 0.45)
  {
    return;
  }

  const double alpha = valid_face_count >= 3 ? 0.35 : 0.12;
  ekf_.state(0) = (1.0 - alpha) * ekf_.state(0) + alpha * fused_x;
  ekf_.state(2) = (1.0 - alpha) * ekf_.state(2) + alpha * fused_y;
  if (have_even_face)
  {
    ekf_.state(8) = (1.0 - alpha) * ekf_.state(8) + alpha * fused_r_even;
  }
  if (have_odd_face)
  {
    rt_.another_r = (1.0 - alpha) * rt_.another_r + alpha * fused_r_odd;
  }
  ekf_.ekf.SetState(ekf_.state);

  XR_LOG_DEBUG(
      "Tracker multi-armor fuse: faces=%d rmse=%.3f center=(%.3f, %.3f) r1=%.3f r2=%.3f",
      valid_face_count, rmse, ekf_.state(0), ekf_.state(2), ekf_.state(8), rt_.another_r);
}

void ArmorTracker::SwitchTrackedFace(int face_index,
                                     const ArmorDetectorResult& current_armor,
                                     double measured_yaw)
{
  if (face_index == 0)
  {
    return;
  }

  // Reuse the candidate yaw already unwrapped near the predicted face.
  const double yaw = measured_yaw;
  rt_.last_yaw = measured_yaw;
  ekf_.state(6) = yaw;
  UpdateArmorsNum(current_armor);

  if (rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    if (SymmetricGeometryEnabled())
    {
      ekf_.state(4) = current_armor.pose.translation.z();
      rt_.another_r = ekf_.state(8);
      rt_.dz = 0.0;
      rt_.dz_abs_ref = 0.0;
    }
    else if (face_index % 2 == 1)
    {
      const double measured_dz = ekf_.state(4) - current_armor.pose.translation.z();
      const double measured_dz_abs = std::abs(measured_dz);
      if (measured_dz_abs > 0.02 && measured_dz_abs < 0.20)
      {
        if (rt_.dz_abs_ref <= 1e-6)
        {
          rt_.dz_abs_ref = measured_dz_abs;
        }
        else if (std::abs(measured_dz_abs - rt_.dz_abs_ref) < 0.03)
        {
          rt_.dz_abs_ref = 0.8 * rt_.dz_abs_ref + 0.2 * measured_dz_abs;
        }
      }
      rt_.dz = (rt_.dz_abs_ref > 0.02) ? std::copysign(rt_.dz_abs_ref, measured_dz)
                                       : measured_dz;
      ekf_.state(4) = current_armor.pose.translation.z();
      std::swap(ekf_.state(8), rt_.another_r);
    }
    else
    {
      ekf_.state(4) = current_armor.pose.translation.z();
      if (rt_.dz_abs_ref > 0.02 && std::abs(rt_.dz) > 1e-6)
      {
        rt_.dz = std::copysign(rt_.dz_abs_ref, rt_.dz);
      }
    }
  }
  else
  {
    ekf_.state(4) = current_armor.pose.translation.z();
  }

  auto p = current_armor.pose.translation;
  Eigen::Vector3d current_p(p.x(), p.y(), p.z());
  Eigen::Vector3d infer_p = GetArmorPositionFromState(ekf_.state, 0);
  if (FaceSwitchRecenterEnabled())
  {
    const double recenter_error = (current_p - infer_p).norm();
    const bool large_recenter_error = recenter_error > cfg_.match.max_match_distance;
    const double r = ekf_.state(8);
    ekf_.state(0) = p.x() + r * std::cos(yaw);  // xc
    ekf_.state(2) = p.y() + r * std::sin(yaw);  // yc
    ekf_.state(4) = p.z();                      // za
    if (large_recenter_error)
    {
      ekf_.state(1) = 0;
      ekf_.state(3) = 0;
      ekf_.state(5) = 0;
    }
    XR_LOG_DEBUG("Tracker face switch recentered state: err=%.3f reset_vel=%d",
                 recenter_error, large_recenter_error ? 1 : 0);
  }

  ekf_.ekf.SetState(ekf_.state);
}

void ArmorTracker::HandleArmorJump(const ArmorDetectorResult& current_armor,
                                   double measured_yaw)
{
  const double yaw = measured_yaw;
  rt_.last_yaw = measured_yaw;
  ekf_.state(6) = yaw;
  UpdateArmorsNum(current_armor);

  if (rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    if (SymmetricGeometryEnabled())
    {
      ekf_.state(4) = current_armor.pose.translation.z();
      rt_.another_r = ekf_.state(8);
      rt_.dz = 0.0;
      rt_.dz_abs_ref = 0.0;
    }
    else
    {
      const double measured_dz = ekf_.state(4) - current_armor.pose.translation.z();
      const double measured_dz_abs = std::abs(measured_dz);
      if (measured_dz_abs > 0.02 && measured_dz_abs < 0.20)
      {
        if (rt_.dz_abs_ref <= 1e-6)
        {
          rt_.dz_abs_ref = measured_dz_abs;
        }
        else if (std::abs(measured_dz_abs - rt_.dz_abs_ref) < 0.03)
        {
          rt_.dz_abs_ref = 0.8 * rt_.dz_abs_ref + 0.2 * measured_dz_abs;
        }
      }
      rt_.dz = (rt_.dz_abs_ref > 0.02) ? std::copysign(rt_.dz_abs_ref, measured_dz)
                                       : measured_dz;
      ekf_.state(4) = current_armor.pose.translation.z();
      std::swap(ekf_.state(8), rt_.another_r);
    }
  }
  XR_LOG_WARN("Armor jump!");

  // 大偏差则重置中心位置
  auto p = current_armor.pose.translation;
  Eigen::Vector3d current_p(p.x(), p.y(), p.z());
  Eigen::Vector3d infer_p = GetArmorPositionFromState(ekf_.state);
  if ((current_p - infer_p).norm() > cfg_.match.max_match_distance)
  {
    double r = ekf_.state(8);
    ekf_.state(0) = p.x() + r * std::cos(yaw);  // xc
    ekf_.state(1) = 0;
    ekf_.state(2) = p.y() + r * std::sin(yaw);  // yc
    ekf_.state(3) = 0;
    ekf_.state(4) = p.z();  // za
    ekf_.state(5) = 0;
    XR_LOG_ERROR("Reset State!");
  }

  ekf_.ekf.SetState(ekf_.state);
}

double ArmorTracker::OrientationToYaw(const LibXR::Quaternion<double>& q)
{
  auto yaw = OrientationToYawNear(q, rt_.last_yaw);
  rt_.last_yaw = yaw;
  return yaw;
}

double ArmorTracker::GetArmorYawFromState(const Eigen::VectorXd& x, int face_index)
{
  const int armor_count = std::max(1, static_cast<int>(rt_.tracked_armors_num));
  const double angle_step = 2.0 * M_PI / armor_count;
  return x(6) - angle_step * face_index;
}

Eigen::Vector3d ArmorTracker::GetArmorPositionFromState(const Eigen::VectorXd& x,
                                                        int face_index)
{
  const double xc = x(0), yc = x(2);
  double za = x(4);
  double r = x(8);
  const double yaw = GetArmorYawFromState(x, face_index);

  if (!SymmetricGeometryEnabled() && rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
      face_index % 2 == 1)
  {
    r = rt_.another_r;
    za = x(4) + rt_.dz;
  }

  const double xa = xc - r * std::cos(yaw);
  const double ya = yc - r * std::sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}

void ArmorTracker::SetConfig(const Config& cfg)
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

int ArmorTracker::CommandFun(ArmorTracker* self, int argc, char** argv)
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
