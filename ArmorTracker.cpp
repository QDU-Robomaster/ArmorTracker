#include "ArmorTracker.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <utility>

ArmorTracker::ArmorTracker(LibXR::HardwareContainer& /*hw*/,
                           LibXR::ApplicationManager& /*app*/, Config cfg)
    : cfg_(std::move(cfg))
{
  XR_LOG_INFO("Starting ArmorTracker!");

  // 轨迹解算器
  io_.solver = std::make_unique<SolveTrajectory>(cfg_.solver.k, cfg_.solver.bias_time,
                                                 cfg_.solver.s_bias, cfg_.solver.z_bias);

  // 初值（和老逻辑一致）
  rt_.tracking_thres = cfg_.thresholds.tracking_thres;
  io_.base_transform_static = cfg_.frames.base_transform_static;

  // ---------------- EKF 设置 ----------------
  // 状态 x = [xc, vxc, yc, vyc, za, vza, yaw, vyaw, r]
  // 观测 z = [xa, ya, za, yaw]
  auto f = [this](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd x_new = x;
    x_new(0) += x(1) * time_.dt;
    x_new(2) += x(3) * time_.dt;
    x_new(4) += x(5) * time_.dt;
    x_new(6) += x(7) * time_.dt;
    return x_new;
  };
  auto j_f = [this](const Eigen::VectorXd&)
  {
    Eigen::MatrixXd F(9, 9);
    // clang-format off
    F << 1, time_.dt, 0, 0, 0, 0, 0, 0, 0,
         0, 1,        0, 0, 0, 0, 0, 0, 0,
         0, 0,        1, time_.dt, 0, 0, 0, 0, 0,
         0, 0,        0, 1,        0, 0, 0, 0, 0,
         0, 0,        0, 0,        1, time_.dt, 0, 0, 0,
         0, 0,        0, 0,        0, 1,        0, 0, 0,
         0, 0,        0, 0,        0, 0,        1, time_.dt, 0,
         0, 0,        0, 0,        0, 0,        0, 1,        0,
         0, 0,        0, 0,        0, 0,        0, 0,        1;
    // clang-format on
    return F;
  };
  auto h = [](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd z(4);
    double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
    z(0) = xc - r * std::cos(yaw);  // xa
    z(1) = yc - r * std::sin(yaw);  // ya
    z(2) = x(4);                    // za
    z(3) = x(6);                    // yaw
    return z;
  };
  auto j_h = [](const Eigen::VectorXd& x)
  {
    Eigen::MatrixXd H(4, 9);
    double yaw = x(6), r = x(8);
    //                 xc vxc yc vyc za vza yaw               vyaw r
    H << /*xa */ 1, 0, 0, 0, 0, 0, r * std::sin(yaw), 0, -std::cos(yaw),
        /*ya */ 0, 0, 1, 0, 0, 0, -r * std::cos(yaw), 0, -std::sin(yaw),
        /*za */ 0, 0, 0, 0, 1, 0, 0, 0, 0,
        /*yaw*/ 0, 0, 0, 0, 0, 0, 1, 0, 0;
    return H;
  };
  auto u_q = [this]()
  {
    Eigen::MatrixXd Q(9, 9);
    double t = time_.dt, x = cfg_.ekf.sigma2_q_xyz, y = cfg_.ekf.sigma2_q_yaw,
           r = cfg_.ekf.sigma2_q_r;
    double q_x_x = std::pow(t, 4) / 4 * x, q_x_vx = std::pow(t, 3) / 2 * x,
           q_vx_vx = std::pow(t, 2) * x;
    double q_y_y = std::pow(t, 4) / 4 * y, q_y_vy = std::pow(t, 3) / 2 * x,
           q_vy_vy = std::pow(t, 2) * y;
    double q_r = std::pow(t, 4) / 4 * r;
    // clang-format off
    Q.setZero();
    Q(0,0)=q_x_x;  Q(0,1)=q_x_vx; Q(1,0)=q_x_vx; Q(1,1)=q_vx_vx;
    Q(2,2)=q_x_x;  Q(2,3)=q_x_vx; Q(3,2)=q_x_vx; Q(3,3)=q_vx_vx;
    Q(4,4)=q_x_x;  Q(4,5)=q_x_vx; Q(5,4)=q_x_vx; Q(5,5)=q_vx_vx;
    Q(6,6)=q_y_y;  Q(6,7)=q_y_vy; Q(7,6)=q_y_vy; Q(7,7)=q_vy_vy;
    Q(8,8)=q_r;
    // clang-format on
    return Q;
  };
  auto u_r = [this](const Eigen::VectorXd& z)
  {
    Eigen::DiagonalMatrix<double, 4> R;
    double x = cfg_.noise.r_xyz_factor;
    R.diagonal() << std::abs(x * z[0]), std::abs(x * z[1]), std::abs(x * z[2]),
        cfg_.noise.r_yaw;
    return R;
  };
  Eigen::DiagonalMatrix<double, 9> P0;
  P0.setIdentity();
  ekf_.ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, P0};

  // ---------------- Topics & 回调 ----------------
  // 装甲板订阅
  LibXR::Topic armors_topic = LibXR::Topic::Find("/detector/armors");
  auto armors_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto armors_msg = reinterpret_cast<ArmorDetectorResults*>(data.addr_);
        self->ArmorsCallback(*armors_msg);
      },
      this);
  armors_topic.RegisterCallback(armors_cb);

  // 速度订阅（用于弹道解算初始化）
  LibXR::Topic velocity_topic = LibXR::Topic::FindOrCreate<double>(
      "/current_velocity", nullptr, false, false, false);
  auto velocity_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto velocity_msg = reinterpret_cast<double*>(data.addr_);
        self->VelocityCallback(*velocity_msg);
      },
      this);
  velocity_topic.RegisterCallback(velocity_cb);

  // 机体基座旋转订阅（坐标变换）
  LibXR::Topic base_rotation_topic =
      LibXR::Topic::FindOrCreate<LibXR::Quaternion<double>>("/base_rotation");
  auto base_rotation_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        LibXR::Mutex::LockGuard lock(self->io_.base_rotation_lock);
        auto base_rotation_msg = reinterpret_cast<LibXR::Quaternion<double>*>(data.addr_);
        self->io_.base_rotation = *base_rotation_msg;
      },
      this);
  base_rotation_topic.RegisterCallback(base_rotation_cb);
}

void ArmorTracker::OnMonitor() {}

void ArmorTracker::Init(const ArmorDetectorResults& armors_msg)
{
  if (armors_msg.empty()) return;

  double min_distance = DBL_MAX;
  rt_.tracked_armor = armors_msg[0];
  for (const auto& armor : armors_msg)
  {
    if (armor.distance_to_image_center < min_distance)
    {
      min_distance = armor.distance_to_image_center;
      rt_.tracked_armor = armor;
    }
  }

  InitEKF(rt_.tracked_armor);
  XR_LOG_DEBUG("Init EKF!");

  rt_.tracked_id = rt_.tracked_armor.number;
  rt_.state = DETECTING;
  UpdateArmorsNum(rt_.tracked_armor);
}

void ArmorTracker::Update(const ArmorDetectorResults& armors_msg)
{
  Eigen::VectorXd ekf_prediction = ekf_.ekf.Predict();  // 预测
  XR_LOG_DEBUG("EKF predict");
  bool matched = false;
  ekf_.state = ekf_prediction;

  if (!armors_msg.empty())
  {
    ArmorDetectorResult same_id_armor{};
    int same_id_armors_count = 0;
    auto predicted_position = GetArmorPositionFromState(ekf_prediction);

    double min_position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;

    for (const auto& armor : armors_msg)
    {
      if (armor.number == rt_.tracked_id)
      {
        same_id_armor = armor;
        same_id_armors_count++;

        auto p = armor.pose.translation;
        Eigen::Vector3d position_vec(p.x(), p.y(), p.z());
        double position_diff = (predicted_position - position_vec).norm();

        if (position_diff < min_position_diff)
        {
          min_position_diff = position_diff;
          yaw_diff = std::abs(OrientationToYaw(armor.pose.rotation) - ekf_prediction(6));
          rt_.tracked_armor = armor;
        }
      }
    }

    // 存储信息
    rt_.info_position_diff = min_position_diff;
    rt_.info_yaw_diff = yaw_diff;

    // 判定匹配
    if (min_position_diff < cfg_.match.max_match_distance &&
        yaw_diff < cfg_.match.max_match_yaw_diff)
    {
      matched = true;
      auto p = rt_.tracked_armor.pose.translation;
      double measured_yaw = OrientationToYaw(rt_.tracked_armor.pose.rotation);
      ekf_.measurement = Eigen::Vector4d(p.x(), p.y(), p.z(), measured_yaw);
      ekf_.state = ekf_.ekf.Update(ekf_.measurement);
      XR_LOG_DEBUG("EKF update");
    }
    else if (same_id_armors_count == 1 && yaw_diff > cfg_.match.max_match_yaw_diff)
    {
      HandleArmorJump(same_id_armor);
    }
    else
    {
      XR_LOG_WARN("No matched armor found!");
    }
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
  if (rt_.state == DETECTING)
  {
    if (matched)
    {
      rt_.detect_count++;
      if (rt_.detect_count > rt_.tracking_thres)
      {
        rt_.detect_count = 0;
        rt_.state = TRACKING;
      }
    }
    else
    {
      rt_.detect_count = 0;
      rt_.state = LOST;
    }
  }
  else if (rt_.state == TRACKING)
  {
    if (!matched)
    {
      rt_.state = TEMP_LOST;
      rt_.lost_count++;
    }
  }
  else if (rt_.state == TEMP_LOST)
  {
    if (!matched)
    {
      rt_.lost_count++;
      if (rt_.lost_count > rt_.lost_thres)
      {
        rt_.lost_count = 0;
        rt_.state = LOST;
      }
    }
    else
    {
      rt_.state = TRACKING;
      rt_.lost_count = 0;
    }
  }
}

void ArmorTracker::VelocityCallback(double velocity_msg)
{
  io_.solver->Init(velocity_msg);
}

void ArmorTracker::ArmorsCallback(ArmorDetectorResults& armors_msg)
{
  XR_LOG_DEBUG("Got %d armors", static_cast<int>(armors_msg.size()));

  // 图像坐标 -> 世界坐标
  io_.base_rotation_lock.Lock();
  for (auto& armor : armors_msg)
  {
    LibXR::Transform<double> tf = armor.pose;
    armor.pose = LibXR::Transform<double>(io_.base_rotation, {0.0, 0.0, 0.0}) +
                 io_.base_transform_static + tf;
  }
  io_.base_rotation_lock.Unlock();

  // 过滤异常装甲
  armors_msg.erase(std::remove_if(armors_msg.begin(), armors_msg.end(),
                                  [this](const ArmorDetectorResult& armor)
                                  {
                                    return std::abs(armor.pose.translation.z()) > 1.2 ||
                                           Eigen::Vector2d(armor.pose.translation.x(),
                                                           armor.pose.translation.y())
                                                   .norm() >
                                               cfg_.limits.max_armor_distance;
                                  }),
                   armors_msg.end());

  // 构造消息
  TrackerInfo info_msg{};
  SolveTrajectory::Target target_msg{};
  Send send_msg{};
  target_msg.id = ArmorNumber::INVALID;

  auto time = LibXR::Timebase::GetMicroseconds();

  // 跟踪更新
  if (rt_.state == LOST)
  {
    Init(armors_msg);
    target_msg.tracking = false;
  }
  else
  {
    // dt
    time_.dt = (time - time_.last_time).ToSecond();
    if (time_.dt <= 0) time_.dt = 1.0 / 100.0;
    rt_.lost_thres = static_cast<int>(cfg_.thresholds.lost_time_thres / time_.dt);
    if (rt_.lost_thres < 1) rt_.lost_thres = 1;

    Update(armors_msg);

    // 发布 Info
    info_msg.position_diff = rt_.info_position_diff;
    info_msg.yaw_diff = rt_.info_yaw_diff;
    info_msg.position.x() = ekf_.measurement(0);
    info_msg.position.y() = ekf_.measurement(1);
    info_msg.position.z() = ekf_.measurement(2);
    info_msg.yaw = ekf_.measurement(3);
    io_.info_topic.Publish(info_msg);

    if (rt_.state == DETECTING)
    {
      target_msg.tracking = false;
    }
    else if (rt_.state == TRACKING || rt_.state == TEMP_LOST)
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

      float pitch = 0, yaw = 0, aim_x = 0, aim_y = 0, aim_z = 0;
      io_.solver->AutoSolveTrajectory(pitch, yaw, aim_x, aim_y, aim_z, &target_msg);

      io_.solver->SetFireCallback([&](bool is_fire) { send_msg.is_fire = is_fire; });
      send_msg.position.x() = aim_x;
      send_msg.position.y() = aim_y;
      send_msg.position.z() = aim_z;
      send_msg.v_yaw = target_msg.v_yaw;
      send_msg.pitch = pitch;
      send_msg.yaw = yaw;
    }
  }

  time_.last_time = time;

  io_.send_topic.Publish(send_msg);
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
  rt_.another_r = r;
  ekf_.state << xc, 0, yc, 0, za, 0, yaw, 0, r;

  ekf_.ekf.SetState(ekf_.state);
}

void ArmorTracker::UpdateArmorsNum(const ArmorDetectorResult&)
{
  if (rt_.tracked_id == ArmorNumber::OUTPOST)
    rt_.tracked_armors_num = ArmorsNum::OUTPOST_3;
  else
    rt_.tracked_armors_num = ArmorsNum::NORMAL_4;
}

void ArmorTracker::HandleArmorJump(const ArmorDetectorResult& current_armor)
{
  double yaw = OrientationToYaw(current_armor.pose.rotation);
  ekf_.state(6) = yaw;
  UpdateArmorsNum(current_armor);

  if (rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    rt_.dz = ekf_.state(4) - current_armor.pose.translation.z();
    ekf_.state(4) = current_armor.pose.translation.z();
    std::swap(ekf_.state(8), rt_.another_r);
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
  LibXR::EulerAngle<double> eulr =
      LibXR::RotationMatrix<double>(q.ToRotationMatrix()).ToEulerAngle();
  auto yaw = eulr.Yaw();
  const double DELTA =
      LibXR::CycleValue<double>(yaw) - LibXR::CycleValue<double>(rt_.last_yaw);
  yaw = rt_.last_yaw + DELTA;
  rt_.last_yaw = yaw;
  return yaw;
}

Eigen::Vector3d ArmorTracker::GetArmorPositionFromState(const Eigen::VectorXd& x)
{
  double xc = x(0), yc = x(2), za = x(4);
  double yaw = x(6), r = x(8);
  double xa = xc - r * std::cos(yaw);
  double ya = yc - r * std::sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}
