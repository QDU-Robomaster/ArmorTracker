#include "ArmorTracker.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

#include "cycle_value.hpp"
#include "logger.hpp"
#include "message.hpp"
#include "transform.hpp"

ArmorTracker::ArmorTracker(LibXR::HardwareContainer &,
                           LibXR::ApplicationManager &, Config cfg)
    : cfg_(std::move(cfg)) {
  XR_LOG_INFO("Starting ArmorTracker!");

  // 轨迹解算器
  io_.solver =
      std::make_unique<SolveTrajectory>(cfg_.solver.k, cfg_.solver.bias_time,
                                        cfg_.solver.s_bias, cfg_.solver.z_bias);

  // 初值（和老逻辑一致）
  rt_.tracking_thres = cfg_.thresholds.tracking_thres;
  io_.gimbal_to_camera_transform_static = cfg_.frames.base_transform_static;

  // ---------------- EKF 设置 ----------------
  // 状态 x = [xc, vxc, yc, vyc, za, vza, yaw, vyaw, r]
  // 观测 z = [xa, ya, za, yaw]
  auto f = [this](const Eigen::VectorXd &x) {
    Eigen::VectorXd x_new = x;
    x_new(0) += x(1) * time_.dt;
    x_new(2) += x(3) * time_.dt;
    x_new(4) += x(5) * time_.dt;
    x_new(6) += x(7) * time_.dt;
    return x_new;
  };
  auto j_f = [this](const Eigen::VectorXd &) {
    Eigen::MatrixXd f(9, 9);
    // clang-format off
    f << 1, time_.dt, 0, 0, 0, 0, 0, 0, 0,
         0, 1,        0, 0, 0, 0, 0, 0, 0,
         0, 0,        1, time_.dt, 0, 0, 0, 0, 0,
         0, 0,        0, 1,        0, 0, 0, 0, 0,
         0, 0,        0, 0,        1, time_.dt, 0, 0, 0,
         0, 0,        0, 0,        0, 1,        0, 0, 0,
         0, 0,        0, 0,        0, 0,        1, time_.dt, 0,
         0, 0,        0, 0,        0, 0,        0, 1,        0,
         0, 0,        0, 0,        0, 0,        0, 0,        1;
    // clang-format on
    return f;
  };
  auto h = [](const Eigen::VectorXd &x) {
    Eigen::VectorXd z(4);
    double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
    z(0) = xc - r * std::cos(yaw); // xa
    z(1) = yc - r * std::sin(yaw); // ya
    z(2) = x(4);                   // za
    z(3) = x(6);                   // yaw
    return z;
  };
  auto j_h = [](const Eigen::VectorXd &x) {
    Eigen::MatrixXd h(4, 9);
    double yaw = x(6), r = x(8);
    //                 xc vxc yc vyc za vza yaw               vyaw r
    h << /*xa */ 1, 0, 0, 0, 0, 0, r * std::sin(yaw), 0, -std::cos(yaw),
        /*ya */ 0, 0, 1, 0, 0, 0, -r * std::cos(yaw), 0, -std::sin(yaw),
        /*za */ 0, 0, 0, 0, 1, 0, 0, 0, 0,
        /*yaw*/ 0, 0, 0, 0, 0, 0, 1, 0, 0;
    return h;
  };
  auto u_q = [this]() {
    Eigen::MatrixXd q(9, 9);
    double t = time_.dt, x = cfg_.ekf.sigma2_q_xyz, y = cfg_.ekf.sigma2_q_yaw,
           r = cfg_.ekf.sigma2_q_r;
    double q_x_x = std::pow(t, 4) / 4 * x, q_x_vx = std::pow(t, 3) / 2 * x,
           q_vx_vx = std::pow(t, 2) * x;
    double q_y_y = std::pow(t, 4) / 4 * y, q_y_vy = std::pow(t, 3) / 2 * x,
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
  auto u_r = [this](const Eigen::VectorXd &z) {
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
  LibXR::Topic::Domain armor_detector_domain =
      LibXR::Topic::Domain("armor_detector");
  LibXR::Topic armors_topic =
      LibXR::Topic::Find("armors_result", &armor_detector_domain);
  auto armors_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker *self, LibXR::RawData &data) {
        auto armors_msg = reinterpret_cast<ArmorDetectorResults *>(data.addr_);
        self->ArmorsCallback(*armors_msg);
      },
      this);
  armors_topic.RegisterCallback(armors_cb);

  // 弹丸速度订阅（用于弹道解算初始化）
  LibXR::Topic::Domain referee_domain = LibXR::Topic::Domain("referee");
  LibXR::Topic bullet_speed_tp =
      LibXR::Topic::FindOrCreate<float>("bullet_speed", &referee_domain);
  auto velocity_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker *self, LibXR::RawData &data) {
        auto velocity_msg = reinterpret_cast<float *>(data.addr_);
        self->VelocityCallback(*velocity_msg);
      },
      this);
  bullet_speed_tp.RegisterCallback(velocity_cb);

  // 云台姿态订阅
  LibXR::Topic::Domain gimbal_domain = LibXR::Topic::Domain("gimbal");
  LibXR::Topic gimbal_rotation_topic =
      LibXR::Topic::FindOrCreate<LibXR::Quaternion<float>>("rotation",
                                                           &gimbal_domain);
  auto base_rotation_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker *self, LibXR::RawData &data) {
        LibXR::Mutex::LockGuard lock(self->io_.gimbal_rotation_lock);
        auto base_rotation_msg =
            reinterpret_cast<LibXR::Quaternion<float> *>(data.addr_);
        self->io_.gimbal_rotation = LibXR::Quaternion<double>(
            base_rotation_msg->w(), base_rotation_msg->x(),
            base_rotation_msg->y(), base_rotation_msg->z());
      },
      this);
  gimbal_rotation_topic.RegisterCallback(base_rotation_cb);

  io_.solver->SetFireCallback([&](bool is_fire) {
    XR_LOG_INFO("is_fire: {}", is_fire);
    // uint8_t fire_notify = is_fire ? 1 : 0;
    uint8_t fire_notify = 0;
    io_.fire_notify_topic.Publish(fire_notify);
  });

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE

  auto info_topic = LibXR::Topic(LibXR::Topic::Find("camera_info"));
  auto info_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker *self, LibXR::RawData &data) {
        auto *camera_info =
            reinterpret_cast<CameraBase::CameraInfo *>(data.addr_);
        static bool inited = false;
        if (!inited) {
          XR_LOG_PASS("Got camera info!");
          inited = true;

          self->cam_info_ =
              std::make_shared<CameraBase::CameraInfo>(*camera_info);
        }
      },
      this);
  info_topic.RegisterCallback(info_cb);

  auto img_topic = LibXR::Topic(LibXR::Topic::Find("image_raw"));
  auto img_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker *self, LibXR::RawData &data) {
        auto *img_msg = reinterpret_cast<cv::Mat *>(data.addr_);
        cv::Mat frame = img_msg->clone();

        EkfPointsMsg &ekf = self->ekf_msg_;

        // —— 用 info_cb 提供的内参/畸变；若还没拿到则直接显示原图 ——
        if (!self->cam_info_) {
          cv::imshow("ekf_overlay", frame);
          cv::waitKey(1);
          return;
        }
        const CameraBase::CameraInfo &cam = *self->cam_info_;

        // 只考虑 PLUMB_BOB；否则当作无畸变
        bool has_distortion =
            (cam.distortion_model == CameraBase::DistortionModel::PLUMB_BOB);

        // --- 构造 K(3x3) ---
        const auto &k_arr = cam.camera_matrix; // 行优先 3x3
        cv::Mat k =
            (cv::Mat_<double>(3, 3) << k_arr[0], k_arr[1], k_arr[2], k_arr[3],
             k_arr[4], k_arr[5], k_arr[6], k_arr[7], k_arr[8]);

        // --- 构造 D（PLUMB_BOB: k1,k2,p1,p2,k3）---
        cv::Mat d;
        if (has_distortion) {
          const auto &pb = cam.distortion_coefficients.plumb_bob;
          std::vector<double> dvec = {pb.k1, pb.k2, pb.p1, pb.p2, pb.k3};
          d = cv::Mat(dvec).clone().reshape(1, 1); // 1x5
        } else {
          d = cv::Mat(); // 空 -> 无畸变
        }

        // 若当前帧分辨率与标定分辨率不同，缩放 K；D 不缩放
        const double SX =
            static_cast<double>(frame.cols) / static_cast<double>(cam.width);
        const double SY =
            static_cast<double>(frame.rows) / static_cast<double>(cam.height);
        cv::Mat k_scaled = k.clone();
        k_scaled.at<double>(0, 0) *= SX; // fx
        k_scaled.at<double>(1, 1) *= SY; // fy
        k_scaled.at<double>(0, 2) *= SX; // cx
        k_scaled.at<double>(1, 2) *= SY; // cy

        auto project = [&](const Eigen::Vector3d &Pc, cv::Point2d &uv) -> bool {
          if (!(Pc.z() > 1e-6) || !std::isfinite(Pc.x()) ||
              !std::isfinite(Pc.y()) || !std::isfinite(Pc.z())) {
            return false;
          }

          std::vector<cv::Point3d> obj{cv::Point3d(Pc.x(), Pc.y(), Pc.z())};
          static cv::Mat rvec = cv::Mat::zeros(1, 3, CV_64F);
          static cv::Mat tvec = cv::Mat::zeros(1, 3, CV_64F);
          std::vector<cv::Point2d> imgpts;
          cv::projectPoints(obj, rvec, tvec, k_scaled, d, imgpts);
          uv = imgpts[0];
          return (0 <= uv.x && uv.x < frame.cols && 0 <= uv.y &&
                  uv.y < frame.rows);
        };

        if (ekf.valid[0]) {
          cv::Point2d uv;
          Eigen::Vector3d pc(ekf.center_cam.x(), ekf.center_cam.y(),
                             ekf.center_cam.z());
          if (project(pc, uv)) {
            cv::circle(frame, uv, 5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::putText(frame, "C", uv + cv::Point2d(6, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1,
                        cv::LINE_AA);
          }
        }

        for (int i = 0; i < std::min<int>(ekf.count, 4); ++i) {
          if (!ekf.valid[i + 1]) {
            continue;
          }
          cv::Point2d uv;
          Eigen::Vector3d pc(ekf.armors_cam[i].x(), ekf.armors_cam[i].y(),
                             ekf.armors_cam[i].z());
          if (project(pc, uv)) {
            cv::circle(frame, uv, 4, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
            char buf[16];
            (void)std::snprintf(buf, sizeof(buf), "A%d", i);
            cv::putText(frame, buf, uv + cv::Point2d(6, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0),
                        1, cv::LINE_AA);
          }
        }

        for (int i = 0; i < std::min<int>(ekf.count, 4); ++i) {
          if (!ekf.valid[0] || !ekf.valid[i + 1]) {
            continue;
          }
          cv::Point2d uc, ua;
          Eigen::Vector3d pc_c(ekf.center_cam.x(), ekf.center_cam.y(),
                               ekf.center_cam.z());
          Eigen::Vector3d pc_a(ekf.armors_cam[i].x(), ekf.armors_cam[i].y(),
                               ekf.armors_cam[i].z());
          if (project(pc_c, uc) && project(pc_a, ua)) {
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

void ArmorTracker::Init(const ArmorDetectorResults &armors_msg) {
  if (armors_msg.empty()) {
    return;
  }

  double min_distance = DBL_MAX;
  rt_.tracked_armor = armors_msg[0];
  for (const auto &armor : armors_msg) {
    if (armor.distance_to_image_center < min_distance) {
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

void ArmorTracker::Update(const ArmorDetectorResults &armors_msg) {
  Eigen::VectorXd ekf_prediction = ekf_.ekf.Predict(); // 预测
  XR_LOG_DEBUG("EKF predict");
  bool matched = false;
  ekf_.state = ekf_prediction;

  if (!armors_msg.empty()) {
    ArmorDetectorResult same_id_armor{};
    int same_id_armors_count = 0;
    auto predicted_position = GetArmorPositionFromState(ekf_prediction);

    double min_position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;

    for (const auto &armor : armors_msg) {
      if (armor.number == rt_.tracked_id) {
        same_id_armor = armor;
        same_id_armors_count++;

        auto p = armor.pose.translation;
        Eigen::Vector3d position_vec(p.x(), p.y(), p.z());
        double position_diff = (predicted_position - position_vec).norm();

        if (position_diff < min_position_diff) {
          min_position_diff = position_diff;
          yaw_diff = std::abs(OrientationToYaw(armor.pose.rotation) -
                              ekf_prediction(6));
          rt_.tracked_armor = armor;
        }
      }
    }

    // 存储信息
    rt_.info_position_diff = min_position_diff;
    rt_.info_yaw_diff = yaw_diff;

    // 判定匹配
    if (min_position_diff < cfg_.match.max_match_distance &&
        yaw_diff < cfg_.match.max_match_yaw_diff) {
      matched = true;
      auto p = rt_.tracked_armor.pose.translation;
      double measured_yaw = OrientationToYaw(rt_.tracked_armor.pose.rotation);
      ekf_.measurement = Eigen::Vector4d(p.x(), p.y(), p.z(), measured_yaw);
      ekf_.state = ekf_.ekf.Update(ekf_.measurement);
      XR_LOG_DEBUG("EKF update");
    } else if (same_id_armors_count == 1 &&
               yaw_diff > cfg_.match.max_match_yaw_diff) {
      HandleArmorJump(same_id_armor);
    } else {
      XR_LOG_INFO("No matched armor found!");
    }
  }

  // 防止半径发散
  if (ekf_.state(8) < 0.12) {
    ekf_.state(8) = 0.12;
    ekf_.ekf.SetState(ekf_.state);
  } else if (ekf_.state(8) > 0.4) {
    ekf_.state(8) = 0.4;
    ekf_.ekf.SetState(ekf_.state);
  }

  // 状态机
  if (rt_.state == DETECTING) {
    if (matched) {
      rt_.detect_count++;
      if (rt_.detect_count > rt_.tracking_thres) {
        rt_.detect_count = 0;
        rt_.state = TRACKING;
      }
    } else {
      rt_.detect_count = 0;
      rt_.state = LOST;
    }
  } else if (rt_.state == TRACKING) {
    if (!matched) {
      rt_.state = TEMP_LOST;
      rt_.lost_count++;
    }
  } else if (rt_.state == TEMP_LOST) {
    if (!matched) {
      rt_.lost_count++;
      if (rt_.lost_count > rt_.lost_thres) {
        rt_.lost_count = 0;
        rt_.state = LOST;
      }
    } else {
      rt_.state = TRACKING;
      rt_.lost_count = 0;
    }
  }
}

void ArmorTracker::VelocityCallback(double velocity_msg) {
  io_.solver->Init(velocity_msg);
}

void ArmorTracker::ArmorsCallback(ArmorDetectorResults &armors_msg) {
  if (!armors_msg.empty()) {
    XR_LOG_INFO("Got %d armors", static_cast<int>(armors_msg.size()));
  }

  // 图像坐标 -> 世界坐标
  // gimbal +X  = camera +Z
  // gimbal +Y  = camera -X
  // gimbal +Z  = camera -Y
  io_.gimbal_rotation_lock.Lock();
  for (auto &armor : armors_msg) {
    LibXR::Transform<double> tf = armor.pose;
    armor.pose =
        LibXR::Transform<double>(io_.gimbal_rotation, {0.0, 0.0, 0.0}) +
        io_.gimbal_to_camera_transform_static + tf;
  }
  io_.gimbal_rotation_lock.Unlock();

  // 过滤异常装甲
  armors_msg.erase(
      std::remove_if(armors_msg.begin(), armors_msg.end(),
                     [this](const ArmorDetectorResult &armor) {
                       return std::abs(armor.pose.translation.z()) >
                                  cfg_.limits.max_z_position ||
                              Eigen::Vector2d(armor.pose.translation.x(),
                                              armor.pose.translation.y())
                                      .norm() > cfg_.limits.max_armor_distance;
                     }),
      armors_msg.end());

  // 构造消息
  TrackerInfo info_msg{};
  SolveTrajectory::Target target_msg{};
  LibXR::EulerAngle<float> target_eulr;
  Send send_msg{};
  target_msg.id = ArmorNumber::INVALID;

  auto time = LibXR::Timebase::GetMicroseconds();

  // 跟踪更新
  if (rt_.state == LOST) {
    Init(armors_msg);
    target_msg.tracking = false;
  } else {
    // dt
    time_.dt = (time - time_.last_time).ToSecond();
    if (time_.dt <= 0) {
      time_.dt = 1.0 / 100.0;
    }
    rt_.lost_thres =
        static_cast<int>(cfg_.thresholds.lost_time_thres / time_.dt);
    if (rt_.lost_thres < 1) {
      rt_.lost_thres = 1;
    }

    Update(armors_msg);

    // 发布 Info
    info_msg.position_diff = rt_.info_position_diff;
    info_msg.yaw_diff = rt_.info_yaw_diff;
    info_msg.position.x() = ekf_.measurement(0);
    info_msg.position.y() = ekf_.measurement(1);
    info_msg.position.z() = ekf_.measurement(2);
    info_msg.yaw = ekf_.measurement(3);
    io_.info_topic.Publish(info_msg);

    if (rt_.state == DETECTING) {
      target_msg.tracking = false;
    } else if (rt_.state == TRACKING || rt_.state == TEMP_LOST) {
      target_msg.tracking = true;
      const auto &state = ekf_.state;
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

      XR_LOG_INFO("Target position: (%.3f, %.3f, %.3f) velocity: (%.3f, %.3f, "
                  "%.3f) yaw: %.3f "
                  "v_yaw: %.3f radius_1: %.3f radius_2: %.3f dz: %.3f",
                  target_msg.position.x(), target_msg.position.y(),
                  target_msg.position.z(), target_msg.velocity.x(),
                  target_msg.velocity.y(), target_msg.velocity.z(),
                  target_msg.yaw, target_msg.v_yaw, target_msg.radius_1,
                  target_msg.radius_2, target_msg.dz);

      float pitch = 0, yaw = 0, aim_x = 0, aim_y = 0, aim_z = 0;
      io_.solver->AutoSolveTrajectory(pitch, yaw, aim_x, aim_y, aim_z,
                                      &target_msg);

      XR_LOG_INFO("AutoSolveTrajectory: pitch: %.3f yaw: %.3f aim_x: %.3f "
                  "aim_y: %.3f aim_z: "
                  "%.3f",
                  pitch, yaw, aim_x, aim_y, aim_z);

      target_eulr.Pitch() = pitch;
      target_eulr.Yaw() = yaw;

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
      Eigen::Vector3d pw_center, pw_armors[4];
      {
        const auto &st = ekf_.state; // [xc,vxc,yc,vyc,za,vza,yaw,vyaw,r1]
        const double XC = st(0), YC = st(2), ZA = st(4);
        const double YAW = st(6);
        const double R1 = st(8);
        const double R2 = rt_.another_r; // 另一半径（交替用）
        const int N = static_cast<int>(rt_.tracked_armors_num);

        pw_center = {XC, YC, ZA};
        for (int i = 0; i < 4; ++i) {
          const double THETA = YAW + (2.0 * M_PI / std::max(1, N)) * i;
          const double R = (i % 2 == 0 ? R1 : R2);
          const double XA = XC - R * std::cos(THETA);
          const double YA = YC - R * std::sin(THETA);
          pw_armors[i] = {XA, YA, ZA};
        }
      }

      // === 计算 相机←世界 外参：T_CW = (T_WG ⊕ T_GC)^-1 ===
      LibXR::Transform<double> t_wg(io_.gimbal_rotation,
                                    {0.0, 0.0, 0.0}); // 世界←云台
      LibXR::Transform<double> t_wc =
          t_wg + io_.gimbal_to_camera_transform_static; // 世界←相机
      auto r_wc = t_wc.rotation.ToRotationMatrix();
      Eigen::Matrix3d r_cw = r_wc.transpose(); // 相机←世界 旋转
      Eigen::Vector3d twc(t_wc.translation.x(), t_wc.translation.y(),
                          t_wc.translation.z());

      // === 变到相机系并发布 ===
      ekf_msg_.count = static_cast<uint8_t>(rt_.tracked_armors_num);

      auto to_cam = [&](const Eigen::Vector3d &pw,
                        LibXR::Position<double> &out_pt) -> bool {
        Eigen::Vector3d pc = r_cw * (pw - twc);
        out_pt = LibXR::Position<double>{pc.x(), pc.y(), pc.z()};
        return pc.z() > 1e-6; // 在相机前方才算可见
      };

      // center
      ekf_msg_.valid[0] = to_cam(pw_center, ekf_msg_.center_cam);

      // armors
      for (int i = 0; i < 4; ++i) {
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

  io_.target_eulr_topic.Publish(target_eulr);
  io_.target_topic.Publish(target_msg);
}

void ArmorTracker::InitEKF(const ArmorDetectorResult &a) {
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

void ArmorTracker::UpdateArmorsNum(const ArmorDetectorResult &) {
  if (rt_.tracked_id == ArmorNumber::OUTPOST) {
    rt_.tracked_armors_num = ArmorsNum::OUTPOST_3;
  } else {
    rt_.tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}

void ArmorTracker::HandleArmorJump(const ArmorDetectorResult &current_armor) {
  double yaw = OrientationToYaw(current_armor.pose.rotation);
  ekf_.state(6) = yaw;
  UpdateArmorsNum(current_armor);

  if (rt_.tracked_armors_num == ArmorsNum::NORMAL_4) {
    rt_.dz = ekf_.state(4) - current_armor.pose.translation.z();
    ekf_.state(4) = current_armor.pose.translation.z();
    std::swap(ekf_.state(8), rt_.another_r);
  }
  XR_LOG_WARN("Armor jump!");

  // 大偏差则重置中心位置
  auto p = current_armor.pose.translation;
  Eigen::Vector3d current_p(p.x(), p.y(), p.z());
  Eigen::Vector3d infer_p = GetArmorPositionFromState(ekf_.state);
  if ((current_p - infer_p).norm() > cfg_.match.max_match_distance) {
    double r = ekf_.state(8);
    ekf_.state(0) = p.x() + r * std::cos(yaw); // xc
    ekf_.state(1) = 0;
    ekf_.state(2) = p.y() + r * std::sin(yaw); // yc
    ekf_.state(3) = 0;
    ekf_.state(4) = p.z(); // za
    ekf_.state(5) = 0;
    XR_LOG_ERROR("Reset State!");
  }

  ekf_.ekf.SetState(ekf_.state);
}

double ArmorTracker::OrientationToYaw(const LibXR::Quaternion<double> &q) {
  LibXR::EulerAngle<double> eulr =
      LibXR::RotationMatrix<double>(q.ToRotationMatrix()).ToEulerAngle();
  auto yaw = eulr.Yaw();
  const double DELTA =
      LibXR::CycleValue<double>(yaw) - LibXR::CycleValue<double>(rt_.last_yaw);
  yaw = rt_.last_yaw + DELTA;
  rt_.last_yaw = yaw;
  return yaw;
}

Eigen::Vector3d
ArmorTracker::GetArmorPositionFromState(const Eigen::VectorXd &x) {
  double xc = x(0), yc = x(2), za = x(4);
  double yaw = x(6), r = x(8);
  double xa = xc - r * std::cos(yaw);
  double ya = yc - r * std::sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}
