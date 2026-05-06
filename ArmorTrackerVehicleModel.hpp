#pragma once

/**
 * @file ArmorTrackerVehicleModel.hpp
 * @brief 整车状态模型、面匹配和 EKF 更新核心实现。
 *
 * ArmorTracker 依赖 CameraInfo 模板参数，因此这部分实现保留在头文件中。
 */

/**
 * @brief 将角度归一化到 (-pi, pi]。
 */
template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::VehicleLimitRad(double angle)
{
  while (angle > M_PI)
  {
    angle -= 2.0 * M_PI;
  }
  while (angle <= -M_PI)
  {
    angle += 2.0 * M_PI;
  }
  return angle;
}

/**
 * @brief 从 detector 四元数提取最接近参考角的装甲面 yaw。
 */
template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::VehicleDetectorYawNear(
    const LibXR::Quaternion<double>& q, double reference_yaw)
{
  return MatchYawAllowPiAmbiguityEnabled()
             ? armor_tracker::MeasuredArmorYawNearAllowPi(q, reference_yaw)
             : armor_tracker::MeasuredArmorYawNear(q, reference_yaw);
}

/**
 * @brief 从 detector 装甲结果提取最接近参考角的装甲面 yaw。
 */
template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::VehicleDetectorYawNear(const ArmorDetectorResult& armor,
                                                    double reference_yaw)
{
  return VehicleDetectorYawNear(armor.pose.rotation, reference_yaw);
}

/**
 * @brief 将 tracker 世界系 xyz 坐标转换为 yaw/pitch/distance。
 */
template <CameraTypes::CameraInfo CameraInfoV>
Eigen::Vector3d ArmorTracker<CameraInfoV>::VehicleXyzToYpd(const Eigen::Vector3d& xyz)
{
  const double x = xyz.x();
  const double y = xyz.y();
  const double z = xyz.z();
  const double xy_norm = std::hypot(x, y);
  return {std::atan2(y, x), std::atan2(z, xy_norm), xyz.norm()};
}

/**
 * @brief 计算 xyz 到 yaw/pitch/distance 的解析雅可比矩阵。
 */
template <CameraTypes::CameraInfo CameraInfoV>
Eigen::MatrixXd ArmorTracker<CameraInfoV>::VehicleXyzToYpdJacobian(const Eigen::Vector3d& xyz)
{
  const double x = xyz.x();
  const double y = xyz.y();
  const double z = xyz.z();
  const double xy2 = std::max(1e-9, x * x + y * y);
  const double xy = std::sqrt(xy2);
  const double r2 = std::max(1e-9, xy2 + z * z);
  const double r = std::sqrt(r2);

  Eigen::MatrixXd j(3, 3);
  j << -y / xy2, x / xy2, 0.0,
      -(x * z) / (xy * r2), -(y * z) / (xy * r2), xy / r2,
      x / r, y / r, z / r;
  return j;
}

/**
 * @brief 判断装甲是否属于平衡步兵等双装甲目标。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehicleIsBalanceArmor(const ArmorDetectorResult& armor)
{
  return armor.type == ArmorType::LARGE &&
         (armor.number == ArmorNumber::THREE ||
          armor.number == ArmorNumber::FOUR ||
          armor.number == ArmorNumber::FIVE);
}

/**
 * @brief 根据目标类型返回 tracker 应使用的装甲面数量。
 */
template <CameraTypes::CameraInfo CameraInfoV>
int ArmorTracker<CameraInfoV>::VehicleArmorCountFor(const ArmorDetectorResult& armor)
{
  if (VehicleIsBalanceArmor(armor))
  {
    return 2;
  }
  if (armor.number == ArmorNumber::OUTPOST || armor.number == ArmorNumber::BASE)
  {
    return 3;
  }
  return 4;
}

/**
 * @brief 根据目标类型返回初始化时使用的整车半径先验。
 */
template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::VehicleInitialRadiusFor(
    const ArmorDetectorResult& armor) const
{
  if (VehicleIsBalanceArmor(armor))
  {
    return 0.2;
  }
  if (armor.number == ArmorNumber::OUTPOST)
  {
    return 0.2765;
  }
  if (armor.number == ArmorNumber::BASE)
  {
    return 0.3205;
  }
  const double min_radius =
      std::min(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
  const double max_radius =
      std::max(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
  return std::clamp(cfg_.geometry.initial_radius, min_radius, max_radius);
}

/**
 * @brief 根据目标类型生成 11 维 EKF 初始协方差对角线。
 */
template <CameraTypes::CameraInfo CameraInfoV>
Eigen::VectorXd ArmorTracker<CameraInfoV>::VehicleInitialP0DiagFor(
    const ArmorDetectorResult& armor)
{
  Eigen::VectorXd p0_diag(11);
  if (VehicleIsBalanceArmor(armor))
  {
    p0_diag << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 1e-4, 1e-4;
  }
  else if (armor.number == ArmorNumber::OUTPOST)
  {
    p0_diag << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0;
  }
  else if (armor.number == ArmorNumber::BASE)
  {
    p0_diag << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
  }
  else
  {
    p0_diag << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 1e-4, 1e-4;
  }
  if (VehicleArmorCountFor(armor) == 4)
  {
    p0_diag[ExtendedKalmanFilter::DELTA_Z] = VehicleDeltaZInitialVariance();
  }
  return p0_diag;
}

/**
 * @brief 从 11 维整车状态计算指定 canonical 面的三维位置。
 */
template <CameraTypes::CameraInfo CameraInfoV>
Eigen::Vector3d ArmorTracker<CameraInfoV>::VehicleArmorPosition(
    const Eigen::VectorXd& state, int id) const
{
  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  const double angle = VehicleLimitRad(state[ExtendedKalmanFilter::YAW] +
                                  id * 2.0 * M_PI / armor_count);
  const bool use_length_height = (armor_count == 4) && (id == 1 || id == 3);
  const double radius =
      use_length_height ? state[ExtendedKalmanFilter::ROBOT_R] +
                              state[ExtendedKalmanFilter::DELTA_R]
                        : state[ExtendedKalmanFilter::ROBOT_R];
  const double armor_x = state[ExtendedKalmanFilter::X_CENTER] +
                         radius * std::cos(angle);
  const double armor_y = state[ExtendedKalmanFilter::Y_CENTER] +
                         radius * std::sin(angle);
  const double armor_z =
      use_length_height ? state[ExtendedKalmanFilter::Z_ARMOR] +
                              state[ExtendedKalmanFilter::DELTA_Z]
                        : state[ExtendedKalmanFilter::Z_ARMOR];
  return {armor_x, armor_y, armor_z};
}

/**
 * @brief 计算指定装甲面对 yaw/pitch/distance/yaw 观测的雅可比矩阵。
 */
template <CameraTypes::CameraInfo CameraInfoV>
Eigen::MatrixXd ArmorTracker<CameraInfoV>::VehicleObservationJacobian(
    const Eigen::VectorXd& state, int id) const
{
  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  const double angle = VehicleLimitRad(state[ExtendedKalmanFilter::YAW] +
                                  id * 2.0 * M_PI / armor_count);
  const bool use_length_height = (armor_count == 4) && (id == 1 || id == 3);
  const double radius =
      use_length_height ? state[ExtendedKalmanFilter::ROBOT_R] +
                              state[ExtendedKalmanFilter::DELTA_R]
                        : state[ExtendedKalmanFilter::ROBOT_R];
  const double dx_da = -radius * std::sin(angle);
  const double dy_da = radius * std::cos(angle);
  const double dx_dr = std::cos(angle);
  const double dy_dr = std::sin(angle);
  const double dx_dl = use_length_height ? std::cos(angle) : 0.0;
  const double dy_dl = use_length_height ? std::sin(angle) : 0.0;
  const double dz_dh = use_length_height ? 1.0 : 0.0;

  Eigen::MatrixXd h_armor_xyza = Eigen::MatrixXd::Zero(4, 11);
  h_armor_xyza(0, ExtendedKalmanFilter::X_CENTER) = 1.0;
  h_armor_xyza(0, ExtendedKalmanFilter::YAW) = dx_da;
  h_armor_xyza(0, ExtendedKalmanFilter::ROBOT_R) = dx_dr;
  h_armor_xyza(0, ExtendedKalmanFilter::DELTA_R) = dx_dl;
  h_armor_xyza(1, ExtendedKalmanFilter::Y_CENTER) = 1.0;
  h_armor_xyza(1, ExtendedKalmanFilter::YAW) = dy_da;
  h_armor_xyza(1, ExtendedKalmanFilter::ROBOT_R) = dy_dr;
  h_armor_xyza(1, ExtendedKalmanFilter::DELTA_R) = dy_dl;
  h_armor_xyza(2, ExtendedKalmanFilter::Z_ARMOR) = 1.0;
  h_armor_xyza(2, ExtendedKalmanFilter::DELTA_Z) = dz_dh;
  h_armor_xyza(3, ExtendedKalmanFilter::YAW) = 1.0;

  const Eigen::MatrixXd h_armor_ypd =
      VehicleXyzToYpdJacobian(VehicleArmorPosition(state, id));
  Eigen::MatrixXd h_armor_ypda(4, 4);
  h_armor_ypda << h_armor_ypd(0, 0), h_armor_ypd(0, 1),
      h_armor_ypd(0, 2), 0.0, h_armor_ypd(1, 0), h_armor_ypd(1, 1),
      h_armor_ypd(1, 2), 0.0, h_armor_ypd(2, 0), h_armor_ypd(2, 1),
      h_armor_ypd(2, 2), 0.0, 0.0, 0.0, 0.0, 1.0;
  return h_armor_ypda * h_armor_xyza;
}

/**
 * @brief 计算一个 detector 装甲结果匹配到指定 canonical 面的误差和总分。
 */
template <CameraTypes::CameraInfo CameraInfoV>
typename ArmorTracker<CameraInfoV>::VehicleArmorMatch
ArmorTracker<CameraInfoV>::VehicleMatchArmorToFace(const ArmorDetectorResult& armor,
                                              const Eigen::VectorXd& state,
                                              int face_index) const
{
  constexpr double kMatchYawScale = 0.12;
  constexpr double kMatchPitchScale = 0.10;
  constexpr double kMatchDistanceScale = 0.30;
  constexpr double kMatchAngleScale = 0.45;
  constexpr double kMatchPositionScale = 0.18;

  const Eigen::Vector3d pred_xyz = VehicleArmorPosition(state, face_index);
  const Eigen::Vector3d pred_ypd = VehicleXyzToYpd(pred_xyz);
  const double pred_yaw = GetArmorYawFromState(state, face_index);
  const Eigen::Vector3d armor_xyz(armor.pose.translation.x(),
                                  armor.pose.translation.y(),
                                  armor.pose.translation.z());
  const double measured_yaw = VehicleDetectorYawNear(armor.pose.rotation, pred_yaw);

  VehicleArmorMatch match{};
  match.id = face_index;
  const Eigen::Vector3d armor_ypd = VehicleXyzToYpd(armor_xyz);
  match.yaw_error = std::abs(VehicleLimitRad(armor_ypd.x() - pred_ypd.x()));
  match.pitch_error = std::abs(VehicleLimitRad(armor_ypd.y() - pred_ypd.y()));
  match.distance_error = std::abs(armor_ypd.z() - pred_ypd.z());
  match.angle_error = std::abs(VehicleLimitRad(measured_yaw - pred_yaw));
  match.xyz_error = (armor_xyz - pred_xyz).norm();
  match.measured_yaw = measured_yaw;
  const double observation_quality_penalty =
      ObservationQualityEnabled()
          ? armor_tracker::ArmorObservationQualityPenalty(
                armor, ObservationStableMaxReprojectionPx(),
                ObservationStableMinAreaPx(), ObservationStableMinConfidence())
          : 0.0;
  match.score = match.yaw_error / kMatchYawScale +
                match.pitch_error / kMatchPitchScale +
                match.distance_error / kMatchDistanceScale +
                match.angle_error / kMatchAngleScale +
                match.xyz_error / kMatchPositionScale +
                ObservationQualityScoreWeight() * observation_quality_penalty;
  return match;
}

/**
 * @brief 在所有可用 canonical 面中为一个 detector 装甲结果选择最佳匹配面。
 */
template <CameraTypes::CameraInfo CameraInfoV>
typename ArmorTracker<CameraInfoV>::VehicleArmorMatch
ArmorTracker<CameraInfoV>::VehicleMatchArmor(const ArmorDetectorResult& armor,
                                        const Eigen::VectorXd& state) const
{
  constexpr double kMatchSwitchPenalty = 0.35;

  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  VehicleArmorMatch best{};

  for (int id = 0; id < armor_count; ++id)
  {
    VehicleArmorMatch match = VehicleMatchArmorToFace(armor, state, id);
    const double switch_penalty =
        (rt_.update_count > 0 && id != rt_.tracked_face_index)
            ? kMatchSwitchPenalty
            : 0.0;
    match.score += switch_penalty;
    if (match.score < best.score)
    {
      best = match;
    }
  }
  return best;
}

/**
 * @brief 用同号多装甲观测尝试建立 canonical 相位和高低差初值。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehicleTryCanonicalizeInitialState(
    const ArmorDetectorResults& armors_msg, bool force)
{
  if (!VehicleCanonicalInitEnabled() || rt_.tracked_id == ArmorNumber::INVALID)
  {
    return false;
  }

  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  if (armor_count != 4)
  {
    return false;
  }

  /**
   * @brief canonical 初始化候选观测。
   */
  struct Observation
  {
    std::size_t armor_index = 0;          ///< detector 结果索引。
    ArmorDetectorResult armor{};          ///< detector 装甲结果。
    int image_track_id = -1;              ///< 图像 track ID。
    bool confirmed_image_track = false;   ///< 图像 track 是否已确认。
  };

  std::vector<Observation> observations;
  observations.reserve(std::min<std::size_t>(armors_msg.size(), 4));
  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    if (armor.number != rt_.tracked_id)
    {
      continue;
    }
    if (rt_.tracked_armor.type != ArmorType::INVALID &&
        armor.type != rt_.tracked_armor.type)
    {
      continue;
    }
    observations.push_back({armor_index, armor, FindDetectionTrackId(armor_index),
                            IsDetectionTrackConfirmed(armor_index)});
  }

  if (observations.size() < 2)
  {
    return false;
  }

  std::sort(observations.begin(), observations.end(),
            [](const Observation& lhs, const Observation& rhs)
            {
              if (lhs.armor.priority != rhs.armor.priority)
              {
                return static_cast<int>(lhs.armor.priority) <
                       static_cast<int>(rhs.armor.priority);
              }
              if (std::abs(lhs.armor.distance_to_image_center -
                           rhs.armor.distance_to_image_center) > 1e-6)
              {
                return lhs.armor.distance_to_image_center <
                       rhs.armor.distance_to_image_center;
              }
              if (std::abs(lhs.armor.center.x - rhs.armor.center.x) > 1e-3)
              {
                return lhs.armor.center.x < rhs.armor.center.x;
              }
              if (std::abs(lhs.armor.center.y - rhs.armor.center.y) > 1e-3)
              {
                return lhs.armor.center.y < rhs.armor.center.y;
              }
              return lhs.armor_index < rhs.armor_index;
            });
  if (observations.size() > static_cast<std::size_t>(armor_count))
  {
    observations.resize(static_cast<std::size_t>(armor_count));
  }

  const double radius = VehicleInitialRadiusFor(rt_.tracked_armor);
  const double angle_step = 2.0 * M_PI / armor_count;
  const double min_height = VehicleCanonicalInitMinHeight();
  const double max_abs_dz = VehicleCanonicalInitMaxAbsDz();
  const double max_score = VehicleCanonicalInitMaxScore();

  const auto make_state = [radius](double center_x, double center_y,
                                   double center_z, double yaw, double dz)
  {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(11);
    state << center_x, 0.0, center_y, 0.0, center_z, 0.0, yaw, 0.0,
        radius, 0.0, dz;
    return state;
  };

  const auto score_observation_face =
      [this](const Observation& observation, const Eigen::VectorXd& state,
             int face_index)
  {
    constexpr double kMatchYawScale = 0.12;
    constexpr double kMatchPitchScale = 0.10;
    constexpr double kMatchDistanceScale = 0.30;
    constexpr double kMatchAngleScale = 0.45;
    constexpr double kMatchPositionScale = 0.18;

    const Eigen::Vector3d armor_xyz(observation.armor.pose.translation.x(),
                                    observation.armor.pose.translation.y(),
                                    observation.armor.pose.translation.z());
    const Eigen::Vector3d armor_ypd = VehicleXyzToYpd(armor_xyz);
    const Eigen::Vector3d pred_xyz = VehicleArmorPosition(state, face_index);
    const Eigen::Vector3d pred_ypd = VehicleXyzToYpd(pred_xyz);
    const double pred_yaw = GetArmorYawFromState(state, face_index);
    const double measured_yaw =
        VehicleDetectorYawNear(observation.armor, pred_yaw);
    const double yaw_error = std::abs(VehicleLimitRad(armor_ypd.x() - pred_ypd.x()));
    const double pitch_error =
        std::abs(VehicleLimitRad(armor_ypd.y() - pred_ypd.y()));
    const double distance_error = std::abs(armor_ypd.z() - pred_ypd.z());
    const double angle_error = std::abs(VehicleLimitRad(measured_yaw - pred_yaw));
    const double xyz_error = (armor_xyz - pred_xyz).norm();
    return yaw_error / kMatchYawScale + pitch_error / kMatchPitchScale +
           distance_error / kMatchDistanceScale +
           angle_error / kMatchAngleScale + xyz_error / kMatchPositionScale;
  };

  /**
   * @brief canonical 初始化搜索得到的最优相位假设。
   */
  struct BestHypothesis
  {
    bool valid = false;                                  ///< 假设是否有效。
    bool has_height = false;                             ///< 是否由奇偶面观测到高度差。
    bool positive_dz = false;                            ///< 高低差是否为正。
    double score = std::numeric_limits<double>::infinity(); ///< 平均匹配分。
    double dz = 0.0;                                     ///< 假设高低差。
    double yaw = 0.0;                                    ///< 假设整车 yaw。
    int tracked_face = 0;                                ///< 当前跟踪观测对应面。
    std::size_t tracked_observation = 0;                 ///< 当前跟踪观测索引。
    Eigen::VectorXd state = Eigen::VectorXd::Zero(11);   ///< 假设状态向量。
    std::array<int, 4> faces{{0, 1, 2, 3}};              ///< 观测 rank 到面索引映射。
  } best;

  std::array<int, 4> face_permutation{{0, 1, 2, 3}};
  for (std::size_t anchor_rank = 0; anchor_rank < observations.size(); ++anchor_rank)
  {
    const auto& anchor = observations[anchor_rank];
    const Eigen::Vector3d anchor_xyz(anchor.armor.pose.translation.x(),
                                     anchor.armor.pose.translation.y(),
                                     anchor.armor.pose.translation.z());
    for (int anchor_face = 0; anchor_face < armor_count; ++anchor_face)
    {
      const double anchor_yaw =
          VehicleDetectorYawNear(anchor.armor, anchor_face * angle_step);
      const double seed_yaw =
          VehicleLimitRad(anchor_yaw - anchor_face * angle_step);
      const Eigen::VectorXd seed_state = make_state(
          anchor_xyz.x() - radius * std::cos(anchor_yaw),
          anchor_xyz.y() - radius * std::sin(anchor_yaw), anchor_xyz.z(),
          seed_yaw, 0.0);

      face_permutation = {{0, 1, 2, 3}};
      do
      {
        if (face_permutation[anchor_rank] != anchor_face)
        {
          continue;
        }

        double yaw_sum = 0.0;
        double center_x_sum = 0.0;
        double center_y_sum = 0.0;
        double even_z_sum = 0.0;
        double odd_z_sum = 0.0;
        int even_count = 0;
        int odd_count = 0;
        double preliminary_score = 0.0;

        for (std::size_t obs_rank = 0; obs_rank < observations.size(); ++obs_rank)
        {
          const int face_index = face_permutation[obs_rank];
          const auto& observation = observations[obs_rank];
          const Eigen::Vector3d xyz(observation.armor.pose.translation.x(),
                                    observation.armor.pose.translation.y(),
                                    observation.armor.pose.translation.z());
          const double pred_yaw = GetArmorYawFromState(seed_state, face_index);
          const double measured_yaw =
              VehicleDetectorYawNear(observation.armor, pred_yaw);
          yaw_sum += armor_tracker::UnwrapYawNear(
              measured_yaw - face_index * angle_step, seed_yaw);
          center_x_sum += xyz.x() - radius * std::cos(measured_yaw);
          center_y_sum += xyz.y() - radius * std::sin(measured_yaw);
          if (face_index % 2 == 0)
          {
            even_z_sum += xyz.z();
            ++even_count;
          }
          else
          {
            odd_z_sum += xyz.z();
            ++odd_count;
          }
          preliminary_score +=
              score_observation_face(observation, seed_state, face_index);
        }

        (void)preliminary_score;
        const double inv_count = 1.0 / static_cast<double>(observations.size());
        const double yaw = VehicleLimitRad(yaw_sum * inv_count);
        const double center_x = center_x_sum * inv_count;
        const double center_y = center_y_sum * inv_count;
        double dz = 0.0;
        double center_z = 0.0;
        const bool has_even = even_count > 0;
        const bool has_odd = odd_count > 0;
        if (has_even && has_odd)
        {
          const double even_z = even_z_sum / static_cast<double>(even_count);
          const double odd_z = odd_z_sum / static_cast<double>(odd_count);
          dz = std::clamp(odd_z - even_z, -max_abs_dz, max_abs_dz);
          center_z = even_z;
        }
        else if (has_even)
        {
          center_z = even_z_sum / static_cast<double>(even_count);
        }
        else
        {
          center_z = odd_z_sum / static_cast<double>(odd_count);
        }
        const Eigen::VectorXd state =
            make_state(center_x, center_y, center_z, yaw, dz);

        double final_score = 0.0;
        for (std::size_t obs_rank = 0; obs_rank < observations.size(); ++obs_rank)
        {
          final_score += score_observation_face(
              observations[obs_rank], state, face_permutation[obs_rank]);
        }
        final_score *= inv_count;

        const bool has_height =
            has_even && has_odd && std::abs(dz) >= min_height;
        const bool positive_dz = dz >= 0.0;
        if (has_height && !positive_dz && VehicleCanonicalInitPreferPositiveDz())
        {
          final_score += 4.0;
        }

        bool better = final_score < best.score - 1e-6;
        if (!better && std::abs(final_score - best.score) <= 1e-6)
        {
          if (has_height != best.has_height)
          {
            better = has_height;
          }
          else if (positive_dz != best.positive_dz)
          {
            better = positive_dz;
          }
          else if (std::abs(dz) > std::abs(best.dz) + 1e-4)
          {
            better = true;
          }
          else if (anchor_rank < best.tracked_observation)
          {
            better = true;
          }
        }

        if (better)
        {
          best.valid = true;
          best.has_height = has_height;
          best.positive_dz = positive_dz;
          best.score = final_score;
          best.dz = dz;
          best.yaw = yaw;
          best.tracked_face = anchor_face;
          best.tracked_observation = anchor_rank;
          best.state = state;
          for (std::size_t obs_rank = 0; obs_rank < observations.size(); ++obs_rank)
          {
            best.faces[obs_rank] = face_permutation[obs_rank];
          }
        }
      } while (std::next_permutation(face_permutation.begin(),
                                    face_permutation.end()));
    }
  }

  if (!best.valid || !best.has_height || best.score > max_score)
  {
    return false;
  }

  ekf_.state = best.state;
  ekf_.covariance = VehicleInitialP0DiagFor(rt_.tracked_armor).asDiagonal();
  ekf_.measurement_face_index = best.tracked_face;
  const auto& tracked_observation = observations[best.tracked_observation];
  ekf_.measurement = Eigen::Vector4d(
      tracked_observation.armor.pose.translation.x(),
      tracked_observation.armor.pose.translation.y(),
      tracked_observation.armor.pose.translation.z(),
      GetArmorYawFromState(ekf_.state, best.tracked_face));
  rt_.tracked_armor = tracked_observation.armor;
  rt_.tracked_face_index = best.tracked_face;
  rt_.last_yaw = GetArmorYawFromState(ekf_.state, rt_.tracked_face_index);
  rt_.face_track_id_valid.fill(false);
  rt_.face_track_id.fill(0);
  rt_.tracked_face_track_id_valid = false;
  rt_.tracked_face_track_id = 0;
  for (std::size_t obs_rank = 0; obs_rank < observations.size(); ++obs_rank)
  {
    const auto& observation = observations[obs_rank];
    const int face_index = best.faces[obs_rank];
    if (face_index < 0 || face_index >= 4 || !observation.confirmed_image_track ||
        observation.image_track_id < 0)
    {
      continue;
    }
    rt_.face_track_id_valid[face_index] = true;
    rt_.face_track_id[face_index] =
        static_cast<uint16_t>(observation.image_track_id);
  }
  if (rt_.face_track_id_valid[rt_.tracked_face_index])
  {
    rt_.tracked_face_track_id_valid = true;
    rt_.tracked_face_track_id = rt_.face_track_id[rt_.tracked_face_index];
  }
  SyncGeometryRuntimeFromState();
  ekf_.ekf.SetState(ekf_.state);
  XR_LOG_DEBUG(
      "whole-body model canonical init: force=%d obs=%u score=%.3f face=%d dz=%.4f yaw=%.3f",
      force ? 1 : 0, static_cast<unsigned>(observations.size()), best.score, best.tracked_face,
      best.dz, best.yaw);
  return true;
}

/**
 * @brief 用左右两片相邻装甲的射线交点求解整车中心和长短半径。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehicleSolvePairGeometry(
    const VehiclePairObservation& left, int left_face, double left_measured_yaw,
    const VehiclePairObservation& right, int right_face, double right_measured_yaw,
    const Eigen::VectorXd& state, VehiclePairGeometryFit& fit) const
{
  fit = VehiclePairGeometryFit{};
  if (rt_.tracked_armors_num != ArmorsNum::NORMAL_4 || left_face == right_face)
  {
    return false;
  }

  const bool left_odd = (left_face % 2) == 1;
  const bool right_odd = (right_face % 2) == 1;
  if (left_odd == right_odd)
  {
    return false;
  }

  const Eigen::Vector2d left_dir(std::cos(left_measured_yaw),
                                 std::sin(left_measured_yaw));
  const Eigen::Vector2d right_dir(std::cos(right_measured_yaw),
                                  std::sin(right_measured_yaw));
  const double det = right_dir.x() * left_dir.y() - left_dir.x() * right_dir.y();
  if (!std::isfinite(det) || std::abs(det) < VehiclePairGeometryMinDeterminant())
  {
    return false;
  }

  const Eigen::Vector2d delta(left.xyz.x() - right.xyz.x(),
                              left.xyz.y() - right.xyz.y());
  Eigen::Matrix2d a;
  a << left_dir.x(), -right_dir.x(), left_dir.y(), -right_dir.y();
  const Eigen::Vector2d radii = a.fullPivLu().solve(delta);
  if (!std::isfinite(radii.x()) || !std::isfinite(radii.y()))
  {
    return false;
  }

  const double min_radius =
      std::min(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
  const double max_radius =
      std::max(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
  const double left_radius = radii.x();
  const double right_radius = radii.y();
  if (left_radius < min_radius || left_radius > max_radius ||
      right_radius < min_radius || right_radius > max_radius)
  {
    return false;
  }

  const Eigen::Vector2d left_center =
      Eigen::Vector2d(left.xyz.x(), left.xyz.y()) - left_radius * left_dir;
  const Eigen::Vector2d right_center =
      Eigen::Vector2d(right.xyz.x(), right.xyz.y()) - right_radius * right_dir;
  const Eigen::Vector2d center = 0.5 * (left_center + right_center);
  const double fit_error = 0.5 * (left_center - right_center).norm();
  if (!std::isfinite(fit_error) || fit_error > VehiclePairGeometryMaxFitError())
  {
    return false;
  }

  const double current_r_even = state(ExtendedKalmanFilter::ROBOT_R);
  const double current_r_odd =
      current_r_even + state(ExtendedKalmanFilter::DELTA_R);
  const double r_even = left_odd ? right_radius : left_radius;
  const double r_odd = left_odd ? left_radius : right_radius;
  const double center_shift =
      (center - Eigen::Vector2d(state(ExtendedKalmanFilter::X_CENTER),
                                state(ExtendedKalmanFilter::Y_CENTER)))
          .norm();
  const double radius_shift =
      std::max(std::abs(r_even - current_r_even),
               std::abs(r_odd - current_r_odd));
  if (!std::isfinite(center_shift) || !std::isfinite(radius_shift) ||
      center_shift > VehiclePairGeometryMaxCenterShift() ||
      radius_shift > VehiclePairGeometryMaxRadiusShift())
  {
    return false;
  }

  const int armor_count = std::max(1, static_cast<int>(rt_.tracked_armors_num));
  const double angle_step = 2.0 * M_PI / armor_count;
  const double current_yaw = state(ExtendedKalmanFilter::YAW);
  const double left_yaw0 = armor_tracker::UnwrapYawNear(
      left_measured_yaw - left_face * angle_step, current_yaw);
  const double right_yaw0 = armor_tracker::UnwrapYawNear(
      right_measured_yaw - right_face * angle_step, left_yaw0);
  const double yaw = VehicleLimitRad(0.5 * (left_yaw0 + right_yaw0));

  fit.valid = true;
  fit.center = center;
  fit.r_even = r_even;
  fit.r_odd = SymmetricGeometryEnabled() ? r_even : r_odd;
  fit.yaw = yaw;
  fit.fit_error = fit_error;
  fit.center_shift = center_shift;
  fit.radius_shift = radius_shift;
  return true;
}

/**
 * @brief 从一帧检测结果中搜索可用于双装甲几何或高低差更新的最佳配对。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehicleResolvePairMatch(
    const ArmorDetectorResults& armors_msg, const Eigen::VectorXd& state,
    VehiclePairMatch& pair_match) const
{
  constexpr double kPairFaceSwitchPenalty = 0.75;
  constexpr double kPairCenterConsistencyScale = 0.10;
  constexpr double kPairMaxCenterSplit = 0.20;
  constexpr double kPairGeometryFitScale = 0.03;
  constexpr double kPairGeometryCenterScale = 0.35;
  constexpr double kPairGeometryRadiusScale = 0.18;
  constexpr double kPairMaxScore = 4.5;
  constexpr double kPairMaxXyzError = 0.45;

  pair_match = VehiclePairMatch{};
  const bool pair_geometry_enabled = VehiclePairGeometryEnabled();
  const bool pair_dz_enabled = VehiclePairDeltaZEnabled();
  if ((!pair_geometry_enabled && !pair_dz_enabled) ||
      rt_.tracked_id == ArmorNumber::INVALID ||
      rt_.tracked_armors_num != ArmorsNum::NORMAL_4)
  {
    return false;
  }

  std::vector<VehiclePairObservation> observations;
  observations.reserve(std::min<std::size_t>(armors_msg.size(), 4));
  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    if (armor.number != rt_.tracked_id)
    {
      continue;
    }
    if (rt_.tracked_armor.type != ArmorType::INVALID &&
        armor.type != rt_.tracked_armor.type)
    {
      continue;
    }
    const double z = armor.pose.translation.z();
    if (!std::isfinite(z))
    {
      continue;
    }
    observations.push_back({armor_index, armor,
                            Eigen::Vector3d(armor.pose.translation.x(),
                                            armor.pose.translation.y(), z)});
  }

  if (observations.size() < 2)
  {
    return false;
  }

  const int armor_count = std::max(1, static_cast<int>(rt_.tracked_armors_num));
  const double angle_step = 2.0 * M_PI / armor_count;
  const double current_yaw = state(ExtendedKalmanFilter::YAW);
  VehiclePairMatch best{};

  for (std::size_t lhs = 0; lhs < observations.size(); ++lhs)
  {
    for (std::size_t rhs = lhs + 1; rhs < observations.size(); ++rhs)
    {
      const VehiclePairObservation* left = &observations[lhs];
      const VehiclePairObservation* right = &observations[rhs];
      if (right->armor.center.x < left->armor.center.x ||
          (std::abs(right->armor.center.x - left->armor.center.x) < 1e-3 &&
           right->xyz.x() < left->xyz.x()))
      {
        std::swap(left, right);
      }

      for (int left_face = 0; left_face < armor_count; ++left_face)
      {
        for (int right_face = 0; right_face < armor_count; ++right_face)
        {
          if (left_face == right_face)
          {
            continue;
          }
          const int face_delta = std::abs(left_face - right_face);
          if (!(face_delta == 1 || face_delta == armor_count - 1))
          {
            continue;
          }
          if (((left_face % 2) == 1) == ((right_face % 2) == 1))
          {
            continue;
          }

          const double left_ref_yaw =
              VehicleLimitRad(current_yaw + left_face * angle_step);
          const double right_ref_yaw =
              VehicleLimitRad(current_yaw + right_face * angle_step);
          const double left_measured_yaw =
              VehicleDetectorYawNear(left->armor, left_ref_yaw);
          const double right_measured_yaw =
              VehicleDetectorYawNear(right->armor, right_ref_yaw);
          const double left_yaw0 = armor_tracker::UnwrapYawNear(
              left_measured_yaw - left_face * angle_step, current_yaw);
          const double right_yaw0 = armor_tracker::UnwrapYawNear(
              right_measured_yaw - right_face * angle_step, left_yaw0);
          const double yaw = VehicleLimitRad(0.5 * (left_yaw0 + right_yaw0));

          VehiclePairGeometryFit geometry{};
          const bool geometry_valid =
              pair_geometry_enabled &&
              VehicleSolvePairGeometry(*left, left_face, left_measured_yaw, *right,
                                  right_face, right_measured_yaw, state,
                                  geometry);
          if (pair_geometry_enabled && !geometry_valid)
          {
            continue;
          }

          const bool left_odd = (left_face % 2) == 1;
          const double even_z = left_odd ? right->xyz.z() : left->xyz.z();
          const double odd_z = left_odd ? left->xyz.z() : right->xyz.z();
          const double observed_dz = std::clamp(
              odd_z - even_z, -VehiclePairDeltaZMaxAbs(), VehiclePairDeltaZMaxAbs());
          const bool dz_valid =
              pair_dz_enabled && std::abs(observed_dz) >= VehiclePairDeltaZMinHeight();

          if (!geometry_valid && !dz_valid)
          {
            continue;
          }

          Eigen::VectorXd candidate = state;
          candidate(ExtendedKalmanFilter::Z_ARMOR) = even_z;
          candidate(ExtendedKalmanFilter::YAW) =
              geometry_valid ? geometry.yaw : yaw;
          candidate(ExtendedKalmanFilter::DELTA_Z) = observed_dz;
          if (geometry_valid)
          {
            candidate(ExtendedKalmanFilter::X_CENTER) = geometry.center.x();
            candidate(ExtendedKalmanFilter::Y_CENTER) = geometry.center.y();
            candidate(ExtendedKalmanFilter::ROBOT_R) = geometry.r_even;
            candidate(ExtendedKalmanFilter::DELTA_R) =
                SymmetricGeometryEnabled() ? 0.0 : (geometry.r_odd - geometry.r_even);
          }

          const VehicleArmorMatch left_match =
              VehicleMatchArmorToFace(left->armor, candidate, left_face);
          const VehicleArmorMatch right_match =
              VehicleMatchArmorToFace(right->armor, candidate, right_face);
          double score = 0.5 * (left_match.score + right_match.score);
          if (geometry_valid)
          {
            score += geometry.fit_error / kPairGeometryFitScale;
            score += geometry.center_shift / kPairGeometryCenterScale;
            score += geometry.radius_shift / kPairGeometryRadiusScale;
          }
          else
          {
            auto estimate_center = [this, angle_step, armor_count](
                                       const VehiclePairObservation& observation,
                                       const Eigen::VectorXd& candidate_state,
                                       int face_index)
            {
              const Eigen::Vector3d measured_xyz(
                  observation.armor.pose.translation.x(),
                  observation.armor.pose.translation.y(),
                  observation.armor.pose.translation.z());
              const bool use_length_height =
                  armor_count == 4 && (face_index == 1 || face_index == 3);
              const double radius =
                  candidate_state(ExtendedKalmanFilter::ROBOT_R) +
                  (use_length_height
                       ? candidate_state(ExtendedKalmanFilter::DELTA_R)
                       : 0.0);
              const double angle = VehicleLimitRad(
                  candidate_state(ExtendedKalmanFilter::YAW) +
                  face_index * angle_step);
              const double z_offset =
                  use_length_height
                      ? candidate_state(ExtendedKalmanFilter::DELTA_Z)
                      : 0.0;
              return Eigen::Vector3d(measured_xyz.x() -
                                         radius * std::cos(angle),
                                     measured_xyz.y() -
                                         radius * std::sin(angle),
                                     measured_xyz.z() - z_offset);
            };
            const double center_split =
                (estimate_center(*left, candidate, left_face) -
                 estimate_center(*right, candidate, right_face))
                    .norm();
            if (center_split > kPairMaxCenterSplit)
            {
              continue;
            }
            score += center_split / kPairCenterConsistencyScale;
          }
          if (rt_.update_count > 0 && rt_.tracked_face_index != left_face &&
              rt_.tracked_face_index != right_face)
          {
            score += kPairFaceSwitchPenalty;
          }
          const double yaw_delta = std::abs(VehicleLimitRad(
              candidate(ExtendedKalmanFilter::YAW) - current_yaw));
          const double best_yaw_delta =
              best.valid ? std::abs(VehicleLimitRad(best.yaw - current_yaw))
                         : std::numeric_limits<double>::infinity();
          const bool better =
              score < best.score - 1e-6 ||
              (std::abs(score - best.score) <= 1e-6 &&
               yaw_delta < best_yaw_delta);
          if (!better)
          {
            continue;
          }

          best.valid = true;
          best.geometry_valid = geometry_valid;
          best.dz_valid = dz_valid;
          best.left = *left;
          best.right = *right;
          best.left_face = left_face;
          best.right_face = right_face;
          best.left_match = left_match;
          best.right_match = right_match;
          best.score = score;
          best.yaw = candidate(ExtendedKalmanFilter::YAW);
          best.dz_observed = observed_dz;
          best.even_z_observed = even_z;
          best.geometry = geometry;

          const bool previous_is_left = rt_.tracked_face_index == left_face;
          const bool previous_is_right = rt_.tracked_face_index == right_face;
          const bool track_left =
              (previous_is_left && !previous_is_right) ||
              (!previous_is_left && !previous_is_right &&
               left_match.score <= right_match.score);
          if (track_left)
          {
            best.tracked_face = left_face;
            best.tracked_armor_index = left->armor_index;
            best.tracked_armor = left->armor;
            best.tracked_match = left_match;
          }
          else
          {
            best.tracked_face = right_face;
            best.tracked_armor_index = right->armor_index;
            best.tracked_armor = right->armor;
            best.tracked_match = right_match;
          }
        }
      }
    }
  }

  if (!best.valid)
  {
    return false;
  }

  if (best.score > kPairMaxScore ||
      std::max(best.left_match.xyz_error, best.right_match.xyz_error) >
          kPairMaxXyzError)
  {
    return false;
  }

  pair_match = best;
  return true;
}

/**
 * @brief 当高低差符号与 canonical 约定相反时整体旋转相位并变换协方差。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::VehicleCanonicalizePairPhaseForPositiveDz()
{
  if (rt_.tracked_armors_num != ArmorsNum::NORMAL_4 ||
      SymmetricGeometryEnabled() || !VehicleCanonicalInitPreferPositiveDz() ||
      ekf_.state(ExtendedKalmanFilter::DELTA_Z) >= 0.0)
  {
    return;
  }

  Eigen::MatrixXd transform =
      Eigen::MatrixXd::Identity(ekf_.covariance.rows(), ekf_.covariance.cols());
  transform(ExtendedKalmanFilter::Z_ARMOR, ExtendedKalmanFilter::DELTA_Z) = 1.0;
  transform(ExtendedKalmanFilter::ROBOT_R, ExtendedKalmanFilter::DELTA_R) = 1.0;
  transform(ExtendedKalmanFilter::DELTA_Z, ExtendedKalmanFilter::DELTA_Z) = -1.0;
  transform(ExtendedKalmanFilter::DELTA_R, ExtendedKalmanFilter::DELTA_R) = -1.0;

  ekf_.state(ExtendedKalmanFilter::YAW) =
      VehicleLimitRad(ekf_.state(ExtendedKalmanFilter::YAW) + 0.5 * M_PI);
  ekf_.state(ExtendedKalmanFilter::Z_ARMOR) +=
      ekf_.state(ExtendedKalmanFilter::DELTA_Z);
  ekf_.state(ExtendedKalmanFilter::DELTA_Z) =
      -ekf_.state(ExtendedKalmanFilter::DELTA_Z);
  ekf_.state(ExtendedKalmanFilter::ROBOT_R) +=
      ekf_.state(ExtendedKalmanFilter::DELTA_R);
  ekf_.state(ExtendedKalmanFilter::DELTA_R) =
      -ekf_.state(ExtendedKalmanFilter::DELTA_R);
  ekf_.covariance = transform * ekf_.covariance * transform.transpose();

  rt_.tracked_face_index =
      armor_tracker::NormalizeFaceIndex(rt_.tracked_face_index - 1, 4);
  ekf_.measurement_face_index =
      armor_tracker::NormalizeFaceIndex(ekf_.measurement_face_index - 1, 4);

  const auto old_valid = rt_.face_track_id_valid;
  const auto old_ids = rt_.face_track_id;
  for (int face = 0; face < 4; ++face)
  {
    const int old_face = armor_tracker::NormalizeFaceIndex(face + 1, 4);
    rt_.face_track_id_valid[face] = old_valid[old_face];
    rt_.face_track_id[face] = old_ids[old_face];
  }
  if (rt_.face_track_id_valid[rt_.tracked_face_index])
  {
    rt_.tracked_face_track_id_valid = true;
    rt_.tracked_face_track_id = rt_.face_track_id[rt_.tracked_face_index];
  }
  else
  {
    rt_.tracked_face_track_id_valid = false;
    rt_.tracked_face_track_id = 0;
  }

  SyncGeometryRuntimeFromState();
  ekf_.ekf.SetState(ekf_.state);
}

/**
 * @brief 将双装甲几何或高低差观测作为标量观测写入 EKF。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::VehicleApplyPairGeometryUpdate(
    const VehiclePairMatch& pair_match)
{
  if (!pair_match.geometry_valid && !pair_match.dz_valid)
  {
    return;
  }

  constexpr int kMaxRows = 4;
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(kMaxRows, 11);
  Eigen::VectorXd z = Eigen::VectorXd::Zero(kMaxRows);
  Eigen::VectorXd r_diag = Eigen::VectorXd::Zero(kMaxRows);
  int rows = 0;

  const auto add_scalar = [&](int state_index, double measurement,
                              double variance)
  {
    h(rows, state_index) = 1.0;
    z(rows) = measurement;
    r_diag(rows) = variance;
    ++rows;
  };

  if (pair_match.geometry_valid)
  {
    const double covariance_floor = VehiclePairGeometryCovarianceFloor();
    ekf_.covariance(ExtendedKalmanFilter::ROBOT_R,
                    ExtendedKalmanFilter::ROBOT_R) =
        std::max(ekf_.covariance(ExtendedKalmanFilter::ROBOT_R,
                                 ExtendedKalmanFilter::ROBOT_R),
                 covariance_floor);
    ekf_.covariance(ExtendedKalmanFilter::DELTA_R,
                    ExtendedKalmanFilter::DELTA_R) =
        std::max(ekf_.covariance(ExtendedKalmanFilter::DELTA_R,
                                 ExtendedKalmanFilter::DELTA_R),
                 covariance_floor);

    // 双面 PnP yaw 的射线交点对系统偏差很敏感：两条方向线总能交出一个
    // 中心，fit_error 很小也可能得到错误的长短半径。相邻两板的 XY 弦长
    // 不依赖对手车宽先验，也不直接信任 PnP yaw；这里只用它弱约束平均半径，
    // 半径差仍交给后续单面观测和状态协方差慢慢收敛。
    const double pair_dx = pair_match.left.xyz.x() - pair_match.right.xyz.x();
    const double pair_dy = pair_match.left.xyz.y() - pair_match.right.xyz.y();
    const double chord_radius = std::hypot(pair_dx, pair_dy) / std::sqrt(2.0);
    const double min_radius =
        std::min(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    const double max_radius =
        std::max(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    if (std::isfinite(chord_radius) && chord_radius >= min_radius &&
        chord_radius <= max_radius)
    {
      // The adjacent-face chord observes only the average radius.  Updating
      // DELTA_R from this scalar would let the EKF invent a long/short split
      // from an unobservable measurement, so shift ROBOT_R while preserving
      // the current delta until a real delta observation is available.
      h(rows, ExtendedKalmanFilter::ROBOT_R) = 1.0;
      z(rows) =
          SymmetricGeometryEnabled()
              ? chord_radius
              : (chord_radius -
                 0.5 * ekf_.state(ExtendedKalmanFilter::DELTA_R));
      r_diag(rows) = VehiclePairGeometryRadiusVariance();
      ++rows;
    }
  }

  if (pair_match.dz_valid)
  {
    const int z_index = ExtendedKalmanFilter::Z_ARMOR;
    const int dz_index = ExtendedKalmanFilter::DELTA_Z;
    const double dz_variance = VehiclePairDeltaZVariance();
    ekf_.covariance(z_index, z_index) =
        std::max(ekf_.covariance(z_index, z_index), dz_variance);
    ekf_.covariance(dz_index, dz_index) =
        std::max(ekf_.covariance(dz_index, dz_index), dz_variance);
    add_scalar(z_index, pair_match.even_z_observed,
               VehiclePairGeometryCenterVariance());
    add_scalar(dz_index, pair_match.dz_observed, dz_variance);
  }

  if (rows <= 0)
  {
    return;
  }

  h.conservativeResize(rows, Eigen::NoChange);
  z.conservativeResize(rows);
  r_diag.conservativeResize(rows);

  const Eigen::VectorXd predicted = h * ekf_.state;
  Eigen::VectorXd innovation = z - predicted;

  const Eigen::MatrixXd r = r_diag.asDiagonal();
  const Eigen::MatrixXd innovation_cov =
      h * ekf_.covariance * h.transpose() + r;
  const Eigen::MatrixXd kalman_gain =
      ekf_.covariance * h.transpose() * innovation_cov.inverse();
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(ekf_.covariance.rows(), ekf_.covariance.cols());

  ekf_.state = ekf_.state + kalman_gain * innovation;
  ekf_.state(ExtendedKalmanFilter::YAW) =
      VehicleLimitRad(ekf_.state(ExtendedKalmanFilter::YAW));
  ekf_.covariance = (identity - kalman_gain * h) * ekf_.covariance *
                        (identity - kalman_gain * h).transpose() +
                    kalman_gain * r * kalman_gain.transpose();
  ClampGeometryState();
  if (pair_match.dz_valid)
  {
    VehicleCanonicalizePairPhaseForPositiveDz();
  }
}

/**
 * @brief 对双装甲匹配结果执行几何更新和可选的左右面单面更新。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::VehicleUpdatePair(const VehiclePairMatch& pair_match,
                                             uint64_t image_timestamp_us,
                                             CandidateDebugMsg* candidate_debug)
{
  // 单装甲板只能约束“当前可见板”；双板几何才约束整车中心和半径。
  VehicleApplyPairGeometryUpdate(pair_match);

  if (VehiclePairDualUpdateEnabled())
  {
    VehicleUpdate(pair_match.left.armor, pair_match.left_match, true,
             image_timestamp_us, candidate_debug);
    VehicleUpdate(pair_match.right.armor, pair_match.right_match, true,
             image_timestamp_us, candidate_debug);
  }
  else if (pair_match.tracked_face == pair_match.left_face)
  {
    VehicleUpdate(pair_match.left.armor, pair_match.left_match, true,
             image_timestamp_us, candidate_debug);
  }
  else
  {
    VehicleUpdate(pair_match.right.armor, pair_match.right_match, true,
             image_timestamp_us, candidate_debug);
  }

  if (pair_match.dz_valid)
  {
    rt_.model_pair_delta_z_valid = true;
  }
  ekf_.measurement_face_index = pair_match.tracked_face;
  const Eigen::Vector3d tracked_xyz(
      pair_match.tracked_armor.pose.translation.x(),
      pair_match.tracked_armor.pose.translation.y(),
      pair_match.tracked_armor.pose.translation.z());
  ekf_.measurement = Eigen::Vector4d(tracked_xyz.x(), tracked_xyz.y(),
                                     tracked_xyz.z(),
                                     pair_match.tracked_match.measured_yaw);
}

/**
 * @brief 按当前 dt 对 11 维整车状态和协方差执行预测。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::VehiclePredict()
{
  const double dt = std::max(1e-4, time_.dt);
  Eigen::MatrixXd f = Eigen::MatrixXd::Identity(11, 11);
  f(ExtendedKalmanFilter::X_CENTER, ExtendedKalmanFilter::V_X_CENTER) = dt;
  f(ExtendedKalmanFilter::Y_CENTER, ExtendedKalmanFilter::V_Y_CENTER) = dt;
  f(ExtendedKalmanFilter::Z_ARMOR, ExtendedKalmanFilter::V_Z_ARMOR) = dt;
  f(ExtendedKalmanFilter::YAW, ExtendedKalmanFilter::V_YAW) = dt;

  double linear_variance = 100.0;
  double angular_variance = 400.0;
  if (rt_.tracked_id == ArmorNumber::OUTPOST)
  {
    linear_variance = 10.0;
    angular_variance = 0.1;
  }
  linear_variance = std::max(
      0.0, armor_tracker_detail::ParseEnvDouble(
               "XR_TRACKER_MODEL_Q_XYZ", linear_variance));
  angular_variance = std::max(
      0.0, armor_tracker_detail::ParseEnvDouble(
               "XR_TRACKER_MODEL_Q_YAW", angular_variance));

  const double a = dt * dt * dt * dt / 4.0;
  const double b = dt * dt * dt / 2.0;
  const double c = dt * dt;
  Eigen::MatrixXd q = Eigen::MatrixXd::Zero(11, 11);
  q(0, 0) = a * linear_variance;
  q(0, 1) = b * linear_variance;
  q(1, 0) = b * linear_variance;
  q(1, 1) = c * linear_variance;
  q(2, 2) = a * linear_variance;
  q(2, 3) = b * linear_variance;
  q(3, 2) = b * linear_variance;
  q(3, 3) = c * linear_variance;
  q(4, 4) = a * linear_variance;
  q(4, 5) = b * linear_variance;
  q(5, 4) = b * linear_variance;
  q(5, 5) = c * linear_variance;
  q(6, 6) = a * angular_variance;
  q(6, 7) = b * angular_variance;
  q(7, 6) = b * angular_variance;
  q(7, 7) = c * angular_variance;
  q(ExtendedKalmanFilter::DELTA_R, ExtendedKalmanFilter::DELTA_R) =
      std::max(0.0, armor_tracker_detail::ParseEnvDouble(
                        "XR_TRACKER_MODEL_DELTA_R_Q", 1.0e-5));
  q(ExtendedKalmanFilter::DELTA_Z, ExtendedKalmanFilter::DELTA_Z) =
      VehicleDeltaZProcessVariance();

  ekf_.covariance = f * ekf_.covariance * f.transpose() + q;
  ekf_.state = f * ekf_.state;
  ekf_.state(ExtendedKalmanFilter::YAW) =
      VehicleLimitRad(ekf_.state(ExtendedKalmanFilter::YAW));

  const bool outpost_converged = rt_.tracked_id == ArmorNumber::OUTPOST &&
                                 rt_.update_count > 10 && !VehicleStateDiverged();
  if (outpost_converged &&
      std::abs(ekf_.state(ExtendedKalmanFilter::V_YAW)) > 2.0)
  {
    ekf_.state(ExtendedKalmanFilter::V_YAW) =
        ekf_.state(ExtendedKalmanFilter::V_YAW) > 0.0 ? 2.51 : -2.51;
  }
}

/**
 * @brief 用单个装甲观测更新整车 EKF，并填充可选 EKF 调试信息。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::VehicleUpdate(const ArmorDetectorResult& armor,
                                         const VehicleArmorMatch& match,
                                         bool freeze_delta_z,
                                         uint64_t image_timestamp_us,
                                         CandidateDebugMsg* candidate_debug)
{
  const Eigen::Vector3d raw_armor_xyz(armor.pose.translation.x(),
                                      armor.pose.translation.y(),
                                      armor.pose.translation.z());
  Eigen::Vector3d armor_xyz = raw_armor_xyz;
  const double raw_range = raw_armor_xyz.norm();
  bool range_clamped = false;
  // Keep the measured bearing, but reject same-face depth jumps that exceed
  // physically plausible target motion; PnP range is the noisiest component.
  if (std::isfinite(raw_range) && raw_range > 1e-6 && image_timestamp_us > 0)
  {
    if (rt_.model_range_filter_valid && rt_.model_range_filter_face == match.id &&
        image_timestamp_us > rt_.model_range_filter_timestamp_us)
    {
      const double dt =
          static_cast<double>(image_timestamp_us -
                              rt_.model_range_filter_timestamp_us) *
          1.0e-6;
      constexpr double kMinDt = 0.002;
      constexpr double kMaxDt = 0.080;
      if (dt >= kMinDt && dt <= kMaxDt)
      {
        constexpr double kMaxRangeRate = 2.0;
        constexpr double kBaseRangeStep = 0.03;
        const double max_delta = kBaseRangeStep + kMaxRangeRate * dt;
        const double delta = raw_range - rt_.model_range_filter_distance;
        if (std::abs(delta) > max_delta)
        {
          const double filtered_range =
              rt_.model_range_filter_distance + std::copysign(max_delta, delta);
          armor_xyz = raw_armor_xyz * (filtered_range / raw_range);
          range_clamped = true;
        }
      }
    }
    rt_.model_range_filter_valid = true;
    rt_.model_range_filter_timestamp_us = image_timestamp_us;
    rt_.model_range_filter_face = match.id;
    rt_.model_range_filter_distance = armor_xyz.norm();
  }
  ekf_.measurement_face_index = match.id;
  const double base_z_before_update = ekf_.state(ExtendedKalmanFilter::Z_ARMOR);
  const Eigen::Vector3d armor_ypd = VehicleXyzToYpd(armor_xyz);
  Eigen::MatrixXd h = VehicleObservationJacobian(ekf_.state, match.id);
  const Eigen::Vector3d predicted_xyz_before =
      VehicleArmorPosition(ekf_.state, match.id);
  const Eigen::Vector3d pre_residual_xyz = armor_xyz - predicted_xyz_before;
  const auto fill_update_debug =
      [this, candidate_debug, &armor_xyz, &pre_residual_xyz, &match,
       freeze_delta_z, raw_range, range_clamped](
          uint8_t update_mode, const Eigen::VectorXd& innovation,
          const Eigen::VectorXd& r_diag,
          const Eigen::MatrixXd& innovation_cov)
  {
    if (candidate_debug == nullptr)
    {
      return;
    }
    const Eigen::Vector3d post_residual_xyz =
        armor_xyz - VehicleArmorPosition(ekf_.state, match.id);
    double mahalanobis = 0.0;
    if (innovation.size() > 0 && innovation_cov.rows() == innovation.size() &&
        innovation_cov.cols() == innovation.size() && innovation.allFinite() &&
        innovation_cov.allFinite())
    {
      const Eigen::VectorXd solved = innovation_cov.ldlt().solve(innovation);
      if (solved.allFinite())
      {
        mahalanobis = innovation.dot(solved);
      }
    }

    candidate_debug->ekf_update_valid = 1;
    candidate_debug->ekf_update_mode = update_mode;
    candidate_debug->ekf_update_face = static_cast<int8_t>(match.id);
    candidate_debug->ekf_freeze_delta_z = freeze_delta_z ? 1 : 0;
    candidate_debug->ekf_range_clamped = range_clamped ? 1 : 0;
    candidate_debug->ekf_raw_range_m = static_cast<float>(raw_range);
    candidate_debug->ekf_range_m = static_cast<float>(armor_xyz.norm());
    candidate_debug->ekf_mahalanobis = static_cast<float>(mahalanobis);
    candidate_debug->ekf_pre_res_x = static_cast<float>(pre_residual_xyz.x());
    candidate_debug->ekf_pre_res_y = static_cast<float>(pre_residual_xyz.y());
    candidate_debug->ekf_pre_res_z = static_cast<float>(pre_residual_xyz.z());
    candidate_debug->ekf_pre_res_norm =
        static_cast<float>(pre_residual_xyz.norm());
    candidate_debug->ekf_post_res_x = static_cast<float>(post_residual_xyz.x());
    candidate_debug->ekf_post_res_y = static_cast<float>(post_residual_xyz.y());
    candidate_debug->ekf_post_res_z = static_cast<float>(post_residual_xyz.z());
    candidate_debug->ekf_post_res_norm =
        static_cast<float>(post_residual_xyz.norm());
    candidate_debug->ekf_innov_0 =
        innovation.size() > 0 ? static_cast<float>(innovation[0]) : 0.0F;
    candidate_debug->ekf_innov_1 =
        innovation.size() > 1 ? static_cast<float>(innovation[1]) : 0.0F;
    candidate_debug->ekf_innov_2 =
        innovation.size() > 2 ? static_cast<float>(innovation[2]) : 0.0F;
    candidate_debug->ekf_innov_3 =
        innovation.size() > 3 ? static_cast<float>(innovation[3]) : 0.0F;
    candidate_debug->ekf_r_0 =
        r_diag.size() > 0 ? static_cast<float>(r_diag[0]) : 0.0F;
    candidate_debug->ekf_r_1 =
        r_diag.size() > 1 ? static_cast<float>(r_diag[1]) : 0.0F;
    candidate_debug->ekf_r_2 =
        r_diag.size() > 2 ? static_cast<float>(r_diag[2]) : 0.0F;
    candidate_debug->ekf_r_3 =
        r_diag.size() > 3 ? static_cast<float>(r_diag[3]) : 0.0F;
  };
  const double dz_before_update =
      ekf_.state(ExtendedKalmanFilter::DELTA_Z);
  const double dz_variance_before_update =
      ekf_.covariance(ExtendedKalmanFilter::DELTA_Z,
                      ExtendedKalmanFilter::DELTA_Z);
  if (freeze_delta_z && rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    h.col(ExtendedKalmanFilter::DELTA_Z).setZero();
  }
  if (rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
      !VehicleXyzMeasurementFullGeometryEnabled())
  {
    // 单装甲板观测无法区分整车中心平移和半径误差；半径只由多装甲几何更新。
    h.col(ExtendedKalmanFilter::ROBOT_R).setZero();
    h.col(ExtendedKalmanFilter::DELTA_R).setZero();
  }

  if (VehicleXyzMeasurementUpdateEnabled())
  {
    const int armor_count =
        std::max(1, static_cast<int>(rt_.tracked_armors_num));
    const double angle = VehicleLimitRad(
        ekf_.state(ExtendedKalmanFilter::YAW) +
        match.id * 2.0 * M_PI / armor_count);
    const bool odd_face =
        rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
        (match.id == 1 || match.id == 3);
    const double radius =
        ekf_.state(ExtendedKalmanFilter::ROBOT_R) +
        (odd_face ? ekf_.state(ExtendedKalmanFilter::DELTA_R) : 0.0);

    Eigen::MatrixXd h_xyz = Eigen::MatrixXd::Zero(4, 11);
    h_xyz(0, ExtendedKalmanFilter::X_CENTER) = 1.0;
    h_xyz(0, ExtendedKalmanFilter::YAW) = -radius * std::sin(angle);
    h_xyz(1, ExtendedKalmanFilter::Y_CENTER) = 1.0;
    h_xyz(1, ExtendedKalmanFilter::YAW) = radius * std::cos(angle);
    h_xyz(2, ExtendedKalmanFilter::Z_ARMOR) = 1.0;
    h_xyz(3, ExtendedKalmanFilter::YAW) = 1.0;

    if (VehicleXyzMeasurementFullGeometryEnabled())
    {
      h_xyz(0, ExtendedKalmanFilter::ROBOT_R) = std::cos(angle);
      h_xyz(1, ExtendedKalmanFilter::ROBOT_R) = std::sin(angle);
      if (odd_face)
      {
        h_xyz(0, ExtendedKalmanFilter::DELTA_R) = std::cos(angle);
        h_xyz(1, ExtendedKalmanFilter::DELTA_R) = std::sin(angle);
      }
    }
    if (odd_face && !freeze_delta_z)
    {
      h_xyz(2, ExtendedKalmanFilter::DELTA_Z) = 1.0;
    }

    const double range = std::max(1e-6, armor_xyz.norm());
    const double detector_variance_scale =
        armor_tracker::DetectorObservationVarianceScale(armor);
    const double quality_alpha =
        ObservationQualityEnabled()
            ? std::clamp(1.0 - 0.35 *
                                   armor_tracker::ArmorObservationQualityPenalty(
                                       armor, ObservationStableMaxReprojectionPx(),
                                       ObservationStableMinAreaPx(),
                                       ObservationStableMinConfidence()),
                         0.45, 1.0)
            : 1.0;
    const double position_sigma =
        std::sqrt(detector_variance_scale) *
        std::max(0.005, VehicleXyzMeasurementRFactor(cfg_.noise.r_xyz_factor) * range);
    const double effective_position_sigma = position_sigma / quality_alpha;
    const double effective_yaw_variance =
        detector_variance_scale *
        VehicleXyzMeasurementYawVariance(cfg_.noise.r_yaw) /
        (quality_alpha * quality_alpha);
    Eigen::VectorXd r_diag(4);
    r_diag << effective_position_sigma * effective_position_sigma,
        effective_position_sigma * effective_position_sigma,
        effective_position_sigma * effective_position_sigma,
        effective_yaw_variance;

    Eigen::VectorXd z(4);
    z << armor_xyz.x(), armor_xyz.y(), armor_xyz.z(), match.measured_yaw;
    Eigen::VectorXd predicted(4);
    predicted << predicted_xyz_before.x(), predicted_xyz_before.y(),
        predicted_xyz_before.z(), angle;
    Eigen::VectorXd innovation = z - predicted;
    innovation[3] = VehicleLimitRad(innovation[3]);

    const Eigen::MatrixXd r = r_diag.asDiagonal();
    const Eigen::MatrixXd innovation_cov =
        h_xyz * ekf_.covariance * h_xyz.transpose() + r;
    const Eigen::MatrixXd kalman_gain =
        ekf_.covariance * h_xyz.transpose() * innovation_cov.inverse();
    const Eigen::MatrixXd identity =
        Eigen::MatrixXd::Identity(ekf_.covariance.rows(), ekf_.covariance.cols());
    ekf_.state = ekf_.state + kalman_gain * innovation;
    ekf_.state(ExtendedKalmanFilter::YAW) =
        VehicleLimitRad(ekf_.state(ExtendedKalmanFilter::YAW));
    if (freeze_delta_z && rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
    {
      ekf_.state(ExtendedKalmanFilter::DELTA_Z) = dz_before_update;
    }
    const bool direct_dz_update =
        VehicleDirectDeltaZEnabled() && rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
        !freeze_delta_z && odd_face;
    if (direct_dz_update)
    {
      const double alpha = VehicleDirectDeltaZAlpha();
      const double max_abs_dz = VehicleDirectDeltaZMaxAbs();
      const double observed_dz =
          std::clamp(armor_xyz.z() - base_z_before_update, -max_abs_dz, max_abs_dz);
      ekf_.state(ExtendedKalmanFilter::DELTA_Z) =
          (1.0 - alpha) * ekf_.state(ExtendedKalmanFilter::DELTA_Z) +
          alpha * observed_dz;
    }
    if (rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
        !SymmetricGeometryEnabled())
    {
      ekf_.state(ExtendedKalmanFilter::DELTA_R) *=
          (1.0 - VehicleDeltaRadiusShrinkAlpha());
    }
    ekf_.covariance = (identity - kalman_gain * h_xyz) * ekf_.covariance *
                          (identity - kalman_gain * h_xyz).transpose() +
                      kalman_gain * r * kalman_gain.transpose();
    if (freeze_delta_z && rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
    {
      const int dz_index = ExtendedKalmanFilter::DELTA_Z;
      ekf_.covariance.row(dz_index).setZero();
      ekf_.covariance.col(dz_index).setZero();
      ekf_.covariance(dz_index, dz_index) = dz_variance_before_update;
    }
    ekf_.measurement =
        Eigen::Vector4d(armor_xyz.x(), armor_xyz.y(), armor_xyz.z(),
                        match.measured_yaw);
    fill_update_debug(2, innovation, r_diag, innovation_cov);
    return;
  }

  const double center_yaw = std::atan2(armor_xyz.y(), armor_xyz.x());
  const double delta_angle = VehicleLimitRad(match.measured_yaw - center_yaw);
  const double detector_variance_scale =
      armor_tracker::DetectorObservationVarianceScale(armor);
  Eigen::VectorXd r_diag(4);
  r_diag << detector_variance_scale * 4e-3,
      detector_variance_scale * 4e-3 * VehiclePitchVarianceScale(),
      detector_variance_scale *
          (std::log(std::abs(delta_angle) + 1.0) + 1.0) *
          VehicleYpdDistanceVarianceScale(),
      detector_variance_scale *
          (std::log(std::abs(armor_ypd.z()) + 1.0) / 200.0 + 9e-2) *
          VehicleYpdArmorYawVarianceScale();
  const Eigen::MatrixXd r = r_diag.asDiagonal();

  auto observe = [this, &match](const Eigen::VectorXd& state)
  {
    const Eigen::Vector3d xyz = VehicleArmorPosition(state, match.id);
    const Eigen::Vector3d ypd = VehicleXyzToYpd(xyz);
    const int armor_count =
        std::max(1, static_cast<int>(rt_.tracked_armors_num));
    const double angle =
        VehicleLimitRad(state(ExtendedKalmanFilter::YAW) +
                   match.id * 2.0 * M_PI / armor_count);
    Eigen::VectorXd out(4);
    out << ypd.x(), ypd.y(), ypd.z(), angle;
    return out;
  };
  auto subtract = [](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs)
  {
    Eigen::VectorXd result = lhs - rhs;
    result[0] = ArmorTracker<CameraInfoV>::VehicleLimitRad(result[0]);
    result[1] = ArmorTracker<CameraInfoV>::VehicleLimitRad(result[1]);
    result[3] = ArmorTracker<CameraInfoV>::VehicleLimitRad(result[3]);
    return result;
  };
  Eigen::VectorXd z(4);
  z << armor_ypd.x(), armor_ypd.y(), armor_ypd.z(), match.measured_yaw;

  const Eigen::MatrixXd innovation_cov =
      h * ekf_.covariance * h.transpose() + r;
  const Eigen::MatrixXd kalman_gain =
      ekf_.covariance * h.transpose() * innovation_cov.inverse();
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(ekf_.covariance.rows(), ekf_.covariance.cols());
  const Eigen::VectorXd innovation = subtract(z, observe(ekf_.state));
  ekf_.state = ekf_.state + kalman_gain * innovation;
  ekf_.state(ExtendedKalmanFilter::YAW) =
      VehicleLimitRad(ekf_.state(ExtendedKalmanFilter::YAW));
  if (freeze_delta_z && rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    ekf_.state(ExtendedKalmanFilter::DELTA_Z) = dz_before_update;
  }
  const bool direct_dz_update =
      VehicleDirectDeltaZEnabled() && rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
      !freeze_delta_z &&
      (match.id == 1 || match.id == 3);
  if (direct_dz_update)
  {
    const double alpha = VehicleDirectDeltaZAlpha();
    const double max_abs_dz = VehicleDirectDeltaZMaxAbs();
    const double observed_dz =
        std::clamp(armor_xyz.z() - base_z_before_update, -max_abs_dz, max_abs_dz);
    ekf_.state(ExtendedKalmanFilter::DELTA_Z) =
        (1.0 - alpha) * ekf_.state(ExtendedKalmanFilter::DELTA_Z) +
        alpha * observed_dz;
  }
  if (rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
      !SymmetricGeometryEnabled())
  {
    ekf_.state(ExtendedKalmanFilter::DELTA_R) *=
        (1.0 - VehicleDeltaRadiusShrinkAlpha());
  }
  ekf_.covariance = (identity - kalman_gain * h) * ekf_.covariance *
                        (identity - kalman_gain * h).transpose() +
                    kalman_gain * r * kalman_gain.transpose();
  if (freeze_delta_z && rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    const int dz_index = ExtendedKalmanFilter::DELTA_Z;
    ekf_.covariance.row(dz_index).setZero();
    ekf_.covariance.col(dz_index).setZero();
    ekf_.covariance(dz_index, dz_index) = dz_variance_before_update;
  }

  double recenter_alpha = VehicleMeasurementRecenterAlpha();
  if (recenter_alpha > 0.0 && ObservationQualityEnabled())
  {
    const double quality_penalty =
        armor_tracker::ArmorObservationQualityPenalty(
            armor, ObservationStableMaxReprojectionPx(),
            ObservationStableMinAreaPx(), ObservationStableMinConfidence());
    recenter_alpha *= std::clamp(1.0 - 0.35 * quality_penalty, 0.45, 1.0);
  }
  if (recenter_alpha > 0.0 && VehicleMeasurementRecenterQualityEnabled())
  {
    const auto quality_ramp = [](double value, double good, double bad)
    {
      if (bad <= good + 1e-9)
      {
        return value <= good ? 1.0 : 0.0;
      }
      return std::clamp((bad - value) / (bad - good), 0.0, 1.0);
    };

    const double quality = std::min(
        {quality_ramp(match.score, VehicleMeasurementRecenterScoreGood(),
                      VehicleMeasurementRecenterScoreBad()),
         quality_ramp(match.angle_error, VehicleMeasurementRecenterYawGood(),
                      VehicleMeasurementRecenterYawBad()),
         quality_ramp(match.xyz_error, VehicleMeasurementRecenterXyzGood(),
                      VehicleMeasurementRecenterXyzBad())});
    const double alpha_bad = VehicleMeasurementRecenterAlphaBad();
    const double alpha_good = VehicleMeasurementRecenterAlphaGood();
    recenter_alpha =
        std::clamp(alpha_bad + (alpha_good - alpha_bad) * quality, 0.0, 1.0);
  }
  if (recenter_alpha > 0.0 && rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    const int armor_count =
        std::max(1, static_cast<int>(rt_.tracked_armors_num));
    const double angle_step = 2.0 * M_PI / armor_count;
    const bool odd_face = (match.id == 1 || match.id == 3);
    const double radius =
        ekf_.state(ExtendedKalmanFilter::ROBOT_R) +
        (odd_face ? ekf_.state(ExtendedKalmanFilter::DELTA_R) : 0.0);
    const double anchor_yaw = armor_tracker::UnwrapYawNear(
        match.measured_yaw - match.id * angle_step,
        ekf_.state(ExtendedKalmanFilter::YAW));
    const double anchor_x =
        armor_xyz.x() - radius * std::cos(match.measured_yaw);
    const double anchor_y =
        armor_xyz.y() - radius * std::sin(match.measured_yaw);
    const double anchor_z =
        armor_xyz.z() -
        (odd_face ? ekf_.state(ExtendedKalmanFilter::DELTA_Z) : 0.0);

    const double correction_x =
        recenter_alpha *
        (anchor_x - ekf_.state(ExtendedKalmanFilter::X_CENTER));
    const double correction_y =
        recenter_alpha *
        (anchor_y - ekf_.state(ExtendedKalmanFilter::Y_CENTER));
    const double correction_z =
        recenter_alpha *
        (anchor_z - ekf_.state(ExtendedKalmanFilter::Z_ARMOR));
    const double correction_yaw =
        recenter_alpha *
        VehicleLimitRad(anchor_yaw - ekf_.state(ExtendedKalmanFilter::YAW));
    ekf_.state(ExtendedKalmanFilter::X_CENTER) += correction_x;
    ekf_.state(ExtendedKalmanFilter::Y_CENTER) += correction_y;
    ekf_.state(ExtendedKalmanFilter::Z_ARMOR) += correction_z;
    ekf_.state(ExtendedKalmanFilter::YAW) =
        VehicleLimitRad(ekf_.state(ExtendedKalmanFilter::YAW) + correction_yaw);
  }

  const double position_anchor_alpha = VehicleMeasurementPositionAnchorAlpha();
  if (position_anchor_alpha > 0.0 &&
      rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    const double xyz_bad = VehicleMeasurementPositionAnchorXyzBad();
    const double quality =
        std::clamp((xyz_bad - match.xyz_error) / xyz_bad, 0.0, 1.0);
    const double alpha = position_anchor_alpha * quality;
    if (alpha > 0.0)
    {
      const double quality_alpha =
          ObservationQualityEnabled()
              ? std::clamp(1.0 - 0.35 *
                                     armor_tracker::ArmorObservationQualityPenalty(
                                         armor, ObservationStableMaxReprojectionPx(),
                                         ObservationStableMinAreaPx(),
                                         ObservationStableMinConfidence()),
                           0.45, 1.0)
              : 1.0;
      const double effective_alpha = alpha * quality_alpha;
      const int armor_count =
          std::max(1, static_cast<int>(rt_.tracked_armors_num));
      const double angle_step = 2.0 * M_PI / armor_count;
      const bool odd_face = (match.id == 1 || match.id == 3);
      const double face_yaw =
          VehicleLimitRad(ekf_.state(ExtendedKalmanFilter::YAW) +
                     match.id * angle_step);
      const double radius =
          ekf_.state(ExtendedKalmanFilter::ROBOT_R) +
          (odd_face ? ekf_.state(ExtendedKalmanFilter::DELTA_R) : 0.0);
      const double z_offset =
          odd_face ? ekf_.state(ExtendedKalmanFilter::DELTA_Z) : 0.0;
      const double anchor_x = armor_xyz.x() - radius * std::cos(face_yaw);
      const double anchor_y = armor_xyz.y() - radius * std::sin(face_yaw);
      const double anchor_z = armor_xyz.z() - z_offset;

      ekf_.state(ExtendedKalmanFilter::X_CENTER) +=
          effective_alpha * (anchor_x - ekf_.state(ExtendedKalmanFilter::X_CENTER));
      ekf_.state(ExtendedKalmanFilter::Y_CENTER) +=
          effective_alpha * (anchor_y - ekf_.state(ExtendedKalmanFilter::Y_CENTER));
      ekf_.state(ExtendedKalmanFilter::Z_ARMOR) +=
          effective_alpha * (anchor_z - ekf_.state(ExtendedKalmanFilter::Z_ARMOR));
    }
  }

  ekf_.measurement =
      Eigen::Vector4d(armor_xyz.x(), armor_xyz.y(), armor_xyz.z(),
                      match.measured_yaw);
  fill_update_debug(1, innovation, r_diag, innovation_cov);
}

/**
 * @brief 基于测量后中心位置差分维护输出速度观测器。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::VehicleUpdateCenterMotionObserver(
    const ArmorDetectorResult& armor, const VehicleArmorMatch& match,
    uint64_t image_timestamp_us)
{
  (void)armor;
  (void)match;
  if (rt_.tracked_id == ArmorNumber::OUTPOST ||
      ekf_.state.size() <= ExtendedKalmanFilter::DELTA_Z)
  {
    return;
  }

  const Eigen::Vector3d anchor(
      ekf_.state(ExtendedKalmanFilter::X_CENTER),
      ekf_.state(ExtendedKalmanFilter::Y_CENTER),
      ekf_.state(ExtendedKalmanFilter::Z_ARMOR));

  if (!anchor.allFinite())
  {
    return;
  }

  if (!rt_.center_motion_observer_valid ||
      image_timestamp_us <= rt_.center_motion_observer_timestamp_us)
  {
    rt_.center_motion_observer_valid = true;
    rt_.center_motion_observer_timestamp_us = image_timestamp_us;
    rt_.center_motion_observer_anchor = anchor;
    rt_.center_motion_observer_velocity =
        Eigen::Vector3d(ekf_.state(ExtendedKalmanFilter::V_X_CENTER),
                        ekf_.state(ExtendedKalmanFilter::V_Y_CENTER),
                        ekf_.state(ExtendedKalmanFilter::V_Z_ARMOR));
    rt_.center_motion_observer_raw_velocity =
        rt_.center_motion_observer_velocity;
    rt_.center_motion_observer_confidence = 0.0;
    rt_.center_motion_observer_samples = 0;
    return;
  }

  const double dt =
      static_cast<double>(image_timestamp_us -
                          rt_.center_motion_observer_timestamp_us) *
      1.0e-6;
  constexpr double kMinDt = 0.002;
  constexpr double kMaxDt = 0.080;
  if (dt < kMinDt || dt > kMaxDt)
  {
    rt_.center_motion_observer_timestamp_us = image_timestamp_us;
    rt_.center_motion_observer_anchor = anchor;
    rt_.center_motion_observer_velocity =
        Eigen::Vector3d(ekf_.state(ExtendedKalmanFilter::V_X_CENTER),
                        ekf_.state(ExtendedKalmanFilter::V_Y_CENTER),
                        ekf_.state(ExtendedKalmanFilter::V_Z_ARMOR));
    rt_.center_motion_observer_raw_velocity =
        rt_.center_motion_observer_velocity;
    rt_.center_motion_observer_confidence = 0.0;
    rt_.center_motion_observer_samples = 0;
    return;
  }

  Eigen::Vector3d raw_velocity =
      (anchor - rt_.center_motion_observer_anchor) / dt;
  if (!raw_velocity.allFinite())
  {
    rt_.center_motion_observer_confidence = 0.0;
    return;
  }
  constexpr double kMaxObservedSpeed = 4.0;
  const double raw_speed = raw_velocity.head<2>().norm();
  const bool raw_speed_clamped = raw_speed > kMaxObservedSpeed;
  if (raw_speed > kMaxObservedSpeed)
  {
    raw_velocity.x() *= kMaxObservedSpeed / raw_speed;
    raw_velocity.y() *= kMaxObservedSpeed / raw_speed;
  }
  rt_.center_motion_observer_raw_velocity = raw_velocity;

  const Eigen::Vector2d anchor_xy(anchor.x(), anchor.y());
  const double range_xy = anchor_xy.norm();
  if (!std::isfinite(range_xy) || range_xy < 1e-6)
  {
    rt_.center_motion_observer_timestamp_us = image_timestamp_us;
    rt_.center_motion_observer_anchor = anchor;
    rt_.center_motion_observer_confidence = 0.0;
    rt_.center_motion_observer_samples = 0;
    return;
  }

  const Eigen::Vector2d radial = anchor_xy / range_xy;
  const Eigen::Vector2d tangential(-radial.y(), radial.x());
  const Eigen::Vector2d raw_xy(raw_velocity.x(), raw_velocity.y());
  const Eigen::Vector2d previous_xy(rt_.center_motion_observer_velocity.x(),
                                    rt_.center_motion_observer_velocity.y());

  // Use the post-update center as a measurement-derived anchor for output
  // velocity only; the EKF geometry state remains measurement-driven.
  constexpr double kTangentialTau = 0.020;
  constexpr double kVerticalTau = 0.100;
  const double tangential_alpha = dt / (kTangentialTau + dt);
  const double vertical_alpha = dt / (kVerticalTau + dt);

  const double tangential_speed =
      (1.0 - tangential_alpha) * previous_xy.dot(tangential) +
      tangential_alpha * raw_xy.dot(tangential);
  const double previous_tangential = previous_xy.dot(tangential);
  const double raw_tangential = raw_xy.dot(tangential);
  double radial_speed = previous_xy.dot(radial);
  if (VehicleCenterMotionObserverRadialVelocityEnabled())
  {
    radial_speed =
        (1.0 - tangential_alpha) * radial_speed +
        tangential_alpha * raw_xy.dot(radial);
  }
  else
  {
    radial_speed = 0.0;
  }
  const Eigen::Vector2d observed_xy =
      tangential_speed * tangential + radial_speed * radial;
  const double observed_z =
      (1.0 - vertical_alpha) * rt_.center_motion_observer_velocity.z() +
      vertical_alpha * raw_velocity.z();

  rt_.center_motion_observer_velocity =
      Eigen::Vector3d(observed_xy.x(), observed_xy.y(), observed_z);
  const double tangential_jump =
      std::abs(raw_tangential - previous_tangential);
  constexpr double kGoodTangentialJump = 0.25;
  constexpr double kBadTangentialJump = 1.20;
  double instant_confidence =
      std::clamp((kBadTangentialJump - tangential_jump) /
                     (kBadTangentialJump - kGoodTangentialJump),
                 0.0, 1.0);
  constexpr double kSignFlipSpeed = 0.10;
  if (previous_tangential * raw_tangential < 0.0 &&
      std::abs(previous_tangential) > kSignFlipSpeed &&
      std::abs(raw_tangential) > kSignFlipSpeed)
  {
    instant_confidence = 0.0;
  }
  if (raw_speed_clamped)
  {
    instant_confidence = 0.0;
  }
  if (rt_.center_motion_observer_samples < UINT32_MAX)
  {
    ++rt_.center_motion_observer_samples;
  }
  const double warmup =
      std::clamp(static_cast<double>(rt_.center_motion_observer_samples) / 4.0,
                 0.0, 1.0);
  instant_confidence *= warmup;
  const double old_confidence = rt_.center_motion_observer_confidence;
  const double confidence_alpha =
      instant_confidence < old_confidence ? 0.65 : 0.20;
  rt_.center_motion_observer_confidence =
      (1.0 - confidence_alpha) * old_confidence +
      confidence_alpha * instant_confidence;
  rt_.center_motion_observer_timestamp_us = image_timestamp_us;
  rt_.center_motion_observer_anchor = anchor;
}

/**
 * @brief 用当前可见面 yaw 差分结果修正发布消息中的 yaw 角速度。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::VehicleApplyYawRateObserver(
    double output_yaw, uint64_t image_timestamp_us,
    ArmorTrackerTarget& target_msg)
{
  (void)output_yaw;
  if (!VehicleYawRateObserverEnabled() || !target_msg.tracking ||
      target_msg.armors_num <= 1 || image_timestamp_us == 0 ||
      !std::isfinite(target_msg.v_yaw))
  {
    rt_.yaw_rate_observer_valid = false;
    rt_.yaw_rate_observer_samples = 0;
    return;
  }

  if (!target_msg.measured_face_valid || target_msg.measured_face_index < 0 ||
      target_msg.measured_face_index >= target_msg.armors_num ||
      !std::isfinite(target_msg.measured_face_yaw))
  {
    rt_.yaw_rate_observer_valid = false;
    rt_.yaw_rate_observer_samples = 0;
    return;
  }

  const double ekf_vyaw = target_msg.v_yaw;
  const double angle_step =
      2.0 * M_PI / std::max(1, target_msg.armors_num);
  const double measured_center_yaw =
      target_msg.measured_face_yaw -
      static_cast<double>(target_msg.measured_face_index) * angle_step;
  if (!rt_.yaw_rate_observer_valid ||
      image_timestamp_us <= rt_.yaw_rate_observer_timestamp_us)
  {
    rt_.yaw_rate_observer_valid = true;
    rt_.yaw_rate_observer_timestamp_us = image_timestamp_us;
    rt_.yaw_rate_observer_yaw =
        armor_tracker::UnwrapYawNear(measured_center_yaw, target_msg.yaw);
    rt_.yaw_rate_observer_value = ekf_vyaw;
    rt_.yaw_rate_observer_samples = 0;
    return;
  }

  const double dt =
      static_cast<double>(image_timestamp_us -
                          rt_.yaw_rate_observer_timestamp_us) *
      1.0e-6;
  constexpr double kMinDt = 0.002;
  constexpr double kMaxDt = 0.080;
  const double unwrapped_yaw =
      armor_tracker::UnwrapYawNear(measured_center_yaw,
                                   rt_.yaw_rate_observer_yaw);
  if (dt < kMinDt || dt > kMaxDt)
  {
    rt_.yaw_rate_observer_timestamp_us = image_timestamp_us;
    rt_.yaw_rate_observer_yaw = unwrapped_yaw;
    rt_.yaw_rate_observer_value = ekf_vyaw;
    rt_.yaw_rate_observer_samples = 0;
    return;
  }

  const double raw_yaw_rate =
      (unwrapped_yaw - rt_.yaw_rate_observer_yaw) / dt;
  rt_.yaw_rate_observer_timestamp_us = image_timestamp_us;
  rt_.yaw_rate_observer_yaw = unwrapped_yaw;
  if (!std::isfinite(raw_yaw_rate) ||
      std::abs(raw_yaw_rate) > VehicleYawRateObserverMaxRaw())
  {
    rt_.yaw_rate_observer_value = ekf_vyaw;
    rt_.yaw_rate_observer_samples = 0;
    return;
  }

  const double alpha =
      std::clamp(dt / (VehicleYawRateObserverTau() + dt), 0.0, 1.0);
  if (rt_.yaw_rate_observer_samples == 0)
  {
    rt_.yaw_rate_observer_value = raw_yaw_rate;
  }
  else
  {
    rt_.yaw_rate_observer_value =
        (1.0 - alpha) * rt_.yaw_rate_observer_value + alpha * raw_yaw_rate;
  }
  if (rt_.yaw_rate_observer_samples < UINT32_MAX)
  {
    ++rt_.yaw_rate_observer_samples;
  }

  const double observer_vyaw = rt_.yaw_rate_observer_value;
  if (rt_.yaw_rate_observer_samples < VehicleYawRateObserverMinSamples() ||
      !std::isfinite(observer_vyaw) ||
      std::abs(observer_vyaw - ekf_vyaw) > VehicleYawRateObserverMaxBlendDelta())
  {
    return;
  }

  const double blend = VehicleYawRateObserverBlend();
  target_msg.v_yaw = (1.0 - blend) * ekf_vyaw + blend * observer_vyaw;
}

/**
 * @brief 检查半径状态是否已经越过物理范围并需要重置。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehicleStateDiverged() const
{
  const double r1 = ekf_.state(ExtendedKalmanFilter::ROBOT_R);
  const double r2 = r1 + ekf_.state(ExtendedKalmanFilter::DELTA_R);
  return !(r1 > 0.05 && r1 < 0.5 && r2 > 0.05 && r2 < 0.5);
}
