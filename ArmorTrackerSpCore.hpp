#pragma once

// SP 风格整车状态模型和 EKF 更新路径。ArmorTracker 依赖 CameraInfo 模板参数，
// 因此这部分实现保留在头文件中。

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::SpLimitRad(double angle)
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

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::SpDetectorYawNear(
    const LibXR::Quaternion<double>& q, double reference_yaw)
{
  return armor_tracker::MeasuredArmorYawNear(q, reference_yaw);
}

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::SpDetectorYawNear(const ArmorDetectorResult& armor,
                                                    double reference_yaw)
{
  return SpDetectorYawNear(armor.pose.rotation, reference_yaw);
}

template <CameraTypes::CameraInfo CameraInfoV>
Eigen::Vector3d ArmorTracker<CameraInfoV>::SpXyzToYpd(const Eigen::Vector3d& xyz)
{
  const double x = xyz.x();
  const double y = xyz.y();
  const double z = xyz.z();
  const double xy_norm = std::hypot(x, y);
  return {std::atan2(y, x), std::atan2(z, xy_norm), xyz.norm()};
}

template <CameraTypes::CameraInfo CameraInfoV>
Eigen::MatrixXd ArmorTracker<CameraInfoV>::SpXyzToYpdJacobian(const Eigen::Vector3d& xyz)
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

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::SpIsBalanceArmor(const ArmorDetectorResult& armor)
{
  return armor.type == ArmorType::LARGE &&
         (armor.number == ArmorNumber::THREE ||
          armor.number == ArmorNumber::FOUR ||
          armor.number == ArmorNumber::FIVE);
}

template <CameraTypes::CameraInfo CameraInfoV>
int ArmorTracker<CameraInfoV>::SpArmorCountFor(const ArmorDetectorResult& armor)
{
  if (SpIsBalanceArmor(armor))
  {
    return 2;
  }
  if (armor.number == ArmorNumber::OUTPOST || armor.number == ArmorNumber::BASE)
  {
    return 3;
  }
  return 4;
}

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::SpInitialRadiusFor(
    const ArmorDetectorResult& armor) const
{
  if (SpIsBalanceArmor(armor))
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

template <CameraTypes::CameraInfo CameraInfoV>
Eigen::VectorXd ArmorTracker<CameraInfoV>::SpInitialP0DiagFor(
    const ArmorDetectorResult& armor)
{
  Eigen::VectorXd p0_diag(11);
  if (SpIsBalanceArmor(armor))
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
  if (SpArmorCountFor(armor) == 4)
  {
    p0_diag[ExtendedKalmanFilter::DELTA_Z] = SpDeltaZInitialVariance();
  }
  return p0_diag;
}

template <CameraTypes::CameraInfo CameraInfoV>
Eigen::Vector3d ArmorTracker<CameraInfoV>::SpArmorPosition(
    const Eigen::VectorXd& state, int id) const
{
  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  const double angle = SpLimitRad(state[ExtendedKalmanFilter::YAW] +
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

template <CameraTypes::CameraInfo CameraInfoV>
Eigen::MatrixXd ArmorTracker<CameraInfoV>::SpObservationJacobian(
    const Eigen::VectorXd& state, int id) const
{
  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  const double angle = SpLimitRad(state[ExtendedKalmanFilter::YAW] +
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
      SpXyzToYpdJacobian(SpArmorPosition(state, id));
  Eigen::MatrixXd h_armor_ypda(4, 4);
  h_armor_ypda << h_armor_ypd(0, 0), h_armor_ypd(0, 1),
      h_armor_ypd(0, 2), 0.0, h_armor_ypd(1, 0), h_armor_ypd(1, 1),
      h_armor_ypd(1, 2), 0.0, h_armor_ypd(2, 0), h_armor_ypd(2, 1),
      h_armor_ypd(2, 2), 0.0, 0.0, 0.0, 0.0, 1.0;
  return h_armor_ypda * h_armor_xyza;
}

template <CameraTypes::CameraInfo CameraInfoV>
typename ArmorTracker<CameraInfoV>::SpArmorMatch
ArmorTracker<CameraInfoV>::SpMatchArmorToFace(const ArmorDetectorResult& armor,
                                              const Eigen::VectorXd& state,
                                              int face_index) const
{
  constexpr double kMatchYawScale = 0.12;
  constexpr double kMatchPitchScale = 0.10;
  constexpr double kMatchDistanceScale = 0.30;
  constexpr double kMatchAngleScale = 0.45;
  constexpr double kMatchPositionScale = 0.18;

  const Eigen::Vector3d pred_xyz = SpArmorPosition(state, face_index);
  const Eigen::Vector3d pred_ypd = SpXyzToYpd(pred_xyz);
  const double pred_yaw = GetArmorYawFromState(state, face_index);
  const Eigen::Vector3d armor_xyz(armor.pose.translation.x(),
                                  armor.pose.translation.y(),
                                  armor.pose.translation.z());
  const double measured_yaw = SpDetectorYawNear(armor.pose.rotation, pred_yaw);

  SpArmorMatch match{};
  match.id = face_index;
  const Eigen::Vector3d armor_ypd = SpXyzToYpd(armor_xyz);
  match.yaw_error = std::abs(SpLimitRad(armor_ypd.x() - pred_ypd.x()));
  match.pitch_error = std::abs(SpLimitRad(armor_ypd.y() - pred_ypd.y()));
  match.distance_error = std::abs(armor_ypd.z() - pred_ypd.z());
  match.angle_error = std::abs(SpLimitRad(measured_yaw - pred_yaw));
  match.xyz_error = (armor_xyz - pred_xyz).norm();
  match.measured_yaw = measured_yaw;
  match.score = match.yaw_error / kMatchYawScale +
                match.pitch_error / kMatchPitchScale +
                match.distance_error / kMatchDistanceScale +
                match.angle_error / kMatchAngleScale +
                match.xyz_error / kMatchPositionScale;
  return match;
}

template <CameraTypes::CameraInfo CameraInfoV>
typename ArmorTracker<CameraInfoV>::SpArmorMatch
ArmorTracker<CameraInfoV>::SpMatchArmor(const ArmorDetectorResult& armor,
                                        const Eigen::VectorXd& state) const
{
  constexpr double kMatchSwitchPenalty = 0.35;

  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  SpArmorMatch best{};

  for (int id = 0; id < armor_count; ++id)
  {
    SpArmorMatch match = SpMatchArmorToFace(armor, state, id);
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

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::SpTryCanonicalizeInitialState(
    const ArmorDetectorResults& armors_msg, bool force)
{
  if (!SpCanonicalInitEnabled() || rt_.tracked_id == ArmorNumber::INVALID)
  {
    return false;
  }

  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  if (armor_count != 4)
  {
    return false;
  }

  struct Observation
  {
    std::size_t armor_index = 0;
    ArmorDetectorResult armor{};
    int image_track_id = -1;
    bool confirmed_image_track = false;
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

  const double radius = SpInitialRadiusFor(rt_.tracked_armor);
  const double angle_step = 2.0 * M_PI / armor_count;
  const double min_height = SpCanonicalInitMinHeight();
  const double max_abs_dz = SpCanonicalInitMaxAbsDz();
  const double max_score = SpCanonicalInitMaxScore();

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
    const Eigen::Vector3d armor_ypd = SpXyzToYpd(armor_xyz);
    const Eigen::Vector3d pred_xyz = SpArmorPosition(state, face_index);
    const Eigen::Vector3d pred_ypd = SpXyzToYpd(pred_xyz);
    const double pred_yaw = GetArmorYawFromState(state, face_index);
    const double measured_yaw =
        SpDetectorYawNear(observation.armor, pred_yaw);
    const double yaw_error = std::abs(SpLimitRad(armor_ypd.x() - pred_ypd.x()));
    const double pitch_error =
        std::abs(SpLimitRad(armor_ypd.y() - pred_ypd.y()));
    const double distance_error = std::abs(armor_ypd.z() - pred_ypd.z());
    const double angle_error = std::abs(SpLimitRad(measured_yaw - pred_yaw));
    const double xyz_error = (armor_xyz - pred_xyz).norm();
    return yaw_error / kMatchYawScale + pitch_error / kMatchPitchScale +
           distance_error / kMatchDistanceScale +
           angle_error / kMatchAngleScale + xyz_error / kMatchPositionScale;
  };

  struct BestHypothesis
  {
    bool valid = false;
    bool has_height = false;
    bool positive_dz = false;
    double score = std::numeric_limits<double>::infinity();
    double dz = 0.0;
    double yaw = 0.0;
    int tracked_face = 0;
    std::size_t tracked_observation = 0;
    Eigen::VectorXd state = Eigen::VectorXd::Zero(11);
    std::array<int, 4> faces{{0, 1, 2, 3}};
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
          SpDetectorYawNear(anchor.armor, anchor_face * angle_step);
      const double seed_yaw =
          SpLimitRad(anchor_yaw - anchor_face * angle_step);
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
              SpDetectorYawNear(observation.armor, pred_yaw);
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
        const double yaw = SpLimitRad(yaw_sum * inv_count);
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
        if (has_height && !positive_dz && SpCanonicalInitPreferPositiveDz())
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
  ekf_.covariance = SpInitialP0DiagFor(rt_.tracked_armor).asDiagonal();
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
      "SP canonical init: force=%d obs=%zu score=%.3f face=%d dz=%.4f yaw=%.3f",
      force ? 1 : 0, observations.size(), best.score, best.tracked_face,
      best.dz, best.yaw);
  return true;
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::SpSolvePairGeometry(
    const SpPairObservation& left, int left_face, double left_measured_yaw,
    const SpPairObservation& right, int right_face, double right_measured_yaw,
    const Eigen::VectorXd& state, SpPairGeometryFit& fit) const
{
  fit = SpPairGeometryFit{};
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
  if (!std::isfinite(det) || std::abs(det) < SpPairGeometryMinDeterminant())
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
  if (!std::isfinite(fit_error) || fit_error > SpPairGeometryMaxFitError())
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
      center_shift > SpPairGeometryMaxCenterShift() ||
      radius_shift > SpPairGeometryMaxRadiusShift())
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
  const double yaw = SpLimitRad(0.5 * (left_yaw0 + right_yaw0));

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

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::SpResolvePairMatch(
    const ArmorDetectorResults& armors_msg, const Eigen::VectorXd& state,
    SpPairMatch& pair_match) const
{
  constexpr double kPairFaceSwitchPenalty = 0.75;
  constexpr double kPairCenterConsistencyScale = 0.10;
  constexpr double kPairMaxCenterSplit = 0.20;
  constexpr double kPairGeometryFitScale = 0.03;
  constexpr double kPairGeometryCenterScale = 0.35;
  constexpr double kPairGeometryRadiusScale = 0.18;
  constexpr double kPairMaxScore = 4.5;
  constexpr double kPairMaxXyzError = 0.45;

  pair_match = SpPairMatch{};
  const bool pair_geometry_enabled = SpPairGeometryEnabled();
  const bool pair_dz_enabled = SpPairDeltaZEnabled();
  if ((!pair_geometry_enabled && !pair_dz_enabled) ||
      rt_.tracked_id == ArmorNumber::INVALID ||
      rt_.tracked_armors_num != ArmorsNum::NORMAL_4)
  {
    return false;
  }

  std::vector<SpPairObservation> observations;
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
  SpPairMatch best{};

  for (std::size_t lhs = 0; lhs < observations.size(); ++lhs)
  {
    for (std::size_t rhs = lhs + 1; rhs < observations.size(); ++rhs)
    {
      const SpPairObservation* left = &observations[lhs];
      const SpPairObservation* right = &observations[rhs];
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
              SpLimitRad(current_yaw + left_face * angle_step);
          const double right_ref_yaw =
              SpLimitRad(current_yaw + right_face * angle_step);
          const double left_measured_yaw =
              SpDetectorYawNear(left->armor, left_ref_yaw);
          const double right_measured_yaw =
              SpDetectorYawNear(right->armor, right_ref_yaw);
          const double left_yaw0 = armor_tracker::UnwrapYawNear(
              left_measured_yaw - left_face * angle_step, current_yaw);
          const double right_yaw0 = armor_tracker::UnwrapYawNear(
              right_measured_yaw - right_face * angle_step, left_yaw0);
          const double yaw = SpLimitRad(0.5 * (left_yaw0 + right_yaw0));

          SpPairGeometryFit geometry{};
          const bool geometry_valid =
              pair_geometry_enabled &&
              SpSolvePairGeometry(*left, left_face, left_measured_yaw, *right,
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
              odd_z - even_z, -SpPairDeltaZMaxAbs(), SpPairDeltaZMaxAbs());
          const bool dz_valid =
              pair_dz_enabled && std::abs(observed_dz) >= SpPairDeltaZMinHeight();

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

          const SpArmorMatch left_match =
              SpMatchArmorToFace(left->armor, candidate, left_face);
          const SpArmorMatch right_match =
              SpMatchArmorToFace(right->armor, candidate, right_face);
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
                                       const SpPairObservation& observation,
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
              const double angle = SpLimitRad(
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
          const double yaw_delta = std::abs(SpLimitRad(
              candidate(ExtendedKalmanFilter::YAW) - current_yaw));
          const double best_yaw_delta =
              best.valid ? std::abs(SpLimitRad(best.yaw - current_yaw))
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

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SpApplyPairGeometryUpdate(
    const SpPairMatch& pair_match)
{
  if (!pair_match.geometry_valid && !pair_match.dz_valid)
  {
    return;
  }

  constexpr int kMaxRows = 7;
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(kMaxRows, 11);
  Eigen::VectorXd z = Eigen::VectorXd::Zero(kMaxRows);
  Eigen::VectorXd r_diag = Eigen::VectorXd::Zero(kMaxRows);
  int rows = 0;
  int yaw_row = -1;

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
    const double covariance_floor = SpPairGeometryCovarianceFloor();
    const int geometry_indices[] = {
        ExtendedKalmanFilter::X_CENTER, ExtendedKalmanFilter::Y_CENTER,
        ExtendedKalmanFilter::YAW, ExtendedKalmanFilter::ROBOT_R,
        ExtendedKalmanFilter::DELTA_R};
    for (const int index : geometry_indices)
    {
      ekf_.covariance(index, index) =
          std::max(ekf_.covariance(index, index), covariance_floor);
    }

    add_scalar(ExtendedKalmanFilter::X_CENTER, pair_match.geometry.center.x(),
               SpPairGeometryCenterVariance());
    add_scalar(ExtendedKalmanFilter::Y_CENTER, pair_match.geometry.center.y(),
               SpPairGeometryCenterVariance());

    h(rows, ExtendedKalmanFilter::YAW) = 1.0;
    z(rows) = pair_match.geometry.yaw;
    r_diag(rows) = SpPairGeometryYawVariance();
    yaw_row = rows;
    ++rows;

    add_scalar(ExtendedKalmanFilter::ROBOT_R, pair_match.geometry.r_even,
               SpPairGeometryRadiusVariance());
    if (!SymmetricGeometryEnabled())
    {
      h(rows, ExtendedKalmanFilter::ROBOT_R) = 1.0;
      h(rows, ExtendedKalmanFilter::DELTA_R) = 1.0;
      z(rows) = pair_match.geometry.r_odd;
      r_diag(rows) = SpPairGeometryRadiusVariance();
      ++rows;
    }
  }

  if (pair_match.dz_valid)
  {
    const int z_index = ExtendedKalmanFilter::Z_ARMOR;
    const int dz_index = ExtendedKalmanFilter::DELTA_Z;
    const double dz_variance = SpPairDeltaZVariance();
    ekf_.covariance(z_index, z_index) =
        std::max(ekf_.covariance(z_index, z_index), dz_variance);
    ekf_.covariance(dz_index, dz_index) =
        std::max(ekf_.covariance(dz_index, dz_index), dz_variance);
    add_scalar(z_index, pair_match.even_z_observed,
               SpPairGeometryCenterVariance());
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
  if (yaw_row >= 0)
  {
    innovation(yaw_row) = SpLimitRad(innovation(yaw_row));
  }

  const Eigen::MatrixXd r = r_diag.asDiagonal();
  const Eigen::MatrixXd innovation_cov =
      h * ekf_.covariance * h.transpose() + r;
  const Eigen::MatrixXd kalman_gain =
      ekf_.covariance * h.transpose() * innovation_cov.inverse();
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(ekf_.covariance.rows(), ekf_.covariance.cols());

  ekf_.state = ekf_.state + kalman_gain * innovation;
  ekf_.state(ExtendedKalmanFilter::YAW) =
      SpLimitRad(ekf_.state(ExtendedKalmanFilter::YAW));
  ekf_.covariance = (identity - kalman_gain * h) * ekf_.covariance *
                        (identity - kalman_gain * h).transpose() +
                    kalman_gain * r * kalman_gain.transpose();
  ClampGeometryState();
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SpUpdatePair(const SpPairMatch& pair_match)
{
  // 单装甲板只能约束“当前可见板”；双板几何才约束整车中心和半径。
  SpApplyPairGeometryUpdate(pair_match);

  if (SpPairDualUpdateEnabled())
  {
    SpUpdate(pair_match.left.armor, pair_match.left_match, true);
    SpUpdate(pair_match.right.armor, pair_match.right_match, true);
  }
  else if (pair_match.tracked_face == pair_match.left_face)
  {
    SpUpdate(pair_match.left.armor, pair_match.left_match, true);
  }
  else
  {
    SpUpdate(pair_match.right.armor, pair_match.right_match, true);
  }

  if (pair_match.dz_valid)
  {
    rt_.sp_pair_delta_z_valid = true;
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

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SpPredict()
{
  const double dt = std::max(1e-4, time_.dt);
  Eigen::MatrixXd f = Eigen::MatrixXd::Identity(11, 11);
  f(ExtendedKalmanFilter::X_CENTER, ExtendedKalmanFilter::V_X_CENTER) = dt;
  f(ExtendedKalmanFilter::Y_CENTER, ExtendedKalmanFilter::V_Y_CENTER) = dt;
  f(ExtendedKalmanFilter::Z_ARMOR, ExtendedKalmanFilter::V_Z_ARMOR) = dt;
  f(ExtendedKalmanFilter::YAW, ExtendedKalmanFilter::V_YAW) = dt;

  double linear_variance = 300.0;
  double angular_variance = 400.0;
  if (rt_.tracked_id == ArmorNumber::OUTPOST)
  {
    linear_variance = 10.0;
    angular_variance = 0.1;
  }
  linear_variance = std::max(
      0.0, armor_tracker_detail::ParseEnvDouble(
               "XR_TRACKER_SP_Q_XYZ", linear_variance));
  angular_variance = std::max(
      0.0, armor_tracker_detail::ParseEnvDouble(
               "XR_TRACKER_SP_Q_YAW", angular_variance));

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
  q(ExtendedKalmanFilter::DELTA_Z, ExtendedKalmanFilter::DELTA_Z) =
      SpDeltaZProcessVariance();

  ekf_.covariance = f * ekf_.covariance * f.transpose() + q;
  ekf_.state = f * ekf_.state;
  ekf_.state(ExtendedKalmanFilter::YAW) =
      SpLimitRad(ekf_.state(ExtendedKalmanFilter::YAW));

  const bool outpost_converged = rt_.tracked_id == ArmorNumber::OUTPOST &&
                                 rt_.update_count > 10 && !SpStateDiverged();
  if (outpost_converged &&
      std::abs(ekf_.state(ExtendedKalmanFilter::V_YAW)) > 2.0)
  {
    ekf_.state(ExtendedKalmanFilter::V_YAW) =
        ekf_.state(ExtendedKalmanFilter::V_YAW) > 0.0 ? 2.51 : -2.51;
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SpUpdate(const ArmorDetectorResult& armor,
                                         const SpArmorMatch& match,
                                         bool freeze_delta_z)
{
  const Eigen::Vector3d armor_xyz(armor.pose.translation.x(),
                                  armor.pose.translation.y(),
                                  armor.pose.translation.z());
  ekf_.measurement_face_index = match.id;
  const double base_z_before_update = ekf_.state(ExtendedKalmanFilter::Z_ARMOR);
  const Eigen::Vector3d armor_ypd = SpXyzToYpd(armor_xyz);
  Eigen::MatrixXd h = SpObservationJacobian(ekf_.state, match.id);
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
      !SpXyzMeasurementFullGeometryEnabled())
  {
    // 单装甲板观测无法区分整车中心平移和半径误差；半径只由多装甲几何更新。
    h.col(ExtendedKalmanFilter::ROBOT_R).setZero();
    h.col(ExtendedKalmanFilter::DELTA_R).setZero();
  }

  if (SpXyzMeasurementUpdateEnabled())
  {
    const int armor_count =
        std::max(1, static_cast<int>(rt_.tracked_armors_num));
    const double angle = SpLimitRad(
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

    if (SpXyzMeasurementFullGeometryEnabled())
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
    const double position_sigma =
        std::max(0.005, SpXyzMeasurementRFactor(cfg_.noise.r_xyz_factor) * range);
    Eigen::VectorXd r_diag(4);
    r_diag << position_sigma * position_sigma,
        position_sigma * position_sigma, position_sigma * position_sigma,
        SpXyzMeasurementYawVariance(cfg_.noise.r_yaw);

    Eigen::VectorXd z(4);
    z << armor_xyz.x(), armor_xyz.y(), armor_xyz.z(), match.measured_yaw;
    Eigen::VectorXd predicted(4);
    const Eigen::Vector3d predicted_xyz = SpArmorPosition(ekf_.state, match.id);
    predicted << predicted_xyz.x(), predicted_xyz.y(), predicted_xyz.z(), angle;
    Eigen::VectorXd innovation = z - predicted;
    innovation[3] = SpLimitRad(innovation[3]);

    const Eigen::MatrixXd r = r_diag.asDiagonal();
    const Eigen::MatrixXd innovation_cov =
        h_xyz * ekf_.covariance * h_xyz.transpose() + r;
    const Eigen::MatrixXd kalman_gain =
        ekf_.covariance * h_xyz.transpose() * innovation_cov.inverse();
    const Eigen::MatrixXd identity =
        Eigen::MatrixXd::Identity(ekf_.covariance.rows(), ekf_.covariance.cols());
    ekf_.state = ekf_.state + kalman_gain * innovation;
    ekf_.state(ExtendedKalmanFilter::YAW) =
        SpLimitRad(ekf_.state(ExtendedKalmanFilter::YAW));
    if (freeze_delta_z && rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
    {
      ekf_.state(ExtendedKalmanFilter::DELTA_Z) = dz_before_update;
    }
    const bool direct_dz_update =
        SpDirectDeltaZEnabled() && rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
        !freeze_delta_z && odd_face;
    if (direct_dz_update)
    {
      const double alpha = SpDirectDeltaZAlpha();
      const double max_abs_dz = SpDirectDeltaZMaxAbs();
      const double observed_dz =
          std::clamp(armor_xyz.z() - base_z_before_update, -max_abs_dz, max_abs_dz);
      ekf_.state(ExtendedKalmanFilter::DELTA_Z) =
          (1.0 - alpha) * ekf_.state(ExtendedKalmanFilter::DELTA_Z) +
          alpha * observed_dz;
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
    return;
  }

  const double center_yaw = std::atan2(armor_xyz.y(), armor_xyz.x());
  const double delta_angle = SpLimitRad(match.measured_yaw - center_yaw);
  Eigen::VectorXd r_diag(4);
  r_diag << 4e-3, 4e-3 * SpPitchVarianceScale(),
      (std::log(std::abs(delta_angle) + 1.0) + 1.0) *
          SpYpdDistanceVarianceScale(),
      (std::log(std::abs(armor_ypd.z()) + 1.0) / 200.0 + 9e-2) *
          SpYpdArmorYawVarianceScale();
  const Eigen::MatrixXd r = r_diag.asDiagonal();

  auto observe = [this, &match](const Eigen::VectorXd& state)
  {
    const Eigen::Vector3d xyz = SpArmorPosition(state, match.id);
    const Eigen::Vector3d ypd = SpXyzToYpd(xyz);
    const int armor_count =
        std::max(1, static_cast<int>(rt_.tracked_armors_num));
    const double angle =
        SpLimitRad(state(ExtendedKalmanFilter::YAW) +
                   match.id * 2.0 * M_PI / armor_count);
    Eigen::VectorXd out(4);
    out << ypd.x(), ypd.y(), ypd.z(), angle;
    return out;
  };
  auto subtract = [](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs)
  {
    Eigen::VectorXd result = lhs - rhs;
    result[0] = ArmorTracker<CameraInfoV>::SpLimitRad(result[0]);
    result[1] = ArmorTracker<CameraInfoV>::SpLimitRad(result[1]);
    result[3] = ArmorTracker<CameraInfoV>::SpLimitRad(result[3]);
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
      SpLimitRad(ekf_.state(ExtendedKalmanFilter::YAW));
  if (freeze_delta_z && rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    ekf_.state(ExtendedKalmanFilter::DELTA_Z) = dz_before_update;
  }
  const bool direct_dz_update =
      SpDirectDeltaZEnabled() && rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
      !freeze_delta_z &&
      (match.id == 1 || match.id == 3);
  if (direct_dz_update)
  {
    const double alpha = SpDirectDeltaZAlpha();
    const double max_abs_dz = SpDirectDeltaZMaxAbs();
    const double observed_dz =
        std::clamp(armor_xyz.z() - base_z_before_update, -max_abs_dz, max_abs_dz);
    ekf_.state(ExtendedKalmanFilter::DELTA_Z) =
        (1.0 - alpha) * ekf_.state(ExtendedKalmanFilter::DELTA_Z) +
        alpha * observed_dz;
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

  double recenter_alpha = SpMeasurementRecenterAlpha();
  if (recenter_alpha > 0.0 && SpMeasurementRecenterQualityEnabled())
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
        {quality_ramp(match.score, SpMeasurementRecenterScoreGood(),
                      SpMeasurementRecenterScoreBad()),
         quality_ramp(match.angle_error, SpMeasurementRecenterYawGood(),
                      SpMeasurementRecenterYawBad()),
         quality_ramp(match.xyz_error, SpMeasurementRecenterXyzGood(),
                      SpMeasurementRecenterXyzBad())});
    const double alpha_bad = SpMeasurementRecenterAlphaBad();
    const double alpha_good = SpMeasurementRecenterAlphaGood();
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
        SpLimitRad(anchor_yaw - ekf_.state(ExtendedKalmanFilter::YAW));
    ekf_.state(ExtendedKalmanFilter::X_CENTER) += correction_x;
    ekf_.state(ExtendedKalmanFilter::Y_CENTER) += correction_y;
    ekf_.state(ExtendedKalmanFilter::Z_ARMOR) += correction_z;
    ekf_.state(ExtendedKalmanFilter::YAW) =
        SpLimitRad(ekf_.state(ExtendedKalmanFilter::YAW) + correction_yaw);
  }

  const double position_anchor_alpha = SpMeasurementPositionAnchorAlpha();
  if (position_anchor_alpha > 0.0 &&
      rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    const double xyz_bad = SpMeasurementPositionAnchorXyzBad();
    const double quality =
        std::clamp((xyz_bad - match.xyz_error) / xyz_bad, 0.0, 1.0);
    const double alpha = position_anchor_alpha * quality;
    if (alpha > 0.0)
    {
      const int armor_count =
          std::max(1, static_cast<int>(rt_.tracked_armors_num));
      const double angle_step = 2.0 * M_PI / armor_count;
      const bool odd_face = (match.id == 1 || match.id == 3);
      const double face_yaw =
          SpLimitRad(ekf_.state(ExtendedKalmanFilter::YAW) +
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
          alpha * (anchor_x - ekf_.state(ExtendedKalmanFilter::X_CENTER));
      ekf_.state(ExtendedKalmanFilter::Y_CENTER) +=
          alpha * (anchor_y - ekf_.state(ExtendedKalmanFilter::Y_CENTER));
      ekf_.state(ExtendedKalmanFilter::Z_ARMOR) +=
          alpha * (anchor_z - ekf_.state(ExtendedKalmanFilter::Z_ARMOR));
    }
  }

  ekf_.measurement =
      Eigen::Vector4d(armor_xyz.x(), armor_xyz.y(), armor_xyz.z(),
                      match.measured_yaw);
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::SpStateDiverged() const
{
  const double r1 = ekf_.state(ExtendedKalmanFilter::ROBOT_R);
  const double r2 = r1 + ekf_.state(ExtendedKalmanFilter::DELTA_R);
  return !(r1 > 0.05 && r1 < 0.5 && r2 > 0.05 && r2 < 0.5);
}
