#pragma once

/**
 * @file ArmorTrackerRuntimeAdapter.hpp
 * @brief ArmorTracker 类状态与 observer / selector 纯算法运行态之间的适配层。
 *
 * 本文件只做成员变量到轻量 runtime 结构的组装、回写和局部几何辅助计算，
 * 让 `ArmorTrackerObserver.hpp`、`ArmorTrackerFaceSelector.hpp` 保持无类依赖。
 */

/**
 * @brief 根据当前配置和运行开关构造观测器策略。
 */
template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker::ObserverPolicy ArmorTracker<CameraInfoV>::BuildObserverPolicy() const
{
  armor_tracker::ObserverPolicy policy{};
  policy.single_armor_mode = SingleArmorModeEnabled();
  policy.symmetric_geometry_enabled = SymmetricGeometryEnabled();
  policy.max_match_distance = cfg_.match.max_match_distance;
  policy.max_match_yaw_diff = cfg_.match.max_match_yaw_diff;
  policy.initial_radius = cfg_.geometry.initial_radius;
  return policy;
}

/**
 * @brief 将类内跟踪状态打包为 observer helper 可消费的运行态。
 */
template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker::ObserverRuntime ArmorTracker<CameraInfoV>::BuildObserverRuntime() const
{
  armor_tracker::ObserverRuntime runtime{};
  runtime.tracked_id = rt_.tracked_id;
  runtime.tracked_armor_type = rt_.tracked_armor.type;
  runtime.tracked_armors_num = static_cast<int>(rt_.tracked_armors_num);
  runtime.tracked_face_index = rt_.tracked_face_index;
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

/**
 * @brief 将 observer helper 修改后的运行态同步回类成员。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ApplyObserverRuntime(
    const armor_tracker::ObserverRuntime& runtime)
{
  rt_.tracked_id = runtime.tracked_id;
  rt_.tracked_armors_num =
      static_cast<ArmorsNum>(runtime.tracked_armors_num);
  rt_.tracked_face_index = runtime.tracked_face_index;
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

/**
 * @brief 打包装甲面与图像 track 绑定运行态。
 */
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

/**
 * @brief 将装甲面绑定 helper 的结果写回类成员。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ApplyFaceBindingRuntime(
    const armor_tracker::FaceBindingRuntime& runtime)
{
  rt_.tracked_face_track_id_valid = runtime.tracked_face_track_id_valid;
  rt_.tracked_face_track_id = runtime.tracked_face_track_id;
  rt_.face_track_id_valid = runtime.face_track_id_valid;
  rt_.face_track_id = runtime.face_track_id;
}

/**
 * @brief 按候选观测刷新当前目标身份，同时避免被非同号观测覆盖。
 */
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

/**
 * @brief 将已接受候选同步到装甲面和图像 track 绑定表。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ApplySelectedFaceBinding(
    const armor_tracker::FaceMatchCandidate& selected_candidate, bool did_face_switch)
{
  const int tracked_face_track_before =
      rt_.tracked_face_track_id_valid ? static_cast<int>(rt_.tracked_face_track_id)
                                      : -1;
  UNUSED(tracked_face_track_before);
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

/**
 * @brief 汇总当前环境变量和配置，构造换面选择策略。
 */
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
  face_policy.observation_quality_enabled = ObservationQualityEnabled();
  face_policy.match_yaw_allow_pi_ambiguity =
      MatchYawAllowPiAmbiguityEnabled();
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
  face_policy.stable_max_reprojection_px = ObservationStableMaxReprojectionPx();
  face_policy.stable_min_area_px = ObservationStableMinAreaPx();
  face_policy.stable_min_confidence = ObservationStableMinConfidence();
  face_policy.observation_quality_score_weight =
      ObservationQualityScoreWeight();
  face_policy.confirmed_track_bonus = ObservationConfirmedTrackBonus();
  return face_policy;
}

/**
 * @brief 打包当前跟踪目标状态，供换面选择器计算候选分数。
 */
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

/**
 * @brief 获取 tracker 世界系中的当前相机位置。
 */
template <CameraTypes::CameraInfo CameraInfoV>
Eigen::Vector3d ArmorTracker<CameraInfoV>::GetCameraWorldPosition()
{
  const LibXR::Transform<double> t_wc =
      io_.current_camera_pose_valid ? io_.current_camera_pose
                                    : io_.gimbal_to_camera_transform_static;
  return Eigen::Vector3d(t_wc.translation.x(), t_wc.translation.y(),
                         t_wc.translation.z());
}

/**
 * @brief 应用换面选择结果，并在接受观测时执行整车模型 EKF 更新。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::ApplyFaceSelection(
    const armor_tracker::FaceSelectionResult& selection,
    CandidateDebugMsg& candidate_debug, bool freeze_delta_z,
    uint64_t image_timestamp_us)
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
  const int observed_face_index =
      LocalFaceToCanonicalFace(selected_candidate.face_index);
  const VehicleArmorMatch model_match =
      VehicleMatchArmorToFace(selected_candidate.armor, ekf_.state, observed_face_index);
  candidate_debug.matched = 1;
  candidate_debug.accepted_mode =
      static_cast<uint8_t>(selection.accepted_mode);
  armor_tracker::LogAcceptedSelection(selection, log_context);

  if (did_face_switch)
  {
    rt_.face_switch_cooldown_remaining = candidate_debug.face_switch_timeout_sec;
    candidate_debug.face_switch_cooldown_remaining =
        static_cast<float>(rt_.face_switch_cooldown_remaining);
  }

  ApplySelectedIdentity(selected_candidate);
  ApplySelectedFaceBinding(selected_candidate, did_face_switch);
  VehicleUpdate(selected_candidate.armor, model_match, freeze_delta_z, image_timestamp_us,
           &candidate_debug);
  if (VehicleCenterMotionObserverEnabled())
  {
    VehicleUpdateCenterMotionObserver(selected_candidate.armor, model_match,
                                 image_timestamp_us);
  }
  rt_.tracked_armor = selected_candidate.armor;
  rt_.tracked_armors_num =
      static_cast<ArmorsNum>(VehicleArmorCountFor(selected_candidate.armor));
  rt_.tracked_face_index = observed_face_index;
  rt_.last_yaw = model_match.measured_yaw;
  rt_.info_position_diff = model_match.xyz_error;
  rt_.info_yaw_diff = model_match.angle_error;
  SyncGeometryRuntimeFromState();
  ekf_.ekf.SetState(ekf_.state);
  return true;
}

/**
 * @brief 从 EKF 几何状态同步高低差参考量。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SyncDzReferenceFromState()
{
  auto runtime = BuildObserverRuntime();
  armor_tracker::SyncDzReferenceFromState(runtime);
  ApplyObserverRuntime(runtime);
}

/**
 * @brief 用当前测量装甲板刚性重定位整车中心、yaw 和基础高度。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::RecenterTrackedStateToMeasurement(
    const ArmorDetectorResult& armor, int observed_face_index,
    double measured_yaw)
{
  const int armor_count = std::max(1, static_cast<int>(rt_.tracked_armors_num));
  const double angle_step = 2.0 * M_PI / armor_count;
  const bool odd_face =
      !SymmetricGeometryEnabled() &&
      rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
      (observed_face_index % 2 == 1);
  const double radius_1 = ekf_.state(ExtendedKalmanFilter::ROBOT_R);
  const double radius_2 =
      SymmetricGeometryEnabled()
          ? radius_1
          : (radius_1 + ekf_.state(ExtendedKalmanFilter::DELTA_R));
  const double radius = odd_face ? radius_2 : radius_1;
  const double base_z =
      odd_face ? (armor.pose.translation.z() -
                  ekf_.state(ExtendedKalmanFilter::DELTA_Z))
               : armor.pose.translation.z();

  ekf_.state(ExtendedKalmanFilter::YAW) =
      measured_yaw - angle_step * observed_face_index;
  ekf_.state(ExtendedKalmanFilter::X_CENTER) =
      armor.pose.translation.x() - radius * std::cos(measured_yaw);
  ekf_.state(ExtendedKalmanFilter::Y_CENTER) =
      armor.pose.translation.y() - radius * std::sin(measured_yaw);
  ekf_.state(ExtendedKalmanFilter::Z_ARMOR) = base_z;

  ekf_.ekf.SetState(ekf_.state);
  SyncGeometryRuntimeFromState();
}

/**
 * @brief 将候选选择器返回的局部面索引转换为当前 canonical 面索引。
 */
template <CameraTypes::CameraInfo CameraInfoV>
int ArmorTracker<CameraInfoV>::LocalFaceToCanonicalFace(int local_face_index) const
{
  return armor_tracker::NormalizeFaceIndex(
      rt_.tracked_face_index + local_face_index,
      std::max(1, static_cast<int>(rt_.tracked_armors_num)));
}

/**
 * @brief 从 EKF 状态同步半径、高低差等几何运行态缓存。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SyncGeometryRuntimeFromState()
{
  const bool four_armors = rt_.tracked_armors_num == ArmorsNum::NORMAL_4;
  const double radius_1 = ekf_.state(ExtendedKalmanFilter::ROBOT_R);
  const double radius_2 =
      !four_armors ? radius_1 : (radius_1 + ekf_.state(ExtendedKalmanFilter::DELTA_R));
  rt_.another_r = radius_2;
  rt_.dz = !four_armors ? 0.0 : ekf_.state(ExtendedKalmanFilter::DELTA_Z);
  rt_.dz_abs_ref = std::abs(rt_.dz);
}

/**
 * @brief 将 EKF 几何状态限制在配置允许的物理范围内。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ClampGeometryState()
{
  const bool four_armors = rt_.tracked_armors_num == ArmorsNum::NORMAL_4;
  double radius_1 = ekf_.state(ExtendedKalmanFilter::ROBOT_R);
  double radius_2 =
      (!four_armors || SymmetricGeometryEnabled())
          ? radius_1
          : (radius_1 + ekf_.state(ExtendedKalmanFilter::DELTA_R));

  const double min_radius = std::min(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
  const double max_radius = std::max(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
  radius_1 = std::clamp(radius_1, min_radius, max_radius);
  radius_2 = std::clamp(radius_2, min_radius, max_radius);
  ekf_.state(ExtendedKalmanFilter::ROBOT_R) = radius_1;
  ekf_.state(ExtendedKalmanFilter::DELTA_R) =
      (!four_armors || SymmetricGeometryEnabled()) ? 0.0 : (radius_2 - radius_1);
  if (!four_armors || SymmetricGeometryEnabled())
  {
    ekf_.state(ExtendedKalmanFilter::DELTA_Z) = 0.0;
  }
  ekf_.ekf.SetState(ekf_.state);
  SyncGeometryRuntimeFromState();
}

/**
 * @brief 从整车状态向量计算指定装甲面的 yaw。
 */
template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::GetArmorYawFromState(const Eigen::VectorXd& x,
                                                       int face_index) const
{
  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  return VehicleLimitRad(x(ExtendedKalmanFilter::YAW) +
                    face_index * 2.0 * M_PI / armor_count);
}

/**
 * @brief 从整车状态向量计算指定装甲面的 tracker 世界系位置。
 */
template <CameraTypes::CameraInfo CameraInfoV>
Eigen::Vector3d ArmorTracker<CameraInfoV>::GetArmorPositionFromState(const Eigen::VectorXd& x,
                                                        int face_index) const
{
  return VehicleArmorPosition(x, face_index);
}

/**
 * @brief 对一帧 detector 结果批量执行固定姿态 yaw 重投影优化。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::OptimizeArmorYawMeasurements(
    ArmorDetectorResults& armors_msg,
    const LibXR::Transform<double>& camera_pose_world) const
{
  if (!VehicleFixedPoseYawOptimizeEnabled())
  {
    return;
  }

  for (auto& armor : armors_msg)
  {
    OptimizeSingleArmorYawMeasurement(armor, camera_pose_world);
  }
}

/**
 * @brief 尝试用固定 pitch 假设重估单个装甲板 yaw。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::OptimizeSingleArmorYawMeasurement(
    ArmorDetectorResult& armor,
    const LibXR::Transform<double>& camera_pose_world) const
{
  if (!armor.pnp_valid)
  {
    return false;
  }

  const bool is_balance =
      armor.type == ArmorType::LARGE &&
      (armor.number == ArmorNumber::THREE ||
       armor.number == ArmorNumber::FOUR ||
       armor.number == ArmorNumber::FIVE);
  if (is_balance)
  {
    return false;
  }

  const double raw_yaw = armor_tracker::QuaternionToYaw(armor.pose.rotation);
  if (!std::isfinite(raw_yaw))
  {
    return false;
  }

  const double pitch_abs = VehicleFixedPoseYawPitchDeg() * M_PI / 180.0;
  const std::array<double, 2> pitch_candidates = {pitch_abs, -pitch_abs};
  double raw_fixed_error = std::numeric_limits<double>::infinity();
  for (const double pitch : pitch_candidates)
  {
    raw_fixed_error =
        std::min(raw_fixed_error,
                 ArmorYawReprojectionError(armor, camera_pose_world, raw_yaw, pitch));
  }
  if (!std::isfinite(raw_fixed_error))
  {
    return false;
  }

  double best_yaw = raw_yaw;
  double best_pitch = pitch_candidates[0];
  double best_error = raw_fixed_error;
  const double search_range = VehicleFixedPoseYawRangeDeg() * M_PI / 180.0;
  const double coarse_step = VehicleFixedPoseYawCoarseStepDeg() * M_PI / 180.0;
  const double fine_step = VehicleFixedPoseYawFineStepDeg() * M_PI / 180.0;

  auto try_candidate = [&](double yaw, double pitch)
  {
    const double error =
        ArmorYawReprojectionError(armor, camera_pose_world, yaw, pitch);
    if (std::isfinite(error) && error < best_error)
    {
      best_error = error;
      best_yaw = yaw;
      best_pitch = pitch;
    }
  };

  for (const double pitch : pitch_candidates)
  {
    for (double offset = -search_range; offset <= search_range + 1e-9;
         offset += coarse_step)
    {
      try_candidate(raw_yaw + offset, pitch);
    }
  }

  const double fine_range = std::max(coarse_step, 4.0 * fine_step);
  for (const double pitch : pitch_candidates)
  {
    for (double offset = -fine_range; offset <= fine_range + 1e-9;
         offset += fine_step)
    {
      try_candidate(best_yaw + offset, pitch);
    }
  }

  if (raw_fixed_error - best_error < VehicleFixedPoseYawMinGainPx())
  {
    return false;
  }

  const Eigen::AngleAxisd yaw_rotation(best_yaw, Eigen::Vector3d::UnitZ());
  const Eigen::AngleAxisd pitch_rotation(best_pitch, Eigen::Vector3d::UnitY());
  armor.pose.rotation =
      LibXR::Quaternion<double>((yaw_rotation * pitch_rotation).toRotationMatrix());
  return true;
}

/**
 * @brief 计算给定 yaw/pitch 假设下装甲四点的重投影误差。
 */
template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::ArmorYawReprojectionError(
    const ArmorDetectorResult& armor,
    const LibXR::Transform<double>& camera_pose_world,
    double yaw_rad, double pitch_rad) const
{
  const double half_width_m =
      ((armor.type == ArmorType::LARGE) ? 225.0 : 135.0) * 0.5 / 1000.0;
  constexpr double half_height_m = 56.0 * 0.5 / 1000.0;
  const std::vector<cv::Point3f> object_points = {
      {0.0F, static_cast<float>(half_width_m), static_cast<float>(-half_height_m)},
      {0.0F, static_cast<float>(half_width_m), static_cast<float>(half_height_m)},
      {0.0F, static_cast<float>(-half_width_m), static_cast<float>(half_height_m)},
      {0.0F, static_cast<float>(-half_width_m), static_cast<float>(-half_height_m)}};
  const std::vector<cv::Point2f> image_points = {
      armor.points[3], armor.points[0], armor.points[1], armor.points[2]};

  const auto r_wc = camera_pose_world.rotation.ToRotationMatrix();
  const Eigen::Matrix3d r_cw = r_wc.transpose();
  const Eigen::Vector3d t_wc(camera_pose_world.translation.x(),
                             camera_pose_world.translation.y(),
                             camera_pose_world.translation.z());
  const Eigen::Vector3d t_aw(armor.pose.translation.x(),
                             armor.pose.translation.y(),
                             armor.pose.translation.z());
  const Eigen::Vector3d t_ac = r_cw * (t_aw - t_wc);
  if (!t_ac.allFinite() || t_ac.z() <= 1e-6)
  {
    return std::numeric_limits<double>::infinity();
  }

  const Eigen::AngleAxisd yaw_rotation(yaw_rad, Eigen::Vector3d::UnitZ());
  const Eigen::AngleAxisd pitch_rotation(pitch_rad, Eigen::Vector3d::UnitY());
  const Eigen::Matrix3d r_aw =
      (yaw_rotation * pitch_rotation).toRotationMatrix();
  const Eigen::Matrix3d r_ac = r_cw * r_aw;

  cv::Mat rmat(3, 3, CV_64F);
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      rmat.at<double>(row, col) = r_ac(row, col);
    }
  }
  cv::Mat rvec;
  cv::Rodrigues(rmat, rvec);
  const cv::Mat tvec =
      (cv::Mat_<double>(3, 1) << t_ac.x(), t_ac.y(), t_ac.z());
  const cv::Mat camera_matrix =
      cv::Mat(3, 3, CV_64F,
              const_cast<double*>(kCameraInfo.camera_matrix.data()))
          .clone();

  cv::Mat dist_coeffs;
  constexpr auto dist_info =
      armor_tracker_detail::BuildTrackerPnPDistCoeffs(kCameraInfo);
  if constexpr (!dist_info.requires_undistort_first && dist_info.size > 0)
  {
    dist_coeffs =
        cv::Mat(1, static_cast<int>(dist_info.size), CV_64F,
                const_cast<double*>(dist_info.values.data()))
            .clone();
  }

  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs,
                    projected);
  if (projected.size() != image_points.size())
  {
    return std::numeric_limits<double>::infinity();
  }

  double error = 0.0;
  for (std::size_t index = 0; index < image_points.size(); ++index)
  {
    const double point_error = cv::norm(projected[index] - image_points[index]);
    if (!std::isfinite(point_error))
    {
      return std::numeric_limits<double>::infinity();
    }
    error += point_error;
  }
  return error;
}
