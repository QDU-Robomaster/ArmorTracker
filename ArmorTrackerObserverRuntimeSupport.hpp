#pragma once

template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker::ObserverPolicy ArmorTracker<CameraInfoV>::BuildObserverPolicy() const
{
  armor_tracker::ObserverPolicy policy{};
  policy.single_armor_mode = SingleArmorModeEnabled();
  policy.symmetric_geometry_enabled = SymmetricGeometryEnabled();
  policy.max_match_distance = cfg_.match.max_match_distance;
  policy.max_match_yaw_diff = cfg_.match.max_match_yaw_diff;
  policy.initial_radius = cfg_.geometry.initial_radius;
  policy.min_radius = cfg_.geometry.min_radius;
  policy.max_radius = cfg_.geometry.max_radius;
  return policy;
}

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
    const ArmorDetectorResults& armors_msg, int observed_face_index,
    bool recenter_before_update)
{
  rt_.last_yaw = selected_candidate.measured_yaw;
  if (recenter_before_update)
  {
    RecenterTrackedStateToMeasurement(selected_candidate.armor,
                                      observed_face_index,
                                      selected_candidate.measured_yaw);
  }
  ekf_.measurement_face_index = observed_face_index;
  ekf_.measurement_geometry_mode = recenter_before_update
                                       ? EKFBlock::MeasurementGeometryMode::FULL_BODY
                                       : EKFBlock::MeasurementGeometryMode::VISIBLE_FACE_ONLY;
  const auto p = selected_candidate.armor.pose.translation;
  ekf_.measurement =
      Eigen::Vector4d(p.x(), p.y(), p.z(), selected_candidate.measured_yaw);
  ekf_.state = ekf_.ekf.Update(ekf_.measurement);
  if (recenter_before_update)
  {
    // 换面这一步允许几何状态被一次性重估，但不要把几何-位姿相关性带进后续单面更新。
    ekf_.ekf.DecorrelatePosterior(
        {ExtendedKalmanFilter::ROBOT_R, ExtendedKalmanFilter::DELTA_R,
         ExtendedKalmanFilter::DELTA_Z});
  }
  ekf_.measurement_geometry_mode = EKFBlock::MeasurementGeometryMode::FULL_BODY;
  SyncGeometryRuntimeFromState();
  SyncDzReferenceFromState();
  if (MultiArmorFuseEnabled())
  {
    FuseMultiArmorObservation(armors_msg);
  }
  ClampGeometryState();
  XR_LOG_DEBUG("EKF update");
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
  const LibXR::Transform<double> t_wc =
      io_.current_camera_pose_valid ? io_.current_camera_pose
                                    : io_.gimbal_to_camera_transform_static;
  return Eigen::Vector3d(t_wc.translation.x(), t_wc.translation.y(),
                         t_wc.translation.z());
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::ApplyFaceSelection(
    const armor_tracker::FaceSelectionResult& selection,
    const ArmorDetectorResults& armors_msg, CandidateDebugMsg& candidate_debug)
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
  candidate_debug.matched = 1;
  candidate_debug.accepted_mode =
      static_cast<uint8_t>(selection.accepted_mode);
  armor_tracker::LogAcceptedSelection(selection, log_context);

  if (did_face_switch)
  {
    SwitchTrackedFace(selected_candidate.face_index, selected_candidate.measured_yaw);
    rt_.face_switch_cooldown_remaining = candidate_debug.face_switch_timeout_sec;
    candidate_debug.face_switch_cooldown_remaining =
        static_cast<float>(rt_.face_switch_cooldown_remaining);
  }

  ApplySelectedIdentity(selected_candidate);
  ApplySelectedFaceBinding(selected_candidate, did_face_switch);
  ApplySelectedMeasurementUpdate(selected_candidate, armors_msg,
                                 observed_face_index, did_face_switch);
  return true;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::UpdateArmorsNum()
{
  auto runtime = BuildObserverRuntime();
  armor_tracker::UpdateArmorsNum(runtime, BuildObserverPolicy());
  ApplyObserverRuntime(runtime);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SyncDzReferenceFromState()
{
  auto runtime = BuildObserverRuntime();
  armor_tracker::SyncDzReferenceFromState(runtime);
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
void ArmorTracker<CameraInfoV>::SwitchTrackedFace(int face_index, double measured_yaw)
{
  auto runtime = BuildObserverRuntime();
  armor_tracker::SwitchTrackedFace(runtime, ekf_.state, BuildObserverPolicy(),
                                   face_index, measured_yaw);
  ApplyObserverRuntime(runtime);
  ekf_.ekf.SetState(ekf_.state);
}

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

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::OrientationToYaw(const LibXR::Quaternion<double>& q)
{
  auto runtime = BuildObserverRuntime();
  const double yaw = armor_tracker::OrientationToYaw(q, runtime);
  ApplyObserverRuntime(runtime);
  return yaw;
}

template <CameraTypes::CameraInfo CameraInfoV>
int ArmorTracker<CameraInfoV>::LocalFaceToCanonicalFace(int local_face_index) const
{
  return armor_tracker::NormalizeFaceIndex(
      rt_.tracked_face_index + local_face_index,
      std::max(1, static_cast<int>(rt_.tracked_armors_num)));
}

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

template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::GetArmorYawFromState(const Eigen::VectorXd& x,
                                                       int face_index) const
{
  const int armor_count =
      std::max(1, static_cast<int>(rt_.tracked_armors_num));
  return SpLimitRad(x(ExtendedKalmanFilter::YAW) +
                    face_index * 2.0 * M_PI / armor_count);
}

template <CameraTypes::CameraInfo CameraInfoV>
Eigen::Vector3d ArmorTracker<CameraInfoV>::GetArmorPositionFromState(const Eigen::VectorXd& x,
                                                        int face_index) const
{
  return SpArmorPosition(x, face_index);
}
