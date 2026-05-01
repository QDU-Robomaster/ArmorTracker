#pragma once

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::Init(const ArmorDetectorResults& armors_msg)
{
  if (armors_msg.empty())
  {
    return;
  }

  double min_distance = DBL_MAX;
  ArmorPriority best_priority = ArmorPriority::FIFTH;
  std::size_t tracked_index = 0;
  rt_.tracked_armor = armors_msg[0];
  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    if (static_cast<int>(armor.priority) < static_cast<int>(best_priority) ||
        (armor.priority == best_priority &&
         armor.distance_to_image_center < min_distance))
    {
      best_priority = armor.priority;
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
  rt_.tracked_face_index = 0;
  rt_.face_track_id_valid.fill(false);
  rt_.face_track_id.fill(0);
  if (rt_.tracked_face_track_id_valid)
  {
    rt_.face_track_id_valid[0] = true;
    rt_.face_track_id[0] = rt_.tracked_face_track_id;
  }

  rt_.tracked_id = rt_.tracked_armor.number;
  rt_.tracked_armors_num =
      static_cast<ArmorsNum>(SpArmorCountFor(rt_.tracked_armor));
  rt_.sp_initial_phase_resolved = false;
  rt_.sp_pair_delta_z_valid = false;
  rt_.measurement_valid_current_frame = false;
  InitEKF(rt_.tracked_armor);
  rt_.sp_initial_phase_resolved =
      SpTryCanonicalizeInitialState(armors_msg, true);
  XR_LOG_DEBUG("Init EKF!");

  rt_.state = State::DETECTING;
  rt_.detect_count = 1;
  rt_.lost_count = 0;
  rt_.update_count = 0;
  rt_.switch_count = 0;
  rt_.suspect_count = 0;
  candidate_debug_msg_ = CandidateDebugMsg{};
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::Update(const ArmorDetectorResults& armors_msg,
                                       uint64_t image_timestamp_us)
{
  if (SingleArmorModeEnabled())
  {
    UpdateSingleArmorMode(armors_msg, image_timestamp_us);
    return;
  }

  SpPredict();
  Eigen::VectorXd ekf_prediction = ekf_.state;
  if (SpCanonicalInitEnabled())
  {
    if (rt_.update_count <=
        static_cast<int>(SpCanonicalInitMaxUpdates()))
    {
      if (SpTryCanonicalizeInitialState(
              armors_msg, !rt_.sp_initial_phase_resolved))
      {
        rt_.sp_initial_phase_resolved = true;
        ekf_prediction = ekf_.state;
      }
    }
  }
  XR_LOG_DEBUG("SP tracker predict");
  bool matched = false;
  rt_.measurement_valid_current_frame = false;
  const bool pair_delta_z_mode =
      SpPairDeltaZEnabled() && rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
      rt_.state == State::TRACKING;
  SpPairMatch pair_match{};
  const bool has_pair_match =
      pair_delta_z_mode && SpResolvePairMatch(armors_msg, ekf_prediction, pair_match);
  ArmorTracker<CameraInfoV>::CandidateDebugMsg candidate_debug{};
  std::fill(candidate_debug.detection_track_ids.begin(),
            candidate_debug.detection_track_ids.end(), static_cast<int16_t>(-1));
  std::fill(candidate_debug.detection_track_confirmed.begin(),
            candidate_debug.detection_track_confirmed.end(), static_cast<uint8_t>(0));
  candidate_debug.tracked_armors_num =
      static_cast<uint8_t>(std::max(1, static_cast<int>(rt_.tracked_armors_num)));
  candidate_debug.predicted_vyaw =
      static_cast<float>(ekf_prediction(ExtendedKalmanFilter::V_YAW));
  candidate_debug.max_match_distance = static_cast<float>(cfg_.match.max_match_distance);
  candidate_debug.max_match_yaw_diff = static_cast<float>(cfg_.match.max_match_yaw_diff);
  candidate_debug.detection_count = static_cast<uint8_t>(
      std::min<std::size_t>(armors_msg.size(),
                            ArmorTracker<CameraInfoV>::CandidateDebugMsg::kMaxDetections));
  for (std::size_t armor_index = 0; armor_index < candidate_debug.detection_count;
       ++armor_index)
  {
    candidate_debug.detection_track_ids[armor_index] =
        static_cast<int16_t>(FindDetectionTrackId(armor_index));
    candidate_debug.detection_track_confirmed[armor_index] =
        IsDetectionTrackConfirmed(armor_index) ? 1 : 0;
  }
  rt_.info_position_diff = DBL_MAX;
  rt_.info_yaw_diff = DBL_MAX;
  const auto face_policy = BuildFaceSelectionPolicy();
  FillCandidateDebugPolicy(candidate_debug, ekf_prediction, face_policy);
  armor_tracker::FaceSelectionResult face_selection{};
  const armor_tracker::FaceSelectionResult* audit_selection = nullptr;

  if (has_pair_match)
  {
    matched = true;
    candidate_debug.has_same_number_candidate = 1;
    candidate_debug.matched = 1;
    candidate_debug.accepted_mode = 1;
    candidate_debug.count = 2;
    candidate_debug.selected_index =
        pair_match.tracked_armor_index == pair_match.left.armor_index ? 0 : 1;
    candidate_debug.best_same_face_score = static_cast<float>(pair_match.score);
    candidate_debug.same_face_matched = 1;

    const auto fill_item = [this, &candidate_debug, &ekf_prediction](
                               std::size_t item_index,
                               const SpPairObservation& observation,
                               const SpArmorMatch& match)
    {
      auto& item = candidate_debug.items[item_index];
      item.armor_index =
          static_cast<uint8_t>(std::min<std::size_t>(observation.armor_index, 255));
      item.face_index = static_cast<uint8_t>(std::max(0, match.id));
      item.same_number = 1;
      item.image_track_id =
          static_cast<int16_t>(FindDetectionTrackId(observation.armor_index));
      item.image_track_confirmed =
          IsDetectionTrackConfirmed(observation.armor_index) ? 1 : 0;
      item.same_persistent_track = 0;
      item.number = observation.armor.number;
      item.type = observation.armor.type;
      item.score = static_cast<float>(match.score);
      item.position_diff = static_cast<float>(match.xyz_error);
      item.yaw_diff = static_cast<float>(match.angle_error);
      item.center_x = observation.armor.center.x;
      item.center_y = observation.armor.center.y;
      item.predicted_yaw =
          static_cast<float>(GetArmorYawFromState(ekf_prediction, match.id));
      item.measured_yaw = static_cast<float>(match.measured_yaw);
    };
    fill_item(0, pair_match.left, pair_match.left_match);
    fill_item(1, pair_match.right, pair_match.right_match);

    const int previous_face = rt_.tracked_face_index;
    SpUpdatePair(pair_match);
    rt_.tracked_armor = pair_match.tracked_armor;
    rt_.tracked_id = pair_match.tracked_armor.number;
    rt_.tracked_armors_num =
        static_cast<ArmorsNum>(SpArmorCountFor(pair_match.tracked_armor));
    rt_.tracked_face_index = pair_match.tracked_face;
    rt_.last_yaw = pair_match.tracked_match.measured_yaw;
    rt_.info_position_diff = pair_match.tracked_match.xyz_error;
    rt_.info_yaw_diff = pair_match.tracked_match.angle_error;
    rt_.update_count++;
    if (pair_match.tracked_face != previous_face)
    {
      rt_.switch_count++;
    }
    SyncGeometryRuntimeFromState();
    ekf_.ekf.SetState(ekf_.state);
    XR_LOG_DEBUG(
        "SP pair tracker update: tracked_face=%d left_face=%d right_face=%d score=%.3f err=(left_xyz=%.3f right_xyz=%.3f left_angle=%.3f right_angle=%.3f) dz=%.4f",
        pair_match.tracked_face, pair_match.left_face, pair_match.right_face,
        pair_match.score, pair_match.left_match.xyz_error,
        pair_match.right_match.xyz_error, pair_match.left_match.angle_error,
        pair_match.right_match.angle_error,
        ekf_.state(ExtendedKalmanFilter::DELTA_Z));
  }

  if (!has_pair_match)
  {
    const int previous_face = rt_.tracked_face_index;
    face_selection = armor_tracker::SelectFaceMatch(
        armors_msg, BuildFaceSelectionTrackedState(), face_policy,
        GetCameraWorldPosition(),
        ekf_prediction(ExtendedKalmanFilter::V_YAW),
        [this](std::size_t armor_index)
        {
          return FindDetectionTrackId(armor_index);
        },
        [this](std::size_t armor_index)
        {
          return IsDetectionTrackConfirmed(armor_index);
        },
        [this, &ekf_prediction](int local_face_index)
        {
          return GetArmorPositionFromState(
              ekf_prediction, LocalFaceToCanonicalFace(local_face_index));
        },
        [this, &ekf_prediction](int local_face_index)
        {
          return GetArmorYawFromState(
              ekf_prediction, LocalFaceToCanonicalFace(local_face_index));
        });
    audit_selection = &face_selection;
    FillCandidateDebugFromSelection(face_selection, candidate_debug);
    matched = ApplyFaceSelection(face_selection, candidate_debug,
                                 pair_delta_z_mode);
    if (matched)
    {
      rt_.update_count++;
      if (rt_.tracked_face_index != previous_face)
      {
        rt_.switch_count++;
      }
    }
  }

  if (matched && SpStateDiverged())
  {
    XR_LOG_DEBUG("SP tracker target diverged: r1=%.3f r2=%.3f",
                 ekf_.state(ExtendedKalmanFilter::ROBOT_R),
                 ekf_.state(ExtendedKalmanFilter::ROBOT_R) +
                     ekf_.state(ExtendedKalmanFilter::DELTA_R));
    matched = false;
    candidate_debug.matched = 0;
    rt_.state = State::LOST;
    rt_.tracked_id = ArmorNumber::INVALID;
    rt_.detect_count = 0;
    rt_.lost_count = 0;
  }
  rt_.measurement_valid_current_frame = matched;
  bool state_matched = matched;
  if (matched)
  {
    rt_.suspect_count = 0;
  }
  else if (audit_selection != nullptr && audit_selection->has_same_number_candidate &&
           rt_.state == State::TRACKING)
  {
    ++rt_.suspect_count;
    state_matched = rt_.suspect_count <= 3;
  }
  else
  {
    rt_.suspect_count = 0;
  }

  if (rt_.state != State::LOST)
  {
    AdvanceTrackerState(state_matched);
  }

  candidate_debug_msg_ = candidate_debug;
  candidate_debug_msg_.tracked_face_track_id_valid =
      rt_.tracked_face_track_id_valid ? 1 : 0;
  candidate_debug_msg_.tracked_face_track_id =
      rt_.tracked_face_track_id_valid ? static_cast<int16_t>(rt_.tracked_face_track_id)
                                      : static_cast<int16_t>(-1);
  WriteStateAuditRow(image_timestamp_us, ekf_prediction, audit_selection, matched);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::AdvanceTrackerState(bool matched)
{
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
      return;
    }

    rt_.detect_count = 0;
    rt_.state = State::LOST;
    return;
  }

  if (rt_.state == State::TRACKING)
  {
    if (!matched)
    {
      rt_.state = State::TEMP_LOST;
      rt_.lost_count++;
    }
    return;
  }

  if (rt_.state == State::TEMP_LOST)
  {
    if (!matched)
    {
      rt_.lost_count++;
      if (rt_.lost_count > rt_.lost_thres)
      {
        rt_.lost_count = 0;
        rt_.state = State::LOST;
      }
      return;
    }

    rt_.state = State::TRACKING;
    rt_.lost_count = 0;
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::TryRecoverTempLost(
    const ArmorDetectorResults& armors_msg, CandidateDebugMsg& candidate_debug)
{
  if (!TempLostRecoveryEnabled() || rt_.state != State::TEMP_LOST || armors_msg.empty())
  {
    return false;
  }

  struct RecoveryCandidate
  {
    std::size_t armor_index = 0;
    ArmorDetectorResult armor{};
    int image_track_id = -1;
    bool confirmed_image_track = false;
    double score = DBL_MAX;
    double image_center = DBL_MAX;
    double area_score = 0.0;
    double frontality = 0.0;
    double measured_yaw = 0.0;
    std::array<double, 4> phase_position_diff{};
    std::array<double, 4> phase_yaw_diff{};
  };

  RecoveryCandidate best{};
  bool found = false;
  const Eigen::Vector3d camera_world = GetCameraWorldPosition();
  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    if (rt_.tracked_id != ArmorNumber::INVALID && armor.number != rt_.tracked_id)
    {
      continue;
    }
    if (rt_.tracked_armor.type != ArmorType::INVALID &&
        armor.type != rt_.tracked_armor.type)
    {
      continue;
    }

    const auto p = armor.pose.translation;
    const Eigen::Vector3d position_vec(p.x(), p.y(), p.z());
    const Eigen::Vector3d armor_front =
        armor.pose.rotation.ToRotationMatrix() * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d armor_to_camera = camera_world - position_vec;
    double frontality = 0.0;
    if (armor_to_camera.norm() > 1e-6)
    {
      frontality =
          std::max(0.0, armor_front.normalized().dot(armor_to_camera.normalized()));
    }

    const double image_area = armor_tracker::ArmorImageArea(armor);
    const double area_score = std::min(image_area / 2500.0, 1.0);
    const double image_center = armor.distance_to_image_center;
    const int image_track_id = FindDetectionTrackId(armor_index);
    const bool confirmed_image_track = IsDetectionTrackConfirmed(armor_index);
    const bool same_persistent_track =
        rt_.tracked_face_track_id_valid && confirmed_image_track &&
        image_track_id >= 0 &&
        static_cast<uint16_t>(image_track_id) == rt_.tracked_face_track_id;
    if (!same_persistent_track &&
        (image_center > 220.0 || area_score < 0.12))
    {
      continue;
    }

    // TEMP_LOST 已经说明 EKF 预测不可靠；恢复时只相信目标身份、视角质量和图像连续性。
    double score = image_center / 320.0 - 0.55 * area_score - 0.45 * frontality;
    if (same_persistent_track)
    {
      score -= 0.70;
    }
    else if (confirmed_image_track)
    {
      score -= 0.20;
    }

    if (score < best.score)
    {
      found = true;
      best.armor_index = armor_index;
      best.armor = armor;
      best.image_track_id = image_track_id;
      best.confirmed_image_track = confirmed_image_track;
      best.score = score;
      best.image_center = image_center;
      best.area_score = area_score;
      best.frontality = frontality;
      best.measured_yaw =
          armor_tracker::MeasuredArmorYawNear(armor, rt_.last_yaw);
      for (int local_face = 0; local_face < 4; ++local_face)
      {
        const Eigen::Vector3d predicted =
            GetArmorPositionFromState(ekf_.state, local_face);
        best.phase_position_diff[local_face] = (predicted - position_vec).norm();
        best.phase_yaw_diff[local_face] = armor_tracker::AngularDiffAbs(
            best.measured_yaw, GetArmorYawFromState(ekf_.state, local_face));
      }
    }
  }

  if (!found)
  {
    return false;
  }

  const int tracked_face_before_recover = rt_.tracked_face_index;
  int recovered_face_phase = 0;
  double recovered_face_cost = DBL_MAX;
  for (int phase = 0; phase < 4; ++phase)
  {
    const double cost =
        best.phase_position_diff[phase] + 0.15 * best.phase_yaw_diff[phase];
    if (cost < recovered_face_cost)
    {
      recovered_face_cost = cost;
      recovered_face_phase = phase;
    }
  }
  rt_.tracked_face_index = recovered_face_phase;
  rt_.face_switch_cooldown_remaining = 0.0;
  rt_.tracked_armor = best.armor;
  rt_.last_yaw = best.measured_yaw;
  rt_.tracked_face_track_id_valid =
      best.image_track_id >= 0 && best.confirmed_image_track;
  rt_.tracked_face_track_id = best.image_track_id >= 0
                                  ? static_cast<uint16_t>(best.image_track_id)
                                  : 0;
  rt_.face_track_id_valid.fill(false);
  rt_.face_track_id.fill(0);
  if (rt_.tracked_face_track_id_valid)
  {
    rt_.face_track_id_valid[0] = true;
    rt_.face_track_id[0] = rt_.tracked_face_track_id;
  }

  RecenterTrackedStateToMeasurement(best.armor, 0, best.measured_yaw);
  ekf_.measurement_face_index = 0;
  ekf_.measurement_geometry_mode = EKFBlock::MeasurementGeometryMode::FULL_BODY;
  const auto p = best.armor.pose.translation;
  ekf_.measurement =
      Eigen::Vector4d(p.x(), p.y(), p.z(), best.measured_yaw);
  ekf_.state = ekf_.ekf.Update(ekf_.measurement);
  ekf_.ekf.DecorrelatePosterior(
      {ExtendedKalmanFilter::ROBOT_R, ExtendedKalmanFilter::DELTA_R,
       ExtendedKalmanFilter::DELTA_Z});
  ekf_.measurement_geometry_mode = EKFBlock::MeasurementGeometryMode::FULL_BODY;
  SyncGeometryRuntimeFromState();
  SyncDzReferenceFromState();
  ClampGeometryState();

  rt_.lost_count = 0;
  rt_.recovery_count++;
  candidate_debug.matched = 1;
  candidate_debug.accepted_mode =
      static_cast<uint8_t>(armor_tracker::FaceSelectionAcceptedMode::RELAXED_SAME_FACE);
  XR_LOG_DEBUG(
      "Tracker TEMP_LOST recover: tracked_face_before=%d tracked_face_after=%d phase_cost=%.3f armor=%zu num=%d img=%d confirmed=%d score=%.3f center=%.1f area=%.3f frontality=%.3f yaw=%.3f phase_pos=[%.3f,%.3f,%.3f,%.3f] phase_yaw=[%.3f,%.3f,%.3f,%.3f]",
      tracked_face_before_recover,
      recovered_face_phase, recovered_face_cost,
      best.armor_index, static_cast<int>(best.armor.number), best.image_track_id,
      best.confirmed_image_track ? 1 : 0, best.score, best.image_center,
      best.area_score, best.frontality, best.measured_yaw,
      best.phase_position_diff[0], best.phase_position_diff[1],
      best.phase_position_diff[2], best.phase_position_diff[3],
      best.phase_yaw_diff[0], best.phase_yaw_diff[1], best.phase_yaw_diff[2],
      best.phase_yaw_diff[3]);
  return true;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::VelocityCallback(double velocity_msg)
{
  io_.solver->Init(velocity_msg);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ArmorsCallback(const DetectionMessage& message)
{
  const auto& source_frame = message.source_frame;
  if (source_frame.image_frame == nullptr)
  {
    XR_LOG_ERROR("ArmorTracker received detector packet without image frame");
    return;
  }
  if (source_frame.imu == nullptr)
  {
    XR_LOG_ERROR("ArmorTracker received detector packet without synced imu");
    return;
  }

  const uint64_t image_timestamp_us = source_frame.image_timestamp_us;
  if (source_frame.image_frame->timestamp_us != image_timestamp_us)
  {
    XR_LOG_ERROR(
        "ArmorTracker detector packet timestamp mismatch image=%llu packet=%llu",
        static_cast<unsigned long long>(source_frame.image_frame->timestamp_us),
        static_cast<unsigned long long>(image_timestamp_us));
    return;
  }

  ArmorDetectorResults armors_msg = message.results;
  const LibXR::Transform<double> camera_pose_world =
      ArmorTrackerCameraRotationToTrackerWorldPose(
          armor_tracker_detail::PackedCameraRotation(source_frame.imu->rotation_wxyz),
          armor_tracker_detail::PackedCameraTranslation(
              source_frame.imu->translation_xyz),
          io_.gimbal_to_camera_transform_static);
  io_.current_camera_pose = camera_pose_world;
  io_.current_camera_pose_valid = true;

  // 图像坐标 -> tracker 世界坐标。
  // 这里直接使用 sync 带来的相机动态位姿，避免把中心偏差继续挤进半径状态。
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

  // 构造消息
  TrackerInfo info_msg{};
  SolveTrajectory::Target target_msg{};
  ekf_msg_ = {};
  ekf_msg_.image_timestamp_us = image_timestamp_us;
  target_msg.image_timestamp_us = image_timestamp_us;
  target_msg.id = ArmorNumber::INVALID;

  auto time = LibXR::Timebase::GetMicroseconds();
  // 同步图像时间戳是 tracker 的运动模型基准；只有没有有效传感器时间时才退回进程时间。
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
  if (rt_.state != State::LOST && time_.dt > max_dt_before_reset)
  {
    // 大跳变说明旧 EKF 和装甲面绑定已经跨过长阻塞/暂停，继续外推只会污染后级。
    XR_LOG_WARN("ArmorTracker large dt %.3f s, reset tracker state", time_.dt);
    rt_ = TrackRuntime{};
    rt_.tracking_thres = cfg_.thresholds.tracking_thres;
    image_tracker_.Reset();
    time_.last_time = time;
    time_.last_image_timestamp_us = image_timestamp_us;
    time_.dt = 1.0 / 100.0;
  }

  UpdateImageIdTracks(armors_msg, image_timestamp_us);

  // 跟踪更新
  if (rt_.state == State::LOST)
  {
    Init(armors_msg);
    target_msg.tracking = false;
  }
  else
  {
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
      Eigen::VectorXd output_state = ekf_.state;
      const double output_dt = SpOutputExtrapolateSeconds();
      if (output_dt > 0.0 && output_state.size() > ExtendedKalmanFilter::DELTA_Z)
      {
        output_state(ExtendedKalmanFilter::X_CENTER) +=
            output_state(ExtendedKalmanFilter::V_X_CENTER) * output_dt;
        output_state(ExtendedKalmanFilter::Y_CENTER) +=
            output_state(ExtendedKalmanFilter::V_Y_CENTER) * output_dt;
        output_state(ExtendedKalmanFilter::Z_ARMOR) +=
            output_state(ExtendedKalmanFilter::V_Z_ARMOR) * output_dt;
        output_state(ExtendedKalmanFilter::YAW) =
            SpLimitRad(output_state(ExtendedKalmanFilter::YAW) +
                       output_state(ExtendedKalmanFilter::V_YAW) * output_dt);
      }
      const auto& state = output_state;
      target_msg.id = rt_.tracked_id;
      target_msg.measured_face_valid = false;
      target_msg.measured_face_index = -1;
      if (SingleArmorModeEnabled())
      {
        const Eigen::Vector3d armor_pos = GetArmorPositionFromState(state, 0);
        target_msg.armors_num = 1;
        target_msg.position.x() = armor_pos.x();
        target_msg.position.y() = armor_pos.y();
        target_msg.position.z() = armor_pos.z();
        target_msg.velocity.x() = state(1);
        target_msg.velocity.y() = state(3);
        target_msg.velocity.z() = state(5);
        target_msg.yaw = GetArmorYawFromState(state, 0);
        target_msg.v_yaw = state(7);
        target_msg.radius_1 = 0.0;
        target_msg.radius_2 = 0.0;
        target_msg.dz = 0.0;
      }
      else
      {
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
        if (SpMeasurementAnchoredOutputEnabled() &&
            rt_.measurement_valid_current_frame)
        {
          target_msg.measured_face_valid = true;
          target_msg.measured_face_index =
              std::clamp(ekf_.measurement_face_index, 0,
                         std::max(1, static_cast<int>(rt_.tracked_armors_num)) - 1);
          target_msg.measured_face_position = Eigen::Vector3d(
              ekf_.measurement(0), ekf_.measurement(1), ekf_.measurement(2));
          target_msg.measured_face_yaw = ekf_.measurement(3);
        }
      }

      XR_LOG_DEBUG(
          "Target position: (%.3f, %.3f, %.3f) velocity: (%.3f, %.3f, "
          "%.3f) yaw: %.3f "
          "v_yaw: %.3f radius_1: %.3f radius_2: %.3f dz: %.3f",
          target_msg.position.x(), target_msg.position.y(), target_msg.position.z(),
          target_msg.velocity.x(), target_msg.velocity.y(), target_msg.velocity.z(),
          target_msg.yaw, target_msg.v_yaw, target_msg.radius_1, target_msg.radius_2,
          target_msg.dz);

      Eigen::Vector3d pw_center, pw_armors[4];
      if (SingleArmorModeEnabled())
      {
        const Eigen::Vector3d armor_pos = GetArmorPositionFromState(state, 0);
        pw_center = armor_pos;
        pw_armors[0] = armor_pos;
        for (int i = 1; i < 4; ++i)
        {
          pw_armors[i] = armor_pos;
        }
      }
      else
      {
        const auto& st = state;  // [xc,vxc,yc,vyc,za,vza,yaw,vyaw,r1]
        const double XC = st(0), YC = st(2), ZA = st(4);
        double center_z = ZA;
        if (rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
        {
          center_z += rt_.dz * 0.5;
        }

        pw_center = {XC, YC, center_z};
        const int publish_armor_count =
            std::max(1, std::min(4, static_cast<int>(rt_.tracked_armors_num)));
        for (int i = 0; i < publish_armor_count; ++i)
        {
          pw_armors[i] = GetArmorPositionFromState(st, i);
        }
        if (SpMeasurementAnchoredOutputEnabled() &&
            rt_.measurement_valid_current_frame)
        {
          const int face_index =
              std::clamp(ekf_.measurement_face_index, 0, publish_armor_count - 1);
          pw_armors[face_index] = Eigen::Vector3d(
              ekf_.measurement(0), ekf_.measurement(1), ekf_.measurement(2));
        }
        for (int i = publish_armor_count; i < 4; ++i)
        {
          pw_armors[i] = pw_center;
        }
      }

      // === 计算 相机←世界 外参：T_CW = T_WC^-1 ===
      const LibXR::Transform<double> t_wc = camera_pose_world;
      auto r_wc = t_wc.rotation.ToRotationMatrix();
      Eigen::Matrix3d r_cw = r_wc.transpose();  // 相机←世界 旋转
      Eigen::Vector3d twc(t_wc.translation.x(), t_wc.translation.y(),
                          t_wc.translation.z());

      // === 变到相机系并发布 ===
      ekf_msg_.count = SingleArmorModeEnabled() ? 1 : static_cast<uint8_t>(rt_.tracked_armors_num);

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
        if (i < static_cast<int>(rt_.tracked_armors_num))
        {
          ekf_msg_.valid[i + 1] = to_cam(pw_armors[i], ekf_msg_.armors_cam[i]);
        }
        else
        {
          ekf_msg_.valid[i + 1] = false;
          ekf_msg_.armors_cam[i] = ekf_msg_.center_cam;
        }
      }
    }
  }

  time_.last_time = time;
  time_.last_image_timestamp_us = image_timestamp_us;

  candidate_debug_msg_.image_timestamp_us = image_timestamp_us;
  io_.candidate_debug_topic.Publish(candidate_debug_msg_);
  // ekf_points 是运行期数据合约，供预览、录像和 truth 对齐工具消费。
  io_.ekf_points_topic.Publish(ekf_msg_);
  io_.target_topic.Publish(target_msg);
}
