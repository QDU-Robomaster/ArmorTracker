#pragma once

/**
 * @file ArmorTrackerPipeline.hpp
 * @brief ArmorTracker 每帧处理流水线、状态机和 detector topic 回调实现。
 *
 * 该文件负责把 detector 输出、相机同步姿态、整车模型 EKF 更新、调试消息和最终
 * tracker/target 发布串成一条运行期流水线。
 */

/**
 * @brief 从当前帧检测结果中选择初始目标并初始化 EKF。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::Init(const ArmorDetectorResults& armors_msg)
{
  if (armors_msg.empty())
  {
    return;
  }

  double best_score = DBL_MAX;
  ArmorPriority best_priority = ArmorPriority::FIFTH;
  std::size_t tracked_index = 0;
  rt_.tracked_armor = armors_msg[0];
  bool found_stable_candidate = false;
  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    const int detection_track_id = FindDetectionTrackId(armor_index);
    const bool confirmed_track = IsDetectionTrackConfirmed(armor_index);
    const bool stable_observation =
        !ObservationQualityEnabled() ||
        armor_tracker::StableArmorObservation(
            armor, ObservationStableMaxReprojectionPx(),
            ObservationStableMinAreaPx(), ObservationStableMinConfidence());
    if (!stable_observation && InitRequiresStableObservation())
    {
      continue;
    }

    const double range =
        std::sqrt(std::pow(armor.pose.translation.x(), 2.0) +
                  std::pow(armor.pose.translation.y(), 2.0) +
                  std::pow(armor.pose.translation.z(), 2.0));
    const double quality_penalty =
        ObservationQualityEnabled()
            ? armor_tracker::ArmorObservationQualityPenalty(
                  armor, ObservationStableMaxReprojectionPx(),
                  ObservationStableMinAreaPx(), ObservationStableMinConfidence())
            : 0.0;
    double score =
        static_cast<double>(static_cast<int>(armor.priority)) +
        0.22 * armor.distance_to_image_center / 320.0 +
        0.10 * range / std::max(cfg_.limits.max_armor_distance, 1e-6) +
        ObservationQualityScoreWeight() * quality_penalty -
        0.20 * static_cast<double>(armor.confidence);
    if (confirmed_track && detection_track_id >= 0)
    {
      score -= ObservationConfirmedTrackBonus();
    }
    if (stable_observation)
    {
      found_stable_candidate = true;
      score -= 0.12;
    }

    if (score < best_score - 1e-6 ||
        (std::abs(score - best_score) <= 1e-6 &&
         (static_cast<int>(armor.priority) < static_cast<int>(best_priority) ||
          (armor.priority == best_priority &&
           armor.distance_to_image_center <
               rt_.tracked_armor.distance_to_image_center))))
    {
      best_priority = armor.priority;
      best_score = score;
      tracked_index = armor_index;
      rt_.tracked_armor = armor;
    }
  }
  if (InitRequiresStableObservation() && !found_stable_candidate)
  {
    XR_LOG_DEBUG("Tracker init skipped: no stable observation among %u detections",
                 static_cast<unsigned>(armors_msg.size()));
    return;
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
      static_cast<ArmorsNum>(VehicleArmorCountFor(rt_.tracked_armor));
  rt_.model_initial_phase_resolved = false;
  rt_.model_pair_delta_z_valid = false;
  rt_.measurement_valid_current_frame = false;
  rt_.center_motion_observer_valid = false;
  rt_.center_motion_observer_samples = 0;
  rt_.center_motion_observer_confidence = 0.0;
  rt_.yaw_rate_observer_valid = false;
  rt_.yaw_rate_observer_samples = 0;
  rt_.model_range_filter_valid = false;
  rt_.output_anchor_delta_valid = false;
  InitEKF(rt_.tracked_armor);
  rt_.model_initial_phase_resolved =
      VehicleTryCanonicalizeInitialState(armors_msg, true);
  if ((VehiclePairGeometryEnabled() || VehiclePairDeltaZEnabled()) &&
      rt_.tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    VehiclePairMatch init_pair_match{};
    if (VehicleResolvePairMatch(armors_msg, ekf_.state, init_pair_match))
    {
      VehicleApplyPairGeometryUpdate(init_pair_match);
      if (init_pair_match.dz_valid)
      {
        rt_.model_pair_delta_z_valid = true;
      }
      SyncGeometryRuntimeFromState();
      ekf_.ekf.SetState(ekf_.state);
      XR_LOG_DEBUG(
          "whole-body model pair shape init: left=(%u/%d) right=(%u/%d) score=%.3f geom=%d dz_valid=%d r=(%.3f,%.3f) dz=%.4f",
          static_cast<unsigned>(init_pair_match.left.armor_index), init_pair_match.left_face,
          static_cast<unsigned>(init_pair_match.right.armor_index), init_pair_match.right_face,
          init_pair_match.score, init_pair_match.geometry_valid ? 1 : 0,
          init_pair_match.dz_valid ? 1 : 0,
          ekf_.state(ExtendedKalmanFilter::ROBOT_R),
          ekf_.state(ExtendedKalmanFilter::ROBOT_R) +
              ekf_.state(ExtendedKalmanFilter::DELTA_R),
          ekf_.state(ExtendedKalmanFilter::DELTA_Z));
    }
  }
  XR_LOG_DEBUG("Init EKF!");

  rt_.state = State::DETECTING;
  rt_.detect_count = 1;
  rt_.lost_count = 0;
  rt_.update_count = 0;
  rt_.switch_count = 0;
  rt_.suspect_count = 0;
  candidate_debug_msg_ = CandidateDebugMsg{};
}

/**
 * @brief 对一帧检测结果执行预测、候选选择、EKF 更新和状态审计。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::Update(const ArmorDetectorResults& armors_msg,
                                       uint64_t image_timestamp_us)
{
  if (SingleArmorModeEnabled())
  {
    UpdateSingleArmorMode(armors_msg, image_timestamp_us);
    return;
  }

  VehiclePredict();
  Eigen::VectorXd ekf_prediction = ekf_.state;
  if (VehicleCanonicalInitEnabled())
  {
    if (rt_.update_count <=
        static_cast<int>(VehicleCanonicalInitMaxUpdates()))
    {
      if (VehicleTryCanonicalizeInitialState(
              armors_msg, !rt_.model_initial_phase_resolved))
      {
        rt_.model_initial_phase_resolved = true;
        ekf_prediction = ekf_.state;
      }
    }
  }
  XR_LOG_DEBUG("whole-body model tracker predict");
  bool matched = false;
  rt_.measurement_valid_current_frame = false;
  const bool freeze_single_observation_delta_z =
      VehicleFreezeSingleObservationDeltaZEnabled() && VehiclePairDeltaZEnabled() &&
      rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
      rt_.state == State::TRACKING;
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

  std::size_t selected_armor_index = 0;
  int selected_canonical_face = -1;
  if (face_selection.has_selected_candidate)
  {
    selected_armor_index = face_selection.selected_candidate.armor_index;
    selected_canonical_face =
        LocalFaceToCanonicalFace(face_selection.selected_candidate.face_index);
  }

  matched = ApplyFaceSelection(face_selection, candidate_debug,
                               freeze_single_observation_delta_z,
                               image_timestamp_us);
  if (matched)
  {
    rt_.update_count++;
    if (rt_.tracked_face_index != previous_face)
    {
      rt_.switch_count++;
    }
  }

  const bool pair_shape_update_mode =
      (VehiclePairGeometryEnabled() || VehiclePairDeltaZEnabled()) &&
      rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
      rt_.tracked_id != ArmorNumber::INVALID;
  VehiclePairMatch pair_match{};
  if (pair_shape_update_mode &&
      VehicleResolvePairMatch(armors_msg, matched ? ekf_.state : ekf_prediction,
                         pair_match))
  {
    bool accept_pair_shape = false;
    if (matched && selected_canonical_face >= 0)
    {
      const bool selected_is_left =
          pair_match.left.armor_index == selected_armor_index &&
          pair_match.left_face == selected_canonical_face;
      const bool selected_is_right =
          pair_match.right.armor_index == selected_armor_index &&
          pair_match.right_face == selected_canonical_face;
      accept_pair_shape = selected_is_left || selected_is_right;
    }
    else
    {
      // 未匹配帧只允许高自洽的双板 shape 观测修半径；它不改变本帧 matched 状态。
      // 已经 TRACKING 时的偶发 miss 不用 pair 改 shape，避免噪声观测污染稳定跟踪。
      accept_pair_shape = rt_.state != State::TRACKING &&
                          pair_match.geometry_valid &&
                          pair_match.score <=
                              armor_tracker_detail::VehiclePairGeometryFallbackMaxScore();
    }

    if (accept_pair_shape)
    {
      VehicleApplyPairGeometryUpdate(pair_match);
      if (pair_match.dz_valid)
      {
        rt_.model_pair_delta_z_valid = true;
      }
      SyncGeometryRuntimeFromState();
      ekf_.ekf.SetState(ekf_.state);
      XR_LOG_DEBUG(
          "whole-body model pair shape update: matched=%d selected=(armor=%u face=%d) left=(%u/%d) right=(%u/%d) score=%.3f geom=%d dz_valid=%d r=(%.3f,%.3f) dz=%.4f",
          matched ? 1 : 0, static_cast<unsigned>(selected_armor_index), selected_canonical_face,
          static_cast<unsigned>(pair_match.left.armor_index), pair_match.left_face,
          static_cast<unsigned>(pair_match.right.armor_index), pair_match.right_face, pair_match.score,
          pair_match.geometry_valid ? 1 : 0, pair_match.dz_valid ? 1 : 0,
          ekf_.state(ExtendedKalmanFilter::ROBOT_R),
          ekf_.state(ExtendedKalmanFilter::ROBOT_R) +
              ekf_.state(ExtendedKalmanFilter::DELTA_R),
          ekf_.state(ExtendedKalmanFilter::DELTA_Z));
    }
    else
    {
      XR_LOG_DEBUG(
          "whole-body model pair shape skipped: matched=%d selected=(armor=%u face=%d) pair=(left=%u/%d right=%u/%d) score=%.3f",
          matched ? 1 : 0, static_cast<unsigned>(selected_armor_index), selected_canonical_face,
          static_cast<unsigned>(pair_match.left.armor_index), pair_match.left_face,
          static_cast<unsigned>(pair_match.right.armor_index), pair_match.right_face, pair_match.score);
    }
  }

  if (matched && VehicleStateDiverged())
  {
    XR_LOG_DEBUG("whole-body model tracker target diverged: r1=%.3f r2=%.3f",
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

/**
 * @brief 根据本帧是否匹配推进 LOST/DETECTING/TRACKING/TEMP_LOST 状态机。
 */
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

/**
 * @brief 在 TEMP_LOST 状态下用身份、图像连续性和视角质量尝试快速恢复。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::TryRecoverTempLost(
    const ArmorDetectorResults& armors_msg, CandidateDebugMsg& candidate_debug)
{
  if (!TempLostRecoveryEnabled() || rt_.state != State::TEMP_LOST || armors_msg.empty())
  {
    return false;
  }

  /**
   * @brief TEMP_LOST 快速恢复的候选观测。
   */
  struct RecoveryCandidate
  {
    std::size_t armor_index = 0;             ///< detector 结果索引。
    ArmorDetectorResult armor{};             ///< detector 装甲结果。
    int image_track_id = -1;                 ///< 图像 track ID。
    bool confirmed_image_track = false;      ///< 图像 track 是否已确认。
    double score = DBL_MAX;                  ///< 恢复候选分，越小越好。
    double image_center = DBL_MAX;           ///< 到图像中心距离。
    double area_score = 0.0;                 ///< 图像面积归一化分。
    double frontality = 0.0;                 ///< 装甲朝向相机程度。
    double measured_yaw = 0.0;               ///< 展开后的测量 yaw。
    std::array<double, 4> phase_position_diff{}; ///< 各相位位置差。
    std::array<double, 4> phase_yaw_diff{};       ///< 各相位 yaw 差。
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

  RecenterTrackedStateToMeasurement(best.armor, recovered_face_phase,
                                    best.measured_yaw);
  ekf_.measurement_face_index = recovered_face_phase;
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
      "Tracker TEMP_LOST recover: tracked_face_before=%d tracked_face_after=%d phase_cost=%.3f armor=%u num=%d img=%d confirmed=%d score=%.3f center=%.1f area=%.3f frontality=%.3f yaw=%.3f phase_pos=[%.3f,%.3f,%.3f,%.3f] phase_yaw=[%.3f,%.3f,%.3f,%.3f]",
      tracked_face_before_recover,
      recovered_face_phase, recovered_face_cost,
      static_cast<unsigned>(best.armor_index), static_cast<int>(best.armor.number), best.image_track_id,
      best.confirmed_image_track ? 1 : 0, best.score, best.image_center,
      best.area_score, best.frontality, best.measured_yaw,
      best.phase_position_diff[0], best.phase_position_diff[1],
      best.phase_position_diff[2], best.phase_position_diff[3],
      best.phase_yaw_diff[0], best.phase_yaw_diff[1], best.phase_yaw_diff[2],
      best.phase_yaw_diff[3]);
  return true;
}

/**
 * @brief detector 结果 topic 回调，完成坐标转换、过滤、跟踪更新和 topic 发布。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ArmorsCallback(
    typename ArmorTracker<CameraInfoV>::DetectionMessageArg message)
{
  const ArmorDetectionsSourceFrame<CameraInfoV>* source_frame_ptr = nullptr;
  const ArmorDetectorResults* detections_ptr = nullptr;
  uint64_t detections_timestamp_us = 0;

  if constexpr (std::is_pointer<DetectionMessage>::value)
  {
    if (message == nullptr)
    {
      XR_LOG_ERROR("ArmorTracker received empty detector packet pointer");
      return;
    }
    if (message->detections == nullptr)
    {
      XR_LOG_ERROR("ArmorTracker received detector packet without detections");
      return;
    }
    source_frame_ptr = &message->source_frame;
    detections_ptr = &message->detections->results;
    detections_timestamp_us = message->detections->image_timestamp_us;
  }
  else
  {
    source_frame_ptr = &message.source_frame;
    detections_ptr = &message.results;
    detections_timestamp_us = message.source_frame.image_timestamp_us;
  }

  const auto& source_frame = *source_frame_ptr;
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
        "ArmorTracker detector packet timestamp mismatch image=%u packet=%u",
        static_cast<unsigned>(source_frame.image_frame->timestamp_us),
        static_cast<unsigned>(image_timestamp_us));
    return;
  }
  if (detections_timestamp_us != image_timestamp_us)
  {
    XR_LOG_ERROR(
        "ArmorTracker detector result timestamp mismatch result=%u packet=%u",
        static_cast<unsigned>(detections_timestamp_us),
        static_cast<unsigned>(image_timestamp_us));
    return;
  }

  const ArmorDetectorResults detector_preview_armors = *detections_ptr;
  ArmorDetectorResults armors_msg = detector_preview_armors;
  armors_msg.erase(
      std::remove_if(armors_msg.begin(), armors_msg.end(),
                     [](const ArmorDetectorResult& armor)
                     {
                       return !armor.pnp_valid;
                     }),
      armors_msg.end());

  const LibXR::Transform<double> camera_pose_world =
      ArmorTrackerCameraRotationToTrackerWorldPose(
          armor_tracker_detail::PackedCameraRotation(source_frame.imu->rotation_wxyz),
          armor_tracker_detail::PackedCameraTranslation(
              source_frame.imu->translation_xyz),
          io_.gimbal_to_camera_transform_static,
          io_.camera_pose_runtime);
  io_.current_camera_pose = camera_pose_world;
  io_.current_camera_pose_valid = true;

  // 图像坐标 -> tracker 世界坐标。
  // 这里直接使用 sync 带来的相机动态位姿，避免把中心偏差继续挤进半径状态。
  for (auto& armor : armors_msg)
  {
    LibXR::Transform<double> tf = armor.pose;
    armor.pose = camera_pose_world + tf;
  }
  OptimizeArmorYawMeasurements(armors_msg, camera_pose_world);

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
  ArmorTrackerTarget target_msg{};
  ekf_msg_ = {};
  ekf_msg_.image_timestamp_us = image_timestamp_us;
  target_msg.image_timestamp_us = image_timestamp_us;
  target_msg.id = ArmorNumber::INVALID;
  target_msg.velocity_confidence = 0.0;
  const LibXR::MicrosecondTimestamp publish_timestamp(image_timestamp_us);

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
    io_.info_topic.Publish(info_msg, publish_timestamp);

    if (rt_.state == State::DETECTING)
    {
      target_msg.tracking = false;
    }
    else if (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST)
    {
      target_msg.tracking = true;
      Eigen::VectorXd output_state = ekf_.state;
      const double output_dt = VehicleOutputExtrapolateSeconds();
      if (output_dt > 0.0 && output_state.size() > ExtendedKalmanFilter::DELTA_Z)
      {
        output_state(ExtendedKalmanFilter::X_CENTER) +=
            output_state(ExtendedKalmanFilter::V_X_CENTER) * output_dt;
        output_state(ExtendedKalmanFilter::Y_CENTER) +=
            output_state(ExtendedKalmanFilter::V_Y_CENTER) * output_dt;
        output_state(ExtendedKalmanFilter::Z_ARMOR) +=
            output_state(ExtendedKalmanFilter::V_Z_ARMOR) * output_dt;
        output_state(ExtendedKalmanFilter::YAW) =
            VehicleLimitRad(output_state(ExtendedKalmanFilter::YAW) +
                       output_state(ExtendedKalmanFilter::V_YAW) * output_dt);
      }
      const auto& state = output_state;
      const Eigen::Vector3d ekf_velocity(
          state(ExtendedKalmanFilter::V_X_CENTER),
          state(ExtendedKalmanFilter::V_Y_CENTER),
          state(ExtendedKalmanFilter::V_Z_ARMOR));
      const Eigen::Vector3d output_velocity =
          VehicleCenterMotionObserverEnabled() && rt_.center_motion_observer_valid
              ? rt_.center_motion_observer_velocity
              : ekf_velocity;
      Eigen::Vector3d velocity_variance = Eigen::Vector3d::Zero();
      if (ekf_.covariance.rows() > ExtendedKalmanFilter::V_Z_ARMOR &&
          ekf_.covariance.cols() > ExtendedKalmanFilter::V_Z_ARMOR)
      {
        velocity_variance.x() = std::max(
            0.0, ekf_.covariance(ExtendedKalmanFilter::V_X_CENTER,
                                 ExtendedKalmanFilter::V_X_CENTER));
        velocity_variance.y() = std::max(
            0.0, ekf_.covariance(ExtendedKalmanFilter::V_Y_CENTER,
                                 ExtendedKalmanFilter::V_Y_CENTER));
        velocity_variance.z() = std::max(
            0.0, ekf_.covariance(ExtendedKalmanFilter::V_Z_ARMOR,
                                 ExtendedKalmanFilter::V_Z_ARMOR));
      }
      const double xy_velocity_sigma =
          std::sqrt(std::max(velocity_variance.x(), velocity_variance.y()));
      double covariance_confidence =
          std::clamp((1.20 - xy_velocity_sigma) / (1.20 - 0.20), 0.0, 1.0);
      if (!std::isfinite(covariance_confidence))
      {
        covariance_confidence = 0.0;
      }
      double velocity_confidence =
          VehicleCenterMotionObserverEnabled() && rt_.center_motion_observer_valid
              ? rt_.center_motion_observer_confidence
              : covariance_confidence;
      if (rt_.state == State::TEMP_LOST || !rt_.measurement_valid_current_frame ||
          !output_velocity.allFinite())
      {
        velocity_confidence *= 0.25;
      }
      target_msg.velocity_variance = velocity_variance;
      target_msg.velocity_confidence = std::clamp(velocity_confidence, 0.0, 1.0);
      target_msg.id = rt_.tracked_id;
      target_msg.tracked_face_index = std::clamp(
          rt_.tracked_face_index, 0,
          std::max(1, static_cast<int>(rt_.tracked_armors_num)) - 1);
      target_msg.face_switch_observed = rt_.switch_count > 0;
      target_msg.use_measured_face_anchor = VehicleAimerMeasuredFaceAnchorEnabled();
      target_msg.measured_face_valid = false;
      target_msg.measured_face_index = -1;
      if (SingleArmorModeEnabled())
      {
        const Eigen::Vector3d armor_pos = GetArmorPositionFromState(state, 0);
        target_msg.armors_num = 1;
        target_msg.position.x() = armor_pos.x();
        target_msg.position.y() = armor_pos.y();
        target_msg.position.z() = armor_pos.z();
        target_msg.velocity = output_velocity;
        target_msg.yaw = GetArmorYawFromState(state, 0);
        target_msg.v_yaw = state(7);
        target_msg.radius_1 = 0.0;
        target_msg.radius_2 = 0.0;
        target_msg.dz = 0.0;
        VehicleApplyYawRateObserver(target_msg.yaw, image_timestamp_us, target_msg);
      }
      else
      {
        target_msg.armors_num = static_cast<int>(rt_.tracked_armors_num);
        target_msg.position.x() = state(0);
        target_msg.position.y() = state(2);
        target_msg.position.z() = state(4);
        target_msg.velocity = output_velocity;
        target_msg.yaw = state(6);
        target_msg.v_yaw = state(7);
        target_msg.radius_1 = state(8);
        target_msg.radius_2 = rt_.another_r;
        target_msg.dz = rt_.dz;
        if (rt_.measurement_valid_current_frame)
        {
          target_msg.measured_face_valid = true;
          target_msg.measured_face_index =
              std::clamp(ekf_.measurement_face_index, 0,
                         std::max(1, static_cast<int>(rt_.tracked_armors_num)) - 1);
          target_msg.measured_face_position = Eigen::Vector3d(
              ekf_.measurement(0), ekf_.measurement(1), ekf_.measurement(2));
          target_msg.measured_face_yaw = ekf_.measurement(3);
        }
        VehicleApplyYawRateObserver(target_msg.yaw, image_timestamp_us, target_msg);
      }

      XR_LOG_DEBUG(
          "Target position: (%.3f, %.3f, %.3f) velocity: (%.3f, %.3f, "
          "%.3f) velocity_confidence: %.3f yaw: %.3f "
          "v_yaw: %.3f radius_1: %.3f radius_2: %.3f dz: %.3f",
          target_msg.position.x(), target_msg.position.y(), target_msg.position.z(),
          target_msg.velocity.x(), target_msg.velocity.y(), target_msg.velocity.z(),
          target_msg.velocity_confidence,
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
        if (VehicleMeasurementAnchoredOutputEnabled() &&
            rt_.measurement_valid_current_frame)
        {
          const int face_index =
              std::clamp(ekf_.measurement_face_index, 0, publish_armor_count - 1);
          const Eigen::Vector3d measured_face_position(
              ekf_.measurement(0), ekf_.measurement(1), ekf_.measurement(2));
          const Eigen::Vector3d desired_delta =
              measured_face_position - pw_armors[face_index];
          Eigen::Vector3d anchor_delta = desired_delta;
          if (desired_delta.allFinite())
          {
            const Eigen::Vector3d camera_world(
                camera_pose_world.translation.x(),
                camera_pose_world.translation.y(),
                camera_pose_world.translation.z());
            Eigen::Vector3d radial_dir = measured_face_position - camera_world;
            const double radial_norm = radial_dir.norm();
            if (radial_norm > 1e-6 && std::isfinite(radial_norm))
            {
              radial_dir /= radial_norm;
            }
            else
            {
              radial_dir = Eigen::Vector3d::Zero();
            }

            if (rt_.output_anchor_delta_valid)
            {
              const double radial_alpha = VehicleOutputMeasAnchorAlpha();
              const double lateral_alpha = VehicleOutputMeasAnchorLateralAlpha();
              const double desired_radial_scalar =
                  radial_dir.squaredNorm() > 0.0 ? desired_delta.dot(radial_dir) : 0.0;
              const Eigen::Vector3d desired_radial =
                  desired_radial_scalar * radial_dir;
              const Eigen::Vector3d desired_lateral =
                  desired_delta - desired_radial;
              const double previous_radial_scalar =
                  radial_dir.squaredNorm() > 0.0
                      ? rt_.output_anchor_delta.dot(radial_dir)
                      : 0.0;
              const Eigen::Vector3d previous_radial =
                  previous_radial_scalar * radial_dir;
              const Eigen::Vector3d previous_lateral =
                  rt_.output_anchor_delta - previous_radial;
              Eigen::Vector3d radial_delta =
                  (1.0 - radial_alpha) * previous_radial +
                  radial_alpha * desired_radial;
              const Eigen::Vector3d lateral_delta =
                  (1.0 - lateral_alpha) * previous_lateral +
                  lateral_alpha * desired_lateral;
              Eigen::Vector3d step = radial_delta - previous_radial;
              const double max_step = VehicleOutputMeasAnchorMaxStep();
              const double step_norm = step.norm();
              if (max_step > 0.0 && step_norm > max_step)
              {
                step *= max_step / step_norm;
                radial_delta = previous_radial + step;
              }
              anchor_delta = lateral_delta + radial_delta;
            }

            const double max_delta = VehicleOutputMeasAnchorMaxDelta();
            const double delta_norm = anchor_delta.norm();
            if (max_delta > 0.0 && delta_norm > max_delta)
            {
              anchor_delta *= max_delta / delta_norm;
            }

            rt_.output_anchor_delta = anchor_delta;
            rt_.output_anchor_delta_timestamp_us = image_timestamp_us;
            rt_.output_anchor_delta_valid = true;
          }
          else
          {
            rt_.output_anchor_delta_valid = false;
            anchor_delta = Eigen::Vector3d::Zero();
          }
          pw_center += anchor_delta;
          for (int i = 0; i < publish_armor_count; ++i)
          {
            pw_armors[i] += anchor_delta;
          }
        }
        else
        {
          rt_.output_anchor_delta_valid = false;
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
  io_.candidate_debug_topic.Publish(candidate_debug_msg_, publish_timestamp);
  // ekf_points 是运行期数据合约，供预览、录像和 truth 对齐工具消费。
  io_.ekf_points_topic.Publish(ekf_msg_, publish_timestamp);
  io_.target_topic.Publish(target_msg, publish_timestamp);
  SubmitPreview(*source_frame.image_frame, detector_preview_armors, target_msg,
                ekf_msg_, candidate_debug_msg_);
}

/**
 * @brief 提交 tracker 实时预览帧。
 *
 * 预览线程只消费这里捕获的消息快照，不读取下一帧复用的 tracker 成员缓存；
 * `VisionPreview::Submit()` 会先深拷贝图像，主链路不会被窗口绘制反压。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SubmitPreview(
    const ImageFrame& image_frame,
    const ArmorDetectorResults& detector_armors,
    const ArmorTrackerTarget& target_msg,
    const EkfPointsMsg& ekf_msg,
    const CandidateDebugMsg& candidate_debug_msg)
{
  if (!preview_.Running())
  {
    return;
  }

  int cv_type = -1;
  switch (kCameraInfo.encoding)
  {
    case CameraTypes::Encoding::RGB8:
    case CameraTypes::Encoding::BGR8:
      cv_type = CV_8UC3;
      break;
    case CameraTypes::Encoding::RGBA8:
    case CameraTypes::Encoding::BGRA8:
      cv_type = CV_8UC4;
      break;
    case CameraTypes::Encoding::MONO8:
      cv_type = CV_8UC1;
      break;
    default:
      break;
  }
  if (cv_type < 0)
  {
    return;
  }

  cv::Mat image(static_cast<int>(kCameraInfo.height),
                static_cast<int>(kCameraInfo.width), cv_type,
                const_cast<uint8_t*>(image_frame.data.data()),
                static_cast<size_t>(kCameraInfo.step));
  cv::Mat bgr_image;
  switch (kCameraInfo.encoding)
  {
    case CameraTypes::Encoding::RGB8:
      cv::cvtColor(image, bgr_image, cv::COLOR_RGB2BGR);
      break;
    case CameraTypes::Encoding::RGBA8:
      cv::cvtColor(image, bgr_image, cv::COLOR_RGBA2BGR);
      break;
    case CameraTypes::Encoding::BGRA8:
      cv::cvtColor(image, bgr_image, cv::COLOR_BGRA2BGR);
      break;
    case CameraTypes::Encoding::MONO8:
      cv::cvtColor(image, bgr_image, cv::COLOR_GRAY2BGR);
      break;
    case CameraTypes::Encoding::BGR8:
      bgr_image = image;
      break;
    default:
      return;
  }
  if (bgr_image.empty())
  {
    return;
  }

  struct PreviewArmorOverlay
  {
    bool valid = false;
    std::array<LibXR::Position<double>, 4> corners_cam{};
  };

  std::array<PreviewArmorOverlay, 4> tracker_overlay{};
  const int tracker_overlay_count =
      target_msg.tracking
          ? std::max(0, std::min(4, static_cast<int>(ekf_msg.count)))
          : 0;
  if (tracker_overlay_count > 0 && io_.current_camera_pose_valid)
  {
    const double half_width_m =
        ((rt_.tracked_armor.type == ArmorType::LARGE) ? 225.0 : 135.0) *
        0.5 / 1000.0;
    constexpr double half_height_m = 56.0 * 0.5 / 1000.0;
    const auto r_wc = io_.current_camera_pose.rotation.ToRotationMatrix();
    const Eigen::Matrix3d r_cw = r_wc.transpose();
    const double angle_step = 2.0 * M_PI / static_cast<double>(tracker_overlay_count);

    const auto to_eigen =
        [](const LibXR::Position<double>& point) -> Eigen::Vector3d
    {
      return {point.x(), point.y(), point.z()};
    };
    const auto to_position =
        [](const Eigen::Vector3d& point) -> LibXR::Position<double>
    {
      return {point.x(), point.y(), point.z()};
    };

    for (int i = 0; i < tracker_overlay_count; ++i)
    {
      if (!ekf_msg.valid[i + 1])
      {
        continue;
      }
      const double yaw = target_msg.yaw + angle_step * static_cast<double>(i);
      const Eigen::Vector3d width_world(-std::sin(yaw), std::cos(yaw), 0.0);
      const Eigen::Vector3d height_world(0.0, 0.0, 1.0);
      Eigen::Vector3d width_cam = r_cw * width_world;
      Eigen::Vector3d height_cam = r_cw * height_world;
      if (!width_cam.allFinite() || !height_cam.allFinite() ||
          width_cam.norm() < 1e-9 || height_cam.norm() < 1e-9)
      {
        continue;
      }
      width_cam.normalize();
      height_cam.normalize();

      const Eigen::Vector3d center_cam = to_eigen(ekf_msg.armors_cam[i]);
      if (!center_cam.allFinite())
      {
        continue;
      }
      tracker_overlay[i].corners_cam[0] =
          to_position(center_cam - half_width_m * width_cam -
                      half_height_m * height_cam);
      tracker_overlay[i].corners_cam[1] =
          to_position(center_cam + half_width_m * width_cam -
                      half_height_m * height_cam);
      tracker_overlay[i].corners_cam[2] =
          to_position(center_cam + half_width_m * width_cam +
                      half_height_m * height_cam);
      tracker_overlay[i].corners_cam[3] =
          to_position(center_cam - half_width_m * width_cam +
                      half_height_m * height_cam);
      tracker_overlay[i].valid = true;
    }
  }

  const auto camera_matrix = kCameraInfo.camera_matrix;
  preview_.Submit(
      bgr_image,
      [detector_armors, target_msg, ekf_msg, candidate_debug_msg,
       camera_matrix, tracker_overlay, tracker_overlay_count](cv::Mat& canvas)
      {
        for (const auto& armor : detector_armors)
        {
          const cv::Scalar color =
              armor.pnp_valid ? cv::Scalar(80, 220, 255) : cv::Scalar(120, 120, 120);
          for (std::size_t i = 0; i < armor.points.size(); ++i)
          {
            cv::line(canvas, armor.points[i],
                     armor.points[(i + 1U) % armor.points.size()], color, 2,
                     cv::LINE_AA);
          }
          cv::circle(canvas, armor.center, 4, color, -1, cv::LINE_AA);
        }

        const auto project = [&camera_matrix](const LibXR::Position<double>& point,
                                              cv::Point& uv) -> bool
        {
          const double z = point.z();
          if (!std::isfinite(z) || z <= 1e-6)
          {
            return false;
          }
          const double u = camera_matrix[0] * point.x() / z + camera_matrix[2];
          const double v = camera_matrix[4] * point.y() / z + camera_matrix[5];
          if (!std::isfinite(u) || !std::isfinite(v))
          {
            return false;
          }
          uv = cv::Point(static_cast<int>(std::lround(u)),
                         static_cast<int>(std::lround(v)));
          return true;
        };

        std::array<cv::Point, 4> armor_center_uv{};
        std::array<bool, 4> armor_center_valid{};
        for (int i = 0; i < tracker_overlay_count; ++i)
        {
          armor_center_valid[i] =
              ekf_msg.valid[i + 1] &&
              project(ekf_msg.armors_cam[i], armor_center_uv[i]);
        }

        for (int i = 0; i < tracker_overlay_count; ++i)
        {
          if (!tracker_overlay[i].valid)
          {
            continue;
          }
          std::array<cv::Point, 4> corners_uv{};
          bool corners_valid = true;
          for (int corner_index = 0; corner_index < 4; ++corner_index)
          {
            corners_valid =
                corners_valid &&
                project(tracker_overlay[i].corners_cam[corner_index],
                        corners_uv[corner_index]);
          }
          if (!corners_valid)
          {
            continue;
          }
          for (int corner_index = 0; corner_index < 4; ++corner_index)
          {
            cv::line(canvas, corners_uv[corner_index],
                     corners_uv[(corner_index + 1) % 4],
                     cv::Scalar(255, 160, 40), 2, cv::LINE_AA);
          }
        }

        if (tracker_overlay_count > 1)
        {
          for (int i = 0; i < tracker_overlay_count; ++i)
          {
            const int next = (i + 1) % tracker_overlay_count;
            if (armor_center_valid[i] && armor_center_valid[next])
            {
              cv::line(canvas, armor_center_uv[i], armor_center_uv[next],
                       cv::Scalar(40, 255, 180), 2, cv::LINE_AA);
            }
          }
        }

        for (int i = 0; i < tracker_overlay_count; ++i)
        {
          if (!armor_center_valid[i])
          {
            continue;
          }
          cv::circle(canvas, armor_center_uv[i], 5, cv::Scalar(255, 120, 40), -1,
                     cv::LINE_AA);
          cv::putText(canvas, "E" + std::to_string(i),
                      armor_center_uv[i] + cv::Point(6, 16),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 120, 40),
                      2, cv::LINE_AA);
        }

        cv::Point center_uv;
        if (ekf_msg.valid[0] && project(ekf_msg.center_cam, center_uv))
        {
          cv::circle(canvas, center_uv, 5, cv::Scalar(40, 255, 40), -1,
                     cv::LINE_AA);
          cv::drawMarker(canvas, center_uv, cv::Scalar(40, 255, 40),
                         cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
          cv::putText(canvas, "TC", center_uv + cv::Point(8, -8),
                      cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(40, 255, 40),
                      2, cv::LINE_AA);
        }

        const auto id_index = static_cast<std::size_t>(target_msg.id);
        const std::string id_name =
            id_index < ARMOR_NUMBER_NAMES.size()
                ? std::string(ARMOR_NUMBER_NAMES[id_index])
                : std::string("invalid");
        const std::string header =
            std::string("tracker ") + (target_msg.tracking ? "TRACK" : "NO_TARGET") +
            " id=" + id_name +
            " face=" + std::to_string(target_msg.tracked_face_index) +
            " det=" + std::to_string(detector_armors.size());
        cv::putText(canvas, header, cv::Point(12, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(40, 240, 40),
                    2, cv::LINE_AA);

        const std::string debug_line =
            "candidate count=" + std::to_string(candidate_debug_msg.count) +
            " selected=" + std::to_string(candidate_debug_msg.selected_index) +
            " matched=" + std::to_string(candidate_debug_msg.matched);
        cv::putText(canvas, debug_line, cv::Point(12, 56),
                    cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(230, 230, 230),
                    2, cv::LINE_AA);
      });
}

/**
 * @brief 将选面器内部调试快照复制到对外 CandidateDebugMsg。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::FillCandidateDebugFromSelection(
    const armor_tracker::FaceSelectionResult& selection,
    CandidateDebugMsg& candidate_debug)
{
  candidate_debug.count = selection.debug.count;
  candidate_debug.selected_index = selection.debug.selected_index;
  candidate_debug.detection_count = selection.debug.detection_count;
  candidate_debug.preferred_adjacent_face = selection.debug.preferred_adjacent_face;
  candidate_debug.has_same_number_candidate =
      selection.debug.has_same_number_candidate;
  candidate_debug.relaxed_same_face_distance =
      selection.debug.relaxed_same_face_distance;
  candidate_debug.relaxed_face_switch_distance =
      selection.debug.relaxed_face_switch_distance;
  candidate_debug.relaxed_face_switch_yaw_diff =
      selection.debug.relaxed_face_switch_yaw_diff;
  candidate_debug.best_same_face_score = selection.debug.best_same_face_score;
  candidate_debug.best_switch_face_score = selection.debug.best_switch_face_score;
  candidate_debug.same_face_matched = selection.debug.same_face_matched;
  candidate_debug.switch_face_matched = selection.debug.switch_face_matched;
  candidate_debug.switch_blocked_by_timeout =
      selection.debug.switch_blocked_by_timeout;
  candidate_debug.switch_allowed = selection.debug.switch_allowed;
  candidate_debug.detection_track_ids = selection.debug.detection_track_ids;
  candidate_debug.detection_track_confirmed =
      selection.debug.detection_track_confirmed;

  for (std::size_t item_index = 0; item_index < selection.debug.count; ++item_index)
  {
    const auto& src = selection.debug.items[item_index];
    auto& dst = candidate_debug.items[item_index];
    dst.armor_index = src.armor_index;
    dst.face_index =
        static_cast<uint8_t>(LocalFaceToCanonicalFace(src.face_index));
    dst.same_number = src.same_number;
    dst.image_track_id = src.image_track_id;
    dst.image_track_confirmed = src.image_track_confirmed;
    dst.same_persistent_track = src.same_persistent_track;
    dst.number = src.number;
    dst.type = src.type;
    dst.score = src.score;
    dst.position_diff = src.position_diff;
    dst.yaw_diff = src.yaw_diff;
    dst.view_bonus = src.view_bonus;
    dst.area_score = src.area_score;
    dst.frontality = src.frontality;
    dst.observation_quality_penalty = src.observation_quality_penalty;
    dst.center_x = src.center_x;
    dst.center_y = src.center_y;
    dst.predicted_yaw = src.predicted_yaw;
    dst.measured_yaw = src.measured_yaw;
  }
  candidate_debug.face_switch_cooldown_remaining =
      static_cast<float>(rt_.face_switch_cooldown_remaining);
}

/**
 * @brief 填充当前策略开关和阈值到调试消息。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::FillCandidateDebugPolicy(
    CandidateDebugMsg& candidate_debug, const Eigen::VectorXd& ekf_prediction,
    const armor_tracker::FaceSelectionPolicy& face_policy) const
{
  candidate_debug.face_switch_enabled = face_policy.face_switch_enabled ? 1 : 0;
  candidate_debug.relaxed_face_switch_enabled =
      face_policy.relaxed_face_switch_enabled ? 1 : 0;
  candidate_debug.odd_face_switch_enabled =
      face_policy.odd_face_switch_enabled ? 1 : 0;
  candidate_debug.view_priority_enabled =
      face_policy.view_priority_enabled ? 1 : 0;
  candidate_debug.directional_face_switch_enabled =
      face_policy.directional_face_switch_enabled ? 1 : 0;
  candidate_debug.tracked_face_track_id_valid =
      rt_.tracked_face_track_id_valid ? 1 : 0;
  candidate_debug.tracked_face_track_id =
      rt_.tracked_face_track_id_valid
          ? static_cast<int16_t>(rt_.tracked_face_track_id)
          : static_cast<int16_t>(-1);
  candidate_debug.tracked_armors_num = static_cast<uint8_t>(rt_.tracked_armors_num);
  candidate_debug.predicted_vyaw =
      static_cast<float>(ekf_prediction(ExtendedKalmanFilter::V_YAW));
  candidate_debug.max_match_distance =
      static_cast<float>(cfg_.match.max_match_distance);
  candidate_debug.max_match_yaw_diff =
      static_cast<float>(cfg_.match.max_match_yaw_diff);
  candidate_debug.face_switch_score_deadzone =
      static_cast<float>(face_policy.face_switch_score_deadzone);
  candidate_debug.face_switch_position_deadzone =
      static_cast<float>(face_policy.face_switch_position_deadzone);
  candidate_debug.face_switch_yaw_deadzone =
      static_cast<float>(face_policy.face_switch_yaw_deadzone);
  candidate_debug.face_switch_timeout_sec =
      static_cast<float>(face_policy.face_switch_timeout_sec);
  candidate_debug.face_switch_cooldown_remaining =
      static_cast<float>(rt_.face_switch_cooldown_remaining);
}

/**
 * @brief 在单装甲模式下使用当前最佳观测直接维护目标状态。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::UpdateSingleArmorMode(
    const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us)
{
  std::size_t selected_index = 0;
  int detection_track_id = -1;
  bool confirmed_track = false;
  float selected_center_diff = 0.0f;
  float selected_area_log = 0.0f;
  float selected_score = 0.0f;
  auto selected_armor = SelectSingleArmorObservation(
      armors_msg, image_timestamp_us, selected_index, detection_track_id,
      confirmed_track, selected_center_diff, selected_area_log, selected_score);

  const bool matched = selected_armor.has_value();
  if (matched)
  {
    const auto& armor = *selected_armor;
    rt_.tracked_armor = armor;
    rt_.tracked_id = armor.number;
    rt_.tracked_armors_num = static_cast<ArmorsNum>(1);
    rt_.tracked_face_index = 0;
    rt_.tracked_face_track_id_valid = confirmed_track && detection_track_id >= 0;
    rt_.tracked_face_track_id =
        detection_track_id >= 0 ? static_cast<uint16_t>(detection_track_id) : 0;
    rt_.face_track_id_valid.fill(false);
    rt_.face_track_id.fill(0);
    if (rt_.tracked_face_track_id_valid)
    {
      rt_.face_track_id_valid[0] = true;
      rt_.face_track_id[0] = rt_.tracked_face_track_id;
    }

    const double raw_measured_yaw =
        MatchYawAllowPiAmbiguityEnabled()
            ? armor_tracker::MeasuredArmorYawNearAllowPi(armor, rt_.last_yaw)
            : armor_tracker::MeasuredArmorYawNear(armor, rt_.last_yaw);
    double measured_yaw = raw_measured_yaw;
    if (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST)
    {
      const double yaw_gate = 0.55;
      const std::array<double, 3> yaw_candidates = {
          raw_measured_yaw,
          armor_tracker::UnwrapYawNear(raw_measured_yaw + M_PI, rt_.last_yaw),
          armor_tracker::UnwrapYawNear(raw_measured_yaw - M_PI, rt_.last_yaw)};
      double best_candidate = yaw_candidates[0];
      double best_delta =
          armor_tracker::AngularDiffAbs(best_candidate, rt_.last_yaw);
      for (double candidate : yaw_candidates)
      {
        const double candidate_delta =
            armor_tracker::AngularDiffAbs(candidate, rt_.last_yaw);
        if (candidate_delta < best_delta)
        {
          best_delta = candidate_delta;
          best_candidate = candidate;
        }
      }
      measured_yaw = best_candidate;
      const double yaw_delta = best_delta;
      if (yaw_delta > yaw_gate)
      {
        XR_LOG_DEBUG(
            "SingleArmor yaw hold: idx=%u track=%d yaw_prev=%.3f yaw_raw=%.3f yaw_meas=%.3f yaw_delta=%.3f gate=%.3f",
            static_cast<unsigned>(selected_index), detection_track_id,
            rt_.last_yaw, raw_measured_yaw, measured_yaw, yaw_delta, yaw_gate);
        measured_yaw = rt_.last_yaw;
      }
    }

    XR_LOG_DEBUG(
        "SingleArmor match: idx=%u track=%d confirmed=%d num=%d type=%d center=(%.1f,%.1f) pos=(%.3f,%.3f,%.3f) score=%.3f center_diff=%.1f area_log=%.3f yaw_prev=%.3f yaw_raw=%.3f yaw_meas=%.3f",
        static_cast<unsigned>(selected_index), detection_track_id,
        confirmed_track ? 1 : 0, static_cast<int>(armor.number),
        static_cast<int>(armor.type), static_cast<double>(armor.center.x),
        static_cast<double>(armor.center.y), armor.pose.translation.x(),
        armor.pose.translation.y(), armor.pose.translation.z(),
        static_cast<double>(selected_score),
        static_cast<double>(selected_center_diff),
        static_cast<double>(selected_area_log), rt_.last_yaw, raw_measured_yaw,
        measured_yaw);
    rt_.last_yaw = measured_yaw;

    ekf_.state(ExtendedKalmanFilter::X_CENTER) = armor.pose.translation.x();
    ekf_.state(ExtendedKalmanFilter::Y_CENTER) = armor.pose.translation.y();
    ekf_.state(ExtendedKalmanFilter::Z_ARMOR) = armor.pose.translation.z();
    ekf_.state(ExtendedKalmanFilter::YAW) = measured_yaw;
    ekf_.state(ExtendedKalmanFilter::ROBOT_R) = 0.0;
    ekf_.state(ExtendedKalmanFilter::DELTA_R) = 0.0;
    ekf_.state(ExtendedKalmanFilter::DELTA_Z) = 0.0;
    ekf_.measurement =
        Eigen::Vector4d(armor.pose.translation.x(), armor.pose.translation.y(),
                        armor.pose.translation.z(), measured_yaw);
    ekf_.ekf.SetState(ekf_.state);

    rt_.info_position_diff = 0.0;
    rt_.info_yaw_diff = 0.0;
    FillSingleArmorDebug(selected_index, detection_track_id, confirmed_track,
                         selected_score, selected_center_diff, selected_area_log);
  }
  else
  {
    XR_LOG_DEBUG("SingleArmor miss: state=%d tracked_track_valid=%d tracked_track=%u",
                 static_cast<int>(rt_.state),
                 rt_.tracked_face_track_id_valid ? 1 : 0,
                 static_cast<unsigned>(rt_.tracked_face_track_id));
    CandidateDebugMsg debug{};
    debug.tracked_armors_num = 1;
    debug.matched = 0;
    debug.tracked_face_track_id_valid =
        rt_.tracked_face_track_id_valid ? 1 : 0;
    debug.tracked_face_track_id =
        rt_.tracked_face_track_id_valid
            ? static_cast<int16_t>(rt_.tracked_face_track_id)
            : static_cast<int16_t>(-1);
    candidate_debug_msg_ = debug;
    rt_.info_position_diff = DBL_MAX;
    rt_.info_yaw_diff = DBL_MAX;
  }

  rt_.measurement_valid_current_frame = matched;
  AdvanceTrackerState(matched);
  WriteStateAuditRow(image_timestamp_us, ekf_.state, nullptr, matched);
}

/**
 * @brief 写入一帧 tracker 状态、候选评分和 EKF 几何审计记录。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::WriteStateAuditRow(
    uint64_t image_timestamp_us, const Eigen::VectorXd& ekf_prediction,
    const armor_tracker::FaceSelectionResult* selection, bool matched)
{
  if (state_audit_.path.empty())
  {
    return;
  }

  if (!state_audit_.file.is_open())
  {
    state_audit_.file.open(state_audit_.path, std::ios::out | std::ios::trunc);
    if (!state_audit_.file)
    {
      if (!state_audit_.open_failed)
      {
        XR_LOG_ERROR("ArmorTracker failed to open state audit: %s",
                     state_audit_.path.c_str());
        state_audit_.open_failed = true;
      }
      return;
    }

    state_audit_.file
        << "image_timestamp_us\tmatched\ttracked_face_index\ttracked_id\t"
        << "accepted_mode\thas_selected_candidate\tselected_face_index\t"
        << "selected_track_id\tselected_confirmed\tselected_score\t"
        << "selected_position_diff\tselected_yaw_diff\tselected_measured_yaw\t"
        << "best_same_face_index\tbest_same_track_id\tbest_same_confirmed\t"
        << "best_same_score\tbest_same_position_diff\tbest_same_yaw_diff\t"
        << "best_switch_face_index\tbest_switch_track_id\tbest_switch_confirmed\t"
        << "best_switch_score\tbest_switch_position_diff\tbest_switch_yaw_diff\t"
        << "same_face_matched\tswitch_face_matched\tswitch_allowed\t"
        << "switch_blocked_by_timeout\tswitch_blocked_by_id_mismatch\t"
        << "state\tlost_count\trecovery_count\t"
        << "pred_x\tpred_y\tpred_z\tpred_yaw\tpred_r1\tpred_r2\tpred_dz\t"
        << "pred_a0_x\tpred_a0_y\tpred_a0_z\tpred_a1_x\tpred_a1_y\tpred_a1_z\t"
        << "pred_a2_x\tpred_a2_y\tpred_a2_z\tpred_a3_x\tpred_a3_y\tpred_a3_z\t"
        << "post_x\tpost_y\tpost_z\tpost_yaw\tpost_r1\tpost_r2\tpost_dz\n";
  }

  const auto calc_radius_2 = [this](const Eigen::VectorXd& state)
  {
    const bool four_armors = rt_.tracked_armors_num == ArmorsNum::NORMAL_4;
    const double radius_1 = state(ExtendedKalmanFilter::ROBOT_R);
    return (!four_armors || SymmetricGeometryEnabled())
               ? radius_1
               : (radius_1 + state(ExtendedKalmanFilter::DELTA_R));
  };
  const auto calc_dz = [this](const Eigen::VectorXd& state)
  {
    return (rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
            !SymmetricGeometryEnabled())
               ? state(ExtendedKalmanFilter::DELTA_Z)
               : 0.0;
  };
  const auto write_pred_armors = [this, &ekf_prediction]()
  {
    for (int face_index = 0; face_index < 4; ++face_index)
    {
      const Eigen::Vector3d p = GetArmorPositionFromState(ekf_prediction, face_index);
      state_audit_.file << p.x() << '\t' << p.y() << '\t' << p.z() << '\t';
    }
  };
  const auto write_candidate =
      [this](const armor_tracker::FaceMatchCandidate& candidate, bool valid)
  {
    if (!valid)
    {
      state_audit_.file << -1 << '\t' << -1 << '\t' << 0 << '\t' << 0.0 << '\t'
                        << 0.0 << '\t' << 0.0;
      return;
    }
    state_audit_.file << LocalFaceToCanonicalFace(candidate.face_index) << '\t'
                      << candidate.image_track_id << '\t'
                      << (candidate.confirmed_image_track ? 1 : 0) << '\t'
                      << candidate.score << '\t' << candidate.position_diff
                      << '\t' << candidate.yaw_diff;
  };
  const armor_tracker::FaceMatchCandidate* selected_candidate =
      (selection != nullptr && selection->has_selected_candidate)
          ? &selection->selected_candidate
          : nullptr;

  state_audit_.file << image_timestamp_us << '\t' << (matched ? 1 : 0) << '\t'
                    << rt_.tracked_face_index << '\t'
                    << static_cast<int>(rt_.tracked_id) << '\t'
                    << (selection != nullptr
                            ? static_cast<int>(selection->accepted_mode)
                            : static_cast<int>(
                                  armor_tracker::FaceSelectionAcceptedMode::NONE))
                    << '\t' << (selected_candidate != nullptr ? 1 : 0) << '\t';
  if (selected_candidate != nullptr)
  {
    state_audit_.file << LocalFaceToCanonicalFace(selected_candidate->face_index)
                      << '\t' << selected_candidate->image_track_id << '\t'
                      << (selected_candidate->confirmed_image_track ? 1 : 0)
                      << '\t' << selected_candidate->score << '\t'
                      << selected_candidate->position_diff << '\t'
                      << selected_candidate->yaw_diff << '\t'
                      << selected_candidate->measured_yaw << '\t';
  }
  else
  {
    state_audit_.file << -1 << '\t' << -1 << '\t' << 0 << '\t' << 0.0 << '\t'
                      << 0.0 << '\t' << 0.0 << '\t' << 0.0 << '\t';
  }

  if (selection != nullptr)
  {
    write_candidate(selection->best_same_face_candidate,
                    selection->best_same_face_candidate.face_index >= 0);
    state_audit_.file << '\t';
    write_candidate(selection->best_switch_candidate,
                    selection->best_switch_candidate.face_index >= 0);
    state_audit_.file << '\t'
                      << (selection->matched_same_face ? 1 : 0) << '\t'
                      << (selection->matched_switch_face ? 1 : 0) << '\t'
                      << (selection->allow_face_switch ? 1 : 0) << '\t'
                      << (selection->switch_blocked_by_timeout ? 1 : 0) << '\t'
                      << (selection->switch_blocked_by_id_mismatch ? 1 : 0)
                      << '\t';
  }
  else
  {
    write_candidate(armor_tracker::FaceMatchCandidate{}, false);
    state_audit_.file << '\t';
    write_candidate(armor_tracker::FaceMatchCandidate{}, false);
    state_audit_.file << "\t0\t0\t0\t0\t0\t";
  }

  state_audit_.file << static_cast<int>(rt_.state) << '\t' << rt_.lost_count
                    << '\t' << static_cast<int>(rt_.recovery_count) << '\t';

  state_audit_.file << ekf_prediction(ExtendedKalmanFilter::X_CENTER) << '\t'
                    << ekf_prediction(ExtendedKalmanFilter::Y_CENTER) << '\t'
                    << ekf_prediction(ExtendedKalmanFilter::Z_ARMOR) << '\t'
                    << ekf_prediction(ExtendedKalmanFilter::YAW) << '\t'
                    << ekf_prediction(ExtendedKalmanFilter::ROBOT_R) << '\t'
                    << calc_radius_2(ekf_prediction) << '\t'
                    << calc_dz(ekf_prediction) << '\t';
  write_pred_armors();
  state_audit_.file
      << ekf_.state(ExtendedKalmanFilter::X_CENTER) << '\t'
      << ekf_.state(ExtendedKalmanFilter::Y_CENTER) << '\t'
      << ekf_.state(ExtendedKalmanFilter::Z_ARMOR) << '\t'
      << ekf_.state(ExtendedKalmanFilter::YAW) << '\t'
      << ekf_.state(ExtendedKalmanFilter::ROBOT_R) << '\t'
      << calc_radius_2(ekf_.state) << '\t' << calc_dz(ekf_.state) << '\n';
  state_audit_.file.flush();
}
