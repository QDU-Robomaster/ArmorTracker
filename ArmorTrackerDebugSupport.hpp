#pragma once

/**
 * @file ArmorTrackerDebugSupport.hpp
 * @brief ArmorTracker 调试 topic 的组包辅助实现。
 */

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
