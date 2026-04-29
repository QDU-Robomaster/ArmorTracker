#pragma once

#include <algorithm>
#include <array>

#include "ArmorTrackerFaceSelector.hpp"

namespace armor_tracker
{
// 选面之后只维护 face 与 image-track 的绑定，不把这段杂糅回主类。
struct FaceBindingRuntime
{
  int tracked_armors_num = 4;
  bool tracked_face_track_id_valid = false;
  uint16_t tracked_face_track_id = 0;
  std::array<bool, 4> face_track_id_valid{};
  std::array<uint16_t, 4> face_track_id{};
};

inline void ApplySelectedFaceBinding(FaceBindingRuntime& runtime,
                                     const FaceMatchCandidate& selected_candidate,
                                     bool did_face_switch)
{
  if (did_face_switch)
  {
    const int face_count = std::max(1, std::min(4, runtime.tracked_armors_num));
    std::array<bool, 4> rotated_valid{};
    std::array<uint16_t, 4> rotated_ids{};
    for (int face_slot = 0; face_slot < face_count; ++face_slot)
    {
      const int old_slot = (face_slot + selected_candidate.face_index) % face_count;
      rotated_valid[face_slot] = runtime.face_track_id_valid[old_slot];
      rotated_ids[face_slot] = runtime.face_track_id[old_slot];
    }
    runtime.face_track_id_valid = rotated_valid;
    runtime.face_track_id = rotated_ids;
  }

  if (runtime.face_track_id_valid[0])
  {
    runtime.tracked_face_track_id_valid = true;
    runtime.tracked_face_track_id = runtime.face_track_id[0];
  }
  else
  {
    runtime.tracked_face_track_id_valid = false;
  }

  if (selected_candidate.image_track_id >= 0 &&
      selected_candidate.confirmed_image_track)
  {
    runtime.tracked_face_track_id_valid = true;
    runtime.tracked_face_track_id =
        static_cast<uint16_t>(selected_candidate.image_track_id);
    runtime.face_track_id_valid[0] = true;
    runtime.face_track_id[0] = runtime.tracked_face_track_id;
  }
}

struct FaceSelectionLogContext
{
  ArmorNumber tracked_id = ArmorNumber::INVALID;
  double face_switch_timeout_sec = 0.0;
  double face_switch_score_deadzone = 0.0;
  double face_switch_position_deadzone = 0.0;
  double face_switch_yaw_deadzone = 0.0;
  double face_switch_cooldown_remaining = 0.0;
};

inline void LogRejectedSelection(const FaceSelectionResult& selection,
                                 const FaceSelectionLogContext& context)
{
  const auto& best_candidate = selection.best_candidate;
  const auto& best_same_face_candidate = selection.best_same_face_candidate;
  const auto& best_switch_candidate = selection.best_switch_candidate;
  XR_LOG_DEBUG(
      "No matched armor found! same_number=%d best_face=%d score=%.3f pos_diff=%.3f yaw_diff=%.3f same_score=%.3f switch_score=%.3f cooldown=%.3f image_track=%d confirmed=%d persistent=%d img_diff=%.1f area_log=%.3f",
      selection.has_same_number_candidate ? 1 : 0, best_candidate.face_index,
      best_candidate.score, best_candidate.position_diff, best_candidate.yaw_diff,
      best_same_face_candidate.score, best_switch_candidate.score,
      context.face_switch_cooldown_remaining, best_candidate.image_track_id,
      best_candidate.confirmed_image_track ? 1 : 0,
      best_candidate.same_persistent_track ? 1 : 0,
      best_candidate.image_center_diff, best_candidate.area_ratio_log);
}

inline void LogAcceptedSelection(const FaceSelectionResult& selection,
                                 const FaceSelectionLogContext& context)
{
  const auto& selected_candidate = selection.selected_candidate;
  const auto& best_same_face_candidate = selection.best_same_face_candidate;
  const auto& best_switch_candidate = selection.best_switch_candidate;
  const bool did_face_switch = selected_candidate.face_index != 0;

  XR_LOG_DEBUG(
      "Tracker pick: armor=%zu num=%d face=%d same=%d score=%.3f pos_diff=%.3f yaw_diff=%.3f view_bonus=%.3f area=%.3f frontality=%.3f cooldown=%.3f",
      selected_candidate.armor_index,
      static_cast<int>(selected_candidate.armor.number),
      selected_candidate.face_index, selected_candidate.same_number ? 1 : 0,
      selected_candidate.score, selected_candidate.position_diff,
      selected_candidate.yaw_diff, selected_candidate.view_bonus,
      selected_candidate.area_score, selected_candidate.frontality,
      context.face_switch_cooldown_remaining);

  if (did_face_switch)
  {
    if (!selection.strict_face_switch_match &&
        !selection.relaxed_face_switch_match &&
        selection.id_assisted_face_rebind_match)
    {
      XR_LOG_DEBUG(
          "Tracker id-assisted face rebind: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f observed_persistent=%d",
          selected_candidate.face_index, selected_candidate.position_diff,
          selected_candidate.yaw_diff,
          static_cast<int>(selected_candidate.armor.number),
          context.face_switch_timeout_sec,
          selection.observed_persistent_track_this_frame ? 1 : 0);
      return;
    }
    if (!selection.strict_face_switch_match &&
        !selection.relaxed_face_switch_match &&
        selection.id_assisted_face_handover_match)
    {
      XR_LOG_DEBUG(
          "Tracker id-assisted face handover: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f same_pos=%.3f",
          selected_candidate.face_index, selected_candidate.position_diff,
          selected_candidate.yaw_diff,
          static_cast<int>(selected_candidate.armor.number),
          context.face_switch_timeout_sec,
          best_same_face_candidate.position_diff);
      return;
    }
    if (!selection.strict_face_switch_match &&
        selection.relaxed_face_switch_match)
    {
      XR_LOG_DEBUG(
          "Tracker relaxed face switch: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f",
          selected_candidate.face_index, selected_candidate.position_diff,
          selected_candidate.yaw_diff,
          static_cast<int>(selected_candidate.armor.number),
          context.face_switch_timeout_sec);
      return;
    }

    XR_LOG_DEBUG(
        "Tracker face switch: face=%d pos_diff=%.3f yaw_diff=%.3f number=%d cooldown=%.3f",
        selected_candidate.face_index, selected_candidate.position_diff,
        selected_candidate.yaw_diff,
        static_cast<int>(selected_candidate.armor.number),
        context.face_switch_timeout_sec);
    return;
  }

  if (!selection.strict_same_face_match && selection.relaxed_same_face_match)
  {
    XR_LOG_DEBUG("Tracker relaxed same-face match: pos_diff=%.3f yaw_diff=%.3f",
                 selected_candidate.position_diff, selected_candidate.yaw_diff);
    return;
  }
  if (selection.id_assisted_same_face_hold)
  {
    XR_LOG_DEBUG(
        "Tracker hold same-face by persistent image id: pos_diff=%.3f yaw_diff=%.3f img_diff=%.1f area_log=%.3f",
        selected_candidate.position_diff, selected_candidate.yaw_diff,
        selected_candidate.image_center_diff, selected_candidate.area_ratio_log);
    return;
  }
  if (selection.switch_blocked_by_timeout && selection.matched_switch_face)
  {
    XR_LOG_DEBUG(
        "Tracker hold same-face by timeout: cooldown=%.3f same_score=%.3f switch_score=%.3f",
        context.face_switch_cooldown_remaining, best_same_face_candidate.score,
        best_switch_candidate.score);
    return;
  }
  if (selection.switch_blocked_by_id_mismatch)
  {
    XR_LOG_DEBUG(
        "Tracker hold same-face by id mismatch: tracked_num=%d switch_num=%d same_score=%.3f switch_score=%.3f",
        static_cast<int>(context.tracked_id),
        static_cast<int>(best_switch_candidate.armor.number),
        best_same_face_candidate.score, best_switch_candidate.score);
    return;
  }
  if (!selection.allow_face_switch && selection.matched_switch_face &&
      selection.matched_same_face)
  {
    XR_LOG_DEBUG(
        "Tracker hold same-face by deadzone: same_score=%.3f switch_score=%.3f score_dz=%.3f pos_dz=%.3f yaw_dz=%.3f",
        best_same_face_candidate.score, best_switch_candidate.score,
        context.face_switch_score_deadzone,
        context.face_switch_position_deadzone,
        context.face_switch_yaw_deadzone);
  }
}
}  // namespace armor_tracker
