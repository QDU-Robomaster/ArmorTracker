#pragma once

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/imgproc.hpp>

#include "ArmorTrackerCommon.hpp"
#include "armor.hpp"

namespace armor_tracker
{
inline double FaceSwitchPenalty(int face_index)
{
  if (face_index == 0)
  {
    return 0.0;
  }
  return face_index == 2 ? 0.45 : 0.20;
}

struct FaceSelectionPolicy
{
  // 这些是“本帧允许怎么选”的策略开关和阈值。
  // ArmorTracker 负责从配置/env 组装，FaceSelector 只消费，不自己读环境变量。
  bool single_armor_mode = false;
  bool id_assist_enabled = false;
  bool face_switch_enabled = false;
  bool relaxed_face_switch_enabled = false;
  bool odd_face_switch_enabled = false;
  bool view_priority_enabled = false;
  bool directional_face_switch_enabled = false;
  bool symmetric_geometry_enabled = false;

  double max_match_distance = 0.15;
  double max_match_yaw_diff = 1.0;
  double single_armor_image_center_gate_px = 180.0;
  double single_armor_area_log_gate = 0.80;
  double face_switch_score_deadzone = 0.15;
  double face_switch_position_deadzone = 0.05;
  double face_switch_yaw_deadzone = 0.35;
  double face_switch_timeout_sec = 0.08;
  double id_assist_same_face_center_gate_px = 85.0;
  double id_assist_same_face_area_log_gate = 0.45;
  double relaxed_same_face_image_gate_px = 90.0;
  double relaxed_same_face_area_log_gate = 0.80;
};

struct FaceSelectionTrackedState
{
  // 这是当前 tracker 对“正在跟谁”的最小认知，
  // FaceSelector 不接触 EKF 内部状态，只吃这些稳定输入。
  ArmorDetectorResult tracked_armor{};
  ArmorNumber tracked_id = ArmorNumber::INVALID;
  int tracked_armors_num = 1;
  bool tracked_face_track_id_valid = false;
  uint16_t tracked_face_track_id = 0;
  double face_switch_cooldown_remaining = 0.0;
  double dz_abs_ref = 0.0;
};

struct FaceMatchCandidate
{
  // 一条 detection 在某个 face_index 假设下的完整匹配评分。
  ArmorDetectorResult armor{};
  std::size_t armor_index = 0;
  uint8_t debug_index = 24;
  int face_index = -1;
  bool same_number = false;
  int image_track_id = -1;
  bool confirmed_image_track = false;
  bool same_persistent_track = false;
  double measured_yaw = 0.0;
  double position_diff = DBL_MAX;
  double yaw_diff = DBL_MAX;
  double view_bonus = 0.0;
  double area_score = 0.0;
  double frontality = 0.0;
  double image_center_diff = DBL_MAX;
  double area_ratio_log = DBL_MAX;
  double score = DBL_MAX;
};

struct FaceSelectionDebugItem
{
  uint8_t armor_index{};
  uint8_t face_index{};
  uint8_t same_number{};
  uint8_t reserved0{};
  int16_t image_track_id{-1};
  uint8_t image_track_confirmed{};
  uint8_t same_persistent_track{};
  ArmorNumber number{ArmorNumber::INVALID};
  ArmorType type{ArmorType::INVALID};
  uint8_t reserved1{};
  uint8_t reserved2{};
  float score{};
  float position_diff{};
  float yaw_diff{};
  float view_bonus{};
  float area_score{};
  float frontality{};
  float center_x{};
  float center_y{};
  float predicted_yaw{};
  float measured_yaw{};
};

struct FaceSelectionDebugSnapshot
{
  // 调试快照只服务可视化与日志，不反向驱动算法。
  static constexpr uint8_t kMaxItems = 24;
  static constexpr uint8_t kMaxDetections = 8;

  uint8_t count = 0;
  uint8_t selected_index = 255;
  uint8_t detection_count = 0;
  int8_t preferred_adjacent_face = -1;
  uint8_t has_same_number_candidate = 0;
  float relaxed_same_face_distance = 0.0f;
  float relaxed_face_switch_distance = 0.0f;
  float relaxed_face_switch_yaw_diff = 0.0f;
  float best_same_face_score = -1.0f;
  float best_switch_face_score = -1.0f;
  uint8_t same_face_matched = 0;
  uint8_t switch_face_matched = 0;
  uint8_t switch_blocked_by_timeout = 0;
  uint8_t switch_allowed = 0;
  std::array<int16_t, kMaxDetections> detection_track_ids{};
  std::array<uint8_t, kMaxDetections> detection_track_confirmed{};
  std::array<FaceSelectionDebugItem, kMaxItems> items{};
};

enum class FaceSelectionAcceptedMode : std::uint8_t
{
  NONE = 0,
  STRICT_SWITCH = 1,
  RELAXED_SWITCH = 2,
  ID_REBIND_SWITCH = 3,
  ID_HANDOVER_SWITCH = 4,
  STRICT_SAME_FACE = 5,
  RELAXED_SAME_FACE = 6,
  ID_ASSISTED_SAME_FACE = 7,
};

struct FaceSelectionResult
{
  // 选面器的输出分两层：
  // 1. 调试快照
  // 2. 最终是否选中、为什么选中
  FaceSelectionDebugSnapshot debug{};

  FaceMatchCandidate best_candidate{};
  FaceMatchCandidate best_same_face_candidate{};
  FaceMatchCandidate best_switch_candidate{};
  FaceMatchCandidate selected_candidate{};

  bool has_selected_candidate = false;
  bool has_same_number_candidate = false;
  bool observed_persistent_track_this_frame = false;

  bool strict_same_face_match = false;
  bool relaxed_same_face_match = false;
  bool id_assisted_same_face_match = false;
  bool id_assisted_same_face_hold = false;
  bool matched_same_face = false;

  bool strict_face_switch_match = false;
  bool relaxed_face_switch_match = false;
  bool id_assisted_face_rebind_match = false;
  bool id_assisted_face_handover_match = false;
  bool matched_switch_face = false;

  bool switch_blocked_by_timeout = false;
  bool switch_blocked_by_id_mismatch = false;
  bool allow_face_switch = false;

  FaceSelectionAcceptedMode accepted_mode = FaceSelectionAcceptedMode::NONE;
  double info_position_diff = DBL_MAX;
  double info_yaw_diff = DBL_MAX;
};

inline bool IsBetterMatchCandidate(const FaceMatchCandidate& candidate,
                                   const FaceMatchCandidate& best)
{
  if (candidate.score < best.score - 1e-6)
  {
    return true;
  }
  if (std::abs(candidate.score - best.score) < 0.10 &&
      candidate.position_diff < best.position_diff - 1e-6)
  {
    return true;
  }
  if (std::abs(candidate.position_diff - best.position_diff) < 0.02 &&
      candidate.yaw_diff < best.yaw_diff)
  {
    return true;
  }
  return false;
}

template <typename TrackIdGetter, typename TrackConfirmedGetter,
          typename PredictPositionGetter, typename PredictYawGetter>
FaceSelectionResult SelectFaceMatch(
    const ArmorDetectorResults& armors_msg,
    const FaceSelectionTrackedState& tracked,
    const FaceSelectionPolicy& policy, const Eigen::Vector3d& camera_world,
    double predicted_vyaw, TrackIdGetter&& get_detection_track_id,
    TrackConfirmedGetter&& is_detection_track_confirmed,
    PredictPositionGetter&& get_predicted_position,
    PredictYawGetter&& get_predicted_yaw)
{
  FaceSelectionResult result;
  result.debug.detection_track_ids.fill(-1);
  result.debug.detection_track_confirmed.fill(0);

  // face_count 表示“这一帧允许枚举多少个理论装甲面”。
  const int armor_count =
      policy.single_armor_mode
          ? 1
          : (policy.face_switch_enabled ? std::max(1, tracked.tracked_armors_num) : 1);

  int preferred_adjacent_face = -1;
  if (policy.directional_face_switch_enabled && tracked.tracked_armors_num == 4)
  {
    if (predicted_vyaw > 0.05)
    {
      preferred_adjacent_face = 1;
    }
    else if (predicted_vyaw < -0.05)
    {
      preferred_adjacent_face = 3;
    }
  }
  result.debug.preferred_adjacent_face =
      static_cast<int8_t>(preferred_adjacent_face);
  result.debug.detection_count = static_cast<uint8_t>(std::min<std::size_t>(
      armors_msg.size(), FaceSelectionDebugSnapshot::kMaxDetections));
  for (std::size_t armor_index = 0; armor_index < result.debug.detection_count;
       ++armor_index)
  {
    const int detection_track_id = get_detection_track_id(armor_index);
    result.debug.detection_track_ids[armor_index] =
        static_cast<int16_t>(detection_track_id);
    result.debug.detection_track_confirmed[armor_index] =
        is_detection_track_confirmed(armor_index) ? 1 : 0;
  }

  if (tracked.tracked_id != ArmorNumber::INVALID)
  {
    for (const auto& armor : armors_msg)
    {
      if (tracked.tracked_armor.type != ArmorType::INVALID &&
          armor.type != tracked.tracked_armor.type)
      {
        continue;
      }
      if (armor.number == tracked.tracked_id)
      {
        result.has_same_number_candidate = true;
        break;
      }
    }
  }
  result.debug.has_same_number_candidate =
      result.has_same_number_candidate ? 1 : 0;

  const double tracked_image_area = ArmorImageArea(tracked.tracked_armor);
  const cv::Point2f tracked_image_center = tracked.tracked_armor.center;

  // 第一阶段：枚举 detection × face 假设，得到所有候选及调试项。
  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    if (tracked.tracked_armor.type != ArmorType::INVALID &&
        armor.type != tracked.tracked_armor.type)
    {
      continue;
    }

    const auto p = armor.pose.translation;
    const Eigen::Vector3d position_vec(p.x(), p.y(), p.z());
    const double image_area = ArmorImageArea(armor);
    const double area_score = std::min(image_area / 2500.0, 1.0);
    const Eigen::Vector3d armor_front =
        armor.pose.rotation.ToRotationMatrix() * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d armor_to_camera = camera_world - position_vec;
    double frontality = 0.0;
    if (armor_to_camera.norm() > 1e-6)
    {
      frontality =
          std::max(0.0, armor_front.normalized().dot(armor_to_camera.normalized()));
    }
    const double view_bonus =
        policy.view_priority_enabled ? (0.35 * area_score + 0.35 * frontality) : 0.0;
    const double image_center_diff =
        std::hypot(static_cast<double>(armor.center.x - tracked_image_center.x),
                   static_cast<double>(armor.center.y - tracked_image_center.y));
    const double area_ratio_log =
        std::abs(std::log(std::max(image_area, 1.0) / tracked_image_area));
    const int image_track_id = get_detection_track_id(armor_index);
    const bool confirmed_image_track = is_detection_track_confirmed(armor_index);
    const bool same_persistent_track =
        policy.id_assist_enabled && tracked.tracked_face_track_id_valid &&
        image_track_id >= 0 &&
        static_cast<uint16_t>(image_track_id) == tracked.tracked_face_track_id;
    result.observed_persistent_track_this_frame =
        result.observed_persistent_track_this_frame ||
        (confirmed_image_track && same_persistent_track);

    for (int face_index = 0; face_index < armor_count; ++face_index)
    {
      if (policy.directional_face_switch_enabled && tracked.tracked_armors_num == 4 &&
          face_index > 0)
      {
        if (face_index == 2 || face_index != preferred_adjacent_face)
        {
          continue;
        }
      }
      if (!policy.odd_face_switch_enabled && face_index > 0 &&
          (face_index % 2 == 1))
      {
        continue;
      }
      if (policy.id_assist_enabled && tracked.tracked_face_track_id_valid)
      {
        if (face_index == 0 && !same_persistent_track)
        {
          continue;
        }
        if (face_index > 0 && same_persistent_track)
        {
          continue;
        }
        if (face_index > 0 && !confirmed_image_track)
        {
          continue;
        }
      }

      const Eigen::Vector3d predicted_position = get_predicted_position(face_index);
      const double predicted_yaw = get_predicted_yaw(face_index);
      const double measured_yaw =
          OrientationToYawNear(armor, predicted_yaw);
      const double position_diff = (predicted_position - position_vec).norm();
      const double current_yaw_diff =
          AngularDiffAbs(measured_yaw, predicted_yaw);
      LogImpossibleYawDiff("match", armor_index, face_index, measured_yaw,
                           predicted_yaw, current_yaw_diff);

      const bool same_number =
          tracked.tracked_id == ArmorNumber::INVALID || armor.number == tracked.tracked_id;
      const bool allow_number_mismatch_same_face =
          tracked.tracked_id != ArmorNumber::INVALID &&
          !result.has_same_number_candidate && face_index == 0 &&
          same_persistent_track &&
          image_center_diff < policy.id_assist_same_face_center_gate_px &&
          area_ratio_log < policy.id_assist_same_face_area_log_gate;
      if (!same_number && tracked.tracked_id != ArmorNumber::INVALID &&
          !allow_number_mismatch_same_face)
      {
        XR_LOG_DEBUG(
            "Tracker reject mismatched number: armor=%zu num=%d tracked=%d face=%d has_same=%d persistent=%d confirmed=%d img_diff=%.1f area_log=%.3f",
            armor_index, static_cast<int>(armor.number),
            static_cast<int>(tracked.tracked_id), face_index,
            result.has_same_number_candidate ? 1 : 0,
            same_persistent_track ? 1 : 0, confirmed_image_track ? 1 : 0,
            image_center_diff, area_ratio_log);
        continue;
      }
      if (policy.single_armor_mode && tracked.tracked_id != ArmorNumber::INVALID &&
          (image_center_diff > policy.single_armor_image_center_gate_px ||
           area_ratio_log > policy.single_armor_area_log_gate))
      {
        XR_LOG_DEBUG(
            "Tracker single-armor reject: armor=%zu num=%d img_diff=%.1f area_log=%.3f",
            armor_index, static_cast<int>(armor.number), image_center_diff,
            area_ratio_log);
        continue;
      }
      double dz_mismatch_penalty = 0.0;
      if (!policy.symmetric_geometry_enabled && tracked.tracked_armors_num == 4 &&
          face_index % 2 == 1 && tracked.dz_abs_ref > 0.02)
      {
        const double measured_dz_abs =
            std::abs(get_predicted_position(0).z() - position_vec.z());
        const double dz_error =
            std::abs(measured_dz_abs - tracked.dz_abs_ref);
        dz_mismatch_penalty = std::min(dz_error / 0.05, 0.35);
      }

      const double position_score =
          position_diff / std::max(policy.max_match_distance, 1e-6);
      const double yaw_score =
          current_yaw_diff / std::max(policy.max_match_yaw_diff, 1e-6);
      const double image_score =
          policy.single_armor_mode
              ? image_center_diff /
                    std::max(policy.single_armor_image_center_gate_px, 1.0)
              : 0.0;
      const double area_ratio_score =
          policy.single_armor_mode
              ? area_ratio_log / std::max(policy.single_armor_area_log_gate, 1e-6)
              : 0.0;
      const double number_penalty = same_number ? 0.0 : 1.5;
      const double persistent_track_bonus =
          (same_persistent_track && face_index == 0) ? 0.25 : 0.0;
      const double confirmed_switch_bonus =
          (policy.id_assist_enabled && face_index > 0 && confirmed_image_track) ? 0.16
                                                                                : 0.0;
      const double score =
          position_score + 0.40 * yaw_score + FaceSwitchPenalty(face_index) +
          number_penalty - view_bonus + 0.35 * image_score +
          0.20 * area_ratio_score + dz_mismatch_penalty - persistent_track_bonus -
          confirmed_switch_bonus;

      uint8_t debug_index = FaceSelectionDebugSnapshot::kMaxItems;
      if (result.debug.count < FaceSelectionDebugSnapshot::kMaxItems)
      {
        debug_index = result.debug.count;
        auto& item = result.debug.items[result.debug.count++];
        item.armor_index =
            static_cast<uint8_t>(std::min<std::size_t>(armor_index, 255));
        item.face_index = static_cast<uint8_t>(face_index);
        item.same_number = same_number ? 1 : 0;
        item.image_track_id = static_cast<int16_t>(image_track_id);
        item.image_track_confirmed = confirmed_image_track ? 1 : 0;
        item.same_persistent_track = same_persistent_track ? 1 : 0;
        item.number = armor.number;
        item.type = armor.type;
        item.score = static_cast<float>(score);
        item.position_diff = static_cast<float>(position_diff);
        item.yaw_diff = static_cast<float>(current_yaw_diff);
        item.view_bonus = static_cast<float>(view_bonus);
        item.area_score = static_cast<float>(area_score);
        item.frontality = static_cast<float>(frontality);
        item.center_x = armor.center.x;
        item.center_y = armor.center.y;
        item.predicted_yaw = static_cast<float>(predicted_yaw);
        item.measured_yaw = static_cast<float>(measured_yaw);
      }

      XR_LOG_DEBUG(
          "Tracker cand: armor=%zu num=%d face=%d same=%d score=%.3f pos_diff=%.3f yaw_diff=%.3f img_diff=%.1f area_log=%.3f view_bonus=%.3f area=%.3f frontality=%.3f",
          armor_index, static_cast<int>(armor.number), face_index,
          same_number ? 1 : 0, score, position_diff, current_yaw_diff,
          image_center_diff, area_ratio_log, view_bonus, area_score, frontality);

      FaceMatchCandidate candidate{};
      candidate.armor = armor;
      candidate.armor_index = armor_index;
      candidate.debug_index = debug_index;
      candidate.face_index = face_index;
      candidate.same_number = same_number;
      candidate.image_track_id = image_track_id;
      candidate.confirmed_image_track = confirmed_image_track;
      candidate.same_persistent_track = same_persistent_track;
      candidate.measured_yaw = measured_yaw;
      candidate.position_diff = position_diff;
      candidate.yaw_diff = current_yaw_diff;
      candidate.view_bonus = view_bonus;
      candidate.area_score = area_score;
      candidate.frontality = frontality;
      candidate.image_center_diff = image_center_diff;
      candidate.area_ratio_log = area_ratio_log;
      candidate.score = score;

      if (IsBetterMatchCandidate(candidate, result.best_candidate))
      {
        result.best_candidate = candidate;
      }
      if (face_index == 0)
      {
        if (IsBetterMatchCandidate(candidate, result.best_same_face_candidate))
        {
          result.best_same_face_candidate = candidate;
        }
      }
      else if (IsBetterMatchCandidate(candidate, result.best_switch_candidate))
      {
        result.best_switch_candidate = candidate;
      }
    }
  }

  const double relaxed_same_face_distance = policy.max_match_distance * 1.25;
  const double relaxed_face_switch_distance = policy.max_match_distance * 1.25;
  const double relaxed_face_switch_yaw_diff =
      std::max(policy.max_match_yaw_diff * 1.2, policy.max_match_yaw_diff + 0.1);
  const double face_switch_position_tie_margin = 0.01;

  // 第二阶段：在“同面保持”和“切面”之间做最终决策。
  result.debug.relaxed_same_face_distance =
      static_cast<float>(relaxed_same_face_distance);
  result.debug.relaxed_face_switch_distance =
      static_cast<float>(relaxed_face_switch_distance);
  result.debug.relaxed_face_switch_yaw_diff =
      static_cast<float>(relaxed_face_switch_yaw_diff);
  result.debug.best_same_face_score =
      result.best_same_face_candidate.face_index == 0
          ? static_cast<float>(result.best_same_face_candidate.score)
          : -1.0f;
  result.debug.best_switch_face_score =
      result.best_switch_candidate.face_index > 0
          ? static_cast<float>(result.best_switch_candidate.score)
          : -1.0f;

  result.strict_same_face_match =
      result.best_same_face_candidate.face_index == 0 &&
      result.best_same_face_candidate.position_diff < policy.max_match_distance &&
      result.best_same_face_candidate.yaw_diff < policy.max_match_yaw_diff;

  const bool relaxed_same_face_image_consistent =
      result.best_same_face_candidate.face_index == 0 &&
      (result.best_same_face_candidate.same_persistent_track ||
       (result.best_same_face_candidate.image_center_diff <
            policy.relaxed_same_face_image_gate_px &&
        result.best_same_face_candidate.area_ratio_log <
            policy.relaxed_same_face_area_log_gate));
  result.relaxed_same_face_match =
      result.best_same_face_candidate.face_index == 0 &&
      result.best_same_face_candidate.position_diff < relaxed_same_face_distance &&
      result.best_same_face_candidate.yaw_diff < policy.max_match_yaw_diff &&
      relaxed_same_face_image_consistent;

  const double id_assisted_same_face_distance =
      std::min(relaxed_same_face_distance, policy.max_match_distance * 1.10);
  const double id_assisted_same_face_yaw_diff =
      std::min(relaxed_face_switch_yaw_diff, policy.max_match_yaw_diff * 0.75);
  result.id_assisted_same_face_match =
      policy.id_assist_enabled && result.best_same_face_candidate.face_index == 0 &&
      result.best_same_face_candidate.same_persistent_track &&
      result.best_same_face_candidate.confirmed_image_track &&
      result.best_same_face_candidate.image_center_diff <
          policy.id_assist_same_face_center_gate_px &&
      result.best_same_face_candidate.area_ratio_log <
          policy.id_assist_same_face_area_log_gate &&
      result.best_same_face_candidate.position_diff <
          id_assisted_same_face_distance &&
      result.best_same_face_candidate.yaw_diff < id_assisted_same_face_yaw_diff;

  const double id_assisted_same_face_hold_distance =
      std::min(policy.max_match_distance * 0.45, 0.20);
  const double id_assisted_same_face_hold_yaw_diff =
      std::min(id_assisted_same_face_yaw_diff,
               policy.face_switch_yaw_deadzone + 0.25);
  result.id_assisted_same_face_hold =
      policy.id_assist_enabled && result.best_same_face_candidate.face_index == 0 &&
      result.best_same_face_candidate.same_persistent_track &&
      result.best_same_face_candidate.confirmed_image_track &&
      result.best_same_face_candidate.image_center_diff < 24.0 &&
      result.best_same_face_candidate.area_ratio_log < 0.18 &&
      result.best_same_face_candidate.position_diff <
          id_assisted_same_face_hold_distance &&
      result.best_same_face_candidate.yaw_diff <
          id_assisted_same_face_hold_yaw_diff;
  result.matched_same_face =
      result.strict_same_face_match || result.relaxed_same_face_match ||
      result.id_assisted_same_face_match || result.id_assisted_same_face_hold;

  result.strict_face_switch_match =
      result.best_switch_candidate.face_index > 0 &&
      result.best_switch_candidate.position_diff < policy.max_match_distance &&
      result.best_switch_candidate.yaw_diff < policy.max_match_yaw_diff;
  result.relaxed_face_switch_match =
      policy.relaxed_face_switch_enabled &&
      result.best_switch_candidate.face_index > 0 &&
      result.best_switch_candidate.position_diff < relaxed_face_switch_distance &&
      result.best_switch_candidate.yaw_diff < relaxed_face_switch_yaw_diff;

  // 图像 track id 只允许增强同面保持，不能单独触发跨面切换。
  // 两块同编号装甲板在图像上接近或交叉时，ID rebind 会把车体 yaw 拉到错误面。
  result.id_assisted_face_rebind_match = false;
  result.id_assisted_face_handover_match = false;
  result.matched_switch_face =
      result.strict_face_switch_match || result.relaxed_face_switch_match ||
      result.id_assisted_face_rebind_match ||
      result.id_assisted_face_handover_match;

  result.switch_blocked_by_id_mismatch =
      result.matched_switch_face && tracked.tracked_id != ArmorNumber::INVALID &&
      !result.best_switch_candidate.same_number;
  const bool switch_has_clear_score_advantage =
      result.matched_switch_face && result.best_same_face_candidate.face_index == 0 &&
      result.best_switch_candidate.score + policy.face_switch_score_deadzone <
          result.best_same_face_candidate.score;
  const bool switch_has_clear_position_advantage =
      result.matched_switch_face && result.best_same_face_candidate.face_index == 0 &&
      result.best_switch_candidate.position_diff +
              policy.face_switch_position_deadzone <
          result.best_same_face_candidate.position_diff &&
      result.best_switch_candidate.yaw_diff <
          result.best_same_face_candidate.yaw_diff + policy.face_switch_yaw_deadzone;
  const bool switch_has_clear_yaw_advantage =
      result.matched_switch_face && result.best_same_face_candidate.face_index == 0 &&
      result.best_switch_candidate.position_diff <
          result.best_same_face_candidate.position_diff +
              face_switch_position_tie_margin &&
      result.best_switch_candidate.yaw_diff + policy.face_switch_yaw_deadzone <
          result.best_same_face_candidate.yaw_diff;

  // 切面冷却是对“连续改写车体面身份”的保护，不应只在同面候选存在时生效。
  // 否则换面后的下一帧如果同面暂时匹配不上，selector 会继续连跳到另一面。
  result.switch_blocked_by_timeout =
      tracked.face_switch_cooldown_remaining > 1e-6 && result.matched_switch_face;
  result.allow_face_switch =
      result.matched_switch_face && !result.switch_blocked_by_timeout &&
      !result.switch_blocked_by_id_mismatch &&
      !result.id_assisted_same_face_hold &&
      (!result.matched_same_face || switch_has_clear_score_advantage ||
       switch_has_clear_position_advantage || switch_has_clear_yaw_advantage);

  result.debug.same_face_matched = result.matched_same_face ? 1 : 0;
  result.debug.switch_face_matched = result.matched_switch_face ? 1 : 0;
  result.debug.switch_blocked_by_timeout =
      result.switch_blocked_by_timeout ? 1 : 0;
  result.debug.switch_allowed = result.allow_face_switch ? 1 : 0;

  const FaceMatchCandidate* debug_candidate = nullptr;
  if (result.allow_face_switch)
  {
    result.selected_candidate = result.best_switch_candidate;
    result.has_selected_candidate = true;
    result.accepted_mode =
        result.strict_face_switch_match
            ? FaceSelectionAcceptedMode::STRICT_SWITCH
            : (result.id_assisted_face_rebind_match
                   ? FaceSelectionAcceptedMode::ID_REBIND_SWITCH
                   : (result.id_assisted_face_handover_match
                          ? FaceSelectionAcceptedMode::ID_HANDOVER_SWITCH
                          : FaceSelectionAcceptedMode::RELAXED_SWITCH));
  }
  else if (result.matched_same_face)
  {
    result.selected_candidate = result.best_same_face_candidate;
    result.has_selected_candidate = true;
    result.accepted_mode =
        result.strict_same_face_match
            ? FaceSelectionAcceptedMode::STRICT_SAME_FACE
            : (result.id_assisted_same_face_match
                   ? FaceSelectionAcceptedMode::ID_ASSISTED_SAME_FACE
                   : FaceSelectionAcceptedMode::RELAXED_SAME_FACE);
  }

  if (result.has_selected_candidate)
  {
    debug_candidate = &result.selected_candidate;
  }
  else if (result.best_candidate.face_index >= 0)
  {
    debug_candidate = &result.best_candidate;
  }
  if (debug_candidate != nullptr)
  {
    result.debug.selected_index = debug_candidate->debug_index;
    result.info_position_diff = debug_candidate->position_diff;
    result.info_yaw_diff = debug_candidate->yaw_diff;
  }

  return result;
}
}  // namespace armor_tracker
