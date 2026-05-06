#pragma once

/**
 * @file ArmorTrackerFaceSelector.hpp
 * @brief 多装甲面候选评分、换面门控和最终选面策略。
 *
 * FaceSelector 是纯算法层：输入 tracker 运行时状态、detector 观测和若干预测回调，
 * 输出本帧最可信的装甲面候选与完整调试快照，不直接读写 topic 或 EKF 对象。
 */

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/imgproc.hpp>

#include "ArmorTrackerCommon.hpp"
#include "ArmorDetectorTypes.hpp"

namespace armor_tracker
{
/**
 * @brief 按候选面索引给换面候选增加先验惩罚。
 */
inline double FaceSwitchPenalty(int face_index)
{
  if (face_index == 0)
  {
    return 0.0;
  }
  return face_index == 2 ? 0.45 : 0.20;
}

/**
 * @brief 计算允许半径误差弱化后的三维位置残差。
 *
 * 四装甲目标的半径状态容易被单帧 PnP 尾部拉动；允许半径误差时会降低径向残差权重，
 * 让切向和高度误差在选面中占主导。
 */
inline double RadiusInvariantPositionDiff(const Eigen::Vector3d& predicted,
                                          const Eigen::Vector3d& measured,
                                          double predicted_yaw,
                                          bool allow_radius_error)
{
  const Eigen::Vector3d residual = predicted - measured;
  if (!allow_radius_error)
  {
    return residual.norm();
  }

  const Eigen::Vector2d radial_dir(std::cos(predicted_yaw),
                                   std::sin(predicted_yaw));
  const Eigen::Vector2d xy_residual(residual.x(), residual.y());
  const double radial_error = xy_residual.dot(radial_dir);
  const Eigen::Vector2d tangent_error = xy_residual - radial_error * radial_dir;
  constexpr double kRadialErrorWeight = 0.35;
  return std::sqrt(tangent_error.squaredNorm() + residual.z() * residual.z() +
                   std::pow(kRadialErrorWeight * radial_error, 2));
}

/**
 * @brief 当前帧选面策略开关与阈值集合。
 */
struct FaceSelectionPolicy
{
  bool single_armor_mode = false;             ///< 是否只跟踪单块装甲。
  bool id_assist_enabled = false;             ///< 是否使用图像 track id 辅助同面保持。
  bool face_switch_enabled = false;           ///< 是否允许跨面匹配。
  bool relaxed_face_switch_enabled = false;   ///< 是否允许放宽阈值换面。
  bool odd_face_switch_enabled = false;       ///< 是否允许切到奇数高低面。
  bool view_priority_enabled = false;         ///< 是否按视角和面积给候选加权。
  bool directional_face_switch_enabled = false;  ///< 是否按 yaw 速度限制换面方向。
  bool symmetric_geometry_enabled = false;       ///< 是否强制长短半径/高度对称。
  bool observation_quality_enabled = true;       ///< 是否启用观测质量评分。
  bool match_yaw_allow_pi_ambiguity = false;     ///< 是否允许 PnP yaw 折叠 pi。

  double max_match_distance = 0.15;           ///< 严格位置匹配阈值，单位 m。
  double max_match_yaw_diff = 1.0;            ///< 严格 yaw 匹配阈值，单位 rad。
  double single_armor_image_center_gate_px = 180.0;  ///< 单装甲图像中心门限。
  double single_armor_area_log_gate = 0.80;          ///< 单装甲面积比例对数门限。
  double face_switch_score_deadzone = 0.15;          ///< 换面分数优势死区。
  double face_switch_position_deadzone = 0.05;       ///< 换面位置优势死区。
  double face_switch_yaw_deadzone = 0.35;            ///< 换面 yaw 优势死区。
  double face_switch_timeout_sec = 0.0;              ///< 换面冷却时间，单位 s。
  double id_assist_same_face_center_gate_px = 85.0;  ///< ID 辅助同面中心门限。
  double id_assist_same_face_area_log_gate = 0.45;   ///< ID 辅助同面面积门限。
  double relaxed_same_face_image_gate_px = 90.0;     ///< 放宽同面图像中心门限。
  double relaxed_same_face_area_log_gate = 0.80;     ///< 放宽同面面积门限。
  double stable_max_reprojection_px = 1.8;           ///< 稳定观测最大重投影误差。
  double stable_min_area_px = 60.0;                  ///< 稳定观测最小图像面积。
  double stable_min_confidence = 0.0;                ///< 稳定观测最小置信度。
  double observation_quality_score_weight = 0.55;    ///< 观测质量惩罚权重。
  double confirmed_track_bonus = 0.24;               ///< confirmed track 奖励。
};

/**
 * @brief 选面器消费的当前 tracker 绑定状态。
 */
struct FaceSelectionTrackedState
{
  ArmorDetectorResult tracked_armor{};       ///< 上一帧绑定的装甲观测。
  ArmorNumber tracked_id = ArmorNumber::INVALID;  ///< 当前跟踪数字 ID。
  int tracked_armors_num = 1;                ///< 当前目标理论装甲面数量。
  bool tracked_face_track_id_valid = false;  ///< 当前面 track id 是否有效。
  uint16_t tracked_face_track_id = 0;        ///< 当前面图像 track id。
  double face_switch_cooldown_remaining = 0.0;  ///< 剩余换面冷却时间。
  double dz_abs_ref = 0.0;                   ///< 高低面高度差绝对值参考。
};

/**
 * @brief 单条 detection 在某个 face 假设下的候选评分。
 */
struct FaceMatchCandidate
{
  ArmorDetectorResult armor{};        ///< 原始装甲观测。
  std::size_t armor_index = 0;        ///< detection 在本帧列表中的索引。
  uint8_t debug_index = 24;           ///< 调试数组中的索引。
  int face_index = -1;                ///< 假设匹配的本地装甲面索引。
  bool same_number = false;           ///< 数字 ID 是否与当前跟踪一致。
  int image_track_id = -1;            ///< 图像域 track id。
  bool confirmed_image_track = false; ///< 图像 track 是否已确认。
  bool same_persistent_track = false; ///< 是否与当前绑定面同一个持续 track。
  double measured_yaw = 0.0;          ///< 测得装甲面 yaw。
  double position_diff = DBL_MAX;     ///< 位置残差。
  double yaw_diff = DBL_MAX;          ///< yaw 残差。
  double view_bonus = 0.0;            ///< 视角/面积奖励。
  double area_score = 0.0;            ///< 图像面积归一化分数。
  double frontality = 0.0;            ///< 装甲面朝向相机程度。
  double observation_quality_penalty = 0.0;  ///< PnP/面积/置信度质量惩罚。
  double image_center_diff = DBL_MAX;        ///< 图像中心连续性残差。
  double area_ratio_log = DBL_MAX;           ///< 面积比例对数残差。
  double score = DBL_MAX;                    ///< 综合候选分数，越小越好。
};

/**
 * @brief 单个候选的紧凑调试载荷。
 */
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
  float observation_quality_penalty{};
  float center_x{};
  float center_y{};
  float predicted_yaw{};
  float measured_yaw{};
};

/**
 * @brief 本帧选面过程的完整调试快照。
 */
struct FaceSelectionDebugSnapshot
{
  static constexpr uint8_t kMaxItems = 24;       ///< 最多记录的候选数量。
  static constexpr uint8_t kMaxDetections = 8;   ///< 最多记录的 detection 数量。

  uint8_t count = 0;                 ///< 已写入候选数量。
  uint8_t selected_index = 255;      ///< 被选中候选的 debug index。
  uint8_t detection_count = 0;       ///< 本帧 detection 记录数量。
  int8_t preferred_adjacent_face = -1;   ///< 按旋转方向偏好的相邻面。
  uint8_t has_same_number_candidate = 0; ///< 是否存在同数字候选。
  float relaxed_same_face_distance = 0.0f;      ///< 放宽同面位置阈值。
  float relaxed_face_switch_distance = 0.0f;    ///< 放宽换面位置阈值。
  float relaxed_face_switch_yaw_diff = 0.0f;    ///< 放宽换面 yaw 阈值。
  float best_same_face_score = -1.0f;           ///< 最佳同面分数。
  float best_switch_face_score = -1.0f;         ///< 最佳换面分数。
  uint8_t same_face_matched = 0;                ///< 是否同面匹配成功。
  uint8_t switch_face_matched = 0;              ///< 是否存在换面匹配。
  uint8_t switch_blocked_by_timeout = 0;        ///< 换面是否被冷却阻塞。
  uint8_t switch_allowed = 0;                   ///< 本帧是否允许换面。
  std::array<int16_t, kMaxDetections> detection_track_ids{};       ///< detection 对应 track id。
  std::array<uint8_t, kMaxDetections> detection_track_confirmed{}; ///< detection track 确认标记。
  std::array<FaceSelectionDebugItem, kMaxItems> items{};           ///< 候选调试项。
};

/**
 * @brief 最终接受候选的原因枚举。
 */
enum class FaceSelectionAcceptedMode : std::uint8_t
{
  NONE = 0,                  ///< 未接受候选。
  STRICT_SWITCH = 1,         ///< 严格阈值换面。
  RELAXED_SWITCH = 2,        ///< 放宽阈值换面。
  ID_REBIND_SWITCH = 3,      ///< ID 辅助重新绑定换面。
  ID_HANDOVER_SWITCH = 4,    ///< ID 辅助交接换面。
  STRICT_SAME_FACE = 5,      ///< 严格阈值保持同面。
  RELAXED_SAME_FACE = 6,     ///< 放宽阈值保持同面。
  ID_ASSISTED_SAME_FACE = 7, ///< ID 辅助保持同面。
};

/**
 * @brief 选面器一帧的完整输出。
 */
struct FaceSelectionResult
{
  FaceSelectionDebugSnapshot debug{};  ///< 可视化和日志使用的调试快照。

  FaceMatchCandidate best_candidate{};            ///< 全局最佳候选。
  FaceMatchCandidate best_same_face_candidate{};  ///< 最佳同面候选。
  FaceMatchCandidate best_switch_candidate{};     ///< 最佳换面候选。
  FaceMatchCandidate selected_candidate{};        ///< 最终接受候选。

  bool has_selected_candidate = false;            ///< 是否有最终候选。
  bool has_same_number_candidate = false;         ///< 是否存在同数字候选。
  bool observed_persistent_track_this_frame = false;  ///< 是否观测到当前持续 track。

  bool strict_same_face_match = false;       ///< 严格同面匹配是否成立。
  bool relaxed_same_face_match = false;      ///< 放宽同面匹配是否成立。
  bool id_assisted_same_face_match = false;  ///< ID 辅助同面匹配是否成立。
  bool id_assisted_same_face_hold = false;   ///< ID 强保持同面是否成立。
  bool matched_same_face = false;            ///< 任一同面条件是否成立。

  bool strict_face_switch_match = false;         ///< 严格换面匹配是否成立。
  bool relaxed_face_switch_match = false;        ///< 放宽换面匹配是否成立。
  bool id_assisted_face_rebind_match = false;    ///< ID 重绑定换面是否成立。
  bool id_assisted_face_handover_match = false;  ///< ID 交接换面是否成立。
  bool matched_switch_face = false;              ///< 任一换面条件是否成立。

  bool switch_blocked_by_timeout = false;     ///< 换面是否被冷却时间阻塞。
  bool switch_blocked_by_id_mismatch = false; ///< 换面是否被数字 ID 阻塞。
  bool allow_face_switch = false;             ///< 本帧最终是否允许换面。

  FaceSelectionAcceptedMode accepted_mode = FaceSelectionAcceptedMode::NONE; ///< 接受模式。
  double info_position_diff = DBL_MAX;  ///< 对外 info 的位置残差。
  double info_yaw_diff = DBL_MAX;       ///< 对外 info 的 yaw 残差。
};

/**
 * @brief 比较两个候选，判断 candidate 是否优于 best。
 */
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

/**
 * @brief 执行本帧多 detection、多 face 假设枚举和最终选面。
 *
 * @tparam TrackIdGetter detection index -> 图像 track id 的回调。
 * @tparam TrackConfirmedGetter detection index -> track confirmed 标记的回调。
 * @tparam PredictPositionGetter face index -> 预测装甲位置的回调。
 * @tparam PredictYawGetter face index -> 预测装甲 yaw 的回调。
 */
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
          policy.match_yaw_allow_pi_ambiguity
              ? MeasuredArmorYawNearAllowPi(armor, predicted_yaw)
              : MeasuredArmorYawNear(armor, predicted_yaw);
      const bool allow_radius_error =
          !policy.single_armor_mode && tracked.tracked_armors_num == 4;
      const double position_diff = RadiusInvariantPositionDiff(
          predicted_position, position_vec, predicted_yaw, allow_radius_error);
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
            "Tracker reject mismatched number: armor=%u num=%d tracked=%d face=%d has_same=%d persistent=%d confirmed=%d img_diff=%.1f area_log=%.3f",
            static_cast<unsigned>(armor_index), static_cast<int>(armor.number),
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
            "Tracker single-armor reject: armor=%u num=%d img_diff=%.1f area_log=%.3f",
            static_cast<unsigned>(armor_index), static_cast<int>(armor.number), image_center_diff,
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
      const double observation_quality_penalty =
          policy.observation_quality_enabled
              ? ArmorObservationQualityPenalty(
                    armor, policy.stable_max_reprojection_px,
                    policy.stable_min_area_px, policy.stable_min_confidence)
              : 0.0;
      const double stable_observation_bonus =
          policy.observation_quality_enabled &&
                  StableArmorObservation(armor, policy.stable_max_reprojection_px,
                                         policy.stable_min_area_px,
                                         policy.stable_min_confidence)
              ? 0.12
              : 0.0;
      const double confirmed_track_quality_bonus =
          policy.observation_quality_enabled && confirmed_image_track
              ? policy.confirmed_track_bonus
              : 0.0;
      const double score =
          position_score + 0.40 * yaw_score + FaceSwitchPenalty(face_index) +
          number_penalty - view_bonus + 0.35 * image_score +
          0.20 * area_ratio_score + dz_mismatch_penalty - persistent_track_bonus -
          confirmed_switch_bonus +
          policy.observation_quality_score_weight * observation_quality_penalty -
          stable_observation_bonus - confirmed_track_quality_bonus;

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
        item.observation_quality_penalty =
            static_cast<float>(observation_quality_penalty);
        item.center_x = armor.center.x;
        item.center_y = armor.center.y;
        item.predicted_yaw = static_cast<float>(predicted_yaw);
        item.measured_yaw = static_cast<float>(measured_yaw);
      }

      XR_LOG_DEBUG(
          "Tracker cand: armor=%u num=%d face=%d same=%d score=%.3f pos_diff=%.3f yaw_diff=%.3f img_diff=%.1f area_log=%.3f view_bonus=%.3f area=%.3f frontality=%.3f q_pen=%.3f reproj=%.3f confirmed=%d",
          static_cast<unsigned>(armor_index), static_cast<int>(armor.number), face_index,
          same_number ? 1 : 0, score, position_diff, current_yaw_diff,
          image_center_diff, area_ratio_log, view_bonus, area_score, frontality,
          observation_quality_penalty, armor.pnp_reprojection_error_px,
          confirmed_image_track ? 1 : 0);

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
      candidate.observation_quality_penalty = observation_quality_penalty;
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

  const double relaxed_same_face_distance =
      std::max(policy.max_match_distance * 1.25, 0.45);
  const double relaxed_face_switch_distance = policy.max_match_distance * 1.25;
  const double relaxed_face_switch_yaw_diff =
      std::max(policy.max_match_yaw_diff * 1.2, policy.max_match_yaw_diff + 0.1);
  const double relaxed_same_face_yaw_diff =
      std::max(policy.max_match_yaw_diff, std::min(relaxed_face_switch_yaw_diff, 0.80));
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
      result.best_same_face_candidate.yaw_diff < relaxed_same_face_yaw_diff &&
      relaxed_same_face_image_consistent;

  const double id_assisted_same_face_distance =
      std::max(relaxed_same_face_distance,
               std::min(policy.max_match_distance * 3.0, 0.45));
  const double id_assisted_same_face_yaw_diff =
      std::max(policy.max_match_yaw_diff, std::min(relaxed_face_switch_yaw_diff, 0.80));
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
