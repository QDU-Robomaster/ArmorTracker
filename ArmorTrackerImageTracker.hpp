#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "armor.hpp"

namespace armor_tracker
{
// 图像域短时身份跟踪参数。只负责“这是不是同一块装甲板”，不参与整车几何估计。
struct ImageTrackConfig
{
  std::uint32_t appear_hits{2};
  double appear_timeout_sec{0.01};
  std::uint32_t tentative_misses{2};
  double tentative_timeout_sec{0.03};
  std::uint32_t disappear_misses{3};
  double disappear_timeout_sec{0.06};
};

struct ImageTrack
{
  bool active = false;
  bool confirmed = false;
  uint16_t track_id = 0;
  ArmorColor color = ArmorColor::UNKNOWN;
  ArmorNumber number = ArmorNumber::INVALID;
  ArmorType type = ArmorType::INVALID;
  float confidence = 0.0f;
  cv::Point2f image_center{};
  cv::Point2f image_velocity{};
  double area = 0.0;
  double area_rate = 0.0;
  uint64_t first_timestamp_us = 0;
  uint64_t last_timestamp_us = 0;
  uint64_t last_seen_timestamp_us = 0;
  uint32_t age = 0;
  uint32_t hit_count = 0;
  uint32_t miss_count = 0;
  bool matched_this_frame = false;
  uint8_t matched_armor_index = 255;
};

// 纯图像层的装甲板身份管理器。
// 输入 detector 结果，输出每个 detection 对应的稳定 track id 及 confirmed 状态。
class ImageTrackManager
{
 public:
  static constexpr std::size_t kMaxTracks = 8;

  void Reset()
  {
    tracks_.fill(ImageTrack{});
    detection_track_ids_.clear();
    detection_track_confirmed_.clear();
    next_track_id_ = 0;
  }

  void Update(const ArmorDetectorResults& armors, uint64_t image_timestamp_us,
              const ImageTrackConfig& cfg)
  {
    detection_track_ids_.assign(armors.size(), -1);
    detection_track_confirmed_.assign(armors.size(), 0);

    auto timeout_satisfied = [](uint64_t now_us, uint64_t since_us, double timeout_sec)
    {
      if (timeout_sec <= 1e-9 || now_us == 0 || since_us == 0)
      {
        return true;
      }
      return ImageTrackManager::TimestampDeltaSeconds(now_us, since_us) + 1e-9 >=
             timeout_sec;
    };

    auto compatible_track_pair = [](const ImageTrack& lhs, const ImageTrack& rhs)
    {
      if (lhs.type != ArmorType::INVALID && rhs.type != ArmorType::INVALID &&
          lhs.type != rhs.type)
      {
        return false;
      }
      if (lhs.color != ArmorColor::UNKNOWN && rhs.color != ArmorColor::UNKNOWN &&
          lhs.color != rhs.color)
      {
        return false;
      }
      if (lhs.number != ArmorNumber::INVALID && rhs.number != ArmorNumber::INVALID &&
          lhs.number != rhs.number)
      {
        return false;
      }
      return true;
    };

    auto reset_track = [](ImageTrack& track)
    {
      track = ImageTrack{};
    };

    auto assign_track = [&](ImageTrack& track, const ArmorDetectorResult& armor,
                            uint8_t armor_index, double score)
    {
      const double dt_raw =
          ImageTrackManager::TimestampDeltaSeconds(image_timestamp_us,
                                                  track.last_timestamp_us);
      const double dt = dt_raw > 1e-4 ? dt_raw : 1.0 / 100.0;
      const double measured_area = std::max(1.0, ImageTrackManager::ArmorArea(armor));
      if (track.age > 0)
      {
        const cv::Point2f measured_image_velocity(
            static_cast<float>((armor.center.x - track.image_center.x) / dt),
            static_cast<float>((armor.center.y - track.image_center.y) / dt));
        track.image_velocity.x =
            0.65f * track.image_velocity.x + 0.35f * measured_image_velocity.x;
        track.image_velocity.y =
            0.65f * track.image_velocity.y + 0.35f * measured_image_velocity.y;
        const double measured_area_rate = (measured_area - track.area) / dt;
        track.area_rate = 0.65 * track.area_rate + 0.35 * measured_area_rate;
      }
      else
      {
        track.image_velocity = cv::Point2f(0.0f, 0.0f);
        track.area_rate = 0.0;
      }

      if (track.first_timestamp_us == 0)
      {
        track.first_timestamp_us = image_timestamp_us;
      }

      track.image_center = armor.center;
      track.area = measured_area;
      track.confidence = armor.confidence;
      if (armor.color != ArmorColor::UNKNOWN || track.color == ArmorColor::UNKNOWN)
      {
        track.color = armor.color;
      }
      if (armor.number != ArmorNumber::INVALID || track.number == ArmorNumber::INVALID)
      {
        track.number = armor.number;
      }
      if (armor.type != ArmorType::INVALID || track.type == ArmorType::INVALID)
      {
        track.type = armor.type;
      }
      track.last_timestamp_us = image_timestamp_us;
      track.last_seen_timestamp_us = image_timestamp_us;
      track.age++;
      track.hit_count++;
      track.miss_count = 0;
      if (!track.confirmed && track.hit_count >= cfg.appear_hits &&
          timeout_satisfied(image_timestamp_us, track.first_timestamp_us,
                            cfg.appear_timeout_sec))
      {
        track.confirmed = true;
      }
      track.matched_this_frame = true;
      track.matched_armor_index = armor_index;
      (void)score;
      track.active = true;
    };

    auto create_track = [&](const ArmorDetectorResult& armor, uint8_t armor_index)
    {
      for (auto& track : tracks_)
      {
        if (track.active)
        {
          continue;
        }
        track = ImageTrack{};
        track.active = true;
        track.confirmed = false;
        track.track_id = next_track_id_++;
        assign_track(track, armor, armor_index, 0.0);
        return;
      }
    };

    auto suppress_spawn = [&](const ArmorDetectorResult& armor)
    {
      for (const auto& track : tracks_)
      {
        if (!track.active || track.miss_count > 6U)
        {
          continue;
        }
        if (!ImageTrackManager::CompatibleLabel(track, armor))
        {
          continue;
        }
        const double dt =
            std::max(ImageTrackManager::TimestampDeltaSeconds(image_timestamp_us,
                                                             track.last_timestamp_us),
                     1.0 / 100.0);
        const cv::Point2f predicted_center(
            track.image_center.x + track.image_velocity.x * static_cast<float>(dt),
            track.image_center.y + track.image_velocity.y * static_cast<float>(dt));
        const double center_diff = std::hypot(
            static_cast<double>(armor.center.x - predicted_center.x),
            static_cast<double>(armor.center.y - predicted_center.y));
        const double predicted_area = std::max(1.0, track.area + track.area_rate * dt);
        const double area = std::max(1.0, ImageTrackManager::ArmorArea(armor));
        const double area_log = std::abs(std::log(area / predicted_area));
        const bool strong_same_track =
            (center_diff < 55.0 && area_log < 0.45) ||
            (center_diff < 28.0 && area_log < 0.90);
        const bool relaxed_confirmed_same_track =
            track.confirmed && ((center_diff < 72.0 && area_log < 0.28) ||
                                (center_diff < 42.0 && area_log < 0.45));
        if (strong_same_track || relaxed_confirmed_same_track)
        {
          return true;
        }
      }
      return false;
    };

    for (auto& track : tracks_)
    {
      if (!track.active)
      {
        continue;
      }
      track.matched_this_frame = false;
      track.matched_armor_index = 255;
    }

    std::array<cv::Point2f, kMaxTracks> predicted_centers{};
    std::array<bool, kMaxTracks> predicted_center_valid{};
    for (std::size_t track_slot = 0; track_slot < tracks_.size(); ++track_slot)
    {
      const auto& track = tracks_[track_slot];
      if (!track.active)
      {
        continue;
      }
      const double dt =
          std::max(ImageTrackManager::TimestampDeltaSeconds(image_timestamp_us,
                                                           track.last_timestamp_us),
                   1.0 / 100.0);
      predicted_centers[track_slot] = cv::Point2f(
          track.image_center.x + track.image_velocity.x * static_cast<float>(dt),
          track.image_center.y + track.image_velocity.y * static_cast<float>(dt));
      predicted_center_valid[track_slot] = true;
    }

    std::array<std::vector<double>, kMaxTracks> order_bias_by_track;
    for (auto& bias_vec : order_bias_by_track)
    {
      bias_vec.assign(armors.size(), 0.0);
    }

    auto apply_dual_order_bias = [&](std::size_t lhs_slot, std::size_t rhs_slot)
    {
      const auto& lhs_track = tracks_[lhs_slot];
      const auto& rhs_track = tracks_[rhs_slot];
      if (!lhs_track.active || !rhs_track.active || !lhs_track.confirmed ||
          !rhs_track.confirmed)
      {
        return;
      }
      if (!predicted_center_valid[lhs_slot] || !predicted_center_valid[rhs_slot])
      {
        return;
      }
      if (!compatible_track_pair(lhs_track, rhs_track))
      {
        return;
      }

      std::size_t compatible_track_count = 0;
      for (const auto& track : tracks_)
      {
        if (!track.active || !track.confirmed)
        {
          continue;
        }
        if (track.type == lhs_track.type && track.number == lhs_track.number &&
            track.color == lhs_track.color)
        {
          compatible_track_count++;
        }
      }
      if (compatible_track_count != 2U)
      {
        return;
      }

      std::vector<std::size_t> compatible_detections;
      for (std::size_t armor_index = 0; armor_index < armors.size(); ++armor_index)
      {
        if (ImageTrackManager::CompatibleLabel(lhs_track, armors[armor_index]) &&
            ImageTrackManager::CompatibleLabel(rhs_track, armors[armor_index]))
        {
          compatible_detections.push_back(armor_index);
        }
      }
      if (compatible_detections.size() != 2U)
      {
        return;
      }

      cv::Point2f axis(predicted_centers[rhs_slot].x - predicted_centers[lhs_slot].x,
                       predicted_centers[rhs_slot].y - predicted_centers[lhs_slot].y);
      double axis_norm =
          std::hypot(static_cast<double>(axis.x), static_cast<double>(axis.y));
      if (axis_norm < 20.0)
      {
        axis = std::abs(axis.x) >= std::abs(axis.y) ? cv::Point2f(1.0f, 0.0f)
                                                    : cv::Point2f(0.0f, 1.0f);
        axis_norm = 1.0;
      }
      axis.x = static_cast<float>(axis.x / axis_norm);
      axis.y = static_cast<float>(axis.y / axis_norm);

      auto project = [&](const cv::Point2f& p)
      {
        return static_cast<double>(p.x) * static_cast<double>(axis.x) +
               static_cast<double>(p.y) * static_cast<double>(axis.y);
      };

      std::array<std::size_t, 2> ordered_track_slots = {lhs_slot, rhs_slot};
      if (project(predicted_centers[ordered_track_slots[0]]) >
          project(predicted_centers[ordered_track_slots[1]]))
      {
        std::swap(ordered_track_slots[0], ordered_track_slots[1]);
      }

      std::array<std::size_t, 2> ordered_detection_indices = {
          compatible_detections[0], compatible_detections[1]};
      if (project(armors[ordered_detection_indices[0]].center) >
          project(armors[ordered_detection_indices[1]].center))
      {
        std::swap(ordered_detection_indices[0], ordered_detection_indices[1]);
      }

      const double detection_sep =
          std::abs(project(armors[ordered_detection_indices[1]].center) -
                   project(armors[ordered_detection_indices[0]].center));
      if (detection_sep < 18.0)
      {
        return;
      }

      const double predicted_sep =
          std::abs(project(predicted_centers[ordered_track_slots[1]]) -
                   project(predicted_centers[ordered_track_slots[0]]));
      const double order_bias =
          predicted_sep > 90.0 && detection_sep > 60.0 ? 0.30 : 0.20;

      order_bias_by_track[ordered_track_slots[0]][ordered_detection_indices[0]] -=
          order_bias;
      order_bias_by_track[ordered_track_slots[0]][ordered_detection_indices[1]] +=
          order_bias;
      order_bias_by_track[ordered_track_slots[1]][ordered_detection_indices[0]] +=
          order_bias;
      order_bias_by_track[ordered_track_slots[1]][ordered_detection_indices[1]] -=
          order_bias;
    };

    for (std::size_t lhs_slot = 0; lhs_slot < tracks_.size(); ++lhs_slot)
    {
      for (std::size_t rhs_slot = lhs_slot + 1; rhs_slot < tracks_.size(); ++rhs_slot)
      {
        apply_dual_order_bias(lhs_slot, rhs_slot);
      }
    }

    struct MatchCandidate
    {
      std::size_t track_slot = 0;
      std::size_t armor_index = 0;
      double score = 0.0;
      double center_diff = 0.0;
      double area_log = 0.0;
    };
    std::vector<MatchCandidate> candidates;

    for (std::size_t track_slot = 0; track_slot < tracks_.size(); ++track_slot)
    {
      const auto& track = tracks_[track_slot];
      if (!track.active)
      {
        continue;
      }
      const double dt =
          std::max(ImageTrackManager::TimestampDeltaSeconds(image_timestamp_us,
                                                           track.last_timestamp_us),
                   1.0 / 100.0);
      const double miss_scale = static_cast<double>(std::min<uint32_t>(track.miss_count, 6U));
      const cv::Point2f predicted_center = predicted_centers[track_slot];
      const double predicted_area = std::max(1.0, track.area + track.area_rate * dt);
      const double center_score_gate = 80.0 + 15.0 * miss_scale;
      const double center_gate = 140.0 + 20.0 * miss_scale;
      const double area_gate = 0.55 + 0.08 * miss_scale;

      for (std::size_t armor_index = 0; armor_index < armors.size(); ++armor_index)
      {
        const auto& armor = armors[armor_index];
        if (!ImageTrackManager::CompatibleLabel(track, armor))
        {
          continue;
        }
        const double center_diff = std::hypot(
            static_cast<double>(armor.center.x - predicted_center.x),
            static_cast<double>(armor.center.y - predicted_center.y));
        const double area = std::max(1.0, ImageTrackManager::ArmorArea(armor));
        const double area_log = std::abs(std::log(area / predicted_area));
        if (center_diff > center_gate || area_log > area_gate)
        {
          continue;
        }
        double score = 0.78 * center_diff / center_score_gate +
                       0.22 * area_log / area_gate -
                       0.05 * static_cast<double>(armor.confidence) +
                       order_bias_by_track[track_slot][armor_index];
        if (track.confirmed)
        {
          score -= 0.18;
        }
        if (track.miss_count == 0U)
        {
          score -= 0.08;
        }
        if (track.confirmed && track.miss_count == 0U && center_diff < 18.0 &&
            area_log < 0.12)
        {
          score -= 0.16;
        }
        candidates.push_back({track_slot, armor_index, score, center_diff, area_log});
      }
    }

    auto candidate_less = [&](const MatchCandidate& lhs, const MatchCandidate& rhs)
    {
      const auto& lhs_track = tracks_[lhs.track_slot];
      const auto& rhs_track = tracks_[rhs.track_slot];
      if (lhs_track.confirmed != rhs_track.confirmed &&
          std::abs(lhs.score - rhs.score) < 0.25)
      {
        return lhs_track.confirmed > rhs_track.confirmed;
      }
      if (lhs_track.miss_count != rhs_track.miss_count &&
          std::abs(lhs.score - rhs.score) < 0.20)
      {
        return lhs_track.miss_count < rhs_track.miss_count;
      }
      if (std::abs(lhs.score - rhs.score) > 1e-6)
      {
        return lhs.score < rhs.score;
      }
      return lhs_track.track_id < rhs_track.track_id;
    };

    std::vector<std::size_t> ordered_candidate_indices;
    ordered_candidate_indices.reserve(candidates.size());
    for (std::size_t candidate_index = 0; candidate_index < candidates.size();
         ++candidate_index)
    {
      ordered_candidate_indices.push_back(candidate_index);
    }
    std::sort(ordered_candidate_indices.begin(), ordered_candidate_indices.end(),
              [&](std::size_t lhs_index, std::size_t rhs_index)
              {
                return candidate_less(candidates[lhs_index], candidates[rhs_index]);
              });

    std::array<bool, kMaxTracks> track_used{};
    std::vector<bool> detection_used(armors.size(), false);
    std::vector<bool> candidate_selected(candidates.size(), false);
    std::array<std::vector<std::size_t>, kMaxTracks> track_candidate_indices;
    std::vector<int> detection_bit_indices(armors.size(), -1);
    std::vector<uint8_t> candidate_detection_bits(candidates.size(), 0);
    std::vector<std::size_t> unique_detection_indices;
    unique_detection_indices.reserve(armors.size());
    for (std::size_t candidate_index = 0; candidate_index < candidates.size();
         ++candidate_index)
    {
      const auto& candidate = candidates[candidate_index];
      track_candidate_indices[candidate.track_slot].push_back(candidate_index);
      if (detection_bit_indices[candidate.armor_index] < 0)
      {
        detection_bit_indices[candidate.armor_index] =
            static_cast<int>(unique_detection_indices.size());
        unique_detection_indices.push_back(candidate.armor_index);
      }
      candidate_detection_bits[candidate_index] =
          static_cast<uint8_t>(detection_bit_indices[candidate.armor_index]);
    }

    for (auto& candidate_indices : track_candidate_indices)
    {
      std::sort(candidate_indices.begin(), candidate_indices.end(),
                [&](std::size_t lhs_index, std::size_t rhs_index)
                {
                  return candidate_less(candidates[lhs_index], candidates[rhs_index]);
                });
    }

    bool used_global_assignment = false;
    std::vector<std::size_t> assignment_track_slots;
    assignment_track_slots.reserve(tracks_.size());
    for (std::size_t track_slot = 0; track_slot < tracks_.size(); ++track_slot)
    {
      if (!track_candidate_indices[track_slot].empty())
      {
        assignment_track_slots.push_back(track_slot);
      }
    }
    std::sort(assignment_track_slots.begin(), assignment_track_slots.end(),
              [&](std::size_t lhs_slot, std::size_t rhs_slot)
              {
                const auto& lhs_track = tracks_[lhs_slot];
                const auto& rhs_track = tracks_[rhs_slot];
                if (lhs_track.confirmed != rhs_track.confirmed)
                {
                  return lhs_track.confirmed > rhs_track.confirmed;
                }
                if (lhs_track.miss_count != rhs_track.miss_count)
                {
                  return lhs_track.miss_count < rhs_track.miss_count;
                }
                if (track_candidate_indices[lhs_slot].size() !=
                    track_candidate_indices[rhs_slot].size())
                {
                  return track_candidate_indices[lhs_slot].size() <
                         track_candidate_indices[rhs_slot].size();
                }
                return lhs_track.track_id < rhs_track.track_id;
              });

    const std::size_t unique_detection_count = unique_detection_indices.size();
    if (!assignment_track_slots.empty() && unique_detection_count <= 16U)
    {
      struct AssignmentState
      {
        bool valid{false};
        uint16_t matched_count{0};
        uint16_t confirmed_matched_count{0};
        double total_score{0.0};
      };

      const std::size_t state_count = std::size_t{1} << unique_detection_count;
      const std::size_t memo_size = (assignment_track_slots.size() + 1U) * state_count;
      std::vector<uint8_t> memo_seen(memo_size, 0);
      std::vector<AssignmentState> memo(memo_size);
      std::vector<int32_t> memo_choice(memo_size, -2);

      auto assignment_better = [&](const AssignmentState& lhs,
                                   const AssignmentState& rhs)
      {
        if (!lhs.valid)
        {
          return false;
        }
        if (!rhs.valid)
        {
          return true;
        }
        if (lhs.matched_count != rhs.matched_count)
        {
          return lhs.matched_count > rhs.matched_count;
        }
        if (lhs.confirmed_matched_count != rhs.confirmed_matched_count)
        {
          return lhs.confirmed_matched_count > rhs.confirmed_matched_count;
        }
        if (std::abs(lhs.total_score - rhs.total_score) > 1e-9)
        {
          return lhs.total_score < rhs.total_score;
        }
        return false;
      };

      auto memo_index_of = [&](std::size_t track_order_index, std::size_t used_mask)
      {
        return track_order_index * state_count + used_mask;
      };

      auto solve_assignment =
          [&](auto&& self, std::size_t track_order_index,
              std::size_t used_mask) -> AssignmentState
      {
        const std::size_t memo_index = memo_index_of(track_order_index, used_mask);
        if (memo_seen[memo_index] != 0U)
        {
          return memo[memo_index];
        }
        memo_seen[memo_index] = 1U;

        AssignmentState best;
        best.valid = true;
        memo_choice[memo_index] = -1;
        if (track_order_index >= assignment_track_slots.size())
        {
          memo[memo_index] = best;
          return best;
        }

        best = self(self, track_order_index + 1U, used_mask);

        const std::size_t track_slot = assignment_track_slots[track_order_index];
        const auto& track = tracks_[track_slot];
        for (std::size_t candidate_index : track_candidate_indices[track_slot])
        {
          const std::size_t detection_bit = candidate_detection_bits[candidate_index];
          const std::size_t detection_mask = std::size_t{1} << detection_bit;
          if ((used_mask & detection_mask) != 0U)
          {
            continue;
          }

          AssignmentState next =
              self(self, track_order_index + 1U, used_mask | detection_mask);
          if (!next.valid)
          {
            continue;
          }
          next.matched_count++;
          if (track.confirmed)
          {
            next.confirmed_matched_count++;
          }
          next.total_score += candidates[candidate_index].score;
          if (assignment_better(next, best))
          {
            best = next;
            memo_choice[memo_index] = static_cast<int32_t>(candidate_index);
          }
        }

        memo[memo_index] = best;
        return best;
      };

      const AssignmentState best_assignment =
          solve_assignment(solve_assignment, 0U, 0U);
      if (best_assignment.valid)
      {
        used_global_assignment = true;
        std::size_t used_mask = 0U;
        for (std::size_t track_order_index = 0;
             track_order_index < assignment_track_slots.size(); ++track_order_index)
        {
          const std::size_t memo_index = memo_index_of(track_order_index, used_mask);
          const int32_t choice = memo_choice[memo_index];
          if (choice < 0)
          {
            continue;
          }
          const std::size_t candidate_index = static_cast<std::size_t>(choice);
          candidate_selected[candidate_index] = true;
          used_mask |= std::size_t{1} << candidate_detection_bits[candidate_index];
        }
      }
    }

    if (!used_global_assignment)
    {
      for (std::size_t candidate_index : ordered_candidate_indices)
      {
        const auto& candidate = candidates[candidate_index];
        if (track_used[candidate.track_slot] || detection_used[candidate.armor_index])
        {
          continue;
        }
        candidate_selected[candidate_index] = true;
        track_used[candidate.track_slot] = true;
        detection_used[candidate.armor_index] = true;
      }
      track_used.fill(false);
      std::fill(detection_used.begin(), detection_used.end(), false);
    }

    for (std::size_t candidate_index : ordered_candidate_indices)
    {
      if (!candidate_selected[candidate_index])
      {
        continue;
      }
      const auto& candidate = candidates[candidate_index];
      auto& track = tracks_[candidate.track_slot];
      assign_track(track, armors[candidate.armor_index],
                   static_cast<uint8_t>(candidate.armor_index), candidate.score);
      track_used[candidate.track_slot] = true;
      detection_used[candidate.armor_index] = true;
    }

    for (std::size_t track_slot = 0; track_slot < tracks_.size(); ++track_slot)
    {
      auto& track = tracks_[track_slot];
      if (!track.active || track_used[track_slot])
      {
        continue;
      }

      const double dt =
          ImageTrackManager::TimestampDeltaSeconds(image_timestamp_us,
                                                  track.last_timestamp_us);
      if (dt > 1e-4)
      {
        track.image_center.x += track.image_velocity.x * static_cast<float>(dt);
        track.image_center.y += track.image_velocity.y * static_cast<float>(dt);
        track.area = std::max(1.0, track.area + track.area_rate * dt);
        track.last_timestamp_us = image_timestamp_us;
      }
      track.miss_count++;
      track.hit_count = 0;
      track.matched_this_frame = false;
      track.matched_armor_index = 255;

      const bool drop_tentative =
          !track.confirmed && track.miss_count >= cfg.tentative_misses &&
          timeout_satisfied(image_timestamp_us, track.last_seen_timestamp_us,
                            cfg.tentative_timeout_sec);
      const bool drop_confirmed =
          track.confirmed && track.miss_count >= cfg.disappear_misses &&
          timeout_satisfied(image_timestamp_us, track.last_seen_timestamp_us,
                            cfg.disappear_timeout_sec);
      if (drop_tentative || drop_confirmed)
      {
        reset_track(track);
      }
    }

    for (std::size_t armor_index = 0; armor_index < armors.size(); ++armor_index)
    {
      if (detection_used[armor_index] || suppress_spawn(armors[armor_index]))
      {
        continue;
      }
      create_track(armors[armor_index], static_cast<uint8_t>(armor_index));
    }

    for (const auto& track : tracks_)
    {
      if (!track.active || !track.matched_this_frame || track.matched_armor_index == 255)
      {
        continue;
      }
      if (track.matched_armor_index < detection_track_ids_.size())
      {
        detection_track_ids_[track.matched_armor_index] = track.track_id;
        detection_track_confirmed_[track.matched_armor_index] = track.confirmed ? 1 : 0;
      }
    }
  }

  int FindDetectionTrackId(std::size_t armor_index) const
  {
    if (armor_index >= detection_track_ids_.size())
    {
      return -1;
    }
    return detection_track_ids_[armor_index];
  }

  bool IsDetectionTrackConfirmed(std::size_t armor_index) const
  {
    if (armor_index >= detection_track_confirmed_.size())
    {
      return false;
    }
    return detection_track_confirmed_[armor_index] != 0;
  }

  const std::array<ImageTrack, kMaxTracks>& Tracks() const
  {
    return tracks_;
  }

 private:
  static double ArmorArea(const ArmorDetectorResult& armor)
  {
    return std::abs(
        cv::contourArea(std::vector<cv::Point2f>(armor.points.begin(), armor.points.end())));
  }

  static double TimestampDeltaSeconds(uint64_t newer, uint64_t older)
  {
    if (newer > older)
    {
      return static_cast<double>(newer - older) / 1000000.0;
    }
    return 0.0;
  }

  static bool CompatibleLabel(const ImageTrack& track, const ArmorDetectorResult& armor)
  {
    if (track.type != ArmorType::INVALID && armor.type != ArmorType::INVALID &&
        track.type != armor.type)
    {
      return false;
    }
    if (track.color != ArmorColor::UNKNOWN && armor.color != ArmorColor::UNKNOWN &&
        track.color != armor.color)
    {
      return false;
    }
    if (track.number != ArmorNumber::INVALID && armor.number != ArmorNumber::INVALID &&
        track.number != armor.number)
    {
      return false;
    }
    return true;
  }

  std::array<ImageTrack, kMaxTracks> tracks_{};
  std::vector<int> detection_track_ids_{};
  std::vector<uint8_t> detection_track_confirmed_{};
  uint16_t next_track_id_ = 0;
};
}  // namespace armor_tracker
