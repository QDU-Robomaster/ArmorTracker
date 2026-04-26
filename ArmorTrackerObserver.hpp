#pragma once

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Eigen>

#include "ArmorTrackerFaceSelector.hpp"

namespace armor_tracker
{
// Observer 层只关心“整车状态如何表达、如何被装甲观测修正”。
// 它不接触 topic、不读配置文件，也不关心 preview/debug 输出。
struct ObserverPolicy
{
  bool single_armor_mode = false;
  bool symmetric_geometry_enabled = false;
  double max_match_distance = 0.15;
  double max_match_yaw_diff = 1.0;
  double initial_radius = 0.26;
};

struct ObserverRuntime
{
  ArmorNumber tracked_id = ArmorNumber::INVALID;
  ArmorType tracked_armor_type = ArmorType::INVALID;
  int tracked_armors_num = 4;
  int tracked_face_index = 0;
  bool tracked_face_track_id_valid = false;
  uint16_t tracked_face_track_id = 0;
  std::array<bool, 4> face_track_id_valid{};
  std::array<uint16_t, 4> face_track_id{};
  double last_yaw = 0.0;
  double face_switch_cooldown_remaining = 0.0;
  double dz = 0.0;
  double dz_abs_ref = 0.0;
  double another_r = 0.0;
};

inline void UpdateArmorsNum(ObserverRuntime& runtime,
                            const ObserverPolicy& policy)
{
  if (policy.single_armor_mode)
  {
    runtime.tracked_armors_num = 1;
    runtime.tracked_face_index = 0;
    return;
  }
  if (runtime.tracked_id == ArmorNumber::OUTPOST)
  {
    runtime.tracked_armors_num = 3;
  }
  else
  {
    runtime.tracked_armors_num = 4;
  }
  const int armor_count = std::max(1, runtime.tracked_armors_num);
  runtime.tracked_face_index =
      ((runtime.tracked_face_index % armor_count) + armor_count) % armor_count;
}

inline int NormalizeFaceIndex(int face_index, int armor_count)
{
  const int bounded_count = std::max(1, armor_count);
  return ((face_index % bounded_count) + bounded_count) % bounded_count;
}

inline double OrientationToYaw(const LibXR::Quaternion<double>& q,
                               ObserverRuntime& runtime)
{
  const double yaw = OrientationToYawNear(q, runtime.last_yaw);
  runtime.last_yaw = yaw;
  return yaw;
}

inline double GetArmorYawFromState(const Eigen::VectorXd& state,
                                   const ObserverRuntime& runtime,
                                   int face_index = 0)
{
  const int armor_count = std::max(1, runtime.tracked_armors_num);
  const double angle_step = 2.0 * M_PI / armor_count;
  return state(6) - angle_step * face_index;
}

inline double GetArmorSecondRadiusFromState(const Eigen::VectorXd& state,
                                            const ObserverPolicy& policy)
{
  if (policy.symmetric_geometry_enabled)
  {
    return state(8);
  }
  return state(8) + state(9);
}

inline double GetArmorDzFromState(const Eigen::VectorXd& state,
                                  const ObserverPolicy& policy)
{
  if (policy.symmetric_geometry_enabled)
  {
    return 0.0;
  }
  return state(10);
}

inline Eigen::Vector3d GetArmorPositionFromState(const Eigen::VectorXd& state,
                                                 const ObserverRuntime& runtime,
                                                 const ObserverPolicy& policy,
                                                 int face_index = 0)
{
  const double xc = state(0);
  const double yc = state(2);
  double za = state(4);
  double r = state(8);
  const double yaw = GetArmorYawFromState(state, runtime, face_index);

  if (!policy.symmetric_geometry_enabled && runtime.tracked_armors_num == 4 &&
      face_index % 2 == 1)
  {
    r = GetArmorSecondRadiusFromState(state, policy);
    za = state(4) + GetArmorDzFromState(state, policy);
  }

  const double xa = xc - r * std::cos(yaw);
  const double ya = yc - r * std::sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}

inline void InitEkfState(Eigen::VectorXd& state, ObserverRuntime& runtime,
                         const ObserverPolicy& policy,
                         const ArmorDetectorResult& armor)
{
  const double xa = armor.pose.translation.x();
  const double ya = armor.pose.translation.y();
  const double za = armor.pose.translation.z();
  runtime.last_yaw = 0.0;
  const double yaw = OrientationToYaw(armor.pose.rotation, runtime);

  state = Eigen::VectorXd::Zero(11);
  const double r = policy.initial_radius;
  const double xc = xa + r * std::cos(yaw);
  const double yc = ya + r * std::sin(yaw);
  runtime.dz = 0.0;
  runtime.dz_abs_ref = 0.0;
  runtime.another_r = r;
  runtime.tracked_face_index = 0;
  runtime.face_switch_cooldown_remaining = 0.0;
  state << xc, 0, yc, 0, za, 0, yaw, 0, r, 0, 0;
}

inline void SyncDzReferenceFromState(ObserverRuntime& runtime)
{
  runtime.dz_abs_ref = std::abs(runtime.dz);
}

inline int LocalFaceToCanonicalFace(const ObserverRuntime& runtime, int face_index)
{
  return NormalizeFaceIndex(runtime.tracked_face_index + face_index,
                            std::max(1, runtime.tracked_armors_num));
}

inline void SwitchTrackedFace(ObserverRuntime& runtime, Eigen::VectorXd& state,
                              const ObserverPolicy& policy, int face_index,
                              double measured_yaw)
{
  (void)state;
  if (face_index == 0)
  {
    runtime.last_yaw = measured_yaw;
    return;
  }

  UpdateArmorsNum(runtime, policy);
  runtime.tracked_face_index =
      LocalFaceToCanonicalFace(runtime, face_index);
  runtime.last_yaw = measured_yaw;
}

template <typename TrackIdGetter, typename TrackConfirmedGetter>
bool FuseMultiArmorObservation(ObserverRuntime& runtime, Eigen::VectorXd& state,
                               const ObserverPolicy& policy,
                               const ArmorDetectorResults& armors_msg,
                               TrackIdGetter&& get_detection_track_id,
                               TrackConfirmedGetter&& is_detection_track_confirmed)
{
  runtime.another_r = GetArmorSecondRadiusFromState(state, policy);
  runtime.dz = GetArmorDzFromState(state, policy);
  runtime.dz_abs_ref = std::abs(runtime.dz);

  if (runtime.tracked_armors_num != 4 || armors_msg.size() < 2 ||
      runtime.tracked_id == ArmorNumber::INVALID)
  {
    return false;
  }

  struct FaceObservation
  {
    bool valid = false;
    ArmorDetectorResult armor{};
    double measured_yaw = 0.0;
    double position_diff = DBL_MAX;
    int image_track_id = -1;
    bool confirmed_image_track = false;
  };

  struct FuseCandidate
  {
    std::size_t armor_index = 0;
    int face_index = -1;
    ArmorDetectorResult armor{};
    double measured_yaw = 0.0;
    double position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;
    int image_track_id = -1;
    bool confirmed_image_track = false;
    bool same_persistent_track = false;
  };

  std::array<FaceObservation, 4> faces{};
  bool have_even_face = false;
  bool have_odd_face = false;
  const double max_position_diff = policy.max_match_distance * 2.5;
  const double max_yaw_diff = std::max(policy.max_match_yaw_diff, 1.2);
  std::vector<FuseCandidate> candidates;

  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    if (armor.number != runtime.tracked_id)
    {
      continue;
    }
    if (runtime.tracked_armor_type != ArmorType::INVALID &&
        armor.type != runtime.tracked_armor_type)
    {
      continue;
    }

    const auto p = armor.pose.translation;
    const Eigen::Vector3d position_vec(p.x(), p.y(), p.z());
    const int image_track_id = get_detection_track_id(armor_index);
    const bool confirmed_image_track = is_detection_track_confirmed(armor_index);
    const bool same_persistent_track =
        runtime.tracked_face_track_id_valid && confirmed_image_track &&
        image_track_id >= 0 &&
        static_cast<uint16_t>(image_track_id) == runtime.tracked_face_track_id;
    int bound_face_index = -1;
    if (confirmed_image_track && image_track_id >= 0)
    {
      for (int face_slot = 0; face_slot < 4; ++face_slot)
      {
        if (runtime.face_track_id_valid[face_slot] &&
            runtime.face_track_id[face_slot] ==
                static_cast<uint16_t>(image_track_id))
        {
          bound_face_index = face_slot;
          break;
        }
      }
    }

    for (int face_index = 0; face_index < 4; ++face_index)
    {
      if (confirmed_image_track)
      {
        if (bound_face_index >= 0 && face_index != bound_face_index)
        {
          continue;
        }
        if (bound_face_index < 0 && runtime.face_track_id_valid[face_index])
        {
          continue;
        }
      }

      const Eigen::Vector3d predicted_position =
          GetArmorPositionFromState(state, runtime, policy, face_index);
      const double predicted_yaw =
          GetArmorYawFromState(state, runtime, face_index);
      const double measured_yaw =
          OrientationToYawNear(armor.pose.rotation, predicted_yaw);
      const double position_diff = (predicted_position - position_vec).norm();
      const double yaw_diff = AngularDiffAbs(measured_yaw, predicted_yaw);
      LogImpossibleYawDiff("fuse", armor_index, face_index, measured_yaw,
                           predicted_yaw, yaw_diff);
      if (position_diff >= max_position_diff || yaw_diff >= max_yaw_diff)
      {
        continue;
      }
      candidates.push_back({armor_index, face_index, armor, measured_yaw,
                            position_diff, yaw_diff, image_track_id,
                            confirmed_image_track, same_persistent_track});
    }
  }

  if (candidates.size() < 2)
  {
    return false;
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const FuseCandidate& lhs, const FuseCandidate& rhs)
            {
              if (lhs.same_persistent_track != rhs.same_persistent_track)
              {
                return lhs.same_persistent_track > rhs.same_persistent_track;
              }
              if (lhs.confirmed_image_track != rhs.confirmed_image_track &&
                  std::abs(lhs.position_diff - rhs.position_diff) < 0.03)
              {
                return lhs.confirmed_image_track > rhs.confirmed_image_track;
              }
              if (std::abs(lhs.position_diff - rhs.position_diff) > 1e-6)
              {
                return lhs.position_diff < rhs.position_diff;
              }
              return lhs.yaw_diff < rhs.yaw_diff;
            });

  std::vector<bool> armor_used(armors_msg.size(), false);
  std::array<bool, 4> face_used{};
  std::vector<int> used_confirmed_image_track_ids;
  int valid_face_count = 0;
  for (const auto& candidate : candidates)
  {
    if (candidate.face_index < 0 || candidate.face_index >= 4)
    {
      continue;
    }
    if (armor_used[candidate.armor_index] || face_used[candidate.face_index])
    {
      continue;
    }
    if (candidate.confirmed_image_track && candidate.image_track_id >= 0 &&
        std::find(used_confirmed_image_track_ids.begin(),
                  used_confirmed_image_track_ids.end(),
                  candidate.image_track_id) !=
            used_confirmed_image_track_ids.end())
    {
      continue;
    }
    armor_used[candidate.armor_index] = true;
    face_used[candidate.face_index] = true;
    if (candidate.confirmed_image_track && candidate.image_track_id >= 0)
    {
      used_confirmed_image_track_ids.push_back(candidate.image_track_id);
    }
    faces[candidate.face_index].valid = true;
    faces[candidate.face_index].armor = candidate.armor;
    faces[candidate.face_index].measured_yaw = candidate.measured_yaw;
    faces[candidate.face_index].position_diff = candidate.position_diff;
    faces[candidate.face_index].image_track_id = candidate.image_track_id;
    faces[candidate.face_index].confirmed_image_track =
        candidate.confirmed_image_track;
    ++valid_face_count;
  }

  if (valid_face_count < 2)
  {
    return false;
  }

  for (int face_index = 0; face_index < 4; ++face_index)
  {
    if (!faces[face_index].valid || !faces[face_index].confirmed_image_track ||
        faces[face_index].image_track_id < 0)
    {
      continue;
    }
    runtime.face_track_id_valid[face_index] = true;
    runtime.face_track_id[face_index] =
        static_cast<uint16_t>(faces[face_index].image_track_id);
  }
  if (runtime.face_track_id_valid[0])
  {
    runtime.tracked_face_track_id_valid = true;
    runtime.tracked_face_track_id = runtime.face_track_id[0];
  }

  int even_col = -1;
  int odd_col = -1;
  int cols = 2;
  for (int face_index = 0; face_index < 4; ++face_index)
  {
    if (!faces[face_index].valid)
    {
      continue;
    }
    if (face_index % 2 == 0)
    {
      have_even_face = true;
    }
    else
    {
      have_odd_face = true;
    }
  }
  if (have_even_face)
  {
    even_col = cols++;
  }
  if (have_odd_face)
  {
    odd_col = cols++;
  }

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(valid_face_count * 2, cols);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(valid_face_count * 2);
  int row = 0;
  for (int face_index = 0; face_index < 4; ++face_index)
  {
    if (!faces[face_index].valid)
    {
      continue;
    }
    const int radius_col = (face_index % 2 == 0) ? even_col : odd_col;
    if (radius_col < 0)
    {
      continue;
    }

    const auto p = faces[face_index].armor.pose.translation;
    const double yaw = faces[face_index].measured_yaw;
    A(row, 0) = 1.0;
    A(row, radius_col) = -std::cos(yaw);
    b(row) = p.x();
    ++row;
    A(row, 1) = 1.0;
    A(row, radius_col) = -std::sin(yaw);
    b(row) = p.y();
    ++row;
  }
  if (row < 4)
  {
    return false;
  }

  const int fit_rows = row;
  if (valid_face_count == 2)
  {
    constexpr double kCenterPriorWeight = 0.35;
    constexpr double kRadiusPriorWeight = 0.50;
    const int prior_rows = 2 + (have_even_face ? 1 : 0) + (have_odd_face ? 1 : 0);
    A.conservativeResize(row + prior_rows, cols);
    b.conservativeResize(row + prior_rows);

    A.row(row).setZero();
    A(row, 0) = kCenterPriorWeight;
    b(row) = kCenterPriorWeight * state(0);
    ++row;

    A.row(row).setZero();
    A(row, 1) = kCenterPriorWeight;
    b(row) = kCenterPriorWeight * state(2);
    ++row;

    if (have_even_face)
    {
      A.row(row).setZero();
      A(row, even_col) = kRadiusPriorWeight;
      b(row) = kRadiusPriorWeight * state(8);
      ++row;
    }
    if (have_odd_face)
    {
      A.row(row).setZero();
      A(row, odd_col) = kRadiusPriorWeight;
      b(row) = kRadiusPriorWeight * GetArmorSecondRadiusFromState(state, policy);
      ++row;
    }
  }
  else
  {
    A.conservativeResize(row, cols);
    b.conservativeResize(row);
  }

  const Eigen::VectorXd sol =
      Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeThinU | Eigen::ComputeThinV)
          .solve(b);
  const Eigen::VectorXd residual = A.topRows(fit_rows) * sol - b.head(fit_rows);
  const double rmse = std::sqrt(residual.squaredNorm() / std::max(1, fit_rows));
  if (!std::isfinite(rmse) || rmse > 0.05)
  {
    return false;
  }

  const double fused_x = sol(0);
  const double fused_y = sol(1);
  const double fused_r_even = have_even_face ? sol(even_col) : state(8);
  const double fused_r_odd =
      have_odd_face ? sol(odd_col) : GetArmorSecondRadiusFromState(state, policy);
  if (!std::isfinite(fused_x) || !std::isfinite(fused_y) ||
      !std::isfinite(fused_r_even) || !std::isfinite(fused_r_odd) ||
      fused_r_even < 0.05 || fused_r_even > 0.45 || fused_r_odd < 0.05 ||
      fused_r_odd > 0.45)
  {
    return false;
  }

  const double alpha = valid_face_count >= 3 ? 0.35 : 0.12;
  state(0) = (1.0 - alpha) * state(0) + alpha * fused_x;
  state(2) = (1.0 - alpha) * state(2) + alpha * fused_y;

  double radius_even = state(8);
  double radius_odd = GetArmorSecondRadiusFromState(state, policy);
  if (have_even_face)
  {
    radius_even = (1.0 - alpha) * radius_even + alpha * fused_r_even;
  }
  if (have_odd_face)
  {
    radius_odd = (1.0 - alpha) * radius_odd + alpha * fused_r_odd;
  }
  state(8) = radius_even;
  state(9) = policy.symmetric_geometry_enabled ? 0.0 : (radius_odd - radius_even);
  runtime.another_r = radius_odd;
  runtime.dz = GetArmorDzFromState(state, policy);
  runtime.dz_abs_ref = std::abs(runtime.dz);

  XR_LOG_DEBUG(
      "Tracker multi-armor fuse: faces=%d rmse=%.3f center=(%.3f, %.3f) r1=%.3f r2=%.3f",
      valid_face_count, rmse, state(0), state(2), state(8), runtime.another_r);
  return true;
}

/* Old observer implementation intentionally removed below. */
}  // namespace armor_tracker
