#pragma once

/**
 * @file ArmorTrackerCore.hpp
 * @brief Header-only facade that adapts detector armors into tracker outputs.
 */

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "ArmorTrackerAaest.hpp"
#include "ArmorTrackerModel.hpp"

namespace armor_tracker_detail
{
/**
 * @brief Detector armor observation consumed by the tracker core.
 */
struct InputArmor
{
  int tag_id = -1;
  int armor_type = 0;
  double confidence = 0.0;
  std::array<cv::Point2f, 4> corners{};
  std::array<cv::Point2f, 4> frame_corners{};
  bool frame_corners_valid = false;
  cv::Point2f center{};
  cv::Point2f center_norm{0.5F, 0.5F};
};

/**
 * @brief One active tracked vehicle slot prepared for preview/debug drawing.
 */
struct TrackOutput
{
  std::string state{"lost"};
  bool selected = false;
  int tag_id = -1;
  int armor_type = 0;
  int armors_num = 0;
  int selected_face = -1;
  double score = 0.0;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double vyaw = 0.0;
  double radius_even = 0.0;
  double radius_odd = 0.0;
  double dz = 0.0;
  Eigen::Vector3d center_world = Eigen::Vector3d::Zero();
  double yaw_world = 0.0;
  std::vector<Eigen::Vector4d> faces_world;
};

/**
 * @brief Tracker result prepared for public-axis target output and preview drawing.
 */
struct Output
{
  std::string state{"lost"};
  bool has_target = false;
  int selected_tag_id = -1;
  int selected_armor_type = 0;
  int armors_num = 0;
  int selected_face = -1;
  int outpost_height_phase = 0;
  bool jumped = false;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double vyaw = 0.0;
  double radius_even = 0.0;
  double radius_odd = 0.0;
  double dz = 0.0;
  Eigen::Vector3d center_world = Eigen::Vector3d::Zero();
  double yaw_world = 0.0;
  Eigen::Vector3d armor = Eigen::Vector3d::Zero();
  double armor_yaw = 0.0;
  std::vector<Eigen::Vector4d> faces_world;
  std::vector<TrackOutput> tracks;
};

/**
 * @brief Convert detector number enum values to the internal armor-name id.
 */
inline int DetectorNumberToModelNameId(int number)
{
  if (number >= 0 && number <= 5)
  {
    return number + 1;
  }
  if (number == 6)
  {
    return 0;
  }
  if (number == 7)
  {
    return 7;
  }
  return number;
}

/**
 * @brief Return whether a detector armor should use the large armor geometry.
 */
inline bool DetectorTypeIsLarge(const InputArmor& input)
{
  return input.armor_type == 1 || input.tag_id == 0 || input.tag_id == 7;
}

/**
 * @brief Convert an inertial W-frame yaw into the public output axes.
 */
inline double WorldYawToOutputYaw(double yaw_world, double output_yaw_world)
{
  return LimitRad(yaw_world - output_yaw_world);
}

/**
 * @brief Convert an inertial W-frame armor face pose into public output axes.
 */
inline Eigen::Vector4d WorldFaceToOutputFace(const Eigen::Vector4d& face_world,
                                             const Eigen::Matrix3d& R_world_to_output,
                                             double output_yaw_world)
{
  const Eigen::Vector3d position_output = R_world_to_output * face_world.head<3>();
  return {position_output.x(), position_output.y(), position_output.z(),
          WorldYawToOutputYaw(face_world[3], output_yaw_world)};
}

/**
 * @brief Convert an internal armor name back to the detector number enum value.
 */
inline int ModelNameToDetectorNumber(ArmorName name)
{
  switch (name)
  {
    case ArmorName::ONE:
      return 0;
    case ArmorName::TWO:
      return 1;
    case ArmorName::THREE:
      return 2;
    case ArmorName::FOUR:
      return 3;
    case ArmorName::FIVE:
      return 4;
    case ArmorName::OUTPOST:
      return 5;
    case ArmorName::SENTRY:
      return 6;
    case ArmorName::BASE:
      return 7;
    case ArmorName::NOT_ARMOR:
    default:
      return 8;
  }
}

/**
 * @brief Build an internal armor observation from detector output.
 */
inline Armor MakeTrackedArmor(const InputArmor& input)
{
  std::vector<cv::Point2f> points;
  points.reserve(4);
  for (const auto& point : input.corners)
  {
    points.push_back(point);
  }
  const cv::Rect box = cv::boundingRect(points);
  Armor armor(DetectorNumberToModelNameId(input.tag_id),
              static_cast<float>(input.confidence), box, points);
  armor.type = DetectorTypeIsLarge(input) ? ArmorType::BIG : ArmorType::SMALL;
  armor.priority = PriorityFromName(armor.name);
  armor.center_norm = input.center_norm;
  if (input.frame_corners_valid)
  {
    armor.selection_points.assign(input.frame_corners.begin(), input.frame_corners.end());
  }
  else
  {
    armor.selection_points = points;
  }
  return armor;
}

/**
 * @brief Stateful facade used by the module and replay tool.
 */
class TrackerCore
{
 public:
  /**
   * @brief Construct with default tracker configuration.
   */
  TrackerCore() { Configure(config_); }

  /**
   * @brief Construct and configure the tracker core.
   */
  explicit TrackerCore(const Config& config) { Configure(config); }

  /**
   * @brief Reset the core using a new configuration.
   */
  void Configure(const Config& config)
  {
    config_ = config;
    solver_ = std::make_unique<Solver>(config_);
    tracker_ = std::make_unique<Tracker>(config_, *solver_);
    has_time_base_ = false;
    base_timestamp_us_ = 0;
    ConfigureAaest();
  }

  /**
   * @brief Advance tracker state with one detector frame.
   *
   * A backward timestamp resets tracking history and reacquires using unchanged
   * calibration and configuration.
   * @param timestamp_us Sensor timestamp of the detector frame.
   * @param q_body_to_world Body-to-world IMU orientation in public B axes.
   * @param inputs Detector armors from the same image frame.
   * @param angular_velocity_body Body angular rate (rad/s, public B axes) sampled with the
   *        attitude. Nonzero enables aaest's image-to-IMU time offset estimate; zero keeps
   *        the attitude-only behaviour.
   * @return Tracker output in the public right-handed inertial B-axis frame.
   */
  Output Step(uint64_t timestamp_us, const Eigen::Quaterniond& q_body_to_world,
              const std::vector<InputArmor>& inputs,
              const Eigen::Vector3d& angular_velocity_body = Eigen::Vector3d::Zero())
  {
    if (has_time_base_ && timestamp_us < last_timestamp_us_)
    {
      tracker_ = std::make_unique<Tracker>(config_, *solver_);
      has_time_base_ = false;
      aaest_.clear();
    }
    last_timestamp_us_ = timestamp_us;
    Eigen::Quaterniond q = q_body_to_world;
    if (!std::isfinite(q.norm()) || q.norm() < 1e-9)
    {
      q = Eigen::Quaterniond::Identity();
    }
    q.normalize();
    solver_->SetRBodyToWorld(q);
    const Eigen::Matrix3d R_world_to_output = Eigen::Matrix3d::Identity();
    constexpr double output_yaw_world = 0.0;

    std::list<Armor> armors;
    for (const auto& input : inputs)
    {
      if (!ValidInput(input))
      {
        continue;
      }
      armors.push_back(MakeTrackedArmor(input));
    }

    if (!has_time_base_)
    {
      has_time_base_ = true;
      base_timestamp_us_ = timestamp_us;
      base_tp_ = std::chrono::steady_clock::now();
    }
    const uint64_t delta_us =
        timestamp_us >= base_timestamp_us_ ? timestamp_us - base_timestamp_us_ : 0;
    const auto tp =
        base_tp_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       std::chrono::microseconds(delta_us));
    (void)tracker_->Track(armors, tp);
    if (config_.use_aaest && aaest_supported_)
    {
      StepAaest(timestamp_us, q, angular_velocity_body, inputs);
    }

    Output out;
    out.state = tracker_->State();
    out.selected_tag_id = config_.target_tag_id;
    const auto snapshots = tracker_->Snapshots();
    const Tracker::TrackSnapshot* selected_snapshot = nullptr;
    for (const auto& snapshot : snapshots)
    {
      if (snapshot.selected)
      {
        selected_snapshot = &snapshot;
      }
      out.tracks.push_back(
          MakeTrackOutput(snapshot, R_world_to_output, output_yaw_world));
    }
    if (selected_snapshot == nullptr)
    {
      return out;
    }
    const Target& target = selected_snapshot->target;
    out.has_target = true;
    out.selected_tag_id = ModelNameToDetectorNumber(target.name);
    out.selected_armor_type = target.armor_type == ArmorType::BIG ? 1 : 0;
    out.selected_face = target.last_id;
    out.outpost_height_phase = target.OutpostHeightPhase();
    out.jumped = target.jumped;
    const Eigen::VectorXd x = target.EkfX();
    out.radius_even = x[8];
    out.radius_odd = x[8] + x[9];
    out.faces_world = target.ArmorXyzaListForOutput();
    out.armors_num = static_cast<int>(out.faces_world.size());
    out.center_world = target.CenterWorldForOutput();
    out.yaw_world = LimitRad(x[6]);
    if (target.name == ArmorName::OUTPOST && out.armors_num == 3)
    {
      out.yaw_world = LimitRad(out.yaw_world + kPi);
    }
    out.center = R_world_to_output * out.center_world;
    out.velocity = R_world_to_output * target.VelocityWorldForOutput();
    out.yaw = WorldYawToOutputYaw(out.yaw_world, output_yaw_world);
    out.vyaw = x[7];
    out.dz = target.DzForOutput();
    if (target.last_id >= 0 && target.last_id < static_cast<int>(out.faces_world.size()))
    {
      const auto face =
          WorldFaceToOutputFace(out.faces_world[static_cast<std::size_t>(target.last_id)],
                                R_world_to_output, output_yaw_world);
      out.armor = face.head<3>();
      out.armor_yaw = face[3];
    }
    if (config_.use_aaest && aaest_supported_)
    {
      ApplyAaest(out, R_world_to_output, output_yaw_world);
    }
    return out;
  }

  /**
   * @brief Reproject one modeled armor face using the current solver pose.
   *
   * The solver pose is updated by Step() before preview submission, so this
   * keeps preview projection on the same geometry path as tracker yaw fitting.
   */
  std::vector<cv::Point2f> ReprojectArmorFace(const Eigen::Vector3d& center_world,
                                              double yaw_world, int armor_type,
                                              int tag_id) const
  {
    if (!solver_)
    {
      return {};
    }
    const ArmorType type = armor_type == 1 ? ArmorType::BIG : ArmorType::SMALL;
    return solver_->ReprojectArmor(center_world, yaw_world, type,
                                   ArmorNameFromDetectorNumber(tag_id));
  }

  bool IsArmorFaceFrontFacing(const Eigen::Vector3d& center_world, double yaw_world,
                              int tag_id) const
  {
    if (!solver_)
    {
      return false;
    }
    return solver_->IsArmorFaceFrontFacing(center_world, yaw_world,
                                           ArmorNameFromDetectorNumber(tag_id));
  }

 private:
  Config config_{};
  std::unique_ptr<Solver> solver_{};
  std::unique_ptr<Tracker> tracker_{};
  bool has_time_base_ = false;
  uint64_t base_timestamp_us_ = 0;
  uint64_t last_timestamp_us_ = 0;
  std::chrono::steady_clock::time_point base_tp_{};

  /**
   * @brief One aaest estimator per detector number of a four-plate vehicle.
   */
  struct AaestTrack
  {
    aaest::Estimator estimator;
    aaest::Target last{};
    uint64_t last_seen_us = 0;
  };
  std::map<int, AaestTrack> aaest_{};
  aaest::Camera aaest_camera_{};
  bool aaest_supported_ = false;

  /**
   * @brief Build the aaest camera from the tracker calibration and mount extrinsic.
   *
   * aaest projects with the OpenCV plumb-bob model (k1 k2 p1 p2 k3); calibrations with
   * further non-zero coefficients keep the EKF output.
   */
  void ConfigureAaest()
  {
    aaest_.clear();
    aaest_camera_.fx = config_.camera_matrix[0];
    aaest_camera_.cx = config_.camera_matrix[2];
    aaest_camera_.fy = config_.camera_matrix[4];
    aaest_camera_.cy = config_.camera_matrix[5];
    aaest_supported_ = config_.camera_model_supported;
    for (std::size_t i = 0; i < config_.distortion_coefficients.size(); ++i)
    {
      const double k = i < config_.distortion_size ? config_.distortion_coefficients[i] : 0.0;
      if (i < aaest_camera_.dist.size())
      {
        aaest_camera_.dist[i] = k;
      }
      else if (k != 0.0)
      {
        aaest_supported_ = false;
      }
    }
    aaest_camera_.R_cb =
        CameraToBodyRotationFromMountExtrinsic(config_.camera_mount_to_body_rotation);
    aaest_camera_.t_cb = Eigen::Vector3d(config_.camera_mount_to_body_translation[0],
                                         config_.camera_mount_to_body_translation[1],
                                         config_.camera_mount_to_body_translation[2]);
  }

  /**
   * @brief Whether a detection belongs to a four-plate vehicle that aaest models.
   *
   * Outpost (5) and base (7) have other plate layouts; big plates on numbers 2-4 are
   * two-plate balance infantry.
   */
  static bool AaestEligible(const InputArmor& input)
  {
    if (input.tag_id == 5 || input.tag_id == 7)
    {
      return false;
    }
    return !(input.armor_type == 1 &&
             (input.tag_id == 2 || input.tag_id == 3 || input.tag_id == 4));
  }

  /**
   * @brief Step every aaest estimator with the detections of its number in this frame.
   *
   * Estimators without detections for two seconds are dropped.
   */
  void StepAaest(uint64_t timestamp_us, const Eigen::Quaterniond& q,
                 const Eigen::Vector3d& angular_velocity_body,
                 const std::vector<InputArmor>& inputs)
  {
    std::map<int, std::vector<aaest::Detection>> by_tag;
    for (const auto& input : inputs)
    {
      if (!ValidInput(input) || !AaestEligible(input))
      {
        continue;
      }
      aaest::Detection det;
      for (std::size_t k = 0; k < 4; ++k)
      {
        det.corners[k] = aaest::Vec2(input.corners[k].x, input.corners[k].y);
      }
      det.type = input.armor_type == 1 ? 1 : 0;
      by_tag[input.tag_id].push_back(det);
    }
    for (const auto& entry : by_tag)
    {
      if (aaest_.find(entry.first) == aaest_.end())
      {
        aaest_.emplace(entry.first,
                       AaestTrack{aaest::Estimator(aaest_camera_), {}, timestamp_us});
      }
    }
    const double t = static_cast<double>(timestamp_us - base_timestamp_us_) * 1e-6;
    const std::array<double, 4> wxyz{q.w(), q.x(), q.y(), q.z()};
    static const std::vector<aaest::Detection> kNone{};
    for (auto it = aaest_.begin(); it != aaest_.end();)
    {
      const auto found = by_tag.find(it->first);
      const auto& dets = found == by_tag.end() ? kNone : found->second;
      if (!dets.empty())
      {
        it->second.last_seen_us = timestamp_us;
      }
      else if (timestamp_us - it->second.last_seen_us > 2000000U)
      {
        it = aaest_.erase(it);
        continue;
      }
      const double distance =
          it->second.last.tracking ? it->second.last.position.head<2>().norm() : 5.0;
      const double horizon =
          config_.aaest_latency_s + distance / std::max(config_.aaest_bullet_speed_m_s, 1.0);
      it->second.last =
          it->second.estimator.step(t, wxyz, angular_velocity_body, dets, horizon);
      ++it;
    }
  }

  /**
   * @brief Replace the selected four-plate target state by its aaest estimate.
   */
  void ApplyAaest(Output& out, const Eigen::Matrix3d& R_world_to_output,
                  double output_yaw_world) const
  {
    if (!out.has_target || out.armors_num != 4)
    {
      return;
    }
    const auto it = aaest_.find(out.selected_tag_id);
    if (it == aaest_.end() || !it->second.last.tracking)
    {
      return;
    }
    const aaest::Target& tg = it->second.last;
    out.center_world = tg.position;
    out.center = R_world_to_output * out.center_world;
    out.velocity =
        R_world_to_output * Eigen::Vector3d(tg.velocity.x(), tg.velocity.y(), 0.0);
    out.yaw_world = LimitRad(tg.yaw);
    out.yaw = WorldYawToOutputYaw(out.yaw_world, output_yaw_world);
    out.vyaw = tg.v_yaw;
    out.radius_even = tg.radius_1;
    out.radius_odd = tg.radius_2;
    out.dz = tg.dz;
    out.selected_face = tg.face;
    out.jumped = false;
    out.faces_world.clear();
    for (int k = 0; k < 4; ++k)
    {
      const double a = tg.yaw + k * aaest::HALF_PI;
      const bool odd = k % 2 == 1;
      const double r = odd ? tg.radius_2 : tg.radius_1;
      out.faces_world.emplace_back(tg.position.x() + r * std::sin(a),
                                   tg.position.y() - r * std::cos(a),
                                   tg.position.z() + (odd ? tg.dz : 0.0), LimitRad(a));
    }
    const auto face = WorldFaceToOutputFace(out.faces_world[static_cast<std::size_t>(tg.face)],
                                            R_world_to_output, output_yaw_world);
    out.armor = face.head<3>();
    out.armor_yaw = face[3];
  }

  /**
   * @brief Convert an internal target snapshot to public output-frame fields.
   */
  TrackOutput MakeTrackOutput(const Tracker::TrackSnapshot& snapshot,
                              const Eigen::Matrix3d& R_world_to_output,
                              double output_yaw_world) const
  {
    const Target& target = snapshot.target;
    TrackOutput out;
    out.state = snapshot.state;
    out.selected = snapshot.selected;
    out.score = snapshot.score;
    out.tag_id = ModelNameToDetectorNumber(target.name);
    out.armor_type = target.armor_type == ArmorType::BIG ? 1 : 0;
    out.selected_face = target.last_id;
    const Eigen::VectorXd x = target.EkfX();
    out.center_world = target.CenterWorldForOutput();
    out.yaw_world = LimitRad(x[6]);
    out.radius_even = x[8];
    out.radius_odd = x[8] + x[9];
    out.faces_world = target.ArmorXyzaListForOutput();
    out.armors_num = static_cast<int>(out.faces_world.size());
    out.center = R_world_to_output * out.center_world;
    out.velocity = R_world_to_output * target.VelocityWorldForOutput();
    out.yaw = WorldYawToOutputYaw(out.yaw_world, output_yaw_world);
    out.vyaw = x[7];
    out.dz = target.DzForOutput();
    return out;
  }

  /**
   * @brief Validate detector fields before constructing an internal armor.
   */
  static bool ValidInput(const InputArmor& input)
  {
    if (input.tag_id < 0)
    {
      return false;
    }
    for (const auto& corner : input.corners)
    {
      if (!std::isfinite(corner.x) || !std::isfinite(corner.y))
      {
        return false;
      }
    }
    return true;
  }
};

}  // namespace armor_tracker_detail
