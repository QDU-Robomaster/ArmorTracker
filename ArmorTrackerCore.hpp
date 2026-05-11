#pragma once

/**
 * @file ArmorTrackerCore.hpp
 * @brief Header-only facade that adapts detector armors into tracker outputs.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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
  std::vector<Eigen::Vector4d> faces;
};

/**
 * @brief Tracker result in the configured output frame.
 */
struct Output
{
  std::string state{"lost"};
  bool has_target = false;
  int selected_tag_id = -1;
  int selected_armor_type = 0;
  int armors_num = 0;
  int selected_face = -1;
  bool jumped = false;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double vyaw = 0.0;
  double radius_even = 0.0;
  double radius_odd = 0.0;
  double dz = 0.0;
  Eigen::Vector3d armor = Eigen::Vector3d::Zero();
  double armor_yaw = 0.0;
  std::vector<Eigen::Vector4d> faces;
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
 * @brief Convert the tracker world frame vector to the camera output frame.
 */
inline Eigen::Vector3d WorldToCameraFrame(const Eigen::Vector3d& point)
{
  return {-point.y(), -point.z(), point.x()};
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
  explicit TrackerCore(const Config& config)
  {
    Configure(config);
  }

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
  }

  /**
   * @brief Advance tracker state with one detector frame.
   *
   * @param timestamp_us Sensor timestamp of the detector frame.
   * @param q_gimbal_to_world Gimbal-to-world IMU orientation.
   * @param inputs Detector armors from the same image frame.
   * @return Tracker output in the configured output frame.
   */
  Output Step(uint64_t timestamp_us, const Eigen::Quaterniond& q_gimbal_to_world,
              const std::vector<InputArmor>& inputs)
  {
    Eigen::Quaterniond q = q_gimbal_to_world;
    if (!std::isfinite(q.norm()) || q.norm() < 1e-9)
    {
      q = Eigen::Quaterniond::Identity();
    }
    q.normalize();
    solver_->SetRGimbal2World(q);

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
    const auto tp = base_tp_ + std::chrono::duration_cast<
                                   std::chrono::steady_clock::duration>(
                                   std::chrono::microseconds(delta_us));
    const auto targets = tracker_->Track(armors, tp);

    Output out;
    out.state = tracker_->State();
    out.selected_tag_id = config_.target_tag_id;
    for (const auto& snapshot : tracker_->Snapshots())
    {
      out.tracks.push_back(MakeTrackOutput(snapshot));
    }
    if (targets.empty())
    {
      return out;
    }
    const Target& target = targets.front();
    out.has_target = true;
    out.selected_tag_id = ModelNameToDetectorNumber(target.name);
    out.selected_armor_type = target.armor_type == ArmorType::BIG ? 1 : 0;
    out.selected_face = target.last_id;
    out.jumped = target.jumped;
    const Eigen::VectorXd x = target.EkfX();
    out.radius_even = x[8];
    out.radius_odd = x[8] + x[9];
    out.faces = target.ArmorXyzaList();
    out.armors_num = static_cast<int>(out.faces.size());
    if (config_.output_frame == 0)
    {
      out.center = {x[0], x[2], x[4]};
      out.velocity = {x[1], x[3], x[5]};
      out.yaw = LimitRad(x[6]);
      out.vyaw = x[7];
      out.dz = x[10];
      if (target.last_id >= 0 &&
          target.last_id < static_cast<int>(out.faces.size()))
      {
        const auto& face = out.faces[static_cast<std::size_t>(target.last_id)];
        out.armor = face.head<3>();
        out.armor_yaw = face[3];
      }
    }
    else
    {
      out.center = WorldToCameraFrame({x[0], x[2], x[4]});
      out.velocity = WorldToCameraFrame({x[1], x[3], x[5]});
      out.yaw = LimitRad(x[6] - kPi * 0.5);
      out.vyaw = x[7];
      out.dz = -x[10];
      if (target.last_id >= 0 &&
          target.last_id < static_cast<int>(out.faces.size()))
      {
        const auto& face = out.faces[static_cast<std::size_t>(target.last_id)];
        out.armor = WorldToCameraFrame(face.head<3>());
        out.armor_yaw = LimitRad(face[3] - kPi * 0.5);
      }
    }
    return out;
  }

 private:
  Config config_{};
  std::unique_ptr<Solver> solver_{};
  std::unique_ptr<Tracker> tracker_{};
  bool has_time_base_ = false;
  uint64_t base_timestamp_us_ = 0;
  std::chrono::steady_clock::time_point base_tp_{};

  /**
   * @brief Convert an internal target snapshot to public output-frame fields.
   */
  TrackOutput MakeTrackOutput(const Tracker::TrackSnapshot& snapshot) const
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
    out.radius_even = x[8];
    out.radius_odd = x[8] + x[9];
    out.faces = target.ArmorXyzaList();
    out.armors_num = static_cast<int>(out.faces.size());
    if (config_.output_frame == 0)
    {
      out.center = {x[0], x[2], x[4]};
      out.velocity = {x[1], x[3], x[5]};
      out.yaw = LimitRad(x[6]);
      out.vyaw = x[7];
      out.dz = x[10];
    }
    else
    {
      out.center = WorldToCameraFrame({x[0], x[2], x[4]});
      out.velocity = WorldToCameraFrame({x[1], x[3], x[5]});
      out.yaw = LimitRad(x[6] - kPi * 0.5);
      out.vyaw = x[7];
      out.dz = -x[10];
    }
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
