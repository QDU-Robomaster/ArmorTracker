#pragma once

/**
 * @file ArmorTrackerCore.hpp
 * @brief 将 detector 装甲板结果转为 tracker 输出。
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
 * @brief tracker core 使用的 detector 装甲板观测。
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
 * @brief 用于 preview 和调试绘制的一个有效车辆状态。
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
 * @brief tracker 单帧输出结果，用于目标输出和 preview 绘制。
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
 * @brief 将 detector 数字枚举转为 tracker 内部装甲板 ID。
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
 * @brief 判断 detector 装甲板是否使用大装甲几何尺寸。
 */
inline bool DetectorTypeIsLarge(const InputArmor& input)
{
  return input.armor_type == 1 || input.tag_id == 0 || input.tag_id == 7;
}

/**
 * @brief 将惯性 W 系 yaw 转为公开输出轴约定。
 */
inline double WorldYawToOutputYaw(double yaw_world, double output_yaw_world)
{
  return LimitRad(yaw_world - output_yaw_world);
}

/**
 * @brief 将惯性 W 系装甲面位姿转为公开输出轴约定。
 */
inline Eigen::Vector4d WorldFaceToOutputFace(
    const Eigen::Vector4d& face_world, const Eigen::Matrix3d& R_world_to_output,
    double output_yaw_world)
{
  const Eigen::Vector3d position_output =
      R_world_to_output * face_world.head<3>();
  return {position_output.x(), position_output.y(), position_output.z(),
          WorldYawToOutputYaw(face_world[3], output_yaw_world)};
}

/**
 * @brief 将内部装甲板 ID 转回 detector 数字枚举。
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
 * @brief 由 detector 输出生成内部装甲板观测。
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
 * @brief 模块和 replay 工具共用的有状态 tracker core。
 */
class TrackerCore
{
 public:
  /**
   * @brief 使用默认配置构造 tracker core。
   */
  TrackerCore() { Configure(config_); }

  /**
   * @brief 构造 tracker core 并设置配置。
   */
  explicit TrackerCore(const Config& config)
  {
    Configure(config);
  }

  /**
   * @brief 使用新配置重置 tracker core。
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
   * @brief 使用一帧 detector 结果推进 tracker 状态。
   *
   * @param timestamp_us Sensor timestamp of the detector frame.
   * @param q_body_to_world Body-to-world IMU orientation in public B axes.
   * @param inputs Detector armors from the same image frame.
   * @return Tracker output in the public right-handed inertial B-axis frame.
   */
  Output Step(uint64_t timestamp_us, const Eigen::Quaterniond& q_body_to_world,
              const std::vector<InputArmor>& inputs)
  {
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
    const auto tp = base_tp_ + std::chrono::duration_cast<
                                   std::chrono::steady_clock::duration>(
                                   std::chrono::microseconds(delta_us));
    (void)tracker_->Track(armors, tp);

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
    out.center = R_world_to_output * out.center_world;
    out.velocity = R_world_to_output * target.VelocityWorldForOutput();
    out.yaw = WorldYawToOutputYaw(out.yaw_world, output_yaw_world);
    out.vyaw = x[7];
    out.dz = target.DzForOutput();
    if (target.last_id >= 0 &&
        target.last_id < static_cast<int>(out.faces_world.size()))
    {
      const auto face = WorldFaceToOutputFace(
          out.faces_world[static_cast<std::size_t>(target.last_id)],
          R_world_to_output, output_yaw_world);
      out.armor = face.head<3>();
      out.armor_yaw = face[3];
    }
    return out;
  }

  /**
   * @brief 使用当前解算位姿重投影一个模型装甲面。
   *
   * The solver pose is updated by Step() before preview submission, so this
   * keeps preview projection on the same geometry path as tracker yaw fitting.
   */
  std::vector<cv::Point2f> ReprojectArmorFace(
      const Eigen::Vector3d& center_world, double yaw_world, int armor_type,
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

  /**
   * @brief 为 preview 绘制重投影一个装甲面。
   */
  std::vector<cv::Point2f> ReprojectPreviewArmorFace(
      const Eigen::Vector3d& center_world, double yaw_world, int armor_type,
      int tag_id) const
  {
    if (!solver_)
    {
      return {};
    }
    const ArmorType type = armor_type == 1 ? ArmorType::BIG : ArmorType::SMALL;
    return solver_->ReprojectPreviewArmor(center_world, yaw_world, type,
                                          ArmorNameFromDetectorNumber(tag_id));
  }

 private:
  Config config_{};
  std::unique_ptr<Solver> solver_{};
  std::unique_ptr<Tracker> tracker_{};
  bool has_time_base_ = false;
  uint64_t base_timestamp_us_ = 0;
  std::chrono::steady_clock::time_point base_tp_{};

  /**
   * @brief 将内部目标快照转为公开输出坐标字段。
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
   * @brief 生成内部装甲板观测前检查 detector 字段。
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
