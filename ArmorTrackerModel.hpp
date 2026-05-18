#pragma once

/**
 * @file ArmorTrackerModel.hpp
 * @brief Internal armor geometry, target state, and tracker state machine.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ArmorTrackerMath.hpp"

namespace armor_tracker_detail
{
/**
 * @brief Internal armor size labels used by PnP geometry.
 */
enum ArmorType
{
  BIG,
  SMALL
};

/**
 * @brief Internal armor identity labels used by the tracker model.
 */
enum ArmorName
{
  ONE,
  TWO,
  THREE,
  FOUR,
  FIVE,
  SENTRY,
  OUTPOST,
  BASE,
  NOT_ARMOR
};

/**
 * @brief Number of persistent vehicle tracks, excluding invalid armor labels.
 */
inline constexpr std::size_t kTrackSlotCount =
    static_cast<std::size_t>(ArmorName::NOT_ARMOR);

/**
 * @brief Convert a detector number enum value to the internal armor name.
 */
inline ArmorName ArmorNameFromDetectorNumber(int number)
{
  if (number >= 0 && number <= 4)
  {
    return static_cast<ArmorName>(number);
  }
  if (number == 5)
  {
    return ArmorName::OUTPOST;
  }
  if (number == 6)
  {
    return ArmorName::SENTRY;
  }
  if (number == 7)
  {
    return ArmorName::BASE;
  }
  return ArmorName::NOT_ARMOR;
}

/**
 * @brief Return the persistent track slot index for an armor name.
 */
inline int TrackSlotIndex(ArmorName name)
{
  const auto index = static_cast<int>(name);
  return index >= 0 && index < static_cast<int>(kTrackSlotCount) ? index : -1;
}

/// 前哨站相邻两块装甲板的中心高度差，单位 m。
inline constexpr double kOutpostArmorHeightStep = 0.102;
/// 前哨站转盘半径，单位 m。
inline constexpr double kOutpostArmorRadius = 0.2765;
/// 前哨站装甲板固定安装倾角，单位 rad。
inline constexpr double kOutpostArmorTilt = 15.0 * kPi / 180.0;
/// Webots 可见贴纸宽度，单位 m。
inline constexpr double kOutpostVisibleFaceWidth = 0.120;
/// Webots 可见贴纸高度，单位 m。
inline constexpr double kOutpostVisibleFaceHeight = 0.105;
/// Webots 可见贴纸相对装甲板中心的法向偏移，单位 m。
inline constexpr double kOutpostVisibleFaceXOffset = 0.008486277200519598;
/// Webots 可见贴纸相对装甲板中心的高度偏移，单位 m。
inline constexpr double kOutpostVisibleFaceZOffset = -0.003102112066953941;

inline int PositiveMod(int value, int mod)
{
  const int result = value % mod;
  return result < 0 ? result + mod : result;
}

inline double OutpostArmorHeightOffset(int face_id, int height_phase)
{
  // 本地面按 yaw 递增顺序展开为中、高、低。
  switch (PositiveMod(face_id + height_phase, 3))
  {
    case 1:
      return kOutpostArmorHeightStep;
    case 2:
      return -kOutpostArmorHeightStep;
    case 0:
    default:
      return 0.0;
  }
}

inline int SignNonZero(double value)
{
  return value > 0.0 ? 1 : -1;
}

/**
 * @brief Target selection priority for a detected armor.
 */
enum ArmorPriority
{
  FIRST = 1,
  SECOND,
  THIRD,
  FOURTH,
  FIFTH
};

/**
 * @brief Internal armor observation enriched with solved 3D pose.
 */
struct Armor
{
  cv::Point2f center{};
  cv::Point2f center_norm{};
  std::vector<cv::Point2f> points{};
  double ratio{};
  double rectangular_error{};
  ArmorType type{};
  ArmorName name{};
  ArmorPriority priority{FIFTH};
  cv::Rect box{};
  double confidence{};
  Eigen::Vector3d xyz_in_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypr_in_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();
  double yaw_raw{};

  /**
   * @brief Construct an empty armor observation.
   */
  Armor() = default;

  /**
   * @brief Build an armor observation from detector geometry.
   */
  Armor(int num_id, float confidence_in, const cv::Rect& box_in,
        std::vector<cv::Point2f> armor_keypoints)
      : points(std::move(armor_keypoints)),
        box(box_in),
        confidence(confidence_in)
  {
    center = (points[0] + points[1] + points[2] + points[3]) / 4.0F;
    const auto left_width = cv::norm(points[0] - points[3]);
    const auto right_width = cv::norm(points[1] - points[2]);
    const auto max_width = std::max(left_width, right_width);
    const auto top_length = cv::norm(points[0] - points[1]);
    const auto bottom_length = cv::norm(points[3] - points[2]);
    const auto max_length = std::max(top_length, bottom_length);
    const auto left_center = (points[0] + points[3]) / 2.0F;
    const auto right_center = (points[1] + points[2]) / 2.0F;
    const auto left2right = right_center - left_center;
    const auto roll = std::atan2(left2right.y, left2right.x);
    const auto left_rectangular_error = std::abs(
        std::atan2((points[3] - points[0]).y, (points[3] - points[0]).x) -
        roll - kPi / 2.0);
    const auto right_rectangular_error = std::abs(
        std::atan2((points[2] - points[1]).y, (points[2] - points[1]).x) -
        roll - kPi / 2.0);
    rectangular_error = std::max(left_rectangular_error, right_rectangular_error);
    ratio = max_length / max_width;
    name = num_id == 0 ? ArmorName::SENTRY
                       : num_id > 5 ? ArmorName(num_id) : ArmorName(num_id - 1);
    type = num_id == 1 ? ArmorType::BIG : ArmorType::SMALL;
  }
};

/**
 * @brief Map armor identity to tracker target-selection priority.
 */
inline ArmorPriority PriorityFromName(ArmorName name)
{
  switch (name)
  {
    case ArmorName::THREE:
    case ArmorName::FOUR:
      return ArmorPriority::FIRST;
    case ArmorName::ONE:
      return ArmorPriority::SECOND;
    case ArmorName::FIVE:
    case ArmorName::SENTRY:
      return ArmorPriority::THIRD;
    case ArmorName::TWO:
      return ArmorPriority::FOURTH;
    case ArmorName::OUTPOST:
    case ArmorName::BASE:
    case ArmorName::NOT_ARMOR:
    default:
      return ArmorPriority::FIFTH;
  }
}

/**
 * @brief Internal tracker configuration independent of module/runtime wiring.
 */
struct Config
{
  struct TargetSelectConfig
  {
    double observed_count_weight = 1.6;
    double distance_weight = 2.0;
    double area_weight = 1.2;
    double spin_weight = 0.8;
    double angle_weight = 2.0;
    double max_distance_m = 8.0;
    double distance_span_m = 7.5;
    double area_norm_px = 6000.0;
    double observed_count_norm = 4.0;
    double max_spin_rad_s = 8.0;
    double max_angle_norm = 0.5;
    double detecting_scale = 0.55;
    double temp_lost_scale = 0.35;
    double switch_margin = 0.25;
  };

  bool require_target_tag = false;
  int target_tag_id = -1;
  int min_detect_count = 2;
  int max_temp_lost_count = 15;
  int outpost_max_temp_lost_count = 75;
  TargetSelectConfig target_select{};
  std::array<double, 9> camera_matrix{
      1164.3428599490444, 0.0, 366.6782312546237,
      0.0, 1164.335053894998, 270.30936434613865,
      0.0, 0.0, 1.0};
  std::array<double, 4> camera_mount_to_body_rotation{1.0, 0.0, 0.0, 0.0};
  std::array<double, 3> camera_mount_to_body_translation{0.0, 0.0, 0.0};
};

/**
 * @brief Convert a wxyz quaternion to a normalized rotation matrix.
 */
inline Eigen::Matrix3d RotationMatrixFromWxyz(
    const std::array<double, 4>& rotation)
{
  Eigen::Quaterniond q(rotation[0], rotation[1], rotation[2], rotation[3]);
  if (!std::isfinite(q.norm()) || q.norm() < 1e-9)
  {
    q = Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q.toRotationMatrix();
}

/**
 * @brief Fixed transform from OpenCV camera frame C to camera mount frame M.
 *
 * C uses x right, y down, z forward. M has the same origin as C and uses the
 * public axis convention: x right, y forward, z up.
 */
inline Eigen::Matrix3d CameraToMountRotation()
{
  Eigen::Matrix3d rotation;
  rotation << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0;
  return rotation;
}

/**
 * @brief Build the internal C-to-B rotation from the configured M-to-B mount.
 */
inline Eigen::Matrix3d CameraToBodyRotationFromMountExtrinsic(
    const std::array<double, 4>& camera_mount_to_body_rotation)
{
  return RotationMatrixFromWxyz(camera_mount_to_body_rotation) *
         CameraToMountRotation();
}

/**
 * @brief Convert the published gimbal IMU attitude into tracker inertial axes.
 *
 * The `gimbal_quat` ABI is already expressed as the public body frame B
 * attitude (`x` right, `y` forward, `z` up). Board-specific IMU mounting
 * corrections belong before AHRS, e.g. in the BMI088 rotation config, so the
 * tracker must not add another fixed basis rotation here.
 */
inline Eigen::Matrix3d BodyToWorldRotationFromImu(const Eigen::Quaterniond& q)
{
  Eigen::Quaterniond normalized = q;
  if (!std::isfinite(normalized.norm()) || normalized.norm() < 1e-9)
  {
    normalized = Eigen::Quaterniond::Identity();
  }
  normalized.normalize();
  return normalized.toRotationMatrix();
}

/**
 * @brief Build an armor-to-frame rotation from public yaw and armor tilt.
 */
inline Eigen::Matrix3d ArmorRotationFromYaw(double yaw, double tilt)
{
  const auto sin_yaw = std::sin(yaw);
  const auto cos_yaw = std::cos(yaw);
  const auto sin_tilt = std::sin(tilt);
  const auto cos_tilt = std::cos(tilt);

  Eigen::Matrix3d rotation;
  rotation << -sin_yaw * cos_tilt, -cos_yaw, -sin_yaw * sin_tilt,
      cos_yaw * cos_tilt, -sin_yaw, cos_yaw * sin_tilt, -sin_tilt, 0.0,
      cos_tilt;
  return rotation;
}

/**
 * @brief Extract public-frame yaw from an armor rotation.
 */
inline double ArmorYawFromRotation(const Eigen::Matrix3d& rotation)
{
  return BearingYaw(rotation.col(0));
}

/**
 * @brief Solves detector armor corners into camera/body/world pose estimates.
 */
class Solver
{
 public:
  /**
   * @brief Construct a pose solver from camera and hand-eye parameters.
   */
  explicit Solver(const Config& config)
      : R_camera_to_body_(Eigen::Matrix3d::Identity()),
        t_camera_to_body_(Eigen::Vector3d::Zero()),
        R_body_to_world_(Eigen::Matrix3d::Identity())
  {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(
        config.camera_matrix.data());
    Eigen::Matrix<double, 1, 5> distort_coeffs;
    distort_coeffs << 0.0, 0.0, 0.0, 0.0, 0.0;
    camera_matrix_ = Mat3dToCv(camera_matrix);
    distort_coeffs_ = cv::Mat(1, 5, CV_64F);
    for (int col = 0; col < 5; ++col)
    {
      distort_coeffs_.at<double>(0, col) = distort_coeffs(0, col);
    }
    R_camera_to_body_ =
        CameraToBodyRotationFromMountExtrinsic(
            config.camera_mount_to_body_rotation);
    t_camera_to_body_ =
        Eigen::Vector3d(config.camera_mount_to_body_translation[0],
                        config.camera_mount_to_body_translation[1],
                        config.camera_mount_to_body_translation[2]);
  }

  /**
   * @brief Update the body-to-world orientation used by PnP postprocessing.
   */
  void SetRBodyToWorld(const Eigen::Quaterniond& q)
  {
    R_body_to_world_ = BodyToWorldRotationFromImu(q);
  }

  /**
   * @brief Solve a detector armor pose and fill its 3D fields in-place.
   */
  void Solve(Armor& armor) const
  {
    const auto& object_points =
        armor.type == ArmorType::BIG ? BigArmorPoints() : SmallArmorPoints();
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::solvePnP(object_points, armor.points, camera_matrix_, distort_coeffs_,
                 rvec, tvec, false, cv::SOLVEPNP_IPPE);

    Eigen::Vector3d xyz_in_camera(tvec[0], tvec[1], tvec[2]);
    armor.xyz_in_body = R_camera_to_body_ * xyz_in_camera + t_camera_to_body_;
    armor.xyz_in_world = R_body_to_world_ * armor.xyz_in_body;

    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    Eigen::Matrix3d R_armor2camera = CvMatToMat3d(rmat);
    Eigen::Matrix3d R_armor2body = R_camera_to_body_ * R_armor2camera;
    Eigen::Matrix3d R_armor2world = R_body_to_world_ * R_armor2body;
    armor.ypr_in_body = {ArmorYawFromRotation(R_armor2body), 0.0, 0.0};
    armor.ypr_in_world = {ArmorYawFromRotation(R_armor2world), 0.0, 0.0};
    armor.ypd_in_world = XyzToYpd(armor.xyz_in_world);

    const auto is_balance =
        armor.type == ArmorType::BIG &&
        (armor.name == ArmorName::THREE || armor.name == ArmorName::FOUR ||
         armor.name == ArmorName::FIVE);
    if (is_balance)
    {
      return;
    }
    OptimizeYaw(armor);
  }

  /**
   * @brief Project an armor face from world coordinates into image pixels.
   */
  std::vector<cv::Point2f> ReprojectArmor(const Eigen::Vector3d& xyz_in_world,
                                          double yaw, ArmorType type,
                                          ArmorName name) const
  {
    const auto tilt =
        name == ArmorName::OUTPOST ? kOutpostArmorTilt : 15.0 * kPi / 180.0;
    const Eigen::Matrix3d R_armor2world = ArmorRotationFromYaw(yaw, tilt);

    const Eigen::Vector3d& t_armor2world = xyz_in_world;
    Eigen::Matrix3d R_armor2camera =
        R_camera_to_body_.transpose() * R_body_to_world_.transpose() *
        R_armor2world;
    Eigen::Vector3d t_armor2camera =
        R_camera_to_body_.transpose() *
        (R_body_to_world_.transpose() * t_armor2world - t_camera_to_body_);

    cv::Vec3d rvec;
    cv::Rodrigues(Mat3dToCv(R_armor2camera), rvec);
    cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

    std::vector<cv::Point2f> image_points;
    const auto& object_points =
        type == ArmorType::BIG ? BigArmorPoints() : SmallArmorPoints();
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_,
                      image_points);
    return image_points;
  }

  /**
   * @brief 重投影 preview 用的装甲板可见贴纸四角。
   */
  std::vector<cv::Point2f> ReprojectPreviewArmor(
      const Eigen::Vector3d& xyz_in_world, double yaw, ArmorType type,
      ArmorName name) const
  {
    if (name == ArmorName::OUTPOST)
    {
      // preview 使用 Webots 可见贴纸几何，而不是 PnP 灯条面。
      const double preview_yaw = LimitRad(yaw + kPi);
      return ReprojectArmorObjectPoints(xyz_in_world, preview_yaw,
                                        -kOutpostArmorTilt,
                                        OutpostVisibleFacePointsMirroredX());
    }
    return ReprojectArmor(xyz_in_world, yaw, type, name);
  }

 private:
  /**
   * @brief 用给定几何模板重投影装甲板四角。
   */
  std::vector<cv::Point2f> ReprojectArmorObjectPoints(
      const Eigen::Vector3d& xyz_in_world, double yaw, double tilt,
      const std::vector<cv::Point3f>& object_points) const
  {
    const Eigen::Matrix3d R_armor2world = ArmorRotationFromYaw(yaw, tilt);
    const Eigen::Vector3d& t_armor2world = xyz_in_world;
    Eigen::Matrix3d R_armor2camera =
        R_camera_to_body_.transpose() * R_body_to_world_.transpose() *
        R_armor2world;
    Eigen::Vector3d t_armor2camera =
        R_camera_to_body_.transpose() *
        (R_body_to_world_.transpose() * t_armor2world - t_camera_to_body_);

    cv::Vec3d rvec;
    cv::Rodrigues(Mat3dToCv(R_armor2camera), rvec);
    cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

    std::vector<cv::Point2f> image_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_,
                      image_points);
    return image_points;
  }

  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_camera_to_body_;
  Eigen::Vector3d t_camera_to_body_;
  Eigen::Matrix3d R_body_to_world_;

  /**
   * @brief Return object points for large armor PnP.
   */
  static const std::vector<cv::Point3f>& BigArmorPoints()
  {
    static const std::vector<cv::Point3f> points{
        {0, kBigArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kBigArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kBigArmorWidth / 2.0, -kLightbarLength / 2.0},
        {0, kBigArmorWidth / 2.0, -kLightbarLength / 2.0}};
    return points;
  }

  /**
   * @brief Return object points for small armor PnP.
   */
  static const std::vector<cv::Point3f>& SmallArmorPoints()
  {
    static const std::vector<cv::Point3f> points{
        {0, kSmallArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kSmallArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kSmallArmorWidth / 2.0, -kLightbarLength / 2.0},
        {0, kSmallArmorWidth / 2.0, -kLightbarLength / 2.0}};
    return points;
  }

  /**
   * @brief 返回前哨站可见贴纸四角，水平顺序按当前 preview 约定镜像。
   */
  static const std::vector<cv::Point3f>& OutpostVisibleFacePointsMirroredX()
  {
    static const std::vector<cv::Point3f> points{
        {kOutpostVisibleFaceXOffset, kOutpostVisibleFaceWidth / 2.0,
         kOutpostVisibleFaceZOffset - kOutpostVisibleFaceHeight / 2.0},
        {kOutpostVisibleFaceXOffset, -kOutpostVisibleFaceWidth / 2.0,
         kOutpostVisibleFaceZOffset - kOutpostVisibleFaceHeight / 2.0},
        {kOutpostVisibleFaceXOffset, -kOutpostVisibleFaceWidth / 2.0,
         kOutpostVisibleFaceZOffset + kOutpostVisibleFaceHeight / 2.0},
        {kOutpostVisibleFaceXOffset, kOutpostVisibleFaceWidth / 2.0,
         kOutpostVisibleFaceZOffset + kOutpostVisibleFaceHeight / 2.0}};
    return points;
  }

  /**
   * @brief Refine armor yaw by minimizing reprojection error.
   */
  void OptimizeYaw(Armor& armor) const
  {
    const double body_yaw = BearingYaw(R_body_to_world_.col(1));
    constexpr double search_range = 140.0;
    auto yaw0 = LimitRad(body_yaw - search_range / 2.0 * kPi / 180.0);

    auto min_error = 1e10;
    auto best_yaw = armor.ypr_in_world[0];
    for (int i = 0; i < static_cast<int>(search_range); ++i)
    {
      const double yaw = LimitRad(yaw0 + i * kPi / 180.0);
      const auto error =
          ArmorReprojectionError(armor, yaw,
                                 (i - search_range / 2.0) * kPi / 180.0);
      if (error < min_error)
      {
        min_error = error;
        best_yaw = yaw;
      }
    }
    armor.yaw_raw = armor.ypr_in_world[0];
    armor.ypr_in_world[0] = best_yaw;
  }

  /**
   * @brief Return reprojection error for an armor under a candidate yaw.
   */
  double ArmorReprojectionError(const Armor& armor, double yaw,
                                const double&) const
  {
    auto image_points =
        ReprojectArmor(armor.xyz_in_world, yaw, armor.type, armor.name);
    auto error = 0.0;
    for (int i = 0; i < 4; ++i)
    {
      error += cv::norm(armor.points[i] - image_points[i]);
    }
    return error;
  }
};

/**
 * @brief Extended target state for a single tracked robot.
 */
class Target
{
 public:
  ArmorName name{};
  ArmorType armor_type{};
  ArmorPriority priority{};
  bool jumped{};
  int last_id{};

  int OutpostHeightPhase() const { return outpost_height_phase_; }

  /**
   * @brief Construct an empty target state.
   */
  Target() = default;

  /**
   * @brief Initialize a target from the first selected armor observation.
   */
  Target(const Armor& armor, std::chrono::steady_clock::time_point t,
         double radius, int armor_num, const Eigen::VectorXd& P0_dig,
         int outpost_initial_id = 0, int outpost_height_phase = 0,
         bool outpost_height_phase_valid = false,
         Eigen::Vector3d outpost_center_hint = Eigen::Vector3d::Zero(),
         bool outpost_center_hint_valid = false)
      : name(armor.name),
        armor_type(armor.type),
        priority(armor.priority),
        jumped(false),
        last_id(std::clamp(outpost_initial_id, 0, std::max(0, armor_num - 1))),
        armor_num_(armor_num),
        update_count_(0),
        is_converged_(false),
        t_(t),
        outpost_height_phase_(outpost_height_phase),
        outpost_height_phase_valid_(outpost_height_phase_valid)
  {
    const auto r = radius;
    const Eigen::VectorXd& xyz = armor.xyz_in_world;
    const Eigen::VectorXd& ypr = armor.ypr_in_world;
    auto center_x = xyz[0] - r * std::sin(ypr[0]);
    auto center_y = xyz[1] + r * std::cos(ypr[0]);
    auto center_z = xyz[2];
    if (name == ArmorName::OUTPOST && armor_num == 3)
    {
      center_z -= OutpostArmorHeightOffset(last_id, outpost_height_phase_);
      if (outpost_center_hint_valid)
      {
        center_x = outpost_center_hint.x();
        center_y = outpost_center_hint.y();
        center_z = outpost_center_hint.z();
      }
    }

    Eigen::VectorXd x0(11);
    x0 << center_x, 0.0, center_y, 0.0, center_z, 0.0, ypr[0], 0.0, r,
        0.0, 0.0;
    Eigen::MatrixXd P0 = P0_dig.asDiagonal();

    auto x_add = [](const Eigen::VectorXd& a,
                    const Eigen::VectorXd& b) -> Eigen::VectorXd
    {
      Eigen::VectorXd c = a + b;
      c[6] = LimitRad(c[6]);
      return c;
    };
    ekf_ = ExtendedKalmanFilter(x0, P0, x_add);
    if (UseOutpostHeightModel())
    {
      outpost_observed_z_[static_cast<std::size_t>(last_id)] = xyz[2];
      outpost_observed_z_valid_[static_cast<std::size_t>(last_id)] = true;
    }
  }

  /**
   * @brief Predict target state to a wall-clock time point.
   */
  void Predict(std::chrono::steady_clock::time_point t)
  {
    auto dt = DeltaTime(t, t_);
    Predict(dt);
    t_ = t;
  }

  /**
   * @brief Predict target state by an elapsed time in seconds.
   */
  void Predict(double dt)
  {
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;

    double v1;
    double v2;
    if (name == ArmorName::OUTPOST)
    {
      v1 = 0.05;
      v2 = 0.5;
    }
    else
    {
      v1 = 100.0;
      v2 = 400.0;
    }
    const auto a = dt * dt * dt * dt / 4.0;
    const auto b = dt * dt * dt / 2.0;
    const auto c = dt * dt;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
    Q(0, 0) = a * v1;
    Q(0, 1) = b * v1;
    Q(1, 0) = b * v1;
    Q(1, 1) = c * v1;
    Q(2, 2) = a * v1;
    Q(2, 3) = b * v1;
    Q(3, 2) = b * v1;
    Q(3, 3) = c * v1;
    Q(4, 4) = a * v1;
    Q(4, 5) = b * v1;
    Q(5, 4) = b * v1;
    Q(5, 5) = c * v1;
    Q(6, 6) = a * v2;
    Q(6, 7) = b * v2;
    Q(7, 6) = b * v2;
    Q(7, 7) = c * v2;

    auto f = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd
    {
      Eigen::VectorXd x_prior = F * x;
      x_prior[6] = LimitRad(x_prior[6]);
      return x_prior;
    };

    if (Converged() && name == ArmorName::OUTPOST && std::abs(ekf_.x[7]) > 2.0)
    {
      ekf_.x[7] = ekf_.x[7] > 0.0 ? 2.51 : -2.51;
    }
    ekf_.Predict(F, Q, f);
    ClampOutpostCenterVelocity();
  }

  /**
   * @brief Update target state with one matching armor observation.
   */
  void Update(const Armor& armor)
  {
    int id = 0;
    auto min_angle_error = 1e10;
    const std::vector<Eigen::Vector4d>& xyza_list = ArmorXyzaList();
    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; ++i)
    {
      xyza_i_list.push_back({xyza_list[i], i});
    }
    std::sort(
        xyza_i_list.begin(), xyza_i_list.end(),
        [](const std::pair<Eigen::Vector4d, int>& a,
           const std::pair<Eigen::Vector4d, int>& b)
        {
          Eigen::Vector3d ypd1 = XyzToYpd(a.first.head(3));
          Eigen::Vector3d ypd2 = XyzToYpd(b.first.head(3));
          return ypd1[2] < ypd2[2];
        });

    const int candidate_count =
        std::min(3, static_cast<int>(xyza_i_list.size()));
    if (UseOutpostHeightModel())
    {
      id = MatchOutpostArmor(armor, xyza_i_list);
    }
    else
    {
      for (int i = 0; i < candidate_count; ++i)
      {
        const auto& xyza = xyza_i_list[i].first;
        Eigen::Vector3d ypd = XyzToYpd(xyza.head(3));
        auto angle_error =
            std::abs(LimitRad(armor.ypr_in_world[0] - xyza[3])) +
            std::abs(LimitRad(armor.ypd_in_world[0] - ypd[0]));
        if (std::abs(angle_error) < std::abs(min_angle_error))
        {
          id = xyza_i_list[i].second;
          min_angle_error = angle_error;
        }
      }
    }

    if (id != 0)
    {
      jumped = true;
    }
    UpdateOutpostHeightPhase(armor, id);
    last_id = id;
    ++update_count_;
    UpdateYpda(armor, id);
    ClampOutpostCenterVelocity();
  }

  /**
   * @brief Return a copy of the EKF state vector.
   */
  Eigen::VectorXd EkfX() const { return ekf_.x; }

  /**
   * @brief Return the EKF object for debug metrics and health checks.
   */
  const ExtendedKalmanFilter& Ekf() const { return ekf_; }

  /**
   * @brief Return the modeled armor face centers and yaws in inertial W frame.
   */
  std::vector<Eigen::Vector4d> ArmorXyzaList() const
  {
    std::vector<Eigen::Vector4d> list;
    list.reserve(static_cast<std::size_t>(armor_num_));
    for (int i = 0; i < armor_num_; ++i)
    {
      auto angle = LimitRad(ekf_.x[6] + i * 2.0 * kPi / armor_num_);
      Eigen::Vector3d xyz = HArmorXyz(ekf_.x, i);
      list.push_back({xyz[0], xyz[1], xyz[2], angle});
    }
    return list;
  }

  std::vector<Eigen::Vector4d> ArmorXyzaListForOutput() const
  {
    if (!UseOutpostHeightModel())
    {
      return ArmorXyzaList();
    }

    std::vector<Eigen::Vector4d> list;
    list.reserve(static_cast<std::size_t>(armor_num_));
    const double center_z = CenterWorldForOutput().z();
    for (int i = 0; i < armor_num_; ++i)
    {
      auto angle = LimitRad(ekf_.x[6] + i * 2.0 * kPi / armor_num_);
      Eigen::Vector3d xyz = HArmorXyz(ekf_.x, i);
      xyz.z() = center_z + OutpostArmorHeightOffset(i, outpost_height_phase_);
      list.push_back({xyz[0], xyz[1], xyz[2], angle});
    }
    return list;
  }

  Eigen::Vector3d CenterWorldForOutput() const
  {
    Eigen::Vector3d center{ekf_.x[0], ekf_.x[2], ekf_.x[4]};
    return center;
  }

  Eigen::Vector3d VelocityWorldForOutput() const
  {
    Eigen::Vector3d velocity{ekf_.x[1], ekf_.x[3], ekf_.x[5]};
    if (UseOutpostHeightModel())
    {
      velocity.z() = 0.0;
    }
    return velocity;
  }

  double DzForOutput() const
  {
    return UseOutpostHeightModel() ? kOutpostArmorHeightStep : ekf_.x[10];
  }

  /**
   * @brief Check whether estimated geometry has left the accepted radius range.
   */
  bool Diverged() const
  {
    auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
    auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
    return !(r_ok && l_ok);
  }

  /**
   * @brief Update and return the target convergence flag.
   */
  bool Converged()
  {
    if (name != ArmorName::OUTPOST && update_count_ > 3 && !Diverged())
    {
      is_converged_ = true;
    }
    if (name == ArmorName::OUTPOST && update_count_ > 10 && !Diverged())
    {
      is_converged_ = true;
    }
    return is_converged_;
  }

 private:
  int armor_num_ = 4;
  int update_count_ = 0;
  bool is_converged_ = false;
  ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_{};
  int outpost_height_phase_ = 0;
  bool outpost_height_phase_valid_ = false;
  std::array<double, 3> outpost_observed_z_{};
  std::array<bool, 3> outpost_observed_z_valid_{};

  bool UseOutpostHeightModel() const
  {
    return name == ArmorName::OUTPOST && armor_num_ == 3;
  }
  int MatchOutpostArmor(
      const Armor& armor,
      const std::vector<std::pair<Eigen::Vector4d, int>>& xyza_i_list) const
  {
    int best_id = 0;
    double best_error = std::numeric_limits<double>::infinity();
    for (const auto& [xyza, candidate_id] : xyza_i_list)
    {
      const Eigen::Vector3d ypd = XyzToYpd(xyza.head(3));
      double error =
          std::abs(LimitRad(armor.ypr_in_world[0] - xyza[3])) +
          std::abs(LimitRad(armor.ypd_in_world[0] - ypd[0]));
      if (outpost_height_phase_valid_)
      {
        error += 2.0 * std::abs(armor.xyz_in_world.z() - xyza[2]);
      }
      if (error < best_error)
      {
        best_error = error;
        best_id = candidate_id;
      }
    }

    return best_id;
  }

  void ClampOutpostCenterVelocity()
  {
    if (!UseOutpostHeightModel())
    {
      return;
    }

    ekf_.x[1] = 0.0;
    ekf_.x[3] = 0.0;
    ekf_.x[5] = 0.0;
  }

  void UpdateOutpostHeightPhase(const Armor& armor, int id)
  {
    if (!UseOutpostHeightModel() || id < 0 || id >= 3)
    {
      return;
    }

    if (outpost_height_phase_valid_)
    {
      return;
    }

    const bool has_previous =
        last_id >= 0 && last_id < 3 &&
        outpost_observed_z_valid_[static_cast<std::size_t>(last_id)];
    const double previous_z =
        has_previous ? outpost_observed_z_[static_cast<std::size_t>(last_id)]
                     : 0.0;

    if (id != last_id && has_previous)
    {
      constexpr double kHeightRelationThreshold = 0.04;
      constexpr double kTwoHeightStepTolerance = 0.05;
      const double measured_delta = armor.xyz_in_world.z() - previous_z;
      const double measured_delta_abs = std::abs(measured_delta);
      const bool is_two_step =
          std::abs(measured_delta_abs - 2.0 * kOutpostArmorHeightStep) <=
          kTwoHeightStepTolerance;
      if (measured_delta_abs >= kHeightRelationThreshold && is_two_step)
      {
        const int measured_sign = SignNonZero(measured_delta);
        for (int phase = 0; phase < 3; ++phase)
        {
          const double candidate_delta =
              OutpostArmorHeightOffset(id, phase) -
              OutpostArmorHeightOffset(last_id, phase);
          const bool candidate_is_two_step =
              std::abs(std::abs(candidate_delta) -
                       2.0 * kOutpostArmorHeightStep) <= 1e-6;
          if (candidate_is_two_step &&
              SignNonZero(candidate_delta) == measured_sign)
          {
            if (phase != outpost_height_phase_)
            {
              outpost_height_phase_ = phase;
              ekf_.x[4] =
                  armor.xyz_in_world.z() -
                  OutpostArmorHeightOffset(id, outpost_height_phase_);
              ekf_.x[5] = 0.0;
            }
            outpost_height_phase_valid_ = true;
            break;
          }
        }
      }
    }

    constexpr double kZUpdateAlpha = 0.35;
    auto& observed_z = outpost_observed_z_[static_cast<std::size_t>(id)];
    auto& observed_valid =
        outpost_observed_z_valid_[static_cast<std::size_t>(id)];
    if (observed_valid)
    {
      observed_z =
          (1.0 - kZUpdateAlpha) * observed_z + kZUpdateAlpha * armor.xyz_in_world.z();
    }
    else
    {
      observed_z = armor.xyz_in_world.z();
      observed_valid = true;
    }
  }

  /**
   * @brief Update EKF with yaw, elevation, distance, and armor yaw measurement.
   */
  void UpdateYpda(const Armor& armor, int id)
  {
    const double center_x_before = ekf_.x[0];
    const double center_y_before = ekf_.x[2];
    const double center_z_before = ekf_.x[4];

    Eigen::MatrixXd H = HJacobian(ekf_.x, id);
    auto center_yaw = BearingYaw(armor.xyz_in_world);
    const Eigen::Vector3d center_before{center_x_before, center_y_before,
                                        center_z_before};
    const double observed_armor_yaw = armor.ypr_in_world[0];
    auto delta_angle = LimitRad(observed_armor_yaw - center_yaw);
    const double side_view = std::abs(delta_angle);
    const bool side_observation = side_view > 0.55;
    Eigen::VectorXd R_dig(4);
    if (name == ArmorName::OUTPOST)
    {
      const double ypd_noise =
          side_observation ? 25.0 : 0.02 + 0.2 * side_view;
      const double distance_noise = side_observation ? 400.0 : 2.0;
      R_dig << ypd_noise, ypd_noise, distance_noise, 3e-2;
    }
    else
    {
      R_dig << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1.0) + 1.0,
          std::log(std::abs(armor.ypd_in_world[2]) + 1.0) / 200.0 + 9e-2;
    }
    Eigen::MatrixXd R = R_dig.asDiagonal();

    auto h = [&](const Eigen::VectorXd& x) -> Eigen::Vector4d
    {
      Eigen::VectorXd xyz = HArmorXyz(x, id);
      Eigen::VectorXd ypd = XyzToYpd(xyz);
      auto angle = LimitRad(x[6] + id * 2.0 * kPi / armor_num_);
      return {ypd[0], ypd[1], ypd[2], angle};
    };
    auto z_subtract = [](const Eigen::VectorXd& a,
                         const Eigen::VectorXd& b) -> Eigen::VectorXd
    {
      Eigen::VectorXd c = a - b;
      c[0] = LimitRad(c[0]);
      c[1] = LimitRad(c[1]);
      c[3] = LimitRad(c[3]);
      return c;
    };

    const Eigen::VectorXd& ypd = armor.ypd_in_world;
    Eigen::VectorXd z(4);
    z << ypd[0], ypd[1], ypd[2], observed_armor_yaw;
    ekf_.Update(z, H, R, h, z_subtract);
    if (UseOutpostHeightModel())
    {
      ekf_.x[0] = center_x_before;
      ekf_.x[1] = 0.0;
      ekf_.x[2] = center_y_before;
      ekf_.x[3] = 0.0;
      ekf_.x[4] = center_z_before;
      ekf_.x[5] = 0.0;
    }
  }

  /**
   * @brief Calculate one modeled armor center from an EKF state.
   */
  Eigen::Vector3d HArmorXyz(const Eigen::VectorXd& x, int id) const
  {
    auto angle = LimitRad(x[6] + id * 2.0 * kPi / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
    auto r = use_l_h ? x[8] + x[9] : x[8];
    auto armor_x = x[0] + r * std::sin(angle);
    auto armor_y = x[2] - r * std::cos(angle);
    auto armor_z = use_l_h ? x[4] + x[10] : x[4];
    if (UseOutpostHeightModel())
    {
      armor_z = x[4] + OutpostArmorHeightOffset(id, outpost_height_phase_);
    }
    return {armor_x, armor_y, armor_z};
  }

  /**
   * @brief Return measurement Jacobian for one modeled armor face.
   */
  Eigen::MatrixXd HJacobian(const Eigen::VectorXd& x, int id) const
  {
    auto angle = LimitRad(x[6] + id * 2.0 * kPi / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
    auto r = use_l_h ? x[8] + x[9] : x[8];
    auto dx_da = r * std::cos(angle);
    auto dy_da = r * std::sin(angle);
    auto dx_dr = std::sin(angle);
    auto dy_dr = -std::cos(angle);
    auto dx_dl = use_l_h ? std::sin(angle) : 0.0;
    auto dy_dl = use_l_h ? -std::cos(angle) : 0.0;
    auto dz_dh = use_l_h ? 1.0 : 0.0;

    Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, 11);
    H_armor_xyza(0, 0) = 1.0;
    H_armor_xyza(0, 6) = dx_da;
    H_armor_xyza(0, 8) = dx_dr;
    H_armor_xyza(0, 9) = dx_dl;
    H_armor_xyza(1, 2) = 1.0;
    H_armor_xyza(1, 6) = dy_da;
    H_armor_xyza(1, 8) = dy_dr;
    H_armor_xyza(1, 9) = dy_dl;
    H_armor_xyza(2, 4) = 1.0;
    H_armor_xyza(2, 10) = dz_dh;
    H_armor_xyza(3, 6) = 1.0;

    Eigen::VectorXd armor_xyz = HArmorXyz(x, id);
    Eigen::MatrixXd H_armor_ypd = XyzToYpdJacobian(armor_xyz);
    Eigen::MatrixXd H_armor_ypda(4, 4);
    H_armor_ypda << H_armor_ypd(0, 0), H_armor_ypd(0, 1),
        H_armor_ypd(0, 2), 0, H_armor_ypd(1, 0), H_armor_ypd(1, 1),
        H_armor_ypd(1, 2), 0, H_armor_ypd(2, 0), H_armor_ypd(2, 1),
        H_armor_ypd(2, 2), 0, 0, 0, 0, 1;
    return H_armor_ypda * H_armor_xyza;
  }
};

/**
 * @brief Target state machine operating on solved armor observations.
 */
class Tracker
{
 public:
  /**
   * @brief Construct a tracker over a pose solver.
   */
  Tracker(const Config& config, Solver& solver)
      : solver_(solver),
        require_target_tag_(config.require_target_tag),
        target_name_(ArmorNameFromDetectorNumber(config.target_tag_id)),
        min_detect_count_(config.min_detect_count),
        outpost_max_temp_lost_count_(config.outpost_max_temp_lost_count),
        normal_temp_lost_count_(config.max_temp_lost_count),
        target_select_(config.target_select),
        state_("lost")
  {
    for (std::size_t i = 0; i < tracks_.size(); ++i)
    {
      tracks_[i].name = static_cast<ArmorName>(i);
    }
  }

  /**
   * @brief Return the current state-machine state string.
   */
  const std::string& State() const { return state_; }

  /**
   * @brief Snapshot of one active target slot for preview and debug output.
   */
  struct TrackSnapshot
  {
    std::string state{"lost"};
    bool selected{false};
    double score{-std::numeric_limits<double>::infinity()};
    Target target{};
  };

  /**
   * @brief Return active target slots after the latest tracking step.
   */
  std::vector<TrackSnapshot> Snapshots() const
  {
    std::vector<TrackSnapshot> snapshots;
    snapshots.reserve(tracks_.size());
    for (std::size_t i = 0; i < tracks_.size(); ++i)
    {
      if (!Selectable(tracks_[i]))
      {
        continue;
      }
      snapshots.push_back({tracks_[i].state,
                           static_cast<int>(i) == selected_index_,
                           tracks_[i].score, tracks_[i].target});
    }
    return snapshots;
  }

  /**
   * @brief Run one tracking step over candidate armors.
   */
  std::list<Target> Track(std::list<Armor>& armors,
                          std::chrono::steady_clock::time_point t)
  {
    armors.sort(
        [](const Armor& a, const Armor& b)
        {
          cv::Point2f img_center(1440.0F / 2.0F, 1080.0F / 2.0F);
          auto distance_1 = cv::norm(a.center - img_center);
          auto distance_2 = cv::norm(b.center - img_center);
          return distance_1 < distance_2;
        });
    armors.sort([](const Armor& a, const Armor& b) {
      return a.priority < b.priority;
    });

    std::array<std::list<Armor>, kTrackSlotCount> armors_by_name;
    for (const auto& armor : armors)
    {
      const int index = TrackSlotIndex(armor.name);
      if (index >= 0)
      {
        armors_by_name[static_cast<std::size_t>(index)].push_back(armor);
      }
    }

    for (std::size_t i = 0; i < tracks_.size(); ++i)
    {
      UpdateSlot(tracks_[i], armors_by_name[i], t);
    }

    const int selected = SelectTrack();
    if (selected < 0)
    {
      state_ = "lost";
      selected_index_ = -1;
      return {};
    }

    selected_index_ = selected;
    const auto& slot = tracks_[static_cast<std::size_t>(selected_index_)];
    state_ = slot.state;
    return {slot.target};
  }

 private:
  /**
   * @brief Persistent state and selection metrics for one vehicle number.
   */
  struct TrackSlot
  {
    ArmorName name{ArmorName::NOT_ARMOR};
    std::string state{"lost"};
    Target target{};
    bool initialized{false};
    int detect_count{0};
    int temp_lost_count{0};
    int max_temp_lost_count{0};
    double score{-std::numeric_limits<double>::infinity()};
    double observed_count_lpf{0.0};
    double hittable_area{0.0};
    double gimbal_angle_error{1.0};
    bool has_timestamp{false};
    std::chrono::steady_clock::time_point last_timestamp{};
    bool outpost_center_hint_valid{false};
    Eigen::Vector3d outpost_center_hint{Eigen::Vector3d::Zero()};
  };

  Solver& solver_;
  bool require_target_tag_;
  ArmorName target_name_;
  int min_detect_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  Config::TargetSelectConfig target_select_;
  std::string state_;
  std::array<TrackSlot, kTrackSlotCount> tracks_{};
  int selected_index_ = -1;

  /**
   * @brief Clamp a scalar into [0, 1].
   */
  static double Clamp01(double value)
  {
    return std::clamp(value, 0.0, 1.0);
  }

  /**
   * @brief Return whether a slot can be selected for the public target output.
   */
  static bool Selectable(const TrackSlot& slot)
  {
    return slot.initialized && slot.state != "lost" &&
           std::isfinite(slot.score);
  }

  /**
   * @brief Update current-frame observation metrics for target selection.
   */
  static void UpdateObservationMetrics(TrackSlot& slot,
                                       const std::list<Armor>& armors)
  {
    slot.observed_count_lpf =
        0.8 * slot.observed_count_lpf +
        0.2 * static_cast<double>(armors.size());
    slot.hittable_area = 0.0;
    auto min_angle_error = std::numeric_limits<double>::infinity();
    for (const auto& armor : armors)
    {
      if (armor.points.size() == 4)
      {
        slot.hittable_area += std::abs(cv::contourArea(armor.points));
      }
      const auto dx = static_cast<double>(armor.center_norm.x) - 0.5;
      const auto dy = static_cast<double>(armor.center_norm.y) - 0.5;
      min_angle_error = std::min(min_angle_error, std::hypot(dx, dy));
    }
    if (std::isfinite(min_angle_error))
    {
      slot.gimbal_angle_error = min_angle_error;
    }
  }

  /**
   * @brief Return whether the slot's EKF has failed recent innovation gates.
   */
  static bool RecentNisFailed(const TrackSlot& slot)
  {
    const auto& ekf = slot.target.Ekf();
    return std::accumulate(ekf.recent_nis_failures.begin(),
                           ekf.recent_nis_failures.end(), 0) >=
           0.4 * static_cast<double>(ekf.window_size);
  }

  /**
   * @brief Return whether the retained EKF state is no longer usable.
   */
  static bool TargetHealthFailed(const TrackSlot& slot)
  {
    if (!slot.initialized)
    {
      return false;
    }
    if (slot.target.Diverged())
    {
      return true;
    }
    // Outpost keeps the fixed tower center and height phase across short NIS failures; otherwise side PnP noise can rebuild it with a wrong phase.
    if (slot.target.name == ArmorName::OUTPOST)
    {
      return false;
    }
    return RecentNisFailed(slot);
  }
  /**
   * @brief Choose the initial outpost face id from the retained center hint.
   */
  static int ChooseOutpostInitialFaceId(const TrackSlot& slot, const Armor& armor)
  {
    if (!slot.outpost_center_hint_valid)
    {
      return 0;
    }
    Eigen::Vector3d armor_to_center = slot.outpost_center_hint - armor.xyz_in_world;
    armor_to_center.z() = 0.0;
    if (armor_to_center.head<2>().norm() < 1e-6)
    {
      return 0;
    }
    const double observed_face_yaw = BearingYaw(armor_to_center);
    int best_id = 0;
    double best_error = std::numeric_limits<double>::infinity();
    for (int id = 0; id < 3; ++id)
    {
      const double candidate_yaw =
          LimitRad(armor.ypr_in_world[0] + id * 2.0 * kPi / 3.0);
      const double error = std::abs(LimitRad(observed_face_yaw - candidate_yaw));
      if (error < best_error)
      {
        best_error = error;
        best_id = id;
      }
    }
    return best_id;
  }
  static std::pair<int, bool> ChooseOutpostInitialHeightPhase(
      const TrackSlot& slot, const Armor& armor, int initial_id)
  {
    if (!slot.outpost_center_hint_valid)
    {
      return {0, true};
    }
    int best_phase = 0;
    double best_error = std::numeric_limits<double>::infinity();
    for (int phase = 0; phase < 3; ++phase)
    {
      const double center_z =
          armor.xyz_in_world.z() - OutpostArmorHeightOffset(initial_id, phase);
      const double error = std::abs(center_z - slot.outpost_center_hint.z());
      if (error < best_error)
      {
        best_error = error;
        best_phase = phase;
      }
    }
    constexpr double kCenterHintGate = 0.06;
    return {best_phase, best_error <= kCenterHintGate};
  }
  /**
   * @brief Clear the retained target state after an EKF health failure.
   *
   * The slot identity, timestamp, and current-frame observation metrics are
   * preserved; only the target/EKF and state-machine counters are discarded.
   */
  static void ResetTargetSlot(TrackSlot& slot)
  {
    slot.target = Target{};
    slot.initialized = false;
    slot.state = "lost";
    slot.detect_count = 0;
    slot.temp_lost_count = 0;
    slot.max_temp_lost_count = 0;
    slot.score = -std::numeric_limits<double>::infinity();
  }

  /**
   * @brief Update one vehicle-number slot for the current frame.
   */
  void UpdateSlot(TrackSlot& slot, std::list<Armor>& armors,
                  std::chrono::steady_clock::time_point t)
  {
    UpdateObservationMetrics(slot, armors);

    if (slot.has_timestamp)
    {
      const auto dt = DeltaTime(t, slot.last_timestamp);
      if (slot.state != "lost" && dt > 0.1)
      {
        slot.state = "lost";
      }
    }
    slot.last_timestamp = t;
    slot.has_timestamp = true;

    if (TargetHealthFailed(slot))
    {
      ResetTargetSlot(slot);
    }

    bool found = false;
    if (!slot.initialized)
    {
      found = SetTarget(slot, armors, t);
    }
    else if (slot.state == "lost" && armors.empty())
    {
      found = false;
    }
    else
    {
      found = UpdateTarget(slot, armors, t);
    }

    if (TargetHealthFailed(slot))
    {
      ResetTargetSlot(slot);
      found = SetTarget(slot, armors, t);
    }
    StateMachine(slot, found);
    if (slot.state != "lost" && TargetHealthFailed(slot))
    {
      ResetTargetSlot(slot);
    }
    if (slot.initialized && slot.state != "lost" &&
        slot.target.name == ArmorName::OUTPOST)
    {
      slot.outpost_center_hint = slot.target.CenterWorldForOutput();
      slot.outpost_center_hint_valid = true;
    }
    slot.score = ScoreSlot(slot, target_select_);
  }

  /**
   * @brief Calculate the current target-selection score for one slot.
   */
  static double ScoreSlot(const TrackSlot& slot,
                          const Config::TargetSelectConfig& cfg)
  {
    if (!slot.initialized || slot.state == "lost")
    {
      return -std::numeric_limits<double>::infinity();
    }

    const Eigen::VectorXd x = slot.target.EkfX();
    const double distance = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);
    const double distance_span = std::max(cfg.distance_span_m, 1e-6);
    const double area_norm = std::max(cfg.area_norm_px, 1e-6);
    const double count_norm = std::max(cfg.observed_count_norm, 1e-6);
    const double spin_norm = std::max(cfg.max_spin_rad_s, 1e-6);
    const double angle_norm = std::max(cfg.max_angle_norm, 1e-6);
    const double distance_score =
        Clamp01((cfg.max_distance_m - distance) / distance_span);
    const double area_score = Clamp01(slot.hittable_area / area_norm);
    const double count_score = Clamp01(slot.observed_count_lpf / count_norm);
    const double spin_score = Clamp01(1.0 - std::abs(x[7]) / spin_norm);
    const double angle_score = Clamp01(1.0 - slot.gimbal_angle_error / angle_norm);

    double state_scale = 1.0;
    if (slot.state == "detecting")
    {
      state_scale = cfg.detecting_scale;
    }
    else if (slot.state == "temp_lost")
    {
      state_scale = cfg.temp_lost_scale;
    }

    return state_scale *
           (cfg.observed_count_weight * count_score +
            cfg.distance_weight * distance_score + cfg.area_weight * area_score +
            cfg.spin_weight * spin_score + cfg.angle_weight * angle_score);
  }

  /**
   * @brief Select the output target with a small hysteresis margin.
   */
  int SelectTrack() const
  {
    if (require_target_tag_)
    {
      const int required_index = TrackSlotIndex(target_name_);
      if (required_index >= 0 &&
          Selectable(tracks_[static_cast<std::size_t>(required_index)]))
      {
        return required_index;
      }
      return -1;
    }

    int best_index = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < tracks_.size(); ++i)
    {
      if (!Selectable(tracks_[i]) || tracks_[i].score <= best_score)
      {
        continue;
      }
      best_score = tracks_[i].score;
      best_index = static_cast<int>(i);
    }
    if (best_index < 0)
    {
      return -1;
    }

    if (selected_index_ >= 0 &&
        selected_index_ < static_cast<int>(tracks_.size()) &&
        Selectable(tracks_[static_cast<std::size_t>(selected_index_)]))
    {
      const auto& selected = tracks_[static_cast<std::size_t>(selected_index_)];
      if (best_index != selected_index_ &&
          tracks_[static_cast<std::size_t>(best_index)].score <=
              selected.score + target_select_.switch_margin)
      {
        return selected_index_;
      }
    }
    return best_index;
  }

  /**
   * @brief Advance the lost/detecting/tracking/temp-lost state machine.
   */
  void StateMachine(TrackSlot& slot, bool found)
  {
    if (slot.state == "lost")
    {
      if (!found)
      {
        return;
      }
      slot.state = "detecting";
      slot.detect_count = 1;
    }
    else if (slot.state == "detecting")
    {
      if (found)
      {
        ++slot.detect_count;
        if (slot.detect_count >= min_detect_count_)
        {
          slot.state = "tracking";
        }
      }
      else
      {
        slot.detect_count = 0;
        slot.state = "lost";
      }
    }
    else if (slot.state == "tracking")
    {
      if (found)
      {
        return;
      }
      slot.temp_lost_count = 1;
      slot.state = "temp_lost";
    }
    else if (slot.state == "switching")
    {
      if (found)
      {
        slot.state = "detecting";
      }
      else
      {
        ++slot.temp_lost_count;
        if (slot.temp_lost_count > 200)
        {
          slot.state = "lost";
        }
      }
    }
    else if (slot.state == "temp_lost")
    {
      if (found)
      {
        slot.state = "tracking";
      }
      else
      {
        ++slot.temp_lost_count;
        if (slot.target.name == ArmorName::OUTPOST)
        {
          slot.max_temp_lost_count = outpost_max_temp_lost_count_;
        }
        else
        {
          slot.max_temp_lost_count = normal_temp_lost_count_;
        }
        if (slot.temp_lost_count > slot.max_temp_lost_count)
        {
          slot.state = "lost";
        }
      }
    }
  }

  /**
   * @brief Initialize target state from the best sorted armor candidate.
   */
  bool SetTarget(TrackSlot& slot, std::list<Armor>& armors,
                 std::chrono::steady_clock::time_point t)
  {
    if (armors.empty())
    {
      return false;
    }
    auto& armor = armors.front();
    solver_.Solve(armor);

    const auto is_balance =
        armor.type == ArmorType::BIG &&
        (armor.name == ArmorName::THREE || armor.name == ArmorName::FOUR ||
         armor.name == ArmorName::FIVE);

    Eigen::VectorXd P0_dig(11);
    if (is_balance)
    {
      P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
      slot.target = Target(armor, t, 0.2, 2, P0_dig);
    }
    else if (armor.name == ArmorName::OUTPOST)
    {
      P0_dig << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0;
      const int initial_id = ChooseOutpostInitialFaceId(slot, armor);
      const auto [height_phase, height_phase_valid] =
          ChooseOutpostInitialHeightPhase(slot, armor, initial_id);
      slot.target = Target(armor, t, kOutpostArmorRadius, 3, P0_dig, initial_id,
                           height_phase, height_phase_valid,
                           slot.outpost_center_hint,
                           height_phase_valid &&
                               slot.outpost_center_hint_valid);
    }
    else if (armor.name == ArmorName::BASE)
    {
      P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
      slot.target = Target(armor, t, 0.3205, 3, P0_dig);
    }
    else
    {
      P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
      slot.target = Target(armor, t, 0.2, 4, P0_dig);
    }
    slot.initialized = true;
    slot.max_temp_lost_count =
        armor.name == ArmorName::OUTPOST ? outpost_max_temp_lost_count_
                                         : normal_temp_lost_count_;
    return true;
  }

  /**
   * @brief Predict and update the current target from matching armors.
   */
  bool UpdateTarget(TrackSlot& slot, std::list<Armor>& armors,
                    std::chrono::steady_clock::time_point t)
  {
    slot.target.Predict(t);
    int found_count = 0;
    for (const auto& armor : armors)
    {
      if (armor.name != slot.target.name || armor.type != slot.target.armor_type)
      {
        continue;
      }
      ++found_count;
    }
    if (found_count == 0)
    {
      return false;
    }

    for (auto& armor : armors)
    {
      if (armor.name != slot.target.name || armor.type != slot.target.armor_type)
      {
        continue;
      }
      solver_.Solve(armor);
      slot.target.Update(armor);
    }
    return true;
  }
};

}  // namespace armor_tracker_detail
