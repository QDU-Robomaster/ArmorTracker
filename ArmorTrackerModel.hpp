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
  Eigen::Vector3d xyz_in_gimbal = Eigen::Vector3d::Zero();
  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypr_in_gimbal = Eigen::Vector3d::Zero();
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
  bool require_target_tag = false;
  int target_tag_id = -1;
  int min_detect_count = 2;
  int max_temp_lost_count = 15;
  int outpost_max_temp_lost_count = 75;
  int output_frame = 1;
  std::array<double, 9> camera_matrix{
      1164.3428599490444, 0.0, 366.6782312546237,
      0.0, 1164.335053894998, 270.30936434613865,
      0.0, 0.0, 1.0};
};

/**
 * @brief Solves detector armor corners into camera/gimbal/world pose estimates.
 */
class Solver
{
 public:
  /**
   * @brief Construct a pose solver from camera and output-frame parameters.
   */
  explicit Solver(const Config& config)
      : R_gimbal2imubody_(Eigen::Matrix3d::Identity()),
        R_camera2gimbal_(Eigen::Matrix3d::Identity()),
        t_camera2gimbal_(Eigen::Vector3d::Zero()),
        R_gimbal2world_(Eigen::Matrix3d::Identity())
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
    if (config.output_frame == 0)
    {
      R_camera2gimbal_ = Eigen::Matrix3d::Identity();
    }
    else
    {
      R_camera2gimbal_ << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0,
          0.0;
    }
  }

  /**
   * @brief Update the gimbal-to-world orientation used by PnP postprocessing.
   */
  void SetRGimbal2World(const Eigen::Quaterniond& q)
  {
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    R_gimbal2world_ =
        R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
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
    armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
    armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    Eigen::Matrix3d R_armor2camera = CvMatToMat3d(rmat);
    Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
    Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
    armor.ypr_in_gimbal = Eulers(R_armor2gimbal, 2, 1, 0);
    armor.ypr_in_world = Eulers(R_armor2world, 2, 1, 0);
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
    const auto sin_yaw = std::sin(yaw);
    const auto cos_yaw = std::cos(yaw);
    const auto pitch =
        name == ArmorName::OUTPOST ? -15.0 * kPi / 180.0 : 15.0 * kPi / 180.0;
    const auto sin_pitch = std::sin(pitch);
    const auto cos_pitch = std::cos(pitch);

    Eigen::Matrix3d R_armor2world;
    R_armor2world << cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch,
        sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch, -sin_pitch, 0.0,
        cos_pitch;

    const Eigen::Vector3d& t_armor2world = xyz_in_world;
    Eigen::Matrix3d R_armor2camera =
        R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() *
        R_armor2world;
    Eigen::Vector3d t_armor2camera =
        R_camera2gimbal_.transpose() *
        (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

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

 private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

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
   * @brief Refine armor yaw by minimizing reprojection error.
   */
  void OptimizeYaw(Armor& armor) const
  {
    Eigen::Vector3d gimbal_ypr = Eulers(R_gimbal2world_, 2, 1, 0);
    constexpr double search_range = 140.0;
    auto yaw0 = LimitRad(gimbal_ypr[0] - search_range / 2.0 * kPi / 180.0);

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

  /**
   * @brief Construct an empty target state.
   */
  Target() = default;

  /**
   * @brief Initialize a target from the first selected armor observation.
   */
  Target(const Armor& armor, std::chrono::steady_clock::time_point t,
         double radius, int armor_num, const Eigen::VectorXd& P0_dig)
      : name(armor.name),
        armor_type(armor.type),
        priority(armor.priority),
        jumped(false),
        last_id(0),
        armor_num_(armor_num),
        update_count_(0),
        is_converged_(false),
        t_(t)
  {
    const auto r = radius;
    const Eigen::VectorXd& xyz = armor.xyz_in_world;
    const Eigen::VectorXd& ypr = armor.ypr_in_world;
    const auto center_x = xyz[0] + r * std::cos(ypr[0]);
    const auto center_y = xyz[1] + r * std::sin(ypr[0]);
    const auto center_z = xyz[2];

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
      v1 = 10.0;
      v2 = 0.1;
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

    if (id != 0)
    {
      jumped = true;
    }
    last_id = id;
    ++update_count_;
    UpdateYpda(armor, id);
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
   * @brief Return the modeled armor face centers and yaws in world frame.
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

  /**
   * @brief Update EKF with yaw, pitch, distance, and armor yaw measurement.
   */
  void UpdateYpda(const Armor& armor, int id)
  {
    Eigen::MatrixXd H = HJacobian(ekf_.x, id);
    auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
    auto delta_angle = LimitRad(armor.ypr_in_world[0] - center_yaw);
    Eigen::VectorXd R_dig(4);
    R_dig << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1.0) + 1.0,
        std::log(std::abs(armor.ypd_in_world[2]) + 1.0) / 200.0 + 9e-2;
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
    const Eigen::VectorXd& ypr = armor.ypr_in_world;
    Eigen::VectorXd z(4);
    z << ypd[0], ypd[1], ypd[2], ypr[0];
    ekf_.Update(z, H, R, h, z_subtract);
  }

  /**
   * @brief Calculate one modeled armor center from an EKF state.
   */
  Eigen::Vector3d HArmorXyz(const Eigen::VectorXd& x, int id) const
  {
    auto angle = LimitRad(x[6] + id * 2.0 * kPi / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
    auto r = use_l_h ? x[8] + x[9] : x[8];
    auto armor_x = x[0] - r * std::cos(angle);
    auto armor_y = x[2] - r * std::sin(angle);
    auto armor_z = use_l_h ? x[4] + x[10] : x[4];
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
    auto dx_da = r * std::sin(angle);
    auto dy_da = -r * std::cos(angle);
    auto dx_dr = -std::cos(angle);
    auto dy_dr = -std::sin(angle);
    auto dx_dl = use_l_h ? -std::cos(angle) : 0.0;
    auto dy_dl = use_l_h ? -std::sin(angle) : 0.0;
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
  };

  Solver& solver_;
  bool require_target_tag_;
  ArmorName target_name_;
  int min_detect_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
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
    return slot.initialized &&
           (slot.target.Diverged() || RecentNisFailed(slot));
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
    slot.score = ScoreSlot(slot);
  }

  /**
   * @brief Calculate the current target-selection score for one slot.
   */
  static double ScoreSlot(const TrackSlot& slot)
  {
    if (!slot.initialized || slot.state == "lost")
    {
      return -std::numeric_limits<double>::infinity();
    }

    const Eigen::VectorXd x = slot.target.EkfX();
    const double distance = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);
    const double distance_score = Clamp01((8.0 - distance) / 7.5);
    const double area_score = Clamp01(slot.hittable_area / 6000.0);
    const double count_score = Clamp01(slot.observed_count_lpf / 4.0);
    const double spin_score = Clamp01(1.0 - std::abs(x[7]) / 8.0);
    const double angle_score = Clamp01(1.0 - slot.gimbal_angle_error / 0.5);

    double state_scale = 1.0;
    if (slot.state == "detecting")
    {
      state_scale = 0.55;
    }
    else if (slot.state == "temp_lost")
    {
      state_scale = 0.35;
    }

    return state_scale * (2.0 * count_score + 1.2 * distance_score +
                          1.5 * area_score + spin_score + angle_score);
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
      constexpr double kSwitchMargin = 0.25;
      const auto& selected = tracks_[static_cast<std::size_t>(selected_index_)];
      if (best_index != selected_index_ &&
          tracks_[static_cast<std::size_t>(best_index)].score <=
              selected.score + kSwitchMargin)
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
      slot.target = Target(armor, t, 0.2765, 3, P0_dig);
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
