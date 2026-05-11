#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace armor_tracker_xr
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kLightbarLength = 56e-3;
constexpr double kBigArmorWidth = 230e-3;
constexpr double kSmallArmorWidth = 135e-3;

inline cv::Mat Mat3dToCv(const Eigen::Matrix3d& matrix)
{
  cv::Mat out(3, 3, CV_64F);
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      out.at<double>(row, col) = matrix(row, col);
    }
  }
  return out;
}

inline Eigen::Matrix3d CvMatToMat3d(const cv::Mat& matrix)
{
  Eigen::Matrix3d out;
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      out(row, col) = matrix.at<double>(row, col);
    }
  }
  return out;
}

inline double LimitRad(double angle)
{
  while (angle > kPi)
  {
    angle -= 2.0 * kPi;
  }
  while (angle <= -kPi)
  {
    angle += 2.0 * kPi;
  }
  return angle;
}

inline Eigen::Vector3d Eulers(Eigen::Quaterniond q, int axis0, int axis1,
                              int axis2, bool extrinsic = false)
{
  if (!extrinsic)
  {
    std::swap(axis0, axis2);
  }

  auto i = axis0;
  auto j = axis1;
  auto k = axis2;
  const bool is_proper = i == k;
  if (is_proper)
  {
    k = 3 - i - j;
  }
  const auto sign = (i - j) * (j - k) * (k - i) / 2;

  double a, b, c, d;
  Eigen::Vector4d xyzw = q.coeffs();
  if (is_proper)
  {
    a = xyzw[3];
    b = xyzw[i];
    c = xyzw[j];
    d = xyzw[k] * sign;
  }
  else
  {
    a = xyzw[3] - xyzw[j];
    b = xyzw[i] + xyzw[k] * sign;
    c = xyzw[j] + xyzw[3];
    d = xyzw[k] * sign - xyzw[i];
  }

  Eigen::Vector3d eulers;
  const auto n2 = a * a + b * b + c * c + d * d;
  eulers[1] = std::acos(2.0 * (a * a + b * b) / n2 - 1.0);

  const auto half_sum = std::atan2(b, a);
  const auto half_diff = std::atan2(-d, c);

  constexpr double eps = 1e-7;
  const auto safe1 = std::abs(eulers[1]) >= eps;
  const auto safe2 = std::abs(eulers[1] - kPi) >= eps;
  const auto safe = safe1 && safe2;
  if (safe)
  {
    eulers[0] = half_sum + half_diff;
    eulers[2] = half_sum - half_diff;
  }
  else
  {
    if (!extrinsic)
    {
      eulers[0] = 0.0;
      if (!safe1)
      {
        eulers[2] = 2.0 * half_sum;
      }
      if (!safe2)
      {
        eulers[2] = -2.0 * half_diff;
      }
    }
    else
    {
      eulers[2] = 0.0;
      if (!safe1)
      {
        eulers[0] = 2.0 * half_sum;
      }
      if (!safe2)
      {
        eulers[0] = 2.0 * half_diff;
      }
    }
  }

  for (int idx = 0; idx < 3; ++idx)
  {
    eulers[idx] = LimitRad(eulers[idx]);
  }

  if (!is_proper)
  {
    eulers[2] *= sign;
    eulers[1] -= kPi / 2.0;
  }

  if (!extrinsic)
  {
    std::swap(eulers[0], eulers[2]);
  }

  return eulers;
}

inline Eigen::Vector3d Eulers(Eigen::Matrix3d rotation, int axis0, int axis1,
                              int axis2, bool extrinsic = false)
{
  return Eulers(Eigen::Quaterniond(rotation), axis0, axis1, axis2, extrinsic);
}

inline Eigen::Vector3d XyzToYpd(const Eigen::Vector3d& xyz)
{
  const auto x = xyz[0];
  const auto y = xyz[1];
  const auto z = xyz[2];
  return {std::atan2(y, x), std::atan2(z, std::sqrt(x * x + y * y)),
          std::sqrt(x * x + y * y + z * z)};
}

inline Eigen::MatrixXd XyzToYpdJacobian(const Eigen::Vector3d& xyz)
{
  const auto x = xyz[0];
  const auto y = xyz[1];
  const auto z = xyz[2];

  const auto dyaw_dx = -y / (x * x + y * y);
  const auto dyaw_dy = x / (x * x + y * y);
  const auto dyaw_dz = 0.0;

  const auto dpitch_dx =
      -(x * z) / ((z * z / (x * x + y * y) + 1.0) *
                  std::pow((x * x + y * y), 1.5));
  const auto dpitch_dy =
      -(y * z) / ((z * z / (x * x + y * y) + 1.0) *
                  std::pow((x * x + y * y), 1.5));
  const auto dpitch_dz =
      1.0 / ((z * z / (x * x + y * y) + 1.0) *
             std::pow((x * x + y * y), 0.5));

  const auto ddistance_dx = x / std::pow((x * x + y * y + z * z), 0.5);
  const auto ddistance_dy = y / std::pow((x * x + y * y + z * z), 0.5);
  const auto ddistance_dz = z / std::pow((x * x + y * y + z * z), 0.5);

  Eigen::MatrixXd jacobian(3, 3);
  jacobian << dyaw_dx, dyaw_dy, dyaw_dz, dpitch_dx, dpitch_dy,
      dpitch_dz, ddistance_dx, ddistance_dy, ddistance_dz;
  return jacobian;
}

inline double DeltaTime(const std::chrono::steady_clock::time_point& a,
                        const std::chrono::steady_clock::time_point& b)
{
  std::chrono::duration<double> delta = a - b;
  return delta.count();
}

class ExtendedKalmanFilter
{
 public:
  Eigen::VectorXd x;
  Eigen::MatrixXd P;
  std::map<std::string, double> data;
  std::deque<int> recent_nis_failures{0};
  size_t window_size = 100;
  double last_nis{};

  ExtendedKalmanFilter() = default;

  ExtendedKalmanFilter(
      const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                    const Eigen::VectorXd&)>
          x_add_in =
              [](const Eigen::VectorXd& a, const Eigen::VectorXd& b)
      { return a + b; })
      : x(x0),
        P(P0),
        I_(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())),
        x_add_(std::move(x_add_in))
  {
    data["residual_yaw"] = 0.0;
    data["residual_pitch"] = 0.0;
    data["residual_distance"] = 0.0;
    data["residual_angle"] = 0.0;
    data["nis"] = 0.0;
    data["nees"] = 0.0;
    data["nis_fail"] = 0.0;
    data["nees_fail"] = 0.0;
    data["recent_nis_failures"] = 0.0;
  }

  Eigen::VectorXd Predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q)
  {
    return Predict(F, Q, [&](const Eigen::VectorXd& prior) { return F * prior; });
  }

  Eigen::VectorXd Predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
                          std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f)
  {
    P = F * P * F.transpose() + Q;
    x = f(x);
    return x;
  }

  Eigen::VectorXd Update(
      const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
      const Eigen::MatrixXd& R,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                    const Eigen::VectorXd&)>
          z_subtract =
              [](const Eigen::VectorXd& a, const Eigen::VectorXd& b)
      { return a - b; })
  {
    return Update(z, H, R, [&](const Eigen::VectorXd& state) { return H * state; },
                  std::move(z_subtract));
  }

  Eigen::VectorXd Update(
      const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
      const Eigen::MatrixXd& R,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                    const Eigen::VectorXd&)>
          z_subtract =
              [](const Eigen::VectorXd& a, const Eigen::VectorXd& b)
      { return a - b; })
  {
    Eigen::VectorXd x_prior = x;
    Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
    P = (I_ - K * H) * P * (I_ - K * H).transpose() + K * R * K.transpose();
    x = x_add_(x, K * z_subtract(z, h(x)));

    Eigen::VectorXd residual = z_subtract(z, h(x));
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    double nis = residual.transpose() * S.inverse() * residual;
    double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

    constexpr double nis_threshold = 0.711;
    constexpr double nees_threshold = 0.711;
    if (nis > nis_threshold)
    {
      ++nis_count_;
      data["nis_fail"] = 1.0;
    }
    if (nees > nees_threshold)
    {
      ++nees_count_;
      data["nees_fail"] = 1.0;
    }
    ++total_count_;
    last_nis = nis;

    recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);
    if (recent_nis_failures.size() > window_size)
    {
      recent_nis_failures.pop_front();
    }
    const int recent_failures =
        std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
    const double recent_rate =
        static_cast<double>(recent_failures) / recent_nis_failures.size();

    data["residual_yaw"] = residual[0];
    data["residual_pitch"] = residual[1];
    data["residual_distance"] = residual[2];
    data["residual_angle"] = residual[3];
    data["nis"] = nis;
    data["nees"] = nees;
    data["recent_nis_failures"] = recent_rate;
    return x;
  }

 private:
  Eigen::MatrixXd I_;
  std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                const Eigen::VectorXd&)>
      x_add_;
  int nees_count_ = 0;
  int nis_count_ = 0;
  int total_count_ = 0;
};

enum Color
{
  red,
  blue,
  extinguish,
  purple
};

enum ArmorType
{
  big,
  small
};

enum ArmorName
{
  one,
  two,
  three,
  four,
  five,
  sentry,
  outpost,
  base,
  not_armor
};

enum ArmorPriority
{
  first = 1,
  second,
  third,
  forth,
  fifth
};

struct Armor
{
  Color color{};
  cv::Point2f center{};
  cv::Point2f center_norm{};
  std::vector<cv::Point2f> points{};
  double ratio{};
  double rectangular_error{};
  ArmorType type{};
  ArmorName name{};
  ArmorPriority priority{fifth};
  cv::Rect box{};
  double confidence{};
  Eigen::Vector3d xyz_in_gimbal = Eigen::Vector3d::Zero();
  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypr_in_gimbal = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();
  double yaw_raw{};

  Armor() = default;

  Armor(int color_id, int num_id, float confidence_in, const cv::Rect& box_in,
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
    color = color_id == 0 ? Color::blue
                          : color_id == 1 ? Color::red : Color::extinguish;
    name = num_id == 0 ? ArmorName::sentry
                       : num_id > 5 ? ArmorName(num_id) : ArmorName(num_id - 1);
    type = num_id == 1 ? ArmorType::big : ArmorType::small;
  }
};

inline ArmorPriority PriorityFromName(ArmorName name)
{
  switch (name)
  {
    case ArmorName::three:
    case ArmorName::four:
      return ArmorPriority::first;
    case ArmorName::one:
      return ArmorPriority::second;
    case ArmorName::five:
    case ArmorName::sentry:
      return ArmorPriority::third;
    case ArmorName::two:
      return ArmorPriority::forth;
    case ArmorName::outpost:
    case ArmorName::base:
    case ArmorName::not_armor:
    default:
      return ArmorPriority::fifth;
  }
}

struct Config
{
  int enemy_color_id = -1;
  bool require_target_tag = false;
  int target_tag_id = -1;
  int min_detect_count = 2;
  int max_temp_lost_count = 15;
  int outpost_max_temp_lost_count = 75;
  int frame_convention = 1;
  std::array<double, 9> camera_matrix{
      1164.3428599490444, 0.0, 366.6782312546237,
      0.0, 1164.335053894998, 270.30936434613865,
      0.0, 0.0, 1.0};
};

class Solver
{
 public:
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
    if (config.frame_convention == 0)
    {
      R_camera2gimbal_ = Eigen::Matrix3d::Identity();
    }
    else
    {
      R_camera2gimbal_ << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0,
          0.0;
    }
  }

  void SetRGimbal2World(const Eigen::Quaterniond& q)
  {
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    R_gimbal2world_ =
        R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
  }

  void Solve(Armor& armor) const
  {
    const auto& object_points =
        armor.type == ArmorType::big ? BigArmorPoints() : SmallArmorPoints();
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
        armor.type == ArmorType::big &&
        (armor.name == ArmorName::three || armor.name == ArmorName::four ||
         armor.name == ArmorName::five);
    if (is_balance)
    {
      return;
    }
    OptimizeYaw(armor);
  }

  std::vector<cv::Point2f> ReprojectArmor(const Eigen::Vector3d& xyz_in_world,
                                          double yaw, ArmorType type,
                                          ArmorName name) const
  {
    const auto sin_yaw = std::sin(yaw);
    const auto cos_yaw = std::cos(yaw);
    const auto pitch =
        name == ArmorName::outpost ? -15.0 * kPi / 180.0 : 15.0 * kPi / 180.0;
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
        type == ArmorType::big ? BigArmorPoints() : SmallArmorPoints();
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

  static const std::vector<cv::Point3f>& BigArmorPoints()
  {
    static const std::vector<cv::Point3f> points{
        {0, kBigArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kBigArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kBigArmorWidth / 2.0, -kLightbarLength / 2.0},
        {0, kBigArmorWidth / 2.0, -kLightbarLength / 2.0}};
    return points;
  }

  static const std::vector<cv::Point3f>& SmallArmorPoints()
  {
    static const std::vector<cv::Point3f> points{
        {0, kSmallArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kSmallArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kSmallArmorWidth / 2.0, -kLightbarLength / 2.0},
        {0, kSmallArmorWidth / 2.0, -kLightbarLength / 2.0}};
    return points;
  }

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

class Target
{
 public:
  ArmorName name{};
  ArmorType armor_type{};
  ArmorPriority priority{};
  bool jumped{};
  int last_id{};
  bool isinit = false;

  Target() = default;

  Target(const Armor& armor, std::chrono::steady_clock::time_point t,
         double radius, int armor_num, const Eigen::VectorXd& P0_dig)
      : name(armor.name),
        armor_type(armor.type),
        priority(armor.priority),
        jumped(false),
        last_id(0),
        armor_num_(armor_num),
        switch_count_(0),
        update_count_(0),
        is_switch_(false),
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

  void Predict(std::chrono::steady_clock::time_point t)
  {
    auto dt = DeltaTime(t, t_);
    Predict(dt);
    t_ = t;
  }

  void Predict(double dt)
  {
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;

    double v1;
    double v2;
    if (name == ArmorName::outpost)
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

    if (Convergened() && name == ArmorName::outpost && std::abs(ekf_.x[7]) > 2.0)
    {
      ekf_.x[7] = ekf_.x[7] > 0.0 ? 2.51 : -2.51;
    }
    ekf_.Predict(F, Q, f);
  }

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

    for (int i = 0; i < 3; ++i)
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
    is_switch_ = id != last_id;
    if (is_switch_)
    {
      ++switch_count_;
    }
    last_id = id;
    ++update_count_;
    UpdateYpda(armor, id);
  }

  Eigen::VectorXd EkfX() const { return ekf_.x; }
  const ExtendedKalmanFilter& Ekf() const { return ekf_; }

  std::vector<Eigen::Vector4d> ArmorXyzaList() const
  {
    std::vector<Eigen::Vector4d> list;
    for (int i = 0; i < armor_num_; ++i)
    {
      auto angle = LimitRad(ekf_.x[6] + i * 2.0 * kPi / armor_num_);
      Eigen::Vector3d xyz = HArmorXyz(ekf_.x, i);
      list.push_back({xyz[0], xyz[1], xyz[2], angle});
    }
    return list;
  }

  bool Diverged() const
  {
    auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
    auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
    return !(r_ok && l_ok);
  }

  bool Convergened()
  {
    if (name != ArmorName::outpost && update_count_ > 3 && !Diverged())
    {
      is_converged_ = true;
    }
    if (name == ArmorName::outpost && update_count_ > 10 && !Diverged())
    {
      is_converged_ = true;
    }
    return is_converged_;
  }

 private:
  int armor_num_ = 4;
  int switch_count_ = 0;
  int update_count_ = 0;
  bool is_switch_ = false;
  bool is_converged_ = false;
  ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_{};

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

class Tracker
{
 public:
  Tracker(const Config& config, Solver& solver)
      : solver_(solver),
        enemy_color_(config.enemy_color_id == 1 ? Color::red : Color::blue),
        min_detect_count_(config.min_detect_count),
        max_temp_lost_count_(config.max_temp_lost_count),
        detect_count_(0),
        temp_lost_count_(0),
        outpost_max_temp_lost_count_(config.outpost_max_temp_lost_count),
        normal_temp_lost_count_(config.max_temp_lost_count),
        state_("lost"),
        pre_state_("lost"),
        last_timestamp_(std::chrono::steady_clock::now()),
        omni_target_priority_(ArmorPriority::fifth)
  {
  }

  std::string State() const { return state_; }

  std::list<Target> Track(std::list<Armor>& armors,
                          std::chrono::steady_clock::time_point t,
                          bool use_enemy_color = true)
  {
    (void)use_enemy_color;
    auto dt = DeltaTime(t, last_timestamp_);
    last_timestamp_ = t;
    if (state_ != "lost" && dt > 0.1)
    {
      state_ = "lost";
    }

    armors.remove_if(
        [&](const Armor& armor) { return armor.color != enemy_color_; });

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

    bool found;
    if (state_ == "lost")
    {
      found = SetTarget(armors, t);
    }
    else
    {
      found = UpdateTarget(armors, t);
    }

    StateMachine(found);
    if (state_ != "lost" && target_.Diverged())
    {
      state_ = "lost";
      return {};
    }
    if (std::accumulate(target_.Ekf().recent_nis_failures.begin(),
                        target_.Ekf().recent_nis_failures.end(), 0) >=
        0.4 * target_.Ekf().window_size)
    {
      state_ = "lost";
      return {};
    }
    if (state_ == "lost")
    {
      return {};
    }
    return {target_};
  }

 private:
  Solver& solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_;
  std::string pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;

  void StateMachine(bool found)
  {
    if (state_ == "lost")
    {
      if (!found)
      {
        return;
      }
      state_ = "detecting";
      detect_count_ = 1;
    }
    else if (state_ == "detecting")
    {
      if (found)
      {
        ++detect_count_;
        if (detect_count_ >= min_detect_count_)
        {
          state_ = "tracking";
        }
      }
      else
      {
        detect_count_ = 0;
        state_ = "lost";
      }
    }
    else if (state_ == "tracking")
    {
      if (found)
      {
        return;
      }
      temp_lost_count_ = 1;
      state_ = "temp_lost";
    }
    else if (state_ == "switching")
    {
      if (found)
      {
        state_ = "detecting";
      }
      else
      {
        ++temp_lost_count_;
        if (temp_lost_count_ > 200)
        {
          state_ = "lost";
        }
      }
    }
    else if (state_ == "temp_lost")
    {
      if (found)
      {
        state_ = "tracking";
      }
      else
      {
        ++temp_lost_count_;
        if (target_.name == ArmorName::outpost)
        {
          max_temp_lost_count_ = outpost_max_temp_lost_count_;
        }
        else
        {
          max_temp_lost_count_ = normal_temp_lost_count_;
        }
        if (temp_lost_count_ > max_temp_lost_count_)
        {
          state_ = "lost";
        }
      }
    }
  }

  bool SetTarget(std::list<Armor>& armors,
                 std::chrono::steady_clock::time_point t)
  {
    if (armors.empty())
    {
      return false;
    }
    auto& armor = armors.front();
    solver_.Solve(armor);

    const auto is_balance =
        armor.type == ArmorType::big &&
        (armor.name == ArmorName::three || armor.name == ArmorName::four ||
         armor.name == ArmorName::five);

    Eigen::VectorXd P0_dig(11);
    if (is_balance)
    {
      P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
      target_ = Target(armor, t, 0.2, 2, P0_dig);
    }
    else if (armor.name == ArmorName::outpost)
    {
      P0_dig << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0;
      target_ = Target(armor, t, 0.2765, 3, P0_dig);
    }
    else if (armor.name == ArmorName::base)
    {
      P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
      target_ = Target(armor, t, 0.3205, 3, P0_dig);
    }
    else
    {
      P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
      target_ = Target(armor, t, 0.2, 4, P0_dig);
    }
    return true;
  }

  bool UpdateTarget(std::list<Armor>& armors,
                    std::chrono::steady_clock::time_point t)
  {
    target_.Predict(t);
    int found_count = 0;
    double min_x = 1e10;
    for (const auto& armor : armors)
    {
      if (armor.name != target_.name || armor.type != target_.armor_type)
      {
        continue;
      }
      ++found_count;
      min_x = armor.center.x < min_x ? armor.center.x : min_x;
    }
    (void)min_x;
    if (found_count == 0)
    {
      return false;
    }

    for (auto& armor : armors)
    {
      if (armor.name != target_.name || armor.type != target_.armor_type)
      {
        continue;
      }
      solver_.Solve(armor);
      target_.Update(armor);
    }
    return true;
  }
};

struct InputArmor
{
  int color_id = -1;
  int tag_id = -1;
  int armor_type = 0;
  double confidence = 0.0;
  std::array<cv::Point2f, 4> corners{};
  cv::Point2f center{};
  cv::Point2f center_norm{0.5F, 0.5F};
};

struct Output
{
  std::string state{"lost"};
  bool has_target = false;
  int selected_tag_id = -1;
  int selected_color_id = -1;
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
};

inline int QduNumberToInternalArmorNameId(int number)
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

inline bool QduTypeIsLarge(const InputArmor& input)
{
  return input.armor_type == 1 || input.tag_id == 0 || input.tag_id == 7;
}

inline Eigen::Vector3d XrWorldToCamera(const Eigen::Vector3d& point)
{
  return {-point.y(), -point.z(), point.x()};
}

inline int ArmorNameToQduNumber(ArmorName name)
{
  switch (name)
  {
    case ArmorName::one:
      return 0;
    case ArmorName::two:
      return 1;
    case ArmorName::three:
      return 2;
    case ArmorName::four:
      return 3;
    case ArmorName::five:
      return 4;
    case ArmorName::outpost:
      return 5;
    case ArmorName::sentry:
      return 6;
    case ArmorName::base:
      return 7;
    case ArmorName::not_armor:
    default:
      return 8;
  }
}

inline Armor MakeTrackedArmor(const InputArmor& input)
{
  std::vector<cv::Point2f> points;
  points.reserve(4);
  for (const auto& point : input.corners)
  {
    points.push_back(point);
  }
  const cv::Rect box = cv::boundingRect(points);
  Armor armor(input.color_id, QduNumberToInternalArmorNameId(input.tag_id),
              static_cast<float>(input.confidence), box, points);
  armor.type = QduTypeIsLarge(input) ? ArmorType::big : ArmorType::small;
  armor.priority = PriorityFromName(armor.name);
  armor.center_norm = input.center_norm;
  return armor;
}

class LockedTracker
{
 public:
  LockedTracker() { Configure(config_); }

  explicit LockedTracker(const Config& config)
  {
    Configure(config);
  }

  void Configure(const Config& config)
  {
    config_ = config;
    solver_ = std::make_unique<Solver>(config_);
    tracker_ = std::make_unique<Tracker>(config_, *solver_);
    has_time_base_ = false;
    base_timestamp_us_ = 0;
  }

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
      if (config_.enemy_color_id >= 0 && input.color_id != config_.enemy_color_id)
      {
        continue;
      }
      if (config_.require_target_tag && input.tag_id != config_.target_tag_id)
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
    const auto targets = tracker_->Track(armors, tp, false);

    Output out;
    out.state = tracker_->State();
    out.selected_color_id = config_.enemy_color_id;
    out.selected_tag_id = config_.target_tag_id;
    if (targets.empty())
    {
      return out;
    }
    const Target& target = targets.front();
    out.has_target = true;
    out.selected_tag_id = ArmorNameToQduNumber(target.name);
    out.armors_num = static_cast<int>(target.ArmorXyzaList().size());
    out.selected_face = target.last_id;
    out.jumped = target.jumped;
    const Eigen::VectorXd x = target.EkfX();
    out.radius_even = x[8];
    out.radius_odd = x[8] + x[9];
    out.faces = target.ArmorXyzaList();
    if (config_.frame_convention == 0)
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
      out.center = XrWorldToCamera({x[0], x[2], x[4]});
      out.velocity = XrWorldToCamera({x[1], x[3], x[5]});
      out.yaw = LimitRad(x[6] - kPi * 0.5);
      out.vyaw = x[7];
      out.dz = -x[10];
      if (target.last_id >= 0 &&
          target.last_id < static_cast<int>(out.faces.size()))
      {
        const auto& face = out.faces[static_cast<std::size_t>(target.last_id)];
        out.armor = XrWorldToCamera(face.head<3>());
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

}  // namespace armor_tracker_xr
