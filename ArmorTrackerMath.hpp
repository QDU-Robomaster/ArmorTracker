#pragma once

/**
 * @file ArmorTrackerMath.hpp
 * @brief Internal math helpers and Kalman filter used by ArmorTracker.
 */

#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace armor_tracker_detail
{
/**
 * @brief Mathematical constants and physical dimensions used by the model.
 */
constexpr double kPi = 3.14159265358979323846;
constexpr double kLightbarLength = 56e-3;
constexpr double kBigArmorWidth = 230e-3;
constexpr double kSmallArmorWidth = 135e-3;

/**
 * @brief Convert an Eigen 3x3 matrix to an OpenCV double matrix.
 */
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

/**
 * @brief Convert an OpenCV double 3x3 matrix to Eigen.
 */
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

/**
 * @brief Normalize an angle into (-pi, pi].
 */
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

/**
 * @brief Convert a quaternion to Euler angles with the requested axis order.
 */
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

/**
 * @brief Convert a rotation matrix to Euler angles with the requested axis order.
 */
inline Eigen::Vector3d Eulers(Eigen::Matrix3d rotation, int axis0, int axis1,
                              int axis2, bool extrinsic = false)
{
  return Eulers(Eigen::Quaterniond(rotation), axis0, axis1, axis2, extrinsic);
}

/**
 * @brief Convert XYZ position to yaw, pitch, distance.
 */
inline Eigen::Vector3d XyzToYpd(const Eigen::Vector3d& xyz)
{
  const auto x = xyz[0];
  const auto y = xyz[1];
  const auto z = xyz[2];
  return {std::atan2(y, x), std::atan2(z, std::sqrt(x * x + y * y)),
          std::sqrt(x * x + y * y + z * z)};
}

/**
 * @brief Return the Jacobian of XyzToYpd().
 */
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

/**
 * @brief Return the signed duration from b to a in seconds.
 */
inline double DeltaTime(const std::chrono::steady_clock::time_point& a,
                        const std::chrono::steady_clock::time_point& b)
{
  std::chrono::duration<double> delta = a - b;
  return delta.count();
}

/**
 * @brief Minimal EKF used by the armor target model.
 */
class ExtendedKalmanFilter
{
 public:
  Eigen::VectorXd x;
  Eigen::MatrixXd P;
  std::map<std::string, double> data;
  std::deque<int> recent_nis_failures{0};
  size_t window_size = 100;
  double last_nis{};

  /**
   * @brief Construct an empty EKF object.
   */
  ExtendedKalmanFilter() = default;

  /**
   * @brief Construct an EKF with initial state, covariance, and state addition.
   */
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

  /**
   * @brief Predict with a linear transition matrix.
   */
  Eigen::VectorXd Predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q)
  {
    return Predict(F, Q, [&](const Eigen::VectorXd& prior) { return F * prior; });
  }

  /**
   * @brief Predict with a custom nonlinear state transition.
   */
  Eigen::VectorXd Predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
                          std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f)
  {
    P = F * P * F.transpose() + Q;
    x = f(x);
    return x;
  }

  /**
   * @brief Update with a linear observation matrix.
   */
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

  /**
   * @brief Update with a custom nonlinear observation function.
   */
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

}  // namespace armor_tracker_detail
