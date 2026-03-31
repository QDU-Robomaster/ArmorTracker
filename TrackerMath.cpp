#include "TrackerMath.hpp"

#include <cmath>

#include <opencv2/core.hpp>

namespace TrackerMath
{
double LimitRad(double angle)
{
  while (angle > CV_PI)
  {
    angle -= 2.0 * CV_PI;
  }
  while (angle <= -CV_PI)
  {
    angle += 2.0 * CV_PI;
  }
  return angle;
}

Eigen::Vector3d XyzToYpd(const Eigen::Vector3d& xyz)
{
  const double x = xyz.x();
  const double y = xyz.y();
  const double z = xyz.z();
  const double yaw = std::atan2(y, x);
  const double pitch = std::atan2(z, std::sqrt(x * x + y * y));
  const double distance = std::sqrt(x * x + y * y + z * z);
  return {yaw, pitch, distance};
}

Eigen::MatrixXd XyzToYpdJacobian(const Eigen::Vector3d& xyz)
{
  const double x = xyz.x();
  const double y = xyz.y();
  const double z = xyz.z();
  const double xy_square = x * x + y * y;
  const double xy_norm = std::sqrt(std::max(xy_square, 1e-12));
  const double xyz_square = x * x + y * y + z * z;
  const double xyz_norm = std::sqrt(std::max(xyz_square, 1e-12));
  const double pitch_den = (z * z / std::max(xy_square, 1e-12)) + 1.0;

  Eigen::MatrixXd jacobian(3, 3);
  jacobian(0, 0) = -y / std::max(xy_square, 1e-12);
  jacobian(0, 1) = x / std::max(xy_square, 1e-12);
  jacobian(0, 2) = 0.0;
  jacobian(1, 0) = -(x * z) / (pitch_den * std::pow(std::max(xy_square, 1e-12), 1.5));
  jacobian(1, 1) = -(y * z) / (pitch_den * std::pow(std::max(xy_square, 1e-12), 1.5));
  jacobian(1, 2) = 1.0 / (pitch_den * xy_norm);
  jacobian(2, 0) = x / xyz_norm;
  jacobian(2, 1) = y / xyz_norm;
  jacobian(2, 2) = z / xyz_norm;
  return jacobian;
}

double DeltaTime(const std::chrono::steady_clock::time_point& lhs,
                 const std::chrono::steady_clock::time_point& rhs)
{
  const std::chrono::duration<double> delta = lhs - rhs;
  return delta.count();
}

double DeltaTime(const LibXR::MicrosecondTimestamp& lhs,
                 const LibXR::MicrosecondTimestamp& rhs)
{
  return (lhs - rhs).ToSecond();
}
}  // namespace TrackerMath
