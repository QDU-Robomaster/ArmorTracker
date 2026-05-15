#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "ArmorTrackerModel.hpp"

namespace
{
constexpr double kTolerance = 1e-10;

Eigen::Matrix3d RotationX(double angle)
{
  return Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
}

Eigen::Matrix3d RotationY(double angle)
{
  return Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()).toRotationMatrix();
}

Eigen::Matrix3d RotationZ(double angle)
{
  return Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

void Check(bool condition, const char* message)
{
  if (!condition)
  {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void CheckNear(double value, double expected, const char* message)
{
  if (std::abs(value - expected) > kTolerance)
  {
    std::cerr << "FAIL: " << message << " value=" << value
              << " expected=" << expected << '\n';
    std::exit(1);
  }
}
}  // namespace

int main()
{
  using armor_tracker_detail::BodyToWorldRotationFromImu;

  const Eigen::Matrix3d expected =
      RotationZ(0.3) * RotationY(-0.1) * RotationX(0.2);
  const Eigen::Matrix3d actual =
      BodyToWorldRotationFromImu(Eigen::Quaterniond(expected));
  Check((actual - expected).cwiseAbs().maxCoeff() < kTolerance,
        "gimbal_quat must already be interpreted in public B axes");

  const Eigen::Vector3d forward_point(0.0, 2.0, 0.0);
  const Eigen::Matrix3d positive_roll =
      BodyToWorldRotationFromImu(Eigen::Quaterniond(RotationX(0.2)));
  CheckNear((positive_roll * forward_point).z(), std::sin(0.2) * 2.0,
            "positive roll must raise a forward point in world z");

  const Eigen::Matrix3d yaw_only =
      BodyToWorldRotationFromImu(Eigen::Quaterniond(RotationZ(0.2)));
  Check((yaw_only - RotationZ(0.2)).cwiseAbs().maxCoeff() < kTolerance,
        "yaw-only attitude must remain yaw-only");

  const Eigen::Matrix3d invalid = BodyToWorldRotationFromImu(
      Eigen::Quaterniond(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0,
                         0.0));
  Check((invalid - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() <
            kTolerance,
        "invalid quaternion must fall back to identity");

  std::cout << "coordinate_semantics_check PASS\n";
  return 0;
}
