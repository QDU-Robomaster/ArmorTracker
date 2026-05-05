#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "ArmorDetectorTypes.hpp"
#include "cycle_value.hpp"
#include "logger.hpp"
#include "transform.hpp"

namespace armor_tracker
{
inline double UnwrapYawNear(double yaw, double reference_yaw)
{
  const double delta =
      LibXR::CycleValue<double>(yaw) - LibXR::CycleValue<double>(reference_yaw);
  return reference_yaw + delta;
}

inline double QuaternionToYaw(const LibXR::Quaternion<double>& q)
{
  LibXR::EulerAngle<double> eulr =
      LibXR::RotationMatrix<double>(q.ToRotationMatrix()).ToEulerAngle();
  return eulr.Yaw();
}

inline double OrientationToYawNear(const LibXR::Quaternion<double>& q,
                                   double reference_yaw)
{
  return UnwrapYawNear(QuaternionToYaw(q), reference_yaw);
}

inline double OrientationToYawNear(const ArmorDetectorResult& armor,
                                   double reference_yaw)
{
  return OrientationToYawNear(armor.pose.rotation, reference_yaw);
}

inline double MeasuredArmorYawNear(const LibXR::Quaternion<double>& q,
                                   double reference_yaw)
{
  // Detector 的 rotation yaw 与整车 EKF 的 face yaw 方向相反。
  // selector、SP update 和审计都必须使用同一套 yaw 约定。
  return UnwrapYawNear(QuaternionToYaw(q) + M_PI, reference_yaw);
}

inline double MeasuredArmorYawNear(const ArmorDetectorResult& armor,
                                   double reference_yaw)
{
  return MeasuredArmorYawNear(armor.pose.rotation, reference_yaw);
}

inline double AngularDiffAbs(double lhs, double rhs)
{
  return std::abs(LibXR::CycleValue<double>(lhs) - LibXR::CycleValue<double>(rhs));
}

inline void LogImpossibleYawDiff(const char* tag, std::size_t armor_index,
                                 int face_index, double measured_yaw,
                                 double predicted_yaw, double yaw_diff)
{
  if (!(std::isfinite(yaw_diff)) || yaw_diff <= M_PI + 1e-3)
  {
    return;
  }
  const double wrapped_measured = LibXR::CycleValue<double>(measured_yaw);
  const double wrapped_predicted = LibXR::CycleValue<double>(predicted_yaw);
  XR_LOG_ERROR(
      "Impossible yaw diff[%s]: armor=%u face=%d measured=%.6f predicted=%.6f wrapped_measured=%.6f wrapped_predicted=%.6f yaw_diff=%.6f direct_cycle_sub=%.6f raw_sub=%.6f",
      tag, static_cast<unsigned>(armor_index), face_index, measured_yaw, predicted_yaw, wrapped_measured,
      wrapped_predicted, yaw_diff,
      std::abs(LibXR::CycleValue<double>(measured_yaw) -
               LibXR::CycleValue<double>(predicted_yaw)),
      std::abs(measured_yaw - predicted_yaw));
}

inline uint64_t TimestampAbsDiff(uint64_t lhs, uint64_t rhs)
{
  return lhs >= rhs ? (lhs - rhs) : (rhs - lhs);
}

inline double ArmorImageArea(const ArmorDetectorResult& armor)
{
  return std::max(
      1.0, std::abs(cv::contourArea(std::vector<cv::Point2f>(
               armor.points.begin(), armor.points.end()))));
}

inline double DetectorObservationVarianceScale(const ArmorDetectorResult& armor)
{
  if (!armor.pnp_valid || !std::isfinite(armor.pnp_reprojection_error_px))
  {
    return 4.0;
  }

  const double excess_error =
      std::max(0.0, std::clamp(armor.pnp_reprojection_error_px, 0.0, 8.0) - 1.5);
  return std::clamp(1.0 + 0.02 * excess_error * excess_error, 1.0, 1.5);
}
}  // namespace armor_tracker
