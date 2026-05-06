#pragma once

/**
 * @file ArmorTrackerCommon.hpp
 * @brief ArmorTracker 跨子模块复用的角度、时间戳和观测质量工具函数。
 */

#include <algorithm>
#include <array>
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
/**
 * @brief 将 yaw 展开到最接近参考 yaw 的连续角度。
 * @param yaw 待展开角度，单位 rad。
 * @param reference_yaw 参考角度，单位 rad。
 * @return 与参考角差最小的等价 yaw。
 */
inline double UnwrapYawNear(double yaw, double reference_yaw)
{
  const double delta =
      LibXR::CycleValue<double>(yaw) - LibXR::CycleValue<double>(reference_yaw);
  return reference_yaw + delta;
}

/**
 * @brief 从四元数中提取 yaw。
 * @param q 目标姿态四元数。
 * @return yaw 角，单位 rad。
 */
inline double QuaternionToYaw(const LibXR::Quaternion<double>& q)
{
  LibXR::EulerAngle<double> eulr =
      LibXR::RotationMatrix<double>(q.ToRotationMatrix()).ToEulerAngle();
  return eulr.Yaw();
}

/**
 * @brief 从姿态中提取并展开到参考 yaw 附近。
 */
inline double OrientationToYawNear(const LibXR::Quaternion<double>& q,
                                   double reference_yaw)
{
  return UnwrapYawNear(QuaternionToYaw(q), reference_yaw);
}

/**
 * @brief 从 detector 装甲结果姿态中提取并展开 yaw。
 */
inline double OrientationToYawNear(const ArmorDetectorResult& armor,
                                   double reference_yaw)
{
  return OrientationToYawNear(armor.pose.rotation, reference_yaw);
}

/**
 * @brief 将 detector 装甲姿态转换为 tracker 面 yaw 约定。
 *
 * Detector 的 rotation yaw 与整车 EKF 的 face yaw 方向相反，因此需要加 pi
 * 后再按参考角展开。
 */
inline double MeasuredArmorYawNear(const LibXR::Quaternion<double>& q,
                                   double reference_yaw)
{
  // Detector 的 rotation yaw 与整车 EKF 的 face yaw 方向相反。
  // selector、整车模型更新和审计都必须使用同一套 yaw 约定。
  return UnwrapYawNear(QuaternionToYaw(q) + M_PI, reference_yaw);
}

/**
 * @brief 从装甲检测结果中计算 tracker 面 yaw。
 */
inline double MeasuredArmorYawNear(const ArmorDetectorResult& armor,
                                   double reference_yaw)
{
  return MeasuredArmorYawNear(armor.pose.rotation, reference_yaw);
}

/**
 * @brief 在允许 pi 二义性时选择最接近参考角的装甲 yaw。
 */
inline double MeasuredArmorYawNearAllowPi(const LibXR::Quaternion<double>& q,
                                          double reference_yaw)
{
  const double base = MeasuredArmorYawNear(q, reference_yaw);
  std::array<double, 3> candidates = {
      base,
      UnwrapYawNear(base + M_PI, reference_yaw),
      UnwrapYawNear(base - M_PI, reference_yaw)};
  double best = candidates[0];
  double best_diff = std::abs(LibXR::CycleValue<double>(best) -
                              LibXR::CycleValue<double>(reference_yaw));
  for (double candidate : candidates)
  {
    const double diff = std::abs(LibXR::CycleValue<double>(candidate) -
                                 LibXR::CycleValue<double>(reference_yaw));
    if (diff < best_diff)
    {
      best = candidate;
      best_diff = diff;
    }
  }
  return best;
}

/**
 * @brief 从装甲检测结果中计算允许 pi 二义性的 tracker 面 yaw。
 */
inline double MeasuredArmorYawNearAllowPi(const ArmorDetectorResult& armor,
                                          double reference_yaw)
{
  return MeasuredArmorYawNearAllowPi(armor.pose.rotation, reference_yaw);
}

/**
 * @brief 计算两个角度在周期域中的绝对差。
 */
inline double AngularDiffAbs(double lhs, double rhs)
{
  return std::abs(LibXR::CycleValue<double>(lhs) - LibXR::CycleValue<double>(rhs));
}

/**
 * @brief 在角差超过理论范围时输出诊断日志。
 */
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

/**
 * @brief 计算两个无符号时间戳的绝对差。
 */
inline uint64_t TimestampAbsDiff(uint64_t lhs, uint64_t rhs)
{
  return lhs >= rhs ? (lhs - rhs) : (rhs - lhs);
}

/**
 * @brief 计算装甲四点轮廓面积，返回值下限为 1。
 */
inline double ArmorImageArea(const ArmorDetectorResult& armor)
{
  return std::max(
      1.0, std::abs(cv::contourArea(std::vector<cv::Point2f>(
               armor.points.begin(), armor.points.end()))));
}

/**
 * @brief 读取并钳制 detector PnP 重投影误差。
 */
inline double ArmorObservationReprojectionErrorPx(
    const ArmorDetectorResult& armor)
{
  if (!armor.pnp_valid || !std::isfinite(armor.pnp_reprojection_error_px))
  {
    return 8.0;
  }
  return std::clamp(armor.pnp_reprojection_error_px, 0.0, 8.0);
}

/**
 * @brief 根据重投影误差、图像面积和置信度生成观测质量惩罚。
 */
inline double ArmorObservationQualityPenalty(
    const ArmorDetectorResult& armor, double stable_max_reprojection_px,
    double stable_min_area_px, double stable_min_confidence)
{
  const double reprojection_error = ArmorObservationReprojectionErrorPx(armor);
  const double good_reprojection =
      std::min(0.8, std::max(0.0, stable_max_reprojection_px * 0.5));
  const double reprojection_den =
      std::max(1e-6, stable_max_reprojection_px - good_reprojection);
  const double reprojection_penalty =
      std::clamp((reprojection_error - good_reprojection) /
                     reprojection_den,
                 0.0, 2.0);

  const double area = ArmorImageArea(armor);
  const double area_penalty =
      stable_min_area_px > 1e-6
          ? std::clamp((stable_min_area_px - area) / stable_min_area_px, 0.0,
                       1.0)
          : 0.0;
  const double confidence_penalty =
      stable_min_confidence > 1e-6
          ? std::clamp((stable_min_confidence -
                        static_cast<double>(armor.confidence)) /
                           stable_min_confidence,
                       0.0, 1.0)
          : 0.0;

  return 0.65 * reprojection_penalty + 0.25 * area_penalty +
         0.10 * confidence_penalty;
}

/**
 * @brief 判断一条装甲观测是否满足稳定初始化或强匹配要求。
 */
inline bool StableArmorObservation(
    const ArmorDetectorResult& armor, double max_reprojection_px,
    double min_area_px, double min_confidence)
{
  if (!armor.pnp_valid)
  {
    return false;
  }
  if (!std::isfinite(armor.pnp_reprojection_error_px) ||
      armor.pnp_reprojection_error_px > max_reprojection_px)
  {
    return false;
  }
  if (ArmorImageArea(armor) < min_area_px)
  {
    return false;
  }
  if (static_cast<double>(armor.confidence) < min_confidence)
  {
    return false;
  }
  return true;
}

/**
 * @brief 将 detector 观测质量映射为 EKF 观测噪声缩放因子。
 */
inline double DetectorObservationVarianceScale(const ArmorDetectorResult& armor)
{
  if (!armor.pnp_valid)
  {
    return 4.0;
  }

  const double excess_error =
      std::max(0.0, ArmorObservationReprojectionErrorPx(armor) - 1.5);
  const double quality_penalty =
      ArmorObservationQualityPenalty(armor, 1.8, 60.0, 0.0);
  return std::clamp(1.0 + 0.35 * quality_penalty +
                        0.04 * excess_error * excess_error,
                    1.0, 3.0);
}
}  // namespace armor_tracker
