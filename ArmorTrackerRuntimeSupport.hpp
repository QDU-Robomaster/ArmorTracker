#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>

#include "ArmorTrackerCommon.hpp"
#include "CameraBase.hpp"

namespace armor_tracker_detail
{
inline LibXR::Quaternion<double> PackedCameraRotation(
    const std::array<float, 4>& rotation_wxyz)
{
  return LibXR::Quaternion<double>(rotation_wxyz[0], rotation_wxyz[1],
                                   rotation_wxyz[2], rotation_wxyz[3]);
}

inline LibXR::Position<double> PackedCameraTranslation(
    const std::array<float, 3>& translation_xyz)
{
  return LibXR::Position<double>(translation_xyz[0], translation_xyz[1],
                                 translation_xyz[2]);
}

inline LibXR::Transform<double> ArmorTrackerCameraRotationToTrackerWorldPose(
    const LibXR::Quaternion<double>& camera_rotation,
    const LibXR::Position<double>& camera_translation,
    const LibXR::Transform<double>& gimbal_to_camera_transform_static)
{
  return LibXR::Transform<double>(camera_rotation, camera_translation) +
         gimbal_to_camera_transform_static;
}

struct FaceConstrainedProjectionResult
{
  bool valid{false};
  Eigen::Vector3d world_position{0.0, 0.0, 0.0};
  double sp_yaw_rad{0.0};
  double reprojection_rmse_px{0.0};
};

inline cv::Mat BuildCameraMatrixCv(const CameraTypes::CameraInfo& camera_info)
{
  return cv::Mat(3, 3, CV_64F,
                 const_cast<double*>(camera_info.camera_matrix.data()))
      .clone();
}

inline cv::Mat BuildDistCoeffsCv(const CameraTypes::CameraInfo& camera_info)
{
  std::vector<double> coeffs;
  switch (camera_info.distortion_model)
  {
    case CameraTypes::DistortionModel::NONE:
      break;
    case CameraTypes::DistortionModel::PLUMB_BOB:
      coeffs = {camera_info.distortion_coefficients[0],
                camera_info.distortion_coefficients[1],
                camera_info.distortion_coefficients[2],
                camera_info.distortion_coefficients[3],
                camera_info.distortion_coefficients[4]};
      break;
    case CameraTypes::DistortionModel::RATIONAL_POLYNOMIAL:
      coeffs = {camera_info.distortion_coefficients[0],
                camera_info.distortion_coefficients[1],
                camera_info.distortion_coefficients[2],
                camera_info.distortion_coefficients[3],
                camera_info.distortion_coefficients[4],
                camera_info.distortion_coefficients[5],
                camera_info.distortion_coefficients[6],
                camera_info.distortion_coefficients[7]};
      break;
    default:
      break;
  }
  if (coeffs.empty())
  {
    return {};
  }
  return cv::Mat(1, static_cast<int>(coeffs.size()), CV_64F, coeffs.data()).clone();
}

inline std::vector<cv::Point3f> ArmorObjectPoints(ArmorType armor_type)
{
  const double width_mm = armor_type == ArmorType::LARGE ? 225.0 : 135.0;
  constexpr double kHeightMm = 56.0;
  const double half_width_m = width_mm * 0.5 / 1000.0;
  const double half_height_m = kHeightMm * 0.5 / 1000.0;
  return {{0.0F, static_cast<float>(half_width_m), static_cast<float>(-half_height_m)},
          {0.0F, static_cast<float>(half_width_m), static_cast<float>(half_height_m)},
          {0.0F, static_cast<float>(-half_width_m), static_cast<float>(half_height_m)},
          {0.0F, static_cast<float>(-half_width_m), static_cast<float>(-half_height_m)}};
}

inline std::vector<cv::Point2f> ArmorImagePointsInPnpOrder(
    const ArmorDetectorResult& armor)
{
  return {armor.points[2], armor.points[3], armor.points[0], armor.points[1]};
}

inline cv::Mat EigenRotationToCvMat(const Eigen::Matrix3d& rotation)
{
  cv::Mat mat(3, 3, CV_64F);
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      mat.at<double>(row, col) = rotation(row, col);
    }
  }
  return mat;
}

inline double ReprojectionRmse(const std::vector<cv::Point3f>& object_points,
                               const std::vector<cv::Point2f>& image_points,
                               const cv::Mat& camera_matrix,
                               const cv::Mat& dist_coeffs, const cv::Mat& rvec,
                               const cv::Mat& tvec)
{
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected);
  if (projected.size() != image_points.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < projected.size(); ++i)
  {
    const cv::Point2f diff = projected[i] - image_points[i];
    sum_sq += static_cast<double>(diff.x) * diff.x +
              static_cast<double>(diff.y) * diff.y;
  }
  return std::sqrt(sum_sq / static_cast<double>(projected.size()));
}

inline bool SolveTranslationForRotation(const std::vector<cv::Point3f>& object_points,
                                        const std::vector<cv::Point2f>& normalized_points,
                                        const Eigen::Matrix3d& r_optical_object,
                                        Eigen::Vector3d& t_optical_object)
{
  if (object_points.size() != normalized_points.size() || object_points.empty())
  {
    return false;
  }

  Eigen::MatrixXd a(2 * static_cast<int>(object_points.size()), 3);
  Eigen::VectorXd b(2 * static_cast<int>(object_points.size()));
  for (std::size_t i = 0; i < object_points.size(); ++i)
  {
    const Eigen::Vector3d object_point(object_points[i].x, object_points[i].y,
                                       object_points[i].z);
    const Eigen::Vector3d rotated_point = r_optical_object * object_point;
    const double u = normalized_points[i].x;
    const double v = normalized_points[i].y;
    const int row = static_cast<int>(2 * i);

    a(row, 0) = 1.0;
    a(row, 1) = 0.0;
    a(row, 2) = -u;
    b(row) = u * rotated_point.z() - rotated_point.x();
    a(row + 1, 0) = 0.0;
    a(row + 1, 1) = 1.0;
    a(row + 1, 2) = -v;
    b(row + 1) = v * rotated_point.z() - rotated_point.y();
  }

  t_optical_object = a.colPivHouseholderQr().solve(b);
  if (!t_optical_object.allFinite())
  {
    return false;
  }
  for (const auto& object_point_cv : object_points)
  {
    const Eigen::Vector3d object_point(object_point_cv.x, object_point_cv.y,
                                       object_point_cv.z);
    const Eigen::Vector3d optical_point =
        r_optical_object * object_point + t_optical_object;
    if (!(optical_point.allFinite() && optical_point.z() > 1e-6))
    {
      return false;
    }
  }
  return true;
}

inline Eigen::Matrix3d ReducedYawWorldRotation(double base_yaw_rad,
                                               int sign_index)
{
  static const std::array<Eigen::Vector3d, 4> kSigns = {
      Eigen::Vector3d(1.0, 1.0, 1.0),
      Eigen::Vector3d(-1.0, -1.0, 1.0),
      Eigen::Vector3d(-1.0, 1.0, -1.0),
      Eigen::Vector3d(1.0, -1.0, -1.0),
  };

  const Eigen::Matrix3d base =
      Eigen::AngleAxisd(base_yaw_rad, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Matrix3d signs = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d selected =
      kSigns[static_cast<std::size_t>(std::clamp(sign_index, 0, 3))];
  signs(0, 0) = selected.x();
  signs(1, 1) = selected.y();
  signs(2, 2) = selected.z();
  return base * signs;
}

inline double RotationToSpYaw(const Eigen::Matrix3d& r_world_object)
{
  return LibXR::CycleValue<double>(
      armor_tracker::QuaternionToYaw(LibXR::Quaternion<double>(r_world_object)) +
      M_PI);
}

inline bool FaceConstrainedProjectionEnabled()
{
  const char* env = std::getenv("XR_TRACKER_SP_FACE_CONSTRAINED_PROJECTION");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline double FaceConstrainedProjectionMaxReprojectionPx()
{
  const char* env =
      std::getenv("XR_TRACKER_SP_FACE_CONSTRAINED_MAX_REPROJ_PX");
  if (env == nullptr || env[0] == '\0')
  {
    return 3.0;
  }
  char* end = nullptr;
  const double value = std::strtod(env, &end);
  if (end == env || !std::isfinite(value))
  {
    return 3.0;
  }
  return std::max(0.0, value);
}

inline double FaceConstrainedProjectionScoreWeight()
{
  const char* env = std::getenv("XR_TRACKER_SP_FACE_CONSTRAINED_SCORE_WEIGHT");
  if (env == nullptr || env[0] == '\0')
  {
    return 0.08;
  }
  char* end = nullptr;
  const double value = std::strtod(env, &end);
  if (end == env || !std::isfinite(value))
  {
    return 0.08;
  }
  return std::max(0.0, value);
}

inline FaceConstrainedProjectionResult SolveFaceConstrainedProjection(
    const ArmorDetectorResult& armor, const CameraTypes::CameraInfo& camera_info,
    const LibXR::Transform<double>& camera_pose_world, double target_sp_yaw_rad)
{
  FaceConstrainedProjectionResult best{};
  if (armor.type == ArmorType::INVALID)
  {
    return best;
  }

  const std::vector<cv::Point2f> image_points = ArmorImagePointsInPnpOrder(armor);
  for (const auto& point : image_points)
  {
    if (!(std::isfinite(point.x) && std::isfinite(point.y)))
    {
      return best;
    }
  }

  const cv::Mat camera_matrix = BuildCameraMatrixCv(camera_info);
  const cv::Mat dist_coeffs = BuildDistCoeffsCv(camera_info);
  std::vector<cv::Point2f> normalized_points;
  cv::undistortPoints(image_points, normalized_points, camera_matrix, dist_coeffs);
  if (normalized_points.size() != image_points.size())
  {
    return best;
  }

  const std::vector<cv::Point3f> object_points = ArmorObjectPoints(armor.type);
  const Eigen::Matrix3d r_w_optical =
      camera_pose_world.rotation.ToRotationMatrix();
  const Eigen::Vector3d t_w_optical(camera_pose_world.translation.x(),
                                    camera_pose_world.translation.y(),
                                    camera_pose_world.translation.z());
  const double base_yaw_rad =
      armor_tracker::UnwrapYawNear(target_sp_yaw_rad - M_PI, 0.0);
  for (int sign_index = 0; sign_index < 4; ++sign_index)
  {
    const Eigen::Matrix3d r_world_object =
        ReducedYawWorldRotation(base_yaw_rad, sign_index);
    const double sp_yaw = armor_tracker::UnwrapYawNear(
        RotationToSpYaw(r_world_object), target_sp_yaw_rad);
    const Eigen::Matrix3d r_optical_object =
        r_w_optical.transpose() * r_world_object;
    Eigen::Vector3d t_optical_object = Eigen::Vector3d::Zero();
    if (!SolveTranslationForRotation(object_points, normalized_points,
                                     r_optical_object, t_optical_object))
    {
      continue;
    }

    cv::Mat rvec;
    cv::Rodrigues(EigenRotationToCvMat(r_optical_object), rvec);
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << t_optical_object.x(),
                    t_optical_object.y(), t_optical_object.z());
    const double reprojection_rmse = ReprojectionRmse(
        object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);
    if (!std::isfinite(reprojection_rmse))
    {
      continue;
    }
    if (!best.valid || reprojection_rmse < best.reprojection_rmse_px)
    {
      best.valid = true;
      best.world_position = r_w_optical * t_optical_object + t_w_optical;
      best.sp_yaw_rad = sp_yaw;
      best.reprojection_rmse_px = reprojection_rmse;
    }
  }
  return best;
}

inline bool SingleArmorModeEnabled()
{
  const char* env = std::getenv("XR_TRACKER_SINGLE_ARMOR_MODE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool MultiArmorFuseEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_ENABLE_MULTI_FUSE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool SymmetricGeometryEnabled()
{
  const char* env = std::getenv("XR_TRACKER_FORCE_SYMMETRIC_GEOMETRY");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool FaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline bool RelaxedFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_ENABLE_RELAXED_FACE_SWITCH");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool TempLostRecoveryEnabled()
{
  const char* env = std::getenv("XR_TRACKER_DISABLE_TEMP_LOST_RECOVER");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline bool OddFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_ODD_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline bool ViewPriorityEnabled()
{
  const char* env = std::getenv("XR_TRACKER_ENABLE_VIEW_PRIORITY");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline const char* ArmorTrackerArmorsTopicName()
{
  const char* env = std::getenv("XR_ARMORS_TOPIC_NAME");
  return (env != nullptr && env[0] != '\0') ? env : "armors_frame";
}

inline bool DirectionalFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_ENABLE_DIRECTIONAL_FACE_SWITCH");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline double ParseEnvDouble(const char* name, double default_value)
{
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0')
  {
    return default_value;
  }
  char* end = nullptr;
  const double parsed = std::strtod(env, &end);
  if (end == env || !std::isfinite(parsed))
  {
    return default_value;
  }
  return parsed;
}

inline bool EnvFlagEnabled(const char* name)
{
  const char* env = std::getenv(name);
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline std::uint32_t ParseEnvUint(const char* name, std::uint32_t default_value)
{
  return static_cast<std::uint32_t>(std::max(
      1.0, std::round(ParseEnvDouble(name, static_cast<double>(default_value)))));
}

inline double SingleArmorImageCenterGatePx()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SINGLE_ARMOR_IMAGE_GATE_PX", 180.0));
}

inline double SingleArmorAreaLogGate()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SINGLE_ARMOR_AREA_LOG_GATE", 0.80));
}

inline double FaceSwitchScoreDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_SCORE_DEADZONE", 0.15));
}

inline double FaceSwitchPositionDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_POSITION_DEADZONE", 0.05));
}

inline double FaceSwitchYawDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_YAW_DEADZONE", 0.35));
}

inline double FaceSwitchTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_TIMEOUT_SEC", 0.08));
}

inline bool IdAssistEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_IMAGE_ID_ASSIST");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

inline double IdAssistSameFaceCenterGatePx()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_ASSIST_CENTER_GATE_PX", 85.0));
}

inline double IdAssistSameFaceAreaLogGate()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_ASSIST_AREA_LOG_GATE", 0.45));
}

inline std::uint32_t IdTrackAppearHits()
{
  return ParseEnvUint("XR_TRACKER_ID_APPEAR_HITS", 2U);
}

inline double IdTrackAppearTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_APPEAR_TIMEOUT_SEC", 0.01));
}

inline std::uint32_t IdTrackTentativeMisses()
{
  return ParseEnvUint("XR_TRACKER_ID_TENTATIVE_MISSES", 2U);
}

inline double IdTrackTentativeTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_TENTATIVE_TIMEOUT_SEC", 0.03));
}

inline std::uint32_t IdTrackDisappearMisses()
{
  return ParseEnvUint("XR_TRACKER_ID_DISAPPEAR_MISSES", 3U);
}

inline double IdTrackDisappearTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_DISAPPEAR_TIMEOUT_SEC", 0.06));
}

inline bool SpStaticDeltaZEnabled()
{
  const char* env = std::getenv("XR_TRACKER_SP_STATIC_DZ");
  return env != nullptr && env[0] != '\0';
}

inline double SpStaticDeltaZ()
{
  return ParseEnvDouble("XR_TRACKER_SP_STATIC_DZ", 0.0);
}

inline double SpDeltaZInitialVariance()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_DZ_P0", 1e-4));
}

inline double SpDeltaZProcessVariance()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_DZ_Q", 0.0));
}

inline double SpPitchVarianceScale()
{
  return std::max(1e-3, ParseEnvDouble("XR_TRACKER_SP_PITCH_R_SCALE", 1.0));
}

inline double SpYpdDistanceVarianceScale()
{
  return std::max(1e-4,
                  ParseEnvDouble("XR_TRACKER_SP_DISTANCE_R_SCALE", 1.0));
}

inline double SpYpdArmorYawVarianceScale()
{
  return std::max(1e-4,
                  ParseEnvDouble("XR_TRACKER_SP_ARMOR_YAW_R_SCALE", 1.0));
}

inline bool SpDirectDeltaZEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_DIRECT_DZ");
}

inline double SpDirectDeltaZAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_DIRECT_DZ_ALPHA", 0.25), 0.0, 1.0);
}

inline double SpDirectDeltaZMaxAbs()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_DIRECT_DZ_MAX_ABS", 0.12));
}

inline bool SpCanonicalInitEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_ENABLE_CANONICAL_INIT");
}

inline std::uint32_t SpCanonicalInitMaxUpdates()
{
  return ParseEnvUint("XR_TRACKER_SP_CANONICAL_INIT_MAX_UPDATES", 16U);
}

inline double SpCanonicalInitMinHeight()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_CANONICAL_INIT_MIN_DZ", 0.015));
}

inline double SpCanonicalInitMaxAbsDz()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_CANONICAL_INIT_MAX_ABS_DZ", 0.12));
}

inline double SpCanonicalInitMaxScore()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_CANONICAL_INIT_MAX_SCORE", 2.5));
}

inline bool SpCanonicalInitPreferPositiveDz()
{
  return !EnvFlagEnabled("XR_TRACKER_SP_CANONICAL_INIT_ALLOW_NEG_DZ");
}

inline bool SpPairDeltaZEnabled()
{
  if (EnvFlagEnabled("XR_TRACKER_SP_DISABLE_PAIR_DZ"))
  {
    return false;
  }
  return EnvFlagEnabled("XR_TRACKER_SP_ENABLE_PAIR_DZ");
}

inline double SpPairDeltaZAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_PAIR_DZ_ALPHA", 0.40), 0.0, 1.0);
}

inline double SpPairDeltaZMinHeight()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_PAIR_DZ_MIN_HEIGHT", 0.015));
}

inline double SpPairDeltaZMaxAbs()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_PAIR_DZ_MAX_ABS", 0.12));
}

inline double SpPairDeltaZVariance()
{
  return std::max(1e-8, ParseEnvDouble("XR_TRACKER_SP_PAIR_DZ_VARIANCE", 4e-4));
}

inline double SpPairRecenterAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_PAIR_RECENTER_ALPHA", 0.5),
                    0.0, 1.0);
}

inline bool SpPairDualUpdateEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_PAIR_DUAL_UPDATE");
}

inline bool SpMeasurementAnchoredOutputEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_ENABLE_OUTPUT_MEAS_ANCHOR");
}

inline double SpOutputExtrapolateSeconds()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_SP_OUTPUT_EXTRAPOLATE_SEC", 0.0), 0.0, 0.050);
}

inline double SpMeasurementRecenterAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_MEAS_RECENTER_ALPHA", 1.0),
                    0.0, 1.0);
}

inline bool SpMeasurementRecenterQualityEnabled()
{
  const char* env = std::getenv("XR_TRACKER_SP_QUALITY_RECENTER");
  if (env != nullptr)
  {
    return env[0] != '\0' && env[0] != '0';
  }
  return false;
}

inline double SpMeasurementRecenterAlphaGood()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_MEAS_RECENTER_ALPHA_GOOD", 0.45),
                    0.0, 1.0);
}

inline double SpMeasurementRecenterAlphaBad()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_MEAS_RECENTER_ALPHA_BAD", 0.08),
                    0.0, 1.0);
}

inline double SpMeasurementRecenterScoreGood()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SP_MEAS_RECENTER_SCORE_GOOD", 0.45));
}

inline double SpMeasurementRecenterScoreBad()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SP_MEAS_RECENTER_SCORE_BAD", 1.30));
}

inline double SpMeasurementRecenterYawGood()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SP_MEAS_RECENTER_YAW_GOOD", 0.05));
}

inline double SpMeasurementRecenterYawBad()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SP_MEAS_RECENTER_YAW_BAD", 0.35));
}

inline double SpMeasurementRecenterXyzGood()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SP_MEAS_RECENTER_XYZ_GOOD", 0.03));
}

inline double SpMeasurementRecenterXyzBad()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SP_MEAS_RECENTER_XYZ_BAD", 0.10));
}

inline double SpMeasurementPositionAnchorAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_MEAS_POS_ANCHOR_ALPHA", 0.0),
                    0.0, 1.0);
}

inline double SpMeasurementPositionAnchorXyzBad()
{
  return std::max(
      1e-6, ParseEnvDouble("XR_TRACKER_SP_MEAS_POS_ANCHOR_XYZ_BAD", 0.12));
}

inline bool SpXyzMeasurementUpdateEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_XYZ_UPDATE");
}

inline double SpXyzMeasurementRFactor(double default_value)
{
  return std::max(1e-4,
                  ParseEnvDouble("XR_TRACKER_SP_XYZ_R_FACTOR", default_value));
}

inline double SpXyzMeasurementYawVariance(double default_value)
{
  return std::max(1e-6,
                  ParseEnvDouble("XR_TRACKER_SP_XYZ_YAW_R", default_value));
}

inline bool SpXyzMeasurementFullGeometryEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_XYZ_FULL_GEOMETRY");
}
}  // namespace armor_tracker_detail
