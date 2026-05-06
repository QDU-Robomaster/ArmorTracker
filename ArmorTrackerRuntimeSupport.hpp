#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "ArmorTrackerCommon.hpp"
#include "CameraBase.hpp"

namespace armor_tracker_detail
{
enum class CameraPoseMode : uint8_t
{
  FULL,
  STATIC_ONLY,
  YAW_ONLY,
  RELATIVE_IMU,
  RELATIVE_YAW_ONLY,
};

struct CameraPoseRuntime
{
  bool relative_rotation_initialized = false;
  LibXR::Quaternion<double> initial_camera_rotation_inverse{};
  bool relative_yaw_initialized = false;
  double initial_camera_yaw = 0.0;
};

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

inline CameraPoseMode ParseCameraPoseMode()
{
  const char* env = std::getenv("XR_TRACKER_CAMERA_POSE_MODE");
  if (env == nullptr || env[0] == '\0' || std::strcmp(env, "full") == 0 ||
      std::strcmp(env, "default") == 0)
  {
    return CameraPoseMode::FULL;
  }
  if (std::strcmp(env, "static_only") == 0 || std::strcmp(env, "static") == 0)
  {
    return CameraPoseMode::STATIC_ONLY;
  }
  if (std::strcmp(env, "yaw_only") == 0 || std::strcmp(env, "yaw") == 0)
  {
    return CameraPoseMode::YAW_ONLY;
  }
  if (std::strcmp(env, "relative_imu") == 0)
  {
    return CameraPoseMode::RELATIVE_IMU;
  }
  if (std::strcmp(env, "relative_yaw_only") == 0 ||
      std::strcmp(env, "relative_yaw") == 0)
  {
    return CameraPoseMode::RELATIVE_YAW_ONLY;
  }
  return CameraPoseMode::FULL;
}

inline LibXR::Quaternion<double> CameraYawOnlyRotation(
    const LibXR::Quaternion<double>& camera_rotation)
{
  const Eigen::Vector3d euler =
      LibXR::RotationMatrix<double>(camera_rotation.ToRotationMatrix()).ToEulerAngle();
  return LibXR::Quaternion<double>(
      LibXR::EulerAngle<double>(0.0, 0.0, euler.z()).ToQuaternion());
}

inline double CameraYaw(const LibXR::Quaternion<double>& camera_rotation)
{
  const Eigen::Vector3d euler =
      LibXR::RotationMatrix<double>(camera_rotation.ToRotationMatrix()).ToEulerAngle();
  return euler.z();
}

inline LibXR::Transform<double> ComposeCameraPose(
    const LibXR::Quaternion<double>& dynamic_rotation,
    const LibXR::Position<double>& camera_translation,
    const LibXR::Transform<double>& gimbal_to_camera_transform_static)
{
  return LibXR::Transform<double>(dynamic_rotation, camera_translation) +
         gimbal_to_camera_transform_static;
}

inline LibXR::Transform<double> ArmorTrackerCameraRotationToTrackerWorldPose(
    const LibXR::Quaternion<double>& camera_rotation,
    const LibXR::Position<double>& camera_translation,
    const LibXR::Transform<double>& gimbal_to_camera_transform_static,
    CameraPoseRuntime& runtime)
{
  switch (ParseCameraPoseMode())
  {
    case CameraPoseMode::STATIC_ONLY:
      return ComposeCameraPose(LibXR::Quaternion<double>(), camera_translation,
                               gimbal_to_camera_transform_static);
    case CameraPoseMode::YAW_ONLY:
      return ComposeCameraPose(CameraYawOnlyRotation(camera_rotation),
                               camera_translation,
                               gimbal_to_camera_transform_static);
    case CameraPoseMode::RELATIVE_IMU:
    {
      if (!runtime.relative_rotation_initialized)
      {
        runtime.initial_camera_rotation_inverse = -camera_rotation;
        runtime.relative_rotation_initialized = true;
      }
      return ComposeCameraPose(
          runtime.initial_camera_rotation_inverse * camera_rotation,
          camera_translation, gimbal_to_camera_transform_static);
    }
    case CameraPoseMode::RELATIVE_YAW_ONLY:
    {
      const double yaw = CameraYaw(camera_rotation);
      if (!runtime.relative_yaw_initialized)
      {
        runtime.initial_camera_yaw = yaw;
        runtime.relative_yaw_initialized = true;
      }
      return ComposeCameraPose(
          LibXR::Quaternion<double>(
              LibXR::EulerAngle<double>(0.0, 0.0,
                                        yaw - runtime.initial_camera_yaw)
                  .ToQuaternion()),
          camera_translation, gimbal_to_camera_transform_static);
    }
    case CameraPoseMode::FULL:
    default:
      return ComposeCameraPose(camera_rotation, camera_translation,
                               gimbal_to_camera_transform_static);
  }
}

inline bool SingleArmorModeEnabled()
{
  const char* env = std::getenv("XR_TRACKER_SINGLE_ARMOR_MODE");
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
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_TIMEOUT_SEC", 0.0));
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
                  ParseEnvDouble("XR_TRACKER_SP_DISTANCE_R_SCALE", 0.1));
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

inline double SpPairGeometryMinDeterminant()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_SP_PAIR_GEOMETRY_MIN_DET", 0.35), 0.0, 1.0);
}

inline double SpPairGeometryMaxFitError()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_SP_PAIR_GEOMETRY_MAX_FIT_ERROR", 0.035));
}

inline double SpPairGeometryMaxCenterShift()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_SP_PAIR_GEOMETRY_MAX_CENTER_SHIFT", 0.60));
}

inline double SpPairGeometryMaxRadiusShift()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_SP_PAIR_GEOMETRY_MAX_RADIUS_SHIFT", 0.30));
}

inline double SpPairGeometryCenterVariance()
{
  const double sigma =
      std::max(1e-4, ParseEnvDouble("XR_TRACKER_SP_PAIR_GEOMETRY_CENTER_SIGMA", 0.025));
  return sigma * sigma;
}

inline double SpPairGeometryYawVariance()
{
  const double sigma =
      std::max(1e-4, ParseEnvDouble("XR_TRACKER_SP_PAIR_GEOMETRY_YAW_SIGMA", 0.025));
  return sigma * sigma;
}

inline double SpPairGeometryRadiusVariance()
{
  const double sigma =
      std::max(1e-4, ParseEnvDouble("XR_TRACKER_SP_PAIR_GEOMETRY_RADIUS_SIGMA", 0.100));
  return sigma * sigma;
}

inline double SpPairGeometryFallbackMaxScore()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_SP_PAIR_GEOMETRY_FALLBACK_MAX_SCORE", 1.20));
}

inline double SpPairGeometryCovarianceFloor()
{
  const double sigma =
      std::max(1e-4, ParseEnvDouble("XR_TRACKER_SP_PAIR_GEOMETRY_STATE_SIGMA", 0.050));
  return sigma * sigma;
}

inline double SpDeltaRadiusShrinkAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_DELTA_R_SHRINK_ALPHA", 0.03),
                    0.0, 1.0);
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

inline bool SpPairDualUpdateEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_PAIR_DUAL_UPDATE");
}

inline bool SpMeasurementAnchoredOutputEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_ENABLE_OUTPUT_MEAS_ANCHOR");
}

inline bool SpFixedPoseYawOptimizeEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_ENABLE_FIXED_POSE_YAW_OPT");
}

inline double SpFixedPoseYawPitchDeg()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_SP_FIXED_POSE_YAW_PITCH_DEG", 15.0), 0.0, 45.0);
}

inline double SpFixedPoseYawRangeDeg()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_SP_FIXED_POSE_YAW_RANGE_DEG", 70.0), 1.0, 180.0);
}

inline double SpFixedPoseYawCoarseStepDeg()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_SP_FIXED_POSE_YAW_COARSE_STEP_DEG", 2.0),
      0.2, 20.0);
}

inline double SpFixedPoseYawFineStepDeg()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_SP_FIXED_POSE_YAW_FINE_STEP_DEG", 0.2),
      0.05, 5.0);
}

inline double SpFixedPoseYawMinGainPx()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_SP_FIXED_POSE_YAW_MIN_GAIN_PX", 0.05));
}

inline bool SpCenterMotionObserverEnabled()
{
  return !EnvFlagEnabled("XR_TRACKER_SP_DISABLE_CENTER_MOTION_OBSERVER");
}

inline bool SpCenterMotionObserverRadialVelocityEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_SP_CENTER_MOTION_OBSERVER_RADIAL");
}

inline bool SpYawRateObserverEnabled()
{
  if (EnvFlagEnabled("XR_TRACKER_SP_DISABLE_YAW_RATE_OBSERVER"))
  {
    return false;
  }
  return EnvFlagEnabled("XR_TRACKER_SP_ENABLE_YAW_RATE_OBSERVER");
}

inline double SpYawRateObserverAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_YAW_RATE_OBSERVER_ALPHA", 0.08),
                    0.0, 1.0);
}

inline double SpYawRateObserverTau()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_YAW_RATE_OBSERVER_TAU", 0.020),
                    0.001, 0.500);
}

inline double SpYawRateObserverBlend()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_SP_YAW_RATE_OBSERVER_BLEND", 1.0),
                    0.0, 1.0);
}

inline double SpYawRateObserverMaxRaw()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_YAW_RATE_OBSERVER_MAX_RAW", 20.0));
}

inline double SpYawRateObserverMaxBlendDelta()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_SP_YAW_RATE_OBSERVER_MAX_BLEND_DELTA", 0.8));
}

inline std::uint32_t SpYawRateObserverMinSamples()
{
  return ParseEnvUint("XR_TRACKER_SP_YAW_RATE_OBSERVER_MIN_SAMPLES", 4U);
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
