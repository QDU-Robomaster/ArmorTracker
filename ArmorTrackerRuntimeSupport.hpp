#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

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
