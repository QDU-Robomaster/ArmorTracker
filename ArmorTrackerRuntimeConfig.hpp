#pragma once

/**
 * @file ArmorTrackerRuntimeConfig.hpp
 * @brief ArmorTracker 运行时环境开关、相机位姿转换和参数读取工具。
 *
 * 本文件集中封装环境变量解析、调试 profile 开关和相机到 tracker 世界系的位姿转换，
 * 让主 tracker 类只消费已解析的布尔值和阈值。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "ArmorTrackerCommon.hpp"
#include "CameraBase.hpp"

namespace armor_tracker_detail
{
/**
 * @brief tracker 使用相机动态姿态的模式。
 */
enum class CameraPoseMode : uint8_t
{
  FULL,              ///< 使用完整相机姿态。
  STATIC_ONLY,       ///< 只使用静态外参。
  YAW_ONLY,          ///< 只保留相机 yaw。
  RELATIVE_IMU,      ///< 使用相对初始 IMU 的完整姿态。
  RELATIVE_YAW_ONLY, ///< 只使用相对初始 yaw。
};

/**
 * @brief 相对相机姿态模式需要保持的运行时初值。
 */
struct CameraPoseRuntime
{
  bool relative_rotation_initialized = false;   ///< 相对完整姿态初值是否已建立。
  LibXR::Quaternion<double> initial_camera_rotation_inverse{}; ///< 初始姿态逆。
  bool relative_yaw_initialized = false;        ///< 相对 yaw 初值是否已建立。
  double initial_camera_yaw = 0.0;              ///< 初始相机 yaw。
};

/**
 * @brief tracker PnP 重投影使用的畸变系数打包结果。
 */
struct TrackerPnPDistCoeffs
{
  std::array<double, 8> values{};       ///< OpenCV distCoeffs 数组。
  uint8_t size = 0;                     ///< 有效畸变系数数量。
  bool requires_undistort_first = false; ///< 当前模型是否不能直接交给 projectPoints。
};

/**
 * @brief 从 CameraInfo 编译期常量构建 OpenCV PnP 畸变参数。
 */
template <typename CameraInfoT>
inline constexpr TrackerPnPDistCoeffs BuildTrackerPnPDistCoeffs(
    const CameraInfoT& info)
{
  TrackerPnPDistCoeffs dc{};
  switch (info.distortion_model)
  {
    case CameraTypes::DistortionModel::NONE:
      break;
    case CameraTypes::DistortionModel::PLUMB_BOB:
      dc.values[0] = info.distortion_coefficients[0];
      dc.values[1] = info.distortion_coefficients[1];
      dc.values[2] = info.distortion_coefficients[2];
      dc.values[3] = info.distortion_coefficients[3];
      dc.values[4] = info.distortion_coefficients[4];
      dc.size = 5;
      break;
    case CameraTypes::DistortionModel::RATIONAL_POLYNOMIAL:
      dc.values[0] = info.distortion_coefficients[0];
      dc.values[1] = info.distortion_coefficients[1];
      dc.values[2] = info.distortion_coefficients[2];
      dc.values[3] = info.distortion_coefficients[3];
      dc.values[4] = info.distortion_coefficients[4];
      dc.values[5] = info.distortion_coefficients[5];
      dc.values[6] = info.distortion_coefficients[6];
      dc.values[7] = info.distortion_coefficients[7];
      dc.size = 8;
      break;
    default:
      dc.requires_undistort_first = true;
      break;
  }
  return dc;
}

/**
 * @brief 将同步帧里的 wxyz 数组转换为 LibXR 四元数。
 */
inline LibXR::Quaternion<double> PackedCameraRotation(
    const std::array<float, 4>& rotation_wxyz)
{
  return LibXR::Quaternion<double>(rotation_wxyz[0], rotation_wxyz[1],
                                   rotation_wxyz[2], rotation_wxyz[3]);
}

/**
 * @brief 将同步帧里的 xyz 数组转换为 LibXR 位置。
 */
inline LibXR::Position<double> PackedCameraTranslation(
    const std::array<float, 3>& translation_xyz)
{
  return LibXR::Position<double>(translation_xyz[0], translation_xyz[1],
                                 translation_xyz[2]);
}

/**
 * @brief 解析 XR_TRACKER_CAMERA_POSE_MODE 环境变量。
 */
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

/**
 * @brief 从完整相机姿态中提取 yaw-only 四元数。
 */
inline LibXR::Quaternion<double> CameraYawOnlyRotation(
    const LibXR::Quaternion<double>& camera_rotation)
{
  const Eigen::Vector3d euler =
      LibXR::RotationMatrix<double>(camera_rotation.ToRotationMatrix()).ToEulerAngle();
  return LibXR::Quaternion<double>(
      LibXR::EulerAngle<double>(0.0, 0.0, euler.z()).ToQuaternion());
}

/**
 * @brief 从相机四元数中提取 yaw。
 */
inline double CameraYaw(const LibXR::Quaternion<double>& camera_rotation)
{
  const Eigen::Vector3d euler =
      LibXR::RotationMatrix<double>(camera_rotation.ToRotationMatrix()).ToEulerAngle();
  return euler.z();
}

/**
 * @brief 组合动态相机旋转、相机平移和静态云台到相机外参。
 */
inline LibXR::Transform<double> ComposeCameraPose(
    const LibXR::Quaternion<double>& dynamic_rotation,
    const LibXR::Position<double>& camera_translation,
    const LibXR::Transform<double>& gimbal_to_camera_transform_static)
{
  return LibXR::Transform<double>(dynamic_rotation, camera_translation) +
         gimbal_to_camera_transform_static;
}

/**
 * @brief 根据运行时模式将相机姿态转换到 tracker 世界系。
 */
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

/**
 * @brief 是否启用单装甲模式。
 */
inline bool SingleArmorModeEnabled()
{
  const char* env = std::getenv("XR_TRACKER_SINGLE_ARMOR_MODE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

/**
 * @brief 是否强制使用对称车体几何。
 */
inline bool SymmetricGeometryEnabled()
{
  const char* env = std::getenv("XR_TRACKER_FORCE_SYMMETRIC_GEOMETRY");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

/**
 * @brief 是否允许换面。
 */
inline bool FaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

/**
 * @brief 是否允许放宽阈值换面。
 */
inline bool RelaxedFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_ENABLE_RELAXED_FACE_SWITCH");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

/**
 * @brief 是否允许 TEMP_LOST 状态快速恢复。
 */
inline bool TempLostRecoveryEnabled()
{
  const char* env = std::getenv("XR_TRACKER_DISABLE_TEMP_LOST_RECOVER");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

/**
 * @brief 是否允许切到奇数高低面。
 */
inline bool OddFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_ODD_FACE_SWITCH");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

/**
 * @brief 是否启用视角优先评分。
 */
inline bool ViewPriorityEnabled()
{
  const char* env = std::getenv("XR_TRACKER_ENABLE_VIEW_PRIORITY");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

/**
 * @brief 返回 ArmorDetector 结果 topic 名称。
 */
inline const char* ArmorTrackerArmorsTopicName()
{
  const char* env = std::getenv("XR_ARMORS_TOPIC_NAME");
  return (env != nullptr && env[0] != '\0') ? env : "armors_frame";
}

/**
 * @brief 是否按 yaw 角速度方向限制换面方向。
 */
inline bool DirectionalFaceSwitchEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_ENABLE_DIRECTIONAL_FACE_SWITCH");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

/**
 * @brief 解析 double 环境变量。
 */
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

/**
 * @brief 判断环境变量是否被设置为启用状态。
 */
inline bool EnvFlagEnabled(const char* name)
{
  const char* env = std::getenv(name);
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

/**
 * @brief 是否启用 detector 观测质量评分。
 */
inline bool ObservationQualityEnabled()
{
  if (EnvFlagEnabled("XR_TRACKER_DISABLE_OBSERVATION_QUALITY"))
  {
    return false;
  }
  return true;
}

/**
 * @brief 稳定观测允许的最大重投影误差。
 */
inline double ObservationStableMaxReprojectionPx()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_STABLE_REPROJ_PX", 1.8));
}

/**
 * @brief 稳定观测要求的最小图像面积。
 */
inline double ObservationStableMinAreaPx()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_STABLE_AREA_PX", 60.0));
}

/**
 * @brief 稳定观测要求的最小 detector 置信度。
 */
inline double ObservationStableMinConfidence()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_STABLE_CONFIDENCE", 0.0), 0.0, 1.0);
}

/**
 * @brief 观测质量惩罚在候选分数中的权重。
 */
inline double ObservationQualityScoreWeight()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_QUALITY_SCORE_WEIGHT", 0.55));
}

/**
 * @brief confirmed 图像 track 的候选分数奖励。
 */
inline double ObservationConfirmedTrackBonus()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_CONFIRMED_TRACK_BONUS", 0.24));
}

/**
 * @brief 初始化是否必须等待稳定观测。
 */
inline bool InitRequiresStableObservation()
{
  return !EnvFlagEnabled("XR_TRACKER_INIT_ALLOW_UNSTABLE_OBSERVATION");
}

/**
 * @brief 匹配 yaw 时是否允许 pi 二义性。
 */
inline bool MatchYawAllowPiAmbiguityEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_ENABLE_PNP_PI_YAW_FOLD");
}

/**
 * @brief 解析无符号整数环境变量。
 */
inline std::uint32_t ParseEnvUint(const char* name, std::uint32_t default_value)
{
  return static_cast<std::uint32_t>(std::max(
      1.0, std::round(ParseEnvDouble(name, static_cast<double>(default_value)))));
}

/**
 * @brief 单装甲模式图像中心门限。
 */
inline double SingleArmorImageCenterGatePx()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SINGLE_ARMOR_IMAGE_GATE_PX", 180.0));
}

/**
 * @brief 单装甲模式面积比例对数门限。
 */
inline double SingleArmorAreaLogGate()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_SINGLE_ARMOR_AREA_LOG_GATE", 0.80));
}

/**
 * @brief 换面分数优势死区。
 */
inline double FaceSwitchScoreDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_SCORE_DEADZONE", 0.15));
}

/**
 * @brief 换面位置优势死区。
 */
inline double FaceSwitchPositionDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_POSITION_DEADZONE", 0.05));
}

/**
 * @brief 换面 yaw 优势死区。
 */
inline double FaceSwitchYawDeadzone()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_YAW_DEADZONE", 0.35));
}

/**
 * @brief 换面冷却时间阈值。
 */
inline double FaceSwitchTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_FACE_SWITCH_TIMEOUT_SEC", 0.0));
}

/**
 * @brief 是否启用图像 track id 辅助。
 */
inline bool IdAssistEnabled()
{
  if (SingleArmorModeEnabled())
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_DISABLE_IMAGE_ID_ASSIST");
  return !(env != nullptr && env[0] != '\0' && env[0] != '0');
}

/**
 * @brief ID 辅助同面保持的图像中心门限。
 */
inline double IdAssistSameFaceCenterGatePx()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_ASSIST_CENTER_GATE_PX", 85.0));
}

/**
 * @brief ID 辅助同面保持的面积比例对数门限。
 */
inline double IdAssistSameFaceAreaLogGate()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_ASSIST_AREA_LOG_GATE", 0.45));
}

/**
 * @brief 图像 track 进入 confirmed 所需命中数。
 */
inline std::uint32_t IdTrackAppearHits()
{
  return ParseEnvUint("XR_TRACKER_ID_APPEAR_HITS", 2U);
}

/**
 * @brief 图像 track 进入 confirmed 所需最短时间。
 */
inline double IdTrackAppearTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_APPEAR_TIMEOUT_SEC", 0.01));
}

/**
 * @brief tentative 图像 track 删除所需丢失数。
 */
inline std::uint32_t IdTrackTentativeMisses()
{
  return ParseEnvUint("XR_TRACKER_ID_TENTATIVE_MISSES", 2U);
}

/**
 * @brief tentative 图像 track 删除所需最短丢失时间。
 */
inline double IdTrackTentativeTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_TENTATIVE_TIMEOUT_SEC", 0.03));
}

/**
 * @brief confirmed 图像 track 删除所需丢失数。
 */
inline std::uint32_t IdTrackDisappearMisses()
{
  return ParseEnvUint("XR_TRACKER_ID_DISAPPEAR_MISSES", 3U);
}

/**
 * @brief confirmed 图像 track 删除所需最短丢失时间。
 */
inline double IdTrackDisappearTimeoutSec()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_ID_DISAPPEAR_TIMEOUT_SEC", 0.06));
}

/**
 * @brief 是否使用固定静态 dz。
 */
inline bool VehicleStaticDeltaZEnabled()
{
  const char* env = std::getenv("XR_TRACKER_MODEL_STATIC_DZ");
  return env != nullptr && env[0] != '\0';
}

/**
 * @brief 静态高低差值。
 */
inline double VehicleStaticDeltaZ()
{
  return ParseEnvDouble("XR_TRACKER_MODEL_STATIC_DZ", 0.0);
}

/**
 * @brief DELTA_Z 初始协方差。
 */
inline double VehicleDeltaZInitialVariance()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_DZ_P0", 1e-4));
}

/**
 * @brief DELTA_Z 过程噪声方差。
 */
inline double VehicleDeltaZProcessVariance()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_DZ_Q", 0.0));
}

/**
 * @brief pitch 观测噪声缩放。
 */
inline double VehiclePitchVarianceScale()
{
  return std::max(1e-3, ParseEnvDouble("XR_TRACKER_MODEL_PITCH_R_SCALE", 1.0));
}

/**
 * @brief ypd 距离观测噪声缩放。
 */
inline double VehicleYpdDistanceVarianceScale()
{
  return std::max(1e-4,
                  ParseEnvDouble("XR_TRACKER_MODEL_DISTANCE_R_SCALE", 0.1));
}

/**
 * @brief 装甲 yaw 观测噪声缩放。
 */
inline double VehicleYpdArmorYawVarianceScale()
{
  return std::max(1e-4,
                  ParseEnvDouble("XR_TRACKER_MODEL_ARMOR_YAW_R_SCALE", 1.0));
}

/**
 * @brief 是否启用 direct dz 低通写入。
 */
inline bool VehicleDirectDeltaZEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_MODEL_DIRECT_DZ");
}

/**
 * @brief direct dz 低通系数。
 */
inline double VehicleDirectDeltaZAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_MODEL_DIRECT_DZ_ALPHA", 0.25), 0.0, 1.0);
}

/**
 * @brief direct dz 最大绝对值。
 */
inline double VehicleDirectDeltaZMaxAbs()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_DIRECT_DZ_MAX_ABS", 0.12));
}

/**
 * @brief 是否启用初始化阶段 canonical 高低面解析。
 */
inline bool VehicleCanonicalInitEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_CANONICAL_INIT");
}

/**
 * @brief canonical 初始化最大更新帧数。
 */
inline std::uint32_t VehicleCanonicalInitMaxUpdates()
{
  return ParseEnvUint("XR_TRACKER_MODEL_CANONICAL_INIT_MAX_UPDATES", 16U);
}

/**
 * @brief canonical 初始化接受高低差的最小高度。
 */
inline double VehicleCanonicalInitMinHeight()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_CANONICAL_INIT_MIN_DZ", 0.015));
}

/**
 * @brief canonical 初始化接受高低差的最大绝对值。
 */
inline double VehicleCanonicalInitMaxAbsDz()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_CANONICAL_INIT_MAX_ABS_DZ", 0.12));
}

/**
 * @brief 返回 canonical 初始化允许的最大平均匹配分。
 */
inline double VehicleCanonicalInitMaxScore()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_CANONICAL_INIT_MAX_SCORE", 2.5));
}

/**
 * @brief canonical 初始化是否优先选择奇数面高于偶数面的相位。
 */
inline bool VehicleCanonicalInitPreferPositiveDz()
{
  return !EnvFlagEnabled("XR_TRACKER_MODEL_CANONICAL_INIT_ALLOW_NEG_DZ");
}

/**
 * @brief 环境变量层面是否强制启用双装甲高低差观测。
 */
inline bool VehiclePairDeltaZEnabled()
{
  if (EnvFlagEnabled("XR_TRACKER_MODEL_DISABLE_PAIR_DZ"))
  {
    return false;
  }
  return EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_PAIR_DZ");
}

/**
 * @brief 返回双装甲几何射线求交的最小行列式阈值。
 */
inline double VehiclePairGeometryMinDeterminant()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_MODEL_PAIR_GEOMETRY_MIN_DET", 0.35), 0.0, 1.0);
}

/**
 * @brief 返回双装甲几何拟合中心误差上限。
 */
inline double VehiclePairGeometryMaxFitError()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_GEOMETRY_MAX_FIT_ERROR", 0.035));
}

/**
 * @brief 返回双装甲几何允许修正整车中心的最大距离。
 */
inline double VehiclePairGeometryMaxCenterShift()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_GEOMETRY_MAX_CENTER_SHIFT", 0.60));
}

/**
 * @brief 返回双装甲几何允许修正半径的最大距离。
 */
inline double VehiclePairGeometryMaxRadiusShift()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_GEOMETRY_MAX_RADIUS_SHIFT", 0.30));
}

/**
 * @brief 返回双装甲中心观测写入 EKF 的方差。
 */
inline double VehiclePairGeometryCenterVariance()
{
  const double sigma =
      std::max(1e-4, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_GEOMETRY_CENTER_SIGMA", 0.025));
  return sigma * sigma;
}

/**
 * @brief 返回双装甲 yaw 观测写入 EKF 的方差。
 */
inline double VehiclePairGeometryYawVariance()
{
  const double sigma =
      std::max(1e-4, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_GEOMETRY_YAW_SIGMA", 0.025));
  return sigma * sigma;
}

/**
 * @brief 返回双装甲半径观测写入 EKF 的方差。
 */
inline double VehiclePairGeometryRadiusVariance()
{
  const double sigma =
      std::max(1e-4, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_GEOMETRY_RADIUS_SIGMA", 0.100));
  return sigma * sigma;
}

/**
 * @brief 返回未匹配帧可接受双装甲几何回退更新的最高分。
 */
inline double VehiclePairGeometryFallbackMaxScore()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_GEOMETRY_FALLBACK_MAX_SCORE", 1.20));
}

/**
 * @brief 返回半径状态在双装甲更新前抬高协方差的下限。
 */
inline double VehiclePairGeometryCovarianceFloor()
{
  const double sigma =
      std::max(1e-4, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_GEOMETRY_STATE_SIGMA", 0.050));
  return sigma * sigma;
}

/**
 * @brief 返回无双装甲显式观测时长短半径差的收缩系数。
 */
inline double VehicleDeltaRadiusShrinkAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_MODEL_DELTA_R_SHRINK_ALPHA", 0.03),
                    0.0, 1.0);
}

/**
 * @brief 返回双装甲高低差观测的最小有效高度。
 */
inline double VehiclePairDeltaZMinHeight()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_DZ_MIN_HEIGHT", 0.015));
}

/**
 * @brief 返回双装甲高低差观测的绝对值钳位上限。
 */
inline double VehiclePairDeltaZMaxAbs()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_DZ_MAX_ABS", 0.12));
}

/**
 * @brief 返回双装甲高低差观测写入 EKF 的方差。
 */
inline double VehiclePairDeltaZVariance()
{
  return std::max(1e-8, ParseEnvDouble("XR_TRACKER_MODEL_PAIR_DZ_VARIANCE", 4e-4));
}

/**
 * @brief 是否对双装甲中的左右两面都执行单面观测更新。
 */
inline bool VehiclePairDualUpdateEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_MODEL_PAIR_DUAL_UPDATE");
}

/**
 * @brief 单面观测更新时是否冻结四装甲高低差状态。
 */
inline bool VehicleFreezeSingleObservationDeltaZEnabled()
{
  if (EnvFlagEnabled("XR_TRACKER_MODEL_DISABLE_FREEZE_SINGLE_DZ"))
  {
    return false;
  }
  const char* env = std::getenv("XR_TRACKER_MODEL_FREEZE_SINGLE_DZ");
  if (env != nullptr)
  {
    return env[0] != '\0' && env[0] != '0';
  }
  return true;
}

/**
 * @brief 环境变量层面是否启用测量面锚定输出。
 */
inline bool VehicleMeasurementAnchoredOutputEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_OUTPUT_MEAS_ANCHOR");
}

/**
 * @brief 返回输出锚定时径向方向的滤波系数。
 */
inline double VehicleOutputMeasAnchorAlpha()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_MODEL_OUTPUT_MEAS_ANCHOR_ALPHA", 0.25),
      0.0, 1.0);
}

/**
 * @brief 返回输出锚定时横向方向的滤波系数。
 */
inline double VehicleOutputMeasAnchorLateralAlpha()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_MODEL_OUTPUT_MEAS_ANCHOR_LATERAL_ALPHA", 1.0),
      0.0, 1.0);
}

/**
 * @brief 返回输出锚定单帧径向步长上限。
 */
inline double VehicleOutputMeasAnchorMaxStep()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_MODEL_OUTPUT_MEAS_ANCHOR_MAX_STEP", 0.018));
}

/**
 * @brief 返回输出锚定总平移量上限。
 */
inline double VehicleOutputMeasAnchorMaxDelta()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_MODEL_OUTPUT_MEAS_ANCHOR_MAX_DELTA", 0.180));
}

/**
 * @brief 是否启用固定俯仰角的装甲 yaw 重投影优化。
 */
inline bool VehicleFixedPoseYawOptimizeEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_FIXED_POSE_YAW_OPT");
}

/**
 * @brief 返回固定俯仰 yaw 优化的候选 pitch 角度，单位 deg。
 */
inline double VehicleFixedPoseYawPitchDeg()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_MODEL_FIXED_POSE_YAW_PITCH_DEG", 15.0), 0.0, 45.0);
}

/**
 * @brief 返回固定俯仰 yaw 优化的粗搜索范围，单位 deg。
 */
inline double VehicleFixedPoseYawRangeDeg()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_MODEL_FIXED_POSE_YAW_RANGE_DEG", 70.0), 1.0, 180.0);
}

/**
 * @brief 返回固定俯仰 yaw 优化的粗搜索步长，单位 deg。
 */
inline double VehicleFixedPoseYawCoarseStepDeg()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_MODEL_FIXED_POSE_YAW_COARSE_STEP_DEG", 2.0),
      0.2, 20.0);
}

/**
 * @brief 返回固定俯仰 yaw 优化的细搜索步长，单位 deg。
 */
inline double VehicleFixedPoseYawFineStepDeg()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_MODEL_FIXED_POSE_YAW_FINE_STEP_DEG", 0.2),
      0.05, 5.0);
}

/**
 * @brief 返回固定俯仰 yaw 优化需要带来的最小重投影收益。
 */
inline double VehicleFixedPoseYawMinGainPx()
{
  return std::max(
      0.0, ParseEnvDouble("XR_TRACKER_MODEL_FIXED_POSE_YAW_MIN_GAIN_PX", 0.05));
}

/**
 * @brief 是否启用中心速度观测器作为输出速度来源。
 */
inline bool VehicleCenterMotionObserverEnabled()
{
  return !EnvFlagEnabled("XR_TRACKER_MODEL_DISABLE_CENTER_MOTION_OBSERVER");
}

/**
 * @brief 中心速度观测器是否保留径向速度分量。
 */
inline bool VehicleCenterMotionObserverRadialVelocityEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_MODEL_CENTER_MOTION_OBSERVER_RADIAL");
}

/**
 * @brief 是否启用可见面 yaw 差分角速度观测器。
 */
inline bool VehicleYawRateObserverEnabled()
{
  if (EnvFlagEnabled("XR_TRACKER_MODEL_DISABLE_YAW_RATE_OBSERVER"))
  {
    return false;
  }
  return EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_YAW_RATE_OBSERVER");
}

/**
 * @brief 返回旧版 yaw rate 观测器平滑系数。
 */
inline double VehicleYawRateObserverAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_MODEL_YAW_RATE_OBSERVER_ALPHA", 0.08),
                    0.0, 1.0);
}

/**
 * @brief 返回 yaw rate 观测器一阶滤波时间常数。
 */
inline double VehicleYawRateObserverTau()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_MODEL_YAW_RATE_OBSERVER_TAU", 0.020),
                    0.001, 0.500);
}

/**
 * @brief 返回 yaw rate 观测器与 EKF 角速度的输出混合比例。
 */
inline double VehicleYawRateObserverBlend()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_MODEL_YAW_RATE_OBSERVER_BLEND", 1.0),
                    0.0, 1.0);
}

/**
 * @brief 返回 yaw rate 单帧原始差分可接受的最大绝对值。
 */
inline double VehicleYawRateObserverMaxRaw()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_YAW_RATE_OBSERVER_MAX_RAW", 20.0));
}

/**
 * @brief 返回 yaw rate 观测值与 EKF 角速度允许混合的最大差值。
 */
inline double VehicleYawRateObserverMaxBlendDelta()
{
  return std::max(0.0, ParseEnvDouble("XR_TRACKER_MODEL_YAW_RATE_OBSERVER_MAX_BLEND_DELTA", 0.8));
}

/**
 * @brief 返回 yaw rate 观测器参与输出前需要的最小样本数。
 */
inline std::uint32_t VehicleYawRateObserverMinSamples()
{
  return ParseEnvUint("XR_TRACKER_MODEL_YAW_RATE_OBSERVER_MIN_SAMPLES", 4U);
}

/**
 * @brief 返回发布目标时额外外推的时间，单位 s。
 */
inline double VehicleOutputExtrapolateSeconds()
{
  return std::clamp(
      ParseEnvDouble("XR_TRACKER_MODEL_OUTPUT_EXTRAPOLATE_SEC", 0.0), 0.0, 0.050);
}

/**
 * @brief 返回测量面重定位的基础系数。
 */
inline double VehicleMeasurementRecenterAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_MODEL_MEAS_RECENTER_ALPHA", 1.0),
                    0.0, 1.0);
}

/**
 * @brief 是否按匹配质量动态调节测量面重定位系数。
 */
inline bool VehicleMeasurementRecenterQualityEnabled()
{
  const char* env = std::getenv("XR_TRACKER_MODEL_QUALITY_RECENTER");
  if (env != nullptr)
  {
    return env[0] != '\0' && env[0] != '0';
  }
  return false;
}

/**
 * @brief 返回高质量观测时的测量面重定位系数。
 */
inline double VehicleMeasurementRecenterAlphaGood()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_MODEL_MEAS_RECENTER_ALPHA_GOOD", 0.45),
                    0.0, 1.0);
}

/**
 * @brief 返回低质量观测时的测量面重定位系数。
 */
inline double VehicleMeasurementRecenterAlphaBad()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_MODEL_MEAS_RECENTER_ALPHA_BAD", 0.08),
                    0.0, 1.0);
}

/**
 * @brief 返回重定位质量评分中的好匹配分阈值。
 */
inline double VehicleMeasurementRecenterScoreGood()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_MODEL_MEAS_RECENTER_SCORE_GOOD", 0.45));
}

/**
 * @brief 返回重定位质量评分中的坏匹配分阈值。
 */
inline double VehicleMeasurementRecenterScoreBad()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_MODEL_MEAS_RECENTER_SCORE_BAD", 1.30));
}

/**
 * @brief 返回重定位质量评分中的好 yaw 误差阈值。
 */
inline double VehicleMeasurementRecenterYawGood()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_MODEL_MEAS_RECENTER_YAW_GOOD", 0.05));
}

/**
 * @brief 返回重定位质量评分中的坏 yaw 误差阈值。
 */
inline double VehicleMeasurementRecenterYawBad()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_MODEL_MEAS_RECENTER_YAW_BAD", 0.35));
}

/**
 * @brief 返回重定位质量评分中的好位置误差阈值。
 */
inline double VehicleMeasurementRecenterXyzGood()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_MODEL_MEAS_RECENTER_XYZ_GOOD", 0.03));
}

/**
 * @brief 返回重定位质量评分中的坏位置误差阈值。
 */
inline double VehicleMeasurementRecenterXyzBad()
{
  return std::max(0.0,
                  ParseEnvDouble("XR_TRACKER_MODEL_MEAS_RECENTER_XYZ_BAD", 0.10));
}

/**
 * @brief 返回额外位置锚定修正系数。
 */
inline double VehicleMeasurementPositionAnchorAlpha()
{
  return std::clamp(ParseEnvDouble("XR_TRACKER_MODEL_MEAS_POS_ANCHOR_ALPHA", 0.0),
                    0.0, 1.0);
}

/**
 * @brief 返回位置锚定质量衰减的位置误差上限。
 */
inline double VehicleMeasurementPositionAnchorXyzBad()
{
  return std::max(
      1e-6, ParseEnvDouble("XR_TRACKER_MODEL_MEAS_POS_ANCHOR_XYZ_BAD", 0.12));
}

/**
 * @brief 是否使用 XYZ+yaw 直接观测更新路径。
 */
inline bool VehicleXyzMeasurementUpdateEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_MODEL_XYZ_UPDATE");
}

/**
 * @brief 返回 XYZ 观测噪声随距离缩放的比例。
 * @param default_value 配置文件给出的默认比例。
 */
inline double VehicleXyzMeasurementRFactor(double default_value)
{
  return std::max(1e-4,
                  ParseEnvDouble("XR_TRACKER_MODEL_XYZ_R_FACTOR", default_value));
}

/**
 * @brief 返回 XYZ 更新路径中的 yaw 观测方差。
 * @param default_value 配置文件给出的默认方差。
 */
inline double VehicleXyzMeasurementYawVariance(double default_value)
{
  return std::max(1e-6,
                  ParseEnvDouble("XR_TRACKER_MODEL_XYZ_YAW_R", default_value));
}

/**
 * @brief XYZ 观测是否允许同时更新整车半径等完整几何状态。
 */
inline bool VehicleXyzMeasurementFullGeometryEnabled()
{
  return EnvFlagEnabled("XR_TRACKER_MODEL_XYZ_FULL_GEOMETRY");
}
}  // namespace armor_tracker_detail
