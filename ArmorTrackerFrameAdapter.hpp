#pragma once

/**
 * @file ArmorTrackerFrameAdapter.hpp
 * @brief Validates detector frame geometry and adapts native detections for tracking.
 */

#include <cstddef>
#include <opencv2/core.hpp>

#include "ArmorDetectorTypes.hpp"
#include "ArmorTrackerCore.hpp"
#include "CameraBase.hpp"

namespace armor_tracker_detail
{
/**
 * @brief Result of validating a detector packet against its referenced image.
 */
enum class DetectorFrameGeometryStatus
{
  VALID,
  INVALID,
  IMAGE_MISMATCH,
};

/**
 * @brief Validate packet geometry and require an exact match with the image snapshot.
 */
template <CameraTypes::FrameLayout FrameLayoutV>
[[nodiscard]] constexpr DetectorFrameGeometryStatus CheckDetectorFrameGeometry(
    const CameraTypes::CameraCalibration& calibration,
    const CameraTypes::FrameGeometry& packet_geometry,
    const CameraTypes::FrameGeometry& image_geometry)
{
  if (!CameraTypes::ValidateFrameGeometry(FrameLayoutV, calibration, packet_geometry))
  {
    return DetectorFrameGeometryStatus::INVALID;
  }
  if (!CameraTypes::SameFrameGeometry(packet_geometry, image_geometry))
  {
    return DetectorFrameGeometryStatus::IMAGE_MISMATCH;
  }
  return DetectorFrameGeometryStatus::VALID;
}

/**
 * @brief Keep native corners for PnP while restoring frame-scale points for selection.
 */
[[nodiscard]] inline InputArmor BuildTrackerInput(
    const ArmorDetectorResult& armor, const CameraTypes::FrameGeometry& geometry)
{
  InputArmor input{};
  input.tag_id = static_cast<int>(armor.number);
  input.armor_type = static_cast<int>(armor.type);
  input.confidence = armor.confidence;
  input.corners = armor.points;
  for (std::size_t index = 0; index < input.frame_corners.size(); ++index)
  {
    input.frame_corners[index] = cv::Point2f(
        static_cast<float>(CameraTypes::NativeToFrameX(geometry, armor.points[index].x)),
        static_cast<float>(CameraTypes::NativeToFrameY(geometry, armor.points[index].y)));
  }
  input.frame_corners_valid = true;
  input.center = armor.center;
  input.center_norm = armor.center_norm;
  return input;
}
}  // namespace armor_tracker_detail
