#pragma once

/**
 * @file ArmorTrackerFrameAdapter.hpp
 * @brief Adapts native detector coordinates for tracking.
 */

#include <cstddef>
#include <opencv2/core.hpp>

#include "ArmorDetectorTypes.hpp"
#include "ArmorTrackerCore.hpp"
#include "CameraBase.hpp"

namespace armor_tracker_detail
{
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
