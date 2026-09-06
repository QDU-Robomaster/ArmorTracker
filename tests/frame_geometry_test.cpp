#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <string_view>

#include "ArmorDetectorPublishGeometry.hpp"
#include "ArmorTrackerFrameAdapter.hpp"

namespace
{
constexpr CameraTypes::FrameLayout kLayout{720, 540, 2160, CameraTypes::Encoding::BGR8};

constexpr CameraTypes::CameraCalibration kCalibration{
    .native_width = 1440,
    .native_height = 1080,
};

constexpr CameraTypes::FrameGeometry kWideGeometry{
    .width = 720,
    .height = 540,
    .step = 2160,
    .roi_offset_x_native = 0,
    .roi_offset_y_native = 0,
    .decimation_x = 2,
    .decimation_y = 2,
    .flags = CameraTypes::FRAME_GEOMETRY_NONE,
    .reserved = 0,
    .sample_phase_x_native = 0.0F,
    .sample_phase_y_native = 0.0F,
};

void Expect(bool condition, std::string_view message)
{
  if (!condition)
  {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void ExpectNear(double actual, double expected, std::string_view message)
{
  Expect(std::abs(actual - expected) < 1e-4, message);
}

ArmorDetectorResult MakeDetectionFromFrameQuad(
    const CameraTypes::FrameGeometry& geometry,
    const std::array<cv::Point2f, 4>& frame_points)
{
  ArmorDetectorResult armor{};
  armor.number = ArmorNumber::THREE;
  armor.type = ArmorType::SMALL;
  armor.confidence = 0.9F;
  const cv::Point2f frame_center =
      (frame_points[0] + frame_points[1] + frame_points[2] + frame_points[3]) / 4.0F;
  const auto published = armor_detector_detail::MapPublishGeometry(
      frame_points, frame_center, cv::Rect{10, 20, 41, 21}, geometry);
  armor.points = published.points;
  armor.center = published.center;
  armor.center_norm = {0.5F, 0.5F};
  return armor;
}

void TestGeometryValidation()
{
  Expect(CameraTypes::ValidateFrameGeometry(kLayout, kCalibration, kWideGeometry),
         "image-owned geometry must be accepted");

  auto invalid = kWideGeometry;
  invalid.decimation_x = 0;
  Expect(!CameraTypes::ValidateFrameGeometry(kLayout, kCalibration, invalid),
         "invalid image-owned geometry must be rejected");
}

void TestFrameAreaRecovery(const CameraTypes::FrameGeometry& geometry,
                           const std::array<cv::Point2f, 4>& expected_frame_points)
{
  const std::array<cv::Point2f, 4> frame_points{
      cv::Point2f{10.0F, 20.0F}, cv::Point2f{50.0F, 20.0F}, cv::Point2f{50.0F, 40.0F},
      cv::Point2f{10.0F, 40.0F}};
  const auto detection = MakeDetectionFromFrameQuad(geometry, frame_points);
  const auto input = armor_tracker_detail::BuildTrackerInput(detection, geometry);

  for (std::size_t index = 0; index < detection.points.size(); ++index)
  {
    ExpectNear(input.corners[index].x, detection.points[index].x,
               "tracker PnP corner x must remain native");
    ExpectNear(input.corners[index].y, detection.points[index].y,
               "tracker PnP corner y must remain native");
    ExpectNear(input.frame_corners[index].x, expected_frame_points[index].x,
               "inverse-mapped frame corner x must preserve producer order");
    ExpectNear(input.frame_corners[index].y, expected_frame_points[index].y,
               "inverse-mapped frame corner y must preserve producer order");
  }

  const std::vector<cv::Point2f> native_points(detection.points.begin(),
                                               detection.points.end());
  ExpectNear(std::abs(cv::contourArea(native_points)), 3200.0,
             "wide native quad area must scale by four");

  const auto tracked = armor_tracker_detail::MakeTrackedArmor(input);
  ExpectNear(std::abs(cv::contourArea(tracked.selection_points)), 800.0,
             "selection area must be restored to frame pixels");
}
}  // namespace

int main()
{
  TestGeometryValidation();
  const std::array<cv::Point2f, 4> frame_points{
      cv::Point2f{10.0F, 20.0F}, cv::Point2f{50.0F, 20.0F}, cv::Point2f{50.0F, 40.0F},
      cv::Point2f{10.0F, 40.0F}};
  TestFrameAreaRecovery(kWideGeometry, frame_points);

  auto reversed = kWideGeometry;
  reversed.flags =
      CameraTypes::FRAME_GEOMETRY_REVERSE_X | CameraTypes::FRAME_GEOMETRY_REVERSE_Y;
  const std::array<cv::Point2f, 4> reversed_frame_points{
      frame_points[2], frame_points[3], frame_points[0], frame_points[1]};
  TestFrameAreaRecovery(reversed, reversed_frame_points);

  std::cout << "ArmorTracker frame geometry tests passed\n";
  return EXIT_SUCCESS;
}
