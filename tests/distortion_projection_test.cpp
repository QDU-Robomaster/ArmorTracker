#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "ArmorTrackerModel.hpp"

namespace
{
constexpr double projection_tolerance = 1e-4;
constexpr double translation_tolerance = 2e-4;

void Expect(bool condition, const char* label)
{
  if (!condition)
  {
    std::cerr << label << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void ExpectNear(double actual, double expected, double tolerance,
                const std::string& label)
{
  if (std::abs(actual - expected) > tolerance)
  {
    std::cerr << label << ": actual=" << actual << " expected=" << expected << '\n';
    std::exit(EXIT_FAILURE);
  }
}

armor_tracker_detail::Config MakeConfig(uint8_t distortion_size,
                                        bool camera_model_supported = true)
{
  armor_tracker_detail::Config config{};
  config.native_width = 1440;
  config.native_height = 1080;
  config.camera_matrix = {910.0, 0.0, 720.0, 0.0, 925.0, 540.0, 0.0, 0.0, 1.0};
  config.distortion_coefficients = {-0.28,  0.095, 0.0015, -0.0011,
                                    -0.018, 0.011, -0.004, 0.0015};
  config.distortion_size = distortion_size;
  config.camera_model_supported = camera_model_supported;
  return config;
}

cv::Mat MakeCameraMatrix(const armor_tracker_detail::Config& config)
{
  cv::Mat camera_matrix(3, 3, CV_64F);
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      camera_matrix.at<double>(row, col) =
          config.camera_matrix[static_cast<std::size_t>(row * 3 + col)];
    }
  }
  return camera_matrix;
}

cv::Mat MakeReferenceDistortion(const armor_tracker_detail::Config& config)
{
  if (config.distortion_size == 0)
  {
    return {};
  }

  cv::Mat distortion(1, config.distortion_size, CV_64F);
  for (uint8_t index = 0; index < config.distortion_size; ++index)
  {
    distortion.at<double>(0, index) =
        config.distortion_coefficients[static_cast<std::size_t>(index)];
  }
  return distortion;
}

std::vector<cv::Point3f> ArmorObjectPoints(armor_tracker_detail::ArmorType type)
{
  const double width = type == armor_tracker_detail::ArmorType::BIG
                           ? armor_tracker_detail::kBigArmorWidth
                           : armor_tracker_detail::kSmallArmorWidth;
  return {{0.0F, static_cast<float>(width / 2.0),
           static_cast<float>(armor_tracker_detail::kLightbarLength / 2.0)},
          {0.0F, static_cast<float>(-width / 2.0),
           static_cast<float>(armor_tracker_detail::kLightbarLength / 2.0)},
          {0.0F, static_cast<float>(-width / 2.0),
           static_cast<float>(-armor_tracker_detail::kLightbarLength / 2.0)},
          {0.0F, static_cast<float>(width / 2.0),
           static_cast<float>(-armor_tracker_detail::kLightbarLength / 2.0)}};
}

std::vector<cv::Point2f> ReferenceReprojection(const armor_tracker_detail::Config& config,
                                               const Eigen::Vector3d& xyz_in_world,
                                               double yaw,
                                               armor_tracker_detail::ArmorType type,
                                               armor_tracker_detail::ArmorName name)
{
  const double tilt = name == armor_tracker_detail::ArmorName::OUTPOST
                          ? armor_tracker_detail::kOutpostArmorTilt
                          : 15.0 * armor_tracker_detail::kPi / 180.0;
  const Eigen::Matrix3d camera_to_body =
      armor_tracker_detail::CameraToBodyRotationFromMountExtrinsic(
          config.camera_mount_to_body_rotation);
  const Eigen::Matrix3d armor_to_camera =
      camera_to_body.transpose() * armor_tracker_detail::ArmorRotationFromYaw(yaw, tilt);
  const Eigen::Vector3d armor_in_camera = camera_to_body.transpose() * xyz_in_world;

  cv::Vec3d rvec;
  cv::Rodrigues(armor_tracker_detail::Mat3dToCv(armor_to_camera), rvec);
  const cv::Vec3d tvec(armor_in_camera.x(), armor_in_camera.y(), armor_in_camera.z());
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(ArmorObjectPoints(type), rvec, tvec, MakeCameraMatrix(config),
                    MakeReferenceDistortion(config), image_points);
  return image_points;
}

void TestReprojectionMatchesOpenCv()
{
  const std::array<uint8_t, 3> distortion_sizes{0, 5, 8};
  const std::array<const char*, 3> names{"none", "plumb_bob", "rational"};
  const Eigen::Vector3d point_in_camera(0.75, -0.4, 2.5);

  std::array<std::vector<cv::Point2f>, 3> projected_by_model{};
  for (std::size_t index = 0; index < distortion_sizes.size(); ++index)
  {
    const auto config = MakeConfig(distortion_sizes[index]);
    const Eigen::Vector3d point_in_world =
        armor_tracker_detail::CameraToBodyRotationFromMountExtrinsic(
            config.camera_mount_to_body_rotation) *
        point_in_camera;
    armor_tracker_detail::Solver solver(config);
    Expect(solver.CameraModelSupported(), "supported model must construct a solver");

    const auto actual = solver.ReprojectArmor(point_in_world, 0.35,
                                              armor_tracker_detail::ArmorType::SMALL,
                                              armor_tracker_detail::ArmorName::TWO);
    const auto expected = ReferenceReprojection(config, point_in_world, 0.35,
                                                armor_tracker_detail::ArmorType::SMALL,
                                                armor_tracker_detail::ArmorName::TWO);
    Expect(actual.size() == expected.size(), "reprojection point count must match");
    for (std::size_t point_index = 0; point_index < actual.size(); ++point_index)
    {
      const std::string prefix =
          std::string(names[index]) + "/point_" + std::to_string(point_index);
      ExpectNear(actual[point_index].x, expected[point_index].x, projection_tolerance,
                 prefix + " x");
      ExpectNear(actual[point_index].y, expected[point_index].y, projection_tolerance,
                 prefix + " y");
    }
    projected_by_model[index] = actual;
  }

  Expect(cv::norm(projected_by_model[0][0] - projected_by_model[1][0]) > 1.0,
         "non-zero tracker D fixture must detect a zero-D regression");
}

void TestSolvePnpUsesConfiguredDistortion()
{
  const std::array<uint8_t, 3> distortion_sizes{0, 5, 8};
  const cv::Vec3d source_rvec(0.12, -0.08, 0.2);
  const cv::Vec3d source_tvec(0.55, -0.3, 3.2);

  for (const uint8_t distortion_size : distortion_sizes)
  {
    const auto config = MakeConfig(distortion_size);
    armor_tracker_detail::Armor armor{};
    armor.type = armor_tracker_detail::ArmorType::BIG;
    armor.name = armor_tracker_detail::ArmorName::THREE;
    cv::projectPoints(ArmorObjectPoints(armor.type), source_rvec, source_tvec,
                      MakeCameraMatrix(config), MakeReferenceDistortion(config),
                      armor.points);

    armor_tracker_detail::Solver solver(config);
    Expect(solver.Solve(armor), "solvePnP must accept supported distortion models");
    const Eigen::Vector3d expected_body =
        armor_tracker_detail::CameraToBodyRotationFromMountExtrinsic(
            config.camera_mount_to_body_rotation) *
        Eigen::Vector3d(source_tvec[0], source_tvec[1], source_tvec[2]);
    ExpectNear(armor.xyz_in_body.x(), expected_body.x(), translation_tolerance,
               "solvePnP body x");
    ExpectNear(armor.xyz_in_body.y(), expected_body.y(), translation_tolerance,
               "solvePnP body y");
    ExpectNear(armor.xyz_in_body.z(), expected_body.z(), translation_tolerance,
               "solvePnP body z");
  }
}

void TestUnsupportedModelFailsClosed()
{
  const auto config = MakeConfig(0, false);
  armor_tracker_detail::Solver solver(config);
  Expect(!solver.CameraModelSupported(), "unsupported model must be rejected");

  armor_tracker_detail::Armor armor{};
  armor.type = armor_tracker_detail::ArmorType::SMALL;
  armor.name = armor_tracker_detail::ArmorName::TWO;
  armor.points = {{100.0F, 100.0F}, {200.0F, 100.0F}, {200.0F, 160.0F}, {100.0F, 160.0F}};
  Expect(!solver.Solve(armor), "unsupported model must not enter solvePnP");
  Expect(solver
             .ReprojectArmor(Eigen::Vector3d(0.0, 3.0, 0.0), 0.0,
                             armor_tracker_detail::ArmorType::SMALL,
                             armor_tracker_detail::ArmorName::TWO)
             .empty(),
         "unsupported model must not enter projectPoints");
  Expect(!solver.IsArmorFaceFrontFacing(Eigen::Vector3d(0.0, 3.0, 0.0), 0.0,
                                        armor_tracker_detail::ArmorName::TWO),
         "unsupported model must not produce preview geometry");

  auto nonfinite_config = MakeConfig(5);
  nonfinite_config.distortion_coefficients[13] = std::numeric_limits<double>::quiet_NaN();
  armor_tracker_detail::Solver nonfinite_solver(nonfinite_config);
  Expect(!nonfinite_solver.CameraModelSupported(),
         "non-finite unused D tail must disable the immutable calibration");

  auto nonstandard_config = MakeConfig(5);
  nonstandard_config.camera_matrix[1] = 1.0;
  armor_tracker_detail::Solver nonstandard_solver(nonstandard_config);
  Expect(!nonstandard_solver.CameraModelSupported(),
         "nonstandard pinhole K must disable the solver");

  auto nonfinite_k_config = MakeConfig(5);
  nonfinite_k_config.camera_matrix[0] = std::numeric_limits<double>::infinity();
  armor_tracker_detail::Solver nonfinite_k_solver(nonfinite_k_config);
  Expect(!nonfinite_k_solver.CameraModelSupported(),
         "non-finite K must disable the solver");

  auto invalid_size_config = MakeConfig(4);
  armor_tracker_detail::Solver invalid_size_solver(invalid_size_config);
  Expect(!invalid_size_solver.CameraModelSupported(),
         "non-OpenCV distortion size must disable the solver");

  auto valid_config = MakeConfig(5);
  armor_tracker_detail::Solver valid_solver(valid_config);
  armor.points = {{100.0F, 100.0F}};
  Expect(!valid_solver.Solve(armor), "wrong corner count must fail before solvePnP");
  armor.points = {{100.0F, 100.0F},
                  {200.0F, 100.0F},
                  {200.0F, 160.0F},
                  {100.0F, std::numeric_limits<float>::quiet_NaN()}};
  Expect(!valid_solver.Solve(armor), "non-finite corners must fail before solvePnP");
  Expect(valid_solver
             .ReprojectArmor(
                 Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 3.0, 0.0), 0.0,
                 armor_tracker_detail::ArmorType::SMALL,
                 armor_tracker_detail::ArmorName::TWO)
             .empty(),
         "non-finite reprojection input must fail before projectPoints");
}
}  // namespace

int main()
{
  TestReprojectionMatchesOpenCv();
  TestSolvePnpUsesConfiguredDistortion();
  TestUnsupportedModelFailsClosed();
  std::cout << "ArmorTracker distortion projection tests passed\n";
  return EXIT_SUCCESS;
}
