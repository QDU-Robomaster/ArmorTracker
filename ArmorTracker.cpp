#include "ArmorTracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "TrackerMath.hpp"
#include "logger.hpp"
#include "timebase.hpp"

namespace
{
constexpr double MAX_TRACKER_DT_S = 0.1;
constexpr double BAD_CONVERGENCE_RATIO = 0.4;
constexpr int INFO_PANEL_WIDTH = 420;
constexpr int MAX_DEBUG_OBSERVATIONS = 6;
constexpr double HEADER_BAR_ALPHA = 0.8;
constexpr double SMALL_ARMOR_HALF_WIDTH_M = 0.135 * 0.5;
constexpr double LARGE_ARMOR_HALF_WIDTH_M = 0.225 * 0.5;
constexpr double ARMOR_HALF_HEIGHT_M = 0.055 * 0.5;
constexpr double PROJECTED_ARMOR_FILL_ALPHA = 0.20;
constexpr double GOOD_PREDICTION_ERROR_PX = 12.0;
constexpr double WARN_PREDICTION_ERROR_PX = 28.0;
constexpr std::size_t MAX_GIMBAL_ROTATION_HISTORY_SIZE = 256U;
constexpr double MAX_GIMBAL_ROTATION_HISTORY_S = 2.0;
constexpr int OPTIMIZED_YAW_SEARCH_RANGE_DEG = 140;
constexpr double OUTPOST_ARMOR_PITCH_RAD = -15.0 * CV_PI / 180.0;
constexpr double DEFAULT_ARMOR_PITCH_RAD = 15.0 * CV_PI / 180.0;

struct PreviewProjector
{
  bool valid{false};
  int image_width{0};
  int image_height{0};
  cv::Mat camera_matrix{};
  cv::Mat dist_coeffs{};
  Eigen::Quaterniond camera_rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d camera_translation = Eigen::Vector3d::Zero();

  bool Project(const Eigen::Vector3d& world_xyz, cv::Point2f& image_point) const
  {
    if (!valid)
    {
      return false;
    }

    const Eigen::Vector3d camera_xyz =
        camera_rotation.conjugate() * (world_xyz - camera_translation);
    if (camera_xyz.z() <= 1e-6)
    {
      return false;
    }

    std::vector<cv::Point3d> object_points = {
        cv::Point3d(camera_xyz.x(), camera_xyz.y(), camera_xyz.z())};
    std::vector<cv::Point2d> image_points;
    cv::projectPoints(object_points, cv::Vec3d(0.0, 0.0, 0.0),
                      cv::Vec3d(0.0, 0.0, 0.0), camera_matrix, dist_coeffs,
                      image_points);
    if (image_points.empty())
    {
      return false;
    }

    const cv::Point2d projected = image_points.front();
    if (!std::isfinite(projected.x) || !std::isfinite(projected.y))
    {
      return false;
    }

    constexpr double IMAGE_BORDER_MARGIN = 64.0;
    if (projected.x < -IMAGE_BORDER_MARGIN ||
        projected.x > static_cast<double>(image_width) + IMAGE_BORDER_MARGIN ||
        projected.y < -IMAGE_BORDER_MARGIN ||
        projected.y > static_cast<double>(image_height) + IMAGE_BORDER_MARGIN)
    {
      return false;
    }

    image_point = cv::Point2f(static_cast<float>(projected.x),
                              static_cast<float>(projected.y));
    return true;
  }
};

bool is_balance_armor(const TrackedArmorObservation& armor)
{
  return armor.result.type == ArmorType::LARGE &&
         (armor.result.number == ArmorNumber::THREE ||
          armor.result.number == ArmorNumber::FOUR ||
          armor.result.number == ArmorNumber::FIVE);
}

std::string state_to_string(ArmorTracker::State state)
{
  switch (state)
  {
    case ArmorTracker::State::LOST:
      return "LOST";
    case ArmorTracker::State::DETECTING:
      return "DETECTING";
    case ArmorTracker::State::TRACKING:
      return "TRACKING";
    case ArmorTracker::State::TEMP_LOST:
      return "TEMP LOST";
    default:
      return "UNKNOWN";
  }
}

cv::Scalar state_to_color(ArmorTracker::State state)
{
  switch (state)
  {
    case ArmorTracker::State::LOST:
      return cv::Scalar(92, 102, 119);
    case ArmorTracker::State::DETECTING:
      return cv::Scalar(255, 204, 92);
    case ArmorTracker::State::TRACKING:
      return cv::Scalar(84, 196, 116);
    case ArmorTracker::State::TEMP_LOST:
      return cv::Scalar(255, 153, 77);
    default:
      return cv::Scalar(160, 160, 160);
  }
}

cv::Scalar armor_color_to_scalar(ArmorColor color)
{
  switch (color)
  {
    case ArmorColor::BLUE:
      return cv::Scalar(255, 185, 40);
    case ArmorColor::RED:
      return cv::Scalar(72, 96, 255);
    default:
      return cv::Scalar(180, 220, 180);
  }
}

cv::Scalar prediction_error_to_color(double error_px)
{
  if (error_px <= GOOD_PREDICTION_ERROR_PX)
  {
    return cv::Scalar(128, 226, 142);
  }
  if (error_px <= WARN_PREDICTION_ERROR_PX)
  {
    return cv::Scalar(255, 214, 102);
  }
  return cv::Scalar(255, 128, 128);
}

std::string armor_number_to_string(ArmorNumber number)
{
  const std::size_t index = static_cast<std::size_t>(number);
  if (index >= ARMOR_NUMBER_NAMES.size())
  {
    return "invalid";
  }
  return std::string(ARMOR_NUMBER_NAMES[index]);
}

std::string format_value(double value, int precision = 3)
{
  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(precision);
  stream << value;
  return stream.str();
}

void draw_info_row(cv::Mat& canvas, int x, int y, const std::string& key,
                   const std::string& value, const cv::Scalar& value_color)
{
  constexpr int FONT = cv::FONT_HERSHEY_DUPLEX;
  constexpr double FONT_SCALE = 0.55;
  constexpr int THICKNESS = 1;

  cv::putText(canvas, key, cv::Point(x, y), FONT, FONT_SCALE,
              cv::Scalar(170, 182, 196), THICKNESS, cv::LINE_AA);
  cv::putText(canvas, value, cv::Point(x + 170, y), FONT, FONT_SCALE, value_color,
              THICKNESS, cv::LINE_AA);
}

void draw_state_chip(cv::Mat& canvas, const std::string& text, const cv::Point& origin,
                     const cv::Scalar& color)
{
  constexpr int FONT = cv::FONT_HERSHEY_DUPLEX;
  constexpr double FONT_SCALE = 0.66;
  constexpr int THICKNESS = 1;
  constexpr int PADDING_X = 10;
  constexpr int PADDING_Y = 8;

  int baseline = 0;
  const cv::Size text_size =
      cv::getTextSize(text, FONT, FONT_SCALE, THICKNESS, &baseline);
  const cv::Rect rect(origin.x, origin.y - text_size.height - PADDING_Y,
                      text_size.width + 2 * PADDING_X,
                      text_size.height + 2 * PADDING_Y);
  cv::rectangle(canvas, rect, color, cv::FILLED, cv::LINE_AA);
  cv::putText(canvas, text,
              cv::Point(origin.x + PADDING_X,
                        origin.y - PADDING_Y + baseline / 2),
              FONT, FONT_SCALE, cv::Scalar(16, 18, 24), THICKNESS, cv::LINE_AA);
}

PreviewProjector build_preview_projector(
    const std::shared_ptr<CameraBase::CameraInfo>& camera_info,
    const LibXR::Quaternion<double>& gimbal_rotation,
    const LibXR::Transform<double>& camera_to_gimbal_transform)
{
  PreviewProjector projector;
  if (camera_info == nullptr)
  {
    return projector;
  }

  projector.image_width = static_cast<int>(camera_info->width);
  projector.image_height = static_cast<int>(camera_info->height);
  projector.camera_matrix =
      cv::Mat(3, 3, CV_64F, const_cast<double*>(camera_info->camera_matrix.data())).clone();

  const auto dist_coeffs = CameraBase::CameraInfo::ToPnPDistCoeffs(
      camera_info->distortion_model, camera_info->distortion_coefficients);
  if (!dist_coeffs.empty())
  {
    projector.dist_coeffs =
        cv::Mat(1, static_cast<int>(dist_coeffs.size()), CV_64F,
                const_cast<double*>(dist_coeffs.data()))
            .clone();
  }

  const LibXR::Transform<double> world_to_gimbal(gimbal_rotation, {0.0, 0.0, 0.0});
  const LibXR::Transform<double> world_to_camera =
      world_to_gimbal + camera_to_gimbal_transform;

  projector.camera_rotation = Eigen::Quaterniond(world_to_camera.rotation);
  projector.camera_translation =
      Eigen::Vector3d(world_to_camera.translation.x(), world_to_camera.translation.y(),
                      world_to_camera.translation.z());
  projector.valid = true;
  return projector;
}

std::array<Eigen::Vector3d, 4> build_predicted_armor_corners(
    const Eigen::Vector4d& armor_xyza, ArmorType armor_type, ArmorNumber armor_number)
{
  (void)armor_number;
  const double half_width = (armor_type == ArmorType::LARGE) ? LARGE_ARMOR_HALF_WIDTH_M
                                                             : SMALL_ARMOR_HALF_WIDTH_M;
  const Eigen::Vector3d center = armor_xyza.head<3>();
  const double yaw = armor_xyza[3];
  const Eigen::Vector3d width_axis(-std::sin(yaw), std::cos(yaw), 0.0);
  const Eigen::Vector3d height_axis(0.0, 0.0, 1.0);

  return {
      center + width_axis * half_width - height_axis * ARMOR_HALF_HEIGHT_M,
      center + width_axis * half_width + height_axis * ARMOR_HALF_HEIGHT_M,
      center - width_axis * half_width + height_axis * ARMOR_HALF_HEIGHT_M,
      center - width_axis * half_width - height_axis * ARMOR_HALF_HEIGHT_M,
  };
}

Eigen::Matrix3d build_armor_to_world_rotation(double pitch, double yaw)
{
  const double SIN_YAW = std::sin(yaw);
  const double COS_YAW = std::cos(yaw);
  const double SIN_PITCH = std::sin(pitch);
  const double COS_PITCH = std::cos(pitch);

  Eigen::Matrix3d armor_to_world;
  armor_to_world << COS_YAW * COS_PITCH, -SIN_YAW, COS_YAW * SIN_PITCH,
      SIN_YAW * COS_PITCH, COS_YAW, SIN_YAW * SIN_PITCH, -SIN_PITCH, 0.0,
      COS_PITCH;
  return armor_to_world;
}

double get_armor_pitch_rad(ArmorNumber armor_number)
{
  return armor_number == ArmorNumber::OUTPOST ? OUTPOST_ARMOR_PITCH_RAD
                                              : DEFAULT_ARMOR_PITCH_RAD;
}

std::array<Eigen::Vector3d, 4> build_observation_ordered_armor_corners(
    const Eigen::Vector3d& center, const Eigen::Matrix3d& armor_to_world,
    ArmorType armor_type)
{
  const double HALF_WIDTH = (armor_type == ArmorType::LARGE) ? LARGE_ARMOR_HALF_WIDTH_M
                                                             : SMALL_ARMOR_HALF_WIDTH_M;
  const auto world_point = [&](double y, double z)
  {
    return center + armor_to_world * Eigen::Vector3d(0.0, y, z);
  };

  return {
      world_point(HALF_WIDTH, ARMOR_HALF_HEIGHT_M),
      world_point(-HALF_WIDTH, ARMOR_HALF_HEIGHT_M),
      world_point(-HALF_WIDTH, -ARMOR_HALF_HEIGHT_M),
      world_point(HALF_WIDTH, -ARMOR_HALF_HEIGHT_M),
  };
}

bool project_predicted_armor_quad(const PreviewProjector& projector,
                                  const std::array<Eigen::Vector3d, 4>& world_corners,
                                  std::array<cv::Point, 4>& image_corners)
{
  for (std::size_t corner_index = 0; corner_index < world_corners.size(); ++corner_index)
  {
    cv::Point2f image_point;
    if (!projector.Project(world_corners[corner_index], image_point))
    {
      return false;
    }

    image_corners[corner_index] =
        cv::Point(cvRound(image_point.x), cvRound(image_point.y));
  }
  return true;
}

bool project_armor_quad(const PreviewProjector& projector,
                        const std::array<Eigen::Vector3d, 4>& world_corners,
                        std::array<cv::Point2f, 4>& image_corners)
{
  for (std::size_t corner_index = 0; corner_index < world_corners.size();
       ++corner_index)
  {
    if (!projector.Project(world_corners[corner_index], image_corners[corner_index]))
    {
      return false;
    }
  }
  return true;
}

void draw_projected_armor_quad(cv::Mat& canvas,
                               const std::array<cv::Point, 4>& image_corners,
                               const cv::Scalar& outline_color,
                               const cv::Scalar& fill_color,
                               int outline_thickness)
{
  std::vector<cv::Point> polygon(image_corners.begin(), image_corners.end());
  cv::Mat overlay = canvas.clone();
  cv::fillConvexPoly(overlay, polygon, fill_color, cv::LINE_AA);
  cv::addWeighted(overlay, PROJECTED_ARMOR_FILL_ALPHA, canvas,
                  1.0 - PROJECTED_ARMOR_FILL_ALPHA, 0.0, canvas);

  const cv::Point* points = image_corners.data();
  const int point_count = static_cast<int>(image_corners.size());
  cv::polylines(canvas, &points, &point_count, 1, true, outline_color, outline_thickness,
                cv::LINE_AA);
}

struct ProjectedArmorCandidate
{
  cv::Point2f center{};
  std::array<cv::Point, 4> quad{};
};

cv::Point2f quad_center(const std::array<cv::Point, 4>& quad)
{
  cv::Point2f center(0.0F, 0.0F);
  for (const auto& point : quad)
  {
    center.x += static_cast<float>(point.x);
    center.y += static_cast<float>(point.y);
  }
  center.x /= 4.0F;
  center.y /= 4.0F;
  return center;
}

double quad_rms_error_px(const std::array<cv::Point, 4>& projected_quad,
                         const std::array<cv::Point2f, 4>& observed_quad)
{
  double best_mean_square_error = std::numeric_limits<double>::max();

  for (int reverse = 0; reverse < 2; ++reverse)
  {
    for (int shift = 0; shift < 4; ++shift)
    {
      double mean_square_error = 0.0;
      for (int index = 0; index < 4; ++index)
      {
        const int observed_index =
            reverse == 0 ? (index + shift) % 4 : (shift - index + 4) % 4;
        const double dx = static_cast<double>(projected_quad[index].x) -
                          static_cast<double>(observed_quad[observed_index].x);
        const double dy = static_cast<double>(projected_quad[index].y) -
                          static_cast<double>(observed_quad[observed_index].y);
        mean_square_error += dx * dx + dy * dy;
      }
      best_mean_square_error =
          std::min(best_mean_square_error, mean_square_error / 4.0);
    }
  }

  return std::sqrt(best_mean_square_error);
}

double quad_rms_error_px(const std::array<cv::Point2f, 4>& projected_quad,
                         const std::array<cv::Point2f, 4>& observed_quad)
{
  double mean_square_error = 0.0;
  for (int index = 0; index < 4; ++index)
  {
    const cv::Point2f delta = projected_quad[index] - observed_quad[index];
    mean_square_error +=
        static_cast<double>(delta.x) * static_cast<double>(delta.x) +
        static_cast<double>(delta.y) * static_cast<double>(delta.y);
  }
  return std::sqrt(mean_square_error / 4.0);
}

double optimize_observation_yaw(const PreviewProjector& projector,
                                const ArmorDetectorResult& armor,
                                const Eigen::Vector3d& xyz_in_world,
                                const LibXR::Quaternion<double>& frame_gimbal_rotation,
                                double raw_yaw)
{
  if (!projector.valid)
  {
    return raw_yaw;
  }

  const double GIMBAL_YAW =
      TrackerMath::LimitRad(frame_gimbal_rotation.ToEulerAngleZYX()[2]);
  const double YAW_START =
      TrackerMath::LimitRad(
          GIMBAL_YAW - 0.5 * OPTIMIZED_YAW_SEARCH_RANGE_DEG * CV_PI / 180.0);
  const double ARMOR_PITCH = get_armor_pitch_rad(armor.number);

  double best_yaw = raw_yaw;
  double best_error = std::numeric_limits<double>::max();
  for (int step = 0; step < OPTIMIZED_YAW_SEARCH_RANGE_DEG; ++step)
  {
    const double candidate_yaw =
        TrackerMath::LimitRad(YAW_START + step * CV_PI / 180.0);
    const auto candidate_corners = build_observation_ordered_armor_corners(
        xyz_in_world, build_armor_to_world_rotation(ARMOR_PITCH, candidate_yaw),
        armor.type);
    std::array<cv::Point2f, 4> projected_corners{};
    if (!project_armor_quad(projector, candidate_corners, projected_corners))
    {
      continue;
    }

    const double error = quad_rms_error_px(projected_corners, armor.points);
    if (error < best_error)
    {
      best_error = error;
      best_yaw = candidate_yaw;
    }
  }

  return best_yaw;
}

LibXR::Quaternion<double> slerp_quaternion(const LibXR::Quaternion<double>& lhs,
                                           const LibXR::Quaternion<double>& rhs,
                                           double alpha)
{
  const double clamped_alpha = std::clamp(alpha, 0.0, 1.0);
  const Eigen::Quaterniond lhs_eigen(lhs);
  const Eigen::Quaterniond rhs_eigen(rhs);
  return LibXR::Quaternion<double>(lhs_eigen.slerp(clamped_alpha, rhs_eigen));
}
}  // namespace

ArmorTracker::ArmorTracker(LibXR::HardwareContainer&, LibXR::ApplicationManager& app,
                           Config cfg)
    : cfg_(std::move(cfg))
{
  camera_to_gimbal_transform_static_ = cfg_.frames.camera_to_gimbal_transform;

  auto info_topic = LibXR::Topic(LibXR::Topic::Find("camera_info"));
  auto info_callback = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto* camera_info = reinterpret_cast<CameraBase::CameraInfo*>(data.addr_);
        self->CameraInfoCallback(camera_info);
      },
      this);
  info_topic.RegisterCallback(info_callback);

  if (cfg_.debug.preview)
  {
    auto image_topic = LibXR::Topic(LibXR::Topic::Find("image_raw"));
    auto image_callback = LibXR::Topic::Callback::Create(
        [](bool, ArmorTracker* self, LibXR::RawData& data)
        {
          auto* image = reinterpret_cast<cv::Mat*>(data.addr_);
          self->ImageCallback(image);
        },
        this);
    image_topic.RegisterCallback(image_callback);
  }

  LibXR::Topic::Domain armor_detector_domain("armor_detector");
  LibXR::Topic armors_topic =
      LibXR::Topic::FindOrCreate<ArmorDetectionsMessage>("armors_result",
                                                         &armor_detector_domain);
  auto armors_callback = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto* armors_msg = reinterpret_cast<ArmorDetectionsMessage*>(data.addr_);
        self->ArmorsCallback(*armors_msg);
      },
      this);
  armors_topic.RegisterCallback(armors_callback);

  LibXR::Topic::Domain gimbal_domain("gimbal");
  LibXR::Topic gimbal_rotation_topic =
      LibXR::Topic::FindOrCreate<LibXR::Quaternion<float>>("rotation", &gimbal_domain);
  auto gimbal_rotation_callback = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        auto* rotation_msg = reinterpret_cast<LibXR::Quaternion<float>*>(data.addr_);
        self->GimbalRotationCallback(rotation_msg);
      },
      this);
  gimbal_rotation_topic.RegisterCallback(gimbal_rotation_callback);

  app.Register(*this);
}

void ArmorTracker::CameraInfoCallback(CameraBase::CameraInfo* camera_info)
{
  if (camera_info == nullptr)
  {
    return;
  }

  LibXR::Mutex::LockGuard lock(preview_frame_lock_);
  camera_info_ = std::make_shared<CameraBase::CameraInfo>(*camera_info);
}

void ArmorTracker::ImageCallback(cv::Mat* img_msg)
{
  if (img_msg == nullptr || img_msg->empty())
  {
    return;
  }

  cv::Mat bgr_img = ConvertToBgr(*img_msg);
  LibXR::Mutex::LockGuard lock(preview_frame_lock_);
  latest_frame_ = std::move(bgr_img);
}

void ArmorTracker::GimbalRotationCallback(LibXR::Quaternion<float>* rotation_msg)
{
  if (rotation_msg == nullptr)
  {
    return;
  }

  PushGimbalRotationSample(
      LibXR::Timebase::GetMicroseconds(),
      LibXR::Quaternion<double>(rotation_msg->w(), rotation_msg->x(),
                                rotation_msg->y(), rotation_msg->z()));
}

cv::Mat ArmorTracker::ConvertToBgr(const cv::Mat& input) const
{
  std::shared_ptr<CameraBase::CameraInfo> camera_info;
  {
    LibXR::Mutex::LockGuard lock(preview_frame_lock_);
    camera_info = camera_info_;
  }

  if (camera_info == nullptr)
  {
    return input.clone();
  }

  switch (camera_info->encoding)
  {
    case CameraBase::Encoding::RGB8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
      return output;
    }
    case CameraBase::Encoding::BGRA8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_BGRA2BGR);
      return output;
    }
    case CameraBase::Encoding::RGBA8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_RGBA2BGR);
      return output;
    }
    default:
      return input.clone();
  }
}

void ArmorTracker::PushGimbalRotationSample(LibXR::MicrosecondTimestamp timestamp,
                                            const LibXR::Quaternion<double>& rotation)
{
  LibXR::Mutex::LockGuard lock(gimbal_rotation_lock_);

  const uint64_t TIMESTAMP_US = static_cast<uint64_t>(timestamp);
  if (!gimbal_rotation_history_.empty())
  {
    const uint64_t LAST_TIMESTAMP_US =
        static_cast<uint64_t>(gimbal_rotation_history_.back().timestamp);
    if (TIMESTAMP_US < LAST_TIMESTAMP_US)
    {
      gimbal_rotation_history_.clear();
    }
    else if (TIMESTAMP_US == LAST_TIMESTAMP_US)
    {
      gimbal_rotation_history_.pop_back();
    }
  }

  gimbal_rotation_history_.push_back({timestamp, rotation});
  while (gimbal_rotation_history_.size() > MAX_GIMBAL_ROTATION_HISTORY_SIZE)
  {
    gimbal_rotation_history_.pop_front();
  }
  while (gimbal_rotation_history_.size() > 1U)
  {
    const double history_span_s =
        TrackerMath::DeltaTime(gimbal_rotation_history_.back().timestamp,
                               gimbal_rotation_history_.front().timestamp);
    if (history_span_s <= MAX_GIMBAL_ROTATION_HISTORY_S)
    {
      break;
    }
    gimbal_rotation_history_.pop_front();
  }

  gimbal_rotation_ = rotation;
  has_gimbal_rotation_ = true;
}

bool ArmorTracker::QueryGimbalRotationAt(LibXR::MicrosecondTimestamp timestamp,
                                         LibXR::Quaternion<double>& rotation) const
{
  LibXR::Mutex::LockGuard lock(gimbal_rotation_lock_);
  if (!has_gimbal_rotation_)
  {
    return false;
  }

  if (gimbal_rotation_history_.empty() || static_cast<uint64_t>(timestamp) == 0U)
  {
    rotation = gimbal_rotation_;
    return true;
  }

  auto upper_it = std::lower_bound(
      gimbal_rotation_history_.begin(), gimbal_rotation_history_.end(), timestamp,
      [](const TimedGimbalRotation& sample, const LibXR::MicrosecondTimestamp& target)
      {
        return static_cast<uint64_t>(sample.timestamp) < static_cast<uint64_t>(target);
      });

  if (upper_it == gimbal_rotation_history_.begin())
  {
    rotation = upper_it->rotation;
    return true;
  }

  if (upper_it == gimbal_rotation_history_.end())
  {
    rotation = gimbal_rotation_history_.back().rotation;
    return true;
  }

  const auto& NEXT_SAMPLE = *upper_it;
  const auto& PREV_SAMPLE = *(upper_it - 1);
  const double DT_S = TrackerMath::DeltaTime(NEXT_SAMPLE.timestamp, PREV_SAMPLE.timestamp);
  if (DT_S <= 1e-6)
  {
    rotation = NEXT_SAMPLE.rotation;
    return true;
  }

  const double INTERP_ALPHA =
      TrackerMath::DeltaTime(timestamp, PREV_SAMPLE.timestamp) / DT_S;
  rotation = slerp_quaternion(PREV_SAMPLE.rotation, NEXT_SAMPLE.rotation,
                              INTERP_ALPHA);
  return true;
}

void ArmorTracker::ArmorsCallback(ArmorDetectionsMessage& armors_msg)
{
  const auto start_time = std::chrono::steady_clock::now();
  const uint64_t FRAME_TIMESTAMP_VALUE =
      armors_msg.image_timestamp_us == 0U
          ? static_cast<uint64_t>(LibXR::Timebase::GetMicroseconds())
          : armors_msg.image_timestamp_us;
  const LibXR::MicrosecondTimestamp FRAME_TIMESTAMP(FRAME_TIMESTAMP_VALUE);

  if (state_ != State::LOST &&
      static_cast<uint64_t>(last_timestamp_) != 0U &&
      TrackerMath::DeltaTime(FRAME_TIMESTAMP, last_timestamp_) > MAX_TRACKER_DT_S)
  {
    XR_LOG_WARN("ArmorTracker reset because dt exceeded %.3f s", MAX_TRACKER_DT_S);
    ++reset_count_;
    state_ = State::LOST;
    detect_count_ = 0;
    temp_lost_count_ = 0;
    tracked_id_ = ArmorNumber::INVALID;
  }

  LibXR::Quaternion<double> frame_gimbal_rotation{};
  if (!QueryGimbalRotationAt(FRAME_TIMESTAMP, frame_gimbal_rotation))
  {
    XR_LOG_WARN("ArmorTracker skipped frame because gimbal rotation is unavailable");
    return;
  }

  auto observations = BuildObservations(armors_msg, frame_gimbal_rotation);
  SortObservations(observations);

  bool found = false;
  if (state_ == State::LOST)
  {
    found = SetTarget(observations, FRAME_TIMESTAMP);
  }
  else
  {
    found = UpdateTarget(observations, FRAME_TIMESTAMP);
  }

  StateMachine(found);

  if (state_ != State::LOST && target_.Diverged())
  {
    XR_LOG_WARN("ArmorTracker reset because target diverged");
    ++reset_count_;
    state_ = State::LOST;
    tracked_id_ = ArmorNumber::INVALID;
  }

  if (state_ != State::LOST)
  {
    const int recent_nis_failures = std::accumulate(
        target_.GetEkf().recent_nis_failures.begin(),
        target_.GetEkf().recent_nis_failures.end(), 0);
    if (recent_nis_failures >=
        static_cast<int>(BAD_CONVERGENCE_RATIO * target_.GetEkf().window_size))
    {
      XR_LOG_WARN("ArmorTracker reset because convergence degraded");
      ++reset_count_;
      state_ = State::LOST;
      tracked_id_ = ArmorNumber::INVALID;
    }
  }

  const double tracker_latency_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                start_time)
          .count();
  uint32_t projection_sample_count = 0U;
  double prediction_center_error_px = 0.0;
  double prediction_corner_error_px = 0.0;
  EvaluateProjectionMetrics(observations, frame_gimbal_rotation,
                            projection_sample_count,
                            prediction_center_error_px,
                            prediction_corner_error_px);
  PublishOutputs(observations.size(), found ? 1U : 0U, tracker_latency_ms,
                 projection_sample_count, prediction_center_error_px,
                 prediction_corner_error_px);
  if (ShouldShowPreview())
  {
    ShowDebugPreview(armors_msg, observations, found ? 1U : 0U,
                     frame_gimbal_rotation);
  }
  last_timestamp_ = FRAME_TIMESTAMP;
}

std::vector<TrackedArmorObservation> ArmorTracker::BuildObservations(
    const ArmorDetectionsMessage& armors_msg,
    const LibXR::Quaternion<double>& frame_gimbal_rotation) const
{
  std::vector<TrackedArmorObservation> observations;
  observations.reserve(armors_msg.results.size());

  std::shared_ptr<CameraBase::CameraInfo> camera_info;
  {
    LibXR::Mutex::LockGuard lock(preview_frame_lock_);
    camera_info = camera_info_;
  }
  const PreviewProjector YAW_PROJECTOR =
      build_preview_projector(camera_info, frame_gimbal_rotation,
                              camera_to_gimbal_transform_static_);

  const LibXR::Transform<double> world_to_gimbal(frame_gimbal_rotation, {0.0, 0.0, 0.0});
  for (const auto& armor : armors_msg.results)
  {
    const auto& pose = armor.pose;
    if (std::abs(pose.translation.x()) < 1e-6 && std::abs(pose.translation.y()) < 1e-6 &&
        std::abs(pose.translation.z()) < 1e-6)
    {
      continue;
    }

    const LibXR::Transform<double> world_pose =
        world_to_gimbal + camera_to_gimbal_transform_static_ + pose;
    const Eigen::Vector3d xyz_in_world(world_pose.translation.x(),
                                       world_pose.translation.y(),
                                       world_pose.translation.z());

    if (std::abs(xyz_in_world.z()) > cfg_.limits.max_z_position ||
        xyz_in_world.head<2>().norm() > cfg_.limits.max_armor_distance)
    {
      continue;
    }

    const auto euler = world_pose.rotation.ToEulerAngleZYX();
    const double RAW_YAW = GetArmorWorldYaw(world_pose.rotation);
    const double OPTIMIZED_YAW =
        optimize_observation_yaw(YAW_PROJECTOR, armor, xyz_in_world,
                                 frame_gimbal_rotation, RAW_YAW);

    TrackedArmorObservation observation;
    observation.result = armor;
    observation.result.pose = world_pose;
    observation.xyz_in_world = xyz_in_world;
    observation.ypr_in_world = Eigen::Vector3d(OPTIMIZED_YAW, euler[1], euler[0]);
    observation.ypd_in_world = TrackerMath::XyzToYpd(observation.xyz_in_world);
    observation.raw_yaw_in_world = RAW_YAW;
    observation.yaw_optimization_delta =
        TrackerMath::LimitRad(OPTIMIZED_YAW - RAW_YAW);
    observations.emplace_back(std::move(observation));
  }

  return observations;
}

void ArmorTracker::SortObservations(std::vector<TrackedArmorObservation>& armors) const
{
  std::stable_sort(
      armors.begin(), armors.end(),
      [](const TrackedArmorObservation& lhs, const TrackedArmorObservation& rhs)
      {
        return lhs.result.distance_to_image_center < rhs.result.distance_to_image_center;
      });

  std::stable_sort(
      armors.begin(), armors.end(),
      [](const TrackedArmorObservation& lhs, const TrackedArmorObservation& rhs)
      {
        return static_cast<int>(lhs.result.priority) <
               static_cast<int>(rhs.result.priority);
      });
}

bool ArmorTracker::SetTarget(std::vector<TrackedArmorObservation>& armors,
                             LibXR::MicrosecondTimestamp timestamp)
{
  if (armors.empty())
  {
    return false;
  }

  const TrackedArmorObservation& armor = armors.front();

  Eigen::VectorXd p0_diag(11);
  double radius = 0.2;
  int armor_count = 4;

  if (is_balance_armor(armor))
  {
    p0_diag << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
  }
  else if (armor.result.number == ArmorNumber::OUTPOST)
  {
    p0_diag << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0;
    radius = 0.2765;
    armor_count = 3;
  }
  else if (armor.result.number == ArmorNumber::BASE)
  {
    p0_diag << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
    radius = 0.3205;
    armor_count = 3;
  }
  else
  {
    p0_diag << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
  }

  target_ = Target(armor, timestamp, radius, armor_count, p0_diag);
  tracked_id_ = armor.result.number;
  return true;
}

bool ArmorTracker::UpdateTarget(std::vector<TrackedArmorObservation>& armors,
                                LibXR::MicrosecondTimestamp timestamp)
{
  target_.Predict(timestamp);
  target_.jumped = false;

  int matched_count = 0;
  for (const auto& armor : armors)
  {
    if (armor.result.number != tracked_id_ || armor.result.type != target_.armor_type)
    {
      continue;
    }
    ++matched_count;
  }

  if (matched_count == 0)
  {
    return false;
  }

  for (const auto& armor : armors)
  {
    if (armor.result.number != tracked_id_ || armor.result.type != target_.armor_type)
    {
      continue;
    }
    if (target_.Update(armor))
    {
      target_.jumped = true;
    }
  }
  return true;
}

void ArmorTracker::StateMachine(bool found)
{
  if (state_ == State::LOST)
  {
    if (!found)
    {
      return;
    }
    state_ = State::DETECTING;
    detect_count_ = 1;
    temp_lost_count_ = 0;
  }
  else if (state_ == State::DETECTING)
  {
    if (found)
    {
      ++detect_count_;
      if (detect_count_ >= cfg_.thresholds.min_detect_count)
      {
        state_ = State::TRACKING;
      }
    }
    else
    {
      detect_count_ = 0;
      state_ = State::LOST;
      tracked_id_ = ArmorNumber::INVALID;
    }
  }
  else if (state_ == State::TRACKING)
  {
    if (!found)
    {
      temp_lost_count_ = 1;
      state_ = State::TEMP_LOST;
    }
  }
  else if (state_ == State::TEMP_LOST)
  {
    if (found)
    {
      temp_lost_count_ = 0;
      state_ = State::TRACKING;
    }
    else
    {
      ++temp_lost_count_;
      const int max_temp_lost_count =
          (tracked_id_ == ArmorNumber::OUTPOST)
              ? cfg_.thresholds.outpost_max_temp_lost_count
              : cfg_.thresholds.max_temp_lost_count;
      if (temp_lost_count_ > max_temp_lost_count)
      {
        temp_lost_count_ = 0;
        state_ = State::LOST;
        tracked_id_ = ArmorNumber::INVALID;
      }
    }
  }
}

void ArmorTracker::PublishOutputs(std::size_t input_armor_count,
                                  std::size_t matched_armor_count,
                                  double tracker_latency_ms,
                                  uint32_t projection_sample_count,
                                  double prediction_center_error_px,
                                  double prediction_corner_error_px)
{
  ++frame_index_;

  info_msg_ = {};
  target_msg_ = {};

  metrics_msg_.frame_index = frame_index_;
  metrics_msg_.input_armor_count = static_cast<uint32_t>(input_armor_count);
  metrics_msg_.matched_armor_count = static_cast<uint32_t>(matched_armor_count);
  metrics_msg_.projection_sample_count = projection_sample_count;
  metrics_msg_.state = static_cast<uint32_t>(state_);
  metrics_msg_.reset_count = reset_count_;
  metrics_msg_.tracking =
      (state_ == State::TRACKING || state_ == State::TEMP_LOST);
  metrics_msg_.tracked_id = tracked_id_;
  metrics_msg_.tracker_latency_ms = tracker_latency_ms;
  metrics_msg_.prediction_center_error_px = prediction_center_error_px;
  metrics_msg_.prediction_corner_error_px = prediction_corner_error_px;

  if (state_ != State::LOST)
  {
    const auto& state = target_.GetState();
    info_msg_.position_diff = target_.GetEkf().data.at("residual_distance");
    info_msg_.yaw_diff = target_.GetEkf().data.at("residual_angle");
    info_msg_.position.x() = state[0];
    info_msg_.position.y() = state[2];
    info_msg_.position.z() = state[4];
    info_msg_.yaw = state[6];

    metrics_msg_.last_nis = target_.GetEkf().last_nis;
    metrics_msg_.recent_nis_failure_rate =
        target_.GetEkf().data.at("recent_nis_failures");
  }

  if (state_ == State::TRACKING || state_ == State::TEMP_LOST)
  {
    const auto& state = target_.GetState();
    target_msg_.tracking = true;
    target_msg_.id = tracked_id_;
    target_msg_.armor_type = target_.armor_type;
    target_msg_.armors_num = target_.GetArmorCount();
    target_msg_.jumped = target_.jumped;
    target_msg_.position.x() = state[0];
    target_msg_.velocity.x() = state[1];
    target_msg_.position.y() = state[2];
    target_msg_.velocity.y() = state[3];
    target_msg_.position.z() = state[4];
    target_msg_.velocity.z() = state[5];
    target_msg_.yaw = state[6];
    target_msg_.v_yaw = state[7];
    target_msg_.radius_1 = state[8];
    target_msg_.radius_2 = state[8] + state[9];
    target_msg_.dz = state[10];
  }

  info_topic_.Publish(info_msg_);
  metrics_topic_.Publish(metrics_msg_);
  target_topic_.Publish(target_msg_);

  if ((frame_index_ % 30U) == 0U)
  {
    XR_LOG_INFO(
        "ArmorTracker frame=%llu state=%d tracked=%d latency_ms=%.2f nis=%.3f pred_center_px=%.2f pred_corner_px=%.2f samples=%u",
        static_cast<unsigned long long>(frame_index_),
        static_cast<int>(state_), static_cast<int>(tracked_id_),
        metrics_msg_.tracker_latency_ms, metrics_msg_.last_nis,
        metrics_msg_.prediction_center_error_px,
        metrics_msg_.prediction_corner_error_px,
        metrics_msg_.projection_sample_count);
  }
}

void ArmorTracker::EvaluateProjectionMetrics(
    const std::vector<TrackedArmorObservation>& observations,
    const LibXR::Quaternion<double>& frame_gimbal_rotation,
    uint32_t& projection_sample_count, double& prediction_center_error_px,
    double& prediction_corner_error_px) const
{
  projection_sample_count = 0U;
  prediction_center_error_px = 0.0;
  prediction_corner_error_px = 0.0;

  if (state_ == State::LOST || tracked_id_ == ArmorNumber::INVALID || observations.empty())
  {
    return;
  }

  std::shared_ptr<CameraBase::CameraInfo> camera_info;
  {
    LibXR::Mutex::LockGuard lock(preview_frame_lock_);
    camera_info = camera_info_;
  }
  if (camera_info == nullptr)
  {
    return;
  }

  const PreviewProjector projector =
      build_preview_projector(camera_info, frame_gimbal_rotation,
                              camera_to_gimbal_transform_static_);
  if (!projector.valid)
  {
    return;
  }

  std::vector<ProjectedArmorCandidate> predicted_candidates;
  const auto predicted_armors = target_.GetArmorXYZAList();
  predicted_candidates.reserve(predicted_armors.size());

  for (const auto& predicted_armor : predicted_armors)
  {
    std::array<cv::Point, 4> projected_quad{};
        const auto predicted_corners =
            build_predicted_armor_corners(predicted_armor, target_.armor_type, tracked_id_);
        if (!project_predicted_armor_quad(projector, predicted_corners, projected_quad))
        {
          continue;
    }

    predicted_candidates.push_back({quad_center(projected_quad), projected_quad});
  }

  if (predicted_candidates.empty())
  {
    return;
  }

  double center_error_sum_px = 0.0;
  double corner_error_sum_px = 0.0;
  for (const auto& observation : observations)
  {
    if (observation.result.number != tracked_id_ ||
        observation.result.type != target_.armor_type)
    {
      continue;
    }

    double best_center_error_px = std::numeric_limits<double>::max();
    double best_corner_error_px = std::numeric_limits<double>::max();
    for (const auto& candidate : predicted_candidates)
    {
      const double center_error_px =
          cv::norm(observation.result.center - candidate.center);
      if (center_error_px >= best_center_error_px)
      {
        continue;
      }

      best_center_error_px = center_error_px;
      best_corner_error_px =
          quad_rms_error_px(candidate.quad, observation.result.points);
    }

    if (!std::isfinite(best_center_error_px) || !std::isfinite(best_corner_error_px))
    {
      continue;
    }

    center_error_sum_px += best_center_error_px;
    corner_error_sum_px += best_corner_error_px;
    ++projection_sample_count;
  }

  if (projection_sample_count == 0U)
  {
    return;
  }

  prediction_center_error_px =
      center_error_sum_px / static_cast<double>(projection_sample_count);
  prediction_corner_error_px =
      corner_error_sum_px / static_cast<double>(projection_sample_count);
}

void ArmorTracker::ShowDebugPreview(
    const ArmorDetectionsMessage& armors_msg,
    const std::vector<TrackedArmorObservation>& observations,
    std::size_t matched_armor_count,
    const LibXR::Quaternion<double>& frame_gimbal_rotation)
{
  try
  {
    cv::Mat frame;
    std::shared_ptr<CameraBase::CameraInfo> camera_info;
    {
      LibXR::Mutex::LockGuard lock(preview_frame_lock_);
      if (!latest_frame_.empty())
      {
        frame = latest_frame_.clone();
      }
      camera_info = camera_info_;
    }

    if (frame.empty())
    {
      frame = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(20, 24, 30));
      cv::putText(frame, "No image frame yet", cv::Point(40, 80),
                  cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(220, 224, 230), 1,
                  cv::LINE_AA);
    }

    cv::Mat canvas(frame.rows, frame.cols + INFO_PANEL_WIDTH, CV_8UC3,
                   cv::Scalar(18, 22, 28));
    frame.copyTo(canvas(cv::Rect(0, 0, frame.cols, frame.rows)));

    cv::Mat header = canvas(cv::Rect(0, 0, frame.cols, 58));
    cv::Mat header_overlay = header.clone();
    cv::rectangle(header_overlay, cv::Rect(0, 0, header.cols, header.rows),
                  cv::Scalar(10, 16, 22), cv::FILLED);
    cv::addWeighted(header_overlay, HEADER_BAR_ALPHA, header, 1.0 - HEADER_BAR_ALPHA, 0.0,
                    header);

    cv::putText(canvas, "ArmorTracker Preview", cv::Point(18, 36),
                cv::FONT_HERSHEY_DUPLEX, 0.88, cv::Scalar(240, 244, 250), 1,
                cv::LINE_AA);
    cv::putText(canvas, "state machine + EKF + aim command", cv::Point(18, 56),
                cv::FONT_HERSHEY_DUPLEX, 0.50, cv::Scalar(151, 170, 192), 1,
                cv::LINE_AA);
    draw_state_chip(canvas, state_to_string(state_), cv::Point(18, 92),
                    state_to_color(state_));

    if (cfg_.debug.draw_candidates)
    {
      for (const auto& armor : armors_msg.results)
      {
        const bool tracked_candidate =
            (state_ != State::LOST && armor.number == tracked_id_);
        const cv::Scalar armor_color =
            tracked_candidate ? state_to_color(State::TRACKING)
                              : armor_color_to_scalar(armor.color);

        std::array<cv::Point, 4> polygon{};
        for (std::size_t point_index = 0; point_index < armor.points.size(); ++point_index)
        {
          polygon[point_index] = armor.points[point_index];
        }

        const cv::Point* polygon_points = polygon.data();
        const int polygon_size = static_cast<int>(polygon.size());
        cv::polylines(canvas, &polygon_points, &polygon_size, 1, true, armor_color,
                      tracked_candidate ? 3 : 2, cv::LINE_AA);
        cv::rectangle(canvas, armor.box, armor_color, tracked_candidate ? 2 : 1,
                      cv::LINE_AA);
        cv::circle(canvas, armor.center, 4, armor_color, cv::FILLED, cv::LINE_AA);

        std::ostringstream label;
        if (tracked_candidate)
        {
          label << "TRACK ";
        }
        label << armor_number_to_string(armor.number) << " "
              << format_value(armor.confidence, 2);
        cv::putText(canvas, label.str(),
                    cv::Point(std::max(armor.box.x, 8),
                              std::max(armor.box.y - 8, 118)),
                    cv::FONT_HERSHEY_DUPLEX, 0.52, armor_color, 1, cv::LINE_AA);
      }
    }

    std::size_t predicted_visible_count = 0U;
    if (state_ != State::LOST)
    {
      const PreviewProjector projector = build_preview_projector(
          camera_info, frame_gimbal_rotation, camera_to_gimbal_transform_static_);
      if (projector.valid)
      {
        const auto& state = target_.GetState();
        const Eigen::Vector3d target_center_world(state[0], state[2], state[4]);
        cv::Point2f projected_center_f(0.0F, 0.0F);
        const bool center_visible =
            projector.Project(target_center_world, projected_center_f);
        const cv::Point projected_center(
            cvRound(projected_center_f.x), cvRound(projected_center_f.y));

        if (center_visible)
        {
          const cv::Rect center_box(projected_center.x - 7, projected_center.y - 7, 14, 14);
          cv::rectangle(canvas, center_box, cv::Scalar(255, 122, 162), 2, cv::LINE_AA);
          cv::putText(canvas, "EKF center",
                      projected_center + cv::Point(10, -10),
                      cv::FONT_HERSHEY_DUPLEX, 0.46, cv::Scalar(255, 182, 202), 1,
                      cv::LINE_AA);
        }

        const auto predicted_armors = target_.GetArmorXYZAList();
        for (std::size_t armor_index = 0; armor_index < predicted_armors.size();
             ++armor_index)
        {
          cv::Point2f projected_armor_f;
          if (!projector.Project(predicted_armors[armor_index].head<3>(),
                                 projected_armor_f))
          {
            continue;
          }

          const cv::Point projected_armor(
              cvRound(projected_armor_f.x), cvRound(projected_armor_f.y));
          const bool is_primary_armor = (armor_index == 0U);
          const cv::Scalar overlay_color =
              is_primary_armor ? cv::Scalar(74, 226, 255)
                               : cv::Scalar(197, 139, 255);

          std::array<cv::Point, 4> projected_quad{};
          const auto predicted_corners = build_predicted_armor_corners(
              predicted_armors[armor_index], target_.armor_type, tracked_id_);
          if (!project_predicted_armor_quad(projector, predicted_corners, projected_quad))
          {
            continue;
          }

          ++predicted_visible_count;
          const cv::Scalar fill_color =
              is_primary_armor ? cv::Scalar(26, 110, 125)
                               : cv::Scalar(86, 58, 117);
          draw_projected_armor_quad(canvas, projected_quad, overlay_color, fill_color,
                                    is_primary_armor ? 2 : 1);

          std::ostringstream projected_label;
          projected_label << "P" << armor_index;
          if (is_primary_armor)
          {
            projected_label << " main";
          }
          cv::Point label_anchor = projected_armor + cv::Point(10, -8);
          label_anchor.x = std::max(label_anchor.x, 8);
          label_anchor.y = std::max(label_anchor.y, 118);
          cv::putText(canvas, projected_label.str(),
                      label_anchor,
                      cv::FONT_HERSHEY_DUPLEX, 0.48, overlay_color, 1, cv::LINE_AA);
        }
      }
    }

    const cv::Point image_center(frame.cols / 2, frame.rows / 2);
    cv::drawMarker(canvas, image_center, cv::Scalar(80, 92, 110), cv::MARKER_CROSS, 22, 1,
                   cv::LINE_AA);

    const int panel_x = frame.cols + 18;
    int panel_y = 42;

    cv::putText(canvas, "Tracker State", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.74, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 28;
    draw_info_row(canvas, panel_x, panel_y, "state", state_to_string(state_),
                  state_to_color(state_));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "tracked_id",
                  armor_number_to_string(tracked_id_),
                  tracked_id_ == ArmorNumber::INVALID ? cv::Scalar(151, 170, 192)
                                                      : cv::Scalar(240, 244, 250));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "input_armors",
                  std::to_string(metrics_msg_.input_armor_count),
                  cv::Scalar(255, 214, 102));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "matched_armors",
                  std::to_string(matched_armor_count),
                  matched_armor_count > 0U ? cv::Scalar(128, 226, 142)
                                           : cv::Scalar(255, 166, 77));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "pred_armors",
                  std::to_string(predicted_visible_count),
                  predicted_visible_count > 0U ? cv::Scalar(74, 226, 255)
                                               : cv::Scalar(151, 170, 192));
    panel_y += 24;
    draw_info_row(
        canvas, panel_x, panel_y, "pred_center_px",
        metrics_msg_.projection_sample_count > 0U
            ? format_value(metrics_msg_.prediction_center_error_px, 1)
            : "--",
        metrics_msg_.projection_sample_count > 0U
            ? prediction_error_to_color(metrics_msg_.prediction_center_error_px)
            : cv::Scalar(151, 170, 192));
    panel_y += 24;
    draw_info_row(
        canvas, panel_x, panel_y, "pred_corner_px",
        metrics_msg_.projection_sample_count > 0U
            ? format_value(metrics_msg_.prediction_corner_error_px, 1)
            : "--",
        metrics_msg_.projection_sample_count > 0U
            ? prediction_error_to_color(metrics_msg_.prediction_corner_error_px)
            : cv::Scalar(151, 170, 192));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "reset_count",
                  std::to_string(metrics_msg_.reset_count),
                  cv::Scalar(255, 166, 77));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "tracker_ms",
                  format_value(metrics_msg_.tracker_latency_ms, 2),
                  cv::Scalar(91, 196, 255));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "last_nis",
                  format_value(metrics_msg_.last_nis, 3),
                  metrics_msg_.last_nis > 9.488 ? cv::Scalar(255, 120, 120)
                                                : cv::Scalar(128, 226, 142));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "nis_fail_rate",
                  format_value(metrics_msg_.recent_nis_failure_rate, 3),
                  metrics_msg_.recent_nis_failure_rate > BAD_CONVERGENCE_RATIO
                      ? cv::Scalar(255, 120, 120)
                      : cv::Scalar(240, 244, 250));

    panel_y += 36;
    cv::putText(canvas, "Tracker Flags", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.74, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 28;
    draw_info_row(canvas, panel_x, panel_y, "jumped", target_.jumped ? "true" : "false",
                  target_.jumped ? cv::Scalar(128, 226, 142)
                                 : cv::Scalar(151, 170, 192));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "armor_type",
                  target_.armor_type == ArmorType::LARGE ? "large" : "small",
                  cv::Scalar(240, 244, 250));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "detect_count",
                  std::to_string(detect_count_), cv::Scalar(240, 244, 250));
    panel_y += 24;
    draw_info_row(canvas, panel_x, panel_y, "temp_lost",
                  std::to_string(temp_lost_count_), cv::Scalar(240, 244, 250));

    panel_y += 36;
    cv::putText(canvas, "Active Target", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.74, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 28;

    if (state_ == State::LOST)
    {
      cv::putText(canvas, "No active EKF target", cv::Point(panel_x, panel_y),
                  cv::FONT_HERSHEY_DUPLEX, 0.56, cv::Scalar(151, 170, 192), 1,
                  cv::LINE_AA);
      panel_y += 24;
    }
    else
    {
      const auto& state = target_.GetState();
      draw_info_row(canvas, panel_x, panel_y, "x", format_value(state[0], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "vx", format_value(state[1], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "y", format_value(state[2], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "vy", format_value(state[3], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "z", format_value(state[4], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "vz", format_value(state[5], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "yaw_rad", format_value(state[6], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "v_yaw", format_value(state[7], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "r1", format_value(state[8], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "r2", format_value(state[8] + state[9], 3),
                    cv::Scalar(240, 244, 250));
      panel_y += 24;
      draw_info_row(canvas, panel_x, panel_y, "dz", format_value(state[10], 3),
                    cv::Scalar(240, 244, 250));
    }

    panel_y += 36;
    cv::putText(canvas, "Observations", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.74, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 26;

    if (observations.empty())
    {
      cv::putText(canvas, "No valid observation after filtering",
                  cv::Point(panel_x, panel_y), cv::FONT_HERSHEY_DUPLEX, 0.54,
                  cv::Scalar(151, 170, 192), 1, cv::LINE_AA);
    }
    else
    {
      const int observation_count =
          std::min(static_cast<int>(observations.size()), MAX_DEBUG_OBSERVATIONS);
      for (int index = 0; index < observation_count; ++index)
      {
        const auto& observation = observations[index];
        const bool tracked_candidate =
            (state_ != State::LOST && observation.result.number == tracked_id_);
        const cv::Scalar item_color =
            tracked_candidate ? state_to_color(State::TRACKING)
                              : armor_color_to_scalar(observation.result.color);
        const cv::Rect item_rect(panel_x - 10, panel_y - 18, INFO_PANEL_WIDTH - 32, 56);
        cv::rectangle(canvas, item_rect, cv::Scalar(32, 39, 48), cv::FILLED,
                      cv::LINE_AA);
        cv::rectangle(canvas, item_rect, item_color, 1, cv::LINE_AA);

        cv::putText(canvas, armor_number_to_string(observation.result.number),
                    cv::Point(panel_x, panel_y), cv::FONT_HERSHEY_DUPLEX, 0.56,
                    cv::Scalar(245, 247, 250), 1, cv::LINE_AA);

        std::ostringstream world_xyz;
        world_xyz << "xyz=" << format_value(observation.xyz_in_world.x(), 2) << ", "
                  << format_value(observation.xyz_in_world.y(), 2) << ", "
                  << format_value(observation.xyz_in_world.z(), 2);
        cv::putText(canvas, world_xyz.str(), cv::Point(panel_x, panel_y + 20),
                    cv::FONT_HERSHEY_DUPLEX, 0.42, item_color, 1, cv::LINE_AA);

        std::ostringstream ypd;
        ypd << "yaw=" << format_value(observation.ypr_in_world.x(), 2)
            << " d=" << format_value(observation.yaw_optimization_delta, 2)
            << "  dist=" << format_value(observation.ypd_in_world.z(), 2);
        cv::putText(canvas, ypd.str(), cv::Point(panel_x, panel_y + 38),
                    cv::FONT_HERSHEY_DUPLEX, 0.42, cv::Scalar(210, 220, 232), 1,
                    cv::LINE_AA);
        panel_y += 64;
      }
    }

    cv::Mat display = canvas;
    if (std::abs(cfg_.debug.overlay_scale - 1.0) > 1e-6)
    {
      cv::resize(canvas, display, cv::Size(), cfg_.debug.overlay_scale,
                 cfg_.debug.overlay_scale);
    }

    cv::imshow("armor_tracker_debug", display);
    cv::waitKey(std::max(cfg_.debug.wait_key_ms, 1));
  }
  catch (const cv::Exception& exception)
  {
    preview_available_ = false;
    if (!preview_warned_)
    {
      preview_warned_ = true;
      XR_LOG_WARN("ArmorTracker preview disabled: %s", exception.what());
    }
  }
}

bool ArmorTracker::ShouldShowPreview()
{
  if (!cfg_.debug.preview || !preview_available_)
  {
    return false;
  }

  const char* display = std::getenv("DISPLAY");
  const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
  if (display == nullptr && wayland_display == nullptr)
  {
    preview_available_ = false;
    if (!preview_warned_)
    {
      preview_warned_ = true;
      XR_LOG_WARN("ArmorTracker preview disabled because DISPLAY is unavailable");
    }
    return false;
  }

  return true;
}

double ArmorTracker::GetArmorWorldYaw(const LibXR::Quaternion<double>& rotation) const
{
  const auto euler = rotation.ToEulerAngleZYX();
  return TrackerMath::LimitRad(euler[2]);
}
