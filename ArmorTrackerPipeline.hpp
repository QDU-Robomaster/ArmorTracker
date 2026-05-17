#pragma once

/**
 * @file ArmorTrackerPipeline.hpp
 * @brief Template implementations for tracker topic wiring and preview output.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker_detail::Config ArmorTracker<CameraInfoV>::BuildTrackerConfig() const
{
  armor_tracker_detail::Config config{};
  config.require_target_tag = cfg_.tracker.require_target_tag;
  config.target_tag_id = cfg_.tracker.target_tag_id;
  config.min_detect_count = cfg_.tracker.min_detect_count;
  config.max_temp_lost_count = cfg_.tracker.max_temp_lost_count;
  config.outpost_max_temp_lost_count = cfg_.tracker.outpost_max_temp_lost_count;
  config.target_select.observed_count_weight =
      cfg_.tracker.target_select.observed_count_weight;
  config.target_select.distance_weight =
      cfg_.tracker.target_select.distance_weight;
  config.target_select.area_weight = cfg_.tracker.target_select.area_weight;
  config.target_select.spin_weight = cfg_.tracker.target_select.spin_weight;
  config.target_select.angle_weight = cfg_.tracker.target_select.angle_weight;
  config.target_select.max_distance_m =
      cfg_.tracker.target_select.max_distance_m;
  config.target_select.distance_span_m =
      cfg_.tracker.target_select.distance_span_m;
  config.target_select.area_norm_px = cfg_.tracker.target_select.area_norm_px;
  config.target_select.observed_count_norm =
      cfg_.tracker.target_select.observed_count_norm;
  config.target_select.max_spin_rad_s =
      cfg_.tracker.target_select.max_spin_rad_s;
  config.target_select.max_angle_norm =
      cfg_.tracker.target_select.max_angle_norm;
  config.target_select.detecting_scale =
      cfg_.tracker.target_select.detecting_scale;
  config.target_select.temp_lost_scale =
      cfg_.tracker.target_select.temp_lost_scale;
  config.target_select.switch_margin = cfg_.tracker.target_select.switch_margin;
  config.camera_matrix = {
      kCameraInfo.camera_matrix[0], kCameraInfo.camera_matrix[1],
      kCameraInfo.camera_matrix[2], kCameraInfo.camera_matrix[3],
      kCameraInfo.camera_matrix[4], kCameraInfo.camera_matrix[5],
      kCameraInfo.camera_matrix[6], kCameraInfo.camera_matrix[7],
      kCameraInfo.camera_matrix[8]};
  config.camera_mount_to_body_rotation =
      cfg_.extrinsic.camera_mount_to_body.rotation;
  config.camera_mount_to_body_translation =
      cfg_.extrinsic.camera_mount_to_body.translation;
  return config;
}

template <CameraTypes::CameraInfo CameraInfoV>
ArmorTracker<CameraInfoV>::ArmorTracker(LibXR::HardwareContainer& hw,
                                        LibXR::ApplicationManager&,
                                        Config cfg, FrameSync& sync)
    : cfg_(std::move(cfg)),
      cmd_file_(LibXR::RamFS::CreateFile(name_, CommandFun, this)),
      sync_(sync)
{
  XR_LOG_INFO("Starting ArmorTracker");
  tracker_.Configure(BuildTrackerConfig());
  preview_.Start(cfg_.preview);
  hw.template FindOrExit<LibXR::RamFS>({"ramfs"})->Add(cmd_file_);
  SubscribeDetectorTopic();
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SubscribeDetectorTopic()
{
  armors_topic_ = LibXR::Topic(LibXR::Topic::WaitTopic(
      kDetectorTopicName, UINT32_MAX, &armor_detector_domain_));
  auto armors_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        if (self->params_is_changed_)
        {
          self->SetConfig(self->cfg_);
          self->params_is_changed_ = false;
        }
        if constexpr (std::is_pointer<DetectionMessage>::value)
        {
          auto* message_addr = reinterpret_cast<DetectionMessage*>(data.addr_);
          self->ArmorsCallback(message_addr != nullptr ? *message_addr : nullptr);
        }
        else
        {
          auto* message_addr = reinterpret_cast<DetectionMessage*>(data.addr_);
          if (message_addr == nullptr)
          {
            XR_LOG_ERROR("ArmorTracker received empty detector message");
            return;
          }
          self->ArmorsCallback(*message_addr);
        }
      },
      this);
  armors_topic_.RegisterCallback(armors_cb);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::OnMonitor()
{
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SetConfig(const Config& cfg)
{
  cfg_ = cfg;
  tracker_.Configure(BuildTrackerConfig());
  preview_.Stop();
  preview_.Start(cfg_.preview);
}

template <CameraTypes::CameraInfo CameraInfoV>
int ArmorTracker<CameraInfoV>::CommandFun(ArmorTracker<CameraInfoV>* self,
                                          int argc, char** argv)
{
  if (argc == 1)
  {
    TRACKER_STDIO_PRINT("ArmorTracker\n\n");
    TRACKER_STDIO_PRINT("Usage\r\n");
    TRACKER_STDIO_PRINT("  show\r\n");
    TRACKER_STDIO_PRINT("  target_tag_id <value>\r\n");
    TRACKER_STDIO_PRINT("  require_target_tag <0|1>\r\n");
    return 0;
  }

  if (argc == 2 && std::string(argv[1]) == "show")
  {
    TRACKER_STDIO_PRINT("name: ArmorTracker\r\n");
    TRACKER_STDIO_PRINT("cfg:\r\n");
    TRACKER_STDIO_PRINT("  tracker:\r\n");
    TRACKER_STDIO_PRINTF("    require_target_tag: %d\r\n",
                         self->cfg_.tracker.require_target_tag ? 1 : 0);
    TRACKER_STDIO_PRINTF("    target_tag_id: %d\r\n",
                         self->cfg_.tracker.target_tag_id);
    TRACKER_STDIO_PRINTF("    min_detect_count: %d\r\n",
                         self->cfg_.tracker.min_detect_count);
    TRACKER_STDIO_PRINTF("    max_temp_lost_count: %d\r\n",
                         self->cfg_.tracker.max_temp_lost_count);
    TRACKER_STDIO_PRINTF("    outpost_max_temp_lost_count: %d\r\n",
                         self->cfg_.tracker.outpost_max_temp_lost_count);
    TRACKER_STDIO_PRINT("    target_select:\r\n");
    TRACKER_STDIO_PRINTF("      observed_count_weight: %f\r\n",
                         self->cfg_.tracker.target_select.observed_count_weight);
    TRACKER_STDIO_PRINTF("      distance_weight: %f\r\n",
                         self->cfg_.tracker.target_select.distance_weight);
    TRACKER_STDIO_PRINTF("      area_weight: %f\r\n",
                         self->cfg_.tracker.target_select.area_weight);
    TRACKER_STDIO_PRINTF("      spin_weight: %f\r\n",
                         self->cfg_.tracker.target_select.spin_weight);
    TRACKER_STDIO_PRINTF("      angle_weight: %f\r\n",
                         self->cfg_.tracker.target_select.angle_weight);
    TRACKER_STDIO_PRINTF("      max_distance_m: %f\r\n",
                         self->cfg_.tracker.target_select.max_distance_m);
    TRACKER_STDIO_PRINTF("      distance_span_m: %f\r\n",
                         self->cfg_.tracker.target_select.distance_span_m);
    TRACKER_STDIO_PRINTF("      area_norm_px: %f\r\n",
                         self->cfg_.tracker.target_select.area_norm_px);
    TRACKER_STDIO_PRINTF("      observed_count_norm: %f\r\n",
                         self->cfg_.tracker.target_select.observed_count_norm);
    TRACKER_STDIO_PRINTF("      max_spin_rad_s: %f\r\n",
                         self->cfg_.tracker.target_select.max_spin_rad_s);
    TRACKER_STDIO_PRINTF("      max_angle_norm: %f\r\n",
                         self->cfg_.tracker.target_select.max_angle_norm);
    TRACKER_STDIO_PRINTF("      detecting_scale: %f\r\n",
                         self->cfg_.tracker.target_select.detecting_scale);
    TRACKER_STDIO_PRINTF("      temp_lost_scale: %f\r\n",
                         self->cfg_.tracker.target_select.temp_lost_scale);
    TRACKER_STDIO_PRINTF("      switch_margin: %f\r\n",
                         self->cfg_.tracker.target_select.switch_margin);
    TRACKER_STDIO_PRINT("  extrinsic:\r\n");
    TRACKER_STDIO_PRINT("    camera_mount_to_body:\r\n");
    TRACKER_STDIO_PRINT("      rotation:\r\n");
    TRACKER_STDIO_PRINTF("        - %f\r\n",
                         self->cfg_.extrinsic.camera_mount_to_body.rotation[0]);
    TRACKER_STDIO_PRINTF("        - %f\r\n",
                         self->cfg_.extrinsic.camera_mount_to_body.rotation[1]);
    TRACKER_STDIO_PRINTF("        - %f\r\n",
                         self->cfg_.extrinsic.camera_mount_to_body.rotation[2]);
    TRACKER_STDIO_PRINTF("        - %f\r\n",
                         self->cfg_.extrinsic.camera_mount_to_body.rotation[3]);
    TRACKER_STDIO_PRINT("      translation:\r\n");
    TRACKER_STDIO_PRINTF("        - %f\r\n",
                         self->cfg_.extrinsic.camera_mount_to_body.translation[0]);
    TRACKER_STDIO_PRINTF("        - %f\r\n",
                         self->cfg_.extrinsic.camera_mount_to_body.translation[1]);
    TRACKER_STDIO_PRINTF("        - %f\r\n",
                         self->cfg_.extrinsic.camera_mount_to_body.translation[2]);
    return 0;
  }

  if (argc == 3)
  {
    const std::string cmd = argv[1];
    if (cmd == "target_tag_id")
    {
      self->cfg_.tracker.target_tag_id = std::stoi(argv[2]);
    }
    else if (cmd == "require_target_tag")
    {
      self->cfg_.tracker.require_target_tag = std::stoi(argv[2]) != 0;
    }
    else
    {
      TRACKER_STDIO_PRINTF("Unknown command: %s\n", argv[1]);
      return -1;
    }
    self->params_is_changed_ = true;
    return 0;
  }

  TRACKER_STDIO_PRINTF("Unknown command: %s\n", argc > 1 ? argv[1] : "");
  return -1;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::ArmorsCallback(
    typename ArmorTracker<CameraInfoV>::DetectionMessageArg message)
{
  const ArmorDetectionsSourceFrame<CameraInfoV>* source_frame_ptr = nullptr;
  const ArmorDetectorResults* detections_ptr = nullptr;
  uint64_t detections_timestamp_us = 0;

  if constexpr (std::is_pointer<DetectionMessage>::value)
  {
    if (message == nullptr)
    {
      XR_LOG_ERROR("ArmorTracker received empty detector packet pointer");
      return;
    }
    if (message->detections == nullptr)
    {
      XR_LOG_ERROR("ArmorTracker received detector packet without detections");
      return;
    }
    source_frame_ptr = &message->source_frame;
    detections_ptr = &message->detections->results;
    detections_timestamp_us = message->detections->image_timestamp_us;
  }
  else
  {
    source_frame_ptr = &message.source_frame;
    detections_ptr = &message.results;
    detections_timestamp_us = message.source_frame.image_timestamp_us;
  }

  const auto& source_frame = *source_frame_ptr;
  if (source_frame.image_frame == nullptr)
  {
    XR_LOG_ERROR("ArmorTracker received detector packet without image frame");
    return;
  }
  if (source_frame.imu == nullptr)
  {
    XR_LOG_ERROR("ArmorTracker received detector packet without synced imu");
    return;
  }

  const uint64_t image_timestamp_us = source_frame.image_timestamp_us;
  if (source_frame.image_frame->timestamp_us != image_timestamp_us)
  {
    XR_LOG_ERROR("ArmorTracker detector packet timestamp mismatch image=%u packet=%u",
                 static_cast<unsigned>(source_frame.image_frame->timestamp_us),
                 static_cast<unsigned>(image_timestamp_us));
    return;
  }
  if (detections_timestamp_us != image_timestamp_us)
  {
    XR_LOG_ERROR("ArmorTracker detector result timestamp mismatch result=%u packet=%u",
                 static_cast<unsigned>(detections_timestamp_us),
                 static_cast<unsigned>(image_timestamp_us));
    return;
  }

  const ArmorDetectorResults& detector_armors = *detections_ptr;
  std::vector<armor_tracker_detail::InputArmor> inputs;
  inputs.reserve(detector_armors.size());
  for (const auto& armor : detector_armors)
  {
    armor_tracker_detail::InputArmor input{};
    input.tag_id = static_cast<int>(armor.number);
    input.armor_type = static_cast<int>(armor.type);
    input.confidence = armor.confidence;
    input.corners = armor.points;
    input.center = armor.center;
    input.center_norm = armor.center_norm;
    inputs.push_back(input);
  }

  Eigen::Quaterniond q_body_to_world(
      source_frame.imu->rotation_wxyz[0], source_frame.imu->rotation_wxyz[1],
      source_frame.imu->rotation_wxyz[2], source_frame.imu->rotation_wxyz[3]);
  if (!std::isfinite(q_body_to_world.norm()) ||
      q_body_to_world.norm() < 1e-9)
  {
    q_body_to_world = Eigen::Quaterniond::Identity();
  }
  q_body_to_world.normalize();

  const auto output = tracker_.Step(image_timestamp_us, q_body_to_world, inputs);

  ArmorTrackerTarget target_msg{};
  target_msg.image_timestamp_us = image_timestamp_us;
  target_msg.id = ArmorNumber::INVALID;

  if (output.has_target)
  {
    target_msg.tracking = true;
    target_msg.id = static_cast<ArmorNumber>(std::clamp(
        output.selected_tag_id, 0, static_cast<int>(ArmorNumber::NEGATIVE)));
    target_msg.armors_num = output.armors_num;
    target_msg.position = output.center;
    target_msg.velocity = output.velocity;
    target_msg.yaw = output.yaw;
    target_msg.v_yaw = output.vyaw;
    target_msg.radius_1 = output.radius_even;
    target_msg.radius_2 = output.radius_odd;
    target_msg.dz = output.dz;
    target_msg.tracked_face_index = output.selected_face;
    target_msg.face_switch_observed = output.jumped;
  }
  else
  {
    target_msg.tracking = false;
  }

  const LibXR::MicrosecondTimestamp publish_timestamp(image_timestamp_us);
  target_frame_target_msg_ = target_msg;
  target_frame_packet_.source_frame = source_frame;
  target_frame_packet_.target = &target_frame_target_msg_;

  const Eigen::Matrix3d R_camera_to_body =
      armor_tracker_detail::CameraToBodyRotationFromMountExtrinsic(
          cfg_.extrinsic.camera_mount_to_body.rotation);
  const Eigen::Vector3d t_camera_to_body(
      cfg_.extrinsic.camera_mount_to_body.translation[0],
      cfg_.extrinsic.camera_mount_to_body.translation[1],
      cfg_.extrinsic.camera_mount_to_body.translation[2]);
  const Eigen::Matrix3d R_body_to_world =
      armor_tracker_detail::BodyToWorldRotationFromImu(q_body_to_world);
  const Eigen::Matrix3d R_output_to_camera =
      R_camera_to_body.transpose() * R_body_to_world.transpose();
  const Eigen::Vector3d t_output_to_camera =
      -R_camera_to_body.transpose() * t_camera_to_body;
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      target_frame_packet_.output_to_camera_rotation[static_cast<std::size_t>(
          row * 3 + col)] = R_output_to_camera(row, col);
    }
    target_frame_packet_.output_to_camera_translation[static_cast<std::size_t>(
        row)] = t_output_to_camera(row);
  }

  TargetFrameMessage target_frame_msg = &target_frame_packet_;
  target_frame_topic_.Publish(target_frame_msg, publish_timestamp);
  SubmitPreview(*source_frame.image_frame, detector_armors, target_msg, output);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SubmitPreview(
    const ImageFrame& image_frame, const ArmorDetectorResults& detector_armors,
    const ArmorTrackerTarget& target_msg,
    const armor_tracker_detail::Output& output)
{
  if (!preview_.Running())
  {
    return;
  }

  int cv_type = -1;
  switch (kCameraInfo.encoding)
  {
    case CameraTypes::Encoding::RGB8:
    case CameraTypes::Encoding::BGR8:
      cv_type = CV_8UC3;
      break;
    case CameraTypes::Encoding::RGBA8:
    case CameraTypes::Encoding::BGRA8:
      cv_type = CV_8UC4;
      break;
    case CameraTypes::Encoding::MONO8:
      cv_type = CV_8UC1;
      break;
    default:
      break;
  }
  if (cv_type < 0)
  {
    return;
  }

  cv::Mat image(static_cast<int>(kCameraInfo.height),
                static_cast<int>(kCameraInfo.width), cv_type,
                const_cast<uint8_t*>(image_frame.data.data()),
                static_cast<size_t>(kCameraInfo.step));
  cv::Mat bgr_image;
  switch (kCameraInfo.encoding)
  {
    case CameraTypes::Encoding::RGB8:
      cv::cvtColor(image, bgr_image, cv::COLOR_RGB2BGR);
      break;
    case CameraTypes::Encoding::RGBA8:
      cv::cvtColor(image, bgr_image, cv::COLOR_RGBA2BGR);
      break;
    case CameraTypes::Encoding::BGRA8:
      cv::cvtColor(image, bgr_image, cv::COLOR_BGRA2BGR);
      break;
    case CameraTypes::Encoding::MONO8:
      cv::cvtColor(image, bgr_image, cv::COLOR_GRAY2BGR);
      break;
    case CameraTypes::Encoding::BGR8:
      bgr_image = image;
      break;
    default:
      return;
  }
  if (bgr_image.empty())
  {
    return;
  }

  struct ArmorOverlay
  {
    bool valid = false;
    cv::Point center_uv{};
    std::array<cv::Point, 4> corners_uv{};
  };

  struct TrackOverlay
  {
    int tag_id = -1;
    std::string state{"lost"};
    bool selected = false;
    double score = 0.0;
    bool center_valid = false;
    cv::Point center_uv{};
    int face_count = 0;
    std::array<ArmorOverlay, 4> faces{};
  };

  const auto to_point = [](const cv::Point2f& point)
  {
    return cv::Point(static_cast<int>(std::lround(point.x)),
                     static_cast<int>(std::lround(point.y)));
  };

  std::vector<TrackOverlay> track_overlays;
  track_overlays.reserve(output.tracks.size());
  for (const auto& track : output.tracks)
  {
    TrackOverlay overlay;
    overlay.tag_id = track.tag_id;
    overlay.state = track.state;
    overlay.selected = track.selected;
    overlay.score = track.score;
    overlay.face_count = std::min(
        4, std::min(track.armors_num,
                    static_cast<int>(track.faces_world.size())));

    for (int i = 0; i < overlay.face_count; ++i)
    {
      const Eigen::Vector4d face =
          track.faces_world[static_cast<std::size_t>(i)];
      const Eigen::Vector3d center_world = face.head<3>();
      const double yaw = face[3];
      if (!center_world.allFinite() || !std::isfinite(yaw))
      {
        continue;
      }

      const auto corners =
          tracker_.ReprojectArmorFace(center_world, yaw, track.armor_type,
                                      track.tag_id);
      if (corners.size() != 4U)
      {
        continue;
      }
      cv::Point2f center_uv(0.0F, 0.0F);
      bool valid = true;
      for (std::size_t corner = 0; corner < corners.size(); ++corner)
      {
        valid = valid && std::isfinite(corners[corner].x) &&
                std::isfinite(corners[corner].y);
        center_uv += corners[corner] * 0.25F;
        overlay.faces[static_cast<std::size_t>(i)].corners_uv[corner] =
            to_point(corners[corner]);
      }
      if (!valid)
      {
        continue;
      }
      overlay.faces[static_cast<std::size_t>(i)].center_uv = to_point(center_uv);
      overlay.faces[static_cast<std::size_t>(i)].valid = true;
    }

    cv::Point2d center_sum(0.0, 0.0);
    int valid_face_count = 0;
    for (int i = 0; i < overlay.face_count; ++i)
    {
      const auto& face = overlay.faces[static_cast<std::size_t>(i)];
      if (!face.valid)
      {
        continue;
      }
      center_sum.x += face.center_uv.x;
      center_sum.y += face.center_uv.y;
      ++valid_face_count;
    }
    if (valid_face_count > 0)
    {
      overlay.center_uv =
          cv::Point(static_cast<int>(std::lround(
                        center_sum.x / static_cast<double>(valid_face_count))),
                    static_cast<int>(std::lround(
                        center_sum.y / static_cast<double>(valid_face_count))));
      overlay.center_valid = true;
      track_overlays.push_back(overlay);
    }
  }

  preview_.Submit(
      bgr_image,
      [detector_armors, target_msg, track_overlays](cv::Mat& canvas)
      {
        for (const auto& armor : detector_armors)
        {
          const cv::Scalar color =
              armor.pnp_valid ? cv::Scalar(80, 220, 255)
                              : cv::Scalar(120, 120, 120);
          for (std::size_t i = 0; i < armor.points.size(); ++i)
          {
            cv::line(canvas, armor.points[i],
                     armor.points[(i + 1U) % armor.points.size()], color, 2,
                     cv::LINE_AA);
          }
          cv::circle(canvas, armor.center, 4, color, -1, cv::LINE_AA);
        }

        for (const auto& track : track_overlays)
        {
          const cv::Scalar face_color =
              track.selected ? cv::Scalar(255, 160, 40)
                             : cv::Scalar(210, 210, 120);
          const cv::Scalar body_color =
              track.selected ? cv::Scalar(40, 255, 40)
                             : cv::Scalar(80, 220, 255);
          const int line_thickness = track.selected ? 2 : 1;

          std::array<cv::Point, 4> armor_center_uv{};
          std::array<bool, 4> armor_center_valid{};
          for (int i = 0; i < track.face_count; ++i)
          {
            armor_center_valid[static_cast<std::size_t>(i)] =
                track.faces[static_cast<std::size_t>(i)].valid;
            armor_center_uv[static_cast<std::size_t>(i)] =
                track.faces[static_cast<std::size_t>(i)].center_uv;
          }

          for (int i = 0; i < track.face_count; ++i)
          {
            const auto& face = track.faces[static_cast<std::size_t>(i)];
            if (!face.valid)
            {
              continue;
            }
            std::array<cv::Point, 4> corners_uv{};
            for (int corner_index = 0; corner_index < 4; ++corner_index)
            {
              corners_uv[static_cast<std::size_t>(corner_index)] =
                  face.corners_uv[static_cast<std::size_t>(corner_index)];
            }
            for (int corner_index = 0; corner_index < 4; ++corner_index)
            {
              cv::line(canvas, corners_uv[static_cast<std::size_t>(corner_index)],
                       corners_uv[static_cast<std::size_t>((corner_index + 1) % 4)],
                       face_color, line_thickness, cv::LINE_AA);
            }
          }

          if (track.face_count > 1)
          {
            for (int i = 0; i < track.face_count; ++i)
            {
              const int next = (i + 1) % track.face_count;
              if (armor_center_valid[static_cast<std::size_t>(i)] &&
                  armor_center_valid[static_cast<std::size_t>(next)])
              {
                cv::line(canvas,
                         armor_center_uv[static_cast<std::size_t>(i)],
                         armor_center_uv[static_cast<std::size_t>(next)],
                         body_color, line_thickness, cv::LINE_AA);
              }
            }
          }

          for (int i = 0; i < track.face_count; ++i)
          {
            if (!armor_center_valid[static_cast<std::size_t>(i)])
            {
              continue;
            }
            cv::circle(canvas, armor_center_uv[static_cast<std::size_t>(i)], 4,
                       face_color, -1, cv::LINE_AA);
            cv::putText(canvas, "E" + std::to_string(i),
                        armor_center_uv[static_cast<std::size_t>(i)] +
                            cv::Point(6, 14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, face_color, 1,
                        cv::LINE_AA);
          }

          cv::Point center_uv;
          bool center_projected = track.center_valid;
          center_uv = track.center_uv;
          const auto in_frame = [&canvas](const cv::Point& point)
          {
            return point.x >= 0 && point.x < canvas.cols && point.y >= 0 &&
                   point.y < canvas.rows;
          };
          if (!center_projected || !in_frame(center_uv))
          {
            cv::Point2d average(0.0, 0.0);
            int count = 0;
            for (int i = 0; i < track.face_count; ++i)
            {
              if (!armor_center_valid[static_cast<std::size_t>(i)] ||
                  !in_frame(armor_center_uv[static_cast<std::size_t>(i)]))
              {
                continue;
              }
              average.x += armor_center_uv[static_cast<std::size_t>(i)].x;
              average.y += armor_center_uv[static_cast<std::size_t>(i)].y;
              ++count;
            }
            if (count > 0)
            {
              center_uv = cv::Point(static_cast<int>(std::lround(
                                       average.x / static_cast<double>(count))),
                                    static_cast<int>(std::lround(
                                       average.y / static_cast<double>(count))));
              center_projected = true;
            }
          }
          if (center_projected && in_frame(center_uv))
          {
            cv::circle(canvas, center_uv, track.selected ? 6 : 5, body_color, -1,
                       cv::LINE_AA);
            cv::drawMarker(canvas, center_uv, body_color, cv::MARKER_CROSS,
                           track.selected ? 20 : 16, line_thickness,
                           cv::LINE_AA);
            char label[64];
            std::snprintf(label, sizeof(label), "%s%d %.2f",
                          track.selected ? "*" : "", track.tag_id, track.score);
            int baseline = 0;
            const cv::Size label_size = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, 0.52, 1, &baseline);
            cv::Point label_pos = center_uv + cv::Point(8, -8);
            label_pos.x =
                std::clamp(label_pos.x, 2, canvas.cols - label_size.width - 2);
            label_pos.y = std::clamp(label_pos.y, label_size.height + 2,
                                     canvas.rows - baseline - 2);
            cv::putText(canvas, label, label_pos, cv::FONT_HERSHEY_SIMPLEX,
                        0.52, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(canvas, label, label_pos, cv::FONT_HERSHEY_SIMPLEX,
                        0.52, body_color, 1, cv::LINE_AA);
          }
        }

        const auto id_index = static_cast<std::size_t>(target_msg.id);
        const std::string id_name =
            id_index < ARMOR_NUMBER_NAMES.size()
                ? std::string(ARMOR_NUMBER_NAMES[id_index])
                : std::string("invalid");
        const std::string header =
            std::string("tracker ") +
            (target_msg.tracking ? "TRACK" : "NO_TARGET") + " id=" + id_name +
            " face=" + std::to_string(target_msg.tracked_face_index) +
            " det=" + std::to_string(detector_armors.size()) +
            " tracks=" + std::to_string(track_overlays.size());
        cv::putText(canvas, header, cv::Point(12, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(40, 240, 40),
                    2, cv::LINE_AA);

      });
}
