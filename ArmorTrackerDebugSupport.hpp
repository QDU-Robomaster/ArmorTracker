#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::RenderPreviewFrame(ArmorTracker<CameraInfoV>* self,
                                                   cv::Mat frame)
{
  if (frame.empty())
  {
    return;
  }

  EkfPointsMsg& ekf = self->ekf_msg_;
  const CameraInfo& cam = ArmorTracker<CameraInfoV>::kCameraInfo;
  const bool has_distortion =
      (cam.distortion_model == CameraTypes::DistortionModel::PLUMB_BOB);

  const auto& k_arr = cam.camera_matrix;
  cv::Mat k = (cv::Mat_<double>(3, 3) << k_arr[0], k_arr[1], k_arr[2], k_arr[3],
               k_arr[4], k_arr[5], k_arr[6], k_arr[7], k_arr[8]);

  cv::Mat d;
  if (has_distortion)
  {
    std::vector<double> dvec = {cam.distortion_coefficients[0],
                                cam.distortion_coefficients[1],
                                cam.distortion_coefficients[2],
                                cam.distortion_coefficients[3],
                                cam.distortion_coefficients[4]};
    d = cv::Mat(dvec).clone().reshape(1, 1);
  }

  const double sx = static_cast<double>(frame.cols) / static_cast<double>(cam.width);
  const double sy = static_cast<double>(frame.rows) / static_cast<double>(cam.height);
  cv::Mat k_scaled = k.clone();
  k_scaled.at<double>(0, 0) *= sx;
  k_scaled.at<double>(1, 1) *= sy;
  k_scaled.at<double>(0, 2) *= sx;
  k_scaled.at<double>(1, 2) *= sy;

  auto project = [&](const Eigen::Vector3d& pc, cv::Point2d& uv) -> bool
  {
    if (!(pc.z() > 1e-6) || !std::isfinite(pc.x()) || !std::isfinite(pc.y()) ||
        !std::isfinite(pc.z()))
    {
      return false;
    }

    std::vector<cv::Point3d> obj{cv::Point3d(pc.x(), pc.y(), pc.z())};
    static cv::Mat rvec = cv::Mat::zeros(1, 3, CV_64F);
    static cv::Mat tvec = cv::Mat::zeros(1, 3, CV_64F);
    std::vector<cv::Point2d> imgpts;
    cv::projectPoints(obj, rvec, tvec, k_scaled, d, imgpts);
    uv = imgpts[0];
    return (0 <= uv.x && uv.x < frame.cols && 0 <= uv.y && uv.y < frame.rows);
  };

  if (ekf.valid[0])
  {
    cv::Point2d uv;
    Eigen::Vector3d pc(ekf.center_cam.x(), ekf.center_cam.y(), ekf.center_cam.z());
    if (project(pc, uv))
    {
      cv::circle(frame, uv, 5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
      cv::putText(frame, "C", uv + cv::Point2d(6, -6), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
  }

  for (int i = 0; i < std::min<int>(ekf.count, 4); ++i)
  {
    if (!ekf.valid[i + 1])
    {
      continue;
    }
    cv::Point2d uv;
    Eigen::Vector3d pc(ekf.armors_cam[i].x(), ekf.armors_cam[i].y(),
                       ekf.armors_cam[i].z());
    if (project(pc, uv))
    {
      cv::circle(frame, uv, 4, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
      char buf[16];
      (void)std::snprintf(buf, sizeof(buf), "A%d", i);
      cv::putText(frame, buf, uv + cv::Point2d(6, -6), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
  }

  for (int i = 0; i < std::min<int>(ekf.count, 4); ++i)
  {
    if (!ekf.valid[0] || !ekf.valid[i + 1])
    {
      continue;
    }
    cv::Point2d uc, ua;
    Eigen::Vector3d pc_c(ekf.center_cam.x(), ekf.center_cam.y(), ekf.center_cam.z());
    Eigen::Vector3d pc_a(ekf.armors_cam[i].x(), ekf.armors_cam[i].y(),
                         ekf.armors_cam[i].z());
    if (project(pc_c, uc) && project(pc_a, ua))
    {
      cv::line(frame, uc, ua, cv::Scalar(80, 180, 255), 1, cv::LINE_AA);
    }
  }

  cv::imshow("ekf_overlay", frame);
  cv::waitKey(1);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::PreviewImageThreadFun(ArmorTracker<CameraInfoV>* self)
{
  if (!ArmorTrackerPreviewUiAvailable())
  {
    XR_LOG_WARN("ArmorTracker preview disabled because no display backend is available");
    return;
  }

  XR_LOG_PASS("ArmorTracker preview uses sync frame topic");

  while (true)
  {
    typename FrameSync::Subscriber subscriber(self->sync_);
    if (!subscriber.Valid())
    {
      LibXR::Thread::Sleep(200);
      continue;
    }

    SyncedFrame synced_frame;
    while (true)
    {
      const auto wait_ans =
          subscriber.Wait(synced_frame, kArmorTrackerSyncFrameWaitTimeoutMs);
      if (wait_ans == LibXR::ErrorCode::TIMEOUT)
      {
        continue;
      }
      if (wait_ans != LibXR::ErrorCode::OK)
      {
        break;
      }

      const auto* image_frame = synced_frame.GetImageFrame();
      if (image_frame != nullptr)
      {
        const int cv_type = ArmorTrackerCvTypeFromEncoding(kCameraInfo.encoding);
        if (cv_type >= 0)
        {
          cv::Mat input(static_cast<int>(kCameraInfo.height),
                        static_cast<int>(kCameraInfo.width), cv_type,
                        const_cast<uint8_t*>(image_frame->data.data()),
                        static_cast<size_t>(kCameraInfo.step));
          cv::Mat frame = ArmorTrackerConvertToBgrWithEncoding(input, kCameraInfo.encoding);
          if (!frame.empty())
          {
            RenderPreviewFrame(self, frame);
          }
        }
      }
    }
  }
}
#endif

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::FillCandidateDebugFromSelection(
    const armor_tracker::FaceSelectionResult& selection,
    CandidateDebugMsg& candidate_debug)
{
  candidate_debug.count = selection.debug.count;
  candidate_debug.selected_index = selection.debug.selected_index;
  candidate_debug.detection_count = selection.debug.detection_count;
  candidate_debug.preferred_adjacent_face = selection.debug.preferred_adjacent_face;
  candidate_debug.has_same_number_candidate =
      selection.debug.has_same_number_candidate;
  candidate_debug.relaxed_same_face_distance =
      selection.debug.relaxed_same_face_distance;
  candidate_debug.relaxed_face_switch_distance =
      selection.debug.relaxed_face_switch_distance;
  candidate_debug.relaxed_face_switch_yaw_diff =
      selection.debug.relaxed_face_switch_yaw_diff;
  candidate_debug.best_same_face_score = selection.debug.best_same_face_score;
  candidate_debug.best_switch_face_score = selection.debug.best_switch_face_score;
  candidate_debug.same_face_matched = selection.debug.same_face_matched;
  candidate_debug.switch_face_matched = selection.debug.switch_face_matched;
  candidate_debug.switch_blocked_by_timeout =
      selection.debug.switch_blocked_by_timeout;
  candidate_debug.switch_allowed = selection.debug.switch_allowed;
  candidate_debug.detection_track_ids = selection.debug.detection_track_ids;
  candidate_debug.detection_track_confirmed =
      selection.debug.detection_track_confirmed;

  for (std::size_t item_index = 0; item_index < selection.debug.count; ++item_index)
  {
    const auto& src = selection.debug.items[item_index];
    auto& dst = candidate_debug.items[item_index];
    dst.armor_index = src.armor_index;
    dst.face_index = src.face_index;
    dst.same_number = src.same_number;
    dst.image_track_id = src.image_track_id;
    dst.image_track_confirmed = src.image_track_confirmed;
    dst.same_persistent_track = src.same_persistent_track;
    dst.number = src.number;
    dst.type = src.type;
    dst.score = src.score;
    dst.position_diff = src.position_diff;
    dst.yaw_diff = src.yaw_diff;
    dst.view_bonus = src.view_bonus;
    dst.area_score = src.area_score;
    dst.frontality = src.frontality;
    dst.center_x = src.center_x;
    dst.center_y = src.center_y;
    dst.predicted_yaw = src.predicted_yaw;
    dst.measured_yaw = src.measured_yaw;
  }
  candidate_debug.face_switch_cooldown_remaining =
      static_cast<float>(rt_.face_switch_cooldown_remaining);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::FillCandidateDebugPolicy(
    CandidateDebugMsg& candidate_debug, const Eigen::VectorXd& ekf_prediction,
    const armor_tracker::FaceSelectionPolicy& face_policy) const
{
  candidate_debug.face_switch_enabled = face_policy.face_switch_enabled ? 1 : 0;
  candidate_debug.relaxed_face_switch_enabled =
      face_policy.relaxed_face_switch_enabled ? 1 : 0;
  candidate_debug.odd_face_switch_enabled =
      face_policy.odd_face_switch_enabled ? 1 : 0;
  candidate_debug.view_priority_enabled =
      face_policy.view_priority_enabled ? 1 : 0;
  candidate_debug.directional_face_switch_enabled =
      face_policy.directional_face_switch_enabled ? 1 : 0;
  candidate_debug.tracked_face_track_id_valid =
      rt_.tracked_face_track_id_valid ? 1 : 0;
  candidate_debug.tracked_face_track_id =
      rt_.tracked_face_track_id_valid
          ? static_cast<int16_t>(rt_.tracked_face_track_id)
          : static_cast<int16_t>(-1);
  candidate_debug.tracked_armors_num = static_cast<uint8_t>(rt_.tracked_armors_num);
  candidate_debug.predicted_vyaw =
      static_cast<float>(ekf_prediction(ExtendedKalmanFilter::V_YAW));
  candidate_debug.max_match_distance =
      static_cast<float>(cfg_.match.max_match_distance);
  candidate_debug.max_match_yaw_diff =
      static_cast<float>(cfg_.match.max_match_yaw_diff);
  candidate_debug.face_switch_score_deadzone =
      static_cast<float>(face_policy.face_switch_score_deadzone);
  candidate_debug.face_switch_position_deadzone =
      static_cast<float>(face_policy.face_switch_position_deadzone);
  candidate_debug.face_switch_yaw_deadzone =
      static_cast<float>(face_policy.face_switch_yaw_deadzone);
  candidate_debug.face_switch_timeout_sec =
      static_cast<float>(face_policy.face_switch_timeout_sec);
  candidate_debug.face_switch_cooldown_remaining =
      static_cast<float>(rt_.face_switch_cooldown_remaining);
}
