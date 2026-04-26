#pragma once

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::UpdateSingleArmorMode(
    const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us)
{
  std::size_t selected_index = 0;
  int detection_track_id = -1;
  bool confirmed_track = false;
  float selected_center_diff = 0.0f;
  float selected_area_log = 0.0f;
  float selected_score = 0.0f;
  auto selected_armor = SelectSingleArmorObservation(
      armors_msg, image_timestamp_us, selected_index, detection_track_id,
      confirmed_track, selected_center_diff, selected_area_log, selected_score);

  const bool matched = selected_armor.has_value();
  if (matched)
  {
    const auto& armor = *selected_armor;
    rt_.tracked_armor = armor;
    rt_.tracked_id = armor.number;
    rt_.tracked_armors_num = static_cast<ArmorsNum>(1);
    rt_.tracked_face_index = 0;
    rt_.tracked_face_track_id_valid = confirmed_track && detection_track_id >= 0;
    rt_.tracked_face_track_id =
        detection_track_id >= 0 ? static_cast<uint16_t>(detection_track_id) : 0;
    rt_.face_track_id_valid.fill(false);
    rt_.face_track_id.fill(0);
    if (rt_.tracked_face_track_id_valid)
    {
      rt_.face_track_id_valid[0] = true;
      rt_.face_track_id[0] = rt_.tracked_face_track_id;
    }

    const double raw_measured_yaw =
        armor_tracker::OrientationToYawNear(armor.pose.rotation, rt_.last_yaw);
    double measured_yaw = raw_measured_yaw;
    if (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST)
    {
      const double yaw_gate = 0.55;
      const std::array<double, 3> yaw_candidates = {
          raw_measured_yaw,
          armor_tracker::UnwrapYawNear(raw_measured_yaw + M_PI, rt_.last_yaw),
          armor_tracker::UnwrapYawNear(raw_measured_yaw - M_PI, rt_.last_yaw)};
      double best_candidate = yaw_candidates[0];
      double best_delta =
          armor_tracker::AngularDiffAbs(best_candidate, rt_.last_yaw);
      for (double candidate : yaw_candidates)
      {
        const double candidate_delta =
            armor_tracker::AngularDiffAbs(candidate, rt_.last_yaw);
        if (candidate_delta < best_delta)
        {
          best_delta = candidate_delta;
          best_candidate = candidate;
        }
      }
      measured_yaw = best_candidate;
      const double yaw_delta = best_delta;
      if (yaw_delta > yaw_gate)
      {
        XR_LOG_DEBUG(
            "SingleArmor yaw hold: idx=%zu track=%d yaw_prev=%.3f yaw_raw=%.3f yaw_meas=%.3f yaw_delta=%.3f gate=%.3f",
            selected_index, detection_track_id, rt_.last_yaw, raw_measured_yaw,
            measured_yaw,
            yaw_delta, yaw_gate);
        measured_yaw = rt_.last_yaw;
      }
    }

    XR_LOG_DEBUG(
        "SingleArmor match: idx=%zu track=%d confirmed=%d num=%d type=%d center=(%.1f,%.1f) pos=(%.3f,%.3f,%.3f) score=%.3f center_diff=%.1f area_log=%.3f yaw_prev=%.3f yaw_raw=%.3f yaw_meas=%.3f",
        selected_index, detection_track_id, confirmed_track ? 1 : 0,
        static_cast<int>(armor.number), static_cast<int>(armor.type),
        static_cast<double>(armor.center.x),
        static_cast<double>(armor.center.y),
        armor.pose.translation.x(), armor.pose.translation.y(),
        armor.pose.translation.z(), static_cast<double>(selected_score),
        static_cast<double>(selected_center_diff), static_cast<double>(selected_area_log),
        rt_.last_yaw, raw_measured_yaw, measured_yaw);
    rt_.last_yaw = measured_yaw;

    ekf_.state(ExtendedKalmanFilter::X_CENTER) = armor.pose.translation.x();
    ekf_.state(ExtendedKalmanFilter::Y_CENTER) = armor.pose.translation.y();
    ekf_.state(ExtendedKalmanFilter::Z_ARMOR) = armor.pose.translation.z();
    ekf_.state(ExtendedKalmanFilter::YAW) = measured_yaw;
    ekf_.state(ExtendedKalmanFilter::ROBOT_R) = 0.0;
    ekf_.state(ExtendedKalmanFilter::DELTA_R) = 0.0;
    ekf_.state(ExtendedKalmanFilter::DELTA_Z) = 0.0;
    ekf_.measurement =
        Eigen::Vector4d(armor.pose.translation.x(), armor.pose.translation.y(),
                        armor.pose.translation.z(), measured_yaw);
    ekf_.ekf.SetState(ekf_.state);

    rt_.info_position_diff = 0.0;
    rt_.info_yaw_diff = 0.0;
    FillSingleArmorDebug(selected_index, detection_track_id, confirmed_track,
                         selected_score, selected_center_diff, selected_area_log);
  }
  else
  {
    XR_LOG_DEBUG("SingleArmor miss: state=%d tracked_track_valid=%d tracked_track=%u",
                 static_cast<int>(rt_.state), rt_.tracked_face_track_id_valid ? 1 : 0,
                 static_cast<unsigned>(rt_.tracked_face_track_id));
    CandidateDebugMsg debug{};
    debug.tracked_armors_num = 1;
    debug.matched = 0;
    debug.tracked_face_track_id_valid =
        rt_.tracked_face_track_id_valid ? 1 : 0;
    debug.tracked_face_track_id =
        rt_.tracked_face_track_id_valid ? static_cast<int16_t>(rt_.tracked_face_track_id)
                                        : static_cast<int16_t>(-1);
    candidate_debug_msg_ = debug;
    rt_.info_position_diff = DBL_MAX;
    rt_.info_yaw_diff = DBL_MAX;
  }

  rt_.measurement_valid_current_frame = matched;
  AdvanceTrackerState(matched);
  WriteStateAuditRow(image_timestamp_us, ekf_.state, nullptr, matched);
}
