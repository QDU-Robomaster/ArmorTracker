#pragma once

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::WriteStateAuditRow(
    uint64_t image_timestamp_us, const Eigen::VectorXd& ekf_prediction,
    const CandidateDebugMsg& candidate_debug, bool matched)
{
  if (state_audit_.path.empty())
  {
    return;
  }

  if (!state_audit_.file.is_open())
  {
    state_audit_.file.open(state_audit_.path, std::ios::out | std::ios::trunc);
    if (!state_audit_.file)
    {
      if (!state_audit_.open_failed)
      {
        XR_LOG_ERROR("ArmorTracker failed to open state audit: %s",
                     state_audit_.path.c_str());
        state_audit_.open_failed = true;
      }
      return;
    }

    state_audit_.file
        << "image_timestamp_us\tmatched\ttracked_face_index\ttracked_id\t"
        << "accepted_mode\thas_selected_candidate\tselected_face_index\t"
        << "selected_track_id\tselected_confirmed\tselected_score\t"
        << "selected_position_diff\tselected_yaw_diff\tselected_measured_yaw\t"
        << "best_same_face_index\tbest_same_track_id\tbest_same_confirmed\t"
        << "best_same_score\tbest_same_position_diff\tbest_same_yaw_diff\t"
        << "best_switch_face_index\tbest_switch_track_id\tbest_switch_confirmed\t"
        << "best_switch_score\tbest_switch_position_diff\tbest_switch_yaw_diff\t"
        << "same_face_matched\tswitch_face_matched\tswitch_allowed\t"
        << "switch_blocked_by_timeout\tswitch_blocked_by_id_mismatch\t"
        << "state\tlost_count\trecovery_count\t"
        << "pred_x\tpred_y\tpred_z\tpred_yaw\tpred_r1\tpred_r2\tpred_dz\t"
        << "pred_a0_x\tpred_a0_y\tpred_a0_z\tpred_a1_x\tpred_a1_y\tpred_a1_z\t"
        << "pred_a2_x\tpred_a2_y\tpred_a2_z\tpred_a3_x\tpred_a3_y\tpred_a3_z\t"
        << "post_x\tpost_y\tpost_z\tpost_yaw\tpost_r1\tpost_r2\tpost_dz\n";
  }

  const auto calc_radius_2 = [this](const Eigen::VectorXd& state)
  {
    const bool four_armors = rt_.tracked_armors_num == ArmorsNum::NORMAL_4;
    const double radius_1 = state(ExtendedKalmanFilter::ROBOT_R);
    return (!four_armors || SymmetricGeometryEnabled())
               ? radius_1
               : (radius_1 + state(ExtendedKalmanFilter::DELTA_R));
  };
  const auto calc_dz = [this](const Eigen::VectorXd& state)
  {
    return (rt_.tracked_armors_num == ArmorsNum::NORMAL_4 &&
            !SymmetricGeometryEnabled())
               ? state(ExtendedKalmanFilter::DELTA_Z)
               : 0.0;
  };
  const auto write_pred_armors = [this, &ekf_prediction]()
  {
    for (int face_index = 0; face_index < 4; ++face_index)
    {
      const Eigen::Vector3d p = GetArmorPositionFromState(ekf_prediction, face_index);
      state_audit_.file << p.x() << '\t' << p.y() << '\t' << p.z() << '\t';
    }
  };
  const auto* selected_item =
      (matched && candidate_debug.matched != 0 &&
       candidate_debug.selected_index < candidate_debug.count &&
       candidate_debug.selected_index < CandidateDebugMsg::kMaxItems)
          ? &candidate_debug.items[candidate_debug.selected_index]
          : nullptr;
  const auto write_item =
      [this](const CandidateDebugItem* item)
  {
    (void)this;
    if (item == nullptr)
    {
      state_audit_.file << -1 << '\t' << -1 << '\t' << 0 << '\t' << 0.0 << '\t'
                        << 0.0 << '\t' << 0.0;
      return;
    }
    state_audit_.file << static_cast<int>(item->face_index) << '\t'
                      << item->image_track_id << '\t'
                      << static_cast<int>(item->image_track_confirmed) << '\t'
                      << item->score << '\t' << item->position_diff << '\t'
                      << item->yaw_diff;
  };

  state_audit_.file << image_timestamp_us << '\t' << (matched ? 1 : 0) << '\t'
                    << rt_.tracked_face_index << '\t'
                    << static_cast<int>(rt_.tracked_id) << '\t'
                    << static_cast<int>(candidate_debug.accepted_mode) << '\t'
                    << (selected_item != nullptr ? 1 : 0) << '\t';
  if (selected_item != nullptr)
  {
    state_audit_.file << static_cast<int>(selected_item->face_index) << '\t'
                      << selected_item->image_track_id << '\t'
                      << static_cast<int>(selected_item->image_track_confirmed)
                      << '\t' << selected_item->score << '\t'
                      << selected_item->position_diff << '\t'
                      << selected_item->yaw_diff << '\t'
                      << selected_item->measured_yaw << '\t';
  }
  else
  {
    state_audit_.file << -1 << '\t' << -1 << '\t' << 0 << '\t' << 0.0 << '\t'
                      << 0.0 << '\t' << 0.0 << '\t' << 0.0 << '\t';
  }

  write_item(candidate_debug.same_face_matched != 0 ? selected_item : nullptr);
  state_audit_.file << '\t';
  write_item(candidate_debug.switch_face_matched != 0 ? selected_item : nullptr);
  state_audit_.file << '\t' << static_cast<int>(candidate_debug.same_face_matched)
                    << '\t' << static_cast<int>(candidate_debug.switch_face_matched)
                    << '\t' << static_cast<int>(candidate_debug.switch_allowed)
                    << '\t'
                    << static_cast<int>(candidate_debug.switch_blocked_by_timeout)
                    << "\t0\t";

  state_audit_.file << static_cast<int>(rt_.state) << '\t' << rt_.lost_count
                    << '\t' << static_cast<int>(rt_.recovery_count) << '\t';

  state_audit_.file << ekf_prediction(ExtendedKalmanFilter::X_CENTER) << '\t'
                    << ekf_prediction(ExtendedKalmanFilter::Y_CENTER) << '\t'
                    << ekf_prediction(ExtendedKalmanFilter::Z_ARMOR) << '\t'
                    << ekf_prediction(ExtendedKalmanFilter::YAW) << '\t'
                    << ekf_prediction(ExtendedKalmanFilter::ROBOT_R) << '\t'
                    << calc_radius_2(ekf_prediction) << '\t'
                    << calc_dz(ekf_prediction) << '\t';
  write_pred_armors();
  state_audit_.file
                    << ekf_.state(ExtendedKalmanFilter::X_CENTER) << '\t'
                    << ekf_.state(ExtendedKalmanFilter::Y_CENTER) << '\t'
                    << ekf_.state(ExtendedKalmanFilter::Z_ARMOR) << '\t'
                    << ekf_.state(ExtendedKalmanFilter::YAW) << '\t'
                    << ekf_.state(ExtendedKalmanFilter::ROBOT_R) << '\t'
                    << calc_radius_2(ekf_.state) << '\t'
                    << calc_dz(ekf_.state) << '\n';
  state_audit_.file.flush();
}
