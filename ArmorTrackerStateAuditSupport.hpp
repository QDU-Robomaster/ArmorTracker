#pragma once

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::WriteStateAuditRow(
    uint64_t image_timestamp_us, const Eigen::VectorXd& ekf_prediction,
    const armor_tracker::FaceSelectionResult* selection, bool matched)
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
  const auto write_candidate =
      [this](const armor_tracker::FaceMatchCandidate& candidate, bool valid)
  {
    if (!valid)
    {
      state_audit_.file << -1 << '\t' << -1 << '\t' << 0 << '\t' << 0.0 << '\t'
                        << 0.0 << '\t' << 0.0;
      return;
    }
    state_audit_.file << LocalFaceToCanonicalFace(candidate.face_index) << '\t'
                      << candidate.image_track_id << '\t'
                      << (candidate.confirmed_image_track ? 1 : 0) << '\t'
                      << candidate.score << '\t' << candidate.position_diff
                      << '\t' << candidate.yaw_diff;
  };
  const armor_tracker::FaceMatchCandidate* selected_candidate =
      (selection != nullptr && selection->has_selected_candidate)
          ? &selection->selected_candidate
          : nullptr;

  state_audit_.file << image_timestamp_us << '\t' << (matched ? 1 : 0) << '\t'
                    << rt_.tracked_face_index << '\t'
                    << static_cast<int>(rt_.tracked_id) << '\t'
                    << (selection != nullptr
                            ? static_cast<int>(selection->accepted_mode)
                            : static_cast<int>(
                                  armor_tracker::FaceSelectionAcceptedMode::NONE))
                    << '\t' << (selected_candidate != nullptr ? 1 : 0) << '\t';
  if (selected_candidate != nullptr)
  {
    state_audit_.file << LocalFaceToCanonicalFace(selected_candidate->face_index)
                      << '\t'
                      << selected_candidate->image_track_id << '\t'
                      << (selected_candidate->confirmed_image_track ? 1 : 0) << '\t'
                      << selected_candidate->score << '\t'
                      << selected_candidate->position_diff << '\t'
                      << selected_candidate->yaw_diff << '\t'
                      << selected_candidate->measured_yaw << '\t';
  }
  else
  {
    state_audit_.file << -1 << '\t' << -1 << '\t' << 0 << '\t' << 0.0 << '\t'
                      << 0.0 << '\t' << 0.0 << '\t' << 0.0 << '\t';
  }

  if (selection != nullptr)
  {
    write_candidate(selection->best_same_face_candidate,
                    selection->best_same_face_candidate.face_index >= 0);
    state_audit_.file << '\t';
    write_candidate(selection->best_switch_candidate,
                    selection->best_switch_candidate.face_index >= 0);
    state_audit_.file << '\t'
                      << (selection->matched_same_face ? 1 : 0) << '\t'
                      << (selection->matched_switch_face ? 1 : 0) << '\t'
                      << (selection->allow_face_switch ? 1 : 0) << '\t'
                      << (selection->switch_blocked_by_timeout ? 1 : 0) << '\t'
                      << (selection->switch_blocked_by_id_mismatch ? 1 : 0)
                      << '\t';
  }
  else
  {
    write_candidate(armor_tracker::FaceMatchCandidate{}, false);
    state_audit_.file << '\t';
    write_candidate(armor_tracker::FaceMatchCandidate{}, false);
    state_audit_.file << "\t0\t0\t0\t0\t0\t";
  }

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
