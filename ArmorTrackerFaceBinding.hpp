#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "ArmorTrackerFaceSelector.hpp"

namespace armor_tracker
{
// FaceBinding 层只负责“当前正面”和图像短时 track id 的绑定关系。
// 它不接触 EKF、选面评分，也不关心 topic / preview。
struct FaceBindingRuntime
{
  int tracked_armors_num = 4;
  bool tracked_face_track_id_valid = false;
  uint16_t tracked_face_track_id = 0;
  std::array<bool, 4> face_track_id_valid{};
  std::array<uint16_t, 4> face_track_id{};
};

inline void ApplySelectedFaceBinding(FaceBindingRuntime& runtime,
                                     const FaceMatchCandidate& selected_candidate,
                                     bool did_face_switch)
{
  if (did_face_switch)
  {
    const int face_count =
        std::max(1, std::min(4, runtime.tracked_armors_num));
    std::array<bool, 4> rotated_valid{};
    std::array<uint16_t, 4> rotated_ids{};
    for (int face_slot = 0; face_slot < face_count; ++face_slot)
    {
      const int old_slot =
          (face_slot + selected_candidate.face_index) % face_count;
      rotated_valid[face_slot] = runtime.face_track_id_valid[old_slot];
      rotated_ids[face_slot] = runtime.face_track_id[old_slot];
    }
    runtime.face_track_id_valid = rotated_valid;
    runtime.face_track_id = rotated_ids;
  }

  if (runtime.face_track_id_valid[0])
  {
    runtime.tracked_face_track_id_valid = true;
    runtime.tracked_face_track_id = runtime.face_track_id[0];
  }
  else
  {
    runtime.tracked_face_track_id_valid = false;
  }

  if (selected_candidate.image_track_id >= 0 &&
      selected_candidate.confirmed_image_track)
  {
    runtime.tracked_face_track_id_valid = true;
    runtime.tracked_face_track_id =
        static_cast<uint16_t>(selected_candidate.image_track_id);
    runtime.face_track_id_valid[0] = true;
    runtime.face_track_id[0] = runtime.tracked_face_track_id;
  }
}
}  // namespace armor_tracker
