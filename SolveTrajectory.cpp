#include "SolveTrajectory.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <utility>

#include "logger.hpp"

namespace
{
bool SolverUseTargetDeltaZ()
{
  const char* env = std::getenv("XR_SOLVER_USE_TARGET_DZ");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}
}  // namespace

SolveTrajectory::SolveTrajectory(const float& k_, const int& bias_time_,
                                 const float& s_bias_, const float& z_bias_,
                                 CalculateMode calculate_mode,
                                 const TrajectoryTable::TableConfig& table_config)
    : k_(k_),
      bias_time_(bias_time_),
      s_bias_(s_bias_),
      z_bias_(z_bias_),
      calculate_mode_(calculate_mode),
      table_(table_config)
{
  if (calculate_mode_ == CalculateMode::TABLE_LOOKUP)
  {
    table_.Init();
  }
}

void SolveTrajectory::Init(double velocity)
{
  const float old_v = current_v_;
  if (!std::isnan(velocity))
  {
    current_v_ = static_cast<float>(velocity);
  }
  else
  {
    current_v_ = 12.0f;  // 默认弹速
  }
  if (std::fabs(current_v_ - old_v) > 1e-3f)
  {
    XR_LOG_INFO("SolveTrajectory bullet speed updated: %.3f -> %.3f",
                static_cast<double>(old_v), static_cast<double>(current_v_));
  }
}

float SolveTrajectory::MonoDirectionalAirResistanceModel(float s, float v, float angle)
{
  // 飞行时间 t = (e^{k s} - 1) / (k v cos(angle))
  const double k = static_cast<double>(k_);
  const double distance = static_cast<double>(s);
  const double speed = static_cast<double>(v);
  const double cos_angle = std::cos(static_cast<double>(angle));
  const double distance_factor =
      std::abs(k) < 1e-12 ? distance : std::expm1(k * distance) / k;
  fly_time_ = static_cast<float>(distance_factor / (speed * cos_angle));

  if (std::isnan(fly_time_))
  {
    XR_LOG_ERROR("Fly time is nan!");
    fly_time_ = 0.0;
    return 0.0f;
  }

  if (fly_time_ < 0.0)
  {
    XR_LOG_ERROR("Fly time is negative!");
    fly_time_ = 0.0;
    return 0.0f;
  }

  // z = v sin(angle) t - 0.5 g t^2
  float z = static_cast<float>(v * std::sin(angle) * fly_time_ -
                               GRAVITY * fly_time_ * fly_time_ / 2.0);
  return z;
}

float SolveTrajectory::CompleteAirResistanceModel(float /*s*/, float /*v*/,
                                                  float /*angle*/)
{
  // TODO: Implement complete air resistance model
  return 0.0f;
}

float SolveTrajectory::PitchTrajectoryCompensation(float s, float z, float v)
{
  if (calculate_mode_ == CalculateMode::NORMAL || !table_.IsInit())
  {
    float z_temp = z;
    float angle_pitch = 0.0f;
    // 经验：20~27 次迭代通常足够收敛
    for (int i = 0; i < 22; ++i)
    {
      if (std::isnan(z_temp))
      {
        XR_LOG_ERROR("z_temp is nan!");
        return 0.0f;
      }

      angle_pitch = std::atan2(z_temp, s);

      const float Z_ACTUAL = MonoDirectionalAirResistanceModel(s, v, angle_pitch);
      const float DZ = 0.3f * (z - Z_ACTUAL);
      z_temp += DZ;

      if (std::fabs(DZ) < 1e-5f)
      {
        break;
      }
    }

    return angle_pitch;
  }
  else if (calculate_mode_ == CalculateMode::TABLE_LOOKUP && table_.IsInit())
  {
    auto res = table_.Check(s, z);
    fly_time_ = res.t;
    return static_cast<float>(res.pitch);
  }

  return 0.0f;
}

bool SolveTrajectory::ShouldFire(float tmp_yaw, float v_yaw, float timeDelay)
{
  // 线性预测：若预测 yaw 接近整圈（2π），则认为到达最佳击发窗口
  return std::fabs((tmp_yaw + v_yaw * timeDelay) - 2.0f * PI) < 0.001f;
}

void SolveTrajectory::CalculateArmorPosition(Target* msg, bool use_1,
                                             bool use_average_radius)
{
  tmp_yaws_.clear();
  min_yaw_in_cycle_ = std::numeric_limits<float>::max();
  max_yaw_in_cycle_ = std::numeric_limits<float>::lowest();

  for (int i = 0; i < msg->armors_num; ++i)
  {
    // 以目标 yaw 为基准，按数量等间隔分布
    const float TMP_YAW = tar_yaw_ + static_cast<float>(i) * 2.0f * PI /
                                         static_cast<float>(msg->armors_num);
    tmp_yaws_.push_back(TMP_YAW);
    min_yaw_in_cycle_ = std::min(min_yaw_in_cycle_, TMP_YAW);
    max_yaw_in_cycle_ = std::max(max_yaw_in_cycle_, TMP_YAW);

    // 半径选择
    float r = 0.0f;
    if (use_average_radius)
    {
      r = static_cast<float>((msg->radius_1 + msg->radius_2) / 2.0);
    }
    else
    {
      r = static_cast<float>(use_1 ? msg->radius_1 : msg->radius_2);
    }

    const bool use_delta_z =
        SolverUseTargetDeltaZ() && msg->armors_num == ARMOR_NUM_NORMAL &&
        (i == 1 || i == 3);

    // 世界坐标推算。默认保持旧行为；实验开关打开时按 SP 的 1/3 号面使用 dz。
    tar_position_[i].x = static_cast<float>(msg->position.x()) + r * std::cos(TMP_YAW);
    tar_position_[i].y = static_cast<float>(msg->position.y()) + r * std::sin(TMP_YAW);
    tar_position_[i].z = static_cast<float>(msg->position.z() +
                                            (use_delta_z ? msg->dz : 0.0));
    tar_position_[i].yaw = TMP_YAW;

    use_1 = !use_1;  // 交替使用 r1/r2
  }

  if (msg->measured_face_valid && msg->measured_face_index >= 0 &&
      msg->measured_face_index < msg->armors_num)
  {
    const int idx = msg->measured_face_index;
    tar_position_[idx].x = static_cast<float>(msg->measured_face_position.x());
    tar_position_[idx].y = static_cast<float>(msg->measured_face_position.y());
    tar_position_[idx].z = static_cast<float>(msg->measured_face_position.z());
    tar_position_[idx].yaw = static_cast<float>(msg->measured_face_yaw);
    tmp_yaws_[idx] = tar_position_[idx].yaw;
    min_yaw_in_cycle_ = std::min(min_yaw_in_cycle_, tar_position_[idx].yaw);
    max_yaw_in_cycle_ = std::max(max_yaw_in_cycle_, tar_position_[idx].yaw);
  }
}

std::pair<float, float> SolveTrajectory::CalculatePitchAndYaw(
    int idx, Target* msg, float timeDelay, float s_bias_, float z_bias_, float current_v_,
    bool use_target_center_for_yaw, float& aim_x, float& aim_y, float& aim_z)
{
  // 线性预测装甲板落点（x,y），z 保持
  aim_x = tar_position_[idx].x + static_cast<float>(msg->velocity.x()) * timeDelay;
  aim_y = tar_position_[idx].y + static_cast<float>(msg->velocity.y()) * timeDelay;
  aim_z = tar_position_[idx].z;

  // yaw 使用目标中心 or 预测装甲板点
  const float YAW_X =
      use_target_center_for_yaw ? static_cast<float>(msg->position.x()) : aim_x;
  const float YAW_Y =
      use_target_center_for_yaw ? static_cast<float>(msg->position.y()) : aim_y;

  // pitch 解算（水平距离需扣除 s_bias；目标高度加上 z_bias）
  const float S_HORIZ = std::sqrt(aim_x * aim_x + aim_y * aim_y) - s_bias_;
  const float Z_GOAL = aim_z + z_bias_;
  const float PITCH = PitchTrajectoryCompensation(S_HORIZ, Z_GOAL, current_v_);

  const float YAW = std::atan2(YAW_Y, YAW_X);

  return {PITCH, YAW};
}

int SolveTrajectory::SelectArmor(Target* msg, bool select_by_min_yaw)
{
  int selected_armor_idx = 0;

  if (select_by_min_yaw)
  {
    float min_yaw_diff = std::fabs(static_cast<float>(msg->yaw) - tar_position_[0].yaw);
    for (int i = 1; i < msg->armors_num; ++i)
    {
      const float D = std::fabs(static_cast<float>(msg->yaw) - tar_position_[i].yaw);
      if (D < min_yaw_diff)
      {
        min_yaw_diff = D;
        selected_armor_idx = i;
      }
    }
  }
  else
  {
    float min_distance = std::numeric_limits<float>::max();
    for (int i = 0; i < msg->armors_num; ++i)
    {
      const float DX = tar_position_[i].x;
      const float DY = tar_position_[i].y;
      const float DZ = tar_position_[i].z;
      const float DIST = std::sqrt(DX * DX + DY * DY + DZ * DZ);
      if (DIST < min_distance)
      {
        min_distance = DIST;
        selected_armor_idx = i;
      }
    }
  }

  return selected_armor_idx;
}

void SolveTrajectory::FireLogicIsTop(float& pitch, float& yaw, float& aim_x, float& aim_y,
                                     float& aim_z, Target* msg)
{
  tar_yaw_ = static_cast<float>(msg->yaw);

  // 通信延迟（ms->s） + 由上一次弹道模型计算得到的飞行时间
  const float TIME_DELAY =
      static_cast<float>(bias_time_) / 1000.0f + static_cast<float>(fly_time_);

  int idx = 0;
  bool is_fire = false;

  if (msg->armors_num == ARMOR_NUM_OUTPOST)
  {
    CalculateArmorPosition(msg, /*use_1=*/false, /*use_average_radius=*/true);

    for (size_t i = 0; i < tmp_yaws_.size(); ++i)
    {
      const float TY = tmp_yaws_[i];
      if (ShouldFire(TY, static_cast<float>(msg->v_yaw), TIME_DELAY))
      {
        is_fire = true;
        idx = static_cast<int>(i);
        if (fire_callback_)
        {
          fire_callback_(is_fire);
        }
        break;
      }
    }
  }
  else
  {
    CalculateArmorPosition(msg, /*use_1=*/false, /*use_average_radius=*/false);

    for (size_t i = 0; i < tmp_yaws_.size(); ++i)
    {
      const float TY = tmp_yaws_[i];
      if (ShouldFire(TY, static_cast<float>(msg->v_yaw), TIME_DELAY))
      {
        is_fire = true;
        idx = static_cast<int>(i);
        if (fire_callback_)
        {
          fire_callback_(is_fire);
        }
        break;
      }
    }
  }

  XR_LOG_DEBUG("FireLogicIsTop idx: %d", idx);

  const auto [p, y] =
      CalculatePitchAndYaw(idx, msg, TIME_DELAY, s_bias_, z_bias_, current_v_,
                           /*use_target_center_for_yaw=*/false, aim_x, aim_y, aim_z);
  pitch = p;
  yaw = y;

  XR_LOG_DEBUG("SolveTrajectory pitch: %f, yaw: %f", pitch, yaw);
}

void SolveTrajectory::FireLogicDefault(float& pitch, float& yaw, float& aim_x,
                                       float& aim_y, float& aim_z, Target* msg)
{
  // 时延
  const float TIME_DELAY =
      static_cast<float>(bias_time_) / 1000.0f + static_cast<float>(fly_time_);

  // 基于角速度对目标 yaw 作线性外推
  tar_yaw_ += static_cast<float>(msg->v_yaw) * TIME_DELAY;

  CalculateArmorPosition(msg, /*use_1=*/false, /*use_average_radius=*/false);
  const int IDX = SelectArmor(msg, /*select_by_min_yaw=*/false);

  const auto [p, y] =
      CalculatePitchAndYaw(IDX, msg, TIME_DELAY, s_bias_, z_bias_, current_v_,
                           /*use_target_center_for_yaw=*/false, aim_x, aim_y, aim_z);
  pitch = p;
  yaw = y;
}

int SolveTrajectory::SelectSpLikeAimArmor(Target* msg)
{
  const int armor_num = std::max(1, std::min(4, msg->armors_num));
  const double camera_facing_yaw =
      std::remainder(std::atan2(msg->position.y(), msg->position.x()) +
                         static_cast<double>(PI),
                     2.0 * static_cast<double>(PI));
  std::vector<double> delta_angle_list;
  delta_angle_list.reserve(static_cast<std::size_t>(armor_num));
  for (int i = 0; i < armor_num; ++i)
  {
    delta_angle_list.emplace_back(
        std::remainder(static_cast<double>(tmp_yaws_[i]) - camera_facing_yaw,
                       2.0 * static_cast<double>(PI)));
  }

  if (std::abs(msg->radius_1) <= 2.0 && msg->id != ArmorNumber::OUTPOST)
  {
    std::vector<int> candidate_indices;
    for (int i = 0; i < armor_num; ++i)
    {
      if (std::abs(delta_angle_list[i]) <= 60.0 / 57.3)
      {
        candidate_indices.push_back(i);
      }
    }
    if (candidate_indices.empty())
    {
      return -1;
    }
    if (candidate_indices.size() > 1)
    {
      const int id0 = candidate_indices[0];
      const int id1 = candidate_indices[1];
      if (lock_id_ != id0 && lock_id_ != id1)
      {
        lock_id_ = std::abs(delta_angle_list[id0]) <
                           std::abs(delta_angle_list[id1])
                       ? id0
                       : id1;
      }
      return lock_id_;
    }
    lock_id_ = -1;
    return candidate_indices[0];
  }

  const double coming_angle =
      msg->id == ArmorNumber::OUTPOST ? 70.0 / 57.3 : 55.0 / 57.3;
  const double leaving_angle =
      msg->id == ArmorNumber::OUTPOST ? 30.0 / 57.3 : 20.0 / 57.3;
  for (int i = 0; i < armor_num; ++i)
  {
    if (std::abs(delta_angle_list[i]) > coming_angle)
    {
      continue;
    }
    if (msg->v_yaw > 0.0 && delta_angle_list[i] < leaving_angle)
    {
      return i;
    }
    if (msg->v_yaw < 0.0 && delta_angle_list[i] > -leaving_angle)
    {
      return i;
    }
  }
  return -1;
}

void SolveTrajectory::AutoSolveTrajectory(float& pitch, float& yaw, float& aim_x,
                                          float& aim_y, float& aim_z, Target* msg)
{
  tar_yaw_ = static_cast<float>(msg->yaw);
  const float time_delay =
      static_cast<float>(bias_time_) / 1000.0f + static_cast<float>(fly_time_);
  tar_yaw_ += static_cast<float>(msg->v_yaw) * time_delay;
  CalculateArmorPosition(msg, /*use_1=*/false,
                         /*use_average_radius=*/msg->id == ArmorNumber::OUTPOST);
  const int idx = SelectSpLikeAimArmor(msg);
  if (idx < 0)
  {
    pitch = std::numeric_limits<float>::quiet_NaN();
    yaw = std::numeric_limits<float>::quiet_NaN();
    aim_x = aim_y = aim_z = 0.0f;
    return;
  }
  const auto [p, y] =
      CalculatePitchAndYaw(idx, msg, time_delay, s_bias_, z_bias_, current_v_,
                           /*use_target_center_for_yaw=*/false, aim_x, aim_y, aim_z);
  pitch = p;
  yaw = y;

  XR_LOG_DEBUG("SolveTrajectory SP-like idx=%d pitch=%f yaw=%f", idx, pitch, yaw);
}
