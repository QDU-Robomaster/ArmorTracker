#pragma once

#include <cstdint>

namespace RobotGameReferee
{
struct [[gnu::packed]] GameStatus
{
  uint8_t game_type : 4;
  uint8_t game_progress : 4;
  uint16_t stage_remain_time;
  uint64_t sync_time_stamp;
};

struct [[gnu::packed]] RobotStatus
{
  uint8_t robot_id;
  uint8_t robot_level;
  uint16_t remain_hp;
  uint16_t max_hp;
  uint16_t shooter_cooling_value;
  uint16_t shooter_heat_limit;
  uint16_t chassis_power_limit;
  uint8_t power_gimbal_output : 1;
  uint8_t power_chassis_output : 1;
  uint8_t power_launcher_output : 1;
};

struct [[gnu::packed]] LauncherData
{
  uint8_t bullet_type;
  uint8_t launcherer_id;
  uint8_t bullet_freq;
  float bullet_speed;
};

struct [[gnu::packed]] Pack
{
  RobotStatus robot_status;
  GameStatus game_status;
  LauncherData launcher_data;
};

static_assert(sizeof(GameStatus) == 11);
static_assert(sizeof(RobotStatus) == 13);
static_assert(sizeof(LauncherData) == 7);
static_assert(sizeof(Pack) == 31);
}  // namespace RobotGameReferee
