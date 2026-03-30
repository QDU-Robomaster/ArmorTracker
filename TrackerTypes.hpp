#pragma once

#include <Eigen/Dense>

#include "armor.hpp"
#include "transform.hpp"

struct TrackerTarget
{
  bool tracking{false};
  ArmorNumber id{ArmorNumber::INVALID};
  ArmorType armor_type{ArmorType::INVALID};
  int armors_num{0};
  bool jumped{false};
  Eigen::Matrix<double, 3, 1> position = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> velocity = Eigen::Matrix<double, 3, 1>::Zero();
  double yaw{0.0};
  double v_yaw{0.0};
  double radius_1{0.0};
  double radius_2{0.0};
  double dz{0.0};
};

struct TrackerSend
{
  bool is_fire{false};
  LibXR::Position<double> position{};
  double v_yaw{0.0};
  double pitch{0.0};
  double yaw{0.0};
  Eigen::Matrix<double, 3, 1> cmd_vel_linear = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> cmd_vel_angular = Eigen::Matrix<double, 3, 1>::Zero();
};
