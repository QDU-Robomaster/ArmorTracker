#pragma once

/**
 * @file ArmorTrackerTarget.hpp
 * @brief 定义 ArmorTracker 对外发布的目标状态消息。
 *
 * 该文件只描述 tracker 输出的目标几何，不包含弹道、开火或云台命令语义。
 * 后级 Aimer 应当以此消息作为输入自行完成瞄准解算。
 */

#include <cstdint>

#include <Eigen/Dense>

#include "ArmorDetectorTypes.hpp"

/**
 * @brief tracker 输出目标状态载荷。
 *
 * 该结构表达当前跟踪到的机器人中心、速度、yaw、半径和高低差。当前通过
 * tracker/target_frame 随同源图像帧一起发布。输出坐标使用与公开 B 系同向的
 * 惯性解算轴：右手系，x 向右，y 向前，z 向上；yaw 以前向为 0，左转为正。
 */
struct ArmorTrackerTarget
{
  uint64_t image_timestamp_us{};  ///< 图像传感器时间戳，单位 us。
  bool tracking{};                ///< 当前帧是否有有效跟踪目标。
  ArmorNumber id{ArmorNumber::INVALID};  ///< 目标机器人数字 ID。
  int armors_num{};                      ///< 目标装甲面数量，通常为 1/3/4。
  Eigen::Matrix<double, 3, 1> position =
      Eigen::Matrix<double, 3, 1>::Zero();  ///< 整车中心位置，单位 m。
  Eigen::Matrix<double, 3, 1> velocity =
      Eigen::Matrix<double, 3, 1>::Zero();  ///< 整车中心速度，单位 m/s。
  double yaw{};                            ///< 整车中心 yaw，单位 rad。
  double v_yaw{};                          ///< 整车 yaw 角速度，单位 rad/s。
  double radius_1{};                       ///< 偶数面或默认装甲半径，单位 m。
  double radius_2{};                       ///< 奇数面装甲半径，单位 m。
  double dz{};                             ///< 奇偶装甲面高度差，单位 m。
  int tracked_face_index{0};               ///< 当前 EKF 绑定的本地装甲面索引。
  int outpost_height_phase{0};
  bool face_switch_observed{false};        ///< 跟踪期间是否观测到换面。
};
