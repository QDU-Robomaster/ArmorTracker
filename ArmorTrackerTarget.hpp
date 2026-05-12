#pragma once

/**
 * @file ArmorTrackerTarget.hpp
 * @brief 定义 ArmorTracker 对外发布的目标状态消息。
 *
 * 该文件只描述 tracker 输出的目标几何与当前可见面锚点，不包含弹道、
 * 开火或云台命令语义。后级 Aimer 应当以此消息作为输入自行完成瞄准解算。
 */

#include <cstdint>

#include <Eigen/Dense>

#include "ArmorDetectorTypes.hpp"

/**
 * @brief tracker/target topic 的目标状态载荷。
 *
 * 该结构表达当前跟踪到的机器人中心、速度、yaw、半径、高低差以及当前观测面。
 * 坐标位于 tracker 配置的输出帧；字段保持平凡聚合，方便 SharedTopic / recorder
 * 直接按结构体尺寸注册和传输。
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
  bool face_switch_observed{false};        ///< 跟踪期间是否观测到换面。
  bool measured_face_valid{false};         ///< 当前帧是否存在有效可见面测量。
  bool use_measured_face_anchor{false};    ///< 后级是否可优先使用可见面锚点。
  int measured_face_index{-1};             ///< 当前可见测量面索引。
  Eigen::Matrix<double, 3, 1> measured_face_position =
      Eigen::Matrix<double, 3, 1>::Zero();  ///< 当前可见面位置，单位 m。
  double measured_face_yaw{};               ///< 当前可见面 yaw，单位 rad。
  Eigen::Matrix<double, 3, 1> velocity_variance =
      Eigen::Matrix<double, 3, 1>::Zero();  ///< 速度状态方差，单位 (m/s)^2。
  double velocity_confidence{};             ///< 速度可外推置信度，范围约为 0~1。
};
