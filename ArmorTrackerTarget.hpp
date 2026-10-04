#pragma once

/**
 * @file ArmorTrackerTarget.hpp
 * @brief 定义 ArmorTracker 对外发布的目标状态消息。
 *        Target state messages published by ArmorTracker.
 *
 * 该文件描述 tracker 输出的目标几何，后级 Aimer 以此消息作为瞄准解算的输入。
 * This file describes the target geometry output by the tracker; the downstream Aimer
 * uses it as input of the aiming solution.
 */

#include <Eigen/Dense>
#include <array>
#include <cstdint>

#include "ArmorDetectorTypes.hpp"

/**
 * @brief tracker 输出的目标状态载荷。
 *        Target state payload output by the tracker.
 *
 * 该结构表达当前跟踪到的机器人中心、速度、yaw、半径和高低差，通过
 * tracker/target_frame 随同源图像帧一起发布。输出坐标使用与公开 B 系同向的
 * 惯性解算轴：右手系，x 向右，y 向前，z 向上；yaw 以前向为 0，左转为正。
 * The structure holds the center, velocity, yaw, radii and height difference of the
 * tracked robot and is published with the source image frame through
 * tracker/target_frame. Output coordinates use inertial solution axes oriented like the
 * public B frame: right-handed, x right, y forward, z up; yaw is 0 forward and positive
 * to the left.
 */
struct ArmorTrackerTarget
{
  uint64_t image_timestamp_us{};         ///< 同步帧 IMU 时间戳 (us)
                                         ///< Synchronized-frame IMU timestamp (us)
  bool tracking{};                       ///< 当前帧是否有有效跟踪目标
                                         ///< Whether the current frame has a valid target
  ArmorNumber id{ArmorNumber::INVALID};  ///< 目标机器人编号
                                         ///< Target robot number
  int armors_num{};                      ///< 目标装甲面数量，通常为 1、3 或 4
                                         ///< Armor face count, usually 1, 3 or 4
  Eigen::Matrix<double, 3, 1> position =
      Eigen::Matrix<double, 3, 1>::Zero();  ///< 整车中心位置 (m)
                                            ///< Vehicle center position (m)
  Eigen::Matrix<double, 3, 1> velocity =
      Eigen::Matrix<double, 3, 1>::Zero();  ///< 整车中心速度 (m/s)
                                            ///< Vehicle center velocity (m/s)
  double yaw{};                             ///< 整车中心 yaw (rad)
                                            ///< Vehicle center yaw (rad)
  double v_yaw{};                           ///< 整车 yaw 角速度 (rad/s)
                                            ///< Vehicle yaw rate (rad/s)
  double radius_1{};                        ///< 偶数面或默认装甲半径 (m)
                                            ///< Even-face or default armor radius (m)
  double radius_2{};                        ///< 奇数面装甲半径 (m)
                                            ///< Odd-face armor radius (m)
  double dz{};                              ///< 奇偶装甲面高度差 (m)
                                            ///< Odd/even face height difference (m)
  int tracked_face_index{0};                ///< 当前 EKF 绑定的本地装甲面索引
                                            ///< Local face index bound to the EKF
  int outpost_height_phase{0};              ///< 前哨站高度相位
                                            ///< Outpost height phase
  bool face_switch_observed{false};         ///< 跟踪期间是否观测到换面
                                            ///< Face switch observed while tracking
};

/**
 * @brief Tracker 完成一帧处理后发布的进程内结果。
 *        In-process result published after the tracker finishes one frame.
 *
 * `image` 持有 CameraBase 对象池槽位。普通 Topic 只在同步回调期间借用
 * `const TrackedFrame*`，消费者在回调内完成使用，逐帧几何从
 * `image.Get()->geometry` 读取。
 * `image` holds a CameraBase object-pool slot. The plain Topic lends
 * `const TrackedFrame*` only during the synchronous callback; consumers finish using it
 * inside the callback and read the per-frame geometry from `image.Get()->geometry`.
 *
 * @tparam FrameLayoutV 帧布局。
 *                      Frame layout.
 */
template <CameraTypes::FrameLayout FrameLayoutV>
struct TrackedFrame
{
  using Base = CameraBase<FrameLayoutV>;
  using ImageFrame = typename Base::ImageFrame;
  using SharedFrame = typename Base::SharedFrame;
  using ImuStamped = typename Base::ImuStamped;

  uint64_t sequence{};          ///< CameraFrameSync 分配的帧序号
                                ///< Frame sequence number assigned by CameraFrameSync
  SharedFrame image{};          ///< 当前跟踪结果对应的共享图像所有权
                                ///< Shared image ownership of the current tracking result
  ImuStamped imu{};             ///< 与图像对齐的 IMU 样本
                                ///< IMU sample aligned with the image
  ArmorTrackerTarget target{};  ///< 本帧跟踪输出
                                ///< Tracking output of this frame
  /// 输出系 O 到 OpenCV 相机系的旋转，row-major
  /// Rotation from the output frame O to the OpenCV camera frame, row-major
  std::array<double, 9> output_to_camera_rotation{1.0, 0.0, 0.0, 0.0, 1.0,
                                                  0.0, 0.0, 0.0, 1.0};
  /// 输出系 O 到 OpenCV 相机系的平移 (m)
  /// Translation from the output frame O to the OpenCV camera frame (m)
  std::array<double, 3> output_to_camera_translation{0.0, 0.0, 0.0};

  /**
   * @brief 获取共享图像帧。
   *        Get the shared image frame.
   *
   * @return 图像帧指针。
   *         Pointer to the image frame.
   */
  [[nodiscard]] const ImageFrame* GetImageFrame() const noexcept { return image.Get(); }

  /**
   * @brief 判断共享图像是否有效。
   *        Check whether the shared image is valid.
   *
   * @return 有效为 true。
   *         True when valid.
   */
  [[nodiscard]] bool Valid() const noexcept { return image.Valid(); }
};

/**
 * @brief target_frame 普通 Topic 在同步回调期间借用的载荷。
 *        Payload lent by the target_frame plain Topic during the synchronous callback.
 */
template <CameraTypes::FrameLayout FrameLayoutV>
using TrackedFrameMessage = const TrackedFrame<FrameLayoutV>*;
