#pragma once

/**
 * @file ArmorTracker.hpp
 * @brief ArmorTracker 模块入口、配置和 topic 数据类型。
 */

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: Armor tracker
constructor_args:
  cfg:
    tracker:
      require_target_tag: false
      target_tag_id: -1
      min_detect_count: 2
      max_temp_lost_count: 15
      outpost_max_temp_lost_count: 75
      target_select:
        observed_count_weight: 1.6
        distance_weight: 2.0
        area_weight: 1.2
        spin_weight: 0.8
        angle_weight: 2.0
        max_distance_m: 8.0
        distance_span_m: 7.5
        area_norm_px: 6000.0
        observed_count_norm: 4.0
        max_spin_rad_s: 8.0
        max_angle_norm: 0.5
        detecting_scale: 0.55
        temp_lost_scale: 0.35
        switch_margin: 0.25

    extrinsic:
      camera_mount_to_body:
        rotation: [1.0, 0.0, 0.0, 0.0]
        translation: [0.0, 0.0, 0.0]

    preview:
      enabled: false
      preview_window_name: "armor_tracker_preview"
      preview_scale: 0.5
      preview_wait_key_ms: 1
      queue_capacity: 1
      output_mode: "window"
      web_bind_address: "0.0.0.0"
      web_port: 8080
      web_stream_name: "armor_tracker"
      max_fps: 30.0
  sync: '@camera_frame_sync'
template_args:
  - Info:
      width: 1280
      height: 720
      step: 3840
      encoding: CameraTypes::Encoding::BGR8
      camera_matrix: [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
      distortion_model: CameraTypes::DistortionModel::PLUMB_BOB
      distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
      rectification_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      projection_matrix: [800.0, 0.0, 640.0, 0.0, 0.0, 800.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0]
required_hardware: []
depends:
  - qdu-future/ArmorDetector
  - qdu-future/CameraFrameSync
  - qdu-future/VisionPreview
=== END MANIFEST === */
// clang-format on

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "ArmorDetectorTypes.hpp"
#include "ArmorTrackerCore.hpp"
#include "ArmorTrackerTarget.hpp"
#include "CameraFrameSync.hpp"
#include "VisionPreview.hpp"
#include "app_framework.hpp"
#include "libxr_time.hpp"
#include "logger.hpp"
#include "message.hpp"
#include "timebase.hpp"

#if defined(__has_include)
#if __has_include("print/print_api.hpp")
#define TRACKER_STDIO_HAS_COMPILED_PRINTF 1
#endif
#endif

#ifndef TRACKER_STDIO_HAS_COMPILED_PRINTF
#define TRACKER_STDIO_HAS_COMPILED_PRINTF 0
#endif

#if TRACKER_STDIO_HAS_COMPILED_PRINTF
#define TRACKER_STDIO_PRINT(format_literal) LibXR::STDIO::Printf<format_literal>()
#define TRACKER_STDIO_PRINTF(format_literal, ...) \
  LibXR::STDIO::Printf<format_literal>(__VA_ARGS__)
#else
#define TRACKER_STDIO_PRINT(format_literal) LibXR::STDIO::Printf(format_literal)
#define TRACKER_STDIO_PRINTF(format_literal, ...) \
  LibXR::STDIO::Printf(format_literal, __VA_ARGS__)
#endif

/**
 * @brief RoboMaster 装甲板目标跟踪模块。
 *
 * 模块订阅 detector 同帧结果，运行整车目标跟踪，发布同帧目标包，并可提交
 * 内置 preview 绘制任务。
 */
template <CameraTypes::CameraInfo CameraInfoV>
class ArmorTracker : public LibXR::Application
{
 public:
  using FrameSync = CameraFrameSync<CameraInfoV>;
  using Base = typename FrameSync::Base;
  using CameraInfo = typename Base::CameraInfo;
  using ImageFrame = typename FrameSync::ImageFrame;
  using DetectionPacket = ArmorDetectionsFramePacket<CameraInfoV>;
  using DetectionMessage = ArmorDetectionsFrameMessage<CameraInfoV>;
  using DetectionMessageArg = typename std::conditional<
      std::is_pointer<DetectionMessage>::value, DetectionMessage,
      const DetectionMessage&>::type;

  static inline constexpr CameraInfo kCameraInfo = CameraInfoV;

  /**
   * @brief 从 YAML 读取的运行配置。
   */
  struct Config
  {
    /**
     * @brief 目标选择和状态机参数。
     */
    struct TrackerParams
    {
      bool require_target_tag = false;
      int target_tag_id = -1;
      int min_detect_count = 2;
      int max_temp_lost_count = 15;
      int outpost_max_temp_lost_count = 75;
      struct TargetSelectParams
      {
        double observed_count_weight = 1.6;
        double distance_weight = 2.0;
        double area_weight = 1.2;
        double spin_weight = 0.8;
        double angle_weight = 2.0;
        double max_distance_m = 8.0;
        double distance_span_m = 7.5;
        double area_norm_px = 6000.0;
        double observed_count_norm = 4.0;
        double max_spin_rad_s = 8.0;
        double max_angle_norm = 0.5;
        double detecting_scale = 0.55;
        double temp_lost_scale = 0.35;
        double switch_margin = 0.25;
      } target_select;
    } tracker;

    /**
     * @brief 相机安装外参，表达在公开本体系 B 中。
     */
    struct ExtrinsicParams
    {
      /**
       * @brief 相机安装系 M 到本体系 B 的变换。
       *
       * rotation 为 wxyz 四元数，translation 单位为 m。M 与 OpenCV 相机系 C
       * 同原点，并使用公开本体系 B 的轴向：x 向右，y 向前，z 向上。OpenCV
       * 相机系 C 到 M 的固定轴变换由 ArmorTracker 内部处理。
       */
      struct CameraMountToBody
      {
        std::array<double, 4> rotation = {1.0, 0.0, 0.0, 0.0};
        std::array<double, 3> translation = {0.0, 0.0, 0.0};
      } camera_mount_to_body;
    } extrinsic;

    VisionPreview::RuntimeParam preview{};
  };

  /**
   * @brief tracker/target_frame 的同帧目标和源图像数据包。
   *
   * 该 topic 只用于同进程视觉链路回调，语义与 detector 给 tracker 的
   * armors_frame 一致：图像和 IMU 指针只在当前发布回调期间有效，跨线程或
   * 异步预览必须立即深拷贝图像。
   */
  struct TargetFramePacket
  {
    /// tracker 输入所用的 detector 同源图像/IMU 帧。
    ArmorDetectionsSourceFrame<CameraInfoV> source_frame{};
    /// 本帧 tracker 输出的目标结果，使用与公开 B 系同向的惯性输出轴 O。
    const ArmorTrackerTarget* target{nullptr};
    /// output 坐标到 OpenCV camera 坐标的旋转，row-major，满足 p_C = R_CO p_O + t_CO。
    std::array<double, 9> output_to_camera_rotation{
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    /// output 坐标到 OpenCV camera 坐标的平移，单位 m。
    std::array<double, 3> output_to_camera_translation{0.0, 0.0, 0.0};
  };

  /**
   * @brief tracker/target_frame topic 的数据类型。
   */
  using TargetFrameMessage = TargetFramePacket*;

  /**
   * @brief 构造模块并订阅 detector topic。
   */
  explicit ArmorTracker(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app,
                        Config cfg, FrameSync& sync);

  /**
   * @brief RamFS 命令入口，用于查看或修改 tracker 参数。
   */
  static int CommandFun(ArmorTracker* self, int argc, char** argv);

  /**
   * @brief 获取当前运行配置。
   */
  const Config& GetConfig() const { return cfg_; }

  /**
   * @brief 替换运行配置并重启相关运行态对象。
   */
  void SetConfig(const Config& cfg);

  /**
   * @brief RamFS 命令回调转接函数。
   */
  static int CommandAdapter(void* instance, int argc, char** argv)
  {
    return CommandFun(static_cast<ArmorTracker*>(instance), argc, argv);
  }

  /**
   * @brief 应用 monitor hook，当前 tracker 没有周期任务。
   */
  void OnMonitor() override;

 private:
  static constexpr const char* kDetectorTopicName = "armors_frame";

  /**
   * @brief 将模块配置转为 tracker core 配置。
   */
  armor_tracker_detail::Config BuildTrackerConfig() const;

  /**
   * @brief 处理一帧 detector 数据并发布 tracker 输出。
   */
  void ArmorsCallback(DetectionMessageArg message);

  /**
   * @brief 解析并订阅 detector 同帧结果 topic。
   */
  void SubscribeDetectorTopic();

  /**
   * @brief preview 启用时提交绘制任务。
   */
  void SubmitPreview(const ImageFrame& image_frame,
                     const ArmorDetectorResults& detector_armors,
                     const ArmorTrackerTarget& target_msg,
                     const armor_tracker_detail::Output& output);

  Config cfg_;
  armor_tracker_detail::TrackerCore tracker_{};
  VisionPreview preview_{};

  LibXR::Topic::Domain armor_detector_domain_ =
      LibXR::Topic::Domain("armor_detector");
  LibXR::Topic::Domain tracker_domain_ = LibXR::Topic::Domain("tracker");
  LibXR::Topic armors_topic_ = LibXR::Topic();
  LibXR::Topic target_frame_topic_ =
      LibXR::Topic("target_frame", sizeof(TargetFrameMessage), &tracker_domain_);

  const char* name_ = "armor_tracker";
  LibXR::RamFS::File cmd_file_;
  std::atomic<bool> params_is_changed_{false};
  ArmorTrackerTarget target_frame_target_msg_{};
  TargetFramePacket target_frame_packet_{};
  FrameSync& sync_;
};

/**
 * @brief tracker/target_frame 的跨模块数据包类型。
 */
template <CameraTypes::CameraInfo CameraInfoV>
using ArmorTrackerTargetFramePacket =
    typename ArmorTracker<CameraInfoV>::TargetFramePacket;

/**
 * @brief tracker/target_frame 的跨模块 topic 数据类型。
 */
template <CameraTypes::CameraInfo CameraInfoV>
using ArmorTrackerTargetFrameMessage =
    typename ArmorTracker<CameraInfoV>::TargetFrameMessage;

#include "ArmorTrackerPipeline.hpp"
