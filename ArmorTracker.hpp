#pragma once

/**
 * @file ArmorTracker.hpp
 * @brief ArmorTracker 模块主类、配置、消息载荷和类内运行态定义。
 *
 * 模块接收 ArmorDetector 的装甲检测结果和 CameraFrameSync 的同步相机/IMU
 * 数据，维护整车模型 EKF，发布 tracker/target、tracker/ekf_points
 * 和调试 topic。
 */

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: Armor tracker
constructor_args:
  cfg:
    limits:
      max_armor_distance: 10.0
      max_z_position: 1.0

    match:
      max_match_distance: 0.15
      max_match_yaw_diff: 1.0

    thresholds:
      tracking_thres: 5
      lost_time_thres: 0.3

    ekf:
      sigma2_q_xyz: 20.0
      sigma2_q_yaw: 100.0
      sigma2_q_r: 800

    geometry:
      initial_radius: 0.26
      min_radius: 0.12
      max_radius: 0.4

    noise:
      r_xyz_factor: 0.05
      r_yaw: 0.02

    frames:
      rotation: [1.0, 0.0, 0.0, 0.0]
      translation: [0.0, 0.0, 0.0]

    model:
      enable_pair_dz: true
      measurement_recenter_alpha: 0.25
      quality_recenter: true
      enable_pair_geometry: false
      enable_output_meas_anchor: false
      enable_aimer_meas_anchor: true
      enable_fixed_pose_yaw_opt: false

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

    xr:
      enemy_color_id: -1
      require_target_tag: false
      target_tag_id: -1
      min_detect_count: 2
      max_temp_lost_count: 15
      outpost_max_temp_lost_count: 75
      frame_convention: 1
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
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// 框架与外部依赖头
#include "ArmorTrackerCommon.hpp"
#include "ArmorTrackerFaceSelector.hpp"
#include "ArmorTrackerImageTracker.hpp"
#include "ArmorTrackerObserver.hpp"
#include "ArmorTrackerRuntimeConfig.hpp"
#include "ArmorTrackerXrLocked.hpp"
#include "ArmorTrackerTarget.hpp"
#include "CameraFrameSync.hpp"
#include "app_framework.hpp"
#include "ArmorDetectorTypes.hpp"
#include "cycle_value.hpp"
#include "extended_kalman_filter.hpp"
#include "libxr_time.hpp"
#include "logger.hpp"
#include "message.hpp"
#include "timebase.hpp"
#include "transform.hpp"
#include "VisionPreview.hpp"

#if defined(__has_include)
#if __has_include("print/print_api.hpp")
#define XR_TRACKER_STDIO_HAS_COMPILED_PRINTF 1
#endif
#endif

#ifndef XR_TRACKER_STDIO_HAS_COMPILED_PRINTF
#define XR_TRACKER_STDIO_HAS_COMPILED_PRINTF 0
#endif

#if XR_TRACKER_STDIO_HAS_COMPILED_PRINTF
#define XR_TRACKER_STDIO_PRINT(format_literal) \
  LibXR::STDIO::Printf<format_literal>()
#define XR_TRACKER_STDIO_PRINTF(format_literal, ...) \
  LibXR::STDIO::Printf<format_literal>(__VA_ARGS__)
#else
#define XR_TRACKER_STDIO_PRINT(format_literal) \
  LibXR::STDIO::Printf(format_literal)
#define XR_TRACKER_STDIO_PRINTF(format_literal, ...) \
  LibXR::STDIO::Printf(format_literal, __VA_ARGS__)
#endif

namespace cv
{
class Mat;
}

/**
 * @brief 基于 detector 结果和同步相机姿态维护整车装甲跟踪状态的应用模块。
 *
 * @tparam CameraInfoV 编译期相机模型和内参。
 */
template <CameraTypes::CameraInfo CameraInfoV>
class ArmorTracker : public LibXR::Application
{
 public:
  using FrameSync = CameraFrameSync<CameraInfoV>;  ///< 当前相机信息对应的同步模块类型。
  using Base = typename FrameSync::Base;           ///< 同步模块继承的相机基础类型。
  using CameraInfo = typename Base::CameraInfo;    ///< 相机信息类型别名。
  using ImageFrame = typename FrameSync::ImageFrame; ///< 同步图像帧类型。
  using ImuStamped = typename FrameSync::ImuStamped; ///< 同步 IMU 样本类型。
  using SyncedFrame = typename FrameSync::SyncedFrame; ///< 同步后的图像+IMU 帧类型。
  using DetectionPacket = ArmorDetectionsFramePacket<CameraInfoV>; ///< detector 包载荷类型。
  using DetectionMessage = ArmorDetectionsFrameMessage<CameraInfoV>; ///< detector topic 消息类型。
  using DetectionMessageArg = typename std::conditional<
      std::is_pointer<DetectionMessage>::value, DetectionMessage,
      const DetectionMessage&>::type; ///< 兼容指针和对象消息 ABI 的回调参数类型。

  static inline constexpr CameraInfo kCameraInfo = CameraInfoV; ///< 当前模板实例的相机信息。

  /**
   * @brief 构造函数接收的 tracker 配置聚合。
   */
  struct Config
  {
    /**
     * @brief detector 结果进入 tracker 前的空间过滤阈值。
     */
    struct Limits
    {
      double max_armor_distance = 10.0;  ///< XOY 平面最大装甲距离，单位 m。
      double max_z_position = 1.0;       ///< 最大绝对 z 坐标，单位 m。
    } limits;                            ///< 输入检测结果空间过滤配置。

    /**
     * @brief 单面候选匹配门限。
     */
    struct Match
    {
      double max_match_distance = 0.15;  ///< 匹配位置阈值，单位 m。
      double max_match_yaw_diff = 1.0;   ///< 匹配 yaw 阈值，单位 rad。
    } match;                             ///< detector 到预测面的匹配配置。

    /**
     * @brief 状态机进入/退出跟踪的阈值。
     */
    struct Thresholds
    {
      int tracking_thres = 5;        ///< 进入 TRACKING 需要的连续匹配帧数。
      double lost_time_thres = 0.3;  ///< TEMP_LOST 转 LOST 的时间阈值，单位 s。
    } thresholds;                    ///< 跟踪状态机阈值配置。

    /**
     * @brief 传统 EKF 过程噪声配置。
     */
    struct Ekf
    {
      double sigma2_q_xyz = 20.0;   ///< 位置/速度过程噪声。
      double sigma2_q_yaw = 100.0;  ///< yaw/角速度过程噪声。
      double sigma2_q_r = 800;      ///< 半径过程噪声。
    } ekf;                          ///< EKF 噪声配置。

    /**
     * @brief 整车几何半径先验和钳位范围。
     */
    struct Geometry
    {
      double initial_radius = 0.26;  ///< EKF 初始化半径先验，单位 m。
      double min_radius = 0.12;      ///< 几何半径下界，单位 m。
      double max_radius = 0.4;       ///< 几何半径上界，单位 m。
    } geometry;                      ///< 整车几何配置。

    /**
     * @brief detector 测量噪声配置。
     */
    struct Noise
    {
      double r_xyz_factor = 0.05;  ///< 位置观测噪声随距离缩放比例。
      double r_yaw = 0.02;         ///< yaw 观测噪声。
    } noise;                       ///< 测量噪声配置。

    /**
     * @brief 静态云台到相机外参。
     */
    struct Frames
    {
      std::array<double, 4> rotation = {1.0, 0.0, 0.0, 0.0};  ///< 四元数 wxyz。
      std::array<double, 3> translation = {0.0, 0.0, 0.0};    ///< 平移 xyz，单位 m。
    } frames;                                                 ///< 坐标系外参配置。

    /**
     * @brief 整车模型跟踪策略开关。
     */
    struct VehicleTuning
    {
      bool enable_pair_dz = true;               ///< 是否启用双装甲高低差软融合。
      double measurement_recenter_alpha = 0.25; ///< 单装甲测量重定位基础权重。
      bool quality_recenter = true;             ///< 是否按匹配质量调节重定位权重。
      bool enable_pair_geometry = false;        ///< 是否显式估计中心和长短半径。
      bool enable_output_meas_anchor = false;   ///< ekf_points 是否用测量面锚定输出。
      bool enable_aimer_meas_anchor = true;     ///< Aimer 是否使用测量面作为瞄准锚点。
      bool enable_fixed_pose_yaw_opt = false;   ///< 是否启用固定俯仰装甲 yaw 重估。
    } model;                                    ///< 整车模型策略配置。

    VisionPreview::RuntimeParam preview{};      ///< 可选实时预览配置。
    struct XrRuntime
    {
      int enemy_color_id = -1;
      bool require_target_tag = false;
      int target_tag_id = -1;
      int min_detect_count = 2;
      int max_temp_lost_count = 15;
      int outpost_max_temp_lost_count = 75;
      int frame_convention = 1;
    } xr;
  };

  /**
   * @brief 目标装甲面数量。
   */
  enum class ArmorsNum : std::uint8_t
  {
    NORMAL_4 = 4, ///< 常规四装甲目标。
    OUTPOST_3 = 3 ///< 前哨站或基地三装甲目标。
  };

  /**
   * @brief tracker 主状态机状态。
   */
  enum class State : std::uint8_t
  {
    LOST,      ///< 当前没有有效目标。
    DETECTING, ///< 连续检测确认阶段。
    TRACKING,  ///< 稳定跟踪阶段。
    TEMP_LOST, ///< 短暂丢失但仍保留预测状态。
  };

  /**
   * @brief tracker/info topic 的简要匹配信息。
   */
  struct TrackerInfo
  {
    double position_diff{};              ///< 当前匹配位置差，单位 m。
    double yaw_diff{};                   ///< 当前匹配 yaw 差，单位 rad。
    LibXR::Position<double> position{};  ///< 最近测量装甲位置。
    double yaw{};                        ///< 最近测量装甲 yaw，单位 rad。
  };

  /**
   * @brief tracker/ekf_points topic 的整车和装甲点云调试消息。
   */
  struct EkfPointsMsg
  {
    uint64_t image_timestamp_us{};          ///< 图像时间戳，单位 us。
    uint8_t count;                          ///< 实际装甲块数量，通常为 1/3/4。
    LibXR::Position<double> center_cam;     ///< 相机系整车中心位置。
    LibXR::Position<double> armors_cam[4];  ///< 相机系装甲位置，最多 4 块。
    bool valid[5];                          ///< center 与 4 个装甲点是否位于相机前方。
  };

  /**
   * @brief 单个候选匹配的调试记录。
   */
  struct CandidateDebugItem
  {
    uint8_t armor_index{};                     ///< detector 结果索引。
    uint8_t face_index{};                      ///< 匹配到的本地面索引。
    uint8_t same_number{};                     ///< 是否与当前跟踪 ID 同号。
    uint8_t reserved0{};                       ///< 对齐保留字段。
    int16_t image_track_id{-1};                ///< 关联图像 track ID。
    uint8_t image_track_confirmed{};           ///< 图像 track 是否已确认。
    uint8_t same_persistent_track{};           ///< 是否命中同一个持久图像 track。
    ArmorNumber number{ArmorNumber::INVALID};  ///< 候选装甲数字。
    ArmorType type{ArmorType::INVALID};        ///< 候选装甲类型。
    uint8_t reserved1{};                       ///< 对齐保留字段。
    uint8_t reserved2{};                       ///< 对齐保留字段。
    float score{};                             ///< 候选总分，越小越好。
    float position_diff{};                     ///< 位置残差，单位 m。
    float yaw_diff{};                          ///< yaw 残差，单位 rad。
    float view_bonus{};                        ///< 视角优先奖励项。
    float area_score{};                        ///< 图像面积评分。
    float frontality{};                        ///< 装甲朝向相机程度。
    float observation_quality_penalty{};       ///< 观测质量惩罚。
    float center_x{};                          ///< detector 图像中心 x。
    float center_y{};                          ///< detector 图像中心 y。
    float predicted_yaw{};                     ///< 预测面 yaw，单位 rad。
    float measured_yaw{};                      ///< 测量面 yaw，单位 rad。
  };

  /**
   * @brief tracker/candidate_debug topic 的候选选择和 EKF 更新调试消息。
   */
  struct CandidateDebugMsg
  {
    static constexpr uint8_t kMaxItems = 24;      ///< 最多记录的候选数量。
    static constexpr uint8_t kMaxDetections = 8;  ///< 最多记录的 detector 输入数量。

    uint64_t image_timestamp_us{};        ///< 图像时间戳，单位 us。
    uint8_t count{};                      ///< 有效候选记录数量。
    uint8_t selected_index{255};          ///< 被选中候选在 items 中的索引。
    uint8_t matched{};                    ///< 本帧是否接受候选并更新跟踪。
    uint8_t accepted_mode{};              ///< 候选接受模式枚举值。
    uint8_t detection_count{};            ///< 本帧 detector 输入数量。
    int8_t preferred_adjacent_face{-1};   ///< 按旋转方向优先的相邻面。
    uint8_t tracked_armors_num{};         ///< 当前目标装甲面数量。
    uint8_t has_same_number_candidate{};  ///< 是否存在同号候选。
    uint8_t face_switch_enabled{};        ///< 是否允许普通换面。
    uint8_t relaxed_face_switch_enabled{}; ///< 是否允许放宽换面。
    uint8_t odd_face_switch_enabled{};    ///< 是否允许切换到奇数高低面。
    uint8_t view_priority_enabled{};      ///< 是否启用视角优先。
    uint8_t directional_face_switch_enabled{}; ///< 是否启用方向换面限制。
    uint8_t tracked_face_track_id_valid{}; ///< 当前面图像 track ID 是否有效。
    int16_t tracked_face_track_id{-1};    ///< 当前面绑定的图像 track ID。
    float predicted_vyaw{};               ///< 预测角速度，单位 rad/s。
    float max_match_distance{};           ///< 当前匹配距离阈值，单位 m。
    float max_match_yaw_diff{};           ///< 当前匹配 yaw 阈值，单位 rad。
    float relaxed_same_face_distance{};   ///< 放宽同面匹配距离阈值。
    float relaxed_face_switch_distance{}; ///< 放宽换面距离阈值。
    float relaxed_face_switch_yaw_diff{}; ///< 放宽换面 yaw 阈值。
    float face_switch_score_deadzone{};   ///< 换面分数死区。
    float face_switch_position_deadzone{}; ///< 换面位置死区。
    float face_switch_yaw_deadzone{};     ///< 换面 yaw 死区。
    float face_switch_timeout_sec{};      ///< 换面冷却时间，单位 s。
    float face_switch_cooldown_remaining{}; ///< 当前剩余换面冷却时间。
    float best_same_face_score{};         ///< 最佳同面候选分。
    float best_switch_face_score{};       ///< 最佳换面候选分。
    uint8_t same_face_matched{};          ///< 是否同面匹配成功。
    uint8_t switch_face_matched{};        ///< 是否换面匹配成功。
    uint8_t switch_blocked_by_timeout{};  ///< 换面是否被冷却阻止。
    uint8_t switch_allowed{};             ///< 当前是否允许执行换面。
    uint8_t ekf_update_valid{};           ///< EKF 调试字段是否有效。
    uint8_t ekf_update_mode{};            ///< 1 为 YPD 观测，2 为 XYZ 观测。
    int8_t ekf_update_face{-1};           ///< EKF 更新使用的面索引。
    uint8_t ekf_freeze_delta_z{};         ///< 本次更新是否冻结高低差。
    uint8_t ekf_range_clamped{};          ///< 本次更新是否钳位 PnP 距离。
    float ekf_raw_range_m{};              ///< 原始测量距离，单位 m。
    float ekf_range_m{};                  ///< 写入 EKF 的测量距离，单位 m。
    float ekf_mahalanobis{};              ///< 本次创新马氏距离。
    float ekf_pre_res_x{};                ///< 更新前 x 残差，单位 m。
    float ekf_pre_res_y{};                ///< 更新前 y 残差，单位 m。
    float ekf_pre_res_z{};                ///< 更新前 z 残差，单位 m。
    float ekf_pre_res_norm{};             ///< 更新前位置残差模长，单位 m。
    float ekf_post_res_x{};               ///< 更新后 x 残差，单位 m。
    float ekf_post_res_y{};               ///< 更新后 y 残差，单位 m。
    float ekf_post_res_z{};               ///< 更新后 z 残差，单位 m。
    float ekf_post_res_norm{};            ///< 更新后位置残差模长，单位 m。
    float ekf_innov_0{};                  ///< EKF 创新向量第 0 维。
    float ekf_innov_1{};                  ///< EKF 创新向量第 1 维。
    float ekf_innov_2{};                  ///< EKF 创新向量第 2 维。
    float ekf_innov_3{};                  ///< EKF 创新向量第 3 维。
    float ekf_r_0{};                      ///< EKF 观测方差第 0 维。
    float ekf_r_1{};                      ///< EKF 观测方差第 1 维。
    float ekf_r_2{};                      ///< EKF 观测方差第 2 维。
    float ekf_r_3{};                      ///< EKF 观测方差第 3 维。
    std::array<int16_t, kMaxDetections> detection_track_ids{}; ///< 输入图像 track ID。
    std::array<uint8_t, kMaxDetections> detection_track_confirmed{}; ///< 输入 track 确认位。
    CandidateDebugItem items[kMaxItems]{}; ///< 候选明细数组。
  };

 public:
  /**
   * @brief 构造 tracker 模块并注册 detector 回调和命令文件。
   */
  explicit ArmorTracker(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app,
                        Config cfg, FrameSync& sync);

  /**
   * @brief RamFS 命令入口，用于运行时查看和修改部分配置。
   */
  static int CommandFun(ArmorTracker* self, int argc, char** argv);

  /**
   * @brief 获取当前配置。
   */
  const Config& GetConfig() const { return cfg_; }

  /**
   * @brief 更新当前配置并同步会影响运行阈值的字段。
   */
  void SetConfig(const Config& cfg);

  /**
   * @brief LibXR RamFS 命令系统使用的静态适配函数。
   */
  static int CommandAdapter(void* instance, int argc, char** argv)
  {
    return CommandFun(static_cast<ArmorTracker*>(instance), argc, argv);
  }

  /**
   * @brief 应用监控回调；当前模块不需要周期性监控逻辑。
   */
  void OnMonitor() override;

 private:
  /**
   * @brief 从检测结果中选择目标并初始化跟踪状态。
   */
  void Init(const ArmorDetectorResults& armors_msg);

  /**
   * @brief 对一帧检测结果执行预测、匹配和更新。
   */
  void Update(const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us);

  /**
   * @brief 单装甲退化模式下的每帧更新入口。
   */
  void UpdateSingleArmorMode(const ArmorDetectorResults& armors_msg,
                             uint64_t image_timestamp_us);

  /**
   * @brief 单装甲模式下选择最可信的观测。
   */
  std::optional<ArmorDetectorResult> SelectSingleArmorObservation(
      const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us,
      std::size_t& selected_index, int& detection_track_id,
      bool& confirmed_track, float& selected_center_diff, float& selected_area_log,
      float& selected_score);

  /**
   * @brief 更新 detector 结果对应的图像级 track ID。
   */
  void UpdateImageIdTracks(const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us);

  /**
   * @brief 查询 detector 结果索引对应的图像 track ID。
   */
  int FindDetectionTrackId(std::size_t armor_index) const;

  /**
   * @brief 查询 detector 结果索引对应的图像 track 是否已确认。
   */
  bool IsDetectionTrackConfirmed(std::size_t armor_index) const;

  /**
   * @brief 将换面选择结果写入 candidate_debug 消息。
   */
  void FillCandidateDebugFromSelection(
      const armor_tracker::FaceSelectionResult& selection,
      CandidateDebugMsg& candidate_debug);

  /**
   * @brief 将当前换面策略和预测量写入 candidate_debug 消息。
   */
  void FillCandidateDebugPolicy(
      CandidateDebugMsg& candidate_debug, const Eigen::VectorXd& ekf_prediction,
      const armor_tracker::FaceSelectionPolicy& face_policy) const;

  /**
   * @brief 写入一行状态审计 TSV。
   */
  void WriteStateAuditRow(
      uint64_t image_timestamp_us, const Eigen::VectorXd& ekf_prediction,
      const armor_tracker::FaceSelectionResult* selection, bool matched);

  /**
   * @brief 构造换面选择策略。
   */
  armor_tracker::FaceSelectionPolicy BuildFaceSelectionPolicy() const;

  /**
   * @brief 构造换面选择所需的当前跟踪状态。
   */
  armor_tracker::FaceSelectionTrackedState BuildFaceSelectionTrackedState() const;

  /**
   * @brief 返回 tracker 世界系中的相机位置。
   */
  Eigen::Vector3d GetCameraWorldPosition();

  /**
   * @brief 推进跟踪状态机。
   */
  void AdvanceTrackerState(bool matched);

  /**
   * @brief 应用换面选择结果并执行对应 EKF 更新。
   */
  bool ApplyFaceSelection(const armor_tracker::FaceSelectionResult& selection,
                          CandidateDebugMsg& candidate_debug,
                          bool freeze_delta_z, uint64_t image_timestamp_us);

  /**
   * @brief 在 TEMP_LOST 状态下尝试快速恢复同一目标。
   */
  bool TryRecoverTempLost(const ArmorDetectorResults& armors_msg,
                          CandidateDebugMsg& candidate_debug);

  /**
   * @brief 构造 observer 纯算法策略。
   */
  armor_tracker::ObserverPolicy BuildObserverPolicy() const;

  /**
   * @brief 构造 observer 纯算法运行态。
   */
  armor_tracker::ObserverRuntime BuildObserverRuntime() const;

  /**
   * @brief 将 observer 运行态回写到类成员。
   */
  void ApplyObserverRuntime(const armor_tracker::ObserverRuntime& runtime);

  /**
   * @brief 构造装甲面和图像 track 绑定运行态。
   */
  armor_tracker::FaceBindingRuntime BuildFaceBindingRuntime() const;

  /**
   * @brief 将装甲面绑定运行态回写到类成员。
   */
  void ApplyFaceBindingRuntime(const armor_tracker::FaceBindingRuntime& runtime);

  /**
   * @brief 按被选中候选更新当前跟踪目标身份。
   */
  void ApplySelectedIdentity(const armor_tracker::FaceMatchCandidate& selected_candidate);

  /**
   * @brief 按被选中候选更新当前面和图像 track 绑定关系。
   */
  void ApplySelectedFaceBinding(const armor_tracker::FaceMatchCandidate& selected_candidate,
                                bool did_face_switch);

  /**
   * @brief 填充单装甲模式下的 candidate_debug 消息。
   */
  void FillSingleArmorDebug(std::size_t selected_index, int detection_track_id,
                            bool confirmed_track, float score, float center_diff,
                            float area_log);

  /**
   * @brief detector 结果 topic 回调。
   */
  void ArmorsCallback(DetectionMessageArg message);

  /**
   * @brief 把当前 tracker 快照提交给实时预览线程。
   * @param image_frame 当前同步图像帧。
   * @param detector_armors 原始 detector 输出，未做 tracker 空间过滤。
   * @param target_msg 当前 tracker/target 输出快照。
   * @param ekf_msg 当前 tracker/ekf_points 输出快照。
   * @param candidate_debug_msg 当前候选选择调试快照。
   */
  void SubmitPreview(const ImageFrame& image_frame,
                     const ArmorDetectorResults& detector_armors,
                     const ArmorTrackerTarget& target_msg,
                     const EkfPointsMsg& ekf_msg,
                     const CandidateDebugMsg& candidate_debug_msg);

  armor_tracker_xr::Config BuildXrTrackerConfig() const;

  /**
   * @brief 用单个装甲观测初始化整车模型 EKF 状态。
   */
  void InitEKF(const ArmorDetectorResult& a);

  /**
   * @brief 从当前状态同步高低差参考。
   */
  void SyncDzReferenceFromState();

  /**
   * @brief 用当前测量装甲板对整车状态做刚性重定位。
   */
  void RecenterTrackedStateToMeasurement(const ArmorDetectorResult& armor,
                                         int observed_face_index,
                                         double measured_yaw);

  /**
   * @brief 将选择器局部面索引转换为 canonical 面索引。
   */
  int LocalFaceToCanonicalFace(int local_face_index) const;

  /**
   * @brief 对当前帧装甲 yaw 测量执行可选固定姿态优化。
   */
  void OptimizeArmorYawMeasurements(
      ArmorDetectorResults& armors_msg,
      const LibXR::Transform<double>& camera_pose_world) const;

  /**
   * @brief 对单个装甲 yaw 测量执行可选固定姿态优化。
   */
  bool OptimizeSingleArmorYawMeasurement(
      ArmorDetectorResult& armor,
      const LibXR::Transform<double>& camera_pose_world) const;

  /**
   * @brief 计算指定 yaw/pitch 假设的装甲四点重投影误差。
   */
  double ArmorYawReprojectionError(
      const ArmorDetectorResult& armor,
      const LibXR::Transform<double>& camera_pose_world,
      double yaw_rad, double pitch_rad) const;

  /**
   * @brief 从 EKF 状态同步几何运行态缓存。
   */
  void SyncGeometryRuntimeFromState();

  /**
   * @brief 钳位 EKF 几何状态到物理范围。
   */
  void ClampGeometryState();

  /**
   * @brief 从状态向量计算指定面的 yaw。
   */
  double GetArmorYawFromState(const Eigen::VectorXd& x, int face_index = 0) const;

  /**
   * @brief 从状态向量计算指定面的三维位置。
   */
  Eigen::Vector3d GetArmorPositionFromState(const Eigen::VectorXd& x,
                                            int face_index = 0) const;

  /**
   * @brief 单个 detector 装甲与预测装甲面的匹配结果。
   */
  struct VehicleArmorMatch
  {
    int id = 0;                                      ///< 匹配到的 canonical 面索引。
    double score = std::numeric_limits<double>::infinity(); ///< 归一化匹配分。
    double yaw_error = 0.0;                          ///< yaw 方位误差，单位 rad。
    double pitch_error = 0.0;                        ///< pitch 方位误差，单位 rad。
    double distance_error = 0.0;                     ///< 距离误差，单位 m。
    double angle_error = 0.0;                        ///< 装甲面 yaw 误差，单位 rad。
    double xyz_error = 0.0;                          ///< 三维位置误差，单位 m。
    double measured_yaw = 0.0;                       ///< 展开后的测量 yaw，单位 rad。
  };

  /**
   * @brief 双装甲配对求解使用的单侧观测。
   */
  struct VehiclePairObservation
  {
    std::size_t armor_index = 0;             ///< detector 结果索引。
    ArmorDetectorResult armor{};             ///< detector 装甲结果。
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero(); ///< tracker 世界系测量位置。
  };

  /**
   * @brief 双装甲几何求解得到的中心、半径和 yaw 拟合结果。
   */
  struct VehiclePairGeometryFit
  {
    bool valid = false;                         ///< 几何拟合是否有效。
    Eigen::Vector2d center = Eigen::Vector2d::Zero(); ///< XOY 平面中心。
    double r_even = 0.0;                        ///< 偶数面半径，单位 m。
    double r_odd = 0.0;                         ///< 奇数面半径，单位 m。
    double yaw = 0.0;                           ///< 整车 yaw，单位 rad。
    double fit_error = 0.0;                     ///< 左右中心交汇误差，单位 m。
    double center_shift = 0.0;                  ///< 相对当前状态的中心修正量。
    double radius_shift = 0.0;                  ///< 相对当前状态的半径修正量。
  };

  /**
   * @brief 一帧中最佳双装甲配对及其可写入 EKF 的观测。
   */
  struct VehiclePairMatch
  {
    bool valid = false;                         ///< 配对是否有效。
    bool geometry_valid = false;                ///< 是否包含有效双面几何。
    bool dz_valid = false;                      ///< 是否包含有效高低差观测。
    VehiclePairObservation left{};                   ///< 图像左侧观测。
    VehiclePairObservation right{};                  ///< 图像右侧观测。
    int left_face = 0;                          ///< 左侧观测匹配的 canonical 面。
    int right_face = 0;                         ///< 右侧观测匹配的 canonical 面。
    VehicleArmorMatch left_match{};                  ///< 左侧单面匹配结果。
    VehicleArmorMatch right_match{};                 ///< 右侧单面匹配结果。
    double score = std::numeric_limits<double>::infinity(); ///< 配对总分。
    double yaw = 0.0;                           ///< 配对估计的整车 yaw。
    double dz_observed = 0.0;                   ///< 观测到的奇偶面高低差。
    double even_z_observed = 0.0;               ///< 偶数面基础高度观测。
    VehiclePairGeometryFit geometry{};               ///< 双面几何拟合结果。
    int tracked_face = 0;                       ///< 本帧用于主跟踪更新的面。
    std::size_t tracked_armor_index = 0;        ///< 主跟踪装甲 detector 索引。
    ArmorDetectorResult tracked_armor{};        ///< 主跟踪装甲观测。
    VehicleArmorMatch tracked_match{};               ///< 主跟踪装甲匹配结果。
  };

  /** @brief 将角度限制到 (-pi, pi]。 */
  static double VehicleLimitRad(double angle);
  /** @brief 从四元数提取接近参考角的 detector 装甲 yaw。 */
  static double VehicleDetectorYawNear(const LibXR::Quaternion<double>& q,
                                  double reference_yaw);
  /** @brief 从 detector 装甲结果提取接近参考角的装甲 yaw。 */
  static double VehicleDetectorYawNear(const ArmorDetectorResult& armor,
                                  double reference_yaw);
  /** @brief 将 xyz 坐标转换为 yaw/pitch/distance。 */
  static Eigen::Vector3d VehicleXyzToYpd(const Eigen::Vector3d& xyz);
  /** @brief 计算 xyz 到 yaw/pitch/distance 的雅可比。 */
  static Eigen::MatrixXd VehicleXyzToYpdJacobian(const Eigen::Vector3d& xyz);
  /** @brief 判断是否为双装甲平衡类目标。 */
  static bool VehicleIsBalanceArmor(const ArmorDetectorResult& armor);
  /** @brief 返回目标应建模的装甲面数量。 */
  static int VehicleArmorCountFor(const ArmorDetectorResult& armor);
  /** @brief 返回初始化半径先验。 */
  double VehicleInitialRadiusFor(const ArmorDetectorResult& armor) const;
  /** @brief 返回初始化协方差对角线。 */
  static Eigen::VectorXd VehicleInitialP0DiagFor(const ArmorDetectorResult& armor);
  /** @brief 从状态向量计算指定面位置。 */
  Eigen::Vector3d VehicleArmorPosition(const Eigen::VectorXd& state, int id) const;
  /** @brief 计算指定面观测雅可比。 */
  Eigen::MatrixXd VehicleObservationJacobian(const Eigen::VectorXd& state, int id) const;
  /** @brief 计算单个装甲到指定面的匹配结果。 */
  VehicleArmorMatch VehicleMatchArmorToFace(const ArmorDetectorResult& armor,
                                  const Eigen::VectorXd& state,
                                  int face_index) const;
  /** @brief 计算单个装甲到所有面的最佳匹配。 */
  VehicleArmorMatch VehicleMatchArmor(const ArmorDetectorResult& armor,
                            const Eigen::VectorXd& state) const;
  /** @brief 用多装甲观测尝试规范化初始相位。 */
  bool VehicleTryCanonicalizeInitialState(const ArmorDetectorResults& armors_msg,
                                     bool force);
  /** @brief 从一帧检测结果中求解最佳双装甲配对。 */
  bool VehicleResolvePairMatch(const ArmorDetectorResults& armors_msg,
                          const Eigen::VectorXd& state,
                          VehiclePairMatch& pair_match) const;
  /** @brief 用相邻两装甲几何求解整车中心和半径。 */
  bool VehicleSolvePairGeometry(const VehiclePairObservation& left, int left_face,
                           double left_measured_yaw,
                           const VehiclePairObservation& right, int right_face,
                           double right_measured_yaw,
                           const Eigen::VectorXd& state,
                           VehiclePairGeometryFit& fit) const;
  /** @brief 将双装甲几何或高低差观测写入 EKF。 */
  void VehicleApplyPairGeometryUpdate(const VehiclePairMatch& pair_match);
  /** @brief 将高低差符号规范化为 canonical 正方向。 */
  void VehicleCanonicalizePairPhaseForPositiveDz();
  /** @brief 执行整车模型状态预测。 */
  void VehiclePredict();
  /** @brief 执行双装甲更新。 */
  void VehicleUpdatePair(const VehiclePairMatch& pair_match,
                    uint64_t image_timestamp_us = 0,
                    CandidateDebugMsg* candidate_debug = nullptr);
  /** @brief 执行单装甲观测更新。 */
  void VehicleUpdate(const ArmorDetectorResult& armor, const VehicleArmorMatch& match,
                bool freeze_delta_z, uint64_t image_timestamp_us = 0,
                CandidateDebugMsg* candidate_debug = nullptr);
  /** @brief 更新中心速度观测器。 */
  void VehicleUpdateCenterMotionObserver(const ArmorDetectorResult& armor,
                                    const VehicleArmorMatch& match,
                                    uint64_t image_timestamp_us);
  /** @brief 用可见面 yaw 差分修正输出角速度。 */
  void VehicleApplyYawRateObserver(double output_yaw, uint64_t image_timestamp_us,
                              ArmorTrackerTarget& target_msg);
  /** @brief 检查整车模型几何状态是否发散。 */
  bool VehicleStateDiverged() const;
  /** @brief 判断是否启用双装甲高低差观测。 */
  bool VehiclePairDeltaZEnabled() const;
  /** @brief 判断是否启用双装甲几何观测。 */
  bool VehiclePairGeometryEnabled() const;
  /** @brief 返回测量面重定位基础系数。 */
  double VehicleMeasurementRecenterAlpha() const;
  /** @brief 判断是否按观测质量调节重定位系数。 */
  bool VehicleMeasurementRecenterQualityEnabled() const;
  /** @brief 判断 ekf_points 是否用测量面锚定输出。 */
  bool VehicleMeasurementAnchoredOutputEnabled() const;
  /** @brief 判断 Aimer 是否使用测量面锚点。 */
  bool VehicleAimerMeasuredFaceAnchorEnabled() const;
  /** @brief 判断是否启用固定姿态 yaw 优化。 */
  bool VehicleFixedPoseYawOptimizeEnabled() const;
  /**
   * @brief EKF 算法对象、状态向量和当前观测缓存。
   */
  struct EKFBlock
  {
    /**
     * @brief 当前观测对几何状态的约束模式。
     */
    enum class MeasurementGeometryMode : std::uint8_t
    {
      FULL_BODY = 0,         ///< 观测约束完整整车几何。
      VISIBLE_FACE_ONLY = 1, ///< 观测仅约束当前可见装甲面。
    };

    ExtendedKalmanFilter ekf;  ///< 传统 EKF 包装对象。
    Eigen::VectorXd measurement = Eigen::VectorXd::Zero(4); ///< 当前观测 z=[xa,ya,za,yaw]。
    Eigen::VectorXd state =
        Eigen::VectorXd::Zero(11); ///< 状态 x=[xc,vxc,yc,vyc,za,vza,yaw,vyaw,r1,dr,dz]。
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(11, 11); ///< 当前后验协方差。
    int measurement_face_index = 0; ///< 当前观测绑定的 canonical 面索引。
    MeasurementGeometryMode measurement_geometry_mode =
        MeasurementGeometryMode::FULL_BODY; ///< 当前观测几何约束模式。
  } ekf_; ///< EKF 相关运行态。

  armor_tracker::ImageTrackManager image_tracker_{}; ///< 图像级装甲 track 管理器。

  /**
   * @brief 跟踪状态机、目标身份和观测器运行态。
   */
  struct TrackRuntime
  {
    State state = State::LOST;      ///< 当前跟踪状态机状态。
    int detect_count = 0;           ///< DETECTING 连续命中计数。
    int lost_count = 0;             ///< TEMP_LOST 连续丢失计数。
    int tracking_thres = 5;         ///< 进入 TRACKING 的帧数阈值。
    int lost_thres = 0;             ///< 进入 LOST 的帧数阈值，由时间阈值换算。
    uint8_t recovery_count = 0;     ///< TEMP_LOST 恢复次数。
    double last_yaw = 0.0;          ///< 最近一次有效测量 yaw。
    double info_position_diff = 0.0; ///< tracker/info 位置差缓存。
    double info_yaw_diff = 0.0;     ///< tracker/info yaw 差缓存。
    double face_switch_cooldown_remaining = 0.0; ///< 剩余换面冷却时间。
    int update_count = 0;           ///< 接受观测更新次数。
    int switch_count = 0;           ///< 已执行换面次数。
    int suspect_count = 0;          ///< TRACKING 中疑似短暂 miss 计数。

    ArmorNumber tracked_id = ArmorNumber::INVALID; ///< 当前跟踪目标数字 ID。
    ArmorDetectorResult tracked_armor{};           ///< 最近一次接受的装甲观测。
    ArmorsNum tracked_armors_num = ArmorsNum::NORMAL_4; ///< 当前目标装甲面数量。
    int tracked_face_index = 0;                     ///< 当前绑定的 canonical 面索引。
    bool tracked_face_track_id_valid = false;       ///< 当前面 track ID 是否有效。
    uint16_t tracked_face_track_id = 0;             ///< 当前面绑定的图像 track ID。
    std::array<bool, 4> face_track_id_valid{};      ///< 每个 canonical 面 track ID 有效位。
    std::array<uint16_t, 4> face_track_id{};        ///< 每个 canonical 面绑定的图像 track ID。
    bool model_initial_phase_resolved = false;         ///< 初始 canonical 相位是否已确定。
    bool model_pair_delta_z_valid = false;             ///< 双装甲高低差是否已观测过。
    bool measurement_valid_current_frame = false;   ///< 当前帧是否接受测量。
    bool center_motion_observer_valid = false;      ///< 中心速度观测器是否有效。
    uint64_t center_motion_observer_timestamp_us = 0; ///< 中心速度观测器时间戳。
    Eigen::Vector3d center_motion_observer_anchor = Eigen::Vector3d::Zero(); ///< 上次中心锚点。
    Eigen::Vector3d center_motion_observer_velocity = Eigen::Vector3d::Zero(); ///< 滤波后速度。
    Eigen::Vector3d center_motion_observer_raw_velocity = Eigen::Vector3d::Zero(); ///< 原始差分速度。
    double center_motion_observer_confidence = 0.0; ///< 中心速度观测器置信度。
    std::uint32_t center_motion_observer_samples = 0; ///< 中心速度观测器样本数。
    bool yaw_rate_observer_valid = false;          ///< yaw rate 观测器是否有效。
    uint64_t yaw_rate_observer_timestamp_us = 0;   ///< yaw rate 观测器时间戳。
    double yaw_rate_observer_yaw = 0.0;            ///< yaw rate 上次展开 yaw。
    double yaw_rate_observer_value = 0.0;          ///< yaw rate 滤波值。
    std::uint32_t yaw_rate_observer_samples = 0;   ///< yaw rate 样本数。
    bool model_range_filter_valid = false;            ///< 单面距离钳位滤波是否有效。
    uint64_t model_range_filter_timestamp_us = 0;     ///< 距离钳位滤波时间戳。
    int model_range_filter_face = -1;                 ///< 距离钳位滤波绑定面。
    double model_range_filter_distance = 0.0;         ///< 距离钳位滤波上次距离。
    bool output_anchor_delta_valid = false;        ///< 输出锚定平移量是否有效。
    uint64_t output_anchor_delta_timestamp_us = 0; ///< 输出锚定更新时间戳。
    Eigen::Vector3d output_anchor_delta = Eigen::Vector3d::Zero(); ///< 输出锚定平移量。

    double dz = 0.0;          ///< 奇偶装甲面高低差，单位 m。
    double dz_abs_ref = 0.0;  ///< 高低差绝对值参考。
    double another_r = 0.0;   ///< 另一组装甲面的半径，单位 m。

  } rt_;

  /**
   * @brief 每帧时间和 dt 估计缓存。
   */
  struct TimeBlock
  {
    LibXR::MicrosecondTimestamp last_time = LibXR::Timebase::GetMicroseconds(); ///< 上次进程时间。
    uint64_t last_image_timestamp_us = 0; ///< 上次图像时间戳。
    double dt = 1.0 / 100.0;              ///< 当前帧间隔，初始假定 100Hz。
  } time_;                                ///< 时间缓存。

  /**
   * @brief topic、坐标系和相机姿态运行态。
   */
  struct IOBlock
  {
    LibXR::Transform<double> gimbal_to_camera_transform_static{}; ///< 静态云台到相机外参。
    LibXR::Transform<double> current_camera_pose{};               ///< 当前相机到 tracker 世界姿态。
    bool current_camera_pose_valid = false;                       ///< 当前相机姿态是否有效。
    armor_tracker_detail::CameraPoseRuntime camera_pose_runtime{}; ///< 相对姿态模式运行态。

    LibXR::Topic::Domain tracker_domain = LibXR::Topic::Domain("tracker"); ///< tracker topic 域。
    LibXR::Topic info_topic = LibXR::Topic("info", sizeof(TrackerInfo), &tracker_domain); ///< info topic。
    LibXR::Topic target_topic =
        LibXR::Topic("target", sizeof(ArmorTrackerTarget), &tracker_domain); ///< target topic。
    LibXR::Topic ekf_points_topic =
        LibXR::Topic("ekf_points", sizeof(EkfPointsMsg), &tracker_domain); ///< ekf_points topic。
    LibXR::Topic candidate_debug_topic =
        LibXR::Topic("candidate_debug", sizeof(CandidateDebugMsg), &tracker_domain); ///< 候选调试 topic。
  } io_; ///< IO 和坐标系运行态。

  Config cfg_; ///< 当前配置副本。
  armor_tracker_xr::LockedTracker xr_tracker_{}; ///< 锁定 XR tracker 运行态。
  VisionPreview preview_{}; ///< 可选实时预览工具，不参与主链路同步。

  const char* name_ = "armor_tracker";      ///< RamFS 命令文件名。
  LibXR::RamFS::File cmd_file_;             ///< RamFS 命令文件句柄。
  std::atomic<bool> params_is_changed_{false}; ///< 运行时配置是否被命令修改。
  EkfPointsMsg ekf_msg_;                    ///< ekf_points 发布缓存。
  CandidateDebugMsg candidate_debug_msg_{}; ///< candidate_debug 发布缓存。

  /**
   * @brief 状态审计 TSV 文件运行态。
   */
  struct StateAuditBlock
  {
    std::string path{};    ///< 审计文件路径。
    std::ofstream file{};  ///< 审计文件流。
    bool open_failed = false; ///< 是否已经打开失败，避免重复刷日志。
  } state_audit_;          ///< 状态审计输出运行态。
  FrameSync& sync_;        ///< 相机同步模块引用。
};

using armor_tracker_detail::ArmorTrackerArmorsTopicName;
using armor_tracker_detail::ArmorTrackerCameraRotationToTrackerWorldPose;
using armor_tracker_detail::DirectionalFaceSwitchEnabled;
using armor_tracker_detail::FaceSwitchEnabled;
using armor_tracker_detail::FaceSwitchPositionDeadzone;
using armor_tracker_detail::FaceSwitchScoreDeadzone;
using armor_tracker_detail::FaceSwitchTimeoutSec;
using armor_tracker_detail::FaceSwitchYawDeadzone;
using armor_tracker_detail::IdAssistEnabled;
using armor_tracker_detail::IdAssistSameFaceAreaLogGate;
using armor_tracker_detail::IdAssistSameFaceCenterGatePx;
using armor_tracker_detail::IdTrackAppearHits;
using armor_tracker_detail::IdTrackAppearTimeoutSec;
using armor_tracker_detail::IdTrackDisappearMisses;
using armor_tracker_detail::IdTrackDisappearTimeoutSec;
using armor_tracker_detail::IdTrackTentativeMisses;
using armor_tracker_detail::IdTrackTentativeTimeoutSec;
using armor_tracker_detail::InitRequiresStableObservation;
using armor_tracker_detail::MatchYawAllowPiAmbiguityEnabled;
using armor_tracker_detail::ObservationConfirmedTrackBonus;
using armor_tracker_detail::ObservationQualityEnabled;
using armor_tracker_detail::ObservationQualityScoreWeight;
using armor_tracker_detail::ObservationStableMaxReprojectionPx;
using armor_tracker_detail::ObservationStableMinAreaPx;
using armor_tracker_detail::ObservationStableMinConfidence;
using armor_tracker_detail::OddFaceSwitchEnabled;
using armor_tracker_detail::RelaxedFaceSwitchEnabled;
using armor_tracker_detail::SingleArmorAreaLogGate;
using armor_tracker_detail::SingleArmorImageCenterGatePx;
using armor_tracker_detail::SingleArmorModeEnabled;
using armor_tracker_detail::SymmetricGeometryEnabled;
using armor_tracker_detail::VehicleDeltaZInitialVariance;
using armor_tracker_detail::VehicleDeltaZProcessVariance;
using armor_tracker_detail::VehicleDeltaRadiusShrinkAlpha;
using armor_tracker_detail::VehicleDirectDeltaZAlpha;
using armor_tracker_detail::VehicleDirectDeltaZEnabled;
using armor_tracker_detail::VehicleDirectDeltaZMaxAbs;
using armor_tracker_detail::VehicleCanonicalInitEnabled;
using armor_tracker_detail::VehicleCanonicalInitMaxAbsDz;
using armor_tracker_detail::VehicleCanonicalInitMaxUpdates;
using armor_tracker_detail::VehicleCanonicalInitMaxScore;
using armor_tracker_detail::VehicleCanonicalInitMinHeight;
using armor_tracker_detail::VehicleCanonicalInitPreferPositiveDz;
using armor_tracker_detail::VehiclePitchVarianceScale;
using armor_tracker_detail::VehicleYpdArmorYawVarianceScale;
using armor_tracker_detail::VehicleYpdDistanceVarianceScale;
using armor_tracker_detail::VehiclePairDeltaZMaxAbs;
using armor_tracker_detail::VehiclePairDeltaZMinHeight;
using armor_tracker_detail::VehiclePairDeltaZVariance;
using armor_tracker_detail::VehiclePairDualUpdateEnabled;
using armor_tracker_detail::VehiclePairGeometryCenterVariance;
using armor_tracker_detail::VehiclePairGeometryCovarianceFloor;
using armor_tracker_detail::VehiclePairGeometryMaxCenterShift;
using armor_tracker_detail::VehiclePairGeometryMaxFitError;
using armor_tracker_detail::VehiclePairGeometryMaxRadiusShift;
using armor_tracker_detail::VehiclePairGeometryMinDeterminant;
using armor_tracker_detail::VehiclePairGeometryRadiusVariance;
using armor_tracker_detail::VehiclePairGeometryYawVariance;
using armor_tracker_detail::VehicleFreezeSingleObservationDeltaZEnabled;
using armor_tracker_detail::VehicleMeasurementAnchoredOutputEnabled;
using armor_tracker_detail::VehicleOutputMeasAnchorAlpha;
using armor_tracker_detail::VehicleOutputMeasAnchorLateralAlpha;
using armor_tracker_detail::VehicleOutputMeasAnchorMaxDelta;
using armor_tracker_detail::VehicleOutputMeasAnchorMaxStep;
using armor_tracker_detail::VehicleFixedPoseYawCoarseStepDeg;
using armor_tracker_detail::VehicleFixedPoseYawFineStepDeg;
using armor_tracker_detail::VehicleFixedPoseYawMinGainPx;
using armor_tracker_detail::VehicleFixedPoseYawOptimizeEnabled;
using armor_tracker_detail::VehicleFixedPoseYawPitchDeg;
using armor_tracker_detail::VehicleFixedPoseYawRangeDeg;
using armor_tracker_detail::VehicleCenterMotionObserverEnabled;
using armor_tracker_detail::VehicleCenterMotionObserverRadialVelocityEnabled;
using armor_tracker_detail::VehicleYawRateObserverAlpha;
using armor_tracker_detail::VehicleYawRateObserverBlend;
using armor_tracker_detail::VehicleYawRateObserverEnabled;
using armor_tracker_detail::VehicleYawRateObserverMaxBlendDelta;
using armor_tracker_detail::VehicleYawRateObserverMaxRaw;
using armor_tracker_detail::VehicleYawRateObserverMinSamples;
using armor_tracker_detail::VehicleYawRateObserverTau;
using armor_tracker_detail::VehicleOutputExtrapolateSeconds;
using armor_tracker_detail::VehicleMeasurementRecenterAlphaBad;
using armor_tracker_detail::VehicleMeasurementRecenterAlphaGood;
using armor_tracker_detail::VehicleMeasurementRecenterScoreBad;
using armor_tracker_detail::VehicleMeasurementRecenterScoreGood;
using armor_tracker_detail::VehicleMeasurementRecenterXyzBad;
using armor_tracker_detail::VehicleMeasurementRecenterXyzGood;
using armor_tracker_detail::VehicleMeasurementRecenterYawBad;
using armor_tracker_detail::VehicleMeasurementRecenterYawGood;
using armor_tracker_detail::VehicleMeasurementPositionAnchorAlpha;
using armor_tracker_detail::VehicleMeasurementPositionAnchorXyzBad;
using armor_tracker_detail::VehicleStaticDeltaZ;
using armor_tracker_detail::VehicleStaticDeltaZEnabled;
using armor_tracker_detail::VehicleXyzMeasurementFullGeometryEnabled;
using armor_tracker_detail::VehicleXyzMeasurementRFactor;
using armor_tracker_detail::VehicleXyzMeasurementUpdateEnabled;
using armor_tracker_detail::VehicleXyzMeasurementYawVariance;
using armor_tracker_detail::TempLostRecoveryEnabled;
using armor_tracker_detail::ViewPriorityEnabled;
using armor_tracker::AngularDiffAbs;
using armor_tracker::LogImpossibleYawDiff;
using armor_tracker::OrientationToYawNear;
using armor_tracker::QuaternionToYaw;
using armor_tracker::TimestampAbsDiff;
using armor_tracker::UnwrapYawNear;

/**
 * @brief 读取配置和环境变量，判断双装甲高低差更新是否启用。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehiclePairDeltaZEnabled() const
{
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_DISABLE_PAIR_DZ"))
  {
    return false;
  }
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_PAIR_DZ"))
  {
    return true;
  }
  return cfg_.model.enable_pair_dz;
}

/**
 * @brief 读取配置和环境变量，判断双装甲几何更新是否启用。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehiclePairGeometryEnabled() const
{
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_DISABLE_PAIR_GEOMETRY"))
  {
    return false;
  }
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_PAIR_GEOMETRY"))
  {
    return true;
  }
  return cfg_.model.enable_pair_geometry;
}

/**
 * @brief 返回单面测量重定位基础系数。
 */
template <CameraTypes::CameraInfo CameraInfoV>
double ArmorTracker<CameraInfoV>::VehicleMeasurementRecenterAlpha() const
{
  return std::clamp(
      armor_tracker_detail::ParseEnvDouble(
          "XR_TRACKER_MODEL_MEAS_RECENTER_ALPHA",
          cfg_.model.measurement_recenter_alpha),
      0.0, 1.0);
}

/**
 * @brief 判断是否按观测质量动态调节重定位系数。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehicleMeasurementRecenterQualityEnabled() const
{
  const char* env = std::getenv("XR_TRACKER_MODEL_QUALITY_RECENTER");
  if (env != nullptr)
  {
    return env[0] != '\0' && env[0] != '0';
  }
  return cfg_.model.quality_recenter;
}

/**
 * @brief 判断 ekf_points 输出是否用当前测量面锚定。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehicleMeasurementAnchoredOutputEnabled() const
{
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_DISABLE_OUTPUT_MEAS_ANCHOR"))
  {
    return false;
  }
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_OUTPUT_MEAS_ANCHOR"))
  {
    return true;
  }
  return cfg_.model.enable_output_meas_anchor;
}

/**
 * @brief 判断 target 输出是否提示 Aimer 使用测量面锚点。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehicleAimerMeasuredFaceAnchorEnabled() const
{
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_DISABLE_AIMER_MEAS_ANCHOR"))
  {
    return false;
  }
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_AIMER_MEAS_ANCHOR"))
  {
    return true;
  }
  return cfg_.model.enable_aimer_meas_anchor;
}

/**
 * @brief 判断是否启用固定俯仰装甲 yaw 重估。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::VehicleFixedPoseYawOptimizeEnabled() const
{
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_DISABLE_FIXED_POSE_YAW_OPT"))
  {
    return false;
  }
  if (armor_tracker_detail::EnvFlagEnabled("XR_TRACKER_MODEL_ENABLE_FIXED_POSE_YAW_OPT"))
  {
    return true;
  }
  return cfg_.model.enable_fixed_pose_yaw_opt;
}

template <CameraTypes::CameraInfo CameraInfoV>
armor_tracker_xr::Config ArmorTracker<CameraInfoV>::BuildXrTrackerConfig() const
{
  armor_tracker_xr::Config config{};
  config.enemy_color_id = cfg_.xr.enemy_color_id;
  config.require_target_tag = cfg_.xr.require_target_tag;
  config.target_tag_id = cfg_.xr.target_tag_id;
  config.min_detect_count = cfg_.xr.min_detect_count;
  config.max_temp_lost_count = cfg_.xr.max_temp_lost_count;
  config.outpost_max_temp_lost_count = cfg_.xr.outpost_max_temp_lost_count;
  config.frame_convention = cfg_.xr.frame_convention == 0 ? 0 : 1;
  config.camera_matrix = {
      kCameraInfo.camera_matrix[0], kCameraInfo.camera_matrix[1],
      kCameraInfo.camera_matrix[2], kCameraInfo.camera_matrix[3],
      kCameraInfo.camera_matrix[4], kCameraInfo.camera_matrix[5],
      kCameraInfo.camera_matrix[6], kCameraInfo.camera_matrix[7],
      kCameraInfo.camera_matrix[8]};
  return config;
}

// 分离出的实现块：让主头文件只保留 tracker 主流程。
#include "ArmorTrackerVehicleModel.hpp"
#include "ArmorTrackerRuntimeAdapter.hpp"
#include "ArmorTrackerPipeline.hpp"

/**
 * @brief 构造 tracker，初始化 EKF 模型、静态外参、topic 和 RamFS 命令。
 */
template <CameraTypes::CameraInfo CameraInfoV>
ArmorTracker<CameraInfoV>::ArmorTracker(LibXR::HardwareContainer& hw,
                                        LibXR::ApplicationManager&,
                                        Config cfg, FrameSync& sync)
    : cfg_(std::move(cfg)),
      cmd_file_(LibXR::RamFS::CreateFile(name_, CommandFun, this)),
      sync_(sync)
{
  XR_LOG_INFO("Starting ArmorTracker!");
  xr_tracker_.Configure(BuildXrTrackerConfig());
  preview_.Start(cfg_.preview);

  hw.template FindOrExit<LibXR::RamFS>({"ramfs"})->Add(cmd_file_);
  // 初值（和老逻辑一致）
  rt_.tracking_thres = cfg_.thresholds.tracking_thres;
  io_.gimbal_to_camera_transform_static = {
      LibXR::Quaternion<double>(cfg_.frames.rotation[0], cfg_.frames.rotation[1],
                                cfg_.frames.rotation[2], cfg_.frames.rotation[3]),
      LibXR::Position<double>(cfg_.frames.translation[0], cfg_.frames.translation[1],
                              cfg_.frames.translation[2])};

  // ---------------- EKF 设置 ----------------
  // 状态 x = [xc, vxc, yc, vyc, za, vza, yaw0, vyaw, r1, dr, dz]
  // 其中 yaw0/r1 对应 canonical face 0，face 1/3 使用 r1 + dr 与 za + dz。
  // 观测 z = [xa, ya, za, yaw(face_i)]，由 measurement_face_index 指定当前观测面。
  auto f = [this](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd x_new = x;
    x_new(ExtendedKalmanFilter::X_CENTER) +=
        x(ExtendedKalmanFilter::V_X_CENTER) * time_.dt;
    x_new(ExtendedKalmanFilter::Y_CENTER) +=
        x(ExtendedKalmanFilter::V_Y_CENTER) * time_.dt;
    x_new(ExtendedKalmanFilter::Z_ARMOR) += x(ExtendedKalmanFilter::V_Z_ARMOR) * time_.dt;
    x_new(ExtendedKalmanFilter::YAW) += x(ExtendedKalmanFilter::V_YAW) * time_.dt;
    return x_new;
  };
  auto j_f = [this](const Eigen::VectorXd&)
  {
    Eigen::MatrixXd f = Eigen::MatrixXd::Identity(11, 11);
    double d = time_.dt;
    f(ExtendedKalmanFilter::X_CENTER, ExtendedKalmanFilter::V_X_CENTER) = d;
    f(ExtendedKalmanFilter::Y_CENTER, ExtendedKalmanFilter::V_Y_CENTER) = d;
    f(ExtendedKalmanFilter::Z_ARMOR, ExtendedKalmanFilter::V_Z_ARMOR) = d;
    f(ExtendedKalmanFilter::YAW, ExtendedKalmanFilter::V_YAW) = d;
    return f;
  };
  auto h = [this](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd z(4);
    const int face_index = std::max(0, ekf_.measurement_face_index);
    const Eigen::Vector3d armor_position =
        GetArmorPositionFromState(x, face_index);
    z(0) = armor_position.x();
    z(1) = armor_position.y();
    z(2) = armor_position.z();
    z(3) = GetArmorYawFromState(x, face_index);
    return z;
  };
  auto j_h = [this](const Eigen::VectorXd& x)
  {
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(4, 11);
    const auto runtime = BuildObserverRuntime();
    const auto policy = BuildObserverPolicy();
    const int face_index = std::max(0, ekf_.measurement_face_index);
    const double yaw =
        armor_tracker::GetArmorYawFromState(x, runtime, face_index);
    const bool odd_face =
        !policy.symmetric_geometry_enabled && runtime.tracked_armors_num == 4 &&
        (face_index % 2 == 1);
    const bool visible_face_only_geometry =
        ekf_.measurement_geometry_mode ==
        EKFBlock::MeasurementGeometryMode::VISIBLE_FACE_ONLY;
    const double radius =
        odd_face ? (x(ExtendedKalmanFilter::ROBOT_R) + x(ExtendedKalmanFilter::DELTA_R))
                 : x(ExtendedKalmanFilter::ROBOT_R);

    h(0, ExtendedKalmanFilter::X_CENTER) = 1.0;
    h(0, ExtendedKalmanFilter::YAW) = -radius * std::sin(yaw);
    // 单面连续跟踪时，只让观测收敛当前可见装甲板。
    // 否则 x/y 残差会被错误吸收到整车半径里，逐帧把 r1/r2 推到 clamp。
    if (!visible_face_only_geometry)
    {
      h(0, ExtendedKalmanFilter::ROBOT_R) = std::cos(yaw);
    }

    h(1, ExtendedKalmanFilter::Y_CENTER) = 1.0;
    h(1, ExtendedKalmanFilter::YAW) = radius * std::cos(yaw);
    if (!visible_face_only_geometry)
    {
      h(1, ExtendedKalmanFilter::ROBOT_R) = std::sin(yaw);
    }

    h(2, ExtendedKalmanFilter::Z_ARMOR) = 1.0;
    h(3, ExtendedKalmanFilter::YAW) = 1.0;

    if (odd_face)
    {
      if (!visible_face_only_geometry)
      {
        h(0, ExtendedKalmanFilter::DELTA_R) = std::cos(yaw);
        h(1, ExtendedKalmanFilter::DELTA_R) = std::sin(yaw);
      }
      h(2, ExtendedKalmanFilter::DELTA_Z) = 1.0;
    }
    return h;
  };
  auto u_q = [this]()
  {
    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(11, 11);
    double t = time_.dt, x = cfg_.ekf.sigma2_q_xyz, y = cfg_.ekf.sigma2_q_yaw,
           r = cfg_.ekf.sigma2_q_r;
    double q_x_x = std::pow(t, 4) / 4 * x, q_x_vx = std::pow(t, 3) / 2 * x,
           q_vx_vx = std::pow(t, 2) * x;
    double q_y_y = std::pow(t, 4) / 4 * y, q_y_vy = std::pow(t, 3) / 2 * y,
           q_vy_vy = std::pow(t, 2) * y;
    double q_r = std::pow(t, 4) / 4 * r;
    const double min_radius =
        std::min(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    const double max_radius =
        std::max(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    const double geometry_span = std::max(0.01, max_radius - min_radius);
    const double q_delta_r = std::max(q_r, std::pow(geometry_span * 0.05, 2));
    const double q_delta_z = std::max(q_x_x, std::pow(geometry_span * 0.06, 2));
    q(0,0)=q_x_x;  q(0,1)=q_x_vx; q(1,0)=q_x_vx; q(1,1)=q_vx_vx;
    q(2,2)=q_x_x;  q(2,3)=q_x_vx; q(3,2)=q_x_vx; q(3,3)=q_vx_vx;
    q(4,4)=q_x_x;  q(4,5)=q_x_vx; q(5,4)=q_x_vx; q(5,5)=q_vx_vx;
    q(6,6)=q_y_y;  q(6,7)=q_y_vy; q(7,6)=q_y_vy; q(7,7)=q_vy_vy;
    q(8,8)=std::max(q_r, q_delta_r);
    q(9,9)=q_delta_r;
    q(10,10)=q_delta_z;
    return q;
  };
  auto u_r = [this](const Eigen::VectorXd& z)
  {
    Eigen::DiagonalMatrix<double, 4> r;
    const double range =
        std::max(1e-6, Eigen::Vector3d(z[0], z[1], z[2]).norm());
    const double position_sigma =
        std::max(0.01, cfg_.noise.r_xyz_factor * range);
    const double position_variance = position_sigma * position_sigma;
    r.diagonal() << position_variance, position_variance, position_variance,
        cfg_.noise.r_yaw;
    return r;
  };
  Eigen::DiagonalMatrix<double, 11> p0;
  p0.setIdentity();
  {
    const double min_radius =
        std::min(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    const double max_radius =
        std::max(cfg_.geometry.min_radius, cfg_.geometry.max_radius);
    const double geometry_span = std::max(0.01, max_radius - min_radius);
    const double delta_r_prior_sigma = std::max(0.03, geometry_span * 0.15);
    const double delta_z_prior_sigma = std::max(0.03, geometry_span * 0.20);
    p0.diagonal()(ExtendedKalmanFilter::DELTA_R) =
        delta_r_prior_sigma * delta_r_prior_sigma;
    p0.diagonal()(ExtendedKalmanFilter::DELTA_Z) =
        delta_z_prior_sigma * delta_z_prior_sigma;
  }
  ekf_.ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};

  // ---------------- Topics & 回调 ----------------
  // 装甲板识别结果订阅
  LibXR::Topic::Domain armor_detector_domain = LibXR::Topic::Domain("armor_detector");
  LibXR::Topic armors_topic = LibXR::Topic(
      LibXR::Topic::WaitTopic(ArmorTrackerArmorsTopicName(), UINT32_MAX, &armor_detector_domain));
  auto armors_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorTracker* self, LibXR::RawData& data)
      {
        if (self->params_is_changed_ == true)
        {
          self->SetConfig(self->cfg_);
          self->params_is_changed_ = false;
        }
        if constexpr (std::is_pointer<DetectionMessage>::value)
        {
          auto* message_addr = reinterpret_cast<DetectionMessage*>(data.addr_);
          self->ArmorsCallback(message_addr != nullptr ? *message_addr : nullptr);
        }
        else
        {
          auto* message_addr = reinterpret_cast<DetectionMessage*>(data.addr_);
          if (message_addr == nullptr)
          {
            XR_LOG_ERROR("ArmorTracker received empty detector message");
            return;
          }
          self->ArmorsCallback(*message_addr);
        }
      },
      this);
  armors_topic.RegisterCallback(armors_cb);

  if (const char* audit_env = std::getenv("XR_TRACKER_STATE_AUDIT_PATH"))
  {
    if (audit_env[0] != '\0')
    {
      state_audit_.path = audit_env;
    }
  }

}

/**
 * @brief 监控回调占位，当前没有周期性自检逻辑。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::OnMonitor() {}

/**
 * @brief 查询指定 detector 索引关联的图像 track ID。
 */
template <CameraTypes::CameraInfo CameraInfoV>
int ArmorTracker<CameraInfoV>::FindDetectionTrackId(std::size_t armor_index) const
{
  return image_tracker_.FindDetectionTrackId(armor_index);
}

/**
 * @brief 查询指定 detector 索引关联的图像 track 是否已确认。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorTracker<CameraInfoV>::IsDetectionTrackConfirmed(std::size_t armor_index) const
{
  return image_tracker_.IsDetectionTrackConfirmed(armor_index);
}

/**
 * @brief 用当前帧检测结果推进图像级 track 管理器并清理失效面绑定。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::UpdateImageIdTracks(const ArmorDetectorResults& armors_msg,
                                       uint64_t image_timestamp_us)
{
  image_tracker_.Update(
      armors_msg, image_timestamp_us,
      armor_tracker::ImageTrackConfig{
          .appear_hits = IdTrackAppearHits(),
          .appear_timeout_sec = IdTrackAppearTimeoutSec(),
          .tentative_misses = IdTrackTentativeMisses(),
          .tentative_timeout_sec = IdTrackTentativeTimeoutSec(),
          .disappear_misses = IdTrackDisappearMisses(),
          .disappear_timeout_sec = IdTrackDisappearTimeoutSec(),
      });

  if (rt_.tracked_face_track_id_valid)
  {
    bool found_bound_track = false;
    for (const auto& track : image_tracker_.Tracks())
    {
      if (track.active && track.track_id == rt_.tracked_face_track_id)
      {
        found_bound_track = true;
        break;
      }
    }
    if (!found_bound_track)
    {
      XR_LOG_DEBUG("Tracker face bind cleared: tracked_id=%d",
                   static_cast<int>(rt_.tracked_face_track_id));
      rt_.tracked_face_track_id_valid = false;
    }
  }

  const int bound_face_count = std::max(1, static_cast<int>(rt_.tracked_armors_num));
  for (int face_index = 0; face_index < 4; ++face_index)
  {
    if (!rt_.face_track_id_valid[face_index])
    {
      continue;
    }
    bool found_bound_track = false;
    for (const auto& track : image_tracker_.Tracks())
    {
      if (track.active && track.track_id == rt_.face_track_id[face_index])
      {
        found_bound_track = true;
        break;
      }
    }
    if (!found_bound_track)
    {
      rt_.face_track_id_valid[face_index] = false;
    }
  }
  if (rt_.tracked_face_track_id_valid)
  {
    rt_.face_track_id_valid[0] = true;
    rt_.face_track_id[0] = rt_.tracked_face_track_id;
  }
  else
  {
    rt_.face_track_id_valid[0] = false;
  }
  for (int face_index = bound_face_count; face_index < 4; ++face_index)
  {
    rt_.face_track_id_valid[face_index] = false;
  }
}


/**
 * @brief 构造单装甲模式下的候选调试消息。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::FillSingleArmorDebug(std::size_t selected_index,
                                                     int detection_track_id,
                                                     bool confirmed_track,
                                                     float score,
                                                     float center_diff,
                                                     float area_log)
{
  CandidateDebugMsg debug{};
  candidate_debug_msg_.matched =
      (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST) ? 1 : 0;
  debug.matched =
      (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST) ? 1 : 0;
  debug.count = 1;
  debug.selected_index = 0;
  debug.detection_count = 1;
  debug.tracked_armors_num = 1;
  debug.tracked_face_track_id_valid = confirmed_track ? 1 : 0;
  debug.tracked_face_track_id =
      confirmed_track && detection_track_id >= 0
          ? static_cast<int16_t>(detection_track_id)
          : static_cast<int16_t>(-1);
  debug.best_same_face_score = score;
  debug.same_face_matched = debug.matched;
  debug.switch_face_matched = 0;
  debug.switch_allowed = 0;
  debug.detection_track_ids.fill(static_cast<int16_t>(-1));
  debug.detection_track_confirmed.fill(static_cast<uint8_t>(0));
  if (selected_index < CandidateDebugMsg::kMaxDetections)
  {
    debug.detection_track_ids[selected_index] =
        static_cast<int16_t>(detection_track_id);
    debug.detection_track_confirmed[selected_index] =
        confirmed_track ? 1 : 0;
  }

  auto& item = debug.items[0];
  item.armor_index = static_cast<uint8_t>(selected_index);
  item.face_index = 0;
  item.same_number = 1;
  item.image_track_id = static_cast<int16_t>(detection_track_id);
  item.image_track_confirmed = confirmed_track ? 1 : 0;
  item.same_persistent_track = confirmed_track ? 1 : 0;
  item.number = rt_.tracked_armor.number;
  item.type = rt_.tracked_armor.type;
  item.score = score;
  item.position_diff = 0.0f;
  item.yaw_diff = 0.0f;
  item.view_bonus = 0.0f;
  item.area_score = static_cast<float>(armor_tracker::ArmorImageArea(rt_.tracked_armor));
  item.frontality = 0.0f;
  item.observation_quality_penalty = static_cast<float>(
      ObservationQualityEnabled()
          ? armor_tracker::ArmorObservationQualityPenalty(
                rt_.tracked_armor, ObservationStableMaxReprojectionPx(),
                ObservationStableMinAreaPx(), ObservationStableMinConfidence())
          : 0.0);
  item.center_x = rt_.tracked_armor.center.x;
  item.center_y = rt_.tracked_armor.center.y;
  item.predicted_yaw = static_cast<float>(rt_.last_yaw);
  item.measured_yaw = static_cast<float>(rt_.last_yaw);
  debug.relaxed_same_face_distance = center_diff;
  debug.relaxed_face_switch_distance = area_log;
  candidate_debug_msg_ = debug;
}

/**
 * @brief 单装甲模式下按身份、图像连续性、面积和距离选择观测。
 */
template <CameraTypes::CameraInfo CameraInfoV>
std::optional<ArmorDetectorResult> ArmorTracker<CameraInfoV>::SelectSingleArmorObservation(
    const ArmorDetectorResults& armors_msg, uint64_t image_timestamp_us,
    std::size_t& selected_index, int& detection_track_id,
    bool& confirmed_track, float& selected_center_diff, float& selected_area_log,
    float& selected_score)
{
  (void)image_timestamp_us;
  selected_index = 0;
  detection_track_id = -1;
  confirmed_track = false;
  selected_center_diff = 0.0f;
  selected_area_log = 0.0f;
  selected_score = 0.0f;
  /**
   * @brief 单装甲模式候选观测。
   */
  struct Candidate
  {
    std::size_t armor_index = 0;       ///< detector 结果索引。
    int detection_track_id = -1;       ///< 图像 track ID。
    bool confirmed_track = false;      ///< 图像 track 是否已确认。
    float score = 0.0f;                ///< 候选分，越小越好。
    float center_diff = 0.0f;          ///< 与上一帧中心的像素差。
    float area_log = 0.0f;             ///< 面积比例的对数差。
    ArmorDetectorResult armor{};       ///< detector 装甲结果。
  };

  bool has_candidate = false;
  Candidate best{};
  const double tracked_area =
      std::max(1.0, armor_tracker::ArmorImageArea(rt_.tracked_armor));
  const bool have_tracked_armor = rt_.tracked_id != ArmorNumber::INVALID;
  const bool have_tracked_pose =
      have_tracked_armor && (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST);
  const double prev_range =
      have_tracked_pose
          ? std::sqrt(std::pow(rt_.tracked_armor.pose.translation.x(), 2.0) +
                      std::pow(rt_.tracked_armor.pose.translation.y(), 2.0) +
                      std::pow(rt_.tracked_armor.pose.translation.z(), 2.0))
          : 0.0;
  for (std::size_t armor_index = 0; armor_index < armors_msg.size(); ++armor_index)
  {
    const auto& armor = armors_msg[armor_index];
    const int detection_track_id = FindDetectionTrackId(armor_index);
    const bool confirmed_track = IsDetectionTrackConfirmed(armor_index);

    float score = 0.0f;
    const bool same_confirmed_track =
        rt_.tracked_face_track_id_valid && confirmed_track &&
        detection_track_id >= 0 &&
        static_cast<uint16_t>(detection_track_id) == rt_.tracked_face_track_id;
    if (same_confirmed_track)
    {
      score -= 8.0f;
    }
    else if (have_tracked_armor && armor.number == rt_.tracked_id &&
             armor.number != ArmorNumber::INVALID)
    {
      score -= 4.0f;
    }

    if (armor.type == rt_.tracked_armor.type &&
        armor.type != ArmorType::INVALID)
    {
      score -= 1.0f;
    }

    const float center_diff = have_tracked_armor
                                  ? static_cast<float>(std::hypot(
                                        armor.center.x - rt_.tracked_armor.center.x,
                                        armor.center.y - rt_.tracked_armor.center.y))
                                  : static_cast<float>(armor.distance_to_image_center);
    const float area_log = have_tracked_armor
                               ? static_cast<float>(std::abs(std::log(
                                     std::max(1.0, armor_tracker::ArmorImageArea(armor)) /
                                     tracked_area)))
                               : 0.0f;
    const double range =
        std::sqrt(std::pow(armor.pose.translation.x(), 2.0) +
                  std::pow(armor.pose.translation.y(), 2.0) +
                  std::pow(armor.pose.translation.z(), 2.0));
    const float range_diff =
        have_tracked_pose ? static_cast<float>(std::abs(range - prev_range)) : 0.0f;

    score += 0.012f * center_diff;
    score += 0.35f * area_log;
    score += 1.40f * range_diff;
    score -= 0.45f * armor.confidence;
    score += 0.0015f *
             static_cast<float>(armor.distance_to_image_center);
    if (ObservationQualityEnabled())
    {
      score += static_cast<float>(
          ObservationQualityScoreWeight() *
          armor_tracker::ArmorObservationQualityPenalty(
              armor, ObservationStableMaxReprojectionPx(),
              ObservationStableMinAreaPx(), ObservationStableMinConfidence()));
      if (armor_tracker::StableArmorObservation(
              armor, ObservationStableMaxReprojectionPx(),
              ObservationStableMinAreaPx(), ObservationStableMinConfidence()))
      {
        score -= 0.12f;
      }
      if (confirmed_track)
      {
        score -= static_cast<float>(ObservationConfirmedTrackBonus());
      }
    }

    if (rt_.state == State::TEMP_LOST)
    {
      const bool same_label =
          have_tracked_armor && armor.number == rt_.tracked_id &&
          armor.type == rt_.tracked_armor.type &&
          armor.number != ArmorNumber::INVALID;
      if (!(same_confirmed_track ||
            (same_label && center_diff <= 120.0f && area_log <= 0.50f)))
      {
        continue;
      }
    }
    if (have_tracked_armor && same_confirmed_track &&
        (rt_.state == State::TRACKING || rt_.state == State::TEMP_LOST))
    {
      // 同一条已确认的 image-track 在连续小像素位移下仍可能触发平面
      // PnP 的反解 yaw；默认沿用 PnP yaw，仅在实验开关下折叠 ±pi。
      const double yaw_now =
          MatchYawAllowPiAmbiguityEnabled()
              ? armor_tracker::MeasuredArmorYawNearAllowPi(armor, rt_.last_yaw)
              : armor_tracker::MeasuredArmorYawNear(armor, rt_.last_yaw);
      const double yaw_jump =
          armor_tracker::AngularDiffAbs(yaw_now, rt_.last_yaw);
      const bool suspicious_pose_jump =
          center_diff < 6.0f &&
          yaw_jump > 1.20 &&
          std::abs(armor.pose.translation.z() - rt_.tracked_armor.pose.translation.z()) > 0.12;
      if (suspicious_pose_jump)
      {
        XR_LOG_DEBUG(
            "SingleArmor reject pose jump: idx=%u track=%d yaw_jump=%.3f center_diff=%.1f area_log=%.3f prev=(%.3f,%.3f,%.3f) now=(%.3f,%.3f,%.3f)",
            static_cast<unsigned>(armor_index), detection_track_id, yaw_jump,
            static_cast<double>(center_diff), static_cast<double>(area_log), rt_.tracked_armor.pose.translation.x(),
            rt_.tracked_armor.pose.translation.y(), rt_.tracked_armor.pose.translation.z(),
            armor.pose.translation.x(), armor.pose.translation.y(),
            armor.pose.translation.z());
        continue;
      }
    }

    if (!has_candidate || score < best.score)
    {
      has_candidate = true;
      best.armor_index = armor_index;
      best.detection_track_id = detection_track_id;
      best.confirmed_track = confirmed_track;
      best.score = score;
      best.center_diff = center_diff;
      best.area_log = area_log;
      best.armor = armor;
    }
  }
  if (!has_candidate)
  {
    return std::nullopt;
  }

  selected_index = best.armor_index;
  detection_track_id = best.detection_track_id;
  confirmed_track = best.confirmed_track;
  selected_center_diff = best.center_diff;
  selected_area_log = best.area_log;
  selected_score = best.score;
  return best.armor;
}

/**
 * @brief 用一个装甲观测初始化 11 维整车模型 EKF 状态和协方差。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::InitEKF(const ArmorDetectorResult& a)
{
  const Eigen::Vector3d xyz(a.pose.translation.x(), a.pose.translation.y(),
                            a.pose.translation.z());
  rt_.last_yaw = 0.0;
  const double yaw = VehicleDetectorYawNear(a, 0.0);
  const double radius = VehicleInitialRadiusFor(a);
  const double center_x = xyz.x() - radius * std::cos(yaw);
  const double center_y = xyz.y() - radius * std::sin(yaw);
  const double center_z = xyz.z();

  ekf_.state = Eigen::VectorXd::Zero(11);
  ekf_.state << center_x, 0.0, center_y, 0.0, center_z, 0.0, yaw, 0.0,
      radius, 0.0, 0.0;
  if (VehicleArmorCountFor(a) == 4 && VehicleStaticDeltaZEnabled())
  {
    ekf_.state(ExtendedKalmanFilter::DELTA_Z) = VehicleStaticDeltaZ();
  }
  ekf_.covariance = VehicleInitialP0DiagFor(a).asDiagonal();
  ekf_.measurement_face_index = 0;
  ekf_.measurement =
      Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);
  rt_.last_yaw = yaw;
  rt_.tracked_armors_num = static_cast<ArmorsNum>(VehicleArmorCountFor(a));
  rt_.tracked_face_index = 0;
  rt_.another_r = radius;
  rt_.dz = ekf_.state(ExtendedKalmanFilter::DELTA_Z);
  rt_.dz_abs_ref = std::abs(rt_.dz);
  rt_.face_switch_cooldown_remaining = 0.0;
  rt_.model_pair_delta_z_valid = false;
  ekf_.ekf.SetState(ekf_.state);
}


/**
 * @brief 更新运行配置，并同步状态机阈值等即时参数。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorTracker<CameraInfoV>::SetConfig(const Config& cfg)
{
  if (cfg.thresholds.tracking_thres != rt_.tracking_thres)
  {
    rt_.tracking_thres = cfg.thresholds.tracking_thres;
  }
  cfg_ = cfg;
  xr_tracker_.Configure(BuildXrTrackerConfig());
  preview_.Stop();
  preview_.Start(cfg_.preview);
}

/**
 * @brief RamFS 命令实现，支持 show 和少量运行时参数修改。
 */
template <CameraTypes::CameraInfo CameraInfoV>
int ArmorTracker<CameraInfoV>::CommandFun(ArmorTracker<CameraInfoV>* self, int argc, char** argv)
{
  if (argc == 1)
  {
    XR_TRACKER_STDIO_PRINT("ArmorTracker\n\n");
    XR_TRACKER_STDIO_PRINT("Usage\r\n");
    XR_TRACKER_STDIO_PRINT("  show\r\n");
    XR_TRACKER_STDIO_PRINT("  max_armor_distance <value>\r\n");
    XR_TRACKER_STDIO_PRINT("  max_z_position <value>\r\n");
    XR_TRACKER_STDIO_PRINT("  max_match_distance <value>\r\n");
    XR_TRACKER_STDIO_PRINT("  max_match_yaw_diff <value>\r\n");
    XR_TRACKER_STDIO_PRINT("  tracking_thres <value>\r\n");
    XR_TRACKER_STDIO_PRINT("  sigma2_q_xyz <value>\r\n");
    XR_TRACKER_STDIO_PRINT("  sigma2_q_yaw <value>\r\n");
    XR_TRACKER_STDIO_PRINT("  sigma2_q_r <value>\r\n");
    XR_TRACKER_STDIO_PRINT("  r_xyz_factor <value>\r\n");
    XR_TRACKER_STDIO_PRINT("  r_yaw <value>\r\n");
    return 0;
  }
  else if (argc == 2)
  {
    std::string cmd = argv[1];
    if (cmd == "show")
    {
      // clang-format off
      XR_TRACKER_STDIO_PRINT("name: ArmorTracker\r\n");
      XR_TRACKER_STDIO_PRINT("cfg:\r\n");
      XR_TRACKER_STDIO_PRINT("  limits:\r\n");
      XR_TRACKER_STDIO_PRINTF("    max_armor_distance: %f\r\n", self->cfg_.limits.max_armor_distance);
      XR_TRACKER_STDIO_PRINTF("    max_z_position: %f\r\n", self->cfg_.limits.max_z_position);
      XR_TRACKER_STDIO_PRINT("  match:\r\n");
      XR_TRACKER_STDIO_PRINTF("    max_match_distance: %f\r\n", self->cfg_.match.max_match_distance);
      XR_TRACKER_STDIO_PRINTF("    max_match_yaw_diff: %f\r\n", self->cfg_.match.max_match_yaw_diff);
      XR_TRACKER_STDIO_PRINT("  thresholds:\r\n");
      XR_TRACKER_STDIO_PRINTF("    tracking_thres: %d\r\n", self->cfg_.thresholds.tracking_thres);
      XR_TRACKER_STDIO_PRINTF("    lost_time_thres: %f\r\n", self->cfg_.thresholds.lost_time_thres);
      XR_TRACKER_STDIO_PRINT("  ekf:\r\n");
      XR_TRACKER_STDIO_PRINTF("    sigma2_q_xyz: %f\r\n", self->cfg_.ekf.sigma2_q_xyz);
      XR_TRACKER_STDIO_PRINTF("    sigma2_q_yaw: %f\r\n", self->cfg_.ekf.sigma2_q_yaw);
      XR_TRACKER_STDIO_PRINTF("    sigma2_q_r: %f\r\n", self->cfg_.ekf.sigma2_q_r);
      XR_TRACKER_STDIO_PRINT("  geometry:\r\n");
      XR_TRACKER_STDIO_PRINTF("    initial_radius: %f\r\n", self->cfg_.geometry.initial_radius);
      XR_TRACKER_STDIO_PRINTF("    min_radius: %f\r\n", self->cfg_.geometry.min_radius);
      XR_TRACKER_STDIO_PRINTF("    max_radius: %f\r\n", self->cfg_.geometry.max_radius);
      XR_TRACKER_STDIO_PRINT("  noise:\r\n");
      XR_TRACKER_STDIO_PRINTF("    r_xyz_factor: %f\r\n", self->cfg_.noise.r_xyz_factor);
      XR_TRACKER_STDIO_PRINTF("    r_yaw: %f\r\n", self->cfg_.noise.r_yaw);
      XR_TRACKER_STDIO_PRINT("  frames:\r\n");
      XR_TRACKER_STDIO_PRINT("    rotation:\r\n");
      XR_TRACKER_STDIO_PRINTF("      - %f\r\n", self->cfg_.frames.rotation[0]);
      XR_TRACKER_STDIO_PRINTF("      - %f\r\n", self->cfg_.frames.rotation[1]);
      XR_TRACKER_STDIO_PRINTF("      - %f\r\n", self->cfg_.frames.rotation[2]);
      XR_TRACKER_STDIO_PRINTF("      - %f\r\n", self->cfg_.frames.rotation[3]);
      XR_TRACKER_STDIO_PRINT("    translation:\r\n");
      XR_TRACKER_STDIO_PRINTF("      - %f\r\n", self->cfg_.frames.translation[0]);
      XR_TRACKER_STDIO_PRINTF("      - %f\r\n", self->cfg_.frames.translation[1]);
      XR_TRACKER_STDIO_PRINTF("      - %f\r\n", self->cfg_.frames.translation[2]);
      XR_TRACKER_STDIO_PRINT("  model:\r\n");
      XR_TRACKER_STDIO_PRINTF("    enable_pair_dz: %d\r\n", self->cfg_.model.enable_pair_dz ? 1 : 0);
      XR_TRACKER_STDIO_PRINTF("    measurement_recenter_alpha: %f\r\n", self->cfg_.model.measurement_recenter_alpha);
      XR_TRACKER_STDIO_PRINTF("    quality_recenter: %d\r\n", self->cfg_.model.quality_recenter ? 1 : 0);
      XR_TRACKER_STDIO_PRINTF("    enable_pair_geometry: %d\r\n", self->cfg_.model.enable_pair_geometry ? 1 : 0);
      XR_TRACKER_STDIO_PRINTF("    enable_output_meas_anchor: %d\r\n", self->cfg_.model.enable_output_meas_anchor ? 1 : 0);
      XR_TRACKER_STDIO_PRINTF("    enable_aimer_meas_anchor: %d\r\n", self->cfg_.model.enable_aimer_meas_anchor ? 1 : 0);
      XR_TRACKER_STDIO_PRINTF("    enable_fixed_pose_yaw_opt: %d\r\n", self->cfg_.model.enable_fixed_pose_yaw_opt ? 1 : 0);
      // clang-format on
    }
    return 0;
  }
  else if (argc == 3)
  {
    std::string cmd = argv[1];
    if (cmd == "max_armor_distance")
    {
      self->cfg_.limits.max_armor_distance = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_z_position")
    {
      self->cfg_.limits.max_z_position = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_match_distance")
    {
      self->cfg_.match.max_match_distance = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_match_yaw_diff")
    {
      self->cfg_.match.max_match_yaw_diff = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "tracking_thres")
    {
      self->cfg_.thresholds.tracking_thres = std::stoi(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "sigma2_q_xyz")
    {
      self->cfg_.ekf.sigma2_q_xyz = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "sigma2_q_yaw")
    {
      self->cfg_.ekf.sigma2_q_yaw = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "sigma2_q_r")
    {
      self->cfg_.ekf.sigma2_q_r = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "r_xyz_factor")
    {
      self->cfg_.noise.r_xyz_factor = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "r_yaw")
    {
      self->cfg_.noise.r_yaw = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else
    {
      XR_TRACKER_STDIO_PRINTF("Unknown command: %s\n", argv[1]);
      return -1;
    }
    return 0;
  }
  XR_TRACKER_STDIO_PRINTF("Unknown command: %s\n", argv[1]);
  return -1;
}
