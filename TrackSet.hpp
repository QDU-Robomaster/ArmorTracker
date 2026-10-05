#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

#include "AutoAimTypes.hpp"
#include "FallbackTarget.hpp"
#include "VehicleEstimator.hpp"

/// 跟踪槽的状态 / State of a track slot.
enum class TrackState : uint8_t
{
  LOST,
  DETECTING,  ///< 刚看到，还没连续看够 / Seen, not yet often enough
  TRACKING,
  TEMP_LOST,  ///< 暂时没看到 / Briefly unseen
};

/// 选目标打分的权重与归一化 / Weights and scales of target selection.
struct SelectWeights
{
  double observed_count_weight = 1.6;
  double distance_weight = 2.0;
  double area_weight = 1.2;
  double spin_weight = 0.8;
  double angle_weight = 2.0;
  double max_distance_m = 8.0;
  double distance_span_m = 7.5;
  double area_norm_px = 24000.0;  ///< 原生像素 / Native pixels
  double observed_count_norm = 4.0;
  double max_spin_rad_s = 8.0;
  double max_angle_rad = 0.3;  ///< 偏离光轴的角度 / Angle off the optical axis
  double detecting_scale = 0.55;
  double temp_lost_scale = 0.35;
  double switch_margin = 0.25;
};

/// 跟踪设置，与 YAML 一一对应 / Tracker settings, one-to-one with the YAML.
struct TrackerSettings
{
  std::string_view camera_name;
  std::array<double, 4> mount_rotation_wxyz;  ///< 相机安装到云台本体 / Camera mount
  std::array<double, 3> mount_translation;    ///< m
  int target_number;                          ///< 只打该编号，-1 为不限 / -1 = any
  int min_detect_count;                       ///< 2
  int max_temp_lost;                          ///< 帧 / frames, 15
  int outpost_max_temp_lost;                  ///< 帧 / frames, 75
  double latency_s;     ///< 帧到命中的固定延迟，整车估计器的速率时域 / Fixed latency
  double bullet_speed;  ///< m/s，同上 / for the rate horizon
  SelectWeights select;
};

/**
 * @brief 多目标管理：每个编号一个槽，四块板的整车用 VehicleEstimator，前哨站、基地、
 *        平衡步兵用兜底 EKF；槽按状态机推进，按得分选出要打的目标。
 *        Multi-target management: one slot per number; four-plate vehicles use the
 *        VehicleEstimator, outpost, base and balance infantry the fallback EKF. Slots
 *        follow a state machine and the best-scoring one is the target.
 */
class TrackSet
{
 public:
  static constexpr int SLOTS = 8;  ///< ONE … BASE

  explicit TrackSet(const TrackerSettings& s) : s_(s) {}

  /**
   * @param t_us 帧的 IMU 时间戳 / IMU timestamp of the frame
   * @param q_wxyz 云台本体系到世界系 / Gimbal body-to-world attitude
   * @param armors 检测（角点为原生像素，左上、左下、右下、右上）/ Detections in native
   *               pixels, ordered top-left, bottom-left, bottom-right, top-right
   */
  ArmorTrackerTarget Step(uint64_t t_us, const std::array<float, 4>& q_wxyz,
                          const CameraTypes::CameraCalibration& calibration,
                          const std::vector<AutoAim::Armor>& armors)
  {
    if (calibration_ != &calibration)
    {
      Configure(calibration);
    }
    if (has_base_ && t_us < last_t_us_)
    {
      ResetAll();  // 时间倒退：重新开始 / Time went backwards: start over
    }
    if (!has_base_)
    {
      has_base_ = true;
      base_t_us_ = t_us;
    }
    last_t_us_ = t_us;
    const double t = static_cast<double>(t_us - base_t_us_) * 1e-6;
    q_ = {q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]};
    r_bw_ = Vehicle::RotationFromQuaternion(q_[0], q_[1], q_[2], q_[3]);
    solver_->SetAttitude(r_bw_);

    std::array<std::vector<const AutoAim::Armor*>, SLOTS> by_number;
    for (const AutoAim::Armor& a : armors)
    {
      const int n = static_cast<int>(a.number);
      if (n >= 0 && n < SLOTS)
      {
        by_number[n].push_back(&a);
      }
    }
    for (int n = 0; n < SLOTS; ++n)
    {
      UpdateSlot(slots_[n], static_cast<ArmorNumber>(n), by_number[n], t);
    }
    selected_ = Select();

    ArmorTrackerTarget out{};
    out.image_timestamp_us = t_us;
    out.id = ArmorNumber::INVALID;
    if (selected_ >= 0)
    {
      Fill(slots_[selected_], static_cast<ArmorNumber>(selected_), out);
    }
    return out;
  }

  /// 世界系到相机光学系的旋转与平移（行优先），供预览投影 / World-to-optical
  /// transform (row-major) for preview projection.
  void WorldToCamera(std::array<double, 9>& rotation, std::array<double, 3>& translation) const
  {
    // p_c = R_cbᵀ (R_bwᵀ p_w − t_cb)
    const Vehicle::Mat3 r = camera_.R_cb.transpose() * r_bw_.transpose();
    const Vehicle::Vec3 t = -(camera_.R_cb.transpose() * camera_.t_cb);
    for (int i = 0; i < 3; ++i)
    {
      translation[i] = t(i);
      for (int j = 0; j < 3; ++j)
      {
        rotation[3 * i + j] = r(i, j);
      }
    }
  }

 private:
  struct Slot
  {
    TrackState state = TrackState::LOST;
    bool large = false;
    std::optional<Vehicle::VehicleEstimator> vehicle;
    Vehicle::VehicleTarget vehicle_target;
    std::optional<Fallback::FallbackTarget> fallback;
    int detect_count = 0;
    int temp_lost = 0;
    double count_lpf = 0.0;
    double area = 0.0;
    double view_angle = 1.0;
    double score = -std::numeric_limits<double>::infinity();
    double last_t = -1.0;
    double last_seen = -1.0;
    bool outpost_hint_valid = false;  ///< 跨重置保留 / Kept across resets
    Vehicle::Vec3 outpost_hint = Vehicle::Vec3::Zero();

    bool Initialized() const { return vehicle.has_value() || fallback.has_value(); }
  };

  void Configure(const CameraTypes::CameraCalibration& c)
  {
    calibration_ = &c;
    camera_.fx = c.fx;
    camera_.fy = c.fy;
    camera_.cx = c.cx;
    camera_.cy = c.cy;
    camera_.dist = c.distortion;
    const auto& m = s_.mount_rotation_wxyz;
    camera_.R_cb = Vehicle::RotationFromQuaternion(m[0], m[1], m[2], m[3]) *
                   Vehicle::OpticalToBody();
    camera_.t_cb = {s_.mount_translation[0], s_.mount_translation[1], s_.mount_translation[2]};
    solver_.emplace(camera_);
    ResetAll();
  }

  void ResetAll()
  {
    for (Slot& slot : slots_)
    {
      Reset(slot);
      slot.outpost_hint_valid = false;
    }
    has_base_ = false;
    selected_ = -1;
  }

  /// 清空槽（时间倒退或兜底目标健康检查失败）/ Clear a slot (time went backwards or
  /// the fallback target failed its health check).
  static void Reset(Slot& slot)
  {
    slot.state = TrackState::LOST;
    slot.vehicle.reset();
    slot.vehicle_target = {};
    slot.fallback.reset();
    slot.detect_count = 0;
    slot.temp_lost = 0;
    slot.score = -std::numeric_limits<double>::infinity();
  }

  /// 1 号与基地只有大板 / Number one and the base carry only large plates.
  static bool Large(const AutoAim::Armor& a)
  {
    return a.type == ArmorType::LARGE || a.number == ArmorNumber::ONE ||
           a.number == ArmorNumber::BASE;
  }

  /// 兜底目标的种类；整车估计器能处理的返回空 / Fallback kind, or none for the vehicle
  /// estimator.
  static std::optional<Fallback::Kind> FallbackKind(ArmorNumber n, bool large)
  {
    if (n == ArmorNumber::OUTPOST)
    {
      return Fallback::Kind::OUTPOST;
    }
    if (n == ArmorNumber::BASE)
    {
      return Fallback::Kind::BASE;
    }
    if (large && (n == ArmorNumber::THREE || n == ArmorNumber::FOUR || n == ArmorNumber::FIVE))
    {
      return Fallback::Kind::BALANCE;  // 两块大板的平衡步兵 / Two-plate balance robot
    }
    return std::nullopt;
  }

  /// AutoAim 角点顺序（左上、左下、右下、右上）转为估计器顺序（左上、右上、右下、左下）。
  /// AutoAim corner order to the estimators' order.
  static std::array<cv::Point2f, 4> EstimatorCorners(const AutoAim::Armor& a)
  {
    const auto p = [&a](int k) { return cv::Point2f(a.corners[k].x, a.corners[k].y); };
    return {p(0), p(3), p(2), p(1)};
  }

  void UpdateMetrics(Slot& slot, const std::vector<const AutoAim::Armor*>& dets) const
  {
    slot.count_lpf = 0.8 * slot.count_lpf + 0.2 * static_cast<double>(dets.size());
    slot.area = 0.0;
    double min_angle = std::numeric_limits<double>::infinity();
    for (const AutoAim::Armor* a : dets)
    {
      const auto c = EstimatorCorners(*a);
      slot.area += std::abs(cv::contourArea(std::vector<cv::Point2f>(c.begin(), c.end())));
      const cv::Point2f centre = (c[0] + c[1] + c[2] + c[3]) * 0.25F;
      min_angle = std::min(min_angle, std::atan(std::hypot((centre.x - camera_.cx) / camera_.fx,
                                                           (centre.y - camera_.cy) / camera_.fy)));
    }
    if (std::isfinite(min_angle))
    {
      slot.view_angle = min_angle;
    }
  }

  void UpdateSlot(Slot& slot, ArmorNumber n, const std::vector<const AutoAim::Armor*>& dets,
                  double t)
  {
    UpdateMetrics(slot, dets);
    if (slot.state != TrackState::LOST && t - slot.last_t > 0.1)
    {
      slot.state = TrackState::LOST;  // 太久没处理 / Too long since the last frame
    }
    slot.last_t = t;
    if (!slot.Initialized() && !dets.empty())
    {
      slot.large = Large(*dets.front());
    }
    std::vector<const AutoAim::Armor*> same_size;
    for (const AutoAim::Armor* a : dets)
    {
      if (Large(*a) == slot.large)
      {
        same_size.push_back(a);
      }
    }
    const auto kind = FallbackKind(n, slot.large);
    const bool found = kind ? UpdateFallback(slot, *kind, same_size, t)
                            : UpdateVehicle(slot, same_size, t);
    Advance(slot, found);
    // 状态机只决定能否被选为目标；整车估计器连续 2 s 没看到才丢弃，丢失后由它自己重新起步。
    // The state machine only decides selectability; a vehicle estimator is dropped after
    // 2 s unseen and otherwise reboots itself after a loss.
    if (slot.vehicle && t - slot.last_seen > 2.0)
    {
      slot.vehicle.reset();
      slot.vehicle_target = {};
    }
    if (slot.fallback && slot.fallback->GetKind() == Fallback::Kind::OUTPOST)
    {
      slot.outpost_hint = slot.fallback->Centre();
      slot.outpost_hint_valid = true;
    }
    slot.score = Score(slot);
  }

  bool UpdateVehicle(Slot& slot, const std::vector<const AutoAim::Armor*>& dets, double t)
  {
    if (!slot.vehicle)
    {
      if (dets.empty())
      {
        return false;
      }
      slot.vehicle.emplace(camera_);
    }
    std::vector<Vehicle::Detection> input;
    for (const AutoAim::Armor* a : dets)
    {
      const auto c = EstimatorCorners(*a);
      Vehicle::Detection d;
      for (int k = 0; k < 4; ++k)
      {
        d.corners[k] = {c[k].x, c[k].y};
      }
      d.type = slot.large ? 1 : 0;
      input.push_back(d);
    }
    // 速率取帧到命中的平均 / Rates averaged over frame-to-impact.
    const double distance =
        slot.vehicle_target.tracking ? slot.vehicle_target.position.head<2>().norm() : 5.0;
    const double horizon = s_.latency_s + distance / std::max(s_.bullet_speed, 1.0);
    slot.vehicle_target = slot.vehicle->Step(t, q_, input, horizon);
    if (!dets.empty())
    {
      slot.last_seen = t;
    }
    return !dets.empty() && slot.vehicle_target.tracking;
  }

  bool UpdateFallback(Slot& slot, Fallback::Kind kind,
                      const std::vector<const AutoAim::Armor*>& dets, double t)
  {
    if (slot.fallback && slot.fallback->HealthFailed())
    {
      slot.fallback.reset();
    }
    bool found = false;
    if (!slot.fallback)
    {
      found = StartFallback(slot, kind, dets, t);
    }
    else if (!dets.empty())
    {
      slot.fallback->Predict(t);
      for (const AutoAim::Armor* a : dets)
      {
        Fallback::PlateObservation obs;
        if (solver_->Solve(EstimatorCorners(*a), slot.large, kind, obs))
        {
          slot.fallback->Update(obs);
          found = true;
        }
      }
    }
    else if (slot.state != TrackState::LOST)
    {
      slot.fallback->Predict(t);
    }
    if (slot.fallback && slot.fallback->HealthFailed())
    {
      slot.fallback.reset();
      found = StartFallback(slot, kind, dets, t);
    }
    return found;
  }

  /// 用最靠近主点的检测起始 / Start from the detection nearest the principal point.
  bool StartFallback(Slot& slot, Fallback::Kind kind,
                     const std::vector<const AutoAim::Armor*>& dets, double t)
  {
    const AutoAim::Armor* nearest = nullptr;
    double best = std::numeric_limits<double>::infinity();
    for (const AutoAim::Armor* a : dets)
    {
      const auto c = EstimatorCorners(*a);
      const cv::Point2f centre = (c[0] + c[1] + c[2] + c[3]) * 0.25F;
      const double d = std::hypot(centre.x - camera_.cx, centre.y - camera_.cy);
      if (d < best)
      {
        best = d;
        nearest = a;
      }
    }
    Fallback::PlateObservation obs;
    if (nearest == nullptr || !solver_->Solve(EstimatorCorners(*nearest), slot.large, kind, obs))
    {
      return false;
    }
    Fallback::FallbackTarget::Start start;
    if (kind == Fallback::Kind::OUTPOST && slot.outpost_hint_valid)
    {
      start = OutpostStart(slot.outpost_hint, obs);
    }
    slot.fallback.emplace(kind, obs, t, start);
    return true;
  }

  /// 前哨站重新起始：按上次的中心推断看到的是哪块板、高度相位 / Restarting the outpost:
  /// infer the observed face and the height phase from the previous centre.
  static Fallback::FallbackTarget::Start OutpostStart(const Vehicle::Vec3& hint,
                                                      const Fallback::PlateObservation& obs)
  {
    using Fallback::LimitRad;
    Fallback::FallbackTarget::Start start;
    Vehicle::Vec3 to_centre = hint - obs.xyz;
    to_centre.z() = 0.0;
    if (to_centre.head<2>().norm() >= 1e-6)
    {
      const double face_yaw = Fallback::BearingYaw(to_centre);
      const double observed = Fallback::OutpostObservedYaw(obs.yaw);
      double best = std::numeric_limits<double>::infinity();
      for (int id = 0; id < 3; ++id)
      {
        const double error = std::abs(LimitRad(face_yaw - LimitRad(observed + id * 2.0 * Fallback::PI / 3.0)));
        if (error < best)
        {
          best = error;
          start.face = id;
        }
      }
    }
    double best = std::numeric_limits<double>::infinity();
    for (int phase = 0; phase < 3; ++phase)
    {
      const double error =
          std::abs(obs.xyz.z() - Fallback::OutpostHeightOffset(start.face, phase) - hint.z());
      if (error < best)
      {
        best = error;
        start.height_phase = phase;
      }
    }
    constexpr double CENTRE_HINT_GATE = 0.06;
    start.height_phase_valid = best <= CENTRE_HINT_GATE;
    start.centre_hint_valid = start.height_phase_valid;
    start.centre_hint = hint;
    return start;
  }

  void Advance(Slot& slot, bool found) const
  {
    switch (slot.state)
    {
      case TrackState::LOST:
        if (found)
        {
          slot.state = TrackState::DETECTING;
          slot.detect_count = 1;
        }
        break;
      case TrackState::DETECTING:
        if (!found)
        {
          slot.state = TrackState::LOST;
        }
        else if (++slot.detect_count >= s_.min_detect_count)
        {
          slot.state = TrackState::TRACKING;
        }
        break;
      case TrackState::TRACKING:
        if (!found)
        {
          slot.state = TrackState::TEMP_LOST;
          slot.temp_lost = 1;
        }
        break;
      case TrackState::TEMP_LOST:
      {
        if (found)
        {
          slot.state = TrackState::TRACKING;
          break;
        }
        const bool outpost =
            slot.fallback && slot.fallback->GetKind() == Fallback::Kind::OUTPOST;
        if (++slot.temp_lost > (outpost ? s_.outpost_max_temp_lost : s_.max_temp_lost))
        {
          slot.state = TrackState::LOST;
        }
        break;
      }
    }
  }

  double Score(const Slot& slot) const
  {
    if (!slot.Initialized() || slot.state == TrackState::LOST)
    {
      return -std::numeric_limits<double>::infinity();
    }
    const auto clamp01 = [](double v) { return std::clamp(v, 0.0, 1.0); };
    const SelectWeights& w = s_.select;
    const Vehicle::Vec3 centre =
        slot.vehicle ? slot.vehicle_target.position : slot.fallback->Centre();
    const double spin =
        slot.vehicle ? slot.vehicle_target.v_yaw : slot.fallback->State()(7);
    const double distance_score =
        clamp01((w.max_distance_m - centre.norm()) / std::max(w.distance_span_m, 1e-6));
    const double area_score = clamp01(slot.area / std::max(w.area_norm_px, 1e-6));
    const double count_score = clamp01(slot.count_lpf / std::max(w.observed_count_norm, 1e-6));
    const double spin_score = clamp01(1.0 - std::abs(spin) / std::max(w.max_spin_rad_s, 1e-6));
    const double angle_score = clamp01(1.0 - slot.view_angle / std::max(w.max_angle_rad, 1e-6));
    const double scale = slot.state == TrackState::DETECTING   ? w.detecting_scale
                         : slot.state == TrackState::TEMP_LOST ? w.temp_lost_scale
                                                               : 1.0;
    return scale * (w.observed_count_weight * count_score + w.distance_weight * distance_score +
                    w.area_weight * area_score + w.spin_weight * spin_score +
                    w.angle_weight * angle_score);
  }

  static bool Selectable(const Slot& slot)
  {
    return slot.Initialized() && slot.state != TrackState::LOST && std::isfinite(slot.score) &&
           (!slot.vehicle || slot.vehicle_target.tracking);
  }

  /// 得分最高者；换目标要领先 switch_margin / Best score; switching needs a margin.
  int Select() const
  {
    if (s_.target_number >= 0)
    {
      return s_.target_number < SLOTS && Selectable(slots_[s_.target_number]) ? s_.target_number
                                                                                : -1;
    }
    int best = -1;
    for (int i = 0; i < SLOTS; ++i)
    {
      if (Selectable(slots_[i]) && (best < 0 || slots_[i].score > slots_[best].score))
      {
        best = i;
      }
    }
    if (best >= 0 && selected_ >= 0 && best != selected_ && Selectable(slots_[selected_]) &&
        slots_[best].score <= slots_[selected_].score + s_.select.switch_margin)
    {
      return selected_;
    }
    return best;
  }

  static void Fill(const Slot& slot, ArmorNumber n, ArmorTrackerTarget& out)
  {
    out.tracking = true;
    out.id = n;
    if (slot.vehicle)
    {
      const Vehicle::VehicleTarget& v = slot.vehicle_target;
      out.armors_num = 4;
      out.position = v.position;
      out.velocity = Eigen::Vector3d(v.velocity.x(), v.velocity.y(), 0.0);
      out.yaw = v.yaw;
      out.v_yaw = v.v_yaw;
      out.radius_1 = v.radius_1;
      out.radius_2 = v.radius_2;
      out.dz = v.dz;
      out.tracked_face_index = v.face;
      return;
    }
    const Fallback::FallbackTarget& f = *slot.fallback;
    out.armors_num = f.Plates();
    out.position = f.Centre();
    out.velocity = f.Velocity();
    out.yaw = f.OutputYaw();
    out.v_yaw = f.State()(7);
    out.radius_1 = f.State()(8);
    out.radius_2 = f.State()(8) + f.State()(9);
    out.dz = f.OutputDz();
    out.tracked_face_index = f.Face();
    out.outpost_height_phase = f.HeightPhase();
    out.face_switch_observed = f.Jumped();
  }

  const TrackerSettings s_;
  const CameraTypes::CameraCalibration* calibration_ = nullptr;
  Vehicle::Camera camera_;
  std::optional<Fallback::PlateSolver> solver_;
  std::array<Slot, SLOTS> slots_{};
  int selected_ = -1;
  bool has_base_ = false;
  uint64_t base_t_us_ = 0;
  uint64_t last_t_us_ = 0;
  std::array<double, 4> q_{1, 0, 0, 0};
  Vehicle::Mat3 r_bw_ = Vehicle::Mat3::Identity();
};
