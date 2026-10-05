#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "CornerEkf.hpp"
#include "ManeuverDetector.hpp"
#include "ModelSelector.hpp"
#include "PoseBootstrap.hpp"
#include "VehicleGeometry.hpp"

/**
 * @brief 四块板整车估计器：四个滤波器（STEADY、VARY、AGILE、HARM）并行处理每帧检测，
 *        由 ModelSelector 选出输出者。按 aasim `docs/glr_estimator.md` 重新实现，原型为
 *        aaest（aasim `cpp/aaest/aaest.hpp`）。
 *        Four-plate vehicle estimator: four filters (STEADY, VARY, AGILE, HARM) process
 *        every frame side by side and ModelSelector picks the reporting one.
 *        Re-implemented from aasim `docs/glr_estimator.md`; prototype aaest.
 */
namespace Vehicle
{
/// 一个滤波器：角点 EKF + 换档检测 + 起步 / One filter: corner EKF, step detection, boot.
class VehicleFilter
{
 public:
  struct Output
  {
    bool tracking = false;
    Vec state = Vec::Zero();
    Rates rates{Vec2::Zero(), 0.0};
  };

  VehicleFilter(ModelKind kind, const FilterParams& p, const Camera& cam)
      : kind(kind), ekf_(p, cam), glr_(p), cam_(cam)
  {
  }

  const ModelKind kind;
  double ll_frame = 0.0;  ///< 本帧角点对数似然 / Corner log-likelihood of this frame

  bool Active() const { return active_; }
  const Vec& State() const { return ekf_.x; }
  bool Periodic() const { return ekf_.Periodic(); }
  Rates PredictedRates(double h) const { return ekf_.PredictedRates(h); }

  /// 处理一帧；h 为输出速率的平均时域 / One frame; h is the horizon of the rates.
  Output Frame(double t, const Mat3& r_bw, const std::vector<Detection>& dets, double h)
  {
    ll_frame = 0.0;
    if (!active_)
    {
      return BootFrame(t, r_bw, dets, h);
    }
    const double t_prev = t_;
    if (t > t_)
    {
      glr_.OnPredict(ekf_.Predict(t - t_), t);
    }
    t_ = t;
    bool seen = false;
    for (const auto& d : dets)
    {
      UpdateTerms u;
      if (ekf_.Update(ekf_.x, ekf_.P, d, r_bw, u))
      {
        seen = true;
        ll_frame += u.ll;
        glr_.OnUpdate(u);
      }
    }
    if (seen)
    {
      t_seen_ = t;
    }
    else if (t - t_seen_ > ekf_.Params().lost_after)
    {
      active_ = false;
      glr_.Clear();
      ekf_.ResetSpinHistory();
      return {};
    }
    glr_.Test(ekf_.x, ekf_.P, t_);
    Output out{true, ekf_.x, ekf_.PredictedRates(h)};
    ekf_.AfterFrame(t, t_prev);
    return out;
  }

 private:
  Output BootFrame(double t, const Mat3& r_bw, const std::vector<Detection>& dets,
                   double h)
  {
    if (!boot_.Frame(ekf_, cam_, t, r_bw, dets, ll_frame))
    {
      ekf_.ResetSpinHistory();
      return {};
    }
    Output out{true, ekf_.x, ekf_.PredictedRates(h)};
    if (!boot_.Done(t))
    {
      ekf_.ResetSpinHistory();
      return out;
    }
    boot_.Clear();
    active_ = true;
    t_ = t_seen_ = t;
    glr_.Clear();
    ekf_.SetMeanSpin(ekf_.x(OMEGA));
    ekf_.AfterFrame(t, t);
    return out;
  }

  CornerEkf ekf_;
  ManeuverDetector glr_;
  PoseBootstrap boot_;
  const Camera cam_;
  bool active_ = false;
  double t_ = 0;
  double t_seen_ = 0;
};

/// 估计器对外的整车状态 / Vehicle state reported by the estimator.
struct VehicleTarget
{
  bool tracking = false;
  Vec3 position = Vec3::Zero();  ///< 中心 x、y 与高度 / Centre x, y and height
  Vec2 velocity = Vec2::Zero();  ///< 时域内平均速度 / Mean over the horizon
  double yaw = 0, v_yaw = 0, radius_1 = 0, radius_2 = 0, dz = 0;
  int face = 0;   ///< 最正对射手的板 / Plate most facing the shooter
  int model = 0;  ///< 输出的滤波器 / Reporting filter
};

class VehicleEstimator
{
 public:
  explicit VehicleEstimator(const Camera& cam, RateMode mode = RateMode::AUTO,
                            const PlateShape& shape = PROTOTYPE_SHAPE)
  {
    FilterParams base;
    base.mode = mode;
    base.shape = shape;
    FilterParams vary = base;
    vary.q_al = 300.0;
    vary.tau_al = 0.25;
    FilterParams agile = base;
    agile.q_a = 2.0;
    agile.threshold = {10.0, 25.0, 30.0};
    FilterParams harm = base;
    harm.harmonic = true;
    harm.q_al = 20.0;
    filters_.emplace_back(ModelKind::STEADY, base, cam);
    filters_.emplace_back(ModelKind::VARY, vary, cam);
    filters_.emplace_back(ModelKind::AGILE, agile, cam);
    filters_.emplace_back(ModelKind::HARM, harm, cam);
  }

  /**
   * @brief 处理一帧。
   *        Process one frame.
   * @param t 图像曝光中点时间，秒 / Mid-exposure time in s
   * @param q 该时刻云台本体系到世界系的姿态 wxyz / Gimbal body-to-world attitude
   * @param dets 本目标这一帧的检测 / This target's detections in the frame
   * @param h 输出速率的平均时域，秒 / Horizon of the reported rates in s
   */
  VehicleTarget Step(double t, const std::array<double, 4>& q,
                     const std::vector<Detection>& dets, double h)
  {
    const Mat3 r_bw = RotationFromQuaternion(q[0], q[1], q[2], q[3]);
    std::vector<VehicleFilter::Output> outputs;
    std::vector<ModelSelector::Candidate> candidates;
    for (auto& f : filters_)
    {
      outputs.push_back(f.Frame(t, r_bw, dets, h));
      candidates.push_back({f.kind, f.Active(), f.Periodic(), f.ll_frame});
    }
    use_ = selector_.Select(t, candidates, filters_[0].State()(OMEGA));
    const VehicleFilter::Output* o = &outputs[use_];
    if (!o->tracking)
    {
      for (const auto& e : outputs)
      {
        if (e.tracking)
        {
          o = &e;
          break;
        }
      }
    }
    VehicleTarget out;
    if (!o->tracking)
    {
      return out;
    }
    const Vec& x = o->state;
    out.tracking = true;
    out.position = Vec3(x(CX), x(CY), x(CZ));
    out.velocity = o->rates.v;
    out.yaw = std::atan2(std::sin(x(YAW)), std::cos(x(YAW)));
    out.v_yaw = o->rates.w;
    out.radius_1 = x(R_EVEN);
    out.radius_2 = x(R_ODD);
    out.dz = x(DZ);
    out.face = FacingPlate(x);
    out.model = use_;
    return out;
  }

  /// 输出滤波器在另一个时域上的速率 / Rates of the reporting filter for another horizon.
  Rates PredictedRates(double h) const { return filters_[use_].PredictedRates(h); }

 private:
  std::vector<VehicleFilter> filters_;
  ModelSelector selector_;
  int use_ = 0;
};
}  // namespace Vehicle
