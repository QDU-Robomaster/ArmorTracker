#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <unsupported/Eigen/MatrixFunctions>
#include <utility>
#include <vector>

#include "VehicleGeometry.hpp"

/**
 * @brief 角点级 EKF：以每个检测的四个角点为观测，状态见 VehicleGeometry.hpp。规格见
 *        aasim `docs/glr_estimator.md` §3–4、§7（谐振子自旋）、§8（预测速率）。
 *        Corner-level EKF with the four corners of each detection as the observation.
 *        Specification: aasim `docs/glr_estimator.md` §3–4, §7 (harmonic spin) and §8
 *        (predicted rates).
 */
namespace Vehicle
{
/// 预测速率的输出方式 / How the predicted rates are formed.
enum class RateMode : uint8_t
{
  CV,    ///< 当前速度 / Current rates
  ACC,   ///< 速度加加速度前推 / Command from velocity plus acceleration
  KEYS,  ///< 按键规则 / Key-driving rule
  AUTO,  ///< 按巡航平台在 ACC 与 KEYS 之间选 / Chosen from the cruise plateaus
};

/// 目标底盘的响应（实车应换成实测值）/ Target chassis response (measure on robots).
struct ChassisResponse
{
  double tau_v = 0.12, a_max = 4.0, v_max = 3.5, tau_w = 0.25, alpha_max = 25.0;
};

/// 一个滤波器的参数（§9）/ Parameters of one filter (§9).
struct FilterParams
{
  RateMode mode = RateMode::AUTO;
  double sig_px = 0.8, gate_px = 30.0;
  double q_v = 0.02, q_a = 0.1, q_w = 0.2, q_al = 2.0, q_geom = 1e-6;
  double tau_a = 0.6, tau_al = 0.25;
  double lost_after = 0.3, sat_frac = 0.75, v_key = 2.0, sig_scale = 1e-4, r0 = 0.22;
  // GLR 突变检测 / GLR step detection
  double window = 0.3;
  std::array<double, 3> threshold{15.0, 25.0,
                                  30.0};  ///< 平移、自旋、两者 / translation, spin, joint
  std::array<double, 3> sig_step{5.0, 2.0,
                                 30.0};  ///< 横向、纵深、自旋 / lateral, depth, spin
  double amb_margin = 6.0, max_wait = 0.05;
  // 巡航平台 / Cruise plateaus
  double plateau_frac = 0.25, plateau_hold = 0.15, plateau_min_speed = 0.4, key_cv = 0.15;
  // 谐振子自旋 / Harmonic spin
  bool harmonic = false;
  double tau_mean = 1.5, fit_window = 6.0, fit_every = 0.2, min_r2 = 0.8, min_amp = 1.0,
         period_tol = 0.15;
  double hist_dt = 0.02;
  double period_min = 0.8, period_max = 3.0, period_step = 0.1;
  ChassisResponse chassis;
  PlateShape shape =
      PROTOTYPE_SHAPE;  ///< 检测器角点对应的关键点 / Keypoints of the corners
};

/// 时域 h 内的平均中心速度与平均转速 / Mean centre velocity and spin over a horizon.
struct Rates
{
  Vec2 v;
  double w;
};

/// 一次观测更新的中间量，供 GLR 使用 / Intermediates of one update, used by the GLR.
struct UpdateTerms
{
  Eigen::Matrix<double, 8, NX> H;
  Eigen::Matrix<double, 8, 8> S;
  Eigen::Matrix<double, NX, 8> K;
  Pix r;
  double ll;  ///< 高斯对数似然 / Gaussian log-likelihood
};

/**
 * @brief 一个滤波器的模型与状态：过程模型（含谐振子自旋）、角点观测更新、预测速率。
 *        Model and state of one filter: process model (with harmonic spin), corner
 *        update and predicted rates.
 */
class CornerEkf
{
 public:
  CornerEkf(const FilterParams& p, const Camera& cam) : p_(p), cam_(cam)
  {
    for (double period = p.period_min; period <= p.period_max + 1e-9;
         period += p.period_step)
    {
      periods_.push_back(std::round(period * 100) / 100);
    }
  }

  const FilterParams& Params() const { return p_; }
  Vec x = Vec::Zero();
  Mat P = Mat::Identity();

  /// 精确离散的 Φ 与 Q（Van Loan），按 10 µs 分辨率缓存。plain 为不含谐振子的普通模型。
  /// Exact discrete Φ and Q (Van Loan), cached at 10 µs. `plain` drops the harmonic spin.
  const std::pair<Mat, Mat>& Transition(double dt, bool plain)
  {
    const long key = std::lround(dt * 1e5) * 2 + (plain ? 1 : 0);
    auto it = cache_.find(key);
    if (it != cache_.end())
    {
      return it->second;
    }
    if (cache_.size() > 256)
    {
      cache_.clear();
    }
    Vec qc = Vec::Zero();
    qc(VX) = qc(VY) = p_.q_v;
    qc(AX) = qc(AY) = p_.q_a;
    qc(OMEGA) = p_.q_w;
    qc(ALPHA) = p_.q_al;
    for (int i = CZ; i <= DZ; ++i)
    {
      qc(i) = p_.q_geom;
    }
    qc(SCALE_W) = qc(SCALE_H) = 1e-8;
    Eigen::Matrix<double, 2 * NX, 2 * NX> m =
        Eigen::Matrix<double, 2 * NX, 2 * NX>::Zero();
    const Mat a = Dynamics(plain);
    m.topLeftCorner<NX, NX>() = -a;
    m.topRightCorner<NX, NX>() = qc.asDiagonal();
    m.bottomRightCorner<NX, NX>() = a.transpose();
    const Eigen::Matrix<double, 2 * NX, 2 * NX> e =
        (m * (std::lround(dt * 1e5) * 1e-5)).exp();
    const Mat phi = e.bottomRightCorner<NX, NX>().transpose();
    const Mat q = phi * e.topRightCorner<NX, NX>();
    return cache_[key] = {phi, 0.5 * (q + q.transpose())};
  }

  /// 普通模型预测任意一组状态（起步假设用）/ Plain prediction of any state (boot).
  void PredictPlain(Vec& state, Mat& cov, double dt)
  {
    if (dt <= 0)
    {
      return;
    }
    const auto& pq = Transition(dt, true);
    state = pq.first * state;
    cov = pq.first * cov * pq.first.transpose() + pq.second;
  }

  /// 预测本滤波器的状态；返回 Φ 供 GLR 传播 / Predict this filter's state; returns Φ.
  const Mat& Predict(double dt)
  {
    const auto& pq = Transition(dt, false);
    x = pq.first * x;
    P = pq.first * P * pq.first.transpose() + pq.second;
    if (p_.harmonic)
    {
      // 谐振子的已知输入 ω̄ / Known mean-rate input of the harmonic block.
      const double c = std::cos(w_ * dt), s = std::sin(w_ * dt);
      x(YAW) += wbar_ * (dt - s / w_);
      x(OMEGA) += wbar_ * (1 - c);
      x(ALPHA) += wbar_ * w_ * s;
    }
    return pq.first;
  }

  /**
   * @brief 关联、数值雅可比、Huber 式放大的 EKF 更新（作用于给定的状态）。
   *        Association, numerical Jacobian and Huber-inflated EKF update on a state.
   * @return 角点误差超过关联门限时返回 false / False when outside the gate.
   */
  bool Update(Vec& state, Mat& cov, const Detection& d, const Mat3& r_bw,
              UpdateTerms& u) const
  {
    const auto object = ObjectPoints(d.type, p_.shape);
    Pix measured;
    for (int k = 0; k < 4; ++k)
    {
      measured(2 * k) = d.corners[k].x();
      measured(2 * k + 1) = d.corners[k].y();
    }
    int plate = 0;
    double best = std::numeric_limits<double>::infinity();
    for (int p = 0; p < 4; ++p)
    {
      const double e =
          (PlateCorners(cam_, r_bw, state, p, object) - measured).cwiseAbs().mean();
      if (e < best)
      {
        best = e;
        plate = p;
      }
    }
    if (best > p_.gate_px)
    {
      return false;
    }
    static constexpr double EPS[NX] = {1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-3,
                                       1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4};
    const Pix h0 = PlateCorners(cam_, r_bw, state, plate, object);
    for (int i = 0; i < NX; ++i)
    {
      Vec shifted = state;
      shifted(i) += EPS[i];
      u.H.col(i) = (PlateCorners(cam_, r_bw, shifted, plate, object) - h0) / EPS[i];
    }
    u.r = measured - h0;
    Pix noise;
    for (int i = 0; i < 8; ++i)
    {
      noise(i) =
          p_.sig_px * p_.sig_px * std::max(1.0, std::abs(u.r(i)) / p_.sig_px / 2.0);
    }
    u.S = u.H * cov * u.H.transpose();
    u.S.diagonal() += noise;
    const Eigen::LDLT<Eigen::Matrix<double, 8, 8>> ldlt(u.S);
    const Pix s_r = ldlt.solve(u.r);
    const Eigen::PartialPivLU<Eigen::Matrix<double, 8, 8>> lu(u.S);
    u.ll = -0.5 * (u.r.dot(s_r) + std::log(std::abs(lu.determinant())));
    u.K = (ldlt.solve(u.H * cov)).transpose();
    state += u.K * u.r;
    cov = (Mat::Identity() - u.K * u.H) * cov;
    cov = 0.5 * (cov + cov.transpose()).eval();
    return true;
  }

  /// 时域 h 内的平均速率（§8）/ Mean rates over the horizon h (§8).
  Rates PredictedRates(double h) const
  {
    const Vec2 v = x.segment<2>(VX), a = x.segment<2>(AX);
    const double w = x(OMEGA), al = x(ALPHA);
    if (p_.mode == RateMode::CV)
    {
      return {v, w};
    }
    const ChassisResponse& d = p_.chassis;
    Vec2 u = v + d.tau_v * a;
    const double na = a.norm();
    bool keys = p_.mode == RateMode::KEYS;
    double v_key = p_.v_key;
    if (p_.mode == RateMode::AUTO)
    {
      TranslationRule(keys, v_key);
    }
    if (keys && na >= p_.sat_frac * d.a_max)
    {
      u = a / na * v_key;
    }
    const double uw = w + d.tau_w * al;
    Rates r;
    r.v = MeanRate<2>(v, u, d.tau_v, d.a_max, h, d.v_max);
    Eigen::Matrix<double, 1, 1> w0, u0;
    w0 << w;
    u0 << uw;
    r.w = MeanRate<1>(w0, u0, d.tau_w, d.alpha_max, h, -1.0)(0);
    if (p_.harmonic && h > 0)
    {
      const double d0 = w - wbar_, wh = w_ * h;
      r.w = wbar_ + d0 * std::sin(wh) / wh + (al / w_) * (1 - std::cos(wh)) / wh;
    }
    return r;
  }

  /// 帧末的巡航平台与谐振子记录 / Cruise and harmonic bookkeeping after a frame.
  void AfterFrame(double t, double t_prev)
  {
    RecordCruise(t);
    if (!p_.harmonic)
    {
      return;
    }
    wbar_ += std::min(1.0, (t - t_prev) / p_.tau_mean) * (x(OMEGA) - wbar_);
    if (history_.empty() || t - history_.back().first >= p_.hist_dt - 1e-9)
    {
      history_.push_back({t, x(OMEGA)});
    }
    while (!history_.empty() && t - history_.front().first > p_.fit_window)
    {
      history_.pop_front();
    }
    if (t - t_fit_ >= p_.fit_every && t - history_.front().first >= 0.75 * p_.fit_window)
    {
      t_fit_ = t;
      FitPeriod();
    }
  }

  /// 丢失或起步时清空谐振子的历史 / Forget the harmonic history on loss or boot.
  void ResetSpinHistory()
  {
    history_.clear();
    if (p_.harmonic)
    {
      periodic_ = false;
    }
  }
  void SetMeanSpin(double wbar) { wbar_ = wbar; }
  /// 非谐振子滤波器恒为 true / Always true for non-harmonic filters.
  bool Periodic() const { return !p_.harmonic || periodic_; }

 private:
  Mat Dynamics(bool plain) const
  {
    Mat a = Mat::Zero();
    a(CX, VX) = a(CY, VY) = 1;
    a(VX, AX) = a(VY, AY) = 1;
    a(AX, AX) = a(AY, AY) = -1 / p_.tau_a;
    a(YAW, OMEGA) = 1;
    a(OMEGA, ALPHA) = 1;
    a(ALPHA, ALPHA) = -1 / p_.tau_al;
    if (p_.harmonic && !plain)
    {
      a(ALPHA, ALPHA) = 0;
      a(ALPHA, OMEGA) = -w_ * w_;
    }
    return a;
  }

  /// 从 v0 出发、保持输入 u 的一阶响应在 [0, h] 上的平均 / Mean first-order response.
  template <int N>
  static Eigen::Matrix<double, N, 1> MeanRate(Eigen::Matrix<double, N, 1> v,
                                              const Eigen::Matrix<double, N, 1>& u,
                                              double tau, double a_max, double h,
                                              double v_max)
  {
    const int n = std::max(2, static_cast<int>(std::ceil(h / 0.002)));
    const double dt = h / n;
    tau = std::max(tau, 1e-4);
    Eigen::Matrix<double, N, 1> sum = Eigen::Matrix<double, N, 1>::Zero();
    for (int i = 0; i < n; ++i)
    {
      Eigen::Matrix<double, N, 1> acc = (u - v) / tau;
      const double na = acc.norm();
      if (na > a_max)
      {
        acc *= a_max / na;
      }
      Eigen::Matrix<double, N, 1> vn = v + acc * dt;
      if (v_max > 0.0)
      {
        const double nv = vn.norm();
        if (nv > v_max)
        {
          vn *= v_max / nv;
        }
      }
      sum += 0.5 * (v + vn) * dt;
      v = vn;
    }
    return sum / h;
  }

  void RecordCruise(double t)
  {
    const double a = x.segment<2>(AX).norm(), v = x.segment<2>(VX).norm();
    const double a_max = p_.chassis.a_max;
    if (a >= 0.5 * a_max)
    {
      cruise_done_ = false;
    }
    if (a < p_.plateau_frac * a_max && v > p_.plateau_min_speed)
    {
      if (cruise_t0_ < 0)
      {
        cruise_t0_ = t;
      }
      else if (!cruise_done_ && t - cruise_t0_ >= p_.plateau_hold)
      {
        plateaus_.push_back(v);
        cruise_done_ = true;
      }
    }
    else
    {
      cruise_t0_ = -1;
    }
  }

  /// 最近 6 次平台：不足 2 次或速度一致时按键规则，否则用加速度前推。
  /// The last 6 plateaus: fewer than 2 or consistent speeds mean keys, else ACC.
  void TranslationRule(bool& keys, double& v_key) const
  {
    const int n = static_cast<int>(plateaus_.size());
    if (n < 2)
    {
      keys = true;
      return;
    }
    std::vector<double> ps(plateaus_.end() - std::min(n, 6), plateaus_.end());
    double mean = 0, var = 0;
    for (double p : ps)
    {
      mean += p;
    }
    mean /= ps.size();
    for (double p : ps)
    {
      var += (p - mean) * (p - mean);
    }
    var /= ps.size();
    if (std::sqrt(var) / std::max(mean, 1e-6) > p_.key_cv)
    {
      keys = false;
      return;
    }
    std::sort(ps.begin(), ps.end());
    const size_t m = ps.size();
    v_key = m % 2 ? ps[m / 2] : 0.5 * (ps[m / 2 - 1] + ps[m / 2]);
    keys = true;
  }

  /// 对转速历史按周期网格拟合“正弦 + 常数”/ Sine-plus-constant fit over the period grid.
  void FitPeriod()
  {
    const size_t n = history_.size();
    const double t_end = history_.back().first;
    double mean = 0;
    for (const auto& h : history_)
    {
      mean += h.second;
    }
    mean /= n;
    double var = 0;
    for (const auto& h : history_)
    {
      var += (h.second - mean) * (h.second - mean);
    }
    var /= n;
    if (var <= 0)
    {
      periodic_ = false;
      return;
    }
    double best_res = std::numeric_limits<double>::infinity(), best_period = 0,
           best_amp = 0;
    double ww = 0;
    for (const auto& h : history_)
    {
      ww += h.second * h.second;
    }
    for (double period : periods_)
    {
      Mat3 normal = Mat3::Zero();
      Vec3 rhs = Vec3::Zero();
      for (const auto& h : history_)
      {
        const double phase = 2 * M_PI * (h.first - t_end) / period;
        const Vec3 row(1.0, std::sin(phase), std::cos(phase));
        normal += row * row.transpose();
        rhs += row * h.second;
      }
      const Vec3 coef = normal.ldlt().solve(rhs);
      const double res = std::max(0.0, (ww - coef.dot(rhs)) / n);
      if (res < best_res)
      {
        best_res = res;
        best_period = period;
        best_amp = std::hypot(coef(1), coef(2));
      }
    }
    const double previous = period_fit_;
    periodic_ = (1 - best_res / var) >= p_.min_r2 && best_amp >= p_.min_amp &&
                previous > 0 &&
                std::abs(best_period - previous) <= p_.period_tol * previous;
    period_fit_ = best_period;
    if (periodic_ && std::abs(2 * M_PI / best_period - w_) > 1e-9)
    {
      w_ = 2 * M_PI / best_period;
      cache_.clear();
    }
  }

  const FilterParams p_;
  const Camera cam_;
  std::map<long, std::pair<Mat, Mat>> cache_;
  // 巡航平台 / Cruise plateaus
  std::vector<double> plateaus_;
  double cruise_t0_ = -1;
  bool cruise_done_ = false;
  // 谐振子自旋 / Harmonic spin
  double w_ = 2 * M_PI / 2.0;
  double wbar_ = 0;
  double t_fit_ = -1e9;
  double period_fit_ = -1;
  bool periodic_ = false;
  std::deque<std::pair<double, double>> history_;
  std::vector<double> periods_;
};
}  // namespace Vehicle
