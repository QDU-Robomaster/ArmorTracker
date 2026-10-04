// Vendored from Jiu-xiao/aasim cpp/aaest/aaest.hpp (based on c03a841, plus the camera mounting extrinsic).
// Edit it there and copy it here: aasim checks this estimator frame by frame against its Python reference
// (tools/parity_aaest.py, tools/parity_real.py). Algorithm: aasim docs/glr_estimator.md.

// aaest: whole-vehicle estimator for armor detections (C++ port of aasim/algos/lagekf.py, glrekf.py, mmglr.py).
//
// Steady acceleration-state EKF over the four corner keypoints of every detection, a GLR detector for steps
// of the translation / spin accelerations (operator key changes), a harmonic spin model with an online
// period fit (periodic variable-speed tops), and a multi-model selection by corner likelihood. Boot: the
// first detection's IPPE pose starts one filter per spin-rate hypothesis; the most likely one continues.
// Specification: docs/glr_estimator.md. Header-only; needs Eigen 3 (with unsupported/MatrixFunctions) and
// OpenCV (calib3d, for the IPPE boot).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace aaest {

constexpr int NX = 15;
using Vec = Eigen::Matrix<double, NX, 1>;
using Mat = Eigen::Matrix<double, NX, NX>;
using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;
using Pix = Eigen::Matrix<double, 8, 1>;              // corners x0 y0 x1 y1 x2 y2 x3 y3
using MatH = Eigen::Matrix<double, 8, NX>;
using MatD = Eigen::Matrix<double, NX, 3>;
constexpr double HALF_PI = 1.5707963267948966;

// ---- geometry (aasim/real/geom.py, ArmorTracker conventions) -------------------------------------

// Optical camera axes to gimbal body axes (x right, y forward, z up).
inline Mat3 R_CB() {
  Mat3 R;
  R << 1, 0, 0, 0, 0, 1, 0, -1, 0;
  return R;
}

// Pinhole camera with OpenCV plumb-bob distortion and its mounting on the gimbal body: a body point
// p_b = R_cb * p_c + t_cb (ArmorTracker: R_cb = mount rotation * R_CB(), t_cb = mount translation).
struct Camera {
  double fx = 1250, fy = 1250, cx = 640, cy = 512;
  std::array<double, 5> dist{0, 0, 0, 0, 0};          // k1 k2 p1 p2 k3
  Mat3 R_cb = R_CB();
  Vec3 t_cb = Vec3::Zero();
};

struct Detection {
  std::array<Vec2, 4> corners;                          // LT, RT, RB, LB pixels
  int type = 0;                                         // 0 small, 1 large armor
};


inline Mat3 quat_to_rot(double w, double x, double y, double z) {
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  w /= n; x /= n; y /= n; z /= n;
  Mat3 R;
  R << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
       2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
       2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
  return R;
}

// Armor-to-world rotation from the plate yaw and the 15 degree tilt (geom.armor_rotation).
inline Mat3 armor_rotation(double yaw, double tilt = 15.0 * M_PI / 180.0) {
  const double s = std::sin(yaw), c = std::cos(yaw), st = std::sin(tilt), ct = std::cos(tilt);
  Mat3 R;
  R << -s * ct, -c, -s * st, c * ct, -s, c * st, -st, 0, ct;
  return R;
}

inline std::array<Vec3, 4> object_points(int type) {
  const double w = type == 1 ? 0.230 : 0.135, l = 0.056;
  return {Vec3(0, w / 2, l / 2), Vec3(0, -w / 2, l / 2), Vec3(0, -w / 2, -l / 2), Vec3(0, w / 2, -l / 2)};
}

// World points to pixels for a body-to-world attitude Rbw (window.WindowEstimator._project).
inline Pix project(const Camera& cam, const Mat3& Rbw, const std::array<Vec3, 4>& P) {
  const Mat3 RcbT = cam.R_cb.transpose();
  Pix out;
  const auto& D = cam.dist;
  for (int k = 0; k < 4; ++k) {
    const Vec3 pc = RcbT * (Rbw.transpose() * P[k] - cam.t_cb);
    const double z = std::max(pc.z(), 1e-3), x = pc.x() / z, y = pc.y() / z;
    const double r2 = x * x + y * y, rad = 1 + D[0] * r2 + D[1] * r2 * r2 + D[4] * r2 * r2 * r2;
    const double xd = x * rad + 2 * D[2] * x * y + D[3] * (r2 + 2 * x * x);
    const double yd = y * rad + D[2] * (r2 + 2 * y * y) + 2 * D[3] * x * y;
    out(2 * k) = cam.fx * xd + cam.cx;
    out(2 * k + 1) = cam.fy * yd + cam.cy;
  }
  return out;
}

// ---- parameters (docs/glr_estimator.md section 9) -------------------------------------------------

enum class Mode { CV, ACC, KB, AUTO };

struct Dyn {                                            // assumed target chassis response
  double tau_v = 0.12, a_max = 4.0, v_max = 3.5, tau_w = 0.25, alpha_max = 25.0;
};

struct FilterParams {
  Mode mode = Mode::AUTO;
  double sig_px = 0.8, gate_px = 30.0;
  double q_v = 0.02, q_a = 0.1, q_w = 0.2, q_al = 2.0, q_geom = 1e-6;
  double tau_a = 0.6, tau_al = 0.25;
  double lost_after = 0.3, sat_frac = 0.75, v_key = 2.0, sig_scale = 1e-4, r0 = 0.22;
  // GLR step detection
  double window = 0.3;
  std::array<double, 3> threshold{15.0, 25.0, 30.0};    // translation, spin, joint
  std::array<double, 3> sig_step{5.0, 2.0, 30.0};       // lateral, along the line of sight, spin
  double amb_margin = 6.0, max_wait = 0.05;
  // cruise plateaus (auto translation rule)
  double plateau_frac = 0.25, plateau_hold = 0.15, plateau_min_speed = 0.4, key_cv = 0.15;
  // harmonic spin (period fitted online)
  bool harmonic = false;
  double tau_mean = 1.5, fit_window = 6.0, fit_every = 0.2, min_r2 = 0.8, min_amp = 1.0, period_tol = 0.15;
  double hist_dt = 0.02;                                // rate history kept every 20 ms for the period fit
  double period_min = 0.8, period_max = 3.0, period_step = 0.1;
  Dyn dyn;
};

// Mean of the first-order response from v0 toward the held input u over [0, h] (oracle._mean_rate_nb).
template <int N>
inline Eigen::Matrix<double, N, 1> mean_rate(Eigen::Matrix<double, N, 1> v, const Eigen::Matrix<double, N, 1>& u,
                                              double tau, double a_max, double h, double v_max) {
  const int n = std::max(2, static_cast<int>(std::ceil(h / 0.002)));
  const double dt = h / n;
  tau = std::max(tau, 1e-4);
  Eigen::Matrix<double, N, 1> s = Eigen::Matrix<double, N, 1>::Zero();
  for (int i = 0; i < n; ++i) {
    Eigen::Matrix<double, N, 1> acc = (u - v) / tau;
    const double na = acc.norm();
    if (na > a_max) acc *= a_max / na;
    Eigen::Matrix<double, N, 1> vn = v + acc * dt;
    if (v_max > 0.0) {
      const double nv = vn.norm();
      if (nv > v_max) vn *= v_max / nv;
    }
    s += 0.5 * (v + vn) * dt;
    v = vn;
  }
  return s / h;
}

struct Rates {
  Vec2 v;      // mean centre velocity over the horizon
  double w;    // mean spin rate over the horizon
};

// ---- one filter: steady EKF + GLR (+ harmonic spin) -------------------------------------------------

class Filter {
 public:
  explicit Filter(const FilterParams& p = FilterParams(), const Camera& cam = Camera()) : p_(p), cam_(cam) {
    for (double P = p.period_min; P <= p.period_max + 1e-9; P += p.period_step) periods_.push_back(std::round(P * 100) / 100);
    W_ = 2 * M_PI / 2.0;
  }

  bool active() const { return active_; }
  bool booting() const { return !hyps_.empty(); }
  const Vec& x() const { return x_; }
  const Mat& P() const { return P_; }
  double t() const { return t_; }
  double ll_frame = 0.0;                                // corner log-likelihood of the latest frame
  bool periodic() const { return !p_.harmonic || periodic_; }
  int n_jumps() const { return n_jumps_; }
  double period() const { return period_fit_; }

  struct Out {
    bool tracking = false;
    Vec state = Vec::Zero();
    Rates rates{Vec2::Zero(), 0.0};
  };

  // One frame (LagEKFEstimator.step and its subclasses): PnP boot while inactive, else predict / update /
  // step test. ``h`` is the horizon of the reported rates. The output is taken where the Python version
  // builds its message: after the step test, before the cruise and harmonic bookkeeping.
  Out frame(double t, const Mat3& Rbw, const std::vector<Detection>& dets, double h) {
    Out out;
    if (!active_) {
      if (!boot_step(t, Rbw, dets, h, out) || !active_) {
        hist_.clear();
        if (p_.harmonic) periodic_ = false;
      }
      return out;
    }
    const double t_prev = t_;
    predict(t - t_);
    t_ = t;
    bool ok = false;
    ll_frame = 0.0;
    for (const auto& d : dets) {
      Upd u;
      if (update_core(x_, P_, d, Rbw, u)) {
        ok = true;
        ll_frame += u.ll;
        glr_accumulate(u);
      }
    }
    if (ok) {
      t_seen_ = t;
    } else if (t - t_seen_ > p_.lost_after) {
      active_ = false;
      clear_bank();
      hist_.clear();
      if (p_.harmonic) periodic_ = false;
      return out;
    }
    glr_test();
    out = {true, x_, rates_for(h)};
    after_frame(t, t_prev);
    return out;
  }

  // Plain prediction (no GLR, no harmonic term) for the boot hypotheses.
  void boot_predict(Vec& x, Mat& P, double dt) {
    if (dt <= 0) return;
    const auto& PQ = transition(dt, true);
    x = PQ.first * x;
    P = PQ.first * P * PQ.first.transpose() + PQ.second;
  }

  // Mean centre velocity and spin rate over the next h seconds (output mode; AccGLREstimator.rates).
  Rates rates(double h) const {
    Vec2 v = x_.segment<2>(2), a = x_.segment<2>(4);
    const double w = x_(7), al = x_(8);
    if (p_.mode == Mode::CV) return {v, w};
    const Dyn& d = p_.dyn;
    Vec2 u = v + d.tau_v * a;
    const double na = a.norm();
    bool kb = p_.mode == Mode::KB;
    double v_key = p_.v_key;
    if (p_.mode == Mode::AUTO) rule(kb, v_key);
    if (kb && na >= p_.sat_frac * d.a_max) u = a / na * v_key;
    const double uw = w + d.tau_w * al;
    Rates r;
    r.v = mean_rate<2>(v, u, d.tau_v, d.a_max, h, d.v_max);
    Eigen::Matrix<double, 1, 1> w0, u0;
    w0 << w; u0 << uw;
    r.w = mean_rate<1>(w0, u0, d.tau_w, d.alpha_max, h, -1.0)(0);
    if (p_.harmonic && h > 0) {
      const double d0 = w - wbar_, Wh = W_ * h;
      r.w = wbar_ + d0 * std::sin(Wh) / Wh + (al / W_) * (1 - std::cos(Wh)) / Wh;
    }
    return r;
  }

  // Index of the plate most facing the shooter (origin) in the current state.
  int face() const {
    int best = 0;
    double bc = -2;
    for (int k = 0; k < 4; ++k) {
      const double a = x_(6) + k * HALF_PI, r = (k % 2) ? x_(11) : x_(10);
      const Vec2 c(x_(0) + r * std::sin(a), x_(1) - r * std::cos(a)), n(std::sin(a), -std::cos(a));
      const double cv = n.dot(-c / c.norm());
      if (cv > bc) { bc = cv; best = k; }
    }
    return best;
  }

  const FilterParams& params() const { return p_; }
  const Camera& camera() const { return cam_; }

 private:
  struct Upd { MatH H; Eigen::Matrix<double, 8, 8> S; Eigen::Matrix<double, NX, 8> K; Pix r; double ll; };
  struct Hyp { Vec x; Mat P; double ll; };

  Rates rates_for(double h) const { return p_.mode == Mode::CV ? Rates{x_.segment<2>(2), x_(7)} : rates(h); }

  // Cruise plateaus and the harmonic mean / period bookkeeping after the frame's message.
  void after_frame(double t, double t_prev) {
    cruise();
    if (!p_.harmonic) return;
    wbar_ += std::min(1.0, (t - t_prev) / p_.tau_mean) * (x_(7) - wbar_);
    if (hist_.empty() || t - hist_.back().first >= p_.hist_dt - 1e-9) hist_.push_back({t, x_(7)});
    while (!hist_.empty() && t - hist_.front().first > p_.fit_window) hist_.pop_front();
    if (t - t_fit_ >= p_.fit_every && t - hist_.front().first >= 0.75 * p_.fit_window) {
      t_fit_ = t;
      fit_period();
    }
  }

  // PnP boot (LagEKFEstimator._pnp_boot_step); returns the tracking flag of the frame.
  bool boot_step(double t, const Mat3& Rbw, const std::vector<Detection>& dets, double h, Out& out) {
    double dt = 0.0;
    if (hyps_.empty()) {
      if (dets.empty()) return false;
      const Detection* d = &dets[0];
      double widest = -1;
      for (const auto& e : dets) {
        double lo = 1e18, hi = -1e18;
        for (const auto& c : e.corners) { lo = std::min(lo, c.x()); hi = std::max(hi, c.x()); }
        if (hi - lo > widest) { widest = hi - lo; d = &e; }
      }
      Vec x0;
      Mat P0;
      if (!pnp_state(*d, Rbw, x0, P0)) return false;
      for (double w : acq_omegas_) {
        Hyp hy{x0, P0, 0.0};
        hy.x(7) = w;
        hyps_.push_back(hy);
      }
      boot_t0_ = t_ = t;
    } else {
      dt = t - t_;
      t_ = t;
    }
    for (auto& hy : hyps_) {
      boot_predict(hy.x, hy.P, dt);
      ll_frame = 0.0;                                   // the frame value left is the last hypothesis's (as Python)
      for (const auto& d : dets) {
        Upd u;
        if (update_core(hy.x, hy.P, d, Rbw, u)) ll_frame += u.ll;
      }
      hy.ll += ll_frame;
    }
    const Hyp& best = *std::max_element(hyps_.begin(), hyps_.end(), [](const Hyp& a, const Hyp& b) { return a.ll < b.ll; });
    x_ = best.x;
    P_ = best.P;
    if (t - boot_t0_ >= boot_time_) {
      const double t_prev = t_;
      hyps_.clear();
      active_ = true;
      t_ = t_seen_ = t;
      clear_bank();
      wbar_ = x_(7);
      out = {true, x_, rates_for(h)};
      after_frame(t, t_prev);
      return true;
    }
    out = {true, x_, rates_for(h)};
    return true;
  }

  // Initial state and covariance from the detection's best IPPE solution (LagEKFEstimator._pnp_state).
  bool pnp_state(const Detection& d, const Mat3& Rbw, Vec& x, Mat& P) const {
    const auto o = object_points(d.type);
    std::vector<cv::Point3d> obj;
    std::vector<cv::Point2d> img;
    for (int k = 0; k < 4; ++k) {
      obj.emplace_back(o[k].x(), o[k].y(), o[k].z());
      img.emplace_back(d.corners[k].x(), d.corners[k].y());
    }
    const cv::Mat K = (cv::Mat_<double>(3, 3) << cam_.fx, 0, cam_.cx, 0, cam_.fy, cam_.cy, 0, 0, 1);
    const cv::Mat D = (cv::Mat_<double>(1, 5) << cam_.dist[0], cam_.dist[1], cam_.dist[2], cam_.dist[3], cam_.dist[4]);
    std::vector<cv::Mat> rv, tv;
    cv::Mat err;
    const int n = cv::solvePnPGeneric(obj, img, K, D, rv, tv, false, cv::SOLVEPNP_IPPE, cv::noArray(), cv::noArray(), err);
    if (n < 1) return false;
    int j = 0;
    for (int i = 1; i < n; ++i)
      if (err.at<double>(i) < err.at<double>(j)) j = i;
    cv::Mat Rc;
    cv::Rodrigues(rv[j], Rc);
    Mat3 R_co;
    Vec3 tc;
    for (int r = 0; r < 3; ++r) {
      tc(r) = tv[j].at<double>(r);
      for (int c = 0; c < 3; ++c) R_co(r, c) = Rc.at<double>(r, c);
    }
    const Mat3 M = Rbw * cam_.R_cb;                     // optical -> world (rotation)
    const Vec3 pw = Rbw * (cam_.R_cb * tc + cam_.t_cb);  // plate centre
    const Vec3 xa = M * R_co.col(0);                    // armor x axis (inward normal)
    const double a = std::atan2(-xa.x(), xa.y()), r0 = p_.r0;
    x = Vec::Zero();
    x(0) = pw.x() - r0 * std::sin(a);
    x(1) = pw.y() + r0 * std::cos(a);
    x(6) = a;
    x(9) = pw.z();
    x(10) = x(11) = r0;
    x(13) = x(14) = 1.0;
    Vec d0;
    d0 << 0.01, 0.01, 1, 1, 4, 4, 0.05 * 0.05, 9, 100, 1e-4, 0.0025, 0.0025, 0.0009, p_.sig_scale * p_.sig_scale,
        p_.sig_scale * p_.sig_scale;
    P = d0.asDiagonal();
    return true;
  }
  struct Onset { double t; MatD D; Mat3 C; Vec3 d; };

  Mat A(bool plain) const {
    Mat A = Mat::Zero();
    A(0, 2) = A(1, 3) = 1;
    A(2, 4) = A(3, 5) = 1;
    A(4, 4) = A(5, 5) = -1 / p_.tau_a;
    A(6, 7) = 1;
    A(7, 8) = 1;
    A(8, 8) = -1 / p_.tau_al;
    if (p_.harmonic && !plain) {
      A(8, 8) = 0;
      A(8, 7) = -W_ * W_;
    }
    return A;
  }

  // Exact discrete transition and process noise (Van Loan), cached per dt in 10 us steps.
  const std::pair<Mat, Mat>& transition(double dt, bool plain) {
    const long key = std::lround(dt * 1e5) * 2 + (plain ? 1 : 0);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    if (cache_.size() > 256) cache_.clear();
    Vec qc = Vec::Zero();
    qc(2) = qc(3) = p_.q_v;
    qc(4) = qc(5) = p_.q_a;
    qc(7) = p_.q_w;
    qc(8) = p_.q_al;
    for (int i = 9; i < 13; ++i) qc(i) = p_.q_geom;
    qc(13) = qc(14) = 1e-8;
    Eigen::Matrix<double, 2 * NX, 2 * NX> M = Eigen::Matrix<double, 2 * NX, 2 * NX>::Zero();
    const Mat Am = A(plain);
    M.topLeftCorner<NX, NX>() = -Am;
    M.topRightCorner<NX, NX>() = qc.asDiagonal();
    M.bottomRightCorner<NX, NX>() = Am.transpose();
    const Eigen::Matrix<double, 2 * NX, 2 * NX> E = (M * (std::lround(dt * 1e5) * 1e-5)).exp();
    const Mat Phi = E.bottomRightCorner<NX, NX>().transpose();
    const Mat Q = Phi * E.topRightCorner<NX, NX>();
    return cache_[key] = {Phi, 0.5 * (Q + Q.transpose())};
  }

  void predict(double dt) {
    if (dt <= 0) return;
    const auto& PQ = transition(dt, false);
    const Mat& Phi = PQ.first;
    x_ = Phi * x_;
    P_ = Phi * P_ * Phi.transpose() + PQ.second;
    // GLR onsets: propagate, add one at this frame, drop the old ones.
    const double t_new = t_ + dt;
    size_t keep = 0;
    for (size_t i = 0; i < bank_.size(); ++i) {
      if (t_new - bank_[i].t > p_.window) continue;
      bank_[keep] = bank_[i];
      bank_[keep].D = Phi * bank_[i].D;
      ++keep;
    }
    bank_.resize(keep);
    Onset o{t_new, MatD::Zero(), Mat3::Zero(), Vec3::Zero()};
    o.D(4, 0) = o.D(5, 1) = o.D(8, 2) = 1;
    bank_.push_back(o);
    if (p_.harmonic) {                                  // known mean-rate input of the harmonic block
      const double c = std::cos(W_ * dt), s = std::sin(W_ * dt);
      x_(6) += wbar_ * (dt - s / W_);
      x_(7) += wbar_ * (1 - c);
      x_(8) += wbar_ * W_ * s;
    }
  }

  Pix corners(const Vec& x, int plate, const std::array<Vec3, 4>& obj, const Mat3& Rbw) const {
    const double a = x(6) + plate * HALF_PI;
    const bool odd = plate % 2 == 1;
    const double r = odd ? x(11) : x(10);
    const Vec3 ctr(x(0) + r * std::sin(a), x(1) - r * std::cos(a), x(9) + (odd ? x(12) : 0.0));
    const Mat3 R = armor_rotation(a);
    std::array<Vec3, 4> P;
    for (int k = 0; k < 4; ++k) {
      Vec3 o = obj[k];
      o.y() *= x(13);
      o.z() *= x(14);
      P[k] = ctr + R * o;
    }
    return project(cam_, Rbw, P);
  }

  // Association, numerical Jacobian, Huber-inflated update (LagEKFEstimator._update).
  bool update_core(Vec& x, Mat& P, const Detection& d, const Mat3& Rbw, Upd& u) const {
    const auto obj = object_points(d.type);
    Pix C;
    for (int k = 0; k < 4; ++k) { C(2 * k) = d.corners[k].x(); C(2 * k + 1) = d.corners[k].y(); }
    int plate = 0;
    double best = std::numeric_limits<double>::infinity();
    for (int p = 0; p < 4; ++p) {
      const double e = (corners(x, p, obj, Rbw) - C).cwiseAbs().mean();
      if (e < best) { best = e; plate = p; }
    }
    if (best > p_.gate_px) return false;
    static const double eps[NX] = {1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4,
                                   1e-4, 1e-4};
    const Pix h0 = corners(x, plate, obj, Rbw);
    for (int i = 0; i < NX; ++i) {
      Vec xs = x;
      xs(i) += eps[i];
      u.H.col(i) = (corners(xs, plate, obj, Rbw) - h0) / eps[i];
    }
    u.r = C - h0;
    Pix Rv;
    for (int i = 0; i < 8; ++i) Rv(i) = p_.sig_px * p_.sig_px * std::max(1.0, std::abs(u.r(i)) / p_.sig_px / 2.0);
    u.S = u.H * P * u.H.transpose();
    u.S.diagonal() += Rv;
    const Eigen::LDLT<Eigen::Matrix<double, 8, 8>> ldlt(u.S);
    const Pix Sr = ldlt.solve(u.r);
    const Eigen::PartialPivLU<Eigen::Matrix<double, 8, 8>> lu(u.S);
    u.ll = -0.5 * (u.r.dot(Sr) + std::log(std::abs(lu.determinant())));
    u.K = (ldlt.solve(u.H * P)).transpose();
    x += u.K * u.r;
    P = (Mat::Identity() - u.K * u.H) * P;
    P = 0.5 * (P + P.transpose()).eval();
    return true;
  }

  void glr_accumulate(const Upd& u) {
    if (bank_.empty()) return;
    const Eigen::Matrix<double, 8, 8> Si = u.S.inverse();
    for (auto& o : bank_) {
      const Eigen::Matrix<double, 8, 3> G = u.H * o.D;
      const Eigen::Matrix<double, 8, 3> SG = Si * G;
      o.C += G.transpose() * SG;
      o.d += SG.transpose() * u.r;
      o.D -= u.K * G;
    }
  }

  void clear_bank() { bank_.clear(); t_wait_ = -1; }

  // Best onset of one hypothesis (indices IX of the step) by the marginal likelihood statistic.
  template <int M>
  void best_onset(const std::array<int, M>& ix, const Mat3& Sd, double& lr_best, int& j_best,
                  Eigen::Matrix<double, M, 1>& g_best, Eigen::Matrix<double, M, M>& Cm_best) const {
    using MM = Eigen::Matrix<double, M, M>;
    using VM = Eigen::Matrix<double, M, 1>;
    MM S;
    for (int a = 0; a < M; ++a)
      for (int b = 0; b < M; ++b) S(a, b) = Sd(ix[a], ix[b]);
    const MM Sinv = S.inverse();
    lr_best = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < bank_.size(); ++j) {
      MM C;
      VM dv;
      for (int a = 0; a < M; ++a) {
        dv(a) = bank_[j].d(ix[a]);
        for (int b = 0; b < M; ++b) C(a, b) = bank_[j].C(ix[a], ix[b]);
      }
      const MM Cm = C + Sinv;
      const VM g = Cm.partialPivLu().solve(dv);
      const double lr = dv.dot(g) - std::log((MM::Identity() + S * C).determinant());
      if (lr > lr_best) { lr_best = lr; j_best = static_cast<int>(j); g_best = g; Cm_best = Cm; }
    }
  }

  template <int M>
  void commit(const std::array<int, M>& ix, int j, const Eigen::Matrix<double, M, 1>& g,
              const Eigen::Matrix<double, M, M>& Cm) {
    Eigen::Matrix<double, NX, M> D;
    for (int a = 0; a < M; ++a) D.col(a) = bank_[j].D.col(ix[a]);
    x_ += D * g;
    P_ += D * Cm.partialPivLu().solve(D.transpose());
    P_ = 0.5 * (P_ + P_.transpose()).eval();
    ++n_jumps_;
    clear_bank();
  }

  // Most likely step among translation / spin / joint hypotheses; commit above threshold (GLR._test).
  void glr_test() {
    if (bank_.empty()) return;
    const Vec2 n = x_.segment<2>(0) / std::max(x_.segment<2>(0).norm(), 1e-6);
    Mat3 Sd = Mat3::Zero();
    const auto& s = p_.sig_step;
    Sd.topLeftCorner<2, 2>() = s[0] * s[0] * (Eigen::Matrix2d::Identity() - n * n.transpose()) + s[1] * s[1] * n * n.transpose();
    Sd(2, 2) = s[2] * s[2];
    static const std::array<int, 2> I2{0, 1};
    static const std::array<int, 1> I1{2};
    static const std::array<int, 3> I3{0, 1, 2};
    double lr[3] = {0.0, 0.0, 0.0};
    int j[3] = {0, 0, 0};
    Eigen::Matrix<double, 2, 1> g2 = Eigen::Matrix<double, 2, 1>::Zero(); Eigen::Matrix2d C2 = Eigen::Matrix2d::Zero();
    Eigen::Matrix<double, 1, 1> g1 = Eigen::Matrix<double, 1, 1>::Zero(); Eigen::Matrix<double, 1, 1> C1 = Eigen::Matrix<double, 1, 1>::Zero();
    Eigen::Vector3d g3 = Eigen::Vector3d::Zero(); Mat3 C3 = Mat3::Zero();
    best_onset<2>(I2, Sd, lr[0], j[0], g2, C2);
    best_onset<1>(I1, Sd, lr[1], j[1], g1, C1);
    best_onset<3>(I3, Sd, lr[2], j[2], g3, C3);
    int b = -1;
    for (int si = 0; si < 3; ++si) {
      const double thr = p_.threshold[si];
      if (lr[si] > thr && (b < 0 || lr[si] - thr > lr[b] - p_.threshold[b])) b = si;
    }
    if (b < 0) { t_wait_ = -1; return; }
    if (b < 2 && p_.max_wait > 0 && std::abs(lr[0] - lr[1]) < p_.amb_margin) {
      if (t_wait_ < 0) t_wait_ = t_;
      if (t_ - t_wait_ < p_.max_wait) return;
    }
    t_wait_ = -1;
    if (b == 0) commit<2>(I2, j[0], g2, C2);
    else if (b == 1) commit<1>(I1, j[1], g1, C1);
    else commit<3>(I3, j[2], g3, C3);
  }

  // Cruise-plateau bookkeeping for the auto translation rule.
  void cruise() {
    const double a = x_.segment<2>(4).norm(), v = x_.segment<2>(2).norm(), amax = p_.dyn.a_max;
    if (a >= 0.5 * amax) cruise_done_ = false;
    if (a < p_.plateau_frac * amax && v > p_.plateau_min_speed) {
      if (cruise_t0_ < 0) cruise_t0_ = t_;
      else if (!cruise_done_ && t_ - cruise_t0_ >= p_.plateau_hold) { plateaus_.push_back(v); cruise_done_ = true; }
    } else {
      cruise_t0_ = -1;
    }
  }

  void rule(bool& kb, double& v_key) const {
    const int n = static_cast<int>(plateaus_.size());
    if (n < 2) { kb = true; return; }
    std::vector<double> ps(plateaus_.end() - std::min(n, 6), plateaus_.end());
    double mean = 0, var = 0;
    for (double x : ps) mean += x;
    mean /= ps.size();
    for (double x : ps) var += (x - mean) * (x - mean);
    var /= ps.size();
    if (std::sqrt(var) / std::max(mean, 1e-6) > p_.key_cv) { kb = false; return; }
    std::sort(ps.begin(), ps.end());
    const size_t m = ps.size();
    v_key = m % 2 ? ps[m / 2] : 0.5 * (ps[m / 2 - 1] + ps[m / 2]);
    kb = true;
  }

  // Least-squares sinusoid + constant on the rate history for every period of the grid.
  void fit_period() {
    const size_t n = hist_.size();
    const double t_end = hist_.back().first;
    double mean = 0;
    for (const auto& h : hist_) mean += h.second;
    mean /= n;
    double var = 0;
    for (const auto& h : hist_) var += (h.second - mean) * (h.second - mean);
    var /= n;
    if (var <= 0) { periodic_ = false; return; }
    double b_res = std::numeric_limits<double>::infinity(), b_P = 0, b_amp = 0;
    double ww = 0;
    for (const auto& h : hist_) ww += h.second * h.second;
    for (double P : periods_) {                         // 3x3 normal equations of [1, sin, cos]
      Mat3 N = Mat3::Zero();
      Vec3 rhs = Vec3::Zero();
      for (const auto& h : hist_) {
        const double ph = 2 * M_PI * (h.first - t_end) / P;
        const Vec3 xr(1.0, std::sin(ph), std::cos(ph));
        N += xr * xr.transpose();
        rhs += xr * h.second;
      }
      const Vec3 coef = N.ldlt().solve(rhs);
      const double res = std::max(0.0, (ww - coef.dot(rhs)) / n);
      if (res < b_res) { b_res = res; b_P = P; b_amp = std::hypot(coef(1), coef(2)); }
    }
    const double prev = period_fit_;
    periodic_ = (1 - b_res / var) >= p_.min_r2 && b_amp >= p_.min_amp && prev > 0 &&
                std::abs(b_P - prev) <= p_.period_tol * prev;
    period_fit_ = b_P;
    if (periodic_ && std::abs(2 * M_PI / b_P - W_) > 1e-9) {
      W_ = 2 * M_PI / b_P;
      cache_.clear();
    }
  }

  FilterParams p_;
  Camera cam_;
  bool active_ = false;
  Vec x_ = Vec::Zero();
  Mat P_ = Mat::Identity();
  double t_ = 0, t_seen_ = 0, t_wait_ = -1;
  std::vector<Onset> bank_;
  std::map<long, std::pair<Mat, Mat>> cache_;
  int n_jumps_ = 0;
  // auto rule
  std::vector<double> plateaus_;
  double cruise_t0_ = -1;
  bool cruise_done_ = false;
  // harmonic spin
  double W_, wbar_ = 0, t_fit_ = -1e9, period_fit_ = -1;
  bool periodic_ = false;
  std::deque<std::pair<double, double>> hist_;
  std::vector<double> periods_;
  std::vector<Hyp> hyps_;
  std::vector<double> acq_omegas_{0, -8, 8, -16, 16, -24, 24};  // ties (first frames) go to the slowest
  double boot_t0_ = 0, boot_time_ = 0.3;
};

// ---- multi-model estimator ------------------------------------------------------------------------------

struct ModelSpec {
  std::string name;                                     // "steady", "vary", "agile", "harm"
  FilterParams p;
};

inline std::vector<ModelSpec> default_models(Mode mode = Mode::AUTO) {
  FilterParams base;
  base.mode = mode;
  ModelSpec steady{"steady", base}, vary{"vary", base}, agile{"agile", base}, harm{"harm", base};
  vary.p.q_al = 300.0;
  vary.p.tau_al = 0.25;
  agile.p.q_a = 2.0;
  agile.p.threshold = {10.0, 25.0, 30.0};
  harm.p.harmonic = true;
  harm.p.q_al = 20.0;
  return {steady, vary, agile, harm};
}

struct EstimatorParams {
  double tau = 0.3, margin = 3.0, spin_min = 2.0, harm_bonus = 5.0, steady_bonus = 3.0;
};

struct Target {
  bool tracking = false;
  Vec3 position = Vec3::Zero();                         // centre x, y and plate height cz
  Vec2 velocity = Vec2::Zero();                          // mean over the requested horizon
  double yaw = 0, v_yaw = 0, radius_1 = 0, radius_2 = 0, dz = 0;
  int face = 0, model = 0;
};

// Filters run side by side on every frame; the one with the best weighted corner likelihood reports
// (MultiModelGLREstimator; docs/glr_estimator.md section 7).
class Estimator {
 public:
  Estimator(const Camera& cam, std::vector<ModelSpec> models = default_models(),
            const EstimatorParams& ep = EstimatorParams())
      : specs_(std::move(models)), ep_(ep) {
    for (const auto& s : specs_) f_.emplace_back(s.p, cam);
    sums_.assign(f_.size(), 0.0);
  }

  // One frame: image timestamp (s), gimbal attitude (w, x, y, z), this target's detections, and the
  // horizon h (s) over which the reported rates are averaged (expected frame-to-impact time).
  Target step(double t, const std::array<double, 4>& q, const std::vector<Detection>& dets, double h) {
    const Mat3 Rbw = quat_to_rot(q[0], q[1], q[2], q[3]);
    std::vector<Filter::Out> outs;
    for (auto& f : f_) {
      f.ll_frame = 0.0;
      outs.push_back(f.frame(t, Rbw, dets, h));
    }
    const double dt = first_ ? 0.0 : t - t_prev_;
    first_ = false;
    const double g = std::exp(-dt / ep_.tau);
    for (size_t i = 0; i < f_.size(); ++i) sums_[i] = f_[i].active() ? g * sums_[i] + f_[i].ll_frame : 0.0;
    t_prev_ = t;
    select();
    const Filter::Out* o = &outs[use_];
    if (!o->tracking)
      for (const auto& e : outs)
        if (e.tracking) { o = &e; break; }
    Target out;
    if (!o->tracking) return out;
    const Vec& x = o->state;
    out.tracking = true;
    out.position = Vec3(x(0), x(1), x(9));
    out.velocity = o->rates.v;
    out.yaw = std::atan2(std::sin(x(6)), std::cos(x(6)));
    out.v_yaw = o->rates.w;
    out.radius_1 = x(10);
    out.radius_2 = x(11);
    out.dz = x(12);
    out.face = face(x);
    out.model = use_;
    return out;
  }

  // Rates of the reporting filter for another horizon (an aimer evaluates several launch times).
  Rates rates(double h) const { return f_[use_].rates(h); }
  int model() const { return use_; }
  const Filter& filter(int i) const { return f_[i]; }
  double sum(int i) const { return sums_[i]; }
  int size() const { return static_cast<int>(f_.size()); }

 private:
  static int face(const Vec& x) {
    int best = 0;
    double bc = -2;
    for (int k = 0; k < 4; ++k) {
      const double a = x(6) + k * HALF_PI, r = (k % 2) ? x(11) : x(10);
      const Vec2 c(x(0) + r * std::sin(a), x(1) - r * std::cos(a)), n(std::sin(a), -std::cos(a));
      const double cv = n.dot(-c / c.norm());
      if (cv > bc) { bc = cv; best = k; }
    }
    return best;
  }

  void select() {
    const bool spinning = f_[0].active() && std::abs(f_[0].x()(7)) >= ep_.spin_min;
    std::vector<int> allowed;
    for (size_t i = 0; i < f_.size(); ++i) {
      const auto& nm = specs_[i].name;
      const bool spin_model = nm.rfind("vary", 0) == 0 || nm.rfind("harm", 0) == 0;
      if (f_[i].active() && (i == 0 || spinning || !spin_model) && f_[i].periodic()) allowed.push_back(static_cast<int>(i));
    }
    if (allowed.empty()) return;
    std::vector<double> score(f_.size());
    for (size_t i = 0; i < f_.size(); ++i)
      score[i] = sums_[i] + (specs_[i].name.rfind("harm", 0) == 0 ? ep_.harm_bonus : 0.0) + (i == 0 ? ep_.steady_bonus : 0.0);
    int best = allowed[0];
    for (int i : allowed)
      if (score[i] > score[best]) best = i;
    const bool use_ok = std::find(allowed.begin(), allowed.end(), use_) != allowed.end();
    if (!use_ok) use_ = best;
    else if (best != use_ && score[best] > score[use_] + ep_.margin) use_ = best;
  }

  std::vector<ModelSpec> specs_;
  EstimatorParams ep_;
  std::vector<Filter> f_;
  std::vector<double> sums_;
  int use_ = 0;
  bool first_ = true;
  double t_prev_ = 0;
};

}  // namespace aaest
