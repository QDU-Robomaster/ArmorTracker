#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "VehicleGeometry.hpp"

/**
 * @brief 整车估计器不覆盖的目标的兜底跟踪：前哨站（三块板，高度相位）、基地（三块板）、
 *        平衡步兵（两块大板）。逐块板 PnP 后用 11 维 EKF 跟踪中心、朝向与半径。算法沿用
 *        原 ArmorTracker（移植自 sp_vision），只做了清理。
 *        Fallback tracking of targets the vehicle estimator does not model: outpost
 *        (three plates with a height phase), base (three plates) and balance infantry
 *        (two large plates). Per-plate PnP feeds an 11-state EKF of centre, heading and
 *        radius. The algorithm is the former ArmorTracker's (ported from sp_vision),
 *        only cleaned up.
 */
namespace Fallback
{
using Vehicle::Camera;
using Vehicle::Mat3;
using Vehicle::Vec3;

inline constexpr double PI = 3.14159265358979323846;

inline double LimitRad(double a)
{
  while (a > PI)
  {
    a -= 2.0 * PI;
  }
  while (a <= -PI)
  {
    a += 2.0 * PI;
  }
  return a;
}

/// 世界系方位角，前向为 0、左转为正 / World bearing, 0 forward, positive to the left.
inline double BearingYaw(const Vec3& v) { return std::atan2(-v.x(), v.y()); }

/// 方位、俯仰、距离 / Yaw, pitch and distance.
inline Vec3 XyzToYpd(const Vec3& p)
{
  const double xy = std::sqrt(p.x() * p.x() + p.y() * p.y());
  return {BearingYaw(p), std::atan2(p.z(), xy), p.norm()};
}

inline Mat3 XyzToYpdJacobian(const Vec3& p)
{
  const double x = p.x(), y = p.y(), z = p.z();
  const double xy2 = x * x + y * y;
  const double k = (z * z / xy2 + 1.0);
  Mat3 j;
  j << -y / xy2, x / xy2, 0.0, -(x * z) / (k * std::pow(xy2, 1.5)),
      -(y * z) / (k * std::pow(xy2, 1.5)), 1.0 / (k * std::pow(xy2, 0.5)), x / p.norm(),
      y / p.norm(), z / p.norm();
  return j;
}

/// 兜底目标的种类 / Kinds of fallback targets.
enum class Kind : uint8_t
{
  BALANCE,  ///< 两块大板，半径 0.2 m / Two large plates, radius 0.2 m
  OUTPOST,  ///< 三块板，半径 0.2765 m，三档高度 / Three plates at three heights
  BASE,     ///< 三块板，半径 0.3205 m / Three plates
};

inline constexpr double OUTPOST_HEIGHT_STEP = 0.102;
inline constexpr double OUTPOST_RADIUS = 0.2765;
inline constexpr double OUTPOST_TILT = 15.0 * PI / 180.0;

inline int PositiveMod(int value, int mod)
{
  const int r = value % mod;
  return r < 0 ? r + mod : r;
}

/// 前哨站第 face 块板相对中心的高度 / Height of an outpost plate relative to the centre.
inline double OutpostHeightOffset(int face, int phase)
{
  switch (PositiveMod(face + phase, 3))
  {
    case 1:
      return OUTPOST_HEIGHT_STEP;
    case 2:
      return -OUTPOST_HEIGHT_STEP;
    default:
      return 0.0;
  }
}

/// 前哨站贴纸朝外，观测到的板朝向要转半圈 / Outpost plates face outward.
inline double OutpostObservedYaw(double yaw) { return LimitRad(yaw + PI); }

/// 一块板的 PnP 观测（世界系）/ One plate's PnP observation in the world frame.
struct PlateObservation
{
  Vec3 xyz;    ///< 板中心 / Plate centre
  double yaw;  ///< 板朝向 / Plate heading
  Vec3 ypd;    ///< 板中心的方位、俯仰、距离 / Yaw, pitch, distance of the centre
};

/**
 * @brief 单块板 PnP（IPPE），朝向在机体前向 ±70° 内按 1° 搜索重投影误差最小者（平衡步兵
 *        直接用 PnP 朝向）。
 *        Single-plate PnP (IPPE); the heading is searched in 1° steps within ±70° of
 *        the body forward for the smallest reprojection error (balance infantry use the
 *        PnP heading).
 */
class PlateSolver
{
 public:
  PlateSolver(const Camera& cam) : cam_(cam)
  {
    k_ = (cv::Mat_<double>(3, 3) << cam.fx, 0, cam.cx, 0, cam.fy, cam.cy, 0, 0, 1);
    d_ = (cv::Mat_<double>(1, 5) << cam.dist[0], cam.dist[1], cam.dist[2], cam.dist[3],
          cam.dist[4]);
  }

  void SetAttitude(const Mat3& r_bw) { r_bw_ = r_bw; }

  /// 角点顺序：左上、右上、右下、左下 / Corners: top-left, top-right, bottom-right,
  /// bottom-left.
  bool Solve(const std::array<cv::Point2f, 4>& corners, bool large, Kind kind,
             PlateObservation& out) const
  {
    std::vector<cv::Point2f> image(corners.begin(), corners.end());
    cv::Vec3d rvec, tvec;
    if (!cv::solvePnP(Points(large), image, k_, d_, rvec, tvec, false, cv::SOLVEPNP_IPPE) ||
        !cv::checkRange(rvec) || !cv::checkRange(tvec))
    {
      return false;
    }
    const Vec3 xyz_camera(tvec[0], tvec[1], tvec[2]);
    out.xyz = r_bw_ * (cam_.R_cb * xyz_camera + cam_.t_cb);
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    Mat3 r_armor_camera;
    for (int r = 0; r < 3; ++r)
    {
      for (int c = 0; c < 3; ++c)
      {
        r_armor_camera(r, c) = rmat.at<double>(r, c);
      }
    }
    out.yaw = BearingYaw(r_bw_ * cam_.R_cb * r_armor_camera.col(0));
    out.ypd = XyzToYpd(out.xyz);
    if (kind != Kind::BALANCE)
    {
      out.yaw = SearchYaw(corners, large, kind, out.xyz, out.yaw);
    }
    return true;
  }

  /// 板在世界系 (xyz, yaw) 时的角点像素 / Corner pixels of a plate at (xyz, yaw).
  std::vector<cv::Point2f> Reproject(const Vec3& xyz, double yaw, bool large, Kind kind) const
  {
    const double tilt = kind == Kind::OUTPOST ? OUTPOST_TILT : Vehicle::ARMOR_TILT;
    const double s = std::sin(yaw), c = std::cos(yaw), st = std::sin(tilt), ct = std::cos(tilt);
    Mat3 r_armor_world;
    r_armor_world << -s * ct, -c, -s * st, c * ct, -s, c * st, -st, 0, ct;
    const Mat3 r_cb_t = cam_.R_cb.transpose();
    const Mat3 r_armor_camera = r_cb_t * r_bw_.transpose() * r_armor_world;
    const Vec3 t_armor_camera = r_cb_t * (r_bw_.transpose() * xyz - cam_.t_cb);
    cv::Mat r_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
    {
      for (int col = 0; col < 3; ++col)
      {
        r_cv.at<double>(r, col) = r_armor_camera(r, col);
      }
    }
    cv::Vec3d rvec;
    cv::Rodrigues(r_cv, rvec);
    const cv::Vec3d tvec(t_armor_camera.x(), t_armor_camera.y(), t_armor_camera.z());
    std::vector<cv::Point2f> image;
    cv::projectPoints(Points(large), rvec, tvec, k_, d_, image);
    return image;
  }

 private:
  static const std::vector<cv::Point3f>& Points(bool large)
  {
    static const auto make = [](float half_width)
    {
      constexpr float HALF_HEIGHT = 0.028F;
      return std::vector<cv::Point3f>{{0, half_width, HALF_HEIGHT},
                                      {0, -half_width, HALF_HEIGHT},
                                      {0, -half_width, -HALF_HEIGHT},
                                      {0, half_width, -HALF_HEIGHT}};
    };
    static const std::vector<cv::Point3f> SMALL = make(0.135F / 2), LARGE = make(0.230F / 2);
    return large ? LARGE : SMALL;
  }

  double SearchYaw(const std::array<cv::Point2f, 4>& corners, bool large, Kind kind,
                   const Vec3& xyz, double pnp_yaw) const
  {
    constexpr int RANGE_DEG = 140;
    const double body_yaw = BearingYaw(r_bw_.col(1));
    const double yaw0 = LimitRad(body_yaw - RANGE_DEG / 2.0 * PI / 180.0);
    double best_error = 1e10;
    double best_yaw = pnp_yaw;
    for (int i = 0; i < RANGE_DEG; ++i)
    {
      const double yaw = LimitRad(yaw0 + i * PI / 180.0);
      const auto image = Reproject(xyz, yaw, large, kind);
      double error = 0.0;
      for (int k = 0; k < 4; ++k)
      {
        error += cv::norm(corners[k] - image[k]);
      }
      if (error < best_error)
      {
        best_error = error;
        best_yaw = yaw;
      }
    }
    return best_yaw;
  }

  const Camera cam_;
  cv::Mat k_;
  cv::Mat d_;
  Mat3 r_bw_ = Mat3::Identity();
};

/**
 * @brief 11 维整车 EKF：[cx, vx, cy, vy, cz, vz, yaw, ω, r, Δr, Δz]，观测为一块板的方位、
 *        俯仰、距离与朝向。
 *        11-state vehicle EKF [cx, vx, cy, vy, cz, vz, yaw, ω, r, Δr, Δz] observing one
 *        plate's yaw, pitch, distance and heading.
 */
class FallbackTarget
{
 public:
  using Vec11 = Eigen::Matrix<double, 11, 1>;
  using Mat11 = Eigen::Matrix<double, 11, 11>;
  using Vec4 = Eigen::Vector4d;

  /// 起始时的选项（前哨站沿用上一次的中心与高度相位）/ Start options; the outpost may
  /// reuse the previous centre and height phase.
  struct Start
  {
    int face = 0;
    int height_phase = 0;
    bool height_phase_valid = false;
    bool centre_hint_valid = false;
    Vec3 centre_hint = Vec3::Zero();
  };

  FallbackTarget(Kind kind, const PlateObservation& obs, double t, const Start& start)
      : kind_(kind), plates_(kind == Kind::BALANCE ? 2 : 3), t_(t),
        face_(std::clamp(start.face, 0, plates_ - 1)), height_phase_(start.height_phase),
        height_phase_valid_(start.height_phase_valid)
  {
    const double r = kind == Kind::BALANCE   ? 0.2
                     : kind == Kind::OUTPOST ? OUTPOST_RADIUS
                                             : 0.3205;
    const double yaw = kind == Kind::OUTPOST ? OutpostObservedYaw(obs.yaw) : obs.yaw;
    double cx = obs.xyz.x() - r * std::sin(yaw);
    double cy = obs.xyz.y() + r * std::cos(yaw);
    double cz = obs.xyz.z();
    if (kind == Kind::OUTPOST)
    {
      cz -= OutpostHeightOffset(face_, height_phase_);
      if (start.centre_hint_valid)
      {
        cx = start.centre_hint.x();
        cy = start.centre_hint.y();
        cz = start.centre_hint.z();
      }
      observed_z_[face_] = obs.xyz.z();
      observed_z_valid_[face_] = true;
    }
    x_ << cx, 0.0, cy, 0.0, cz, 0.0, yaw, 0.0, r, 0.0, 0.0;
    Vec11 p0;
    if (kind == Kind::BALANCE)
    {
      p0 << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
    }
    else if (kind == Kind::OUTPOST)
    {
      p0 << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0;
    }
    else
    {
      p0 << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
    }
    P_ = p0.asDiagonal();
  }

  void Predict(double t)
  {
    const double dt = t - t_;
    t_ = t;
    Mat11 f = Mat11::Identity();
    f(0, 1) = f(2, 3) = f(4, 5) = f(6, 7) = dt;
    const double v1 = kind_ == Kind::OUTPOST ? 0.05 : 100.0;
    const double v2 = kind_ == Kind::OUTPOST ? 0.5 : 400.0;
    const double a = dt * dt * dt * dt / 4.0, b = dt * dt * dt / 2.0, c = dt * dt;
    Mat11 q = Mat11::Zero();
    for (int i : {0, 2, 4})
    {
      q(i, i) = a * v1;
      q(i, i + 1) = q(i + 1, i) = b * v1;
      q(i + 1, i + 1) = c * v1;
    }
    q(6, 6) = a * v2;
    q(6, 7) = q(7, 6) = b * v2;
    q(7, 7) = c * v2;
    // 收敛后前哨站转速钳到 ±2.51 rad/s / A converged outpost spins at ±2.51 rad/s.
    if (Converged() && kind_ == Kind::OUTPOST && std::abs(x_(7)) > 2.0)
    {
      x_(7) = x_(7) > 0.0 ? 2.51 : -2.51;
    }
    P_ = f * P_ * f.transpose() + q;
    x_ = f * x_;
    x_(6) = LimitRad(x_(6));
    HoldOutpostCentre();
  }

  void Update(const PlateObservation& obs)
  {
    const int id = Associate(obs);
    if (id != 0)
    {
      jumped_ = true;
    }
    UpdateHeightPhase(obs, id);
    face_ = id;
    ++updates_;
    UpdateEkf(obs, id);
    HoldOutpostCentre();
  }

  /// 半径越界即发散 / Diverged when a radius leaves (0.05, 0.5) m.
  bool Diverged() const
  {
    const bool r_ok = x_(8) > 0.05 && x_(8) < 0.5;
    const bool l_ok = x_(8) + x_(9) > 0.05 && x_(8) + x_(9) < 0.5;
    return !(r_ok && l_ok);
  }

  /// 发散，或（前哨站除外）最近 100 次更新里 NIS 超限不少于 40 次 / Diverged, or (except
  /// the outpost) at least 40 NIS failures in the last 100 updates.
  bool HealthFailed() const
  {
    if (Diverged())
    {
      return true;
    }
    if (kind_ == Kind::OUTPOST)
    {
      return false;
    }
    int failures = 0;
    for (bool f : nis_failures_)
    {
      failures += f ? 1 : 0;
    }
    return failures >= 40;
  }

  bool Converged()
  {
    if (!converged_ && !Diverged() && updates_ > (kind_ == Kind::OUTPOST ? 10 : 3))
    {
      converged_ = true;
    }
    return converged_;
  }

  Kind GetKind() const { return kind_; }
  int Plates() const { return plates_; }
  int Face() const { return face_; }
  int HeightPhase() const { return height_phase_; }
  bool Jumped() const { return jumped_; }
  const Vec11& State() const { return x_; }

  Vec3 Centre() const { return {x_(0), x_(2), x_(4)}; }
  Vec3 Velocity() const
  {
    return {x_(1), x_(3), kind_ == Kind::OUTPOST ? 0.0 : x_(5)};
  }
  /// 输出朝向：前哨站转半圈 / Output heading; the outpost turns half a circle.
  double OutputYaw() const
  {
    return kind_ == Kind::OUTPOST ? LimitRad(x_(6) + PI) : LimitRad(x_(6));
  }
  double OutputDz() const { return kind_ == Kind::OUTPOST ? OUTPOST_HEIGHT_STEP : x_(10); }

 private:
  bool OutpostHeightModel() const { return kind_ == Kind::OUTPOST; }

  /// 第 id 块板的 (x, y, z, 朝向) / Plate id's position and heading.
  Vec4 Plate(const Vec11& x, int id) const
  {
    const double angle = LimitRad(x(6) + id * 2.0 * PI / plates_);
    const double r = x(8);
    double z = x(4);
    if (OutpostHeightModel())
    {
      z = x(4) + OutpostHeightOffset(id, height_phase_);
    }
    return {x(0) + r * std::sin(angle), x(2) - r * std::cos(angle), z, angle};
  }

  /// 按距离取最近的至多 3 块板，选朝向与方位误差最小者 / Of the (up to 3) nearest
  /// plates, the one with the smallest heading plus bearing error.
  int Associate(const PlateObservation& obs) const
  {
    std::vector<std::pair<Vec4, int>> plates;
    for (int i = 0; i < plates_; ++i)
    {
      plates.push_back({Plate(x_, i), i});
    }
    std::sort(plates.begin(), plates.end(),
              [](const auto& a, const auto& b)
              { return XyzToYpd(a.first.template head<3>())[2] < XyzToYpd(b.first.template head<3>())[2]; });
    int best = 0;
    double best_error = std::numeric_limits<double>::infinity();
    const int candidates = std::min(3, static_cast<int>(plates.size()));
    for (int i = 0; i < candidates; ++i)
    {
      const Vec4& p = plates[i].first;
      const Vec3 ypd = XyzToYpd(p.head<3>());
      const double yaw = OutpostHeightModel() ? OutpostObservedYaw(obs.yaw) : obs.yaw;
      double error = std::abs(LimitRad(yaw - p(3))) + std::abs(LimitRad(obs.ypd(0) - ypd(0)));
      if (OutpostHeightModel() && height_phase_valid_)
      {
        error += 2.0 * std::abs(obs.xyz.z() - p(2));
      }
      if (error < best_error)
      {
        best_error = error;
        best = plates[i].second;
      }
    }
    return best;
  }

  void HoldOutpostCentre()
  {
    if (OutpostHeightModel())
    {
      x_(1) = x_(3) = x_(5) = 0.0;
    }
  }

  /// 前哨站换面时按高度跳变确定高度相位 / Determine the outpost height phase from the
  /// height jump at a face change.
  void UpdateHeightPhase(const PlateObservation& obs, int id)
  {
    if (!OutpostHeightModel() || height_phase_valid_)
    {
      return;
    }
    if (id != face_ && observed_z_valid_[face_])
    {
      constexpr double THRESHOLD = 0.04;
      constexpr double TWO_STEP_TOLERANCE = 0.05;
      const double delta = obs.xyz.z() - observed_z_[face_];
      const bool two_step =
          std::abs(std::abs(delta) - 2.0 * OUTPOST_HEIGHT_STEP) <= TWO_STEP_TOLERANCE;
      if (std::abs(delta) >= THRESHOLD && two_step)
      {
        const int sign = delta > 0.0 ? 1 : -1;
        for (int phase = 0; phase < 3; ++phase)
        {
          const double candidate = OutpostHeightOffset(id, phase) - OutpostHeightOffset(face_, phase);
          const bool candidate_two_step =
              std::abs(std::abs(candidate) - 2.0 * OUTPOST_HEIGHT_STEP) <= 1e-6;
          if (candidate_two_step && (candidate > 0.0 ? 1 : -1) == sign)
          {
            if (phase != height_phase_)
            {
              height_phase_ = phase;
              x_(4) = obs.xyz.z() - OutpostHeightOffset(id, height_phase_);
              x_(5) = 0.0;
            }
            height_phase_valid_ = true;
            break;
          }
        }
      }
    }
    constexpr double Z_ALPHA = 0.35;
    if (observed_z_valid_[id])
    {
      observed_z_[id] = (1.0 - Z_ALPHA) * observed_z_[id] + Z_ALPHA * obs.xyz.z();
    }
    else
    {
      observed_z_[id] = obs.xyz.z();
      observed_z_valid_[id] = true;
    }
  }

  Eigen::Matrix<double, 4, 11> Jacobian(const Vec11& x, int id) const
  {
    const double angle = LimitRad(x(6) + id * 2.0 * PI / plates_);
    const double r = x(8);
    Eigen::Matrix<double, 4, 11> h_xyza = Eigen::Matrix<double, 4, 11>::Zero();
    h_xyza(0, 0) = 1.0;
    h_xyza(0, 6) = r * std::cos(angle);
    h_xyza(0, 8) = std::sin(angle);
    h_xyza(1, 2) = 1.0;
    h_xyza(1, 6) = r * std::sin(angle);
    h_xyza(1, 8) = -std::cos(angle);
    h_xyza(2, 4) = 1.0;
    h_xyza(3, 6) = 1.0;
    const Mat3 h_ypd = XyzToYpdJacobian(Plate(x, id).head<3>());
    Eigen::Matrix4d h_ypda = Eigen::Matrix4d::Zero();
    h_ypda.topLeftCorner<3, 3>() = h_ypd;
    h_ypda(3, 3) = 1.0;
    return h_ypda * h_xyza;
  }

  Vec4 Measure(const Vec11& x, int id) const
  {
    const Vec4 p = Plate(x, id);
    const Vec3 ypd = XyzToYpd(p.head<3>());
    return {ypd(0), ypd(1), ypd(2), LimitRad(x(6) + id * 2.0 * PI / plates_)};
  }

  static Vec4 Residual(const Vec4& a, const Vec4& b)
  {
    Vec4 c = a - b;
    c(0) = LimitRad(c(0));
    c(1) = LimitRad(c(1));
    c(3) = LimitRad(c(3));
    return c;
  }

  void UpdateEkf(const PlateObservation& obs, int id)
  {
    const Vec3 centre_before = Centre();
    const double observed_yaw = OutpostHeightModel() ? OutpostObservedYaw(obs.yaw) : obs.yaw;
    const double side_view = std::abs(LimitRad(observed_yaw - BearingYaw(obs.xyz)));
    Vec4 r_diag;
    if (kind_ == Kind::OUTPOST)
    {
      const bool side = side_view > 0.55;
      const double ypd_noise = side ? 25.0 : 0.02 + 0.2 * side_view;
      r_diag << ypd_noise, ypd_noise, side ? 400.0 : 2.0, 3e-2;
    }
    else
    {
      r_diag << 4e-3, 4e-3, std::log(side_view + 1.0) + 1.0,
          std::log(std::abs(obs.ypd(2)) + 1.0) / 200.0 + 9e-2;
    }
    const Eigen::Matrix4d r = r_diag.asDiagonal();
    const Eigen::Matrix<double, 4, 11> h = Jacobian(x_, id);
    const Vec4 z(obs.ypd(0), obs.ypd(1), obs.ypd(2), observed_yaw);

    const Eigen::Matrix<double, 11, 4> k = P_ * h.transpose() * (h * P_ * h.transpose() + r).inverse();
    const Mat11 ikh = Mat11::Identity() - k * h;
    P_ = ikh * P_ * ikh.transpose() + k * r * k.transpose();
    x_ = x_ + k * Residual(z, Measure(x_, id));
    x_(6) = LimitRad(x_(6));
    const Vec4 residual = Residual(z, Measure(x_, id));
    const double nis = residual.transpose() * (h * P_ * h.transpose() + r).inverse() * residual;
    nis_failures_[nis_index_] = nis > 0.711;
    nis_index_ = (nis_index_ + 1) % nis_failures_.size();

    if (OutpostHeightModel())
    {
      // 前哨站中心不动；高度只在正对时缓慢跟随 / The outpost centre stays; the height
      // follows slowly only when facing.
      constexpr double Z_FOLLOW_ALPHA = 0.08;
      constexpr double Z_FOLLOW_FACING = 0.30;
      x_(0) = centre_before.x();
      x_(1) = 0.0;
      x_(2) = centre_before.y();
      x_(3) = 0.0;
      if (height_phase_valid_ && side_view < Z_FOLLOW_FACING)
      {
        const double observed_cz = obs.xyz.z() - OutpostHeightOffset(id, height_phase_);
        x_(4) = (1.0 - Z_FOLLOW_ALPHA) * centre_before.z() + Z_FOLLOW_ALPHA * observed_cz;
      }
      else
      {
        x_(4) = centre_before.z();
      }
      x_(5) = 0.0;
    }
  }

  const Kind kind_;
  const int plates_;
  double t_;
  int face_;
  int height_phase_;
  bool height_phase_valid_;
  bool jumped_ = false;
  bool converged_ = false;
  int updates_ = 0;
  Vec11 x_;
  Mat11 P_;
  std::array<bool, 100> nis_failures_{};
  std::size_t nis_index_ = 0;
  std::array<double, 3> observed_z_{};
  std::array<bool, 3> observed_z_valid_{};
};
}  // namespace Fallback
