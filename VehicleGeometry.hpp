#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>

/**
 * @brief 整车估计器的状态、相机与装甲板几何。
 *        State layout, camera and armor geometry of the vehicle estimator.
 *
 * 规格见 aasim `docs/glr_estimator.md`（原型 aaest）。世界系为 IMU 世界系，z 向上；本体系
 * x 右、y 前、z 上；装甲板系 x 沿（带 15° 倾角的）法向，y 横向，z 向上。
 * Specification: aasim `docs/glr_estimator.md` (prototype aaest). The world is the IMU
 * world frame with z up; the body frame is x right, y forward, z up; an armor frame has x
 * along its (15° tilted) normal, y lateral and z up.
 */
namespace Vehicle
{
/// 15 维状态的下标 / Indices of the 15-dimensional state.
enum State : int
{
  CX = 0,     ///< 中心 x / Centre x
  CY = 1,     ///< 中心 y / Centre y
  VX = 2,     ///< 中心速度 / Centre velocity
  VY = 3,
  AX = 4,     ///< 中心加速度 / Centre acceleration
  AY = 5,
  YAW = 6,    ///< 0 号板朝向 / Heading of plate 0
  OMEGA = 7,  ///< 转速 / Spin rate
  ALPHA = 8,  ///< 角加速度 / Spin acceleration
  CZ = 9,     ///< 中心高度 / Centre height
  R_EVEN = 10,  ///< 偶数板半径 / Even-plate radius
  R_ODD = 11,   ///< 奇数板半径 / Odd-plate radius
  DZ = 12,      ///< 奇数板高度差 / Odd-plate height offset
  SCALE_W = 13,  ///< 关键点横向尺度 / Keypoint lateral scale
  SCALE_H = 14,  ///< 关键点纵向尺度 / Keypoint vertical scale
  NX = 15,
};

using Vec = Eigen::Matrix<double, NX, 1>;
using Mat = Eigen::Matrix<double, NX, NX>;
using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;
/// 四个角点 x0 y0 … x3 y3 / Four corners x0 y0 … x3 y3.
using Pix = Eigen::Matrix<double, 8, 1>;

inline constexpr double HALF_PI = 1.5707963267948966;
inline constexpr double ARMOR_TILT = 15.0 * M_PI / 180.0;

/// 光学系到本体系的固定轴变换 / Fixed axes from the optical frame to the body frame.
inline Mat3 OpticalToBody()
{
  Mat3 r;
  r << 1, 0, 0, 0, 0, 1, 0, -1, 0;
  return r;
}

/**
 * @brief 针孔相机加五参数畸变，以及相机在云台本体上的安装：p_b = R_cb·p_c + t_cb。
 *        Pinhole camera with five-term distortion and its mounting on the gimbal body:
 *        p_b = R_cb·p_c + t_cb.
 */
struct Camera
{
  double fx = 1250, fy = 1250, cx = 640, cy = 512;
  std::array<double, 5> dist{0, 0, 0, 0, 0};  ///< k1 k2 p1 p2 k3
  Mat3 R_cb = OpticalToBody();
  Vec3 t_cb = Vec3::Zero();
};

/**
 * @brief 一块检测到的装甲板，角点按估计器内部顺序：左上、右上、右下、左下。
 *        One detected armor, corners in the estimator's order: top-left, top-right,
 *        bottom-right, bottom-left.
 */
struct Detection
{
  std::array<Vec2, 4> corners;
  int type = 0;  ///< 0 小装甲，1 大装甲 / 0 small, 1 large
};

/// 本体系到世界系的旋转（四元数 wxyz，先归一化）/ Body-to-world rotation from wxyz.
inline Mat3 RotationFromQuaternion(double w, double x, double y, double z)
{
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  w /= n;
  x /= n;
  y /= n;
  z /= n;
  Mat3 r;
  r << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
  return r;
}

/// 装甲板系到世界系：朝向 yaw，倾角 15° / Armor-to-world rotation for a heading.
inline Mat3 ArmorRotation(double yaw)
{
  const double s = std::sin(yaw), c = std::cos(yaw);
  const double st = std::sin(ARMOR_TILT), ct = std::cos(ARMOR_TILT);
  Mat3 r;
  r << -s * ct, -c, -s * st, c * ct, -s, c * st, -st, 0, ct;
  return r;
}

/// 装甲板系里的四个关键点 / The four keypoints in the armor frame.
inline std::array<Vec3, 4> ObjectPoints(int type)
{
  const double w = type == 1 ? 0.230 : 0.135;
  const double l = 0.056;
  return {Vec3(0, w / 2, l / 2), Vec3(0, -w / 2, l / 2), Vec3(0, -w / 2, -l / 2),
          Vec3(0, w / 2, -l / 2)};
}

/// 世界点投影到像素 / World points to pixels for a body-to-world attitude.
inline Pix Project(const Camera& cam, const Mat3& r_bw, const std::array<Vec3, 4>& points)
{
  const Mat3 r_bc = cam.R_cb.transpose();
  const auto& d = cam.dist;
  Pix out;
  for (int k = 0; k < 4; ++k)
  {
    const Vec3 pc = r_bc * (r_bw.transpose() * points[k] - cam.t_cb);
    const double z = std::max(pc.z(), 1e-3), x = pc.x() / z, y = pc.y() / z;
    const double r2 = x * x + y * y;
    const double radial = 1 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 * r2 * r2;
    const double xd = x * radial + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x);
    const double yd = y * radial + d[2] * (r2 + 2 * y * y) + 2 * d[3] * x * y;
    out(2 * k) = cam.fx * xd + cam.cx;
    out(2 * k + 1) = cam.fy * yd + cam.cy;
  }
  return out;
}

/// 状态 x 下第 plate 块板的角点像素 / Corner pixels of one plate for state x.
inline Pix PlateCorners(const Camera& cam, const Mat3& r_bw, const Vec& x, int plate,
                        const std::array<Vec3, 4>& object)
{
  const double a = x(YAW) + plate * HALF_PI;
  const bool odd = plate % 2 == 1;
  const double r = odd ? x(R_ODD) : x(R_EVEN);
  const Vec3 centre(x(CX) + r * std::sin(a), x(CY) - r * std::cos(a),
                    x(CZ) + (odd ? x(DZ) : 0.0));
  const Mat3 rot = ArmorRotation(a);
  std::array<Vec3, 4> points;
  for (int k = 0; k < 4; ++k)
  {
    Vec3 o = object[k];
    o.y() *= x(SCALE_W);
    o.z() *= x(SCALE_H);
    points[k] = centre + rot * o;
  }
  return Project(cam, r_bw, points);
}

/// 最正对原点（射手）的板 / The plate most facing the origin (the shooter).
inline int FacingPlate(const Vec& x)
{
  int best = 0;
  double best_cos = -2;
  for (int k = 0; k < 4; ++k)
  {
    const double a = x(YAW) + k * HALF_PI;
    const double r = (k % 2) ? x(R_ODD) : x(R_EVEN);
    const Vec2 c(x(CX) + r * std::sin(a), x(CY) - r * std::cos(a));
    const Vec2 n(std::sin(a), -std::cos(a));
    const double cos_view = n.dot(-c / c.norm());
    if (cos_view > best_cos)
    {
      best_cos = cos_view;
      best = k;
    }
  }
  return best;
}
}  // namespace Vehicle
