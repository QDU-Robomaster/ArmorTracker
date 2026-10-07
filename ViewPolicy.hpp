#pragma once

#include <cmath>
#include <limits>
#include <optional>

#include "CameraBase.hpp"

/// 远距离 NARROW 跟随的设置 / Settings of NARROW following at long range.
struct ViewSettings
{
  bool enabled;          ///< 是否控制视角 / Whether the tracker controls the view
  double enter_m;        ///< 进入 NARROW 的距离，6.0 / Distance to enter NARROW, m
  double exit_m;         ///< 退出 NARROW 的距离，5.0 / Distance to leave NARROW, m
  double lost_s;         ///< 目标没看到多久退出，0.1 / Unseen time to leave, s
  double recenter_frac;  ///< 目标偏离窗口中心超过窗口宽高的这个比例时平移，0.25
                         ///< Offset from the window centre, as a fraction of the
                         ///< window size, that moves the window
  double edge_margin;    ///< 目标中心到窗口边缘的最小比例，0.1 / Smallest distance of
                         ///< the target centre from the window edge, as a fraction
  double min_move_s;     ///< 两次平移的最小间隔，0.1 / Shortest time between moves, s
};

/**
 * @brief 远距离 NARROW 跟随：选中的目标在 enter_m 以外且处于 TRACKING 时切到 NARROW，
 *        窗口居中于目标；目标偏离窗口中心时平移窗口；目标近于 exit_m、丢失超过 lost_s
 *        或窗口已顶到传感器边缘仍装不下时回到 WIDE。纯逻辑，请求由调用方转给
 *        CameraFrameSync。
 *        NARROW following at long range: switch to NARROW, with the window centred on
 *        the target, when the selected target is TRACKING beyond enter_m; move the
 *        window when the target drifts from its centre; go back to WIDE when the target
 *        comes closer than exit_m, stays unseen for lost_s, or no longer fits a window
 *        pressed against the sensor edge. Pure logic; the caller forwards the requests
 *        to CameraFrameSync.
 *
 * 判断按自己请求过的视角与窗口进行，不看帧的几何，所以请求生效前不会重复请求。
 * Decisions use the view and window this policy requested, not the frame geometry, so
 * nothing is requested twice while a request takes effect.
 */
class ViewPolicy
{
 public:
  /// 时间比较的容差 / Tolerance of time comparisons.
  static constexpr double TIME_EPS = 1e-6;
  /// 小于这个距离的平移不做，原生像素 / Moves shorter than this are skipped, native px.
  static constexpr double MIN_SHIFT_PX = 8.0;

  /// 一帧跟踪结果中与视角有关的部分 / The part of a tracked frame the policy uses.
  struct Input
  {
    double t;                     ///< s
    bool selected;                ///< 有选中的目标 / A target is selected
    bool seen;                    ///< 选中目标处于 TRACKING / It is TRACKING
    double distance;              ///< 目标中心距离 / Distance of the centre, m
    bool in_front;                ///< 目标中心在相机前方 / The centre is in front
    CameraTypes::Point2d native;  ///< 目标中心的原生像素 / Centre in native pixels
  };

  /// 要转给 CameraFrameSync 的请求，先移窗再切档 / Requests for CameraFrameSync; the
  /// move goes first, then the view.
  struct Request
  {
    std::optional<NarrowPosition> move;
    std::optional<View> view;
  };

  explicit ViewPolicy(const ViewSettings& settings) : s_(settings) {}

  Request Step(const CameraTypes::CameraCalibration& calibration, const Input& in)
  {
    if (!s_.enabled)
    {
      return {};
    }
    if (in.selected && in.seen)
    {
      last_seen_ = in.t;
    }
    if (!narrow_)
    {
      if (!in.selected || !in.seen || !in.in_front || in.distance < s_.enter_m)
      {
        return {};
      }
      const CameraTypes::Point2d centre = WindowCentre(calibration, in.native);
      if (!Fits(centre, in.native))
      {
        return {};
      }
      narrow_ = true;
      centre_ = centre;
      last_move_ = in.t;
      return {CameraBase::CenteredOn(calibration, in.native), View::NARROW};
    }

    if (!in.selected || in.t - last_seen_ >= s_.lost_s - TIME_EPS ||
        in.distance < s_.exit_m)
    {
      return Leave();
    }
    if (!in.seen || !in.in_front || !Drifted(in.native) ||
        in.t - last_move_ < s_.min_move_s - TIME_EPS)
    {
      return {};
    }
    const CameraTypes::Point2d centre = WindowCentre(calibration, in.native);
    if (!Fits(centre, in.native))
    {
      return Leave();  // 窗口已顶到传感器边缘 / The window is against the sensor edge
    }
    if (std::hypot(centre.x - centre_.x, centre.y - centre_.y) < MIN_SHIFT_PX)
    {
      return {};
    }
    centre_ = centre;
    last_move_ = in.t;
    return {CameraBase::CenteredOn(calibration, in.native), std::nullopt};
  }

  /// 是否请求了 NARROW / Whether NARROW is requested.
  bool Narrow() const { return narrow_; }

 private:
  static constexpr double HALF_W = CameraTypes::FRAME_WIDTH / 2.0;
  static constexpr double HALF_H = CameraTypes::FRAME_HEIGHT / 2.0;

  /// 居中于 native、受传感器边界限制的窗口中心 / Window centre for native, limited by
  /// the sensor edges.
  static CameraTypes::Point2d WindowCentre(const CameraTypes::CameraCalibration& c,
                                           CameraTypes::Point2d native)
  {
    const NarrowPosition p = CameraBase::CenteredOn(c, native);
    return {p.u * (c.native_width - CameraTypes::FRAME_WIDTH) + HALF_W,
            p.v * (c.native_height - CameraTypes::FRAME_HEIGHT) + HALF_H};
  }

  /// 目标中心离窗口边缘不少于 edge_margin / The target centre keeps edge_margin from
  /// the window edges.
  bool Fits(CameraTypes::Point2d centre, CameraTypes::Point2d native) const
  {
    return std::abs(native.x - centre.x) <= (0.5 - s_.edge_margin) * 2.0 * HALF_W &&
           std::abs(native.y - centre.y) <= (0.5 - s_.edge_margin) * 2.0 * HALF_H;
  }

  bool Drifted(CameraTypes::Point2d native) const
  {
    return std::abs(native.x - centre_.x) > s_.recenter_frac * 2.0 * HALF_W ||
           std::abs(native.y - centre_.y) > s_.recenter_frac * 2.0 * HALF_H;
  }

  Request Leave()
  {
    narrow_ = false;
    return {std::nullopt, View::WIDE};
  }

  const ViewSettings s_;
  bool narrow_ = false;
  CameraTypes::Point2d centre_{};  ///< 请求的窗口中心 / Requested window centre
  double last_seen_ = -std::numeric_limits<double>::infinity();
  double last_move_ = -std::numeric_limits<double>::infinity();
};
