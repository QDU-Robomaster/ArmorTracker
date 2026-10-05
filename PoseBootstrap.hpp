#pragma once

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "CornerEkf.hpp"

/**
 * @brief PnP 多假设起步（aasim `docs/glr_estimator.md` §5）：用最宽检测的 IPPE 位姿给出
 *        初始状态，按若干转速假设各起一个滤波器，boot_time 秒后似然最高者胜出。
 *        PnP multi-hypothesis boot (§5): the widest detection's IPPE pose gives the
 *        initial state, one filter starts per spin-rate hypothesis, and the most likely
 *        one wins after boot_time seconds.
 */
namespace Vehicle
{
class PoseBootstrap
{
 public:
  /// 起步时长 / Boot duration.
  static constexpr double BOOT_TIME = 0.3;

  bool Running() const { return !hypotheses_.empty(); }
  void Clear() { hypotheses_.clear(); }

  /**
   * @brief 起步一帧。第一次调用时建立假设，之后预测并更新所有假设。
   *        One boot frame. The first call creates the hypotheses, later calls predict
   *        and update all of them.
   * @param ekf 提供模型与相机；其 x、P 被设为当前最好的假设 / Provides the model; its
   *            x and P are set to the best hypothesis.
   * @param ll_frame 最后一个假设本帧的似然（与原型一致）/ The last hypothesis's
   *                 likelihood of this frame (as in the prototype).
   * @return 本帧是否有输出（还没有任何假设且无法建立时为 false）/ Whether this frame
   *         reports a state.
   */
  bool Frame(CornerEkf& ekf, const Camera& cam, double t, const Mat3& r_bw,
             const std::vector<Detection>& dets, double& ll_frame)
  {
    double dt = 0.0;
    if (hypotheses_.empty())
    {
      if (dets.empty())
      {
        return false;
      }
      Vec x0;
      Mat p0;
      if (!InitialState(Widest(dets), cam, r_bw, ekf.Params(), x0, p0))
      {
        return false;
      }
      for (double w : SPIN_HYPOTHESES)
      {
        Hypothesis h{x0, p0, 0.0};
        h.x(OMEGA) = w;
        hypotheses_.push_back(h);
      }
      t0_ = t_ = t;
    }
    else
    {
      dt = t - t_;
      t_ = t;
    }
    for (auto& h : hypotheses_)
    {
      ekf.PredictPlain(h.x, h.P, dt);
      ll_frame = 0.0;
      for (const auto& d : dets)
      {
        UpdateTerms u;
        if (ekf.Update(h.x, h.P, d, r_bw, u))
        {
          ll_frame += u.ll;
        }
      }
      h.ll += ll_frame;
    }
    const Hypothesis& best = *std::max_element(
        hypotheses_.begin(), hypotheses_.end(),
        [](const Hypothesis& a, const Hypothesis& b) { return a.ll < b.ll; });
    ekf.x = best.x;
    ekf.P = best.P;
    return true;
  }

  /// 起步是否已满 BOOT_TIME / Whether the boot has lasted BOOT_TIME.
  bool Done(double t) const { return t - t0_ >= BOOT_TIME; }

 private:
  struct Hypothesis
  {
    Vec x;
    Mat P;
    double ll;
  };

  /// 转速假设；头几帧似然相同时取排在前面的（转得慢的）/ Spin hypotheses; ties in the
  /// first frames go to the earlier (slower) ones.
  static constexpr std::array<double, 7> SPIN_HYPOTHESES = {0, -8, 8, -16, 16, -24, 24};

  static const Detection& Widest(const std::vector<Detection>& dets)
  {
    const Detection* widest = &dets[0];
    double width = -1;
    for (const auto& d : dets)
    {
      double lo = 1e18, hi = -1e18;
      for (const auto& c : d.corners)
      {
        lo = std::min(lo, c.x());
        hi = std::max(hi, c.x());
      }
      if (hi - lo > width)
      {
        width = hi - lo;
        widest = &d;
      }
    }
    return *widest;
  }

  /// IPPE 两个解中重投影误差小的一个给出板位姿 / The better IPPE solution's pose.
  static bool InitialState(const Detection& d, const Camera& cam, const Mat3& r_bw,
                           const FilterParams& p, Vec& x, Mat& cov)
  {
    const auto object = ObjectPoints(d.type, p.shape);
    std::vector<cv::Point3d> obj;
    std::vector<cv::Point2d> img;
    for (int k = 0; k < 4; ++k)
    {
      obj.emplace_back(object[k].x(), object[k].y(), object[k].z());
      img.emplace_back(d.corners[k].x(), d.corners[k].y());
    }
    const cv::Mat k_mat =
        (cv::Mat_<double>(3, 3) << cam.fx, 0, cam.cx, 0, cam.fy, cam.cy, 0, 0, 1);
    const cv::Mat d_mat = (cv::Mat_<double>(1, 5) << cam.dist[0], cam.dist[1],
                           cam.dist[2], cam.dist[3], cam.dist[4]);
    std::vector<cv::Mat> rv, tv;
    cv::Mat err;
    const int n =
        cv::solvePnPGeneric(obj, img, k_mat, d_mat, rv, tv, false, cv::SOLVEPNP_IPPE,
                            cv::noArray(), cv::noArray(), err);
    if (n < 1)
    {
      return false;
    }
    int j = 0;
    for (int i = 1; i < n; ++i)
    {
      if (err.at<double>(i) < err.at<double>(j))
      {
        j = i;
      }
    }
    cv::Mat rc;
    cv::Rodrigues(rv[j], rc);
    Mat3 r_co;
    Vec3 tc;
    for (int r = 0; r < 3; ++r)
    {
      tc(r) = tv[j].at<double>(r);
      for (int c = 0; c < 3; ++c)
      {
        r_co(r, c) = rc.at<double>(r, c);
      }
    }
    const Mat3 optical_to_world = r_bw * cam.R_cb;
    const Vec3 plate = r_bw * (cam.R_cb * tc + cam.t_cb);
    const Vec3 normal = optical_to_world * r_co.col(0);  // 向内的法向 / Inward normal
    const double a = std::atan2(-normal.x(), normal.y());
    x = Vec::Zero();
    x(CX) = plate.x() - p.r0 * std::sin(a);
    x(CY) = plate.y() + p.r0 * std::cos(a);
    x(YAW) = a;
    x(CZ) = plate.z();
    x(R_EVEN) = x(R_ODD) = p.r0;
    x(SCALE_W) = x(SCALE_H) = 1.0;
    Vec diag;
    diag << 0.01, 0.01, 1, 1, 4, 4, 0.05 * 0.05, 9, 100, 1e-4, 0.0025, 0.0025, 0.0009,
        p.sig_scale * p.sig_scale, p.sig_scale * p.sig_scale;
    cov = diag.asDiagonal();
    return true;
  }

  std::vector<Hypothesis> hypotheses_;
  double t0_ = 0;
  double t_ = 0;
};
}  // namespace Vehicle
