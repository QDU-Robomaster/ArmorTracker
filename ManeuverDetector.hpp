#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "CornerEkf.hpp"

/**
 * @brief 换档检测（GLR，aasim `docs/glr_estimator.md` §6）：假设过去 window 秒内某一帧，
 *        中心加速度 (ax, ay) 与角加速度 α 发生了阶跃，检验并把估计的阶跃并入状态。
 *        Step detection (GLR, §6): tests whether the centre acceleration (ax, ay) and
 *        the spin acceleration α stepped at some frame within the window and merges the
 *        estimated step into the state.
 */
namespace Vehicle
{
class ManeuverDetector
{
 public:
  using MatD = Eigen::Matrix<double, NX, 3>;

  explicit ManeuverDetector(const FilterParams& p) : p_(p) {}

  /// 预测时：传播候选、加一个本帧候选、丢弃过旧的 / On prediction: propagate the
  /// onsets, add one at this frame, drop the old ones.
  void OnPredict(const Mat& phi, double t_new)
  {
    std::size_t keep = 0;
    for (std::size_t i = 0; i < onsets_.size(); ++i)
    {
      if (t_new - onsets_[i].t > p_.window)
      {
        continue;
      }
      onsets_[keep] = onsets_[i];
      onsets_[keep].D = phi * onsets_[i].D;
      ++keep;
    }
    onsets_.resize(keep);
    Onset o{t_new, MatD::Zero(), Mat3::Zero(), Vec3::Zero()};
    o.D(AX, 0) = o.D(AY, 1) = o.D(ALPHA, 2) = 1;
    onsets_.push_back(o);
  }

  /// 每次观测更新后累积各候选的信息 / Accumulate each onset's information after an
  /// update.
  void OnUpdate(const UpdateTerms& u)
  {
    if (onsets_.empty())
    {
      return;
    }
    const Eigen::Matrix<double, 8, 8> s_inv = u.S.inverse();
    for (auto& o : onsets_)
    {
      const Eigen::Matrix<double, 8, 3> g = u.H * o.D;
      const Eigen::Matrix<double, 8, 3> sg = s_inv * g;
      o.C += g.transpose() * sg;
      o.d += sg.transpose() * u.r;
      o.D -= u.K * g;
    }
  }

  /**
   * @brief 帧末检验：只平移、只自旋、两者三种假设，超过阈值最多者提交（歧义时等待）。
   *        End-of-frame test over translation-only, spin-only and joint steps; the one
   *        furthest above its threshold is committed (waiting when ambiguous).
   * @return 提交了一次阶跃 / A step was committed.
   */
  bool Test(Vec& x, Mat& P, double t)
  {
    if (onsets_.empty())
    {
      return false;
    }
    const Vec2 n = x.segment<2>(CX) / std::max(x.segment<2>(CX).norm(), 1e-6);
    Mat3 prior = Mat3::Zero();
    const auto& s = p_.sig_step;
    prior.topLeftCorner<2, 2>() =
        s[0] * s[0] * (Eigen::Matrix2d::Identity() - n * n.transpose()) +
        s[1] * s[1] * n * n.transpose();
    prior(2, 2) = s[2] * s[2];
    static constexpr std::array<int, 2> TRANSLATION{0, 1};
    static constexpr std::array<int, 1> SPIN{2};
    static constexpr std::array<int, 3> JOINT{0, 1, 2};
    double lr[3] = {0.0, 0.0, 0.0};
    int at[3] = {0, 0, 0};
    Eigen::Matrix<double, 2, 1> g2 = Eigen::Matrix<double, 2, 1>::Zero();
    Eigen::Matrix2d c2 = Eigen::Matrix2d::Zero();
    Eigen::Matrix<double, 1, 1> g1 = Eigen::Matrix<double, 1, 1>::Zero();
    Eigen::Matrix<double, 1, 1> c1 = Eigen::Matrix<double, 1, 1>::Zero();
    Vec3 g3 = Vec3::Zero();
    Mat3 c3 = Mat3::Zero();
    BestOnset<2>(TRANSLATION, prior, lr[0], at[0], g2, c2);
    BestOnset<1>(SPIN, prior, lr[1], at[1], g1, c1);
    BestOnset<3>(JOINT, prior, lr[2], at[2], g3, c3);
    int b = -1;
    for (int i = 0; i < 3; ++i)
    {
      const double thr = p_.threshold[i];
      if (lr[i] > thr && (b < 0 || lr[i] - thr > lr[b] - p_.threshold[b]))
      {
        b = i;
      }
    }
    if (b < 0)
    {
      t_wait_ = -1;
      return false;
    }
    // 只平移与只自旋难分时先等一等 / Wait while translation and spin are ambiguous.
    if (b < 2 && p_.max_wait > 0 && std::abs(lr[0] - lr[1]) < p_.amb_margin)
    {
      if (t_wait_ < 0)
      {
        t_wait_ = t;
      }
      if (t - t_wait_ < p_.max_wait)
      {
        return false;
      }
    }
    t_wait_ = -1;
    if (b == 0)
    {
      Commit<2>(TRANSLATION, at[0], g2, c2, x, P);
    }
    else if (b == 1)
    {
      Commit<1>(SPIN, at[1], g1, c1, x, P);
    }
    else
    {
      Commit<3>(JOINT, at[2], g3, c3, x, P);
    }
    return true;
  }

  void Clear()
  {
    onsets_.clear();
    t_wait_ = -1;
  }

 private:
  struct Onset
  {
    double t;
    MatD D;   ///< 真实响应减滤波器响应 / True minus filter response
    Mat3 C;   ///< 阶跃的信息矩阵 / Information matrix of the step
    Vec3 d;   ///< 信息向量 / Information vector
  };

  template <int M>
  void BestOnset(const std::array<int, M>& ix, const Mat3& prior, double& lr_best, int& j_best,
                 Eigen::Matrix<double, M, 1>& g_best, Eigen::Matrix<double, M, M>& cm_best) const
  {
    using MM = Eigen::Matrix<double, M, M>;
    using VM = Eigen::Matrix<double, M, 1>;
    MM s;
    for (int a = 0; a < M; ++a)
    {
      for (int b = 0; b < M; ++b)
      {
        s(a, b) = prior(ix[a], ix[b]);
      }
    }
    const MM s_inv = s.inverse();
    lr_best = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < onsets_.size(); ++j)
    {
      MM c;
      VM dv;
      for (int a = 0; a < M; ++a)
      {
        dv(a) = onsets_[j].d(ix[a]);
        for (int b = 0; b < M; ++b)
        {
          c(a, b) = onsets_[j].C(ix[a], ix[b]);
        }
      }
      const MM cm = c + s_inv;
      const VM g = cm.partialPivLu().solve(dv);
      const double lr = dv.dot(g) - std::log((MM::Identity() + s * c).determinant());
      if (lr > lr_best)
      {
        lr_best = lr;
        j_best = static_cast<int>(j);
        g_best = g;
        cm_best = cm;
      }
    }
  }

  template <int M>
  void Commit(const std::array<int, M>& ix, int j, const Eigen::Matrix<double, M, 1>& g,
              const Eigen::Matrix<double, M, M>& cm, Vec& x, Mat& P)
  {
    Eigen::Matrix<double, NX, M> d;
    for (int a = 0; a < M; ++a)
    {
      d.col(a) = onsets_[j].D.col(ix[a]);
    }
    x += d * g;
    P += d * cm.partialPivLu().solve(d.transpose());
    P = 0.5 * (P + P.transpose()).eval();
    Clear();
  }

  const FilterParams p_;
  std::vector<Onset> onsets_;
  double t_wait_ = -1;
};
}  // namespace Vehicle
