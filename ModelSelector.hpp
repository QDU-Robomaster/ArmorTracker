#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

/**
 * @brief 多模型选择（aasim `docs/glr_estimator.md` §7）：各滤波器角点对数似然的指数加权和
 *        加上偏置，可选的滤波器中得分最高者输出，切换需领先 margin。
 *        Multi-model selection (§7): exponentially weighted corner log-likelihoods plus
 *        biases; the best selectable filter reports, and switching needs a margin.
 */
namespace Vehicle
{
/// 四种滤波器 / The four filters.
enum class ModelKind : uint8_t
{
  STEADY,  ///< 静止、匀速陀螺、按键平移 / Still, steady spin, key driving
  VARY,    ///< 变速陀螺 / Varying spin
  AGILE,   ///< 频繁换向的平移 / Frequent direction changes
  HARM,    ///< 周期变速陀螺 / Periodic spin
};

struct SelectorParams
{
  double tau = 0.3, margin = 3.0, spin_min = 2.0, harm_bonus = 5.0, steady_bonus = 3.0;
};

class ModelSelector
{
 public:
  /// 每个滤波器本帧的情况 / One filter's state this frame.
  struct Candidate
  {
    ModelKind kind;
    bool active;    ///< 已完成起步 / Booted
    bool periodic;  ///< 非 HARM 恒为 true / Always true except HARM
    double ll;      ///< 本帧角点对数似然 / This frame's corner log-likelihood
  };

  SelectorParams params;

  /**
   * @brief 更新得分并选择。第 0 个滤波器须为 STEADY。
   *        Update the scores and select. Filter 0 must be STEADY.
   * @param steady_spin STEADY 当前的转速 / The STEADY filter's spin rate
   * @return 输出的滤波器下标 / Index of the reporting filter
   */
  int Select(double t, const std::vector<Candidate>& filters, double steady_spin)
  {
    if (sums_.size() != filters.size())
    {
      sums_.assign(filters.size(), 0.0);
    }
    const double dt = first_ ? 0.0 : t - t_prev_;
    first_ = false;
    t_prev_ = t;
    const double decay = std::exp(-dt / params.tau);
    for (std::size_t i = 0; i < filters.size(); ++i)
    {
      sums_[i] = filters[i].active ? decay * sums_[i] + filters[i].ll : 0.0;
    }

    // 不转的目标上，单块板的横向加速度可被“转速变化”同样解释，所以变速模型只在 STEADY
    // 看到转速时可选。/ Spin models are selectable only while STEADY sees spin.
    const bool spinning = filters[0].active && std::abs(steady_spin) >= params.spin_min;
    std::vector<int> allowed;
    for (std::size_t i = 0; i < filters.size(); ++i)
    {
      const bool spin_model =
          filters[i].kind == ModelKind::VARY || filters[i].kind == ModelKind::HARM;
      if (filters[i].active && (i == 0 || spinning || !spin_model) && filters[i].periodic)
      {
        allowed.push_back(static_cast<int>(i));
      }
    }
    if (allowed.empty())
    {
      return use_;
    }
    std::vector<double> score(filters.size());
    for (std::size_t i = 0; i < filters.size(); ++i)
    {
      score[i] = sums_[i] +
                 (filters[i].kind == ModelKind::HARM ? params.harm_bonus : 0.0) +
                 (i == 0 ? params.steady_bonus : 0.0);
    }
    int best = allowed[0];
    for (int i : allowed)
    {
      if (score[i] > score[best])
      {
        best = i;
      }
    }
    bool current_allowed = false;
    for (int i : allowed)
    {
      current_allowed = current_allowed || i == use_;
    }
    if (!current_allowed || (best != use_ && score[best] > score[use_] + params.margin))
    {
      use_ = best;
    }
    return use_;
  }

 private:
  std::vector<double> sums_;
  int use_ = 0;
  bool first_ = true;
  double t_prev_ = 0;
};
}  // namespace Vehicle
