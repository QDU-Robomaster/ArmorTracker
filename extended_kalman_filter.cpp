#include "extended_kalman_filter.hpp"

#include <array>

/*
f:过程函数
h:观测函数
j_f:过程函数的雅可比矩阵
j_h:测量函数的雅可比矩阵
u_q:过程噪声协方差矩阵
u_r:测量噪声协方差矩阵
P0:初始状态协方差矩阵
*/
ExtendedKalmanFilter::ExtendedKalmanFilter(const VecVecFunc& f, const VecVecFunc& h,
                                           const VecMatFunc& j_f, const VecMatFunc& j_h,
                                           const VoidMatFunc& u_q, const VecMatFunc& u_r,
                                           const Eigen::MatrixXd& P0)
    : f_(f),
      h_(h),
      jacobian_f_(j_f),
      jacobian_h_(j_h),
      update_q_(u_q),
      update_r_(u_r),
      p_post_(P0),
      n_(P0.rows()),
      i_(Eigen::MatrixXd::Identity(n_, n_)),
      x_pri_(n_),
      x_post_(n_)
{
}

void ExtendedKalmanFilter::SetState(const Eigen::VectorXd& x0)
{
  x_post_ = x0;
  x_pri_ = x0;
}

Eigen::MatrixXd ExtendedKalmanFilter::Predict()
{
  m_f_ = jacobian_f_(x_post_), m_q_ = update_q_();

  x_pri_ = f_(x_post_);
  p_pri_ = m_f_ * p_post_ * m_f_.transpose() + m_q_;

  // handle the case when there will be no measurement before the next predict
  x_post_ = x_pri_;
  p_post_ = p_pri_;

  return x_pri_;
}

Eigen::MatrixXd ExtendedKalmanFilter::Update(const Eigen::VectorXd& z)
{
  m_h_ = jacobian_h_(x_pri_), m_r_ = update_r_(z);

  m_k_ = p_pri_ * m_h_.transpose() *
         (m_h_ * p_pri_ * m_h_.transpose() + m_r_).inverse();  // inverse计算逆矩阵

  const Eigen::VectorXd innovation = z - h_(x_pri_);
  const Eigen::MatrixXd correction = i_ - m_k_ * m_h_;
  x_post_ = x_pri_ + m_k_ * innovation;
  p_post_ = correction * p_pri_ * correction.transpose() + m_k_ * m_r_ * m_k_.transpose();

  return x_post_;
}

void ExtendedKalmanFilter::DecorrelatePosterior(
    std::initializer_list<int> state_indices)
{
  std::array<bool, 64> selected{};
  if (n_ > static_cast<int>(selected.size()))
  {
    return;
  }

  for (const int state_index : state_indices)
  {
    if (state_index >= 0 && state_index < n_)
    {
      selected[static_cast<std::size_t>(state_index)] = true;
    }
  }

  for (int row = 0; row < n_; ++row)
  {
    if (!selected[static_cast<std::size_t>(row)])
    {
      continue;
    }
    for (int col = 0; col < n_; ++col)
    {
      if (selected[static_cast<std::size_t>(col)])
      {
        continue;
      }
      p_post_(row, col) = 0.0;
      p_post_(col, row) = 0.0;
    }
  }
}
