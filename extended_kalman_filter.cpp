#include "extended_kalman_filter.hpp"

#include "Eigen/Core"

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
      n_(static_cast<int>(P0.rows())),
      i_(Eigen::MatrixXd::Identity(n_, n_)),
      x_pri_(n_),
      x_post_(n_)
{
}

void ExtendedKalmanFilter::PrintCovariance() const
{
  printf("P_post:\n");
  for (int i = 0; i < n_; i++)
  {
    for (int j = 0; j < n_; j++)
    {
      printf("%.4f ", p_post_(i, j));
    }
    printf("\n");
  }
}

void ExtendedKalmanFilter::SetState(const Eigen::VectorXd& x0) { x_post_ = x0; }
Eigen::VectorXd ExtendedKalmanFilter::GetState() const { return x_post_; }
void ExtendedKalmanFilter::SetStateWithUncertainty(const Eigen::VectorXd& x0,
                                                   const Eigen::VectorXd& diagP)
{
  x_post_ = x0;
  for (int i = 0; i < diagP.size(); ++i)
  {
    p_post_(i, i) = diagP(i);
  }
}

const Eigen::MatrixXd ExtendedKalmanFilter::GetCovariance() const { return p_post_; }
Eigen::MatrixXd ExtendedKalmanFilter::GetCovariance() { return p_post_; }
void ExtendedKalmanFilter::SetCovariance(const Eigen::MatrixXd& p) { p_post_ = p; }

ExtendedKalmanFilter::VecVecFunc ExtendedKalmanFilter::Observation() const { return h_; }
ExtendedKalmanFilter::VecVecFunc ExtendedKalmanFilter::StateTransition() const
{
  return f_;
}

void ExtendedKalmanFilter::PriToPost()
{
  x_post_ = x_pri_;
  p_post_ = p_pri_;
}

bool ExtendedKalmanFilter::GateMeasurement(const Eigen::VectorXd& z,
                                           double gate_threshold, double* out_d2) const
{
  Eigen::MatrixXd h = jacobian_h_(x_pri_);
  Eigen::MatrixXd r = update_r_(z);
  Eigen::MatrixXd s = h * p_pri_ * h.transpose() + r;

  Eigen::LLT<Eigen::MatrixXd> llt(s);
  if (llt.info() != Eigen::Success)
  {
    return false;
  }

  Eigen::VectorXd innov = z - h_(x_pri_);
  // w = S^{-1} * innov
  Eigen::VectorXd w = llt.solve(innov);
  double d2 = innov.dot(w);

  if (out_d2)
  {
    *out_d2 = d2;
  }

  return d2 < gate_threshold;
}

Eigen::MatrixXd ExtendedKalmanFilter::Predict()
{
  m_f_ = jacobian_f_(x_post_), m_q_ = update_q_();

  x_pri_ = f_(x_post_);
  p_pri_ = m_f_ * p_post_ * m_f_.transpose() + m_q_;

  // handle the case when there will be no measurement before the next predict
  // x_post_ = x_pri_;
  // p_post_ = p_pri_;

  return x_pri_;
}

Eigen::MatrixXd ExtendedKalmanFilter::Update(const Eigen::VectorXd& z)
{
  m_h_ = jacobian_h_(x_pri_);
  m_r_ = update_r_(z);

  Eigen::MatrixXd s = m_h_ * p_pri_ * m_h_.transpose() + m_r_;  // 创新协方差
  Eigen::LLT<Eigen::MatrixXd> llt(s);

  if (llt.info() != Eigen::Success)
  {
    // printf("LLT decomposition failed!\n");
    return x_post_;
  }

  Eigen::MatrixXd ph_t = p_pri_ * m_h_.transpose();
  Eigen::MatrixXd k = llt.solve(ph_t.transpose()).transpose();

  m_k_ = k;
  x_post_ = x_pri_ + m_k_ * (z - h_(x_pri_));

  Eigen::MatrixXd i_kh = i_ - m_k_ * m_h_;
  p_post_ = i_kh * p_pri_ * i_kh.transpose() + m_k_ * m_r_ * m_k_.transpose();
  p_post_ = 0.5 * (p_post_ + p_post_.transpose());  // 对称化

  return x_post_;
}
