#include "extended_kalman_filter.hpp"

#include <numeric>

namespace
{
// chi-square 95th percentile for 4D observation residuals.
constexpr double NIS_THRESHOLD = 9.488;
constexpr double NEES_THRESHOLD = 0.711;
}

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const Eigen::VectorXd& x0, const Eigen::MatrixXd& p0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> x_add)
    : x(x0), P(p0), identity_(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())),
      x_add_(std::move(x_add))
{
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::Predict(const Eigen::MatrixXd& f,
                                              const Eigen::MatrixXd& q)
{
  return Predict(f, q,
                 [&](const Eigen::VectorXd& state)
                 {
                   return f * state;
                 });
}

Eigen::VectorXd ExtendedKalmanFilter::Predict(
    const Eigen::MatrixXd& f, const Eigen::MatrixXd& q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> transfer)
{
  P = f * P * f.transpose() + q;
  x = transfer(x);
  return x;
}

Eigen::VectorXd ExtendedKalmanFilter::Update(
    const Eigen::VectorXd& z, const Eigen::MatrixXd& h, const Eigen::MatrixXd& r,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> z_subtract)
{
  return Update(z, h, r,
                [&](const Eigen::VectorXd& state)
                {
                  return h * state;
                },
                std::move(z_subtract));
}

Eigen::VectorXd ExtendedKalmanFilter::Update(
    const Eigen::VectorXd& z, const Eigen::MatrixXd& h, const Eigen::MatrixXd& r,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> observe,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> z_subtract)
{
  const Eigen::VectorXd x_prior = x;
  const Eigen::MatrixXd kalman_gain =
      P * h.transpose() * (h * P * h.transpose() + r).inverse();

  P = (identity_ - kalman_gain * h) * P * (identity_ - kalman_gain * h).transpose() +
      kalman_gain * r * kalman_gain.transpose();
  x = x_add_(x, kalman_gain * z_subtract(z, observe(x)));

  const Eigen::VectorXd residual = z_subtract(z, observe(x));
  const Eigen::MatrixXd s = h * P * h.transpose() + r;
  const double nis = residual.transpose() * s.inverse() * residual;
  const double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  if (nis > NIS_THRESHOLD)
  {
    ++nis_count_;
    data["nis_fail"] = 1.0;
  }
  if (nees > NEES_THRESHOLD)
  {
    ++nees_count_;
    data["nees_fail"] = 1.0;
  }
  ++total_count_;
  last_nis = nis;

  recent_nis_failures.push_back(nis > NIS_THRESHOLD ? 1 : 0);
  if (recent_nis_failures.size() > window_size)
  {
    recent_nis_failures.pop_front();
  }

  const int recent_failures = std::accumulate(recent_nis_failures.begin(),
                                              recent_nis_failures.end(), 0);
  const double recent_rate = recent_nis_failures.empty()
                                 ? 0.0
                                 : static_cast<double>(recent_failures) /
                                       static_cast<double>(recent_nis_failures.size());

  data["residual_yaw"] = residual.size() > 0 ? residual[0] : 0.0;
  data["residual_pitch"] = residual.size() > 1 ? residual[1] : 0.0;
  data["residual_distance"] = residual.size() > 2 ? residual[2] : 0.0;
  data["residual_angle"] = residual.size() > 3 ? residual[3] : 0.0;
  data["nis"] = nis;
  data["nees"] = nees;
  data["recent_nis_failures"] = recent_rate;

  return x;
}
