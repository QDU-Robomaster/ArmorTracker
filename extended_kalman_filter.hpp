#pragma once

#include <Eigen/Dense>

#include <deque>
#include <functional>
#include <map>

class ExtendedKalmanFilter
{
 public:
  Eigen::VectorXd x{};
  Eigen::MatrixXd P{};
  std::map<std::string, double> data{};
  std::deque<int> recent_nis_failures{0};
  size_t window_size{100};
  double last_nis{0.0};

  ExtendedKalmanFilter() = default;
  ExtendedKalmanFilter(
      const Eigen::VectorXd& x0, const Eigen::MatrixXd& p0,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> x_add =
          [](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs)
          {
            return lhs + rhs;
          });

  Eigen::VectorXd Predict(const Eigen::MatrixXd& f, const Eigen::MatrixXd& q);

  Eigen::VectorXd Predict(const Eigen::MatrixXd& f, const Eigen::MatrixXd& q,
                          std::function<Eigen::VectorXd(const Eigen::VectorXd&)> transfer);

  Eigen::VectorXd Update(
      const Eigen::VectorXd& z, const Eigen::MatrixXd& h, const Eigen::MatrixXd& r,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> z_subtract =
          [](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs)
          {
            return lhs - rhs;
          });

  Eigen::VectorXd Update(
      const Eigen::VectorXd& z, const Eigen::MatrixXd& h, const Eigen::MatrixXd& r,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)> observe,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> z_subtract =
          [](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs)
          {
            return lhs - rhs;
          });

 private:
  Eigen::MatrixXd identity_{};
  std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> x_add_{};
  int nees_count_{0};
  int nis_count_{0};
  int total_count_{0};
};
