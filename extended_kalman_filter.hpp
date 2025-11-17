#ifndef ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
#define ARMOR_PROCESSOR__KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <cstdint>
#include <functional>

class ExtendedKalmanFilter
{
 public:
  // enum XVectorIndex : std::uint8_t
  // {
  //   X_CENTER = 0,
  //   V_X_CENTER = 1,
  //   Y_CENTER = 2,
  //   V_Y_CENTER = 3,
  //   Z_ARMOR = 4,
  //   V_Z_ARMOR = 5,
  //   YAW = 6,
  //   V_YAW = 7,
  //   ROBOT_R = 8
  // };

  enum XVectorIndex : std::uint8_t
  {
    X_CENTER = 0,
    V_X_CENTER = 1,
    A_X_CENTER = 2,
    Y_CENTER = 3,
    V_Y_CENTER = 4,
    A_Y_CENTER = 5,
    Z_ARMOR = 6,
    V_Z_ARMOR = 7,
    A_Z_ARMOR = 8,
    YAW = 9,
    V_YAW = 10,
    A_YAW = 11,
    ROBOT_R = 12,
  };

  ExtendedKalmanFilter() = default;

  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  explicit ExtendedKalmanFilter(const VecVecFunc& f, const VecVecFunc& h,
                                const VecMatFunc& j_f, const VecMatFunc& j_h,
                                const VoidMatFunc& u_q, const VecMatFunc& u_r,
                                const Eigen::MatrixXd& P0);

  // Set the initial state
  void SetState(const Eigen::VectorXd& x0);
  Eigen::VectorXd GetState() const;
  void SetStateWithUncertainty(const Eigen::VectorXd& x0, const Eigen::VectorXd& diagP);

  const Eigen::MatrixXd GetCovariance() const;
  Eigen::MatrixXd GetCovariance();

  void SetCovariance(const Eigen::MatrixXd& p);

  void PrintCovariance() const;

  VecVecFunc Observation() const;
  VecVecFunc StateTransition() const;

  void PriToPost();

  // Compute a predicted state
  Eigen::MatrixXd Predict();

  // Update the estimated state based on measurement
  Eigen::MatrixXd Update(const Eigen::VectorXd& z);

 private:
  // Process nonlinear vector function
  VecVecFunc f_;
  // Observation nonlinear vector function
  VecVecFunc h_;
  // Jacobian of f()
  VecMatFunc jacobian_f_;
  Eigen::MatrixXd m_f_;
  // Jacobian of h()
  VecMatFunc jacobian_h_;
  Eigen::MatrixXd m_h_;
  // Process noise covariance matrix
  VoidMatFunc update_q_;
  Eigen::MatrixXd m_q_;
  // Measurement noise covariance matrix
  VecMatFunc update_r_;
  Eigen::MatrixXd m_r_;

  // Priori error estimate covariance matrix
  Eigen::MatrixXd p_pri_;
  // Posteriori error estimate covariance matrix
  Eigen::MatrixXd p_post_;

  // Kalman gain
  Eigen::MatrixXd m_k_;

  // System dimensions
  int n_;

  // N-size identity
  Eigen::MatrixXd i_;

  // Priori state
  Eigen::VectorXd x_pri_;
  // Posteriori state
  Eigen::VectorXd x_post_;
};

#endif  // ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
