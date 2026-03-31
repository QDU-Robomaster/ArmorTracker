#pragma once

#include <Eigen/Dense>

#include <chrono>
#include <vector>

#include "armor.hpp"
#include "extended_kalman_filter.hpp"
#include "libxr.hpp"

struct TrackedArmorObservation
{
  ArmorDetectorResult result{};
  Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypr_in_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypd_in_world{Eigen::Vector3d::Zero()};
  double raw_yaw_in_world{0.0};
  double yaw_optimization_delta{0.0};
};

class Target
{
 public:
  ArmorNumber number{ArmorNumber::INVALID};
  ArmorType armor_type{ArmorType::INVALID};
  ArmorPriority priority{ArmorPriority::FIFTH};
  bool jumped{false};
  int last_id{0};

  Target() = default;
  Target(const TrackedArmorObservation& armor,
         LibXR::MicrosecondTimestamp timestamp, double radius,
         int armor_count, const Eigen::VectorXd& p0_diag);

  void Predict(LibXR::MicrosecondTimestamp timestamp);
  void Predict(double dt);
  bool Update(const TrackedArmorObservation& armor);

  const ExtendedKalmanFilter& GetEkf() const;
  const Eigen::VectorXd& GetState() const;
  std::vector<Eigen::Vector4d> GetArmorXYZAList() const;

  bool Diverged() const;
  bool Converged();
  int GetArmorCount() const;

 private:
  int armor_count_{4};
  int switch_count_{0};
  int update_count_{0};
  bool is_switch_{false};
  bool is_converged_{false};
  ExtendedKalmanFilter ekf_{};
  LibXR::MicrosecondTimestamp timestamp_{};

  void UpdateYpda(const TrackedArmorObservation& armor, int id);
  Eigen::Vector3d GetArmorPosition(const Eigen::VectorXd& state, int id) const;
  Eigen::MatrixXd GetObservationJacobian(const Eigen::VectorXd& state, int id) const;
};
