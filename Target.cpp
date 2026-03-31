#include "Target.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include <opencv2/core.hpp>

#include "TrackerMath.hpp"

Target::Target(const TrackedArmorObservation& armor,
               LibXR::MicrosecondTimestamp timestamp, double radius,
               int armor_count, const Eigen::VectorXd& p0_diag)
    : number(armor.result.number), armor_type(armor.result.type),
      priority(armor.result.priority), armor_count_(armor_count), timestamp_(timestamp)
{
  const Eigen::Vector3d& xyz = armor.xyz_in_world;
  const Eigen::Vector3d& ypr = armor.ypr_in_world;

  const double center_x = xyz.x() + radius * std::cos(ypr.x());
  const double center_y = xyz.y() + radius * std::sin(ypr.x());
  const double center_z = xyz.z();

  Eigen::VectorXd x0(11);
  x0 << center_x, 0.0, center_y, 0.0, center_z, 0.0, ypr.x(), 0.0, radius, 0.0,
      0.0;
  const Eigen::MatrixXd p0 = p0_diag.asDiagonal();

  ekf_ = ExtendedKalmanFilter(
      x0, p0,
      [](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs)
      {
        Eigen::VectorXd result = lhs + rhs;
        result[6] = TrackerMath::LimitRad(result[6]);
        return result;
      });
}

void Target::Predict(LibXR::MicrosecondTimestamp timestamp)
{
  Predict(TrackerMath::DeltaTime(timestamp, timestamp_));
  timestamp_ = timestamp;
}

void Target::Predict(double dt)
{
  Eigen::MatrixXd f = Eigen::MatrixXd::Identity(11, 11);
  f(0, 1) = dt;
  f(2, 3) = dt;
  f(4, 5) = dt;
  f(6, 7) = dt;

  double linear_variance = 100.0;
  double angular_variance = 400.0;
  if (number == ArmorNumber::OUTPOST)
  {
    linear_variance = 10.0;
    angular_variance = 0.1;
  }

  const double a = dt * dt * dt * dt / 4.0;
  const double b = dt * dt * dt / 2.0;
  const double c = dt * dt;

  Eigen::MatrixXd q = Eigen::MatrixXd::Zero(11, 11);
  q(0, 0) = a * linear_variance;
  q(0, 1) = b * linear_variance;
  q(1, 0) = b * linear_variance;
  q(1, 1) = c * linear_variance;
  q(2, 2) = a * linear_variance;
  q(2, 3) = b * linear_variance;
  q(3, 2) = b * linear_variance;
  q(3, 3) = c * linear_variance;
  q(4, 4) = a * linear_variance;
  q(4, 5) = b * linear_variance;
  q(5, 4) = b * linear_variance;
  q(5, 5) = c * linear_variance;
  q(6, 6) = a * angular_variance;
  q(6, 7) = b * angular_variance;
  q(7, 6) = b * angular_variance;
  q(7, 7) = c * angular_variance;

  ekf_.Predict(
      f, q,
      [&](const Eigen::VectorXd& state)
      {
        Eigen::VectorXd prior = f * state;
        prior[6] = TrackerMath::LimitRad(prior[6]);
        return prior;
      });

  if (Converged() && number == ArmorNumber::OUTPOST && std::abs(ekf_.x[7]) > 2.0)
  {
    ekf_.x[7] = ekf_.x[7] > 0.0 ? 2.51 : -2.51;
  }
}

bool Target::Update(const TrackedArmorObservation& armor)
{
  int best_id = 0;
  double min_angle_error = 1e10;
  const auto armor_xyza_list = GetArmorXYZAList();

  std::vector<std::pair<Eigen::Vector4d, int>> sorted_candidates;
  sorted_candidates.reserve(armor_xyza_list.size());
  for (int index = 0; index < armor_count_; ++index)
  {
    sorted_candidates.push_back({armor_xyza_list[index], index});
  }

  std::sort(sorted_candidates.begin(), sorted_candidates.end(),
            [](const std::pair<Eigen::Vector4d, int>& lhs,
               const std::pair<Eigen::Vector4d, int>& rhs)
            {
              return TrackerMath::XyzToYpd(lhs.first.head(3)).z() <
                     TrackerMath::XyzToYpd(rhs.first.head(3)).z();
            });

  const int search_count = std::min(3, static_cast<int>(sorted_candidates.size()));
  for (int index = 0; index < search_count; ++index)
  {
    const auto& xyza = sorted_candidates[index].first;
    const Eigen::Vector3d ypd = TrackerMath::XyzToYpd(xyza.head(3));
    const double angle_error =
        std::abs(TrackerMath::LimitRad(armor.ypr_in_world.x() - xyza[3])) +
        std::abs(TrackerMath::LimitRad(armor.ypd_in_world.x() - ypd.x()));

    if (std::abs(angle_error) < std::abs(min_angle_error))
    {
      best_id = sorted_candidates[index].second;
      min_angle_error = angle_error;
    }
  }

  const bool did_jump = (best_id != 0);
  is_switch_ = (best_id != last_id);
  if (is_switch_)
  {
    ++switch_count_;
  }

  last_id = best_id;
  ++update_count_;
  UpdateYpda(armor, best_id);
  return did_jump;
}

const ExtendedKalmanFilter& Target::GetEkf() const { return ekf_; }

const Eigen::VectorXd& Target::GetState() const { return ekf_.x; }

std::vector<Eigen::Vector4d> Target::GetArmorXYZAList() const
{
  std::vector<Eigen::Vector4d> armors;
  armors.reserve(armor_count_);
  for (int index = 0; index < armor_count_; ++index)
  {
    const double angle =
        TrackerMath::LimitRad(ekf_.x[6] + index * 2.0 * CV_PI / armor_count_);
    const Eigen::Vector3d xyz = GetArmorPosition(ekf_.x, index);
    armors.push_back({xyz.x(), xyz.y(), xyz.z(), angle});
  }
  return armors;
}

bool Target::Diverged() const
{
  const bool radius_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  const bool length_ok = (ekf_.x[8] + ekf_.x[9]) > 0.05 && (ekf_.x[8] + ekf_.x[9]) < 0.5;
  return !(radius_ok && length_ok);
}

bool Target::Converged()
{
  if (number == ArmorNumber::OUTPOST)
  {
    if (update_count_ > 10 && !Diverged())
    {
      is_converged_ = true;
    }
  }
  else if (update_count_ > 3 && !Diverged())
  {
    is_converged_ = true;
  }
  return is_converged_;
}

int Target::GetArmorCount() const { return armor_count_; }

void Target::UpdateYpda(const TrackedArmorObservation& armor, int id)
{
  const Eigen::MatrixXd h = GetObservationJacobian(ekf_.x, id);
  const double center_yaw = std::atan2(armor.xyz_in_world.y(), armor.xyz_in_world.x());
  const double delta_angle = TrackerMath::LimitRad(armor.ypr_in_world.x() - center_yaw);
  constexpr double ARMOR_YAW_NOISE_BASE = 9e-2;

  Eigen::VectorXd r_diag(4);
  r_diag << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1.0) + 1.0,
      std::log(std::abs(armor.ypd_in_world.z()) + 1.0) / 200.0 +
          ARMOR_YAW_NOISE_BASE;
  const Eigen::MatrixXd r = r_diag.asDiagonal();

  const Eigen::VectorXd z =
      (Eigen::Vector4d() << armor.ypd_in_world.x(), armor.ypd_in_world.y(),
       armor.ypd_in_world.z(), armor.ypr_in_world.x())
          .finished();

  ekf_.Update(
      z, h, r,
      [&](const Eigen::VectorXd& state)
      {
        const Eigen::Vector3d xyz = GetArmorPosition(state, id);
        const Eigen::Vector3d ypd = TrackerMath::XyzToYpd(xyz);
        const double angle = TrackerMath::LimitRad(state[6] + id * 2.0 * CV_PI / armor_count_);
        return (Eigen::Vector4d() << ypd.x(), ypd.y(), ypd.z(), angle).finished();
      },
      [](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs)
      {
        Eigen::VectorXd result = lhs - rhs;
        result[0] = TrackerMath::LimitRad(result[0]);
        result[1] = TrackerMath::LimitRad(result[1]);
        result[3] = TrackerMath::LimitRad(result[3]);
        return result;
      });
}

Eigen::Vector3d Target::GetArmorPosition(const Eigen::VectorXd& state, int id) const
{
  const double angle = TrackerMath::LimitRad(state[6] + id * 2.0 * CV_PI / armor_count_);
  const bool use_length_height = (armor_count_ == 4) && (id == 1 || id == 3);
  const double radius = use_length_height ? state[8] + state[9] : state[8];
  const double armor_x = state[0] - radius * std::cos(angle);
  const double armor_y = state[2] - radius * std::sin(angle);
  const double armor_z = use_length_height ? state[4] + state[10] : state[4];
  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::GetObservationJacobian(const Eigen::VectorXd& state, int id) const
{
  const double angle = TrackerMath::LimitRad(state[6] + id * 2.0 * CV_PI / armor_count_);
  const bool use_length_height = (armor_count_ == 4) && (id == 1 || id == 3);
  const double radius = use_length_height ? state[8] + state[9] : state[8];
  const double dx_da = radius * std::sin(angle);
  const double dy_da = -radius * std::cos(angle);
  const double dx_dr = -std::cos(angle);
  const double dy_dr = -std::sin(angle);
  const double dx_dl = use_length_height ? -std::cos(angle) : 0.0;
  const double dy_dl = use_length_height ? -std::sin(angle) : 0.0;
  const double dz_dh = use_length_height ? 1.0 : 0.0;

  Eigen::MatrixXd h_armor_xyza = Eigen::MatrixXd::Zero(4, 11);
  h_armor_xyza(0, 0) = 1.0;
  h_armor_xyza(0, 6) = dx_da;
  h_armor_xyza(0, 8) = dx_dr;
  h_armor_xyza(0, 9) = dx_dl;
  h_armor_xyza(1, 2) = 1.0;
  h_armor_xyza(1, 6) = dy_da;
  h_armor_xyza(1, 8) = dy_dr;
  h_armor_xyza(1, 9) = dy_dl;
  h_armor_xyza(2, 4) = 1.0;
  h_armor_xyza(2, 10) = dz_dh;
  h_armor_xyza(3, 6) = 1.0;

  const Eigen::MatrixXd h_armor_ypd =
      TrackerMath::XyzToYpdJacobian(GetArmorPosition(state, id));
  Eigen::MatrixXd h_armor_ypda(4, 4);
  h_armor_ypda << h_armor_ypd(0, 0), h_armor_ypd(0, 1), h_armor_ypd(0, 2), 0,
      h_armor_ypd(1, 0), h_armor_ypd(1, 1), h_armor_ypd(1, 2), 0,
      h_armor_ypd(2, 0), h_armor_ypd(2, 1), h_armor_ypd(2, 2), 0, 0, 0, 0, 1;

  return h_armor_ypda * h_armor_xyza;
}
