#pragma once

#include <Eigen/Dense>

#include <chrono>

#include "libxr.hpp"

namespace TrackerMath
{
double LimitRad(double angle);

Eigen::Vector3d XyzToYpd(const Eigen::Vector3d& xyz);

Eigen::MatrixXd XyzToYpdJacobian(const Eigen::Vector3d& xyz);

double DeltaTime(const std::chrono::steady_clock::time_point& lhs,
                 const std::chrono::steady_clock::time_point& rhs);

double DeltaTime(const LibXR::MicrosecondTimestamp& lhs,
                 const LibXR::MicrosecondTimestamp& rhs);
}  // namespace TrackerMath
