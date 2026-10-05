// 合成场景测试：整车估计器收敛、多目标选择与换目标、按编号定大小板、基地兜底、模块接线。
//
// Synthetic tests: vehicle estimator convergence, target selection and switching, plate
// size by number, the base fallback, and the Module wiring.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "ArmorTracker.hpp"
#include "libxr.hpp"

namespace
{
void Expect(bool condition, const char* message)
{
  if (!condition)
  {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

const CameraTypes::CameraCalibration CALIBRATION{
    1440, 1080, 2328.0, 2328.0, 720.0, 540.0, {0.0, 0.0, 0.0, 0.0, 0.0}};

Vehicle::Camera MakeCamera()
{
  Vehicle::Camera cam;
  cam.fx = cam.fy = 2328.0;
  cam.cx = 720.0;
  cam.cy = 540.0;
  return cam;
}

/// 真值整车：中心、转速、两个半径、高度差 / A ground-truth vehicle.
struct Truth
{
  Vehicle::Vec3 centre;
  double yaw0;
  double omega;
  double r_even = 0.25, r_odd = 0.22, dz = 0.05;
  bool large = false;
  int plates = 4;  ///< 基地为 3 / 3 for the base
  Vehicle::PlateShape shape = LIGHTBAR4_SHAPE;

  Vehicle::Vec StateAt(double t) const
  {
    Vehicle::Vec x = Vehicle::Vec::Zero();
    x(Vehicle::CX) = centre.x();
    x(Vehicle::CY) = centre.y();
    x(Vehicle::CZ) = centre.z();
    x(Vehicle::YAW) = yaw0 + omega * t;
    x(Vehicle::OMEGA) = omega;
    x(Vehicle::R_EVEN) = r_even;
    x(Vehicle::R_ODD) = r_odd;
    x(Vehicle::DZ) = dz;
    x(Vehicle::SCALE_W) = x(Vehicle::SCALE_H) = 1.0;
    return x;
  }

  /// 正对相机的板的角点（估计器顺序），加像素噪声 / Corners of the facing plates.
  std::vector<std::array<Vehicle::Vec2, 4>> Visible(double t, std::mt19937& rng) const
  {
    std::normal_distribution<double> noise(0.0, 0.3);
    const Vehicle::Vec x = StateAt(t);
    const Vehicle::Camera cam = MakeCamera();
    const auto object = Vehicle::ObjectPoints(large ? 1 : 0, shape);
    std::vector<std::array<Vehicle::Vec2, 4>> out;
    for (int k = 0; k < plates; ++k)
    {
      const double a = x(Vehicle::YAW) + k * 2.0 * M_PI / plates;
      const bool odd = plates == 4 && k % 2 == 1;
      const double r = odd ? r_odd : r_even;
      const Vehicle::Vec3 c(centre.x() + r * std::sin(a), centre.y() - r * std::cos(a),
                            centre.z() + (odd ? dz : 0.0));
      const Vehicle::Vec2 n(std::sin(a), -std::cos(a));
      if (n.dot(-c.head<2>() / c.head<2>().norm()) < 0.3)
      {
        continue;
      }
      const Vehicle::Mat3 rot = Vehicle::ArmorRotation(a);
      std::array<Vehicle::Vec3, 4> points;
      for (int i = 0; i < 4; ++i)
      {
        points[i] = c + rot * object[i];
      }
      const Vehicle::Pix p = Vehicle::Project(cam, Vehicle::Mat3::Identity(), points);
      std::array<Vehicle::Vec2, 4> corners;
      for (int i = 0; i < 4; ++i)
      {
        corners[i] = {p(2 * i) + noise(rng), p(2 * i + 1) + noise(rng)};
      }
      out.push_back(corners);
    }
    return out;
  }
};

/// 估计器顺序（左上、右上、右下、左下）转为 AutoAim 顺序（左上、左下、右下、右上）。
AutoAim::Armor MakeArmor(ArmorNumber number, bool large,
                         const std::array<Vehicle::Vec2, 4>& c)
{
  AutoAim::Armor a{
      ArmorColor::RED, number, large ? ArmorType::LARGE : ArmorType::SMALL, 0.9F, {}};
  const int order[4] = {0, 3, 2, 1};
  for (int k = 0; k < 4; ++k)
  {
    a.corners[k] = {static_cast<float>(c[order[k]].x()),
                    static_cast<float>(c[order[k]].y())};
  }
  return a;
}

TrackerSettings Settings(const char* camera)
{
  return {camera, {1, 0, 0, 0}, {0, 0, 0}, -1, 2, 15, 75, 0.07, 23.0, SelectWeights{}};
}

void TestVehicleEstimatorConverges()
{
  const Truth truth{{0.3, 4.0, 0.1}, 0.4, 4.0};
  Vehicle::VehicleEstimator est(MakeCamera(), Vehicle::RateMode::AUTO, LIGHTBAR4_SHAPE);
  std::mt19937 rng(1);
  Vehicle::VehicleTarget out;
  for (int i = 0; i <= 200; ++i)
  {
    const double t = 0.01 * i;
    std::vector<Vehicle::Detection> dets;
    for (const auto& c : truth.Visible(t, rng))
    {
      dets.push_back({c, 0});
    }
    out = est.Step(t, {1, 0, 0, 0}, dets, 0.2);
  }
  std::printf("vehicle: centre err %.4f m, v_yaw %.3f rad/s, r1 %.3f r2 %.3f\n",
              (out.position - truth.centre).norm(), out.v_yaw, out.radius_1,
              out.radius_2);
  Expect(out.tracking, "tracking after 2 s");
  Expect((out.position.head<2>() - truth.centre.head<2>()).norm() < 0.02,
         "centre within 2 cm");
  Expect(std::abs(out.v_yaw - truth.omega) < 0.3, "spin within 0.3 rad/s");
}

void TestSelectionAndSwitching()
{
  TrackSet set(Settings("sel"));
  const Truth near{{0.0, 4.0, 0.1}, 0.2, 0.0};
  const Truth far{{1.5, 7.0, 0.1}, 0.0, 0.0};
  std::mt19937 rng(2);
  ArmorTrackerTarget out;
  for (int i = 0; i < 60; ++i)
  {
    const double t = 0.01 * i;
    std::vector<AutoAim::Armor> armors;
    if (i < 30)
    {
      for (const auto& c : near.Visible(t, rng))
      {
        armors.push_back(MakeArmor(ArmorNumber::THREE, false, c));
      }
    }
    for (const auto& c : far.Visible(t, rng))
    {
      armors.push_back(MakeArmor(ArmorNumber::FOUR, false, c));
    }
    out = set.Step(1000000 + 10000ULL * i, {1, 0, 0, 0}, CALIBRATION, armors);
    if (i == 29)
    {
      Expect(out.tracking && out.id == ArmorNumber::THREE, "the nearer target is chosen");
      Expect(out.armors_num == 4, "four-plate vehicle");
    }
  }
  // 3 号消失 15 帧后判丢失，换到 4 号 / THREE is lost after 15 frames, FOUR takes over.
  Expect(out.tracking && out.id == ArmorNumber::FOUR, "switch to the remaining target");
  Expect(std::abs(out.position.y() - far.centre.y()) < 0.1, "far target position");
}

void TestSizeByNumber()
{
  // 检测器把 3 号报成大板：仍按四块小板的整车跟踪 / The detector reports number three as
  // large: it is still tracked as a four-plate vehicle with small plates.
  TrackSet set(Settings("size"));
  const Truth infantry{{0.0, 3.0, 0.15}, 0.2, 3.0};
  std::mt19937 rng(3);
  ArmorTrackerTarget out;
  for (int i = 0; i < 100; ++i)
  {
    std::vector<AutoAim::Armor> armors;
    for (const auto& c : infantry.Visible(0.01 * i, rng))
    {
      armors.push_back(MakeArmor(ArmorNumber::THREE, true, c));
    }
    out = set.Step(10000ULL * i, {1, 0, 0, 0}, CALIBRATION, armors);
  }
  std::printf("size: armors_num %d centre err %.4f m\n", out.armors_num,
              (out.position.head<2>() - infantry.centre.head<2>()).norm());
  Expect(out.tracking && out.armors_num == 4, "number three is a four-plate vehicle");
  Expect((out.position.head<2>() - infantry.centre.head<2>()).norm() < 0.02,
         "small-plate geometry despite the size output");
}

void TestBaseFallback()
{
  Truth base{{0.2, 5.0, 0.3}, 0.0, 0.0};
  base.plates = 3;
  base.r_even = base.r_odd = 0.3205;
  base.dz = 0.0;
  base.large = true;
  TrackSet set(Settings("base"));
  std::mt19937 rng(5);
  ArmorTrackerTarget out;
  for (int i = 0; i < 100; ++i)
  {
    std::vector<AutoAim::Armor> armors;
    for (const auto& c : base.Visible(0.01 * i, rng))
    {
      armors.push_back(MakeArmor(ArmorNumber::BASE, false, c));
    }
    out = set.Step(10000ULL * i, {1, 0, 0, 0}, CALIBRATION, armors);
  }
  std::printf("base: armors_num %d centre (%.3f, %.3f)\n", out.armors_num,
              out.position.x(), out.position.y());
  Expect(out.tracking && out.armors_num == 3,
         "the base is a three-plate fallback target");
  Expect((out.position.head<2>() - base.centre.head<2>()).norm() < 0.1, "base centre");
}

void TestModule()
{
  LibXR::Topic detected =
      LibXR::Topic::CreateTopic<const AutoAim::DetectedFrame*>("mod_detected");
  auto* tracker = new ArmorTracker(Settings("mod"));
  struct Received
  {
    std::mutex mutex;
    std::vector<ArmorTrackerTarget> targets;
  };
  auto* received = new Received();
  auto callback = LibXR::Topic::Callback::Create(
      [](bool, Received* r, const AutoAim::TrackedFrame* f)
      {
        Expect(f->detected.synced.image.Valid(), "tracked frame holds the image");
        std::lock_guard<std::mutex> lock(r->mutex);
        r->targets.push_back(f->target);
      },
      received);
  AutoAim::RequireTopic<const AutoAim::TrackedFrame*>("mod_tracked")
      .RegisterCallback(callback);

  ImagePool pool(4);
  const Truth truth{{0.0, 4.0, 0.1}, 0.3, 2.0};
  std::mt19937 rng(4);
  for (int i = 0; i < 50; ++i)
  {
    ImagePool::Handle writing;
    Expect(pool.Acquire(writing) == LibXR::ErrorCode::OK, "image slot");
    writing->calibration = &CALIBRATION;
    AutoAim::DetectedFrame frame;
    frame.synced = {static_cast<uint64_t>(i + 1), SharedFrame(std::move(writing)), {}};
    frame.synced.imu.timestamp_us = LibXR::MicrosecondTimestamp(10000ULL * (i + 1));
    frame.synced.imu.rotation_wxyz = {1, 0, 0, 0};
    for (const auto& c : truth.Visible(0.01 * i, rng))
    {
      frame.armors.push_back(MakeArmor(ArmorNumber::TWO, false, c));
    }
    const AutoAim::DetectedFrame* payload = &frame;
    detected.Publish(payload);
  }
  for (int t = 0; t < 2000; ++t)
  {
    {
      std::lock_guard<std::mutex> lock(received->mutex);
      if (received->targets.size() == 50)
      {
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::lock_guard<std::mutex> lock(received->mutex);
  Expect(received->targets.size() == 50, "one tracked frame per detected frame");
  Expect(received->targets.back().tracking &&
             received->targets.back().id == ArmorNumber::TWO,
         "tracking number two");
  Expect(received->targets.back().image_timestamp_us == 500000, "IMU timestamp");
  delete tracker;
}
}  // namespace

int main()
{
  LibXR::PlatformInit();
  TestVehicleEstimatorConverges();
  TestSelectionAndSwitching();
  TestSizeByNumber();
  TestBaseFallback();
  TestModule();
  std::puts("armor_tracker_test passed");
  return 0;
}
