// Replay recorded detections and gimbal IMU through Vehicle::VehicleEstimator and print
// the target per frame.
//
//   vehicle_replay --det detector.tsv --imu imu.csv --number N --cam
//   fx,fy,cx,cy[,k1,k2,p1,p2,k3]
//                [--h 0.3] [--mode auto|cv|acc|kb] [--type T] [--float 1] [--trackset 1]
//                [--shape lightbar4]   (default: the prototype's 135 / 230 x 56 mm
//                keypoints)
//
// detector.tsv: the recording format of the replay data package (header with
// image_timestamp_us, number, type, p0_x .. p3_y). imu.csv: timestamp_us,qw,qx,qy,qz,...
// with or without a header; the attitude used for a frame is the latest sample at or
// before its image timestamp. --type overrides the armor type of every detection. Output
// (tab separated, one line per frame with detections of N): ts_us tracking model x y z
// yaw v_yaw vx vy r1 r2 dz face
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "TrackSet.hpp"
#include "VehicleEstimator.hpp"

using namespace Vehicle;

namespace
{

std::vector<std::string> split(const std::string& s, char sep)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) out.push_back(item);
  return out;
}

struct Imu
{
  std::vector<long long> ts;
  std::vector<std::array<double, 4>> q;
};

Imu load_imu(const std::string& fname)
{
  std::ifstream f(fname);
  std::string line;
  std::vector<std::pair<long long, std::array<double, 4>>> rows;
  while (std::getline(f, line))
  {
    if (line.empty() || std::isalpha(static_cast<unsigned char>(line[0]))) continue;
    const auto c = split(line, ',');
    if (c.size() < 5) continue;
    rows.push_back({std::atoll(c[0].c_str()),
                    {std::atof(c[1].c_str()), std::atof(c[2].c_str()),
                     std::atof(c[3].c_str()), std::atof(c[4].c_str())}});
  }
  std::stable_sort(rows.begin(), rows.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });
  Imu imu;
  for (const auto& r : rows)
  {
    imu.ts.push_back(r.first);
    imu.q.push_back(r.second);
  }
  return imu;
}

}  // namespace

int main(int argc, char** argv)
{
  std::map<std::string, std::string> arg;
  for (int i = 1; i + 1 < argc; i += 2) arg[argv[i]] = argv[i + 1];
  if (!arg.count("--det") || !arg.count("--imu") || !arg.count("--number") ||
      !arg.count("--cam"))
  {
    std::fprintf(stderr,
                 "usage: vehicle_replay --det tsv --imu csv --number N --cam "
                 "fx,fy,cx,cy[,k1,k2,p1,p2,k3] "
                 "[--h 0.3] [--mode auto] [--type T]\n");
    return 2;
  }
  const int number = std::atoi(arg["--number"].c_str());
  const double h = arg.count("--h") ? std::atof(arg["--h"].c_str()) : 0.3;
  const int type_override = arg.count("--type") ? std::atoi(arg["--type"].c_str()) : -1;
  const bool as_float =
      arg.count("--float") && arg["--float"] == "1";  // corners as float, as ArmorTracker
  const std::string m = arg.count("--mode") ? arg["--mode"] : "auto";
  const RateMode mode = m == "cv"    ? RateMode::CV
                        : m == "acc" ? RateMode::ACC
                        : m == "kb"  ? RateMode::KEYS
                                     : RateMode::AUTO;
  Camera cam;
  const auto cv = split(arg["--cam"], ',');
  cam.fx = std::atof(cv[0].c_str());
  cam.fy = std::atof(cv[1].c_str());
  cam.cx = std::atof(cv[2].c_str());
  cam.cy = std::atof(cv[3].c_str());
  for (size_t i = 4; i < cv.size() && i < 9; ++i)
    cam.dist[i - 4] = std::atof(cv[i].c_str());

  const Imu imu = load_imu(arg["--imu"]);
  std::ifstream f(arg["--det"]);
  std::string line;
  std::getline(f, line);
  const auto head = split(line, '\t');
  std::map<std::string, size_t> col;
  for (size_t i = 0; i < head.size(); ++i) col[head[i]] = i;
  std::map<long long, std::vector<Detection>> frames;  // ordered by time stamp
  while (std::getline(f, line))
  {
    const auto c = split(line, '\t');
    if (c.size() < head.size() || std::atoi(c[col["number"]].c_str()) != number) continue;
    Detection d;
    d.type = type_override >= 0 ? type_override : std::atoi(c[col["type"]].c_str());
    for (int k = 0; k < 4; ++k)
    {
      double u = std::atof(c[col["p" + std::to_string(k) + "_x"]].c_str());
      double v = std::atof(c[col["p" + std::to_string(k) + "_y"]].c_str());
      if (as_float)
      {
        u = static_cast<float>(u);
        v = static_cast<float>(v);
      }
      d.corners[k] = Vec2(u, v);
    }
    frames[std::atoll(c[col["image_timestamp_us"]].c_str())].push_back(d);
  }

  const PlateShape shape = arg.count("--shape") && arg["--shape"] == "lightbar4"
                               ? LIGHTBAR4_SHAPE
                               : PROTOTYPE_SHAPE;
  // --trackset 1：经 TrackSet（模块的路径：AutoAim
  // 角点顺序、相对时间、按距离的时域）回放， 只输出位置与正对板。/ Replay through
  // TrackSet, the Module's path; prints position and face only.
  if (arg.count("--trackset") && arg["--trackset"] == "1")
  {
    const CameraTypes::CameraCalibration calibration{1440,   1080,   cam.fx,  cam.fy,
                                                     cam.cx, cam.cy, cam.dist};
    TrackSet set({"replay",
                  {1, 0, 0, 0},
                  {0, 0, 0},
                  number,
                  2,
                  15,
                  75,
                  0.07,
                  23.0,
                  SelectWeights{}},
                 shape);
    std::printf("ts_us\ttracking\tx\ty\tz\tface\n");
    for (const auto& fr : frames)
    {
      const long long ts = fr.first;
      size_t i = std::upper_bound(imu.ts.begin(), imu.ts.end(), ts) - imu.ts.begin();
      i = i == 0 ? 0 : i - 1;
      std::vector<AutoAim::Armor> armors;
      for (const Detection& d : fr.second)
      {
        AutoAim::Armor a{ArmorColor::RED,
                         static_cast<ArmorNumber>(number),
                         d.type == 1 ? ArmorType::LARGE : ArmorType::SMALL,
                         1.0F,
                         {}};
        const int order[4] = {0, 3, 2,
                              1};  // 估计器顺序转 AutoAim 顺序 / to AutoAim order
        for (int k = 0; k < 4; ++k)
        {
          a.corners[k] = {static_cast<float>(d.corners[order[k]].x()),
                          static_cast<float>(d.corners[order[k]].y())};
        }
        armors.push_back(a);
      }
      const std::array<float, 4> q{
          static_cast<float>(imu.q[i][0]), static_cast<float>(imu.q[i][1]),
          static_cast<float>(imu.q[i][2]), static_cast<float>(imu.q[i][3])};
      const ArmorTrackerTarget o =
          set.Step(static_cast<uint64_t>(ts), q, calibration, armors);
      std::printf("%lld\t%d\t%.9f\t%.9f\t%.9f\t%d\n", ts, o.tracking ? 1 : 0,
                  o.position.x(), o.position.y(), o.position.z(), o.tracked_face_index);
    }
    return 0;
  }

  VehicleEstimator est(cam, mode, shape);
  std::printf("ts_us\ttracking\tmodel\tx\ty\tz\tyaw\tv_yaw\tvx\tvy\tr1\tr2\tdz\tface\n");
  for (const auto& fr : frames)
  {
    const long long ts = fr.first;
    size_t i = std::upper_bound(imu.ts.begin(), imu.ts.end(), ts) - imu.ts.begin();
    i = i == 0 ? 0 : i - 1;
    const VehicleTarget o = est.Step(ts / 1e6, imu.q[i], fr.second, h);
    std::printf(
        "%lld\t%d\t%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.6f\t%.6f\t%.6f\t%d\n",
        ts, o.tracking ? 1 : 0, o.model, o.position.x(), o.position.y(), o.position.z(),
        o.yaw, o.v_yaw, o.velocity.x(), o.velocity.y(), o.radius_1, o.radius_2, o.dz,
        o.face);
  }
  return 0;
}
