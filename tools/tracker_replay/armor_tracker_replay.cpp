/**
 * @file armor_tracker_replay.cpp
 * @brief Deterministic TSV replay tool for ArmorTrackerCore validation.
 */

#include "ArmorTrackerCore.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
struct Row
{
  std::map<std::string, std::string> fields;
};

struct Camera
{
  double fx{};
  double fy{};
  double cx{};
  double cy{};
};

struct CaseSpec
{
  std::string dataset;
  fs::path detector;
  int color{};
  int number{};
  Camera camera{};
  int output_frame{1};
};

struct QuatRow
{
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

std::vector<std::string> split_tab(const std::string& line)
{
  std::vector<std::string> out;
  std::string cur;
  for (char ch : line)
  {
    if (ch == '\t')
    {
      out.push_back(cur);
      cur.clear();
    }
    else if (ch != '\r')
    {
      cur.push_back(ch);
    }
  }
  out.push_back(cur);
  return out;
}

std::vector<std::string> split_csv(const std::string& line)
{
  std::vector<std::string> out;
  std::string cur;
  bool in_quote = false;
  for (char ch : line)
  {
    if (ch == '"')
    {
      in_quote = !in_quote;
    }
    else if (ch == ',' && !in_quote)
    {
      out.push_back(cur);
      cur.clear();
    }
    else if (ch != '\r')
    {
      cur.push_back(ch);
    }
  }
  out.push_back(cur);
  return out;
}

std::vector<Row> read_tsv(const fs::path& path)
{
  std::ifstream in(path);
  if (!in)
  {
    throw std::runtime_error("cannot open " + path.string());
  }
  std::string line;
  if (!std::getline(in, line))
  {
    return {};
  }
  auto header = split_tab(line);
  std::vector<Row> rows;
  while (std::getline(in, line))
  {
    auto vals = split_tab(line);
    Row row;
    for (std::size_t i = 0; i < header.size(); ++i)
    {
      row.fields[header[i]] = i < vals.size() ? vals[i] : "";
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

std::string get(const Row& row, const std::string& key,
                const std::string& def = "")
{
  auto it = row.fields.find(key);
  return it == row.fields.end() || it->second.empty() ? def : it->second;
}

double as_double(const Row& row, const std::string& key, double def = NAN)
{
  try
  {
    const std::string text = get(row, key);
    return text.empty() ? def : std::stod(text);
  }
  catch (...)
  {
    return def;
  }
}

int as_int(const Row& row, const std::string& key, int def = 0)
{
  const double value = as_double(row, key, NAN);
  return std::isfinite(value) ? static_cast<int>(std::llround(value)) : def;
}

uint64_t as_u64(const Row& row, const std::string& key, uint64_t def = 0)
{
  const double value = as_double(row, key, NAN);
  return std::isfinite(value) ? static_cast<uint64_t>(std::llround(value)) : def;
}

int header_index(const std::vector<std::string>& header, const std::string& key)
{
  for (std::size_t i = 0; i < header.size(); ++i)
  {
    if (header[i] == key)
    {
      return static_cast<int>(i);
    }
  }
  return -1;
}

double csv_double(const std::vector<std::string>& cols, int idx, double def = 0.0)
{
  if (idx < 0 || idx >= static_cast<int>(cols.size()) ||
      cols[static_cast<std::size_t>(idx)].empty())
  {
    return def;
  }
  try
  {
    return std::stod(cols[static_cast<std::size_t>(idx)]);
  }
  catch (...)
  {
    return def;
  }
}

std::map<uint64_t, QuatRow> read_imu_quats(const fs::path& path)
{
  std::map<uint64_t, QuatRow> out;
  if (path.empty())
  {
    return out;
  }
  std::ifstream in(path);
  if (!in)
  {
    throw std::runtime_error("cannot open imu " + path.string());
  }
  std::string line;
  if (!std::getline(in, line))
  {
    return out;
  }
  auto first = split_csv(line);
  const bool has_header =
      std::find(first.begin(), first.end(), "qw") != first.end() ||
      std::find(first.begin(), first.end(), "image_timestamp_us") != first.end() ||
      std::find(first.begin(), first.end(), "timestamp_us") != first.end();
  int ts_idx = 0;
  int qw_idx = 1;
  int qx_idx = 2;
  int qy_idx = 3;
  int qz_idx = 4;
  if (has_header)
  {
    ts_idx = header_index(first, "image_timestamp_us");
    if (ts_idx < 0)
    {
      ts_idx = header_index(first, "timestamp_us");
    }
    if (ts_idx < 0)
    {
      ts_idx = 0;
    }
    qw_idx = header_index(first, "qw");
    qx_idx = header_index(first, "qx");
    qy_idx = header_index(first, "qy");
    qz_idx = header_index(first, "qz");
  }
  else
  {
    const double ts_val = csv_double(first, 0, -1.0);
    if (ts_val >= 0.0)
    {
      out[static_cast<uint64_t>(std::llround(ts_val))] = {
          csv_double(first, 1, 1.0), csv_double(first, 2),
          csv_double(first, 3), csv_double(first, 4)};
    }
  }

  while (std::getline(in, line))
  {
    if (line.empty())
    {
      continue;
    }
    auto cols = split_csv(line);
    const double ts_val = csv_double(cols, ts_idx, -1.0);
    if (ts_val < 0.0)
    {
      continue;
    }
    out[static_cast<uint64_t>(std::llround(ts_val))] = {
        csv_double(cols, qw_idx, 1.0), csv_double(cols, qx_idx),
        csv_double(cols, qy_idx), csv_double(cols, qz_idx)};
  }
  return out;
}

std::optional<QuatRow> find_causal_quat(
    const std::map<uint64_t, QuatRow>& quats, uint64_t timestamp_us)
{
  if (quats.empty())
  {
    return std::nullopt;
  }
  auto it = quats.upper_bound(timestamp_us);
  if (it == quats.begin())
  {
    return std::nullopt;
  }
  --it;
  if (timestamp_us > it->first && timestamp_us - it->first > 100000ULL)
  {
    return std::nullopt;
  }
  return it->second;
}

fs::path imu_path_for(const std::string& dataset)
{
  const char* env = nullptr;
  if (dataset == "new_internal")
  {
    env = std::getenv("TRACKER_RAW_CORNER_IMU_NEW");
  }
  else if (dataset == "new_internal_5p9_lossless")
  {
    env = std::getenv("TRACKER_RAW_CORNER_IMU_5P9");
  }
  else if (dataset == "old_internal")
  {
    env = std::getenv("TRACKER_RAW_CORNER_IMU_OLD");
  }
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path{};
}

std::vector<CaseSpec> read_cases(const fs::path& path)
{
  std::vector<CaseSpec> cases;
  for (const auto& row : read_tsv(path))
  {
    CaseSpec spec;
    spec.dataset = get(row, "dataset");
    spec.detector = get(row, "detector");
    spec.color = as_int(row, "color", -1);
    spec.number = as_int(row, "number", -1);
    spec.camera.fx = as_double(row, "fx");
    spec.camera.fy = as_double(row, "fy");
    spec.camera.cx = as_double(row, "cx");
    spec.camera.cy = as_double(row, "cy");
    spec.output_frame = as_int(row, "output_frame", 1);
    cases.push_back(spec);
  }
  return cases;
}

bool valid_detector_row(const Row& row, int color, int number)
{
  if (as_int(row, "color", -1) != color)
  {
    return false;
  }
  if (as_int(row, "number", -1) != number)
  {
    return false;
  }
  for (int i = 0; i < 4; ++i)
  {
    if (!std::isfinite(as_double(row, "p" + std::to_string(i) + "_x")) ||
        !std::isfinite(as_double(row, "p" + std::to_string(i) + "_y")))
    {
      return false;
    }
  }
  return true;
}

armor_tracker_detail::InputArmor make_input(const Row& row)
{
  armor_tracker_detail::InputArmor input;
  input.tag_id = as_int(row, "number", -1);
  input.armor_type = as_int(row, "type", 0);
  input.confidence = as_double(row, "confidence", 0.0);
  input.center_norm.x = static_cast<float>(as_double(row, "center_norm_x", 0.5));
  input.center_norm.y = static_cast<float>(as_double(row, "center_norm_y", 0.5));
  for (int i = 0; i < 4; ++i)
  {
    input.corners[static_cast<std::size_t>(i)] = cv::Point2f(
        static_cast<float>(as_double(row, "p" + std::to_string(i) + "_x")),
        static_cast<float>(as_double(row, "p" + std::to_string(i) + "_y")));
  }
  return input;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "usage: " << argv[0] << " <cases.tsv> <out.tsv>\n";
    return 2;
  }
  const fs::path cases_path(argv[1]);
  const fs::path out_path(argv[2]);
  fs::create_directories(out_path.parent_path());
  std::ofstream out(out_path);
  out << std::setprecision(17);
  out << "dataset\timage_timestamp_us\tsource\tcenter_x\tcenter_y\tcenter_z\t"
         "vel_x\tvel_y\tvel_z\tvelocity_confidence\tvelocity_var_x\t"
         "velocity_var_y\tvelocity_var_z\tyaw\tv_yaw\tarmor_x\tarmor_y\t"
         "armor_z\tarmor_yaw\tselected_face\tjumped\tmeasured_face_valid\t"
         "measured_face_index\tmeasured_face_x\tmeasured_face_y\t"
         "measured_face_z\tmeasured_face_yaw\tradius_even\tradius_odd\tdz\n";

  for (const CaseSpec& spec : read_cases(cases_path))
  {
    armor_tracker_detail::Config config;
    config.require_target_tag = true;
    config.target_tag_id = spec.number;
    config.min_detect_count = 2;
    config.max_temp_lost_count = 15;
    config.outpost_max_temp_lost_count = 75;
    config.output_frame = spec.output_frame;
    config.camera_matrix = {spec.camera.fx, 0.0, spec.camera.cx,
                            0.0, spec.camera.fy, spec.camera.cy,
                            0.0, 0.0, 1.0};
    armor_tracker_detail::TrackerCore tracker(config);
    const auto imu = read_imu_quats(imu_path_for(spec.dataset));
    std::map<uint64_t, std::vector<armor_tracker_detail::InputArmor>> frames;
    for (const auto& row : read_tsv(spec.detector))
    {
      if (!valid_detector_row(row, spec.color, spec.number))
      {
        continue;
      }
      frames[as_u64(row, "image_timestamp_us")].push_back(make_input(row));
    }
    for (const auto& [timestamp_us, inputs] : frames)
    {
      QuatRow quat{};
      if (const auto q = find_causal_quat(imu, timestamp_us))
      {
        quat = *q;
      }
      const armor_tracker_detail::Output output = tracker.Step(
          timestamp_us, Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z),
          inputs);
      if (!output.has_target)
      {
        continue;
      }
      out << spec.dataset << '\t' << timestamp_us << "\traw_corner_cabi_"
          << output.state << '\t' << output.center.x() << '\t'
          << output.center.y() << '\t' << output.center.z() << '\t'
          << output.velocity.x() << '\t' << output.velocity.y() << '\t'
          << output.velocity.z() << '\t' << output.velocity_confidence << '\t'
          << output.velocity_variance.x() << '\t'
          << output.velocity_variance.y() << '\t'
          << output.velocity_variance.z() << '\t'
          << output.yaw << '\t' << output.vyaw << '\t' << output.armor.x()
          << '\t'
          << output.armor.y() << '\t' << output.armor.z() << '\t'
          << output.armor_yaw << '\t' << output.selected_face << '\t'
          << (output.jumped ? 1 : 0) << '\t'
          << (output.measured_face_valid ? 1 : 0) << '\t'
          << output.measured_face_index << '\t'
          << output.measured_face_position.x() << '\t'
          << output.measured_face_position.y() << '\t'
          << output.measured_face_position.z() << '\t'
          << output.measured_face_yaw << '\t' << output.radius_even << '\t'
          << output.radius_odd << '\t' << output.dz << '\n';
    }
  }
  return 0;
}
