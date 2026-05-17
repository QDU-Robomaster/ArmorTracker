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
};

struct SingleSpec
{
  std::string dataset{"single"};
  fs::path detector;
  fs::path out;
  fs::path preview_out;
  fs::path imu;
  int color{-1};
  int number{-1};
  Camera camera{};
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
    cases.push_back(spec);
  }
  return cases;
}

bool parse_single_args(int argc, char** argv, SingleSpec& spec)
{
  if (argc < 4 || std::string(argv[1]) != "--single")
  {
    return false;
  }
  spec.detector = argv[2];
  spec.out = argv[3];
  for (int i = 4; i < argc; ++i)
  {
    const std::string arg = argv[i];
    auto need_value = [&](const char* name) -> const char*
    {
      if (i + 1 >= argc)
      {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--dataset")
    {
      spec.dataset = need_value("--dataset");
    }
    else if (arg == "--color")
    {
      spec.color = std::stoi(need_value("--color"));
    }
    else if (arg == "--number")
    {
      spec.number = std::stoi(need_value("--number"));
    }
    else if (arg == "--fx")
    {
      spec.camera.fx = std::stod(need_value("--fx"));
    }
    else if (arg == "--fy")
    {
      spec.camera.fy = std::stod(need_value("--fy"));
    }
    else if (arg == "--cx")
    {
      spec.camera.cx = std::stod(need_value("--cx"));
    }
    else if (arg == "--cy")
    {
      spec.camera.cy = std::stod(need_value("--cy"));
    }
    else if (arg == "--imu")
    {
      spec.imu = need_value("--imu");
    }
    else if (arg == "--preview-out")
    {
      spec.preview_out = need_value("--preview-out");
    }
    else
    {
      throw std::runtime_error("unknown arg: " + arg);
    }
  }
  if (spec.detector.empty() || spec.out.empty())
  {
    throw std::runtime_error("single mode requires detector and out paths");
  }
  if (!std::isfinite(spec.camera.fx) || !std::isfinite(spec.camera.fy) ||
      !std::isfinite(spec.camera.cx) || !std::isfinite(spec.camera.cy))
  {
    throw std::runtime_error("single mode requires fx fy cx cy");
  }
  return true;
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

void write_preview_header(std::ofstream& out)
{
  out << "dataset\timage_timestamp_us\tstate\tselected_face\t"
         "outpost_height_phase\tface_id\tface_x\tface_y\tface_z\tface_yaw\t"
         "uv0_x\tuv0_y\tuv1_x\tuv1_y\tuv2_x\tuv2_y\tuv3_x\tuv3_y\n";
}

void write_preview_rows(std::ofstream& out,
                        armor_tracker_detail::TrackerCore& tracker,
                        const std::string& dataset, uint64_t timestamp_us,
                        const armor_tracker_detail::Output& output)
{
  for (const auto& track : output.tracks)
  {
    if (!track.selected)
    {
      continue;
    }
    for (int face = 0; face < static_cast<int>(track.faces_world.size()); ++face)
    {
      const Eigen::Vector4d xyza =
          track.faces_world[static_cast<std::size_t>(face)];
      const auto corners = tracker.ReprojectPreviewArmorFace(
          xyza.head<3>(), xyza[3], track.armor_type, track.tag_id);
      if (corners.size() != 4U)
      {
        continue;
      }
      out << dataset << '\t' << timestamp_us << '\t' << output.state << '\t'
          << output.selected_face << '\t' << output.outpost_height_phase
          << '\t' << face << '\t' << xyza.x() << '\t' << xyza.y() << '\t'
          << xyza.z() << '\t' << xyza[3];
      for (const auto& corner : corners)
      {
        out << '\t' << corner.x << '\t' << corner.y;
      }
      out << '\n';
    }
  }
}

}  // namespace

int main(int argc, char** argv)
{
  SingleSpec single_spec;
  if (parse_single_args(argc, argv, single_spec))
  {
    if (!single_spec.out.parent_path().empty())
    {
      fs::create_directories(single_spec.out.parent_path());
    }
    std::ofstream out(single_spec.out);
    out << std::setprecision(17);
    out << "dataset\timage_timestamp_us\tsource\tcenter_x\tcenter_y\tcenter_z\t"
           "vel_x\tvel_y\tvel_z\tyaw\tv_yaw\tarmor_x\tarmor_y\tarmor_z\t"
           "armor_yaw\tselected_face\toutpost_height_phase\tjumped\t"
           "radius_even\tradius_odd\tdz\n";
    std::ofstream preview_out;
    if (!single_spec.preview_out.empty())
    {
      if (!single_spec.preview_out.parent_path().empty())
      {
        fs::create_directories(single_spec.preview_out.parent_path());
      }
      preview_out.open(single_spec.preview_out);
      preview_out << std::setprecision(17);
      write_preview_header(preview_out);
    }

    armor_tracker_detail::Config config;
    config.require_target_tag = true;
    config.target_tag_id = single_spec.number;
    config.min_detect_count = 2;
    config.max_temp_lost_count = 15;
    config.outpost_max_temp_lost_count = 75;
    config.camera_matrix = {single_spec.camera.fx, 0.0, single_spec.camera.cx,
                            0.0, single_spec.camera.fy, single_spec.camera.cy,
                            0.0, 0.0, 1.0};
    armor_tracker_detail::TrackerCore tracker(config);
    const auto imu = read_imu_quats(single_spec.imu);
    std::map<uint64_t, std::vector<armor_tracker_detail::InputArmor>> frames;
    std::map<uint64_t, QuatRow> inline_imu;
    for (const auto& row : read_tsv(single_spec.detector))
    {
      if (!valid_detector_row(row, single_spec.color, single_spec.number))
      {
        continue;
      }
      const auto timestamp_us = as_u64(row, "image_timestamp_us");
      frames[timestamp_us].push_back(make_input(row));
      if (std::isfinite(as_double(row, "imu_qw", NAN)))
      {
        inline_imu[timestamp_us] = {as_double(row, "imu_qw", 1.0),
                                   as_double(row, "imu_qx", 0.0),
                                   as_double(row, "imu_qy", 0.0),
                                   as_double(row, "imu_qz", 0.0)};
      }
    }
    for (const auto& [timestamp_us, inputs] : frames)
    {
      QuatRow quat{};
      if (const auto inline_it = inline_imu.find(timestamp_us);
          inline_it != inline_imu.end())
      {
        quat = inline_it->second;
      }
      else if (const auto q = find_causal_quat(imu, timestamp_us))
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
      out << single_spec.dataset << '\t' << timestamp_us << "\traw_corner_cabi_"
          << output.state << '\t' << output.center.x() << '\t'
          << output.center.y() << '\t' << output.center.z() << '\t'
          << output.velocity.x() << '\t' << output.velocity.y() << '\t'
          << output.velocity.z() << '\t' << output.yaw << '\t'
          << output.vyaw << '\t' << output.armor.x() << '\t'
          << output.armor.y() << '\t' << output.armor.z() << '\t'
          << output.armor_yaw << '\t' << output.selected_face << '\t'
          << output.outpost_height_phase << '\t'
          << (output.jumped ? 1 : 0) << '\t' << output.radius_even << '\t'
          << output.radius_odd << '\t' << output.dz << '\n';
      if (preview_out)
      {
        write_preview_rows(preview_out, tracker, single_spec.dataset,
                           timestamp_us, output);
      }
    }
    return 0;
  }

  if (argc != 3)
  {
    std::cerr << "usage:\n  " << argv[0]
              << " <cases.tsv> <out.tsv>\n  " << argv[0]
              << " --single <detector.tsv> <out.tsv> --number N --color N --fx F --fy F --cx F --cy F [--dataset name] [--imu imu.csv] [--preview-out preview.tsv]\n";
    return 2;
  }
  const fs::path cases_path(argv[1]);
  const fs::path out_path(argv[2]);
  fs::create_directories(out_path.parent_path());
  std::ofstream out(out_path);
  out << std::setprecision(17);
  out << "dataset\timage_timestamp_us\tsource\tcenter_x\tcenter_y\tcenter_z\t"
         "vel_x\tvel_y\tvel_z\tyaw\tv_yaw\tarmor_x\tarmor_y\tarmor_z\t"
         "armor_yaw\tselected_face\toutpost_height_phase\tjumped\t"
         "radius_even\tradius_odd\tdz\n";

  for (const CaseSpec& spec : read_cases(cases_path))
  {
    armor_tracker_detail::Config config;
    config.require_target_tag = true;
    config.target_tag_id = spec.number;
    config.min_detect_count = 2;
    config.max_temp_lost_count = 15;
    config.outpost_max_temp_lost_count = 75;
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
          << output.velocity.z() << '\t' << output.yaw << '\t'
          << output.vyaw << '\t' << output.armor.x() << '\t'
          << output.armor.y() << '\t' << output.armor.z() << '\t'
          << output.armor_yaw << '\t' << output.selected_face << '\t'
          << output.outpost_height_phase << '\t'
          << (output.jumped ? 1 : 0) << '\t' << output.radius_even << '\t'
          << output.radius_odd << '\t' << output.dz << '\n';
    }
  }
  return 0;
}
