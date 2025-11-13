#include <cmath>
#include <cstddef>
#include <fstream>
#include <future>
#include <iostream>
#include <queue>
#include <string>
#include <thread>
#include <vector>

constexpr double MIN_PITCH = -0.6; // 限位
constexpr double MAX_PITCH = 1.2;
constexpr double MAX_X = 13.0; // 解算距离范围
constexpr double MIN_X = 0.0;
constexpr double MAX_Y = 2;         // 解算高度范围
constexpr double MIN_Y = -1;        // 中心为车的pitch轴电机
constexpr double RESOLUTION = 0.01; // 精度，m
constexpr double MAX_ERROR = 0.005; // 允许误差，m
constexpr int ERROR_LEVEL = 5;      // 误差等级
constexpr double GUN = 0.30;        // 枪口到pitch轴电机的距离，m

constexpr double G = 9.8;       // 重力加速度，m/s^2
constexpr double STEP = 0.0001; // RK4步长 (s)

static void build_table();

struct State {
  double x, y, vx, vy;
  State(double x, double y, double vx, double vy)
      : x(x), y(y), vx(vx), vy(vy) {}
};

class SolveTrajectory {
private:
  double v0_;
  double k_;        // 阻力系数
  double target_x_; // 目标x，y坐标 ,相对小车pitch轴电机(小车中心点)
  double target_y_;
  double dt_; // RK4步长 (s)

public:
  SolveTrajectory(double v0, bool type, double target_x, double target_y,
                  double dt = STEP)
      : v0_(v0), target_x_(target_x), target_y_(target_y), dt_(dt) {
    if (type == 0) { // 英雄
      // 20度时空气密度、空气阻力系数、子弹直径、重量
      k_ = 1.205 * 0.40 * 0.0425 * 0.0425 / (2 * 0.0445);
    } else if (type == 1) {
      k_ = 1.205 * 0.47 * 0.0168 * 0.0168 / (2 * 0.0032);
    }
  }

  // 运动方程: dy/dt = f(t, y)
  std::vector<double> AirODE(const State &state) {
    double v = std::sqrt(state.vx * state.vx + state.vy * state.vy);
    double ax = -k_ * v * state.vx;      // dvx/dt
    double ay = -G - k_ * v * state.vy;  // dvy/dt
    return {state.vx, state.vy, ax, ay}; // [dx/dt, dy/dt, dvx/dt, dvy/dt]
  }

  // RK4 单步积分
  State RK4Step(const State &state, double h) {
    auto k1 = AirODE(state);
    State state1(state.x + 0.5 * h * k1[0], state.y + 0.5 * h * k1[1],
                 state.vx + 0.5 * h * k1[2], state.vy + 0.5 * h * k1[3]);

    auto k2 = AirODE(state1);
    State state2(state.x + 0.5 * h * k2[0], state.y + 0.5 * h * k2[1],
                 state.vx + 0.5 * h * k2[2], state.vy + 0.5 * h * k2[3]);

    auto k3 = AirODE(state2);
    State state3(state.x + h * k3[0], state.y + h * k3[1], state.vx + h * k3[2],
                 state.vy + h * k3[3]);

    auto k4 = AirODE(state3);

    State new_state(state.x, state.y, state.vx, state.vy);
    new_state.x += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0;
    new_state.y += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0;
    new_state.vx += h * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0;
    new_state.vy += h * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0;

    return new_state;
  }

  // 单次目标弹道解算，实际解算从枪口计算，这里考虑枪管长度做补偿
  std::vector<double> SolvePitch(double pitch, double error) {
    double t_b = GUN / v0_;
    while (pitch < MAX_PITCH) {
      double count = 0;
      double x_b = -GUN * std::cos(pitch);
      double y_b = -GUN * std::sin(pitch);
      double x_to_gun = target_x_ + x_b;
      double y_to_gun = target_y_ + y_b;
      State state(0, 0, v0_ * std::cos(pitch), v0_ * std::sin(pitch));
      while (state.x < x_to_gun + error) {
        state = RK4Step(state, dt_);
        count++;
        if (pow(state.x - x_to_gun, 2) + pow(state.y - y_to_gun, 2) <=
            pow(error, 2))
        // if (std::fabs(state.x - x_to_gun) <= error &&
        //     std::fabs(state.y - y_to_gun) <= error)
        {
          std::cout << pitch << std::endl;
          return {pitch, count * STEP + t_b,
                  std::sqrt(state.vx * state.vx + state.vy * state.vy)};
        }
      }
      pitch += 0.01;
    }
    return {NAN, NAN, NAN};
  }

  // 重载SolvePitch，当不提供pitch时默认使用二分法搜索
  std::vector<double> SolvePitch(double error) {
    double t_b = GUN / v0_;
    double pitch_top = MAX_PITCH;
    double pitch_low = MIN_PITCH;
    while ((pitch_top - pitch_low) > 0.001) {
      double count = 0;
      double pitch_binary = (pitch_top + pitch_low) / 2;
      double x_b = -GUN * std::cos(pitch_binary);
      double y_b = -GUN * std::sin(pitch_binary);
      double x_to_gun = target_x_ + x_b;
      double y_to_gun = target_y_ + y_b;
      State state(0, 0, v0_ * std::cos(pitch_binary),
                  v0_ * std::sin(pitch_binary));

      // 这里用MIN_Y-1就能运行，MIN_Y就不行，不要问为什么
      while (state.y >= MIN_Y - 1) {
        state = RK4Step(state, dt_);
        count++;
        if (pow(state.x - x_to_gun, 2) + pow(state.y - y_to_gun, 2) <=
            pow(error, 2)) {
          return {pitch_binary, count * dt_ + t_b,
                  std::sqrt(state.vx * state.vx + state.vy * state.vy)};
        }

        if (state.x >= x_to_gun) {
          if (state.y > y_to_gun)
            pitch_top = pitch_binary;
          else
            pitch_low = pitch_binary;
          break;
        } else if (state.y < MIN_Y - 1 && state.x < x_to_gun) {
          // if (pitch_binary > 0.5) {
          //   pitch_top = pitch_binary;
          // } else {
          //   pitch_low = pitch_binary;
          // }
          pitch_low = pitch_binary;
          break;
        }
        // std::cerr << pitch_top << "和" << pitch_low << std::endl;
      }
    }
    // std::cerr << "nan" << std::endl;
    return {NAN, NAN, NAN};
  }

  // 对solvePitch的优化，考虑多级误差以保证精确和有解
  std::vector<double> SolvePitchLevel(int error_level, double pitch0) {
    auto ge = SolvePitch(MAX_ERROR);
    if (std::isnan(ge[0])) {
      return {NAN, NAN, NAN};
    }
    for (int i = 1; i <= error_level; i++) {
      ge = SolvePitch(MAX_ERROR / error_level * i);
      if (!std::isnan(ge[0])) {
        return ge;
      }
    }
    return {NAN, NAN, NAN};
  }

  // 给角度解位置
  std::vector<double> SolveHeightAndLength(double pitch) {
    double max_heigh = 0;
    double max_length = 0;
    State state(0, 0, v0_ * std::cos(pitch), v0_ * std::sin(pitch));
    while (state.y > -0.5) {
      state = RK4Step(state, dt_);
      if (max_heigh < state.y) {
        max_heigh = state.y;
      }
      if (max_length < state.x) {
        max_length = state.x;
      }
    }
    return {max_heigh, max_length};
  }

  static void BuildTable() { build_table(); }
};

using TableData = std::vector<std::vector<std::vector<double>>>;

// 解算多行 (现在 num_rows 基本总是 1)
static TableData solve_rows(double start_x, size_t num_rows) {
  double pitch0 = MIN_PITCH;
  double x = start_x;
  size_t y_dim = std::round((MAX_Y - MIN_Y) / RESOLUTION + 1);
  TableData table;
  table.reserve(num_rows);

  for (size_t i = 0; i < num_rows; i++, x += RESOLUTION) {
    double y = MIN_Y;
    std::vector<std::vector<double>> row;
    row.reserve(y_dim);
    for (size_t j = 0; j < y_dim; j++, y += RESOLUTION) {
      SolveTrajectory solve = SolveTrajectory(11.8, 0, x, y);
      std::vector<double> ge = solve.SolvePitchLevel(ERROR_LEVEL, pitch0);
      // if (!std::isnan(ge[0])) {
      //   pitch0 = ge[0];
      // }
      row.push_back(ge);
    }
    pitch0 = MIN_PITCH;
    table.push_back(std::move(row));
  }
  return table;
}

// 输出得到的表，按array的格式来,检查是否异常解的情况
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << '{';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0)
      os << ',';
    os << v[i];
  }
  return os << '}';
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<std::vector<T>> &v) {
  os << "{{";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0)
      os << ',';
    os << v[i];
  }
  return os << "}}";
}

std::ostream &operator<<(std::ostream &os, const TableData &v) {
  os << "{{\n";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0)
      os << ",\n";
    os << "  " << v[i];
  }
  return os << "\n}}";
}

// 输出表格解的情况，检查是否有无解的情况
template <typename T>
std::ostream &operator<<=(std::ostream &os, const std::vector<T> &v) {
  for (T x : v)
    os <<= x;
  os << "\n";
  return os;
}

template <>
std::ostream &operator<<=(std::ostream &os, const std::vector<double> &v) {
  return os << (std::isnan(v[0]) ? ' ' : '.');
}
static void build_table() {
  TableData table;
  std::ios_base::sync_with_stdio(false);
  std::queue<std::future<TableData>> futures;
  size_t threads = std::thread::hardware_concurrency();
  if (threads == 0) {
    threads = 16;
  }

  size_t total_rows = std::round((MAX_X - MIN_X) / RESOLUTION + 1);
  table.reserve(total_rows);

  std::cerr << "采用新的多线程逻辑 (批处理, 每任务1行), 使用 " << threads
            << " 个线程..." << '\n';
  std::cerr << "总行数: " << total_rows << '\n';

  double current_x = MIN_X;
  size_t rows_processed = 0;

  while (rows_processed < total_rows) {
    // 确定当前批次的大小：不超过线程数，也不超过剩余行数
    size_t batch_size = std::min(threads, total_rows - rows_processed);

    // 提交当前批次的所有任务，每个任务计算 1 行
    for (size_t i = 0; i < batch_size; ++i) {
      futures.push(std::async(std::launch::async, solve_rows, current_x, 1));
      current_x += RESOLUTION;
    }

    // 等待并收集当前批次的所有结果
    while (!futures.empty()) {
      TableData single_row_table = futures.front().get();
      futures.pop();
      // 将获取到的一行数据移动到主 table 中
      table.insert(table.end(),
                   std::make_move_iterator(single_row_table.begin()),
                   std::make_move_iterator(single_row_table.end()));
    }

    rows_processed += batch_size;
    std::cerr << "进度: " << rows_processed << " / " << total_rows
              << " 行已完成" << '\n';
  }

  std::cerr << "计算完成。" << '\n';
  (std::cerr <<= table) << '\n';
  (std::cout << table) << '\n';

  // ----- 二进制文件写入 -----
  std::string output_filename = std::to_string(MAX_X) + "_table.bin";
  std::ofstream file_out(output_filename.c_str(),
                         std::ios::out | std::ios::binary);
  if (!file_out) {
    std::cerr << "错误: 无法打开文件进行写入: " << output_filename << '\n';
    return;
  }

  struct CellToSolve {
    double pitch;
    double t;
    double v;
  };

  struct Cell {
    float pitch;
    float t;
    float v;
  };

  for (const auto &row : table) {
    for (const auto &ge : row) {
      Cell cell_to_write;
      if (ge.empty() || std::isnan(ge[0])) {
        cell_to_write = {NAN, NAN, NAN};
      } else {
        cell_to_write = {static_cast<float>(ge[0]), static_cast<float>(ge[1]),
                         static_cast<float>(ge[2])};
      }
      file_out.write(reinterpret_cast<const char *>(&cell_to_write),
                     sizeof(Cell));
    }
  }

  file_out.close();
  std::cerr << "二进制查找表已成功生成到 " << output_filename << '\n';
}

int main() {
  build_table();
  return 0;
}

// int main() {
//   SolveTrajectory test(11.8, 0, 13, -1);
//   test.SolvePitch(0.05);
//   test.SolvePitch(0, 0.05);
//   return 0;
// }