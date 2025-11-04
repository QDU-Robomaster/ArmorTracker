#include <cmath>
#include <cstddef>
#include <future>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>

constexpr double MIN_PITCH = -0.6;  // 限位
constexpr double MAX_PITCH = 1.2;
constexpr double MAX_X = 13.0;  // 解算距离范围
constexpr double MIN_X = 0.0;
constexpr double MAX_Y = 1.0;       // 解算高度范围
constexpr double MIN_Y = -1;        // 中心为车的pitch轴电机
constexpr double JINGDU = 0.01;     // 精度，m
constexpr double MAX_ERROR = 0.05;  // 允许误差，m
constexpr int ERROR_LEVEL = 4;      // 误差等级
constexpr double GUN = 0.30;        // 枪口到pitch轴电机的距离，m

constexpr double G = 9.8;        // 重力加速度，m/s^2
constexpr double STEP = 0.0001;  // RK4步长 (s)

static void build_table();

struct State
{
  double x, y, vx, vy;
  State(double x, double y, double vx, double vy) : x(x), y(y), vx(vx), vy(vy) {}
};

class SolveTrajectory
{
 private:
  double v0_;
  double k_;         // 阻力系数
  double target_x_;  // 目标x，y坐标 ,相对小车pitch轴电机(小车中心点)
  double target_y_;
  double dt_;  // RK4步长 (s)

 public:
  SolveTrajectory(double v0, bool type, double target_x, double target_y,
                  double dt = STEP)
      : v0_(v0), target_x_(target_x), target_y_(target_y), dt_(dt)
  {
    if (type == 0)
    {  // 英雄
      // 20度时空气密度、空气阻力系数、子弹直径、重量
      k_ = 1.205 * 0.40 * 0.0425 * 0.0425 / (2 * 0.0445);
    }
    else if (type == 1)
    {
      k_ = 1.205 * 0.47 * 0.0168 * 0.0168 / (2 * 0.0032);
    }
  }

  // 运动方程: dy/dt = f(t, y)
  std::vector<double> AirODE(const State& state)
  {
    double v = std::sqrt(state.vx * state.vx + state.vy * state.vy);
    double ax = -k_ * v * state.vx;       // dvx/dt
    double ay = -G - k_ * v * state.vy;   // dvy/dt
    return {state.vx, state.vy, ax, ay};  // [dx/dt, dy/dt, dvx/dt, dvy/dt]
  }

  // RK4 单步积分
  State RK4Step(const State& state, double h)
  {
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
  std::vector<double> SolvePitch(double pitch = MIN_PITCH,
                                 double error = MAX_ERROR / ERROR_LEVEL)
  {
    double count = 0;
    double t_b = GUN / v0_;
    while (pitch < MAX_PITCH)
    {
      count = 0;
      double x_b = -GUN * std::cos(pitch);
      double y_b = -GUN * std::sin(pitch);
      double x_xiangdui = target_x_ + x_b;
      double y_xiangdui = target_y_ + y_b;
      State state(0, 0, v0_ * std::cos(pitch), v0_ * std::sin(pitch));
      while (state.x < x_xiangdui + error)
      {
        state = RK4Step(state, dt_);
        count++;

        if (std::fabs(state.x - x_xiangdui) <= error &&
            std::fabs(state.y - y_xiangdui) <= error)
        {
          return {pitch, count * STEP / 0.001 + t_b,
                  std::sqrt(state.vx * state.vx + state.vy * state.vy)};
        }
      }
      pitch += 0.01;
    }
    return {NAN, NAN, NAN};
  }

  // 对solvePitch的优化，考虑多级误差以保证精确和有解
  std::vector<double> SolvePitchLevel(int error_level, double pitch0 = MIN_PITCH)
  {
    auto ge = SolvePitch(pitch0, MAX_ERROR / error_level * error_level);
    if (std::isnan(ge[0]))
    {
      return {NAN, NAN, NAN};
    }
    for (int i = 1; i <= error_level; i++)
    {
      ge = SolvePitch(pitch0, MAX_ERROR / error_level * i);
      if (!std::isnan(ge[0]))
      {
        return ge;
      }
    }
    return {NAN, NAN, NAN};
  }

  // 给角度解位置
  std::vector<double> SolveHeightAndLength(double pitch)
  {
    double max_heigh = 0;
    double max_length = 0;
    State state(0, 0, v0_ * std::cos(pitch), v0_ * std::sin(pitch));
    while (state.y > -0.5)
    {
      state = RK4Step(state, dt_);
      if (max_heigh < state.y)
      {
        max_heigh = state.y;
      }
      if (max_length < state.x)
      {
        max_length = state.x;
      }
    }
    return {max_heigh, max_length};
  }

  static void BuildTable() { build_table(); }
};

// 解算多行
static std::vector<std::vector<std::vector<double>>> SolveRows(double s, size_t xc)
{
  double pitch0 = MIN_PITCH;
  double x = s;
  double y = NAN;
  size_t yc = std::round((MAX_Y - MIN_Y) / JINGDU) + 1;
  std::vector<std::vector<std::vector<double>>> table;
  table.reserve(xc);
  for (size_t i = 0; i < xc; i++, x += JINGDU)
  {
    y = MIN_Y;
    std::vector<std::vector<double>> row;
    row.reserve(yc);
    for (size_t j = 0; j < yc; j++, y += JINGDU)
    {
      SolveTrajectory solve = SolveTrajectory(11.8, 0, x, y);
      std::vector<double> ge = solve.SolvePitchLevel(ERROR_LEVEL, pitch0);
      if (!std::isnan(ge[0]))
      {
        pitch0 = ge[0];
      }
      row.push_back(ge);
    }
    pitch0 = MIN_PITCH;
    table.push_back(std::move(row));
    std::cerr << '[' << x << '/' << s + (xc - 1) * JINGDU << ']' << std::endl;
  }
  // std::cout << "right------------" << i << std::endl;
  return table;
}

// 输出得到的表，按array的格式来
template <typename T>
static std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
  if (v.empty() || std::isnan(v[0]))
  {
    os << "{NAN, NAN, NAN}";
  }
  else
  {
    os << '{' << v[0] << "f, " << v[1] << "f, " << v[2] << "f}";
  }
  return os;
}

template <typename T>
static std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& v)
{
  os << "  {";
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (i > 0)
    {
      os << ", ";
    }
    if (i > 0 && i % 10 == 0)
    {
      os << "\n   ";
    }
    os << v[i];
  }
  os << "}";
  return os;
}

static std::ostream& operator<<(std::ostream& os,
                                const std::vector<std::vector<std::vector<double>>>& v)
{
  os << "{\n";  // 整个数组的起始大括号
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (i > 0)
    {
      os << ",\n";
    }
    os << v[i];  // 调用上面的重载输出一整行
  }
  os << "\n}";  // 整个数组的结束大括号
  return os;
}

// 输出表格解的情况，检查是否有无解的情况
template <typename T>
static std::ostream& operator<<=(std::ostream& os, const std::vector<T>& v)
{
  for (const T& x : v)
  {
    os <<= x;
  }
  os << "\n";
  return os;
}

template <>
std::ostream& operator<<=(std::ostream& os, const std::vector<double>& v)
{
  return os << (std::isnan(v[0]) ? ' ' : '.');
}

// 多线程提高效率
static void build_table()
{
  std::vector<std::vector<std::vector<double>>> table;
  std::ios::sync_with_stdio(false);
  std::queue<std::future<std::vector<std::vector<std::vector<double>>>>> futures;
  size_t threads = std::thread::hardware_concurrency();
  if (!threads)
  {
    threads = 16;
  }
  size_t total = std::round((MAX_X - MIN_X) / JINGDU) + 1, count = total / threads,
         remaining = total % threads;
  if (!count)
  {
    threads = remaining;
  }
  std::cerr << "Threads: " << threads << " Count: " << count << " ... " << remaining
            << '\n';
  double x = MIN_X;
  for (size_t i = 0; i < threads; i++)
  {
    futures.push(std::async(SolveRows, x, count + !!remaining));
    if (remaining > 0)
    {
      x += JINGDU * (count + !!remaining);
      remaining--;
    }
  }
  table.reserve(count);
  while (futures.size())
  {
    std::vector<std::vector<std::vector<double>>> rows = futures.front().get();
    futures.pop();
    std::move(rows.begin(), rows.end(), std::back_inserter(table));
  }
  std::cerr << "行数x:" << table.size() << '\n';
  std::cerr << "列数y:" << table[0].size() << '\n';
  (std::cerr <<= table) << '\n';
  (std::cout << table) << '\n';
}