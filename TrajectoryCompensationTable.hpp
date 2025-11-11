#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "logger.hpp"

class TrajectoryTable
{
 public:
  struct TableConfig
  {
    double max_x, min_x, max_y, min_y, resolution;
    size_t x_dim, y_dim;
    std::string filename;
    TableConfig(double max_x, double min_x, double max_y, double min_y, double resolution,
                std::string filename)
        : max_x(max_x),
          min_x(min_x),
          max_y(max_y),
          min_y(min_y),
          resolution(resolution),
          x_dim(static_cast<size_t>((max_x - min_x) / resolution) + 1),
          y_dim(static_cast<size_t>((max_y - min_y) / resolution) + 1),
          filename(std::move(filename))
    {
    }
  };
  TrajectoryTable(const TableConfig& config)
      : MAX_X(config.max_x),
        MIN_X(config.min_x),
        MAX_Y(config.max_y),
        MIN_Y(config.min_y),
        RESOLUTION(config.resolution),
        X_DIM(config.x_dim),
        Y_DIM(config.y_dim),
        filename_(config.filename) {};
  ~TrajectoryTable() {};

  struct Cell
  {
    float pitch;
    float t;
    float v;
  };

  struct State
  {
    double x, y, vx, vy;
    State(double x, double y, double vx, double vy) : x(x), y(y), vx(vx), vy(vy) {}
  };

  Cell Check(float x, float y, float x_bias = 0, float y_bias = 0,
             float pitch_bias = 0.02, float t_bias = 0)
  {
    size_t xc = static_cast<size_t>(std::round((x + x_bias - MIN_X) / RESOLUTION));
    size_t yc = static_cast<size_t>(std::round((y + y_bias - MIN_Y) / RESOLUTION));
    xc = std::min(xc, X_DIM - 1);
    yc = std::min(yc, Y_DIM - 1);

    Cell ge = table_[xc * Y_DIM + yc];

    return {ge.pitch + pitch_bias, ge.t + t_bias, ge.v};
  }

  void Init()
  {
    table_.resize(X_DIM * Y_DIM);

    std::ifstream file_in(filename_, std::ios::in | std::ios::binary);

    if (!file_in)
    {
      XR_LOG_ERROR("错误: 无法打开文件，使用默认弹道解算");
      init = false;
      return;
    }

    const std::size_t BYTES_TO_READ = X_DIM * Y_DIM * sizeof(Cell);

    file_in.read(reinterpret_cast<char*>(table_.data()),
                 static_cast<std::streamsize>(BYTES_TO_READ));

    if (!file_in || file_in.gcount() != static_cast<std::streamsize>(BYTES_TO_READ))
    {
      XR_LOG_ERROR("错误: 读取数据失败或文件大小不匹配，使用默认弹道解算");
      init = false;
      return;
    }

    file_in.close();
    init = true;
  }

  bool IsInit() const { return init; }

 private:
  const double MAX_X;
  const double MIN_X;
  const double MAX_Y;
  const double MIN_Y;
  const double RESOLUTION;

  const size_t X_DIM;
  const size_t Y_DIM;

  bool init = false;

  std::string filename_{};

  std::vector<Cell> table_;
};
