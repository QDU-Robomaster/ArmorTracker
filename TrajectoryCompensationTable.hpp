#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

class Table
{
 public:
  Table() {};
  ~Table() {};
  static constexpr double MAX_X = 13.0;
  static constexpr double MIN_X = 0.0;
  static constexpr double MAX_Y = 1.0;
  static constexpr double MIN_Y = -1.0;
  static constexpr double JINGDU = 0.01;

  struct Cell
  {
    double pitch;
    double t;
    double v;
  };

  struct State
  {
    double x, y, vx, vy;
    State(double x, double y, double vx, double vy) : x(x), y(y), vx(vx), vy(vy) {}
  };

  static Cell Check(float x, float y, float x_bias = 0, float y_bias = 0,
                    float pitch_bias = 0.02, float t_bias = 0)
  {
    size_t xc =
        static_cast<size_t>(std::round((x + x_bias - MIN_X) * 100 / JINGDU) / 100);
    size_t yc =
        static_cast<size_t>(std::round((y + y_bias - MIN_Y) * 100 / JINGDU) / 100);
    xc = std::min(xc, X_DIM - 1);
    yc = std::min(yc, Y_DIM - 1);

    Cell ge = TABLE[xc][yc];

    std::cout << xc << yc << '\n';
    return {ge.pitch + pitch_bias, ge.t + t_bias, ge.v};
  }

 private:
  static constexpr size_t X_DIM = static_cast<size_t>((MAX_X - MIN_X) / JINGDU) + 1;
  static constexpr size_t Y_DIM = static_cast<size_t>((MAX_Y - MIN_Y) / JINGDU) + 1;
  static constexpr Cell TABLE[X_DIM][Y_DIM] = {};
};
