#ifndef ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
#define ARMOR_PROCESSOR__KALMAN_FILTER_HPP_

/**
 * @file extended_kalman_filter.hpp
 * @brief 通用扩展卡尔曼滤波器接口。
 *
 * ArmorTracker 使用该 EKF 保存整车中心、速度、yaw、半径和高低差状态。
 * 该类不绑定任何装甲板语义，所有过程模型、观测模型、雅可比和噪声矩阵均由
 * 调用者以函数对象形式注入。
 */

#include <Eigen/Dense>
#include <functional>
#include <initializer_list>

/**
 * @brief 可注入模型的通用扩展卡尔曼滤波器。
 *
 * 状态维度由初始协方差矩阵 `P0` 的行数确定。`Predict()` 会用过程函数推进状态，
 * `Update()` 会用观测函数修正状态，并采用 Joseph 形式更新后验协方差。
 */
class ExtendedKalmanFilter
{
 public:
  /**
   * @brief ArmorTracker 约定的 11 维状态索引。
   */
  enum XVectorIndex : std::uint8_t
  {
    X_CENTER = 0,   ///< 整车中心 x。
    V_X_CENTER = 1, ///< 整车中心 x 速度。
    Y_CENTER = 2,   ///< 整车中心 y。
    V_Y_CENTER = 3, ///< 整车中心 y 速度。
    Z_ARMOR = 4,    ///< 参考装甲面高度。
    V_Z_ARMOR = 5,  ///< 参考装甲面高度速度。
    YAW = 6,        ///< 整车 yaw。
    V_YAW = 7,      ///< 整车 yaw 角速度。
    ROBOT_R = 8,    ///< 第一组装甲半径。
    DELTA_R = 9,    ///< 第二组装甲半径相对差。
    DELTA_Z = 10    ///< 第二组装甲高度相对差。
  };

  /**
   * @brief 构造一个空 EKF，后续需重新赋值完整模型后才能使用。
   */
  ExtendedKalmanFilter() = default;

  ///< 非线性状态或观测函数类型。
  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
  ///< 依赖状态的雅可比或噪声矩阵函数类型。
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
  ///< 不依赖状态的矩阵生成函数类型。
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  /**
   * @brief 构造完整 EKF 模型。
   * @param f 过程函数。
   * @param h 观测函数。
   * @param j_f 过程函数雅可比。
   * @param j_h 观测函数雅可比。
   * @param u_q 过程噪声协方差生成函数。
   * @param u_r 观测噪声协方差生成函数。
   * @param P0 初始后验协方差。
   */
  explicit ExtendedKalmanFilter(const VecVecFunc& f, const VecVecFunc& h,
                                const VecMatFunc& j_f, const VecMatFunc& j_h,
                                const VoidMatFunc& u_q, const VecMatFunc& u_r,
                                const Eigen::MatrixXd& P0);

  /**
   * @brief 设置先验和后验初始状态。
   * @param x0 初始状态向量。
   */
  void SetState(const Eigen::VectorXd& x0);

  /**
   * @brief 执行一次预测。
   * @return 预测后的先验状态。
   */
  Eigen::MatrixXd Predict();

  /**
   * @brief 根据观测量执行一次后验更新。
   * @param z 观测向量。
   * @return 更新后的后验状态。
   */
  Eigen::MatrixXd Update(const Eigen::VectorXd& z);

  /**
   * @brief 清除指定状态维度与其它维度之间的后验协方差。
   * @param state_indices 需要保留内部相关性但切断外部相关性的状态索引集合。
   */
  void DecorrelatePosterior(std::initializer_list<int> state_indices);

 private:
  VecVecFunc f_;              ///< 过程非线性函数。
  VecVecFunc h_;              ///< 观测非线性函数。
  VecMatFunc jacobian_f_;     ///< 过程函数雅可比生成函数。
  Eigen::MatrixXd m_f_;       ///< 最近一次过程雅可比矩阵。
  VecMatFunc jacobian_h_;     ///< 观测函数雅可比生成函数。
  Eigen::MatrixXd m_h_;       ///< 最近一次观测雅可比矩阵。
  VoidMatFunc update_q_;      ///< 过程噪声协方差生成函数。
  Eigen::MatrixXd m_q_;       ///< 最近一次过程噪声协方差。
  VecMatFunc update_r_;       ///< 观测噪声协方差生成函数。
  Eigen::MatrixXd m_r_;       ///< 最近一次观测噪声协方差。
  Eigen::MatrixXd p_pri_;     ///< 先验误差协方差。
  Eigen::MatrixXd p_post_;    ///< 后验误差协方差。
  Eigen::MatrixXd m_k_;       ///< 卡尔曼增益。
  int n_;                     ///< 状态维度。
  Eigen::MatrixXd i_;         ///< n 维单位矩阵。
  Eigen::VectorXd x_pri_;     ///< 先验状态。
  Eigen::VectorXd x_post_;    ///< 后验状态。
};

#endif  // ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
