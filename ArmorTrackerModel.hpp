#pragma once

/**
 * @file ArmorTrackerModel.hpp
 * @brief 装甲板几何、目标状态和 tracker 状态机。
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ArmorTrackerMath.hpp"

namespace armor_tracker_detail
{
/**
 * @brief PnP 几何使用的装甲板尺寸类型。
 */
enum ArmorType
{
  BIG,
  SMALL
};

/**
 * @brief tracker 模型使用的装甲板身份类型。
 */
enum ArmorName
{
  ONE,
  TWO,
  THREE,
  FOUR,
  FIVE,
  SENTRY,
  OUTPOST,
  BASE,
  NOT_ARMOR
};

/**
 * @brief 持久车辆轨道数量，不包含无效装甲板类型。
 */
inline constexpr std::size_t kTrackSlotCount =
    static_cast<std::size_t>(ArmorName::NOT_ARMOR);

/**
 * @brief 将 detector 数字枚举转为内部装甲板名称。
 */
inline ArmorName ArmorNameFromDetectorNumber(int number)
{
  if (number >= 0 && number <= 4)
  {
    return static_cast<ArmorName>(number);
  }
  if (number == 5)
  {
    return ArmorName::OUTPOST;
  }
  if (number == 6)
  {
    return ArmorName::SENTRY;
  }
  if (number == 7)
  {
    return ArmorName::BASE;
  }
  return ArmorName::NOT_ARMOR;
}

/**
 * @brief 返回装甲板名称对应的持久轨道槽位。
 */
inline int TrackSlotIndex(ArmorName name)
{
  const auto index = static_cast<int>(name);
  return index >= 0 && index < static_cast<int>(kTrackSlotCount) ? index : -1;
}

// 前哨站三块装甲相邻高度差，单位 m。
inline constexpr double kOutpostArmorHeightStep = 0.102;
inline constexpr double kOutpostArmorRadius = 0.2680137228;
inline constexpr double kOutpostArmorTilt = 15.0 * kPi / 180.0;
inline constexpr double kOutpostLightbarWidth = 0.135;

inline int PositiveMod(int value, int mod)
{
  const int result = value % mod;
  return result < 0 ? result + mod : result;
}

inline double OutpostArmorHeightOffset(int face_id, int height_phase)
{
  // 本地面按 yaw 递增展开；前哨站实物在该顺序下是中、高、低。
  switch (PositiveMod(face_id + height_phase, 3))
  {
    case 1:
      return kOutpostArmorHeightStep;
    case 2:
      return -kOutpostArmorHeightStep;
    case 0:
    default:
      return 0.0;
  }
}

inline int SignNonZero(double value)
{
  return value > 0.0 ? 1 : -1;
}

/**
 * @brief detector 装甲板的目标选择优先级。
 */
enum ArmorPriority
{
  FIRST = 1,
  SECOND,
  THIRD,
  FOURTH,
  FIFTH
};

/**
 * @brief 带 3D 位姿解算结果的内部装甲板观测。
 */
struct Armor
{
  cv::Point2f center{};
  cv::Point2f center_norm{};
  std::vector<cv::Point2f> points{};
  double ratio{};
  double rectangular_error{};
  ArmorType type{};
  ArmorName name{};
  ArmorPriority priority{FIFTH};
  cv::Rect box{};
  double confidence{};
  Eigen::Vector3d xyz_in_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypr_in_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();
  double yaw_raw{};

  /**
   * @brief 构造空装甲板观测。
   */
  Armor() = default;

  /**
   * @brief 由 detector 几何结果构造装甲板观测。
   */
  Armor(int num_id, float confidence_in, const cv::Rect& box_in,
        std::vector<cv::Point2f> armor_keypoints)
      : points(std::move(armor_keypoints)),
        box(box_in),
        confidence(confidence_in)
  {
    center = (points[0] + points[1] + points[2] + points[3]) / 4.0F;
    const auto left_width = cv::norm(points[0] - points[3]);
    const auto right_width = cv::norm(points[1] - points[2]);
    const auto max_width = std::max(left_width, right_width);
    const auto top_length = cv::norm(points[0] - points[1]);
    const auto bottom_length = cv::norm(points[3] - points[2]);
    const auto max_length = std::max(top_length, bottom_length);
    const auto left_center = (points[0] + points[3]) / 2.0F;
    const auto right_center = (points[1] + points[2]) / 2.0F;
    const auto left2right = right_center - left_center;
    const auto roll = std::atan2(left2right.y, left2right.x);
    const auto left_rectangular_error = std::abs(
        std::atan2((points[3] - points[0]).y, (points[3] - points[0]).x) -
        roll - kPi / 2.0);
    const auto right_rectangular_error = std::abs(
        std::atan2((points[2] - points[1]).y, (points[2] - points[1]).x) -
        roll - kPi / 2.0);
    rectangular_error = std::max(left_rectangular_error, right_rectangular_error);
    ratio = max_length / max_width;
    name = num_id == 0 ? ArmorName::SENTRY
                       : num_id > 5 ? ArmorName(num_id) : ArmorName(num_id - 1);
    type = num_id == 1 ? ArmorType::BIG : ArmorType::SMALL;
  }
};

/**
 * @brief 将装甲板身份映射到 tracker 目标选择优先级。
 */
inline ArmorPriority PriorityFromName(ArmorName name)
{
  switch (name)
  {
    case ArmorName::THREE:
    case ArmorName::FOUR:
      return ArmorPriority::FIRST;
    case ArmorName::ONE:
      return ArmorPriority::SECOND;
    case ArmorName::FIVE:
    case ArmorName::SENTRY:
      return ArmorPriority::THIRD;
    case ArmorName::TWO:
      return ArmorPriority::FOURTH;
    case ArmorName::OUTPOST:
    case ArmorName::BASE:
    case ArmorName::NOT_ARMOR:
    default:
      return ArmorPriority::FIFTH;
  }
}

/**
 * @brief tracker 内部配置。
 */
struct Config
{
  bool require_target_tag = false;
  int target_tag_id = -1;
  int min_detect_count = 2;
  int max_temp_lost_count = 15;
  int outpost_max_temp_lost_count = 75;
  std::array<double, 9> camera_matrix{
      1164.3428599490444, 0.0, 366.6782312546237,
      0.0, 1164.335053894998, 270.30936434613865,
      0.0, 0.0, 1.0};
  std::array<double, 4> camera_mount_to_body_rotation{1.0, 0.0, 0.0, 0.0};
  std::array<double, 3> camera_mount_to_body_translation{0.0, 0.0, 0.0};
};

/**
 * @brief 将 wxyz 四元数转为归一化旋转矩阵。
 */
inline Eigen::Matrix3d RotationMatrixFromWxyz(
    const std::array<double, 4>& rotation)
{
  Eigen::Quaterniond q(rotation[0], rotation[1], rotation[2], rotation[3]);
  if (!std::isfinite(q.norm()) || q.norm() < 1e-9)
  {
    q = Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q.toRotationMatrix();
}

/**
 * @brief OpenCV 相机系 C 到相机安装系 M 的固定轴变换。
 *
 * C 使用 x 向右、y 向下、z 向前。M 与 C 同原点，轴向使用公开约定：
 * x 向右、y 向前、z 向上。
 */
inline Eigen::Matrix3d CameraToMountRotation()
{
  Eigen::Matrix3d rotation;
  rotation << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0;
  return rotation;
}

/**
 * @brief 由配置的 M 到 B 安装旋转生成 C 到 B 的内部旋转。
 */
inline Eigen::Matrix3d CameraToBodyRotationFromMountExtrinsic(
    const std::array<double, 4>& camera_mount_to_body_rotation)
{
  return RotationMatrixFromWxyz(camera_mount_to_body_rotation) *
         CameraToMountRotation();
}

/**
 * @brief 将发布的云台 IMU 姿态转为 tracker 惯性解算轴。
 *
 * `gimbal_quat` 已经按公开本体系 B 表达：x 向右，y 向前，z 向上。
 * 单板 IMU 安装修正应在 AHRS 前完成，例如 BMI088 rotation 配置。
 * tracker 这里只能归一化四元数，不能再叠加固定轴变换。
 */
inline Eigen::Matrix3d BodyToWorldRotationFromImu(const Eigen::Quaterniond& q)
{
  Eigen::Quaterniond normalized = q;
  if (!std::isfinite(normalized.norm()) || normalized.norm() < 1e-9)
  {
    normalized = Eigen::Quaterniond::Identity();
  }
  normalized.normalize();
  return normalized.toRotationMatrix();
}

/**
 * @brief 由公开 yaw 和装甲板倾角生成装甲板旋转。
 */
inline Eigen::Matrix3d ArmorRotationFromYaw(double yaw, double tilt)
{
  const auto sin_yaw = std::sin(yaw);
  const auto cos_yaw = std::cos(yaw);
  const auto sin_tilt = std::sin(tilt);
  const auto cos_tilt = std::cos(tilt);

  Eigen::Matrix3d rotation;
  rotation << -sin_yaw * cos_tilt, -cos_yaw, -sin_yaw * sin_tilt,
      cos_yaw * cos_tilt, -sin_yaw, cos_yaw * sin_tilt, -sin_tilt, 0.0,
      cos_tilt;
  return rotation;
}

/**
 * @brief 从装甲板旋转中提取公开坐标系 yaw。
 */
inline double ArmorYawFromRotation(const Eigen::Matrix3d& rotation)
{
  return BearingYaw(rotation.col(0));
}

/**
 * @brief 将 detector 装甲板角点解算为相机、本体和惯性位姿。
 */
class Solver
{
 public:
  /**
   * @brief 使用相机内参和手眼参数构造位姿解算器。
   */
  explicit Solver(const Config& config)
      : R_camera_to_body_(Eigen::Matrix3d::Identity()),
        t_camera_to_body_(Eigen::Vector3d::Zero()),
        R_body_to_world_(Eigen::Matrix3d::Identity())
  {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(
        config.camera_matrix.data());
    Eigen::Matrix<double, 1, 5> distort_coeffs;
    distort_coeffs << 0.0, 0.0, 0.0, 0.0, 0.0;
    camera_matrix_ = Mat3dToCv(camera_matrix);
    distort_coeffs_ = cv::Mat(1, 5, CV_64F);
    for (int col = 0; col < 5; ++col)
    {
      distort_coeffs_.at<double>(0, col) = distort_coeffs(0, col);
    }
    R_camera_to_body_ =
        CameraToBodyRotationFromMountExtrinsic(
            config.camera_mount_to_body_rotation);
    t_camera_to_body_ =
        Eigen::Vector3d(config.camera_mount_to_body_translation[0],
                        config.camera_mount_to_body_translation[1],
                        config.camera_mount_to_body_translation[2]);
  }

  /**
   * @brief 更新 PnP 后处理使用的本体到惯性姿态。
   */
  void SetRBodyToWorld(const Eigen::Quaterniond& q)
  {
    R_body_to_world_ = BodyToWorldRotationFromImu(q);
  }

  /**
   * @brief 解算 detector 装甲板位姿并填充其 3D 字段。
   */
  void Solve(Armor& armor) const
  {
    const auto& object_points =
        armor.name == ArmorName::OUTPOST ? OutpostPnpPoints()
        : armor.type == ArmorType::BIG    ? BigArmorPoints()
                                          : SmallArmorPoints();
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::solvePnP(object_points, armor.points, camera_matrix_, distort_coeffs_,
                 rvec, tvec, false, cv::SOLVEPNP_IPPE);

    Eigen::Vector3d xyz_in_camera(tvec[0], tvec[1], tvec[2]);
    armor.xyz_in_body = R_camera_to_body_ * xyz_in_camera + t_camera_to_body_;
    armor.xyz_in_world = R_body_to_world_ * armor.xyz_in_body;

    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    Eigen::Matrix3d R_armor2camera = CvMatToMat3d(rmat);
    Eigen::Matrix3d R_armor2body = R_camera_to_body_ * R_armor2camera;
    Eigen::Matrix3d R_armor2world = R_body_to_world_ * R_armor2body;
    armor.ypr_in_body = {ArmorYawFromRotation(R_armor2body), 0.0, 0.0};
    armor.ypr_in_world = {ArmorYawFromRotation(R_armor2world), 0.0, 0.0};
    armor.ypd_in_world = XyzToYpd(armor.xyz_in_world);

    const auto is_balance =
        armor.type == ArmorType::BIG &&
        (armor.name == ArmorName::THREE || armor.name == ArmorName::FOUR ||
         armor.name == ArmorName::FIVE);
    if (is_balance)
    {
      return;
    }
    OptimizeYaw(armor);
  }

  /**
   * @brief 将惯性坐标中的装甲面投影到图像像素。
   */
  std::vector<cv::Point2f> ReprojectArmor(const Eigen::Vector3d& xyz_in_world,
                                          double yaw, ArmorType type,
                                          ArmorName name) const
  {
    const auto tilt =
        name == ArmorName::OUTPOST ? kOutpostArmorTilt : 15.0 * kPi / 180.0;
    const Eigen::Matrix3d R_armor2world = ArmorRotationFromYaw(yaw, tilt);

    const Eigen::Vector3d& t_armor2world = xyz_in_world;
    Eigen::Matrix3d R_armor2camera =
        R_camera_to_body_.transpose() * R_body_to_world_.transpose() *
        R_armor2world;
    Eigen::Vector3d t_armor2camera =
        R_camera_to_body_.transpose() *
        (R_body_to_world_.transpose() * t_armor2world - t_camera_to_body_);

    cv::Vec3d rvec;
    cv::Rodrigues(Mat3dToCv(R_armor2camera), rvec);
    cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

    std::vector<cv::Point2f> image_points;
    const auto& object_points =
        name == ArmorName::OUTPOST ? OutpostPnpPoints()
        : type == ArmorType::BIG    ? BigArmorPoints()
                                  : SmallArmorPoints();
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_,
                      image_points);
    return image_points;
  }

  /**
   * @brief 投影 preview 装甲面，不改变 PnP/yaw 拟合模型。
   */
  std::vector<cv::Point2f> ReprojectPreviewArmor(
      const Eigen::Vector3d& xyz_in_world, double yaw, ArmorType type,
      ArmorName name) const
  {
    return ReprojectArmor(xyz_in_world, yaw, type, name);
  }

 private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_camera_to_body_;
  Eigen::Vector3d t_camera_to_body_;
  Eigen::Matrix3d R_body_to_world_;

  /**
   * @brief 返回大装甲 PnP 物点。
   */
  static const std::vector<cv::Point3f>& BigArmorPoints()
  {
    static const std::vector<cv::Point3f> points{
        {0, kBigArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kBigArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kBigArmorWidth / 2.0, -kLightbarLength / 2.0},
        {0, kBigArmorWidth / 2.0, -kLightbarLength / 2.0}};
    return points;
  }

  /**
   * @brief 返回小装甲 PnP 物点。
   */
  static const std::vector<cv::Point3f>& SmallArmorPoints()
  {
    static const std::vector<cv::Point3f> points{
        {0, kSmallArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kSmallArmorWidth / 2.0, kLightbarLength / 2.0},
        {0, -kSmallArmorWidth / 2.0, -kLightbarLength / 2.0},
        {0, kSmallArmorWidth / 2.0, -kLightbarLength / 2.0}};
    return points;
  }

  /**
   * @brief 返回前哨站装甲 PnP 物点。
   */
  static const std::vector<cv::Point3f>& OutpostPnpPoints()
  {
    static const std::vector<cv::Point3f> points{
        {0, kOutpostLightbarWidth / 2.0, kLightbarLength / 2.0},
        {0, -kOutpostLightbarWidth / 2.0, kLightbarLength / 2.0},
        {0, -kOutpostLightbarWidth / 2.0, -kLightbarLength / 2.0},
        {0, kOutpostLightbarWidth / 2.0, -kLightbarLength / 2.0}};
    return points;
  }

  /**
   * @brief 通过最小化重投影误差修正装甲板 yaw。
   */
  void OptimizeYaw(Armor& armor) const
  {
    const double body_yaw = BearingYaw(R_body_to_world_.col(1));
    constexpr double search_range = 140.0;
    auto yaw0 = LimitRad(body_yaw - search_range / 2.0 * kPi / 180.0);

    auto min_error = 1e10;
    auto best_yaw = armor.ypr_in_world[0];
    for (int i = 0; i < static_cast<int>(search_range); ++i)
    {
      const double yaw = LimitRad(yaw0 + i * kPi / 180.0);
      const auto error =
          ArmorReprojectionError(armor, yaw,
                                 (i - search_range / 2.0) * kPi / 180.0);
      if (error < min_error)
      {
        min_error = error;
        best_yaw = yaw;
      }
    }
    armor.yaw_raw = armor.ypr_in_world[0];
    armor.ypr_in_world[0] = best_yaw;
  }

  /**
   * @brief 返回候选 yaw 下装甲板的重投影误差。
   */
  double ArmorReprojectionError(const Armor& armor, double yaw,
                                const double&) const
  {
    auto image_points =
        ReprojectArmor(armor.xyz_in_world, yaw, armor.type, armor.name);
    auto error = 0.0;
    for (int i = 0; i < 4; ++i)
    {
      error += cv::norm(armor.points[i] - image_points[i]);
    }
    return error;
  }
};

/**
 * @brief 单个被跟踪机器人的扩展目标状态。
 */
class Target
{
 public:
  ArmorName name{};
  ArmorType armor_type{};
  ArmorPriority priority{};
  bool jumped{};
  int last_id{};

  /**
   * @brief 返回当前前哨站高度相位，用于展开三块装甲。
   */
  int OutpostHeightPhase() const { return outpost_height_phase_; }

  /**
   * @brief 返回当前前哨站高度相位是否已经由观测约束过。
   */
  bool OutpostHeightPhaseValid() const { return outpost_height_phase_valid_; }

  /**
   * @brief 构造空目标状态。
   */
  Target() = default;

  /**
   * @brief 使用首个选中装甲板观测初始化目标。
   */
  Target(const Armor& armor, std::chrono::steady_clock::time_point t,
         double radius, int armor_num, const Eigen::VectorXd& P0_dig,
         int outpost_height_phase = 0, bool outpost_height_phase_valid = false,
         Eigen::Vector3d outpost_center_hint = Eigen::Vector3d::Zero(),
         bool outpost_center_hint_valid = false)
      : name(armor.name),
        armor_type(armor.type),
        priority(armor.priority),
        jumped(false),
        last_id(0),
        armor_num_(armor_num),
        update_count_(0),
        is_converged_(false),
        t_(t),
        outpost_height_phase_(outpost_height_phase),
        outpost_height_phase_valid_(outpost_height_phase_valid)
  {
    const auto r = radius;
    const Eigen::VectorXd& xyz = armor.xyz_in_world;
    const Eigen::VectorXd& ypr = armor.ypr_in_world;
    auto center_x = xyz[0] - r * std::sin(ypr[0]);
    auto center_y = xyz[1] + r * std::cos(ypr[0]);
    auto center_z = xyz[2];
    if (name == ArmorName::OUTPOST && armor_num == 3)
    {
      center_z -= OutpostArmorHeightOffset(last_id, outpost_height_phase_);
      if (outpost_center_hint_valid)
      {
        center_x = outpost_center_hint.x();
        center_y = outpost_center_hint.y();
        center_z = outpost_center_hint.z();
      }
    }

    Eigen::VectorXd x0(11);
    x0 << center_x, 0.0, center_y, 0.0, center_z, 0.0, ypr[0], 0.0, r,
        0.0, 0.0;
    Eigen::MatrixXd P0 = P0_dig.asDiagonal();

    auto x_add = [](const Eigen::VectorXd& a,
                    const Eigen::VectorXd& b) -> Eigen::VectorXd
    {
      Eigen::VectorXd c = a + b;
      c[6] = LimitRad(c[6]);
      return c;
    };
    ekf_ = ExtendedKalmanFilter(x0, P0, x_add);
    if (UseOutpostHeightModel())
    {
      outpost_observed_z_[0] = xyz[2];
      outpost_observed_z_valid_[0] = true;
    }
  }

  /**
   * @brief 将目标状态预测到指定时刻。
   */
  void Predict(std::chrono::steady_clock::time_point t)
  {
    auto dt = DeltaTime(t, t_);
    Predict(dt);
    t_ = t;
  }

  /**
   * @brief 按经过时间预测目标状态，单位 s。
   */
  void Predict(double dt)
  {
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;

    double v1;
    double v2;
    if (name == ArmorName::OUTPOST)
    {
      // 前哨站本体固定，侧面 PnP 距离噪声较大；中心位置预测应强约束。
      v1 = 0.05;
      v2 = 0.5;
    }
    else
    {
      v1 = 100.0;
      v2 = 400.0;
    }
    const auto a = dt * dt * dt * dt / 4.0;
    const auto b = dt * dt * dt / 2.0;
    const auto c = dt * dt;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
    Q(0, 0) = a * v1;
    Q(0, 1) = b * v1;
    Q(1, 0) = b * v1;
    Q(1, 1) = c * v1;
    Q(2, 2) = a * v1;
    Q(2, 3) = b * v1;
    Q(3, 2) = b * v1;
    Q(3, 3) = c * v1;
    Q(4, 4) = a * v1;
    Q(4, 5) = b * v1;
    Q(5, 4) = b * v1;
    Q(5, 5) = c * v1;
    Q(6, 6) = a * v2;
    Q(6, 7) = b * v2;
    Q(7, 6) = b * v2;
    Q(7, 7) = c * v2;

    auto f = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd
    {
      Eigen::VectorXd x_prior = F * x;
      x_prior[6] = LimitRad(x_prior[6]);
      return x_prior;
    };

    if (Converged() && name == ArmorName::OUTPOST && std::abs(ekf_.x[7]) > 2.0)
    {
      ekf_.x[7] = ekf_.x[7] > 0.0 ? 2.51 : -2.51;
    }
    ekf_.Predict(F, Q, f);
    ClampOutpostCenterVelocity();
  }

  /**
   * @brief 使用一个匹配的装甲板观测更新目标状态。
   */
  void Update(const Armor& armor)
  {
    int id = 0;
    auto min_angle_error = 1e10;
    const std::vector<Eigen::Vector4d>& xyza_list = ArmorXyzaList();
    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; ++i)
    {
      xyza_i_list.push_back({xyza_list[i], i});
    }
    std::sort(
        xyza_i_list.begin(), xyza_i_list.end(),
        [](const std::pair<Eigen::Vector4d, int>& a,
           const std::pair<Eigen::Vector4d, int>& b)
        {
          Eigen::Vector3d ypd1 = XyzToYpd(a.first.head(3));
          Eigen::Vector3d ypd2 = XyzToYpd(b.first.head(3));
          return ypd1[2] < ypd2[2];
        });

    const int candidate_count =
        std::min(3, static_cast<int>(xyza_i_list.size()));
    for (int i = 0; i < candidate_count; ++i)
    {
      const auto& xyza = xyza_i_list[i].first;
      Eigen::Vector3d ypd = XyzToYpd(xyza.head(3));
      auto angle_error =
          std::abs(LimitRad(armor.ypr_in_world[0] - xyza[3])) +
          std::abs(LimitRad(armor.ypd_in_world[0] - ypd[0]));
      if (UseOutpostHeightModel() && outpost_height_phase_valid_)
      {
        angle_error += 2.0 * std::abs(armor.xyz_in_world.z() - xyza[2]);
      }
      if (std::abs(angle_error) < std::abs(min_angle_error))
      {
        id = xyza_i_list[i].second;
        min_angle_error = angle_error;
      }
    }

    if (id != 0)
    {
      jumped = true;
    }
    UpdateOutpostHeightPhase(armor, id);
    last_id = id;
    ++update_count_;
    UpdateYpda(armor, id);
    ClampOutpostCenterVelocity();
  }

  /**
   * @brief 返回 EKF 状态向量副本。
   */
  Eigen::VectorXd EkfX() const { return ekf_.x; }

  /**
   * @brief 返回 EKF 对象，用于调试指标和健康检查。
   */
  const ExtendedKalmanFilter& Ekf() const { return ekf_; }

  /**
   * @brief 返回惯性 W 系下的模型装甲面中心和 yaw。
   */
  std::vector<Eigen::Vector4d> ArmorXyzaList() const
  {
    std::vector<Eigen::Vector4d> list;
    list.reserve(static_cast<std::size_t>(armor_num_));
    for (int i = 0; i < armor_num_; ++i)
    {
      auto angle = LimitRad(ekf_.x[6] + i * 2.0 * kPi / armor_num_);
      Eigen::Vector3d xyz = HArmorXyz(ekf_.x, i);
      list.push_back({xyz[0], xyz[1], xyz[2], angle});
    }
    return list;
  }

  /**
   * @brief 返回叠加前哨站三高度相位后的装甲板输出几何。
   */
  std::vector<Eigen::Vector4d> ArmorXyzaListForOutput() const
  {
    if (!UseOutpostHeightModel())
    {
      return ArmorXyzaList();
    }

    std::vector<Eigen::Vector4d> list;
    list.reserve(static_cast<std::size_t>(armor_num_));
    const double center_z = CenterWorldForOutput().z();
    for (int i = 0; i < armor_num_; ++i)
    {
      auto angle = LimitRad(ekf_.x[6] + i * 2.0 * kPi / armor_num_);
      Eigen::Vector3d xyz = HArmorXyz(ekf_.x, i);
      xyz.z() = center_z + OutpostArmorHeightOffset(i, outpost_height_phase_);
      list.push_back({xyz[0], xyz[1], xyz[2], angle});
    }
    return list;
  }

  /**
   * @brief 返回用于输出的目标中心。
   */
  Eigen::Vector3d CenterWorldForOutput() const
  {
    Eigen::Vector3d center{ekf_.x[0], ekf_.x[2], ekf_.x[4]};
    return center;
  }

  /**
   * @brief 返回用于输出的目标速度。
   */
  Eigen::Vector3d VelocityWorldForOutput() const
  {
    Eigen::Vector3d velocity{ekf_.x[1], ekf_.x[3], ekf_.x[5]};
    if (UseOutpostHeightModel())
    {
      velocity.z() = 0.0;
    }
    return velocity;
  }

  /**
   * @brief 返回输出使用的高度参数。
   */
  double DzForOutput() const
  {
    return UseOutpostHeightModel() ? kOutpostArmorHeightStep : ekf_.x[10];
  }

  /**
   * @brief 检查估计几何是否超出允许半径范围。
   */
  bool Diverged() const
  {
    auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
    auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
    return !(r_ok && l_ok);
  }

  /**
   * @brief 更新并返回目标收敛标志。
   */
  bool Converged()
  {
    if (name != ArmorName::OUTPOST && update_count_ > 3 && !Diverged())
    {
      is_converged_ = true;
    }
    if (name == ArmorName::OUTPOST && update_count_ > 10 && !Diverged())
    {
      is_converged_ = true;
    }
    return is_converged_;
  }

 private:
  int armor_num_ = 4;
  int update_count_ = 0;
  bool is_converged_ = false;
  ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_{};
  int outpost_height_phase_ = 0;
  bool outpost_height_phase_valid_ = false;
  std::array<double, 3> outpost_observed_z_{};
  std::array<bool, 3> outpost_observed_z_valid_{};

  bool UseOutpostHeightModel() const
  {
    return name == ArmorName::OUTPOST && armor_num_ == 3;
  }

  void ClampOutpostCenterVelocity()
  {
    if (!UseOutpostHeightModel())
    {
      return;
    }

    // 前哨站塔心在惯性系固定，侧面 PnP 不应给中心积分出平移速度。
    ekf_.x[1] = 0.0;
    ekf_.x[3] = 0.0;
    ekf_.x[5] = 0.0;
  }

  void UpdateOutpostHeightPhase(const Armor& armor, int id)
  {
    if (!UseOutpostHeightModel() || id < 0 || id >= 3)
    {
      return;
    }

    const bool has_previous =
        last_id >= 0 && last_id < 3 &&
        outpost_observed_z_valid_[static_cast<std::size_t>(last_id)];
    const double previous_z =
        has_previous ? outpost_observed_z_[static_cast<std::size_t>(last_id)]
                     : 0.0;

    if (id != last_id && has_previous)
    {
      constexpr double kHeightRelationThreshold = 0.04;
      constexpr double kTwoHeightStepTolerance = 0.05;
      const double measured_delta = armor.xyz_in_world.z() - previous_z;
      const double measured_delta_abs = std::abs(measured_delta);
      const bool is_two_step =
          std::abs(measured_delta_abs - 2.0 * kOutpostArmorHeightStep) <=
          kTwoHeightStepTolerance;
      if (measured_delta_abs >= kHeightRelationThreshold && is_two_step)
      {
        const int measured_sign = SignNonZero(measured_delta);
        for (int phase = 0; phase < 3; ++phase)
        {
          const double candidate_delta =
              OutpostArmorHeightOffset(id, phase) -
              OutpostArmorHeightOffset(last_id, phase);
          const bool candidate_is_two_step =
              std::abs(std::abs(candidate_delta) -
                       2.0 * kOutpostArmorHeightStep) <= 1e-6;
          if (candidate_is_two_step &&
              SignNonZero(candidate_delta) == measured_sign)
          {
            if (phase != outpost_height_phase_)
            {
              outpost_height_phase_ = phase;
              ekf_.x[4] =
                  armor.xyz_in_world.z() -
                  OutpostArmorHeightOffset(id, outpost_height_phase_);
              ekf_.x[5] = 0.0;
            }
            outpost_height_phase_valid_ = true;
            break;
          }
        }
      }
    }

    constexpr double kZUpdateAlpha = 0.35;
    auto& observed_z = outpost_observed_z_[static_cast<std::size_t>(id)];
    auto& observed_valid =
        outpost_observed_z_valid_[static_cast<std::size_t>(id)];
    if (observed_valid)
    {
      observed_z =
          (1.0 - kZUpdateAlpha) * observed_z + kZUpdateAlpha * armor.xyz_in_world.z();
    }
    else
    {
      observed_z = armor.xyz_in_world.z();
      observed_valid = true;
    }
  }

  /**
   * @brief 使用 yaw、仰角、距离和装甲板 yaw 观测更新 EKF。
   */
  void UpdateYpda(const Armor& armor, int id)
  {
    const double center_x_before = ekf_.x[0];
    const double center_y_before = ekf_.x[2];
    const double center_z_before = ekf_.x[4];

    Eigen::MatrixXd H = HJacobian(ekf_.x, id);
    auto center_yaw = BearingYaw(armor.xyz_in_world);
    auto delta_angle = LimitRad(armor.ypr_in_world[0] - center_yaw);
    const double side_view = std::abs(delta_angle);
    const bool side_observation = side_view > 0.55;
    const bool lock_outpost_center = UseOutpostHeightModel() && side_observation;
    Eigen::VectorXd R_dig(4);
    if (name == ArmorName::OUTPOST)
    {
      const double ypd_noise =
          side_observation ? 25.0 : 0.02 + 0.2 * side_view;
      const double distance_noise = side_observation ? 400.0 : 2.0;
      // 前哨站侧面观测的 PnP 平移容易偏；侧面只保留 yaw 作为相位和角速度证据。
      R_dig << ypd_noise, ypd_noise, distance_noise, 3e-2;
    }
    else
    {
      R_dig << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1.0) + 1.0,
          std::log(std::abs(armor.ypd_in_world[2]) + 1.0) / 200.0 + 9e-2;
    }
    Eigen::MatrixXd R = R_dig.asDiagonal();

    auto h = [&](const Eigen::VectorXd& x) -> Eigen::Vector4d
    {
      Eigen::VectorXd xyz = HArmorXyz(x, id);
      Eigen::VectorXd ypd = XyzToYpd(xyz);
      auto angle = LimitRad(x[6] + id * 2.0 * kPi / armor_num_);
      return {ypd[0], ypd[1], ypd[2], angle};
    };
    auto z_subtract = [](const Eigen::VectorXd& a,
                         const Eigen::VectorXd& b) -> Eigen::VectorXd
    {
      Eigen::VectorXd c = a - b;
      c[0] = LimitRad(c[0]);
      c[1] = LimitRad(c[1]);
      c[3] = LimitRad(c[3]);
      return c;
    };

    const Eigen::VectorXd& ypd = armor.ypd_in_world;
    const Eigen::VectorXd& ypr = armor.ypr_in_world;
    Eigen::VectorXd z(4);
    z << ypd[0], ypd[1], ypd[2], ypr[0];
    ekf_.Update(z, H, R, h, z_subtract);
    if (UseOutpostHeightModel() && !side_observation)
    {
      const double angle = LimitRad(ekf_.x[6] + id * 2.0 * kPi / armor_num_);
      const double r = ekf_.x[8];
      ekf_.x[0] = armor.xyz_in_world.x() - r * std::sin(angle);
      ekf_.x[1] = 0.0;
      ekf_.x[2] = armor.xyz_in_world.y() + r * std::cos(angle);
      ekf_.x[3] = 0.0;
      ekf_.x[4] = armor.xyz_in_world.z() -
                  OutpostArmorHeightOffset(id, outpost_height_phase_);
      ekf_.x[5] = 0.0;
    }
    if (lock_outpost_center)
    {
      ekf_.x[0] = center_x_before;
      ekf_.x[1] = 0.0;
      ekf_.x[2] = center_y_before;
      ekf_.x[3] = 0.0;
      ekf_.x[4] = center_z_before;
      ekf_.x[5] = 0.0;
    }
  }

  /**
   * @brief 由 EKF 状态计算一个模型装甲面中心。
   */
  Eigen::Vector3d HArmorXyz(const Eigen::VectorXd& x, int id) const
  {
    auto angle = LimitRad(x[6] + id * 2.0 * kPi / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
    auto r = use_l_h ? x[8] + x[9] : x[8];
    auto armor_x = x[0] + r * std::sin(angle);
    auto armor_y = x[2] - r * std::cos(angle);
    auto armor_z = use_l_h ? x[4] + x[10] : x[4];
    if (UseOutpostHeightModel())
    {
      armor_z = x[4] + OutpostArmorHeightOffset(id, outpost_height_phase_);
    }
    return {armor_x, armor_y, armor_z};
  }

  /**
   * @brief 返回一个模型装甲面的观测雅可比矩阵。
   */
  Eigen::MatrixXd HJacobian(const Eigen::VectorXd& x, int id) const
  {
    auto angle = LimitRad(x[6] + id * 2.0 * kPi / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
    auto r = use_l_h ? x[8] + x[9] : x[8];
    auto dx_da = r * std::cos(angle);
    auto dy_da = r * std::sin(angle);
    auto dx_dr = std::sin(angle);
    auto dy_dr = -std::cos(angle);
    auto dx_dl = use_l_h ? std::sin(angle) : 0.0;
    auto dy_dl = use_l_h ? -std::cos(angle) : 0.0;
    auto dz_dh = use_l_h ? 1.0 : 0.0;

    Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, 11);
    H_armor_xyza(0, 0) = 1.0;
    H_armor_xyza(0, 6) = dx_da;
    H_armor_xyza(0, 8) = dx_dr;
    H_armor_xyza(0, 9) = dx_dl;
    H_armor_xyza(1, 2) = 1.0;
    H_armor_xyza(1, 6) = dy_da;
    H_armor_xyza(1, 8) = dy_dr;
    H_armor_xyza(1, 9) = dy_dl;
    H_armor_xyza(2, 4) = 1.0;
    H_armor_xyza(2, 10) = dz_dh;
    H_armor_xyza(3, 6) = 1.0;

    Eigen::VectorXd armor_xyz = HArmorXyz(x, id);
    Eigen::MatrixXd H_armor_ypd = XyzToYpdJacobian(armor_xyz);
    Eigen::MatrixXd H_armor_ypda(4, 4);
    H_armor_ypda << H_armor_ypd(0, 0), H_armor_ypd(0, 1),
        H_armor_ypd(0, 2), 0, H_armor_ypd(1, 0), H_armor_ypd(1, 1),
        H_armor_ypd(1, 2), 0, H_armor_ypd(2, 0), H_armor_ypd(2, 1),
        H_armor_ypd(2, 2), 0, 0, 0, 0, 1;
    return H_armor_ypda * H_armor_xyza;
  }
};

/**
 * @brief 基于装甲板位姿观测运行的目标状态机。
 */
class Tracker
{
 public:
  /**
   * @brief 使用位姿解算器构造 tracker。
   */
  Tracker(const Config& config, Solver& solver)
      : solver_(solver),
        require_target_tag_(config.require_target_tag),
        target_name_(ArmorNameFromDetectorNumber(config.target_tag_id)),
        min_detect_count_(config.min_detect_count),
        outpost_max_temp_lost_count_(config.outpost_max_temp_lost_count),
        normal_temp_lost_count_(config.max_temp_lost_count),
        state_("lost")
  {
    for (std::size_t i = 0; i < tracks_.size(); ++i)
    {
      tracks_[i].name = static_cast<ArmorName>(i);
    }
  }

  /**
   * @brief 返回当前状态机状态字符串。
   */
  const std::string& State() const { return state_; }

  /**
   * @brief 一个有效目标槽位的快照，用于 preview 和调试输出。
   */
  struct TrackSnapshot
  {
    std::string state{"lost"};
    bool selected{false};
    double score{-std::numeric_limits<double>::infinity()};
    Target target{};
  };

  /**
   * @brief 返回最近一次跟踪后的有效目标槽位。
   */
  std::vector<TrackSnapshot> Snapshots() const
  {
    std::vector<TrackSnapshot> snapshots;
    snapshots.reserve(tracks_.size());
    for (std::size_t i = 0; i < tracks_.size(); ++i)
    {
      if (!Selectable(tracks_[i]))
      {
        continue;
      }
      snapshots.push_back({tracks_[i].state,
                           static_cast<int>(i) == selected_index_,
                           tracks_[i].score, tracks_[i].target});
    }
    return snapshots;
  }

  /**
   * @brief 对候选装甲板执行一次跟踪。
   */
  std::list<Target> Track(std::list<Armor>& armors,
                          std::chrono::steady_clock::time_point t)
  {
    armors.sort(
        [](const Armor& a, const Armor& b)
        {
          cv::Point2f img_center(1440.0F / 2.0F, 1080.0F / 2.0F);
          auto distance_1 = cv::norm(a.center - img_center);
          auto distance_2 = cv::norm(b.center - img_center);
          return distance_1 < distance_2;
        });
    armors.sort([](const Armor& a, const Armor& b) {
      return a.priority < b.priority;
    });

    std::array<std::list<Armor>, kTrackSlotCount> armors_by_name;
    for (const auto& armor : armors)
    {
      const int index = TrackSlotIndex(armor.name);
      if (index >= 0)
      {
        armors_by_name[static_cast<std::size_t>(index)].push_back(armor);
      }
    }

    for (std::size_t i = 0; i < tracks_.size(); ++i)
    {
      UpdateSlot(tracks_[i], armors_by_name[i], t);
    }

    const int selected = SelectTrack();
    if (selected < 0)
    {
      state_ = "lost";
      selected_index_ = -1;
      return {};
    }

    selected_index_ = selected;
    const auto& slot = tracks_[static_cast<std::size_t>(selected_index_)];
    state_ = slot.state;
    return {slot.target};
  }

 private:
  /**
   * @brief 单个车辆编号的持久状态和选择指标。
   */
  struct TrackSlot
  {
    ArmorName name{ArmorName::NOT_ARMOR};
    std::string state{"lost"};
    Target target{};
    bool initialized{false};
    int detect_count{0};
    int temp_lost_count{0};
    int max_temp_lost_count{0};
    double score{-std::numeric_limits<double>::infinity()};
    double observed_count_lpf{0.0};
    double hittable_area{0.0};
    double gimbal_angle_error{1.0};
    bool has_timestamp{false};
    std::chrono::steady_clock::time_point last_timestamp{};
    bool outpost_center_hint_valid{false};
    Eigen::Vector3d outpost_center_hint{Eigen::Vector3d::Zero()};
  };

  Solver& solver_;
  bool require_target_tag_;
  ArmorName target_name_;
  int min_detect_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_;
  std::array<TrackSlot, kTrackSlotCount> tracks_{};
  int selected_index_ = -1;

  /**
   * @brief 将标量限制到 [0, 1]。
   */
  static double Clamp01(double value)
  {
    return std::clamp(value, 0.0, 1.0);
  }

  /**
   * @brief 判断槽位是否可作为公开目标输出。
   */
  static bool Selectable(const TrackSlot& slot)
  {
    return slot.initialized && slot.state != "lost" &&
           std::isfinite(slot.score);
  }

  /**
   * @brief 更新目标选择使用的当前帧观测指标。
   */
  static void UpdateObservationMetrics(TrackSlot& slot,
                                       const std::list<Armor>& armors)
  {
    slot.observed_count_lpf =
        0.8 * slot.observed_count_lpf +
        0.2 * static_cast<double>(armors.size());
    slot.hittable_area = 0.0;
    auto min_angle_error = std::numeric_limits<double>::infinity();
    for (const auto& armor : armors)
    {
      if (armor.points.size() == 4)
      {
        slot.hittable_area += std::abs(cv::contourArea(armor.points));
      }
      const auto dx = static_cast<double>(armor.center_norm.x) - 0.5;
      const auto dy = static_cast<double>(armor.center_norm.y) - 0.5;
      min_angle_error = std::min(min_angle_error, std::hypot(dx, dy));
    }
    if (std::isfinite(min_angle_error))
    {
      slot.gimbal_angle_error = min_angle_error;
    }
  }

  /**
   * @brief 判断槽位 EKF 最近是否未通过 innovation gate。
   */
  static bool RecentNisFailed(const TrackSlot& slot)
  {
    const auto& ekf = slot.target.Ekf();
    return std::accumulate(ekf.recent_nis_failures.begin(),
                           ekf.recent_nis_failures.end(), 0) >=
           0.4 * static_cast<double>(ekf.window_size);
  }

  /**
   * @brief 判断保留的 EKF 状态是否已经不可用。
   */
  static bool TargetHealthFailed(const TrackSlot& slot)
  {
    if (!slot.initialized)
    {
      return false;
    }
    if (slot.target.Diverged())
    {
      return true;
    }
    // 前哨站单面/侧面 PnP 平移噪声明显，短窗 NIS 失败不能直接清空
    // 固定塔心和高度相位；否则会在换面附近误重建为另一个 height phase。
    if (slot.target.name == ArmorName::OUTPOST)
    {
      return false;
    }
    return RecentNisFailed(slot);
  }

  /**
   * @brief 用上一次稳定中心给前哨站重建目标时选择初始高度相位。
   */
  static std::pair<int, bool> ChooseOutpostInitialHeightPhase(
      const TrackSlot& slot, const Armor& armor)
  {
    if (!slot.outpost_center_hint_valid)
    {
      return {0, false};
    }

    int best_phase = 0;
    double best_error = std::numeric_limits<double>::infinity();
    for (int phase = 0; phase < 3; ++phase)
    {
      const double center_z =
          armor.xyz_in_world.z() - OutpostArmorHeightOffset(0, phase);
      const double error = std::abs(center_z - slot.outpost_center_hint.z());
      if (error < best_error)
      {
        best_error = error;
        best_phase = phase;
      }
    }

    constexpr double kCenterHintGate = 0.06;
    return {best_phase, best_error <= kCenterHintGate};
  }

  /**
   * @brief EKF 健康检查失败后清空保留目标状态。
   *
   * The slot identity, timestamp, and current-frame observation metrics are
   * preserved; only the target/EKF and state-machine counters are discarded.
   */
  static void ResetTargetSlot(TrackSlot& slot)
  {
    slot.target = Target{};
    slot.initialized = false;
    slot.state = "lost";
    slot.detect_count = 0;
    slot.temp_lost_count = 0;
    slot.max_temp_lost_count = 0;
    slot.score = -std::numeric_limits<double>::infinity();
  }

  /**
   * @brief 更新当前帧中的一个车辆编号槽位。
   */
  void UpdateSlot(TrackSlot& slot, std::list<Armor>& armors,
                  std::chrono::steady_clock::time_point t)
  {
    UpdateObservationMetrics(slot, armors);

    if (slot.has_timestamp)
    {
      const auto dt = DeltaTime(t, slot.last_timestamp);
      if (slot.state != "lost" && dt > 0.1)
      {
        slot.state = "lost";
      }
    }
    slot.last_timestamp = t;
    slot.has_timestamp = true;

    if (TargetHealthFailed(slot))
    {
      ResetTargetSlot(slot);
    }

    bool found = false;
    if (!slot.initialized)
    {
      found = SetTarget(slot, armors, t);
    }
    else if (slot.state == "lost" && armors.empty())
    {
      found = false;
    }
    else
    {
      found = UpdateTarget(slot, armors, t);
    }

    if (TargetHealthFailed(slot))
    {
      ResetTargetSlot(slot);
      found = SetTarget(slot, armors, t);
    }
    StateMachine(slot, found);
    if (slot.state != "lost" && TargetHealthFailed(slot))
    {
      ResetTargetSlot(slot);
    }
    if (slot.initialized && slot.state != "lost" &&
        slot.target.name == ArmorName::OUTPOST)
    {
      slot.outpost_center_hint = slot.target.CenterWorldForOutput();
      slot.outpost_center_hint_valid = true;
    }
    slot.score = ScoreSlot(slot);
  }

  /**
   * @brief 计算一个槽位的当前目标选择分数。
   */
  static double ScoreSlot(const TrackSlot& slot)
  {
    if (!slot.initialized || slot.state == "lost")
    {
      return -std::numeric_limits<double>::infinity();
    }

    const Eigen::VectorXd x = slot.target.EkfX();
    const double distance = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);
    const double distance_score = Clamp01((8.0 - distance) / 7.5);
    const double area_score = Clamp01(slot.hittable_area / 6000.0);
    const double count_score = Clamp01(slot.observed_count_lpf / 4.0);
    const double spin_score = Clamp01(1.0 - std::abs(x[7]) / 8.0);
    const double angle_score = Clamp01(1.0 - slot.gimbal_angle_error / 0.5);

    double state_scale = 1.0;
    if (slot.state == "detecting")
    {
      state_scale = 0.55;
    }
    else if (slot.state == "temp_lost")
    {
      state_scale = 0.35;
    }

    return state_scale * (2.0 * count_score + 1.2 * distance_score +
                          1.5 * area_score + spin_score + angle_score);
  }

  /**
   * @brief 使用小滞回选择输出目标。
   */
  int SelectTrack() const
  {
    if (require_target_tag_)
    {
      const int required_index = TrackSlotIndex(target_name_);
      if (required_index >= 0 &&
          Selectable(tracks_[static_cast<std::size_t>(required_index)]))
      {
        return required_index;
      }
      return -1;
    }

    int best_index = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < tracks_.size(); ++i)
    {
      if (!Selectable(tracks_[i]) || tracks_[i].score <= best_score)
      {
        continue;
      }
      best_score = tracks_[i].score;
      best_index = static_cast<int>(i);
    }
    if (best_index < 0)
    {
      return -1;
    }

    if (selected_index_ >= 0 &&
        selected_index_ < static_cast<int>(tracks_.size()) &&
        Selectable(tracks_[static_cast<std::size_t>(selected_index_)]))
    {
      constexpr double kSwitchMargin = 0.25;
      const auto& selected = tracks_[static_cast<std::size_t>(selected_index_)];
      if (best_index != selected_index_ &&
          tracks_[static_cast<std::size_t>(best_index)].score <=
              selected.score + kSwitchMargin)
      {
        return selected_index_;
      }
    }
    return best_index;
  }

  /**
   * @brief 推进 lost/detecting/tracking/temp-lost 状态机。
   */
  void StateMachine(TrackSlot& slot, bool found)
  {
    if (slot.state == "lost")
    {
      if (!found)
      {
        return;
      }
      slot.state = "detecting";
      slot.detect_count = 1;
    }
    else if (slot.state == "detecting")
    {
      if (found)
      {
        ++slot.detect_count;
        if (slot.detect_count >= min_detect_count_)
        {
          slot.state = "tracking";
        }
      }
      else
      {
        slot.detect_count = 0;
        slot.state = "lost";
      }
    }
    else if (slot.state == "tracking")
    {
      if (found)
      {
        return;
      }
      slot.temp_lost_count = 1;
      slot.state = "temp_lost";
    }
    else if (slot.state == "switching")
    {
      if (found)
      {
        slot.state = "detecting";
      }
      else
      {
        ++slot.temp_lost_count;
        if (slot.temp_lost_count > 200)
        {
          slot.state = "lost";
        }
      }
    }
    else if (slot.state == "temp_lost")
    {
      if (found)
      {
        slot.state = "tracking";
      }
      else
      {
        ++slot.temp_lost_count;
        if (slot.target.name == ArmorName::OUTPOST)
        {
          slot.max_temp_lost_count = outpost_max_temp_lost_count_;
        }
        else
        {
          slot.max_temp_lost_count = normal_temp_lost_count_;
        }
        if (slot.temp_lost_count > slot.max_temp_lost_count)
        {
          slot.state = "lost";
        }
      }
    }
  }

  /**
   * @brief 使用排序后的最佳装甲板候选初始化目标状态。
   */
  bool SetTarget(TrackSlot& slot, std::list<Armor>& armors,
                 std::chrono::steady_clock::time_point t)
  {
    if (armors.empty())
    {
      return false;
    }
    auto& armor = armors.front();
    solver_.Solve(armor);

    const auto is_balance =
        armor.type == ArmorType::BIG &&
        (armor.name == ArmorName::THREE || armor.name == ArmorName::FOUR ||
         armor.name == ArmorName::FIVE);

    Eigen::VectorXd P0_dig(11);
    if (is_balance)
    {
      P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
      slot.target = Target(armor, t, 0.2, 2, P0_dig);
    }
    else if (armor.name == ArmorName::OUTPOST)
    {
      P0_dig << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0;
      const auto [height_phase, height_phase_valid] =
          ChooseOutpostInitialHeightPhase(slot, armor);
      slot.target = Target(armor, t, kOutpostArmorRadius, 3, P0_dig, height_phase,
                           height_phase_valid, slot.outpost_center_hint,
                           height_phase_valid &&
                               slot.outpost_center_hint_valid);
    }
    else if (armor.name == ArmorName::BASE)
    {
      P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
      slot.target = Target(armor, t, 0.3205, 3, P0_dig);
    }
    else
    {
      P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
      slot.target = Target(armor, t, 0.2, 4, P0_dig);
    }
    slot.initialized = true;
    slot.max_temp_lost_count =
        armor.name == ArmorName::OUTPOST ? outpost_max_temp_lost_count_
                                         : normal_temp_lost_count_;
    return true;
  }

  /**
   * @brief 使用匹配装甲板预测并更新当前目标。
   */
  bool UpdateTarget(TrackSlot& slot, std::list<Armor>& armors,
                    std::chrono::steady_clock::time_point t)
  {
    slot.target.Predict(t);
    int found_count = 0;
    for (const auto& armor : armors)
    {
      if (armor.name != slot.target.name || armor.type != slot.target.armor_type)
      {
        continue;
      }
      ++found_count;
    }
    if (found_count == 0)
    {
      return false;
    }

    for (auto& armor : armors)
    {
      if (armor.name != slot.target.name || armor.type != slot.target.armor_type)
      {
        continue;
      }
      solver_.Solve(armor);
      slot.target.Update(armor);
    }
    return true;
  }
};

}  // namespace armor_tracker_detail
