#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 装甲板跟踪：按对方颜色选板，四块板的整车用整车估计器，其余用兜底 EKF，处理受击灭灯与阵亡，多目标中选出要打的一个，发布跟踪帧 / Armor tracking that picks plates of the opponent colour, uses the vehicle estimator for four-plate vehicles and a fallback EKF for the rest, handles hit flashes and destroyed robots, picks one of several targets and publishes tracked frames
depends:
- id: QDU-Robomaster/CameraBase
  ref: same-or-dev
- id: QDU-Robomaster/AutoAimTypes
  ref: same-or-dev
=== END MANIFEST === */
// clang-format on

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "AutoAimTypes.hpp"
#include "TrackSet.hpp"
#include "libxr_def.hpp"
#include "logger.hpp"
#include "message.hpp"

/**
 * @brief 装甲板跟踪。订阅 `<相机名>_detected`，发布 `<相机名>_tracked`。
 *        Armor tracking. Subscribes to `<camera>_detected`, publishes `<camera>_tracked`.
 *
 * 检测帧进一个有界队列，满了在检测器的发布线程里等待，所以每一帧都被跟踪；工作线程按顺序
 * 处理并发布，每收一帧发一帧。检测器发布所有颜色，这里按 target_color 只跟踪
 * 对方颜色的亮板。
 * Detected frames go into a bounded queue; when it is full the detector's publishing
 * thread waits, so every frame is tracked. The worker thread processes and publishes in
 * order, one frame out per frame in. The detector publishes every colour; only lit
 * plates of the opponent colour (target_color) are tracked here.
 */
class ArmorTracker
{
 public:
  /// 队列容量：排队的帧占着相机图像槽 / Queue capacity; queued frames hold image slots.
  static constexpr std::size_t QUEUE_CAPACITY = 2;
  /// 裁判系统摘要包 Topic，首字节为本机 robot_id / Referee summary Topic; its first byte
  /// is the robot_id.
  static constexpr const char* REFEREE_TOPIC = "robot_game_ref";
  static constexpr const char* REFEREE_DOMAIN = "host";

  explicit ArmorTracker(const TrackerSettings& settings)
      : camera_name_(settings.camera_name),
        tracks_(settings),
        tracked_topic_(LibXR::Topic::CreateTopic<const AutoAim::TrackedFrame*>(
            StageTopicName(camera_name_, AutoAim::STAGE_TRACKED).c_str()))
  {
    switch (settings.target_color)
    {
      case TargetColor::RED:
        enemy_.store(ArmorColor::RED);
        break;
      case TargetColor::BLUE:
        enemy_.store(ArmorColor::BLUE);
        break;
      case TargetColor::FROM_REFEREE:
        // 收到第一包之前不跟踪 / Nothing is tracked before the first packet.
        SubscribeReferee();
        break;
    }
    auto on_detected = LibXR::Topic::Callback::Create(
        [](bool, ArmorTracker* self, const AutoAim::DetectedFrame* frame)
        { self->Push(*frame); }, this);
    AutoAim::RequireTopic<const AutoAim::DetectedFrame*>(
        StageTopicName(camera_name_, AutoAim::STAGE_DETECTED))
        .RegisterCallback(on_detected);
    running_.store(true);
    worker_ = std::thread([this]() { WorkerLoop(); });
  }

  ~ArmorTracker()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_.store(false);
    }
    not_empty_.notify_all();
    not_full_.notify_all();
    worker_.join();
  }

  ArmorTracker(const ArmorTracker&) = delete;
  ArmorTracker& operator=(const ArmorTracker&) = delete;

  /// 打印周期摘要 / Print the periodic summary.
  void OnMonitor()
  {
    XR_LOG_INFO("%s tracker: frames=%u tracking=%u waits=%u", camera_name_.c_str(),
                frames_.exchange(0), tracking_.exchange(0), waits_.exchange(0));
  }

 private:
  void SubscribeReferee()
  {
    referee_domain_.emplace(REFEREE_DOMAIN);
    LibXR::Topic::TopicHandle topic =
        LibXR::Topic::Find(REFEREE_TOPIC, &*referee_domain_);
    if (topic == nullptr)
    {
      XR_LOG_ERROR("target_color FROM_REFEREE needs the Topic %s/%s", REFEREE_DOMAIN,
                   REFEREE_TOPIC);
      REQUIRE(false);
    }
    auto on_referee = LibXR::Topic::Callback::Create(
        [](bool, ArmorTracker* self, const LibXR::ConstRawData& data)
        {
          if (data.addr_ == nullptr || data.size_ < 1)
          {
            return;
          }
          // 1–99 为红方，101–199 为蓝方 / 1–99 red, 101–199 blue.
          const uint8_t id = *static_cast<const uint8_t*>(data.addr_);
          if (id >= 1 && id < 100)
          {
            self->enemy_.store(ArmorColor::BLUE);
          }
          else if (id >= 101 && id < 200)
          {
            self->enemy_.store(ArmorColor::RED);
          }
        },
        this);
    LibXR::Topic(topic).RegisterCallback(on_referee);
  }

  void Push(const AutoAim::DetectedFrame& frame)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.size() >= QUEUE_CAPACITY)
    {
      waits_.fetch_add(1, std::memory_order_relaxed);
      not_full_.wait(
          lock, [this]() { return queue_.size() < QUEUE_CAPACITY || !running_.load(); });
    }
    if (!running_.load())
    {
      return;
    }
    queue_.push_back(frame);
    lock.unlock();
    not_empty_.notify_one();
  }

  void WorkerLoop()
  {
    while (true)
    {
      AutoAim::DetectedFrame detected;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() { return !queue_.empty() || !running_.load(); });
        if (!running_.load())
        {
          queue_.clear();
          return;
        }
        detected = std::move(queue_.front());
        queue_.pop_front();
      }
      not_full_.notify_one();
      Process(std::move(detected));
    }
  }

  void Process(AutoAim::DetectedFrame&& detected)
  {
    AutoAim::TrackedFrame tracked;
    tracked.detected = std::move(detected);
    const AutoAim::SyncedFrame& synced = tracked.detected.synced;
    tracked.target =
        tracks_.Step(static_cast<uint64_t>(synced.imu.timestamp_us),
                     synced.imu.rotation_wxyz, synced.imu.angular_velocity_xyz,
                     *synced.image->calibration, tracked.detected.armors, enemy_.load());
    tracks_.WorldToCamera(tracked.output_to_camera_rotation,
                          tracked.output_to_camera_translation);
    const AutoAim::TrackedFrame* payload = &tracked;
    tracked_topic_.Publish(payload);
    frames_.fetch_add(1, std::memory_order_relaxed);
    if (tracked.target.tracking)
    {
      tracking_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  const std::string camera_name_;
  TrackSet tracks_;
  LibXR::Topic tracked_topic_;
  std::optional<LibXR::Topic::Domain> referee_domain_;
  std::atomic<ArmorColor> enemy_{ArmorColor::UNKNOWN};
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<AutoAim::DetectedFrame> queue_;
  std::atomic<bool> running_{false};
  std::atomic<uint32_t> frames_{0};
  std::atomic<uint32_t> tracking_{0};
  std::atomic<uint32_t> waits_{0};
  std::thread worker_;
};
