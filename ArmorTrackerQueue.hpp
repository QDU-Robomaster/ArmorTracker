#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace armor_tracker_pipeline
{

struct FixedSlotQueueSnapshot
{
  std::size_t ready{0};
  std::size_t occupied{0};
  std::size_t high_water{0};
  uint64_t full_wait_count{0};
  uint64_t producer_wait_ns{0};
  bool producer_active{false};
  bool worker_active{false};
};

template <typename T, std::size_t Capacity>
class FixedSlotQueue
{
 public:
  static_assert(Capacity > 0, "FixedSlotQueue requires non-zero capacity");

  struct WriteReservation
  {
    T* slot{nullptr};
    uint64_t producer_wait_ns{0};
    bool waited_for_full{false};
  };

  struct CommitResult
  {
    std::size_t ready{0};
    std::size_t occupied{0};
    std::size_t high_water{0};
  };

  WriteReservation WaitAcquire()
  {
    const auto wait_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    const bool waited_for_full = occupied_ == Capacity;
    if (waited_for_full)
    {
      ++full_wait_count_;
    }
    not_full_cv_.wait(lock,
                      [this]() { return occupied_ < Capacity && !producer_active_; });

    producer_active_ = true;
    ++occupied_;
    const uint64_t producer_wait_ns =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - wait_start)
                                  .count());
    producer_wait_ns_ += producer_wait_ns;
    return WriteReservation{.slot = &slots_[write_index_],
                            .producer_wait_ns = producer_wait_ns,
                            .waited_for_full = waited_for_full};
  }

  CommitResult Commit(T* slot)
  {
    CommitResult result{};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      assert(producer_active_);
      assert(slot == &slots_[write_index_]);
      (void)slot;
      producer_active_ = false;
      write_index_ = Next(write_index_);
      ++ready_;
      high_water_ = std::max(high_water_, occupied_);
      result =
          CommitResult{.ready = ready_, .occupied = occupied_, .high_water = high_water_};
    }
    not_empty_cv_.notify_one();
    not_full_cv_.notify_one();
    return result;
  }

  T* WaitFront()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_cv_.wait(lock, [this]() { return ready_ != 0U && !worker_active_; });
    worker_active_ = true;
    --ready_;
    return &slots_[read_index_];
  }

  void ReleaseFront(T* slot)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      assert(worker_active_);
      assert(slot == &slots_[read_index_]);
      assert(occupied_ != 0U);

      // Drop every resource owned by the consumed slot before producers can reuse it.
      std::destroy_at(slot);
      std::construct_at(slot);

      worker_active_ = false;
      --occupied_;
      read_index_ = Next(read_index_);
    }
    not_full_cv_.notify_one();
  }

  [[nodiscard]] FixedSlotQueueSnapshot Snapshot() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return FixedSlotQueueSnapshot{
        .ready = ready_,
        .occupied = occupied_,
        .high_water = high_water_,
        .full_wait_count = full_wait_count_,
        .producer_wait_ns = producer_wait_ns_,
        .producer_active = producer_active_,
        .worker_active = worker_active_,
    };
  }

 private:
  static constexpr std::size_t Next(std::size_t index) { return (index + 1U) % Capacity; }

  std::array<T, Capacity> slots_;
  std::size_t read_index_{0};
  std::size_t write_index_{0};
  std::size_t ready_{0};
  std::size_t occupied_{0};
  std::size_t high_water_{0};
  uint64_t full_wait_count_{0};
  uint64_t producer_wait_ns_{0};
  bool producer_active_{false};
  bool worker_active_{false};
  mutable std::mutex mutex_{};
  std::condition_variable not_empty_cv_{};
  std::condition_variable not_full_cv_{};
};

inline bool PipelineDrained(uint64_t enqueued, uint64_t processed,
                            const FixedSlotQueueSnapshot& queue) noexcept
{
  return enqueued == processed && queue.ready == 0U && queue.occupied == 0U &&
         !queue.producer_active && !queue.worker_active;
}

}  // namespace armor_tracker_pipeline
