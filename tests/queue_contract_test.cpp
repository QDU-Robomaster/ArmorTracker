#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

#include "ArmorTrackerQueue.hpp"

namespace
{

using armor_tracker_pipeline::FixedSlotQueue;
using armor_tracker_pipeline::PipelineDrained;
using namespace std::chrono_literals;

void Expect(bool condition, const char* message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

struct NonMovableSlot
{
  int value{-1};

  NonMovableSlot() = default;
  NonMovableSlot(const NonMovableSlot&) = delete;
  NonMovableSlot& operator=(const NonMovableSlot&) = delete;
  NonMovableSlot(NonMovableSlot&&) = delete;
  NonMovableSlot& operator=(NonMovableSlot&&) = delete;
};

template <std::size_t Capacity>
void Push(FixedSlotQueue<NonMovableSlot, Capacity>& queue, int value)
{
  const auto reservation = queue.WaitAcquire();
  reservation.slot->value = value;
  queue.Commit(reservation.slot);
}

template <std::size_t Capacity>
int Pop(FixedSlotQueue<NonMovableSlot, Capacity>& queue)
{
  NonMovableSlot* const slot = queue.WaitFront();
  const int value = slot->value;
  queue.ReleaseFront(slot);
  return value;
}

void TestSeventeenthProducerBlocksUntilRelease()
{
  FixedSlotQueue<NonMovableSlot, 16> queue;
  for (int value = 0; value < 16; ++value)
  {
    const auto reservation = queue.WaitAcquire();
    Expect(!reservation.waited_for_full,
           "the first sixteen reservations must not wait for capacity");
    reservation.slot->value = value;
    queue.Commit(reservation.slot);
  }

  std::atomic<bool> producer_entered{false};
  auto seventeenth = std::async(std::launch::async,
                                [&queue, &producer_entered]()
                                {
                                  producer_entered.store(true, std::memory_order_release);
                                  const auto reservation = queue.WaitAcquire();
                                  reservation.slot->value = 16;
                                  queue.Commit(reservation.slot);
                                  return reservation.waited_for_full;
                                });

  const auto entry_deadline = std::chrono::steady_clock::now() + 1s;
  while (!producer_entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < entry_deadline)
  {
    std::this_thread::yield();
  }
  Expect(producer_entered.load(std::memory_order_acquire),
         "seventeenth producer did not start");
  Expect(seventeenth.wait_for(30ms) == std::future_status::timeout,
         "seventeenth producer must block while all slots are occupied");

  NonMovableSlot* const first = queue.WaitFront();
  Expect(first->value == 0, "first queued value must be zero");
  queue.ReleaseFront(first);

  Expect(seventeenth.wait_for(1s) == std::future_status::ready,
         "releasing one slot must unblock the seventeenth producer");
  Expect(seventeenth.get(), "seventeenth producer must report a full wait");

  for (int expected = 1; expected <= 16; ++expected)
  {
    NonMovableSlot* const slot = queue.WaitFront();
    Expect(slot->value == expected, "blocked producer must preserve FIFO order");
    queue.ReleaseFront(slot);
  }

  const auto snapshot = queue.Snapshot();
  Expect(snapshot.occupied == 0U && snapshot.ready == 0U,
         "queue must be empty after all seventeen values are released");
  Expect(snapshot.high_water == 16U, "queue high-water must reach capacity");
  Expect(snapshot.full_wait_count == 1U,
         "exactly one producer must observe a full queue");
}

void TestWraparoundPreservesOrder()
{
  FixedSlotQueue<NonMovableSlot, 3> queue;
  Push(queue, 0);
  Push(queue, 1);
  Push(queue, 2);
  Expect(Pop(queue) == 0, "first wraparound prefix value");
  Expect(Pop(queue) == 1, "second wraparound prefix value");
  Push(queue, 3);
  Push(queue, 4);
  Expect(Pop(queue) == 2, "wraparound retained value");
  Expect(Pop(queue) == 3, "first wrapped value");
  Expect(Pop(queue) == 4, "second wrapped value");
}

void TestWaitingProducerWakesAfterCommit()
{
  FixedSlotQueue<NonMovableSlot, 2> queue;
  const auto first = queue.WaitAcquire();
  first.slot->value = 10;

  std::atomic<bool> producer_entered{false};
  auto second = std::async(std::launch::async,
                           [&queue, &producer_entered]()
                           {
                             producer_entered.store(true, std::memory_order_release);
                             const auto reservation = queue.WaitAcquire();
                             reservation.slot->value = 11;
                             queue.Commit(reservation.slot);
                           });

  const auto entry_deadline = std::chrono::steady_clock::now() + 1s;
  while (!producer_entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < entry_deadline)
  {
    std::this_thread::yield();
  }
  Expect(producer_entered.load(std::memory_order_acquire),
         "second producer did not start");
  Expect(second.wait_for(30ms) == std::future_status::timeout,
         "a second producer must wait while the first reservation is active");

  queue.Commit(first.slot);
  Expect(second.wait_for(1s) == std::future_status::ready,
         "committing a reservation must wake the next producer");
  second.get();

  Expect(Pop(queue) == 10, "first committed producer must remain first");
  Expect(Pop(queue) == 11, "producer awakened by commit must remain second");
}

void TestActiveWorkerPreventsDrain()
{
  FixedSlotQueue<NonMovableSlot, 1> queue;
  const auto reservation = queue.WaitAcquire();
  reservation.slot->value = 7;
  queue.Commit(reservation.slot);

  NonMovableSlot* const active = queue.WaitFront();
  Expect(!PipelineDrained(1U, 1U, queue.Snapshot()),
         "equal counters cannot drain while the worker owns a slot");
  queue.ReleaseFront(active);
  Expect(PipelineDrained(1U, 1U, queue.Snapshot()),
         "equal counters and an idle empty queue must drain");
}

struct CountingImage
{
  int pixel{0};
  int* copy_count{nullptr};

  CountingImage(int value, int* copies) : pixel(value), copy_count(copies) {}

  CountingImage(const CountingImage& source)
      : pixel(source.pixel), copy_count(source.copy_count)
  {
    ++*copy_count;
  }
};

struct OwnershipSlot
{
  uint64_t sequence{0};
  uint64_t timestamp_us{0};
  std::shared_ptr<CountingImage> image{};
};

void TestOwnerMovesOutWithoutPixelCopy()
{
  FixedSlotQueue<OwnershipSlot, 2> queue;
  int copy_count = 0;
  auto publisher_owner = std::make_shared<CountingImage>(42, &copy_count);
  const std::weak_ptr<CountingImage> observed = publisher_owner;

  const auto reservation = queue.WaitAcquire();
  reservation.slot->sequence = 17U;
  reservation.slot->timestamp_us = 900U;
  reservation.slot->image = publisher_owner;
  queue.Commit(reservation.slot);

  Expect(copy_count == 0, "copying an owner must not copy image bytes");
  publisher_owner.reset();

  OwnershipSlot* const slot = queue.WaitFront();
  auto worker_owner = std::move(slot->image);
  Expect(!slot->image, "moving to the worker must empty FIFO ownership");
  queue.ReleaseFront(slot);
  Expect(!observed.expired(), "the worker-local owner must keep the image alive");
  Expect(worker_owner->pixel == 42, "the worker must resolve the original image");
  Expect(copy_count == 0, "the worker handoff must not copy image bytes");

  worker_owner.reset();
  Expect(observed.expired(), "the final owner reset must release the image immediately");
}

void TestReleaseDropsUnmovedOwner()
{
  FixedSlotQueue<OwnershipSlot, 1> queue;
  int copy_count = 0;
  auto publisher_owner = std::make_shared<CountingImage>(7, &copy_count);
  const std::weak_ptr<CountingImage> observed = publisher_owner;

  const auto reservation = queue.WaitAcquire();
  reservation.slot->image = publisher_owner;
  queue.Commit(reservation.slot);
  publisher_owner.reset();

  OwnershipSlot* const slot = queue.WaitFront();
  queue.ReleaseFront(slot);
  Expect(observed.expired(),
         "releasing a consumed slot must destroy any owner left on the slot");
  Expect(copy_count == 0, "release must not copy image bytes");
}

void TestDuplicateTimestampsRemainDistinctAndOrdered()
{
  FixedSlotQueue<OwnershipSlot, 2> queue;
  for (uint64_t sequence : {51U, 52U})
  {
    const auto reservation = queue.WaitAcquire();
    reservation.slot->sequence = sequence;
    reservation.slot->timestamp_us = 1234U;
    queue.Commit(reservation.slot);
  }

  OwnershipSlot* first = queue.WaitFront();
  Expect(first->timestamp_us == 1234U && first->sequence == 51U,
         "the first duplicate timestamp must keep its own sequence");
  queue.ReleaseFront(first);

  OwnershipSlot* second = queue.WaitFront();
  Expect(second->timestamp_us == 1234U && second->sequence == 52U,
         "the second duplicate timestamp must remain FIFO ordered");
  queue.ReleaseFront(second);
}

}  // namespace

int main()
{
  try
  {
    TestSeventeenthProducerBlocksUntilRelease();
    TestWraparoundPreservesOrder();
    TestWaitingProducerWakesAfterCommit();
    TestActiveWorkerPreventsDrain();
    TestOwnerMovesOutWithoutPixelCopy();
    TestReleaseDropsUnmovedOwner();
    TestDuplicateTimestampsRemainDistinctAndOrdered();
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "PASS: ArmorTracker queue contract (7/7)\n";
  return EXIT_SUCCESS;
}
