#include <cstdlib>

#include "../../../User/ReplayBenchmark.hpp"

int main()
{
  AutoAimReplayBenchmark::RecordTrackerQueueAdmission(1U, 1U, 0.25, true, 15U, 16U,
                                                       16U);
  AutoAimReplayBenchmark::RecordTrackerWorkerService(1U, 1U, 1U, 1.5);
  return EXIT_SUCCESS;
}
