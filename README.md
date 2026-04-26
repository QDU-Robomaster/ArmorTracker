# ArmorTracker

`ArmorTracker` is the target-level tracker module used by the Webots/Linux
auto-aim pipeline.

## Current Layout

- `ArmorTracker.hpp`: tracker orchestration, EKF wiring, topics, state machine
- `ArmorTrackerCommon.hpp`: shared yaw/time/image-area utilities
- `ArmorTrackerRuntimeSupport.hpp`: runtime/env helper functions
- `ArmorTrackerSelectionSupport.hpp`: face-binding and selection logging helpers
- `ArmorTrackerDebugSupport.hpp`: preview overlay and candidate-debug helpers
- `ArmorTrackerFaceSelector.hpp`: face selection policy and candidate scoring
- `ArmorTrackerImageTracker.hpp`: image-space armor identity tracking
- `ArmorTrackerObserver.hpp`: vehicle geometry observation model
- `extended_kalman_filter.*`: generic EKF implementation
- `SolveTrajectory.*`, `TrajectoryCompensationTable.hpp`, `table.bin`:
  projectile compensation support

## Trajectory Table

`table.bin` is a runtime asset for trajectory compensation.

If the table must be regenerated, build and run `TableGenerator.cpp`
separately, then replace `table.bin`.

## Build Boundary

- Runtime module sources:
  - `SolveTrajectory.cpp`
  - `extended_kalman_filter.cpp`
- The tracker itself is header-only.
- `TableGenerator.cpp` is an offline tool and must not be linked into the
  runtime target.
