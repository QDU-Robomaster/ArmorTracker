#include <cstdlib>
#include <type_traits>
#include <utility>

#include "ArmorDetectorTypes.hpp"
#include "ArmorTrackerTarget.hpp"

namespace
{
constexpr CameraTypes::FrameLayout kLayout{720, 540, 2160, CameraTypes::Encoding::BGR8};

using Detection = DetectedFrame<kLayout>;
using Tracking = TrackedFrame<kLayout>;

static_assert(std::is_same_v<DetectedFrameMessage<kLayout>, const Detection*>);
static_assert(std::is_same_v<TrackedFrameMessage<kLayout>, const Tracking*>);
static_assert(std::is_same_v<decltype(std::declval<const Detection&>().GetImageFrame()),
                             const Detection::ImageFrame*>);
static_assert(std::is_same_v<decltype(std::declval<const Tracking&>().GetImageFrame()),
                             const Tracking::ImageFrame*>);
static_assert(std::is_same_v<decltype(Tracking::target), ArmorTrackerTarget>);
static_assert(!std::is_pointer_v<decltype(Tracking::target)>);
static_assert(!std::is_pointer_v<decltype(Tracking::imu)>);
}  // namespace

int main() { return EXIT_SUCCESS; }
