find_package(Threads REQUIRED)

add_executable(armor_tracker_test tracker_test.cpp)
target_include_directories(armor_tracker_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(armor_tracker_test PRIVATE xr ArmorTracker_deps Threads::Threads)
target_compile_features(armor_tracker_test PRIVATE cxx_std_20)
add_test(NAME armor_tracker_test COMMAND armor_tracker_test)
set_tests_properties(armor_tracker_test PROPERTIES TIMEOUT 60)

# 与 aasim aaest_replay 逐帧对拍用的回放工具（不是测试）/ Replay tool for the parity
# check against aasim aaest_replay (not a test).
add_executable(vehicle_replay ${CMAKE_CURRENT_LIST_DIR}/../tools/vehicle_replay/vehicle_replay.cpp)
target_include_directories(vehicle_replay PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(vehicle_replay PRIVATE xr ArmorTracker_deps)
target_compile_features(vehicle_replay PRIVATE cxx_std_20)
