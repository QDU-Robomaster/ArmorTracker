add_executable(armor_tracker_frame_geometry_test frame_geometry_test.cpp)

target_include_directories(armor_tracker_frame_geometry_test
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)

target_link_libraries(armor_tracker_frame_geometry_test PRIVATE xr)
target_compile_features(armor_tracker_frame_geometry_test PRIVATE cxx_std_20)

add_test(NAME armor_tracker_frame_geometry_test
         COMMAND armor_tracker_frame_geometry_test)

add_executable(armor_tracker_queue_contract_test queue_contract_test.cpp)

find_package(Threads REQUIRED)
target_include_directories(armor_tracker_queue_contract_test
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(armor_tracker_queue_contract_test PRIVATE Threads::Threads)
target_compile_features(armor_tracker_queue_contract_test PRIVATE cxx_std_20)

add_test(NAME armor_tracker_queue_contract_test
         COMMAND armor_tracker_queue_contract_test)

add_executable(armor_tracker_replay_metrics_compile_test
               replay_metrics_compile_test.cpp)
target_link_libraries(armor_tracker_replay_metrics_compile_test
                      PRIVATE Threads::Threads)
target_compile_features(armor_tracker_replay_metrics_compile_test PRIVATE cxx_std_20)
if(MSVC)
  target_compile_definitions(armor_tracker_replay_metrics_compile_test
                             PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_test(NAME armor_tracker_replay_metrics_compile_test
         COMMAND armor_tracker_replay_metrics_compile_test)

add_executable(armor_tracker_stage_frame_contract_test
               stage_frame_contract_test.cpp)
target_include_directories(armor_tracker_stage_frame_contract_test
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(armor_tracker_stage_frame_contract_test PRIVATE xr)
target_compile_features(armor_tracker_stage_frame_contract_test PRIVATE cxx_std_20)

add_test(NAME armor_tracker_stage_frame_contract_test
         COMMAND armor_tracker_stage_frame_contract_test)
