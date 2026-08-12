add_executable(armor_tracker_frame_geometry_test frame_geometry_test.cpp)

target_include_directories(armor_tracker_frame_geometry_test
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)

target_link_libraries(armor_tracker_frame_geometry_test PRIVATE xr)
target_compile_features(armor_tracker_frame_geometry_test PRIVATE cxx_std_20)

add_test(NAME armor_tracker_frame_geometry_test
         COMMAND armor_tracker_frame_geometry_test)
