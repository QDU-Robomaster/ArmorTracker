# ArmorTracker runtime module
target_include_directories(xr PUBLIC ${CMAKE_CURRENT_LIST_DIR})

set(MODULE_ARMORTRACKER_SRC
  "${CMAKE_CURRENT_LIST_DIR}/SolveTrajectory.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/extended_kalman_filter.cpp"
)

target_sources(xr PRIVATE ${MODULE_ARMORTRACKER_SRC})

get_filename_component(_MODULE_DIRNAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(_DEPS_TARGET "${_MODULE_DIRNAME}_deps")
add_library(${_DEPS_TARGET} INTERFACE)
set_property(GLOBAL APPEND PROPERTY XR_MODULE_DEPS ${_DEPS_TARGET})
