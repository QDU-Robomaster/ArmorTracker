# ArmorTracker 运行时模块
target_include_directories(xr PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# libxr 在包含模块前会先写入默认 Print profile；这里移除默认值后再声明
# tracker 需要的 double 观测量和 uint64_t 图像时间戳输出能力。
foreach(_prop COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
  get_target_property(_defs xr ${_prop})
  if(_defs)
    list(FILTER _defs EXCLUDE REGEX "^LIBXR_PRINT_(FLOAT_ENABLE_DOUBLE|INTEGER_ENABLE_64BIT)=")
    set_property(TARGET xr PROPERTY ${_prop} ${_defs})
  endif()
endforeach()
target_compile_definitions(
  xr PUBLIC LIBXR_PRINT_FLOAT_ENABLE_DOUBLE=1 LIBXR_PRINT_INTEGER_ENABLE_64BIT=1
)

set(MODULE_ARMORTRACKER_SRC
  "${CMAKE_CURRENT_LIST_DIR}/SolveTrajectory.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/extended_kalman_filter.cpp"
)

target_sources(xr PRIVATE ${MODULE_ARMORTRACKER_SRC})

get_filename_component(_MODULE_DIRNAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(_DEPS_TARGET "${_MODULE_DIRNAME}_deps")
add_library(${_DEPS_TARGET} INTERFACE)
set_property(GLOBAL APPEND PROPERTY XR_MODULE_DEPS ${_DEPS_TARGET})
