# CMakeLists.txt for ArmorTracker

# Add module to include path
target_include_directories(xr PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Auto-include source files
file(GLOB MODULE_ARMORTRACKER_SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.cxx"
  "${CMAKE_CURRENT_LIST_DIR}/*.cc"
  "${CMAKE_CURRENT_LIST_DIR}/*.c"
)

# The active runtime path is Tracker -> Aimer. Keep legacy ballistic utilities
# available in-tree for reference, but do not link them into the tracker module.
list(FILTER MODULE_ARMORTRACKER_SRC EXCLUDE REGEX ".*/SolveTrajectory\\.cpp$")
list(FILTER MODULE_ARMORTRACKER_SRC EXCLUDE REGEX ".*/TableGenerator\\.cpp$")

target_sources(xr PRIVATE ${MODULE_ARMORTRACKER_SRC})

# INTERFACE deps shell
get_filename_component(_MODULE_DIRNAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(_DEPS_TARGET "${_MODULE_DIRNAME}_deps")
add_library(${_DEPS_TARGET} INTERFACE)

find_package(OpenCV REQUIRED)

target_link_libraries(${_DEPS_TARGET} INTERFACE
  ${OpenCV_LIBS}
)

# target_compile_definitions(${_DEPS_TARGET} INTERFACE
#   # FOO=1
# )
# target_compile_options(${_DEPS_TARGET} INTERFACE
#   # -Wall
# )

# Register to global
set_property(GLOBAL APPEND PROPERTY XR_MODULE_DEPS ${_DEPS_TARGET})
