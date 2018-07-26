cmake_minimum_required(VERSION 2.8.3)

project(orb_slam2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(PACKAGE_DEPENDENCIES
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  tf
  cmake_modules
)

find_package(catkin REQUIRED COMPONENTS ${PACKAGE_DEPENDENCIES})

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost COMPONENTS system)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/../../../
  ${PROJECT_SOURCE_DIR}/../../../include
  ${Pangolin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

set(LIBS 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
  ${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
  ${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM2.so
)

# Node for monocular camera
add_executable(mono
  src/ros_mono.cc
)

target_link_libraries(mono
 ${LIBS}
)

# Node for monocular camera (Augmented Reality Demo)
add_executable(monoAR
  src/AR/ros_mono_ar.cc
  src/AR/ViewerAR.h
  src/AR/ViewerAR.cc
)

target_link_libraries(monoAR
  ${LIBS}
)

# Node for stereo camera
add_executable(stereo
  src/ros_stereo.cc
)

target_link_libraries(stereo
${LIBS}
)

# Node for RGB-D camera
add_executable(rgbd
  src/ros_rgbd.cc
)

target_link_libraries(rgbd
  ${LIBS}
)

