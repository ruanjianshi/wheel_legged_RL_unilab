cmake_minimum_required(VERSION 3.0.2)
project(quadruped_deploy)

add_compile_options(-std=c++17 -O2 -Wno-deprecated-declarations)
add_definitions(-DPLATFORM_RK)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  std_msgs
  sim2real_msg
  livelybot_serial
  livelybot_msg
)

find_package(Eigen3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs std_msgs sim2real_msg livelybot_serial livelybot_msg
)

# RKNN runtime: 仓库自带的 librknnrt.so
set(RKNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../rknn)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${RKNN_DIR}
)

# livelybot_serial 安装时把头文件放到 install/include/livelybot_serial/{crc,hardware,...}
# 但里面的 .h 互相用 #include "crc16.h"/"motor.h"，需要把这些子目录显式加入搜索路径
foreach(_d ${catkin_INCLUDE_DIRS})
  if(EXISTS "${_d}/livelybot_serial/crc")
    include_directories("${_d}/livelybot_serial/crc")
    include_directories("${_d}/livelybot_serial/hardware")
    include_directories("${_d}/livelybot_serial")
  endif()
endforeach()

add_executable(quadruped_deploy_node src/quadruped_deploy_node.cpp)
target_link_libraries(quadruped_deploy_node
  ${catkin_LIBRARIES}
  ${RKNN_DIR}/librknnrt.so
  pthread
)
