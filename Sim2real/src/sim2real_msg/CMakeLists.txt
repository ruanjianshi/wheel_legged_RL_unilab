cmake_minimum_required(VERSION 3.0.2)
project(sim2real_msg)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
  actionlib_msgs
  sensor_msgs
)


add_message_files(
  FILES
  RL2PD.msg
  PD2RL.msg
  Orient.msg
  MtrState.msg
  RbtState.msg
  MtrTarget.msg
  RbtTarget.msg
  Joy.msg
  WayPoint.msg
)

add_service_files(
  FILES
  Common.srv
  ModelOutput.srv  # 指定服务文件路径
)

add_action_files(
  FILES
  lowlevel_controller.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
  actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
  geometry_msgs
  sensor_msgs
  actionlib_msgs
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
