# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
cmake_minimum_required(VERSION 3.0.2)
project(sim2real)
set(CMAKE_CXX_STANDARD 17)
# 单测相关 +
# 启用代码覆盖率
option(COVERAGE "Enable coverage reporting" ON)
if(COVERAGE)
    message(STATUS "Coverage enabled")
    add_compile_options(--coverage -O0 -g)
    link_libraries(gcov)
endif()
# 单测相关 -
# ----------------------------------
# 目录结构
# ----------------------------------
get_filename_component(PRIVATE_DIR ${CATKIN_DEVEL_PREFIX} DIRECTORY)
get_filename_component(DEVEL_DIR ${PRIVATE_DIR} DIRECTORY)
get_filename_component(WORK_SPACE_DIR ${DEVEL_DIR} DIRECTORY)
set(PythonInterp_DIR /usr/bin/python3)
set(RKNN_PATH ${WORK_SPACE_DIR}/rknn)
add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# ----------------------------------
# 平台检测
# ----------------------------------
set(PLATFORM_RK OFF)
set(PLATFORM_JETSON OFF)
set(PLATFORM_X86_64 OFF)
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    find_path(TENSORRT_HEADER NvInfer.h /usr/include/aarch64-linux-gpu)
    if(TENSORRT_HEADER)
        message( "Building for NVIDIA Jetson (aarch64)" TENSORRT_HEADER)
        add_definitions(-DPLATFORM_JETSON)
        set(PLATFORM_JETSON ON)
    else()
        message( "Building for rk series chip (aarch64)")
        add_definitions(-DPLATFORM_RK)
        set(PLATFORM_RK ON)
    endif()
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    message( "Building for x86_64")
    add_definitions(-DPLATFORM_X86_64)
    set(PLATFORM_X86_64 ON)
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()
if(PLATFORM_X86_64)
    find_package(OpenVINO REQUIRED)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
    message("OpenVINO Include Dirs: ${OpenVINO_INCLUDE_DIRS}")
    message("OpenVINO Library Dirs: ${OpenVINO_LIBRARIES}")
endif()
if(PLATFORM_JETSON)
    set(TENSORRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu")
    set(TENSORRT_LIBRARIES "/usr/lib/aarch64-linux-gnu/libnvinfer.so")
    find_package(CUDA REQUIRED)
endif()
# ----------------------------------
# 依赖查找
# ----------------------------------
if(PLATFORM_RK)
    set(pinocchio_DIR "/opt/ros/noetic/share/cmake/pinocchio/")
elseif(PLATFORM_X86_64)
    set(pinocchio_DIR "/opt/openrobots/lib/cmake/pinocchio/")
    set(casadi_DIR "/opt/openrobots/lib/cmake/casadi/")
    set(coal_DIR "/opt/openrobots/lib/cmake/coal/")
    set(eigenpy_DIR "/opt/openrobots/lib/cmake/eigenpy/")
    set(hpp-fcl_DIR "/opt/openrobots/lib/cmake/hpp-fcl/")
elseif(PLATFORM_JETSON)
    set(pinocchio_DIR "/opt/ros/noetic/share/cmake/pinocchio/")
endif()
find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    std_msgs
    serial
    tf2_ros
    tf2
    # dynamic_reconfigure
    livelybot_serial
    sim2real_msg
    actionlib
    actionlib_msgs
    livelybot_logger
    pinocchio
    Eigen3
)
find_library(SERIALPORT_LIBRARY serialport)
if(NOT SERIALPORT_LIBRARY)
    message(FATAL_ERROR "libserialport not found")
endif()
catkin_package(
    LIBRARIES ${PROJECT_NAME}_lib  
    INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}  ${PINOCCHIO_INCLUDE_DIRS}
    DEPENDS rt pthread pinocchio 
    CATKIN_DEPENDS roscpp std_msgs serial livelybot_serial sim2real_msg livelybot_logger sim2real_msg roslib
)
# ----------------------------------
# 源文件
# ----------------------------------
set(rt_LIBRARIES rt)
set(pthread_LIBRARIES pthread)
file(GLOB_RECURSE MOTOR_SRC "${WORK_SPACE_DIR}/src/sim2real_sdk/src/motor/*.cpp")
file(GLOB_RECURSE IMPL_SRC "${WORK_SPACE_DIR}/src/sim2real_sdk/src/impl/*.cpp")
file(GLOB_RECURSE POLICY_SRC "${WORK_SPACE_DIR}/src/sim2real_sdk/src/policy/*.cpp")
file(GLOB_RECURSE ENGINE_SRC "${WORK_SPACE_DIR}/src/sim2real_sdk/src/engine/*.cpp")
set(COMMON_SRC ${MOTOR_SRC} ${IMPL_SRC} ${POLICY_SRC} ${ENGINE_SRC})
message("PINOCCHIO include directories: ${PINOCCHIO_INCLUDE_DIRS}")
message("PINOCCHIO libraries: ${pinocchio_LIBRARIES}")
message("OpenVINO Include Dirs: ${OpenVINO_INCLUDE_DIRS}")
message("OpenVINO Library Dirs: ${OpenVINO_LIBRARIES}")
message("livelybot_serial_INCLUDE_DIRS : ${livelybot_serial_INCLUDE_DIRS}")
message("WORK_SPACE_DIR" :${WORK_SPACE_DIR})
set(RL_CTRL_SRCS
  src/common.cpp
)
set(PD_CTRL_SRCS
  src/jacobian.cpp
  src/kinematics.cpp 
  src/common.cpp
  ${WORK_SPACE_DIR}/src/sim2real_master/src/CubicSplineInterpolator.cpp
)
# imu贴反补救
add_executable(imu_rotated src/imu_rotated.cpp)
target_link_libraries(imu_rotated
  ${catkin_LIBRARIES}
)
# ----------------------------------
# 包含目录
# ----------------------------------
include_directories(
    ${PINOCCHIO_INCLUDE_DIRS}
    include
    ${catkin_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenVINO_INCLUDE_DIRS}
    ${RKNN_PATH}
    ${livelybot_serial_INCLUDE_DIRS}
    /opt/ros/noetic/include
    ${WORK_SPACE_DIR}/devel/include 
    ${WORK_SPACE_DIR}/src/sim2real_master/include
    ${WORK_SPACE_DIR}/src/sim2real_sdk/include
    ${WORK_SPACE_DIR}/src/livelybot_logger/include
    ${WORK_SPACE_DIR}/install/include/livelybot_serial/  #for install
    ${WORK_SPACE_DIR}/install/include/livelybot_serial/crc  #for install
    ${WORK_SPACE_DIR}/install/include/livelybot_serial/hardware #for install
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
)
add_compile_options(-Wno-ignored-attributes)
link_directories(${PROJECT_SOURCE_DIR}/lib
  /usr/lib/aarch64-linux-gnu
  /usr/local/cuda-11.4/targets/aarch64-linux/lib
)
if(PLATFORM_X86_64)
    # add_executable(rl_pd_controller
    #     src/main.cpp
    #     ${RL_CTRL_SRCS}
    #     ${PD_CTRL_SRCS}
    #     ${COMMON_SRC}
    # )
    # target_link_libraries(
    #     rl_pd_controller
    #     # libsim2real_x86_lib.so
    #     ${catkin_LIBRARIES}
    #     ${SERIALPORT_LIBRARY}
    #     yaml-cpp
    #     ${pinocchio_LIBRARIES}
    #     ${OpenVINO_LIBRARIES}
    #     /usr/lib/libopenvino.so
    # )
    add_library(${PROJECT_NAME}_x86_lib
        ${RL_CTRL_SRCS}
        ${PD_CTRL_SRCS}
        ${COMMON_SRC}
    )
    target_link_libraries(${PROJECT_NAME}_x86_lib PUBLIC
        ${catkin_LIBRARIES}
        ${SERIALPORT_LIBRARY}
        yaml-cpp
        ${pinocchio_LIBRARIES}
        ${livelybot_serial}
        ${OpenVINO_LIBRARIES}
    )
    target_include_directories( ${PROJECT_NAME}_x86_lib PUBLIC
        include
        ${catkin_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${PINOCCHIO_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenVINO_INCLUDE_DIRS}
    )
endif()
if(PLATFORM_RK)
    # add_executable(rl_pd_controller
    #     src/main.cpp
    #     ${RL_CTRL_SRCS}
    #     ${PD_CTRL_SRCS}
    #     ${COMMON_SRC}
    # )
    # target_link_libraries(rl_pd_controller
    #     # libsim2real_arm_lib.so
    #     ${catkin_LIBRARIES}
    #     ${SERIALPORT_LIBRARY}
    #     yaml-cpp
    #     ${pinocchio_LIBRARIES}
    #     ${RKNN_PATH}/librknn_api.so
    #     ${RKNN_PATH}/librknnrt.so
    # )
    add_library(${PROJECT_NAME}_arm_lib
        ${RL_CTRL_SRCS}
        ${PD_CTRL_SRCS}
        ${COMMON_SRC}
    )
    target_link_libraries(${PROJECT_NAME}_arm_lib PUBLIC
        ${catkin_LIBRARIES}
        ${SERIALPORT_LIBRARY}
        yaml-cpp
        ${pinocchio_LIBRARIES}
        ${livelybot_serial}
    )
    target_include_directories(${PROJECT_NAME}_arm_lib PUBLIC
        include
        ${catkin_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${PINOCCHIO_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
    )
endif()
if(PLATFORM_JETSON)
    # add_executable(rl_pd_controller  src/main.cpp
    #     ${RL_CTRL_SRCS}
    #     ${PD_CTRL_SRCS}
    #     ${COMMON_SRC}
    # )
    # target_link_libraries(rl_pd_controller
    #     # libsim2real_arm_jetson_lib.so
    #     ${catkin_LIBRARIES}
    #     ${SERIALPORT_LIBRARY}
    #     yaml-cpp
    #     ${pinocchio_LIBRARIES}
    #     ${TENSORRT_LIBRARIES}
    #     ${CUDA_LIBRARIES}
    # )
    #
    add_library(${PROJECT_NAME}_arm_jetson_lib
        ${RL_CTRL_SRCS}
        ${PD_CTRL_SRCS}
        ${COMMON_SRC}
    )
    target_link_libraries(${PROJECT_NAME}_arm_jetson_lib PUBLIC
        ${catkin_LIBRARIES}
        ${SERIALPORT_LIBRARY}
        yaml-cpp
        ${pinocchio_LIBRARIES}
        ${livelybot_serial}
        ${TENSORRT_LIBRARIES}
        ${CUDA_LIBRARIES}
    )
    target_include_directories(${PROJECT_NAME}_arm_jetson_lib PUBLIC
        include
        ${catkin_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${PINOCCHIO_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIRS}
    )
endif()
# install(TARGETS rl_pd_controller
#    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
install(DIRECTORY launch config policy action_config resources robot_param
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
if(PLATFORM_X86_64)
    install(TARGETS ${PROJECT_NAME}_x86_lib
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    )
endif()
if(PLATFORM_RK)
    install(TARGETS ${PROJECT_NAME}_arm_lib
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    )
endif()
if(PLATFORM_JETSON)
    install(TARGETS ${PROJECT_NAME}_arm_jetson_lib
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    )
endif()
# if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
# #install(TARGETS ${PROJECT_NAME}_x86_lib
# install(TARGETS ${PROJECT_NAME}_arm_lib
# #   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# #   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )
# elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
# install(TARGETS ${PROJECT_NAME}_x86_lib
# #install(TARGETS ${PROJECT_NAME}_arm_lib
# #   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# #   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )
# endif()
# get_target_property(LINK_LIBS ${PROJECT_NAME}_lib LINK_LIBRARIES)
# message("Link libraries for ${PROJECT_NAME}_lib: ")
# foreach(a ${LINK_LIBS})
# message(${a})
# endforeach(a ${${LINK_LIBS}})
if(CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    find_package(GTest REQUIRED)
    # 添加测试文件
    #catkin_add_gtest(test_rl_exec_loop
    #  test/test_rl_exec_loop.cpp
    #  ${RL_CTRL_SRCS}
    #  ${PD_CTRL_SRCS}
    #)
    #target_include_directories(test_rl_exec_loop PRIVATE
    #  include
    #  ${catkin_INCLUDE_DIRS}
    #  ${EIGEN3_INCLUDE_DIR}
    #  ${PINOCCHIO_INCLUDE_DIRS}
    #)
    #target_link_libraries(test_rl_exec_loop
    #  ${catkin_LIBRARIES}
    #  ${SERIALPORT_LIBRARY}
    #  yaml-cpp
    #  ${pinocchio_LIBRARIES}
    #  gcov  # 用于覆盖率
    #)
    # 添加 GTest 测试目标
    catkin_add_gtest(test_jacobian
        test/test_jacobian.cpp
        #src/jacobian.cpp
        ${PD_CTRL_SRCS}
    )
    target_include_directories(test_jacobian PRIVATE
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(test_jacobian
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        yaml-cpp
        gcov # 用于覆盖率
    )
endif()
