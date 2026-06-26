# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
cmake_minimum_required(VERSION 3.0.2)
project(sim2real_master)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
find_package(
    catkin REQUIRED COMPONENTS
    roscpp
    roslib
    std_msgs
    serial
    livelybot_serial
    livelybot_logger
    #sim2real_logger
    sim2real_msg
    actionlib
    actionlib_msgs
    message_generation
)
get_filename_component(WORKSPACE_DIR_FROM_DEVEL_PREFIX ${CATKIN_DEVEL_PREFIX} DIRECTORY)
get_filename_component(PARENT_DIR ${WORKSPACE_DIR_FROM_DEVEL_PREFIX} DIRECTORY)
get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)
set(RKNN_PATH ${GRANDPARENT_DIR}/rknn)
set(PLATFORM_RK OFF)
set(PLATFORM_JETSON OFF)
set(PLATFORM_X86_64 OFF)
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    find_path(TENSORRT_HEADER NvInfer.h /usr/include/aarch64-linux-gpu)
    if(TENSORRT_HEADER)
      message( "Building for NVIDIA Jetson (aarch64)")
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
message("RKNN_PATH : ${RKNN_PATH}")
if(PLATFORM_X86_64)
    find_package(OpenVINO REQUIRED)
    include_directories(${OpenVINO_INCLUDE_DIRS})
    message(STATUS "OpenVINO_INCLUDE_DIRS: ${OpenVINO_INCLUDE_DIRS}")
endif()
if(PLATFORM_RK)
get_filename_component(WORKSPACE_DIR_FROM_DEVEL_PREFIX ${CATKIN_DEVEL_PREFIX} DIRECTORY)
get_filename_component(PARENT_DIR ${WORKSPACE_DIR_FROM_DEVEL_PREFIX} DIRECTORY)
get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)
set(RKNN_PATH ${GRANDPARENT_DIR}/rknn)
endif()
if(PLATFORM_JETSON)
set(TENSORRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu")
set(TENSORRT_LIBRARIES "/usr/lib/aarch64-linux-gnu/libnvinfer.so")
find_package(CUDA REQUIRED)
endif()

set(VERSION "2.1.2")
set(TYPE "motion")
add_definitions(-DVERSION="${VERSION}")
string(TIMESTAMP CURRENT_DATE "%Y%m%d")
if(PLATFORM_X86_64)
  set(FULL_VERSION_STRING "${TYPE}-x86-${VERSION}-${CURRENT_DATE}")
elseif(PLATFORM_RK)
  set(FULL_VERSION_STRING "${TYPE}-rknn-${VERSION}-${CURRENT_DATE}")
elseif(PLATFORM_JETSON)
  set(FULL_VERSION_STRING "${TYPE}-orin-${VERSION}-${CURRENT_DATE}")
endif()
set(JSON_CONTENT "[\"${FULL_VERSION_STRING}\"]")
set(VERSION_FILE_PATH "${GRANDPARENT_DIR}/version.json")
file(WRITE "${VERSION_FILE_PATH}" "${JSON_CONTENT}")
message("============================================")
message("Version info generated: ${VERSION_FILE_PATH}")
message("Content: ${JSON_CONTENT}")
message("============================================")

find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
include_directories(${Boost_INCLUDE_DIRS})
catkin_package(
    LIBRARIES ${PROJECT_NAME}_lib sim2real_lib 
    INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}  ${PINOCCHIO_INCLUDE_DIRS}
    #CATKIN_DEPENDS roscpp std_msgs  sim2real_logger livelybot_serial 
    CATKIN_DEPENDS roscpp std_msgs  
    DEPENDS pinocchio 
)
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${RKNN_PATH}
    ${OpenVINO_INCLUDE_DIRS}
    ${PINOCCHIO_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${GRANDPARENT_DIR}/devel/include
    ${GRANDPARENT_DIR}/src/sim2real/include
    ${GRANDPARENT_DIR}/install/include/livelybot_logger/
    ${GRANDPARENT_DIR}/src/sim2real_sdk/include
    ${GRANDPARENT_DIR}/install/include/livelybot_serial/  #for install
    ${GRANDPARENT_DIR}/install/include/livelybot_serial/crc  #for install
    ${GRANDPARENT_DIR}/install/include/livelybot_serial/hardware #for install
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
)
link_directories(
  /usr/lib/aarch64-linux-gnu
  /usr/local/cuda-11.4/targets/aarch64-linux/lib
)
foreach(a ${catkin_INCLUDE_DIRS})
    message(${a})
endforeach(a ${})
set(S2R_MASTER_SRCS
    include/sim2real/fsm.h
    include/sim2real/master_node.h
    include/sim2real/base_controller_interface.h
    include/sim2real/default_controller_interface.h
    include/sim2real/custom_controller_interface.h
    include/sim2real/remote_controller_interface.h
    include/sim2real/protection_controller_interface.h
    include/sim2real/zerocalibration_controller_interface.h
    include/sim2real/teach_controller_interface.h
    include/sim2real/develop_controller_interface.h
    include/sim2real/CubicSplineInterpolator.h
    #src/main.cpp
    src/master_node.cpp
    src/default_controller_interface.cpp
    src/custom_controller_interface.cpp
    src/remote_controller_interface.cpp
    src/protection_controller_interface.cpp
    src/zerocalibration_controller_interface.cpp
    src/teach_controller_interface_n.cpp
    src/develop_controller_interface.cpp
    src/CubicSplineInterpolator.cpp
    )
file(GLOB_RECURSE MOTOR_SRC "${GRANDPARENT_DIR}/src/sim2real_sdk/src/motor/*.cpp")
file(GLOB_RECURSE IMPL_SRC "${GRANDPARENT_DIR}/src/sim2real_sdk/src/impl/*.cpp")
file(GLOB_RECURSE POLICY_SRC "${GRANDPARENT_DIR}/src/sim2real_sdk/src/policy/*.cpp")
file(GLOB_RECURSE ENGINE_SRC "${GRANDPARENT_DIR}/src/sim2real_sdk/src/engine/*.cpp")
set(COMMON_SRC ${MOTOR_SRC} ${IMPL_SRC} ${POLICY_SRC} ${ENGINE_SRC})
message("Motor source : ${COMMON_SRC}") 
set(S2R_SRCS
    ${GRANDPARENT_DIR}/src/sim2real/src/common.cpp
    # 
    # 
    # 
    # 
    # 
    ${GRANDPARENT_DIR}/src/sim2real/src/kinematics.cpp
    ${GRANDPARENT_DIR}/src/sim2real/src/default_controller.cpp
    ${GRANDPARENT_DIR}/src/sim2real/src/highlevel_controller.cpp
    ${GRANDPARENT_DIR}/src/sim2real/src/sim2real.cpp
    ${GRANDPARENT_DIR}/src/sim2real/src/jacobian.cpp
    ${GRANDPARENT_DIR}/src/sim2real/src/lowlevel_controller.cpp
    ${GRANDPARENT_DIR}/src/sim2real/src/mixlevel_controller.cpp
    ${COMMON_SRC}
)
find_library(SERIALPORT_LIBRARY serialport)
if(NOT SERIALPORT_LIBRARY)
  message(FATAL_ERROR "libserialport not found")
endif()
if(PLATFORM_RK)
add_library(${PROJECT_NAME}_arm_lib ${S2R_MASTER_SRCS})
    target_link_libraries(${PROJECT_NAME}_arm_lib 
    # ${PROJECT_SOURCE_DIR}/lib/libsim2real_arm_lib.so 
    ${catkin_LIBRARIES} 
    ${SERIALPORT_LIBRARY}
    ${RKNN_PATH}/librknn_api.so
    ${RKNN_PATH}/librknnrt.so
    )
elseif(PLATFORM_JETSON)
add_library(${PROJECT_NAME}_arm_jetson_lib ${S2R_MASTER_SRCS})
    target_link_libraries(${PROJECT_NAME}_arm_jetson_lib 
    # ${PROJECT_SOURCE_DIR}/lib/libsim2real_arm_jetson_lib.so 
    ${catkin_LIBRARIES} 
    ${SERIALPORT_LIBRARY}
    ${TENSORRT_LIBRARIES}
    ${CUDA_LIBRARIES} )
elseif(PLATFORM_X86_64)
add_library(${PROJECT_NAME}_x86_lib ${S2R_MASTER_SRCS})
    target_link_libraries(${PROJECT_NAME}_x86_lib
    # ${PROJECT_SOURCE_DIR}/lib/libsim2real_x86_lib.so 
    ${catkin_LIBRARIES} 
    ${SERIALPORT_LIBRARY}
    ${OpenVINO_LIBRARIES}
    /usr/lib/libopenvino.so
    )
endif()
add_executable(${PROJECT_NAME}_node
src/main.cpp ${S2R_MASTER_SRCS} ${S2R_SRCS}
)
# if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
#     target_link_libraries(${PROJECT_NAME}_node
#     #${PROJECT_SOURCE_DIR}/lib/libsim2real_master_arm_lib.so
#     #${PROJECT_SOURCE_DIR}/lib/libsim2real_arm_lib.so 
#     ${catkin_LIBRARIES} 
#     ${SERIALPORT_LIBRARY}
#     # ${RKNN_PATH}/librknn_api.so
#     # ${RKNN_PATH}/librknnrt.so
#     ${TENSORRT_LIBRARIES}
#     ${CUDA_LIBRARIES}
#     ${pinocchio_LIBRARIES}
#     yaml-cpp 
#     )
# elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
# target_link_libraries(${PROJECT_NAME}_node 
#     #${PROJECT_SOURCE_DIR}/lib/libsim2real_master_x86_lib.so
#     #${PROJECT_SOURCE_DIR}/lib/libsim2real_x86_lib.so
#     ${catkin_LIBRARIES}
#     ${SERIALPORT_LIBRARY}
#     Boost::serialization
#     ${OpenVINO_LIBRARIES}
#     /usr/lib/libopenvino.so 
#     ${pinocchio_LIBRARIES}
#     yaml-cpp
#     #sim2real_logger 
#     )
# endif()
# if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
# add_library(${PROJECT_NAME}_arm_lib ${S2R_MASTER_SRCS} ${S2R_SRCS})
#     target_link_libraries(${PROJECT_NAME}_arm_lib 
#     #${PROJECT_SOURCE_DIR}/lib/libsim2real_arm_lib.so 
#     ${catkin_LIBRARIES} 
#     ${SERIALPORT_LIBRARY}
#     ${RKNN_PATH}/librknn_api.so
#     ${RKNN_PATH}/librknnrt.so
#     ${pinocchio_LIBRARIES}
#     yaml-cpp 
#     )
# elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
# #add_library(${PROJECT_NAME}_x86_lib ${S2R_MASTER_SRCS} ${S2R_SRCS})
# add_library(${PROJECT_NAME}_x86_lib ${S2R_MASTER_SRCS})
#     target_link_libraries(${PROJECT_NAME}_x86_lib
#     #${PROJECT_SOURCE_DIR}/lib/libsim2real_x86_lib.so 
#     ${catkin_LIBRARIES} 
#     ${SERIALPORT_LIBRARY}
#     ${OpenVINO_LIBRARIES}
#     /usr/lib/libopenvino.so
#     ${pinocchio_LIBRARIES}
#     yaml-cpp 
#     #sim2real_logger 
#     )
# endif()
if(PLATFORM_RK)
    target_link_libraries(${PROJECT_NAME}_node
    # ${PROJECT_SOURCE_DIR}/lib/libsim2real_master_arm_lib.so
    # ${PROJECT_SOURCE_DIR}/lib/libsim2real_arm_lib.so 
    ${catkin_LIBRARIES} 
    ${SERIALPORT_LIBRARY}
    ${RKNN_PATH}/librknn_api.so
    ${RKNN_PATH}/librknnrt.so
    ${pinocchio_LIBRARIES}
    Boost::serialization
    yaml-cpp 
    )
elseif(PLATFORM_JETSON)
    target_link_libraries(${PROJECT_NAME}_node
    # ${PROJECT_SOURCE_DIR}/lib/libsim2real_master_arm_jetson_lib.so
    # ${PROJECT_SOURCE_DIR}/lib/libsim2real_arm_jetson_lib.so 
    ${catkin_LIBRARIES} 
    ${SERIALPORT_LIBRARY}
    ${TENSORRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${pinocchio_LIBRARIES}
    Boost::serialization
    yaml-cpp 
    )
elseif(PLATFORM_X86_64)
    target_link_libraries(${PROJECT_NAME}_node 
    # ${PROJECT_SOURCE_DIR}/lib/libsim2real_master_x86_lib.so
    # ${PROJECT_SOURCE_DIR}/lib/libsim2real_x86_lib.so
    ${catkin_LIBRARIES}
    ${SERIALPORT_LIBRARY}
    Boost::serialization
    ${OpenVINO_LIBRARIES}
    /usr/lib/libopenvino.so 
    ${pinocchio_LIBRARIES}
    yaml-cpp 
    )
endif()
add_dependencies(${PROJECT_NAME}_node livelybot_logger_generate_messages_cpp)
message("sim2real_logger_LIBRARIES: ${livelybot_logger_LIBRARIES}")
message("sim2real_logger_LIBRARY: livelybot_logger")
foreach(a ${catkin_LIBRARIES})
    message(${a})
endforeach(a ${catkin})
# # Verify include directories and link libraries for the executable
# get_target_property(EXEC_INCLUDE_DIRS ${PROJECT_NAME}_node INCLUDE_DIRECTORIES)
# message("Include directories for ${PROJECT_NAME}_node: ${EXEC_INCLUDE_DIRS}")
# get_target_property(EXEC_LINK_LIBS ${PROJECT_NAME}_node LINK_LIBRARIES)
# message("Link libraries for ${PROJECT_NAME}_node: ${EXEC_LINK_LIBS}")
install(TARGETS ${PROJECT_NAME}_node
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
 )
install(FILES joy.yaml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES joy_footstep.yaml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY way_point/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/way_point
)
install(TARGETS ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
install(FILES
${RKNN_PATH}/librknn_api.so
${RKNN_PATH}/librknnrt.so
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
#install(DIRECTORY config/
#    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
#)
