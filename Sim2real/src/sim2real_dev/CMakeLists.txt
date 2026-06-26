cmake_minimum_required(VERSION 3.1.0)
project(sim2real_dev)


get_filename_component(WORKSPACE_DIR_FROM_DEVEL_PREFIX ${CATKIN_DEVEL_PREFIX} DIRECTORY)
get_filename_component(PARENT_DIR ${WORKSPACE_DIR_FROM_DEVEL_PREFIX} DIRECTORY)
get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
)

include_directories(
    include
    include/dev/
    ${catkin_INCLUDE_DIRS}
    ${GRANDPARENT_DIR}/devel/include
)

catkin_package()
set(DEV_CTRL_SRC
    include/dev/develop_controller.h
    src/main.cpp
    src/develop_controller.cpp

)

add_executable(${PROJECT_NAME}_node ${DEV_CTRL_SRC})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})
add_library(${PROJECT_NAME}_lib STATIC ${DEV_CTRL_SRC})
target_link_libraries(${PROJECT_NAME}_lib ${catkin_LIBRARIES})


