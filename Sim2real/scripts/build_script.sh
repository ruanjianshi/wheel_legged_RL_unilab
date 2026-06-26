#!/bin/bash
source /opt/ros/noetic/setup.bash
catkin build livelybot_logger -DCMAKE_CXX_FLAGS="-DRELEASE"
catkin build livelybot_bringup  -DCMAKE_CXX_FLAGS="-DRELEASE"
catkin config --install 
catkin clean -b -y
#catkin build --catkin-make-args tests
catkin build -DCMAKE_CXX_FLAGS="-DRELEASE" --catkin-make-args tests
lcov --directory build/sim2real --zerocounters
catkin run_tests sim2real --no-deps
lcov --directory build/sim2real --capture --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/test/*' --output-file coverage_filtered.info
genhtml coverage_filtered.info --output-directory coverage_report
