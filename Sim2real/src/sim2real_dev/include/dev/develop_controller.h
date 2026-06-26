#ifndef DEVELOP_CONTROLLER_H_
#define DEVELOP_CONTROLLER_H_
#include <ros/ros.h>
#include <iostream>
#include <string>
#include<thread>
#include<sensor_msgs/JointState.h>

namespace hightorque{

class DevelopControllerNode
{
public:
    DevelopControllerNode();
    ~DevelopControllerNode();
    void Init();

    void exec_loop();

private:
    ros::Publisher command_pub_;
    std::string command_topic_str_;
    std::unique_ptr<std::thread> exec_loop_thread_ptr_;
    bool is_running_;
    ros::NodeHandle nh;
    int robot_dof_;
    
};
};
#endif
