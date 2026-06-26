#ifndef REMOTE_CONTROLLER_INTERFACE_H_
#define REMOTE_CONTROLLER_INTERFACE_H_

#include<pinocchio/fwd.hpp>
#include<iostream>
#include<memory>
// #include<pd_controller.h>
// #include<rl_controller.h>
#include <sensor_msgs/JointState.h>
#include "hardware/robot.h"
#include "sim2real/base_controller_interface.h"

namespace hightorque{


class RemoteControllerInterface:public BaseControllerInterface
{
public:
    RemoteControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr);
    ~RemoteControllerInterface();

    int get_parameters() override;
    bool get_is_shutdowm() override;

    void JointCommandCallback(const sensor_msgs::JointState::ConstPtr &msg);
    void MotorCommandCallback(const sensor_msgs::JointState::ConstPtr &msg);
    void CopyJointStateFromMsg(const sensor_msgs::JointState::ConstPtr &msg);
private:
    std::string joint_command_topic_;
    std::string motor_command_topic_;
    ros::Subscriber joint_command_sub_;
    ros::Subscriber motor_command_sub_;
    // std::shared_ptr<hightorque::PD> pd_ptr_;
    sensor_msgs::JointState now_jointstate_;
};

};



#endif