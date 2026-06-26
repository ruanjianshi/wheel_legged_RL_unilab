#include "dev/develop_controller.h"

namespace hightorque
{
    DevelopControllerNode::DevelopControllerNode() : nh()
    {
        Init();
    }

    void DevelopControllerNode::Init()
    {

        nh.param<int>("robot_dof", this->robot_dof_, 12);

        nh.param<std::string>("command_topic", this->command_topic_str_, "joint_goals");
        command_pub_ = nh.advertise<sensor_msgs::JointState>(command_topic_str_, 10);
        is_running_ = true;
        exec_loop_thread_ptr_.reset(new std::thread(&DevelopControllerNode::exec_loop, this));
    }
    // 1000Hz
    void DevelopControllerNode::exec_loop()
    {
        ros::Rate rate(1000);
        while (is_running_ && ros::ok())
        {
            sensor_msgs::JointState msg;
            msg.position.resize(robot_dof_);
            msg.velocity.resize(robot_dof_);
            msg.effort.resize(robot_dof_);
            for (int i = 0; i < robot_dof_; ++i)
            {
                msg.position.at(i) = 0;
                msg.velocity.at(i) = 0;
                msg.effort.at(i) = 0;
            }

            command_pub_.publish(msg);
            rate.sleep();
        }
    }

    DevelopControllerNode::~DevelopControllerNode()
    {
        is_running_ = false;
        if (exec_loop_thread_ptr_->joinable())
        {
            exec_loop_thread_ptr_->join();
        }
    }

}
