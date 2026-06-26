#ifndef PROTECTION_CONTROLLER_INTERFACE_H_
#define PROTECTION_CONTROLLER_INTERFACE_H_

#include<pinocchio/fwd.hpp>
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <ros/ros.h>
#include <hardware/robot.h>
// #include <pd_controller.h>
#include "base_controller_interface.h"

namespace hightorque
{


    class ProtectionControllerInterface : public BaseControllerInterface
    {
    public:
        // ProtectionControllerInterface();
        ProtectionControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr);
        ~ProtectionControllerInterface();

        void init_data();
        void motor_pd();
        void send_motor_zero();
        void set_motor_commomd_type();
        void exec_loop();

        int get_parameters() override;
        bool get_is_shutdowm() override;

        ros::Subscriber joy_sub_;

        // MotorCommandType motor_commomd_type_;

    private:
        std::shared_ptr<livelybot_serial::robot> rb_ptr_;
        std::shared_ptr<std::thread> thread_exec_ptr_;
        std::atomic<bool> quit;
    };

};

#endif