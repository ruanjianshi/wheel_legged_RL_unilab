#ifndef ZEROCALIBRATION_CONTROLLER_INTERFACE_H_
#define ZEROCALIBRATION_CONTROLLER_INTERFACE_H_

#include<pinocchio/fwd.hpp>
#include <iostream>
#include <memory>
// #include <pd_controller.h>//应该是hardware/robot.h包含了boost头文件 pd_controller.h包含了Pinocchio 头文件 所以要放在robot.h的前面
#include "hardware/robot.h"
#include "sim2real/base_controller_interface.h"

namespace hightorque
{
    class ZeroCalibrationControllerInterface : public BaseControllerInterface
    {
    public:
        ZeroCalibrationControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr);
        ~ZeroCalibrationControllerInterface();

        void reset_zero();
        bool motor_calibrated(motor *m);
        
        int get_parameters() override;
        bool get_is_shutdowm() override;

    private:
        std::shared_ptr<livelybot_serial::robot> rb_ptr_;
        std::shared_ptr<std::thread> thread_exec_ptr_;
        int status;
        std::atomic<bool> quit;
    };

};

#endif