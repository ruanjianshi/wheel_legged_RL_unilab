#ifndef DEVELOP_CONTROLLER_INTERFACE_H_
#define DEVELOP_CONTROLLER_INTERFACE_H_

#include<pinocchio/fwd.hpp>
#include <iostream>
#include <memory>
//#include <pd_controller.h>
//#include <rl_controller.h>
#include "hardware/robot.h"
#include "sim2real/base_controller_interface.h"
#include "highlevel_controller.h"
#include "lowlevel_controller.h"
#include "mixlevel_controller.h"
#include "sim2real_msg/Common.h"
#include "std_msgs/Int8.h"

namespace hightorque
{
    class DevelopControllerInterface : public BaseControllerInterface
    {
        public:
            DevelopControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr);
            ~DevelopControllerInterface();

            int get_parameters() override;
            bool get_is_shutdowm() override;
            bool changeStateHandle(sim2real_msg::Common::Request& req,
                                   sim2real_msg::Common::Response& res);
            void actionCallback(std_msgs::Int8);
        private:
            ros::ServiceServer changeStateService_;
            ros::Subscriber actionTypeSub_;
            std::shared_ptr<LevelControllerNode> levelNode_;
    };

};

#endif
