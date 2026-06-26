#ifndef DEFAULT_CONTROLLER_INTERFACE_H_
#define DEFAULT_CONTROLLER_INTERFACE_H_

// #include <pd_controller.h>
// #include <rl_controller.h>
// #include "hardware/robot.h"
#include "sim2real/base_controller_interface.h"
#include "default_controller.h"

namespace hightorque
{

    class DefaultControllerInterface : public BaseControllerInterface
    {
        public:
            DefaultControllerInterface(std::shared_ptr<livelybot_serial::robot> rbPtr);
            ~DefaultControllerInterface();

            int get_parameters() override;
            bool get_is_shutdowm() override;

        private:
            std::shared_ptr<Sim2Real> node;
    };

};

#endif
