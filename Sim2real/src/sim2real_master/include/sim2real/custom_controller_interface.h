#ifndef CUSTOM_CONTROLLER_INTERFACE_H_
#define CUSTOM_CONTROLLER_INTERFACE_H_

#include "sim2real/base_controller_interface.h"
#include "default_controller.h"

namespace hightorque
{

    class CustomControllerInterface : public BaseControllerInterface
    {
        public:
            CustomControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr);

            ~CustomControllerInterface();

            int get_parameters() override;
            bool get_is_shutdowm() override;

        private:
            std::shared_ptr<Sim2Real> node;
    };

};

#endif
