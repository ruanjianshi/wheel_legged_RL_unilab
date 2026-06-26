#include "sim2real/default_controller_interface.h"

namespace hightorque
{

    DefaultControllerInterface::DefaultControllerInterface(std::shared_ptr<livelybot_serial::robot> rbPtr) : BaseControllerInterface(rbPtr)
    {
        auto nh = std::make_shared<ros::NodeHandle>("~");
        node = std::make_shared<DefaultControllerNode>(rbPtr, nh);
        node->start();
    }

    int DefaultControllerInterface::get_parameters()
    {
        return node->get_parameters();
    }

    bool DefaultControllerInterface::get_is_shutdowm()
    {
        return node->get_is_shutdowm();
    }

    DefaultControllerInterface::~DefaultControllerInterface()
    {
        std::cout << "~DefaultControllerInterface()" << std::endl;
    }
}
