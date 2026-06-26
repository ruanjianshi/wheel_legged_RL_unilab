#include "sim2real/custom_controller_interface.h"

namespace hightorque
{

    CustomControllerInterface::CustomControllerInterface(std::shared_ptr<livelybot_serial::robot> rbPtr) : BaseControllerInterface(rbPtr)
    {
        auto nh = std::make_shared<ros::NodeHandle>("~");
        std::string rlConfigFile;
        nh->param<std::string>("custom_rl_config_file", rlConfigFile, "~");
        ROS_INFO("now config come form %s", rlConfigFile.c_str());
        node = std::make_shared<DefaultControllerNode>(rbPtr, nh, rlConfigFile);
        node->start();
    }

    int CustomControllerInterface::get_parameters()
    {
        return node->get_parameters();
    }

    bool CustomControllerInterface::get_is_shutdowm()
    {
        return node->get_is_shutdowm();
    }

    CustomControllerInterface::~CustomControllerInterface()
    {
        std::cout << "~CustomControllerInterface()" << std::endl;
    }
}
