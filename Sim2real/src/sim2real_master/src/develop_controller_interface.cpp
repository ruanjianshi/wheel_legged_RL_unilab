#include "sim2real/develop_controller_interface.h"
#include <memory>

namespace hightorque
{

    DevelopControllerInterface::DevelopControllerInterface(std::shared_ptr<livelybot_serial::robot> rbPtr) : BaseControllerInterface(rbPtr)
    {
        auto nh = std::make_shared<ros::NodeHandle>("~");
        std::string level, rlConfigFile;
        nh->param<std::string>("devel_level", level, "high");
        nh->param<std::string>("custom_rl_config_file", rlConfigFile, "~");
        ROS_INFO("now config come form %s", rlConfigFile.c_str());
        ROS_INFO("devel_level : %s", level.c_str());
        if (level == "high")
        {
            levelNode_ = std::make_shared<HighlevelControllerNode>(rbPtr, nh, rlConfigFile);
        }
        else if (level == "low")
        {
            levelNode_ = std::make_shared<LowlevelControllerNode>(rbPtr, nh);
        }
        else if (level == "mix")
        {
            levelNode_ = std::make_shared<MixlevelControllerNode>(std::make_shared<LowlevelControllerNode>(rbPtr, nh), std::make_shared<HighlevelControllerNode>(rbPtr, nh, rlConfigFile));
        }
        levelNode_->init();
        levelNode_->start();
    }

    int DevelopControllerInterface::get_parameters()
    {
        return levelNode_->get_parameters();
    }

    bool DevelopControllerInterface::get_is_shutdowm()
    {
        return levelNode_->get_is_shutdowm();
    }

    DevelopControllerInterface::~DevelopControllerInterface()
    {
        std::cout << "~DevelopControllerInterface()" << std::endl;
    }
}
