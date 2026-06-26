#include "mixlevel_controller.h"

namespace hightorque
{
    MixlevelControllerNode::MixlevelControllerNode(std::shared_ptr<LowlevelControllerNode> low, std::shared_ptr<HighlevelControllerNode> high) : low_(low), high_(high)
    {
        ROS_INFO("MixlevelControllerNode %s()", __FUNCTION__);
    }

    bool MixlevelControllerNode::init()
    {
        ROS_INFO("MixlevelControllerNode %s()", __FUNCTION__);
        high_->setUseJoy(true);
        low_->initMotor();
        if (modelType_ == "pi_plus" || modelType_ == "hi")
        {
            low_->keepMotor("upperBody");
        }
        high_->configureMotor(low_->getRobotMotor());
        high_->initPinocchio();
        high_->initTrajectory();
        low_->initTrajectory();
        high_->initEngine();
        high_->initPolicy(policy_, pdInfo_, rlInfo_);
        high_->initObs();
        low_->initRos();
        high_->initRos();
        low_->setIsMix(true);
        high_->setIsMix(true);

        return true;
    }

    bool MixlevelControllerNode::start()
    {
        ROS_INFO("MixlevelControllerNode %s()", __FUNCTION__);
        return (high_->start() && low_->start());
    }

    MixlevelControllerNode::~MixlevelControllerNode()
    {
        high_.reset();
        low_.reset();
        ROS_INFO("%s()", __FUNCTION__);
    };

}

