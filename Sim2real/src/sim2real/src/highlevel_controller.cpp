#include "highlevel_controller.h"

namespace hightorque
{
    HighlevelControllerNode::HighlevelControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr, std::shared_ptr<ros::NodeHandle> nh) : LevelControllerNode(rbPtr, nh)
    {
        ROS_INFO("HighlevelControllerNode %s()", __FUNCTION__);
    }

    HighlevelControllerNode::HighlevelControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr, std::shared_ptr<ros::NodeHandle> nh, const std::string& rlConfigFile) : LevelControllerNode(rbPtr, nh, rlConfigFile)
    {
        ROS_INFO("HighlevelControllerNode %s()", __FUNCTION__);
    }

    bool HighlevelControllerNode::initRos()
    {
        ROS_INFO("HighlevelControllerNode %s()", __FUNCTION__);
        Sim2Real::initRos();
        std::string serviceName;
        serviceName = "change_state";
        if (!useJoy_)
        {
            joySub_.shutdown();
        }
        changeStateService_ = nh_->advertiseService(serviceName, &HighlevelControllerNode::changeStateHandle, this);
        actionCallbackSub_ = nh_->subscribe<std_msgs::Int8>("action", 10, &HighlevelControllerNode::actionCallback, this);
        return true;
    }

    void HighlevelControllerNode::configureMotor(std::shared_ptr<RobotMotorGroup> robotMotor)
    {
        robotMotor_ = robotMotor;
    }

    bool HighlevelControllerNode::changeStateHandle(sim2real_msg::Common::Request& req, sim2real_msg::Common::Response& res)
    {
        std::unique_lock<std::shared_timed_mutex> lk(stateMutex_);
        if (req.str == "standby")
        {
            state_ = sim2Real::State::STANDING;
            trajectory_->setSingle(robotMotor_->getMotorGroupStateByName("all"), "zero", 5, "all", utils::TransitionType::SmoothStep);
        }
        else if (req.str == "sitdown")
        {
            state_ = sim2Real::State::SITTING;
            trajectory_->setSingle(robotMotor_->getMotorGroupStateByName("all"), "sitdown", 5, "all", utils::TransitionType::SmoothStep);
        }
        else if (req.str == "switch")
        {
            if (state_ == sim2Real::State::RUNNING)
            {
                state_ = sim2Real::State::STANDBY;
            }
            else if (state_ == sim2Real::State::STANDBY)
            {
                state_ = sim2Real::State::RUNNING;
            }
            else
            {
                res.error_message = "The current state " + sim2Real::toString(state_) + " is not standby or running." + req.str;
            }
        }
        else if (req.str == "get")
        {
            res.message = sim2Real::toString(state_);
        }
        else
        {
            res.error_message = "not such cmd" + req.str;
        }
        res.result = true;
        return true;
    }

    void HighlevelControllerNode::actionCallback(const std_msgs::Int8::ConstPtr& msg)
    {
        std::unique_lock<std::shared_timed_mutex> lk(stateMutex_);
        if (state_ != sim2Real::State::STANDBY)
        {
            return;
        }
        switch (msg->data)
        {
            case (0):
            case (1):
            case (2):
            case (3):
                {
                    std::string name = std::to_string(msg->data);
                    std::string group = trajectory_->getMotorPosesVecGroup(name);
                    setKpKd();
                    trajectory_->setMulti(robotMotor_->getMotorGroupStateByName(group), name, 1, 0.0, group, utils::TransitionType::Linear);
                }
                break;
            case (10):
            case (11):
            case (12):
            case (13):
                {
                    std::string name = std::to_string(msg->data - 10);
                    std::string group = trajectory_->getMotorPosesVecGroup(name);
                    setKpKd();
                    trajectory_->setMulti(robotMotor_->getMotorGroupStateByName(group), name, 1, 0.0, group, utils::TransitionType::EaseIn);
                }
                break;
            case (20):
            case (21):
            case (22):
            case (23):
                {
                    std::string name = std::to_string(msg->data - 20);
                    std::string group = trajectory_->getMotorPosesVecGroup(name);
                    setKpKd();
                    trajectory_->setMulti(robotMotor_->getMotorGroupStateByName(group), name, 1, 0.0, group, utils::TransitionType::EaseOut);
                }
                break;
            case (30):
            case (31):
            case (32):
            case (33):
                {
                    std::string name = std::to_string(msg->data - 30);
                    std::string group = trajectory_->getMotorPosesVecGroup(name);
                    setKpKd();
                    trajectory_->setMulti(robotMotor_->getMotorGroupStateByName(group), name, 1, 0.0, group, utils::TransitionType::SmoothStep);
                }
                break;
            case (40):
            case (41):
            case (42):
            case (43):
                {
                    std::string name = std::to_string(msg->data - 40);
                    std::string group = trajectory_->getMotorPosesVecGroup(name);
                    setKpKd();
                    trajectory_->setMulti(robotMotor_->getMotorGroupStateByName(group), name, 0.05, 0.0, group, utils::TransitionType::CubicSpline);
                }
                break;
            default:
                return;
        }
        state_ = sim2Real::State::TRAJ_MULTI;
    }

    HighlevelControllerNode::~HighlevelControllerNode()
    {
        ROS_INFO("high %s()", __FUNCTION__);
        stop();
    }
}
