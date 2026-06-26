#include "default_controller.h"

namespace hightorque
{
    DefaultControllerNode::DefaultControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr, std::shared_ptr<ros::NodeHandle> nh) : Sim2Real(rbPtr, nh)
    {
        ROS_INFO("DefaultControllerNode %s()", __FUNCTION__);
        nh->param<double>("kick_duration", kickDuration_, 0.2);
        nh->param<bool>("use_fell_down_policy", useFellDownPolicy_, false);
        nh->param<bool>("use_fell_down_action", useFellDownAction_, false);
        nh->param<std::string>("policy_change_default_policy", policyChangeDefautePolicy, "");

        ROS_INFO("DefaultControllerNode kickDuration: %f, useFelldownPolicy: %d, useFellDownAction: %d", kickDuration_, useFellDownPolicy_, useFellDownAction_);
        audioThread_ = std::thread(&DefaultControllerNode::playAudio, this); // 启动音频播放线程
        init();
    }

    DefaultControllerNode::DefaultControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr, std::shared_ptr<ros::NodeHandle> nh, const std::string& rlConfigFile) : Sim2Real(rbPtr, nh, rlConfigFile)
    {
        ROS_INFO("DefaultControllerNode new config %s()", __FUNCTION__);
        nh->param<double>("kick_duration", kickDuration_, 0.2);
        nh->param<bool>("use_fell_down_policy", useFellDownPolicy_, false);
        nh->param<bool>("use_fell_down_action", useFellDownAction_, false);
        if (useFellDownPolicy_)
        {
            useFellDownAction_ = false;
            ROS_WARN("DefaultControllerNode useFelldownPolicy is true, useFellDownAction will be set to false");
        }

        ROS_INFO("DefaultControllerNode kickDuration: %f, useFelldownPolicy: %d, useFellDownAction: %d", kickDuration_, useFellDownPolicy_, useFellDownAction_);
        init();
    }

    void DefaultControllerNode::playAudio()
    {
        while (audioQuit_ == false)
        {
            if (isPlaying_.load())
            {
                ROS_INFO("======= Playing audio file: %s", audioFile_.c_str());
                std::string command = "aplay " + audioFile_;
                system(command.c_str());
                isPlaying_.store(false); // 播放完成后重置标志
                ROS_INFO("======= Audio playback completed.");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void DefaultControllerNode::joyCallback(const sim2real_msg::Joy::ConstPtr& msg)
    {
        static bool lastSwitched = false, lastUped = false;
        static std::string _lastLoopMulti = "";
        static std::string _lastPolicyName = "";
        std::unique_lock<std::shared_timed_mutex> stateLk(stateMutex_);
        joy_.set(msg);

        // 如果有输出 则使用输出q
        auto getMotorState = [&]() {
            Motor_State state(pdInfo_.dofs);
            std::unique_lock<std::shared_timed_mutex> outputLk(outputMutex_);
            if (!motorOutput_.isZero() && motorOutput_.target_q.size() == pdInfo_.dofs)
            {
                state.q = motorOutput_.target_q;
            }
            else
            {
                state = robotMotor_->getMotorGroupStateByName("all");
            }
            return state;
        };

        if (msg->dpad_horizontal == -1)
        {
            if (!isPlaying_.load())
            {
                isPlaying_.store(true);                // 设置为正在播放
                audioFile_ = "/home/hightorque/1.wav"; // 设置音频文件路径
                ROS_INFO("======= Button -> pressed, playing audio.");
            }
            else
            {
                ROS_WARN("======= Audio is already playing, please wait.");
            }
        }

        // LT+RT
        if (msg->lt < -0.5 && msg->rt < -0.5)
        {
            joy_.reset();
            // 蹲下 lb键
            if (msg->rb == 1)
            {
                state_ = sim2Real::State::SITTING;
                _lastLoopMulti = "";
                trajectory_->setSingle(getMotorState(), "sitdown", 3, "all", utils::TransitionType::SmoothStep);
                stateLk.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                return;
            }
            // 起立 start键
            if (msg->start == 1 && state_ != sim2Real::State::RUNNING)
            {
                state_ = sim2Real::State::STANDING;
                _lastLoopMulti = "";
                trajectory_->setSingle(getMotorState(), "zero", 3, "all", utils::TransitionType::SmoothStep);
                stateLk.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                return;
            }

            // 行走立正切换
            if (msg->lb && !lastSwitched)
            {
                lastSwitched = true;
                if (state_ == sim2Real::State::RUNNING)
                {
                    ROS_INFO("sim2real switch to standby state");
                    _lastLoopMulti = "";
                    trajectory_->setSingle(robotMotor_->getMotorGroupStateByName("all"), "zero", 0.2, "all", utils::TransitionType::SmoothStep);
                    state_ = sim2Real::State::STANDING;
                }
                else if (state_ == sim2Real::State::STANDBY)
                {
                    ROS_INFO("sim2real switch to running state");
                    // 通知策略：进入 RUNNING，策略可自行处理初始 yaw
                    policy_->onEnterRunning(robotState_);
                    state_ = sim2Real::State::RUNNING;
                }
                stateLk.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                return;
            }
            else
            {
                lastSwitched = false;
            }
            // 算法切换 十字键左上一个算法，十字键右切换下一个算法
            if ((state_ == sim2Real::State::STANDBY || state_ == sim2Real::State::INIT))
            {
                bool change = false;
                if (msg->dpad_horizontal == 1) // 十字键左
                {
                    change = changePolicy("", "", false); // 切换到上一个算法
                    _lastPolicyName = "";
                }
                else if (msg->dpad_horizontal == -1) // 十字键右
                {
                    change = changePolicy("", "", true); // 切换到下一个算法
                    _lastPolicyName = "";
                }
                else if (msg->dpad_vertical == 1) // 十字键上
                {
                    change = changePolicy("", "walk"); // 切换走路算法
                }
                if (change && state_ == sim2Real::State::STANDBY)
                {
                    state_ = sim2Real::State::STANDING;
                    _lastLoopMulti = "";
                    trajectory_->setSingle(getMotorState(), "zero", 1, "all", utils::TransitionType::SmoothStep);
                    stateLk.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                    return;
                }
            }
        }
        if (state_ == sim2Real::State::RUNNING || state_ == sim2Real::State::STANDBY ||
            state_ == sim2Real::State::TRAJ_MULTI || state_ == sim2Real::State::TRAJ_SINGLE)
        {
            std::set<std::string> keySet{};
            std::string key = "";
            if (msg->lt < -0.5)
            {
                key = "lt";
                keySet.insert("lt");
            }
            if (msg->rt < -0.5)
            {
                key += key.empty() ? "rt" : "+rt";
                keySet.insert("rt");
            }
            // 如果key是空的 就只有rt
            if (msg->a == 1)
            {
                key += "+a";
                keySet.insert("a");
            }
            if (msg->b == 1)
            {
                key += "+b";
                keySet.insert("b");
            }
            if (msg->x == 1)
            {
                key += "+x";
                keySet.insert("x");
            }
            if (msg->y == 1)
            {
                key += "+y";
                keySet.insert("y");
            }
            if (msg->L == 1)
            {
                key += "+l";
                keySet.insert("l");
            }
            if (msg->R == 1)
            {
                key += "+r";
                keySet.insert("r");
            }
            if (msg->rb == 1)
            {
                key += "+rb";
                keySet.insert("rb");
            }
            if (msg->lb == 1)
            {
                key += "+lb";
                keySet.insert("lb");
            }
            if (msg->back == 1)
            {
                key += "+back";
                keySet.insert("back");
            }
            if (msg->start == 1)
            {
                key += "+start";
                keySet.insert("start");
            }
            if (msg->center == 1)
            {
                key += "+center";
                keySet.insert("center");
            }
            if (msg->dpad_horizontal == 1)
            {
                key += "+dpl";
                keySet.insert("dpl");
            }
            if (msg->dpad_horizontal == -1)
            {
                key += "+dpr";
                keySet.insert("dpr");
            }
            if (msg->dpad_vertical == 1)
            {
                key += "+dpu";
                keySet.insert("dpu");
            }
            if (msg->dpad_vertical == -1)
            {
                key += "+dpd";
                keySet.insert("dpd");
            }
            if (key.empty())
            {
                stateLk.unlock();
                return;
            }
            // ROS_INFO("%s", key.c_str());
            // for (const auto& k : keySet)
            // {
            //     ROS_INFO("keySet contains: %s", k.c_str());
            // }
            // 多动作
            auto itemMulti = std::find_if(pdInfo_.multi_configs[productionType_].begin(), pdInfo_.multi_configs[productionType_].end(),
                                          [&keySet](const std::pair<std::string, config::MultiWaypointConfig>& pair) {
                                              return pair.second.keySet == keySet;
                                          });

            // 上半身控制动作特殊处理：允许在 RUNNING 状态下执行并切换策略
            if (itemMulti != pdInfo_.multi_configs[productionType_].end())
            {
                const auto& name = itemMulti->second.name;
                auto group = trajectory_->getMotorPosesVecGroup(name);

                if ((group == "upperBody" || group == "arm_joint") && (state_ == sim2Real::State::RUNNING || state_ == sim2Real::State::STANDBY))
                {
                    joy_.reset();

                    // 特殊处理上半身控制动作：切换策略
                    if (name == _lastLoopMulti && itemMulti->second.loop)
                    {
                        auto preparedPolicy = policyChangeDefautePolicy.empty() ? rlConfigGroup_.getAllBodyWalkPolicy() : rlConfigGroup_.getRLConfigByName(policyChangeDefautePolicy);
                        // 上半身动作正在循环，再次按下则停止并切换回全身控制策略
                        ROS_INFO("DefaultControllerNode %s action stopping, switching back to %s[%s] walk policy", name.c_str(), preparedPolicy.name.c_str(), preparedPolicy.motor_group.c_str());

                        if (preparedPolicy.name.empty())
                        {
                            ROS_ERROR("No all-body walk policy configured, cannot switch back");
                            stateLk.unlock();
                            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                            return;
                        }
                        // 先平滑恢复上半身到零位，防止切换时震动
                        motorGroupRecoverByPolicyConfig(preparedPolicy.name, "upperBody", 0.5);
                        _lastLoopMulti = "";

                        // 等待上半身恢复完成（0.5秒），然后切换回全身控制走路算法
                        stateLk.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        stateLk.lock();

                        try
                        {
                            if (changePolicy(preparedPolicy.name))
                            {
                                ROS_INFO("Switched back to all-body walk policy: %s", preparedPolicy.name.c_str());
                            }
                            else
                            {
                                ROS_ERROR("Failed to switch back to all-body walk policy");
                            }
                        }
                        catch (const std::exception& e)
                        {
                            ROS_ERROR("Error switching back to all-body policy: %s", e.what());
                        }

                        stateLk.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        return;
                    }
                    else
                    {
                        // 上半身动作即将开始，切换到下半身控制策略
                        ROS_INFO("DefaultControllerNode %s action starting, switching to lower-body walk policy", name.c_str());

                        try
                        {
                            // 切换到下半身控制走路算法 (lr)
                            auto lowerBodyPolicy = rlConfigGroup_.getLowerBodyWalkPolicy();
                            if (changePolicy(lowerBodyPolicy.name))
                            {
                                ROS_INFO("Switched to lower-body walk policy: %s", lowerBodyPolicy.name.c_str());
                            }
                            else
                            {
                                ROS_ERROR("Failed to switch to lower-body walk policy");
                            }
                        }
                        catch (const std::exception& e)
                        {
                            ROS_ERROR("Error switching to lower-body policy: %s", e.what());
                        }
                    }

                    // 执行上半身动作
                    if (group != "lowerBody" && group != "all" || modelType_ != "pi")
                    {
                        stateLk.unlock();
                    }
                    setKpKd(group);
                    trajectory_->setMultiConfig(itemMulti->second);
                    trajectory_->setMulti(robotMotor_->getMotorGroupStateByName(group),
                                          name, itemMulti->second.duration, -0.01,
                                          group, utils::TransitionType(itemMulti->second.type),
                                          itemMulti->second.back_to_zero);
                    ROS_INFO("DefaultController executing upperBody action on group: %s", group.c_str());

                    // 如果是循环动作，记录上次循环的动作名称
                    if (itemMulti->second.loop && _lastLoopMulti != name)
                    {
                        _lastLoopMulti = name;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                    return;
                }
            }

            // 其他动作的正常处理
            if (itemMulti != pdInfo_.multi_configs[productionType_].end() &&
                (rlInfo_.motor_group.find("all") == std::string::npos || (rlInfo_.motor_group.find("all") != std::string::npos && state_ == sim2Real::State::STANDBY)))
            {
                joy_.reset();
                const auto& name = itemMulti->second.name;
                auto group = trajectory_->getMotorPosesVecGroup(name);

                if (name == _lastLoopMulti && itemMulti->second.loop)
                {
                    // 其他循环动作的正常处理
                    ROS_INFO("DefaultControllerNode Button (%s) pressed, but action %s is already looping, ignoring.", key.c_str(), name.c_str());
                    motorGroupRecover(group);
                    _lastLoopMulti = "";
                    stateLk.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                    return;
                }

                if (group != "lowerBody" && group != "all" || modelType_ != "pi")
                {
                    stateLk.unlock();
                }
                setKpKd(group);
                trajectory_->setMultiConfig(itemMulti->second);
                trajectory_->setMulti(robotMotor_->getMotorGroupStateByName(group),
                                      name, itemMulti->second.duration, -0.01,
                                      group, utils::TransitionType(itemMulti->second.type),
                                      itemMulti->second.back_to_zero);
                ROS_INFO("defautlController Button (%s) pressed, executing action %s", key.c_str(), name.c_str());
                if (group == "lowerBody" || group == "all" || modelType_ == "pi")
                {
                    state_ = sim2Real::State::TRAJ_MULTI;
                }

                // 如果是循环动作，记录上次循环的动作名称
                if (itemMulti->second.loop && _lastLoopMulti != name)
                {
                    _lastLoopMulti = name;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                return;
            }
            // 系列动作
            auto itemSeries = std::find_if(pdInfo_.series_configs[productionType_].begin(), pdInfo_.series_configs[productionType_].end(),
                                           [&keySet](const std::pair<std::string, config::SeriesWaypointConfig>& pair) {
                                               return keySet == pair.second.keySet;
                                           });
            if (itemSeries != pdInfo_.series_configs[productionType_].end() &&
                (rlInfo_.motor_group.find("all") == std::string::npos || (rlInfo_.motor_group.find("all") != std::string::npos && state_ == sim2Real::State::STANDBY)))
            {
                joy_.reset();
                auto firstSeries = itemSeries->second.waypointVec[0].name;
                auto group = trajectory_->getMotorPosesVecGroup(firstSeries);
                if (trajectory_->setSeries(itemSeries->second))
                {
                    setKpKd(group);
                    ROS_INFO("defautlController Button (%s) pressed, executing series action %s", key.c_str(), itemSeries->first.c_str());
                    if (group == "lowerBody" || group == "all" || modelType_ == "pi")
                    {
                        ROS_INFO("DefaultControllerNode change to TRAJ_SERIES state");
                        state_ = sim2Real::State::TRAJ_SERIES;
                    }
                }
                else
                {
                    ROS_ERROR("Failed to start series: %s", itemSeries->first.c_str());
                }
                stateLk.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                return;
            }
            // policy动作
            if (state_ == sim2Real::State::RUNNING)
            {
                auto itemPolicy = std::find_if(rlConfigGroup_.getPolicyChangeConfigs(productionType_).begin(), rlConfigGroup_.getPolicyChangeConfigs(productionType_).end(),
                                               [&keySet](const std::pair<std::string, config::PolicyChangeConfig>& pair) {
                                                   return pair.second.keySet == keySet;
                                               });
                if (itemPolicy != rlConfigGroup_.getPolicyChangeConfigs(productionType_).end())
                {
                    ROS_INFO("%s", itemPolicy->second.toString().c_str());
                    joy_.reset();
                    // 说明执行完了
                    if (!policyChangeByKeyPtr_ || _lastPolicyName != rlInfo_.name)
                    {
                        _lastPolicyName = "";
                    }
                    policyChangeByKeyPtr_ = std::make_shared<config::PolicyChangeConfig>(itemPolicy->second);
                    bool prePolicyChange = policyChangeByKeyPtr_->prePolicyChangeShouldExecute(cmdVel_, robotState_); // 策略切换过度
                    if (_lastPolicyName == policyChangeByKeyPtr_->name && policyChangeByKeyPtr_->loop)
                    {
                        ROS_INFO("DefaultControllerNode Button (%s) pressed, (%s) stop loop", key.c_str(), _lastPolicyName.c_str());
                        _lastPolicyName = "";
                        policyChangeByKeyPtr_->loop = false;
                    }
                    // policy 不同 且 当前状态是行走中且是walk类型 才允许切换
                    else if (_lastPolicyName != itemPolicy->second.name &&
                             ((state_ == sim2Real::State::RUNNING && rlInfo_.type == "walk")))
                    {
                        if (prePolicyChangeUpperBody())
                        {
                            state_ = sim2Real::State::PRE_POLICY_CHANGE;
                            ROS_INFO("DefaultControllerNode Button (%s) pressed, executing prePolicyChange: [%s]", key.c_str(), itemPolicy->second.pre_policy_change_description.c_str());
                            _lastPolicyName = itemPolicy->second.name;
                            // 上身动作
                        }
                        else
                        {
                            ROS_ERROR("DefaultControllerNode prePolicyChangeUpperBody failed, cannot change to policy %s(%s)", itemPolicy->second.name.c_str(), itemPolicy->second.policy.c_str());
                            policyChangeByKeyPtr_.reset();
                        }
                    }
                    else
                    {
                        ROS_ERROR("DefaultControllerNode change to policy %s(%s) failed last (%s)", itemPolicy->second.name.c_str(), itemPolicy->second.policy.c_str(), _lastPolicyName.c_str());
                    }
                    stateLk.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                    return;
                }
            }
        }
    }

    bool DefaultControllerNode::prePolicyChangeUpperBody()
    {
        // 切换算法前 如果不是全身控制的走路算法 如lr，则先把上肢调整到待切换算法的第一帧动作
        if (rlInfo_.type == "walk" && rlInfo_.motor_group.find("all") == std::string::npos)
        {
            motorGroupRecoverByPolicyConfig(policyChangeByKeyPtr_->policy, "upperBody");
        }
        else if (rlInfo_.type == "walk" && rlInfo_.motor_group.find("all") != std::string::npos)
        {
            if (!motorGroupRecoverByPolicyConfig(policyChangeByKeyPtr_->policy, "upperBody"))
            {
                return false;
            }
            changePolicy(rlConfigGroup_.getLowerBodyWalkPolicy().name);
        }
        return true;
    }

    DefaultControllerNode::~DefaultControllerNode()
    {
        stop();
        audioQuit_ = true;
        if (audioThread_.joinable())
        {
            audioThread_.join();
        }
        ROS_INFO("%s", __FUNCTION__);
    }
}
