#include "lowlevel_controller.h"
#include "common.h"
#include <sstream>

namespace hightorque
{
    LowlevelControllerNode::LowlevelControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr, std::shared_ptr<ros::NodeHandle> nh)
        : LevelControllerNode([&]() {
              // 在调用父类构造函数之前，检查是否有 develop 模式的配置文件
              std::string develPdConfig, develRlConfig;
              if (nh->getParam("devel_pd_config_file", develPdConfig) &&
                  nh->getParam("devel_rl_config_file", develRlConfig))
              {
                  ROS_INFO("LowlevelControllerNode: Using develop mode config files");
                  ROS_INFO("  PD config: %s", develPdConfig.c_str());
                  ROS_INFO("  RL config: %s", develRlConfig.c_str());
                  // 覆盖原有的配置文件参数
                  nh->setParam("pd_config_file", develPdConfig);
                  nh->setParam("rl_config_file", develRlConfig);
              }
              else
              {
                  ROS_INFO("LowlevelControllerNode: Using default config files");
              }
              return rbPtr;
          }(),
                              nh),
          actionServer_(*nh_, "lowlevel_controller", std::bind(&LowlevelControllerNode::execActionCallback, this, std::placeholders::_1), false)
    {
        ROS_INFO("LowlevelControllerNode %s()", __FUNCTION__);
    }

    bool LowlevelControllerNode::init()
    {
        initTrajectory();
        initMotor();
        initRos();
        return true;
    }

    bool LowlevelControllerNode::initRos()
    {
        ROS_INFO("LowlevelControllerNode %s()", __FUNCTION__);
        // ros
        robotStatePub_ = nh_->advertise<sensor_msgs::JointState>("rbt_state", 100);
        motorStatePub_ = nh_->advertise<sensor_msgs::JointState>("mtr_state", 100);
        motorOutputPub_ = nh_->advertise<sensor_msgs::JointState>("mtr_output", 100);

        joySub_ = nh_->subscribe<sim2real_msg::Joy>("/joy_msg", 10, &LowlevelControllerNode::joyCallback, this);

        // 订阅 preset 话题，用于执行预设姿态（如 zero）
        presetCommandSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_preset", 10,
                                                                    std::bind(&LowlevelControllerNode::robotPresetCommandCallback,
                                                                              this, std::placeholders::_1,
                                                                              "absolute"));

        // 订阅 _all 话题，用于电机的直接控制
        // 增大队列大小以匹配外部推理节点的 100Hz 发布频率，避免丢失控制命令
        allJointSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_all", 100,
                                                               std::bind(&LowlevelControllerNode::robotAllJointCommandCallback,
                                                                         this, std::placeholders::_1,
                                                                         "absolute"));
        allJointRelativeSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_all_relative", 100,
                                                                       std::bind(&LowlevelControllerNode::robotAllJointCommandCallback,
                                                                                 this, std::placeholders::_1,
                                                                                 "relative"));
        leftAnkleJointSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_left_ankle", 10,
                                                                     std::bind(
                                                                         &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                         this, std::placeholders::_1,
                                                                         "left_ankle_joint",
                                                                         "absolute"));
        leftAnkleJointRelativeSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_left_ankle_relative", 10,
                                                                             std::bind(
                                                                                 &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                                 this, std::placeholders::_1,
                                                                                 "left_ankle_joint",
                                                                                 "relative"));
        rightAnkleJointSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_right_ankle", 10,
                                                                      std::bind(
                                                                          &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                          this, std::placeholders::_1,
                                                                          "right_ankle_joint",
                                                                          "absolute"));
        rightAnkleJointRelativeSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_right_ankle_relative", 10,
                                                                              std::bind(
                                                                                  &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                                  this, std::placeholders::_1,
                                                                                  "right_ankle_joint",
                                                                                  "relative"));
        ankleJointSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_ankle", 10,
                                                                 std::bind(
                                                                     &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                     this, std::placeholders::_1,
                                                                     "ankle_joint",
                                                                     "absolute"));
        ankleJointRelativeSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_ankle_relative", 10,
                                                                         std::bind(
                                                                             &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                             this, std::placeholders::_1,
                                                                             "ankle_joint",
                                                                             "relative"));
        hipJointSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_hip", 10,
                                                               std::bind(
                                                                   &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                   this, std::placeholders::_1,
                                                                   "hip_joint",
                                                                   "absolute"));
        hipJointRelativeSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_hip_relative", 10,
                                                                       std::bind(
                                                                           &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                           this, std::placeholders::_1,
                                                                           "hip_joint",
                                                                           "relative"));
        thighJointSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_thigh", 10,
                                                                 std::bind(
                                                                     &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                     this, std::placeholders::_1,
                                                                     "thigh_joint",
                                                                     "absolute"));
        thighJointRelativeSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_thigh_relative", 10,
                                                                         std::bind(
                                                                             &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                             this, std::placeholders::_1,
                                                                             "thigh_joint",
                                                                             "relative"));
        if (modelType_ == "pi_plus")
        {
            armJointSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_arm", 10,
                                                                   std::bind(
                                                                       &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                       this, std::placeholders::_1,
                                                                       "arm_joint",
                                                                       "absolute"));
            armJointRelativeSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_arm_relative", 10,
                                                                           std::bind(
                                                                               &LowlevelControllerNode::robotGroupJointCommandCallback,
                                                                               this, std::placeholders::_1,
                                                                               "arm_joint",
                                                                               "relative"));
        }
        // TODO:配置需要的关节组
        if (modelType_ == "hi")
        {
        }
        jointSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_, 10,
                                                            std::bind(
                                                                &LowlevelControllerNode::robotJointCommandCallback,
                                                                this, std::placeholders::_1,
                                                                "absolute"));
        jointRelativeSub_ = nh_->subscribe<sensor_msgs::JointState>("/" + modelType_ + "_relative", 10,
                                                                    std::bind(
                                                                        &LowlevelControllerNode::robotJointCommandCallback,
                                                                        this, std::placeholders::_1,
                                                                        "relative"));
        actionServer_.start();
        return true;
    }

    void LowlevelControllerNode::joyCallback(const sim2real_msg::Joy::ConstPtr& msg)
    {
        // 暂时关闭所有手柄控制关节功能
        return;

        // std::unique_lock<std::shared_timed_mutex> lk(mutex_);
        // auto jointsNum = robotMotor_->getMotorGroupSizeByName("left_ankle_joint");
        // if (jointsNum < 0)
        // {
        //     ROS_ERROR("left_ankle_joint error");
        //     return;
        // }

        // Motor_Output out1, out2;
        // out1.resize(jointsNum);
        // out2.resize(jointsNum);
        // out1.target_q.setConstant(0.1);
        // out2.target_q.setConstant(-0.1);

        // MotorControlCmd cmd1, cmd2;
        // cmd1.position = 0.1;
        // cmd2.position = -0.1;

        // // 设置单一电机 方法1
        // auto setMotor1 = [&](const std::string& name, MotorControlCmd cmd) {
        //     if (auto motor = robotMotor_->getMotorByName(name))
        //     {
        //         int index = motor->getIndex();
        //         cmd.kp = pdInfo_.byMotorId.kp[index];
        //         cmd.kd = pdInfo_.byMotorId.kd[index];
        //         motor->setMotorRelative(cmd);
        //     }
        // };

        // // 设置单一电机 方法2
        // auto setMotor2 = [&](const std::string& name, MotorControlCmd cmd) {
        //     robotMotor_->foreachMotorByName("all", [&](const std::shared_ptr<MotorBase>& motor) {
        //         if (motor->getName() == name)
        //         {
        //             int index = motor->getIndex();
        //             cmd.kp = pdInfo_.byMotorId.kp[index];
        //             cmd.kd = pdInfo_.byMotorId.kd[index];
        //             motor->setMotorRelative(cmd);
        //             return true;
        //         }
        //         return false;
        //     });
        // };

        // if (msg->R == 1)
        // {
        //     auto group = trajectory_->getMotorPosesVecGroup("8");
        //     auto state = robotMotor_->getMotorGroupStateByName(group);
        //     if (group.empty())
        //     {
        //         return;
        //     }
        //     // setKpKd(MotorControlType::POS_VEL_TQE_KP_KD);
        //     setKpKd();
        //     trajectory_->setMulti(state, "8", 0.05, -0.01, group, utils::TransitionType::CubicSpline, false);
        // }
        // // 左边
        // else if (msg->lt == -1 && msg->rt == 1)
        // {
        //     if (msg->a == 1) // A
        //     {
        //         setMotor2("l_ankle_pitch_joint", cmd1);
        //     }
        //     else if (msg->y == 1) // Y
        //     {
        //         setMotor2("l_ankle_pitch_joint", cmd2);
        //     }
        //     else if (msg->dpad_horizontal == 1) // 十字键左
        //     {
        //         setMotor2("l_ankle_roll_joint", cmd1);
        //     }
        //     else if (msg->dpad_horizontal == -1) // 十字键右
        //     {
        //         setMotor2("l_ankle_roll_joint", cmd2);
        //     }
        //     else if (msg->lb == 1) // LB
        //     {
        //         setMotor2("l_shoulder_roll_joint", cmd1);
        //     }
        //     else if (msg->rb == 1) // RB
        //     {
        //         setMotor2("l_shoulder_roll_joint", cmd2);
        //     }
        // }
        // // 右边
        // else if (msg->lt == 1 && msg->rt == -1)
        // {
        //     if (msg->a == 1) // A
        //     {
        //         setMotor2("r_ankle_pitch_joint", cmd2);
        //     }
        //     else if (msg->y == 1) // Y
        //     {
        //         setMotor2("r_ankle_pitch_joint", cmd1);
        //     }
        //     else if (msg->dpad_horizontal == 1) // 十字键左
        //     {
        //         setMotor2("r_ankle_roll_joint", cmd1);
        //     }
        //     else if (msg->dpad_horizontal == -1) // 十字键右
        //     {
        //         setMotor2("r_ankle_roll_joint", cmd2);
        //     }
        //     else if (msg->lb == 1) // LB
        //     {
        //         setMotor2("r_shoulder_roll_joint", cmd1);
        //     }
        //     else if (msg->rb == 1) // RB
        //     {
        //         setMotor2("r_shoulder_roll_joint", cmd2);
        //     }
        // }
        // else if (msg->lt == -1 && msg->rt == -1)
        // {
        //     if (msg->a == 1) // A
        //     {
        //         setMotor1("l_ankle_pitch_joint", cmd1);
        //         setMotor1("r_ankle_pitch_joint", cmd2);
        //     }
        //     else if (msg->y == 1) // Y
        //     {
        //         setMotor1("l_ankle_pitch_joint", cmd2);
        //         setMotor1("r_ankle_pitch_joint", cmd1);
        //     }
        //     else if (msg->dpad_horizontal == 1) // 十字键左
        //     {
        //         setMotor1("l_ankle_roll_joint", cmd1);
        //         setMotor1("r_ankle_roll_joint", cmd1);
        //     }
        //     else if (msg->dpad_horizontal == -1) // 十字键右
        //     {
        //         setMotor1("l_ankle_roll_joint", cmd2);
        //         setMotor1("r_ankle_roll_joint", cmd2);
        //     }
        //     else if (msg->lb == 1) // LB
        //     {
        //         setMotor1("l_shoulder_roll_joint", cmd2);
        //         setMotor1("r_shoulder_roll_joint", cmd1);
        //     }
        //     else if (msg->rb == 1) // RB
        //     {
        //         setMotor1("l_shoulder_roll_joint", cmd1);
        //         setMotor1("r_shoulder_roll_joint", cmd2);
        //     }
        // }
        // else if (msg->lb == 1) // RB
        // {
        //     auto index = robotMotor_->getMotorGroupIndexByName("thigh_joint");
        //     common::motorOutputSetKpKd(out1, pdInfo_.byMotorId.kp, pdInfo_.byMotorId.p_kd, index);
        //     robotMotor_->setMotorGroupRelativeByName("thigh_joint", out1);
        // }
        // else if (msg->rb == 1) // RB
        // {
        //     auto index = robotMotor_->getMotorGroupIndexByName("thigh_joint");
        //     common::motorOutputSetKpKd(out2, pdInfo_.byMotorId.kp, pdInfo_.byMotorId.p_kd, index);
        //     robotMotor_->setMotorGroupRelativeByName("thigh_joint", out2);
        // }
        // else if (msg->r_vertical > 0.5) // 右摇杆向前
        // {
        //     Motor_Output out(4);
        //     out.target_q << 0.1, 0, 0.1, 0;
        //     auto index = robotMotor_->getMotorGroupIndexByName("hip_joint");
        //     common::motorOutputSetKpKd(out, pdInfo_.byMotorId.kp, pdInfo_.byMotorId.p_kd, index);
        //     robotMotor_->setMotorGroupRelativeByName("hip_joint", out);
        // }
        // else if (msg->r_vertical < -0.5) // 右摇杆向后
        // {
        //     Motor_Output out(4);
        //     out.target_q << -0.1, 0, -0.1, 0;
        //     auto index = robotMotor_->getMotorGroupIndexByName("hip_joint");
        //     common::motorOutputSetKpKd(out, pdInfo_.byMotorId.kp, pdInfo_.byMotorId.p_kd, index);
        //     robotMotor_->setMotorGroupRelativeByName("hip_joint", out);
        // }
    }

    void LowlevelControllerNode::robotJointCommandCallback(const sensor_msgs::JointState::ConstPtr& msg, const std::string& type)
    {
        auto jointsNum = msg->name.size();
        bool v = (msg->velocity.size() == jointsNum), t = (msg->effort.size() == jointsNum);
        if (msg->position.size() != jointsNum && jointsNum > 0)
        {
            ROS_ERROR("position size:(%zu) != joint size:(%zu)", msg->position.size(), jointsNum);
            return;
        }

        for (auto i = 0; i < jointsNum; i++)
        {
            auto name = msg->name[i];
            const auto motor = robotMotor_->getMotorByName(name);
            if (!motor)
            {
                return;
            }
            MotorControlCmd cmd;
            // common::transformRobotOutputToMotorOutput(msg->position[i], cmd.position, pdInfo_.byMotorId.urdf_offset[motor->getIndex()], pdInfo_.byMotorId.direction[motor->getIndex()]);
            cmd.position = msg->position[i];
            cmd.velocity = v ? msg->velocity[i] : 0.0;
            cmd.torque = t ? msg->effort[i] : 0.0;
            cmd.kp = pdInfo_.byMotorId.kp[motor->getIndex()];
            cmd.kd = pdInfo_.byMotorId.kd[motor->getIndex()];
            if (type == "absolute")
            {
                motor->setMotor(cmd);
            }
            else if (type == "relative")
            {
                motor->setMotorRelative(cmd);
            }
        }
    }

    void LowlevelControllerNode::robotAllJointCommandCallback(const sensor_msgs::JointState::ConstPtr& msg, const std::string& type)
    {
        double duration = msg->header.stamp.toSec();
        std::unique_lock<std::shared_timed_mutex> lk(mutex_);

        auto setMotor = [&](const Motor_Output out) {
            if (type == "absolute")
            {
                robotMotor_->setMotorGroupByName("all", out);
            }
            else if (type == "relative")
            {
                robotMotor_->setMotorGroupRelativeByName("all", out);
            }
        };

        auto jointsNum = robotMotor_->getMotorGroupSizeByName("all");
        if (jointsNum != msg->position.size())
        {
            ROS_ERROR("position size:(%zu) != all joint size:(%zu)", msg->position.size(), jointsNum);
            return;
        }

        Robot_Output robotOutput(jointsNum);
        Motor_Output motorOutput(jointsNum);
        robotOutput.parseFromJointState(*msg, jointsNum);

        auto index = robotMotor_->getMotorGroupIndexByName("all");

        // 重要：过滤输出，限制在安全范围内（与 sim2real.cpp 保持一致）
        robotOutput.filter(pdInfo_.byMotorId.clip_output_lower, pdInfo_.byMotorId.clip_output_upper, index);

        common::transformRobotOutputToMotorOutput(robotOutput, motorOutput, pdInfo_.byMotorId.urdf_offset, pdInfo_.byMotorId.direction, index, modelType_ == "hi" ? 1 : 0);

        if (std::fabs(duration) < 1e-6)
        {
            // 缓存命令，由 PD 循环以 1kHz 统一下发（支持 relative 转 absolute）
            common::motorOutputSetKpKd(motorOutput, pdInfo_.byMotorId.p_kp, pdInfo_.byMotorId.p_kd, index);
            // ROS_INFO_THROTTLE(1, "Set pending all joint command: %s", motorOutput.toString().c_str());
            pendingAllMotorOutput_ = motorOutput;
        }
        else
        {
            if (type == "relative")
            {
                motorOutput = motorOutput + robotMotor_->getMotorGroupStateByName("all").toOut();
            }
            trajectory_->setSingle(robotMotor_->getMotorGroupStateByName("all"), motorOutput, duration, "all", utils::TransitionType::SmoothStep);
        }
    }

    void LowlevelControllerNode::robotPresetCommandCallback(const sensor_msgs::JointState::ConstPtr& msg, const std::string& type)
    {
        if (msg->header.frame_id.empty())
        {
            ROS_WARN("Preset command requires frame_id (e.g., 'zero', 'sitdown')");
            return;
        }

        double duration = msg->header.stamp.toSec();
        std::unique_lock<std::shared_timed_mutex> lk(mutex_);

        auto setMotor = [&](const Motor_Output out) {
            if (type == "absolute")
            {
                robotMotor_->setMotorGroupByName("all", out);
            }
            else if (type == "relative")
            {
                robotMotor_->setMotorGroupRelativeByName("all", out);
            }
        };

        if (std::fabs(duration) < 1e-6)
        {
            auto out = trajectory_->getMotorPoses(msg->header.frame_id);
            common::motorOutputSetKpKd(out, pdInfo_.kp, pdInfo_.kd);
            setMotor(out);
        }
        else
        {
            trajectory_->setSingle(robotMotor_->getMotorGroupStateByName("all"), msg->header.frame_id, duration, "all", utils::TransitionType::SmoothStep);
        }
    }

    void LowlevelControllerNode::robotGroupJointCommandCallback(const sensor_msgs::JointState::ConstPtr& msg, const std::string& group, const std::string& type)
    {
        double duration = msg->header.stamp.toSec(); // 获取持续时间
        std::unique_lock<std::shared_timed_mutex> lk(mutex_);

        auto setMotor = [&](const Motor_Output out) {
            if (type == "absolute")
            {
                robotMotor_->setMotorGroupByName(group, out);
            }
            else if (type == "relative")
            {
                robotMotor_->setMotorGroupRelativeByName(group, out);
            }
        };

        auto jointsNum = robotMotor_->getMotorGroupSizeByName(group);
        if (jointsNum != msg->position.size())
        {
            ROS_ERROR("position size:(%zu) != group(%s) joint size:(%zu)", msg->position.size(), group.c_str(), jointsNum);
            return;
        }
        Robot_Output robotOutput(jointsNum);
        Motor_Output motorOutput(jointsNum);
        robotOutput.parseFromJointState(*msg, jointsNum);
        auto index = robotMotor_->getMotorGroupIndexByName(group);
        common::transformRobotOutputToMotorOutput(robotOutput, motorOutput, pdInfo_.byMotorId.urdf_offset, pdInfo_.byMotorId.direction, index);
        if (std::fabs(duration) < 1e-6)
        {
            common::motorOutputSetKpKd(motorOutput, pdInfo_.byMotorId.kp, pdInfo_.byMotorId.kd, index);
            setMotor(motorOutput);
        }
        else
        {
            if (type == "relative")
            {
                motorOutput = motorOutput + robotMotor_->getMotorGroupStateByName(group).toOut();
            }
            trajectory_->setSingle(robotMotor_->getMotorGroupStateByName(group), motorOutput, duration, group, utils::TransitionType::SmoothStep);
        }
    }

    bool LowlevelControllerNode::start()
    {
        quit_ = false;
        try
        {
            pdThreadExecPtr_.reset(new std::thread(&LowlevelControllerNode::execPdLoop, this));
            ROS_INFO("LowlevelControllerNode started successfully");
            return true;
        }
        catch (const std::exception& e)
        {
            ROS_ERROR("Failed to start DefautlController thread: %s", e.what());
            return false;
        }
    }

    void LowlevelControllerNode::execPdLoop()
    {
        keepMotor("all");
        ros::Rate rate(1000);
        const int pubStateDuration = 0;
        unsigned int pubCount = 0;
        ROS_INFO("LowlevelControllerNode execLoop");
        while (!quit_ && ros::ok())
        {
            if (++pubCount >= pubStateDuration && !isMix_)
            {
                pubCount = 0;
                publish();
            }

            bool set = false;
            if (trajectory_->isSingleRunning())
            {
                auto out = trajectory_->generatorSingle();
                auto group = trajectory_->getSingleGroup();
                if (!out.isZero())
                {
                    std::unique_lock<std::shared_timed_mutex> lk(mutex_);
                    auto index = robotMotor_->getMotorGroupIndexByName(group);
                    common::motorOutputSetKpKd(out, pdInfo_.byMotorId.kp, pdInfo_.byMotorId.kd, index);
                    robotMotor_->setMotorGroupByName(group, out);
                    set = true;
                }
            }
            else if (trajectory_->isMultiRunning())
            {
                auto group = trajectory_->getMultiGroup();
                auto out = trajectory_->generatorMulti(robotMotor_->getMotorGroupStateByName(group));
                if (!out.isZero())
                {
                    std::unique_lock<std::shared_timed_mutex> lk(mutex_);
                    auto index = robotMotor_->getMotorGroupIndexByName(group);
                    common::motorOutputSetKpKd(out, pdInfo_.byMotorId.kp, pdInfo_.byMotorId.kd, index);
                    out.target_tau.setConstant(10); // 设置最大力矩
                    robotMotor_->setMotorGroupByName(group, out, MotorControlType::POS_VEL_MAX_TQE);
                    set = true;
                }
            }
            else if (!set && !isMix_)
            {
                // 从缓存中读取 _all 命令并以 1kHz 下发
                std::unique_lock<std::shared_timed_mutex> lk(mutex_);
                auto out = pendingAllMotorOutput_;
                auto jointsNum = robotMotor_->getMotorGroupSizeByName("all");
                if (jointsNum > 0 && out.target_q.size() == jointsNum)
                {
                    // auto index = robotMotor_->getMotorGroupIndexByName("all");
                    // common::motorOutputSetKpKd(out, pdInfo_.byMotorId.p_kp, pdInfo_.byMotorId.p_kd, index);
                    ROS_INFO_THROTTLE(10, "Set output all joint command: %s", out.toString().c_str());
                    robotMotor_->setMotorGroupByName("all", out);
                }
                else
                {
                    robotMotor_->keepMotorGroupLastCmdByName("all");
                }
            }

            std::shared_lock<std::shared_timed_mutex> lk(mutex_);
            if (!isMix_)
            {
                robotMotor_->sendMotors();
            }
            lk.unlock();
            rate.sleep();
        }
    }

    void LowlevelControllerNode::execActionCallback(const sim2real_msg::lowlevel_controllerGoalConstPtr& goal)
    {
        sim2real_msg::lowlevel_controllerFeedback feedback;
        sim2real_msg::lowlevel_controllerResult result;

        // 获取关节组数量
        auto jointsNum = robotMotor_->getMotorGroupSizeByName(goal->group);
        if (jointsNum < 0 || goal->group.empty())
        {
            result.success = false;
            result.errorMessage = "not such group:" + goal->group;
            actionServer_.setAborted(result);
            return;
        }

        std::vector<std::vector<double>> pointsVec;
        if (goal->motorOutput.size() != 0)
        {
            size_t size = goal->motorOutput.size();
            pointsVec.reserve(size);
            // multi
            if (size > 1)
            {
                ROS_INFO("LowlevelControllerNode Motor_Output multi, size: %zu", size);
                for (const auto& out : goal->motorOutput)
                {
                    if (out.position.size() != jointsNum)
                    {
                        ROS_WARN("position size: %zu != joints num %zu", out.position.size(), jointsNum);
                        continue;
                    }
                    pointsVec.push_back(out.position);
                }
                ROS_INFO("LowlevelControllerNode actuality size: %zu", pointsVec.size());
                if (pointsVec.size() == 0)
                {
                    return;
                }
                trajectory_->setMulti(robotMotor_->getMotorGroupStateByName(goal->group), pointsVec, goal->duration, 0.0, goal->group, static_cast<utils::TransitionType>(goal->type), false);
            }
            // single
            else
            {
                ROS_INFO("LowlevelControllerNode Motor_Output single");
            }
        }
        else if (goal->robotOutput.size() != 0)
        {
            size_t size = goal->robotOutput.size();
            Robot_Output robotOutput(jointsNum);
            Motor_Output motorOutput(jointsNum);
            auto index = robotMotor_->getMotorGroupIndexByName(goal->group);
            // multi
            if (size > 1)
            {
                ROS_INFO("LowlevelControllerNode Robot_Output multi, size: %zu", size);
                for (auto i = 0; i < size; i++)
                {
                    robotOutput.parseFromJointState(goal->robotOutput[i], jointsNum);
                    common::transformRobotOutputToMotorOutput(robotOutput, motorOutput, pdInfo_.byMotorId.urdf_offset, pdInfo_.byMotorId.direction, index);
                    pointsVec.push_back(common::toStdVector(motorOutput.target_q));
                }
                ROS_INFO("LowlevelControllerNode actuality size: %zu", pointsVec.size());
                trajectory_->setMulti(robotMotor_->getMotorGroupStateByName(goal->group), pointsVec, goal->duration, 0.0, goal->group, static_cast<utils::TransitionType>(goal->type), false);
            }
            // signle
            else
            {
                ROS_INFO("LowlevelControllerNode Robot_Output single");
            }
        }
        else
        {
            result.success = false;
            result.errorMessage = "motorOutput's size && robotOutput's size is zero'";
            actionServer_.setAborted(result);
            return;
        }
        ROS_INFO("action start");

        ros::Rate rate(50);
        Robot_State robotState(jointsNum);
        auto index = robotMotor_->getMotorGroupIndexByName(goal->group);
        while (ros::ok())
        {
            if (actionServer_.isPreemptRequested())
            {
                ROS_WARN("LowlevelControllerNode Goal canceled by client");
                result.success = false;
                result.errorMessage = "canceled by client";
                actionServer_.setPreempted();
                trajectory_->stopMulti();
                return;
            }
            if (!trajectory_->isMultiRunning())
            {
                ROS_INFO("LowlevelControllerNode Reach Goal!");
                result.success = true;
                result.errorMessage = "";
                actionServer_.setSucceeded(result);
                return;
            }
            auto motorState = robotMotor_->getMotorGroupStateByName(goal->group);
            if (motorState.q.size() == 0)
            {
                ROS_WARN("LowlevelControllerNode Goal canceled because group:(%s) not found", goal->group.c_str());
                actionServer_.setPreempted();
                return;
            }
            feedback.motorState.position = common::toStdVector(motorState.q);
            feedback.motorState.velocity = common::toStdVector(motorState.dq);
            feedback.motorState.effort = common::toStdVector(motorState.tau);

            common::transformMotorStateToRobotState(motorState, robotState, pdInfo_.byMotorId.urdf_offset, pdInfo_.byMotorId.direction, index);
            feedback.robotState.position = common::toStdVector(robotState.q);
            feedback.robotState.velocity = common::toStdVector(robotState.dq);
            feedback.robotState.effort = common::toStdVector(robotState.tau);

            actionServer_.publishFeedback(feedback);

            rate.sleep();
        }
    }

    LowlevelControllerNode::~LowlevelControllerNode()
    {
        ROS_INFO("low %s()", __FUNCTION__);
        stop();
    };
}

