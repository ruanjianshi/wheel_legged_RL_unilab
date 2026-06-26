#include "sim2real/teach_controller_interface_n.h"
#include "motor/pi_motor_group.h"
#include "motor/pi_plus_motor_group.h"
#include "ros/package.h"
#include "std_msgs/String.h"
#include <future>
#include <filesystem>
#include "common.h"

namespace hightorque
{
    namespace teach
    {
        TeachControllerInterface::TeachControllerInterface(std::shared_ptr<livelybot_serial::robot> rbPtr) : BaseControllerInterface(rbPtr)
        {
            ROS_INFO("Teach %s()", __FUNCTION__);
            fileName_ = "test.boost";
            group_ = "all";

            // ros
            std::string configPath, pdConfigFile, rlConfigFile, policyPath, actionConfigPath, productionType;
            nh_ = std::make_shared<ros::NodeHandle>("~");
            nh_->param<std::string>("model_type", modelType_, "pi");
            nh_->param<std::string>("pd_config_file", pdConfigFile, ".yaml");
            nh_->param<std::string>("rl_config_file", rlConfigFile, ".yaml");
            nh_->param<std::string>("action_path", actionConfigPath, "");
            ROS_INFO("pdConfigFile : %s", pdConfigFile.c_str());
            nh_->param<double>("kick_duration", kickDuration_, 0.2); // 机器人模型类型
            nh_->param<std::string>("production_type", productionType, "test");
            pdInfo_.loadFromYaml(pdConfigFile, actionConfigPath);
            pdInfo_.refreshByMotorIdVec();
            pdInfo_.print();

            // motor
            if (modelType_ == "pi")
            {
                robotMotor_ = std::make_shared<PiMotorGroup>(rbPtr, pdInfo_.map_index, pdInfo_.joint_names, pdInfo_.lower, pdInfo_.upper);
            }
            else if (modelType_ == "pi_plus")
            {
                robotMotor_ = std::make_shared<PiPlusMotorGroup>(rbPtr, pdInfo_.map_index, pdInfo_.joint_names, pdInfo_.lower, pdInfo_.upper);
            }
            else
            {
                quit_ = true;
                return;
            }
            if (!initMotor())
            {
                quit_ = true;
                return;
            }

            // trajectory
            filePath_ = ros::package::getPath("sim2real");
            std::vector<config::MultiWaypointConfig> waypoints;
            for (auto& config : pdInfo_.multi_configs[productionType])
            {
                config.second.path = (filePath_ + "/" + config.second.path);
                waypoints.push_back(config.second);
                ROS_INFO("Sim2Real trajectory (%s) file: %s", config.first.c_str(), config.second.path.c_str());
            }
            filePath_ += "/action_config";

            trajectory_ = &utils::TrajectoryGenerator::getInstance(pdInfo_.urdf_offset, pdInfo_.direction, waypoints, modelType_);

            // topic
            waypointSub_ = nh_->subscribe<sim2real_msg::Joy>("/joy_msg", 10, &TeachControllerInterface::joyCallback, this);
            fileNameSub_ = nh_->subscribe<std_msgs::String>("file_name", 10, [this](const std_msgs::String::ConstPtr& msg) {
                this->fileName_ = msg->data;
            });
            motorGroupSub_ = nh_->subscribe<std_msgs::String>("motor_group", 10, [this](const std_msgs::String::ConstPtr& msg) {
                this->group_ = msg->data;
            });

            quit_ = false;
            start();
        }

        bool TeachControllerInterface::start()
        {
            threadExecPtr_.reset(new std::thread(&TeachControllerInterface::execLoop, this));
            ROS_INFO("Teach started successfully");
            return true;
        }

        bool TeachControllerInterface::stop()
        {
            // 标记线程退出
            quit_ = true;
            trajectory_->stopMulti();
            trajectory_->stopSingle();

            // 等待线程结束
            if (threadExecPtr_ && threadExecPtr_->joinable())
            {
                try
                {
                    // 设置超时，避免无限等待
                    auto timeout = std::chrono::seconds(5);
                    auto future = std::async(std::launch::async, [this]() {
                        threadExecPtr_->join();
                    });

                    if (future.wait_for(timeout) == std::future_status::timeout)
                    {
                        ROS_WARN("Teach thread join timed out after 5 seconds");
                        return false;
                    }

                    ROS_INFO("Teach stopped successfully");
                }
                catch (const std::exception& e)
                {
                    ROS_ERROR("Error while stopping Teach thread: %s", e.what());
                    return false;
                }
            }
            return true;
        }

        bool TeachControllerInterface::initMotor()
        {
            std::vector<std::shared_ptr<MotorBase>> leftAnkleMotors, rightAnkleMotors, ankleMotors, hipMotors, thighMotors, armMotors;
            leftAnkleMotors.push_back(robotMotor_->getMotorByName("l_ankle_pitch_joint"));
            leftAnkleMotors.push_back(robotMotor_->getMotorByName("l_ankle_roll_joint"));

            rightAnkleMotors.push_back(robotMotor_->getMotorByName("r_ankle_pitch_joint"));
            rightAnkleMotors.push_back(robotMotor_->getMotorByName("r_ankle_roll_joint"));

            // 注册脚踝关节组
            ankleMotors.push_back(robotMotor_->getMotorByName("l_ankle_pitch_joint"));
            ankleMotors.push_back(robotMotor_->getMotorByName("l_ankle_roll_joint"));
            ankleMotors.push_back(robotMotor_->getMotorByName("r_ankle_pitch_joint"));
            ankleMotors.push_back(robotMotor_->getMotorByName("r_ankle_roll_joint"));

            // 注册臀部关节组
            hipMotors.push_back(robotMotor_->getMotorByName("l_hip_pitch_joint"));
            hipMotors.push_back(robotMotor_->getMotorByName("l_hip_roll_joint"));
            hipMotors.push_back(robotMotor_->getMotorByName("r_hip_pitch_joint"));
            hipMotors.push_back(robotMotor_->getMotorByName("r_hip_roll_joint"));

            // 注册大腿关节组
            thighMotors.push_back(robotMotor_->getMotorByName("l_thigh_joint"));
            thighMotors.push_back(robotMotor_->getMotorByName("r_thigh_joint"));

            // 注册手部关节组
            if (modelType_ == "pi_plus")
            {
                armMotors.push_back(robotMotor_->getMotorByName("l_shoulder_pitch_joint"));
                armMotors.push_back(robotMotor_->getMotorByName("l_shoulder_roll_joint"));
                armMotors.push_back(robotMotor_->getMotorByName("l_upper_arm_joint"));
                armMotors.push_back(robotMotor_->getMotorByName("l_elbow_joint"));
                armMotors.push_back(robotMotor_->getMotorByName("r_shoulder_pitch_joint"));
                armMotors.push_back(robotMotor_->getMotorByName("r_shoulder_roll_joint"));
                armMotors.push_back(robotMotor_->getMotorByName("r_upper_arm_joint"));
                armMotors.push_back(robotMotor_->getMotorByName("r_elbow_joint"));
                robotMotor_->registerMotorGroup("arm_joint", armMotors);
            }

            robotMotor_->registerMotorGroup("left_ankle_joint", leftAnkleMotors);
            robotMotor_->registerMotorGroup("right_ankle_joint", rightAnkleMotors);
            robotMotor_->registerMotorGroup("ankle_joint", ankleMotors);
            robotMotor_->registerMotorGroup("hip_joint", hipMotors);
            robotMotor_->registerMotorGroup("thigh_joint", thighMotors);
            return true;
        }

        void TeachControllerInterface::setKpKd() const
        {
            auto out = robotMotor_->getMotorGroupStateByName("all").toOut();
            out.target_dq.setZero();
            out.target_tau.setZero();
            common::motorOutputSetKpKd(out, pdInfo_.kp, pdInfo_.kd);
            robotMotor_->setMotorGroupByName("all", out);
            ROS_INFO("Teach set kp kd");
            robotMotor_->sendMotors();
            // 为了保持其他电机都是硬的，需要设置一次相同方法的电机配置方法
            out.target_tau.setConstant(10);
            robotMotor_->setMotorGroupByName("all", out, MotorControlType::POS_VEL_MAX_TQE);
        };

        void TeachControllerInterface::joyCallback(const sim2real_msg::Joy::ConstPtr& msg)
        {
            std::unique_lock<std::shared_timed_mutex> lk(mutex_);
            auto nums = motorStateVec_.size();
            // RT
            if (msg->rt < 0)
            {
                // X
                if (msg->x == 1.0)
                {
                    motorStateVec_.clear();
                }
            }
            // LT
            else if (msg->lt < 0)
            {
            }
            // LT + RT
            else if (msg->rt < 0 && msg->lt < 0)
            {
            }
            // 中央左侧
            else if (msg->start == 1.0)
            {
            }
            // 中央右侧
            else if (msg->back == 1.0)
            {
            }
            // A
            else if (msg->a == 1.0)
            {
                if (nums == 0)
                {
                    ROS_WARN("waypoint's size must > 0");
                }
                else
                {
                    std::filesystem::path file = filePath_ + "/" + fileName_;
                    std::vector<std::vector<double>> matrix;
                    matrix.reserve(motorStateVec_.size());
                    for (auto state : motorStateVec_)
                    {
                        matrix.push_back(common::toStdVector(state.q));
                    }
                    if (common::saveMatrixWithMetadata(file, group_, matrix))
                    {
                        ROS_INFO("save successfully");
                    }
                    else
                    {
                        ROS_ERROR("save unsuccessfully");
                    }
                }
            }
            // B
            else if (msg->b == 1.0)
            {
                std::filesystem::path file = filePath_ + "/" + fileName_;
                std::vector<std::vector<double>> matrix;
                std::string group;
                if (common::loadMatrixWithMetadata(file, group, matrix))
                {
                    ROS_INFO("load file: (%s), group: %s, matrix size: %zu", file.filename().c_str(), group.c_str(), matrix.size());
                    auto state = robotMotor_->getMotorGroupStateByName(group);
                    setKpKd();
                    // trajectory_->setMulti(state, matrix, 0.013125, -0.01, group, utils::TransitionType::CubicSpline, false);
                    trajectory_->setMulti(state, matrix, 0.2, -0.01, group, utils::TransitionType::SmoothStep, false);
                }
                else
                {
                    ROS_ERROR("load file: (%s) unsuccessfully", file.filename().c_str());
                }
            }
            // X
            else if (msg->x == 1.0)
            {
                if (nums > 0)
                {
                    motorStateVec_.pop_back();
                }
            }
            // Y 添加新点位
            else if (msg->y == 1.0)
            {
                motorStateVec_.push_back(robotMotor_->getMotorGroupStateByName(group_));
            }
            // L 左摇杆按下
            else if (msg->L == 1.0)
            {
                if (nums == 0)
                {
                    ROS_WARN("waypoint's size must > 0");
                }
                else if (!trajectory_->isMultiRunning())
                {
                    auto state = robotMotor_->getMotorGroupStateByName(group_);
                    auto motorOutputTrajector = trajectory_->generatorTrajectory(state, motorStateVec_, 0.05);
                    setKpKd();
                    trajectory_->setMulti(state, motorOutputTrajector, 0.05, -0.01, group_, utils::TransitionType::CubicSpline, false);
                    trajectory_->setMultiGroup(group_);
                }
            }
            // 十字键上
            else if (msg->dpad_vertical == 1.0)
            {
            }
            // 十字键下
            else if (msg->dpad_vertical == -1.0)
            {
            }
            // 十字键左
            else if (msg->dpad_horizontal == 1.0)
            {
            }
            // 十字键右
            else if (msg->dpad_horizontal == -1.0)
            {
            }
            else
            {
                return;
            }
            ROS_INFO("MotorGroup:[%s]-------->[Y]: add waypoint, [X] delete waypoint, [RT+X] delete all waypoints, [A] add [[%zu]] waypoints to file %s<-------", group_.c_str(), motorStateVec_.size(), (filePath_ + "/" + fileName_).c_str());
        }

        void TeachControllerInterface::execLoop()
        {
            auto pdHz = 1000;
            ros::Rate rate(pdHz);
            ROS_INFO("Teach %s()", __FUNCTION__);
            while (!quit_ && ros::ok())
            {
                std::unique_lock<std::shared_timed_mutex> lk(mutex_);
                if (trajectory_->isMultiRunning())
                {
                    auto group = trajectory_->getMultiGroup();
                    auto motorState = robotMotor_->getMotorGroupStateByName("all");
                    auto out = trajectory_->generatorMulti(motorState);
                    auto index = robotMotor_->getMotorGroupIndexByName(group);
                    // 如果不是全部电机，想让其他电机是硬的需要让电机保持上次命令
                    if (robotMotor_->getMotorGroupSizeByName(group) < robotMotor_->getMotorGroupSizeByName("all"))
                    {
                        robotMotor_->keepMotorGroupLastCmdByName("all");
                    }
                    if (!out.isZero())
                    {
                        // out.target_tau.setConstant(10);
                        common::motorOutputSetKpKd(out, pdInfo_.byMotorId.kp, pdInfo_.byMotorId.kd, index);
                        out.target_dq.setConstant(0);
                        out.target_tau.setConstant(0);
                        robotMotor_->setMotorGroupByName(group, out);
                        ROS_INFO_THROTTLE(3, "%s Multi %s: state:%sout:%s", __FUNCTION__, group.c_str(), robotMotor_->getMotorGroupStateByName(group).toString().c_str(), out.toString().c_str());
                    }
                }
                else
                {
                    robotMotor_->protectMotorGroupByName("all");
                }
                lk.unlock();

                robotMotor_->sendMotors();
                rate.sleep();
            }
        }

        TeachControllerInterface::~TeachControllerInterface()
        {
            ROS_INFO("Teach %s()", __FUNCTION__);
            stop();
        }

    }
}
