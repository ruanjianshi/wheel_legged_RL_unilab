#ifndef SIM_TO_REAL_H_
#define SIM_TO_REAL_H_
// Pinocchio头文件必须在所有其他头文件之前包含
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include "pinocchio/multibody/fwd.hpp"

#include "motor/pi_motor_group.h"
#include "motor/pi_plus_motor_group.h"
#include "motor/hi_motor_group.h"
#include "robot_data.h"
#include "hardware/robot.h"

#include "sensor_msgs/Imu.h"
#include "sim2real_msg/Joy.h"
#include "sim2real_msg/Orient.h"
#include "std_msgs/Int8.h"

#include "engine/inference_engine.h"

#include "impl/hightorque_config.h"
// #include "read_pd_yaml.h"
// #include "read_rl_yaml.h"
#include "impl/utils.h"
#include "common.h"

#include "policy/dreamwaq_policy.h"
#include "policy/footstep_policy.h"
#include "policy/humanoidgym_policy.h"
#include "policy/pbhc_policy.h"
#include "policy/lr_policy.h"

#include <shared_mutex>
#include <future>

#include "engine/lr_inference_engine.h"
#if defined(PLATFORM_X86_64)
#include "engine/openvino_inference_engine.h"
#endif
#if defined(PLATFORM_RK)
#include "engine/rknn_inference_engine.h"
#endif
#if defined(PLATFORM_JETSON)
#include "engine/tensorrt_inference_engine.h"
#endif

namespace hightorque
{
    namespace sim2Real
    {
        enum FellDown
        {
            NONE = 0,
            FRONT = 1, // 脸朝地 俯卧
            BACK = 2,  // 背朝地 仰卧
            LEFT = 3,  // 左测朝地
            RIGHT = 4  // 右侧朝地
        };
        enum State
        {
            INIT = 0,                // 初始状态
            STANDING = 1,            // 站立中
            STANDBY = 2,             // 站立
            RUNNING = 3,             // 踏步 接收cmdVel指令
            SITTING = 4,             // 蹲下中
            PRE_STANDING = 5,        // 执行起立前的准备动作
            POLICY_STANDING = 6,     // 跌倒状态
            LYING = 7,               // 躺下中
            LIE_DOWN = 8,            // 躺下
            TRAJ_SINGLE = 9,         // 执行单一动作指令
            TRAJ_MULTI = 10,         // 执行多动作指令
            TRAJ_SERIES = 11,        // 执行一系列多动作指令
            PRE_POLICY_CHANGE = 12,  // 执行动作前的准备动作
            POST_POLICY_CHANGE = 13, // 执行动作后的收尾动作
        };

        static std::string toString(State s)
        {
            switch (s)
            {
                case INIT:
                    return "init";
                case STANDING:
                    return "standing";
                case STANDBY:
                    return "standby";
                case RUNNING:
                    return "running";
                case SITTING:
                    return "sitting";
                case PRE_STANDING:
                    return "pre up";
                case POLICY_STANDING:
                    return "uping";
                case LYING:
                    return "lying";
                case LIE_DOWN:
                    return "lie down";
                case TRAJ_SINGLE:
                    return "single trajectory";
                case TRAJ_MULTI:
                    return "multiple trajectory";
                case TRAJ_SERIES:
                    return "series trajectory";
                case PRE_POLICY_CHANGE:
                    return "pre policy change";
                case POST_POLICY_CHANGE:
                    return "post policy change";
                default:
                    return "unknown";
            }
        }
    }

    /**
     * @brief Sim2Real 类：实现基础的站立蹲下行走
     */
    class Sim2Real
    {
        public:
            /** ======================== 构造与析构 ======================== */

            /**
             * @brief 构造函数：初始化机器人串口和 ROS 节点句柄
             * @param rbPtr 串口机器人对象指针
             * @param nh    ROS 节点句柄指针
             */
            Sim2Real(std::shared_ptr<livelybot_serial::robot> rbPtr,
                     std::shared_ptr<ros::NodeHandle> nh);

            /**
             * @brief 构造函数：初始化机器人串口和 ROS 节点句柄 使用非默认配置文件
             * @param rbPtr     串口机器人对象指针
             * @param nh        ROS 节点句柄指针
             * @param path      配置文件名称
             */
            Sim2Real(std::shared_ptr<livelybot_serial::robot> rbPtr,
                     std::shared_ptr<ros::NodeHandle> nh,
                     const std::string& path);

            /**
             * @brief 默认构造（仅供派生或测试用）
             */
            Sim2Real() = default;

            /**
             * @brief 析构函数：释放资源，终止执行线程
             */
            virtual ~Sim2Real();

            /** ======================== 公有接口 ======================== */

            /**
             * @brief 检查是否已请求退出 用于状态机
             * @return true 表示应退出主循环
             */
            virtual bool get_is_shutdowm()
            {
                return quit_ || this->isPowerProtect(); // 退出标志
            }

            /**
             * @brief 获取参数 用于状态机
             * @return 返回值暂未使用，默认 0
             */
            virtual bool get_parameters()
            {
                return 0;
            }

            /**
             * @brief 初始化所有模块（调用各子初始化函数）
             * @return true 表示成功
             */
            virtual bool init();

            /**
             * @brief 初始化 Pinocchio 动力学模型
             * @return true 表示成功
             */
            virtual bool initPinocchio();

            /**
             * @brief 初始化 ROS 订阅与发布
             * @return true 表示成功
             */
            virtual bool initRos();

            /**
             * @brief 初始化轨迹生成器
             * @return true 表示成功
             */
            virtual bool initTrajectory();

            /**
             * @brief 初始化电机相关设置
             * @return true 表示成功
             */
            virtual bool initMotor();

            /**
             * @brief 初始化推理引擎
             * @return true 表示成功
             */
            virtual bool initEngine();

            /**
             * @brief 初始化策略模块
             * @return true 表示成功
             */
            virtual bool initPolicy(std::shared_ptr<PolicyBase>& policy, const config::PDConfig& pd, const config::RLConfig& rl);

            /**
             * @brief 初始化观测数据结构
             * @return true 表示成功
             */
            virtual bool initObs();

            /**
             * @brief 执行主循环：读取传感器、计算控制、发送指令、发布状态
             */
            virtual void execPdLoop();

            /**
             * @brief 执行主循环：读取传感器、计算控制、发送指令、发布状态
             */
            virtual void execRlLoop();

            virtual void execTrajLoop();

            /**
             * @brief 遥控输入回调
             * @param msg 控制状态消息指针
             */
            virtual void joyCallback(const sim2real_msg::Joy::ConstPtr& msg);

            /**
             * @brief IMU 数据回调
             * @param msg IMU 消息指针
             */
            virtual void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);

            /**
             * @brief 速度指令回调
             * @param msg Twist 消息指针
             */
            virtual void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg);

            /**
             * @brief 设置电机 PID 参数（Kp/Kd） 用于执行轨迹动作时 设置电机的kpkd
             *        同时起着设置当前状态，配合keepMotorGroupLastCmdByName能保持机器人不动
             */
            virtual void setKpKd(const std::string& group = "all", MotorControlType type = MotorControlType::POS_VEL_MAX_TQE) const;

            /**
             * @brief 启动控制流程
             * @return true 表示成功启动
             */
            virtual bool start();

            /**
             * @brief 停止控制流程
             * @return true 表示已停止
             */
            virtual bool stop();

            /**
             * @brief 发布当前机器人与电机状态到 ROS 话题
             */
            virtual void publish() const;

            /**
             * @brief 保持指定的电机当前的位置
             * @param group 电机组名称
             */
            virtual void keepMotor(const std::string& group);

            /**
             * @brief 检测是否跌倒
             * @return FellDown 枚举跌倒状态
             */
            virtual sim2Real::FellDown isFellDown();

            /**
             * @brief 切换policy算法
             * @param policyName 指定的策略名称，空字符串表示不切换名称
             * @param policyType 指定的策略类型，空字符串表示不切换类型
             * @param next 是否切换到下一个策略，默认为 true
             * @return true 表示重置成功
             */
            virtual bool changePolicy(const std::string& policyName = "", const std::string& policyType = "", bool next = true);

            /**
             * @brief 获取内部 RobotMotorGroup 对象
             * @return RobotMotorGroup 共享指针
             */
            const auto& getRobotMotor()
            {
                return robotMotor_;
            }

            /**
             * @brief 接收并执行所有关节命令
             * @param msg  关节状态消息指针
             * @param type 命令类型，默认为 "absolute"(绝对位置) "relative"(相对位置)
             */
            void robotJointCommandCallback(const sensor_msgs::JointState::ConstPtr& msg,
                                           const std::string& type = "absolute");

            /**
             * @brief 判断动作是否做完，且对动作完成后进行判断是否进入过度动作
             */
            virtual bool isMostionFinished();

            /**
             * @brief 上半身回正动作,用于无需切换的下半身走路策略
             * @param group 针对的电机组，默认上半身
             */
            virtual void motorGroupRecover(const std::string& group = "upperBody", const double duration = 1.0);

            /**
             * @brief 对即将进行切换的算法进行single trajectory过度
             * @param policy 准备切换的策略
             * @param group 针对的电机组，一般来说 PRE_POLICY_CHANGE 是上半身，防止抖动
             *                                      POST_POLICY_CHANGE 是全身，手臂回正以及下蹲
             */
            virtual bool motorGroupRecoverByPolicyConfig(const std::string& policy, const std::string& group, double duration = 0.5);

            /*
             * @brief 功率是否过高
             */
            virtual bool isPowerProtect()
            {
                double power = I_ * U_;
                if (power > pdInfo_.max_power)
                {
                    auto now = std::chrono::system_clock::now();
                    if (now - lastPowerNormalTime_ > std::chrono::duration<double>(pdInfo_.max_power_duration))
                    {
                        powerProtect_ = true;
                        ROS_WARN_THROTTLE(0.2, "Power protect triggered! Current power: %.2f W, Max power: %.2f W", power, pdInfo_.max_power);
                    }
                }
                else
                {
                    lastPowerNormalTime_ = std::chrono::system_clock::now();
                    powerProtect_ = false;
                }
                ROS_INFO_THROTTLE(10, "Power  %.2f W, Max power: %.2f W", power, pdInfo_.max_power);
                return powerProtect_;
            }

        protected:
            /** ======================== 保护成员变量 ======================== */
            std::string testString_ = "";         // 测试字符串
            std::string productionType_ = "test"; //
            int testInt_ = 0;                     // 测试整数

            std::string beforeFellPolicyName_ = "";       // 跌倒前的策略名称
            std::string beforeFellPolicyType_ = "";       // 跌倒前的策略类型
            std::string savedFile_ = "";                  // 测试字符串
            std::string modelType_;                       // 机器人模型类型 pi pi_plus
            std::string policyChangeDefautePolicy = "";        // 策略执行完后回到的默认策略
            sim2Real::State state_;                       // 全局状态
            std::shared_ptr<RobotMotorGroup> robotMotor_; // 电机组管理对象
            bool quit_ = false;                           // 退出标志
            bool powerProtect_ = false;                   // 退出标志
            bool customMode_;                             // custom模式
            bool useFellDownAction_ = false;              // 是否使用跌倒爬起动作
            bool useFellDownPolicy_ = false;              // 是否使用 policy的跌倒爬起动作

            double I_ = 0.0, U_ = 0.0; // 电流与电压

            std::shared_timed_mutex imuMutex_;    // IMU 数据互斥锁
            std::shared_timed_mutex stateMutex_;  // 状态数据互斥锁
            std::shared_timed_mutex outputMutex_; // 输出数据互斥锁

            sensor_msgs::Imu imu_; // 最新 IMU 数据

            std::shared_ptr<livelybot_serial::robot> rbPtr_;                   // 串口机器人对象
            std::shared_ptr<PolicyBase> policy_;                               // 策略模块
            std::shared_ptr<InferenceEngine> engine_;                          // 推理引擎
            utils::TrajectoryGenerator* trajectory_ = nullptr;                 // 轨迹生成器（单例模式非拥有）
            std::shared_ptr<config::PolicyChangeConfig> policyChangeByKeyPtr_; // 策略切换配置指针（弱引用）

            std::shared_ptr<std::thread> pdThreadExecPtr_ = nullptr;   // 执行线程指针
            std::shared_ptr<std::thread> rlThreadExecPtr_ = nullptr;   // 执行线程指针
            std::shared_ptr<std::thread> trajThreadExecPtr_ = nullptr; // 执行轨迹

            std::map<std::string, std::shared_ptr<InferenceEngine>> engineMap_;                            // 推理引擎映射
            std::chrono::system_clock::time_point lastPowerNormalTime_ = std::chrono::system_clock::now(); // 上次功率正常时间

            // Pinocchio 动力学
            pinocchio::Model pinocchioModel_; // 动力学模型
            pinocchio::Data pinocchioData_;   // 动力学数据

            config::PDConfig pdInfo_;             // PID 配置
            config::RLConfigGroup rlConfigGroup_; // 强化学习配置组
            config::RLConfig rlInfo_;             // 强化学习配置

            Robot_State robotState_;      // 当前机器人关节状态
            Robot_Output robotOutput_;    // 当前机器人关节输出
            Motor_Output motorOutput_;    // 电机输出
            DynamicOffset dynamicOffset_; // 动态偏移补偿
            Observations obs;             // 观测数据存储

            // ROS 订阅与发布
            ros::Subscriber jointAbsoluteSub_;    // 订阅：所有关节绝对命令
            ros::Subscriber jointRelativeSub_;    // 订阅：所有关节相对命令
            ros::Subscriber imuSub_;              // IMU 话题订阅
            ros::Subscriber joySub_;              // 手柄话题订阅
            ros::Subscriber cmdVelSub_;           // 速度指令订阅
            ros::Subscriber testStringSub_;       // 测试
            ros::Subscriber testIntSub_;          // 测试
            ros::Subscriber iSub_;                // 电流话题订阅
            ros::Subscriber uSub_;                // 电压话题订阅
            std::shared_ptr<ros::NodeHandle> nh_; // ROS 节点句柄
            ros::Publisher robotStatePub_;        // 发布机器人状态
            ros::Publisher motorStatePub_;        // 发布电机状态
            ros::Publisher motorOutputPub_;       // 发布电机指令

            Command cmdVel_; // 缓存的速度指令
            Joy joy_;
            double kickDuration_ = 0.2; // 踢腿动作执行时间
    };

};
#endif
