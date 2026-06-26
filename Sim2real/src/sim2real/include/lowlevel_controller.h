#ifndef LOWLEVEL_CONTROLLER_H_
#define LOWLEVEL_CONTROLLER_H_

#include "level_controller.h"
#include "sim2real_msg/lowlevel_controllerAction.h"
#include <actionlib/server/simple_action_server.h>

namespace hightorque
{
    using lowActionServer = actionlib::SimpleActionServer<sim2real_msg::lowlevel_controllerAction>;
    /**
     * @brief 低级控制节点：继承 LevelControllerNode，实现关节命令的底层处理
     * 可对单一关节、多关节的关节组进行单独控制
     */
    class LowlevelControllerNode : public LevelControllerNode
    {
        public:
            /** ======================== 构造与析构 ======================== */

            /**
             * @brief 构造函数：初始化串口机器人和 ROS 节点
             * @param rbPtr 串口机器人对象指针
             * @param nh    ROS 节点句柄指针
             */
            LowlevelControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr,
                                   std::shared_ptr<ros::NodeHandle> nh);

            /**
             * @brief 析构函数：清理资源
             */
            ~LowlevelControllerNode();

            /** ======================== ROS 话题回调 ======================== */

            /**
             * @brief 接收并执行单个或多个关节命令
             * @param msg  关节状态消息指针
             * @param type 命令类型，默认为 "absolute"(绝对位置) "relative"(相对位置)
             */
            void robotJointCommandCallback(const sensor_msgs::JointState::ConstPtr& msg,
                                           const std::string& type = "absolute");

            /**
             * @brief 接收并执行所有关节命令（仅处理位置数据）
             * @param msg  关节状态消息指针
             * @param type 命令类型，默认为 "absolute"(绝对位置) "relative"(相对位置)
             */
            void robotAllJointCommandCallback(const sensor_msgs::JointState::ConstPtr& msg,
                                              const std::string& type = "absolute");

            /**
             * @brief 接收并执行预设姿态命令（zero/sitdown等）
             * @param msg  关节状态消息指针，使用frame_id指定姿态，stamp指定过渡时间
             * @param type 命令类型，默认为 "absolute"(绝对位置)
             */
            void robotPresetCommandCallback(const sensor_msgs::JointState::ConstPtr& msg,
                                           const std::string& type = "absolute");

            /**
             * @brief 接收并执行指定电机组的关节命令
             * @param msg   关节状态消息指针
             * @param block 关节组组名称
             * @param type  命令类型，默认为 "absolute"(绝对位置) "relative"(相对位置)
             */
            void robotGroupJointCommandCallback(const sensor_msgs::JointState::ConstPtr& msg,
                                                const std::string& block,
                                                const std::string& type = "absolute");

            /**
             * @brief 遥控输入回调（覆盖基类，实现底层控制）
             * @param msg 控制状态消息指针
             */
            void joyCallback(const sim2real_msg::Joy::ConstPtr& msg) override;

            /** ======================== 主循环与初始化 ======================== */

            /**
             * @brief 执行主循环：读取命令并发送电机指令
             */
            void execPdLoop() override;

            /**
             * @brief 初始化所有模块（覆盖基类）
             * @return true 表示初始化成功
             */
            bool init() override;

            /**
             * @brief 初始化 ROS 订阅与发布（覆盖基类）
             * @return true 表示初始化成功
             */
            bool initRos() override;

            /**
             * @brief 启动控制流程（覆盖基类）
             * @return true 表示启动成功
             */
            bool start() override;

            /** ======================== Action Server 回调 ======================== */

            /**
             * @brief 执行 Action 服务回调
             * @param goal 低级控制目标指针
             */
            void execActionCallback(const sim2real_msg::lowlevel_controllerGoalConstPtr& goal);

            inline void setIsMix(bool isMix)
            {
                isMix_ = isMix;
            }

        private:
            /** ======================== 私有成员变量 ======================== */

            ros::Subscriber allJointSub_;                // 订阅：所有关节命令（位置数据）
            ros::Subscriber presetCommandSub_;           // 订阅：预设姿态命令（zero/sitdown等）
            ros::Subscriber leftAnkleJointSub_;          // 订阅：左踝关节命令
            ros::Subscriber rightAnkleJointSub_;         // 订阅：右踝关节命令
            ros::Subscriber ankleJointSub_;              // 订阅：踝关节命令
            ros::Subscriber hipJointSub_;                // 订阅：髋关节命令
            ros::Subscriber thighJointSub_;              // 订阅：大腿关节命令
            ros::Subscriber armJointSub_;                // 订阅：手臂关节命令
            ros::Subscriber allJointRelativeSub_;        // 订阅：所有关节相对命令
            ros::Subscriber leftAnkleJointRelativeSub_;  // 订阅：左踝相对命令
            ros::Subscriber rightAnkleJointRelativeSub_; // 订阅：右踝相对命令
            ros::Subscriber ankleJointRelativeSub_;      // 订阅：踝关节相对命令
            ros::Subscriber hipJointRelativeSub_;        // 订阅：髋关节相对命令
            ros::Subscriber thighJointRelativeSub_;      // 订阅：大腿相对命令
            ros::Subscriber armJointRelativeSub_;        // 订阅：手臂相对命令
            ros::Subscriber jointSub_;                   // 订阅：通过关节名命令
            ros::Subscriber jointRelativeSub_;           // 订阅：通过关节名相对命令

            bool isMix_ = false;           // 是否为混合控制模式
            lowActionServer actionServer_; // Action Server：低级控制

            std::shared_timed_mutex mutex_; // 数据互斥锁
            
            Motor_Output pendingAllMotorOutput_;
    };

};
#endif

