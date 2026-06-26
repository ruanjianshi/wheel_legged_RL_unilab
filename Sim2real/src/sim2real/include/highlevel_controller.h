#ifndef HIGHLEVEL_CONTROLLER_H_
#define HIGHLEVEL_CONTROLLER_H_

#include "level_controller.h"
#include "sim2real_msg/Common.h"

namespace hightorque
{
    /**
     * @brief 高级控制节点：在基础 LevelControllerNode 上实现更高级别行为
     * 实现站立蹲下，行走，执行示教动作（轨迹）
     */
    class HighlevelControllerNode : public LevelControllerNode
    {
        public:
            /** ======================== 构造与析构 ======================== */

            /**
             * @brief 构造函数：初始化串口机器人和 ROS 节点
             * @param rbPtr 串口机器人对象指针
             * @param nh    ROS 节点句柄指针
             */
            HighlevelControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr,
                                    std::shared_ptr<ros::NodeHandle> nh);

            /**
             * @brief 构造函数：初始化串口机器人和 ROS 节点
             * @param rbPtr 串口机器人对象指针
             * @param nh    ROS 节点句柄指针
             * @param path  配置文件名称
             */
            HighlevelControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr,
                                    std::shared_ptr<ros::NodeHandle> nh,
                                    const std::string& path);

            /**
             * @brief 析构函数：清理资源
             */
            ~HighlevelControllerNode();

            /** ======================== ROS 回调与服务 ======================== */

            /**
             * @brief 动作命令回调函数
             * @param msg std_msgs::Int8 类型的动作指令消息指针
             */
            void actionCallback(const std_msgs::Int8::ConstPtr& msg);

            /**
             * @brief 状态切换服务处理函数
             * @param req 服务请求，包含 Common::Request 字段
             * @param res 服务响应，需设置 Common::Response 字段
             * @return true 表示处理成功
             */
            bool changeStateHandle(sim2real_msg::Common::Request& req,
                                   sim2real_msg::Common::Response& res);

            /** ======================== 基类接口重写 ======================== */

            /**
             * @brief 配置电机组接口实现（从 LevelControllerNode 继承）
             * @param robotMotor 电机组管理对象共享指针
             */
            void configureMotor(std::shared_ptr<RobotMotorGroup> robotMotor) override;

            /**
             * @brief 初始化 ROS 订阅与发布（重写基类方法）
             * @return true 表示初始化成功
             */
            bool initRos() override;

            inline void setIsMix(bool isMix)
            {
                isMix_ = isMix;
            }

            inline void setUseJoy(bool useJoy)
            {
                useJoy_ = useJoy;
            }

        private:
            /** ======================== 私有成员变量 ======================== */

            ros::ServiceServer changeStateService_; // 状态切换服务服务器
            ros::Subscriber actionCallbackSub_;     // 动作命令订阅者
            bool isMix_ = false;
            bool useJoy_ = false;
    };

};
#endif
