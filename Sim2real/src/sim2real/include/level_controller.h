#ifndef LEVEL_CONTROLLER_H_
#define LEVEL_CONTROLLER_H_

#include "sim2real.h"

namespace hightorque
{
    /**
     * @brief 基础控制节点：继承 Sim2Real，提供电机配置接口, 用于开发者模式
     */
    class LevelControllerNode : public Sim2Real
    {
        public:
            /** ======================== 构造与析构 ======================== */

            /**
             * @brief 构造函数：使用串口机器人和 ROS 节点句柄初始化
             * @param rbPtr 串口机器人对象指针
             * @param nh    ROS 节点句柄指针
             */
            LevelControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr,
                                std::shared_ptr<ros::NodeHandle> nh)
                : Sim2Real(rbPtr, nh)
            {
                ROS_INFO("%s()", __FUNCTION__); // 日志：调用函数名
            }

            /**
             * @brief 构造函数：使用串口机器人和 ROS 节点句柄初始化
             * @param rbPtr 串口机器人对象指针
             * @param nh    ROS 节点句柄指针
             * @param path  配置文件名称
             */
            LevelControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr,
                                std::shared_ptr<ros::NodeHandle> nh,
                                const std::string& rlConfigFile)
                : Sim2Real(rbPtr, nh, rlConfigFile)
            {
                ROS_INFO("%s()", __FUNCTION__); // 日志：调用函数名
            }

            /**
             * @brief 默认构造
             */
            LevelControllerNode()
                : Sim2Real()
            {
                ROS_INFO("%s()", __FUNCTION__); // 日志：调用函数名
            }

            /**
             * @brief 析构函数：输出日志
             */
            virtual ~LevelControllerNode()
            {
                ROS_INFO("%s()", __FUNCTION__); // 日志：调用函数名
            }

            /** ======================== 电机配置接口 ======================== */

            /**
             * @brief 配置电机组, 在mixlevel模式下 保证lowlevel和highlevel使用同一个电机组
             * @param robotMotor 电机组管理对象共享指针
             */
            virtual void configureMotor(std::shared_ptr<RobotMotorGroup> robotMotor)
            {
                robotMotor_ = robotMotor; // 保存电机组指针
            }
    }; // class LevelControllerNode

};
#endif
