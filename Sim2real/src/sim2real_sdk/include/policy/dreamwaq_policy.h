#ifndef DREAMWAQ_POLICY_H_
#define DREAMWAQ_POLICY_H_

#include "impl/hightorque_config.h"
#include "policy/policy_base.h"
namespace hightorque
{
    /**
     * @brief 基于 DreaWaQ 算法的策略实现
     */
    class DreaWaQPolicy : public PolicyBase
    {
        public:
            /**
             * @brief 构造函数：从 YAML 配置初始化参数
             * @param info RL_YamlInfo 配置结构体
             */
            DreaWaQPolicy(const config::RLConfig rl, const config::PDConfig pd);

            /**
             * @brief 构造函数：手动传入各类缩放和参数
             * @param frequency            更新频率（Hz）
             * @param robotAngleVelScale   机器人角速度缩放系数
             * @param cmdVelLineScale      指令线速度缩放系数
             * @param cmdVelAngleScale     指令角速度缩放系数
             * @param robotLinePoseScale   机器人线性位置缩放系数
             * @param robotLineVelScale    机器人线性速度缩放系数
             * @param clipObs              观测裁剪阈值
             * @param numSingleObs         单次观测长度
             */
            DreaWaQPolicy(double frequency,
                          double robotAngleVelScale,
                          double cmdVelLineScale,
                          double cmdVelAngleScale,
                          double robotLinePoseScale,
                          double robotLineVelScale,
                          double clipObs,
                          int numSingleObs);

            /**
             * @brief 更新观测值，实现 DreaWaQ 自定义观测逻辑
             * @param timeDiff     与上次调用的时间差（秒）
             * @param robotState   当前机器人状态
             * @param robotOutput  当前机器人输出
             * @param cmdVel       当前控制指令
             * @param obs          输出观测向量
             * @param standby      是否处于待机模式（默认 false）
             * @return             true 表示观测已更新
             */
            bool updateObservation(const double timeDiff,
                                   const Robot_State& robotState,
                                   const Robot_Output& robotOutput,
                                   const Command& cmdVel,
                                   Eigen::VectorXd& obs,
                                   bool standby = false) override;

        private:
            double frequency_;          // 更新频率（Hz）
            double robotAngleVelScale_; // 机器人角速度缩放系数
            double cmdVelLineScale_;    // 指令线速度缩放系数
            double cmdVelAngleScale_;   // 指令角速度缩放系数
            double robotLinePoseScale_; // 机器人线性位置缩放系数
            double robotLineVelScale_;  // 机器人线性速度缩放系数
            
            // 速度限制参数
            double cmd_vel_x_min_, cmd_vel_x_max_;
            double cmd_vel_y_min_, cmd_vel_y_max_;
            double cmd_vel_yaw_min_, cmd_vel_yaw_max_;
    };
}

#endif
