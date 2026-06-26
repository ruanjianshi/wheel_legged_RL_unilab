#ifndef HOST_POLICY_H_
#define HOST_POLICY_H_
#include "impl/hightorque_config.h"
#include "policy/policy_base.h"

namespace hightorque
{
    /**
     * @brief 针对步态周期的策略实现
     */
    class HostPolicy : public PolicyBase
    {
        public:
            /**
             * @brief 构造函数：从 YAML 配置初始化步数和缩放参数
             * @param info           RL_YamlInfo 配置结构体
             */
            HostPolicy(const config::RLConfig rl, const config::PDConfig pd);

            /**
             * @brief 构造函数：手动传入步数和各类缩放参数
             * @param stepsPerCycle      每个周期的步数
             * @param robotAngleVelScale 机器人角速度缩放系数
             * @param cmdVelLineScale    指令线速度缩放系数
             * @param cmdVelAngleScale   指令角速度缩放系数
             * @param robotLinePoseScale 机器人线性位置缩放系数
             * @param robotLineVelScale  机器人线性速度缩放系数
             * @param clipObs            观测裁剪阈值
             * @param numSingleObs       单次观测长度
             */
            HostPolicy(const int stepsPerCycle,
                       double robotAngleVelScale,
                       double cmdVelLineScale,
                       double cmdVelAngleScale,
                       double robotLinePoseScale,
                       double robotLineVelScale,
                       double clipObs,
                       int numSingleObs);

            /**
             * @brief 更新观测值，实现步态周期逻辑
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

            /**
             * @brief 前处理 将q = q*0.25+action
             * @param output 机器人输出对象
             */
            virtual void preModifyOutput(Robot_Output& output) override;

        private:
            Robot_State curentState_;  // 机器人当前状态
            double robotAngleVelScale_; // 机器人角速度缩放系数
            double cmdVelLineScale_;    // 指令线速度缩放系数
            double cmdVelAngleScale_;   // 指令角速度缩放系数
            double robotLinePoseScale_; // 机器人线性位置缩放系数
            double robotLineVelScale_;  // 机器人线性速度缩放系数
            std::vector<double> lower_, upper_;
    };
}

#endif
