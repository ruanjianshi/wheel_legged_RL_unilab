#ifndef FOOTSTEP_POLICY_JZ_H_
#define FOOTSTEP_POLICY_JZ_H_
#include "impl/hightorque_config.h"
#include "policy/policy_base.h"

namespace hightorque
{
    /**
     * @brief 针对步态周期的策略实现
     */
    class FootStepPolicyJz : public PolicyBase
    {
        public:
            /**
             * @brief 构造函数：从 YAML 配置初始化步数和缩放参数
             * @param info           RL_YamlInfo 配置结构体
             */
            FootStepPolicyJz(const config::RLConfig rl, const config::PDConfig pd);

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
            FootStepPolicyJz(const int stepsPerCycle,
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
             * @brief 修改输出，修改policy输出顺序为actual输出顺序
             * @param output 机器人输出对象
             */
            void postModifyOutput(Robot_Output& output) override;

        private:
            int stepsPeriod_;                 // 每个周期的步数
            double step_ = 0.0;               // 当前步数计数
            double robotAngleVelScale_;       // 机器人角速度缩放系数
            double cmdVelLineScale_;          // 指令线速度缩放系数
            double cmdVelAngleScale_;         // 指令角速度缩放系数
            double robotLinePoseScale_;       // 机器人线性位置缩放系数
            double robotLineVelScale_;        // 机器人线性速度缩放系数
            Robot_Output lastOutput_;         // 上次输出
            std::vector<double> actionSales_; // 动作缩放

            bool isMoving_ = false;                            // 是否处于运动状态
            bool lastGaitMode_ = false;                        // 是否处于运动状态
            bool completingCycle_ = false;                     // 是否正在完成周期
            std::vector<int> policy2ActualMap_;                // 策略到实际机器人的映射索引
            std::vector<int> actual2PolicyMap_;                // 实际机器人到策略的映射索引
            double lastPitch_ = 0.0;                           // 上次俯仰角
            double lastRoll_ = 0.0;                            // 上次滚转角
            static constexpr double ATTITUDE_THRESHOLD = 0.25; // 姿态偏转阈值(弧度)
    };
}

#endif
