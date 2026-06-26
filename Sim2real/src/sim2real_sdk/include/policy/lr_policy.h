#ifndef LR_POLICY_H_
#define LR_POLICY_H_
#include "impl/hightorque_config.h"
#include "policy/policy_base.h"

namespace hightorque
{
    /**
     * @brief 针对步态周期的策略实现
     */
    class LrPolicy : public PolicyBase
    {
        public:
            /**
             * @brief 构造函数：从 YAML 配置初始化参数
             * @param rl RLConfig 配置结构体
             * @param pd PDConfig 配置结构体
             */
            LrPolicy(const config::RLConfig rl, const config::PDConfig pd);

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
            std::vector<int> policy2ActualMap_; // 策略到实际机器人的映射索引
            std::vector<int> actual2PolicyMap_; // 实际机器人到策略的映射索引
            
            // 速度限制参数
            double cmd_vel_x_min_, cmd_vel_x_max_;
            double cmd_vel_y_min_, cmd_vel_y_max_;
            double cmd_vel_yaw_min_, cmd_vel_yaw_max_;
    };
}

#endif
