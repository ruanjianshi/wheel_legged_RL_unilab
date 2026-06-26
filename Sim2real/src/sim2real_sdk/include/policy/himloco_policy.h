#ifndef HIMLOCO_POLICY_H_
#define HIMLOCO_POLICY_H_

#include "policy/policy_base.h"
#include "impl/hightorque_config.h"

namespace hightorque
{
    /**
     * @brief HIMloco (HIMActorCritic) 四足机器人策略实现
     * 
     * 对应训练配置 HTDW4438Cfg，使用 HIMActorCritic 算法。
     * 观测顺序: [cmd(3), ang_vel(3), gravity(3), dof_pos_err(12), dof_vel(12), last_action(12)] = 45维
     * 帧堆叠: 6 帧 → 最终输入 270 维
     * 控制频率: 50Hz
     */
    class HIMlocoPolicy : public PolicyBase
    {
        public:
            /**
             * @brief 构造函数：从 YAML 配置初始化参数
             * @param rl RL 配置结构体
             * @param pd PD 配置结构体
             */
            HIMlocoPolicy(const config::RLConfig& rl, const config::PDConfig& pd);

            /**
             * @brief 更新观测值并构建最终输入
             * @param timeDiff     与上次调用的时间差（秒）
             * @param robotState   当前机器人状态
             * @param robotOutput  当前机器人输出
             * @param cmdVel       当前控制指令
             * @param obs          输出观测向量
             * @param standby      是否处于待机模式
             * @return             true 表示观测已更新
             */
            bool updateObservation(const double timeDiff,
                                   const Robot_State& robotState,
                                   const Robot_Output& robotOutput,
                                   const Command& cmdVel,
                                   Eigen::VectorXd& obs,
                                   bool standby = false) override;

        private:
            /// 策略自由度（12 for HTDW_4438）
            int rlDofs_;

            /// 实际硬件自由度
            int pdDofs_;

            /// 策略到实际硬件的关节映射
            std::vector<int> policy2ActualMap_;

            /// 实际硬件到策略的关节映射
            std::vector<int> actual2PolicyMap_;

            /// 默认站立姿态关节角
            Eigen::VectorXd defaultAngles_;

            /// 控制频率 (Hz)
            double frequency_;

            /// 指令线速度缩放
            double cmdLinVelScale_;

            /// 指令角速度缩放
            double cmdAngVelScale_;

            /// 机器人关节位置缩放
            double rbtLinPosScale_;

            /// 机器人关节速度缩放
            double rbtLinVelScale_;

            /// 机器人角速度缩放
            double rbtAngVelScale_;
    };

} // namespace hightorque

#endif // HIMLOCO_POLICY_H_
