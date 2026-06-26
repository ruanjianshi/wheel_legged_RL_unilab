#ifndef AMP_POLICY_H_
#define AMP_POLICY_H_

#include "impl/hightorque_config.h"
#include "policy/policy_base.h"
#include <vector>
#include <Eigen/Core>

namespace hightorque
{
    class AMPPolicy : public PolicyBase
    {
        public:
            /**
             * @brief 从配置初始化（修改：同时接受 RLConfig 和 PDConfig）
             * @param rlInfo RL配置结构体
             * @param pdInfo PD配置结构体
             */
            explicit AMPPolicy(const config::RLConfig& rlInfo, const config::PDConfig& pdInfo);

            bool updateObservation(double timeDiff,
                                   const Robot_State& robotState,
                                   const Robot_Output& robotOutput,
                                   const Command& cmdVel,
                                   Eigen::VectorXd& obs,
                                   bool standby = false) override;

            /**
             * @brief 后处理输出（关键修改：添加缩放和映射转换）
             * @param output 输入为策略输出，输出为实际电机指令
             */
            void postModifyOutput(Robot_Output& output) override;

            void refreshObservation() override
            {
                obs_ = Eigen::VectorXd::Zero(numSingleObs_); // 初始化当前观测向量
                histObs_.clear();
                for (int i = 0; i < frameStack_; ++i)
                {
                    histObs_.emplace_back(Eigen::VectorXd::Zero(numSingleObs_)); // 初始化历史观测缓冲区
                }
                finalObsInput_ = Eigen::MatrixXd::Zero(1, numSingleObs_ * frameStack_); // 初始化最终观测输入矩阵
                lastRobotOutput_.resize(rlDofs_);
            }

        private:
            void initializeMotorMappings();

            // 缩放系数
            double frequency_;
            double robotAngleVelScale_;
            double cmdVelLineScale_;
            double cmdVelAngleScale_;
            double robotLinePoseScale_;
            double robotLineVelScale_;

            std::string mode_; // walk or normal

            // 映射关系
            std::vector<int> actual2PolicyMap_;
            std::vector<int> policy2ActualMap_;

            // 新增关键成员
            std::vector<double> actionSales_; // 动作缩放
            std::vector<double> defaultPose_; // 默认姿态
            Robot_Output lastRobotOutput_;    // 上次输出（策略空间）
            Command lastCmdVel_;              // 上次速度

            // 控制模式相关
            std::string controlMode_;            // 控制模式: Soccer, Fight
            double cmdVelFilterScaleSoccer_;     // Soccer模式命令速度滤波系数
            double cmdVelFilterScaleFight_;      // Fight模式命令速度滤波系数
            double cmdVelXMaxSoccerWalk_;        // Soccer模式走路最大x速度（不按LT）
            double cmdVelXMaxSoccerNormal_;      // Soccer模式正常最大x速度（不按LT）
            double cmdVelXMaxSoccerBoost_;       // Soccer模式LT加速最大x速度
            double cmdVelXMaxFight_;             // Fight模式最大x速度（固定，不响应LT）
            double cmdVelXMinWalk_;              // x速度最小值
            double cmdVelXMinNormal_;            // x速度最小值
            double cmdVelYMin_, cmdVelYMax_;     // y速度范围（所有模式通用）
            double cmdVelYawMin_, cmdVelYawMax_; // yaw速度范围
    };
}

#endif
