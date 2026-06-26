#ifndef PBHC_POLICY_H_
#define PBHC_POLICY_H_

#include "policy/policy_base.h"
#include "impl/hightorque_config.h"
#include <deque>

namespace hightorque
{
    /**
     * @brief 基于 PBHC 算法的策略实现，支持历史观测缓冲
     */
    class PBHCPolicy : public PolicyBase
    {
        public:
            /**
             * @brief 构造函数：从 YAML 配置初始化参数
             * @param rl RL_YamlInfo 配置结构体
             * @param pd PD_YamlInfo 配置结构体
             */
            PBHCPolicy(const config::RLConfig& rl, const config::PDConfig& pd);

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
            PBHCPolicy(double frequency,
                       double robotAngleVelScale,
                       double cmdVelLineScale,
                       double cmdVelAngleScale,
                       double robotLinePoseScale,
                       double robotLineVelScale,
                       double clipObs,
                       int numSingleObs);

            /**
             * @brief 更新观测值并构建最终输入，参考sim2sim_pi20dof.py的方式
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
             * @brief 初始化历史缓冲区
             */
            void initHistoryBuffer();

            /**
             * @brief 获取最终构建的观测输入，供推理引擎使用
             * @return 最终的观测输入矩阵
             */
            const Eigen::MatrixXd& getFinalObsInput() override;

            /**
             * @brief 修改输出，修改policy输出顺序为actual输出顺序
             * @param output 机器人输出对象
             */
            void postModifyOutput(Robot_Output& output) override;

            /**
             * @brief 判断当前动作是否结束
             * @return true 表示动作已结束，false 表示动作仍在进行中
             */
            bool isMotionFinished() const override;

            void refreshObservation() override;

        private:
            double frequency_;                  // 更新频率（Hz）
            double robotAngleVelScale_;         // 机器人角速度缩放系数
            double cmdVelLineScale_;            // 指令线速度缩放系数
            double cmdVelAngleScale_;           // 指令角速度缩放系数
            double robotLinePoseScale_;         // 机器人线性位置缩放系数
            double robotLineVelScale_;          // 机器人线性速度缩放系数
            std::vector<int> policy2ActualMap_; // 策略到实际机器人的映射索引
            std::vector<int> actual2PolicyMap_; // 实际机器人到策略的映射索引

            bool test_ = false; // debug

            // 历史缓冲区参数
            int frameStack_; // 历史帧数

            // 默认关节角度 (与sim2sim_pi20dof.py保持一致)
            Eigen::VectorXd defaultAngles_;

            // 运动时间和长度 (与sim2sim_pi20dof.py保持一致)
            double currentMotionLen_; // 运动累计时间
            double motionLen_;        // 运动总长度 (从yaml读取)

            // 控制时间步长和计数器 (与sim2sim_pi20dof.py保持一致)
            double control_dt_;    // 控制时间步长 (dt * decimation)
            unsigned int counter_; // 控制计数器

            // 历史缓冲区，参考sim2sim_pi20dof.py
            std::deque<Eigen::VectorXd> histActions_;       // actions历史
            std::deque<Eigen::VectorXd> histBaseAngVel_;    // base_ang_vel历史
            std::deque<Eigen::VectorXd> histDofPos_;        // dof_pos历史
            std::deque<Eigen::VectorXd> histDofVel_;        // dof_vel历史
            std::deque<Eigen::VectorXd> histProjectedGrav_; // projected_gravity历史
            std::deque<Eigen::VectorXd> histRefPhase_;      // ref_motion_phase历史

            /**
             * @brief 构建最终观测输入，参考sim2sim_pi20dof.py的结构
             * @param currentObs 当前观测向量
             */
            void buildFinalObsInput(const Eigen::VectorXd& currentObs);

            /**
             * @brief 更新历史缓冲区
             * @param actions 当前动作
             * @param baseAngVel 当前基座角速度
             * @param dofPos 当前关节位置
             * @param dofVel 当前关节速度
             * @param projectedGrav 当前投影重力
             * @param refPhase 当前运动相位
             */
            void updateHistoryBuffer(const Eigen::VectorXd& actions,
                                     const Eigen::VectorXd& baseAngVel,
                                     const Eigen::VectorXd& dofPos,
                                     const Eigen::VectorXd& dofVel,
                                     const Eigen::VectorXd& projectedGrav,
                                     const Eigen::VectorXd& refPhase);
    };
}

#endif
