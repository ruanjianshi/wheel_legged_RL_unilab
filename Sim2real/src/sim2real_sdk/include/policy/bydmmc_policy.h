#ifndef BYDMMC_FUTURE_POLICY_H_
#define BYDMMC_FUTURE_POLICY_H_

#include "policy/policy_base.h"
#include "impl/hightorque_config.h"

namespace hightorque
{
    /**
     * @brief 基于 BYDMMC 算法的策略实现，支持历史观测缓冲
     */
    class BYDMMCFuturePolicy : public PolicyBase
    {
        public:
            class bydpose
            {
                public:
                    double x;
                    double y;
                    double z;
            };
            class bydquat
            {
                public:
                    double x;
                    double y;
                    double z;
                    double w;
            };
            /**
             * @brief 构造函数：从 YAML 配置初始化参数
             * @param rl RL_YamlInfo 配置结构体
             * @param pd PD_YamlInfo 配置结构体
             */
            BYDMMCFuturePolicy(const config::RLConfig& rl, const config::PDConfig& pd);

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

            /**
             * @brief 状态机进入 RUNNING 时设置初始 yaw
             */
            void onEnterRunning(const Robot_State& robotState) override;

            Robot_Output getFirstRobotOutput() override;

            bool isReady() const override
            {
                return (bodyPoseW_ != nullptr && !bodyPoseW_->empty() &&
                        bodyQuatW_ != nullptr && !bodyQuatW_->empty() &&
                        jointPos_ != nullptr && !jointPos_->empty() &&
                        jointVel_ != nullptr && !jointVel_->empty());
            }

            /**
             * @brief 重置并设置初始 yaw 角度为当前机器人的 yaw 角度
             * @param robotState 当前机器人状态，用于读取当前 yaw 角度
             */
            void resetInitialYaw(const Robot_State& robotState);

            static void loadAllFutureFile(const std::vector<std::string>& filePath);

        private:
            double frequency_;                                              // 更新频率（Hz）
            double robotAngleVelScale_;                                     // 机器人角速度缩放系数
            double cmdVelLineScale_;                                        // 指令线速度缩放系数
            double cmdVelAngleScale_;                                       // 指令角速度缩放系数
            double robotLinePoseScale_;                                     // 机器人线性位置缩放系数
            double robotLineVelScale_;                                      // 机器人线性速度缩放系数
            std::vector<int> policy2ActualMap_;                             // 策略到实际机器人的映射索引
            std::vector<int> actual2PolicyMap_;                             // 实际机器人到策略的映射索引
            std::shared_ptr<std::map<int, std::vector<double>>> futureMap_; // 未来帧

            bool test_ = false; // debug

            // 历史缓冲区参数
            int frameStack_; // 历史帧数
            int totalSteps_; // 未来帧总数
            int futureSize_; // 未来帧大小

            Robot_Output lastRobotOutput_;

            std::vector<double> actionSales_; // 动作缩放
            std::vector<double> defaultPose_; // 默认姿态
            double motionLen_;                // 运动总长度 (从yaml读取)

            unsigned int counter_;        // 控制计数器
            unsigned int bodyQuatWIndex_; // 未来帧四元数索引

            double offsetPitch_;   // 初始 pitch 角度（作为零点参考）
            double offsetRoll_;    // 初始 roll 角度（作为零点参考）
            double offsetYaw_;     // 初始 yaw 角度（作为零点参考）
            bool isInitialYawSet_; // 是否已记录初始 yaw 角度
            std::shared_ptr<std::vector<std::vector<bydpose>>> bodyPoseW_;
            std::shared_ptr<std::vector<std::vector<bydquat>>> bodyQuatW_;
            std::shared_ptr<std::vector<std::vector<double>>> jointPos_;
            std::shared_ptr<std::vector<std::vector<double>>> jointVel_;
            static std::map<std::string, std::shared_ptr<std::vector<std::vector<bydpose>>>> bodyPoseMap_; // 未来帧数据
            static std::map<std::string, std::shared_ptr<std::vector<std::vector<bydquat>>>> bodyQuatMap_; // 未来帧数据
            static std::map<std::string, std::shared_ptr<std::vector<std::vector<double>>>> jointPosMap_;  // 未来帧数据
            static std::map<std::string, std::shared_ptr<std::vector<std::vector<double>>>> jointVelMap_;  // 未来帧数据

            bool loadFutureFile(const std::string& filePath);

            int getObsSize() const override
            {
                return numSingleObs_;
            }
    };
}

#endif
