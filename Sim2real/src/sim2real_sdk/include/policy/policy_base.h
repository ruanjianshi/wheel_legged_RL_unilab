#ifndef POLICY_BASE_H_
#define POLICY_BASE_H_

#include "robot_data.h"
#include </usr/include/eigen3/Eigen/Dense>
namespace hightorque
{
    class PolicyBase
    {
        public:
            /** ======================== 公有接口 ======================== */

            /**
             * @brief 默认虚析构函数，确保派生类正确析构
             */
            virtual ~PolicyBase() = default;

            /**
             * @brief 构造函数：初始化观测维度和裁剪阈值
             * @param numSingleObs   单次观测数据长度
             * @param clipObsValue   裁剪观测值的阈值
             */
            PolicyBase(int numSingleObs, int frameStack, double clipObsValue)
                : numSingleObs_(numSingleObs),
                  frameStack_(frameStack),
                  clipObsValue_(clipObsValue) // 裁剪阈值初始化
            {
                obs_ = Eigen::VectorXd::Zero(numSingleObs_); // 初始化当前观测向量
                for (int i = 0; i < frameStack_; ++i)
                {
                    histObs_.emplace_back(Eigen::VectorXd::Zero(numSingleObs_)); // 初始化历史观测缓冲区
                }
                finalObsInput_ = Eigen::MatrixXd::Zero(1, numSingleObs_ * frameStack_); // 初始化最终观测输入矩阵
            }

            /**
             * @brief 更新观测值的纯虚接口
             * @param timeDiff     与上次调用的时间差（秒）
             * @param robotState   当前机器人状态
             * @param robotOutput  当前机器人输出
             * @param cmdVel       当前控制指令
             * @param obs          输出观测向量
             * @param standby      是否处于待机模式（默认 false）
             * @return              返回 true 表示成功更新观测
             */
            virtual bool updateObservation(const double timeDiff,
                                           const Robot_State& robotState,
                                           const Robot_Output& robotOutput,
                                           const Command& cmdVel,
                                           Eigen::VectorXd& obs,
                                           bool standby = false) = 0;

            /**
             * @brief 裁剪观测向量中各维度的值，限制到[-clipObsValue_, clipObsValue_]
             * @param obs  待裁剪的观测向量
             */
            virtual void clipObservations(Eigen::VectorXd& obs)
            {
                for (int i = 0; i < numSingleObs_; ++i)
                {
                    obs[i] = std::clamp(obs[i], -clipObsValue_, clipObsValue_); // 裁剪第 i 维观测
                }
            }

            /**
             * @brief 获取最终构建的观测输入，供推理引擎使用
             * @return 最终的观测输入矩阵
             */
            virtual const Eigen::MatrixXd& getFinalObsInput()
            {
                histObs_.push_back(obs_); // 将当前观测添加到历史缓冲区
                while (histObs_.size() > frameStack_)
                {
                    histObs_.pop_front(); // 保持缓冲区大小不超过 frameStack_
                }
                for (int i = 0; i < frameStack_; ++i)
                {
                    finalObsInput_.block(0, i * numSingleObs_, 1, numSingleObs_) = histObs_[i].transpose(); // 构建最终观测输入
                }
                return finalObsInput_; // 返回最终观测输入矩阵
            }

            /**
             * @brief 在限位前修改action输出
             */
            virtual void preModifyOutput(Robot_Output& output) {};

            /**
             * @brief 在限位后修改输出映射，供派生类实现 默认不需要修改
             */
            virtual void postModifyOutput(Robot_Output& output) {};

            /**
             * @brief 判断当前动作是否结束，供派生类实现
             * @return true 表示动作已结束，false 表示动作仍在进行中
             */
            virtual bool isMotionFinished() const { return false; }

            /**
             * @brief 刷新观测数据，供派生类覆盖实现
             * 该函数在每次更新观测后调用，默认实现不做任何操作
             */
            virtual void refreshObservation()
            {
                // 默认实现不做任何操作，供派生类覆盖
            }

            /**
             * @brief 状态机进入 RUNNING 时的钩子，默认无动作
             */
            virtual void onEnterRunning(const Robot_State& robotState) {}

            /**
             * @brief 是否可以完成初始化
             */
            virtual bool isReady() const { return true; }

            /**
             * @brief 状态机进入 RUNNING 时的钩子，默认无动作
             */
            virtual Robot_Output getFirstRobotOutput()
            {
                std::cout << pdDofs_ << std::endl;
                return Robot_Output(pdDofs_);
            }

            /**
             * @brief 获取单次观测数据的长度
             * @return 单次观测数据的长度
             */
            virtual int getObsSize() const { return numSingleObs_ * frameStack_; }

            /** ======================== 公有成员变量 ======================== */

            int rlDofs_ = 0;                      // policy的关节数
            int pdDofs_ = 0;                      // 机器人实际关节数目
            int numSingleObs_;                    // 观测数据总长度
            int frameStack_;                      // 历史帧数，默认为4帧
            double clipObsValue_;                 // 裁剪阈值
            Eigen::MatrixXd finalObsInput_;       // 最终观测输入矩阵
            Eigen::VectorXd obs_;                 // 当前观测向量
            std::deque<Eigen::VectorXd> histObs_; // 历史观测缓冲区
    };

}

#endif
