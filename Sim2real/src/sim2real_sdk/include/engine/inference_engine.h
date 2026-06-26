#ifndef INFERENCE_ENGINE_H_
#define INFERENCE_ENGINE_H_

#include "robot_data.h"
#include "common.h"
#include "sim2real_msg/ModelOutput.h"
#include </usr/include/eigen3/Eigen/Dense>
namespace hightorque
{
    /**
     * @brief 推理引擎基类 InferenceEngine
     *
     * 该抽象类定义了推理引擎的通用接口，用于处理模型输入、输出，并进行动作映射限制。
     * 主要功能包括：
     *  - 统一输入输出接口定义
     *  - 支持浮点数范围约束
     *  - 提供 float 输出转为 Robot_Output 的映射方法
     */
    class InferenceEngine
    {
        public:
            /**
             * @brief 禁用默认构造函数，防止未设置参数实例化
             */
            InferenceEngine() = delete;

            /**
             * @brief 构造函数（使用 double 类型下限与上限）
             * @param nums 动作维度数量
             * @param lower 各动作维度的下限（double 类型）
             * @param upper 各动作维度的上限（double 类型）
             */
            InferenceEngine(int nums, const std::vector<double> lower, const std::vector<double> upper)
                : nums_(nums), lower_(lower), upper_(upper)
            {
                //
                std::cout << "InferenceEngine initialized with " << nums_ << std::endl;
                std::cout << "lower: " << common::toString(lower_) << std::endl;
                std::cout << "upper: " << common::toString(upper_) << std::endl;
            }

            /**
             * @brief 构造函数（接受 float 类型上下限并转换为 double）
             * @param nums 动作维度数量
             * @param lower 各动作维度的下限（float 类型）
             * @param upper 各动作维度的上限（float 类型）
             */
            InferenceEngine(int nums, const std::vector<float>& lower, const std::vector<float>& upper)
                : nums_(nums)
            {
                // 把 float 类型的向量转换为 double 类型，统一内部数据精度
                lower_.assign(lower.begin(), lower.end());
                upper_.assign(upper.begin(), upper.end());
                std::cout << "InferenceEngine initialized with " << nums_ << std::endl;
                std::cout << "lower: " << common::toString(lower_) << std::endl;
                std::cout << "upper: " << common::toString(upper_) << std::endl;
            }

            /**
             * @brief 虚析构函数，确保派生类析构时资源正确释放
             */
            virtual ~InferenceEngine() = default;

            /**
             * @brief 更新动作输出接口（纯虚函数）
             * @param input 输入特征数据（Eigen 矩阵）
             * @param robotOut 输出结构体，写入最终控制命令
             * @return bool 是否推理成功
             */
            virtual bool updateAction(Eigen::MatrixXd input, Robot_Output& robotOut, const Joy& joy) = 0;

            /**
             * @brief 将模型 float 输出数据映射到机器人输出结构体，并做范围限制
             * @param outputData 模型输出的动作数据（float 数组）
             * @param robotOut 输出结构体，动作值将被写入其中
             */
            virtual void floatOutput2RobotOutput(const float* outputData, Robot_Output& robotOut)
            {
                for (int i = 0; i < nums_; i++)
                {
                    // 限制每个动作值在下限与上限之间
                    robotOut.action[i] = std::clamp(outputData[i], static_cast<float>(lower_[i]), static_cast<float>(upper_[i]));
                }
            }

            virtual void clipActions(Robot_Output& robotOut)
            {
                for (int i = 0; i < nums_; i++)
                {
                    // 限制每个动作值在下限与上限之间
                    robotOut.action[i] = std::clamp(robotOut.action[i], lower_[i], upper_[i]);
                }
            }

            bool init_ = false;                 // 初始化标志，防止未成功初始化使用
            int nums_;                          // 动作维度数
            std::vector<double> lower_, upper_; // 每个动作维度的取值下限与上限
    };
}

#endif
