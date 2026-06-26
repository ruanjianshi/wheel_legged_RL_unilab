#ifndef RKNN_INFERENCE_ENGINE_H_
#define RKNN_INFERENCE_ENGINE_H_

#if defined(PLATFORM_RK)
#include "engine/inference_engine.h"
#include <rknn_api.h>

namespace hightorque
{
    /**
     * @brief 基于 RKNN（瑞芯微 NPU）的推理引擎实现类 RKNNInferenceEngine
     *
     * 该类继承自 InferenceEngine，使用 RKNN 推理框架进行模型加载和执行。
     * 主要用于嵌入式设备（如 RK3568）上加速模型推理，并将输出转换为机器人动作指令。
     */
    class RKNNInferenceEngine : public InferenceEngine
    {
        public:
            /**
             * @brief 构造函数
             * @param path RKNN 模型文件路径（通常为 .rknn）
             * @param obsSize 输入特征向量维度
             * @param actionNums 动作维度数量
             * @param lower 每个动作维度的下限（float）
             * @param upper 每个动作维度的上限（float）
             */
            RKNNInferenceEngine(const std::string& path, size_t obsSize, int actionNums, const std::vector<double> lower, const std::vector<double> upper);

            /**
             * @brief 执行推理并更新机器人输出动作
             * @param input 输入特征（Eigen 矩阵）
             * @param robotOut 输出动作结构体
             * @return bool 推理是否成功
             */
            bool updateAction(Eigen::MatrixXd input, Robot_Output& robotOut, const Joy& joy) override;

        private:
            size_t obsSize_;              // 输入特征向量大小
            rknn_context rknnContext_;    // RKNN 推理上下文句柄
            rknn_input_output_num ioNum_; // 输入输出数量描述结构体
            rknn_input rknnInputs_[1];    // RKNN 输入结构体（假设只有一个输入）
            rknn_output rknnOutputs_[1];  // RKNN 输出结构体（假设只有一个输出）
    };

}

#endif

#endif
