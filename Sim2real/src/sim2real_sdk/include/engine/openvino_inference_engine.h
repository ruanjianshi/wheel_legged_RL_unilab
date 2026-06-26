#ifndef OPENVINO_INFERENCE_ENGINE_H_
#define OPENVINO_INFERENCE_ENGINE_H_

#if defined(PLATFORM_X86_64)
#include "engine/inference_engine.h"
#include <openvino/runtime/core.hpp>
#include <openvino/openvino.hpp>

namespace hightorque
{
    /**
     * @brief 基于 OpenVINO 的推理引擎实现类 OpenvinoInferenceEngine
     *
     * 该类继承自 InferenceEngine，使用 OpenVINO 推理框架进行模型加载和前向推理。
     * 支持输入特征向量，输出动作控制指令，并自动完成数据限制与转换。
     */
    class OpenvinoInferenceEngine : public InferenceEngine
    {
        public:
            /**
             * @brief 构造函数
             * @param path 模型文件路径（.xml/.bin 或 IR 格式）
             * @param deviceName 推理设备名称（如 "CPU", "GPU", "AUTO"）
             * @param nums 动作维度数量
             * @param lower 每个动作维度的取值下限（double）
             * @param upper 每个动作维度的取值上限（double）
             */
            OpenvinoInferenceEngine(const std::string& path, const std::string& deviceName, int nums, const std::vector<double> lower, const std::vector<double> upper);

            /**
             * @brief 执行推理并更新动作指令
             * @param input 输入特征数据（Eigen 矩阵）
             * @param robotOut 输出结构体，写入最终控制指令
             * @return bool 推理是否成功
             */
            bool updateAction(Eigen::MatrixXd input, Robot_Output& robotOut, const Joy& joy) override;

        private:
            ov::Core core;                    // OpenVINO 核心对象，负责设备初始化和模型管理
            std::shared_ptr<ov::Model> model; // 原始模型对象
            ov::CompiledModel compiledModel;  // 编译后的模型
            ov::InferRequest inferRequest;    // 推理请求对象，执行一次前向推理任务
    };
}
#endif // PLATFORM_X86_64

#endif
