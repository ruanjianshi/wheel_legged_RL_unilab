#ifndef TENSORRT_INFERENCE_ENGINE_H_
#define TENSORRT_INFERENCE_ENGINE_H_

#if defined(PLATFORM_JETSON)
// #if 1
#include "engine/inference_engine.h"
#include <NvInfer.h>
#include <fstream>
#include <thread>

namespace hightorque
{
    /**
     * @brief TensorRT 日志器类 Logger，用于输出警告及错误信息
     *
     * 继承自 nvinfer1::ILogger，仅打印警告及错误等级的信息。
     */
    class Logger : public nvinfer1::ILogger
    {
        public:
            /**
             * @brief 重载 log 函数，过滤日志级别，只打印警告及错误
             * @param severity 日志等级（错误、警告、信息等）
             * @param msg 日志内容
             */
            void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
            {
                // 只处理错误和警告信息
                if (severity <= nvinfer1::ILogger::Severity::kWARNING)
                {
                    std::cout << "TensorRTInferenceEngine: " << msg << std::endl;
                }
            }
    };

    /**
     * @brief 基于 TensorRT 的推理引擎实现类 TensorRTInferenceEngine
     *
     * 该类继承自 InferenceEngine，适用于 NVIDIA Jetson 或支持 TensorRT 的平台。
     * 使用序列化的 TensorRT 引擎文件进行快速推理，并将输出转换为机器人动作指令。
     */
    class TensorRTInferenceEngine : public InferenceEngine
    {
        public:
            /**
             * @brief 构造函数
             * @param path TensorRT 序列化引擎文件路径（.trt 文件）
             * @param obsSize 输入观测向量的维度
             * @param numActions 输出动作维度数量
             * @param lower 每个动作维度的下限（float）
             * @param upper 每个动作维度的上限（float）
             */
            TensorRTInferenceEngine(const std::string& path, size_t obsSize, int numActions, const std::vector<double> lower, const std::vector<double> upper);
            // 禁用拷贝/移动（避免资源共享冲突）
            TensorRTInferenceEngine(const TensorRTInferenceEngine&) = delete;
            TensorRTInferenceEngine& operator=(const TensorRTInferenceEngine&) = delete;
            TensorRTInferenceEngine(TensorRTInferenceEngine&&) = delete;
            TensorRTInferenceEngine& operator=(TensorRTInferenceEngine&&) = delete;

            /**
             * @brief 析构函数，释放 GPU 内存及资源
             */
            ~TensorRTInferenceEngine();

            /**
             * @brief 显式销毁函数，释放内部成员
             */
            void destroy();

            void initAsync(const std::string& path);
            bool init(const std::string& path);

            /**
             * @brief 执行推理并输出机器人控制指令
             * @param input 输入特征（Eigen 矩阵）
             * @param robotOut 输出动作结构体
             * @return bool 推理是否成功
             */
            bool updateAction(Eigen::MatrixXd input, Robot_Output& robotOut, const Joy& joy) override;

        private:
            // 状态机：标记实例生命周期状态
            enum class State
            {
                Uninitialized, // 未初始化
                Initializing,  // 初始化中
                Initialized,   // 初始化完成
                Destroyed      // 已销毁
            };

            size_t obsSize_; // 输入特征维度

            Logger logger_;                                  // TensorRT 日志对象
            nvinfer1::IRuntime* runtime_ = nullptr;          // 关键：之前遗漏的运行时
            nvinfer1::ICudaEngine* cudaEngine_ = nullptr;    // TensorRT 推理引擎
            nvinfer1::IExecutionContext* context_ = nullptr; // 推理上下文
            nvinfer1::IGpuAllocator* allocator_ = nullptr;   // 自定义 GPU 内存分配器（如未使用则为 nullptr）

            int inputIndex_ = -1;  // 输入张量在 TensorRT 引擎中的索引
            int outputIndex_ = -1; // 输出张量在 TensorRT 引擎中的索引

            size_t inputSize_ = 0;  // 输入张量大小（单位：字节）
            size_t outputSize_ = 0; // 输出张量大小（单位：字节）

            void* dInput_ = nullptr;  // GPU 上的输入缓冲区
            void* dOutput_ = nullptr; // GPU 上的输出缓冲区

            std::thread thread_;     // 存储构造线程，析构时 join
            mutable std::mutex mtx_; // 保护成员变量访问
            State state_;            // 实例状态机
    };

}

#endif

#endif
