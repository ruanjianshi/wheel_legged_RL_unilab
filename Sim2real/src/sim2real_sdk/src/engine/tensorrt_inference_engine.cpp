// #if 1
#if defined(PLATFORM_JETSON)

#include "engine/tensorrt_inference_engine.h"
#include "common.h"
namespace hightorque
{

    TensorRTInferenceEngine::TensorRTInferenceEngine(const std::string& path, size_t obsSize, int numActions, const std::vector<double> lower, const std::vector<double> upper)
        : obsSize_(obsSize), InferenceEngine(numActions, lower, upper),
          state_(State::Uninitialized), logger_(),
          runtime_(nullptr), cudaEngine_(nullptr), context_(nullptr),
          dInput_(nullptr), dOutput_(nullptr),
          allocator_(nullptr),
          inputSize_(0), outputSize_(0)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        state_ = State::Initializing; // 标记为初始化中
        // 启动构造线程，存储线程对象
        auto trtPath = common::replaceOrAddSuffix(path, "trt");
        std::cout << "Initializing TensorRT Inference Engine with model: " << trtPath << std::endl;
        thread_ = std::thread(&TensorRTInferenceEngine::initAsync, this, trtPath);
    }

    TensorRTInferenceEngine::~TensorRTInferenceEngine()
    {
        // 1. 等待构造线程结束（避免构造未完成就析构）
        if (thread_.joinable())
        {
            thread_.join(); // 阻塞等待，确保资源创建/清理完成
        }
        // 2. 安全销毁资源
        destroy();
    }

    void TensorRTInferenceEngine::initAsync(const std::string& path)
    {
        bool success = false;
        try
        {
            success = init(path); // 执行实际初始化
        }
        catch (const std::exception& e)
        {
            std::cerr << "Async init failed: " << e.what() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300)); // 模拟异步延迟

        // 更新状态 + 清理失败资源
        std::lock_guard<std::mutex> lock(mtx_);
        if (success)
        {
            state_ = State::Initialized;
        }
        else
        {
            state_ = State::Uninitialized;
            destroy(); // 清理已创建的部分资源
        }
    }

    bool TensorRTInferenceEngine::init(const std::string& path)
    {
        // 1. 打开模型文件
        std::ifstream modelFileStream(path, std::ios::binary);
        if (!modelFileStream)
        {
            std::cerr << "Failed to open model file" << path << std::endl;
            return false;
        }

        // 2. 读取引擎文件
        // 读取 TensorRT engine
        std::string modelStr((std::istreambuf_iterator<char>(modelFileStream)),
                             std::istreambuf_iterator<char>());
        modelFileStream.close(); // 提前关闭，释放文件资源

        // 3. 创建 TensorRT 核心组件（加锁保护，避免与析构冲突）
        // 创建 TensorRT builder 和解析器
        std::lock_guard<std::mutex> lock(mtx_);
        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_)
        {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        cudaEngine_ = runtime_->deserializeCudaEngine(modelStr.data(), modelStr.size(), nullptr);
        if (!cudaEngine_)
        {
            std::cerr << "Failed to create TensorRT engine from model file" << std::endl;
            return false;
        }

        inputSize_ = obsSize_ * sizeof(float);
        outputSize_ = nums_ * sizeof(float);

        if (cudaMalloc(&dInput_, inputSize_) != cudaSuccess)
        {
            std::cerr << "Failed to allocate device memory for input." << std::endl;
            return false;
        }
        if (cudaMalloc(&dOutput_, outputSize_) != cudaSuccess)
        {
            std::cerr << "Failed to allocate device memory for output." << std::endl;
            return false;
        }

        context_ = cudaEngine_->createExecutionContext();
        if (!context_)
        {
            std::cerr << "Failed to create execution context." << std::endl;
            return false;
        }

        allocator_ = nullptr; // TODO: 设置适当的分配器
        init_ = true;
        return true;
    }

    void TensorRTInferenceEngine::destroy()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (state_ == State::Destroyed)
        {
            return; // 避免重复销毁
        }

        // 1. 释放 CUDA 设备内存（独立资源，先释放）
        if (dInput_ != nullptr)
        {
            cudaFree(dInput_);
            dInput_ = nullptr; // 置空，避免重复释放
        }
        if (dOutput_ != nullptr)
        {
            cudaFree(dOutput_);
            dOutput_ = nullptr;
        }

        // 2. 销毁执行上下文（依赖 engine，先于 engine 释放）
        if (context_ != nullptr)
        {
            context_->destroy();
            context_ = nullptr;
        }

        // 3. 销毁引擎（依赖 runtime，先于 runtime 释放）
        if (cudaEngine_ != nullptr)
        {
            cudaEngine_->destroy();
            cudaEngine_ = nullptr;
        }

        // 4. 销毁运行时（最后释放，因为 engine 依赖它创建）
        if (runtime_ != nullptr)
        {
            runtime_->destroy();
            runtime_ = nullptr;
        }
        init_ = false;
        state_ = State::Destroyed; // 标记为已销毁
    }

    bool TensorRTInferenceEngine::updateAction(Eigen::MatrixXd input, Robot_Output& robotOut, const Joy& joy)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (state_ != State::Initialized)
        {
            return false;
        }
        std::vector<float> inputData(obsSize_);
        if (!init_ || input.rows() != 1 || input.cols() != obsSize_)
        {
            std::cerr << "TensorRTInferenceEngine not initialized or input size mismatch!" << std::endl;
            return false;
        }
        for (size_t i = 0; i < obsSize_; i++)
        {
            inputData[i] = input(i);
        }

        if (cudaMemcpy(dInput_, inputData.data(), inputSize_, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            std::cerr << "Failed to copy input data to device." << std::endl;
            return false;
        }

        void* buffers[2];
        buffers[0] = dInput_;
        buffers[1] = dOutput_;

        if (!context_->executeV2(buffers))
        {
            std::cerr << "Failed to execute inference." << std::endl;
            return false;
        }

        auto outputData = new float[nums_ * sizeof(float)];
        // 将输出数据从GPU复制回CPU
        if (cudaMemcpy(outputData, dOutput_, outputSize_, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "Failed to copy output data from device." << std::endl;
            return false;
        }
        if (robotOut.action.size() != nums_)
        {
            robotOut.action.resize(nums_);
        }
        for (int i = 0; i < nums_; i++)
        {
            robotOut.action[i] = outputData[i];
        }

        free(outputData);
        return true;
    }
}
#endif
