#if defined(PLATFORM_RK)

#include "engine/rknn_inference_engine.h"
#include "common.h"
#include "rknn_api.h"
namespace hightorque
{
    RKNNInferenceEngine::RKNNInferenceEngine(const std::string& path, size_t obsSize, int nums, const std::vector<double> lower, const std::vector<double> upper) : obsSize_(obsSize), InferenceEngine(nums, lower, upper)
    {
        int modelDataSize = 0;
        auto rknnPath = common::replaceOrAddSuffix(path, "rknn");
        std::cout << "Initializing RKNN Inference Engine with model: " << rknnPath << std::endl;
        auto modelData = common::readFileData(rknnPath.c_str(), &modelDataSize);
        int ret = rknn_init(&rknnContext_, modelData, modelDataSize, 0, NULL);
        if (ret < 0)
        {
            std::cerr << "rknn_init error ret=" << ret << std::endl;
            return;
        }

        rknn_sdk_version version;
        ret = rknn_query(rknnContext_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
        if (ret < 0)
        {
            std::cerr << "rknn_query RKNN_QUERY_SDK_VERSION error ret=" << ret << std::endl;
            return;
        }
        ret = rknn_query(rknnContext_, RKNN_QUERY_IN_OUT_NUM, &ioNum_, sizeof(ioNum_));
        if (ret < 0)
        {
            std::cerr << "rknn_query RKNN_QUERY_IN_OUT_NUM error ret=" << ret << std::endl;
            return;
        }
        std::cout << "model input num:" << ioNum_.n_input << " output num: " << ioNum_.n_output << std::endl;

        memset(rknnInputs_, 0, sizeof(rknn_input));
        rknnInputs_[0].index = 0;
        rknnInputs_[0].size = obsSize * sizeof(float);
        rknnInputs_[0].pass_through = false;
        rknnInputs_[0].type = RKNN_TENSOR_FLOAT32;
        rknnInputs_[0].fmt = RKNN_TENSOR_NHWC;

        memset(rknnOutputs_, 0, sizeof(rknn_output));
        rknnOutputs_[0].want_float = true;

        init_ = true;
    }

    bool RKNNInferenceEngine::updateAction(Eigen::MatrixXd input, Robot_Output& robotOut, const Joy& joy)
    {
        std::vector<float> inputData(obsSize_);
        if (!init_ || input.rows() != 1 || input.cols() != obsSize_)
        {
            std::cerr << "RKNNInferenceEngine not initialized or input size mismatch! " << obsSize_ << " != " << input.cols() << std::endl;
            return false;
        }

        for (size_t i = 0; i < obsSize_; i++)
        {
            inputData[i] = input(i);
        }

        rknnInputs_[0].buf = inputData.data();

        int ret = rknn_inputs_set(rknnContext_, ioNum_.n_input, rknnInputs_);
        if (ret != RKNN_SUCC)
        {
            std::cerr << "Failed to set RKNN input!" << ret << std::endl;
            rknn_destroy(rknnContext_);
        }

        ret = rknn_run(rknnContext_, nullptr);
        if (ret != RKNN_SUCC)
        {
            std::cerr << "Failed to run RKNN inference!" << std::endl;
            rknn_destroy(rknnContext_);
        }

        ret = rknn_outputs_get(rknnContext_, ioNum_.n_output, rknnOutputs_, nullptr);
        if (ret != RKNN_SUCC)
        {
            std::cerr << "Failed to get RKNN output!" << std::endl;
            rknn_destroy(rknnContext_);
        }

        const float* outputData = static_cast<float*>(rknnOutputs_[0].buf);
        if (robotOut.action.size() != nums_)
        {
            robotOut.action.resize(nums_);
        }
        for (int i = 0; i < nums_; i++)
        {
            robotOut.action[i] = outputData[i];
        }

        // floatOutput2RobotOutput(outputData, robotOut);
        return true;
    }
}
#endif
