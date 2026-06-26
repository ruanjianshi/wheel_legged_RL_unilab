#include "engine/openvino_inference_engine.h"

#if defined(PLATFORM_X86_64)
namespace hightorque
{

    OpenvinoInferenceEngine::OpenvinoInferenceEngine(const std::string& path, const std::string& deviceName, int nums, const std::vector<double> lower, const std::vector<double> upper) : InferenceEngine(nums, lower, upper)
    {
        auto onnxPath = common::replaceOrAddSuffix(path, "onnx");
        std::cout << "Initializing ONNX Inference Engine with model: " << onnxPath << std::endl;
        model = core.read_model(onnxPath);
        compiledModel = core.compile_model(model, deviceName);
        inferRequest = compiledModel.create_infer_request();
        init_ = true;
    }

    bool OpenvinoInferenceEngine::updateAction(Eigen::MatrixXd input, Robot_Output& robotOut, const Joy& joy)
    {
        if (!init_)
        {
            return false;
        }
        auto inputPort = compiledModel.input();
        auto inputTensor = inferRequest.get_tensor(inputPort);
        float* inputData = inputTensor.data<float>();
        for (size_t i = 0; i < inputTensor.get_size(); i++)
        {
            inputData[i] = input(i);
        }
        inferRequest.infer();
        auto outputPort = compiledModel.output();
        auto outputTensor = inferRequest.get_tensor(outputPort);

        const float* outputData = outputTensor.data<const float>();

        floatOutput2RobotOutput(outputData, robotOut);
        return true;
    }
}
#endif // PLATFORM_X86_64
