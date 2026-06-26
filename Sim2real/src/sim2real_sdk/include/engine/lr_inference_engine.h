#ifndef LR_INFERENCE_ENGINE_H_
#define LR_INFERENCE_ENGINE_H_

#include "engine/inference_engine.h"
#include <ros/ros.h>

namespace hightorque
{
    /**
     * @brief 基于 亮源的算法 使用service去获取action
     */
    class LrInferenceEngine : public InferenceEngine
    {
        public:
            LrInferenceEngine(const std::string& serviceName, int dofs, int numSingleObs, const std::vector<double> lower, const std::vector<double> upper, 
                            const std::vector<int>& normalMode = {0, 0}, const std::vector<int>& fastMode = {1, 1});

            /**
             * @brief 更新动作输出接口（纯虚函数） 仅仅用于lr算法
             * @param input
             * @param robotOut 输出结构体，写入最终控制命令
             * @return bool 是否推理成功
             */
            bool updateAction(Eigen::MatrixXd input, Robot_Output& robotOut, const Joy& joy);

        private:
            std::shared_ptr<ros::NodeHandle> nh_;
            ros::ServiceClient client_;
            ros::Timer timer_;
            int numSingleObs_;
            int dofs_;
            std::vector<int> normalMode_;   // 正常模式 [frequency, stride]
            std::vector<int> fastMode_;     // 加快模式 [frequency, stride]
            bool isFastMode_;               // 当前是否是加快模式
            bool lastButtonPressed_;       // 上次Y按钮状态，用于边沿检测
            int currentFrequency_;          // 当前步频状态
            int currentStride_;             // 当前步幅状态
            bool modeJustSwitched_;         // 模式是否刚刚切换
            int remainingFrequencySteps_;   // 剩余步频调整次数
            int remainingStrideSteps_;      // 剩余步幅调整次数
            ros::Time lastButtonTime_;      // 上次发送按钮的时间
    };
}

#endif
