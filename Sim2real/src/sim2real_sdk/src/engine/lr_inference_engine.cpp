#include "engine/lr_inference_engine.h"
#include "common.h"

namespace hightorque
{
    LrInferenceEngine::LrInferenceEngine(const std::string& serviceName, int dofs, int numSingleObs, const std::vector<double> lower, const std::vector<double> upper,
                                         const std::vector<int>& normalMode, const std::vector<int>& fastMode)
        : InferenceEngine(dofs, lower, upper), normalMode_(normalMode), fastMode_(fastMode), isFastMode_(false), lastButtonPressed_(false), currentFrequency_(0), currentStride_(0), modeJustSwitched_(false), remainingFrequencySteps_(0), remainingStrideSteps_(0), lastButtonTime_(ros::Time::now())
    {
        nh_ = std::make_shared<ros::NodeHandle>("");
        numSingleObs_ = numSingleObs;
        dofs_ = dofs;
        client_ = nh_->serviceClient<sim2real_msg::ModelOutput>(serviceName, true);
        init_ = true;
        ROS_INFO("LrInferenceEngine initialized with service %s, dofs: %d, numSingleObs: %d", serviceName.c_str(), dofs_, numSingleObs_);
        ROS_INFO("Normal mode: [frequency=%d, stride=%d], Fast mode: [frequency=%d, stride=%d]",
                 normalMode_[0], normalMode_[1], fastMode_[0], fastMode_[1]);
    }

    bool LrInferenceEngine::updateAction(Eigen::MatrixXd input, Robot_Output& robotOut, const Joy& joy)
    {
        sim2real_msg::ModelOutput srv;
        int idx = 0;
        srv.request.joint_positions.resize(dofs_);
        srv.request.joint_velocities.resize(dofs_);
        srv.request.imu_data.resize(6);
        srv.response.target_pos.resize(dofs_);
        for (auto i = 0; i < dofs_; i++)
        {
            srv.request.joint_positions[i] = static_cast<float>(input(idx + i));
        }
        idx += dofs_;
        for (auto i = 0; i < dofs_; i++)
        {
            srv.request.joint_velocities[i] = static_cast<float>(input(idx + i));
        }
        idx += dofs_;

        for (auto i = 0; i < 6; i++)
        {
            srv.request.imu_data[i] = static_cast<float>(input(idx + i));
        }
        idx += 6;

        srv.request.joy_axes_0 = static_cast<float>(input(idx + 0)); // y
        srv.request.joy_axes_1 = static_cast<float>(input(idx + 1)); // x
        srv.request.joy_axes_3 = static_cast<float>(input(idx + 2)); // angle_z

        // 基于 LT/RT 的模式切换（持续检测）：LT<0 且 RT>0 加速，否则普通
        // bool ltPressedEdgeBlock = (joy.LT < 0) && (joy.RT > 0);
        
        // 基于 L 的模式切换（持续检测）：L==1 ，否则普通
        if (joy.RT < 0 && joy.DPU)
        {
            lastButtonPressed_ = true;
        }
        else if (joy.RT < 0 && joy.DPD)
        {
            lastButtonPressed_ = false;
        }

        bool ltPressedEdgeBlock = lastButtonPressed_;
        if (ltPressedEdgeBlock && !isFastMode_)
        {
            isFastMode_ = true;
            modeJustSwitched_ = true;
            int targetFrequency = fastMode_[0];
            int targetStride = fastMode_[1];
            int frequencyDiff = targetFrequency - currentFrequency_;
            int strideDiff = targetStride - currentStride_;
            remainingFrequencySteps_ = (frequencyDiff > 0) ? frequencyDiff : -frequencyDiff;
            remainingStrideSteps_ = (strideDiff > 0) ? strideDiff : -strideDiff;
            ROS_INFO("Switch to FAST [frequency=%d, stride=%d], current: freq=%d, stride=%d, remaining: freq=%d, stride=%d",
                     targetFrequency, targetStride, currentFrequency_, currentStride_, remainingFrequencySteps_, remainingStrideSteps_);
        }
        else if (!ltPressedEdgeBlock && isFastMode_)
        {
            isFastMode_ = false;
            modeJustSwitched_ = true;
            int targetFrequency = normalMode_[0];
            int targetStride = normalMode_[1];
            int frequencyDiff = targetFrequency - currentFrequency_;
            int strideDiff = targetStride - currentStride_;
            remainingFrequencySteps_ = (frequencyDiff > 0) ? frequencyDiff : -frequencyDiff;
            remainingStrideSteps_ = (strideDiff > 0) ? strideDiff : -strideDiff;
            ROS_INFO("Switch to NORMAL [frequency=%d, stride=%d], current: freq=%d, stride=%d, remaining: freq=%d, stride=%d",
                     targetFrequency, targetStride, currentFrequency_, currentStride_, remainingFrequencySteps_, remainingStrideSteps_);
        }

        // 初始化按钮状态为0
        srv.request.joy_buttons_0 = 0; // A: 减小步幅
        srv.request.joy_buttons_1 = 0; // B: 减小步频
        srv.request.joy_buttons_2 = 0; // X: 增加步频
        srv.request.joy_buttons_3 = 0; // Y: 增加步幅

        if (remainingFrequencySteps_ > 0 || remainingStrideSteps_ > 0)
        {
            ros::Time currentTime = ros::Time::now();
            double timeSinceLastButton = (currentTime - lastButtonTime_).toSec();
            if (timeSinceLastButton >= 0.3)
            {
                // 根据当前模式获取目标状态
                const std::vector<int>& currentMode = isFastMode_ ? fastMode_ : normalMode_;
                int targetFrequency = currentMode[0]; // 目标步频状态
                int targetStride = currentMode[1];    // 目标步幅状态

                // 计算当前差值
                int frequencyDiff = targetFrequency - currentFrequency_;
                int strideDiff = targetStride - currentStride_;

                // 执行步频调整
                if (remainingFrequencySteps_ > 0)
                {
                    if (frequencyDiff > 0)
                    {
                        srv.request.joy_buttons_2 = 1;
                        currentFrequency_++;
                    }
                    else if (frequencyDiff < 0)
                    {
                        srv.request.joy_buttons_1 = 1;
                        currentFrequency_--;
                    }
                    remainingFrequencySteps_--;
                }

                // 执行步幅调整
                if (remainingStrideSteps_ > 0)
                {
                    if (strideDiff > 0)
                    {
                        srv.request.joy_buttons_3 = 1;
                        currentStride_++;
                    }
                    else if (strideDiff < 0)
                    {
                        srv.request.joy_buttons_0 = 1;
                        currentStride_--;
                    }
                    remainingStrideSteps_--;
                }

                lastButtonTime_ = currentTime;

                ROS_INFO("Button actions: A=%d, B=%d, X=%d, Y=%d (current: freq=%d, stride=%d, remaining: freq=%d, stride=%d)",
                         srv.request.joy_buttons_0, srv.request.joy_buttons_1,
                         srv.request.joy_buttons_2, srv.request.joy_buttons_3,
                         currentFrequency_, currentStride_, remainingFrequencySteps_, remainingStrideSteps_);
            }
            else
            {
            }
        }

        if (!init_)
        {
            ROS_ERROR_THROTTLE(1, "lr inference_engine not init");
            return false;
        }
        auto now = ros::Time::now();
        if (!client_.call(srv))
        {
            if (!ros::service::exists(client_.getService(), true))
            {
                ROS_ERROR("Failed to call %s: Service is not available ", client_.getService().c_str());
            }
            // else if (ros::service::waitForService(client_.getService(), ros::Duration(0.01)) == false)
            // {
            //     ROS_ERROR("Failed to call %s: Service response timeout", client_.getService().c_str());
            // }
            else
            {
                ROS_ERROR("Failed to call %s: Service rejected request", client_.getService().c_str());
            }
            ROS_ERROR("Failed to call service %s\njoint_positions:%s\njoint_velocities:%s\nimu_data:%s\njoy_axes_0:%f, joy_axes_1:%f, joy_axes_3:%f\njoy_buttons_0:%d, joy_buttons_1:%d, joy_buttons_2:%d, joy_buttons_3:%d\nresponse: %s\nsystem: %s",
                      client_.getService().c_str(),
                      common::toString(srv.request.joint_positions).c_str(),
                      common::toString(srv.request.joint_velocities).c_str(),
                      common::toString(srv.request.imu_data).c_str(),
                      srv.request.joy_axes_0,
                      srv.request.joy_axes_1,
                      srv.request.joy_axes_3,
                      srv.request.joy_buttons_0,
                      srv.request.joy_buttons_1,
                      srv.request.joy_buttons_2,
                      srv.request.joy_buttons_3,
                      common::toString(srv.response.target_pos).c_str(),
                      common::getSystemInfo().c_str());
            return false;
        }
        if (ros::Time::now().toSec() - now.toSec() > 0.02)
        {
            ROS_WARN("call %s took too long: %.4f seconds", client_.getService().c_str(), (ros::Time::now() - now).toSec());
        }
        robotOut.resize(dofs_);
        for (auto i = 0; i < dofs_; i++)
        {
            robotOut.action[i] = srv.response.target_pos[i];
        }

        return true;
    }
}
