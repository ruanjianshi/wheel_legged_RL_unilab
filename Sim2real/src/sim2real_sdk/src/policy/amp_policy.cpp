#include "policy/amp_policy.h"
#include "common.h"
#include "impl/hightorque_config.h"
#include <nlohmann/json.hpp>
#include <tf/transform_datatypes.h>

namespace hightorque
{
    AMPPolicy::AMPPolicy(const config::RLConfig& rlInfo, const config::PDConfig& pdInfo)
        : PolicyBase(rlInfo.num_single_obs, rlInfo.frame_stack, rlInfo.clip_obs)
    {
        frequency_ = rlInfo.frequency;
        robotAngleVelScale_ = rlInfo.rbt_ang_vel_scale;
        cmdVelLineScale_ = rlInfo.cmd_lin_vel_scale;
        cmdVelAngleScale_ = rlInfo.cmd_ang_vel_scale;
        robotLinePoseScale_ = rlInfo.rbt_lin_pos_scale;
        robotLineVelScale_ = rlInfo.rbt_lin_vel_scale;
        lastCmdVel_ = {0.0, 0.0, 0.0};
        mode_ = "normal";

        // 控制模式相关
        controlMode_ = rlInfo.control_mode;
        cmdVelFilterScaleSoccer_ = rlInfo.cmd_vel_filter_scale_soccer;
        cmdVelFilterScaleFight_ = rlInfo.cmd_vel_filter_scale_fight;
        cmdVelXMaxSoccerWalk_ = rlInfo.cmd_vel_x_max_soccer_walk;
        cmdVelXMaxSoccerNormal_ = rlInfo.cmd_vel_x_max_soccer_normal;
        cmdVelXMaxSoccerBoost_ = rlInfo.cmd_vel_x_max_soccer_boost;
        cmdVelXMaxFight_ = rlInfo.cmd_vel_x_max_fight;
        cmdVelXMinNormal_ = rlInfo.cmd_vel_x_min;
        cmdVelXMinWalk_ = rlInfo.cmd_vel_x_min_walk;
        cmdVelYMin_ = rlInfo.cmd_vel_y_min;
        cmdVelYMax_ = rlInfo.cmd_vel_y_max;
        cmdVelYawMin_ = rlInfo.cmd_vel_yaw_min;
        cmdVelYawMax_ = rlInfo.cmd_vel_yaw_max;

        // 从 RLConfig 获取策略自由度
        rlDofs_ = rlInfo.dofs;

        // 从 PDConfig 获取物理自由度和映射关系
        pdDofs_ = pdInfo.dofs;
        policy2ActualMap_ = pdInfo.policy_to_actual_map;
        actual2PolicyMap_ = pdInfo.actual_to_policy_map;

        // 验证映射关系大小
        if (policy2ActualMap_.size() != rlDofs_)
        {
            std::cout << "Warning: policy_to_actual_map size mismatch. Expected "
                      << rlDofs_ << ", got " << policy2ActualMap_.size()
                      << ". Using default mapping." << std::endl;
            // 创建默认映射
            policy2ActualMap_.resize(rlDofs_);
            for (int i = 0; i < rlDofs_; ++i)
            {
                policy2ActualMap_[i] = i;
            }
        }

        if (actual2PolicyMap_.size() != pdDofs_)
        {
            std::cout << "Warning: actual_to_policy_map size mismatch. Expected "
                      << pdDofs_ << ", got " << actual2PolicyMap_.size()
                      << ". Using default mapping." << std::endl;
            // 创建默认映射
            actual2PolicyMap_.resize(pdDofs_);
            std::iota(actual2PolicyMap_.begin(), actual2PolicyMap_.end(), 0); // 从0开始递增
        }

        // 初始化动作缩放
        actionSales_.resize(rlDofs_);
        if (rlInfo.action_scales.size() == rlDofs_)
        {
            actionSales_ = rlInfo.action_scales;
        }
        else
        {
            actionSales_.resize(rlDofs_, 1.0); // 默认缩放为1.0
            std::cout << "Warning: action_scales size mismatch. Using default scaling factors." << std::endl;
        }

        // 初始化默认位姿
        defaultPose_.resize(rlDofs_);
        if (rlInfo.default_pose.size() == rlDofs_)
        {
            defaultPose_ = rlInfo.default_pose;
        }
        else
        {
            defaultPose_.resize(rlDofs_, 0.0); // 默认位姿为0.0
            std::cout << "Warning: default_pose size mismatch. Using zero default pose." << std::endl;
        }

        lastRobotOutput_.resize(rlDofs_);

        std::cout << "AMPPolicy initialized with: "
                  << "frequency=" << frequency_ << " Hz, "
                  << "rlDofs=" << rlDofs_ << ", "
                  << "pdDofs=" << pdDofs_ << ", "
                  << "controlMode=" << controlMode_ << ", "
                  << "cmdVelFilterScaleSoccer=" << cmdVelFilterScaleSoccer_ << ", "
                  << "cmdVelFilterScaleFight=" << cmdVelFilterScaleFight_ << ", "
                  << "cmdVelXMaxSoccerNormal=" << cmdVelXMaxSoccerNormal_ << ", "
                  << "cmdVelXMaxSoccerBoost=" << cmdVelXMaxSoccerBoost_ << ", "
                  << "cmdVelXMaxFight=" << cmdVelXMaxFight_ << ", "
                  << "cmdVelYMax=" << cmdVelYMax_ << ", "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_ << std::endl;
    }

    bool AMPPolicy::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby)
    {
        if (obs_.size() < numSingleObs_)
        {
            obs_.resize(numSingleObs_);
        }

        Robot_State policyRobotState(rlDofs_), actualRobotState = robotState;
        Robot_Output policyRobotOutput(rlDofs_), actualRobotOutput = robotOutput;

        common::transformActualRobotStateToPolicyRobotState(actualRobotState, policyRobotState, actual2PolicyMap_);
        common::transformActualRobotOutputToPolicyRobotOutput(actualRobotOutput, policyRobotOutput, actual2PolicyMap_);

        int idx = 0;
        // 当前速度
        obs_.segment(idx, 3) = policyRobotState.base_ang_vel * robotAngleVelScale_;
        idx += 3;

        // 重力
        Eigen::Vector3d v(0, 0, -1);
        Eigen::Vector4d qt(policyRobotState.quat.x(), policyRobotState.quat.y(), policyRobotState.quat.z(), policyRobotState.quat.w());
        obs_.segment(idx, 3) = common::rotateVectorByQuatVec4(qt, v);
        idx += 3;

        Command modifiedCmdVel = cmdVel;
        bool ltPressed = (cmdVel.joy.LT < -0.5);
        if (cmdVel.joy.LT < -0.5 && cmdVel.joy.DPU == 1.0)
        {
            mode_ = "normal";
            ltPressed = false;
        }
        else if (cmdVel.joy.LT < -0.5 && cmdVel.joy.DPD == 1.0)
        {
            mode_ = "walk";
            ltPressed = false;
        }

        // 根据控制模式确定x速度上限（仅x方向区分模式）
        double cmdVelFilterScale, cmdVelXMax = mode_ == "walk" ? cmdVelXMaxSoccerWalk_ : cmdVelXMaxSoccerNormal_,
                                  cmdVelXMin = mode_ == "walk" ? cmdVelXMinWalk_ : cmdVelXMinNormal_;

        if (controlMode_ == "Soccer")
        {
            // Soccer模式：按LT加速
            cmdVelXMax = ltPressed ? cmdVelXMaxSoccerBoost_ : cmdVelXMax;
            cmdVelXMin = ltPressed ? cmdVelXMinNormal_ : cmdVelXMin;
            cmdVelFilterScale = ltPressed ? 0.7 : cmdVelFilterScaleSoccer_; // 按下直接氮气加速
        }
        else if (controlMode_ == "Fight")
        {
            // Fight模式：固定速度，不响应LT
            cmdVelXMax = cmdVelXMaxFight_;
            cmdVelFilterScale = cmdVelFilterScaleFight_;
        }
        // 应用速度限制（y方向所有模式通用）
        modifiedCmdVel.vx = std::clamp(modifiedCmdVel.vx, cmdVelXMin, cmdVelXMax);
        modifiedCmdVel.vy = std::clamp(modifiedCmdVel.vy, cmdVelYMin_, cmdVelYMax_);
        modifiedCmdVel.dyaw = std::clamp(modifiedCmdVel.dyaw, cmdVelYawMin_, cmdVelYawMax_);

        // ROS_INFO("cmdVelXMax: %.2f, cmdVelXMin: %.2f, modifiedCmdVel.vx: %.2f, ltPressed: %d, mode: %s", cmdVelXMax, cmdVelXMin, modifiedCmdVel.vx, ltPressed ? 1 : 0, mode_.c_str());

        // 使用配置的滤波系数进行滤波
        // filterScale 越大，越关注当前值；越小，越关注历史值
        // 这里严格约定：lastCmdVel_ 始终保存「上一时刻已经滤波后的速度」（未乘缩放系数）
        double filtered_vx = (modifiedCmdVel.vx * cmdVelFilterScale + lastCmdVel_.vx * (1.0 - cmdVelFilterScale)) * cmdVelLineScale_;
        double filtered_vy = (modifiedCmdVel.vy * cmdVelFilterScale + lastCmdVel_.vy * (1.0 - cmdVelFilterScale)) * cmdVelLineScale_;
        double filtered_dyaw = (modifiedCmdVel.dyaw * cmdVelFilterScale + lastCmdVel_.dyaw * (1.0 - cmdVelFilterScale)) * cmdVelAngleScale_;

        if (std::abs(filtered_vx) < 0.08)
            filtered_vx = 0.0;
        if (std::abs(filtered_vy) < 0.08)
            filtered_vy = 0.0;
        if (std::abs(filtered_dyaw) < 0.17)
            filtered_dyaw = 0.0;

        obs_[6] = filtered_vx;
        obs_[7] = filtered_vy;
        obs_[8] = filtered_dyaw;

        idx += 3;

        // 更新 lastCmdVel_ 为本次滤波后的速度，保证下一帧使用的“历史值”与真正写入 obs 的一致（除去缩放系数）
        lastCmdVel_.vx = filtered_vx;
        lastCmdVel_.vy = filtered_vy;
        lastCmdVel_.dyaw = filtered_dyaw;

        // 机器人运动关节的位置
        obs_.segment(idx, rlDofs_) = policyRobotState.q.head(rlDofs_) * robotLinePoseScale_;
        idx += rlDofs_;
        // 机器人运动关节的速度
        obs_.segment(idx, rlDofs_) = policyRobotState.dq.head(rlDofs_) * robotLineVelScale_;
        idx += rlDofs_;
        // 机器人运动关节的上次输出
        obs_.segment(idx, rlDofs_) = lastRobotOutput_.action.head(rlDofs_);
        idx += rlDofs_;

        // 对观测数据裁剪
        clipObservations(obs_);

        obs = obs_;
        return true;
    }

    void AMPPolicy::postModifyOutput(Robot_Output& output)
    {
        auto lastRobotOutput = lastRobotOutput_;
        double filterScale = 1.0;
        lastRobotOutput_ = output; // 保存上次输出，供下次使用
        for (size_t i = 0; i < rlDofs_; ++i)
        {
            output.action[i] = (output.action[i] * filterScale + lastRobotOutput.action[i] * (1 - filterScale)) * actionSales_[i] + defaultPose_[i];
        }
        Robot_Output policyOutput(rlDofs_);
        policyOutput.action = output.action.head(rlDofs_);
        output.resize(pdDofs_); // 调整输出大小为实际机器人的关节数目
        output.action.setConstant(-999.0);
        // 修改输出顺序为实际机器人的顺序
        common::transformPolicyRobotOutputToActualRobotOutput(policyOutput, output, policy2ActualMap_);
    }
}

