#include "policy/lr_policy.h"
#include "policy/policy_base.h"
#include "common.h"

namespace hightorque
{
    LrPolicy::LrPolicy(const config::RLConfig rl, const config::PDConfig pd) : PolicyBase(rl.num_single_obs,
                                                                                          rl.frame_stack,
                                                                                          rl.clip_obs)
    {
        rlDofs_ = rl.dofs;
        pdDofs_ = pd.dofs;
        frameStack_ = rl.frame_stack;                // 1
        policy2ActualMap_ = pd.policy_to_actual_map; // 策略到实际机器人的映射索引
        actual2PolicyMap_ = pd.actual_to_policy_map; // 实际机器人到策略的映射索引

        // 速度限制参数
        cmd_vel_x_min_ = rl.cmd_vel_x_min;
        cmd_vel_x_max_ = rl.cmd_vel_x_max;
        cmd_vel_y_min_ = rl.cmd_vel_y_min;
        cmd_vel_y_max_ = rl.cmd_vel_y_max;
        cmd_vel_yaw_min_ = rl.cmd_vel_yaw_min;
        cmd_vel_yaw_max_ = rl.cmd_vel_yaw_max;
        ROS_INFO_STREAM("LrPolicy initialized with: "
                        << "frameStack_ :" << frameStack_ << " ,"
                        << "rl dofs:" << rlDofs_ << " ,"
                        << "pd dofs:" << pdDofs_ << " ,"
                        << "numSingleObs=" << numSingleObs_);
    }

    bool LrPolicy::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby)
    {
        if (obs_.size() != numSingleObs_)
        {
            obs_.resize(numSingleObs_);
        }
        Robot_State policyRobotState(rlDofs_), actualRobotState = robotState;
        common::transformActualRobotStateToPolicyRobotState(actualRobotState, policyRobotState, actual2PolicyMap_);

        int idx = 0;
        obs_.segment(idx, rlDofs_) = policyRobotState.q.head(rlDofs_);
        idx += rlDofs_;

        obs_.segment(idx, rlDofs_) = policyRobotState.dq.head(rlDofs_);
        idx += rlDofs_;

        obs_[idx + 0] = policyRobotState.base_ang_vel[0];
        obs_[idx + 1] = policyRobotState.base_ang_vel[1];
        obs_[idx + 2] = policyRobotState.base_ang_vel[2];
        // obs_.segment(idx, 3) = policyRobotState.base_ang_vel.head(3);
        idx += 3;

        obs_[idx + 0] = policyRobotState.eu_ang[0];
        obs_[idx + 1] = policyRobotState.eu_ang[1];
        obs_[idx + 2] = policyRobotState.eu_ang[2];
        // obs_.segment(idx, 3) = policyRobotState.eu_ang.head(3);
        idx += 3;

        // 速度裁剪（使用策略自己的配置参数）
        Command clampedCmdVel;
        clampedCmdVel.vx = std::clamp(cmdVel.vx, cmd_vel_x_min_, cmd_vel_x_max_);
        clampedCmdVel.vy = std::clamp(cmdVel.vy, cmd_vel_y_min_, cmd_vel_y_max_);
        clampedCmdVel.dyaw = std::clamp(cmdVel.dyaw, cmd_vel_yaw_min_, cmd_vel_yaw_max_);

        // cmdVel vx vy dyaw (注意：lr_policy 的顺序是 vy, vx, dyaw)
        obs_.segment(idx, 3) = Eigen::Vector3d(clampedCmdVel.vy, clampedCmdVel.vx, clampedCmdVel.dyaw);
        obs = obs_;

        // joy 相关的在update action中实现
        return true;
    }

    void LrPolicy::postModifyOutput(Robot_Output& output)
    {
        Robot_Output policyOutput(rlDofs_);
        policyOutput.action = output.action.head(rlDofs_);
        output.resize(pdDofs_); // 调整输出大小为实际机器人的关节数目
        output.action.setConstant(-999.0);
        // 修改输出顺序为实际机器人的顺序
        common::transformPolicyRobotOutputToActualRobotOutput(policyOutput, output, policy2ActualMap_);
    }
}
