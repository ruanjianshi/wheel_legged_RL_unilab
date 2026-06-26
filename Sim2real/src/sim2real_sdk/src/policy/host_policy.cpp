#include "policy/host_policy.h"
#include "policy/policy_base.h"
#include "common.h"
#include <algorithm>

namespace hightorque
{
    HostPolicy::HostPolicy(const config::RLConfig rl, const config::PDConfig pd)
        : robotAngleVelScale_(rl.rbt_ang_vel_scale),
          cmdVelLineScale_(rl.cmd_lin_vel_scale),
          cmdVelAngleScale_(rl.cmd_ang_vel_scale),
          robotLinePoseScale_(rl.rbt_lin_pos_scale),
          robotLineVelScale_(rl.rbt_lin_vel_scale),
          PolicyBase(rl.num_single_obs,
                     rl.frame_stack,
                     rl.clip_obs)
    {
        rlDofs_ = rl.dofs;
        pdDofs_ = pd.dofs;
        lower_ = (rl.clip_actions_lower);
        upper_ = (rl.clip_actions_upper);
        std::cout << "HostPolicy initialized with: "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_;
    }

    HostPolicy::HostPolicy(const int stepsPerCycle, double robotAngleVelScale, double cmdVelLineScale, double cmdVelAngleScale, double robotLinePoseScale, double robotLineVelScale, double clipObs, int numSingleObs)
        : robotAngleVelScale_(robotAngleVelScale),
          cmdVelLineScale_(cmdVelLineScale),
          cmdVelAngleScale_(cmdVelAngleScale),
          robotLinePoseScale_(robotLinePoseScale),
          robotLineVelScale_(robotLineVelScale),
          PolicyBase(clipObs, 1, numSingleObs)

    {
        std::cout << "HostPolicy initialized with: "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_;
    }
    bool HostPolicy::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby)
    {
        if (obs_.size() != numSingleObs_)
        {
            obs_.resize(numSingleObs_);
        }
        // 当前速度
        obs_.segment(0, 3) = robotState.base_ang_vel * robotAngleVelScale_;

        // 重力
        Eigen::Vector3d v(0, 0, -1);
        Eigen::Vector4d qt(robotState.quat.x(), robotState.quat.y(), robotState.quat.z(), robotState.quat.w());
        obs_.segment(3, 3) = common::rotateVectorByQuatVec4(qt, v);
        // 机器人运动关节的位置
        obs_.segment(6, rlDofs_) = robotState.q.head(rlDofs_) * robotLinePoseScale_;
        // 机器人运动关节的速度
        obs_.segment(18, rlDofs_) = robotState.dq.head(rlDofs_) * robotLineVelScale_;
        // 机器人运动关节的上次输出
        obs_.segment(30, rlDofs_) = robotOutput.action.head(rlDofs_);
        obs_[42] = 0.25;
        curentState_ = robotState;

        // 对观测数据裁剪
        clipObservations(obs_);

        obs = obs_;
        return true;
    }

    void HostPolicy::preModifyOutput(Robot_Output& output)
    {
        if (output.action.size() != rlDofs_ || curentState_.q.size() != rlDofs_)
        {
            std::cout << "HostPolicy::modifyOutput: output.action size is not " << rlDofs_ << ", size: " << output.action.size() << ", currentState_.q size: " << curentState_.q.size() << std::endl;
            return;
        }
        for (int i = 0; i < rlDofs_; i++)
        {
            output.action[i] = output.action[i] * 0.25 + curentState_.q[i];
        }
    }
}
