#include "policy/footstep_policy.h"
#include "policy/policy_base.h"

namespace hightorque
{
    FootStepPolicy::FootStepPolicy(const config::RLConfig rl, const config::PDConfig pd)
        : stepsPeriod_(rl.step_period),
          robotAngleVelScale_(rl.rbt_ang_vel_scale),
          cmdVelLineScale_(rl.cmd_lin_vel_scale),
          cmdVelAngleScale_(rl.cmd_ang_vel_scale),
          robotLinePoseScale_(rl.rbt_lin_pos_scale),
          robotLineVelScale_(rl.rbt_lin_vel_scale),
          cmd_vel_x_min_(rl.cmd_vel_x_min),
          cmd_vel_x_max_(rl.cmd_vel_x_max),
          cmd_vel_y_min_(rl.cmd_vel_y_min),
          cmd_vel_y_max_(rl.cmd_vel_y_max),
          cmd_vel_yaw_min_(rl.cmd_vel_yaw_min),
          cmd_vel_yaw_max_(rl.cmd_vel_yaw_max),
          PolicyBase(rl.num_single_obs,
                     rl.frame_stack,
                     rl.clip_obs)
    {
        rlDofs_ = rl.dofs;
        pdDofs_ = pd.dofs;
        std::cout << "FootStepPolicy initialized with: "
                  << "stepsPerCycle=" << stepsPeriod_ << ", "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_;
    }

    FootStepPolicy::FootStepPolicy(const int stepsPerCycle, double robotAngleVelScale, double cmdVelLineScale, double cmdVelAngleScale, double robotLinePoseScale, double robotLineVelScale, double clipObs, int numSingleObs)
        : stepsPeriod_(stepsPerCycle),
          robotAngleVelScale_(robotAngleVelScale),
          cmdVelLineScale_(cmdVelLineScale),
          cmdVelAngleScale_(cmdVelAngleScale),
          robotLinePoseScale_(robotLinePoseScale),
          robotLineVelScale_(robotLineVelScale),
          PolicyBase(clipObs, 1, numSingleObs)

    {
        std::cout << "FootStepPolicy initialized with: "
                  << "stepsPerCycle=" << stepsPeriod_ << ", "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_;
    }
    bool FootStepPolicy::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby)
    {
        if (obs_.size() != numSingleObs_)
        {
            obs_.resize(numSingleObs_);
        }

        step_ += 1.0 / stepsPeriod_;

        // 正弦余弦
        obs_[0] = standby ? 1.0 : std::sin(2 * M_PI * step_);
        obs_[1] = standby ? -1.0 : std::cos(2 * M_PI * step_);

        // 速度裁剪（使用策略自己的配置参数）
        Command clampedCmdVel;
        clampedCmdVel.vx = std::clamp(cmdVel.vx, cmd_vel_x_min_, cmd_vel_x_max_);
        clampedCmdVel.vy = std::clamp(cmdVel.vy, cmd_vel_y_min_, cmd_vel_y_max_);
        clampedCmdVel.dyaw = std::clamp(cmdVel.dyaw, cmd_vel_yaw_min_, cmd_vel_yaw_max_);

        // 命令
        obs_[2] = clampedCmdVel.vx * cmdVelLineScale_ * (clampedCmdVel.vx < 0 ? 0.5 : 1.0);
        obs_[3] = clampedCmdVel.vy * cmdVelLineScale_;
        obs_[4] = clampedCmdVel.dyaw * cmdVelAngleScale_;

        // 机器人运动关节的位置
        obs_.segment(5, 12) = robotState.q.head(12) * robotLinePoseScale_;
        // 机器人运动关节的速度
        obs_.segment(17, 12) = robotState.dq.head(12) * robotLineVelScale_;
        // 当前速度
        obs_.segment(29, 3) = robotState.base_ang_vel * robotAngleVelScale_;
        // 机器人当前姿态
        obs_.segment(32, 3) = robotState.eu_ang;

        // 对观测数据裁剪
        clipObservations(obs_);

        obs = obs_;
        return true;
    }
}
