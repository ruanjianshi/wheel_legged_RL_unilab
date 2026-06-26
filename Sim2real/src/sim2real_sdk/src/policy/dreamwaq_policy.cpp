#include "policy/dreamwaq_policy.h"
#include "common.h"
#include "impl/hightorque_config.h"

namespace hightorque
{
    DreaWaQPolicy::DreaWaQPolicy(const config::RLConfig rl, const config::PDConfig pd) : PolicyBase(rl.num_single_obs, rl.frame_stack, rl.clip_obs)
    {
        frequency_ = rl.frequency;
        robotAngleVelScale_ = rl.rbt_ang_vel_scale;
        cmdVelLineScale_ = rl.cmd_lin_vel_scale;
        cmdVelAngleScale_ = rl.cmd_ang_vel_scale;
        robotLinePoseScale_ = rl.rbt_lin_pos_scale;
        robotLineVelScale_ = rl.rbt_lin_vel_scale;

        rlDofs_ = rl.dofs;
        pdDofs_ = pd.dofs;
        // 速度限制参数
        cmd_vel_x_min_ = rl.cmd_vel_x_min;
        cmd_vel_x_max_ = rl.cmd_vel_x_max;
        cmd_vel_y_min_ = rl.cmd_vel_y_min;
        cmd_vel_y_max_ = rl.cmd_vel_y_max;
        cmd_vel_yaw_min_ = rl.cmd_vel_yaw_min;
        cmd_vel_yaw_max_ = rl.cmd_vel_yaw_max;

        std::cout << "DreaWaQPolicy initialized with: "
                  << "frequency=" << frequency_ << " Hz, "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_;
    }

    DreaWaQPolicy::DreaWaQPolicy(double frequency, double robotAngleVelScale, double cmdVelLineScale, double cmdVelAngleScale, double robotLinePoseScale, double robotLineVelScale, double clipObs, int numSingleObs)
        : frequency_(frequency),
          robotAngleVelScale_(robotAngleVelScale),
          cmdVelLineScale_(cmdVelLineScale),
          cmdVelAngleScale_(cmdVelAngleScale),
          robotLinePoseScale_(robotLinePoseScale),
          robotLineVelScale_(robotLineVelScale),
          PolicyBase(numSingleObs, 6, clipObs)
    {
        std::cout << "DreaWaQPolicy initialized with: "
                  << "frequency=" << frequency_ << " Hz, "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_;
    }

    bool DreaWaQPolicy::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby)
    {
        if (obs_.size() < numSingleObs_)
        {
            obs_.resize(numSingleObs_);
        }

        // 当前速度
        obs_.segment(0, 3) = robotState.base_ang_vel * robotAngleVelScale_;

        // 重力
        Eigen::Vector3d v(0, 0, -1);
        Eigen::Vector4d qt(robotState.quat.x(), robotState.quat.y(), robotState.quat.z(), robotState.quat.w());
        obs_.segment(3, 3) = common::rotateVectorByQuatVec4(qt, v);

        // 速度裁剪（使用策略自己的配置参数）
        Command clampedCmdVel;
        clampedCmdVel.vx = std::clamp(cmdVel.vx, cmd_vel_x_min_, cmd_vel_x_max_);
        clampedCmdVel.vy = std::clamp(cmdVel.vy, cmd_vel_y_min_, cmd_vel_y_max_);
        clampedCmdVel.dyaw = std::clamp(cmdVel.dyaw, cmd_vel_yaw_min_, cmd_vel_yaw_max_);

        // 命令
        obs_[6] = clampedCmdVel.vx * cmdVelLineScale_;
        obs_[7] = clampedCmdVel.vy * cmdVelLineScale_;
        obs_[8] = clampedCmdVel.dyaw * cmdVelAngleScale_;

        // standing command mask
        obs_[9] = 0.0;

        // 正弦余弦
        obs_[10] = standby ? 1.0 : std::sin(2 * M_PI * timeDiff / frequency_);
        obs_[11] = standby ? -1.0 : std::cos(2 * M_PI * timeDiff / frequency_);

        // TODO 目前脚部关节为12, 有新的数量改为配置
        // 机器人运动关节的位置
        obs_.segment(12, 12) = robotState.q.head(12) * robotLinePoseScale_;
        // 机器人运动关节的速度
        obs_.segment(24, 12) = robotState.dq.head(12) * robotLineVelScale_;
        // 机器人运动关节的上次输出
        obs_.segment(36, 12) = robotOutput.action.head(12);

        // 对观测数据裁剪
        clipObservations(obs_);

        obs = obs_;
        return true;
    }

}
