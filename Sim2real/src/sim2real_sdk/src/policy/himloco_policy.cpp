#include "policy/himloco_policy.h"
#include "common.h"

namespace hightorque
{
    HIMlocoPolicy::HIMlocoPolicy(const config::RLConfig& rl, const config::PDConfig& pd)
        : PolicyBase(rl.num_single_obs, rl.frame_stack, rl.clip_obs)
    {
        rlDofs_ = rl.dofs;   // 12 for HTDW_4438
        pdDofs_ = pd.dofs;

        frequency_ = rl.frequency;
        cmdLinVelScale_ = rl.cmd_lin_vel_scale;
        cmdAngVelScale_ = rl.cmd_ang_vel_scale;
        rbtLinPosScale_ = rl.rbt_lin_pos_scale;
        rbtLinVelScale_ = rl.rbt_lin_vel_scale;
        rbtAngVelScale_ = rl.rbt_ang_vel_scale;

        policy2ActualMap_ = pd.policy_to_actual_map;
        actual2PolicyMap_ = pd.actual_to_policy_map;

        // 默认站立姿态 (与训练 default_joint_angles 一致)
        defaultAngles_ = Eigen::VectorXd::Zero(rlDofs_);
        for (int i = 0; i < rlDofs_ && i < static_cast<int>(rl.default_pose.size()); ++i)
        {
            defaultAngles_[i] = rl.default_pose[i];
        }

        std::cout << "HIMlocoPolicy initialized: "
                  << "rlDofs=" << rlDofs_ << ", pdDofs=" << pdDofs_ << ", "
                  << "frequency=" << frequency_ << "Hz, "
                  << "cmdLinScale=" << cmdLinVelScale_ << ", "
                  << "cmdAngScale=" << cmdAngVelScale_ << ", "
                  << "posScale=" << rbtLinPosScale_ << ", "
                  << "velScale=" << rbtLinVelScale_ << ", "
                  << "angVelScale=" << rbtAngVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_ << ", "
                  << "frameStack=" << frameStack_
                  << std::endl;
    }

    bool HIMlocoPolicy::updateObservation(const double /*timeDiff*/,
                                          const Robot_State& robotState,
                                          const Robot_Output& robotOutput,
                                          const Command& cmdVel,
                                          Eigen::VectorXd& obs,
                                          bool /*standby*/)
    {
        if (obs_.size() != numSingleObs_)
        {
            obs_.resize(numSingleObs_);
        }

        // 将实际机器人状态/输出映射到策略空间
        Robot_State policyRobotState(rlDofs_);
        Robot_State robotStateCopy = robotState;
        Robot_Output policyRobotOutput(rlDofs_);
        Robot_Output robotOutputCopy = robotOutput;
        common::transformActualRobotStateToPolicyRobotState(robotStateCopy, policyRobotState, actual2PolicyMap_);
        common::transformActualRobotOutputToPolicyRobotOutput(robotOutputCopy, policyRobotOutput, actual2PolicyMap_);

        int idx = 0;

        // 1. 速度指令 (3维): vx, vy, dyaw
        //    cmd_lin_vel_scale 用于 vx/vy, cmd_ang_vel_scale 用于 dyaw
        obs_[idx + 0] = cmdVel.vx * cmdLinVelScale_;
        obs_[idx + 1] = cmdVel.vy * cmdLinVelScale_;
        obs_[idx + 2] = cmdVel.dyaw * cmdAngVelScale_;
        idx += 3;

        // 2. 角速度 (3维)
        obs_.segment(idx, 3) = policyRobotState.base_ang_vel * rbtAngVelScale_;
        idx += 3;

        // 3. 投影重力向量 (3维)
        Eigen::Vector3d gravity_vec(0, 0, -1);
        Eigen::Vector4d quat_vec(policyRobotState.quat.x(), policyRobotState.quat.y(),
                                  policyRobotState.quat.z(), policyRobotState.quat.w());
        obs_.segment(idx, 3) = common::rotateVectorByQuatVec4(quat_vec, gravity_vec);
        idx += 3;

        // 4. 关节位置误差 (12维): q - default_angles, 乘以位置缩放
        for (int i = 0; i < rlDofs_; ++i)
        {
            obs_[idx + i] = (policyRobotState.q[i] - defaultAngles_[i]) * rbtLinPosScale_;
        }
        idx += rlDofs_;

        // 5. 关节速度 (12维)
        for (int i = 0; i < rlDofs_; ++i)
        {
            obs_[idx + i] = policyRobotState.dq[i] * rbtLinVelScale_;
        }
        idx += rlDofs_;

        // 6. 上一步动作输出 (12维)
        obs_.segment(idx, rlDofs_) = policyRobotOutput.action.head(rlDofs_);
        idx += rlDofs_;

        // 裁剪观测
        clipObservations(obs_);

        obs = obs_;
        return true;
    }

} // namespace hightorque
