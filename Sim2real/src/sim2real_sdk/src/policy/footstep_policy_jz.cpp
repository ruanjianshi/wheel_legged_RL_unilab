#include "policy/footstep_policy_jz.h"
#include "policy/policy_base.h"
#include "common.h"

namespace hightorque
{
    FootStepPolicyJz::FootStepPolicyJz(const config::RLConfig rl, const config::PDConfig pd)
        : stepsPeriod_(rl.step_period),
          robotAngleVelScale_(rl.rbt_ang_vel_scale),
          cmdVelLineScale_(rl.cmd_lin_vel_scale),
          cmdVelAngleScale_(rl.cmd_ang_vel_scale),
          robotLinePoseScale_(rl.rbt_lin_pos_scale),
          robotLineVelScale_(rl.rbt_lin_vel_scale),
          actionSales_(rl.action_scales),
          PolicyBase(rl.num_single_obs,
                     rl.frame_stack,
                     rl.clip_obs)
    {

        rlDofs_ = rl.dofs;
        pdDofs_ = pd.dofs;
        policy2ActualMap_ = pd.policy_to_actual_map; // 策略到实际机器人的映射索引
        actual2PolicyMap_ = pd.actual_to_policy_map; // 实际机器人到策略的映射索引
        std::cout << "FootStepPolicyJz initialized with: "
                  << "stepsPerCycle=" << stepsPeriod_ << ", "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "actionSales=" << common::toString(actionSales_).c_str() << ", "
                  << "numSingleObs=" << numSingleObs_;
    }

    FootStepPolicyJz::FootStepPolicyJz(const int stepsPerCycle, double robotAngleVelScale, double cmdVelLineScale, double cmdVelAngleScale, double robotLinePoseScale, double robotLineVelScale, double clipObs, int numSingleObs)
        : stepsPeriod_(stepsPerCycle),
          robotAngleVelScale_(robotAngleVelScale),
          cmdVelLineScale_(cmdVelLineScale),
          cmdVelAngleScale_(cmdVelAngleScale),
          robotLinePoseScale_(robotLinePoseScale),
          robotLineVelScale_(robotLineVelScale),
          PolicyBase(clipObs, 1, numSingleObs)

    {
        std::cout << "FootStepPolicyJz initialized with: "
                  << "stepsPerCycle=" << stepsPeriod_ << ", "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_;
    }
    bool FootStepPolicyJz::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby)
    {
        if (obs_.size() != numSingleObs_)
        {
            obs_.resize(numSingleObs_);
        }

        // 1. 检测有效速度命令（阈值调整为0.01 m/s或rad/s）
        const double VEL_THRESHOLD = 0.01;
        bool hasVelocityCommand = (std::abs(cmdVel.vx) > VEL_THRESHOLD ||
                                   std::abs(cmdVel.vy) > VEL_THRESHOLD ||
                                   std::abs(cmdVel.dyaw) > VEL_THRESHOLD);

        // 转换机器人状态到策略坐标系
        Robot_State policyRobotState(rlDofs_), actualRobotState = robotState;
        common::transformActualRobotStateToPolicyRobotState(actualRobotState, policyRobotState, actual2PolicyMap_);

        // 获取当前姿态角（俯仰角、滚转角）
        double currentPitch = policyRobotState.eu_ang[1]; // 俯仰角（索引1）
        double currentRoll = policyRobotState.eu_ang[0];  // 滚转角（索引0）

        // 2. 静止时检测角度扰动（阈值0.18弧度）
        bool attitudeExceeded = false;
        if (!isMoving_ && !completingCycle_) // 仅在静止模式下检测
        {
            attitudeExceeded = (std::abs(currentPitch) > 0.18 || std::abs(currentRoll) > 0.18);
        }

        // 保存上一时刻步态模式，用于检测模式切换
        bool prevGaitMode = lastGaitMode_;
        // 定义当前步态模式（true=行走模式，false=静止模式）
        lastGaitMode_ = isMoving_ || completingCycle_;

        // 3. 状态机逻辑
        if (!isMoving_ && !completingCycle_) // 静止模式
        {
            if (hasVelocityCommand || attitudeExceeded)
            {
                // 有速度命令或姿态扰动，切换到行走模式
                isMoving_ = true;
                step_ = 0.0; // 重置相位计数
                // 更新静止基准姿态（用于下次扰动检测）
                lastPitch_ = currentPitch;
                lastRoll_ = currentRoll;
            }
        }
        else if (isMoving_) // 行走模式
        {
            if (!hasVelocityCommand)
            {
                // 失去速度命令，进入周期完成模式（平滑过渡）
                isMoving_ = false;
                completingCycle_ = true;
            }
        }

        // 4. 相位计数控制（仅在行走模式或周期完成模式下更新）
        if (isMoving_ || completingCycle_)
        {
            step_ += 1.0 / stepsPeriod_;

            // 完成当前相位周期
            if (step_ >= 1.0)
            {
                step_ = 0.0; // 重置相位
                if (completingCycle_)
                {
                    // 周期完成后切换到静止模式
                    completingCycle_ = false;
                    // 记录当前姿态作为新的静止基准
                    lastPitch_ = currentPitch;
                    lastRoll_ = currentRoll;
                }
            }
        }

        // 5. 填充观测数据
        // 机身yaw角
        obs_[0] = policyRobotState.eu_ang[2];

        // 机器人角速度（缩放）
        obs_.segment(1, 3) = policyRobotState.base_ang_vel * robotAngleVelScale_;

        // 重力投影向量
        Eigen::Vector3d gravityVec(0.0, 0.0, -1.0);
        Eigen::Vector4d quatVec(policyRobotState.quat.x(), policyRobotState.quat.y(),
                                policyRobotState.quat.z(), policyRobotState.quat.w());
        obs_.segment(4, 3) = common::rotateVectorByQuatVec4(quatVec, gravityVec);

        // 控制命令（缩放）
        obs_[7] = cmdVel.vx * cmdVelLineScale_;
        obs_[8] = cmdVel.vy * cmdVelLineScale_;
        obs_[9] = cmdVel.dyaw * cmdVelAngleScale_;

        // 步态相位（静止时固定值，运动时为正弦余弦）
        bool isStationary = (!isMoving_ && !completingCycle_) || standby;
        // obs_[10] = isStationary ? 1.0 : std::sin(2 * M_PI * step_);  // 相位正弦值
        // obs_[11] = isStationary ? -1.0 : std::cos(2 * M_PI * step_); // 相位余弦值
        obs_[10] = isStationary ? 0.0 : std::sin(2 * M_PI * step_); // 相位正弦值
        obs_[11] = isStationary ? 1.0 : std::cos(2 * M_PI * step_); // 相位余弦值

        // 关节位置（缩放）
        obs_.segment(12, 12) = policyRobotState.q.head(12) * robotLinePoseScale_;

        // 关节速度（缩放）
        obs_.segment(24, 12) = policyRobotState.dq.head(12) * robotLineVelScale_;

        // 观测数据裁剪（按需启用）
        // clipObservations(obs_);

        obs = obs_;
        return true;
    }

    void FootStepPolicyJz::postModifyOutput(Robot_Output& output)
    {
        auto lastOutput = lastOutput_;
        lastOutput_ = output;
        if (lastOutput.action.size() == rlDofs_ && output.action.size() == rlDofs_)
        {
            for (size_t i = 0; i < rlDofs_; ++i)
            {
                output.action[i] = (output.action[i] * 0.8 + lastOutput.action[i] * 0.2) * actionSales_[i]; // 动作缩放
            }
        }
        Robot_Output policyOutput(rlDofs_);
        policyOutput.action = output.action.head(rlDofs_);
        // 修改输出顺序为实际机器人的顺序
        common::transformPolicyRobotOutputToActualRobotOutput(policyOutput, output, policy2ActualMap_);
    }

    /*     bool FootStepPolicyJz::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby) */
    /* { */
    /*     if (obs_.size() != numSingleObs_) */
    /*     { */
    /*         obs_.resize(numSingleObs_); */
    /*     } */
    /*  */
    /*     // 判断是否有速度命令 */
    /*     bool hasVelocityCommand = (std::abs(cmdVel.vx) > 0.08 || std::abs(cmdVel.vy) > 0.03 || std::abs(cmdVel.dyaw) > 0.1); */
    /*     Robot_State policyRobotState(rlDofs_), actualRobotState = robotState; */
    /*     common::transformActualRobotStateToPolicyRobotState(actualRobotState, policyRobotState, actual2PolicyMap_); */
    /*  */
    /*     // 获取当前俯仰角和滚转角 */
    /*     double currentPitch = policyRobotState.eu_ang[1]; // 假设俯仰角在索引1 */
    /*     double currentRoll = policyRobotState.eu_ang[0];  // 假设滚转角在索引0 */
    /*  */
    /*     // 检查姿态偏转 */
    /*     bool attitudeExceeded = false; */
    /*     if (!isMoving_ && !completingCycle_) */
    /*     { */
    /*         double pitchDiff = std::abs(currentPitch - lastPitch_); */
    /*         double rollDiff = std::abs(currentRoll - lastRoll_); */
    /*         attitudeExceeded = (pitchDiff > ATTITUDE_THRESHOLD || rollDiff > ATTITUDE_THRESHOLD); */
    /*     } */
    /*  */
    /*     // 状态机逻辑 */
    /*     if (!isMoving_ && !completingCycle_) */
    /*     { */
    /*         // 静止状态 */
    /*         if (hasVelocityCommand || attitudeExceeded) */
    /*         { */
    /*             // 开始运动或因姿态偏转而开始 */
    /*             isMoving_ = true; */
    /*             if (attitudeExceeded) */
    /*             { */
    /*                 step_ = 0.0; // 重置步数 */
    /*             } */
    /*             lastPitch_ = currentPitch; */
    /*             lastRoll_ = currentRoll; */
    /*         } */
    /*     } */
    /*     else if (isMoving_) */
    /*     { */
    /*         // 运动状态 */
    /*         if (!hasVelocityCommand) */
    /*         { */
    /*             // 速度命令消失，进入完成周期状态 */
    /*             isMoving_ = false; */
    /*             completingCycle_ = true; */
    /*         } */
    /*     } */
    /*  */
    /*     // 更新step_计数 */
    /*     if (isMoving_ || completingCycle_) */
    /*     { */
    /*         step_ += 1.0 / stepsPeriod_; */
    /*  */
    /*         // 检查是否完成一个周期 */
    /*         if (step_ >= 1.0) */
    /*         { */
    /*             step_ = 0.0; // 重置步数 */
    /*             if (completingCycle_) */
    /*             { */
    /*                 // 完成周期后停止 */
    /*                 completingCycle_ = false; */
    /*             } */
    /*         } */
    /*     } */
    /*  */
    /*     // 0. base_heading (yaw角) - 机身方位角的yaw角 */
    /*     obs_[0] = policyRobotState.eu_ang[2]; // yaw角通常在索引2 */
    /*  */
    /*     // 1-3. base_ang_vel (3维) - 机器人角速度 */
    /*     obs_.segment(1, 3) = policyRobotState.base_ang_vel * robotAngleVelScale_; */
    /*  */
    /*     // 4-6. projected_gravity (3维) - 重力投影向量 */
    /*     Eigen::Vector3d gravityVec(0.0, 0.0, -1.0); // 重力向量 */
    /*     Eigen::Vector4d quatVec( policyRobotState.quat.x(), policyRobotState.quat.y(), policyRobotState.quat.z(), policyRobotState.quat.w()); */
    /*     obs_.segment(4, 3) = common::rotateVectorByQuatVec4(quatVec, gravityVec); */
    /*  */
    /*     // 7-9. commands (3维) - 控制命令 */
    /*     obs_[7] = cmdVel.vx * cmdVelLineScale_; */
    /*     obs_[8] = cmdVel.vy * cmdVelLineScale_; */
    /*     obs_[9] = cmdVel.dyaw * cmdVelAngleScale_; */
    /*  */
    /*     // 10-11. phase (2维) - 步态相位的正弦余弦值 */
    /*     bool isStationary = (!isMoving_ && !completingCycle_) || standby; */
    /*     obs_[10] = isStationary ? 1.0 : std::sin(2 * M_PI * step_);  // phase_sin */
    /*     obs_[11] = isStationary ? -1.0 : std::cos(2 * M_PI * step_); // phase_cos */
    /*  */
    /*     // 12-23. dof_pos (12维) - 机器人运动关节的位置 */
    /*     obs_.segment(12, 12) = policyRobotState.q.head(12) * robotLinePoseScale_; */
    /*  */
    /*     // 24-35. dof_vel (12维) - 机器人运动关节的速度 */
    /*     obs_.segment(24, 12) = policyRobotState.dq.head(12) * robotLineVelScale_; */
    /*  */
    /*     // 对观测数据裁剪 */
    /*     // clipObservations(obs_); */
    /*  */
    /*     obs = obs_; */
    /*     return true; */
    /* } */
    /*  */
}
