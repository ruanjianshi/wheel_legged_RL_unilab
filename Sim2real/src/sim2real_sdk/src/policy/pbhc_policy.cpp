#include "policy/pbhc_policy.h"
#include "common.h"

namespace hightorque
{
    PBHCPolicy::PBHCPolicy(const config::RLConfig& rl, const config::PDConfig& pd) : PolicyBase(rl.num_single_obs, rl.frame_stack, rl.clip_obs)
    {
        frequency_ = rl.frequency;
        robotAngleVelScale_ = rl.rbt_ang_vel_scale;
        cmdVelLineScale_ = rl.cmd_lin_vel_scale;
        cmdVelAngleScale_ = rl.cmd_ang_vel_scale;
        robotLinePoseScale_ = rl.rbt_lin_pos_scale;
        robotLineVelScale_ = rl.rbt_lin_vel_scale;

        // 历史缓冲区参数
        frameStack_ = rl.frame_stack; // 5
        rlDofs_ = rl.dofs;            // Pi机器人20个动作维度
        pdDofs_ = pd.dofs;            // 实际机器人关节数目

        // 运动长度和时间参数
        motionLen_ = rl.motion_len;
        currentMotionLen_ = 0.0;

        // 控制时间步长和计数器参数
        control_dt_ = 1.0 / rl.rl_ctrl_f;
        counter_ = 0;

        policy2ActualMap_ = pd.policy_to_actual_map; // 策略到实际机器人的映射索引
        actual2PolicyMap_ = pd.actual_to_policy_map; // 实际机器人到策略的映射索引

        // 初始化默认角度 (与sim2sim_pi20dof.py保持一致，全零)
        defaultAngles_ = Eigen::VectorXd::Zero(rlDofs_);

        std::cout << "PBHCPolicy initialized with: "
                  << "frequency=" << frequency_ << " Hz, "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_ << ", "
                  << "frameStack=" << frameStack_ << ", "
                  << "motion_len=" << motionLen_ << ", "
                  << "control_dt=" << control_dt_;

        // 初始化历史缓冲区
        initHistoryBuffer();
    }

    PBHCPolicy::PBHCPolicy(double frequency, double robotAngleVelScale, double cmdVelLineScale, double cmdVelAngleScale, double robotLinePoseScale, double robotLineVelScale, double clipObs, int numSingleObs)
        : frequency_(frequency),
          robotAngleVelScale_(robotAngleVelScale),
          cmdVelLineScale_(cmdVelLineScale),
          cmdVelAngleScale_(cmdVelAngleScale),
          robotLinePoseScale_(robotLinePoseScale),
          robotLineVelScale_(robotLineVelScale),
          PolicyBase(numSingleObs, 5, clipObs)
    {
        // 默认参数
        frameStack_ = 5;
        rlDofs_ = 20;

        // 运动长度和时间参数（使用默认值）
        motionLen_ = 7.2; // 默认运动长度
        currentMotionLen_ = 0.0;

        // 控制时间步长和计数器参数
        control_dt_ = 0.01; // 默认控制时间步长
        counter_ = 0;

        // 初始化默认角度 (与sim2sim_pi20dof.py保持一致，全零)
        defaultAngles_ = Eigen::VectorXd::Zero(rlDofs_);

        std::cout << "PBHCPolicy initialized with: "
                  << "frequency=" << frequency_ << " Hz, "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_ << ", "
                  << "frameStack=" << frameStack_ << ", "
                  << "motion_len=" << motionLen_ << ", "
                  << "control_dt=" << control_dt_;

        // 初始化历史缓冲区
        initHistoryBuffer();
    }

    void PBHCPolicy::initHistoryBuffer()
    {
        histActions_.clear();
        histBaseAngVel_.clear();
        histDofPos_.clear();
        histDofVel_.clear();
        histProjectedGrav_.clear();
        histRefPhase_.clear();
        // 初始化历史缓冲区，参考sim2sim_pi20dof.py
        for (int i = 0; i < frameStack_ - 1; ++i) // 4帧历史
        {
            histActions_.push_back(Eigen::VectorXd::Zero(rlDofs_));
            histBaseAngVel_.push_back(Eigen::VectorXd::Zero(3));
            histDofPos_.push_back(Eigen::VectorXd::Zero(rlDofs_));
            histDofVel_.push_back(Eigen::VectorXd::Zero(rlDofs_));
            histProjectedGrav_.push_back(Eigen::VectorXd::Zero(3));
            histRefPhase_.push_back(Eigen::VectorXd::Zero(1));
        }

        // 初始化最终观测输入矩阵
        // 参考sim2sim_pi20dof.py: split_point + history_size + remaining_size
        int split_point = rlDofs_ + 3 + rlDofs_ + rlDofs_; // actions + base_ang_vel + dof_pos + dof_vel = 63
        // 修正单次观测维度：actions(20) + base_ang_vel(3) + dof_pos(20) + dof_vel(20) + projected_gravity(3) + ref_motion_phase(1) = 67
        int single_obs_dim = rlDofs_ + 3 + rlDofs_ + rlDofs_ + 3 + 1; // 67，不是numSingleObs_
        int history_size = single_obs_dim * (frameStack_ - 1);        // 67 * 4 = 268
        int remaining_size = 3 + 1;                                   // projected_gravity + ref_motion_phase = 4
        int total_size = split_point + history_size + remaining_size; // 63 + 268 + 4 = 335

        finalObsInput_.resize(1, total_size);

        // ROS_INFO_STREAM("History buffer initialized: split_point=" << split_point
        //                                                            << ", single_obs_dim=" << single_obs_dim
        //                                                            << ", history_size=" << history_size
        //                                                            << ", remaining_size=" << remaining_size
        //                                                            << ", total_size=" << total_size);
    }

    bool PBHCPolicy::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby)
    {
        if (obs_.size() != numSingleObs_)
        {
            obs_.resize(numSingleObs_);
        }

        Robot_State policyRobotState(rlDofs_), actualRobotState = robotState;
        Robot_Output policyRobotOutput(rlDofs_), actualRobotOutput = robotOutput;

        common::transformActualRobotStateToPolicyRobotState(actualRobotState, policyRobotState, actual2PolicyMap_);
        common::transformActualRobotOutputToPolicyRobotOutput(actualRobotOutput, policyRobotOutput, actual2PolicyMap_);
        // 按照sim2sim_pi20dof.py的观测顺序构建观测量
        int idx = 0;

        // 1. actions (20维) - 机器人运动关节的上次输出
        Eigen::VectorXd actions_obs = policyRobotOutput.action.head(rlDofs_);
        obs_.segment(idx, rlDofs_) = actions_obs;
        idx += rlDofs_;

        // 2. base_ang_vel (3维) - 当前角速度
        Eigen::VectorXd ang_vel_obs = policyRobotState.base_ang_vel * robotAngleVelScale_;
        obs_.segment(idx, 3) = ang_vel_obs;
        idx += 3;

        // 3. dof_pos (20维) - 机器人运动关节的位置
        Eigen::VectorXd dof_pos_obs = (policyRobotState.q.head(rlDofs_) - defaultAngles_) * robotLinePoseScale_;
        obs_.segment(idx, rlDofs_) = dof_pos_obs;
        idx += rlDofs_;

        // 4. dof_vel (20维) - 机器人运动关节的速度
        Eigen::VectorXd dof_vel_obs = policyRobotState.dq.head(rlDofs_) * robotLineVelScale_;
        obs_.segment(idx, rlDofs_) = dof_vel_obs;
        idx += rlDofs_;

        // 5. projected_gravity (3维) - 重力
        Eigen::Vector3d gravity_vec(0, 0, -1);
        Eigen::Vector4d quat_vec(policyRobotState.quat.x(), policyRobotState.quat.y(), policyRobotState.quat.z(), policyRobotState.quat.w());
        Eigen::VectorXd gravity_obs = common::rotateVectorByQuatVec4(quat_vec, gravity_vec);
        obs_.segment(idx, 3) = gravity_obs;
        idx += 3;

        // 6. ref_motion_phase (1维) - 运动相位
        // 更新计数器
        counter_++;

        // 计算线性相位：((counter * control_dt) % motion_len) / motion_len
        Eigen::VectorXd phase_obs(1);

        currentMotionLen_ = counter_ * control_dt_;
        double phase_value = std::fmod(currentMotionLen_, motionLen_) / motionLen_;
        phase_obs[0] = phase_value;

        obs_[idx] = phase_obs[0];
        idx += 1;

        // 对观测数据裁剪
        clipObservations(obs_);

        // 先构建最终观测输入 (使用当前历史，与sim2sim_pi20dof.py一致)
        buildFinalObsInput(obs_);

        // 然后更新历史缓冲区 (在构建观测之后更新，避免时序错误)
        updateHistoryBuffer(actions_obs, ang_vel_obs, dof_pos_obs, dof_vel_obs, gravity_obs, phase_obs);
        obs = obs_; // 返回更新后的观测

        return true;
    }

    void PBHCPolicy::updateHistoryBuffer(const Eigen::VectorXd& actions,
                                         const Eigen::VectorXd& baseAngVel,
                                         const Eigen::VectorXd& dofPos,
                                         const Eigen::VectorXd& dofVel,
                                         const Eigen::VectorXd& projectedGrav,
                                         const Eigen::VectorXd& refPhase)
    {
        // 参考sim2sim_pi20dof.py的history更新方式：appendleft
        histActions_.push_front(actions);
        if (histActions_.size() > frameStack_ - 1)
            histActions_.pop_back();

        histBaseAngVel_.push_front(baseAngVel);
        if (histBaseAngVel_.size() > frameStack_ - 1)
            histBaseAngVel_.pop_back();

        histDofPos_.push_front(dofPos);
        if (histDofPos_.size() > frameStack_ - 1)
            histDofPos_.pop_back();

        histDofVel_.push_front(dofVel);
        if (histDofVel_.size() > frameStack_ - 1)
            histDofVel_.pop_back();

        histProjectedGrav_.push_front(projectedGrav);
        if (histProjectedGrav_.size() > frameStack_ - 1)
            histProjectedGrav_.pop_back();

        histRefPhase_.push_front(refPhase);
        if (histRefPhase_.size() > frameStack_ - 1)
            histRefPhase_.pop_back();
    }

    void PBHCPolicy::buildFinalObsInput(const Eigen::VectorXd& currentObs)
    {
        // 参考sim2sim_pi20dof.py的最终观测构建方式
        // obs_buf = np.concatenate([
        //     curr_obs[:split_point],    # 当前观测前4部分
        //     obs_hist,                  # 历史观测
        //     curr_obs[split_point:]     # 当前观测剩余部分
        // ])

        int split_point = rlDofs_ + 3 + rlDofs_ + rlDofs_; // 63
        int remaining_size = 3 + 1;                        // 4

        int idx = 0;

        // 1. 当前观测的前4个部分 (actions + base_ang_vel + dof_pos + dof_vel)
        finalObsInput_.block(0, idx, 1, split_point) = currentObs.head(split_point).transpose();
        idx += split_point;

        // 2. 历史观测 (按照sim2sim_pi20dof.py的顺序)
        // obs_hist = np.concatenate([
        //     np.concatenate(list(history["actions"])),
        //     np.concatenate(list(history["base_ang_vel"])),
        //     np.concatenate(list(history["dof_pos"])),
        //     np.concatenate(list(history["dof_vel"])),
        //     np.concatenate(list(history["projected_gravity"])),
        //     np.concatenate(list(history["ref_motion_phase"]))
        // ])

        // 按观测类型分组concatenate历史数据
        for (const auto& hist : histActions_)
        {
            finalObsInput_.block(0, idx, 1, rlDofs_) = hist.transpose();
            idx += rlDofs_;
        }
        for (const auto& hist : histBaseAngVel_)
        {
            finalObsInput_.block(0, idx, 1, 3) = hist.transpose();
            idx += 3;
        }
        for (const auto& hist : histDofPos_)
        {
            finalObsInput_.block(0, idx, 1, rlDofs_) = hist.transpose();
            idx += rlDofs_;
        }
        for (const auto& hist : histDofVel_)
        {
            finalObsInput_.block(0, idx, 1, rlDofs_) = hist.transpose();
            idx += rlDofs_;
        }
        for (const auto& hist : histProjectedGrav_)
        {
            finalObsInput_.block(0, idx, 1, 3) = hist.transpose();
            idx += 3;
        }
        for (const auto& hist : histRefPhase_)
        {
            finalObsInput_.block(0, idx, 1, 1) = hist.transpose();
            idx += 1;
        }

        // 3. 当前观测的剩余部分 (projected_gravity + ref_motion_phase)
        finalObsInput_.block(0, idx, 1, remaining_size) = currentObs.segment(split_point, remaining_size).transpose();
    }

    const Eigen::MatrixXd& PBHCPolicy::getFinalObsInput()
    {
        return finalObsInput_;
    }

    void PBHCPolicy::postModifyOutput(Robot_Output& output)
    {
        Robot_Output policyOutput(rlDofs_);
        policyOutput.action = output.action.head(rlDofs_);
        output.resize(pdDofs_); // 调整输出大小为实际机器人的关节数目
        output.action.setConstant(-999.0);
        // 修改输出顺序为实际机器人的顺序
        common::transformPolicyRobotOutputToActualRobotOutput(policyOutput, output, policy2ActualMap_);
    }

    bool PBHCPolicy::isMotionFinished() const
    {
        // 运动结束条件：累计时间超过运动长度
        return currentMotionLen_ >= motionLen_;
    }

    void PBHCPolicy::refreshObservation()
    {
        counter_ = 0;            // 重置计数器
        currentMotionLen_ = 0.0; // 重置累计时间
        test_ = true;
        initHistoryBuffer();
        // 默认实现不做任何操作，供派生类覆盖
        // 这里可以添加一些特定的刷新逻辑，如果需要的话
    }
}
