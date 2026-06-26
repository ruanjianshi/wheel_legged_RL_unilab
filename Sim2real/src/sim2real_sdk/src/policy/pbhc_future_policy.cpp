#include "policy/pbhc_future_policy.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include "common.h"

namespace hightorque
{
    PBHCFuturePolicy::PBHCFuturePolicy(const config::RLConfig& rl, const config::PDConfig& pd) : PolicyBase(rl.num_single_obs, rl.frame_stack, rl.clip_obs)
    {
        frequency_ = rl.frequency;
        robotAngleVelScale_ = rl.rbt_ang_vel_scale;
        cmdVelLineScale_ = rl.cmd_lin_vel_scale;
        cmdVelAngleScale_ = rl.cmd_ang_vel_scale;
        robotLinePoseScale_ = rl.rbt_lin_pos_scale;
        robotLineVelScale_ = rl.rbt_lin_vel_scale;
        loadFutureFile(rl.future_path);

        // 历史缓冲区参数
        frameStack_ = rl.frame_stack; // 5
        rlDofs_ = rl.dofs;            // Pi机器人20个动作维度
        pdDofs_ = pd.dofs;            // 实际机器人关节数目

        // 运动长度和时间参数
        motionLen_ = rl.motion_len;
        currentMotionLen_ = 0.0;

        // 控制时间步长和计数器参数
        control_dt_ = 1.0 / rl.rl_ctrl_f; // 0.02
        counter_ = 0;

        policy2ActualMap_ = pd.policy_to_actual_map; // 策略到实际机器人的映射索引
        actual2PolicyMap_ = pd.actual_to_policy_map; // 实际机器人到策略的映射索引

        // 初始化默认角度 (与sim2sim_pi20dof.py保持一致，全零)
        defaultAngles_ = Eigen::VectorXd::Zero(rlDofs_);

        std::cout << "PBHCFuturePolicy initialized with: "
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
                  << "control_dt=" << control_dt_ << ", "
                  << "future_size=" << futureSize_ << ", "
                  << "total_steps=" << totalSteps_;

        // 初始化历史缓冲区
        initHistoryBuffer();
    }

    bool PBHCFuturePolicy::loadFutureFile(const std::string& filePath)
    {
        // 读取rlInfo.future_path
        nlohmann::json futureJson;
        std::ifstream futureFile(filePath);
        if (futureFile.is_open())
        {
            try
            {
                futureFile >> futureJson;
                int rows = 0, cols = 0;
                if (futureJson.contains("metadata"))
                {
                    if (futureJson["metadata"].contains("total_steps"))
                    {
                        totalSteps_ = futureJson["metadata"]["total_steps"].get<int>();
                        std::cout << "Loaded total_steps: " << totalSteps_ << std::endl;
                    }
                    else
                    {
                        std::cerr << "Future JSON does not contain 'total_steps' in metadata." << std::endl;
                        return false;
                    }
                    if (futureJson["metadata"].contains("data_shape") && futureJson["metadata"]["data_shape"].is_array() && futureJson["metadata"]["data_shape"].size() == 2)
                    {
                        rows = futureJson["metadata"]["data_shape"][0].get<int>();
                        cols = futureJson["metadata"]["data_shape"][1].get<int>();
                        futureSize_ = cols;
                        std::cout << "Future data shape: [" << rows << ", " << cols << "]" << std::endl;
                    }
                    else
                    {
                        std::cerr << "Future JSON does not contain valid 'data_shape' in metadata." << std::endl;
                        return false;
                    }
                }
                if (futureJson.contains("data") && futureJson["data"].is_array())
                {
                    auto dataArray = futureJson["data"];
                    if (dataArray.size() != rows)
                    {
                        std::cerr << "Data row count does not match metadata: expected " << rows << ", got " << dataArray.size() << std::endl;
                        return false;
                    }
                    futureMap_ = std::make_shared<std::map<int, std::vector<double>>>();
                    for (int i = 0; i < rows; ++i)
                    {
                        if (!dataArray[i].contains("values") || !dataArray[i]["values"].is_array())
                        {
                            std::cerr << "Data row " << i << " does not contain valid 'values' array." << std::endl;
                            return false;
                        }
                        auto values = dataArray[i]["values"];
                        if (values.size() != cols)
                        {
                            std::cerr << "Data row " << i << " column count does not match metadata: expected " << cols << ", got " << values.size() << std::endl;
                            return false;
                        }
                        std::vector<double> rowValues;
                        rowValues.reserve(cols);
                        for (const auto& val : values)
                        {
                            rowValues.push_back(val.get<double>());
                        }
                        (*futureMap_)[i] = std::move(rowValues);
                    }
                    std::cout << "Successfully loaded future data with " << futureMap_->size() << " entries." << std::endl;
                }
                else
                {
                    std::cerr << "Future JSON does not contain valid 'data' array." << std::endl;
                    return false;
                }
            }
            catch (const nlohmann::json::parse_error& e)
            {
                std::cerr << "Failed to parse future JSON file: " << e.what() << std::endl;
                return false;
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error reading future JSON file: " << e.what() << std::endl;
                return false;
            }
        }
        else
        {
            ROS_ERROR("Could not open future JSON file: %s", filePath.c_str());
            return false;
        }
        return true;
    }

    void PBHCFuturePolicy::initHistoryBuffer()
    {
        histActions_.clear();
        histBaseAngVel_.clear();
        histDofPos_.clear();
        histDofVel_.clear();
        histProjectedGrav_.clear();
        // 初始化历史缓冲区，参考sim2sim_pi20dof.py
        for (int i = 0; i < frameStack_ - 1; ++i) // 4帧历史
        {
            histActions_.push_back(Eigen::VectorXd::Zero(rlDofs_));
            histBaseAngVel_.push_back(Eigen::VectorXd::Zero(3));
            histDofPos_.push_back(Eigen::VectorXd::Zero(rlDofs_));
            histDofVel_.push_back(Eigen::VectorXd::Zero(rlDofs_));
            histProjectedGrav_.push_back(Eigen::VectorXd::Zero(3));
        }

        // 初始化最终观测输入矩阵
        // 参考sim2sim_pi20dof.py: split_point + history_size + remaining_size
        int split_point = rlDofs_ + 3 + rlDofs_ + rlDofs_; // actions + base_ang_vel + dof_pos + dof_vel = 63
        // 修正单次观测维度：actions(20) + base_ang_vel(3) + dof_pos(20) + dof_vel(20) + projected_gravity(3) + ref_motion_phase(1) = 67
        int single_obs_dim = rlDofs_ + 3 + rlDofs_ + rlDofs_ + 3;                   // 66，不是numSingleObs_
        int history_size = single_obs_dim * (frameStack_ - 1);                      // 67 * 4 = 268
        int remaining_size = 3;                                                     // projected_gravity = 3
        int future_size = futureSize_;                                              // ref_motion_phase = 1
        int total_size = split_point + history_size + remaining_size + future_size; // 63 + 268 + 4 + x = 335

        finalObsInput_.resize(1, total_size);

        // ROS_INFO_STREAM("History buffer initialized: split_point=" << split_point
        //                                                            << ", single_obs_dim=" << single_obs_dim
        //                                                            << ", history_size=" << history_size
        //                                                            << ", remaining_size=" << remaining_size
        //                                                            << ", total_size=" << total_size);
    }

    bool PBHCFuturePolicy::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby)
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

        currentMotionLen_ = counter_ * control_dt_;

        // 对观测数据裁剪
        clipObservations(obs_);

        // 先构建最终观测输入 (使用当前历史，与sim2sim_pi20dof.py一致)
        buildFinalObsInput(obs_);

        // 然后更新历史缓冲区 (在构建观测之后更新，避免时序错误)
        updateHistoryBuffer(actions_obs, ang_vel_obs, dof_pos_obs, dof_vel_obs, gravity_obs);
        obs = obs_; // 返回更新后的观测

        counter_++;
        return true;
    }

    void PBHCFuturePolicy::updateHistoryBuffer(const Eigen::VectorXd& actions,
                                               const Eigen::VectorXd& baseAngVel,
                                               const Eigen::VectorXd& dofPos,
                                               const Eigen::VectorXd& dofVel,
                                               const Eigen::VectorXd& projectedGrav)
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
    }

    void PBHCFuturePolicy::buildFinalObsInput(const Eigen::VectorXd& currentObs)
    {
        // 参考sim2sim_pi20dof.py的最终观测构建方式
        // obs_buf = np.concatenate([
        //     curr_obs[:split_point],    # 当前观测前4部分
        //     obs_hist,                  # 历史观测
        //     curr_obs[split_point:]     # 当前观测剩余部分
        // ])

        int split_point = rlDofs_ + 3 + rlDofs_ + rlDofs_; // 63
        int remaining_size = 3;                            // 3

        int idx = 0;

        // 1. 当前观测的前4个部分 (actions + base_ang_vel + dof_pos + dof_vel)
        finalObsInput_.block(0, idx, 1, split_point) = currentObs.head(split_point).transpose();
        idx += split_point;
        ROS_INFO_THROTTLE(10, "After adding current, finalObsInput_ size: %td x %td, idx=%d", finalObsInput_.rows(), finalObsInput_.cols(), idx);

        // 2. 历史观测 (按照sim2sim_pi20dof.py的顺序)
        // obs_hist = np.concatenate([
        //     np.concatenate(list(history["actions"])),
        //     np.concatenate(list(history["base_ang_vel"])),
        //     np.concatenate(list(history["dof_pos"])),
        //     np.concatenate(list(history["dof_vel"])),
        //     np.concatenate(list(history["projected_gravity"])),
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
        ROS_INFO_THROTTLE(10, "After adding history, finalObsInput_ size: %td x %td, idx=%d", finalObsInput_.rows(), finalObsInput_.cols(), idx);

        // 3. 当前观测的剩余部分 (projected_gravity)
        finalObsInput_.block(0, idx, 1, remaining_size) = currentObs.segment(split_point, remaining_size).transpose();
        idx += remaining_size;
        ROS_INFO_THROTTLE(10, "After adding gravity, finalObsInput_ size: %td x %td, idx=%d", finalObsInput_.rows(), finalObsInput_.cols(), idx);

        // 4. 未来帧 (future_frames)使用counter_索引
        if (futureMap_ && futureMap_->find(counter_) != futureMap_->end())
        {
            const std::vector<double>& futureFrames = (*futureMap_)[counter_];
            if (futureFrames.size() != futureSize_)
            {
                ROS_ERROR_THROTTLE(10, "Future frames size mismatch for counter %u: expected %d, got %zu", counter_, futureSize_, futureFrames.size());
                finalObsInput_.block(0, idx, 1, futureSize_).setZero(); // 填充零
                return;
            }
            finalObsInput_.block(0, idx, 1, futureSize_) = Eigen::Map<const Eigen::VectorXd>(futureFrames.data(), futureSize_).transpose();
            idx += futureSize_;
        }
        else
        {
            ROS_WARN_THROTTLE(10, "No future frames found for counter %u, %f(%f)", counter_, currentMotionLen_, motionLen_);
            finalObsInput_.block(0, idx + remaining_size, 1, 20).setZero(); // 填充零
        }
        ROS_INFO_THROTTLE(10, "After adding future, finalObsInput_ size: %td x %td, idx=%d", finalObsInput_.rows(), finalObsInput_.cols(), idx);
    }

    const Eigen::MatrixXd& PBHCFuturePolicy::getFinalObsInput()
    {
        return finalObsInput_;
    }

    void PBHCFuturePolicy::postModifyOutput(Robot_Output& output)
    {
        Robot_Output policyOutput(rlDofs_);
        policyOutput.action = output.action.head(rlDofs_);
        output.resize(pdDofs_); // 调整输出大小为实际机器人的关节数目
        output.action.setConstant(-999.0);

        // 修改输出顺序为实际机器人的顺序
        common::transformPolicyRobotOutputToActualRobotOutput(policyOutput, output, policy2ActualMap_);
    }

    bool PBHCFuturePolicy::isMotionFinished() const
    {
        // 运动结束条件：累计时间超过运动长度
        return currentMotionLen_ >= motionLen_;
    }

    void PBHCFuturePolicy::refreshObservation()
    {
        counter_ = 0;            // 重置计数器
        currentMotionLen_ = 0.0; // 重置累计时间
        test_ = true;
        initHistoryBuffer();
        // 默认实现不做任何操作，供派生类覆盖
        // 这里可以添加一些特定的刷新逻辑，如果需要的话
    }
}
