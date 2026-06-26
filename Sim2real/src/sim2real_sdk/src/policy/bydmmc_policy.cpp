#include "policy/bydmmc_policy.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include "common.h"
#include "policy/policy_base.h"
#include <tf/transform_datatypes.h>

namespace hightorque
{

    std::map<std::string, std::shared_ptr<std::vector<std::vector<BYDMMCFuturePolicy::bydpose>>>> BYDMMCFuturePolicy::bodyPoseMap_; // 未来帧数据
    std::map<std::string, std::shared_ptr<std::vector<std::vector<BYDMMCFuturePolicy::bydquat>>>> BYDMMCFuturePolicy::bodyQuatMap_; // 未来帧数据
    std::map<std::string, std::shared_ptr<std::vector<std::vector<double>>>> BYDMMCFuturePolicy::jointPosMap_;                      // 未来帧数据
    std::map<std::string, std::shared_ptr<std::vector<std::vector<double>>>> BYDMMCFuturePolicy::jointVelMap_;                      // 未来帧数据

    void BYDMMCFuturePolicy::loadAllFutureFile(const std::vector<std::string>& filePath)
    {
        for (const auto& path : filePath)
        {
            if (bodyPoseMap_.find(path) != bodyPoseMap_.end() &&
                bodyQuatMap_.find(path) != bodyQuatMap_.end() &&
                jointPosMap_.find(path) != jointPosMap_.end() &&
                jointVelMap_.find(path) != jointVelMap_.end())
            {
                // 已经加载过该文件，跳过
                std::cout << "Future file already loaded: " << path << std::endl;
                continue;
            }
            std::cout << "Loading future file: " << path << std::endl;
            nlohmann::json futureJson;
            std::ifstream futureFile(path);
            double futureSize = 0;
            if (futureFile.is_open())
            {
                try
                {
                    futureFile >> futureJson;
                    int rows = 0, cols = 0;
                    if (futureJson.contains("body_pos_w") && futureJson["body_pos_w"].is_array())
                    {
                        auto jsonBodyPoseW = futureJson["body_pos_w"];
                        futureSize = jsonBodyPoseW.size();
                        auto bodyPoseW = std::make_shared<std::vector<std::vector<bydpose>>>(futureSize);
                        for (const auto& posese : jsonBodyPoseW)
                        {
                            bodyPoseW->at(rows).resize(posese.size()); // 每一行有rlDofs_+1个link位置
                            if (posese.is_array())
                            {
                                cols = 0; // 重置列计数器
                                for (const auto& pos : posese)
                                {
                                    if (pos.is_array() && pos.size() == 3)
                                    {
                                        bydpose p;
                                        p.x = pos[0].get<double>();
                                        p.y = pos[1].get<double>();
                                        p.z = pos[2].get<double>();
                                        bodyPoseW->at(rows)[cols] = p;
                                        cols++;
                                    }
                                    else
                                    {
                                        std::cerr << "Invalid body_pos_w position entry in future JSON. " << pos.dump() << std::endl;
                                        continue;
                                    }
                                }
                            }
                            else
                            {
                                std::cerr << "Invalid body_pos_w entry in future JSON. " << posese.dump() << std::endl;
                                continue;
                            }
                            rows++;
                        }
                        bodyPoseMap_.emplace(path, bodyPoseW);
                    }
                    rows = 0;
                    cols = 0;
                    if (futureJson.contains("body_quat_w") && futureJson["body_quat_w"].is_array())
                    {
                        auto jsonBodyQuatW = futureJson["body_quat_w"];
                        if (jsonBodyQuatW.size() != futureSize)
                        {
                            std::cerr << "Mismatched sizes in future JSON: body_quat_w size " << jsonBodyQuatW.size() << " does not match body_pos_w size " << futureSize << "." << std::endl;
                            continue;
                        }
                        auto bodyQuatW = std::make_shared<std::vector<std::vector<bydquat>>>(futureSize);
                        for (const auto& quates : jsonBodyQuatW)
                        {
                            bodyQuatW->at(rows).resize(quates.size()); // 每一行有rlDofs_ + 1个link位置
                            if (quates.is_array())
                            {
                                cols = 0; // 重置列计数器
                                for (const auto& quat : quates)
                                {
                                    if (quat.is_array() && quat.size() == 4)
                                    {
                                        bydquat q;
                                        q.w = quat[0].get<double>();
                                        q.x = quat[1].get<double>();
                                        q.y = quat[2].get<double>();
                                        q.z = quat[3].get<double>();
                                        bodyQuatW->at(rows)[cols] = q;
                                        cols++;
                                    }
                                }
                            }
                            else
                            {
                                std::cerr << "Invalid body_quat_w entry in future JSON. " << quates.dump() << std::endl;
                                continue;
                            }
                            rows++;
                        }
                        bodyQuatMap_.emplace(path, bodyQuatW);
                    }
                    rows = 0;
                    cols = 0;
                    if (futureJson.contains("joint_pos") && futureJson["joint_pos"].is_array())
                    {
                        auto jsonJointPos = futureJson["joint_pos"];
                        if (jsonJointPos.size() != futureSize)
                        {
                            std::cerr << "Mismatched sizes in future JSON: joint_pos size " << jsonJointPos.size() << " does not match body_pos_w size " << futureSize << "." << std::endl;
                            continue;
                        }
                        auto jointPos = std::make_shared<std::vector<std::vector<double>>>(futureSize);
                        for (const auto& poses : jsonJointPos)
                        {
                            jointPos->at(rows).resize(poses.size()); // 每一行有rlDofs_个joint位置
                            if (poses.is_array())
                            {
                                cols = 0; // 重置列计数器
                                for (const auto& pos : poses)
                                {
                                    jointPos->at(rows)[cols] = pos.get<double>();
                                    cols++;
                                }
                            }
                            else
                            {
                                std::cerr << "Invalid joint_pos entry in future JSON. " << poses.dump() << std::endl;
                                continue;
                            }
                            rows++;
                        }
                        jointPosMap_.emplace(path, jointPos);
                    }
                    rows = 0;
                    cols = 0;
                    if (futureJson.contains("joint_vel") && futureJson["joint_vel"].is_array())
                    {
                        auto jsonJointVel = futureJson["joint_vel"];
                        if (jsonJointVel.size() != futureSize)
                        {
                            std::cerr << "Mismatched sizes in future JSON: joint_vel size " << jsonJointVel.size() << " does not match body_pos_w size " << futureSize << "." << std::endl;
                            continue;
                        }
                        auto jointVel = std::make_shared<std::vector<std::vector<double>>>(futureSize);
                        for (const auto& vels : jsonJointVel)
                        {
                            jointVel->at(rows).resize(vels.size()); // 每一行有rlDofs_个joint位置
                            if (vels.is_array())
                            {
                                for (const auto& vel : vels)
                                {
                                    jointVel->at(rows)[cols] = vel.get<double>();
                                    cols++;
                                }
                                cols = 0; // 重置列计数器
                            }
                            else
                            {
                                std::cerr << "Invalid joint_vel entry in future JSON. " << vels.dump() << std::endl;
                                continue;
                            }
                            rows++;
                        }
                        jointVelMap_.emplace(path, jointVel);
                    }
                    else
                    {
                        std::cerr << "Future JSON does not contain required fields." << std::endl;
                        continue;
                    }
                }
                catch (const nlohmann::json::parse_error& e)
                {
                    std::cerr << "Failed to parse future JSON file: " << e.what() << std::endl;
                    continue;
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Error reading future JSON file: " << e.what() << std::endl;
                    continue;
                }
            }
            else
            {
                std::cerr << "Could not open future JSON file: " << path << std::endl;
                continue;
            }
        }
        std::cout << "Finished loading all future files." << std::endl;
    }

    /**
     * @brief 计算两个单个四元数的乘积（适配Eigen::Quaterniond）
     * @param q1 第一个四元数（Eigen::Quaterniond，格式w, x, y, z）
     * @param q2 第二个四元数（Eigen::Quaterniond，格式w, x, y, z）
     * @return 乘积四元数（Eigen::Quaterniond，格式w, x, y, z）
     * @note 完全复刻Python的quat_mul_np公式，确保结果一致
     */
    Eigen::Quaterniond quatMul(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2)
    {
        // 提取q1的分量 (w1, x1, y1, z1)
        double w1 = q1.w();
        double x1 = q1.x();
        double y1 = q1.y();
        double z1 = q1.z();

        // 提取q2的分量 (w2, x2, y2, z2)
        double w2 = q2.w();
        double x2 = q2.x();
        double y2 = q2.y();
        double z2 = q2.z();

        // 按照Python原公式计算乘积的w, x, y, z（保持完全一致）
        double ww = (z1 + x1) * (x2 + y2);
        double yy = (w1 - y1) * (w2 + z2);
        double zz = (w1 + y1) * (w2 - z2);
        double xx = ww + yy + zz;
        double qq = 0.5 * (xx + (z1 - x1) * (x2 - y2));

        // 计算结果分量
        double w = qq - ww + (z1 - y1) * (y2 - z2);
        double x = qq - xx + (x1 + w1) * (x2 + w2);
        double y = qq - yy + (w1 - x1) * (y2 + z2);
        double z = qq - zz + (z1 + y1) * (w2 - x2);

        // 返回新的四元数（注意Eigen::Quaterniond构造函数参数是w, x, y, z）
        return Eigen::Quaterniond(w, x, y, z);
    }

    /**
     * @brief 复刻PyTorch的matrix_from_quat函数，将四元数转换为旋转矩阵
     * @param quat 输入四元数（w, x, y, z顺序，Eigen::Quaterniond）
     * @return 3x3旋转矩阵（Eigen::Matrix3d），与Python结果完全一致
     * @throw std::runtime_error 若四元数模长过小（避免除零）
     */
    Eigen::Matrix3d matrixFromQuat(const Eigen::Quaterniond& quat)
    {
        // 1. 拆分四元数分量（对应Python的r=w, i=x, j=y, k=z）
        const double r = quat.w(); // 实部w
        const double i = quat.x(); // 虚部x
        const double j = quat.y(); // 虚部y
        const double k = quat.z(); // 虚部z

        // 2. 计算四元数模的平方（sum_sq = w² + x² + y² + z²）
        const double sum_sq = r * r + i * i + j * j + k * k;

        // 3. 检查数值稳定性（避免除零）
        const double eps = 1e-8;
        if (sum_sq < eps)
        {
            throw std::runtime_error(
                "matrix_from_quat: 四元数模长过小（sum_sq=" + std::to_string(sum_sq) + "）");
        }

        // 4. 计算归一化因子 two_s = 2.0 / sum_sq
        const double two_s = 2.0 / sum_sq;

        // 5. 严格按Python公式计算旋转矩阵的9个元素
        Eigen::Matrix3d mat;
        mat(0, 0) = 1.0 - two_s * (j * j + k * k); // 第一行第一列
        mat(1, 0) = two_s * (i * j - k * r);       // 第二行第一列
        mat(2, 0) = two_s * (i * k + j * r);       // 第三行第一列
        mat(0, 1) = two_s * (i * j + k * r);       // 第一行第二列
        mat(1, 1) = 1.0 - two_s * (i * i + k * k); // 第二行第二列
        mat(2, 1) = two_s * (j * k - i * r);       // 第三行第二列
        mat(0, 2) = two_s * (i * k - j * r);       // 第一行第三列
        mat(1, 2) = two_s * (j * k + i * r);       // 第二行第三列
        mat(2, 2) = 1.0 - two_s * (i * i + j * j); // 第三行第三列

        return mat;
    }
    /**
     * @brief 复刻 Python 的 quat_inv_np 函数，计算单个四元数的逆
     * @param q 输入四元数（Eigen::Quaterniond，格式 w, x, y, z）
     * @param eps 避免除零的最小阈值，默认 1e-9（与 Python 一致）
     * @return 逆四元数（Eigen::Quaterniond，格式 w, x, y, z）
     */
    // 四元数共轭函数（对应quat_conjugate_np）
    Eigen::Quaterniond quatConjugate(const Eigen::Quaterniond& q)
    {
        // 对于Eigen::Quaterniond，共轭就是取虚部为负
        return Eigen::Quaterniond(q.w(), -q.x(), -q.y(), -q.z());
    }

    // 四元数求逆函数（对应quat_inv_np）
    Eigen::Quaterniond quatInv(const Eigen::Quaterniond& q, double eps = 1e-9)
    {
        // 计算四元数的模平方
        double norm_squared = q.squaredNorm();

        // 使用eps防止除以零
        double scale = 1.0 / std::max(norm_squared, eps);

        // 四元数的逆 = 共轭 / 模平方
        Eigen::Quaterniond conjugate = quatConjugate(q);
        return Eigen::Quaterniond(conjugate.w() * scale,
                                  conjugate.x() * scale,
                                  conjugate.y() * scale,
                                  conjugate.z() * scale);
    }

    BYDMMCFuturePolicy::BYDMMCFuturePolicy(const config::RLConfig& rl, const config::PDConfig& pd)
        : actionSales_(rl.action_scales),
          PolicyBase(rl.num_single_obs, rl.frame_stack, rl.clip_obs)
    {
        frequency_ = rl.frequency;
        robotAngleVelScale_ = rl.rbt_ang_vel_scale;
        cmdVelLineScale_ = rl.cmd_lin_vel_scale;
        cmdVelAngleScale_ = rl.cmd_ang_vel_scale;
        robotLinePoseScale_ = rl.rbt_lin_pos_scale;
        robotLineVelScale_ = rl.rbt_lin_vel_scale;

        // 历史缓冲区参数
        frameStack_ = rl.frame_stack; // 5
        rlDofs_ = rl.dofs;            // policy的动作维度
        pdDofs_ = pd.dofs;            // 实际机器人关节数目

        if (!loadFutureFile(rl.future_path))
        {
            std::cerr << "Failed to load future file: " << rl.future_path << std::endl;
            return;
        }
        finalObsInput_.resize(1, numSingleObs_);
        lastRobotOutput_.resize(rlDofs_);

        // 没有腰部四元数W索引
        if (rlDofs_ == 22)
        {
            bodyQuatWIndex_ = 0;
        }
        // 有腰部四元数W索引
        else if (rlDofs_ == 23)
        {
            bodyQuatWIndex_ = 3;
        }
        else
        {
            bodyQuatWIndex_ = 0;
        }

        counter_ = 0;
        offsetYaw_ = 0.0;
        isInitialYawSet_ = false;

        policy2ActualMap_ = pd.policy_to_actual_map; // 策略到实际机器人的映射索引
        actual2PolicyMap_ = pd.actual_to_policy_map; // 实际机器人到策略的映射索引
        defaultPose_ = rl.default_pose;              // 保存默认姿态

        std::cout << "BYDMMCFuturePolicy initialized with: "
                  << "frequency=" << frequency_ << " Hz, "
                  << "actionSales=" << common::toString(actionSales_) << ", "
                  << "defaultPose=" << common::toString(defaultPose_) << ", "
                  << "robotAngleVelScale=" << robotAngleVelScale_ << ", "
                  << "cmdVelLineScale=" << cmdVelLineScale_ << ", "
                  << "cmdVelAngleScale=" << cmdVelAngleScale_ << ", "
                  << "robotLinePoseScale=" << robotLinePoseScale_ << ", "
                  << "robotLineVelScale=" << robotLineVelScale_ << ", "
                  << "clipObs=" << clipObsValue_ << ", "
                  << "numSingleObs=" << numSingleObs_ << ", "
                  << "frameStack=" << frameStack_ << ", "
                  << "motion_len=" << motionLen_ << ", "
                  << "future_size=" << futureSize_ << ", "
                  << "total_steps=" << totalSteps_ << std::endl;
    }

    bool BYDMMCFuturePolicy::loadFutureFile(const std::string& filePath)
    {
        if (bodyPoseMap_.find(filePath) != bodyPoseMap_.end() &&
            bodyQuatMap_.find(filePath) != bodyQuatMap_.end() &&
            jointPosMap_.find(filePath) != jointPosMap_.end() &&
            jointVelMap_.find(filePath) != jointVelMap_.end())
        {
            bodyPoseW_ = bodyPoseMap_[filePath];
            bodyQuatW_ = bodyQuatMap_[filePath];
            jointPos_ = jointPosMap_[filePath];
            jointVel_ = jointVelMap_[filePath];
            futureSize_ = bodyPoseW_->size();
            motionLen_ = futureSize_;
            std::cout << "Future file already loaded in static map: " << filePath << std::endl;
            return true;
        }
        std::cerr << "Future file not found in static map: " << filePath << std::endl;
        return false;
    }

    bool BYDMMCFuturePolicy::updateObservation(const double timeDiff, const Robot_State& robotState, const Robot_Output& robotOutput, const Command& cmdVel, Eigen::VectorXd& obs, bool standby)
    {
        if (obs_.size() != numSingleObs_)
        {
            obs_.resize(numSingleObs_);
        }
        if (counter_ == 0)
        {
            resetInitialYaw(robotState);
        }

        Robot_State policyRobotState(rlDofs_), actualRobotState = robotState;
        Robot_Output policyRobotOutput(rlDofs_), actualRobotOutput = robotOutput;

        common::transformActualRobotStateToPolicyRobotState(actualRobotState, policyRobotState, actual2PolicyMap_);
        common::transformActualRobotOutputToPolicyRobotOutput(actualRobotOutput, policyRobotOutput, actual2PolicyMap_);
        // 按照sim2sim_pi20dof.py的观测顺序构建观测量
        int idx = 0;
        // 1. command(2 * rlDofs_) - 直接使用当前机器人状态
        Eigen::VectorXd cmd_pos_vel_obs(rlDofs_ * 2);
        for (int i = 0; i < rlDofs_; ++i)
        {
            cmd_pos_vel_obs[i] = (*jointPos_)[counter_][i] * robotLinePoseScale_;          // 使用当前位置
            cmd_pos_vel_obs[i + rlDofs_] = (*jointVel_)[counter_][i] * robotLineVelScale_; // 使用当前速度
        }
        obs_.segment(idx, rlDofs_ * 2) = cmd_pos_vel_obs;
        idx += rlDofs_ * 2;

        // 2. command(6维度) - 使用当前机器人姿态（yaw角度相对于初始角度）
        // 将 Eigen 四元数转换为 tf 四元数
        tf::Quaternion tf_quat_original(
            policyRobotState.quat.x(),
            policyRobotState.quat.y(),
            policyRobotState.quat.z(),
            policyRobotState.quat.w());

        // 转换为欧拉角
        double roll, pitch, justifiedYaw;
        tf::Matrix3x3(tf_quat_original).getRPY(roll, pitch, justifiedYaw);

        // 将 yaw 加初始角度（相对于初始位置的角度）
        justifiedYaw = justifiedYaw + offsetYaw_;
        // roll = roll + offsetRoll_;
        // pitch = pitch + offsetPitch_;
        // 对 yaw 角度进行 ±π 限幅
        // while (justifiedYaw > M_PI + offsetYaw_)
        //     justifiedYaw -= 2.0 * (M_PI + offsetYaw_);
        // while (justifiedYaw < -M_PI + offsetYaw_)
        //     justifiedYaw += 2.0 * (M_PI + offsetYaw_);

        while (justifiedYaw > M_PI)
            justifiedYaw -= 2.0 * (M_PI);
        while (justifiedYaw < -M_PI)
            justifiedYaw += 2.0 * (M_PI);

        while (roll > M_PI)
            roll -= 2.0 * M_PI;
        while (roll < -M_PI)
            roll += 2.0 * M_PI;
        while (pitch > M_PI / 2)
            pitch -= 1.0 * M_PI;
        while (pitch < -M_PI / 2)
            pitch += 1.0 * M_PI;
        //
        // 用新的欧拉角创建新的四元数
        tf::Quaternion justifiedRobotQuat;
        justifiedRobotQuat.setRPY(roll, pitch, justifiedYaw);

        if (numSingleObs_ == rlDofs_ * 5 + 9)
        {
            // 转换回 Eigen 四元数
            Eigen::Quaterniond q01(
                justifiedRobotQuat.w(),
                justifiedRobotQuat.x(),
                justifiedRobotQuat.y(),
                justifiedRobotQuat.z());

            double roll_json, pitch_json, yaw_json;
            tf::Quaternion tf_quat_json(
                (*bodyQuatW_)[counter_][bodyQuatWIndex_].x,
                (*bodyQuatW_)[counter_][bodyQuatWIndex_].y,
                (*bodyQuatW_)[counter_][bodyQuatWIndex_].z,
                (*bodyQuatW_)[counter_][bodyQuatWIndex_].w);
            tf::Matrix3x3(tf_quat_json).getRPY(roll_json, pitch_json, yaw_json);

            // std::cout << "[" << counter_ << "]" << (*bodyQuatW_)[counter_][bodyQuatWIndex_].w << ", " << (*bodyQuatW_)[counter_][bodyQuatWIndex_].x << ", " << (*bodyQuatW_)[counter_][bodyQuatWIndex_].y << ", " << (*bodyQuatW_)[counter_][bodyQuatWIndex_].z << " | " << yaw_json << ", " << pitch_json << ", " << roll_json << "|" << yaw << ", " << pitch << ", " << roll << std::endl;
            auto q10 = quatInv(q01);
            const auto& quat = (*bodyQuatW_)[counter_][bodyQuatWIndex_];
            Eigen::Quaterniond q02(quat.w, quat.x, quat.y, quat.z);
            auto q12 = quatMul(q10, q02);
            // auto mat = matrixFromQuat(q12);
            auto mat = q12.toRotationMatrix();

            Eigen::VectorXd motion_ref_ori_obs(6);
            motion_ref_ori_obs << mat(0, 0), mat(0, 1), mat(1, 0),
                mat(1, 1), mat(2, 0), mat(2, 1);
            obs_.segment(idx, 6) = motion_ref_ori_obs;
            idx += 6;
        }
        else if (numSingleObs_ == rlDofs_ * 5 + 3 + 1 + 3 + 1 + 3)
        {
            // 2. json的重力投影
            Eigen::Vector3d g(0, 0, -1);
            double jsonRoll, jsonPitch, jsonYaw;
            const auto& jsonBydQuat = (*bodyQuatW_)[counter_][bodyQuatWIndex_];
            Eigen::Vector4d jsonVec4(jsonBydQuat.x, jsonBydQuat.y, jsonBydQuat.z, jsonBydQuat.w);
            auto ggg = common::rotateVectorByQuatVec4(jsonVec4, g);
            obs_.segment(idx, 3) = ggg;
            idx += 3;
            // 2. json的yaw角
            tf::Quaternion jsonQuat(jsonBydQuat.x, jsonBydQuat.y, jsonBydQuat.z, jsonBydQuat.w);
            tf::Matrix3x3(jsonQuat).getRPY(jsonRoll, jsonPitch, jsonYaw);
            while (jsonYaw > M_PI)
                jsonYaw -= 2.0 * M_PI;
            while (jsonYaw < -M_PI)
                jsonYaw += 2.0 * M_PI;
            obs_[idx++] = jsonYaw;

            // 2. 机器人的重力投影
            Eigen::Vector4d robotVec4(justifiedRobotQuat.x(), justifiedRobotQuat.y(),
                                      justifiedRobotQuat.z(), justifiedRobotQuat.w());
            auto ggg2 = common::rotateVectorByQuatVec4(robotVec4, g);
            obs_.segment(idx, 3) = ggg2;
            idx += 3;

            // 2. 机器人的yaw角
            obs_[idx++] = justifiedYaw;

            std::cout << "[" << counter_ << "]"
                      << " " << ggg.transpose() << " | JSON yaw: " << jsonYaw
                      << " | " << ggg2.transpose() << " | Robot yaw: " << justifiedYaw
                      << std::endl;
        }
        else if (numSingleObs_ == rlDofs_ * 5 + 6 + 3 + 3 + 3)
        {
            // 2. 四元数
            // 转换回 Eigen 四元数
            Eigen::Quaterniond q01(
                justifiedRobotQuat.w(),
                justifiedRobotQuat.x(),
                justifiedRobotQuat.y(),
                justifiedRobotQuat.z());

            double roll_json, pitch_json, yaw_json;
            tf::Quaternion tf_quat_json(
                (*bodyQuatW_)[counter_][bodyQuatWIndex_].x,
                (*bodyQuatW_)[counter_][bodyQuatWIndex_].y,
                (*bodyQuatW_)[counter_][bodyQuatWIndex_].z,
                (*bodyQuatW_)[counter_][bodyQuatWIndex_].w);
            tf::Matrix3x3(tf_quat_json).getRPY(roll_json, pitch_json, yaw_json);

            // std::cout << "[" << counter_ << "]" << (*bodyQuatW_)[counter_][bodyQuatWIndex_].w << ", " << (*bodyQuatW_)[counter_][bodyQuatWIndex_].x << ", " << (*bodyQuatW_)[counter_][bodyQuatWIndex_].y << ", " << (*bodyQuatW_)[counter_][bodyQuatWIndex_].z << " | " << yaw_json << ", " << pitch_json << ", " << roll_json << "|" << yaw << ", " << pitch << ", " << roll << std::endl;
            auto q10 = quatInv(q01);
            const auto& quat = (*bodyQuatW_)[counter_][bodyQuatWIndex_];
            Eigen::Quaterniond q02(quat.w, quat.x, quat.y, quat.z);
            auto q12 = quatMul(q10, q02);
            // auto mat = matrixFromQuat(q12);
            auto mat = q12.toRotationMatrix();

            Eigen::VectorXd motion_ref_ori_obs(6);
            motion_ref_ori_obs << mat(0, 0), mat(0, 1), mat(1, 0),
                mat(1, 1), mat(2, 0), mat(2, 1);
            obs_.segment(idx, 6) = motion_ref_ori_obs;
            idx += 6;

            // 2. json的重力投影
            Eigen::Vector3d g(0, 0, -1);
            double jsonRoll, jsonPitch, jsonYaw;
            const auto& jsonBydQuat = (*bodyQuatW_)[counter_][bodyQuatWIndex_];
            Eigen::Vector4d jsonVec4(jsonBydQuat.x, jsonBydQuat.y, jsonBydQuat.z, jsonBydQuat.w);
            auto ggg = common::rotateVectorByQuatVec4(jsonVec4, g);
            obs_.segment(idx, 3) = ggg;
            idx += 3;
            // 2. 机器人的重力投影
            Eigen::Vector4d robotVec4(justifiedRobotQuat.x(), justifiedRobotQuat.y(),
                                      justifiedRobotQuat.z(), justifiedRobotQuat.w());
            auto ggg2 = common::rotateVectorByQuatVec4(robotVec4, g);
            obs_.segment(idx, 3) = ggg2;
            idx += 3;
        }
        // 3. base_ang_vel (3维) - 当前角速度
        Eigen::VectorXd ang_vel_obs = policyRobotState.base_ang_vel * robotAngleVelScale_;
        obs_.segment(idx, 3) = ang_vel_obs;
        idx += 3;

        // 4. dof_pos (rlDofs_) - 机器人运动关节的位置
        Eigen::VectorXd dof_pos_obs = (policyRobotState.q.head(rlDofs_) * robotLinePoseScale_) - Eigen::Map<const Eigen::VectorXd>(defaultPose_.data(), rlDofs_);
        obs_.segment(idx, rlDofs_) = dof_pos_obs;
        idx += rlDofs_;

        // 5. dof_vel (rlDofs_) - 机器人运动关节的速度
        Eigen::VectorXd dof_vel_obs = policyRobotState.dq.head(rlDofs_) * robotLineVelScale_;
        obs_.segment(idx, rlDofs_) = dof_vel_obs;
        idx += rlDofs_;

        // 6. actions (rlDofs_) - 机器人运动关节的上次输出
        Eigen::VectorXd actions_obs = lastRobotOutput_.action.head(rlDofs_);
        obs_.segment(idx, rlDofs_) = actions_obs;
        idx += rlDofs_;

        // 对观测数据裁剪
        clipObservations(obs_);

        obs = obs_; // 返回更新后的观测

        counter_++;
        return true;
    }

    const Eigen::MatrixXd& BYDMMCFuturePolicy::getFinalObsInput()
    {
        // 放入obs_
        finalObsInput_.block(0, 0, 1, numSingleObs_) = obs_.transpose();
        return finalObsInput_;
    }

    void BYDMMCFuturePolicy::postModifyOutput(Robot_Output& output)
    {
        lastRobotOutput_ = output; // 保存上次输出，供下次使用
        for (size_t i = 0; i < rlDofs_; ++i)
        {
            output.action[i] = output.action[i] * actionSales_[i] + defaultPose_[i];
        }
        Robot_Output policyOutput(rlDofs_);
        policyOutput.action = output.action.head(rlDofs_);
        output.resize(pdDofs_); // 调整输出大小为实际机器人的关节数目
        output.action.setConstant(-999.0);
        // 修改输出顺序为实际机器人的顺序
        common::transformPolicyRobotOutputToActualRobotOutput(policyOutput, output, policy2ActualMap_);
    }

    bool BYDMMCFuturePolicy::isMotionFinished() const
    {
        // 运动结束条件：累计时间超过运动长度
        return counter_ >= motionLen_;
    }

    void BYDMMCFuturePolicy::refreshObservation()
    {
        counter_ = 0; // 重置计数器
        test_ = true;
        // 默认实现不做任何操作，供派生类覆盖
        // 这里可以添加一些特定的刷新逻辑，如果需要的话
    }

    void BYDMMCFuturePolicy::resetInitialYaw(const Robot_State& robotState)
    {
        // 将当前机器人的四元数转换为 yaw 角度
        tf::Quaternion robotQuat(
            robotState.quat.x(),
            robotState.quat.y(),
            robotState.quat.z(),
            robotState.quat.w());

        // 转换为欧拉角并提取 yaw
        double robotRoll, robotPitch, robotYaw;
        tf::Matrix3x3(robotQuat).getRPY(robotRoll, robotPitch, robotYaw);

        // 设置当前 yaw 为初始 yaw，并叠加 JSON 第一帧基座四元数的 yaw
        if (bodyQuatW_ && !bodyQuatW_->empty() && !bodyQuatW_->at(0).empty())
        {
            const auto& quat0 = bodyQuatW_->at(0).at(bodyQuatWIndex_);
            tf::Quaternion json0Quat(
                quat0.x,
                quat0.y,
                quat0.z,
                quat0.w);

            double jsonRoll, jsonPitch, jsonYaw;
            tf::Matrix3x3(json0Quat).getRPY(jsonRoll, jsonPitch, jsonYaw);
            offsetYaw_ = jsonYaw - robotYaw;
            offsetPitch_ = jsonPitch - robotPitch;
            offsetRoll_ = jsonRoll - robotRoll;
        }
        while (offsetYaw_ > M_PI)
            offsetYaw_ -= 2.0 * M_PI;
        while (offsetYaw_ < -M_PI)
            offsetYaw_ += 2.0 * M_PI;
        while (offsetPitch_ > M_PI)
            offsetPitch_ -= 2.0 * M_PI;
        while (offsetPitch_ < -M_PI)
            offsetPitch_ += 2.0 * M_PI;
        while (offsetRoll_ > M_PI)
            offsetRoll_ -= 2.0 * M_PI;
        while (offsetRoll_ < -M_PI)
            offsetRoll_ += 2.0 * M_PI;

        isInitialYawSet_ = true;
        std::cerr << robotRoll << "  Initial roll set to current roll : " << offsetRoll_ << " rad (" << offsetRoll_ * 180.0 / M_PI << " deg)" << std::endl;
        std::cerr << robotPitch << "  Initial pitch set to current pitch: " << offsetPitch_ << " rad (" << offsetPitch_ * 180.0 / M_PI << " deg)" << std::endl;
        std::cerr << robotYaw << "  Initial yaw set to current yaw: " << offsetYaw_ << " rad (" << offsetYaw_ * 180.0 / M_PI << " deg)" << std::endl;
    }

    void BYDMMCFuturePolicy::onEnterRunning(const Robot_State& robotState)
    {
        resetInitialYaw(robotState);
    }

    Robot_Output BYDMMCFuturePolicy::getFirstRobotOutput()
    {
        if (jointPos_ == nullptr || jointPos_->empty())
        {
            std::cerr << "Error: jointPos_ is null or empty in getFirstRobotOutput(). Returning zero output." << std::endl;
            return Robot_Output(pdDofs_); // 返回零输出
        }
        Robot_Output policyOutput(rlDofs_), actualOutput(pdDofs_);
        for (int i = 0; i < rlDofs_; ++i)
        {
            policyOutput.target_q[i] = (*jointPos_)[0][i] * robotLinePoseScale_; // 使用当前位置
        }
        std::cout << "First policy output target_q: " << (policyOutput.toString()) << std::endl;
        // 修改输出顺序为实际机器人的顺序
        common::transformPolicyRobotOutputToActualRobotOutput(policyOutput, actualOutput, policy2ActualMap_);
        std::cout << "First actual output target_q: " << (actualOutput.toString()) << std::endl;
        return actualOutput;
    }
}
