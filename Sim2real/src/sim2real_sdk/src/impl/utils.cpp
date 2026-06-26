#include "impl/utils.h"
#include "common.h"
#include "ros/console.h"
#include "sim2real/CubicSplineInterpolator.h"
#include "motor/robot_motor_group.h"
#include <chrono>
#include <mutex>
#include <numeric>
#include <functional>

namespace hightorque
{
    namespace utils
    {
        std::map<std::string, std::pair<std::string, Motor_Output>> TrajectoryGenerator::motorOutputPoses_;                 // 电机 单一动作
        std::map<std::string, std::pair<std::string, Robot_Output>> TrajectoryGenerator::robotOutputPoses_;                 // 关节 单一动作
        std::map<std::string, std::pair<std::string, std::vector<Motor_Output>>> TrajectoryGenerator::motorOutputVec_;      // 电机 多动作
        std::map<std::string, std::pair<std::string, std::vector<Robot_Output>>> TrajectoryGenerator::robotOutputVec_;      // 关节 多动作
        std::map<std::string, std::pair<std::string, std::vector<std::vector<double>>>> TrajectoryGenerator::rawMatrixVec_; // 电机 原始点位

        TrajectoryGenerator& TrajectoryGenerator::getInstance(const std::vector<double>& offset, const std::vector<int>& direction, const std::vector<config::MultiWaypointConfig>& waypointConfigs, const std::string& modelType)
        {
            static TrajectoryGenerator instance(offset, direction, waypointConfigs, modelType);
            return instance;
        }

        bool TrajectoryGenerator::refreshSingle(const std::vector<double>& offset, const std::vector<int>& direction, const std::string& modelType, const std::vector<double>& zeroPose)
        {
            auto jointsNum = offset.size();
            motorOutputPoses_.clear();
            // 测试用 后续转yaml文件
            // 站立
            {
                Robot_Output r(jointsNum);
                if (zeroPose.size() != 0)
                {
                    for (int i = 0; i < jointsNum; ++i)
                    {
                        r.target_q[i] = zeroPose[i];
                    }
                }
                Motor_Output m;
                common::transformRobotOutputToMotorOutput(r, m, offset, direction, modelType == "hi" ? 1 : 0);
                robotOutputPoses_.emplace("zero", std::make_pair("all", r));
                motorOutputPoses_.emplace("zero", std::make_pair("all", m));
            }

            // 蹲下
            {
                Robot_Output r(jointsNum);
                if (modelType == "pi")
                {
                    r.target_q[0] = -1.25545132;
                    r.target_q[1] = 0.01696460;
                    r.target_q[2] = -0.0194778;
                    r.target_q[3] = 1.92243332;
                    r.target_q[4] = -0.6550442;
                    r.target_q[5] = -0.0496371;
                    r.target_q[6] = -1.2516813;
                    r.target_q[7] = -0.0169646;
                    r.target_q[8] = 0.02324778;
                    r.target_q[9] = 1.91677875;
                    r.target_q[10] = -0.6594423;
                    r.target_q[11] = 0.01570796;
                }
                else if (modelType == "pi_plus")
                {
                    r.target_q[0] = -1.583;
                    r.target_q[1] = -0.197;
                    r.target_q[2] = -0.099;
                    r.target_q[3] = 1.923;
                    r.target_q[4] = -0.572;
                    r.target_q[5] = -0.096;
                    r.target_q[6] = -1.583;
                    r.target_q[7] = 0.197;
                    r.target_q[8] = -0.099;
                    r.target_q[9] = 1.923;
                    r.target_q[10] = -0.572;
                    r.target_q[11] = 0.096;
                }
                else if (modelType == "hi")
                {
                    r.target_q[0] = -1.000;
                    r.target_q[1] = 0.007;
                    r.target_q[2] = -0.066;
                    r.target_q[3] = 2.012;
                    r.target_q[4] = -0.690;
                    r.target_q[5] = -0.074;
                    r.target_q[6] = -1.000;
                    r.target_q[7] = -0.016;
                    r.target_q[8] = 0.020;
                    r.target_q[9] = 1.983;
                    r.target_q[10] = -0.697;
                    r.target_q[11] = 0.005;
                }
                Motor_Output m;
                common::transformRobotOutputToMotorOutput(r, m, offset, direction, modelType == "hi" ? 1 : 0);
                robotOutputPoses_.emplace("sitdown", std::make_pair("all", r));
                motorOutputPoses_.emplace("sitdown", std::make_pair("all", m));
            }
            // 右脚前左脚后
            {
                Robot_Output r(jointsNum);
                r.target_q[0] = -0.011381;
                r.target_q[1] = -0.073513;
                r.target_q[2] = -0.086080;
                r.target_q[3] = 0.686434;
                r.target_q[4] = -0.615363;
                r.target_q[5] = -0.018850;
                r.target_q[6] = -0.559274;
                r.target_q[7] = -0.094248;
                r.target_q[8] = -0.000000;
                r.target_q[9] = 0.916398;
                r.target_q[10] = -0.402991;
                r.target_q[11] = -0.054664;
                Motor_Output m;
                common::transformRobotOutputToMotorOutput(r, m, offset, direction, modelType == "hi" ? 1 : 0);
                robotOutputPoses_.emplace("rightFront", std::make_pair("all", r));
                motorOutputPoses_.emplace("rightFront", std::make_pair("all", m));
            }
            // 左脚前右脚后
            {
                Robot_Output r(jointsNum);
                r.target_q[0] = -0.674257;
                r.target_q[1] = -0.086708;
                r.target_q[2] = -0.020735;
                r.target_q[3] = 0.464009;
                r.target_q[4] = 0.176947;
                r.target_q[5] = -0.043982;
                r.target_q[6] = -0.008867;
                r.target_q[7] = -0.032044;
                r.target_q[8] = -0.079796;
                r.target_q[9] = 0.578363;
                r.target_q[10] = -0.569496;
                r.target_q[11] = -0.062832;
                Motor_Output m;
                common::transformRobotOutputToMotorOutput(r, m, offset, direction, modelType == "hi" ? 1 : 0);
                robotOutputPoses_.emplace("leftFront", std::make_pair("all", r));
                motorOutputPoses_.emplace("leftFront", std::make_pair("all", m));
            }
            return true;
        }
        TrajectoryGenerator::TrajectoryGenerator(const std::vector<double>& offset, const std::vector<int>& direction, const std::vector<config::MultiWaypointConfig>& waypointConfigs, const std::string& modelType)
        {
            std::cout << "TrajectoryGenerator " << __FUNCTION__ << " offset: ";
            refreshSingle(offset, direction, modelType);

            // 读取动作预设
            for (auto const& config : waypointConfigs)
            {
                const auto& name = config.name;
                const auto& fileName = config.path;
                std::string group;
                std::vector<std::vector<double>> matrix; // 原始数据
                std::filesystem::path file = fileName;
                if (!common::loadMatrixWithMetadata(file, group, matrix))
                {
                    std::cout << "load file: (" << file.filename() << ") unsuccessfully" << std::endl;
                    continue;
                }
                std::cout << "name: (" << name << "), key: " << config.key << " path:" << fileName << ", group: " << group << ", size: " << matrix.size() << "x" << (matrix.size() > 0 ? matrix[0].size() : 0) << std::endl;
                std::vector<Motor_Output> mVec;
                mVec.reserve(matrix.size());
                for (const auto& vec : matrix)
                {
                    Motor_Output m;
                    m.resize(vec.size());
                    for (int j = 0; j < vec.size(); j++)
                    {
                        m.target_q[j] = vec[j];
                    }
                    mVec.push_back(m);
                }
                motorOutputVec_.emplace(name, std::make_pair(group, mVec));
                rawMatrixVec_.emplace(name, std::make_pair(group, matrix));
            }

            for (auto poses : motorOutputVec_)
            {
                std::cout << "fileName: " << poses.first << " size:" << poses.second.second.size() << std::endl;
            }
        }

        const Motor_Output& TrajectoryGenerator::getMotorPoses(const std::string& name) const
        {
            static const Motor_Output defaultPoses; // 默认构造的姿态
            auto it = motorOutputPoses_.find(name);
            if (it == motorOutputPoses_.end())
            {
                std::cout << "TrajectoryGenerator " << __FUNCTION__ << "not such Motor out: (" << name << ")" << std::endl;
                return defaultPoses;
            }
            return motorOutputPoses_.at(name).second;
        }

        const std::string TrajectoryGenerator::getMotorPosesGroup(const std::string& name) const
        {
            static const Motor_Output defaultPoses; // 默认构造的姿态
            auto it = motorOutputPoses_.find(name);
            if (it == motorOutputPoses_.end())
            {
                std::cout << "TrajectoryGenerator " << __FUNCTION__ << "not such Motor out: (" << name << ")" << std::endl;
                return "";
            }
            return motorOutputPoses_.at(name).first;
        }

        const Robot_Output& TrajectoryGenerator::getRobotPoses(const std::string& name) const
        {
            static const Robot_Output defaultPoses; // 默认构造的姿态
            auto it = robotOutputPoses_.find(name);
            if (it == robotOutputPoses_.end())
            {
                std::cout << "TrajectoryGenerator getMotorPosesGroup not such Motor out: (" << name << ")" << std::endl;
                return defaultPoses;
            }
            return robotOutputPoses_.at(name).second;
        }

        const std::string TrajectoryGenerator::getRobotPosesGroup(const std::string& name) const
        {
            static const Robot_Output defaultPoses; // 默认构造的姿态
            auto it = robotOutputPoses_.find(name);
            if (it == robotOutputPoses_.end())
            {
                std::cout << "TrajectoryGenerator getMotorPosesGroup not such Motor out: (" << name << ")" << std::endl;
                return "";
            }
            return robotOutputPoses_.at(name).first;
        }

        const std::vector<Motor_Output>& TrajectoryGenerator::getMotorPosesVec(const std::string& name) const
        {
            static const std::vector<Motor_Output> defaultPosesVec; // 默认构造的姿态
            auto it = motorOutputVec_.find(name);
            if (it == motorOutputVec_.end())
            {
                std::cout << "TrajectoryGenerator getMotorPosesGroup not such Motor out: (" << name << ")" << std::endl;
                return defaultPosesVec;
            }
            return motorOutputVec_.at(name).second;
        }

        const std::string TrajectoryGenerator::getMotorPosesVecGroup(const std::string& name) const
        {
            static const std::vector<Motor_Output> defaultPosesVec; // 默认构造的姿态
            auto it = motorOutputVec_.find(name);
            if (it == motorOutputVec_.end())
            {
                std::cout << "TrajectoryGenerator getMotorPosesGroup not such Motor out: (" << name << ")" << std::endl;
                return "";
            }
            return motorOutputVec_.at(name).first;
        }

        const std::vector<std::vector<double>>& TrajectoryGenerator::getRawPointsVec(const std::string& name) const
        {
            static const std::vector<std::vector<double>> defaultPointsVec; // 默认构造的姿态
            auto it = rawMatrixVec_.find(name);
            if (it == rawMatrixVec_.end())
            {
                std::cout << "TrajectoryGenerator getMotorPosesGroup not such Motor out: (" << name << ")" << std::endl;
                return defaultPointsVec;
            }
            return rawMatrixVec_.at(name).second;
        }

        const std::string TrajectoryGenerator::getRawPointsVecGroup(const std::string& name) const
        {
            static const std::vector<std::vector<double>> defaultPointsVec; // 默认构造的姿态
            auto it = rawMatrixVec_.find(name);
            if (it == rawMatrixVec_.end())
            {
                std::cout << "TrajectoryGenerator getMotorPosesGroup not such Motor out: (" << name << ")" << std::endl;
                return "";
            }
            return rawMatrixVec_.at(name).first;
        }

        const std::vector<Robot_Output>& TrajectoryGenerator::getRobotPosesVec(const std::string& name) const
        {
            static const std::vector<Robot_Output> defaultPosesVec; // 默认构造的姿态
            auto it = robotOutputVec_.find(name);
            if (it == robotOutputVec_.end())
            {
                std::cout << "TrajectoryGenerator getMotorPosesGroup not such Motor out: (" << name << ")" << std::endl;
                return defaultPosesVec;
            }
            return robotOutputVec_.at(name).second;
        }

        const std::string TrajectoryGenerator::getRobotPosesVecGroup(const std::string& name) const
        {
            static const std::vector<Robot_Output> defaultPosesVec; // 默认构造的姿态
            auto it = robotOutputVec_.find(name);
            if (it == robotOutputVec_.end())
            {
                std::cout << "TrajectoryGenerator getMotorPosesGroup not such Motor out: (" << name << ")" << std::endl;
                return "";
            }
            return robotOutputVec_.at(name).first;
        }

        //****************************************single*********************************

        void TrajectoryGenerator::setSingle(const Motor_State& startState, const Motor_Output& targetOutput, double duration, std::string group, TransitionType type)
        {
            if (targetOutput.target_q.size() != startState.q.size())
            {
                std::cout << "setSingle targetOutput size(" << targetOutput.target_q.size() << ") != startState size(" << startState.q.size() << ")" << std::endl;
                return;
            }
            std::unique_lock<std::shared_timed_mutex> lk(singleMutex_);
            singleStartState_ = startState;
            singleDuration_ = duration;
            singleType_ = type;
            singleGroup_ = group;
            singleStartTime_ = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            singleTargetOutput_ = targetOutput;

            singleRunning_ = true;
            multiRunning_ = false;
            toLoopDefaultState_ = false;
            stopSeries();
            std::cout << "(" << group << ") single start " << singleTargetName_ << std::endl;
        }
        void TrajectoryGenerator::setSingle(const Motor_State& startState, const std::string& targetName, double duration, std::string group, TransitionType type)
        {
            const Motor_Output& targetOutput = getMotorPoses(targetName);
            if (targetOutput.target_q.size() != startState.q.size())
            {
                std::cout << "setSingle targetOutput size(" << targetOutput.target_q.size() << ") != startState size(" << startState.q.size() << ")" << std::endl;
                return;
            }
            singleTargetName_ = targetName;
            setSingle(startState, targetOutput, duration, group, type);
        }

        Motor_Output TrajectoryGenerator::singleGeneratorMotorOut(Motor_State startState, Motor_Output targetOutput, double startTime, double currentTime, double duration, TransitionType type)
        {
            if (targetOutput.target_q.size() != startState.q.size())
            {
                std::cout << "singleGeneratorMotorOut targetOutput size(" << targetOutput.target_q.size() << ") != startState size(" << startState.q.size() << ")" << std::endl;
                return Motor_Output();
            }

            Motor_Output result;
            result.resize(startState.q.size());

            auto single = [&](std::function<double(double, double, double, double, double)> f) {
                for (int i = 0; i < startState.q.size(); i++)
                {
                    result.target_q[i] = f(startState.q[i], targetOutput.target_q[i], startTime, currentTime, duration);
                }
            };
            switch (type)
            {
                case TransitionType::Linear:
                    single(common::linearTransitionPosition);
                    break;
                case TransitionType::EaseIn:
                    single(common::easeInTransitionPosition);
                    break;
                case TransitionType::EaseOut:
                    single(common::easeOutTransitionPosition);
                    break;
                case TransitionType::SmoothStep:
                    single(common::smoothStepTransitionPosition);
                    break;
                case TransitionType::CubicSpline:
                    break;
            }
            return result;
        }

        Motor_Output TrajectoryGenerator::singleGeneratorMotorOut(Motor_State startState, std::string targetName, double startTime, double currentTime, double duration, TransitionType type)
        {
            const Motor_Output& targetOutput = getMotorPoses(targetName);
            if (targetOutput.target_q.size() != startState.q.size())
            {
                std::cout << "singleGeneratorMotorOut targetOutput size(" << targetOutput.target_q.size() << ") != startState size(" << startState.q.size() << ")" << std::endl;
                return Motor_Output();
            }
            return singleGeneratorMotorOut(startState, targetOutput, startTime, currentTime, duration, type);
        }

        Motor_Output TrajectoryGenerator::generatorSingle()
        {
            std::unique_lock<std::shared_timed_mutex> lk(singleMutex_);
            if (!singleRunning_)
            {
                return Motor_Output();
            }
            auto now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            if (now - singleStartTime_ > singleDuration_)
            {
                singleRunning_ = false;
                std::cout << "(" << singleGroup_ << ") single end " << singleTargetName_ << std::endl;
                return Motor_Output();
            }
            return singleGeneratorMotorOut(singleStartState_, singleTargetOutput_, singleStartTime_, now, singleDuration_, singleType_);
        }

        //****************************************multi*********************************

        void TrajectoryGenerator::setMulti(const Motor_State& startState, const std::vector<Motor_Output> targetMotorOutputVec, double duration, double similarThreshold, std::string group, TransitionType type, bool backToZero)
        {
            if (duration < 1e-6)
            {
                std::cout << "duration(" << duration << ") must > 0.0" << std::endl;
            }
            std::unique_lock<std::shared_timed_mutex> lk(multiMutex_);
            multiType_ = type;
            multiGroup_ = group;
            multiStartState_ = std::make_shared<Motor_State>(startState);
            multiDuration_ = duration;
            multiStartTime_ = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            similarThreshold_ = similarThreshold;

            if (targetMotorOutputVec.empty() || startState.isZero())
            {
                return;
            }
            multiTargetPoessVec_ = std::make_shared<std::vector<Motor_Output>>(targetMotorOutputVec);

            multiTargetIndex_ = 0;
            multiRunning_ = true;
            singleRunning_ = false;
            backToZero_ = backToZero;
            std::cout << "(" << group << ") multi type(" << static_cast<int>(multiType_) << ") start " << multiTargetName_ << ", size:" << multiTargetPoessVec_->size() << std::endl;
        }

        // 传入double vector点位，根据类型判断是否要计算轨迹
        void TrajectoryGenerator::setMulti(const Motor_State& startState, std::vector<std::vector<double>> targetPointsVec, double duration, double similarThreshold, std::string group, TransitionType type, bool backToZero)
        {
            multiDuration_ = duration;
            multiGroup_ = group;
            if (type == TransitionType::CubicSpline)
            {
                setMulti(startState, generatorTrajectory(startState, targetPointsVec), duration, similarThreshold, group, type, backToZero);
            }
            else
            {
                setMulti(startState, common::toMotorOutputVector(targetPointsVec), duration, similarThreshold, group, type, backToZero);
            }
        }

        // 传入名称，从库中找到对应的点位
        void TrajectoryGenerator::setMulti(const Motor_State& startState, const std::string& targetName, double duration, double similarThreshold, std::string group, TransitionType type, bool backToZero)
        {
            auto _group = getRawPointsVecGroup(targetName);
            if (_group != group)
            {
                std::cout << "group : control group (" << group << ") != trajectory group(" << _group << ")" << std::endl;
                return;
            }
            multiTargetName_ = targetName;
            multiGroup_ = group;
            multiDuration_ = duration;
            if (type == TransitionType::CubicSpline)
            {
                auto pointsRawVec = getRawPointsVec(targetName);
                if (pointsRawVec.size() == 0)
                {
                    return;
                }
                setMulti(startState, pointsRawVec, duration, similarThreshold, group, type, backToZero);
            }
            else
            {
                setMulti(startState, getMotorPosesVec(targetName), duration, similarThreshold, group, type, backToZero);
            }
        }

        Motor_Output TrajectoryGenerator::multiGeneratorMotorOut(Motor_State startState, Motor_Output targetOut, double startTime, double currentTime, double duration, TransitionType type)
        {
            Motor_Output result;
            result.resize(startState.q.size());
            if (startState.q.size() != targetOut.target_q.size())
            {
                std::cout << "multiGeneratorMotorOut targetOutput size(" << targetOut.target_q.size() << ") != startState size(" << startState.q.size() << ")" << std::endl;
                return result;
            }

            auto single = [&](std::function<double(double, double, double, double, double)> f, std::string name) {
                if (mutliVelocity_ >= 0 && name == "dq")
                {
                    result.target_dq.setConstant(mutliVelocity_);
                    return;
                }
                for (int i = 0; i < startState.q.size(); i++)
                {
                    if (name == "q")
                    {
                        result.target_q[i] = f(startState.q[i], targetOut.target_q[i], startTime, currentTime, duration);
                    }
                    if (name == "dq")
                    {
                        result.target_dq[i] = f(startState.q[i], targetOut.target_q[i], startTime, currentTime, duration);
                    }
                }
            };
            switch (type)
            {
                case TransitionType::Linear:
                    single(common::linearTransitionPosition, "q");
                    single(common::linearTransitionVelocity, "dq");
                    break;
                case TransitionType::EaseIn:
                    single(common::easeInTransitionPosition, "q");
                    single(common::easeInTransitionVelocity, "dq");
                    break;
                case TransitionType::EaseOut:
                    single(common::easeOutTransitionPosition, "q");
                    single(common::easeOutTransitionVelocity, "dq");
                    break;
                case TransitionType::SmoothStep:
                    single(common::smoothStepTransitionPosition, "q");
                    single(common::smoothStepTransitionVelocity, "dq");
                    break;
                case TransitionType::CubicSpline:
                    targetOut.target_tau.setConstant(multiConfig_.torque);
                    return targetOut;
            }
            result.target_tau.setConstant(multiConfig_.torque);
            return result;
        }

        std::vector<Motor_Output> TrajectoryGenerator::generatorTrajectory(Motor_State currentState, std::vector<std::vector<double>>& pointsRawVec)
        {
            std::vector<Motor_Output> trajectory;
            std::vector<double> timeList;

            // 将当前状态也放入其中
            // pointsRawVec.insert(pointsRawVec.begin(), common::toStdVector(currentState.q));

            auto rotatedPointsRawVec = common::rotateMatrix(pointsRawVec);

            auto jointsNum = rotatedPointsRawVec.size();
            if (jointsNum == 0)
            {
                return trajectory;
            }
            // 所有点位数量
            auto pointsNum = rotatedPointsRawVec[0].size();
            std::cout << "joints: " << jointsNum << ", points: " << pointsNum << ", duration: " << multiDuration_ << std::endl;
            timeList.resize(pointsNum);
            // 时间列表
            // 0 1 2 3 ....
            // std::iota(timeList.begin(), timeList.end(), 0);
            if (multiTimeInterval_ < 1e-6)
            {
                multiTimeInterval_ = 1;
            }
            for (int i = 0; i < pointsNum; i++)
            {
                timeList[i] = i * multiTimeInterval_;
            }

            // 最终计算的 位置 速度
            std::vector<std::vector<double>> positions;
            std::vector<std::vector<double>> velocities;
            // 关节数量
            positions.resize(jointsNum);
            velocities.resize(jointsNum);

            for (auto& pose : positions)
            {
                pose.resize(pointsNum);
            }
            for (auto& vel : velocities)
            {
                vel.resize(pointsNum);
            }

            std::stringstream ss;
            // 创建插值实例，计算每个关节的所有点位的插值
            for (auto i = 0; i < jointsNum; i++)
            {
                const auto& points = rotatedPointsRawVec[i];
                ss.str("");
                ss.clear();

                // 创建插值实例
                CubicSplineInterpolator interpolator(points, 0.0, 0.0, timeList);

                // 计算插值参数
                interpolator.calculatePosParameters();
                interpolator.calculateVelParameters();

                interpolator.generateInterpolatedPoints(multiDuration_);
                interpolator.generateInterpolatedVels(multiDuration_);
                // interpolator.generateInterpolatedPoints(0.013125);
                // interpolator.generateInterpolatedVels(0.013125);

                positions[i] = interpolator.get_poly_pos_trace();
                velocities[i] = interpolator.get_poly_vel_trace();
                if (mutliVelocity_ >= 0)
                {
                    velocities[i] = std::vector<double>(positions[i].size(), mutliVelocity_);
                }
                // ss << i;
                // ss << "\nposition: \n";
                // for (auto pos : positions[i])
                // {
                //     ss << std::to_string(pos) << ",";
                // }
                // ss << "\nvelocity: \n";
                // for (auto vel : velocities[i])
                // {
                //     ss << std::to_string(vel) << ",";
                // }
            }
            size_t generatedPointsNum = positions[0].size();
            auto rotatedPositions = common::rotateMatrix(positions);
            auto rotatedVelocities = common::rotateMatrix(velocities);
            std::cout << "generatedPoints size: " << generatedPointsNum << std::endl;

            std::vector<Motor_Output> mVec;
            mVec.reserve(pointsNum);
            for (auto i = 0; i < generatedPointsNum; i++)
            {
                Motor_Output m;
                m.resize(jointsNum);
                for (int j = 0; j < jointsNum; j++)
                {
                    m.target_q[j] = rotatedPositions[i][j];
                    m.target_dq[j] = rotatedVelocities[i][j];
                }
                mVec.push_back(m);
            }
            return mVec;
        }

        std::vector<Motor_Output> TrajectoryGenerator::generatorTrajectory(Motor_State currentState, std::string targetName)
        {
            auto pointsRawVec = getRawPointsVec(targetName);
            return generatorTrajectory(currentState, pointsRawVec);
        }

        std::vector<Motor_Output> TrajectoryGenerator::generatorTrajectory(Motor_State currentState, std::vector<Motor_State>& motorStaeVec, double duration)
        {
            multiDuration_ = duration;
            std::vector<std::vector<double>> pointsRawVec;
            pointsRawVec.reserve(motorStaeVec.size());
            for (auto state : motorStaeVec)
            {
                pointsRawVec.push_back(common::toStdVector(state.q));
            }
            return generatorTrajectory(currentState, pointsRawVec);
        }

        Motor_Output TrajectoryGenerator::generatorMulti(const Motor_State& currentState)
        {
            std::unique_lock<std::shared_timed_mutex> lk(multiMutex_);
            if (!multiRunning_ || !multiStartState_ || !multiTargetPoessVec_)
            {
                return Motor_Output();
            }
            auto now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            if (toLoopDefaultState_)
            {
                if (now - multiStartTime_ > 1)
                {
                    multiStartTime_ = now - multiDuration_ - 0.01;
                    toLoopDefaultState_ = false;
                }
                else
                {
                    return singleGeneratorMotorOut(*multiStartState_, (*multiTargetPoessVec_)[0], multiStartTime_, now, 1, TransitionType::SmoothStep);
                }
            }
            // 如果循环，需要先执行到第一个点的位置
            // 切换动作或终止
            // if (now - multiStartTime_ > multiDuration_ || common::isMotorOutputClose(multiStartState_->toOut(), (*multiTargetPoessVec_)[multiTargetIndex_]), similarThreshold_)
            if (now - multiStartTime_ > multiDuration_)
            {
                auto out = (*multiTargetPoessVec_)[multiTargetIndex_];
                multiStartState_ = std::make_shared<Motor_State>(common::toState(out));
                multiStartTime_ = now;
                multiTargetIndex_++;
                // if (multiTargetIndex_ > 30) // test
                if (multiTargetIndex_ >= multiTargetPoessVec_->size())
                {
                    if (multiConfig_.loop)
                    {
                        std::cout << "multi loop start " << multiTargetName_ << std::endl;
                        multiTargetIndex_ = 0;
                        lk.unlock();
                        return generatorMulti(currentState);
                    }
                    if (backToZero_ && !currentState.isZero())
                    {
                        setSingle(currentState, "zero", 1, "all", TransitionType::SmoothStep);
                    }
                    std::cout << "(" << multiGroup_ << ") multi end " << multiTargetName_ << std::endl;
                    multiRunning_ = false;
                    return Motor_Output();
                }
            }
            return multiGeneratorMotorOut(*multiStartState_, (*multiTargetPoessVec_)[multiTargetIndex_], multiStartTime_, now, multiDuration_, multiType_);
        }

        //****************************************series*********************************
        bool TrajectoryGenerator::setSeries(const config::SeriesWaypointConfig& seriesConfig)
        {
            std::unique_lock<std::shared_timed_mutex> lk(multiMutex_);
            seriesConfig_ = seriesConfig;
            multiRunning_ = false;
            singleRunning_ = false;
            seriesRunning_ = true;
            currentSeriesIndex_ = 0;
            multiGroup_ = getMotorPosesVecGroup(seriesConfig_.waypointVec[currentSeriesIndex_].name);
            return true;
        }

        Motor_Output TrajectoryGenerator::generatorSeries(const Motor_State& currentState, const Motor_State& allState)
        {
            auto setNextMulti = [&]() {
                if (currentSeriesIndex_ < seriesConfig_.waypointVec.size())
                {
                    setMultiConfig(seriesConfig_.waypointVec[currentSeriesIndex_]);
                    auto group = getMotorPosesVecGroup(seriesConfig_.waypointVec[currentSeriesIndex_].name);
                    std::cout << "TrajectoryGenerator: Starting next [" << currentSeriesIndex_ << "]trajectory: " << seriesConfig_.waypointVec[currentSeriesIndex_].name << ", group: " << group << std::endl;
                    bool backToZero = false;
                    if (currentSeriesIndex_ == seriesConfig_.waypointVec.size() - 1)
                    {
                        backToZero = true; // 最后一个动作返回零位
                    }
                    setMulti(currentState, seriesConfig_.waypointVec[currentSeriesIndex_].name,
                             seriesConfig_.waypointVec[currentSeriesIndex_].duration, -0.01, group,
                             utils::TransitionType(seriesConfig_.waypointVec[currentSeriesIndex_].type),
                             backToZero);
                    currentSeriesIndex_++;
                    return true;
                }
                return false;
            };

            if (!isMultiRunning())
            {
                setNextMulti();
            }
            // 继续执行当前multi
            if (isMultiRunning())
            {
                auto out = generatorMulti(allState);
                // 上一个动作执行完了
                if (out.isZero())
                {
                    // 成功加载下一个动作
                    if (setNextMulti())
                    {
                        return generatorMulti(allState);
                    }
                    // 动作结束
                    else
                    {
                        // series结束
                        stopSeries();
                        return Motor_Output();
                    }
                }
                return out;
            }
            // series结束
            stopSeries();
            return Motor_Output();
        }

        void TrajectoryGenerator::stopSeries()
        {
            if (seriesRunning_)
            {
                seriesRunning_ = false;
                currentSeriesName_.clear();
                currentSeriesFiles_.clear();
                // currentSeriesIndex_ = 0;
                std::cout << "TrajectoryGenerator: Series trajectory stopped" << std::endl;
            }
        }

        bool TrajectoryGenerator::isSeriesRunning() const
        {
            return seriesRunning_;
        }

        std::string TrajectoryGenerator::getSeriesStatus() const
        {
            if (!seriesRunning_)
            {
                return "Series not running";
            }

            std::ostringstream oss;
            oss << "Series: " << currentSeriesName_
                << ", Progress: " << (currentSeriesIndex_ + 1) << "/" << currentSeriesFiles_.size()
                << ", Current: " << (currentSeriesIndex_ < currentSeriesFiles_.size() ? currentSeriesFiles_[currentSeriesIndex_] : "N/A");
            return oss.str();
        }
    }
}
