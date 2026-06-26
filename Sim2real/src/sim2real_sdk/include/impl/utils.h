#ifndef UTILS_H_
#define UTILS_H_

#include "impl/hightorque_config.h"
#include "robot_data.h"
#include <shared_mutex>
#include <functional>

namespace hightorque
{
    // 前向声明
    class RobotMotorGroup;

    namespace utils
    {
        enum class TransitionType
        {
            Linear = 1,     // 线性
            EaseIn = 2,     // 慢入快出
            EaseOut = 3,    // 快入慢出
            SmoothStep = 4, // 慢入慢出
            CubicSpline = 5 // 三次样条插值
        };
        class TrajectoryGenerator
        {
            public:
                /** ======================== 公有接口 ======================== */

                /**
                 * @brief 获取单例实例（首次调用时加载偏移、方向和文件名）
                 * @param offset    各电机偏移量列表
                 * @param direction 各电机方向列表（1 或 -1）
                 * @param fileNames 轨迹文件名列表
                 * @param modelType 机器人类型，pi pi+ hi
                 * @return 单例 TrajectoryGenerator 引用
                 */
                static TrajectoryGenerator& getInstance(const std::vector<double>& offset, const std::vector<int>& direction, const std::vector<config::MultiWaypointConfig>& waypointConfigs, const std::string& modelType);

                // 删除拷贝构造和赋值运算符，防止复制
                TrajectoryGenerator(const TrajectoryGenerator&) = delete;
                TrajectoryGenerator& operator=(const TrajectoryGenerator&) = delete;

                /**
                 * @brief 获取预定义电机输出（位置）
                 * @param name 预定义姿态名称
                 * @return 对应 Motor_Output 引用
                 */
                const Motor_Output& getMotorPoses(const std::string& name) const;

                const std::string getMotorPosesGroup(const std::string& name) const;

                /**
                 * @brief 获取预定义机器人输出（位置）
                 * @param name 预定义姿态名称
                 * @return 对应 Robot_Output 引用
                 */
                const Robot_Output& getRobotPoses(const std::string& name) const;

                const std::string getRobotPosesGroup(const std::string& name) const;

                /**
                 * @brief 获取原始轨迹点列表
                 * @param name 轨迹名称
                 * @return 二维坐标点列表
                 */
                const std::vector<std::vector<double>>& getRawPointsVec(const std::string& name) const;

                const std::string getRawPointsVecGroup(const std::string& name) const;

                /**
                 * @brief 获取预定义电机输出向量列表
                 * @param name 轨迹名称
                 * @return Motor_Output 向量列表
                 */
                const std::vector<Motor_Output>& getMotorPosesVec(const std::string& name) const;

                const std::string getMotorPosesVecGroup(const std::string& name) const;

                /**
                 * @brief 获取预定义机器人输出向量列表
                 * @param name 轨迹名称
                 * @return Robot_Output 向量列表
                 */
                const std::vector<Robot_Output>& getRobotPosesVec(const std::string& name) const;

                const std::string getRobotPosesVecGroup(const std::string& name) const;

                /** ----- 单动作接口 ----- */

                /**
                 * @brief 设置单个动作（目标为预定义姿态）
                 * @param startState   起始电机状态
                 * @param targetName   目标姿态名称
                 * @param duration     过渡时长（秒）
                 * @param group        作用电机组名称
                 * @param type         过渡类型（默认为 EaseOut）
                 */
                void setSingle(const Motor_State& startState, const std::string& targetName, double duration, std::string group = "all", TransitionType type = TransitionType::EaseOut);

                /**
                 * @brief 设置单个动作（目标为自定义输出）
                 * @param startState   起始电机状态
                 * @param targetOutput 目标电机输出
                 * @param duration     过渡时长（秒）
                 * @param group        作用电机组名称
                 * @param type         过渡类型（默认为 EaseOut）
                 */
                void setSingle(const Motor_State& startState, const Motor_Output& targetOutput, double duration, std::string group = "all", TransitionType type = TransitionType::EaseOut);

                /**
                 * @brief 生成单动作当前输出
                 * @return 当前时刻的 Motor_Output
                 */
                Motor_Output generatorSingle();

                /**
                 * @brief 单动作是否正在运行
                 * @return true 表示正在运行
                 */
                inline bool isSingleRunning()
                {
                    std::shared_lock<std::shared_timed_mutex> lk(singleMutex_);
                    return singleRunning_;
                }

                /**
                 * @brief 获取当前单动作所属电机组
                 * @return 电机组名称
                 */
                inline std::string getSingleGroup()
                {
                    std::shared_lock<std::shared_timed_mutex> lk(singleMutex_);
                    return singleGroup_;
                }

                inline void setSingleGroup(const std::string& group)
                {
                    std::shared_lock<std::shared_timed_mutex> lk(singleMutex_);
                    singleGroup_ = group;
                }

                /**
                 * @brief 停止单动作执行
                 */
                inline void stopSingle()
                {
                    std::unique_lock<std::shared_timed_mutex> lk(singleMutex_);
                    singleRunning_ = false;
                }

                /** ----- 多动作接口 ----- */

                /**
                 * @brief 设置多动作（目标为预定义轨迹）
                 * @param startState       起始电机状态
                 * @param targetName       轨迹名称
                 * @param duration         总时长（秒）
                 * @param similarThreshold 相似度阈值
                 * @param group            作用电机组名称
                 * @param type             过渡类型（默认为 Linear）
                 * @param backToZero       是否回零
                 */
                void setMulti(const Motor_State& startState,
                              const std::string& targetName,
                              double duration,
                              double similarThreshold,
                              std::string group = "all",
                              TransitionType type = TransitionType::Linear,
                              bool backToZero = true);

                /**
                 * @brief 设置多动作（目标为原始点列表）
                 */
                void setMulti(const Motor_State& startState,
                              std::vector<std::vector<double>> targetPointsVec,
                              double duration,
                              double similarThreshold,
                              std::string group = "all",
                              TransitionType type = TransitionType::Linear,
                              bool backToZero = true);

                /**
                 * @brief 设置多动作（目标为自定义输出列表）
                 */
                void setMulti(const Motor_State& startState,
                              const std::vector<Motor_Output> targetMotorOutputVec,
                              double duration,
                              double similarThreshold,
                              std::string group,
                              TransitionType type = TransitionType::Linear,
                              bool backToZero = true);

                /**
                 * @brief 生成多动作当前输出
                 * @param currentState 当前电机状态
                 * @return 当前时刻的 Motor_Output
                 */
                Motor_Output generatorMulti(const Motor_State& currentState);

                bool refreshSingle(const std::vector<double>& offset, const std::vector<int>& direction, const std::string& modelType, const std::vector<double>& zeroPose = {});

                /**
                 * @brief 多动作是否正在运行
                 * @return true 表示正在运行
                 */
                inline bool isMultiRunning()
                {
                    std::shared_lock<std::shared_timed_mutex> lk(multiMutex_);
                    return multiRunning_; // 多动作运行标志
                }

                /**
                 * @brief 获取当前多动作所属电机组
                 * @return 电机组名称
                 */
                inline std::string getMultiGroup()
                {
                    std::shared_lock<std::shared_timed_mutex> lk(multiMutex_);
                    return multiGroup_; // 多动作组名
                }

                inline void setMultiGroup(std::string group)
                {
                    std::shared_lock<std::shared_timed_mutex> lk(multiMutex_);
                    multiGroup_ = group;
                }

                /**
                 * @brief 设置一些参数，为了方便，不在setmulti中设置了
                 */
                inline void setMultiConfig(const config::MultiWaypointConfig& config)
                {
                    std::shared_lock<std::shared_timed_mutex> lk(multiMutex_);
                    loop = config.loop;
                    backToZero_ = config.back_to_zero;
                    backToWalk_ = config.back_to_walk;
                    multiTimeInterval_ = config.time_interval;
                    mutliVelocity_ = config.velocity;
                    multiConfig_ = config;
                    toLoopDefaultState_ = true;
                }

                /**
                 * @brief 停止多动作执行
                 */
                inline void stopMulti()
                {
                    std::unique_lock<std::shared_timed_mutex> lk(multiMutex_);
                    multiRunning_ = false;        // 停止运行
                    multiTargetPoessVec_.reset(); // 清除目标序列
                }

                /** ----- Series轨迹接口 ----- */

                bool setSeries(const config::SeriesWaypointConfig& seriesConfig);

                /**
                 * @brief 生成多动作当前输出
                 * @param currentState 当前电机状态
                 * @return 当前时刻的 Motor_Output
                 */
                Motor_Output generatorSeries(const Motor_State& currentState, const Motor_State& allState);

                /**
                 * @brief 停止series轨迹执行
                 */
                void stopSeries();

                /**
                 * @brief 检查series是否正在运行
                 * @return true 表示正在运行
                 */
                bool isSeriesRunning() const;

                /**
                 * @brief 获取series执行状态信息
                 * @return 状态信息字符串
                 */
                std::string getSeriesStatus() const;

                // 实际执行轨迹生成的函数
                std::vector<Motor_Output> generatorTrajectory(Motor_State currentState,
                                                              std::vector<std::vector<double>>&);

                // 通过预设的名称 计算position和Velocity的轨迹
                std::vector<Motor_Output> generatorTrajectory(Motor_State currentState,
                                                              std::string targetName);

                // 通过多个电机状态，计算position和Velocity的轨迹
                std::vector<Motor_Output> generatorTrajectory(Motor_State currentState,
                                                              std::vector<Motor_State>&,
                                                              double duration);

            private:
                /** ======================== 私有实现 ======================== */

                /**
                 * @brief 私有构造，仅可通过 getInstance 调用
                 */
                TrajectoryGenerator(const std::vector<double>& offset,
                                    const std::vector<int>& direction,
                                    const std::vector<config::MultiWaypointConfig>& waypointConfigs,
                                    const std::string& modelType);

                // 单动作内部生成函数
                Motor_Output singleGeneratorMotorOut(Motor_State currentState,
                                                     std::string targetName,
                                                     double startTime,
                                                     double currentTime,
                                                     double duration,
                                                     TransitionType type = TransitionType::EaseOut);
                Motor_Output singleGeneratorMotorOut(Motor_State currentState,
                                                     Motor_Output targetOut,
                                                     double startTime,
                                                     double currentTime,
                                                     double duration,
                                                     TransitionType type = TransitionType::EaseOut);

                // 多动作内部生成函数
                Motor_Output multiGeneratorMotorOut(Motor_State currentState,
                                                    Motor_Output targetOut,
                                                    double startTime,
                                                    double currentTime,
                                                    double duration,
                                                    TransitionType type = TransitionType::EaseOut);

                std::shared_timed_mutex singleMutex_; // 线程安全读写锁
                std::shared_timed_mutex multiMutex_;  // 线程安全读写锁

                /** ----- 预定义单一姿态存储 ----- */
                static std::map<std::string, std::pair<std::string, Motor_Output>> motorOutputPoses_; // 名称→(电机组 + 电机输出)
                static std::map<std::string, std::pair<std::string, Robot_Output>> robotOutputPoses_; // 名称→(电机组 + 机器人输出)
                bool singleRunning_;                                                                  // 单动作运行标志
                std::string singleTargetName_;                                                        // 当前单动作目标名称
                double singleDuration_;                                                               // 单动作时长
                double singleStartTime_;                                                              // 单动作起始时间
                TransitionType singleType_;                                                           // 单动作过渡类型
                Motor_State singleStartState_;                                                        // 单动作起始状态
                Motor_Output singleTargetOutput_;                                                     // 单动作目标输出
                std::string singleGroup_;                                                             // 单动作作用组名

                /** ----- 预定义多姿态存储 ----- */
                static std::map<std::string, std::pair<std::string, std::vector<Motor_Output>>> motorOutputVec_;      // 名称→(电机组原始电机输出列表)
                static std::map<std::string, std::pair<std::string, std::vector<Robot_Output>>> robotOutputVec_;      // 名称→(电机组机器人输出列表)
                static std::map<std::string, std::pair<std::string, std::vector<std::vector<double>>>> rawMatrixVec_; // 名称→(电机组原始点列表)

                std::shared_ptr<Motor_State> multiStartState_;                   // 多动作起始状态
                bool toLoopDefaultState_;                                        // 如果是循环执行的动作，先顺滑的执行到起始点位
                bool multiRunning_;                                              // 多动作运行标志
                bool backToZero_;                                                // 多动作结束是否回零
                bool backToWalk_;                                                // 多动作结束是否回到走路
                bool loop;                                                       // 多动作结束是否循环执行
                std::string multiTargetName_;                                    // 当前多动作目标名称
                double multiDuration_;                                           // 多动作时长
                TransitionType multiType_;                                       // 多动作过渡类型
                double multiStartTime_;                                          // 多动作起始时间
                std::shared_ptr<std::vector<Motor_Output>> multiTargetPoessVec_; // 多动作目标输出序列
                size_t multiTargetIndex_;                                        // 多动作当前索引
                double similarThreshold_;                                        // 相似度阈值
                std::string multiGroup_;                                         // 多动作作用组名
                // type 5
                double multiTimeInterval_;                // 多动作插帧时间间隔
                double mutliVelocity_;                    // 多动作插帧固定速度
                config::MultiWaypointConfig multiConfig_; // 多动作配置

                /** ----- Series轨迹执行相关 ----- */
                bool seriesRunning_ = false;                  // series是否正在运行
                std::string currentSeriesName_;               // 当前series名称
                std::vector<std::string> currentSeriesFiles_; // 当前series的文件列表
                size_t currentSeriesIndex_ = 0;               // 当前执行到第几个文件
                config::SeriesWaypointConfig seriesConfig_;

                /** ======================== TrajectoryGenerator 实现结束 ======================== */
        };
    }
}

#endif
