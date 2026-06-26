#ifndef HIGHTORQUE_CONFIG_H
#define HIGHTORQUE_CONFIG_H

#include <yaml-cpp/yaml.h>
#include "robot_data.h"
#include "common.h"

namespace hightorque
{
    namespace config
    {
        // 此配置文件的变量用下划线命名方式，方便配置文件中书写与sim2sim的编码风格对齐
        //=============================================MultiWaypointConfig================================================
        class MultiWaypointConfig
        {
            public:
                std::string key;              // 按键组合
                std::string name;             // 轨迹名称
                std::string path;             // 轨迹文件路径
                int type = 1;                 // 插帧类型
                double duration = 0.0;        // 点与点之间的持续时间, 模式4为插值后，点与点之间的持续时间
                double torque = 0.0;          // 点与点之间的持续时间, 模式4为插值后，点与点之间的持续时间
                double time_interval = 0.0;   // 模式5 下使用的，点与点之间的时间间隔
                double velocity = -1;         // 模式5 下使用, 固定速度
                bool back_to_zero = false;    // 是否回零
                bool back_to_walk = false;    // 是否回到行走
                bool loop = false;            // 是否循环
                std::set<std::string> keySet; // 按键集合，方便查询

                const std::string toString() const
                {
                    std::stringstream ss;
                    ss << "  Key: " << key << "\n";
                    ss << "    Path: " << path << "\n";
                    ss << "    Name: " << name << "\n";
                    ss << "    Type: " << type << "\n";
                    ss << "    Duration: " << duration << "\n";
                    ss << "    Torque: " << torque << "\n";
                    ss << "    Time Interval: " << time_interval << "\n";
                    ss << "    velocity: " << velocity << "\n";
                    ss << "    Back To Zero: " << (back_to_zero ? "true" : "false") << "\n";
                    ss << "    Back To Walk: " << (back_to_walk ? "true" : "false") << "\n";
                    ss << "    Loop: " << (loop ? "true" : "false") << "\n";
                    ss << "    Key: " << key << "\n";
                    for (const auto& k : keySet)
                    {
                        ss << "     Key Set Item: " << k << "\n";
                    }
                    if (type == 5)
                    {
                        ss << "    Time Interval: " << time_interval << "\n";
                    }
                    if (type == 5)
                    {
                        ss << "    Velocity: " << velocity << "\n";
                    }

                    return ss.str();
                }
        };
        //=============================================SeriesWaypointConfig================================================
        class SeriesWaypointConfig
        {
            public:
                std::string name;                             // 轨迹名称
                std::string key;                              // 轨迹按键
                std::vector<MultiWaypointConfig> waypointVec; // 轨迹点列表
                std::set<std::string> keySet;                 // 按键集合，方便查询

                const std::string toString() const
                {
                    std::stringstream ss;
                    ss << "Series Name: " << name << "\n";
                    ss << "  Key: " << key << "\n";
                    for (const auto& k : keySet)
                    {
                        ss << "    Key Set Item: " << k << "\n";
                    }
                    for (const auto& waypoint : waypointVec)
                    {
                        ss << waypoint.toString();
                    }
                    return ss.str();
                }
        };
        //=============================================PolicyChangeConfig================================================
        class PolicyChangeConfig
        {
            public:
                std::string name;                          // 切换policy名称
                std::string policy;                        // policy名称
                std::string key;                           // 切换按键
                std::string pre_policy_change_description; // 切换前的动作描述
                std::string pre_policy_change_condition;   // 切换前的动作条件
                bool pre_policy_change_use_walk;           // 切换前的动作是否走路算法
                bool back_to_walk;                         // 切换后是否回到walk
                bool back_to_zero;                         // 切换后是否回到zero
                bool loop;                                 // 切换后是否循环
                std::set<std::string> keySet;              // 按键集合，方便查询
                std::vector<double> threasholds;           // 条件阈值

                const std::string toString() const
                {
                    std::stringstream ss;
                    ss << "  Name: " << name << "\n";
                    ss << "    Policy: " << policy << "\n";
                    ss << "    Key: " << key << "\n";
                    ss << "    Prepare policy change description: " << pre_policy_change_description << "\n";
                    ss << "    Prepare policy change condition: " << pre_policy_change_condition << "\n";
                    ss << "    Prepare policy change use walk: " << (pre_policy_change_use_walk ? "true" : "false") << "\n";
                    ss << "    Back To Walk: " << (back_to_walk ? "true" : "false") << "\n";
                    ss << "    Back To Zero: " << (back_to_zero ? "true" : "false") << "\n";
                    ss << "    Loop: " << (loop ? "true" : "false") << "\n";
                    ss << "    threasholds: " << common::toString(threasholds) << "\n";
                    for (const auto& k : keySet)
                    {
                        ss << "       Key Set Item: " << k << "\n";
                    }
                    return ss.str();
                }
                PolicyChangeConfig() = default;
                PolicyChangeConfig(const config::PolicyChangeConfig& other)
                {
                    keySet = other.keySet;
                    name = other.name;
                    key = other.key;
                    pre_policy_change_description = other.pre_policy_change_description;
                    pre_policy_change_condition = other.pre_policy_change_condition;
                    pre_policy_change_use_walk = other.pre_policy_change_use_walk;
                    policy = other.policy;
                    back_to_walk = other.back_to_walk;
                    back_to_zero = other.back_to_zero;
                    loop = other.loop;
                    threasholds = other.threasholds;
                }

                void analyse()
                {
                    if (pre_policy_change_description == "left leg front")
                    {
                        pre_policy_change_use_walk = true;
                    }
                    else if (pre_policy_change_description == "right leg front")
                    {
                        pre_policy_change_use_walk = true;
                    }
                }

                bool prePolicyChangeShouldExecute(const Command& cmd, const Robot_State& state);
                bool finished(const Motor_State& state) const;
        };

        //=============================================RLConfig================================================
        class RLConfig
        {
            public:
                std::string name;                         // policy名称
                std::string type;                         // policy类型 walk, motion, up, down
                std::string policy_file_name;             // policy文件名称
                std::string future_file_name;             // future文件名称
                std::string policy_path;                  // policy路径
                std::string future_path;                  // future路径
                std::string policy_service_name;          // policy serive 名称
                std::string policy_y_service_name_suffix; // policy serive y轴速度后缀 名称
                std::string motor_group;                  // policy控制的电机组
                std::string algorithm;                    // policy算法类
                std::string control_mode;                 // 控制模式: Soccer, Fight
                bool detect_falls = true;                 // 是否检测跌倒 默认开启
                int dofs;                                 // policy控制的DOF数量
                int step_period;                          // 步态周期
                int frame_stack;                          // 历史观测帧数
                int num_single_obs;                       // 单个观测量的数量
                unsigned int time;                        // 加载时间，用于判断是否切换policy
                double motion_len;                        // 动作长度
                double frequency;                         // dreamwaq,humanoidgym使用的频率
                double cmd_lin_vel_scale;                 // 命令线速度缩放
                double cmd_ang_vel_scale;                 // 命令角速度缩放
                double rbt_lin_pos_scale;                 // 机器人线位置缩放
                double rbt_lin_vel_scale;                 // 机器人线速度缩放
                double rbt_ang_vel_scale;                 // 机器人角速度缩放
                double cmd_vel_x_min_walk;                // 命令线速度x最小值
                double cmd_vel_x_min;                     // 命令线速度x最小值
                double cmd_vel_x_max;                     // 命令线速度x最大值
                double cmd_vel_y_min;                     // 命令线速度y最小值
                double cmd_vel_y_max;                     // 命令角速度y最大值
                double cmd_vel_yaw_min;                   // 命令角速度yaw最小值
                double cmd_vel_yaw_max;                   // 命令角速度yaw最大值
                double cmd_vel_filter_scale_soccer;       // Soccer模式命令速度滤波系数
                double cmd_vel_filter_scale_fight;        // Fight模式命令速度滤波系数
                double cmd_vel_boost_multiplier;          // LT按键加速倍数（已废弃，保留兼容性）
                double cmd_vel_x_max_soccer_walk;         // Soccer模式正常最大x速度（不按LT）
                double cmd_vel_x_max_soccer_normal;       // Soccer模式正常最大x速度（不按LT）
                double cmd_vel_x_max_soccer_boost;        // Soccer模式LT加速最大x速度
                double cmd_vel_x_max_fight;               // Fight模式最大x速度（固定，不响应LT）
                double clip_obs;                          // 观测裁剪
                double pd_ctrl_f;                         // PD控制频率
                double rl_ctrl_f;                         // RL控制频率
                double action_scale;                      // 动作缩放
                std::vector<double> action_scales;        // 动作缩放
                std::vector<std::string> joint_names;     // policy控制的关节名称
                std::vector<double> clip_actions_upper;   // 动作上限
                std::vector<double> clip_actions_lower;   // 动作下限
                std::vector<double> clip_output_lower;    // 输出最小限位
                std::vector<double> clip_output_upper;    // 输出最大限位
                std::vector<double> kp;                   // kp
                std::vector<double> kd;                   // kd
                std::vector<double> urdf_offset;          // URDF偏移, 从电机0位偏移到policy初始位置
                std::vector<double> default_pose;         // 动作的默认初始位置
                std::vector<int> normal_mode;             // 正常模式 [frequency, stride]
                std::vector<int> fast_mode;               // 加快模式 [frequency, stride]

                /**
                 * @brief RLConfig 类的构造函数
                 *
                 * 该构造函数用于初始化 RLConfig 对象。
                 */
                RLConfig();
                RLConfig(const std::string& filePath);

                virtual ~RLConfig() = default;

                /**
                 * @brief 从 YAML 文件加载配置
                 *
                 * @param file_path 配置文件的路径
                 * @return true 加载成功
                 * @return false 加载失败
                 */
                bool loadFromYaml(const std::string& filePath);

                /**
                 * @brief 打印配置内容
                 *
                 * 该函数用于打印 RLConfig 对象的配置信息。
                 */
                void print() const;
        };

        //=============================================PDConfig================================================
        /**
         * @brief PDConfig 类的构造函数
         *
         * 该构造函数用于初始化 PDConfig 对象。
         */
        class PDConfig
        {
            public:
                std::string name;                     // 机器人名称
                std::string algorithm;                // 当前算法名称
                int dofs;                             // DOF数量
                double max_power = 300.0;             // 最大功率
                double max_power_duration = 1.0;      // 最大功率持续时间
                std::vector<std::string> joint_names; // 电机方向
                std::vector<int> map_index;           // 电机ID映射
                std::vector<int> direction;           // 电机方向
                std::vector<double> lower;            // 最底层的电机下限
                std::vector<double> upper;            // 最底层的电机上限
                std::vector<double> p_kp;             // 融合了policy的kp
                std::vector<double> p_kd;             // 融合了policy的kd
                std::vector<double> kp;               // kp
                std::vector<double> kd;               // kd

                std::vector<std::string> waypoint_files; // 轨迹文件

                std::map<std::string, std::map<std::string, MultiWaypointConfig>> multi_configs;   // 产线， 名称 -> MultiWaypointConfig
                std::map<std::string, std::map<std::string, SeriesWaypointConfig>> series_configs; // 产线， 名称 -> SeriesWaypointConfig
                std::vector<int> policy_to_actual_map;                                             // policy到实际电机的映射
                std::vector<int> actual_to_policy_map;                                             // actual到policy的映射
                std::vector<double> pd_urdf_offset;                                                // 最原始的urdf偏移
                std::vector<double> urdf_offset;                                                   // 不同policy有不同的urdf偏移
                std::vector<double> clip_output_lower;                                             // 输出最小限位
                std::vector<double> clip_output_upper;                                             // 输出最大限位
                std::map<std::string, std::vector<std::string>> series_trajectory;                 // 多段轨迹
                // 通过motor_id 排序的 vector
                class ByMotorId
                {
                    public:
                        std::vector<int> direction;            // 电机方向
                        std::vector<double> kp;                // kp
                        std::vector<double> kd;                // kd
                        std::vector<double> p_kp;              // 融合了policy的kp
                        std::vector<double> p_kd;              // 融合了policy的kd
                        std::vector<double> urdf_offset;       // urdf偏移
                        std::vector<double> clip_output_lower; // 输出最小限位
                        std::vector<double> clip_output_upper; // 输出最大限位
                };

                ByMotorId byMotorId;

                PDConfig();
                PDConfig(const std::string& filePath, const std::string& actionPath);

                virtual ~PDConfig() = default;

                /**
                 * @brief 从 YAML 文件加载配置
                 *
                 * @param filePath 配置文件的路径
                 * @param actionPath 动作库配置文件的路径
                 * @return true 加载成功
                 * @return false 加载失败
                 */
                bool loadFromYaml(const std::string& filePath, const std::string& actionPath);

                /**
                 * @brief 刷新配置内容
                 * 先执行RL配置相关的
                 * @param rlInfo RLConfig 对象
                 * @return true 刷新成功
                 * @return false 刷新失败
                 */
                bool refreshRLConfig(const RLConfig& rlInfo);

                /**
                 * @brief 刷新配置内容
                 * 再执行ByMotorId
                 * @return true 刷新成功
                 * @return false 刷新失败
                 */
                bool refreshByMotorIdVec();

                /**
                 * @brief 打印配置内容
                 *
                 * 该函数用于打印 PDConfig 对象的配置信息。
                 */
                void print() const;
        };

        //=============================================RLConfig Group================================================
        class RLConfigGroup
        {
            public:
                RLConfigGroup() = default;
                /**
                 * @brief RLConfigGroup 类的构造函数
                 *
                 * 该构造函数用于初始化 RLConfigGroup 对象。
                 * @param configPath 所有policy配置文件路径
                 * @param rlConfigFile 读取的rlGroup配置文件名称
                 */
                RLConfigGroup(const std::string& configPath, const std::string policyPath, const std::string& rlConfigFile, const std::string& actionPath);

                bool loadFromYaml(const std::string& configPath, const std::string policyPath, const std::string& rlConfigFile, const std::string& actionPath);

                void print() const;

                /** @brief 获取 RLConfig 对象
                 * @param name 策略名称
                 * @return RLConfig 对象
                 * @throws std::out_of_range 如果策略名称不存在
                 */
                const RLConfig& getRLConfigByName(const std::string& name) const;

                /** @brief 获取当前 RLConfig
                 * @return 当前 RLConfig
                 * @throws std::out_of_range 如果当前策略索引超出范围
                 */
                const RLConfig& getCurrentRLConfig() const;

                /** @brief 获取当前 RLConfig
                 * @return 当前 RLConfig
                 * @throws std::out_of_range 如果当前策略索引超出范围
                 */
                const RLConfig& getNextRLConfig();

                /** @brief 获取上一个 RLConfig
                 * @return 上一个 RLConfig, 循环
                 * @throws std::out_of_range 如果没有上一个 RLConfig
                 */
                const RLConfig& getPrevRLConfig();

                /** @brief 获取当前步态 RLConfig
                 * @return 当前步态 RLConfig
                 * @throws std::out_of_range 如果当前步态策略索引超出范围
                 */
                const RLConfig& getWalkCurrentRLConfig();

                /** @brief 获取全身控制走路算法 RLConfig (motor_group == "all")
                 * @return 全身控制走路算法 RLConfig
                 * @throws std::runtime_error 如果没有找到全身控制走路算法
                 */
                const RLConfig& getAllBodyWalkPolicy() const;

                /** @brief 获取下半身控制走路算法 RLConfig (motor_group != "all")
                 * @return 下半身控制走路算法 RLConfig
                 * @throws std::runtime_error 如果没有找到下半身控制走路算法
                 */
                const RLConfig& getLowerBodyWalkPolicy() const;

                /** @brief 获取当前动作 RLConfig
                 * @return 当前动作 RLConfig
                 * @throws std::out_of_range 如果当前动作策略索引超出范围
                 */
                const RLConfig& getMotionCurrentRLConfig();

                /** @brief 获取下一个步态 RLConfig
                 * @return 下一个步态 RLConfig, 循环
                 * @throws std::out_of_range 如果没有下一个步态 RLConfig
                 */
                const RLConfig& getWalkNextRLConfig();

                /** @brief 获取上一个步态 RLConfig
                 * @return 上一个步态 RLConfig, 循环
                 * @throws std::out_of_range 如果没有上一个步态 RLConfig
                 */
                const RLConfig& getWalkPrevRLConfig();

                /** @brief 获取下一个动作 RLConfig
                 * @return 下一个动作 RLConfig, 循环
                 * @throws std::out_of_range 如果没有下一个动作 RLConfig
                 */
                const RLConfig& getMotionPrevRLConfig();
                /** @brief 获取上一个动作 RLConfig
                 * @return 上一个动作 RLConfig, 循环
                 * @throws std::out_of_range 如果没有上一个动作 RLConfig
                 */
                const RLConfig& getMotionNextRLConfig();

                inline const std::map<std::string, PolicyChangeConfig>& getPolicyChangeConfigs(const std::string productionType)
                {
                    return policyChangeConfigs_[productionType];
                }

                /**
                 * @brief 遍历所有 RLConfig 并执行给定的函数
                 * @param func 要执行的函数，接受一个 RLConfig 对象作为参数
                 * @note 如果函数返回 true，则停止遍历
                 */
                const void foreacheRLConfig(const std::function<bool(const RLConfig&)>& func) const
                {
                    std::set<std::string> names;
                    names.insert(walkPolicies_.begin(), walkPolicies_.end());
                    names.insert(motionPolicies_.begin(), motionPolicies_.end());
                    names.insert(upPolicies_.begin(), upPolicies_.end());
                    for (const auto& policyName : names)
                    {
                        if (func(getRLConfigByName(policyName)))
                        {
                            break;
                        }
                    }
                }

            private:
                std::vector<RLConfig> walk_;
                std::vector<RLConfig> motion_;
                std::vector<RLConfig> up_;

                std::map<std::string, std::map<std::string, PolicyChangeConfig>> policyChangeConfigs_;

                std::vector<std::string> policies_;         // 存储所有策略名称
                std::vector<std::string> walkPolicies_;     // 存储所有步态策略名称
                std::vector<std::string> motionPolicies_;   // 存储所有动作策略名称
                std::vector<std::string> upPolicies_;       // 存储所有动作策略名称
                unsigned int currentPolicyIndex_ = 0;       // 当前策略索引
                unsigned int currentWalkPolicyIndex_ = 0;   // 当前步态策略索引
                unsigned int currentMotionPolicyIndex_ = 0; // 当前动作策略索引
        };
    }
}
#endif
