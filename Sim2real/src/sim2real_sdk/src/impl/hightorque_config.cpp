#include "impl/hightorque_config.h"
#include "policy/bydmmc_policy.h"
#include "ros/time.h"
#include <algorithm>
#include <iostream>
#include <thread>

namespace hightorque
{
    namespace config
    {
        /**
         * @brief 将按键组合字符串拆分为按键集合
         *
         * @param combo 按键组合字符串，按键间用'+'连接（如 "LT+RT+A"）
         * @return std::unordered_set<std::string> 拆分后的按键集合
         *
         * 该函数将输入的按键组合字符串按'+'分割，并返回包含各个按键的集合，方便后续查询。
         */
        std::set<std::string> splitKeyCombination(const std::string& combo)
        {
            std::set<std::string> keySet;
            std::stringstream ss(combo);
            std::string key;
            // 用'+'分割字符串
            while (std::getline(ss, key, '+'))
            {
                if (!key.empty())
                { // 过滤空字符串（如组合末尾有'+'的情况）
                    keySet.insert(key);
                }
            }
            return keySet;
        }
        /**
         * @brief 根据 motorId 填充数据
         *
         * @tparam ValueT 数据类型
         * @param mapIndex 电机ID映射
         * @param data 原始数据
         * @param dataById 按 motorId 排序后的数据输出
         *
         * 该函数将根据给定的 motorId 映射，将原始数据填充到按 motorId 排序的输出中。
         */
        template <typename ValueT>
        bool fillByMotorId(const std::vector<int>& mapIndex, const std::vector<ValueT>& data, std::vector<ValueT>& dataById)
        {
            if (mapIndex.size() != data.size())
            {
                std::cerr << "fillByMotorId: mapIndex 与 data 长度不一致" << std::endl;
                return false;
            }

            // 找到最大的 motorId
            int maxId = -1;
            for (int id : mapIndex)
            {
                if (id < 0)
                    continue;
                maxId = std::max(maxId, id);
            }
            if (maxId < 0)
            {
                // 没有有效 motorId，清空输出
                dataById.clear();
                return false;
            }

            // 分配输出空间，初值为 ValueT()（如 0 或 0.0）
            dataById.assign(static_cast<size_t>(maxId) + 1, ValueT());

            // 按 mapIndex 映射 data
            for (size_t i = 0; i < mapIndex.size(); ++i)
            {
                int id = mapIndex[i];
                if (id >= 0)
                {
                    dataById[static_cast<size_t>(id)] = data[i];
                    // std::cout << "[" << id << "]:" << data[i] << ", ";
                }
            }
            // std::cout << std::endl;
            return true;
        }
        // 创建映射关系：策略索引 -> 实际索引
        std::vector<int> createPolicy2ActualMapping(const std::vector<std::string>& policy, const std::vector<std::string>& actual)
        {
            std::vector<int> mapping(policy.size(), -1);

            // 直接匹配策略关节到实际关节
            for (size_t i = 0; i < policy.size(); ++i)
            {
                const std::string& jointName = policy[i];

                // 在实际关节中查找
                auto it = std::find(actual.begin(), actual.end(), jointName);
                if (it != actual.end())
                {
                    size_t actual_idx = std::distance(actual.begin(), it);
                    if (actual_idx < actual.size())
                    {
                        mapping[i] = static_cast<int>(actual_idx);
                    }
                    else
                    {
                        std::cerr << "错误：计算的实际索引超出范围: " << actual_idx << std::endl;
                    }
                }
                else
                {
                    // 关节名称应该都能匹配，这里添加错误处理
                    std::cerr << "错误：策略关节 '" << jointName
                              << "' 未在实际关节列表中找到！" << std::endl;
                }
            }
            return mapping;
        }

        // 创建映射关系：实际索引 -> 策略索引
        std::vector<int> createActual2PolicyMapping(std::vector<std::string> actual, std::vector<std::string> policy)
        {
            std::vector<int> mapping(actual.size(), -1);

            // 直接匹配策略关节到实际关节
            for (size_t i = 0; i < actual.size(); ++i)
            {
                const std::string& jointName = actual[i];

                // 在实际关节中查找
                auto it = std::find(policy.begin(), policy.end(), jointName);
                if (it != policy.end())
                {
                    size_t policy_idx = std::distance(policy.begin(), it);
                    if (policy_idx < policy.size())
                    {
                        mapping[i] = static_cast<int>(policy_idx);
                    }
                    else
                    {
                        std::cerr << "错误：计算的策略索引超出范围: " << jointName << std::endl;
                    }
                }
            }
            return mapping;
        }

        //**********************************************************************RLConfig***************************************************************
        RLConfig::RLConfig()
        {
        }

        RLConfig::RLConfig(const std::string& filePath)
        {
            if (!loadFromYaml(filePath))
            {
                std::cerr << "Failed to load RLConfig from " << filePath << std::endl;
            }
        }

        bool RLConfig::loadFromYaml(const std::string& file_path)
        {
            try
            {
                YAML::Node config = YAML::LoadFile(file_path);
                time = ros::Time::now().nsec;
                if (config["policy_name"])
                {
                    policy_file_name = config["policy_name"].as<std::string>();
                }
                if (config["future_file_name"])
                {
                    future_file_name = config["future_file_name"].as<std::string>();
                }
                if (config["service_name"])
                {
                    policy_service_name = config["service_name"].as<std::string>();
                }
                if (config["y_service_name_suffix"])
                {
                    policy_y_service_name_suffix = config["y_service_name_suffix"].as<std::string>();
                }
                if (config["algorithm"])
                {
                    algorithm = config["algorithm"].as<std::string>();
                }
                if (config["control_mode"])
                {
                    control_mode = config["control_mode"].as<std::string>();
                }
                if (config["motor_group"])
                {
                    motor_group = config["motor_group"].as<std::string>();
                }
                if (config["dofs"])
                {
                    dofs = config["dofs"].as<int>();
                }
                if (config["detect_falls"])
                {
                    detect_falls = config["detect_falls"].as<bool>();
                }
                if (config["step_period"])
                {
                    step_period = config["step_period"].as<int>();
                }
                if (config["frame_stack"])
                {
                    frame_stack = config["frame_stack"].as<int>();
                }
                if (config["num_single_obs"])
                {
                    num_single_obs = config["num_single_obs"].as<int>();
                }
                if (config["motion_len"])
                {
                    motion_len = config["motion_len"].as<double>();
                }
                if (config["frequency"])
                {
                    frequency = config["frequency"].as<double>();
                }
                if (config["cmd_lin_vel_scale"])
                {
                    cmd_lin_vel_scale = config["cmd_lin_vel_scale"].as<double>();
                }
                if (config["cmd_ang_vel_scale"])
                {
                    cmd_ang_vel_scale = config["cmd_ang_vel_scale"].as<double>();
                }
                if (config["rbt_lin_pos_scale"])
                {
                    rbt_lin_pos_scale = config["rbt_lin_pos_scale"].as<double>();
                }
                if (config["rbt_lin_vel_scale"])
                {
                    rbt_lin_vel_scale = config["rbt_lin_vel_scale"].as<double>();
                }
                if (config["rbt_ang_vel_scale"])
                {
                    rbt_ang_vel_scale = config["rbt_ang_vel_scale"].as<double>();
                }
                if (config["cmd_vel_x_min_walk"])
                {
                    cmd_vel_x_min_walk = config["cmd_vel_x_min_walk"].as<double>();
                }
                if (config["cmd_vel_x_min"])
                {
                    cmd_vel_x_min = config["cmd_vel_x_min"].as<double>();
                }
                if (config["cmd_vel_x_max"])
                {
                    cmd_vel_x_max = config["cmd_vel_x_max"].as<double>();
                }
                if (config["cmd_vel_y_min"])
                {
                    cmd_vel_y_min = config["cmd_vel_y_min"].as<double>();
                }
                if (config["cmd_vel_y_max"])
                {
                    cmd_vel_y_max = config["cmd_vel_y_max"].as<double>();
                }
                if (config["cmd_vel_yaw_min"])
                {
                    cmd_vel_yaw_min = config["cmd_vel_yaw_min"].as<double>();
                }
                if (config["cmd_vel_yaw_max"])
                {
                    cmd_vel_yaw_max = config["cmd_vel_yaw_max"].as<double>();
                }
                if (config["cmd_vel_filter_scale_soccer"])
                {
                    cmd_vel_filter_scale_soccer = config["cmd_vel_filter_scale_soccer"].as<double>();
                }
                if (config["cmd_vel_filter_scale_fight"])
                {
                    cmd_vel_filter_scale_fight = config["cmd_vel_filter_scale_fight"].as<double>();
                }
                if (config["cmd_vel_boost_multiplier"])
                {
                    cmd_vel_boost_multiplier = config["cmd_vel_boost_multiplier"].as<double>();
                }
                if (config["cmd_vel_x_max_soccer_walk"])
                {
                    cmd_vel_x_max_soccer_walk = config["cmd_vel_x_max_soccer_walk"].as<double>();
                }
                // Soccer/Fight 模式速度上限配置（仅x方向）
                if (config["cmd_vel_x_max_soccer_normal"])
                {
                    cmd_vel_x_max_soccer_normal = config["cmd_vel_x_max_soccer_normal"].as<double>();
                }
                if (config["cmd_vel_x_max_soccer_boost"])
                {
                    cmd_vel_x_max_soccer_boost = config["cmd_vel_x_max_soccer_boost"].as<double>();
                }
                if (config["cmd_vel_x_max_fight"])
                {
                    cmd_vel_x_max_fight = config["cmd_vel_x_max_fight"].as<double>();
                }
                if (config["clip_obs"])
                {
                    clip_obs = config["clip_obs"].as<double>();
                }
                if (config["pd_ctrl_f"])
                {
                    pd_ctrl_f = config["pd_ctrl_f"].as<double>();
                }
                if (config["rl_ctrl_f"])
                {
                    rl_ctrl_f = config["rl_ctrl_f"].as<double>();
                }
                if (config["action_scale"])
                {
                    action_scale = config["action_scale"].as<double>();
                }
                if (config["joint_names"])
                {
                    joint_names = config["joint_names"].as<std::vector<std::string>>();
                }
                if (config["action_scales"])
                {
                    action_scales = config["action_scales"].as<std::vector<double>>();
                }
                if (config["clip_actions_lower"])
                {
                    clip_actions_lower = config["clip_actions_lower"].as<std::vector<double>>();
                }
                if (config["clip_actions_upper"])
                {
                    clip_actions_upper = config["clip_actions_upper"].as<std::vector<double>>();
                }
                if (config["clip_output_lower"])
                {
                    clip_output_lower = config["clip_output_lower"].as<std::vector<double>>();
                }
                if (config["clip_output_upper"])
                {
                    clip_output_upper = config["clip_output_upper"].as<std::vector<double>>();
                }
                if (config["kp"])
                {
                    kp = config["kp"].as<std::vector<double>>();
                }
                if (config["kd"])
                {
                    kd = config["kd"].as<std::vector<double>>();
                }
                if (config["urdf_offset"])
                {
                    urdf_offset = config["urdf_offset"].as<std::vector<double>>();
                }
                if (config["default_pose"])
                {
                    default_pose = config["default_pose"].as<std::vector<double>>();
                }
                if (config["normal_mode"])
                {
                    normal_mode = config["normal_mode"].as<std::vector<int>>();
                }
                else
                {
                    normal_mode = {0, 0}; // 默认值
                }
                if (config["fast_mode"])
                {
                    fast_mode = config["fast_mode"].as<std::vector<int>>();
                }
                else
                {
                    fast_mode = {1, 1}; // 默认值
                }
            }
            catch (const YAML::Exception& e)
            {
                std::cerr << "Error loading RLConfig (" << file_path << ") from YAML: " << e.what() << std::endl;
                return false;
            }
            return true;
        }

        void RLConfig::print() const
        {
            std::cout << "----------------------------------rl start---------------------------" << std::endl;
            std::cout << "RLConfig: " << name << ": " << policy_file_name << std::endl;
            std::cout << "Future file name: " << future_file_name << std::endl;
            std::cout << "service_name: " << policy_service_name << std::endl;
            std::cout << "service_y_name: " << policy_service_name + policy_y_service_name_suffix << std::endl;
            std::cout << "Algorithm: " << algorithm << std::endl;
            std::cout << "Policy path: " << policy_path << std::endl;
            std::cout << "Future path: " << future_path << std::endl;
            std::cout << "Policy Ctrl Motor Group: " << motor_group << std::endl;
            std::cout << "Detect Falls: " << (detect_falls ? "true" : "false") << std::endl;
            std::cout << "DOFs: " << dofs << std::endl;
            std::cout << "Step Period: " << step_period << std::endl;
            std::cout << "Frame Stack: " << frame_stack << std::endl;
            std::cout << "Num Single Obs: " << num_single_obs << std::endl;
            std::cout << "Motion Length: " << motion_len << std::endl;
            std::cout << "Frequency: " << frequency << std::endl;
            std::cout << "Command Linear Velocity Scale: " << cmd_lin_vel_scale << std::endl;
            std::cout << "Command Angular Velocity Scale: " << cmd_ang_vel_scale << std::endl;
            std::cout << "Robot Linear Position Scale: " << rbt_lin_pos_scale << std::endl;
            std::cout << "Robot Linear Velocity Scale: " << rbt_lin_vel_scale << std::endl;
            std::cout << "Robot Angular Velocity Scale: " << rbt_ang_vel_scale << std::endl;
            std::cout << "Cmd Vel X (" << cmd_vel_x_min << ", " << cmd_vel_x_max << ")" << std::endl;
            std::cout << "Cmd Vel Y (" << cmd_vel_y_min << ", " << cmd_vel_y_max << ")" << std::endl;
            std::cout << "Cmd Vel Yaw (" << cmd_vel_yaw_min << ", " << cmd_vel_yaw_max << ")" << std::endl;
            std::cout << "Clip Observations: " << clip_obs << std::endl;
            std::cout << "PD Control Frequency: " << pd_ctrl_f << std::endl;
            std::cout << "RL Control Frequency: " << rl_ctrl_f << std::endl;
            std::cout << "Action Scale: " << action_scale << std::endl;

            if (!joint_names.empty())
            {
                std::cout << "Joint Names: ";
                for (const auto& joint : joint_names)
                {
                    std::cout << joint << ", ";
                }
                std::cout << "\n";
            }

            if (!action_scales.empty())
            {
                std::cout << "Action Scales: ";
                for (const auto& scale : action_scales)
                {
                    std::cout << scale << ", ";
                }
                std::cout << "\n";
            }

            if (!clip_actions_lower.empty())
            {
                std::cout << "Clip Actions Lower: ";
                for (const auto& lower : clip_actions_lower)
                {
                    std::cout << lower << ", ";
                }
                std::cout << "\n";
            }

            if (!clip_actions_upper.empty())
            {
                std::cout << "Clip Actions Upper: ";
                for (const auto& upper : clip_actions_upper)
                {
                    std::cout << upper << ", ";
                }
                std::cout << "\n";
            }

            if (!clip_output_lower.empty())
            {
                std::cout << "Clip Output Lower: ";
                for (const auto& lower : clip_output_lower)
                {
                    std::cout << lower << ", ";
                }
                std::cout << "\n";
            }
            if (!clip_output_upper.empty())
            {
                std::cout << "Clip Output Upper: ";
                for (const auto& upper : clip_output_upper)
                {
                    std::cout << upper << ", ";
                }
                std::cout << "\n";
            }
            if (!kp.empty())
            {
                std::cout << "Kp: ";
                for (const auto& k : kp)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!kd.empty())
            {
                std::cout << "Kd: ";
                for (const auto& k : kd)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!urdf_offset.empty())
            {
                std::cout << "URDF Offset: ";
                for (const auto& offset : urdf_offset)
                {
                    std::cout << offset << ", ";
                }
                std::cout << "\n";
            }
            if (!default_pose.empty())
            {
                std::cout << "Default Pose: ";
                for (const auto& pose : default_pose)
                {
                    std::cout << pose << ", ";
                }
                std::cout << "\n";
            }
            std::cout << "----------------------------------rl   end---------------------------" << std::endl;
        }

        //**********************************************************************PDConfig***************************************************************

        PDConfig::PDConfig()
        {
        }

        PDConfig::PDConfig(const std::string& filePath, const std::string& actionPath)
        {
            if (!loadFromYaml(filePath, actionPath))
            {
                std::cerr << "Failed to load PDConfig from " << filePath << " and action file: " << actionPath << std::endl;
            }
        }

        bool PDConfig::loadFromYaml(const std::string& pdFile, const std::string& actionPath)
        {
            try
            {
                YAML::Node config = YAML::LoadFile(pdFile);
                if (config["name"])
                {
                    name = config["name"].as<std::string>();
                }
                if (config["dofs"])
                {
                    dofs = config["dofs"].as<int>();
                }
                if (config["joint_names"])
                {
                    joint_names = config["joint_names"].as<std::vector<std::string>>();
                }
                if (config["map_index"])
                {
                    map_index = config["map_index"].as<std::vector<int>>();
                }
                if (config["max_power"])
                {
                    max_power = config["max_power"].as<double>();
                }
                if (config["max_power_duration"])
                {
                    max_power_duration = config["max_power_duration"].as<double>();
                }
                if (config["direction"])
                {
                    direction = config["direction"].as<std::vector<int>>();
                }
                if (config["lower"])
                {
                    lower = config["lower"].as<std::vector<double>>();
                }
                if (config["upper"])
                {
                    upper = config["upper"].as<std::vector<double>>();
                }
                if (config["kp"])
                {
                    kp = config["kp"].as<std::vector<double>>();
                }
                if (config["kd"])
                {
                    kd = config["kd"].as<std::vector<double>>();
                }
                if (config["urdf_offset"])
                {
                    pd_urdf_offset = config["urdf_offset"].as<std::vector<double>>();
                }
                else
                {
                    pd_urdf_offset.resize(dofs, 0.0); // 如果没有提供urdf_offset，则默认填充为0.0
                }
                const auto getWaypointConfig = [](const YAML::Node& node) {
                    MultiWaypointConfig config;
                    if (node["key"])
                    {
                        config.key = node["key"].as<std::string>();
                        config.keySet = splitKeyCombination(config.key);
                    }
                    if (node["name"])
                    {
                        config.name = node["name"].as<std::string>();
                    }
                    if (node["path"])
                    {
                        config.path = node["path"].as<std::string>();
                    }
                    if (node["type"])
                    {
                        config.type = node["type"].as<int>();
                    }
                    if (node["duration"])
                    {
                        config.duration = node["duration"].as<double>();
                    }
                    if (node["torque"])
                    {
                        config.torque = node["torque"].as<double>();
                    }
                    if (node["back_to_zero"])
                    {
                        config.back_to_zero = node["back_to_zero"].as<bool>();
                    }
                    if (node["back_to_walk"])
                    {
                        config.back_to_walk = node["back_to_walk"].as<bool>();
                    }
                    if (node["loop"])
                    {
                        config.loop = node["loop"].as<bool>();
                    }
                    // type 4
                    if (node["time_interval"])
                    {
                        config.time_interval = node["time_interval"].as<double>();
                    }
                    if (node["velocity"])
                    {
                        config.velocity = node["velocity"].as<double>();
                    }
                    return config;
                };
                // 读取动作库所有动作
                std::cout << "Loading base waypoint config from " << actionPath + "base_waypoint.yaml" << std::endl;
                YAML::Node baseWaypointConfig = YAML::LoadFile(actionPath + "base_waypoint.yaml");
                std::map<std::string, MultiWaypointConfig> multi; // 动作名称 -> MultiWaypointConfig
                // multi_waypoint_config
                if (baseWaypointConfig["multi_waypoint_config"] && baseWaypointConfig["multi_waypoint_config"].IsSequence())
                {
                    try
                    {
                        for (size_t i = 0; i < baseWaypointConfig["multi_waypoint_config"].size(); ++i)
                        {
                            const YAML::Node& item = baseWaypointConfig["multi_waypoint_config"][i];
                            if (!item.IsMap())
                            {
                                std::cerr << "Error: multi_waypoint_config[" << item.size() << "] is not a map" << std::endl;
                                continue;
                            }
                            try
                            {
                                auto baseWaypointConfig = getWaypointConfig(item);
                                multi[baseWaypointConfig.name] = baseWaypointConfig;
                            }
                            catch (const YAML::Exception& e)
                            {
                                std::cerr << "Error parsing multi_waypoint_config[" << i << "]: " << e.what() << std::endl;
                                continue;
                            }
                        }
                    }
                    catch (const YAML::Exception& e)
                    {
                        std::cerr << "Error parsing multi_waypoint_config: " << e.what() << std::endl;
                    }
                }
                std::map<std::string, SeriesWaypointConfig> series; // 产线， 名称 -> SeriesWaypointConfig
                // series_waypoint_config
                if (baseWaypointConfig["series_waypoint_config"] && baseWaypointConfig["series_waypoint_config"].IsSequence())
                {
                    try
                    {
                        for (size_t i = 0; i < baseWaypointConfig["series_waypoint_config"].size(); ++i)
                        {
                            SeriesWaypointConfig seriesConfig;
                            const YAML::Node& item = baseWaypointConfig["series_waypoint_config"][i];
                            if (!item.IsMap())
                            {
                                std::cerr << "Error: series_waypoint_config[" << item.size() << "] is not a map" << std::endl;
                                continue;
                            }
                            try
                            {
                                seriesConfig.name = item["name"].as<std::string>();
                                for (size_t j = 0; j < item["waypoint_config"].size(); ++j)
                                {
                                    try
                                    {
                                        auto baseWaypointConfig = getWaypointConfig(item["waypoint_config"][j]);
                                        seriesConfig.waypointVec.push_back(baseWaypointConfig);
                                    }
                                    catch (const YAML::Exception& e)
                                    {
                                        std::cerr << "Error parsing series_waypoint_config[" << i << "][" << j << "]: " << e.what() << std::endl;
                                        continue;
                                    }
                                }
                                series.emplace(seriesConfig.name, seriesConfig);
                            }
                            catch (const YAML::Exception& e)
                            {
                                std::cerr << "Error parsing series_waypoint_config[" << i << "]: " << e.what() << std::endl;
                                continue;
                            }
                        }
                    }
                    catch (const YAML::Exception& e)
                    {
                        std::cerr << "Error parsing series_waypoint_config: " << e.what() << std::endl;
                    }
                }
                // 读取用户自定义动作列表
                std::cout << "Loading custom actions from " << actionPath + "custom_action.yaml" << std::endl;
                YAML::Node customConfig = YAML::LoadFile(actionPath + "custom_action.yaml");
                if (customConfig.IsSequence())
                {
                    for (size_t i = 0; i < customConfig.size(); ++i)
                    {
                        try
                        {
                            if (!customConfig[i].IsMap())
                            {
                                std::cerr << "Error: custom waypoint config[" << i << "] is not a map" << std::endl;
                                continue;
                            }
                            if (!customConfig[i]["production_type"])
                            {
                                std::cerr << "Error: custom waypoint config[" << i << "] missing production_type" << std::endl;
                                continue;
                            }
                            std::string production_type = customConfig[i]["production_type"].as<std::string>();
                            std::cout << "Loading custom action for production type: " << production_type << std::endl;
                            if (customConfig[i]["multi_waypoint_config"] && customConfig[i]["multi_waypoint_config"].IsSequence())
                            {
                                for (size_t j = 0; j < customConfig[i]["multi_waypoint_config"].size(); ++j)
                                {
                                    std::string name = customConfig[i]["multi_waypoint_config"][j]["name"].as<std::string>();
                                    if (multi.find(name) == multi.end())
                                    {
                                        std::cerr << "Warning: custom multi waypoint config name [" << name << "] not found in base_waypoint.yaml" << std::endl;
                                        continue;
                                    }
                                    auto& multiVec = multi[name];
                                    std::string key = customConfig[i]["multi_waypoint_config"][j]["key"].as<std::string>();
                                    multiVec.key = key; // 使用自定义的key覆盖
                                    multiVec.keySet = splitKeyCombination(key);
                                    multi_configs[production_type][name] = multiVec;
                                }
                            }
                            if (customConfig[i]["series_waypoint_config"] && customConfig[i]["series_waypoint_config"].IsSequence())
                            {
                                for (size_t j = 0; j < customConfig[i]["series_waypoint_config"].size(); ++j)
                                {
                                    std::string name = customConfig[i]["series_waypoint_config"][j]["name"].as<std::string>();
                                    if (series.find(name) == series.end())
                                    {
                                        std::cerr << "Warning: custom series waypoint config name [" << name << "] not found in base_waypoint.yaml" << std::endl;
                                        continue;
                                    }
                                    auto& seriesVec = series[name];
                                    std::string key = customConfig[i]["series_waypoint_config"][j]["key"].as<std::string>();
                                    seriesVec.key = key; // 使用自定义的key覆盖
                                    seriesVec.keySet = splitKeyCombination(key);
                                    series_configs[production_type][name] = seriesVec;
                                }
                            }
                        }
                        catch (const YAML::Exception& e)
                        {
                            std::cerr << "Error parsing custom waypoint config[" << i << "]: " << e.what() << std::endl;
                            continue;
                        }
                    }
                }
                else
                {
                    std::cerr << "Error: custom waypoint config is not a sequence (list)" << std::endl;
                }
            }
            catch (const YAML::Exception& e)
            {
                std::cerr << "Error loading PDConfig from YAML: " << e.what() << std::endl;
                return false;
            }
            return true;
        }

        bool PDConfig::refreshByMotorIdVec()
        {
            // 使用 fillByMotorId 函数填充数据
            fillByMotorId(map_index, direction, byMotorId.direction);
            fillByMotorId(map_index, kp, byMotorId.kp);
            fillByMotorId(map_index, kd, byMotorId.kd);
            fillByMotorId(map_index, p_kp, byMotorId.p_kp);
            fillByMotorId(map_index, p_kd, byMotorId.p_kd);
            fillByMotorId(map_index, urdf_offset, byMotorId.urdf_offset);
            fillByMotorId(map_index, clip_output_lower, byMotorId.clip_output_lower);
            fillByMotorId(map_index, clip_output_upper, byMotorId.clip_output_upper);

            return true;
        }

        bool PDConfig::refreshRLConfig(const RLConfig& rlInfo)
        {
            if (rlInfo.joint_names.size() > joint_names.size())
            {
                std::cerr << "Error: RLConfig joint_names size [" << rlInfo.joint_names.size() << "] is larger than PDConfig joint_names size [" << joint_names.size() << "]" << std::endl;
                // return false;
            }
            algorithm = rlInfo.algorithm;

            policy_to_actual_map = createPolicy2ActualMapping(rlInfo.joint_names, joint_names);
            actual_to_policy_map = createActual2PolicyMapping(joint_names, rlInfo.joint_names);
            urdf_offset.resize(dofs);
            urdf_offset = pd_urdf_offset; // 先将urdf偏移填充为最原始的urdf偏移

            clip_output_lower.resize(dofs, -0.1); // 如果policy的dof小于所有的dof，则设置为-0.1
            clip_output_upper.resize(dofs, 0.1);  // 如果policy的dof小于所有的dof，则设置为0.1
            p_kp = kp;                            // 将policy的kp填充电机最合适的kp
            p_kd = kd;                            // 将policy的kd填充电机最合适的kd
            // 将policy的urdf偏移填充到实际的urdf偏移
            for (auto i = 0; i < policy_to_actual_map.size(); i++)
            {
                if (policy_to_actual_map[i] >= 0 && policy_to_actual_map[i] < dofs)
                {
                    urdf_offset[policy_to_actual_map[i]] = rlInfo.urdf_offset[i];
                    clip_output_lower[policy_to_actual_map[i]] = rlInfo.clip_output_lower[i];
                    clip_output_upper[policy_to_actual_map[i]] = rlInfo.clip_output_upper[i];
                    p_kp[policy_to_actual_map[i]] = rlInfo.kp[i];
                    p_kd[policy_to_actual_map[i]] = rlInfo.kd[i];
                }
                else
                {
                    std::cerr << "Error: policy_to_actual_map[" << i << "] is out of bounds for urdf_offset" << std::endl;
                    // return false;
                }
            }
            return true;
        }

        void PDConfig::print() const
        {
            std::cout << "=====================================pd start================================" << std::endl;
            std::cout << "PDConfig: " << name << std::endl;
            std::cout << "DOFs: " << dofs << std::endl;
            std::cout << "Algorithm: " << algorithm << std::endl;
            std::cout << "Joint Names: ";
            if (!joint_names.empty())
            {
                for (const auto& joint : joint_names)
                {
                    std::cout << joint << ", ";
                }
                std::cout << "\n";
            }
            if (!map_index.empty())
            {
                std::cout << "Map Index: ";
                for (const auto& index : map_index)
                {
                    std::cout << index << ", ";
                }
                std::cout << "\n";
            }
            std::cout << "Max Power Duration: " << max_power_duration << "\n";
            std::cout << "Max Power: " << max_power << "\n";
            if (!direction.empty())
            {
                std::cout << "Direction: ";
                for (const auto& dir : direction)
                {
                    std::cout << dir << ", ";
                }
                std::cout << "\n";
            }
            if (!lower.empty())
            {
                std::cout << "Lower Limits: ";
                for (const auto& l : lower)
                {
                    std::cout << l << ", ";
                }
                std::cout << "\n";
            }
            if (!upper.empty())
            {
                std::cout << "Upper Limits: ";
                for (const auto& u : upper)
                {
                    std::cout << u << ", ";
                }
                std::cout << "\n";
            }
            if (!kp.empty())
            {
                std::cout << "Kp: ";
                for (const auto& k : kp)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!kd.empty())
            {
                std::cout << "Kd: ";
                for (const auto& k : kd)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!urdf_offset.empty())
            {
                std::cout << "Urdf Offset: ";
                for (const auto& u : urdf_offset)
                {
                    std::cout << u << ", ";
                }
                std::cout << "\n";
            }
            if (!pd_urdf_offset.empty())
            {
                std::cout << "PD Urdf Offset: ";
                for (const auto& u : pd_urdf_offset)
                {
                    std::cout << u << ", ";
                }
                std::cout << "\n";
            }
            if (!p_kp.empty())
            {
                std::cout << "Policy Kp: ";
                for (const auto& k : p_kp)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!p_kd.empty())
            {
                std::cout << "Policy Kd: ";
                for (const auto& k : p_kd)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!policy_to_actual_map.empty())
            {
                std::cout << "Policy to Actual Map: ";
                for (const auto& map : policy_to_actual_map)
                {
                    std::cout << map << ", ";
                }
                std::cout << "\n";
            }
            if (!actual_to_policy_map.empty())
            {
                std::cout << "Actual to Policy Map: ";
                for (const auto& map : actual_to_policy_map)
                {
                    std::cout << map << ", ";
                }
                std::cout << "\n";
            }
            if (!urdf_offset.empty())
            {
                std::cout << "URDF Offset: ";
                for (const auto& offset : urdf_offset)
                {
                    std::cout << offset << ", ";
                }
                std::cout << "\n";
            }
            if (!clip_output_lower.empty())
            {
                std::cout << "Clip Output Lower: ";
                for (const auto& lower : clip_output_lower)
                {
                    std::cout << lower << ", ";
                }
                std::cout << "\n";
            }
            if (!clip_output_upper.empty())
            {
                std::cout << "Clip Output Upper: ";
                for (const auto& upper : clip_output_upper)
                {
                    std::cout << upper << ", ";
                }
                std::cout << "\n";
            }
            if (!waypoint_files.empty())
            {
                std::cout << "Waypoint Files: ";
                for (const auto& file : waypoint_files)
                {
                    std::cout << file << ", ";
                }
                std::cout << "\n";
            }
            if (!series_trajectory.empty())
            {
                std::cout << "Series Trajectory: \n";
                for (const auto& item : series_trajectory)
                {
                    std::cout << "  Name: " << item.first << "\n";
                    std::cout << "  Waypoint Files: ";
                    for (const auto& file : item.second)
                    {
                        std::cout << file << ", ";
                    }
                    std::cout << "\n";
                }
            }
            if (!multi_configs.empty())
            {
                std::cout << "Waypoint Configs: \n";
                for (const auto& pitem : multi_configs)
                {
                    std::cout << "  Production type: " << pitem.first << "\n";
                    for (const auto& item : pitem.second)
                    {
                        std::cout << item.second.toString() << std::endl;
                    }
                }
            }
            if (!series_configs.empty())
            {
                std::cout << "Series Waypoint Configs: \n";
                for (const auto& pitem : series_configs)
                {
                    std::cout << "  Production type: " << pitem.first << "\n";
                    for (const auto& item : pitem.second)
                    {
                        std::cout << item.second.toString() << std::endl;
                    }
                }
            }
            if (!byMotorId.direction.empty())
            {
                std::cout << "By Motor ID Direction: ";
                for (const auto& dir : byMotorId.direction)
                {
                    std::cout << dir << ", ";
                }
                std::cout << "\n";
            }
            if (!byMotorId.kp.empty())
            {
                std::cout << "By Motor ID Kp: ";
                for (const auto& k : byMotorId.kp)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!byMotorId.kd.empty())
            {
                std::cout << "By Motor ID Kd: ";
                for (const auto& k : byMotorId.kd)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!byMotorId.p_kp.empty())
            {
                std::cout << "By Motor ID Policy Kp: ";
                for (const auto& k : byMotorId.p_kp)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!byMotorId.p_kd.empty())
            {
                std::cout << "By Motor ID Policy Kd: ";
                for (const auto& k : byMotorId.p_kd)
                {
                    std::cout << k << ", ";
                }
                std::cout << "\n";
            }
            if (!byMotorId.urdf_offset.empty())
            {
                std::cout << "By Motor ID URDF Offset: ";
                for (const auto& offset : byMotorId.urdf_offset)
                {
                    std::cout << offset << ", ";
                }
                std::cout << "\n";
            }
            if (!byMotorId.clip_output_lower.empty())
            {
                std::cout << "By Motor ID Clip Output Lower: ";
                for (const auto& lower : byMotorId.clip_output_lower)
                {
                    std::cout << lower << ", ";
                }
                std::cout << "\n";
            }
            if (!byMotorId.clip_output_upper.empty())
            {
                std::cout << "By Motor ID Clip Output Upper: ";
                for (const auto& upper : byMotorId.clip_output_upper)
                {
                    std::cout << upper << ", ";
                }
                std::cout << "\n";
            }
            std::cout << "=====================================pd   end================================" << std::endl;
        }
        //=============================================RLConfigGroup================================================
        RLConfigGroup::RLConfigGroup(const std::string& configPath, const std::string policyPath, const std::string& rlConfigFile, const std::string& actionPath)
        {
            if (!loadFromYaml(configPath, policyPath, rlConfigFile, actionPath))
            {
                std::cerr << "Failed to load RLConfig from " << rlConfigFile << " policy path: " << policyPath << " action path: " << actionPath << std::endl;
            }
        }

        bool RLConfigGroup::loadFromYaml(const std::string& configPath, const std::string policyPath, const std::string& rlConfigFile, const std::string& actionPath)
        {
            try
            {
                std::vector<std::string> futurePaths;
                std::cout << "Loading RLConfigGroup from: " << rlConfigFile << std::endl;
                YAML::Node config = YAML::LoadFile(rlConfigFile);
                if (config["walk"])
                {
                    for (const auto& item : config["walk"])
                    {
                        RLConfig rlConfig;
                        std::string path = item["path"].as<std::string>();
                        std::string name = item["name"].as<std::string>();
                        if (!rlConfig.loadFromYaml(configPath + path))
                        {
                            continue;
                        }
                        rlConfig.name = name;                                          // 设置名称
                        rlConfig.type = "walk";                                        // 设置类型
                        rlConfig.policy_path = policyPath + rlConfig.policy_file_name; // 设置策略路径
                        walk_.push_back(std::move(rlConfig));
                        policies_.push_back(rlConfig.name); // 添加到策略列表
                        walkPolicies_.push_back(rlConfig.name);
                    }
                }
                if (config["motion"])
                {
                    for (const auto& item : config["motion"])
                    {
                        RLConfig rlConfig;
                        std::string path = item["path"].as<std::string>();
                        std::string name = item["name"].as<std::string>();
                        if (!rlConfig.loadFromYaml(configPath + path))
                        {
                            continue;
                        }
                        rlConfig.name = name;                                                      // 设置名称
                        rlConfig.type = "motion";                                                  // 设置类型
                        rlConfig.policy_path = policyPath + rlConfig.policy_file_name;             // 设置策略路径
                        rlConfig.future_path = configPath + "future/" + rlConfig.future_file_name; // future路径
                        futurePaths.push_back(rlConfig.future_path);
                        motion_.push_back(std::move(rlConfig));
                        policies_.push_back(rlConfig.name); // 添加到策略列表
                        motionPolicies_.push_back(rlConfig.name);
                    }
                }
                if (config["up"])
                {
                    for (const auto& item : config["up"])
                    {
                        RLConfig rlConfig;
                        std::string path = item["path"].as<std::string>();
                        std::string name = item["name"].as<std::string>();
                        if (!rlConfig.loadFromYaml(configPath + path))
                        {
                            continue;
                        }
                        rlConfig.name = name;                                                      // 设置名称
                        rlConfig.type = "up";                                                      // 设置类型
                        rlConfig.policy_path = policyPath + rlConfig.policy_file_name;             // 设置策略路径
                        rlConfig.future_path = configPath + "future/" + rlConfig.future_file_name; // future路径
                        futurePaths.push_back(rlConfig.future_path);
                        up_.push_back(std::move(rlConfig));
                        upPolicies_.push_back(rlConfig.name);
                        // policies_.push_back(rlConfig.name); // 添加到策略列表
                    }
                }
                std::map<std::string, PolicyChangeConfig> policyChange;
                std::cout << "Loading base policy changes from " << actionPath + "base_policy.yaml" << std::endl;
                YAML::Node basePolicyConfig = YAML::LoadFile(actionPath + "base_policy.yaml");
                if (basePolicyConfig["policy_change_config"] && basePolicyConfig["policy_change_config"].IsSequence())
                {
                    try
                    {

                        for (size_t i = 0; i < basePolicyConfig["policy_change_config"].size(); ++i)
                        {
                            const YAML::Node& item = basePolicyConfig["policy_change_config"][i];
                            PolicyChangeConfig pcc;
                            if (item["key"])
                            {
                                pcc.key = item["key"].as<std::string>();
                                pcc.keySet = splitKeyCombination(pcc.key);
                            }
                            if (item["name"])
                            {
                                pcc.name = item["name"].as<std::string>();
                            }
                            if (item["policy"])
                            {
                                pcc.policy = item["policy"].as<std::string>();
                            }
                            if (item["pre_policy_change_description"])
                            {
                                pcc.pre_policy_change_description = item["pre_policy_change_description"].as<std::string>();
                            }
                            if (item["pre_policy_change_condition"])
                            {
                                pcc.pre_policy_change_condition = item["pre_policy_change_condition"].as<std::string>();
                            }
                            if (item["back_to_walk"])
                            {
                                pcc.back_to_walk = item["back_to_walk"].as<bool>();
                            }
                            if (item["back_to_zero"])
                            {
                                pcc.back_to_zero = item["back_to_zero"].as<bool>();
                            }
                            if (item["loop"])
                            {
                                pcc.loop = item["loop"].as<bool>();
                            }
                            if (item["threasholds"])
                            {
                                pcc.threasholds = item["threasholds"].as<std::vector<double>>();
                            }
                            if (item["path"] && std::find(policies_.begin(), policies_.end(), pcc.name) == policies_.end())
                            {
                                std::string path = item["path"].as<std::string>();
                                RLConfig rlConfig;
                                if (!rlConfig.loadFromYaml(configPath + path))
                                {
                                    continue;
                                }
                                rlConfig.name = pcc.policy;                                                // 设置名称
                                rlConfig.type = "motion";                                                  // 设置类型
                                rlConfig.policy_path = policyPath + rlConfig.policy_file_name;             // 设置策略路径
                                rlConfig.future_path = configPath + "future/" + rlConfig.future_file_name; // future路径
                                futurePaths.push_back(rlConfig.future_path);
                                motion_.push_back(std::move(rlConfig));
                                // policies_.push_back(pcc.policy); // 添加到策略列表
                                motionPolicies_.push_back(pcc.policy);
                            }
                            pcc.analyse();
                            policyChange.emplace(pcc.name, pcc);
                        }
                    }
                    catch (const YAML::Exception& e)
                    {
                        std::cerr << "Error parsing policy_change_config: " << e.what() << std::endl;
                    }
                }
                std::cout << "Loading custom policy changes from " << actionPath + "custom_action.yaml" << std::endl;
                YAML::Node customConfig = YAML::LoadFile(actionPath + "custom_action.yaml");
                if (customConfig.IsSequence())
                {
                    for (size_t i = 0; i < customConfig.size(); ++i)
                    {
                        try
                        {
                            std::string production_type = customConfig[i]["production_type"].as<std::string>();
                            std::cout << "Loading custom policy change for production type: " << production_type << std::endl;
                            if (customConfig[i]["policy_change_config"] && customConfig[i]["policy_change_config"].IsSequence())
                            {
                                for (size_t j = 0; j < customConfig[i]["policy_change_config"].size(); ++j)
                                {
                                    std::string name = customConfig[i]["policy_change_config"][j]["name"].as<std::string>();
                                    if (policyChange.find(name) == policyChange.end())
                                    {
                                        std::cerr << "Warning: custom policy change config name [" << name << "] not found in base_policy.yaml" << std::endl;
                                        continue;
                                    }
                                    auto& policyChangeConfig = policyChange[name];
                                    std::string key = customConfig[i]["policy_change_config"][j]["key"].as<std::string>();
                                    policyChangeConfig.key = key; // 使用自定义的key覆盖
                                    policyChangeConfig.keySet = splitKeyCombination(key);
                                    policyChangeConfigs_[production_type][name] = policyChangeConfig;
                                }
                            }
                        }
                        catch (const YAML::Exception& e)
                        {
                            std::cerr << "Error parsing custom policy change config[" << i << "]: " << e.what() << std::endl;
                            continue;
                        }
                    }
                }
                else
                {
                    std::cerr << "Error: custom policy change config is not a sequence (list)" << std::endl;
                }
                // 新建线程去加载future文件
                if (!futurePaths.empty())
                {
                    std::thread([futurePaths]() {
                        BYDMMCFuturePolicy::loadAllFutureFile(futurePaths);
                    }).detach();
                }
            }
            catch (const YAML::Exception& e)
            {
                std::cerr << "Error loading RLConfigGroup from YAML: " << e.what() << std::endl;
                return false;
            }

            return true;
        }

        void RLConfigGroup::print() const
        {
            std::cout << "RLConfigGroup:" << std::endl;
            for (const auto& policy : policies_)
            {
                std::cout << "Policy: " << policy << std::endl;
            }
            std::cout << "Walk configurations:" << std::endl;
            for (const auto& config : walk_)
            {
                config.print();
            }
            std::cout << "==============walk: end==================" << std::endl;
            std::cout << "Motion configurations:" << std::endl;
            for (const auto& config : motion_)
            {
                config.print();
            }
            std::cout << "===============motion end==================" << std::endl;
            std::cout << "Up configurations:" << std::endl;
            for (const auto& config : up_)
            {
                config.print();
            }
            std::cout << "policy change configurations:" << std::endl;
            for (const auto& pair : policyChangeConfigs_)
            {
                std::cout << "production type: " << pair.first << std::endl;
                for (const auto& item : pair.second)
                {
                    std::cout << item.second.toString() << std::endl;
                }
            }
            std::cout << "===============up end==================" << std::endl;
        }

        const RLConfig& RLConfigGroup::getRLConfigByName(const std::string& name) const
        {
            for (const auto& config : walk_)
            {
                if (config.name == name)
                {
                    return config;
                }
            }
            for (const auto& config : motion_)
            {
                if (config.name == name)
                {
                    return config;
                }
            }
            for (const auto& config : up_)
            {
                if (config.name == name)
                {
                    return config;
                }
            }
            std::cout << "Warning: RLConfig with name '" << name << "' not found. Returning default configuration." << std::endl;
            if (!walk_.empty())
            {
                return walk_.front(); // 默认返回第一个配置
            }
            if (!motion_.empty())
            {
                return motion_.front(); // 默认返回第一个配置
            }
            if (!up_.empty())
            {
                return up_.front(); // 默认返回第一个配置
            }
            throw std::runtime_error("No RLConfig found with name: " + name);
        }

        const RLConfig& RLConfigGroup::getCurrentRLConfig() const
        {
            if (currentPolicyIndex_ < policies_.size())
            {
                return getRLConfigByName(policies_[currentPolicyIndex_]);
            }
            throw std::out_of_range("Current policy index is out of range");
        }

        const RLConfig& RLConfigGroup::getNextRLConfig()
        {
            if (currentPolicyIndex_ + 1 < policies_.size())
            {
                currentPolicyIndex_++;
                return getCurrentRLConfig();
            }
            if (!policies_.empty())
            {
                currentPolicyIndex_ = 0; // 循环到第一个
                return getCurrentRLConfig();
            }
            throw std::out_of_range("No next RLConfig available");
        }

        const RLConfig& RLConfigGroup::getPrevRLConfig()
        {
            if (currentPolicyIndex_ > 0)
            {
                currentPolicyIndex_--;
                return getCurrentRLConfig();
            }
            if (!policies_.empty())
            {
                currentPolicyIndex_ = policies_.size() - 1; // 循环到最后一个
                return getCurrentRLConfig();
            }
            throw std::out_of_range("No previous RLConfig available");
        }

        const RLConfig& RLConfigGroup::getWalkCurrentRLConfig()
        {
            if (currentWalkPolicyIndex_ < walkPolicies_.size())
            {
                return getRLConfigByName(walkPolicies_[currentWalkPolicyIndex_]);
            }
            if (!walkPolicies_.empty())
            {
                currentWalkPolicyIndex_ = 0; // 循环到第一个
                return getRLConfigByName(walkPolicies_[currentWalkPolicyIndex_]);
            }
            throw std::out_of_range("Current walk policy index " + std::to_string(currentWalkPolicyIndex_) + " is out of range" + std::to_string(walkPolicies_.size()));
        }

        const RLConfig& RLConfigGroup::getMotionCurrentRLConfig()
        {
            if (currentMotionPolicyIndex_ < motionPolicies_.size())
            {
                return getRLConfigByName(motionPolicies_[currentMotionPolicyIndex_]);
            }
            if (!motionPolicies_.empty())
            {
                currentMotionPolicyIndex_ = 0; // 循环到第一个
                return getRLConfigByName(motionPolicies_[currentMotionPolicyIndex_]);
            }
            throw std::out_of_range("Current motion policy index " + std::to_string(currentMotionPolicyIndex_) + " is out of range" + std::to_string(motionPolicies_.size()));
        }

        const RLConfig& RLConfigGroup::getAllBodyWalkPolicy() const
        {
            // 返回第一个 motor_group == "all" 的走路算法
            for (const auto& policy : walk_)
            {
                if (policy.motor_group.find("all") != std::string::npos)
                {
                    ROS_INFO("Found all-body walk policy: %s (algorithm: %s, motor_group: %s)",
                             policy.name.c_str(), policy.algorithm.c_str(), policy.motor_group.c_str());
                    return policy;
                }
            }
            ROS_ERROR("No all-body walk policy found (motor_group == \"all\")");
            return walk_.front(); // 返回第一个作为兜底
        }

        const RLConfig& RLConfigGroup::getLowerBodyWalkPolicy() const
        {
            // 优先查找算法名为 "lr" 的下半身控制策略
            for (const auto& policy : walk_)
            {
                if (policy.motor_group == "lowerBody" && policy.algorithm == "lr")
                {
                    ROS_INFO("Found lower-body walk policy: %s (algorithm: %s, motor_group: %s)",
                             policy.name.c_str(), policy.algorithm.c_str(), policy.motor_group.c_str());
                    return policy;
                }
            }

            // 如果没有 lr，返回第一个 motor_group == "lowerBody" 的策略
            for (const auto& policy : walk_)
            {
                if (policy.motor_group == "lowerBody")
                {
                    ROS_WARN("Using fallback lower-body walk policy: %s (algorithm: %s, motor_group: %s)",
                             policy.name.c_str(), policy.algorithm.c_str(), policy.motor_group.c_str());
                    return policy;
                }
            }

            ROS_ERROR("No lower-body walk policy found (motor_group == \"lowerBody\")");
            throw std::runtime_error("No lower bydy policy");
        }

        const RLConfig& RLConfigGroup::getWalkNextRLConfig()
        {
            if (currentWalkPolicyIndex_ + 1 < walkPolicies_.size())
            {
                currentWalkPolicyIndex_++;
                return getWalkCurrentRLConfig();
            }
            if (!walkPolicies_.empty())
            {
                currentWalkPolicyIndex_ = 0; // 循环到第一个
                return getWalkCurrentRLConfig();
            }
            throw std::runtime_error("No walk RLConfig available");
        }

        const RLConfig& RLConfigGroup::getWalkPrevRLConfig()
        {
            if (currentWalkPolicyIndex_ > 0)
            {
                currentWalkPolicyIndex_--;
                return getWalkCurrentRLConfig();
            }
            if (!walkPolicies_.empty())
            {
                currentWalkPolicyIndex_ = walkPolicies_.size() - 1; // 循环到最后一个
                return getWalkCurrentRLConfig();
            }
            throw std::runtime_error("No previous walk RLConfig available");
        }

        const RLConfig& RLConfigGroup::getMotionNextRLConfig()
        {
            if (currentMotionPolicyIndex_ + 1 < motionPolicies_.size())
            {
                currentMotionPolicyIndex_++;
                return getMotionCurrentRLConfig();
            }
            if (!motionPolicies_.empty())
            {
                currentMotionPolicyIndex_ = 0; // 循环到第一个
                return getMotionCurrentRLConfig();
            }
            throw std::runtime_error("No motion RLConfig available");
        }

        const RLConfig& RLConfigGroup::getMotionPrevRLConfig()
        {
            if (currentMotionPolicyIndex_ > 0)
            {
                currentMotionPolicyIndex_--;
                return getMotionCurrentRLConfig();
            }
            if (!motionPolicies_.empty())
            {
                currentMotionPolicyIndex_ = motionPolicies_.size() - 1; // 循环到最后一个
                return getMotionCurrentRLConfig();
            }
            throw std::runtime_error("No previous motion RLConfig available");
        }

        bool PolicyChangeConfig::prePolicyChangeShouldExecute(const Command& cmd, const Robot_State& state)
        {
            std::cout << "preAction condition: [" << pre_policy_change_condition << "] cmd: " << "vx: " << cmd.vx << " vy: " << cmd.vy << " dyaw: " << cmd.dyaw << std::endl;
            if (pre_policy_change_condition == "velocity more than 0.0")
            {
                if (std::abs(cmd.vx) > 0.05 || std::abs(cmd.vy) > 0.05 || std::abs(cmd.dyaw) > 0.05)
                {
                    pre_policy_change_use_walk = true;
                }
                else
                {
                    pre_policy_change_use_walk = false;
                }
            }
            return pre_policy_change_use_walk;
        }

        bool PolicyChangeConfig::finished(const Motor_State& state) const
        {
            if (!pre_policy_change_use_walk)
            {
                return true;
            }
            if (state.q.size() < 12)
            {
                std::cout << "preActionFinished state.q.size() < 12 " << std::endl;
                return true;
            }
            auto close = [](double a, double b, double tol = 0.15) {
                return std::abs(a - b) < tol;
            };
            double leftHip = state.q[0], leftCaft = state.q[3], leftAnkle = state.q[4], rightHip = state.q[6], rightCaft = state.q[9], rightAnkle = state.q[10];
            std::cout << "preActionFinished condition: [" << pre_policy_change_description << "] " << state.q.head(12).transpose() << std::endl;
            if (threasholds.size() < 11)
            {
                std::cout << "preActionFinished threasholds.size() < 11 " << std::endl;
                return true;
            }
            if (pre_policy_change_description == "left leg front")
            {
                if (close(leftHip, threasholds[0]) && close(rightHip, threasholds[6]) &&
                    close(leftCaft, threasholds[3]) && close(rightCaft, threasholds[9]) &&
                    close(leftAnkle, threasholds[4]) && close(rightAnkle, threasholds[10]))
                {
                    return true;
                }
                return false;
            }
            else if (pre_policy_change_description == "right leg front")
            {
                if (leftHip < -0.2 && rightHip > 0.6)
                {
                    return true;
                }
                return false;
            }
            return true;
        }
    }
}
