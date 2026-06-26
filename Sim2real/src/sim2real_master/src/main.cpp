#include "sim2real/master_node.h"

// debug
#ifdef PLATFORM_X86_64
#include "impl/hightorque_config.h"
#include "nlohmann/json.hpp"
#include "policy/bydmmc_policy.h"
#include "policy/amp_policy.h"
#include "robot_data.h"
std::string productionType_ = "test";

bool initTrajectory(hightorque::config::PDConfig& pdInfo_)
{
    // trajectory
    ROS_INFO("Sim2Real %s()", __FUNCTION__);
    std::string packagePath = ros::package::getPath("sim2real");
    std::vector<hightorque::config::MultiWaypointConfig> waypoints;

    for (auto& config : pdInfo_.multi_configs[productionType_])
    {
        config.second.path = (packagePath + "/" + config.second.path);
        waypoints.push_back(config.second);
        ROS_INFO("Sim2Real trajectory (%s) file: %s", config.first.c_str(), config.second.path.c_str());
    }
    // TODO
    for (auto series : pdInfo_.series_configs[productionType_])
    {
        auto seriesName = series.first;
        for (auto& config : series.second.waypointVec)
        {
            config.path = (packagePath + "/" + config.path);
            waypoints.push_back(config);
            ROS_INFO("Sim2Real series trajectory (%s) file: %s", config.name.c_str(), config.path.c_str());
        }
    }

    auto trajectory_ = &hightorque::utils::TrajectoryGenerator::getInstance(pdInfo_.urdf_offset, pdInfo_.direction, waypoints, "pi");
    // trajectory_->refreshSingle(pdInfo_.urdf_offset, pdInfo_.direction, modelType_);
    return true;
}
#endif
int main(int argc, char** argv)
{

    ros::init(argc, argv, "masternode");
#ifdef PLATFORM_X86_64
    auto close = [](double a, double b, double tol = 0.5) {
        return std::abs(a - b) < tol;
    };

    std::cout << close(-0.57, -0.53) << " " << close(-0.907292, -0.85) << " "
              << close(-0.43, -0.45) << " " << close(0.19, 0.43) << " "
              << close(1.02, 1.5) << " " << close(0.4926, 0.4) << " "
              << std::endl;
    ros::NodeHandle nh_param("~");
    nh_param.param<std::string>("production_type", productionType_, "test");
    std::string config_path = "", pd_config_file, rl_config_file, policy_path, actionConfigPath;
    nh_param.param<std::string>("config_path", config_path, "_config.yaml");
    nh_param.param<std::string>("pd_config_file", pd_config_file, "_config.yaml");
    nh_param.param<std::string>("rl_config_file", rl_config_file, "_config.yaml");
    nh_param.param<std::string>("policy_path", policy_path, "_config.yaml");
    nh_param.param<std::string>("action_path", actionConfigPath, "");
    hightorque::config::PDConfig pdInfo(pd_config_file, actionConfigPath);
    hightorque::config::RLConfigGroup rlInfoGroup(config_path, policy_path, rl_config_file, actionConfigPath);
    // rlInfoGroup.print();
    try
    {
        pdInfo.refreshRLConfig(rlInfoGroup.getCurrentRLConfig());
        pdInfo.refreshByMotorIdVec();
        auto rl = rlInfoGroup.getCurrentRLConfig();
        auto policy = std::make_shared<hightorque::AMPPolicy>(rl, pdInfo);
        Eigen::VectorXd obs = Eigen::VectorXd::Zero(rl.num_single_obs);
        auto robotState = Robot_State(pdInfo.dofs);
        Command cmd1, cmd2;
        cmd1.joy.reset();
        cmd2.joy.reset();

        cmd1.vx = -2.0;
        cmd2.vx = -2.0;
        cmd1.joy.LT = -1.0;
        cmd1.joy.DPD = 1.0;
        policy->updateObservation(0.02, Robot_State(pdInfo.dofs), Robot_Output(pdInfo.dofs),
                                  cmd1, obs, false);
        policy->updateObservation(0.02, Robot_State(pdInfo.dofs), Robot_Output(pdInfo.dofs),
                                  cmd2, obs, false);
        cmd2.joy.LT = -1.0;
        cmd2.joy.DPU = 1.0;
        policy->updateObservation(0.02, Robot_State(pdInfo.dofs), Robot_Output(pdInfo.dofs),
                                  cmd2, obs, false);
        return 0;

        // pdInfo.print();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("Failed to refresh RLConfig: %s", e.what());
        return -1;
    }

    ROS_INFO("init trajectory");
    initTrajectory(pdInfo);
    std::set<std::string> key{"lt", "rt", "a"};
    auto itemMulti = std::find_if(pdInfo.multi_configs[productionType_].begin(), pdInfo.multi_configs[productionType_].end(),
                                  [&key](const std::pair<std::string, hightorque::config::MultiWaypointConfig>& pair) {
                                      return pair.second.keySet == key;
                                  });
    ROS_INFO("find multi %s", itemMulti->first.c_str());
    key = {"lt", "y"};
    auto itemPolicy = std::find_if(rlInfoGroup.getPolicyChangeConfigs(productionType_).begin(), rlInfoGroup.getPolicyChangeConfigs(productionType_).end(),
                                   [&key](const std::pair<std::string, hightorque::config::PolicyChangeConfig>& pair) {
                                       // ROS_INFO("Checking policy name: %s against %s", key.c_str(), pair.second.key.c_str());
                                       return pair.second.keySet == key;
                                   });
    ROS_INFO("find policy %s", itemPolicy->first.c_str());

    // ROS_INFO("init byd");
    // hightorque::BYDMMCFuturePolicy pdPolicy(rlInfoGroup.getCurrentRLConfig(), pdInfo);
    return 0;

    try
    {
        pdInfo.refreshRLConfig(rlInfoGroup.getNextRLConfig());
        pdInfo.refreshByMotorIdVec();
        pdInfo.print();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("Failed to create MasterNode: %s", e.what());
        return -1;
    }
#endif
    ROS_INFO("masternode start");
    hightorque::MasterNode masternode;

    // 使用异步 spinner，指定线程数（例如 4 个线程处理回调）
    ros::AsyncSpinner spinner(4);
    spinner.start();

    // 等待退出（比如 ctrl+c 或某个 shutdown 条件）
    ros::waitForShutdown();
    ROS_INFO("masternode shutdown");
    return 0;
}
