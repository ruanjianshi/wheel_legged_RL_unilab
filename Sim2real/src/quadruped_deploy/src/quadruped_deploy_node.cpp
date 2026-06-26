// quadruped_deploy_node.cpp
//
// 四足机器人独立部署节点（HTDW-4438 + HIMloco RKNN 策略）
// 设计参考: deploy_mujoco/deploy_galileo_keyboard.py
//
// 数据流:
//   IMU -> /imu/data
//   Joy -> /joy_msg
//   Cmd -> /cmd_vel
//   Motor -> livelybot_serial::robot::Motors[i]->pos_vel_kp_kd(...)
//
// 状态机 (本节点内部):
//   IDLE   : 不输出 PD（电机 0 电流），等待手柄指令
//   STAND  : 平滑插值到 default_angles（持续 stand_duration 秒）
//   RUN    : 调用 RKNN 进行策略推理，跟随 /cmd_vel
//   SIT    : 平滑插值到 sit_angles（持续 sit_duration 秒）
//
// 手柄按键 (LT+RT 作为安全锁):
//   LT+RT + start  => STAND
//   LT+RT + lb     => 切换 STAND <-> RUN
//   LT+RT + rb     => SIT
//   center         => 紧急停止 (回到 IDLE, 0 电流)
//
// 电气零位 == 训练 default_angles 站立姿态 (已通过电机 set_zero 校准)
// 因此运行时坐标转换非常简洁:
//   urdf_q  = m.q   * direction + default_angles
//   m.target = (urdf_target - default_angles) * direction
//   obs.qj   = (urdf_q - default_angles) * scale = m.q * direction * scale
//   obs.dqj  = m.dq * direction * scale

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32.h>
#include <sim2real_msg/Joy.h>

#include <livelybot_serial/hardware/robot.h>
#include <rknn_api.h>

#include <Eigen/Dense>

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace qd
{

enum class State : int
{
    IDLE = 0,
    STAND = 1,
    RUN = 2,
    SIT = 3,
};

static const char* stateName(State s)
{
    switch (s) {
        case State::IDLE:  return "IDLE";
        case State::STAND: return "STAND";
        case State::RUN:   return "RUN";
        case State::SIT:   return "SIT";
    }
    return "?";
}

struct Cfg
{
    int                 num_actions{12};
    int                 num_one_step_obs{45};
    int                 frame_stack{6};
    int                 num_obs{270};

    std::vector<int>    direction;        // size = num_actions
    std::vector<double> default_angles;   // URDF 默认角度
    std::vector<double> sit_angles;
    std::vector<double> kp;
    std::vector<double> kd;
    std::vector<double> joint_lower;      // URDF 限位
    std::vector<double> joint_upper;

    double lin_vel_scale{2.0};
    double ang_vel_scale{0.25};
    double dof_pos_scale{1.0};
    double dof_vel_scale{0.05};
    double action_scale{0.25};
    std::vector<double> cmd_scale;        // size 3

    double clip_obs{100.0};
    double clip_action{0.5};

    double pd_hz{1000.0};
    double rl_hz{50.0};

    double cmd_vx_max{1.0};
    double cmd_vy_max{0.7};
    double cmd_wz_max{2.0};

    double stand_duration{3.0};
    double sit_duration{3.0};

    std::string policy_path_full;

    bool loadFromRos(ros::NodeHandle& nh)
    {
        bool ok = true;
        ok &= nh.getParam("num_actions",      num_actions);
        ok &= nh.getParam("num_one_step_obs", num_one_step_obs);
        ok &= nh.getParam("frame_stack",      frame_stack);
        ok &= nh.getParam("num_obs",          num_obs);

        ok &= nh.getParam("direction",        direction);
        ok &= nh.getParam("default_angles",   default_angles);
        ok &= nh.getParam("sit_angles",       sit_angles);
        ok &= nh.getParam("kp",               kp);
        ok &= nh.getParam("kd",               kd);
        ok &= nh.getParam("joint_lower",      joint_lower);
        ok &= nh.getParam("joint_upper",      joint_upper);

        nh.param("lin_vel_scale",   lin_vel_scale,   2.0);
        nh.param("ang_vel_scale",   ang_vel_scale,   0.25);
        nh.param("dof_pos_scale",   dof_pos_scale,   1.0);
        nh.param("dof_vel_scale",   dof_vel_scale,   0.05);
        nh.param("action_scale",    action_scale,    0.25);
        ok &= nh.getParam("cmd_scale",        cmd_scale);

        nh.param("clip_obs",        clip_obs,        100.0);
        nh.param("clip_action",     clip_action,     0.5);
        nh.param("pd_hz",           pd_hz,           1000.0);
        nh.param("rl_hz",           rl_hz,           50.0);
        nh.param("cmd_vx_max",      cmd_vx_max,      1.0);
        nh.param("cmd_vy_max",      cmd_vy_max,      0.7);
        nh.param("cmd_wz_max",      cmd_wz_max,      2.0);
        nh.param("stand_duration",  stand_duration,  3.0);
        nh.param("sit_duration",    sit_duration,    3.0);

        ok &= nh.getParam("policy_path_full", policy_path_full);

        if (!ok) {
            ROS_ERROR("[Cfg] required parameter missing");
            return false;
        }
        if ((int)direction.size() != num_actions ||
            (int)default_angles.size() != num_actions ||
            (int)sit_angles.size() != num_actions ||
            (int)kp.size() != num_actions ||
            (int)kd.size() != num_actions ||
            (int)joint_lower.size() != num_actions ||
            (int)joint_upper.size() != num_actions ||
            (int)cmd_scale.size() != 3) {
            ROS_ERROR("[Cfg] vector size mismatch num_actions=%d", num_actions);
            return false;
        }
        return true;
    }
};

}  // namespace qd

namespace qd
{

class RKNNRunner
{
public:
    bool init(const std::string& path, int input_dim, int output_dim)
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { ROS_ERROR("RKNN file open failed: %s", path.c_str()); return false; }
        auto sz = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> buf(sz);
        f.read(buf.data(), sz);
        int ret = rknn_init(&ctx_, buf.data(), (uint32_t)sz, 0, nullptr);
        if (ret < 0) { ROS_ERROR("rknn_init failed: %d", ret); return false; }
        rknn_input_output_num io;
        ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
        if (ret < 0) { ROS_ERROR("rknn_query IO failed: %d", ret); return false; }
        ROS_INFO("[RKNN] inputs=%u outputs=%u (expect 1/1)", io.n_input, io.n_output);
        input_dim_ = input_dim;
        output_dim_ = output_dim;
        memset(&in_,  0, sizeof(in_));
        memset(&out_, 0, sizeof(out_));
        in_.index = 0;
        in_.type  = RKNN_TENSOR_FLOAT32;
        in_.fmt   = RKNN_TENSOR_NHWC;
        in_.size  = input_dim_ * sizeof(float);
        in_.pass_through = 0;
        out_.index = 0;
        out_.want_float = 1;
        return true;
    }

    bool run(const float* obs, float* action_out)
    {
        in_.buf = const_cast<float*>(obs);
        int ret = rknn_inputs_set(ctx_, 1, &in_);
        if (ret < 0) { ROS_ERROR_THROTTLE(1, "rknn_inputs_set %d", ret); return false; }
        ret = rknn_run(ctx_, nullptr);
        if (ret < 0) { ROS_ERROR_THROTTLE(1, "rknn_run %d", ret); return false; }
        ret = rknn_outputs_get(ctx_, 1, &out_, nullptr);
        if (ret < 0) { ROS_ERROR_THROTTLE(1, "rknn_outputs_get %d", ret); return false; }
        const float* p = static_cast<const float*>(out_.buf);
        std::memcpy(action_out, p, output_dim_ * sizeof(float));
        rknn_outputs_release(ctx_, 1, &out_);
        return true;
    }

    ~RKNNRunner() { if (ctx_) rknn_destroy(ctx_); }

private:
    rknn_context ctx_{0};
    rknn_input   in_{};
    rknn_output  out_{};
    int input_dim_{0};
    int output_dim_{0};
};

}  // namespace qd

namespace qd
{

class DeployNode
{
public:
    DeployNode(ros::NodeHandle& nh, ros::NodeHandle& nhp)
        : nh_(nh), nhp_(nhp)
    {}

    bool init()
    {
        if (!cfg_.loadFromRos(nhp_)) return false;

        // === 硬件初始化 (livelybot_serial::robot 自动从 rosparam 读取硬件配置) ===
        ROS_INFO("[Deploy] creating livelybot_serial::robot ...");
        robot_ = std::make_shared<livelybot_serial::robot>();
        if ((int)robot_->Motors.size() != cfg_.num_actions) {
            ROS_ERROR("[Deploy] motor count %zu != num_actions %d",
                      robot_->Motors.size(), cfg_.num_actions);
            return false;
        }
        ROS_INFO("[Deploy] %zu motors ready", robot_->Motors.size());

        // === RKNN 加载 ===
        if (!rknn_.init(cfg_.policy_path_full, cfg_.num_obs, cfg_.num_actions)) return false;
        ROS_INFO("[Deploy] policy loaded: %s", cfg_.policy_path_full.c_str());

        // === 缓冲区分配 ===
        N_ = cfg_.num_actions;
        q_.assign(N_, 0.0);
        dq_.assign(N_, 0.0);
        target_q_.assign(N_, 0.0);   // URDF 坐标系
        last_action_.assign(N_, 0.0f);
        obs_.assign(cfg_.num_obs, 0.0f);

        cmd_[0] = cmd_[1] = cmd_[2] = 0.0;
        ang_vel_[0] = ang_vel_[1] = ang_vel_[2] = 0.0;
        gravity_[0] = 0; gravity_[1] = 0; gravity_[2] = -1.0;

        state_.store((int)State::IDLE);
        stand_start_q_.assign(N_, 0.0);
        sit_start_q_.assign(N_, 0.0);

        // === ROS 订阅 ===
        sub_imu_  = nh_.subscribe("/imu/data", 1,  &DeployNode::onImu,  this);
        sub_joy_  = nh_.subscribe("/joy_msg", 2,  &DeployNode::onJoy,  this);
        sub_cmd_  = nh_.subscribe("/cmd_vel", 5,  &DeployNode::onCmdVel, this);

        pub_state_ = nh_.advertise<sensor_msgs::JointState>("qd_motor_state", 10);
        pub_target_ = nh_.advertise<sensor_msgs::JointState>("qd_motor_target", 10);

        return true;
    }

    void start()
    {
        running_.store(true);
        pd_thread_ = std::thread(&DeployNode::pdLoop, this);
        rl_thread_ = std::thread(&DeployNode::rlLoop, this);
        log_thread_ = std::thread(&DeployNode::logLoop, this);
        ROS_INFO("[Deploy] started, initial state = IDLE");
    }

    void stop()
    {
        running_.store(false);
        if (pd_thread_.joinable())  pd_thread_.join();
        if (rl_thread_.joinable())  rl_thread_.join();
        if (log_thread_.joinable()) log_thread_.join();
        // 安全释放电机
        if (robot_) {
            for (auto* m : robot_->Motors) m->stop();
        }
        ROS_INFO("[Deploy] stopped");
    }

private:
    // ===== 回调 =====
    void onImu(const sensor_msgs::Imu::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lk(imu_mtx_);
        ang_vel_[0] = msg->angular_velocity.x;
        ang_vel_[1] = msg->angular_velocity.y;
        ang_vel_[2] = msg->angular_velocity.z;
        // 重力方向: 由四元数算出的"重力在 base 坐标系中的方向(单位向量)"
        // 与 deploy_galileo_keyboard.py::get_gravity_orientation 对齐
        const double qw = msg->orientation.w;
        const double qx = msg->orientation.x;
        const double qy = msg->orientation.y;
        const double qz = msg->orientation.z;
        gravity_[0] = 2.0 * (-qz * qx + qw * qy);
        gravity_[1] = -2.0 * (qz * qy + qw * qx);
        gravity_[2] = 1.0 - 2.0 * (qw * qw + qz * qz);
    }

    void onCmdVel(const geometry_msgs::Twist::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lk(cmd_mtx_);
        cmd_[0] = std::clamp(msg->linear.x,  -cfg_.cmd_vx_max, cfg_.cmd_vx_max);
        cmd_[1] = std::clamp(msg->linear.y,  -cfg_.cmd_vy_max, cfg_.cmd_vy_max);
        cmd_[2] = std::clamp(msg->angular.z, -cfg_.cmd_wz_max, cfg_.cmd_wz_max);
    }

    void onJoy(const sim2real_msg::Joy::ConstPtr& msg)
    {
        // 紧急停止: center 单独按下
        if (msg->center > 0.5) {
            transitionTo(State::IDLE);
            return;
        }
        // 必须 LT+RT 同时按下作为安全锁
        const bool armed = (msg->lt < -0.5) && (msg->rt < -0.5);
        if (!armed) { lb_latch_ = false; return; }

        if (msg->start > 0.5) {
            transitionTo(State::STAND);
            return;
        }
        if (msg->rb > 0.5) {
            transitionTo(State::SIT);
            return;
        }
        if (msg->lb > 0.5 && !lb_latch_) {
            lb_latch_ = true;
            int s = state_.load();
            if (s == (int)State::RUN) {
                transitionTo(State::STAND);
            } else if (s == (int)State::STAND) {
                transitionTo(State::RUN);
            } else {
                ROS_WARN("[Joy] LB ignored, must be in STAND or RUN, current=%s",
                         stateName((State)s));
            }
        } else if (msg->lb < 0.5) {
            lb_latch_ = false;
        }
    }

    // ===== 状态机切换 =====
    void transitionTo(State target)
    {
        State cur = (State)state_.load();
        if (cur == target) return;
        // 安全规则:
        //   - RUN 只能从 STAND 进入 (确保腿已就位)
        //   - SIT/STAND/IDLE 任意切换
        if (target == State::RUN && cur != State::STAND) {
            ROS_WARN("[FSM] reject %s -> RUN (must be STAND first)", stateName(cur));
            return;
        }
        ROS_INFO("[FSM] %s -> %s", stateName(cur), stateName(target));
        // 进入 STAND/SIT 时记录起始姿态用于插值
        readJointState();
        if (target == State::STAND) {
            std::lock_guard<std::mutex> lk(traj_mtx_);
            for (int i = 0; i < N_; ++i) stand_start_q_[i] = q_[i];
            traj_t0_ = ros::Time::now();
        } else if (target == State::SIT) {
            std::lock_guard<std::mutex> lk(traj_mtx_);
            for (int i = 0; i < N_; ++i) sit_start_q_[i] = q_[i];
            traj_t0_ = ros::Time::now();
        }
        if (target == State::RUN) {
            // 复位 last_action 与 obs 历史
            std::lock_guard<std::mutex> lk(rl_mtx_);
            std::fill(last_action_.begin(), last_action_.end(), 0.0f);
            std::fill(obs_.begin(), obs_.end(), 0.0f);
            for (int i = 0; i < N_; ++i) target_q_[i] = cfg_.default_angles[i];
        }
        state_.store((int)target);
    }

    // ===== 读取电机当前状态 (URDF 坐标系) =====
    // urdf_q  = m.q  * direction + default_angles
    // urdf_dq = m.dq * direction
    void readJointState()
    {
        std::lock_guard<std::mutex> lk(joint_mtx_);
        for (int i = 0; i < N_; ++i) {
            auto* st = robot_->Motors[i]->get_current_motor_state();
            const double mq = st ? st->position : 0.0;
            const double mv = st ? st->velocity : 0.0;
            q_[i]  = mq * cfg_.direction[i] + cfg_.default_angles[i];
            dq_[i] = mv * cfg_.direction[i];
        }
    }

    // 平滑 SmoothStep 插值 t in [0,1]
    static double smoothStep(double t)
    {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }

    // ===== PD 循环 1000Hz =====
    void pdLoop()
    {
        ros::Rate rate(cfg_.pd_hz);
        while (ros::ok() && running_.load()) {
            const State s = (State)state_.load();
            readJointState();

            switch (s) {
                case State::IDLE: {
                    // 0 电流: 用 0 增益位控 (kp=0, kd=0)
                    for (int i = 0; i < N_; ++i) {
                        robot_->Motors[i]->pos_vel_kp_kd(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                    break;
                }
                case State::STAND: {
                    interpAndCommand(/*to=*/cfg_.default_angles, cfg_.stand_duration);
                    break;
                }
                case State::SIT: {
                    interpAndCommand(/*to=*/cfg_.sit_angles, cfg_.sit_duration);
                    break;
                }
                case State::RUN: {
                    // RL 线程已经写好 target_q_ (URDF 坐标系)
                    std::vector<double> tgt(N_);
                    {
                        std::lock_guard<std::mutex> lk(target_mtx_);
                        tgt = target_q_;
                    }
                    sendUrdfTargets(tgt);
                    break;
                }
            }
            robot_->motor_send_2();
            rate.sleep();
        }
    }

    // 依据 traj_t0_ 在 [0,duration] 区间从 *_start_q_ 到 to 平滑插值
    void interpAndCommand(const std::vector<double>& to, double duration)
    {
        const State s = (State)state_.load();
        const double now = (ros::Time::now() - traj_t0_).toSec();
        const double t = std::clamp(now / std::max(duration, 1e-3), 0.0, 1.0);
        const double a = smoothStep(t);
        std::vector<double> tgt(N_);
        std::vector<double>* start_q = nullptr;
        {
            std::lock_guard<std::mutex> lk(traj_mtx_);
            start_q = (s == State::STAND) ? &stand_start_q_ : &sit_start_q_;
            for (int i = 0; i < N_; ++i) {
                tgt[i] = (*start_q)[i] * (1.0 - a) + to[i] * a;
            }
        }
        sendUrdfTargets(tgt);
    }

    // 发送 URDF 坐标系下的目标到电机
    void sendUrdfTargets(const std::vector<double>& urdf_target)
    {
        // 先在 URDF 系做 clip
        for (int i = 0; i < N_; ++i) {
            double t = std::clamp(urdf_target[i], cfg_.joint_lower[i], cfg_.joint_upper[i]);
            const double m_target = (t - cfg_.default_angles[i]) * cfg_.direction[i];
            const float kp = (float)cfg_.kp[i];
            const float kd = (float)cfg_.kd[i];
            robot_->Motors[i]->pos_vel_kp_kd((float)m_target, 0.0f, kp, kd);
        }
    }

    // ===== RL 循环 50Hz =====
    void rlLoop()
    {
        ros::Rate rate(cfg_.rl_hz);
        std::vector<float> current_obs(cfg_.num_one_step_obs, 0.0f);
        std::vector<float> action(cfg_.num_actions, 0.0f);
        while (ros::ok() && running_.load()) {
            if ((State)state_.load() != State::RUN) {
                rate.sleep();
                continue;
            }

            // === 构造单步观测 (与 deploy_galileo_keyboard.py current_obs 一致) ===
            // [cmd*scale (3), ang_vel*scale (3), gravity (3),
            //  qj=(q-default)*scale (12), dqj=dq*scale (12), last_action (12)]
            double cmd_local[3], avel[3], grav[3];
            { std::lock_guard<std::mutex> lk(cmd_mtx_); std::memcpy(cmd_local, cmd_, sizeof(cmd_)); }
            { std::lock_guard<std::mutex> lk(imu_mtx_);
              std::memcpy(avel, ang_vel_, sizeof(ang_vel_));
              std::memcpy(grav, gravity_, sizeof(gravity_));
            }
            std::vector<double> q_local, dq_local;
            { std::lock_guard<std::mutex> lk(joint_mtx_); q_local = q_; dq_local = dq_; }

            current_obs[0] = (float)(cmd_local[0] * cfg_.cmd_scale[0]);
            current_obs[1] = (float)(cmd_local[1] * cfg_.cmd_scale[1]);
            current_obs[2] = (float)(cmd_local[2] * cfg_.cmd_scale[2]);
            current_obs[3] = (float)(avel[0] * cfg_.ang_vel_scale);
            current_obs[4] = (float)(avel[1] * cfg_.ang_vel_scale);
            current_obs[5] = (float)(avel[2] * cfg_.ang_vel_scale);
            current_obs[6] = (float)grav[0];
            current_obs[7] = (float)grav[1];
            current_obs[8] = (float)grav[2];
            for (int i = 0; i < N_; ++i) {
                current_obs[9 + i]                 = (float)((q_local[i] - cfg_.default_angles[i]) * cfg_.dof_pos_scale);
                current_obs[9 + N_ + i]            = (float)(dq_local[i] * cfg_.dof_vel_scale);
                current_obs[9 + 2 * N_ + i]        = last_action_[i];
            }
            // clip
            for (auto& v : current_obs) v = std::clamp(v, (float)-cfg_.clip_obs, (float)cfg_.clip_obs);

            // === frame_stack: 把 current_obs 塞到 obs 头部, 旧的整体后移 ===
            {
                std::lock_guard<std::mutex> lk(rl_mtx_);
                const int S = cfg_.num_one_step_obs;
                // shift: obs[S:] = obs[:-S]; obs[:S] = current_obs
                std::memmove(obs_.data() + S, obs_.data(),
                             (cfg_.num_obs - S) * sizeof(float));
                std::memcpy(obs_.data(), current_obs.data(), S * sizeof(float));
            }

            // === 推理 ===
            std::vector<float> obs_copy;
            { std::lock_guard<std::mutex> lk(rl_mtx_); obs_copy = obs_; }
            if (!rknn_.run(obs_copy.data(), action.data())) continue;

            // clip action + 计算 target_q (URDF)
            std::vector<double> new_target(N_);
            for (int i = 0; i < N_; ++i) {
                action[i] = std::clamp(action[i], (float)-cfg_.clip_action, (float)cfg_.clip_action);
                new_target[i] = action[i] * cfg_.action_scale + cfg_.default_angles[i];
            }
            {
                std::lock_guard<std::mutex> lk(rl_mtx_);
                std::memcpy(last_action_.data(), action.data(), N_ * sizeof(float));
            }
            {
                std::lock_guard<std::mutex> lk(target_mtx_);
                target_q_ = new_target;
            }

            rate.sleep();
        }
    }

    // ===== 日志循环 (10Hz 发布关节状态) =====
    void logLoop()
    {
        ros::Rate rate(10.0);
        while (ros::ok() && running_.load()) {
            sensor_msgs::JointState st, tg;
            st.header.stamp = ros::Time::now();
            tg.header.stamp = st.header.stamp;
            std::vector<double> q_local, dq_local, t_local;
            { std::lock_guard<std::mutex> lk(joint_mtx_);  q_local = q_; dq_local = dq_; }
            { std::lock_guard<std::mutex> lk(target_mtx_); t_local = target_q_; }
            st.name.reserve(N_); st.position.reserve(N_); st.velocity.reserve(N_);
            tg.position.reserve(N_);
            for (int i = 0; i < N_; ++i) {
                st.name.push_back("j" + std::to_string(i));
                st.position.push_back(q_local[i]);
                st.velocity.push_back(dq_local[i]);
                tg.position.push_back(t_local[i]);
            }
            pub_state_.publish(st);
            pub_target_.publish(tg);
            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_, nhp_;
    Cfg cfg_;
    int N_{12};
    std::shared_ptr<livelybot_serial::robot> robot_;
    RKNNRunner rknn_;

    std::atomic<int>  state_{(int)State::IDLE};
    std::atomic<bool> running_{false};
    bool lb_latch_{false};

    // 共享数据
    std::mutex imu_mtx_;
    double ang_vel_[3]{0,0,0}, gravity_[3]{0,0,-1};
    std::mutex cmd_mtx_;
    double cmd_[3]{0,0,0};
    std::mutex joint_mtx_;
    std::vector<double> q_, dq_;
    std::mutex target_mtx_;
    std::vector<double> target_q_;
    std::mutex rl_mtx_;
    std::vector<float> obs_;
    std::vector<float> last_action_;
    std::mutex traj_mtx_;
    std::vector<double> stand_start_q_, sit_start_q_;
    ros::Time traj_t0_;

    ros::Subscriber sub_imu_, sub_joy_, sub_cmd_;
    ros::Publisher  pub_state_, pub_target_;
    std::thread     pd_thread_, rl_thread_, log_thread_;
};

}  // namespace qd

int main(int argc, char** argv)
{
    ros::init(argc, argv, "quadruped_deploy_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");

    qd::DeployNode node(nh, nhp);
    if (!node.init()) {
        ROS_ERROR("init failed");
        return 1;
    }
    node.start();

    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();

    node.stop();
    return 0;
}
