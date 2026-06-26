#ifndef ROBOT_DATA_H
#define ROBOT_DATA_H

#include </usr/include/eigen3/Eigen/Dense>
#include <algorithm>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/Twist.h>
#include "sim2real_msg/Joy.h"
#include "sim2real_msg/MtrState.h"
#include "sim2real_msg/RbtState.h"
#include "sim2real_msg/MtrTarget.h"
#include "sim2real_msg/RbtTarget.h"
#include "sim2real_msg/Orient.h"
#include "sim2real_msg/PD2RL.h"
#include "sim2real_msg/RL2PD.h"
#include <deque>
#include <sstream>
struct Joy
{
        int A;
        int B;
        int X;
        int Y;
        int L;
        int R;
        int LB;
        int RB;
        float LT;
        float RT;
        bool DPU, DPD, DPL, DPR;
        void set(const sim2real_msg::Joy::ConstPtr& msg)
        {
            A = msg->a;
            B = msg->b;
            X = msg->x;
            Y = msg->y;
            L = msg->L;
            R = msg->R;
            LB = msg->lb;
            RB = msg->rb;
            LT = msg->lt;
            RT = msg->rt;
            DPL = msg->dpad_horizontal == 1;
            DPR = msg->dpad_horizontal == -1;
            DPU = msg->dpad_vertical == 1;
            DPD = msg->dpad_vertical == -1;
        }
        void reset()
        {
            A = 0;
            B = 0;
            X = 0;
            Y = 0;
            L = 0;
            R = 0;
            LB = 0;
            RB = 0;
            LT = 0.0;
            RT = 0.0;
            DPU = false;
            DPD = false;
            DPL = false;
            DPR = false;
        }
};
struct Command
{
        double vx = 0.0;
        double vy = 0.0;
        double dyaw = 0.0;
        Joy joy;
};

enum Init_State
{
    I_STATE_SAMPLE = 2002,
    I_STATE_RUNNING,
};

enum Ctrl_State
{
    C_STATE_WAITING = 206,
    C_STATE_INIT,
    C_STATE_RUNNING,
    C_STATE_STANDBY,
    C_STATE_SHUT_DOWN,
    C_STATE_FORCE_SHUT_DOWN,
    C_DEVELOP_CONTROL_JOINT,
    C_DEVELOP_CONTROL_MOTOR,
    C_STATE_TRANSITION
};

inline static std::string ctrlStateToString(Ctrl_State state)
{
    switch (state)
    {
        case C_STATE_WAITING:
            return "waiting";
        case C_STATE_INIT:
            return "init";
        case C_STATE_RUNNING:
            return "running";
        case C_STATE_STANDBY:
            return "standby";
        case C_STATE_SHUT_DOWN:
            return "shutdown";
        case C_STATE_FORCE_SHUT_DOWN:
            return "force_shut_down";
        case C_DEVELOP_CONTROL_JOINT:
            return "develop control joint";
        case C_DEVELOP_CONTROL_MOTOR:
            return "develop control motor";
        case C_STATE_TRANSITION:
            return "transition";
            break;
    }
}

enum Shut_Down_State
{
    D_STATE_STAND = 608,
    D_STATE_CALC,
    D_STATE_SQUAT,
    D_STATE_OFF,
};
struct Motor_State;
struct Motor_Output
{
        Eigen::VectorXd target_q;
        Eigen::VectorXd target_dq;
        Eigen::VectorXd target_tau;
        Eigen::VectorXd target_kp;
        Eigen::VectorXd target_kd;
        Motor_Output() = default;
        Motor_Output(int size)
        {
            resize(size);
        }
        Motor_Output(float q, float dq, float tau, float kp, float kd)
        {
            resize(1);
            target_q[0] = q;
            target_dq[0] = dq;
            target_tau[0] = tau;
            target_kp[0] = kp;
            target_kd[0] = kd;
        }
        void resize(int size)
        {
            target_q.resize(size);
            target_dq.resize(size);
            target_tau.resize(size);
            target_kp.resize(size);
            target_kd.resize(size);
            target_q.setZero();
            target_dq.setZero();
            target_tau.setZero();
            target_kp.setZero();
            target_kd.setZero();
        }
        bool isZero(double eps = 1e-6) const
        {
            return target_q.isZero(eps) && target_dq.isZero(eps) && target_tau.isZero(eps);
        }
        bool isAllValueZero() const
        {
            return target_q.array().abs().matrix().sum() == 0.0 && target_dq.array().abs().matrix().sum() == 0.0 && target_tau.array().abs().matrix().sum() == 0.0;
        }

        bool parseFromJointState(const sensor_msgs::JointState::ConstPtr& msg, int max)
        {
            auto j = std::min(int(msg->position.size()), max);
            for (int i = 0; i < j; ++i)
            {
                target_q[i] = msg->position.at(i);
            }
            j = std::min(int(msg->velocity.size()), max);
            for (int i = 0; i < j; ++i)
            {
                target_dq[i] = msg->velocity.at(i);
            }
            j = std::min(int(msg->effort.size()), max);
            for (int i = 0; i < j; ++i)
            {
                target_tau[i] = msg->effort.at(i);
            }
            return isZero();
        }

        Motor_Output& filter(std::vector<double> lowest, std::vector<double> highest)
        {
            if (lowest.size() != highest.size() || lowest.size() != target_q.size())
            {
                ROS_ERROR_THROTTLE(1, "lowest[%zu] != highest[%td] || lowest[%td] != target_q[%td]", lowest.size(), target_q.size(), lowest.size(), target_q.size());
                ;
                return *this;
            }
            for (int i = 0; i < target_q.size(); i++)
            {
                target_q[i] = std::clamp(target_q[i], lowest[i], highest[i]);
                // if (target_q[i] < lowest[i])
                // {
                //     ROS_WARN("[%d]%f < lower [%f]", i, target_q[i], lowest[i]);
                //     target_q[i] = lowest[i];
                // }
                // else if (target_q[i] > highest[i])
                // {
                //     ROS_WARN("[%d]%f > upper[%f]", i, target_q[i], highest[i]);
                //     target_q[i] = highest[i];
                // }
            }
            return *this;
        }

        // 有index 说明不是全部的电机，lowest highest的size是全部的电机数量，而index和target_q是指定电机组的size
        Motor_Output& filter(std::vector<double> lowest, std::vector<double> highest, const std::vector<int>& index)
        {
            if (lowest.size() < index.size() || lowest.size() < target_q.size() || target_q.size() != index.size())
            {
                ROS_ERROR_THROTTLE(1, "lowest[%zu] < index[%td] || lowest[%td] < target_q[%td]|| target_q.size[%td] != index.size[%td]", lowest.size(), index.size(), lowest.size(), target_q.size(), target_q.size(), index.size());
                ;
                return *this;
            }
            for (int i = 0; i < target_q.size(); i++)
            {
                target_q[i] = std::clamp(target_q[i], lowest[index[i]], highest[index[i]]);
                // if (target_q[i] < lowest[index[i]])
                // {
                //     ROS_WARN("[%d]%f < lowest [%f]", i, target_q[i], lowest[index[i]]);
                //     target_q[i] = lowest[index[i]];
                // }
                // else if (target_q[i] > highest[index[i]])
                // {
                //     ROS_WARN("[%d]%f > highest [%f]", i, target_q[i], highest[index[i]]);
                //     target_q[i] = highest[index[i]];
                // }
            }
            return *this;
        }

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "Motor_Output(\n"
                << "  target_q: " << target_q.transpose() << "\n"
                << "  target_dq: " << target_dq.transpose() << "\n"
                << "  target_tau: " << target_tau.transpose() << "\n"
                << "  target_kp: " << target_kp.transpose() << "\n"
                << "  target_kd: " << target_kd.transpose() << "\n"
                << ")";
            return oss.str();
        }

        Motor_Output& operator=(const Motor_Output& other)
        {
            if (this != &other)
            {
                target_q = other.target_q;
                target_dq = other.target_dq;
                target_tau = other.target_tau;
                target_kp = other.target_kp;
                target_kd = other.target_kd;
            }
            return *this;
        }

        bool operator==(const Motor_Output& other) const
        {
            return target_q.isApprox(other.target_q) &&
                   target_dq.isApprox(other.target_dq) &&
                   target_tau.isApprox(other.target_tau) &&
                   target_kp.isApprox(other.target_kp) &&
                   target_kd.isApprox(other.target_kd);
        }

        bool operator!=(const Motor_Output& other) const
        {
            return !(*this == other);
        }

        Motor_Output operator+(const Motor_Output& other) const
        {
            Motor_Output result;
            result.target_q = target_q + other.target_q;
            result.target_dq = target_dq + other.target_dq;
            result.target_tau = target_tau + other.target_tau;
            result.target_kp = target_kp + other.target_kp;
            result.target_kd = target_kd + other.target_kd;
            return result;
        }

        Motor_Output operator-(const Motor_Output& other) const
        {
            Motor_Output result;
            result.target_q = target_q - other.target_q;
            result.target_dq = target_dq - other.target_dq;
            result.target_tau = target_tau - other.target_tau;
            result.target_kp = target_kp - other.target_kp;
            result.target_kd = target_kd - other.target_kd;
            return result;
        }
};

struct Motor_State
{
        Eigen::VectorXd q;
        Eigen::VectorXd dq;
        Eigen::VectorXd tau;
        Motor_State() = default;
        Motor_State(int size)
        {
            resize(size);
        }
        void resize(int size)
        {
            q.resize(size);
            dq.resize(size);
            tau.resize(size);
            q.setZero();
            dq.setZero();
            tau.setZero();
        }
        Motor_Output toOut()
        {
            Motor_Output out;
            out.resize(q.size());
            out.target_q = q;
            out.target_dq = dq;
            out.target_tau = tau;
            return out;
        }
        Motor_Output toOut(std::vector<double> kp, std::vector<double> kd)
        {
            Motor_Output out;
            out.resize(q.size());
            out.target_q = q;
            out.target_dq = dq;
            out.target_tau = tau;
            // 将 std::vector<float> 转换为 Eigen::VectorXd
            if (!kp.empty())
            {
                out.target_kp = Eigen::Map<const Eigen::VectorXd>(kp.data(), kp.size()).cast<double>();
            }

            if (!kd.empty())
            {
                out.target_kd = Eigen::Map<const Eigen::VectorXd>(kd.data(), kd.size()).cast<double>();
            }

            return out;
        }
        bool isZero(double eps = 1e-6) const
        {
            return q.isZero(eps) && dq.isZero(eps) && tau.isZero(eps);
        }
        std::string toString() const
        {
            std::ostringstream oss;
            oss << "Motor_State(\n"
                << "  q: " << q.transpose() << "\n"
                << "  dq: " << dq.transpose() << "\n"
                << "  tau: " << tau.transpose() << "\n"
                << ")";
            return oss.str();
        }
};

struct Robot_State
{
        Eigen::VectorXd eu_ang;
        Eigen::VectorXd base_ang_vel;
        Eigen::Quaterniond quat;
        Eigen::VectorXd q;
        Eigen::VectorXd dq;
        Eigen::VectorXd tau;
        Robot_State() = default;
        Robot_State(int size)
        {
            resize(size);
        }
        void resize(int size)
        {
            q.resize(size);
            dq.resize(size);
            tau.resize(size);
            eu_ang.resize(3);
            base_ang_vel.resize(3);
            q.setZero();
            dq.setZero();
            tau.setZero();
            eu_ang.setZero();
            base_ang_vel.setZero();
        }
        std::string toString() const
        {
            std::ostringstream oss;
            oss << "Robot_State(\n"
                << "  q: " << q.transpose() << "\n"
                << "  dq: " << dq.transpose() << "\n"
                << "  tau: " << tau.transpose() << "\n"
                << "  angVel: " << base_ang_vel.transpose() << "\n"
                << "  eu_ang: " << eu_ang.transpose() << "\n"
                << ")";
            return oss.str();
        }
};

struct Robot_Output
{
        Eigen::VectorXd target_q;
        Eigen::VectorXd target_dq;
        Eigen::VectorXd target_tau;
        Eigen::VectorXd action;

        Robot_Output() = default;
        Robot_Output(int size)
        {
            resize(size);
        }

        bool parseFromJointState(const sensor_msgs::JointState& msg, int max)
        {
            auto j = std::min(int(msg.position.size()), max);
            for (int i = 0; i < j; ++i)
            {
                target_q[i] = msg.position.at(i);
            }
            j = std::min(int(msg.velocity.size()), max);
            for (int i = 0; i < j; ++i)
            {
                target_dq[i] = msg.velocity.at(i);
            }
            j = std::min(int(msg.effort.size()), max);
            for (int i = 0; i < j; ++i)
            {
                target_tau[i] = msg.effort.at(i);
            }
            return isZero();
        }

        bool pasrseFromRobotState(const Robot_State rbt, int max)
        {
            auto j = std::min(int(rbt.q.size()), max);
            for (int i = 0; i < j; ++i)
            {
                target_q[i] = rbt.q[i];
            }
            j = std::min(int(rbt.dq.size()), max);
            for (int i = 0; i < j; ++i)
            {
                target_dq[i] = rbt.dq[i];
            }
            j = std::min(int(rbt.tau.size()), max);
            for (int i = 0; i < j; ++i)
            {
                target_tau[i] = rbt.tau[i];
            }
            return isZero();
        }

        void resize(int size)
        {
            target_q.resize(size);
            target_dq.resize(size);
            target_tau.resize(size);
            action.resize(size);
            target_q.setZero();
            target_dq.setZero();
            target_tau.setZero();
            action.setZero();
        }

        bool isZero(double eps = 1e-6) const
        {
            return target_q.isZero(eps) && target_dq.isZero(eps) && target_tau.isZero(eps) && action.isZero(eps);
        }
        bool isAllValueZero() const
        {
            return target_q.array().abs().matrix().sum() == 0.0 && target_dq.array().abs().matrix().sum() == 0.0 && target_tau.array().abs().matrix().sum() == 0.0 && action.array().abs().matrix().sum() == 0.0;
        }

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "Robot_Output(\n"
                << "  target_q: " << target_q.transpose() << "\n"
                << "  target_dq: " << target_dq.transpose() << "\n"
                << "  target_tau: " << target_tau.transpose() << "\n"
                << "  action: " << action.transpose() << "\n"
                << ")";
            return oss.str();
        }

        Robot_Output filter(std::vector<double> lowest, std::vector<double> highest, const std::vector<int>& index)
        {
            if (lowest.size() < index.size() || lowest.size() < target_q.size() || target_q.size() != index.size())
            {
                ROS_ERROR_THROTTLE(1, "lowest[%zu] < index[%td] || lowest[%td] < target_q[%td]|| target_q.size[%td] != index.size[%td]", lowest.size(), index.size(), lowest.size(), target_q.size(), target_q.size(), index.size());
                ;
                return *this;
            }
            for (int i = 0; i < target_q.size(); i++)
            {
                target_q[i] = std::clamp(target_q[i], lowest[index[i]], highest[index[i]]);
                // if (target_q[i] < lowest[index[i]])
                // {
                //     ROS_WARN("[%d]%f < lowest [%f]", i, target_q[i], lowest[index[i]]);
                //     target_q[i] = lowest[index[i]];
                // }
                // else if (target_q[i] > highest[index[i]])
                // {
                //     ROS_WARN("[%d]%f > highest [%f]", i, target_q[i], highest[index[i]]);
                //     target_q[i] = highest[index[i]];
                // }
            }
            return *this;
        }
};

struct Transform
{
        Eigen::Matrix<double, 2, 2> knee_J;
        Eigen::Matrix<double, 2, 2> ankle_J_left;
        Eigen::Matrix<double, 2, 2> ankle_J_right;
        Eigen::Matrix<double, 2, 2> ankle_J_left_inv;
        Eigen::Matrix<double, 2, 2> ankle_J_right_inv;
        Eigen::Matrix<double, 2, 2> ankle_J_left_T_inv;
        Eigen::Matrix<double, 2, 2> ankle_J_right_T_inv;
};

struct Observations
{
        Eigen::VectorXd observations;
        Eigen::MatrixXd input;
        std::deque<Eigen::VectorXd> hist_obs;
};

struct DynamicOffset
{
        double hip;
        double knee;
        double ankle;
};

struct RL_Subscriber
{
        ros::Subscriber pd2rl_sub_;
        ros::Subscriber joy_model_sub_;
        ros::Subscriber joy_control_sub_;
        ros::Subscriber user_control_sub_;
        ros::Subscriber odom_sub_;
        ros::Subscriber usr_ctrl_sub_;
};

struct RL_Publisher
{
        ros::Publisher rl2pd_pub_;
        ros::Publisher ori_pub_;
        ros::Publisher ctrl_usr_pub_;
        ros::Publisher final_command_pub_;
};

struct RL_Msgs
{
        sensor_msgs::Imu yesenseIMU;
        sim2real_msg::Orient ori_msg;
        sim2real_msg::PD2RL pd2rl_msg;
        sim2real_msg::RL2PD rl2pd_msg;
};

struct PD_Subscriber
{
        ros::Subscriber rl2pd_sub_;
        ros::Subscriber joy_sub_;
        ros::Subscriber odom_sub_;
        ros::Subscriber ori_sub_;
};

struct PD_Publisher
{
        ros::Publisher pd2rl_pub_;
        ros::Publisher rviz_pos_pub_;
        ros::Publisher mtr_state_pub_;
        ros::Publisher rbt_state_pub_;
        ros::Publisher mtr_target_pub_;
        ros::Publisher rbt_target_pub_;
};

struct PD_Msgs
{
        sensor_msgs::Imu yesenseIMU;
        sim2real_msg::Orient ori_msg;
        sensor_msgs::JointState rviz_pos_msg;
        sim2real_msg::PD2RL pd2rl_msg;
        sim2real_msg::RL2PD rl2pd_msg;
        sim2real_msg::MtrState mtr_state_msg;
        sim2real_msg::RbtState rbt_state_msg;
        sim2real_msg::MtrTarget mtr_target_msg;
        sim2real_msg::RbtTarget rbt_target_msg;
};

#endif
