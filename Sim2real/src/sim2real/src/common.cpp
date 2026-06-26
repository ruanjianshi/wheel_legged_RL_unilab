#include "jacobian.h"
#include "kinematics.h"
#include "ros/console.h"
#include <algorithm>
#include "common.h"
#include <boost/fusion/functional/invocation/invoke.hpp>
#include <cstddef>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/sysinfo.h>

namespace hightorque
{
    namespace common
    {
        double linearTransitionPosition(double startValue, double targetValue, double startTime, double currentTime, double duration)
        {
            if (duration <= 0.0)
            {
                return targetValue;
            }
            double t = (currentTime - startTime) / duration;
            t = std::clamp(t, 0.0, 1.0);
            return startValue + (targetValue - startValue) * t;
        }

        double linearTransitionVelocity(double startValue, double targetValue, double startTime, double currentTime, double duration)
        {
            if (duration <= 0.0)
            {
                return targetValue;
            }
            double t = (currentTime - startTime) / duration;
            if (t >= 1.0 || t < 0.0)
            {
                return 0.0;
            }
            return (targetValue - startValue) / duration;
        }
        double easeInTransitionPosition(double startValue, double targetValue, double startTime, double currentTime, double duration)
        {
            if (duration <= 0.0)
            {
                return targetValue;
            }
            double t = (currentTime - startTime) / duration;
            t = std::clamp(t, 0.0, 1.0);
            double factor = t * t;
            return startValue + (targetValue - startValue) * factor;
        }
        double easeInTransitionVelocity(double startValue, double targetValue, double startTime, double currentTime, double duration)
        {
            if (duration <= 0.0)
            {
                return targetValue;
            }
            double t = (currentTime - startTime) / duration;
            if (t >= 1.0 || t < 0.0)
            {
                return 0.0;
            }
            double factor = 2 * t;
            return (targetValue - startValue) * factor / duration;
        }
        double easeOutTransitionPosition(double startValue, double targetValue, double startTime, double currentTime, double duration)
        {
            if (duration <= 0.0)
            {
                return targetValue;
            }
            double t = (currentTime - startTime) / duration;
            t = std::clamp(t, 0.0, 1.0);
            double factor = 1.0 - (1.0 - t) * (1.0 - t);
            return startValue + (targetValue - startValue) * factor;
        }
        double easeOutTransitionVelocity(double startValue, double targetValue, double startTime, double currentTime, double duration)
        {
            if (duration <= 0.0)
            {
                return targetValue;
            }
            double t = (currentTime - startTime) / duration;
            if (t >= 1.0 || t < 0.0)
            {
                return 0.0;
            }
            double factor = 2.0 * (1.0 - t);
            return (targetValue - startValue) * factor / duration;
        }

        double smoothStepTransitionPosition(double startValue, double targetValue, double startTime, double currentTime, double duration)
        {
            if (duration <= 0.0)
            {
                return targetValue;
            }
            double t = (currentTime - startTime) / duration;
            double factor = t * t * (3.0 - 2.0 * t);
            return startValue + (targetValue - startValue) * factor;
        }
        double smoothStepTransitionVelocity(double startValue, double targetValue, double startTime, double currentTime, double duration)
        {
            if (duration <= 0.0)
            {
                return targetValue;
            }
            double t = (currentTime - startTime) / duration;
            if (t >= 1.0 || t < 0.0)
            {
                return 0.0;
            }
            double factor = 6 * t * (1.0 - t);
            return (targetValue - startValue) * factor / duration;
        }

        Eigen::Vector3d quatToEulerZYX(const double x, const double y, const double z, const double w, bool normalizeAngle)
        {
            double t0 = +2.0 * (w * x + y * z);
            double t1 = +1.0 - 2.0 * (x * x + y * y);
            double roll = std::atan2(t0, t1);

            double t2 = +2.0 * (w * y - z * x);
            t2 = std::clamp(t2, -1.0, 1.0);
            double pitch = std::asin(t2);

            double t3 = +2.0 * (w * z + x * y);
            double t4 = +1.0 - 2.0 * (y * y + z * z);
            double yaw = std::atan2(t3, t4);

            Eigen::Vector3d euler;
            euler << roll, pitch, yaw;

            if (normalizeAngle)
            {
                euler = euler.unaryExpr([](double angle) {
                    return angle > M_PI ? angle - 2 * M_PI : angle;
                });
            }

            return euler;
        }

        Eigen::Vector3d quatToEulerZYX(const Eigen::Quaterniond& q, bool normalizeAngle)
        {
            double x = q.x(), y = q.y(), z = q.z(), w = q.w();
            return quatToEulerZYX(x, y, z, w, normalizeAngle);
        }

        Eigen::Quaterniond eulerZYXToQuat(const Eigen::Vector3d& euler)
        {
            double roll = euler[0];
            double pitch = euler[1];
            double yaw = euler[2];

            Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

            return yawAngle * pitchAngle * rollAngle;
        }

        Eigen::Quaterniond eulerZYXToQuat(double roll, double pitch, double yaw)
        {
            return eulerZYXToQuat(Eigen::Vector3d(roll, pitch, yaw));
        }

        Eigen::Vector3d rotateVectorByQuatVec4(const Eigen::Vector4d& q, const Eigen::Vector3d& v)
        {
            double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
            Eigen::Vector3d qVec(qx, qy, qz);

            Eigen::Vector3d term1 = v * (2.0 * qw * qw - 1.0);
            Eigen::Vector3d term2 = qVec.cross(v) * qw * 2.0;
            Eigen::Vector3d term3 = qVec * qVec.dot(v) * 2.0;

            return term1 - term2 + term3;
        }

        Eigen::Vector3d rotateVectorByQuat(const Eigen::Quaterniond& q, const Eigen::Vector3d& v)
        {
            return q.conjugate() * v;
        }

        void transformRobotOutputToMotorOutput(const float& r, float& m, const double& offset, const int direction)
        {
            double adjustedPos = r + offset;
            m = adjustedPos * direction;
        }
        void transformRobotOutputToMotorOutput(Robot_Output& r, Motor_Output& m, const std::vector<double>& offset, const std::vector<int> direction, int special)
        {
            if (m.target_q.size() < r.target_q.size())
            {
                m.resize(r.target_q.size());
            }

            for (size_t i = 0; i < r.target_q.size(); ++i)
            {
                double adjustedPos = r.target_q[i] + offset[i];
                r.target_q[i] = adjustedPos;
                m.target_q[i] = adjustedPos * direction[i];
                m.target_dq[i] = r.target_dq[i];
                m.target_tau[i] = r.target_tau[i];
            }
            // hi TODO:电机组4 5 10 11 得是脚踝才行，在lowlevel中如果控制手部电机大概率报错，需要修改
            if (special == 1)
            {
                ankleInverseKinematics(r.target_q[4], r.target_q[5], m.target_q[4], m.target_q[5]);
                ankleInverseKinematics(-r.target_q[10], r.target_q[11], m.target_q[10], m.target_q[11]);
                m.target_q[4] *= direction[4];
                m.target_q[5] *= direction[5];
                m.target_q[10] *= direction[10];
                m.target_q[11] *= direction[11];
            }
        }

        void transformRobotOutputToMotorOutput(Robot_Output& r, Motor_Output& m, const std::vector<double>& offset, const std::vector<int> direction, const std::vector<int>& index, int special)
        {
            auto jointsNum = index.size(); // 使用关节数控制长度
            if (m.target_q.size() < jointsNum)
            {
                m.resize(jointsNum);
            }

            int j = std::min(long(jointsNum), m.target_q.size());
            for (auto i = 0; i < j; i++)
            {
                double adjustedPos = r.target_q[i] + offset[index[i]];
                r.target_q[i] = adjustedPos;
                m.target_q[i] = adjustedPos * direction[index[i]];
                m.target_dq[i] = r.target_dq[i];
                m.target_tau[i] = r.target_tau[i];
            }
            // hi TODO:电机组4 5 10 11 得是脚踝才行，在lowlevel中如果控制手部电机大概率报错，需要修改
            if (special == 1)
            {
                ankleInverseKinematics(r.target_q[4], r.target_q[5], m.target_q[4], m.target_q[5]);
                ankleInverseKinematics(-r.target_q[10], r.target_q[11], m.target_q[10], m.target_q[11]);
                m.target_q[4] *= direction[index[4]];
                m.target_q[5] *= direction[index[4]];
                m.target_q[10] *= direction[index[10]];
                m.target_q[11] *= direction[index[11]];
            }
        }

        void transformMotorStateToRobotState(Motor_State& m, Robot_State& r, const std::vector<double>& offset, const std::vector<int> direction, int special)
        {
            if (r.q.size() < m.q.size())
            {
                r.resize(m.q.size());
            }

            for (int i = 0; i < m.q.size(); i++)
            {
                r.q[i] = m.q[i] * direction[i];
                r.dq[i] = m.dq[i] * direction[i];
                r.q[i] -= offset[i];
                r.tau[i] = m.tau[i];
            }

            // hi TODO:电机组4 5 10 11 得是脚踝才行，在lowlevel中如果控制手部电机大概率报错，需要修改
            if (special == 1)
            {
                // position
                ankleKinematics(m.q[4] * direction[4], m.q[5] * direction[5],
                                r.q[4], r.q[5]);
                ankleKinematics(m.q[10] * direction[10], m.q[11] * direction[11],
                                r.q[10], r.q[11]);
                r.q[10] *= -1;
                r.q[4] -= offset[4];
                r.q[5] -= offset[5];
                r.q[10] -= offset[10];
                r.q[11] -= offset[11];

                // velocity
                Transform trans;
                trans.ankle_J_left = ankle_J_calc(r.q[4], r.q[5]);
                trans.ankle_J_right = ankle_J_calc(r.q[10], r.q[11]);
                trans.ankle_J_left_inv = ensureFullRank(trans.ankle_J_left).inverse();
                trans.ankle_J_right_inv = ensureFullRank(trans.ankle_J_right).inverse();
                trans.ankle_J_left_T_inv = ensureFullRank(trans.ankle_J_left.transpose()).inverse();
                trans.ankle_J_right_T_inv = ensureFullRank(trans.ankle_J_right.transpose()).inverse();
                if ((Eigen::FullPivLU<Eigen::Matrix<double, 2, 2>>(trans.ankle_J_left_inv).rank() < 2) ||
                    (Eigen::FullPivLU<Eigen::Matrix<double, 2, 2>>(trans.ankle_J_right_inv).rank() < 2))
                {
                    ROS_ERROR("The initial Jacobian matrix does not have full rank.\n"
                              "ankle_J_left:\n[%.3f, %.3f]\n[%.3f, %.3f]\n"
                              "ankle_J_right:\n[%.3f, %.3f]\n[%.3f, %.3f]\n"
                              "ankle_J_left_inv:\n[%.3f, %.3f]\n[%.3f, %.3f]\n"
                              "ankle_J_right_inv:\n[%.3f, %.3f]\n[%.3f, %.3f]\n",
                              trans.ankle_J_left(0, 0), trans.ankle_J_left(0, 1),
                              trans.ankle_J_left(1, 0), trans.ankle_J_left(1, 1),
                              trans.ankle_J_right(0, 0), trans.ankle_J_right(0, 1),
                              trans.ankle_J_right(1, 0), trans.ankle_J_right(1, 1),
                              trans.ankle_J_left_inv(0, 0), trans.ankle_J_left_inv(0, 1),
                              trans.ankle_J_left_inv(1, 0), trans.ankle_J_left_inv(1, 1),
                              trans.ankle_J_right_inv(0, 0), trans.ankle_J_right_inv(0, 1),
                              trans.ankle_J_right_inv(1, 0), trans.ankle_J_right_inv(1, 1));
                    ROS_ERROR("\n"
                              "rbt_state.q[4]:%.3f\nrbt_state.q[5]]:%.3f\n"
                              "rbt_state.q[10]:%.3f\nrbt_state.q[11]]:%.3f\n"
                              "mtr_state.q[4]:%.3f\nmrt_state.q[5]]:%.3f\n"
                              "mtr_state.q[10]:%.3f\nmrt_state.q[11]]:%.3f\n",
                              r.q[4] * 180. / 3.14, r.q[5] * 180. / 3.14, r.q[10] * 180. / 3.14, r.q[11] * 180. / 3.14,
                              m.q[4] * 180. / 3.14, m.q[5] * 180. / 3.14, m.q[10] * 180. / 3.14, m.q[11] * 180. / 3.14);
                }
                Eigen::Vector2d mtr_ankle_dq_left(m.dq[4] * direction[4],
                                                  m.dq[5] * direction[5]);
                Eigen::Vector2d mtr_ankle_dq_right(m.dq[10] * direction[10],
                                                   m.dq[11] * direction[11]);
                Eigen::Vector2d rbt_ankle_dq_left = trans.ankle_J_left_inv * mtr_ankle_dq_left;
                Eigen::Vector2d rbt_ankle_dq_right = trans.ankle_J_right_inv * mtr_ankle_dq_right;
                r.dq[4] = rbt_ankle_dq_left[0];
                r.dq[5] = rbt_ankle_dq_left[1];
                r.dq[10] = -rbt_ankle_dq_right[0];
                r.dq[11] = rbt_ankle_dq_right[1];
            }
        }

        void transformMotorStateToRobotState(Motor_State& m, Robot_State& r, const std::vector<double>& offset, const std::vector<int> direction, const std::vector<int>& index, int special)
        {
            if (r.q.size() < m.q.size())
            {
                r.resize(m.q.size());
            }

            int j = std::min(long(index.size()), m.q.size());
            for (auto i = 0; i < j; i++)
            {
                r.q[i] = m.q[i] * direction[index[i]];
                r.dq[i] = m.dq[i] * direction[index[i]];
                r.q[i] -= offset[index[i]];
                r.tau[i] = m.tau[i];
            }
            // hi TODO:电机组4 5 10 11 得是脚踝才行，在lowlevel中如果控制手部电机大概率报错，需要修改
            if (special == 1)
            {
                // position
                ankleKinematics(m.q[4] * direction[4], m.q[5] * direction[5],
                                r.q[4], r.q[5]);
                ankleKinematics(m.q[10] * direction[10], m.q[11] * direction[11],
                                r.q[10], r.q[11]);
                r.q[10] *= -1;
                r.q[4] -= offset[index[4]];
                r.q[5] -= offset[index[5]];
                r.q[10] -= offset[index[10]];
                r.q[11] -= offset[index[11]];

                // velocity
                Transform trans;
                trans.ankle_J_left = ankle_J_calc(r.q[4], r.q[5]);
                trans.ankle_J_right = ankle_J_calc(r.q[10], r.q[11]);
                trans.ankle_J_left_inv = ensureFullRank(trans.ankle_J_left).inverse();
                trans.ankle_J_right_inv = ensureFullRank(trans.ankle_J_right).inverse();
                trans.ankle_J_left_T_inv = ensureFullRank(trans.ankle_J_left.transpose()).inverse();
                trans.ankle_J_right_T_inv = ensureFullRank(trans.ankle_J_right.transpose()).inverse();
                if ((Eigen::FullPivLU<Eigen::Matrix<double, 2, 2>>(trans.ankle_J_left_inv).rank() < 2) ||
                    (Eigen::FullPivLU<Eigen::Matrix<double, 2, 2>>(trans.ankle_J_right_inv).rank() < 2))
                {
                    ROS_ERROR("The initial Jacobian matrix does not have full rank.\n"
                              "ankle_J_left:\n[%.3f, %.3f]\n[%.3f, %.3f]\n"
                              "ankle_J_right:\n[%.3f, %.3f]\n[%.3f, %.3f]\n"
                              "ankle_J_left_inv:\n[%.3f, %.3f]\n[%.3f, %.3f]\n"
                              "ankle_J_right_inv:\n[%.3f, %.3f]\n[%.3f, %.3f]\n",
                              trans.ankle_J_left(0, 0), trans.ankle_J_left(0, 1),
                              trans.ankle_J_left(1, 0), trans.ankle_J_left(1, 1),
                              trans.ankle_J_right(0, 0), trans.ankle_J_right(0, 1),
                              trans.ankle_J_right(1, 0), trans.ankle_J_right(1, 1),
                              trans.ankle_J_left_inv(0, 0), trans.ankle_J_left_inv(0, 1),
                              trans.ankle_J_left_inv(1, 0), trans.ankle_J_left_inv(1, 1),
                              trans.ankle_J_right_inv(0, 0), trans.ankle_J_right_inv(0, 1),
                              trans.ankle_J_right_inv(1, 0), trans.ankle_J_right_inv(1, 1));
                    ROS_ERROR("\n"
                              "rbt_state.q[4]:%.3f\nrbt_state.q[5]]:%.3f\n"
                              "rbt_state.q[10]:%.3f\nrbt_state.q[11]]:%.3f\n"
                              "mtr_state.q[4]:%.3f\nmrt_state.q[5]]:%.3f\n"
                              "mtr_state.q[10]:%.3f\nmrt_state.q[11]]:%.3f\n",
                              r.q[4] * 180. / 3.14, r.q[5] * 180. / 3.14, r.q[10] * 180. / 3.14, r.q[11] * 180. / 3.14,
                              m.q[4] * 180. / 3.14, m.q[5] * 180. / 3.14, m.q[10] * 180. / 3.14, m.q[11] * 180. / 3.14);
                }
                Eigen::Vector2d mtr_ankle_dq_left(
                    m.dq[4] * direction[index[4]],
                    m.dq[5] * direction[index[5]]);
                Eigen::Vector2d mtr_ankle_dq_right(
                    m.dq[10] * direction[index[10]],
                    m.dq[11] * direction[index[11]]);
                Eigen::Vector2d rbt_ankle_dq_left = trans.ankle_J_left_inv * mtr_ankle_dq_left;
                Eigen::Vector2d rbt_ankle_dq_right = trans.ankle_J_right_inv * mtr_ankle_dq_right;
                r.dq[4] = rbt_ankle_dq_left[0];
                r.dq[5] = rbt_ankle_dq_left[1];
                r.dq[10] = -rbt_ankle_dq_right[0];
                r.dq[11] = rbt_ankle_dq_right[1];
            }
        }

        void transformPolicyRobotOutputToActualRobotOutput(Robot_Output& p, Robot_Output& a, std::vector<int> map)
        {
            std::stringstream ss;
            for (auto i = 0; i < p.target_q.size(); i++)
            {
                if (map[i] >= 0 && map[i] < a.target_q.size())
                {
                    ss << (map[i]) << " to " << i << "  ";
                    a.target_q[map[i]] = p.target_q[i];
                    a.target_dq[map[i]] = p.target_dq[i];
                    a.target_tau[map[i]] = p.target_tau[i];
                    a.action[map[i]] = p.action[i];
                }
            }
            ss << "\n";
            // ROS_INFO_THROTTLE(10, "p2a %s", ss.str());
        }
        void transformActualRobotStateToPolicyRobotState(Robot_State& a, Robot_State& p, std::vector<int> map)
        {
            p.quat = a.quat;
            p.eu_ang = a.eu_ang;
            p.base_ang_vel = a.base_ang_vel;
            std::stringstream ss;
            for (auto i = 0; i < a.q.size(); i++)
            {
                if (map[i] >= 0 && map[i] < p.q.size())
                {
                    ss << (map[i]) << " to " << i << "  ";
                    p.q[map[i]] = a.q[i];
                    p.dq[map[i]] = a.dq[i];
                    p.tau[map[i]] = a.tau[i];
                }
            }
            ss << "\n";
            // ROS_INFO_THROTTLE(10, "a2p %s", ss.str());
        }
        void transformActualRobotOutputToPolicyRobotOutput(Robot_Output& a, Robot_Output& p, std::vector<int> map)
        {
            for (auto i = 0; i < a.target_q.size(); i++)
            {
                if (map[i] >= 0 && map[i] < p.target_q.size())
                {
                    p.target_q[map[i]] = a.target_q[i];
                    p.target_dq[map[i]] = a.target_dq[i];
                    p.target_tau[map[i]] = a.target_tau[i];
                    p.action[map[i]] = a.action[i];
                }
            }
        }
        void ankleKinematics(double motorPitch, double motorRoll, double& robotPitch, double& robotRoll)
        {
            double d1 = 0.038015;

            double d2 = 0.025;
            double L1 = 0.025;

            double h1 = 0.122;
            double a1 = 0.122;

            double h2 = 0.07;
            double a2 = 0.07;

            a2 = h2;
            a1 = h1;
            d2 = L1;

            double D_L = 2 * d1 * d2;
            double E_L = 2 * L1 * d1 * sin(motorRoll) - 2 * d1 * h2;
            double F_L = 2 * d2 * L1 * sin(motorRoll) - 2 * d2 * h2;
            double G_L = 2 * d1 * d1;
            double H_L = 2 * d2 * L1 * cos(motorRoll);
            double I_L = L1 * L1 + h2 * h2 + 2 * d1 * d1 + d2 * d2 - a2 * a2 - 2 * h2 * L1 * sin(motorRoll);
            double D_R = D_L;
            double E_R = 2 * L1 * d1 * sin(motorPitch) + 2 * d1 * h1;
            double F_R = 2 * d2 * L1 * sin(motorPitch) + 2 * d2 * h1;
            double G_R = G_L;
            double H_R = 2 * d2 * L1 * cos(motorPitch);
            double I_R = L1 * L1 + h1 * h1 + 2 * d1 * d1 + d2 * d2 - a1 * a1 + 2 * h1 * L1 * sin(motorPitch);

            robotPitch = 0;
            robotRoll = 0;

            int max_iterations = 100;
            double tolerance = 1e-5;

            for (int i = 0; i < max_iterations; ++i)
            {
                double f1 = -D_L * sin(robotPitch) * sin(robotRoll) - E_L * sin(robotRoll) + F_L * sin(robotPitch) - G_L * cos(robotRoll) - H_L * cos(robotPitch) + I_L;
                double f2 = D_R * sin(robotPitch) * sin(robotRoll) - E_R * sin(robotRoll) - F_R * sin(robotPitch) - G_R * cos(robotRoll) - H_R * cos(robotPitch) + I_R;

                if (fabs(f1) < tolerance && fabs(f2) < tolerance)
                {
                    break;
                }

                double J11 = -D_L * cos(robotPitch) * sin(robotRoll) + F_L * cos(robotPitch) + H_L * sin(robotPitch);
                double J12 = -D_L * sin(robotPitch) * cos(robotRoll) - E_L * cos(robotRoll) + G_L * sin(robotRoll);
                double J21 = D_R * cos(robotPitch) * sin(robotRoll) - F_R * cos(robotPitch) - H_R * sin(robotPitch);
                double J22 = D_R * sin(robotPitch) * cos(robotRoll) - E_R * cos(robotRoll) + G_R * sin(robotRoll);

                double det = J11 * J22 - J12 * J21;
                if (fabs(det) < 1e-10)
                {
                    robotPitch = 0.0;
                    robotRoll = 0.0;
                    break;
                }

                double J11_inv = J22 / det;
                double J12_inv = -J12 / det;
                double J21_inv = -J21 / det;
                double J22_inv = J11 / det;

                double delta_theta_p = -(J11_inv * f1 + J12_inv * f2);
                double delta_theta_r = -(J21_inv * f1 + J22_inv * f2);

                robotPitch = robotPitch + delta_theta_p;
                robotRoll = robotRoll + delta_theta_r;
            }
        }

        void ankleInverseKinematics(double robotPitch, double robotRoll, double& motorPitch, double& motorRoll)
        {
            double d1 = 0.038015;

            double d2 = 0.025;
            double L1 = 0.025;

            double h1 = 0.122;
            double a1 = 0.122;

            double h2 = 0.07;
            double a2 = 0.07;

            a2 = h2;
            a1 = h1;
            d2 = L1;

            double A_L = 2 * d2 * L1 * sin(robotPitch) - 2 * L1 * d1 * sin(robotRoll) - 2 * h2 * L1;
            double A_R = -2 * d2 * L1 * sin(robotPitch) - 2 * L1 * d1 * sin(robotRoll) + 2 * h1 * L1;
            double B_L = 2 * d2 * L1 * cos(robotPitch);
            double B_R = 2 * d2 * L1 * cos(robotPitch);
            double C_L = 2 * d1 * d1 + h2 * h2 - a2 * a2 + L1 * L1 + d2 * d2 - 2 * d1 * d1 * cos(robotRoll) + 2 * d1 * h2 * sin(robotRoll) - 2 * d2 * h2 * sin(robotPitch) - 2 * d1 * d2 * sin(robotPitch) * sin(robotRoll);
            double C_R = 2 * d1 * d1 + h1 * h1 - a1 * a1 + L1 * L1 + d2 * d2 - 2 * d1 * d1 * cos(robotRoll) - 2 * d1 * h1 * sin(robotRoll) - 2 * d2 * h1 * sin(robotPitch) + 2 * d1 * d2 * sin(robotPitch) * sin(robotRoll);

            double Len_L = A_L * A_L - C_L * C_L + B_L * B_L;
            double Len_R = A_R * A_R - C_R * C_R + B_R * B_R;

            if (Len_L > 0 && Len_R > 0)
            {
                motorRoll = 2 * atan((-A_L - sqrt(Len_L)) / (B_L + C_L));
                motorPitch = 2 * atan((-A_R + sqrt(Len_R)) / (B_R + C_R));
            }
            else
            {
                motorPitch = 0.0;
                motorRoll = 0.0;
            }
        }

        Motor_State toState(const Motor_Output& out)
        {
            Motor_State state;
            state.resize(out.target_q.size());
            state.q = out.target_q;
            state.dq = out.target_dq;
            state.tau = out.target_tau;
            return state;
        }

        Robot_State toState(const Robot_Output& out)
        {
            Robot_State state;
            state.resize(out.target_q.size());
            state.q = out.target_q;
            state.dq = out.target_dq;
            state.tau = out.target_tau;
            return state;
        }

        Motor_Output toOut(const Motor_State& state)
        {
            Motor_Output out;
            out.resize(state.q.size());
            out.target_q = state.q;
            out.target_dq = state.dq;
            out.target_tau = state.tau;
            return out;
        }

        Robot_Output toOut(const Robot_State& state)
        {
            Robot_Output out;
            out.resize(state.q.size());
            out.target_q = state.q;
            out.target_dq = state.dq;
            out.target_tau = state.tau;
            return out;
        }

        void motorOutputSetKpKd(Motor_Output& m, const std::vector<float> kp, const std::vector<float> kd)
        {
            for (auto i = 0; i < m.target_q.size(); i++)
            {
                m.target_kp[i] = kp[i];
                m.target_kd[i] = kd[i];
            }
        }
        void motorOutputSetKpKd(Motor_Output& m, const std::vector<double> kp, const std::vector<double> kd)
        {
            for (auto i = 0; i < m.target_q.size(); i++)
            {
                m.target_kp[i] = kp[i];
                m.target_kd[i] = kd[i];
            }
        }

        void motorOutputSetTorque(Motor_Output& m, const std::vector<float> torque)
        {
            for (auto i = 0; i < m.target_q.size(); i++)
            {
                m.target_tau[i] = torque[i];
            }
        }

        void motorOutputSetTorque(Motor_Output& m, const std::vector<double> torque)
        {
            for (auto i = 0; i < m.target_q.size(); i++)
            {
                m.target_tau[i] = torque[i];
            }
        }

        void motorOutputSetKpKd(Motor_Output& m, const std::vector<float> kp, const std::vector<float> kd, const std::vector<int> index)
        {
            int j = std::min(long(index.size()), m.target_q.size());
            for (auto i = 0; i < j; i++)
            {
                m.target_kp[i] = kp[index[i]];
                m.target_kd[i] = kd[index[i]];
            }
        }
        void motorOutputSetKpKd(Motor_Output& m, const std::vector<double> kp, const std::vector<double> kd, const std::vector<int> index)
        {
            int j = std::min(long(index.size()), m.target_q.size());
            for (auto i = 0; i < j; i++)
            {
                m.target_kp[i] = kp[index[i]];
                m.target_kd[i] = kd[index[i]];
            }
        }

        bool isRobotStateClose(const Robot_State& s1, const Robot_State& s2, double diff)
        {
            // 检查向量维度是否相同
            if (s1.q.size() != s2.q.size() ||
                s1.dq.size() != s2.dq.size() ||
                s1.tau.size() != s2.tau.size())
            {
                return false;
            }

            // 使用 Eigen 的向量化操作检查每个向量
            if ((s1.q - s2.q).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((s1.dq - s2.dq).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((s1.tau - s2.tau).cwiseAbs().maxCoeff() > diff)
                return false;

            return true;
        }

        std::vector<Motor_Output> toMotorOutputVector(const std::vector<std::vector<double>> pointsVec)
        {
            std::vector<Motor_Output> motorOutputs;
            motorOutputs.reserve(pointsVec.size());

            for (const auto& point : pointsVec)
            {
                Motor_Output output;
                output.resize(point.size());
                output.target_q = Eigen::VectorXd::Map(point.data(), point.size());
                motorOutputs.push_back(output);
            }

            return motorOutputs;
        }

        bool isMotorStateClose(const Motor_State& s1, const Motor_State& s2, double diff)
        {
            // 检查向量维度是否相同
            if (s1.q.size() != s2.q.size() ||
                s1.dq.size() != s2.dq.size() ||
                s1.tau.size() != s2.tau.size())
            {
                return false;
            }

            // 使用 Eigen 的向量化操作检查每个向量
            if ((s1.q - s2.q).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((s1.dq - s2.dq).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((s1.tau - s2.tau).cwiseAbs().maxCoeff() > diff)
                return false;

            return true;
        }

        bool isRobotOutputClose(const Robot_Output& o1, const Robot_Output& o2, double diff)
        {
            // 检查向量维度是否相同
            if (o1.target_q.size() != o2.target_q.size() ||
                o1.target_dq.size() != o2.target_dq.size() ||
                o1.target_tau.size() != o2.target_tau.size())
            {
                return false;
            }

            // 使用 Eigen 的向量化操作检查每个向量
            if ((o1.target_q - o2.target_q).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((o1.target_dq - o2.target_dq).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((o1.target_tau - o2.target_tau).cwiseAbs().maxCoeff() > diff)
                return false;

            return true;
        }

        bool isMotorOutputClose(const Motor_Output& o1, const Motor_Output& o2, double diff)
        {
            // 检查向量维度是否相同
            if (o1.target_q.size() != o2.target_q.size() ||
                o1.target_dq.size() != o2.target_dq.size() ||
                o1.target_tau.size() != o2.target_tau.size())
            {
                return false;
            }

            // 使用 Eigen 的向量化操作检查每个向量
            if ((o1.target_q - o2.target_q).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((o1.target_dq - o2.target_dq).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((o1.target_tau - o2.target_tau).cwiseAbs().maxCoeff() > diff)
                return false;

            return true;
        }

        template <typename T, typename U>
        bool isStateCloseToOutput(const T& s, const U& o, double diff)
        {
            // 检查向量维度是否相同
            if (o.target_q.size() != s.q.size() ||
                o.target_dq.size() != s.dq.size() ||
                o.target_tau.size() != s.tau.size())
            {
                return false;
            }

            // 使用 Eigen 的向量化操作检查每个向量
            if ((o.target_q - s.q).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((o.target_dq - s.dq).cwiseAbs().maxCoeff() > diff)
                return false;
            if ((o.target_tau - s.tau).cwiseAbs().maxCoeff() > diff)
                return false;

            return true;
        }

        std::string toString(std::vector<std::vector<double>> data)
        {
            std::stringstream ss;
            for (const auto& row : data)
            {
                for (size_t i = 0; i < row.size(); i++)
                {
                    ss << row[i];
                    if (i < row.size() - 1)
                    {
                        ss << ",";
                    }
                }
                ss << "\n";
            }
            return ss.str();
        }

        std::string toString(const std::vector<double>& data)
        {
            std::stringstream ss;
            for (size_t i = 0; i < data.size(); i++)
            {
                ss << data[i];
                if (i < data.size() - 1)
                {
                    ss << ",";
                }
            }
            return ss.str();
        }

        std::vector<std::vector<double>> rotateMatrix(const std::vector<std::vector<double>>& matrix)
        {
            if (matrix.empty())
            {
                return {};
            }

            size_t rows = matrix.size();
            size_t cols = matrix[0].size();

            std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));

            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    transposed[j][i] = matrix[i][j];
                }
            }

            return transposed;
        }

        unsigned char* loadData(FILE* file, size_t offset, size_t size)
        {
            if (file == nullptr || size == 0)
            {
                return nullptr;
            }

            unsigned char* data = new (std::nothrow) unsigned char[size];
            if (data == nullptr)
            {
                std::cerr << "Memory allocation failed (size: " << size << " bytes)" << std::endl;
                return nullptr;
            }

            if (fseek(file, offset, SEEK_SET) != 0)
            {
                delete[] data;
                std::cerr << "Failed to seek to offset " << offset
                          << " (errno: " << errno << ", " << std::strerror(errno) << ")" << std::endl;
                return nullptr;
            }

            size_t bytesRead = fread(data, 1, size, file);
            if (bytesRead != size)
            {
                delete[] data;
                std::cerr << "Read error: expected " << size
                          << " bytes, read " << bytesRead
                          << " bytes (errno: " << errno << ", " << std::strerror(errno) << ")" << std::endl;
                return nullptr;
            }

            return data;
        }

        unsigned char* readFileData(const char* filename, int* sizeOut)
        {
            if (filename == nullptr || sizeOut == nullptr)
            {
                return nullptr;
            }

            FILE* file = fopen(filename, "rb");
            if (file == nullptr)
            {
                std::cerr << "Failed to open file '" << filename
                          << "' (errno: " << errno << ", " << std::strerror(errno) << ")" << std::endl;
                return nullptr;
            }

            // Determine file size
            if (fseek(file, 0, SEEK_END) != 0)
            {
                std::cerr << "Failed to determine file size (errno: "
                          << errno << ", " << std::strerror(errno) << ")" << std::endl;
                fclose(file);
                return nullptr;
            }

            long size = ftell(file);
            if (size < 0)
            {
                std::cerr << "Failed to get file size (errno: "
                          << errno << ", " << std::strerror(errno) << ")" << std::endl;
                fclose(file);
                return nullptr;
            }

            if (fseek(file, 0, SEEK_SET) != 0)
            {
                std::cerr << "Failed to rewind file (errno: "
                          << errno << ", " << std::strerror(errno) << ")" << std::endl;
                fclose(file);
                return nullptr;
            }

            unsigned char* data = loadData(file, 0, static_cast<size_t>(size));
            fclose(file);

            *sizeOut = static_cast<int>(size);
            return data;
        }

        std::vector<std::vector<double>> loadVectorWithBoost(const std::string& filename)
        {
            std::ifstream inFile(filename);
            if (!inFile.is_open())
            {
                std::cerr << "Failed to rewind file (errno: " << filename.c_str() << ")" << std::endl;
                return {};
            }
            boost::archive::text_iarchive ia(inFile);
            std::vector<std::vector<double>> data;
            ia >> data;
            return data;
        }

        bool saveMatrixWithMetadata(const std::filesystem::path& filePath, const std::string& metadata,
                                    const std::vector<std::vector<double>>& data)
        {
            try
            {
                // 确保父目录存在
                if (auto parent = filePath.parent_path(); !parent.empty())
                {
                    std::filesystem::create_directories(parent);
                }

                std::ofstream ofs(filePath, std::ios::out | std::ios::trunc);
                if (!ofs.is_open())
                {
                    std::cerr << "Failed to  open file (" << filePath.filename() << ")" << std::endl;
                    return false;
                }

                boost::archive::text_oarchive oa(ofs);
                oa << metadata;
                oa << data;
            }
            catch (const boost::archive::archive_exception& e)
            {
                std::error_code ec;
                std::filesystem::remove(filePath, ec);
                std::cerr << "Failed to save file (" << filePath.filename() << ") err:" << e.what() << std::endl;
                return false;
            }
            catch (...)
            {
                return false;
            }
            return true;
        }

        bool loadMatrixWithMetadata(const std::filesystem::path& filePath, std::string& outMetadata,
                                    std::vector<std::vector<double>>& outData)
        {
            try
            {
                std::ifstream ifs(filePath, std::ios::in);
                if (!ifs.is_open())
                {
                    std::cerr << "Failed to  open file (" << filePath.filename() << ")" << std::endl;
                    return false;
                }

                boost::archive::text_iarchive ia(ifs);
                ia >> outMetadata;
                ia >> outData;
            }
            catch (const boost::archive::archive_exception& e)
            {
                std::cerr << "Failed to load file (" << filePath.filename() << ") err:" << e.what() << std::endl;
                return false;
            }
            catch (...)
            {
                return false;
            }
            return true;
        }

        void saveOutputToFile(const std::string& filename, const Robot_Output& output)
        {
            try
            {
                // 确保父目录存在
                std::filesystem::path filePath(filename);
                if (auto parent = filePath.parent_path(); !parent.empty())
                {
                    std::filesystem::create_directories(parent);
                }

                // 续写
                std::ofstream ofs(filePath, std::ios::out | std::ios::app);
                if (!ofs.is_open())
                {
                    std::cerr << "Failed to open file (" << filePath.filename() << ")" << std::endl;
                    return;
                }
                ros::Time now = ros::Time::now();

                ofs << std::fixed << std::setprecision(6);
                ofs << now.sec << "." << std::setw(9) << std::setfill('0') << now.nsec;
                for (auto i = 0; i < output.target_q.size(); i++)
                {
                    ofs << " " << output.action[i];
                }
                ofs << std::endl;
                ofs.close();
            }
            catch (const boost::archive::archive_exception& e)
            {
                std::cerr << "Failed to save file (" << filename << ") err:" << e.what() << std::endl;
            }
        }

        void saveVectorXdToFile(const std::string& filename, const Eigen::VectorXd& observation)
        {
            try
            {
                // 确保父目录存在
                std::filesystem::path filePath(filename);
                if (auto parent = filePath.parent_path(); !parent.empty())
                {
                    std::filesystem::create_directories(parent);
                }

                std::ofstream ofs(filePath, std::ios::out | std::ios::app);
                if (!ofs.is_open())
                {
                    std::cerr << "Failed to open file (" << filePath.filename() << ")" << std::endl;
                    return;
                }
                ros::Time now = ros::Time::now();
                ofs << std::fixed << std::setprecision(6);
                ofs << now.sec << "." << std::setw(9) << std::setfill('0') << now.nsec;
                for (int i = 0; i < observation.size(); i++)
                {
                    ofs << " " << observation[i];
                }
                ofs << std::endl;
                ofs.close();
            }
            catch (const boost::archive::archive_exception& e)
            {
                std::cerr << "Failed to save file (" << filename << ") err:" << e.what() << std::endl;
            }
        }

        void saveMatrixXdToFile(const std::string& filename, const Eigen::MatrixXd& input)
        {
            try
            {
                // 确保父目录存在
                std::filesystem::path filePath(filename);
                if (auto parent = filePath.parent_path(); !parent.empty())
                {
                    std::filesystem::create_directories(parent);
                }

                std::ofstream ofs(filePath, std::ios::out | std::ios::app);
                if (!ofs.is_open())
                {
                    std::cerr << "Failed to open file (" << filePath.filename() << ")" << std::endl;
                    return;
                }
                ros::Time now = ros::Time::now();

                ofs << std::fixed << std::setprecision(6);
                ofs << now.sec << "." << std::setw(9) << std::setfill('0') << now.nsec;
                for (int i = 0; i < input.rows(); i++)
                {
                    for (int j = 0; j < input.cols(); j++)
                    {
                        ofs << " " << input(i, j);
                    }
                }
                ofs << std::endl;
                ofs.close();
            }
            catch (const boost::archive::archive_exception& e)
            {
                std::cerr << "Failed to save file (" << filename << ") err:" << e.what() << std::endl;
            }
        }

        void saveStringToFile(const std::string& filename, const std::string& str)
        {
            try
            {
                // 确保父目录存在
                std::filesystem::path filePath(filename);
                if (auto parent = filePath.parent_path(); !parent.empty())
                {
                    std::filesystem::create_directories(parent);
                }

                std::ofstream ofs(filePath, std::ios::out);
                if (!ofs.is_open())
                {
                    std::cerr << "Failed to open file (" << filePath.filename() << ")" << std::endl;
                    return;
                }
                ros::Time now = ros::Time::now();

                ofs << std::fixed << std::setprecision(6);
                ofs << str;
                ofs << std::endl;
                ofs.close();
            }
            catch (const boost::archive::archive_exception& e)
            {
                std::cerr << "Failed to save file (" << filename << ") err:" << e.what() << std::endl;
            }
        }

        std::string getSystemInfo()
        {
            std::stringstream ss;

            // 获取内存使用情况
            struct sysinfo memInfo;
            if (sysinfo(&memInfo) == 0)
            {
                long long totalRam = memInfo.totalram * memInfo.mem_unit;
                long long freeRam = memInfo.freeram * memInfo.mem_unit;
                long long usedRam = totalRam - freeRam;
                double ramUsagePercent = (double)usedRam / totalRam * 100;

                ss << "Memory: " << ramUsagePercent << "% used ("
                   << usedRam / (1024 * 1024) << "MB/" << totalRam / (1024 * 1024) << "MB), ";
            }

            // 获取CPU使用率（简化版本）
            std::ifstream statFile("/proc/stat");
            std::string line;
            if (std::getline(statFile, line))
            {
                std::istringstream iss(line);
                std::string cpu;
                long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
                iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;

                long totalIdle = idle + iowait;
                long totalNonIdle = user + nice + system + irq + softirq + steal;
                long total = totalIdle + totalNonIdle;

                // 这里只是简单计算，实际应该与上一次的值比较
                double cpuUsagePercent = (double)totalNonIdle / total * 100;
                ss << "CPU: " << cpuUsagePercent << "%, ";
            }

            // 获取网络使用情况（简化版本）
            std::ifstream netFile("/proc/net/dev");
            if (netFile)
            {
                std::string line;
                long long totalRx = 0, totalTx = 0;

                // 跳过前两行标题
                std::getline(netFile, line);
                std::getline(netFile, line);

                while (std::getline(netFile, line))
                {
                    std::istringstream iss(line);
                    std::string iface;
                    long long rx, tx;
                    iss >> iface >> rx;
                    // 跳过中间字段
                    for (int i = 0; i < 7; i++)
                    {
                        long long tmp;
                        iss >> tmp;
                    }
                    iss >> tx;

                    totalRx += rx;
                    totalTx += tx;
                }

                ss << "Network: RX=" << totalRx / (1024 * 1024) << "MB, TX=" << totalTx / (1024 * 1024) << "MB";
            }

            return ss.str();
        }

        std::string processSuffix(const std::string& suffix)
        {
            if (suffix.empty())
            {
                return "";
            }

            // 去掉前缀所有的'.'字符
            size_t startPos = suffix.find_first_not_of('.');
            if (startPos == std::string::npos)
            {
                // 后缀全是'.'（如"..."），视为无效后缀
                return "";
            }

            // 截取纯后缀部分，添加统一的'.'前缀
            std::string pureSuffix = suffix.substr(startPos);
            return "." + pureSuffix;
        }

        std::string replaceOrAddSuffix(const std::string& path, const std::string& suffix)
        {
            // 处理目标后缀，获取标准化格式
            std::string targetSuffix = processSuffix(suffix);
            if (targetSuffix.empty())
            {
                // 无效后缀，返回原路径
                return path;
            }

            // 查找路径中最后一个'.'的位置（判断是否已有后缀）
            size_t lastDotPos = path.rfind('.');

            // 判断是否需要添加后缀：
            // 1. 没有找到'.'（无后缀）
            // 2. '.'在路径末尾（如"file."，视为无有效后缀）
            if (lastDotPos == std::string::npos || lastDotPos == path.size() - 1)
            {
                return path + targetSuffix;
            }

            // 已有有效后缀，替换最后一个后缀
            return path.substr(0, lastDotPos) + targetSuffix;
        }

    }
}
