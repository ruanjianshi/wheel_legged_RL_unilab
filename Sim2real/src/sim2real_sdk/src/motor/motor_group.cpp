#include "motor/motor_group.h"

namespace hightorque
{
    std::vector<motor_back_t> MotorGroup::getMotorsBackState() const
    {
        std::vector<motor_back_t> motorsBackState;
        for (const auto& motor : motors_)
        {
            motorsBackState.push_back(motor->getMotorBackState());
        }
        return std::move(motorsBackState);
    }

    Motor_State MotorGroup::getMotorState() const
    {
        Motor_State motorsState;
        motorsState.resize(size_);
        for (int i = 0; i < size_; i++)
        {
            const auto motor = motors_[i];
            auto backState = motor->getMotorBackState();
            motorsState.q[i] = backState.position;
            motorsState.dq[i] = backState.velocity;
            motorsState.tau[i] = backState.torque;
        }
        return std::move(motorsState);
    }

    Motor_State MotorGroup::getMotorStateByIndexUp() const
    {
        Motor_State motorsState;
        motorsState.resize(size_);

        // 创建按索引排序的电机副本
        std::vector<std::shared_ptr<MotorBase>> sortedMotors = motors_;
        std::sort(sortedMotors.begin(), sortedMotors.end(),
                  [](const auto& a, const auto& b) {
                      return a->getIndex() < b->getIndex(); // 按index升序排序
                  });
        for (int i = 0; i < sortedMotors.size(); i++)
        {
            const auto motor = sortedMotors[i];
            auto backState = motor->getMotorBackState();
            motorsState.q[i] = backState.position;
            motorsState.dq[i] = backState.velocity;
            motorsState.tau[i] = backState.torque;
        }
        return std::move(motorsState);
    }

    std::string MotorGroup::getMotorStateString() const
    {
        std::stringstream ss;
        ss << "MotorGroup State (size=" << size_ << "):\n";

        for (int i = 0; i < size_; i++)
        {
            const auto& motor = motors_[i];
            auto backState = motor->getMotorBackState();

            ss << "  [" << i << "] " << motor->getName()
               << " (index=" << motor->getIndex() << "):\n"
               << "    Position: " << backState.position << "\n"
               << "    Velocity: " << backState.velocity << "\n"
               << "    Torque:   " << backState.torque << "\n";
        }

        return ss.str();
    }

    void MotorGroup::setMotors(const Motor_Output out, MotorControlType type) const
    {
        for (int i = 0; i < size_; i++)
        {
            const auto motor = motors_[i];
            motor->setMotor(out.target_q[i], out.target_dq[i], out.target_tau[i],
                            out.target_kp[i], 0, out.target_kd[i],
                            0, 0, 0, type);
        }
    }

    void MotorGroup::setMotors(const std::vector<MotorControlCmd> cmd) const
    {
        for (int i = 0; i < size_; i++)
        {
            const auto motor = motors_[i];
            motor->setMotor(cmd[i]);
        }
    }

    void MotorGroup::setMotorsRelative(const Motor_Output out, MotorControlType type) const
    {
        if (out.target_q.size() < size_)
        {
            std::cerr << "MotorGroup out size " << out.target_q.size() << " error, expected at least " << size_ << std::endl;
        }
        for (int i = 0; i < size_; i++)
        {
            const auto motor = motors_[i];
            motor->setMotorRelative(out.target_q[i], out.target_dq[i], out.target_tau[i],
                                    out.target_kp[i], 0, out.target_kd[i],
                                    0, 0, 0, type);
        }
    }

    void MotorGroup::setMotorsRelative(const std::vector<MotorControlCmd> cmd) const
    {
        if (cmd.size() < size_)
        {
            std::cerr << "MotorGroup out size " << cmd.size() << " error, expected at least " << size_ << std::endl;
        }
        for (int i = 0; i < size_; i++)
        {
            const auto motor = motors_[i];
            motor->setMotorRelative(cmd[i]);
        }
    }

    // 按索引升序设置电机（绝对位置）
    void MotorGroup::setMotorsByIndexUp(const Motor_Output out, MotorControlType type) const
    {
        // 创建按索引排序的电机副本
        std::vector<std::shared_ptr<MotorBase>> sortedMotors = motors_;
        std::sort(sortedMotors.begin(), sortedMotors.end(),
                  [](const auto& a, const auto& b) {
                      return a->getIndex() < b->getIndex(); // 按index升序排序
                  });

        // 确保输出数据大小匹配
        if (out.target_q.size() < sortedMotors.size())
        {
            std::cerr << "MotorGroup out size " << out.target_q.size() << " error, expected at least " << sortedMotors.size() << std::endl;
            return;
        }

        // 按索引升序应用控制命令
        for (size_t i = 0; i < sortedMotors.size(); i++)
        {
            const auto motor = sortedMotors[i];
            motor->setMotor(out.target_q[i], out.target_dq[i], out.target_tau[i],
                            out.target_kp[i], 0, out.target_kd[i],
                            0, 0, 0, type);
        }
    }

    // 按索引升序设置电机（使用命令向量）
    void MotorGroup::setMotorsByIndexUp(const std::vector<MotorControlCmd> cmd) const
    {
        // 创建按索引排序的电机副本
        std::vector<std::shared_ptr<MotorBase>> sortedMotors = motors_;
        std::sort(sortedMotors.begin(), sortedMotors.end(),
                  [](const auto& a, const auto& b) {
                      return a->getIndex() < b->getIndex(); // 按index升序排序
                  });

        // 确保命令向量大小匹配
        if (cmd.size() < sortedMotors.size())
        {
            std::cerr << "MotorGroup cmd size " << cmd.size() << " error, expected at least " << sortedMotors.size() << std::endl;
            return;
        }

        // 按索引升序应用控制命令
        for (size_t i = 0; i < sortedMotors.size(); i++)
        {
            const auto motor = sortedMotors[i];
            motor->setMotor(cmd[i]);
        }
    }

    // 按索引升序设置电机（相对位置）
    void MotorGroup::setMotorsRelativeByIndexUp(const Motor_Output out, MotorControlType type) const
    {
        // 创建按索引排序的电机副本
        std::vector<std::shared_ptr<MotorBase>> sortedMotors = motors_;
        std::sort(sortedMotors.begin(), sortedMotors.end(),
                  [](const auto& a, const auto& b) {
                      return a->getIndex() < b->getIndex(); // 按index升序排序
                  });

        // 确保输出数据大小匹配
        if (out.target_q.size() < sortedMotors.size())
        {
            std::cerr << "MotorGroup out size " << out.target_q.size() << " error, expected at least " << sortedMotors.size() << std::endl;
            return;
        }

        // 按索引升序应用相对控制命令
        for (size_t i = 0; i < sortedMotors.size(); i++)
        {
            const auto motor = sortedMotors[i];
            motor->setMotorRelative(out.target_q[i], 0, 0,
                                    out.target_kp[i], 0, out.target_kd[i],
                                    0, 0, 0, type);
        }
    }

    // 按索引升序设置电机（相对位置，使用命令向量）
    void MotorGroup::setMotorsRelativeByIndexUp(const std::vector<MotorControlCmd> cmd) const
    {
        // 创建按索引排序的电机副本
        std::vector<std::shared_ptr<MotorBase>> sortedMotors = motors_;
        std::sort(sortedMotors.begin(), sortedMotors.end(),
                  [](const auto& a, const auto& b) {
                      return a->getIndex() < b->getIndex(); // 按index升序排序
                  });

        // 确保命令向量大小匹配
        if (cmd.size() < sortedMotors.size())
        {
            std::cerr << "MotorGroup cmd size " << cmd.size() << " error, expected at least " << sortedMotors.size() << std::endl;
            return;
        }

        // 按索引升序应用相对控制命令
        for (size_t i = 0; i < sortedMotors.size(); i++)
        {
            const auto motor = sortedMotors[i];
            motor->setMotorRelative(cmd[i]);
        }
    }

    void MotorGroup::protectMotor() const
    {
        for (int i = 0; i < size_; i++)
        {
            const auto motor = motors_[i];
            motor->protectMotor();
        }
    }

    std::shared_ptr<MotorBase> MotorGroup::getMotor(const std::string& name) const
    {
        auto it = motorsMap_.find(name);
        if (it == motorsMap_.end())
        {
            std::cerr << "group not such motor:" << name << std::endl;
            return nullptr;
        }
        return it->second;
    }

}
