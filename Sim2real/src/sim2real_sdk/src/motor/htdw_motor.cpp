#include "motor/motor_base.h"

namespace hightorque
{

    motor_back_t HtdwMotor::getMotorBackState() const
    {
        motor_back_t motorBackData = *rbPtr_->Motors[index_]->get_current_motor_state();
        if (motorBackData.fault > 0)
        {
            ROS_WARN_THROTTLE(2.0, "[%d]%s fault code: %d", index_, name_.c_str(), motorBackData.fault);
        }
        return motorBackData;
    }

    void HtdwMotor::setMotor(float position, float velocity, float torque, float kp, float ki, float kd, float acc, float voltage, float current, MotorControlType type)
    {
        // TODO:修改为传入参数
        // 执行预设动作 限位大一点
        auto highest = type == POS_VEL_MAX_TQE ? 6.0 : highest_;
        auto lowest = type == POS_VEL_MAX_TQE ? -6.0 : lowest_;
        if (position > highest)
        {
            // std::cerr << "[" << index_ << "]" << name_ << ":" << position << " > " << highest << std::endl;
            position = highest;
        }
        if (position < lowest)
        {
            // std::cerr << "[" << index_ << "]" << name_ << ":" << position << " < " << lowest << std::endl;
            position = lowest;
        }
        lastCmd_ = MotorControlCmd(position, velocity, torque,
                                   kp, ki, kd,
                                   acc, voltage, current, type);
        switch (type)
        {
            case POS:
                rbPtr_->Motors[index_]->position(position);
                break;
            case VEL:
                rbPtr_->Motors[index_]->velocity(velocity);
                break;
            case TQE:
                rbPtr_->Motors[index_]->torque(torque);
                break;
            case VOL:
                rbPtr_->Motors[index_]->voltage(voltage);
                break;
            case CUR:
                rbPtr_->Motors[index_]->current(current);
                break;
            case POS_VEL_MAX_TQE:
                rbPtr_->Motors[index_]->pos_vel_MAXtqe(position, velocity, torque);
                break;
            case POS_VEL_TQE_RKP_RKD:
                rbPtr_->Motors[index_]->pos_vel_tqe_kp_kd(position, velocity, torque, kp, kd);
                break;
            case POS_VEL_RKP_RKD:
                rbPtr_->Motors[index_]->pos_vel_kp_kd(position, velocity, kp, kd);
                break;
            case POS_VEL_TQE_KP_KD:
                rbPtr_->Motors[index_]->pos_vel_tqe_kp_kd(position, velocity, torque, kp, kd);
                break;
            case POS_VEL_KP_KD:
                rbPtr_->Motors[index_]->pos_vel_kp_kd(position, velocity, kp, kd);
                break;
            case POS_VEL_ACC:
                rbPtr_->Motors[index_]->pos_vel_acc(position, velocity, acc);
                break;
            case POS_VEL_TQE_KP_KD2:
                rbPtr_->Motors[index_]->pos_vel_tqe_kp_kd2(position, velocity, torque, kp, kd);
                break;
        }
    }

    void HtdwMotor::setMotor(MotorControlCmd cmd)
    {
        // TODO:修改为传入参数
        // 执行预设动作 限位大一点
        auto highest = cmd.type == POS_VEL_MAX_TQE ? 6.0 : highest_;
        auto lowest = cmd.type == POS_VEL_MAX_TQE ? -6.0 : lowest_;
        if (cmd.position > highest)
        {
            std::cerr << "[" << index_ << "]" << name_ << ":" << cmd.position << " > " << highest << std::endl;
            cmd.position = highest;
        }
        if (cmd.position < lowest)
        {
            std::cerr << "[" << index_ << "]" << name_ << ":" << cmd.position << " < " << lowest << std::endl;
            cmd.position = lowest;
        }
        float position = cmd.position, velocity = cmd.velocity, torque = cmd.torque,
              kp = cmd.kp, ki = cmd.ki, kd = cmd.kd,
              acc = cmd.acc, voltage = cmd.voltage, current = cmd.current;

        lastCmd_ = MotorControlCmd(position, velocity, torque,
                                   kp, ki, kd,
                                   acc, voltage, current, cmd.type);
        switch (cmd.type)
        {

            case POS:
                rbPtr_->Motors[index_]->position(position);
                break;
            case VEL:
                rbPtr_->Motors[index_]->velocity(velocity);
                break;
            case TQE:
                rbPtr_->Motors[index_]->torque(torque);
                break;
            case VOL:
                rbPtr_->Motors[index_]->voltage(voltage);
                break;
            case CUR:
                rbPtr_->Motors[index_]->current(current);
                break;
            case POS_VEL_MAX_TQE:
                rbPtr_->Motors[index_]->pos_vel_MAXtqe(position, velocity, torque);
                break;
            case POS_VEL_TQE_RKP_RKD:
                rbPtr_->Motors[index_]->pos_vel_tqe_kp_kd(position, velocity, torque, kp, kd);
                break;
            case POS_VEL_RKP_RKD:
                rbPtr_->Motors[index_]->pos_vel_kp_kd(position, velocity, kp, kd);
                break;
            case POS_VEL_TQE_KP_KD:
                rbPtr_->Motors[index_]->pos_vel_tqe_kp_kd(position, velocity, torque, kp, kd);
                break;
            case POS_VEL_KP_KD:
                rbPtr_->Motors[index_]->pos_vel_kp_kd(position, velocity, kp, kd);
                break;
            case POS_VEL_ACC:
                rbPtr_->Motors[index_]->pos_vel_acc(position, velocity, acc);
                break;
            case POS_VEL_TQE_KP_KD2:
                rbPtr_->Motors[index_]->pos_vel_tqe_kp_kd2(position, velocity, torque, kp, kd);
                break;
        }
    }

    void HtdwMotor::setMotorRelative(float relativePosition, float velocity, float torque, float kp, float ki, float kd, float acc, float voltage, float current, MotorControlType type)
    {
        // 获取当前位置
        float currentPosition = getMotorBackState().position;

        // 计算目标位置 = 当前位置 + 相对位移
        float position = currentPosition + relativePosition;

        setMotor(position, velocity, torque,
                 kp, ki, kd,
                 acc, voltage, current, type);
    }

    void HtdwMotor::setMotorRelative(MotorControlCmd cmd)
    {
        // 获取当前位置
        float currentPosition = getMotorBackState().position;

        // 计算目标位置 = 当前位置 + 相对位移
        cmd.position += currentPosition;

        setMotor(cmd);
    }

    void HtdwMotor::protectMotor()
    {
        float kp = 0.0, kd = 1.0;
        setMotor(0, 0, 0,
                 kp, 0, kd,
                 0, 0, 0);
        lastCmd_ = MotorControlCmd(0, 0, 0,
                                   kp, 0, kd,
                                   0, 0, 0);
    }

    void HtdwMotor::keepMotorLastCmd()
    {
        setMotor(lastCmd_.position, lastCmd_.velocity, lastCmd_.torque,
                 lastCmd_.kp, lastCmd_.ki, lastCmd_.kd,
                 lastCmd_.acc, lastCmd_.voltage, lastCmd_.current, lastCmd_.type);
    }
}
