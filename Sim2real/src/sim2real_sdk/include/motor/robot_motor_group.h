#ifndef ROBOT_MOTOR_GROUP_H_
#define ROBOT_MOTOR_GROUP_H_
#include "motor/motor_group.h"
#include <unordered_map>

namespace hightorque
{
    /** =========================================================================
     * @brief RobotMotorGroup 机器人整体电机组控制类
     *
     * 该类负责管理多个命名的电机组（MotorGroup），用于按名称访问和控制子模块（如腿、手臂等）。
     * 提供对电机组状态获取、指令发送、保持状态等接口。
     * ========================================================================= */
    class RobotMotorGroup
    {
        public:
            RobotMotorGroup() = delete;

            /**
             * @brief 构造函数
             *
             * @param rbPtr Livelybot 机器人底层指针
             * @param mapIndex 电机映射索引列表（与底层通讯映射用）
             * @param names 每组电机组的名称，用于标识电机（如 "r_ankle_roll_joint"）
             */
            RobotMotorGroup(std::shared_ptr<livelybot_serial::robot> rbPtr,
                            std::vector<int> mapIndex,
                            std::vector<std::string> names);

            /**
             * @brief 发送所有电机命令到底层（立即执行）
             */
            inline void sendMotors() const
            {
                rbPtr_->motor_send_2();
            }

            /**
             * @brief 注册一个电机组
             * @param name       电机组名称
             * @param motorGroup 电机基类指针列表，用于组内管理
             */
            void registerMotorGroup(const std::string& name, const std::vector<std::shared_ptr<MotorBase>>& motorGroup)
            {
                motorGroups_.emplace(name, std::make_shared<MotorGroup>(motorGroup.size(), motorGroup));
            }

            /**
             * @brief 获取所有电机组的状态字符串
             * @return std::string 格式化状态信息（调试用）
             */
            std::string getAllMotorGroupsStateString() const;

            /**
             * @brief 根据名称获取电机组大小
             * @param name 电机组名称
             * @return 对应电机组中的电机数量；若未找到则返回 0
             */
            size_t getMotorGroupSizeByName(const std::string& name) const;

            /**
             * @brief 根据名称获取电机组内电机索引列表
             * @param name 电机组名称
             * @return 包含电机在全局列表中索引的 std::vector；若未找到则返回空列表
             */
            std::vector<int> getMotorGroupIndexByName(const std::string& name) const;

            /**
             * @brief 根据名称获取某个电机组状态
             * @param name 电机组名称
             * @return Motor_State 类
             */
            Motor_State getMotorGroupStateByName(const std::string& name) const;

            /**
             * @brief 根据名称获取某个电机组按Index升序状态
             * @param name 电机组名称
             * @return Motor_State 类
             */
            Motor_State getMotorGroupStateByIndexUpByName(const std::string& name) const;

            /**
             * @brief 设置某个电机组的绝对控制指令
             * @param name 电机组名称
             * @param Motor_Output 结构体
             * @param type 电机控制类型
             */
            void setMotorGroupByName(const std::string& name, const Motor_Output& out, MotorControlType type = MotorControlType::POS_VEL_TQE_KP_KD2) const;

            /**
             * @brief 设置某个电机组的绝对控制指令
             * @param name 电机组名称
             * @param MotorControlCmd vector 类
             */
            void setMotorGroupByName(const std::string& name, const std::vector<MotorControlCmd>& cmd) const;

            /**
             * @brief 设置某个电机组的相对控制指令
             * @param name 电机组名称
             * @param Motor_Output 结构体
             * @param type 电机控制类型
             */
            void setMotorGroupRelativeByName(const std::string& name, const Motor_Output& out, MotorControlType type = MotorControlType::POS_VEL_TQE_KP_KD2) const;

            /**
             * @brief 设置某个电机组的相对控制指令
             * @param name 电机组名称
             * @param MotorControlCmd vector 类
             */
            void setMotorGroupRelativeByName(const std::string& name, const std::vector<MotorControlCmd>& cmd) const;

            /**
             * @brief 设置某个电机组按index升序的绝对控制指令, 主要用于示教类播放点位
             * @param name 电机组名称
             * @param Motor_Output 结构体
             * @param type 电机控制类型
             */
            void setMotorGroupByIndexUpByName(const std::string& name, const Motor_Output& out, MotorControlType type = MotorControlType::POS_VEL_TQE_KP_KD2) const;

            /**
             * @brief 设置某个电机组按index升序的绝对控制指令, 主要用于示教类播放点位
             * @param name 电机组名称
             * @param MotorControlCmd vector 类
             */
            void setMotorGroupByIndexUpByName(const std::string& name, const std::vector<MotorControlCmd>& cmd) const;

            /**
             * @brief 设置某个电机组按index升序的相对控制指令, 主要用于示教类播放点位
             * @param name 电机组名称
             * @param Motor_Output 结构体
             * @param type 电机控制类型
             */
            void setMotorGroupRelativeByIndexUpByName(const std::string& name, const Motor_Output& out, MotorControlType type = MotorControlType::POS_VEL_TQE_KP_KD2) const;

            /**
             * @brief 设置某个电机组按index升序的相对控制指令, 主要用于示教类播放点位
             * @param name 电机组名称
             * @param MotorControlCmd vector 类
             */
            void setMotorGroupRelativeByIndexUpByName(const std::string& name, const std::vector<MotorControlCmd>& cmd) const;

            /**
             * @brief 让某电机组保持当前状态,（会比较“硬”但可动）
             * @param name 电机组名称
             */
            void protectMotorGroupByName(const std::string& name) const;

            /**
             * @brief 让某电机组保持上一次控制命令,让电机保持不动
             * @param name 电机组名称
             */
            void keepMotorGroupLastCmdByName(const std::string& name) const;

            /**
             * @brief 遍历指定名称的电机并对其执行给定操作
             * @param name   电机名称
             * @param action 对电机执行的操作函数，返回 false 时中断遍历
             */
            void foreachMotorByName(const std::string& name,
                                    std::function<bool(const std::shared_ptr<MotorBase>& motor)> action) const;

            /**
             * @brief 根据名称获取单个电机对象
             * @param name 电机名称
             * @return 对应名称的电机共享指针；若未找到则返回 nullptr
             */
            std::shared_ptr<MotorBase> getMotorByName(const std::string& name) const;

            std::unordered_map<std::string, std::shared_ptr<MotorGroup>> motorGroups_; // 电机组名 -> MotorGroup 映射表

            std::shared_ptr<livelybot_serial::robot> rbPtr_; // Livelybot 串口机器人底层控制对象

            std::vector<int> mapIndex_; // 电机索引映射表（对应底层硬件）

            std::vector<std::string> names_; // 电机组名称列表
    };
}

#endif
