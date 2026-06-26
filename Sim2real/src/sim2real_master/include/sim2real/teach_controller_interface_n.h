#ifndef TEACH_CONTROLLER_INTERFACE_H_N
#define TEACH_CONTROLLER_INTERFACE_H_N

#include "sim2real/base_controller_interface.h"
#include "impl/hightorque_config.h"
#include "motor/robot_motor_group.h"
#include "impl/utils.h"
#include "sim2real_msg/WayPoint.h"

namespace hightorque
{
    namespace teach
    {
        class TeachControllerInterface : public BaseControllerInterface
        {
            public:
                TeachControllerInterface(std::shared_ptr<livelybot_serial::robot> rbPtr);
                ~TeachControllerInterface();

                int get_parameters() override
                {
                    return 0;
                }
                bool get_is_shutdowm() override
                {
                    return quit_;
                }

                bool start();
                bool stop();
                bool initMotor();

            private:
                bool quit_;
                double kickDuration_;
                std::string fileName_, filePath_, group_, modelType_;

                std::shared_timed_mutex mutex_; // 互斥锁

                std::shared_ptr<RobotMotorGroup> robotMotor_;
                std::shared_ptr<ros::NodeHandle> nh_;
                std::vector<Motor_State> motorStateVec_;

                config::PDConfig pdInfo_;                     // PID 配置（TODO: 重构）
                utils::TrajectoryGenerator* trajectory_; // 轨迹生成器（单例模式非拥有）

                ros::Subscriber joySub_;
                ros::Subscriber waypointSub_;
                ros::Subscriber fileNameSub_;
                ros::Subscriber motorGroupSub_;

                std::shared_ptr<std::thread> threadExecPtr_; // 执行线程指针

                void joyCallback(const sim2real_msg::Joy::ConstPtr& msg);
                void setKpKd() const;
                void execLoop();
        };
    }

}

#endif
