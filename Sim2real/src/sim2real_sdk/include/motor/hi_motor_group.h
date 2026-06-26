#ifndef HI_MOTOR_GROUP_H_
#define HI_MOTOR_GROUP_H_
#include "motor/robot_motor_group.h"

namespace hightorque
{
    class HiMotorGroup : public RobotMotorGroup
    {
    public:
        HiMotorGroup() = delete;

        HiMotorGroup(std::shared_ptr<livelybot_serial::robot> rbPtr,
                     std::vector<int> mapIndex,
                     std::vector<std::string> names,
                     std::vector<double> lowest,
                     std::vector<double> highest);
    };
}

#endif 