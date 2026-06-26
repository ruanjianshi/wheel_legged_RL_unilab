#include "motor/pi_plus_motor_group.h"
#include "motor/motor_base.h"

namespace hightorque
{
    PiPlusMotorGroup::PiPlusMotorGroup(std::shared_ptr<livelybot_serial::robot> rbPtr, std::vector<int> mapIndex, std::vector<std::string> names, std::vector<double> lowest, std::vector<double> highest) : RobotMotorGroup(rbPtr, mapIndex, names)
    {
        // 创建全身
        std::vector<std::shared_ptr<HtdwMotor>> allMotors, lowerBodyMotors, upperBodyMotors, allWithoutHeadMotors;
        allMotors.reserve(mapIndex.size());

        for (auto i = 0; i < mapIndex.size(); i++)
        {
            auto motor = std::make_shared<HtdwMotor>(rbPtr, mapIndex[i], names[i], lowest[mapIndex[i]], highest[mapIndex[i]]);
            std::cout << "PiPlusMotorGroup create [" << i << "/" << mapIndex.size() << "] index: " << mapIndex[i] << " name: " << names[i] << " motor (" << lowest[mapIndex[i]] << "~" << highest[mapIndex[i]] << ")" << std::endl;
            if (mapIndex[i] < 12)
            {
                lowerBodyMotors.push_back(motor);
            }
            else
            {
                upperBodyMotors.push_back(motor);
            }

            if (names[i].find("head") == std::string::npos)
            {
                allWithoutHeadMotors.push_back(motor);
            }

            allMotors.push_back(motor);
        }

        motorGroups_.emplace("all", std::make_shared<MotorGroup>(allMotors.size(), allMotors));
        motorGroups_.emplace("lowerBody", std::make_shared<MotorGroup>(lowerBodyMotors.size(), lowerBodyMotors));
        motorGroups_.emplace("upperBody", std::make_shared<MotorGroup>(upperBodyMotors.size(), upperBodyMotors));
    }

}
