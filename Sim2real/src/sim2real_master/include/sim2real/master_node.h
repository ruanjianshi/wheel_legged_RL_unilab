#ifndef MASTER_NODE_H_
#define MASTER_NODE_H_

#include "sim2real/fsm.h"
#include "sim2real/base_controller_interface.h"
#include "sim2real/default_controller_interface.h"
#include "sim2real/custom_controller_interface.h"
#include "sim2real/protection_controller_interface.h"
#include "sim2real/remote_controller_interface.h"
#include "sim2real/zerocalibration_controller_interface.h"
#include "sim2real/teach_controller_interface.h"
#include <atomic>
#include <thread>
#include "hardware/robot.h"
#include <sensor_msgs/JointState.h>
#include <std_msgs/String.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Int32.h>
#include <cstring>
#include <chrono>
#include "sim2real_msg/Joy.h"
#include "livelybot_serial/VersionInfo.h"
// #include "hightorque_hardware_sdk/usr2ctrl.h"
// #include "hightorque_hardware_sdk/ctrl2usr.h"

#define DEVICE_TYPE 0 // 运控为0
// #define VERSION "2.1.1"

namespace hightorque
{
    using ModeFsm = boost::msm::back::state_machine<ModeFsmDef>;

    class MasterNode
    {
        public:
            MasterNode();

            ~MasterNode();

            void init();

            void joyMsgCallback(const sim2real_msg::Joy::ConstPtr&);

            void imuMsgCallback(const sensor_msgs::Imu::ConstPtr&);

            void jointStateMsgCallback(const sensor_msgs::JointState::ConstPtr&);

            // void usr_callback(const hightorque_hardware_sdk::usr2ctrl::ConstPtr&);

            void fsmModePulish();

            void versionPulish();

            void fsmRuntimeLoop();

            bool allDeviceNormal();

        private:
            boost::msm::back::state_machine<ModeFsmDef> modeFsm_;

            std::shared_ptr<BaseControllerInterface> ctrlInterfacePtr_;

            std::shared_ptr<std::thread> fsmThreadPtr_;

            FsmNodeType fsmNodeType_;

            std::shared_ptr<livelybot_serial::robot> robotPtr_;

            std::atomic<bool> isRunning_;

            ros::NodeHandle nh;

            ros::Subscriber joyCommandSub_;
            ros::Subscriber jointStateSub_;
            ros::Subscriber imuMsgSub_;
            ros::Publisher fsmStatePub_;
            ros::Publisher versionPub_;

            std::string joyCommandTopic_;
            std::string jointStateTopic_;
            std::string imuMsgTopic_;
            std::string fsmStateTopic_;
            std::string versionTopic_;

            std::shared_ptr<std::thread> execLoopPtr_;
            std::map<int, int> motorErrorMap_;

            std::atomic<bool> imuNormal_;
            std::atomic<bool> motorNormal_;
            std::atomic<int> errorCount_;          // 连续错误数据包计数
            static const int MAX_ERROR_COUNT = 10; // 最大允许的连续错误数据包数量

            double lastImuDataTimeS_;

            double currentImuDataTimeS_;

            ros::Time stateEntryTime_;

            friend struct ModeFsmDef;
    };
}

#endif
