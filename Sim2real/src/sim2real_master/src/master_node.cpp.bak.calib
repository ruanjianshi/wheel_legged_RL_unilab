#include "sim2real/master_node.h"
#include "livelybot_logger/logger_interface.h"
#include "std_msgs/String.h"

namespace hightorque
{

    MasterNode::MasterNode() : nh(), modeFsm_(this)
    {
        init();
    }

    MasterNode::~MasterNode()
    {
        isRunning_.store(false);
        if (execLoopPtr_->joinable())
        {
            execLoopPtr_->join();
        }
    }

    void MasterNode::init()
    {
        robotPtr_ = std::make_shared<livelybot_serial::robot>();
        // ctrl_interface_ptr_.reset(new DefaultControllerInterface(robot_ptr_));
        // fsm_node_type_ = FsmNodeType::EXEC_DEFAULT;
        modeFsm_.start();
        modeFsm_.process_event(StartEvt{});

        livelybot_logger::LoggerInterface::init(nh, "sim2real_master");
        // nh.param<std::string>("joy_command_topic", this->joy_command_topic_, "/fsm_command");
        nh.param<std::string>("joint_state_topic", this->jointStateTopic_, "/error_joint_states");
        nh.param<std::string>("imu_msg_topic", this->imuMsgTopic_, "/imu/data");
        nh.param<std::string>("fsm_state_topic", this->fsmStateTopic_, "/fsm_state");
        nh.param<std::string>("version_topic", this->versionTopic_, "/version");

        joyCommandSub_ = nh.subscribe<sim2real_msg::Joy>("/joy_msg", 1, &MasterNode::joyMsgCallback, this);

        fsmStatePub_ = nh.advertise<std_msgs::Int32>(fsmStateTopic_, 10);
        versionPub_ = nh.advertise<livelybot_serial::VersionInfo>(versionTopic_, 10, true);
        jointStateSub_ = nh.subscribe<sensor_msgs::JointState>(jointStateTopic_, 10, &MasterNode::jointStateMsgCallback, this);
        imuMsgSub_ = nh.subscribe<sensor_msgs::Imu>(imuMsgTopic_, 10, &MasterNode::imuMsgCallback, this);
        motorErrorMap_.clear();

        // usr_pub_ = nh.subscribe<hightorque_hardware_sdk::usr2ctrl>("usr2ctrl_data", 10, &MasterNode::usr_callback, this);
        lastImuDataTimeS_ = -1;
        imuNormal_.store(true);
        motorNormal_.store(true);
        errorCount_.store(0);
        isRunning_.store(true);
        execLoopPtr_.reset(new std::thread(&MasterNode::fsmRuntimeLoop, this));
    }

    void MasterNode::fsmModePulish()
    {
        std_msgs::Int32 msg;
        // msg.data = static_cast<int32_t>(fsm_node_type_);
        msg.data = static_cast<int32_t>(modeFsm_.getState());
        fsmStatePub_.publish(msg);
    }

    void MasterNode::versionPulish()
    {
        livelybot_serial::VersionInfo msg;
        msg.device_type = DEVICE_TYPE;
        msg.version = VERSION;
        versionPub_.publish(msg);
    }
    // 通过时间戳判断是否丢失
    void MasterNode::imuMsgCallback(const sensor_msgs::Imu::ConstPtr& imu_ptr)
    {
        currentImuDataTimeS_ = imu_ptr->header.stamp.toSec();
        if (lastImuDataTimeS_ == -1)
        {
            lastImuDataTimeS_ = currentImuDataTimeS_;
            imuNormal_.store(true);
        }
        else
        {
            if (currentImuDataTimeS_ - lastImuDataTimeS_ > 0.3)
            {
                imuNormal_.store(false);
                lastImuDataTimeS_ = currentImuDataTimeS_;
            }
            else
            {
                imuNormal_.store(true);
                lastImuDataTimeS_ = currentImuDataTimeS_;
            }
        }
    }
    // 通过电机角度判断是否丢失
    void MasterNode::jointStateMsgCallback(const sensor_msgs::JointState::ConstPtr& joint_state_msg_ptr)
    {
        bool packet_has_error = false;
        // for (size_t i = 0; i < joint_state_msg_ptr->name.size(); ++i)
        for (size_t i = 0; i < 12; ++i)
        {
            if (joint_state_msg_ptr->position[i] < -20 || joint_state_msg_ptr->position[i] > 29) //@TODO:why -20 and 29
            {
                packet_has_error = true;
                break;
            }
        }

        if (packet_has_error)
        {
            ++errorCount_;
            if (errorCount_ >= MAX_ERROR_COUNT)
            {
                motorNormal_.store(false);
            }
        }
        else
        {
            errorCount_.store(0);
            motorNormal_.store(true);
        }
    }

    bool MasterNode::allDeviceNormal()
    {
        return (imuNormal_.load() && motorNormal_.load());
    }

    void MasterNode::fsmRuntimeLoop()
    {
        ros::Rate rate(20);
        while (isRunning_ && ros::ok())
        {
            fsmModePulish();
            versionPulish();
            modeFsm_.process_event(TickEvt{});
            rate.sleep();
            continue;
        }
    }
    // 遥控器操作
    void MasterNode::joyMsgCallback(const sim2real_msg::Joy::ConstPtr& msg) // 通过摇杆控制状态机
    {
        if (msg->center == 1)
        {
            modeFsm_.process_event(DefaultModeEvt{});
            fsmModePulish();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            return;
        }

        bool bothPressed = (msg->lt < -0.1 && msg->rt < -0.1);
        if (!bothPressed)
        {
            return;
        }

        if (msg->dpad_horizontal == 1)
        {
            modeFsm_.process_event(LastOptionEvt{});
        }
        else if (msg->dpad_horizontal == -1)
        {
            modeFsm_.process_event(NextOptionEvt{});
        }
        else if (msg->a == 1)
        {
            modeFsm_.process_event(ConfirmEvt{});
        }
        else if (msg->b == 1)
        {
            modeFsm_.process_event(ProtectionShutdownEvt{});
        }
        else
        {
            return;
        }
        fsmModePulish();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return;
    }
}
