#ifndef TEACH_CONTROLLER_INTERFACE_H_
#define TEACH_CONTROLLER_INTERFACE_H_


#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <condition_variable>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

//#include <pd_controller.h>//应该是hardware/robot.h包含了boost头文件 pd_controller.h包含了Pinocchio 头文件 所以要放在robot.h的前面

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/String.h>
#include "sim2real_msg/WayPoint.h"

#include "hardware/robot.h"
#include "sim2real/base_controller_interface.h"
#include "sim2real/CubicSplineInterpolator.h"

namespace hightorque
{
    class TeachControllerInterface : public BaseControllerInterface
    {
    public:
        ros::NodeHandle nh;

        TeachControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr);
        ~TeachControllerInterface();

        int get_parameters() override;
        bool get_is_shutdowm() override;

    private:
        void teaching();
        void save_backups(const std::vector<std::vector<double>>& data);
        void joyCallback(const sensor_msgs::Joy::ConstPtr& joy_msg);
        void waypointCallback(const sim2real_msg::WayPoint::ConstPtr& waypoint_msg);
        void keyboardCallback(const std_msgs::String::ConstPtr& msg);
        std::vector<std::vector<double>> vector_transpose(const std::vector<std::vector<double>>& matrix);
        void saveVectorWithBoost(const std::vector<std::vector<double>>& data, const std::string& filename);
        std::vector<std::vector<double>> loadVectorWithBoost(const std::string& filename);

        std::vector<std::vector<double>> act;
        std::vector<std::vector<double>> way_point;
        std::vector<std::vector<double>> pos_way_point_continue;
        std::vector<std::vector<double>> vel_way_point_continue;

        int run_mode;
        int cnt;
        int sample_stop;
        uint8_t lock_type;
        int leg_cnt;
        std::string waypoint_save_filename;
        std::string waypoint_load_filename;
        std::string way_point_back_filename;
        std::vector<double> stop_pos;

        std::vector<std::vector<double>>::iterator joint_pos;
        std::vector<std::vector<double>>::iterator joint_vel;

        ros::Subscriber joy_subscriber_;
        ros::Subscriber waypoint_control_subscriber_;
        ros::Subscriber keyboard_subscriber_;

    private:
        std::shared_ptr<livelybot_serial::robot> rb_ptr_;
        std::shared_ptr<std::thread> thread_exec_ptr_;
        std::atomic<bool> quit;
    };

};

#endif