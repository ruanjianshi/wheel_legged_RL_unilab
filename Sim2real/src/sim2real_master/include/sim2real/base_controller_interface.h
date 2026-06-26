#ifndef BASE_CONTROLLER_INTERFACE_H_

#define BASE_CONTROLLER_INTERFACE_H_
#include<pinocchio/fwd.hpp>
// #include<pd_controller.h>
// #include<rl_controller.h>
#include<iostream>
#include<memory>
#include<ros/ros.h>
#include<hardware/robot.h>



namespace hightorque{

class BaseControllerInterface
{
public:
    BaseControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr):nh()
    {};

    virtual ~BaseControllerInterface()
    {
        std::cout<<  "~BaseControllerInterface()"<<std::endl;
    };

    virtual int get_parameters()=0;
    virtual bool get_is_shutdowm()=0;

protected:
    ros::NodeHandle nh;
private:

};

};



#endif