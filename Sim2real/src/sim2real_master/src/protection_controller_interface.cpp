#include "sim2real/protection_controller_interface.h"

namespace hightorque
{

  ProtectionControllerInterface::ProtectionControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr) : BaseControllerInterface(rb_ptr)
  {
    rb_ptr_=rb_ptr;
    quit.store(false);
    set_motor_commomd_type();
    thread_exec_ptr_.reset(new std::thread(&ProtectionControllerInterface::exec_loop, this));
  }

  ProtectionControllerInterface::~ProtectionControllerInterface()
  {
    quit.store(true);
    if(thread_exec_ptr_->joinable())
    {
      thread_exec_ptr_->join();
    }
    send_motor_zero();
    std::cout << "~ProtectionControllerInterface" << std::endl;
  }

  int ProtectionControllerInterface::get_parameters()
  {
    return 0;
  }

  bool ProtectionControllerInterface::get_is_shutdowm()
  {
    return quit.load();
  }



  void ProtectionControllerInterface::set_motor_commomd_type()
  {
    //motor_commomd_type_ = //MotorCommandType::pd_param;
  }

  void ProtectionControllerInterface::motor_pd()
  {
    // if (motor_commomd_type_ != MotorCommandType::pd_param)
    //   return;

    for (int i = 0; i < rb_ptr_->Motors.size(); i++)
    {
      //rb_ptr_->fresh_cmd_dynamic_config(0, 0, 0, 0, 1, i);
      //int map_index = info.params.map_index[0][i];
      rb_ptr_->Motors[i]->fresh_cmd_int16(0, 0, 0, 
                                          0, 0, 1,
                                          0, 0, 0);

    }

    rb_ptr_->motor_send_2();
  }

  void ProtectionControllerInterface::send_motor_zero()
  {
    for (int i = 0; i <  rb_ptr_->Motors.size(); i++)
    {
      //rb_ptr_->fresh_cmd_dynamic_config(0, 0, 0, 0, 0, i);
      rb_ptr_->Motors[i]->fresh_cmd_int16(0, 0, 0, 
                                          0, 0, 1,
                                          0, 0, 0);
    }
    rb_ptr_->motor_send_2();
  }

  void ProtectionControllerInterface::exec_loop()
  {
    ros::Rate rate(100);

    while (!quit&&ros::ok())
    {
      motor_pd();
      rate.sleep();
    }
    ROS_INFO("ProtectionControllerInterface exec loop terminate");
  }

}