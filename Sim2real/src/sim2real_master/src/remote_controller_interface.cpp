#include "sim2real/remote_controller_interface.h"
namespace hightorque
{

     RemoteControllerInterface::RemoteControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr) : BaseControllerInterface(rb_ptr)//初始化参数
     {
          std::string config_path;
          nh.param<std::string>("/sim2real_master_node/dev_config_path", config_path, "~");
          //pd_ptr_.reset(new hightorque::PD(rb_ptr,config_path));

          nh.param<std::string>("/sim2real_master_node/joint_command",joint_command_topic_,"/joint_goals");
          nh.param<std::string>("/sim2real_master_node/motor_command",motor_command_topic_,"/motor_goals");
          joint_command_sub_=nh.subscribe<sensor_msgs::JointState>(joint_command_topic_,10,&RemoteControllerInterface::JointCommandCallback,this);
          motor_command_sub_=nh.subscribe<sensor_msgs::JointState>(motor_command_topic_,10,&RemoteControllerInterface::MotorCommandCallback,this);
          now_jointstate_.position.resize(rb_ptr->Motors.size());
          now_jointstate_.velocity.resize(rb_ptr->Motors.size());
          now_jointstate_.effort.resize(rb_ptr->Motors.size());
          for(int i=0;i<now_jointstate_.position.size();++i)
          {
               now_jointstate_.position[i]=0;
          
               now_jointstate_.velocity[i]=0;
          
               now_jointstate_.effort[i]=0;
          }
          //pd_ptr_->set_ctrl_state(C_DEVELOP_CONTROL_MOTOR);
          //pd_ptr_->set_motor_command(now_jointstate_);
     }
     //msg指针数据复制到now_jointstate_
     void RemoteControllerInterface::CopyJointStateFromMsg(const sensor_msgs::JointState::ConstPtr &msg)
     {
          for(int i=0;i<now_jointstate_.position.size();++i)
          {
               now_jointstate_.position[i]=0;
          
               now_jointstate_.velocity[i]=0;
          
               now_jointstate_.effort[i]=0;
          }
          for(int i=0;i< msg->position.size();++i)
          {
               now_jointstate_.position[i]=msg->position[i];
          }
          for(int i=0;i< msg->velocity.size();++i)
          {
               now_jointstate_.velocity[i]=msg->velocity[i];
          }
          for(int i=0;i< msg->effort.size();++i)
          {
               now_jointstate_.effort[i]=msg->effort[i];
          }
     }
     
     int RemoteControllerInterface::get_parameters()
     {
          return 0;
     }

     bool RemoteControllerInterface::get_is_shutdowm()
     {
          return false; //pd_ptr_->exec_loop_is_terminate();
     }
     //电机命令回调函数
     void RemoteControllerInterface::MotorCommandCallback(const sensor_msgs::JointState::ConstPtr &msg)
     {
          CopyJointStateFromMsg(msg);
          //pd_ptr_->set_ctrl_state(C_DEVELOP_CONTROL_MOTOR);
          //pd_ptr_->set_motor_command(now_jointstate_);
     }
     //关节命令回调函数
     void RemoteControllerInterface::JointCommandCallback(const sensor_msgs::JointState::ConstPtr &msg)
     {
          CopyJointStateFromMsg(msg);
          //pd_ptr_->set_ctrl_state(C_DEVELOP_CONTROL_JOINT);
          //pd_ptr_->set_joint_command(now_jointstate_);
     }
     //析构
     RemoteControllerInterface::~RemoteControllerInterface()
     {
          //pd_ptr_->set_ctrl_state(C_STATE_SHUT_DOWN);
          std::cout << "~RemoteControllerInterface()" << std::endl;
     }
}
