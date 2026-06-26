编译
编译时，需要区分x86和arm架构，使用编译好的libtorch库文件，放置于整个工作区的根目录下(准确来说sim2real_master的cmacklists.txt文件的父路径的父路径)

使用
在部署到arm上时，需要修改sim2real_master/scripts/joy_control.py的第一行,由#!/root/miniconda3/envs/py39cu12/bin/python改为#!/usr/bin/python3
需要修改sim2real_master/launch/joy_control.launch文件内的17,18行，改为存放自定义配置文件的路径和存放开发模式配置件的路径


文件夹说明
custom_config 
机器人处于自定义模式时，用户可以使用hightorque_rl训练出来的policy和对应的参数，放入到custom_config文件夹中
文件夹的组织形式为
custom_config
|--policy
    |--your_policy.pt
    |--your_second_policy.pt
|--config.yaml
|--dynamic_config.yaml

dev_config 
机器人处于开发者模式时，用户可以自定义机器人的kp，kd，命令方向等，参考sim2real的pd_controller的参数的含义
文件组织形式为
dev_config
|--config.yaml
|--dynamic_config.yaml

livelybot_robot
机器人sdk，主要指定了机器人的id和限位，以及imusdk，屏幕的sdk

pai_12dof
包含机器人模型文件，以及机器人的状态可视化脚本

sim2real
最核心的包，把sim2real的功能包装在rl_controller和pd_controller，衍生出默认，自定义，开发模式

sim2real_master
机器人的管理节点，通过手柄控制机器人切换不同模式，并且通过背板屏幕显示当前状态

sim2real_msg
sim2real包中使用到的自定义话题

sim2real_dev
使用这个包和机器人的develop mode，可以实现机器人的远程可视化，关节层面和电机层面的控制
