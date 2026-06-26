#ifndef DEFAULT_CONTROLLER_H_
#define DEFAULT_CONTROLLER_H_

#include "sim2real.h"

namespace hightorque
{
    /**
     * @brief 默认控制节点：继承 Sim2Real，提供基础控制逻辑
     * 跌倒爬起，示教动作（轨迹）
     */
    class DefaultControllerNode : public Sim2Real
    {
        public:
            /** ======================== 构造与析构 ======================== */

            /**
             * @brief 构造函数：初始化串口机器人和 ROS 节点
             * @param rbPtr 串口机器人对象指针
             * @param nh    ROS 节点句柄指针
             */
            DefaultControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr,
                                  std::shared_ptr<ros::NodeHandle> nh);

            /**
             * @brief 构造函数：初始化串口机器人和 ROS 节点
             * @param rbPtr 串口机器人对象指针
             * @param nh    ROS 节点句柄指针
             * @param path  配置文件名称
             */
            DefaultControllerNode(std::shared_ptr<livelybot_serial::robot> rbPtr,
                                  std::shared_ptr<ros::NodeHandle> nh,
                                  const std::string& path);

            /**
             * @brief 析构函数：清理资源
             */
            ~DefaultControllerNode();

            /** ======================== 覆写接口 ======================== */

            /**
             * @brief 检查是否请求退出（覆盖 Sim2Real）
             * @return true 表示应退出主循环
             */
            // bool get_is_shutdowm() override
            // {
            //     return quit_; // 退出标志
            // }

            /**
             * @brief 遥控输入回调（覆盖 Sim2Real）
             * @param msg 控制状态消息指针
             */
            void joyCallback(const sim2real_msg::Joy::ConstPtr& msg) override;

            /**
             * @brief 启动控制流程（覆盖 Sim2Real）
             * @return true 表示启动成功
             */
            // bool start() override;

            bool prePolicyChangeUpperBody();

            void playAudio();

        private:

            // double kickDuration_ = 0.2; // 踢腿动作执行时间
            bool useHostUpUp_ = false; // 是否使用 host 的跌倒爬起动作

            // audio
            std::thread audioThread_;  // 播放音频线程
            std::atomic<bool> isPlaying_{false};
            bool audioQuit_ = false; // 音频播放线程退出标志
            std::string audioFile_ = "/home/hightorque/pi+.mp3"; // 音频文件路径
    };

};
#endif
