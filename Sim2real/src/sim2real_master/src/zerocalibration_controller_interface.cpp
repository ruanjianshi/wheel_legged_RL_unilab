#include "sim2real/zerocalibration_controller_interface.h"

namespace hightorque
{

    ZeroCalibrationControllerInterface::ZeroCalibrationControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr) : BaseControllerInterface(rb_ptr), rb_ptr_(rb_ptr)
    {
        thread_exec_ptr_.reset(new std::thread(&ZeroCalibrationControllerInterface::reset_zero, this));
        status = 0;
    }

    void ZeroCalibrationControllerInterface::reset_zero()
    {
        ROS_INFO("Starting zero calibration...");

        // 调用机器人对象的零位重置函数
        rb_ptr_->set_reset_zero(); // 全部电机重置零位

        // 设置一个超时时间，防止无限循环
        ros::Time start_time = ros::Time::now();
        ros::Duration timeout(10.0); // 超时时间，例如10秒

        ros::Rate rate(10); // 控制循环频率

        while (!quit.load() && ros::ok())
        {
            // 检查是否超时
            if (ros::Time::now() - start_time > timeout)
            {
                ROS_WARN("Zero calibration timed out.");
                status = 2;
                break;
            }

            // 检查每个电机是否完成校准
            bool all_motors_calibrated = true;
            int num = 0;
            for (motor* m : rb_ptr_->Motors)
            {
                double position = m->get_current_motor_state()->position;
                double velocity = m->get_current_motor_state()->velocity;
                double torque = m->get_current_motor_state()->torque;

                // 打印电机状态（可选）
                printf("Motors[%2.0d]: pos %f, vel %f, tqe %f\n", num++, position, velocity, torque);
                rb_ptr_->send_get_motor_state_cmd();
                // 在这里您可以添加判断条件，确定电机是否完成校准
                // 例如，如果每个电机的位置达到了预期值，认为校准完成
                if (!motor_calibrated(m))
                {
                    all_motors_calibrated = false;
                }
            }
            for (auto motor : rb_ptr_->Motors)
            {
                motor->fresh_cmd_int16(0, 0, 0,
                                       0, 0, 0,
                                       0, 0, 0);
            }
            rb_ptr_->motor_send_2();
            if (all_motors_calibrated)
            {
                ROS_INFO("Zero calibration completed successfully.");
                status = 1;
            }

            rate.sleep();
        }
    }

    bool ZeroCalibrationControllerInterface::motor_calibrated(motor* m)
    {
        // 获取电机的位置
        double position = m->get_current_motor_state()->position;

        // 判断位置的绝对值是否小于等于 0.01
        if (std::abs(position) <= 0.01)
        {
            return true; // 该电机校准成功
        }
        else
        {
            return false; // 该电机尚未校准成功
        }
    }

    int ZeroCalibrationControllerInterface::get_parameters()
    {
        return status;
    }

    bool ZeroCalibrationControllerInterface::get_is_shutdowm()
    {
        switch (status)
        {
            case 0:
                quit.store(false);
                break;
            case 1:
            case 2:
                quit.store(true);
                break;
        }
        return quit.load();
    }

    ZeroCalibrationControllerInterface::~ZeroCalibrationControllerInterface()
    {
        if (thread_exec_ptr_->joinable())
        {
            thread_exec_ptr_->join();
        }
        std::cout << "~ZeroCalibrationControllerInterface()" << std::endl;
    }
}
