#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
// #include <tf2/convert.h>
#include <tf2/convert.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.h>  // 必须包含才能支持 tf2::doTransform for Imu
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
// #include <tf2_sensor_msgs/tf2_sensor_msgs.h>  // 需要安装 tf2_sensor_msgs 包
// #include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

namespace tf2_ext
{

    /**
     * @brief Transform sensor_msgs::Imu using a TransformStamped
     *
     * This function applies a rigid body transform to the orientation,
     * angular velocity, and linear acceleration in the IMU message.
     *
     * @param input    The original IMU message
     * @param transform The transform to apply
     * @return sensor_msgs::Imu  The transformed IMU message
     */
    inline sensor_msgs::Imu transformImu(
        const sensor_msgs::Imu& input,
        const geometry_msgs::TransformStamped& transform)
    {
        sensor_msgs::Imu output = input;

        // Transform orientation
        tf2::Quaternion q_input, q_tf, q_result;
        tf2::fromMsg(input.orientation, q_input);
        tf2::fromMsg(transform.transform.rotation, q_tf);
        q_result = q_tf * q_input;
        q_result.normalize();
        output.orientation = tf2::toMsg(q_result);

        // Transform angular velocity
        tf2::Vector3 av_input(input.angular_velocity.x,
                              input.angular_velocity.y,
                              input.angular_velocity.z);
        tf2::Vector3 av_result = tf2::quatRotate(q_tf, av_input);
        output.angular_velocity.x = av_result.x();
        output.angular_velocity.y = av_result.y();
        output.angular_velocity.z = av_result.z();

        // Transform linear acceleration
        tf2::Vector3 la_input(input.linear_acceleration.x,
                              input.linear_acceleration.y,
                              input.linear_acceleration.z);
        tf2::Vector3 la_result = tf2::quatRotate(q_tf, la_input);
        output.linear_acceleration.x = la_result.x();
        output.linear_acceleration.y = la_result.y();
        output.linear_acceleration.z = la_result.z();

        // Set new frame_id
        output.header.frame_id = transform.header.frame_id;

        return output;
    }

} // namespace tf2_ext

class ImuTransformer
{
    public:
        ImuTransformer()
            : tf_listener_(tf_buffer_)
        {
            ros::NodeHandle nh;
            imu_sub_ = nh.subscribe("/imu/data", 10, &ImuTransformer::imuCallback, this);
            imu_pub_ = nh.advertise<sensor_msgs::Imu>("/imu/rotated", 10);
        }

        void imuCallback(const sensor_msgs::Imu::ConstPtr& msg)
        {
            try
            {
                // 查找从原始 frame（如 imu_link）到目标 frame（如 imu_rotated）的变换
                geometry_msgs::TransformStamped transform_stamped = tf_buffer_.lookupTransform("imu_rotated",        // 目标坐标系
                                                                                               msg->header.frame_id, // 源坐标系
                                                                                               msg->header.stamp,    // 对应时间
                                                                                               ros::Duration(0.1)    // 等待时长
                );

                // 使用 tf2 进行变换
                sensor_msgs::Imu transformed_msg = tf2_ext::transformImu(*msg, transform_stamped);
                // tf2::doTransform(*msg, transformed_msg, transform_stamped);

                // 发布变换后的 IMU 数据
                imu_pub_.publish(transformed_msg);
            }
            catch (tf2::TransformException& ex)
            {
                ROS_WARN("Transform failed: %s", ex.what());
            }
        }

    private:
        ros::Subscriber imu_sub_;
        ros::Publisher imu_pub_;
        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "imu_tf_transformer");
    ImuTransformer transformer;
    ros::spin();
    return 0;
}
