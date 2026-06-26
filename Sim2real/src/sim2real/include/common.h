#ifndef COMMON_H_
#define COMMON_H_

#include "robot_data.h"
#include <cmath>
#include <filesystem>
#include <string>
#include </usr/include/eigen3/Eigen/Dense>

namespace hightorque
{
    namespace common
    {
        // ======================== Transition 插值函数区域 ========================
        // 提供多种数值平滑过渡函数，用于控制信号、动画、位置插值等应用场景。
        // 包含线性插值（Linear）、缓入（Ease-In）、缓出（Ease-Out）、平滑过渡（SmoothStep）等常用函数。

        /**
         * @brief 位置线性插值过渡（Linear Transition）
         * @param startValue 当前起始值
         * @param targetValue  目标结束值
         * @param startTime    过渡开始时间（单位：秒）
         * @param currentTime  当前时间（单位：秒）
         * @param duration     过渡持续时间（单位：秒）
         * @return 当前时间点插值计算结果（线性增长）
         */
        double linearTransitionPosition(double startValue, double targetValue, double startTime, double currentTime, double duration);

        /**
         * @brief 速度线性插值过渡（Linear Transition）
         * @param startValue 当前起始值
         * @param targetValue  目标结束值
         * @param startTime    过渡开始时间（单位：秒）
         * @param currentTime  当前时间（单位：秒）
         * @param duration     过渡持续时间（单位：秒）
         * @return 当前时间点插值计算结果（线性增长）
         */
        double linearTransitionVelocity(double startValue, double targetValue, double startTime, double currentTime, double duration);

        /**
         * @brief 位置缓入插值过渡（Ease-In Transition）
         *        初期缓慢，后期加速的插值曲线（用于模拟自然的启动过程）
         * @param startValue 当前起始值
         * @param targetValue  目标结束值
         * @param startTime    过渡开始时间
         * @param currentTime  当前时间
         * @param duration     过渡持续时间
         * @return 当前时间点插值结果（缓慢起步后加速）
         */
        double easeInTransitionPosition(double startValue, double targetValue, double startTime, double currentTime, double duration);
        /**
         * @brief 速度缓入插值过渡（Ease-In Transition）
         *        初期缓慢，后期加速的插值曲线（用于模拟自然的启动过程）
         * @param startValue 当前起始值
         * @param targetValue  目标结束值
         * @param startTime    过渡开始时间
         * @param currentTime  当前时间
         * @param duration     过渡持续时间
         * @return 当前时间点插值结果（缓慢起步后加速）
         */
        double easeInTransitionVelocity(double startValue, double targetValue, double startTime, double currentTime, double duration);

        /**
         * @brief 位置缓出插值过渡（Ease-Out Transition）
         *        初期快速，后期减速的插值曲线（适合结束阶段的自然减速）
         * @param startValue 当前起始值
         * @param targetValue  目标结束值
         * @param startTime    过渡开始时间
         * @param currentTime  当前时间
         * @param duration     过渡持续时间
         * @return 当前时间点插值结果（快速起步后减速）
         */
        double easeOutTransitionPosition(double startValue, double targetValue, double startTime, double currentTime, double duration);
        /**
         * @brief 速度缓出插值过渡（Ease-Out Transition）
         *        初期快速，后期减速的插值曲线（适合结束阶段的自然减速）
         * @param startValue 当前起始值
         * @param targetValue  目标结束值
         * @param startTime    过渡开始时间
         * @param currentTime  当前时间
         * @param duration     过渡持续时间
         * @return 当前时间点插值结果（快速起步后减速）
         */
        double easeOutTransitionVelocity(double startValue, double targetValue, double startTime, double currentTime, double duration);

        /**
         * @brief 位置平滑插值过渡（SmoothStep）
         *        一个自然平滑的 S 型过渡曲线，两端缓中间快
         * @param startValue 当前起始值
         * @param targetValue  目标结束值
         * @param startTime    过渡开始时间
         * @param currentTime  当前时间
         * @param duration     过渡持续时间
         * @return 当前时间点插值结果（两端缓中间快）
         */
        double smoothStepTransitionPosition(double startValue, double targetValue, double startTime, double currentTime, double duration);
        /**
         * @brief 速度平滑插值过渡（SmoothStep）
         *        一个自然平滑的 S 型过渡曲线，两端缓中间快
         * @param startValue 当前起始值
         * @param targetValue  目标结束值
         * @param startTime    过渡开始时间
         * @param currentTime  当前时间
         * @param duration     过渡持续时间
         * @return 当前时间点插值结果（两端缓中间快）
         */
        double smoothStepTransitionVelocity(double startValue, double targetValue, double startTime, double currentTime, double duration);

        // ======================== Transition 插值函数区域结束 ========================

        // ======================== 欧拉角与四元数转换函数区域 ========================
        // 提供 ZYX 顺序下的欧拉角（roll, pitch, yaw）与四元数之间的转换函数。
        // 用于姿态处理、坐标变换等运动学相关模块。

        /**
         * @brief 将四元数转换为 ZYX 顺序的欧拉角（roll, pitch, yaw）
         *
         * @param q 输入四元数
         * @param normalizeAngle 是否对欧拉角归一化到 [-π, π] 区间（默认 true）
         * @return Eigen::Vector3d 对应欧拉角：roll, pitch, yaw
         */
        Eigen::Vector3d quatToEulerZYX(const Eigen::Quaterniond& q, bool normalizeAngle = true);
        /**
         * @brief 将四元数的四个分量直接转换为 ZYX 顺序的欧拉角
         *
         * @param x 四元数 x 分量
         * @param y 四元数 y 分量
         * @param z 四元数 z 分量
         * @param w 四元数 w 分量
         * @param normalizeAngle 是否对欧拉角归一化到 [-π, π] 区间（默认 true）
         * @return Eigen::Vector3d 对应欧拉角：roll, pitch, yaw
         */
        Eigen::Vector3d quatToEulerZYX(const double x, const double y, double z, double w, bool normalizeAngle = true);
        /**
         * @brief 将 ZYX 欧拉角（roll, pitch, yaw）转换为四元数
         *
         * @param euler 欧拉角向量，顺序为 roll, pitch, yaw
         * @return Eigen::Quaterniond 对应四元数
         */
        Eigen::Quaterniond eulerZYXToQuat(const Eigen::Vector3d& euler);
        /**
         * @brief 将单独的 roll, pitch, yaw 值转换为四元数
         *
         * @param roll 绕 X 轴旋转角度
         * @param pitch 绕 Y 轴旋转角度
         * @param yaw 绕 Z 轴旋转角度
         * @return Eigen::Quaterniond 对应四元数
         */
        Eigen::Quaterniond eulerZYXToQuat(double roll, double pitch, double yaw);
        // ======================== 欧拉角与四元数转换函数区域结束 ========================

        // ======================== 四元数旋转向量函数区域 ========================
        // 提供将 3D 向量 v 通过四元数 q 进行旋转的计算方式。
        // 支持 Eigen::Vector4d 形式的四元数和 Eigen::Quaterniond 格式。

        /**
         * @brief 使用四元数（以 Vector4d 表示）旋转向量
         *
         * 使用显式数学展开公式进行旋转（无需构造 Quaternion 对象）。
         * 输入四元数必须为单位四元数，格式为 (x, y, z, w)。
         *
         * @param q 四元数，格式为 Eigen::Vector4d（x, y, z, w）
         * @param v 待旋转的向量
         * @return Eigen::Vector3d 旋转后的向量
         */
        Eigen::Vector3d rotateVectorByQuatVec4(const Eigen::Vector4d& q, const Eigen::Vector3d& v);
        /**
         * @brief 使用 Eigen::Quaterniond 旋转向量
         *
         * 实际计算为 q.conjugate() * v，默认表示 v 在全局坐标系中，
         * 通过 q 的共轭对其变换至局部坐标系。
         *
         * @param q 四元数（Eigen::Quaterniond 类型）
         * @param v 待旋转向量
         * @return Eigen::Vector3d 旋转后的向量
         */
        Eigen::Vector3d rotateVectorByQuat(const Eigen::Quaterniond& q, const Eigen::Vector3d& v);
        // ======================== 四元数旋转向量函数区域 ========================

        // ======================== 状态与输出转换函数区域 ========================
        /**
         * @brief 单通道转换：机器人输出到电机输出
         * @param r        单通道机器人输出
         * @param m        单通道电机输出（输出参数）
         * @param offset   偏移值
         * @param direction 方向（1 或 -1）
         */

        void transformRobotOutputToMotorOutput(const float& r,
                                               float& m,
                                               const double& offset,
                                               const int direction);

        /**
         * @brief 批量转换：机器人输出到电机输出
         * @param r       机器人输出对象
         * @param m       电机输出对象（输出参数）
         * @param offset  各通道偏移列表
         * @param direction 各通道方向列表
         * @param special  0为普通情况 1为hi的脚踝特殊
         */
        void transformRobotOutputToMotorOutput(Robot_Output& r,
                                               Motor_Output& m,
                                               const std::vector<double>& offset,
                                               const std::vector<int> direction,
                                               int special = 0);

        /**
         * @brief 指定索引转换：机器人输出到电机输出
         * @param r        机器人输出对象
         * @param m        电机输出对象（输出参数）
         * @param offset   各通道偏移列表
         * @param direction 各通道方向列表
         * @param index    映射索引列表
         * @param special  0为普通情况 1为hi的脚踝特殊
         */
        void transformRobotOutputToMotorOutput(Robot_Output& r,
                                               Motor_Output& m,
                                               const std::vector<double>& offset,
                                               const std::vector<int> direction,
                                               const std::vector<int>& index,
                                               int special = 0);

        /**
         * @brief 单通道转换：电机状态到机器人状态
         * @param m        单通道电机状态
         * @param r        单通道机器人状态（输出参数）
         * @param offset   偏移值
         * @param direction 方向（1 或 -1）
         * @param special  0为普通情况 1为hi的脚踝特殊
         */
        void transformMotorStateToRobotState(Motor_State& m,
                                             Robot_State& r,
                                             const std::vector<double>& offset,
                                             const std::vector<int> direction,
                                             int special = 0);

        /**
         * @brief 指定索引转换：电机状态到机器人状态
         * @param m        电机状态对象
         * @param r        机器人状态对象（输出参数）
         * @param offset   各通道偏移列表
         * @param direction 各通道方向列表
         * @param index    映射索引列表
         * @param special  0为普通情况 1为hi的脚踝特殊
         */
        void transformMotorStateToRobotState(Motor_State& m,
                                             Robot_State& r,
                                             const std::vector<double>& offset,
                                             const std::vector<int> direction,
                                             const std::vector<int>& index,
                                             int special = 0);

        void transformPolicyRobotOutputToActualRobotOutput(Robot_Output& p,
                                                           Robot_Output& a,
                                                           std::vector<int> map);

        void transformActualRobotStateToPolicyRobotState(Robot_State& a,
                                                         Robot_State& p,
                                                         std::vector<int> map);

        void transformActualRobotOutputToPolicyRobotOutput(Robot_Output& a,
                                                           Robot_Output& p,
                                                           std::vector<int> map);

        /**
         * @brief 用于hi的脚踝转换
         * @param motorPitch 电机
         * @param motorRoll  电机
         * @param robotPitch 机器人
         * @param robotRoll  机器人
         */
        void ankleKinematics(double motorPitch, double motorRoll, double& robotPitch, double& robotRoll);

        /**
         * @brief 用于hi的脚踝转换
         * @param robotPitch 机器人
         * @param robotRoll  机器人
         * @param motorPitch 电机
         * @param motorRoll  电机
         */
        void ankleInverseKinematics(double robotPitch, double robotRoll, double& motorPitch, double& motorRoll);

        /**
         * @brief 将 Motor_Output 转为 Motor_State
         * @param out 电机输出
         * @return    对应电机状态
         */
        Motor_State toState(const Motor_Output& out);

        /**
         * @brief 将 Robot_Output 转为 Robot_State
         * @param out 机器人输出
         * @return    对应机器人状态
         */
        Robot_State toState(const Robot_Output& out);

        /**
         * @brief 将 Motor_State 转为 Motor_Output
         * @param state 电机状态
         * @return      对应电机输出
         */
        Motor_Output toOut(const Motor_State& state);

        /**
         * @brief 将 Robot_State 转为 Robot_Output
         * @param state 机器人状态
         * @return      对应机器人输出
         */
        Robot_Output toOut(const Robot_State& state);

        /**
         * @brief 将 Eigen::VectorXd 转为 std::vector<double>
         * @param v Eigen 向量
         * @return  std::vector<double> 结果
         */
        inline std::vector<double> toStdVector(const Eigen::VectorXd& v)
        {
            return std::vector<double>(v.data(), v.data() + v.size());
        }

        /**
         * @brief 将二维点列表转换为 Motor_Output 向量列表
         * @param pointsVec 原始二维点列表
         * @return           Motor_Output 列表
         */
        std::vector<Motor_Output> toMotorOutputVector(const std::vector<std::vector<double>> pointsVec);

        // ======================== 状态与输出转换函数区域 ========================

        // ======================== 相似性判断函数区域 ========================

        /**
         * @brief 判断两个电机状态是否接近
         * @param s1   状态 1
         * @param s2   状态 2
         * @param diff 容差（默认 0.05）
         * @return     true 表示两状态差值小于容差
         */
        bool isMotorStateClose(const Motor_State& s1,
                               const Motor_State& s2,
                               double diff = 0.05);

        /**
         * @brief 判断两个机器人状态是否接近
         * @param s1   状态 1
         * @param s2   状态 2
         * @param diff 容差（默认 0.05）
         * @return     true 表示两状态差值小于容差
         */
        bool isRobotStateClose(const Robot_State& s1,
                               const Robot_State& s2,
                               double diff = 0.05);

        /**
         * @brief 判断两个电机输出是否接近
         * @param o1   输出 1
         * @param o2   输出 2
         * @param diff 容差（默认 0.05）
         * @return     true 表示两输出差值小于容差
         */
        bool isMotorOutputClose(const Motor_Output& o1,
                                const Motor_Output& o2,
                                double diff = 0.05);

        /**
         * @brief 判断两个机器人输出是否接近
         * @param o1   输出 1
         * @param o2   输出 2
         * @param diff 容差（默认 0.05）
         * @return     true 表示两输出差值小于容差
         */
        bool isRobotOutputClose(const Robot_Output& o1,
                                const Robot_Output& o2,
                                double diff = 0.05);

        // ======================== 相似性判断函数区域 ========================

        // ======================== Motor_Output 设置函数区域 ========================

        /**
         * @brief 设置电机输出的 PID 参数（float 版）
         */
        void motorOutputSetKpKd(Motor_Output& m,
                                const std::vector<float> kp,
                                const std::vector<float> kd);

        /**
         * @brief 设置电机输出的 PID 参数（double 版）
         */
        void motorOutputSetKpKd(Motor_Output& m,
                                const std::vector<double> kp,
                                const std::vector<double> kd);

        /**
         * @brief 设置电机输出的扭矩（float 版）
         */
        void motorOutputSetTorque(Motor_Output& m,
                                  const std::vector<float> torque);

        /**
         * @brief 设置电机输出的扭矩（double 版）
         */
        void motorOutputSetTorque(Motor_Output& m,
                                  const std::vector<double> torque);

        /**
         * @brief 设置电机输出的速度（float 版）
         */
        void motorOutputSetVelcity(Motor_Output& m,
                                   const std::vector<float> velocity);

        /**
         * @brief 设置电机输出的速度（double 版）
         */
        void motorOutputSetVelcity(Motor_Output& m,
                                   const std::vector<double> velocity);

        /**
         * @brief 设置电机输出的 PID 参数并按 mapIndex 映射（float 版）
         */
        void motorOutputSetKpKd(Motor_Output& m,
                                const std::vector<float> kp,
                                const std::vector<float> kd,
                                const std::vector<int> mapIndex);

        /**
         * @brief 设置电机输出的 PID 参数并按 mapIndex 映射（double 版）
         */
        void motorOutputSetKpKd(Motor_Output& m,
                                const std::vector<double> kp,
                                const std::vector<double> kd,
                                const std::vector<int> mapIndex);

        // ======================== Motor_Output 设置函数区域 ========================

        /**
         * @brief 将二维点列表转为字符串（辅助调试）
         * @param mat 二维点列表
         * @return    格式化字符串
         */
        std::string toString(std::vector<std::vector<double>> mat);

        /**
         * @brief 将一维点列表转为字符串（辅助调试）
         * @param vec 一维点列表
         * @return    格式化字符串
         */
        std::string toString(const std::vector<double>& vec);

        template <typename T>
        std::string toString(const std::vector<T>& vec)
        {
            std::stringstream ss;
            for (size_t i = 0; i < vec.size(); ++i)
            {
                ss << vec[i];
                if (i < vec.size() - 1)
                {
                    ss << ", ";
                }
            }
            ss << "\n";
            return ss.str();
        }

        /**
         * @brief 旋转矩阵（二维点集）90°（行列互换）
         * @param matrix 输入矩阵
         * @return       旋转后的矩阵
         */
        std::vector<std::vector<double>> rotateMatrix(const std::vector<std::vector<double>>& matrix);

        /**
         * @brief 从文件流中加载二进制数据
         * @param file   已打开的文件指针
         * @param offset 偏移字节数
         * @param size   读取字节数
         * @return       指向读取数据的缓冲区指针
         */
        unsigned char* loadData(FILE* file, size_t offset, size_t size);

        /**
         * @brief 读取整个文件到内存
         * @param filename 文件路径
         * @param sizeOut  输出文件大小指针
         * @return         指向读取数据的缓冲区指针
         */
        unsigned char* readFileData(const char* filename, int* sizeOut);

        /**
         * @brief 使用 Boost 序列化加载向量数据
         * @param filename 序列化文件路径
         * @return         二维点列表
         */
        std::vector<std::vector<double>> loadVectorWithBoost(const std::string& filename);

        /**
         * @brief 保存矩阵数据并附带一段元信息（metadata）
         * @param filePath 目标文件路径
         * @param metadata 要保存的字符串元信息
         * @param data     要保存的二维 double 矩阵
         * @ return 成功返回 true，失败返回 false
         */
        bool saveMatrixWithMetadata(const std::filesystem::path& filePath, const std::string& metadata,
                                    const std::vector<std::vector<double>>& data);

        /**
         * @brief 从文件中读取元信息和矩阵数据
         * @param filePath 目标文件路径
         * @param outMetadata  输出读取到的字符串元信息
         * @param outData      输出读取到的二维 double 矩阵
         * @return 成功返回 true，失败返回 false
         */
        bool loadMatrixWithMetadata(const std::filesystem::path& filePath, std::string& outMetadata,
                                    std::vector<std::vector<double>>& outData);

        void saveOutputToFile(const std::string& filename, const Robot_Output& output);
        void saveVectorXdToFile(const std::string& filename, const Eigen::VectorXd& observation);
        void saveMatrixXdToFile(const std::string& filename, const Eigen::MatrixXd& input);
        void saveStringToFile(const std::string& filename, const std::string& str);

        std::string getSystemInfo();
        /**
         * @brief 处理后缀字符串，统一格式（去掉前缀的'.'，添加统一的'.'前缀）
         * @param suffix 原始后缀（可能带'.'或不带）
         * @return 标准化后的后缀（格式为".xxx"，空后缀返回空字符串）
         */
        std::string processSuffix(const std::string& suffix);
        /**
         * @brief 给文件路径替换或添加后缀
         * @param path 原始文件路径
         * @param suffix 目标后缀（可带'.'或不带，如"trt"或".trt"）
         * @return 处理后的文件路径
         */
        std::string replaceOrAddSuffix(const std::string& path, const std::string& suffix);

        inline double calculateScaleFactor(double val, double min, double max)
        {
            double scale = 1.0;
            if (val > max)
            {
                // 超过上限：缩放因子 = 当前值 / 上限（>1）
                scale = val / max;
            }
            else if (val < min)
            {
                // 低于下限：缩放因子 = 当前值 / 下限（<1，绝对值>1）
                scale = val / min;
            }
            // 注意：取绝对值（因为下限是负数时，scale为负，绝对值才是超限比例）
            return std::abs(scale);
        }

    }
}
#endif
