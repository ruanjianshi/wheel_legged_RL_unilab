/*
 * @file add_waypoint.cpp
 * @author 唐 <707567735@qq.com>
 * @brief 4-3-4多项式轨迹插值
 * @version 2.0
 * @date 2023-11-28
 *
 * @copyright Copyright (c) 2023 你的公司或组织名称
 *
 * 这里可以添加更详细的描述或版权声明
 */

#include "sim2real/CubicSplineInterpolator.h"

// 构造函数，接收离散点p、首尾加速度a_s和a_e以及时间序列t_list
CubicSplineInterpolator::CubicSplineInterpolator(const std::vector<double>& points, double a_s, double a_e, const std::vector<double>& time_list)
    : points_(points), a_start(a_s), a_end(a_e), time_list_(time_list)
{
    //if (points_.size() != time_list_.size())
    //{

    //}
    N = points_.size();
}

// 计算4-3-4多项式插值所需的参数向量
void CubicSplineInterpolator::calculatePosParameters(void) 
{
    int N = points_.size();
    int param_nums = 4 * (N-3) + 5 * 2;
    Eigen::VectorXd B = Eigen::VectorXd::Zero(param_nums);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(param_nums, param_nums);

    B.segment(0, 3) << points_[0], 0, a_start;
    B.segment(param_nums-3, 3) << points_[N-1], 0, a_end;
    for (int i = 1; i < N-1; ++i) 
    {
        B.segment(i*4-1, 4) << points_[i], points_[i], 0, 0;
    }
    A.block(0, 0, 3, 5) << 0, 0, 0, 0, 1,
        0, 0, 0, 1, 0,
        0, 0, 2, 0, 0;

    double T = time_list_[1]-time_list_[0];
    A.block(3, 0, 4, 9) << T * T * T * T, T* T* T, T* T, T, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1,
        4 * T * T * T, 3 * T * T, 2 * T, 1, 0, 0, 0, -1, 0,
        12 * T * T, 6 * T, 2, 0, 0, 0, -2, 0, 0;
    T = time_list_[N-2]-time_list_[N-3];
    A.block(param_nums-7, param_nums-9, 4, 9) << T * T * T, T* T, T, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1,
        3 * T * T, 2 * T, 1, 0, 0, 0, 0, -1, 0,
        6 * T, 2, 0, 0, 0, 0, -2, 0, 0;
    T = time_list_[N-1]-time_list_[N-2];
    A.block(param_nums-3, param_nums-5, 3, 5) << T * T * T * T, T* T* T, T* T, T, 1,
        4 * T * T * T, 3 * T * T, 2 * T, 1, 0,
        12 * T * T, 6 * T, 2, 0, 0;
    for (int i = 3; i <= N-2; ++i) {
        T = time_list_[i-1]-time_list_[i-2];
        A.block(3 + 4 * (i-2), 5 + 4 * (i-3), 4, 8) << T * T * T, T* T, T, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1,
            3 * T * T, 2 * T, 1, 0, 0, 0, -1, 0,
            6 * T, 2, 0, 0, 0, -2, 0, 0;
    }

    poly_pos_param = A.colPivHouseholderQr().solve(B);
}

void CubicSplineInterpolator::calculateVelParameters(void) 
{
    int N = points_.size();
    int param_nums = 3 * (N - 3) + 4 * 2;
    poly_vel_param = Eigen::VectorXd::Zero(param_nums);
    for (int i = 0; i < N - 1; i++)
    {
        if (i == 0)
        {
            poly_vel_param.segment(0, 4) = calculate_diff(poly_pos_param.segment(0,5));
        }
        else if(i == N-2)
        {
            poly_vel_param.segment((i - 1) * 3 + 4, 4) = calculate_diff(poly_pos_param.segment((i - 1) * 4 + 5, 5));
        }
        else
        {
            poly_vel_param.segment((i - 1) * 3 + 4, 3) = calculate_diff(poly_pos_param.segment((i - 1) * 4 + 5, 4));
        }
    }
    
}

// 根据参数向量生成插值轨迹点
void CubicSplineInterpolator::generateInterpolatedPoints(double ts)
{
    //Eigen::VectorXd parameters = calculateParameters();
    int param_nums = poly_pos_param.rows();
    double T = time_list_[time_list_.size()-1];
    double num_T = (time_list_.size()-1) * T / ts; 
    Eigen::VectorXd temp;
    interpolated_points.clear();
    //Eigen::VectorXd powers;
    for (double t = 0; t <= T; t += ts)
    {
        int n = 1;
        while (t > time_list_[n])
        {
            n++;
        }
        if (t <= time_list_[1])
        {
            // ...（第一个区间插值计算）
            //powers = Eigen::VectorXd(pow(t, 4), pow(t, 3), pow(t, 2), t, 1);
            Eigen::VectorXd powers(5);
            powers << pow(t, 4), pow(t, 3), pow(t, 2), t, 1;
            temp = Eigen::VectorXd(poly_pos_param.block(0, 0, 5, 1));
            interpolated_points.push_back(temp.dot(powers));
        }
        else if (t > time_list_[time_list_.size()-2])
        {
            // ...（最后一个区间插值计算）
            double dt = (t-time_list_[N-2]);
            Eigen::VectorXd powers(5);
            powers << pow(dt, 4), pow(dt, 3), pow(dt, 2), dt, 1;
            temp = Eigen::VectorXd(poly_pos_param.block(param_nums-5, 0, 5, 1));
            interpolated_points.push_back(temp.dot(powers));
        }
        else
        {
            // ...（中间区间插值计算）
            double dt = t-time_list_[n-1];
            Eigen::VectorXd powers(4);
            powers << pow(dt, 3), pow(dt, 2), dt, 1;
            temp = Eigen::VectorXd(poly_pos_param.block(4 * (n-1) + 1, 0, 4, 1));
            interpolated_points.push_back(temp.dot(powers));
        }
    }
}

void CubicSplineInterpolator::generateInterpolatedVels(double ts)
{
    int param_nums = poly_vel_param.rows();
    double T = time_list_[time_list_.size() - 1];
    double num_T = (time_list_.size() - 1) * T / ts;
    Eigen::VectorXd temp;
    interpolated_vels.clear();
    for (double t = 0; t <= T; t += ts)
    {
        int n = 1;
        while (t > time_list_[n])
        {
            n++;
        }
        if (t <= time_list_[1])
        {
            // ...（第一个区间插值计算）
            //powers = Eigen::VectorXd(pow(t, 3), pow(t, 2), t, 1);
            Eigen::VectorXd powers(4);
            powers << pow(t, 3), pow(t, 2), t, 1;
            temp = Eigen::VectorXd(poly_vel_param.block(0, 0, 4, 1));
            auto v = temp.dot(powers);
            v = std::isnan(v) ? 1.0 : v;
            interpolated_vels.push_back(v);
        }
        else if (t > time_list_[time_list_.size() - 2])
        {
            // ...（最后一个区间插值计算）
            double dt = (t - time_list_[N - 2]);
            Eigen::VectorXd powers(4);
            powers << pow(dt, 3), pow(dt, 2), dt, 1;
            temp = Eigen::VectorXd(poly_vel_param.block(param_nums - 4, 0, 4, 1));
            interpolated_vels.push_back(temp.dot(powers));
        }
        else
        {
            // ...（中间区间插值计算）
            double dt = t - time_list_[n - 1];
            Eigen::VectorXd powers(3);
            powers << pow(dt, 2), dt, 1;
            temp = Eigen::VectorXd(poly_vel_param.block(3 * (n - 1) + 1, 0, 3, 1));
            interpolated_vels.push_back(temp.dot(powers));
        }
    }
}

Eigen::VectorXd CubicSplineInterpolator::get_poly_pos_param(void)
{
    return poly_pos_param;
}

Eigen::VectorXd CubicSplineInterpolator::get_poly_vel_param(void)
{
    return poly_vel_param;
}

std::vector<double> CubicSplineInterpolator::get_poly_pos_trace(void)
{
    return interpolated_points;
}

std::vector<double> CubicSplineInterpolator::get_poly_vel_trace(void)
{
    return interpolated_vels;
}

Eigen::VectorXd CubicSplineInterpolator::calculate_diff(Eigen::VectorXd raw_param)
{
    Eigen::VectorXd diff_param;
    Eigen::Matrix<double,5,5> diff_matrix;
    diff_matrix << 4, 0, 0, 0, 0,
        0, 3, 0, 0, 0,
        0, 0, 2, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0;
    int dim = raw_param.rows();
    
    //switch (dim)
    //{
    //case 5:
    //    diff_param = diff_matrix * raw_param;
    //    break;
    //case 4:
    //    diff_param = diff_matrix.block(1, 1, 4, 4) * raw_param;
    //    break;
    //case 3:
    //    diff_param = diff_matrix.block(2, 2, 3, 3) * raw_param;
    //    break;
    //default:
    //    diff_param = Eigen::VectorXd::Zero(dim);
    //    break;
    //}
    diff_param = diff_matrix.block(5 - dim, 5 - dim, dim, dim) * raw_param;
    // std::cout << diff_param << std::endl;
    return diff_param.segment(0,dim-1);
}
