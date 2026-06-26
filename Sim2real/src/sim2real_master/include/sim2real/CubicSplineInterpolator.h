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
#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
// #include <matplot/matplot.h>

class CubicSplineInterpolator {
public:
    // 构造函数，接收离散点p、首尾加速度a_s和a_e以及时间序列t_list
    CubicSplineInterpolator(const std::vector<double>& points, double a_s, double a_e, const std::vector<double>& time_list);
    void calculatePosParameters(void);
    void calculateVelParameters(void);
    void generateInterpolatedPoints(double ts);
    void generateInterpolatedVels(double ts);
    Eigen::VectorXd get_poly_pos_param(void);
    Eigen::VectorXd get_poly_vel_param(void);
    std::vector<double> get_poly_pos_trace(void);
    std::vector<double> get_poly_vel_trace(void);

private:
    std::vector<double> points_;
    std::vector<double> time_list_;
    Eigen::VectorXd poly_pos_param;
    Eigen::VectorXd poly_vel_param;
    std::vector<double> interpolated_points;
    std::vector<double> interpolated_vels;
    double a_start, a_end;
    int N;

    Eigen::VectorXd calculate_diff(Eigen::VectorXd raw_param);
};
