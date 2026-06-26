#include "sim2real/teach_controller_interface.h"
#include "livelybot_logger/logger_interface.h"

namespace hightorque
{

    TeachControllerInterface::TeachControllerInterface(std::shared_ptr<livelybot_serial::robot> rb_ptr)
        : BaseControllerInterface(rb_ptr), nh(), rb_ptr_(rb_ptr), run_mode(0), cnt(0), sample_stop(0),
          lock_type(3), leg_cnt(0)
    {
        quit.store(false);
         // 获取包的路径
        std::string package_path = ros::package::getPath("sim2real_master");

        // 构建文件路径
        waypoint_save_filename = package_path + "/way_point/way_point.boost";
        waypoint_load_filename = package_path + "/way_point/way_point.boost";
        way_point_back_filename = package_path + "/way_point/way_point.boost";
        thread_exec_ptr_.reset(new std::thread(&TeachControllerInterface::teaching, this));
    }

    void TeachControllerInterface::teaching()
    {
        ros::Rate r(100);

        // joy_subscriber_ = nh.subscribe<sensor_msgs::Joy>("/joy", 10, &TeachControllerInterface::joyCallback, this);
        waypoint_control_subscriber_ = nh.subscribe<sim2real_msg::WayPoint>("/waypoint_control", 10, &TeachControllerInterface::waypointCallback, this);
        // keyboard_subscriber_ = nh.subscribe<std_msgs::String>("/keyboard_input", 10, &TeachControllerInterface::keyboardCallback, this);

        ROS_INFO("\033[1;32mSTART\033[0m");

        ROS_INFO("-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace<-------");
        while(!quit.load() && ros::ok())
        {
            switch(run_mode)
            {
                // 模式0， 无输出
                case 0:
                {
                    for(auto m:rb_ptr_->Motors)
                    {
                        m->pos_vel_tqe_kp_kd(0, 0, 0, 0.0, 0.0);
                    }
                    rb_ptr_->motor_send_2();
                }
                    break;
                // 示教模式
                case 1:
                {
                    if(sample_stop == 0)
                    {
                        for(auto m:rb_ptr_->Motors)
                        {
                            m->pos_vel_tqe_kp_kd(0, 0, 0, 0.3, 0.5);
                        }
                        rb_ptr_->motor_send_2();
                    }
                    else
                    {                    
                        auto stop_it = stop_pos.begin();
                        leg_cnt = 0;
                        for(auto m:rb_ptr_->Motors)
                        {
                            if(leg_cnt < 6)
                            {   
                                if(lock_type & 0x01)
                                {
                                    m->pos_vel_MAXtqe(*stop_it, 0, 10);
                                }
                                else
                                {
                                    m->pos_vel_tqe_kp_kd(0, 0, 0, 0.3, 0.5);
                                }
                            }
                            else if(leg_cnt < 12)
                            {
                                if(lock_type & 0x02)
                                {
                                    m->pos_vel_MAXtqe(*stop_it, 0, 10);
                                }
                                else
                                {
                                    m->pos_vel_tqe_kp_kd(0, 0, 0, 0.3, 0.5);
                                }
                            }
                            stop_it ++;
                            leg_cnt ++;
                        }
                        rb_ptr_->motor_send_2();
                    }
                }
                    break;
                // 模式2，轨迹规划
                case 2:
                {
                    way_point = loadVectorWithBoost(waypoint_load_filename);
                    for (const auto& innerVec : way_point) {
                        for (const auto& value : innerVec) {
                            std::cout << value << " ";
                        }
                        std::cout << std::endl;
                    }

                    // 加入多项式插值
                    auto trasposed_tr = vector_transpose(way_point);
                    std::vector<std::vector<double>> pos_tr;
                    std::vector<std::vector<double>> vel_tr;
                    // 每一行作为一个关节轨迹
                    int which_joint = 0;
                    for(auto single_tr:trasposed_tr)
                    {
                        std::cout << "source trace ----------------->" << which_joint << "<------------------"<< std::endl;
                        for (auto it = single_tr.begin(); it != single_tr.end(); it ++)
                        {
                            std::cout << *it << ",";
                        }
                        std::cout << std::endl;
                        // 创建一个0~N的向量
                        std::vector<double> t_list(single_tr.size());
                        std::iota(t_list.begin(), t_list.end(), 0);
                        std::cout << "time list"<< std::endl;
                        for (auto it = t_list.begin(); it != t_list.end(); it ++)
                        {
                            std::cout << *it << ",";
                        }

                        CubicSplineInterpolator interpolator(single_tr, 0, 0, t_list);
                        interpolator.calculatePosParameters();
                        std::cout << "Pos param x: " << interpolator.get_poly_pos_param().transpose() << std::endl;
                        interpolator.calculateVelParameters();
                        std::cout << "Vel param x: " << interpolator.get_poly_vel_param().transpose() << std::endl;
                        interpolator.generateInterpolatedPoints(0.1);
                        interpolator.generateInterpolatedVels(0.1);
                        auto p_continue = interpolator.get_poly_pos_trace();
                        std::cout << "poly pos trace ----------------->" << which_joint << "<------------------"<< std::endl;
                        for (auto it = p_continue.begin(); it != p_continue.end(); it ++)
                        {
                            std::cout << *it << ",";
                        }
                        std::cout << std::endl;
                        auto v_continue = interpolator.get_poly_vel_trace();
                        std::cout << "poly vel trace ----------------->" << which_joint << "<------------------"<< std::endl;
                        for (auto it = v_continue.begin(); it != v_continue.end(); it ++)
                        {
                            std::cout << *it << ",";
                        }
                        std::cout << std::endl;
                        pos_tr.push_back(p_continue);
                        vel_tr.push_back(v_continue);
                        which_joint ++;
                    }
                    livelybot_logger::LoggerInterface::logOperation("TeachMode", "Plan_Way_Point");
                    pos_way_point_continue = vector_transpose(pos_tr);
                    vel_way_point_continue = vector_transpose(vel_tr);

                    run_mode = 0;
                }
                    break;
                // 模式3，轨迹执行
                case 3:
                {
                    if(cnt % 10 == 0)
                    {
                        auto pos = (*joint_pos).begin();
                        auto vel = (*joint_vel).begin();
                        for(auto m:rb_ptr_->Motors)
                        {                
                            m->pos_vel_MAXtqe(*pos, *vel, 10);
                            pos ++;
                            vel ++;
                        }
                        rb_ptr_->motor_send_2();
                        joint_pos ++;
                        joint_vel ++;
                        if(joint_pos == pos_way_point_continue.end())
                        {
                            run_mode = 0;
                            ROS_INFO("-------->Execute end.<-------");
                            ROS_INFO("-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace<-------");
                            livelybot_logger::LoggerInterface::logOperation("TeachMode", "Teaching_End");
                        }                    
                    }
                    cnt ++;
                }
                    break;
                default:
                    break;
            }
            r.sleep();
        }
    }

    void TeachControllerInterface::joyCallback(const sensor_msgs::Joy::ConstPtr& joy_msg)
    {       
        if(joy_msg->buttons[3] == 1 && joy_msg->axes[5] > 0.5)
        {
            std::vector<double> joint_pos;
            for (auto m : rb_ptr_->Motors)
            {
                joint_pos.push_back(m->get_current_motor_state()->position);
            }
            act.push_back(joint_pos);
            // ROS_INFO("增加一个路点，总共有%d个路点", int(act.size()));
            // ROS_INFO("-------->按Y增加路点，按X删除最后一个路点，按A完成路点采集<-------");
            ROS_INFO("add a way_point, now own %d way_point(s)", int(act.size()));
            ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode", "Add_Joint_Point");
        }
        if(joy_msg->buttons[2] == 1 && joy_msg->axes[5] > 0.5)
        {
            if(int(act.size()) > 0)
            {
                act.pop_back();
                // ROS_INFO("删除一个路点，总共有%d个路点", int(act.size()));
                ROS_INFO("del a way_point, now own %d way_point(s)", int(act.size()));
                livelybot_logger::LoggerInterface::logOperation("TeachMode", "Delete_Joint_Point");
            }
            else
            {
                // ROS_INFO("路点已清空");
                ROS_INFO("Way point has cleared.");
                livelybot_logger::LoggerInterface::logOperation("TeachMode", "Clear_Joint_Point");
            }
            // ROS_INFO("-------->按Y增加路点，按X删除最后一个路点，按A完成路点采集<-------");
            ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
        }

        if(joy_msg->buttons[2] == 1 && joy_msg->axes[5] < -0.5)
        {
            act.clear();
            ROS_INFO("Way point has cleared.");
            ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode", "Clear_Joint_Point");
        }

        if(joy_msg->buttons[0] == 1 && joy_msg->axes[5] > 0.5)
        {
            if(int(act.size()) >= 4)
            {
                saveVectorWithBoost(act, waypoint_save_filename);    
                save_backups(act);        
                ROS_INFO("Waypoint sampling finished.");
                rb_ptr_->set_reset();
                rb_ptr_->motor_send_2();
                run_mode = 0;
                ROS_INFO("-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace<-------");
                livelybot_logger::LoggerInterface::logOperation("TeachMode", "Save_Points");
            }
            else
            {
                // ROS_INFO("路点个数不够，至少需要四个，当前总共有%d个路点", int(act.size()));
                // ROS_INFO("-------->按Y增加路点，按X删除最后一个路点，按A完成路点采集<-------");
                ROS_INFO("nums of way_point, at least need 4, now own %d way_point(s)", int(act.size()));
                ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
                livelybot_logger::LoggerInterface::logOperation("TeachMode", "Save_Error_not_enough_points");
            }
        }

        if(joy_msg->buttons[1] == 1 && joy_msg->axes[5] > 0.5)
        {
            sample_stop = !sample_stop;
            if(sample_stop)
            {
                ROS_INFO("----> lock <----");
                stop_pos.clear();
                for (auto m : rb_ptr_->Motors)
                {
                    stop_pos.push_back(m->get_current_motor_state()->position);
                }
                rb_ptr_->set_reset();
                rb_ptr_->motor_send_2();
                livelybot_logger::LoggerInterface::logOperation("TeachMode", "Add_Stop_Point");
            }        
            else
            {
                ROS_INFO("----> free <----");
            }
        }

        if(joy_msg->axes[7] < -0.5 && joy_msg->axes[5] < -0.5)
        {
            lock_type ^= 0x01;
            ROS_INFO("Lock:%d", lock_type);
            ROS_INFO("-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode", "-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace 1<-------");
        }

        if(joy_msg->axes[6] < -0.5 && joy_msg->axes[5] < -0.5)
        {
            lock_type ^= (0x01<<1);
            ROS_INFO("Lock:%d", lock_type);
            ROS_INFO("-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode", "-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace 2<-------");
        }

        if(joy_msg->buttons[6] == 1 && joy_msg->axes[2] < -0.5) 
        {
            rb_ptr_->set_reset();
            rb_ptr_->motor_send_2();
            run_mode = 1;
            // ROS_INFO("-------->按Y增加路点，按X删除最后一个路点，按A完成路点采集<-------");
            ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode", "-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
        }

        if(joy_msg->buttons[7] == 1 && joy_msg->axes[2] < -0.5)
        {
            run_mode = 2;
            ROS_INFO("-------->Loading way_point.<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode", "Loading_way_point");
        }

        if(joy_msg->buttons[9] == 1)
        {
            rb_ptr_->set_reset();
            rb_ptr_->motor_send_2();
            run_mode = 3;
            cnt = 0;
            joint_pos = pos_way_point_continue.begin();
            joint_vel = vel_way_point_continue.begin();
            ROS_INFO("-------->Starting execute.<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode", "Starting_execute");
        }
    }

    void TeachControllerInterface::waypointCallback(const sim2real_msg::WayPoint::ConstPtr& waypoint_msg)
     {       
        if(waypoint_msg->add_point == 1 && waypoint_msg->RT > -0.001)
        {
            std::vector<double> joint_pos;
            for (auto m : rb_ptr_->Motors)
            {
                joint_pos.push_back(m->get_current_motor_state()->position);
            }
            act.push_back(joint_pos);
            // ROS_INFO("增加一个路点，总共有%d个路点", int(act.size()));
            // ROS_INFO("-------->按Y增加路点，按X删除最后一个路点，按A完成路点采集<-------");
            ROS_INFO("add a way_point, now own %d way_point(s)", int(act.size()));
            ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode_waypointCallback", "Add_Joint_Point");
        }
        if(waypoint_msg->delete_point == 1 && waypoint_msg->RT > -0.001)
        {
            if(int(act.size()) > 0)
            {
                act.pop_back();
                // ROS_INFO("删除一个路点，总共有%d个路点", int(act.size()));
                ROS_INFO("del a way_point, now own %d way_point(s)", int(act.size()));
                livelybot_logger::LoggerInterface::logOperation("TeachMode_waypointCallback", "Delete_Joint_Point");
            }
            else
            {
                // ROS_INFO("路点已清空");
                ROS_INFO("Way point has cleared.");
                livelybot_logger::LoggerInterface::logOperation("TeachMode_waypointCallback", "Clear_Joint_Point");
            }
            // ROS_INFO("-------->按Y增加路点，按X删除最后一个路点，按A完成路点采集<-------");
            ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
        }

        if(waypoint_msg->delete_all_point == 1 && waypoint_msg->RT < -0.5)
        {
            act.clear();
            ROS_INFO("Way point has cleared.");
            ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode_waypointCallback", "Clear_Joint_Point");
        }

        if(waypoint_msg->recording_finish == 1 && waypoint_msg->RT > -0.001)
        {
            if(int(act.size()) >= 4)
            {
                saveVectorWithBoost(act, waypoint_save_filename);    
                save_backups(act);        
                ROS_INFO("Waypoint sampling finished.");
                rb_ptr_->set_reset();
                rb_ptr_->motor_send_2();
                run_mode = 0;
                ROS_INFO("-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace<-------");
                livelybot_logger::LoggerInterface::logOperation("TeachMode_waypointCallback", "Waypoint_Sampling_Finished");
            }
            else
            {
                // ROS_INFO("路点个数不够，至少需要四个，当前总共有%d个路点", int(act.size()));
                // ROS_INFO("-------->按Y增加路点，按X删除最后一个路点，按A完成路点采集<-------");
                ROS_INFO("nums of way_point, at least need 4, now own %d way_point(s)", int(act.size()));
                ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
                livelybot_logger::LoggerInterface::logOperation("TeachMode_waypointCallback", "Save_Error_not_enough_points");
            }
        }

        if(waypoint_msg->lock_unlock_joints == 1 && waypoint_msg->RT > -0.001)
        {
            sample_stop = !sample_stop;
            if(sample_stop)
            {
                ROS_INFO("----> lock <----");
                stop_pos.clear();
                for (auto m : rb_ptr_->Motors)
                {
                    stop_pos.push_back(m->get_current_motor_state()->position);
                }
                rb_ptr_->set_reset();
                rb_ptr_->motor_send_2();
            }        
            else
            {
                ROS_INFO("----> free <----");
            }
        }

        if(waypoint_msg->lock_unlock_left < -0.5 && waypoint_msg->RT < -0.5)
        {
            lock_type ^= 0x01;
            ROS_INFO("Lock:%d", lock_type);
            ROS_INFO("-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace<-------");
        }

        if(waypoint_msg->lock_unlock_right < -0.5 && waypoint_msg->RT < -0.5)
        {
            lock_type ^= (0x01<<1);
            ROS_INFO("Lock:%d", lock_type);
            ROS_INFO("-------->Click 'LT' + '-' to add way_point, Click 'LT' + '+' to execute trace<-------");
        }

        if(waypoint_msg->start == 1 && waypoint_msg->LT < -0.5) 
        {
            rb_ptr_->set_reset();
            rb_ptr_->motor_send_2();
            run_mode = 1;
            // ROS_INFO("-------->按Y增加路点，按X删除最后一个路点，按A完成路点采集<-------");
            ROS_INFO("-------->Click Y to add way_point, click x to delete a way_point, RT+x to clear way_point, click A to finish.<-------");
        }

        if(waypoint_msg->load_path == 1 && waypoint_msg->LT < -0.5)
        {
            run_mode = 2;
            ROS_INFO("-------->Loading way_point.<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode_waypointCallback", "Loading_way_point");
        }

        if(waypoint_msg->execute_path == 1)
        {
            rb_ptr_->set_reset();
            rb_ptr_->motor_send_2();
            run_mode = 3;
            cnt = 0;
            joint_pos = pos_way_point_continue.begin();
            joint_vel = vel_way_point_continue.begin();
            ROS_INFO("-------->Starting execute.<-------");
            livelybot_logger::LoggerInterface::logOperation("TeachMode_waypointCallback", "Starting_execute");
        }
    }




    void TeachControllerInterface::keyboardCallback(const std_msgs::String::ConstPtr& msg)
    {
        // 处理键盘输入数据
        ROS_INFO("Keyboard input received:%s", msg->data.c_str());

        // 去除前后空格
        std::string input_str = msg->data;
        input_str.erase(0, input_str.find_first_not_of(' ')); // 去除前导空格
        input_str.erase(input_str.find_last_not_of(' ') + 1);  // 去除尾随空格

        // 检查输入是否为纯数字
        bool is_number = !input_str.empty() && std::all_of(input_str.begin(), input_str.end(), ::isdigit);

        if (is_number)
        {
            // 构建文件名
            std::string filename = "waypoint_" + input_str + ".boost";

            // 更新保存和加载文件名
            waypoint_save_filename = "/home/pi/sim2real_master_dreamwaq_series/way_point/" + filename;
            waypoint_load_filename = waypoint_save_filename; // 两个文件名相同

            ROS_INFO("The file name is set to:%s", filename.c_str());
        }
        else
        {
            ROS_WARN("The input is not a valid number. Please enter only numbers to set the file name.");
        }
    }

    void TeachControllerInterface::saveVectorWithBoost(const std::vector<std::vector<double>>& data, const std::string& filename) {
        std::ofstream outFile(filename);
        boost::archive::text_oarchive oa(outFile);
        oa << data;
    }

    std::vector<std::vector<double>> TeachControllerInterface::loadVectorWithBoost(const std::string& filename) {
        std::ifstream inFile(filename);
        boost::archive::text_iarchive ia(inFile);
        std::vector<std::vector<double>> data;
        ia >> data;
        return data;
    }


    std::vector<std::vector<double>> TeachControllerInterface::vector_transpose(const std::vector<std::vector<double>>& matrix) {
        if (matrix.empty()) {
            return {};
        }

        size_t rows = matrix.size();
        size_t cols = matrix[0].size();

        std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }

    void TeachControllerInterface::save_backups(const std::vector<std::vector<double>>& data)
    {
        auto t = std::time(nullptr);
        char timeStr[15]; // YYYYMMDDHHMMSS + null terminator
        std::strftime(timeStr, sizeof(timeStr), "%Y%m%d%H%M%S", std::localtime(&t));
         // 获取包的路径
        std::string package_path = ros::package::getPath("sim2real_master");
        std::string backups_file_name = package_path + "/way_point/way_point_backups_" + std::string(timeStr) + ".boost";
        saveVectorWithBoost(data, backups_file_name);
        // std::cout << "backup file name:" << backups_file_name;
    }

    int TeachControllerInterface::get_parameters()
    {
        return 0;
    }

    bool TeachControllerInterface::get_is_shutdowm()
    {
        return  quit.load();
    }

    TeachControllerInterface::~TeachControllerInterface()
    {
        quit.store(true); 
        if (thread_exec_ptr_->joinable()) {
            thread_exec_ptr_->join();
        }
        std::cout << "~TeachControllerInterface()" << std::endl;
    }
}