#include "dev/develop_controller.h"


int main(int argc, char **argv)
{
    ros::init(argc, argv,"develop_controller_node");
    hightorque::DevelopControllerNode DCN;
    ros::spin();
    return 0;
}
