#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0)/.. && pwd)
echo $SCRIPT_DIR

sleep 1
gnome-terminal --title="sim2real" -- bash -c "
exec bash -i -c 'cd $SCRIPT_DIR;
source ./devel/setup.bash;
roslaunch sim2real_master joy_control_pi_plus_waist_orin.launch'
"
wait
