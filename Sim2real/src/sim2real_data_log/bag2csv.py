# -*- coding: utf-8 -*-

###### pip install bagpy #####
import bagpy
from bagpy import bagreader
import os

bag = bagreader('/home/imagining/Desktop/Project/pi/sim2real/src/sim2real_data_log/sim2real_log_2024-08-16-20-28-34.bag')

topics = ['/mtr_state', '/rbt_state', '/mtr_target', '/rbt_target', '/pd2rl', '/rl2pd']

for topic in topics:
    csv_file = bag.message_by_topic(topic)
    print(f"Data from {topic} exported to {csv_file}")