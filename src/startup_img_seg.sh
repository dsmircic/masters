#!/bin/bash


colcon build --packages-select img_segment
cd img_segment/ && source install/setup.bash && cd ..
ros2 run img_segment segment_node
