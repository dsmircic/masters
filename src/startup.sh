#!/bin/bash


colcon build --packages-select img_segment
ros2 run img_segment segment_node
