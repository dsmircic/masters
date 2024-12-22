#!/bin/bash

colcon build --packages-select pointcloud_segmentation
cd pointcloud_segmentation/ && source install/setup.bash && cd ..
ros2 run pointcloud_segmentation pointcloud_segmentation
