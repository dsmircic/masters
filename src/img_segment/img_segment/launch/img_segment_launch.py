#!/usr/bin/env python3


from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='img_segment',  # Replace with your package name
            executable='image_segment.py',  # Replace with your Python file
            name='img_segment',
            output='screen',
            parameters=[
                # Add parameters here if needed
            ],
            remappings=[
                # Remap topics here if needed
                # ('/camera/depth/image_rect_raw', '/new_depth_topic'),
                # ('/camera/color/image_raw', '/new_color_topic'),
            ]
        ),
    ])
