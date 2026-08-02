"""debug_ik_node(ros2 run으로 별도 실행)와 함께 쓰는 RViz 런치 파일. joint_states는
debug_ik_node가 직접 publish하므로 joint_state_publisher_gui는 띄우지 않음(경쟁 방지).
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(pkg_dir, 'model', 'irc_man.urdf')
    rviz_config = os.path.join(pkg_dir, 'config', 'display.rviz')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        ),
    ])
