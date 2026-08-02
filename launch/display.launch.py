"""URDF 미리보기용 런치 파일. joint_state_publisher_gui의 슬라이더로 관절을 수동
조작하면서 RViz에서 로봇 모델을 확인할 때 씀. 실제 FSM/IK 노드는 안 띄우므로
manipulation.launch.py와는 별개(그건 실제 실행용).
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
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        ),
    ])
