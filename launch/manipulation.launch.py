"""main_node(C++, FSM+IK+모터 I/O를 전부 담당)를 띄우는 실제 실행용 런치 파일.

구 ik_solve_node.py는 main_node.cpp(fsm.cpp)로 완전히 흡수되어서 더 이상 별도
노드로 안 뜸(joint_command/manipulation_state/pick_place_ready_done 토픽도 전부
내부 함수 호출로 대체되어 없어짐).

기본으로는 task_planner가 activate_cmd(true)를 보내야 active로 전환되며, 그
전까지는 configure 상태로 대기함. task_planner 없이 테스트할 때는:
  ros2 topic pub --once /activate_cmd std_msgs/msg/Bool "{data: true}"

robot_state_publisher/rviz2는 URDF 정적 표시용으로만 남겨둠 - main_node가 더
이상 joint_states류 토픽을 publish하지 않으므로(RViz 실시간 연동은 이번에
포기함) 관절이 실시간으로 움직이는 모습은 안 보임.
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(pkg_dir, 'model', 'irc_man.urdf')
    rviz_config = os.path.join(pkg_dir, 'config', 'display.rviz')
    main_node_config = os.path.join(pkg_dir, 'config', 'main_node.yaml')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()
    common_params = {
        'urdf_path': urdf_path,
        'ee_frame': 'ee_link_1',
    }

    return LaunchDescription([
        Node(
            package='humanoid_manipulation',
            executable='main_node',
            name='manipulation_node_cpp',
            output='screen',
            parameters=[main_node_config, common_params],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
    ])
