from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from robot_properties_solo.robot_resources import Resources
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context, *args, **kwargs):
    robot_version = LaunchConfiguration('robot_version', default="solo12")
    sim = LaunchConfiguration('sim', default='False')
    fixed = LaunchConfiguration('fixed', default='False')
    start_paused = LaunchConfiguration('start_paused', default='False')
    pos = LaunchConfiguration('pos', default='[0.0, 0.0, 0.4]')
    rpy = LaunchConfiguration('rpy', default='[0.0, 0.0, 0.0]')
    robot_version_value = robot_version.perform(context)
    resources = Resources(robot_version_value)
    with open(resources.urdf_path, 'r') as infp:
        robot_desc = infp.read()
    return [Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': sim}]), Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=[
                '-d', os.path.join(get_package_share_directory("ftn_solo"), "config", "rviz", "robot.rviz")],
            parameters=[{'use_sim_time': sim}]
    ),
        Node(
            package='ftn_solo',
            executable='connector_node',
            name='connector_node',
            parameters=[{'use_sim_time': sim, 'sim': sim,
                         'robot_version': robot_version_value,
                         'fixed': fixed,
                         'start_paused': start_paused,
                         'pos': pos,
                         'rpy': rpy}]
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_version',
            default_value='solo12',
            description='Version of the robot to be used'),
        DeclareLaunchArgument(
            'sim',
            default_value='False',
            description='Use simulation instead of real robot'),
        OpaqueFunction(function=launch_setup)
    ])
