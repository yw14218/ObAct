import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    interbotix_xsarm_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('interbotix_xsarm_control'),
                'launch',
                'xsarm_control.launch.py'
            )
        ),
        launch_arguments={'robot_model': 'vx300s'}.items()
    )

    interbotix_xsarm_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('interbotix_xsarm_moveit'),
                'launch',
                'xsarm_moveit.launch.py'
            )
        ),
        launch_arguments={
            'robot_model': 'vx300s',
            'hardware_type': 'actual'
        }.items()
    )

    realsense_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'rgb_camera.color.profile': '848x480x30',
            'depth_module.profile': '848x480x30',
            'align_depth.enable': 'true',
            # 'pointcloud.enable': 'true',
            'spatial_filter.enable': 'true',
            'temporal_filter.enable': 'true',
            'hole_filling_filter.enable': 'true',
            'device_type': 'd405'
        }.items()
    )

    hand_eye_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('interbotix_xsarm_moveit'),
                'launch',
                'hand_eye.launch.py'
            )
        )
    )

    return LaunchDescription([
        # DeclareLaunchArgument(
        #     'robot_model',
        #     default_value='vx300s',
        #     description='Model type of the Interbotix robot'
        # ),
        # interbotix_xsarm_control_launch,
        TimerAction(period=5.0, actions=[interbotix_xsarm_moveit_launch]),
        TimerAction(period=6.0, actions=[realsense_camera_launch]),
        TimerAction(period=9.0, actions=[hand_eye_launch])
    ])