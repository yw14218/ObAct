""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: vx300s/ee_gripper_link -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "vx300s/ee_gripper_link",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "-0.0683247",
                "--y",
                "0.00957573",
                "--z",
                "0.0818357",
                "--qx",
                "0.606729",
                "--qy",
                "-0.596466",
                "--qz",
                "0.372879",
                "--qw",
                "-0.370229",
                # "--roll",
                # "0.00991171",
                # "--pitch",
                # "2.03511",
                # "--yaw",
                # "-1.59398",
            ],
        ),
    ]
    return LaunchDescription(nodes)
