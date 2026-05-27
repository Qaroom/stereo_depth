"""
Launch the stereo_depth node.

Usage:
    ros2 launch stereo_depth stereo_depth.launch.py

Override topics / parameters:
    ros2 launch stereo_depth stereo_depth.launch.py \\
        left_image_topic:=/my_left/image_raw \\
        right_image_topic:=/my_right/image_raw \\
        baseline_mm:=80.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---- Launch arguments ----
    baseline_mm_arg = DeclareLaunchArgument(
        "baseline_mm",
        default_value="80.0",
        description="Stereo baseline in millimetres (distance between cameras).",
    )
    left_image_arg = DeclareLaunchArgument(
        "left_image_topic",
        default_value="/camera0/image_raw",
        description="Left image topic (sensor_msgs/Image).",
    )
    right_image_arg = DeclareLaunchArgument(
        "right_image_topic",
        default_value="/camera1/image_raw",
        description="Right image topic (sensor_msgs/Image).",
    )
    left_info_arg = DeclareLaunchArgument(
        "left_info_topic",
        default_value="/camera0/camera_info",
        description="Left camera_info topic (sensor_msgs/CameraInfo).",
    )
    right_info_arg = DeclareLaunchArgument(
        "right_info_topic",
        default_value="/camera1/camera_info",
        description="Right camera_info topic (sensor_msgs/CameraInfo).",
    )
    sync_slop_arg = DeclareLaunchArgument(
        "sync_slop",
        default_value="0.05",
        description="Allowed time offset (s) between left and right frames.",
    )
    num_disp_arg = DeclareLaunchArgument(
        "sgbm_num_disparities",
        default_value="128",
        description="StereoSGBM numDisparities (must be a multiple of 16).",
    )
    block_size_arg = DeclareLaunchArgument(
        "sgbm_block_size",
        default_value="5",
        description="StereoSGBM blockSize (odd, typically 3..11).",
    )
    show_disp_arg = DeclareLaunchArgument(
        "show_disparity_window",
        default_value="true",
        description="If true, also open a debug window showing the disparity map.",
    )

    stereo_node = Node(
        package="stereo_depth",
        executable="stereo_node",
        name="stereo_depth_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "baseline_mm": LaunchConfiguration("baseline_mm"),
            "left_image_topic": LaunchConfiguration("left_image_topic"),
            "right_image_topic": LaunchConfiguration("right_image_topic"),
            "left_info_topic": LaunchConfiguration("left_info_topic"),
            "right_info_topic": LaunchConfiguration("right_info_topic"),
            "sync_slop": LaunchConfiguration("sync_slop"),
            "sgbm_num_disparities": LaunchConfiguration("sgbm_num_disparities"),
            "sgbm_block_size": LaunchConfiguration("sgbm_block_size"),
            "show_disparity_window": LaunchConfiguration("show_disparity_window"),
        }],
    )

    return LaunchDescription([
        baseline_mm_arg,
        left_image_arg,
        right_image_arg,
        left_info_arg,
        right_info_arg,
        sync_slop_arg,
        num_disp_arg,
        block_size_arg,
        show_disp_arg,
        stereo_node,
    ])
