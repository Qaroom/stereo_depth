"""
İki USB kamerayı node olarak başlatır:
  /camera0/image_raw , /camera0/camera_info   (sol kamera)
  /camera1/image_raw , /camera1/camera_info   (sağ kamera)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
pkg_share = get_package_share_directory('stereo_depth')
default_left_yaml = 'file://' + os.path.join(pkg_share, 'config', 'camera0.yaml')
    
default_right_yaml = 'file://' + os.path.join(pkg_share, 'config', 'camera1.yaml')

def generate_launch_description():
    left_device_arg = DeclareLaunchArgument(
        "left_device", default_value="/dev/video2",
        description="Sol kameranın V4L2 cihaz yolu.")
    right_device_arg = DeclareLaunchArgument(
        "right_device", default_value="/dev/video0",
        description="Sağ kameranın V4L2 cihaz yolu.")
    width_arg = DeclareLaunchArgument(
        "width", default_value="640",
        description="Görüntü genişliği (px).")
    height_arg = DeclareLaunchArgument(
        "height", default_value="480",
        description="Görüntü yüksekliği (px).")
    pixel_format_arg = DeclareLaunchArgument(
        "pixel_format", default_value="YUYV",
        description="Kameradan istenecek FOURCC formatı (YUYV / UYVY / GREY).")
    left_info_url_arg = DeclareLaunchArgument(
        "left_info_url", default_value=default_left_yaml,
        description="Sol kamera kalibrasyon dosyası (file:///... .yaml).")
    right_info_url_arg = DeclareLaunchArgument(
        "right_info_url", default_value=default_right_yaml,
        description="Sağ kamera kalibrasyon dosyası (file:///... .yaml).")

    image_size = PythonExpression(
        ["[", LaunchConfiguration("width"), ", ",
         LaunchConfiguration("height"), "]"]
    )

    left_camera = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name="v4l2_camera",
        namespace="camera0",
        output="screen",
        parameters=[{
            "video_device": LaunchConfiguration("left_device"),
            "image_size": image_size,
            "pixel_format": LaunchConfiguration("pixel_format"),
            "camera_info_url": LaunchConfiguration("left_info_url"),
            "camera_frame_id": "camera0",
        }],
    )

    right_camera = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name="v4l2_camera",
        namespace="camera1",
        output="screen",
        parameters=[{
            "video_device": LaunchConfiguration("right_device"),
            "image_size": image_size,
            "pixel_format": LaunchConfiguration("pixel_format"),
            "camera_info_url": LaunchConfiguration("right_info_url"),
            "camera_frame_id": "camera1",
        }],
    )

    return LaunchDescription([
        left_device_arg,
        right_device_arg,
        width_arg,
        height_arg,
        pixel_format_arg,
        left_info_url_arg,
        right_info_url_arg,
        left_camera,
        right_camera,
    ])
