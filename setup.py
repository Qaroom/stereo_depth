from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'stereo_depth'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
        glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
        glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Stereo vision based distance measurement (mouse-selected ROI) for ROS2 Jazzy.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_node = stereo_depth.stereo_node:main',
        ],
    },
)
