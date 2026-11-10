import os
from glob import glob
from setuptools import setup

package_name = 'my_robot_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 下面这两行是核心！让系统把 launch 和 urdf 文件装进去
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # 合并模型是 xacro（urdf/*.urdf.xacro），必须一起装进去
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf.xacro')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        # Nav2 参数：AMCL 的 base_frame_id 改成了 base_link
        # 机械臂控制器参数（yaml）和 SRDF（srdf）也放在这里
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.srdf')),
        # 地图和世界文件必须安装到 share，否则 map_server 找不到 yaml
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mobile_manipulation_node = '
            'my_robot_sim.mobile_manipulation_node:main',
            'visual_grasp = my_robot_sim.visual_grasp:main',
            'search_and_grasp = my_robot_sim.search_and_grasp:main',
        ],
    },
)
