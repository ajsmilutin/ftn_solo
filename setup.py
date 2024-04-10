from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ftn_solo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name,'config', 'rviz'), glob('config/rviz/*.rviz')),
        (os.path.join('share', package_name,'config', 'tasks'), glob('config/tasks/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ajsmilutin',
    maintainer_email='ajsmilutin@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulation_node = ftn_solo.visualize:simulate',
            'connector_node = ftn_solo.connector:main'
        ],
    },
)
