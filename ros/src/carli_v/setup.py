from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'carli_v'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='landson',
    maintainer_email='guotl321@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = carli_v.my_node:main',
            'radar_cube_node = carli_v.radar_cube_node:main',
            'optical_flow_node = carli_v.optical_flow_node:main',	
        ],
    },
)
