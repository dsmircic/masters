from setuptools import setup
import os
from glob import glob

package_name = 'img_segment'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),  # Ensure launch files are installed
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A package for image segmentation',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segment_node = img_segment.image_segment:main',  # Update this based on your script
        ],
    },
)
