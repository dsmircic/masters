from setuptools import setup

package_name = 'pointcloud_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, f"{package_name}.helper"],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dino',
    maintainer_email='dino.smircic@fer.hr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_segmentation = pointcloud_segmentation.segment:main'
        ],
    },
)
