from setuptools import setup

package_name = 'segmentation'  # Simpler package name

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools', 'opencv-python', 'pyrealsense2', 'numpy'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'segmentation_node = segmentation.image_segment:main',  # Your node entry point
        ],
    },
)
