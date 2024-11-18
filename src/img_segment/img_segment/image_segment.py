import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RealSenseSegmentation(Node):

    def __init__(self):
        super().__init__('realsense_segmentation')
        
        self.bridge = CvBridge()

        # Subscribe to the color and depth image topics
        self.color_subscriber = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, 10)
        
        self.depth_subscriber = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        self.color_image = None
        self.depth_image = None

    def color_callback(self, msg):
        """Callback to process the color image"""
        try:
            # Convert ROS image message to OpenCV format
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_images()
        except Exception as e:
            self.get_logger().error(f"Error in color image callback: {e}")

    def depth_callback(self, msg):
        """Callback to process the depth image"""
        try:
            # Convert ROS image message to OpenCV format (depth image)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.process_images()
        except Exception as e:
            self.get_logger().error(f"Error in depth image callback: {e}")

    def process_images(self):
        """Perform segmentation and distance measurement"""
        if self.color_image is None or self.depth_image is None:
            return
        
        # Define the color for segmentation (e.g., red)
        lower_color = np.array([0, 0, 100])
        upper_color = np.array([50, 50, 255])

        # Segment the image based on the color range (Red in this case)
        mask = cv2.inRange(self.color_image, lower_color, upper_color)
        segmented_image = cv2.bitwise_and(self.color_image, self.color_image, mask=mask)

        # Find the coordinates of non-zero mask pixels
        non_zero_coords = np.column_stack(np.where(mask > 0))

        # Assuming depth_image is in mm (RealSense depth data format)
        for coord in non_zero_coords:
            # Get depth (in mm) at the coordinate (y, x)
            depth = self.depth_image[coord[0], coord[1]]
            distance_in_meters = depth * 0.001  # Convert from mm to meters
            self.get_logger().info(f"Distance at ({coord[1]}, {coord[0]}): {distance_in_meters:.2f} meters")
        
        # Show the segmented image using OpenCV
        cv2.imshow("Segmented Image", segmented_image)
        cv2.waitKey(1)  # Refresh window

def main(args=None):
    rclpy.init(args=args)

    node = RealSenseSegmentation()

    # Spin the node to keep it alive and process incoming messages
    rclpy.spin(node)

    # Shutdown ROS when done
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    