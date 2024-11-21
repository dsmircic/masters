import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

base_topic = "/camera/camera"
depth_topic = "/depth/color/points"
image_topic = "/color/image_raw"

class RealSenseSegmentation(Node):
    def __init__(self):
        super().__init__('realsense_segmentation')
        self.bridge = CvBridge()

        # Subscribe to the color and depth image topics
        self.color_subscriber = self.create_subscription(
            Image, base_topic + image_topic, self.color_callback, 10)
        
        print("Initialized color subscriber!")
        
        self.depth_subscriber = self.create_subscription(
            PointCloud2, base_topic + depth_topic, self.depth_callback, 10)
        
        print("Initialized depth subscriber!")

        self.color_image = None
        self.depth_data = None

    def color_callback(self, msg):
        """Callback to process the color image"""
        try:
            # Convert ROS image message to OpenCV format
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Call detect_obstacless to detect and display objects
            self.detect_obstacles()
        except Exception as e:
            self.get_logger().error(f"Error in color image callback: {e}")

    def depth_callback(self, msg):
        """Callback to process the depth data"""
        try:
            # Convert ROS PointCloud2 message to a list of 3D points
            # print(msg)
            self.depth_data = pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)
            
            self.detect_obstacles()
            # print(self.depth_data)
        except Exception as e:
            self.get_logger().error(f"Error in depth image callback: {e}")

    def detect_obstacles(self):
        """Detect obstacles and calculate distances."""
        if self.color_image is None or self.depth_data is None:
            return

        # Convert image to grayscale
        gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection (Canny)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter out small contours
            if cv2.contourArea(contour) < 500:
                continue

            # Draw bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate centroid for depth filtering
            centroid_x = x + w // 2
            centroid_y = y + h // 2

            # Get the depth value at the centroid
            distance = self.get_depth_at_point(centroid_x, centroid_y)
            
            # print(distance)
            
            if distance != -1 and 0.5 < distance < 5.0:  # Filter objects within 0.5m to 5m range
                cv2.putText(self.color_image, f"{distance:.2f} m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.get_logger().info(f"Obstacle at ({centroid_x}, {centroid_y}) - Distance: {distance:.2f} m")


        # Display the results
        cv2.imshow("Obstacle Detection", self.color_image)
        cv2.waitKey(1)

    def get_depth_at_point(self, x, y):
        """Find the depth value at a given pixel."""
        for point in self.depth_data:
            px, py, pz = point
            # print(f"x: {px}, y: {py}, z: {pz}\n")
            if int(px) == x and int(py) == y:
                return pz  # Return the Z-coordinate (distance in meters)
        return -1  # Return -1 if no depth found


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseSegmentation()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
