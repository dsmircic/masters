import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from robot_interfaces.msg import BoundingBox, BoundingBoxes

topic_realsense_base    = "/camera/camera"
topic_depth             = "/depth/image_rect_raw"
topic_image             = "/color/image_raw"

image_topic_pub         = "/yolo/detect/image"
topic_bounding_box      = "/yolo/detect/bounding_box"

timer_period = 0.5

class RealSenseSegmentation(Node):
    def __init__(self):
        super().__init__('realsense_segmentation')
        self.bridge = CvBridge()

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # Replace with your YOLO model file

        # Subscribe to the color and depth image topics
        self.color_subscriber = self.create_subscription(
            Image, 
            topic_realsense_base + topic_image, 
            self.color_callback, 
            10
        )
        
        self.depth_subscriber = self.create_subscription(
            Image,
            topic_realsense_base + topic_depth,
            self.depth_callback,
            10
        )
        
        self.image_publisher = self.create_publisher(
            Image,
            image_topic_pub,
            10
        )
        
        self.bounding_box_publisher = self.create_publisher(
            BoundingBoxes,
            topic_bounding_box,
            10
        )
        
        self.timer = self.create_timer(timer_period, self.bbox_publisher_callback)

        self.color_image    = None
        self.depth_image    = None
        self.boxes          = None

    def bbox_publisher_callback(self):
        """Publish the YOLO-processed image."""
        
        if self.boxes == None:
            return
    

        try:
            # Convert the annotated OpenCV image to a ROS Image message
            msg = self.bridge.cv2_to_imgmsg(self.color_image, encoding='bgr8')
            self.image_publisher.publish(msg)
            self.get_logger().info('Publishing YOLO-processed image.')
        except Exception as e:
            self.get_logger().error(f"Error while publishing YOLO-processed image: {e}")
            

    def color_callback(self, msg):
        """Callback to process the color image"""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.detect_obstacles()
        except Exception as e:
            self.get_logger().error(f"Error in color image callback: {e}")

    def depth_callback(self, msg):
        """Callback to process the depth image"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Error in depth image callback: {e}")

    def detect_obstacles(self):
        """Detect obstacles and calculate distances."""
        if self.color_image is None or self.depth_image is None:
            return
        
        self.boxes = BoundingBoxes()
        bounding_boxes_list = []  # Create a list to hold BoundingBox objects

        # Perform YOLO inference
        results = self.model(self.color_image)  # Perform YOLO inference

        for result in results:
            # Extract bounding boxes, confidences, and class IDs
            boxes = result.boxes.xyxy.cpu().numpy()  # Get box coordinates as numpy array
            confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Get class IDs

            # Loop through detected boxes
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                
                # Find center pixels for the given object
                center_x = int(((x1 + x2) / 2))
                center_y = int(((y1 + y2) / 2))

                # Annotate detected objects
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(self.color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Get depth at the center of the detected object
                if 0 <= center_x < self.depth_image.shape[1] and 0 <= center_y < self.depth_image.shape[0]:
                    distance = self.depth_image[center_y, center_x] / 1000.0  # Convert to meters
                    
                    cv2.putText(self.color_image, f"{distance:.2f} m", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Create a BoundingBox object
                    detected_box = BoundingBox()
                    detected_box.x_min = x1
                    detected_box.x_max = x2
                    detected_box.y_min = y1
                    detected_box.y_max = y2
                    detected_box.distance = distance
                    detected_box.class_label = label
                    
                    # Append the BoundingBox object to the list
                    bounding_boxes_list.append(detected_box)

        # Assign the list of BoundingBox objects to the boxes attribute
        self.boxes.boxes = bounding_boxes_list

        # Publish the bounding boxes
        self.bounding_box_publisher.publish(self.boxes)

        # Display the annotated image
        cv2.imshow("Obstacle Detection", self.color_image)
        cv2.waitKey(1)



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
