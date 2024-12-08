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
        self.model = YOLO('yolov8n-seg.pt')  # Replace with your YOLO model file

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
        """Detect obstacles, calculate distances, and visualize segmentations."""
        if self.color_image is None or self.depth_image is None:
            self.get_logger().error("No valid color or depth image received.")
            return

        try:
            # Perform YOLO inference
            results = self.model(self.color_image)
            if not results or len(results) == 0:
                self.get_logger().warning("No detections made by the YOLO model.")
                return

            self.boxes = BoundingBoxes()
            bounding_boxes_list = []

            for result in results:
                # Extract bounding boxes, confidences, class IDs, and masks
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
                confidences = result.boxes.conf.cpu().numpy() if result.boxes else []
                classes = result.boxes.cls.cpu().numpy() if result.boxes else []
                masks = result.masks.data.cpu().numpy() if result.masks else []

                for box, conf, cls, mask in zip(boxes, confidences, classes, masks):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = int(((x1 + x2) / 2))
                    center_y = int(((y1 + y2) / 2))

                    # Annotate segmentation on the color image
                    colored_mask = self.get_colored_mask(mask)
                    self.color_image = cv2.addWeighted(self.color_image, 1.0, colored_mask, 0.5, 0)

                    # Get depth value at the center
                    distance = None
                    if 0 <= center_x < self.depth_image.shape[1] and 0 <= center_y < self.depth_image.shape[0]:
                        distance = self.depth_image[center_y, center_x] / 1000.0  # Convert mm to meters

                    # Draw bounding box and annotations
                    label = f"{self.model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(self.color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(self.color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if distance is not None:
                        cv2.putText(self.color_image, f"{distance:.2f} m", (x1, y2 + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Create a BoundingBox object
                    detected_box = BoundingBox()
                    detected_box.x_min = x1
                    detected_box.x_max = x2
                    detected_box.y_min = y1
                    detected_box.y_max = y2
                    detected_box.distance = distance if distance else 0.0
                    detected_box.class_label = label
                    bounding_boxes_list.append(detected_box)

            # Publish bounding boxes
            self.boxes.boxes = bounding_boxes_list
            self.bounding_box_publisher.publish(self.boxes)

            # Display the image with segmentations
            cv2.imshow("YOLO Segmentation and Distances", self.color_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in obstacle detection: {e}")

    def get_colored_mask(self, mask):
        """Apply a random color to the mask and create an overlay."""
        mask = mask.astype(np.uint8) * 255
        color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8)
        colored_mask = cv2.merge([mask * color[0, 0], mask * color[0, 1], mask * color[0, 2]])
        return colored_mask




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
