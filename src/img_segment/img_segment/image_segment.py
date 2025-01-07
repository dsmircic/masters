import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from robot_interfaces.msg import BoundingBox, BoundingBoxes

# https://opencvpython.blogspot.com/2012/06/contours-3-extraction.html segmentation coordinates

topic_realsense_base    = "/camera"
topic_depth             = "/depth/image_rect_raw"
topic_image             = "/color/image_raw"

image_topic_pub         = "/yolo/detect/image"
topic_bounding_box      = "/yolo/detect/bounding_box"

safety_margin = 30

timer_period = 0.5

class Point2D():
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"({self.x}, {self.y})"
    
    def __iter__(self):
        # Make the object iterable, yielding x and y
        yield self.x
        yield self.y

class RealSenseSegmentation(Node):
    def __init__(self):
        super().__init__('realsense_segmentation')
        self.bridge = CvBridge()

        # Load YOLO model
        self.model = YOLO('yolov8n-seg.pt', verbose=False)
        for name in self.model.names:
            print(self.model.names[name])
        
        self.declare_parameter('input_realsense_img',   '/camera/color/image_raw')
        self.declare_parameter('input_realsense_depth', '/camera/depth/image_rect_raw')
        self.declare_parameter('output_bbox_topic',     '/yolo/detect/bounding_box')
        self.declare_parameter('interested_classes',    ['person', 'dog', 'clock', 'tv', 'laptop', 'bottle', 'umbrella'])
        self.declare_parameter('minimal_confidence',     0.4)
        
        self.topic_realsense_img        = self.get_parameter('input_realsense_img').value
        self.topic_realsense_depth      = self.get_parameter('input_realsense_depth').value
        self.topic_yolo_bbox            = self.get_parameter('output_bbox_topic').value
        self.interested_classes         = self.get_parameter('interested_classes').value
        self.minimal_confidence         = self.get_parameter('minimal_confidence').value
        
        # Subscribe to the color and depth image topics
        self.color_subscriber = self.create_subscription(Image, self.topic_realsense_img,   self.color_callback, 10)
        self.depth_subscriber = self.create_subscription(Image, self.topic_realsense_depth, self.depth_callback, 10)
        
        self.bounding_box_publisher = self.create_publisher(BoundingBoxes, self.topic_yolo_bbox, 10)

        self.color_image    = None
        self.boxes          = None
        
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
        if self.color_image is None:
            self.get_logger().error("No valid color image received.")
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
                    
                    leftmost, rightmost, topmost, bottommost, center_x, center_y = self.get_mask_boundaries(mask)
                    
                    # Annotate segmentation on the color image
                    self.mask_objects(mask)
                    
                    cv2.circle(self.color_image, (leftmost.x, leftmost.y), 5, (0, 0, 255), 2)
                    cv2.circle(self.color_image, (rightmost.x, rightmost.y), 5, (0, 0, 255), 2)
                    cv2.circle(self.color_image, (bottommost.x, bottommost.y), 5, (0, 0, 255), 2)
                    cv2.circle(self.color_image, (topmost.x, topmost.y), 5, (0, 0, 255), 2)

                    # Draw bounding box and annotations
                    label = f"{self.model.names[int(cls)]} {conf:.2f}"
                    cv2.putText(self.color_image, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    depth = self.depth_image[center_y, center_x] / 1000.0

                    # Create a BoundingBox object
                    detected_box                = BoundingBox()
                    detected_box.leftmost       = [int(value) for value in leftmost]
                    detected_box.rightmost      = [int(value) for value in rightmost]
                    detected_box.topmost        = [int(value) for value in topmost]
                    detected_box.bottommost     = [int(value) for value in bottommost]
                    detected_box.center_x       = int(center_x)
                    detected_box.center_y       = int(center_y)
                    detected_box.confidence     = conf / 1.0
                    detected_box.class_label    = self.model.names[int(cls)]
                    detected_box.depth          = depth
                    bounding_boxes_list.append(detected_box)

                # Publish bounding boxes
                self.boxes.boxes = bounding_boxes_list
                self.bounding_box_publisher.publish(self.boxes)

                # Display the image with segmentations
                cv2.imshow("YOLO Segmentation and Distances", self.color_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in obstacle detection: {e}")


    def mask_objects(self, mask):
        """Apply a random color to the mask and create an overlay."""
        if mask.shape[:2] != self.color_image.shape[:2]:
            mask = cv2.resize(mask, (self.color_image.shape[1], self.color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Ensure mask is binary and scaled to 255
        mask = (mask.astype(np.uint8) * 255)

        # Generate a random RGB color
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

        # Create a colored mask
        colored_mask = cv2.merge([
            mask * color[0],  # Red channel
            mask * color[1],  # Green channel
            mask * color[2]   # Blue channel
        ])
        
        self.color_image = cv2.addWeighted(self.color_image, 1.0, colored_mask, 0.5, 0)
        
    def get_mask_boundaries(self, mask) -> Point2D:
        if mask.shape[:2] != self.color_image.shape[:2]:
            mask = cv2.resize(mask, (self.color_image.shape[1], self.color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        binary_mask = mask > 0  # Assuming mask is normalized between 0 and 1

        # Get coordinates of all non-zero points
        y_indices, x_indices = np.where(binary_mask)
        
        # Calculate the extreme points
        leftmost_x = x_indices.min()
        rightmost_x = x_indices.max()
        topmost_y = y_indices.min()
        bottommost_y = y_indices.max()
        
        leftmost:Point2D    = Point2D(leftmost_x, y_indices[x_indices.argmin()])
        rightmost:Point2D   = Point2D(rightmost_x, y_indices[x_indices.argmax()])
        topmost:Point2D     = Point2D(x_indices[y_indices.argmin()], topmost_y)
        bottommost:Point2D  = Point2D(x_indices[y_indices.argmax()], bottommost_y)
        
        mask_pixels = cv2.findNonZero(mask.astype(np.uint8))  # Find all non-zero (True) pixels in the mask
        if mask_pixels is not None:
            moments = cv2.moments(mask.astype(np.uint8))  # Compute moments of the mask
            if moments["m00"] != 0:  # Ensure non-zero area
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                # Fallback in case mask area is zero
                center_x = (leftmost.x + rightmost.x) // 2
                center_y = (topmost.y + bottommost.y) // 2
        else:
            # Fallback for an empty mask
            center_x = (leftmost.x + rightmost.x) // 2
            center_y = (topmost.y + bottommost.y) // 2

        
        return leftmost, rightmost, topmost, bottommost, center_x, center_y
        

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
