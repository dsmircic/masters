import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
from robot_interfaces.msg import BoundingBox, BoundingBoxes
from visualization_msgs.msg import Marker, MarkerArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from sensor_msgs.msg import PointCloud2, PointField

class Darknet3DNode(Node):
    def __init__(self):
        super().__init__('segment3d')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('input_topic_bbox',          '/yolo/detect/bounding_box')
        self.declare_parameter('point_cloud_topic',         '/camera/depth/color/points')
        self.declare_parameter('output_topic_3d_marker',    '/yolo/detect/markers')
        self.declare_parameter('input_realsense_depth',     '/camera/depth/image_rect_raw')
        self.declare_parameter('output_no_go_zones',        '/no/go/zones')
        
        self.declare_parameter('working_frame',                 'camera_link')
        self.declare_parameter('maximum_detection_threshold',    0.3)
        self.declare_parameter('minimum_probability',            0.3)
        self.declare_parameter('interested_classes',            ['person', 'dog', 'clock', 'laptop'])

        self.pointcloud_topic           = self.get_parameter('point_cloud_topic').value
        self.depth_topic                = self.get_parameter('input_realsense_depth').value
        self.bounding_boxes_topic       = self.get_parameter('input_topic_bbox').value
        self.output_bbx3d_topic         = self.get_parameter('output_topic_3d_marker').value
        self.working_frame              = self.get_parameter('working_frame').value
        self.max_detection_threshold    = self.get_parameter('maximum_detection_threshold').value
        self.min_probability            = self.get_parameter('minimum_probability').value
        self.interested_classes         = self.get_parameter('interested_classes').value
        self.no_go_topic                = self.get_parameter('output_no_go_zones').value
        
        self.fx:float = 898.6607
        self.fy:float = 898.1004
        self.cx:float = 642.3966
        self.cy:float = 359.4919

        # Subscribers and Publishers
        self.pointcloud_sub             = self.create_subscription(PointCloud2,     self.pointcloud_topic,      self.pointcloud_callback,   10)
        self.depth_subscriber           = self.create_subscription(Image,           self.depth_topic,           self.depth_callback,        10)
        self.bounding_box_subscriber    = self.create_subscription(BoundingBoxes,   self.bounding_boxes_topic,  self.calculate_bbox_3d,     10)
        self.no_go_zone_subscriber      = self.create_subscription(BoundingBoxes,   self.bounding_boxes_topic,  self.publish_no_go_zones,   10)
        
        self.markers_publisher          = self.create_publisher(MarkerArray, self.output_bbx3d_topic,   10)
        self.no_go_zone_publisher       = self.create_publisher(PointCloud2, self.no_go_topic,          10)

        self.original_bounding_boxes    = []
        self.depth_image                = None
        self.marker_array               = MarkerArray()

    def pointcloud_callback(self, msg):
        self.point_cloud = msg
        
    def depth_callback(self, msg):
        """Callback to process the depth image"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Error in depth image callback: {e}")


    def bounding_boxes_callback(self, msg):
        self.original_bounding_boxes = msg.bounding_boxes
        self.process_bounding_boxes()
        

    def calculate_bbox_3d(self, bounding_boxes_msg):
        for idx, bbox in enumerate(bounding_boxes_msg.boxes):
            marker = Marker()
            marker.header.frame_id = 'camera_link'  # The reference frame for the marker (can be set to 'base_link', 'map', etc.)
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "spheres"
            marker.type = Marker.SPHERE  # Marker type: SPHERE
            marker.action = Marker.ADD  # Action to add the marker
            
            depth_value = self.depth_image[bbox.center_y, bbox.center_x] / 1000.0  # Convert mm to meters (if ne
            
            # Convert 2D pixel coordinates to 3D coordinates
            x = round((bbox.center_x - self.cx) * depth_value / self.fx, 3)
            y = round((bbox.center_y - self.cy) * depth_value / self.fy, 3)
            z = round(depth_value, 3)
            
            marker.pose.position = Point(x=z, y=x, z=1.0)  # Change as needed
            marker.pose.orientation.w = 1.0  # No rotation

            # Set the scale of the sphere
            
            if bbox.class_label in self.interested_classes and bbox.confidence > self.min_probability:
                print(f"X: {x}, Y: {y}, Z: {z}")
                marker.scale.x = 1.0  # Radius in the X direction (diameter = 2*radius)
                marker.scale.y = 1.0  # Radius in the Y direction
                marker.scale.z = 1.0  # Radius in the Z direction
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red sphere with full opacity
                marker.id = idx # Unique ID for the marker

                # Add the marker to the MarkerArray
                self.marker_array.markers.append(marker)

        self.publish_markers()
        

    def publish_no_go_zones(self, bounding_boxes_msg):
        point_cloud                 = PointCloud2()
        point_cloud.header.frame_id = self.working_frame
        point_cloud.header.stamp    = self.get_clock().now().to_msg()
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

        points = []

        for bbox in bounding_boxes_msg.boxes:
            depth_value = self.depth_image[bbox.center_y, bbox.center_x] / 1000.0
            x = (bbox.center_x - self.cx) * depth_value / self.fx
            y = (bbox.center_y - self.cy) * depth_value / self.fy
            z = depth_value
            
            if bbox.class_label in self.interested_classes and bbox.confidence > self.min_probability:
                # Generate no-go zone points (e.g., around the detected person)
                for dx in [-0.5, 0.5]:
                    for dy in [-0.5, 0.5]:
                        points.append((x + dx, y + dy, z))

        point_cloud = pc2.create_cloud(point_cloud.header, fields, points)
        self.no_go_zone_publisher.publish(point_cloud)

    
    def publish_markers(self):        
        self.markers_publisher.publish(self.marker_array)
        self.marker_array = MarkerArray()


def main(args=None):
    rclpy.init(args=args)
    node = Darknet3DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
