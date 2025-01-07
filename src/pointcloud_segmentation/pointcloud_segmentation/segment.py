import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from .helper.AbstractObject import AbstractObject
from .helper.GetObject import GetObject
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
from robot_interfaces.msg import BoundingBox, BoundingBoxes
from visualization_msgs.msg import Marker, MarkerArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from sensor_msgs.msg import PointCloud2, PointField

# https://nicolas.burrus.name/oldstuff/kinect_calibration/ 2d to 3d coordinates

class Segment3D(Node):
    def __init__(self):
        super().__init__('segment3d')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('input_topic_bbox',              '/yolo/detect/bounding_box')
        self.declare_parameter('output_no_go_zones',            '/no/go/zones')
        
        self.declare_parameter('working_frame',                 'camera_link')
        self.declare_parameter('interested_classes',            ['person', 'dog', 'clock', 'laptop', 'bottle', 'umbrella'])
        self.declare_parameter('maximum_detection_threshold',    0.3)
        self.declare_parameter('minimum_probability',            0.3)
        self.declare_parameter('object_radius',                  0.5)
        self.declare_parameter('cube_step',                      0.1)
        self.declare_parameter('qos',                            10)
        
        self.bounding_boxes_topic                               = self.get_parameter('input_topic_bbox').value
        self.working_frame                                      = self.get_parameter('working_frame').value
        self.max_detection_threshold                            = self.get_parameter('maximum_detection_threshold').value
        self.min_probability                                    = self.get_parameter('minimum_probability').value
        self.interested_classes                                 = self.get_parameter('interested_classes').value
        self.no_go_topic                                        = self.get_parameter('output_no_go_zones').value
        self.cube_step                                          = self.get_parameter('cube_step').value
        self.object_radius                                      = self.get_parameter('object_radius').value
        self.qos                                                = self.get_parameter('qos').value

        self.fx:float = 898.6607
        self.fy:float = 898.1004
        self.cx:float = 642.3966
        self.cy:float = 359.4919

        # Subscribers and Publishers
        self.no_go_zone_subscriber      = self.create_subscription(BoundingBoxes,   self.bounding_boxes_topic,  self.publish_no_go_zones,   self.qos)
        
        self.no_go_zone_publisher       = self.create_publisher(PointCloud2, self.no_go_topic,          self.qos)

        self.original_bounding_boxes    = []
        self.marker_array               = MarkerArray()

    def publish_no_go_zones(self, bounding_boxes_msg):
        """Creates a pointcloud of no-go areas for the mobile robot.
            bounding_boxes_msg: BoundingBox.msg
                message from /yolo/detect/bounding_boxes
        """
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
            objectClass                 = GetObject().createObject(bbox.class_label)
            
            if bbox.class_label not in self.interested_classes or bbox.confidence < self.min_probability or objectClass == None:
                continue
            
            object_radius: float    = objectClass.get_depth()
            measured_distance       = bbox.depth
            
            leftmost_x      = (bbox.leftmost[0]     - self.cx) * measured_distance / self.fx
            rightmost_x     = (bbox.rightmost[0]    - self.cx) * measured_distance / self.fx
            topmost_y       = (bbox.topmost[1]      - self.cy) * measured_distance / self.fy
            bottommost_y    = (bbox.bottommost[1]   - self.cy) * measured_distance / self.fy
            
            # x = (bbox.center_x - self.cx) * measured_distance / self.fx
            # y = (bbox.center_y - self.cy) * measured_distance / self.fy
            # z = measured_distance

            for dx in np.arange(leftmost_x, (rightmost_x), self.cube_step):
                for dy in np.arange(topmost_y, (bottommost_y), self.cube_step):
                    for dz in np.arange((measured_distance), (measured_distance + object_radius), self.cube_step):
                        points.append((dz, -dx, -dy))
                        print(f"dx: {dx}\ndy: {dy}\ndz: {dz}\n")

            point_cloud = pc2.create_cloud(point_cloud.header, fields, points)
            self.no_go_zone_publisher.publish(point_cloud)
    
    def publish_markers(self):        
        self.markers_publisher.publish(self.marker_array)
        self.marker_array = MarkerArray()

def main(args=None):
    rclpy.init(args=args)
    node = Segment3D()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
