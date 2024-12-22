import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from sensor_msgs.msg import PointCloud2
from bbox_custom_interface.msg import BoundingBoxesCustom, BoundingBox3dCustom, BoundingBoxes3dCustom
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import tf_transformations as tf_trans
import tf2_geometry_msgs

class Darknet3DNode(Node):
    def __init__(self):
        super().__init__('darknet3d_node')

        # Parameters
        self.declare_parameter('darknet_ros_topic', '/darknet_ros/bounding_boxes')
        self.declare_parameter('output_bbx3d_topic', '/darknet_ros_3d/bounding_boxes')
        self.declare_parameter('point_cloud_topic', '/camera/depth/color/points')
        self.declare_parameter('working_frame', 'camera_link')
        self.declare_parameter('maximum_detection_threshold', 0.3)
        self.declare_parameter('minimum_probability', 0.3)
        self.declare_parameter('interested_classes', [])

        self.pointcloud_topic = self.get_parameter('point_cloud_topic').value
        self.bounding_boxes_topic = self.get_parameter('darknet_ros_topic').value
        self.output_bbx3d_topic = self.get_parameter('output_bbx3d_topic').value
        self.working_frame = self.get_parameter('working_frame').value
        self.max_detection_threshold = self.get_parameter('maximum_detection_threshold').value
        self.min_probability = self.get_parameter('minimum_probability').value
        self.interested_classes = self.get_parameter('interested_classes').value

        # Subscribers and Publishers
        self.pointcloud_sub = self.create_subscription(PointCloud2, self.pointcloud_topic, self.pointcloud_callback, 10)
        self.bounding_boxes_sub = self.create_subscription(BoundingBoxesCustom, self.bounding_boxes_topic, self.bounding_boxes_callback, 10)
        self.bounding_boxes_pub = self.create_publisher(BoundingBoxes3dCustom, self.output_bbx3d_topic, 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/darknet_ros_3d/markers', 10)

        # TF Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.point_cloud = None
        self.original_bounding_boxes = []

    def pointcloud_callback(self, msg):
        self.point_cloud = msg

    def bounding_boxes_callback(self, msg):
        self.original_bounding_boxes = msg.bounding_boxes
        self.process_bounding_boxes()

    def process_bounding_boxes(self):
        if not self.point_cloud or not self.original_bounding_boxes:
            return

        try:
            # Get transform to working frame
            transform = self.tf_buffer.lookup_transform(
                self.working_frame, 
                self.point_cloud.header.frame_id, 
                rclpy.time.Time())
            transformed_cloud = self.transform_point_cloud(self.point_cloud, transform)
        except Exception as e:
            self.get_logger().error(f"Transform failed: {str(e)}")
            return

        cloud_points = self.pointcloud_to_array(transformed_cloud)
        bounding_boxes_msg = BoundingBoxes3dCustom()
        bounding_boxes_msg.header = transformed_cloud.header

        for bbox in self.original_bounding_boxes:
            if bbox.probability < self.min_probability:
                continue

            # Calculate 3D bounding box
            box = self.calculate_3d_bounding_box(cloud_points, bbox)
            if box:
                bounding_boxes_msg.bounding_boxes.append(box)

        self.bounding_boxes_pub.publish(bounding_boxes_msg)
        self.publish_markers(bounding_boxes_msg)

    def transform_point_cloud(self, pointcloud, transform):
        # Extract translation and rotation
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Build transformation matrix
        translation_matrix = tf_trans.translation_matrix([translation.x, translation.y, translation.z])
        rotation_matrix = tf_trans.quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        transform_matrix = np.dot(translation_matrix, rotation_matrix)

        # Read points as a NumPy array
        points = np.array(list(pc2.read_points(pointcloud, skip_nans=True)))

        # Separate x, y, z and other fields
        xyz = points[:, :3]
        extra_fields = points[:, 3:] if points.shape[1] > 3 else None

        # Add homogeneous coordinate to xyz
        xyz_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))

        # Transform all points in one matrix multiplication
        transformed_xyz_homogeneous = xyz_homogeneous @ transform_matrix.T

        # Drop homogeneous coordinate
        transformed_xyz = transformed_xyz_homogeneous[:, :3]

        # Combine transformed xyz with extra fields (if any)
        if extra_fields is not None:
            transformed_points = np.hstack((transformed_xyz, extra_fields))
        else:
            transformed_points = transformed_xyz

        # Create new PointCloud2 message
        fields = pointcloud.fields  # Retain all fields
        transformed_cloud = pc2.create_cloud(pointcloud.header, fields, transformed_points)

        return transformed_cloud


    def calculate_3d_bounding_box(self, cloud_points, bbox):
        center_x = (bbox.xmin + bbox.xmax) // 2
        center_y = (bbox.ymin + bbox.ymax) // 2
        center_idx = center_y * cloud_points.shape[1] + center_x

        if center_idx >= len(cloud_points):
            return None

        center_point = cloud_points[center_idx]
        if np.isnan(center_point).any():
            return None

        # Initialize bounding box dimensions
        min_coords = np.full(3, np.inf)
        max_coords = np.full(3, -np.inf)

        for y in range(bbox.ymin, bbox.ymax):
            for x in range(bbox.xmin, bbox.xmax):
                idx = y * cloud_points.shape[1] + x
                point = cloud_points[idx]

                if np.isnan(point).any() or np.linalg.norm(point[:3] - center_point[:3]) > self.max_detection_threshold:
                    continue

                min_coords = np.minimum(min_coords, point[:3])
                max_coords = np.maximum(max_coords, point[:3])

        if np.isinf(min_coords).any() or np.isinf(max_coords).any():
            return None


        bounding_box = BoundingBox3dCustom()
        bounding_box.object_name = bbox.class_id
        bounding_box.probability = bbox.probability
        bounding_box.xmin, bounding_box.ymin, bounding_box.zmin = min_coords
        bounding_box.xmax, bounding_box.ymax, bounding_box.zmax = max_coords
        print(bounding_box)
        return bounding_box

    def pointcloud_to_array(self, pointcloud):
        points = list(pc2.read_points(pointcloud, field_names=("x", "y", "z"), skip_nans=True))
        return np.array(points)

    def publish_markers(self, bounding_boxes_msg):
        marker_array = MarkerArray()
        for idx, bbox in enumerate(bounding_boxes_msg.bounding_boxes):
            marker = Marker()
            marker.header = bounding_boxes_msg.header
            marker.ns = 'darknet3d'
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = (bbox.xmax + bbox.xmin) / 2.0
            marker.pose.position.y = (bbox.ymax + bbox.ymin) / 2.0
            marker.pose.position.z = (bbox.zmax + bbox.zmin) / 2.0
            marker.scale.x = bbox.xmax - bbox.xmin
            marker.scale.y = bbox.ymax - bbox.ymin
            marker.scale.z = bbox.zmax - bbox.zmin
            marker.color.a = 0.4
            marker.color.r = (1.0 - bbox.probability) * 255.0
            marker.color.g = bbox.probability * 255.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        self.markers_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = Darknet3DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
