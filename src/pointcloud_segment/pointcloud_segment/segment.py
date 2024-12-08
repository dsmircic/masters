from robot_interfaces import BoundingBox, BoundingBoxes
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

topic_img_segment   = "/yolo/detect/bounding_box"
topic_pointcloud    = "/camera/camera/depth/color/points"

class PointCloudSegmentation(Node):
    def __init__(self):
        super.__init__('pointcloud-segmentation')
    
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            topic_pointcloud,
            self.pointcloud_callback,
            10
        )
        
        self.bounding_box_sub = self.create_subscription(
            BoundingBoxes,
            topic_img_segment,
            self.bounding_box_callback,
            10
        )
        
    def pointcloud_callback(self):
        return

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSegmentation()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
