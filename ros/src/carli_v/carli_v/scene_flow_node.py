import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, CameraInfo, PointField, CompressedImage, Image
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from collections import deque
import struct
from pyquaternion import Quaternion
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scene_flow_models.GMSF.models.gmsf import GMSF
import torch

class SceneFlowNode(Node):
    def __init__(self):
        super().__init__('SceneFlowNode')

        self.lidar_subscriber = self.create_subscription(
            PointCloud2, '/lidar_points', self.lidar_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.device = "cuda"

        self.model = GMSF(backbone='DGCNN',
                feature_channels=128,
                ffn_dim_expansion=4,
                num_transformer_pt_layers=1,
                num_transformer_layers=10).to(self.device)
        
        #print('Load checkpoint: %s' % args.resume)
        loc = 'cuda:0' if self.device == 'cuda' else 'cpu'
        checkpoint = torch.load("/home/landson/PLR/CaRLi-V/ros/src/build/carli_v/scene_flow_models/GMSF/FTD_o.pth", map_location=loc)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()

        self.lidar_point_publisher = self.create_publisher(PointCloud2, '/scene_flow_lidar_points', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/scene_flow_velocity_arrows', 10)


        self.last_pcd = None
        self.last_pcd_timestamp = None

        self.get_logger().info('Scene Flow Node Started')

    def numpy_to_pointcloud2(self, points, frame_id="zed_camera_link", timestamp=None):
        """
        Converts a Nx3 or Nx4 numpy array (XYZ or XYZ+Intensity) into a PointCloud2 ROS2 message.
        :param points: NumPy array of shape (N, 3) or (N, 4) with [x, y, z, intensity]
        :param frame_id: Reference frame of the point cloud
        :return: PointCloud2 ROS message
        """
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="vx", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="vy", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="vz", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="v_full", offset=24, datatype=PointField.FLOAT32, count=1),
        ]

        packed_points = [struct.pack('fffffff', *p) for p in points]
        point_step = 7*4 # 7 floats (x, y, z, vx, vy, vz, v_full) = 28 bytes

        data = b"".join(packed_points)

        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = msg.header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
        msg.height = 1  # Unordered point cloud (single row)
        msg.width = points.shape[0]
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = msg.point_step * msg.width
        msg.data = data
        msg.is_dense = True  # No NaN values

        return msg

    def transform_points(self, points, source_frame, target_frame):
        """
        Take in a pointcloud and transform it from source_frame to target_frame
        Input: points: DxN numpy array, where D is the dimensionality of the points and N is the number of points
        """
        try:
            # Lookup transform from source_frame to target_frame
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, self.get_clock().now(),  rclpy.duration.Duration(seconds=1.0))

            # Convert transform to matrix
            translation = np.array([transform.transform.translation.x,
                                     transform.transform.translation.y,
                                     transform.transform.translation.z])
            rotation = transform.transform.rotation
            quaternion = Quaternion([rotation.w, rotation.x, rotation.y, rotation.z])
            rotation_matrix = quaternion.rotation_matrix

            # Create transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix[:3, :3]
            transformation_matrix[:3, 3] = translation

            # Transform points
            points_homogeneous = np.vstack((points[:3, :], np.ones((1, points.shape[1]))))  # Convert to homogeneous coordinates
            transformed_points = transformation_matrix @ points_homogeneous

            transformed_points = np.vstack((transformed_points[:3, :], points[3, :]))  # Convert back to 3D and keep any additional dimensions

            return transformed_points

        except Exception as e:
            self.get_logger().error(f"Failed to transform points: {e}")
            return None

    def lidar_callback(self, lidar_msg):
        fmt = '<fff f H Q'  # Matches 26 bytes (float32 x3, float32, uint16, uint64)
        point_step = lidar_msg.point_step  # Should be 26 bytes

        points = [struct.unpack(fmt, lidar_msg.data[i:i+point_step]) for i in range(0, len(lidar_msg.data), point_step)]

        points = np.array(points)  # Shape should be (N, 6) with columns: [x, y, z, intensity, ring, timestamp]
        points = self.transform_points(points.T, "hesai_lidar", "zed_camera_link")
        if points is None:
            return
        points = points.T
        points = points[points[:, 0] > 1]
        points = points[points[:, 0] < 4]
        points = points[points[:, 1] > -1.5]
        points = points[points[:, 1] < 1.5]
        points = points[points[:, 2] > -1.25]
        points = points[points[:, 2] < 1.25]
        points = torch.from_numpy(points[:,:3]).unsqueeze(0).to(self.device).float()  # Keep only x, y, z

        if self.last_pcd is not None:
            current_time = Time.from_msg(lidar_msg.header.stamp).nanoseconds / 1e9
            prev_time = Time.from_msg(self.last_pcd_timestamp).nanoseconds / 1e9
            dt = current_time - prev_time

            print(self.last_pcd.shape, points.shape)
            min_pcd_length = min(self.last_pcd.shape[1], points.shape[1])
            results_dict_point = self.model(pc0 = self.last_pcd[:, :min_pcd_length, :], pc1 = points[:, :min_pcd_length, :])
            flow_3d_pred = results_dict_point['flow_preds'][-1].cpu()
            points = points.cpu()
            points_np = np.concatenate((points[0, :, :3], np.zeros((points.shape[1], 4))), axis=1)  # Expand array to be (Nx7) to match desired output
            # self.get_logger().info(f'Published: Points shape {points[mask][:, 3:6].shape}, velocities shape{full_velocities.shape}')
            self.get_logger().info(f"Shape {flow_3d_pred.shape}")
            velocities = flow_3d_pred[0].detach().numpy().T / dt
            points_np[:min_pcd_length, 3:6] = velocities
            # self.get_logger().info(f"velocities {full_velocities}, {(full_velocities == 0).all()}")
            new_msg = self.numpy_to_pointcloud2(points_np)
            self.lidar_point_publisher.publish(new_msg)
            # self.publish_velocity_arrows(points[mask, :3], full_velocities)
            self.get_logger().info(f'Published: Lidar Point Cloud with shape {points.shape}')

        self.last_pcd = points.to(self.device)
        self.last_pcd_timestamp = lidar_msg.header.stamp

    def camera_info_callback(self, msg):
        # Extract the K matrix
        self.K_matrix = msg.k.reshape(3, 3)  # Convert to 3x3 matrix

    def publish_velocity_arrows(self, lidar_points, velocities, frame_id="zed_left_camera_frame"):
        marker_array = MarkerArray()
        for i, (point, v) in enumerate(zip(lidar_points, velocities)):
            if v[3] < 0.1:  # Threshold to avoid noisy low velocities
                continue

            arrow = Marker()
            arrow.header.frame_id = frame_id
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.ns = "lidar_full_velocity"
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            # Start and end point of the arrow
            arrow.points.append(Point(x=point[0], y=point[1], z=point[2]))
            end = [point[0] + 0.2*v[2], point[1] - 0.2*v[0], point[2] - 0.2*v[1]] # transform from (z forward, y left, x down) to (z up, y left, x forward)
            arrow.points.append(Point(x=end[0], y=end[1], z=end[2]))

            # Arrow appearance
            arrow.scale.x = 0.005  # shaft diameter
            arrow.scale.y = 0.02   # head diameter
            arrow.scale.z = 0.02   # head length

            # if v[2] >= 0:
            #     arrow.color.r = 0.0
            #     arrow.color.g = 0.0
            #     arrow.color.b = 1.0
            # else:
            arrow.color.r = 1.0
            arrow.color.g = 0.0
            arrow.color.b = 0.0
            arrow.color.a = 1.0

            arrow.lifetime = rclpy.duration.Duration(seconds=7).to_msg()  # Lifetime of the marker
            marker_array.markers.append(arrow)
        self.marker_publisher.publish(marker_array)


def main():
    rclpy.init()
    node = SceneFlowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
