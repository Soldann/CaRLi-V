import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, CameraInfo, PointField, CompressedImage, Image
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import tf2_ros
from collections import deque
import struct
from std_msgs.msg import Float32MultiArray
from pyquaternion import Quaternion
import numpy as np
from carli_v_msgs.msg import StampedFloat32MultiArray

def cal_full_v_in_radar(vx, vy, d, u1, v1, u2, v2, T_c2r, T_c2c, dt):
    # output in radar coordinates
     r11, r12, r13 = T_c2r[0,:3]
     r21, r22, r23 = T_c2r[1,:3]

     ra11, ra12, ra13, btx = T_c2c[0,:]
     ra21, ra22, ra23, bty = T_c2c[1,:]
     ra31, ra32, ra33, btz = T_c2c[2,:]

     A = np.array([[ra11-u2*ra31, ra12-u2*ra32, ra13-u2*ra33], \
                   [ra21-v2*ra31, ra22-v2*ra32, ra23-v2*ra33], \
                   [r11*vx+r21*vy, r12*vx+r22*vy, r13*vx+r23*vy]] )

     b = np.array([[((ra31*u1+ra32*v1+ra33)*u2-(ra11*u1+ra12*v1+ra13))*d+u2*btz-btx],\
                   [((ra31*u1+ra32*v1+ra33)*v2-(ra21*u1+ra22*v1+ra23))*d+v2*btz-bty],\
                   [(vx**2 + vy**2)*dt]])

     x = np.squeeze( np.dot( np.linalg.inv(A), b ) )

     vx_c, vy_c, vz_c = x[0]/dt, x[1]/dt, x[2]/dt

     vr = np.squeeze( np.dot(T_c2r[:3,:3], np.array([[vx_c], [vy_c], [vz_c]])) )

     vx_f, vy_f, vz_f = vr[0], vr[1], vr[2]

     return vx_f, vy_f, vz_f



class RadarFullVelocityNode(Node):
    def __init__(self):
        super().__init__('RadarFullVelocityNode')
        self.optical_flow_subscription = self.create_subscription(
            StampedFloat32MultiArray,
            '/optical_flow_uv_map',
            self.optical_flow_callback,
            10)

        self.lidar_subscriber = self.create_subscription(
            PointCloud2, '/lidar_points_with_radial_velocity', self.lidar_callback, 10)

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/boxi/zed2i/left/camera_info',  # Topic name
            self.camera_info_callback,
            10)

        # Subscribe to the input image topic
        self.image_subscription = self.create_subscription(
            CompressedImage,
            '/boxi/zed2i/left/image_rect_color/compressed',
            self.image_callback,
            10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image = None

        self.lidar_point_publisher = self.create_publisher(PointCloud2, '/lidar_points_with_full_velocity', 10)

        self.bridge = CvBridge()
        self.point_projection_publisher = self.create_publisher(Image, '/projected_points', 10)

        # Image delay parameters
        self.image_buffer = deque(maxlen=200)  # store recent image msgs
        self.uv_image_buffer = deque(maxlen=200)
        self.image_delay = 0  # delay in seconds (100ms)

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

            transformed_points = np.vstack(transformed_points[:3, :], points[3, :])  # Convert back to 3D and keep any additional dimensions

            return transformed_points

        except Exception as e:
            self.get_logger().error(f"Failed to transform points: {e}")
            return

    def image_callback(self, msg):
        self.image_buffer.append(msg)

    def project_points_to_image(self, points, image, camera_matrix):
        # print(points)
        depths = points[2, :]
        min_dist = 1.0 # Distance from the camera below which points are discarded.

        # Project 3D points onto 2D image plane using camera matrix
        points_2d = camera_matrix @ points
        points_2d = points_2d / points_2d[2:3, :].repeat(3, 0)  # Normalize by z (depth)

        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points_2d[0, :] > 1)
        mask = np.logical_and(mask, points_2d[0, :] < image.shape[1] - 1)
        mask = np.logical_and(mask, points_2d[1, :] > 1)
        mask = np.logical_and(mask, points_2d[1, :] < image.shape[0] - 1)
        points_2d = points_2d[:, mask]

        # Convert points to integers in one step to avoid repeated conversions
        points_2d_int = points_2d[:2, :].astype(int)

        # Use numpy to batch process instead of looping
        for x, y in points_2d_int.T:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        return image

    def lidar_callback(self, msg):
        fmt = '<fff f'  # Matches 16 bytes (float32 x3, float32)
        point_step = msg.point_step  # Should be 16 bytes

        points = [struct.unpack(fmt, msg.data[i:i+point_step]) for i in range(0, len(msg.data), point_step)]

        points = np.array(points)  # Shape should be (N, 6) with columns: [x, y, z, radial_velocity]

        if self.K_matrix is not None:
            lidar_time = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
            target_time = lidar_time - self.image_delay

            closest_image = None
            closest_time_diff = float('inf')

            for image_msg in self.image_buffer:
                image_time = Time.from_msg(image_msg.header.stamp).nanoseconds / 1e9
                diff = abs(image_time - target_time)
                # self.get_logger().info(f'diff: {target_time - image_time}, image_time: {image_time}, target_time: {target_time}')
                if diff < closest_time_diff and image_time <= target_time:
                    closest_image = image_msg
                    closest_time_diff = diff

            if closest_image is not None:
                # Convert compressed image to raw OpenCV image
                np_arr = np.frombuffer(closest_image.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                self.image = cv_image
                # # Project lidar points onto the image
                # transformed_pts = self.transform_points(points[:,:3].T, "vmd3_radar", "zed_camera_link")
                # transformed_pts[[0,1,2],:] = transformed_pts[[1,2,0],:]  # Reorder to (x, y, z)
                # transformed_pts[[0,1],:] = -transformed_pts[[0,1],:]  # Invert x and y axis for camera coordinates
                # projected_image = self.project_points_to_image(transformed_pts, cv_image, self.K_matrix)

                # # Publish the projected image
                # projected_msg = self.bridge.cv2_to_imgmsg(projected_image, encoding='bgr8')
                # self.point_projection_publisher.publish(projected_msg)
                # self.get_logger().info('Published: Projected Image')
            else:
                self.get_logger().warn('No suitable delayed IMAGE message found')

        if self.K_matrix is not None:
            lidar_time = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
            target_time = lidar_time - self.image_delay # Optical flow should be delayed just as much as the images

            closest_image_uv = None
            closest_time_diff = float('inf')

            for optical_flow_msg in self.uv_image_buffer:
                image_time = Time.from_msg(optical_flow_msg.stamp).nanoseconds / 1e9
                diff = abs(image_time - target_time)
                # self.get_logger().info(f'diff: {target_time - image_time}, image_time: {image_time}, target_time: {target_time}')
                if diff < closest_time_diff and image_time <= target_time:
                    closest_image_uv = optical_flow_msg
                    closest_time_diff = diff


            if closest_image_uv is not None:
                full_velocities = self.optical_flow_lidar_fusion(closest_image_uv.array, msg)
            else:
                self.get_logger().warn('No suitable delayed OPTICAL FLOW message found')

        points = np.concatenate((points, np.zeros((points.shape[0], 3))), axis=1)  # Add vx, vy, vz

        new_msg = self.numpy_to_pointcloud2(points)
        self.lidar_point_publisher.publish(new_msg)
        self.get_logger().info(f'Published: Lidar Point Cloud with shape {points.shape}')

    def camera_info_callback(self, msg):
        # Extract the K matrix
        self.K_matrix = msg.k.reshape(3, 3)  # Convert to 3x3 matrix

    def optical_flow_lidar_fusion(self, optical_flow_msg, lidar_pcd):
        dims = [dim.size for dim in optical_flow_msg.layout.dim]  # Extract dimensions
        data = np.array(optical_flow_msg.data).reshape(dims)  # Reshape array

        if data.shape[0] != 4:
            self.get_logger().error(f"Expected 4 channels for optical flow (u1, v1, u2, v2), got {data.shape[0]}")
            return

        u1, v1, u2, v2 = data[0], data[1], data[2], data[3] # (u1, v1) are the xy coordinates of the first image, (u2, v2) are the xy coordinates of where they end up in the second image

        transformed_pts = self.transform_points(lidar_pcd.T, "vmd3_radar", "zed_camera_link")
        transformed_pts[[0,1,2],:] = transformed_pts[[1,2,0],:]  # Reorder to (x, y, z)
        transformed_pts[[0,1],:] = -transformed_pts[[0,1],:]  # Invert x and y axis for camera coordinates
        projected_image = self.project_points_to_image(transformed_pts, self.image, self.K_matrix)

        projected_msg = self.bridge.cv2_to_imgmsg(projected_image, encoding='bgr8')
        self.point_projection_publisher.publish(projected_msg)
        self.get_logger().info('Published: Projected Image')
        self.get_logger().info(f'Received: Optical Flow Data with shape {data.shape}')

    def optical_flow_callback(self, msg):
        self.uv_image_buffer.append(msg)


def main():
    rclpy.init()
    node = RadarFullVelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
