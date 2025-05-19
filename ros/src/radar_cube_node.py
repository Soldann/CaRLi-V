import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, PointField
import scipy.signal.windows as windows
from scipy.ndimage import median_filter
import struct
from utils import cartesian_to_polar, polar_to_cartesian, interpolate_array
import numpy as np
import tf2_ros
from pyquaternion import Quaternion


class RadarProcessor(Node):
    def __init__(self):
        super().__init__('radar_processor')
        self.adc_subscriber = self.create_subscription(
            Image, '/radar_ADC', self.radar_callback, 10)
        self.lidar_subscriber = self.create_subscription(
            PointCloud2, '/lidar_points', self.lidar_callback, 10)

        self.ros_publishers = {}

        self.get_logger().info('Radar Processor Node Started')

        self.lidar_point_publisher = self.create_publisher(PointCloud2, '/lidar_points_with_radial_velocity', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Compute RADAR config
        self.radar_h_fov = 60*2
        self.radar_v_fov = 30

        self.radar_config = {
            'num_tx': 3,
            'num_rx': 4,
            'num_adc_samples': 128,
            'num_chirps': 32,
            'adc_sample_rate': 2e3,  # KHz
            'chirp_slope': 49.97,  # MHz/us
            'sweep_repetition_time': 147,  # us
            'start_freq': 60.095,  # GHz
        }

        # speed of wave propagation
        self.c = 299792458 # m/s
        # compute ADC sample period T_c in msec
        adc_sample_period = 1 / self.radar_config['adc_sample_rate'] * self.radar_config['num_adc_samples'] # msec
        # next compute the Bandwidth in GHz
        bandwidth = adc_sample_period * self.radar_config['chirp_slope'] # GHz
        # Coompute range resolution in meters
        self.range_resolution = self.c / (2 * (bandwidth * 1e9)) # meters
        # Compute max range
        self.max_range = self.range_resolution * self.radar_config['num_adc_samples'] # meters

        # compute center frequency in GHz
        center_freq = (self.radar_config['start_freq'] + bandwidth/2) # GHz
        # compute center wavelength 
        lmbda = self.c/(center_freq * 1e9) # meters
        # interval for an entire chirp including deadtime
        chirp_interval = self.radar_config['sweep_repetition_time'] * 1e-6 # seconds
        # compute doppler resolution
        self.doppler_resolution = lmbda / (2 * self.radar_config['num_chirps'] * self.radar_config['num_tx'] * chirp_interval) # m/s
        # compute max doppler reading
        self.max_doppler = self.radar_config['num_chirps']  * self.doppler_resolution / 2 # m/s


        # Angle resolution Calculation
        Na = 8                      # number of antennas
        lambda_m = 0.00486           # 77 GHz -> 3.9 mm wavelength
        la = lambda_m / 2           # spacing
        N_bins = Na

        # Bin indices centered around zero
        k_centered = np.linspace(-N_bins/2 + 0.5, N_bins/2 - 0.5, N_bins)

        # Spatial frequencies
        f_k = k_centered / (Na * la)

        # Corresponding angles
        sin_theta = lambda_m * f_k
        valid = np.abs(sin_theta) <= 1
        theta_rad = np.full(N_bins, np.nan)
        theta_rad[valid] = np.arcsin(sin_theta[valid])

        self.angle_array = interpolate_array(theta_rad, 6)


    def numpy_to_pointcloud2(self, points, frame_id="zed_camera_link"):
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
            PointField(name="velocity", offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        packed_points = [struct.pack('ffff', *p) for p in points]
        point_step = 16

        data = b"".join(packed_points)
        
        msg = PointCloud2()
        msg.header.frame_id = frame_id
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

            # print(rotation_matrix)
            # print(translation)

            # rotation_matrix = np.array([[ 0.99997086, 0.00348797,  0.00679117],
            #                             [ 0.00672519, 0.01859214, -0.99980453],
            #                             [-0.00361355, 0.99982107,  0.01856814],])
            # translation = np.array([ 0.01190664, -0.32498627,-0.75900204])
            

            # Create transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix[:3, :3]
            transformation_matrix[:3, 3] = translation

            # Transform points
            points_homogeneous = np.vstack((points, np.ones((1, points.shape[1]))))  # Convert to homogeneous coordinates
            transformed_points = transformation_matrix @ points_homogeneous
            return transformed_points[:3, :]  # Convert back to 3D coordinates

        except Exception as e:
            self.get_logger().error(f"Failed to transform points: {e}")
            return points


    def lidar_callback(self, msg):
        # Process the Lidar data
        try:
            lidar_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            lidar_points = np.array(list(lidar_data))

            fmt = '<fff f H Q'  # Matches 26 bytes (float32 x3, float32, uint16, uint64)
            point_step = msg.point_step  # Should be 26 bytes

            points = [struct.unpack(fmt, msg.data[i:i+point_step]) for i in range(0, len(msg.data), point_step)]

            points = np.array(points)  # Shape should be (N, 6) with columns: [x, y, z, intensity, ring, timestamp]
            points = points[:,:3]  # Keep only x, y, z

            points = self.transform_points(points.T, 'hesai_lidar', 'vmd3_radar').T  # Transform points to radar frame

            points = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)  # Add velocity column
            points = cartesian_to_polar(points)  # Convert to polar coordinates

            points = points[points[:, 0] < self.max_range]  # Filter points based on max range
            # points = points[points[:, 1] < 2*np.pi/6]  # Filter points based on max elevation angle
            # points = points[points[:, 1] > 2*np.pi/6]  # Filter points based on max elevation angle
            # points = points[points[:, 2] < np.pi/12]  # Filter points based on max elevation angle
            # points = points[points[:, 2] > np.pi/12]  # Filter points based on max elevation angle

            filtered_points = points[
                (np.abs(points[:, 1]) <= np.radians(self.radar_h_fov / 2)) & 
                (points[:,2] >  np.radians(-self.radar_v_fov / 2 + 10)) &
                (points[:, 2] < np.radians(self.radar_v_fov / 2))
            ]

            self.lidar_points = filtered_points.copy()
            # print("There are {} points in the lidar frame".format(self.lidar_points.shape[0]))
            points = polar_to_cartesian(filtered_points)  # Convert back to cartesian coordinates
            points = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)  # Add velocity column
            # print("There are now {} points in the lidar frame".format(points.shape[0]))
            msg = self.numpy_to_pointcloud2(points)
            self.lidar_point_publisher.publish(msg)
            self.get_logger().info('Published transformed Lidar points')

        except Exception as e:
            self.get_logger().error(f'Error processing Lidar data: {str(e)}')

    def radar_callback(self, msg):
        try:
            # Convert ROS Image to NumPy array
            np_image = np.frombuffer(msg.data, dtype=np.int16).reshape((msg.height, msg.width, 2))
            adc_data = np_image.reshape(32, 12, 128, 2)

            # Assuming adc_data has shape (32, 12, 128, 2)
            new_data = np.zeros((32, 2, 8, 128, 2))  # Create new structured array

            # Extract relevant elements
            new_data[:, 0, :, :, :] = adc_data[:, [0,1,2,3,8,9,10,11], :, :]  # First row
            new_data[:, 1, 2:6, :, :] = adc_data[:, [4,5,6,7], :, :]  # Second row (padded in middle)

            # Shape is now (32, 2, 8, 128, 2)
            print("New array shape:", new_data.shape)
            adc_data = new_data

            adc_complex = adc_data[..., 0] + 1j * adc_data[..., 1]

            # Process the radar ADC image into range-azimuth representation
            output_images = self.process_ADC(adc_complex)

            # Convert processed image back to ROS Image message
            for topic, image in output_images.items():
                if topic not in self.ros_publishers:
                    self.get_logger().info('Adding new topic publisher ' + topic)
                    self.ros_publishers[topic] = self.create_publisher(Image, topic, 10)
                out_msg = self.create_ros_image(image, msg.header, False)
                self.ros_publishers[topic].publish(out_msg)

            self.get_logger().info('Published processed images')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_ADC(self, current_radar_frame):
        """
        Take Radar ADC data and process it into various images.
        Args:
            current_radar_frame (np.ndarray): Radar ADC data of dimensions (doppler, elevation, azimuth, range)
        Returns:
            output_images (dict): Dictionary of processed images, where keys are topic names and values are the corresponding images. 
                                    Publishers will be automatically created for each topic if they don't already exist.
        """
        output_images = {}

        # Compute magnitude (range representation)
        # Range FFT
        # window = windows.gaussian(128, 1/4*128)  # N is the number of samples
        window = windows.hann(128)  # N is the number of samples
        # Reshape window for broadcasting
        window = window[np.newaxis, np.newaxis, np.newaxis, :]
        range_fft = np.fft.fft(current_radar_frame * window, axis=3)


        # window = windows.gaussian(32, 1/4*32)  # N is the number of samples
        window = windows.hann(32)  # N is the number of samples
        # Reshape window for broadcasting
        window = window[:, np.newaxis, np.newaxis, np.newaxis]
        # Doppler FFT
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft * window, axis=0), axes=0)

        # Angle FFT
        target_azimuth_bins = 50
        target_elevation_bins = 20
        padded_range_fft = np.pad(doppler_fft,
                                  pad_width=[(0, 0),
                                             (0, target_elevation_bins-doppler_fft.shape[1]),
                                             (0, target_azimuth_bins-doppler_fft.shape[2]),
                                             (0, 0)],
                                  mode='constant'
                                )

        # Do 2D FFT on the azimuth-elevation dimension

        window = np.outer(windows.hamming(target_elevation_bins), windows.hamming(target_azimuth_bins)) # N is the number of samples
        window = window[np.newaxis, :, :, np.newaxis]
        angle_fft = np.fft.fftshift(np.fft.fftshift(np.fft.fft2(padded_range_fft * window, axes=(1,2)), axes=1), axes=2)

        # VELOCITY COMPUTATION
        # Step 1: Compute magnitude in dB
        magnitude_db = 20 * np.log10(np.abs(angle_fft) + 1e-12)

        # Step 2: Find max Doppler bin index for each azimuth-range pair
        velocity_cube = np.argmax(magnitude_db, axis=0) - 16  # shape: (elevation, azimuth, range)

        # Step 3: Thresholding — set weak points to center bin (index 16)
        min_intensity_dB = magnitude_db.max() - 5
        velocity_cube[np.max(magnitude_db, axis=0) < min_intensity_dB] = 0  # Assign center bin for weak detections
        
        # np.where(np.max(velocity_cube, axis=-1) < np.abs(np.min(velocity_cube, axis=-1)), np.min(velocity_cube, axis=-1), np.max(velocity_cube, axis=-1))
        # np.where(np.abs(np.max(velocity_cube, axis=-1)) < np.abs(np.min(velocity_cube, axis=-1)), np.min(velocity_cube, axis=-1), np.max(velocity_cube, axis=-1))

        self.velocity_cube = velocity_cube

        # output_images['range_azimuth'] = 20 * np.log(np.mean(np.mean(np.abs(angle_fft), axis=1), axis=0))

        output_images['velocity_azimuth_elevation'] = np.where(np.abs(np.max(velocity_cube, axis=-1)) < np.abs(np.min(velocity_cube, axis=-1)), np.min(velocity_cube, axis=-1), np.max(velocity_cube, axis=-1))[::-1, ::-1] + 16
        output_images['velocity_range_azimuth'] = np.where(np.abs(np.max(velocity_cube, axis=0)) < np.abs(np.min(velocity_cube, axis=0)), np.min(velocity_cube, axis=0), np.max(velocity_cube, axis=0))[:, ::-1] + 16
        output_images['velocity_range_elevation'] = np.where(np.abs(np.max(velocity_cube, axis=1)) < np.abs(np.min(velocity_cube, axis=1)), np.min(velocity_cube, axis=1), np.max(velocity_cube, axis=1)) + 16

        output_images['velocity_range_azimuth_linear'] = self.scale_image(np.mean(np.mean(magnitude_db, axis=1), axis=0)).T[::-1]
        range_bins = np.arange(0, self.max_range + self.range_resolution, self.range_resolution)  # 1-degree bins
        # angle_bins = np.arcsin(2* np.linspace(-25/50, 25/50, num=50))  # 20 range bins
        angle_bins = np.linspace(-np.radians(self.radar_h_fov/2), np.radians(self.radar_h_fov/2), num=50)

        range_indices = np.digitize(self.lidar_points[:,0], range_bins)  # Assign ranges to bins
        angle_indices = np.digitize(self.lidar_points[:,1], self.angle_array)  # Assign angles to bins
        range_indices[range_indices == 128] = 127  # Ensure the last bin is not out of bounds
        angle_indices[angle_indices == 50] = 49  # Ensure the last bin is not out of bounds

        output_images['velocity_range_azimuth'][angle_indices, range_indices] = 255 
        output_images['velocity_range_azimuth'] = output_images['velocity_range_azimuth'][::-1].T[::-1]

        angle_indices = np.digitize(self.lidar_points[:,1], angle_bins)  # Assign angles to bins
        angle_indices[angle_indices == 50] = 49  # Ensure the last bin is not out of bounds

        output_images['velocity_range_azimuth_linear'][range_indices, angle_indices] = 255 
        output_images['velocity_range_azimuth_linear'] = output_images['velocity_range_azimuth_linear'][::-1,::-1]

        return output_images
    
    def apply_threshold(self, image, threshold):
        """
        Apply a threshold to the image.
        Args:
            image (np.ndarray): Input image.
            threshold (float): Threshold value.
        Returns:
            np.ndarray: Thresholded image.
        """
        threshold_db = image.max() - threshold
        return np.clip(image, a_min=threshold_db, a_max=None)
    
    def scale_image(self, image):
        """
        Scale the image to 0-255 range for visualization.
        Args:
            image (np.ndarray): Input image.
        Returns:
            np.ndarray: Scaled image.
        """
        min_val = np.min(image)
        max_val = np.max(image)
        scaled_image = (image - min_val) / (max_val - min_val) * 255
        return scaled_image

    def create_ros_image(self, np_array, header, scale_image=True):
        msg = Image()
        msg.header = header
        msg.height, msg.width = np_array.shape
        msg.encoding = 'mono8'
        msg.is_bigendian = 0
        msg.step = msg.width
        if scale_image:
            msg.data = self.scale_image(np_array).tobytes()
        else:
            msg.data = np_array.astype(np.uint8).tobytes()
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = RadarProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
