#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, PointField
import scipy.signal.windows as windows
import struct
from scipy.ndimage import minimum_filter, maximum_filter
from carli_v.utils import cartesian_to_polar, polar_to_cartesian, interpolate_array
import numpy as np
import tf2_ros
from pyquaternion import Quaternion
from collections import deque
from rclpy.time import Time

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
        self.radar_h_fov = 87*2
        self.radar_v_fov = 87*2
        self.target_azimuth_bins = 50
        self.target_elevation_bins = 2

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
        print("Min Doppler:", - self.max_doppler)
        print("Max Doppler:", self.max_doppler - self.doppler_resolution/2)

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

        # LiDAR delay parameters
        self.lidar_buffer = deque(maxlen=50)  # store recent lidar msgs
        self.lidar_delay_sec = 0.1  # delay in seconds (100ms)

    def numpy_to_pointcloud2(self, points, frame_id="vmd3_radar"):
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
            PointField(name="radial_velocity", offset=12, datatype=PointField.FLOAT32, count=1)
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
        self.lidar_buffer.append(msg)

    def process_lidar_with_radar(self, lidar_msg, radar_msg):
        # Process the Lidar data
        try:
            fmt = '<fff f H Q'  # Matches 26 bytes (float32 x3, float32, uint16, uint64)
            point_step = lidar_msg.point_step  # Should be 26 bytes

            points = [struct.unpack(fmt, lidar_msg.data[i:i+point_step]) for i in range(0, len(lidar_msg.data), point_step)]

            points = np.array(points)  # Shape should be (N, 6) with columns: [x, y, z, intensity, ring, timestamp]
            points = points[:,:3]  # Keep only x, y, z

            points = self.transform_points(points.T, 'hesai_lidar', 'vmd3_radar').T  # Transform points to radar frame

            polar_points = cartesian_to_polar(points)  # Convert to polar coordinates

            polar_points = polar_points[polar_points[:, 0] < self.max_range]  # Filter points based on max range

            filtered_points = polar_points[
                (np.abs(polar_points[:, 1]) <= np.radians(self.radar_h_fov / 2)) & 
                (polar_points[:,2] >  np.radians(-self.radar_v_fov / 2 + 10)) &
                (polar_points[:, 2] < np.radians(self.radar_v_fov / 2))
            ]

            velocities, debug_images = self.radar_lidar_fusion(filtered_points, self.velocity_cube[::-1, :, ::-1])  # Perform radar-lidar fusion

            for topic, image in debug_images.items():
                if topic not in self.ros_publishers:
                    self.get_logger().info('Adding new topic publisher ' + topic)
                    self.ros_publishers[topic] = self.create_publisher(Image, topic, 10)
                out_msg = self.create_ros_image(image, radar_msg.header, False)
                self.ros_publishers[topic].publish(out_msg)

            msg = self.numpy_to_pointcloud2(np.concatenate((polar_to_cartesian(filtered_points), velocities[:, np.newaxis]), axis=1))
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
            velocity_cube, output_images = self.process_ADC(adc_complex)
            self.velocity_cube = velocity_cube  # store it for fusion

            radar_time = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
            target_time = radar_time - self.lidar_delay_sec

            closest_lidar = None
            closest_time_diff = float('inf')

            for lidar_msg in self.lidar_buffer:
                lidar_time = Time.from_msg(lidar_msg.header.stamp).nanoseconds / 1e9
                diff = abs(lidar_time - target_time)
                if diff < closest_time_diff and lidar_time <= target_time:
                    closest_lidar = lidar_msg
                    closest_time_diff = diff

            if closest_lidar is not None:
                self.process_lidar_with_radar(closest_lidar, msg)
            else:
                self.get_logger().warn('No suitable delayed LiDAR message found')

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
        padded_range_fft = np.pad(doppler_fft,
                                  pad_width=[(0, 0),
                                             (0, self.target_elevation_bins-doppler_fft.shape[1]),
                                             (0, self.target_azimuth_bins-doppler_fft.shape[2]),
                                             (0, 0)],
                                  mode='constant'
                                )

        # Do 2D FFT on the azimuth-elevation dimension

        window = np.outer(windows.hamming(self.target_elevation_bins), windows.hamming(self.target_azimuth_bins)) # N is the number of samples
        window = window[np.newaxis, :, :, np.newaxis]
        angle_fft = np.fft.fftshift(np.fft.fftshift(np.fft.fft2(padded_range_fft * window, axes=(1,2)), axes=1), axes=2)

        # VELOCITY COMPUTATION
        # Compute magnitude in dB
        magnitude_db = 20 * np.log10(np.abs(angle_fft) + 1e-12)

        # Find max Doppler bin index for each azimuth-range pair
        velocity_cube = np.argmax(magnitude_db, axis=0) - (angle_fft.shape[0]//2)  # shape: (elevation, azimuth, range)

        # Changing velocity cube from bins to m/s
        velocity_cube = velocity_cube * self.doppler_resolution  # Convert to m/s

        # Thresholding — set weak points to center bin (index 16)
        min_intensity_dB = magnitude_db.max() - 5
        velocity_cube[np.max(magnitude_db, axis=0) < min_intensity_dB] = 0  # Assign center bin for weak detections
        
        # np.where(np.max(velocity_cube, axis=-1) < np.abs(np.min(velocity_cube, axis=-1)), np.min(velocity_cube, axis=-1), np.max(velocity_cube, axis=-1))
        # np.where(np.abs(np.max(velocity_cube, axis=-1)) < np.abs(np.min(velocity_cube, axis=-1)), np.min(velocity_cube, axis=-1), np.max(velocity_cube, axis=-1))

        self.velocity_cube = velocity_cube

        # magnitude_azimuth_elevation = np.mean(np.mean(magnitude_db, axis=-1), axis=0)[::-1, ::-1]

        # # Relative thresholding
        # # thresholded_magnitude_azimuth_elevation = self.apply_threshold(magnitude_azimuth_elevation, 3)
        # # Absolute thresholding
        # threshold_db = 19 #db
        # thresholded_magnitude_azimuth_elevation = np.clip(magnitude_azimuth_elevation, threshold_db, None)
        # # Output image
        # output_images['magnitude_azimuth_elevation'] = self.scale_image(thresholded_magnitude_azimuth_elevation)

        # amplitude_azimuth_elevation = (np.mean(np.mean(np.real(angle_fft), axis=-1), axis=0)[::-1, ::-1])
        # amplitude_azimuth_elevation += abs(np.min(amplitude_azimuth_elevation))
        # amplitude_azimuth_elevation = np.log10(amplitude_azimuth_elevation + 1e-12) * 20
        # amplitude_azimuth_elevation += abs(np.min(amplitude_azimuth_elevation))

        # phase_azimuth_elevation = (np.mean(np.mean(np.imag(angle_fft), axis=-1), axis=0)[::-1, ::-1])
        # phase_azimuth_elevation += abs(np.min(phase_azimuth_elevation))
        # phase_azimuth_elevation = np.log10(phase_azimuth_elevation + 1e-12) * 20
        # phase_azimuth_elevation += abs(np.min(phase_azimuth_elevation))

        # print(amplitude_azimuth_elevation)
        # print(phase_azimuth_elevation)

        # thresholded_amplitude_azimuth_elevation = np.clip(amplitude_azimuth_elevation, 0, None)
        # thresholded_phase_azimuth_elevation = np.clip(phase_azimuth_elevation, 0, None)

        # thresholded_amplitude_azimuth_elevation = self.apply_percentage_threshold(amplitude_azimuth_elevation, 0.1)
        # thresholded_phase_azimuth_elevation = self.apply_percentage_threshold(phase_azimuth_elevation, 0.1)

        # # print(phase_azimuth_elevation)
        # output_images['amplitude_azimuth_elevation'] = self.scale_image(amplitude_azimuth_elevation)
        # output_images['phase_azimuth_elevation'] = self.scale_image(phase_azimuth_elevation)

        # output_images['range_azimuth'] = 20 * np.log(np.mean(np.mean(np.abs(angle_fft), axis=1), axis=0))

        # output_images['velocity_azimuth_elevation'] = np.where(np.abs(np.max(velocity_cube, axis=-1)) < np.abs(np.min(velocity_cube, axis=-1)), np.min(velocity_cube, axis=-1), np.max(velocity_cube, axis=-1))[::-1, ::-1] + 16
        # output_images['velocity_range_azimuth'] = np.where(np.abs(np.max(velocity_cube, axis=0)) < np.abs(np.min(velocity_cube, axis=0)), np.min(velocity_cube, axis=0), np.max(velocity_cube, axis=0))[:, ::-1] + 16
        # output_images['velocity_range_elevation'] = np.where(np.abs(np.max(velocity_cube, axis=1)) < np.abs(np.min(velocity_cube, axis=1)), np.min(velocity_cube, axis=1), np.max(velocity_cube, axis=1)) + 16

        # output_images['velocity_range_azimuth_linear'] = self.scale_image(np.mean(np.mean(magnitude_db, axis=1), axis=0)).T[::-1]
        # range_bins = np.arange(0, self.max_range + self.range_resolution, self.range_resolution)  # 1-degree bins
        # # azimuth_bins = np.arcsin(2* np.linspace(-25/50, 25/50, num=50))  # 20 range bins
        # azimuth_bins = np.linspace(-np.radians(self.radar_h_fov/2), np.radians(self.radar_h_fov/2), num=50)

        # range_indices = np.digitize(self.lidar_points[:,0], range_bins)  # Assign ranges to bins
        # angle_indices = np.digitize(self.lidar_points[:,1], self.angle_array)  # Assign angles to bins
        # range_indices[range_indices == 128] = 127  # Ensure the last bin is not out of bounds
        # angle_indices[angle_indices == 50] = 49  # Ensure the last bin is not out of bounds

        # output_images['velocity_range_azimuth'][angle_indices, range_indices] = 255 
        # output_images['velocity_range_azimuth'] = output_images['velocity_range_azimuth'][::-1].T[::-1]

        # angle_indices = np.digitize(self.lidar_points[:,1], azimuth_bins)  # Assign angles to bins
        # angle_indices[angle_indices == 50] = 49  # Ensure the last bin is not out of bounds

        # output_images['velocity_range_azimuth_linear'][range_indices, angle_indices] = 255 
        # output_images['velocity_range_azimuth_linear'] = output_images['velocity_range_azimuth_linear'][::-1,::-1]

        return velocity_cube, output_images
    
    def radar_lidar_fusion(self, lidar_points, velocity_cube):
        """
        Perform radar-lidar fusion.
        Args:
            lidar_points (np.ndarray): Lidar points in polar coordinates. [Nx3]
            velocity_cube (np.ndarray): Radar velocity cube in polar coordinates. (elevation, azimuth, range)
        Returns:
            np.ndarray: Fused point cloud. [Nx4]
            debug_images (dict): Dictionary of intermediate images, where keys are topic names and values are the corresponding images. 
                                    Publishers will be automatically created for each topic if they don't already exist.
        """
        debug_images = {}

        range_bins = np.arange(0, self.max_range, self.range_resolution)  # 1-degree bins
        azimuth_bins = np.arange(-np.radians(self.radar_h_fov/2), np.radians(self.radar_h_fov/2), np.radians(self.radar_h_fov/2)/(self.target_azimuth_bins/2))  # 1-degree bins
        elevation_bins = np.arange(-np.radians(self.radar_v_fov/2), np.radians(self.radar_v_fov/2), np.radians(self.radar_v_fov/2)/(self.target_elevation_bins/2))  # 1-degree bins

        range_indices = np.digitize(lidar_points[:,0], range_bins)  # Assign ranges to bins
        angle_indices = np.digitize(lidar_points[:,1], azimuth_bins)  # Assign angles to bins
        elevation_indices = np.digitize(lidar_points[:,2], elevation_bins)  # Assign angles to bins
        range_indices[range_indices == len(range_bins)] = len(range_bins) - 1  # Ensure the last bin is not out of bounds
        angle_indices[angle_indices == len(azimuth_bins)] = len(azimuth_bins) - 1  # Ensure the last bin is not out of bounds
        elevation_indices[elevation_indices == len(elevation_bins)] = len(elevation_bins) - 1  # Ensure the last bin is not out of bounds

        velocities = np.zeros((lidar_points.shape[0]))
        filtered_min = minimum_filter(velocity_cube, size=(10,10,20))
        filtered_max = maximum_filter(velocity_cube, size=(10,10,20))
        extreme_values = np.where(np.abs(filtered_max) > np.abs(filtered_min), filtered_max, filtered_min)

        debug_images["velocity_readings"] = np.where(np.abs(np.max(velocity_cube, axis=0)) < np.abs(np.min(velocity_cube, axis=0)), np.min(velocity_cube, axis=0), np.max(velocity_cube, axis=0)) + 16
        debug_images["velocity_readings_and_lidar"] = np.where(np.abs(np.max(extreme_values, axis=0)) < np.abs(np.min(extreme_values, axis=0)), np.min(extreme_values, axis=0), np.max(extreme_values, axis=0)) + 16
        debug_images['velocity_readings_and_lidar'][angle_indices, range_indices] = 32
        debug_images['velocity_readings_and_lidar'] = debug_images['velocity_readings_and_lidar'] * 255/32
        debug_images["velocity_readings"] = debug_images['velocity_readings'] * 255/32

        velocities = extreme_values[elevation_indices, angle_indices, range_indices]  # Assign velocities to bins
        return velocities, debug_images
    
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

    # def apply_percentage_threshold(self, image, threshold):
    #     """
    #     Apply a threshold to the image.
    #     Args:
    #         image (np.ndarray): Input image.
    #         threshold (float): Threshold value.
    #     Returns:
    #         np.ndarray: Thresholded image.
    #     """
    #     threshold_db = image.max() - (1-threshold)*image.max()
    #     return np.clip(image, a_min=threshold_db, a_max=None)
    
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
