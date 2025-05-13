import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import scipy.signal.windows as windows
from scipy.ndimage import median_filter
import numpy as np

class RadarProcessor(Node):
    def __init__(self):
        super().__init__('radar_processor')
        self.subscription = self.create_subscription(
            Image, '/radar_ADC', self.image_callback, 10)
        self.rfft_subscriber = self.create_subscription(
            Image, '/radar_RFFT_tx2_rx4', self.RFFT_callback, 10)

        self.ros_publishers = {}

        self.get_logger().info('Radar Processor Node Started')

        self.RFFT_image = None

    def RFFT_callback(self, msg):
        try:
            # Convert ROS Image to NumPy array
            np_image = np.frombuffer(msg.data, dtype=np.int16).reshape((msg.height, msg.width, 2))
            np_image_complex = np_image[..., 0] + 1j * np_image[..., 1]

            self.RFFT_image = np_image_complex

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def image_callback(self, msg):
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
            output_images = self.process_image(adc_complex)

            # Convert processed image back to ROS Image message
            for topic, image in output_images.items():
                if topic not in self.ros_publishers:
                    self.get_logger().info('Adding new topic publisher ' + topic)
                    self.ros_publishers[topic] = self.create_publisher(Image, topic, 10)
                out_msg = self.create_ros_image(image*256/32, msg.header, False)
                self.ros_publishers[topic].publish(out_msg)

            self.get_logger().info('Published processed images')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_image(self, current_radar_frame):
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

        output_images['velocity_azimuth_elevation'] = np.where(np.abs(np.max(velocity_cube, axis=-1)) < np.abs(np.min(velocity_cube, axis=-1)), np.min(velocity_cube, axis=-1), np.max(velocity_cube, axis=-1))[::-1, ::-1] + 16
        output_images['velocity_range_azimuth'] = np.where(np.abs(np.max(velocity_cube, axis=0)) < np.abs(np.min(velocity_cube, axis=0)), np.min(velocity_cube, axis=0), np.max(velocity_cube, axis=0)) + 16
        output_images['velocity_range_elevation'] = np.where(np.abs(np.max(velocity_cube, axis=1)) < np.abs(np.min(velocity_cube, axis=1)), np.min(velocity_cube, axis=1), np.max(velocity_cube, axis=1)) + 16

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
        return scaled_image.astype(np.uint8)

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
