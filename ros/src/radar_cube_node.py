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
                out_msg = self.create_ros_image(image, msg.header)
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
        padded_range_fft = np.pad(doppler_fft, pad_width=[(0, 0), (0, 18),  (0, 42), (0, 0)], mode='constant')

        # Do 1D FFT on the azimuth dimension

        window = windows.taylor(50, 6, 55) # N is the number of samples
        window = window[np.newaxis, np.newaxis, :, np.newaxis]
        azimuth_fft = np.fft.fftshift(np.fft.fft(padded_range_fft * window, axis=2), axes=2)

        window = windows.taylor(20, 6, 55) # N is the number of samples
        window = window[np.newaxis, :, np.newaxis, np.newaxis]
        elevation_fft = np.fft.fftshift(np.fft.fft(azimuth_fft * window, axis=1), axes=1)

        # Do 2D FFT on the azimuth-elevation dimension

        window = np.outer(windows.hamming(20), windows.hamming(50)) # N is the number of samples
        window = window[np.newaxis, :, :, np.newaxis]
        azimuth_fft_2d = np.fft.fftshift(np.fft.fft2(padded_range_fft * window, axes=(1,2)), axes=(1,2))

        # radar_cube = azimuth_fft

        # Collapse data into a 2D slice for visualization


        # VELOCITY PLOT
        # # Step 1: Compute magnitude in dB
        # magnitude_db = np.mean(20 * np.log10(np.abs(azimuth_fft) + 1e-12), axis=1)

        # # Step 3: Find max Doppler bin index for each azimuth-range pair
        # collapsed_doppler = np.argmax(magnitude_db, axis=0)  # shape: (azimuth, range)

        # # Step 4: Compute total intensity at max bin (for thresholding)
        # # max_magnitude_linear = np.max(np.mean(np.abs(azimuth_fft), axis=1), axis=0)  # shape: (azimuth, range)

        # output_images['/velocity'] = collapsed_doppler.copy()

        # # Step 5: Thresholding — set weak points to center bin (index 16)
        # min_intensity_dB = magnitude_db.max() - 60 
        # # min_intensity_linear = 10**(min_intensity_dB / 20)
        # collapsed_doppler[np.max(magnitude_db, axis=0) < min_intensity_dB] = 16  # Assign center bin for weak detections
        # output_images['/velocity_thresh'] = collapsed_doppler

        # RANGE-DOPPLER PLOT
        # collapsed_range_doppler = np.mean(np.mean(20 * np.log10(np.abs(doppler_fft) + 1), axis=1), axis=1).T[::-1]  # Range-Doppler Plot
        # output_images['/range_doppler'] = collapsed_range_doppler

        # # Threshold: Clip background below a dynamic floor (e.g., max - 60 dB)
        # threshold_db = collapsed_range_doppler.max() - 45
        # collapsed_range_doppler = np.clip(collapsed_range_doppler, a_min=threshold_db, a_max=None)
        # output_images['/range_doppler_thresholding'] = collapsed_range_doppler

        # # Median filter to suppress speckle noise
        # output_images['/range_doppler_thresholding_median'] = median_filter(collapsed_range_doppler, size=(3, 3))  # You can try (5, 5) too


        output_images["/range_azimuth"] = np.mean(np.mean(20 * np.log10(np.abs(azimuth_fft)+1e-6), axis=1), axis=0)
        output_images["/range_azimuth_21d"] = (np.mean(np.mean(20 * np.log10(np.abs(elevation_fft)+1e-6), axis=1), axis=0))
        output_images["/range_azimuth_2d"] = (np.mean(np.mean(20 * np.log10(np.abs(azimuth_fft_2d)+1e-6), axis=1), axis=0))
        print(np.max(output_images["/range_azimuth"]), np.max(output_images["/range_azimuth_21d"]), np.max(output_images["/range_azimuth_2d"]))
        output_images["/range_azimuth_diff"] = np.abs(output_images["/range_azimuth"] - output_images["/range_azimuth_2d"])
        output_images["/range_azimuth_1d21d_diff"] = np.abs(output_images["/range_azimuth"] - output_images["/range_azimuth_21d"])
        output_images["/range_azimuth_2d_diff"] = np.abs(output_images["/range_azimuth_21d"] - output_images["/range_azimuth_2d"])

        # output_images["/range_azimuth"] = np.mean(np.mean(20 * np.log10(np.abs(azimuth_fft)+1e-6), axis=1), axis=0)
        # output_images["/azimuth_doppler"] = np.mean(np.mean(20 * np.log10(np.abs(azimuth_fft)+1e-6), axis=1), axis=-1)
        # output_images["/azimuth_elevation"] = np.mean(np.mean(20 * np.log10(np.abs(azimuth_fft)), axis=-1), axis=0)
        # output_images["/range_elevation"] = np.mean(np.mean(20 * np.log10(np.abs(azimuth_fft)), axis=2), axis=0)
        # output_images["/doppler_elevation"] = np.mean(np.mean(20 * np.log10(np.abs(azimuth_fft)), axis=-1), axis=-1).T


        # if self.RFFT_image is not None:
        #     output_images["/RFFT"] = np.abs(doppler_fft[:, 1, 0, :].T[::-1])
        #     output_images["/RFFT_RAW"] = np.abs(self.RFFT_image)
        #     print(np.isclose(output_images["/RFFT"], output_images["/RFFT_RAW"]).all())

        return output_images
    
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

    def create_ros_image(self, np_array, header):
        msg = Image()
        msg.header = header
        msg.height, msg.width = np_array.shape
        msg.encoding = '16UC1'
        msg.is_bigendian = 0
        msg.step = msg.width
        msg.data = self.scale_image(np_array).astype(np.uint16).tobytes()
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = RadarProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
