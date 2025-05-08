import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import scipy.signal.windows as windows
import numpy as np

class RadarProcessor(Node):
    def __init__(self):
        super().__init__('radar_processor')
        self.subscription = self.create_subscription(
            Image, '/radar_ADC', self.image_callback, 10)
        self.rfft_subscriber = self.create_subscription(
            Image, '/radar_RFFT_tx2_rx4', self.RFFT_callback, 10)
        self.publisher = self.create_publisher(Image, '/range_azimuth', 10)
        self.rfft_publisher = self.create_publisher(Image, '/RFFT', 10)

        self.rfft_raw_publisher = self.create_publisher(Image, '/RFFT_RAW', 10)

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
            range_azimuth_image = self.process_image(adc_complex, msg)

            # Convert processed image back to ROS Image message
            out_msg = self.create_ros_image(range_azimuth_image.T[::-1], msg.header)
            self.publisher.publish(out_msg)
            self.get_logger().info('Published processed image')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_image(self, current_radar_frame, msg):
        # Compute magnitude (range representation)
        # Range FFT
        window = windows.gaussian(128, 1/4*128)  # N is the number of samples
        # Reshape window for broadcasting
        window = window[np.newaxis, np.newaxis, np.newaxis, :]
        range_fft = np.fft.fft(current_radar_frame * window, axis=3)


        window = windows.gaussian(32, 1/4*32)  # N is the number of samples
        # Reshape window for broadcasting
        window = window[:, np.newaxis, np.newaxis, np.newaxis]
        # Doppler FFT
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft * window, axis=0), axes=0)

        # Angle FFT
        padded_range_fft = np.pad(doppler_fft, pad_width=[(0, 0), (0, 18),  (0, 42), (0, 0)], mode='constant')
        window = windows.taylor(50, 6, 55) # N is the number of samples
        window = window[np.newaxis, np.newaxis, :, np.newaxis]
        azimuth_fft = np.fft.fftshift(np.fft.fft(padded_range_fft * window, axis=2), axes=2)
        # window = np.outer(windows.hamming(20), windows.hamming(50)) # N is the number of samples
        # window = window[np.newaxis, :, :, np.newaxis]
        # azimuth_fft = np.fft.fftshift(np.fft.fft2(padded_range_fft * window, axes=(1,2)), axes=(1,2))
        # padded_range_fft = np.pad(azimuth_fft, pad_width=[(0, 0), (0, 0),  (0, 10), (0, 0)], mode='constant')
        # azimuth_fft = np.fft.fftshift(np.fft.fft(padded_range_fft, axis=3), axes=3)


        # radar_cube = azimuth_fft
        # Collapse data into a 2D slice, collapsing over the azimuth dimension
        # collapsed_range_doppler = np.mean(np.mean(20 * np.log10(np.abs(doppler_fft)), axis=1), axis=1)  # Sum over azimuth dimension
        # collapsed_range_doppler = np.mean(np.mean(20 * np.log10(np.abs(doppler_fft)+1), axis=1), axis=1)  # Sum over azimuth dimension
        
        # collapsed_range_doppler = np.mean(np.mean(20 * np.log10(np.abs(azimuth_fft)), axis=-1), axis=0).T

        collapsed_range_doppler = np.flip(np.mean(np.mean(20 * np.log10(np.abs(azimuth_fft)+1), axis=1), axis=0), axis=1)  # Sum over azimuth dimension

        if self.RFFT_image is not None:
            out_msg = self.create_ros_image(np.abs(doppler_fft[:, 1, 0, :].T[::-1]), msg.header)
            self.rfft_publisher.publish(out_msg)
            out_msg = self.create_ros_image(np.abs(self.RFFT_image), msg.header)
            self.rfft_raw_publisher.publish(out_msg)
            print(np.isclose(np.abs(doppler_fft[:, 1, 0, :].T[::-1]), np.abs(self.RFFT_image)).all())

        # Normalize for visualization
        return collapsed_range_doppler

    def create_ros_image(self, np_array, header):
        msg = Image()
        msg.header = header
        msg.height, msg.width = np_array.shape
        msg.encoding = '16UC1'
        msg.is_bigendian = 0
        msg.step = msg.width
        msg.data = np_array.astype(np.int16).tobytes()
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = RadarProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
