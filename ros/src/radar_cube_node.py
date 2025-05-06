import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class RadarProcessor(Node):
    def __init__(self):
        super().__init__('radar_processor')
        self.subscription = self.create_subscription(
            Image, '/radar_ADC', self.image_callback, 10)
        self.publisher = self.create_publisher(Image, '/range_azimuth', 10)
        self.get_logger().info('Radar Processor Node Started')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to NumPy array
            np_image = np.frombuffer(msg.data, dtype=np.int16).reshape((msg.height, msg.width, 2))
            adc_data = np_image.reshape(32, 128, 3, 4, 2)
            adc_complex = adc_data[..., 0] + 1j * adc_data[..., 1]

            # Process the radar ADC image into range-azimuth representation
            range_azimuth_image = self.process_image(adc_complex)

            # Convert processed image back to ROS Image message
            out_msg = self.create_ros_image(range_azimuth_image.T, msg.header)
            self.publisher.publish(out_msg)
            self.get_logger().info('Published processed image')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_image(self, current_radar_frame):
        # Compute magnitude (range representation)
        # Range FFT
        range_fft = np.fft.fft(current_radar_frame, axis=1)

        # Doppler FFT
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

        # Angle FFT
        # padded_range_fft = np.pad(doppler_fft, pad_width=[(0, 0), (0, 0),  (0, 100), (0, 0)], mode='constant')
        # azimuth_fft = np.fft.fftshift(np.fft.fft(padded_range_fft, axis=2), axes=2)
        # padded_range_fft = np.pad(azimuth_fft, pad_width=[(0, 0), (0, 0),  (0, 0), (0, 100)], mode='constant')
        # azimuth_fft = np.fft.fftshift(np.fft.fft(padded_range_fft, axis=3), axes=3)


        # radar_cube = azimuth_fft
        # Collapse data into a 2D slice, collapsing over the azimuth dimension
        collapsed_range_doppler = 20 * np.log10(np.abs(np.mean(np.mean(doppler_fft, axis=-1), axis=-1)))  # Sum over azimuth dimension

        # plt.figure(figsize=(20, 3))
        # plt.imshow(collapsed_range_doppler, aspect='auto', origin='lower', cmap='viridis')
        # plt.colorbar(label='Magnitude (dB)')
        # plt.xlabel('Range Bin')
        # plt.ylabel('Doppler Bin')
        # plt.title(f'Range-Azimuth Map at Doppler Bin {doppler_bin}')
        # plt.tight_layout()
        # plt.show()

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
