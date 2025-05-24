import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge
import cv2
import numpy as np
from NeuFlow import neuflow
from NeuFlow.data_utils import frame_utils
from NeuFlow.backbone_v7 import ConvBlock
import torch
from carli_v_msgs.msg import StampedFloat32MultiArray

def get_cuda_image(image_path):
    image = cv2.imread(image_path)

    image = cv2.resize(image, (768, 432))

    image = torch.from_numpy(image).permute(2, 0, 1).half()
    return image[None].cuda()

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def flow2uv_full(flow, K):
    '''
    Convert flow into pixel coordinates of before and after:
    Returns:
    - x_map: x coordinates of the first image
    - y_map: y coordinates of the first image
    - u_map: new x coordinates of where those pixels end up in the second image
    - v_map: new y coordinates of where those pixels end up in the second image
    '''
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]

    h,w = flow.shape[:2]
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map, y_map = x_map.astype('float32'), y_map.astype('float32')
    x_map += flow[..., 0]
    y_map += flow[..., 1]

    u_map = (x_map - cx) / f
    v_map = (y_map - cy) / f

    # uv_map = np.stack([u_map,v_map], axis=2)

    return x_map, y_map, u_map, v_map


# def downsample_flow(flow_full, downsample_scale, y_cutoff):
#     H, W, nc = flow_full.shape
#     h = int( H / downsample_scale )
#     w = int( W / downsample_scale )

#     x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))

#     x_map_old = np.round( np.clip( x_map * downsample_scale, 0, W-1) ).astype(int).ravel()
#     y_map_old = np.round( np.clip( y_map * downsample_scale, 0, H-1) ).astype(int).ravel()

#     flow_list = []
#     for i in range(nc):
#         flow_list.append(flow_full[y_map_old, x_map_old, i])

#     flow = np.stack(flow_list, axis=1)
#     flow = np.reshape(flow, (h,w,-1))
#     flow = flow[y_cutoff:,...]

#     return flow

def plot_flow_cv2(im, flow, color=(255, 0, 255), step=20):
    """
    Draw optical flow arrows on an image using OpenCV.

    Args:
    - im (numpy array): Image in BGR format.
    - flow (numpy array): Optical flow array (H, W, 2).
    - color (tuple): Arrow color in BGR format. Default is cyan.
    - step (int): Step size for drawing arrows.

    Returns:
    - Annotated image with flow visualization.
    """
    h, w = im.shape[:2]
    x1, y1 = np.meshgrid(np.arange(0, w), np.arange(0, h))

    dx = flow[..., 0]
    dy = flow[..., 1]

    # Convert image to a copy for visualization
    # img_vis = im.copy()
    img_vis = im

    for i in range(0, h, step):
        for j in range(0, w, step):
            pt1 = (int(x1[i, j]), int(y1[i, j]))
            pt2 = (int(x1[i, j] + dx[i, j]), int(y1[i, j] + dy[i, j]))
            cv2.arrowedLine(img_vis, pt1, pt2, color, 1, tipLength=0.3)

    return img_vis


class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('OpticalFlowNode')
        self.bridge = CvBridge()

        # Subscribe to the input image topic
        self.image_subscription = self.create_subscription(
            CompressedImage,
            '/boxi/zed2i/left/image_rect_color/compressed',
            self.image_callback,
            10
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/boxi/zed2i/left/camera_info',  # Topic name
            self.camera_info_callback,
            10)
        self.K_matrix = None

        self.last_image = None

        self.device = torch.device('cuda')
        self.model = neuflow.NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(self.device)

        for m in self.model.modules():
            if type(m) is ConvBlock:
                m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
                m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
                delattr(m, "norm1")  # remove batchnorm
                delattr(m, "norm2")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.model.eval()
        self.model.half()

        # Neuflow parameters
        self.image_width = 768
        self.image_height = 432

        # Publisher for the republished image
        self.optical_flow_publisher = self.create_publisher(Image, '/optical_flow_image', 10)

        self.optical_flow_uv_publisher = self.create_publisher(StampedFloat32MultiArray, '/optical_flow_uv_map', 10)

        self.get_logger().info('Optical Flow Node Started')

    def camera_info_callback(self, msg):
        # Extract the K matrix
        self.K_matrix = msg.k.reshape(3, 3)  # Convert to 3x3 matrix
        # self.get_logger().info(f'Intrinsic Camera Matrix (K):\n{self.K_matrix}')

    def image_callback(self, msg):
        # Convert compressed image to raw OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR).astype(np.float32)

        if self.last_image is None or self.K_matrix is None:
            # If this is the first image or camera info is not available, just store the image
            self.last_image = cv_image
            return
        else:
            # Compute optical flow using NeuFlow
            im1 = torch.from_numpy(self.last_image).permute(2, 0, 1)
            im2 = torch.from_numpy(cv_image).permute(2, 0, 1)
            im1 = im1[None].to(self.device)
            im2 = im2[None].to(self.device)


            padder = frame_utils.InputPadder(im1.shape, mode='kitti', padding_factor=16)
            im1, im2 = padder.pad(im1, im2)
            self.model.init_bhwd(im1.shape[0], im1.shape[-2], im1.shape[-1], self.device)
            with torch.no_grad():
                flow = self.model(im1.half(), im2.half())[-1][0]
                flow = flow.permute(1,2,0).cpu().numpy()[:,...]

            u_1, v_1, u_2, v_2 = flow2uv_full(flow, self.K_matrix)

            plotted_flow = plot_flow_cv2(self.last_image.astype(np.uint8), flow, color=(255, 0, 255), step=20)

            # Convert BGR8 OpenCV image to ROS 2 Image message
            ros_image = self.bridge.cv2_to_imgmsg(plotted_flow, encoding='bgr8')
            self.optical_flow_publisher.publish(ros_image)

            # publish uv_map
            uv_map_msg = StampedFloat32MultiArray()
            uv_data = np.array([u_1, v_1, u_2, v_2])
            uv_map_msg.stamp = msg.header.stamp 
            uv_map_msg.array.data = uv_data.flatten().tolist()
            uv_map_msg.array.layout.dim.append(MultiArrayDimension(label="depth", size=uv_data.shape[0], stride=uv_data.shape[1] * uv_data.shape[2]))
            uv_map_msg.array.layout.dim.append(MultiArrayDimension(label="rows", size=uv_data.shape[1], stride=uv_data.shape[2]))
            uv_map_msg.array.layout.dim.append(MultiArrayDimension(label="cols", size=uv_data.shape[2], stride=1))
            self.optical_flow_uv_publisher.publish(uv_map_msg)

            self.get_logger().info('Converted and published optical flow image.')

            # Update the last image
            self.last_image = cv_image

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
