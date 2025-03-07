import abc
import rclpy
import cv2
import numpy as np
import config
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Lock
from utils import pose_inv
from lightglue import LightGlue, SIFT, SuperPoint
from lightglue.utils import rbd
from ik_solver import InverseKinematicsSolver

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float, device="cuda")

class CartesianVisualServoer(Node, abc.ABC):
    def __init__(self, use_depth=False, silent=False):
        super().__init__('visual_servoer')
        
        self.bridge = CvBridge()
        self.lock = Lock()
        self.silent = silent
        
        # Track image timestamps to detect new images
        self.last_rgb_stamp = None
        self.last_depth_stamp = None
        self.new_rgb_received = False
        self.new_depth_received = False
        
        self.images = {
            "rgb": None,
            "depth": None
        }
        self.use_depth = use_depth
        
        # Higher queue size to avoid dropping messages
        self.rgb_subscriber = self.create_subscription(
            Image, config.d405_rgb_topic_name, self.rgb_image_callback, 10)
        
        if self.use_depth:
            self.depth_subscriber = self.create_subscription(
                Image, config.d405_depth_topic_name, self.depth_image_callback, 10)

    def log_info(self, message):
        """Log info messages only if not in silent mode."""
        if not self.silent:
            self.get_logger().info(message)

    def log_warn(self, message):
        """Always log warnings, even in silent mode."""
        self.get_logger().warn(message)
            
    def log_error(self, message):
        """Always log errors, even in silent mode."""
        self.get_logger().error(message)

    def rgb_image_callback(self, msg):
        with self.lock:
            try:
                # Store the image
                self.images["rgb"] = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                
                # Mark that we've received a new RGB image
                self.new_rgb_received = True
                self.last_rgb_stamp = msg.header.stamp
                
                self.log_info("RGB Image received and stored.")
            except Exception as e:
                self.log_error(f"Error in rgb_image_callback: {e}")

    def depth_image_callback(self, msg):
        with self.lock:
            try:
                # Store the image
                self.images["depth"] = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                
                # Mark that we've received a new depth image
                self.new_depth_received = True
                self.last_depth_stamp = msg.header.stamp
                
                self.log_info("Depth Image received and stored.")
            except Exception as e:
                self.log_error(f"Error in depth_image_callback: {e}")

    def observe(self, timeout=5.0):
        """Get the latest images by spinning the ROS loop until we have new data.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            tuple: (RGB image, depth image) or (None, None) if timeout occurs
        """
        self.log_info("Observe called, waiting for new images...")
        
        # Mark current images as "not new"
        with self.lock:
            self.new_rgb_received = False
            if self.use_depth:
                self.new_depth_received = False
        
        # Start time for timeout tracking
        start_time = self.get_clock().now()
        timeout_duration = rclpy.duration.Duration(seconds=timeout)
        
        # Wait for new images by spinning the node
        while rclpy.ok():
            # Check if we've exceeded the timeout
            elapsed = self.get_clock().now() - start_time
            if elapsed > timeout_duration:
                self.log_warn(f"Timeout after {timeout} seconds while waiting for images.")
                return (None, None)
            
            # Spin once to process callbacks
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if we've received the images we need
            with self.lock:
                if self.new_rgb_received and (not self.use_depth or self.new_depth_received):
                    # We have all the images we need
                    self.log_info("All required images received.")
                    
                    # Make copies to avoid race conditions
                    rgb_copy = self.images["rgb"].copy() if self.images["rgb"] is not None else None
                    depth_copy = self.images["depth"].copy() if self.use_depth and self.images["depth"] is not None else None
                    
                    return (rgb_copy, depth_copy)
        
        # If we get here, rclpy is no longer ok
        self.log_error("ROS context is no longer valid")
        return (None, None)
    
    def run(self):
        """Run the visual servoing loop."""
        pass


class LightGlueVisualServoer(CartesianVisualServoer):
    def __init__(self, rgb_ref, seg_ref, use_depth=False, features='superpoint', silent=False):
        super().__init__(use_depth=use_depth, silent=silent)

        if features == 'sift':
            self.extractor_sift = SIFT(backend='pycolmap', max_num_keypoints=1024).eval().cuda()
            self.matcher_sift = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()
            self.feats0_sift = self.extractor_sift.extract(numpy_image_to_torch(rgb_ref))
        elif features == 'superpoint':
            self.extractor_sp = SuperPoint(max_num_keypoints=1024).eval().cuda()
            self.matcher_sp = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda() 
            self.feats0_sp = self.extractor_sp.extract(numpy_image_to_torch(rgb_ref))
        else:
            raise NotImplementedError
        
        self.features = features
        self.rgb_ref = rgb_ref
        self.seg_ref = seg_ref

    def match_lightglue(self, filter_seg=True):
        live_rgb, live_depth = self.observe()

        if live_rgb is None:
            self.log_error("No RGB image received. Check camera and topics.")
            return None, None, None

        try:
            if self.features == 'sift':
                feats1 = self.extractor_sift.extract(numpy_image_to_torch(live_rgb))
                matches01 = self.matcher_sift({'image0': self.feats0_sift, 'image1': feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [self.feats0_sift, feats1, matches01]]
            elif self.features == 'superpoint':
                feats1 = self.extractor_sp.extract(numpy_image_to_torch(live_rgb))
                matches01 = self.matcher_sp({'image0': self.feats0_sp, 'image1': feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [self.feats0_sp, feats1, matches01]]

            matches, scores = matches01['matches'], matches01['scores']
            
            if matches.shape[0] == 0:
                self.log_warn("No matches found between reference and current frame.")
                return None, None, None
            
            mkpts_0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
            mkpts_1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

            if filter_seg:
                coords = mkpts_0.astype(int)
                # Boundary check to avoid index errors
                valid_coords = (coords[:, 0] >= 0) & (coords[:, 0] < self.seg_ref.shape[1]) & \
                              (coords[:, 1] >= 0) & (coords[:, 1] < self.seg_ref.shape[0])
                
                if not np.any(valid_coords):
                    self.log_warn("No valid coordinates found within segmentation mask bounds.")
                    return None, None, None
                    
                coords = coords[valid_coords]
                mask = np.zeros(matches.shape[0], dtype=bool)
                mask[valid_coords] = self.seg_ref[coords[:, 1], coords[:, 0]].astype(bool)
                
                mkpts_0 = mkpts_0[mask]
                mkpts_1 = mkpts_1[mask]
                
                if mask.sum() == 0:
                    self.log_warn("No keypoints found within segmentation mask.")
                    return None, None, None
                    
                valid_indices = np.where(mask)[0]
                scores = scores[valid_indices]

            scores = scores.detach().cpu().numpy()[..., None]
            mkpts_scores_0 = np.concatenate((mkpts_0, scores), axis=1)
            mkpts_scores_1 = np.concatenate((mkpts_1, scores), axis=1)
            
            self.log_info(f"Matched {len(mkpts_scores_0)} keypoints after filtering.")
            return mkpts_scores_0, mkpts_scores_1, live_depth
            
        except Exception as e:
            self.log_error(f"Error in match_lightglue: {e}")
            return None, None, None


if __name__ == "__main__":
    rclpy.init()
    
    dir = "robot_control/mug"
    
    # Load the reference RGB image and segmentation mask
    rgb_ref = cv2.imread(f"{dir}/demo_wrist_rgb.png")[...,::-1].copy()
    seg_ref = cv2.imread(f"{dir}/demo_wrist_mask.png", cv2.IMREAD_GRAYSCALE).astype(bool)
    
    # Initialize the visual servoer with reference images
    lgvs = LightGlueVisualServoer(rgb_ref, seg_ref, use_depth=True, silent=False)
    
    try:
        while rclpy.ok():
            if not lgvs.silent:
                print("Starting new observation cycle...")
            
            # Test observation functionality
            live_rgb, live_depth = lgvs.observe(timeout=2.0)
            if live_rgb is not None and not lgvs.silent:
                print(f"Received RGB image with shape: {live_rgb.shape}")
                
                # Test matching functionality
                mkpts_scores_0, mkpts_scores_1, _ = lgvs.match_lightglue(filter_seg=True)
                if mkpts_scores_0 is not None:
                    print(f"Matched {len(mkpts_scores_0)} keypoints")
            
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        rclpy.shutdown()