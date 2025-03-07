import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage
import subprocess
import os
import logging
from typing import TypedDict
from config import d405_intrinsic
import open3d as o3d

class SceneData(TypedDict):
    image: np.ndarray
    depth: np.ndarray
    seg: np.ndarray
    intrinsics: np.ndarray
    T_WC: np.ndarray

class ProcessedData(TypedDict):
    pointcloud: np.ndarray

class RGBDSaver(Node):
    def __init__(self):
        super().__init__('rgbd_saver')
        self.bridge = CvBridge()
        self.intrinsics = d405_intrinsic

    def run(self):
        # rgb_message_wrist = self.wait_for_message("camera/camera/color/image_rect_raw", Image, timeout=5)
        # depth_message_wrist = self.wait_for_message("camera/camera/aligned_depth_to_color/image_raw", Image, timeout=5)

        # if rgb_message_wrist is None or depth_message_wrist is None:
        #     self.get_logger().error("Failed to receive images in time.")
        #     return None

        # rgb_image_wrist = self.bridge.imgmsg_to_cv2(rgb_message_wrist, desired_encoding="bgr8")
        # depth_image_wrist = self.bridge.imgmsg_to_cv2(depth_message_wrist, desired_encoding="16UC1")
        
        # rgb_pil = PILImage.fromarray(cv2.cvtColor(rgb_image_wrist, cv2.COLOR_BGR2RGB))
        # depth_pil = PILImage.fromarray(depth_image_wrist)
        
        rgb_dir = f"/home/yilong/ObAct/robot_control/mug/demo_wrist_rgb.png"
        depth_dir = f"/home/yilong/ObAct/robot_control/mug/demo_wrist_depth.png"
        rgb_image = np.array(PILImage.open(rgb_dir))
        depth_image = np.array(PILImage.open(depth_dir))

        # depth_dir = f"{self.DIR}/mug/demo_wrist_depth.png"
        
        # os.makedirs(f"{self.DIR}/mug", exist_ok=True)
        # rgb_pil.save(rgb_dir)
        # depth_pil.save(depth_dir)

        # Run GroundingDINO segmentation
        seg_mask = self.run_grounded_sam(rgb_dir)
        if seg_mask is None:
            raise RuntimeError
        
        # Prepare SceneData
        scene_data = SceneData(
            image=rgb_image,
            depth=depth_image,
            seg=seg_mask,
            intrinsics=self.intrinsics,
            T_WC=np.eye(4)
        )

        # Create processor instance
        processor = PointCloudProcessor(rgb_dir, depth_dir, "robot_control/outputs/mask_0.png")

        # Visualize with default parameters (radius=0.1m, 100 points)
        processor.visualize()

    def wait_for_message(self, topic, msg_type, timeout=5):
        future = rclpy.Future()

        def callback(msg):
            if not future.done():
                future.set_result(msg)

        sub = self.create_subscription(msg_type, topic, callback, 10)
        self.get_logger().info(f"Waiting for message on {topic}...")

        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                return future.result()
            if self.get_clock().now().seconds_nanoseconds()[0] - start_time > timeout:
                self.get_logger().warning(f"Timeout while waiting for message on {topic}")
                return None

    def run_grounded_sam(self, input_image_path):
        try:
            output_dir = "/home/yilong/ObAct/robot_control/outputs"
            os.makedirs(output_dir, exist_ok=True)

            command = [
                "python3",
                "grounded_sam_demo.py",
                "--config", "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "--grounded_checkpoint", "groundingdino_swint_ogc.pth",
                "--sam_checkpoint", "sam_vit_h_4b8939.pth",
                "--input_image", input_image_path,
                "--output_dir", output_dir,
                "--box_threshold", "0.3",
                "--text_threshold", "0.25",
                "--text_prompt", "green mug",
                "--device", "cuda"
            ]
            
            subprocess.run(command, check=True, cwd=os.path.expanduser("~/Grounded-Segment-Anything"))
            
            # Load the generated mask 
            mask_path = os.path.join(output_dir, "mask_0.png")
            if not os.path.exists(mask_path):
                self.get_logger().error("Segmentation mask not found!")
                return None
                
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return mask > 0  # Convert to boolean mask
            
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Grounded SAM failed: {str(e)}")
            return None
        except Exception as e:
            self.get_logger().error(f"Error in segmentation: {str(e)}")
            return None

class PointCloudProcessor:
    def __init__(self, rgb_path, depth_path, mask_path):
        """Initialize with file paths"""
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.mask_path = mask_path
        
        # Load data
        self._load_data()
        # Process point cloud
        self._process_point_cloud()
        
    def _load_data(self):
        """Load images and intrinsic parameters"""
        # Load images
        self.rgb_image = np.array(PILImage.open(self.rgb_path))
        self.depth_image = np.array(PILImage.open(self.depth_path))
        self.mask = np.array(PILImage.open(self.mask_path))

        # Process RGB image
        if self.rgb_image.shape[-1] == 4:  # If RGBA
            self.rgb_image = self.rgb_image[..., :3]
        self.rgb_image = self.rgb_image.astype(np.uint8)

        # Process depth image
        self.depth_image = self.depth_image.astype(np.float32)
        self.depth_image = self.depth_image * (self.mask > 0)  # Apply mask

    def _process_point_cloud(self):
        """Create and process point cloud"""
        # Create RGBD image
        color = o3d.geometry.Image(self.rgb_image)
        depth = o3d.geometry.Image(self.depth_image)
        
        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=1000.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )

        # Set camera intrinsics
        fx, fy = d405_intrinsic[0, 0], d405_intrinsic[1, 1]
        cx, cy = d405_intrinsic[0, 2], d405_intrinsic[1, 2]
        
        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic_o3d.set_intrinsics(
            width=self.rgb_image.shape[1],
            height=self.rgb_image.shape[0],
            fx=fx, fy=fy, cx=cx, cy=cy
        )

        # Create and clean point cloud
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            self.rgbd_image, self.intrinsic_o3d
        )
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.005)
        self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    def generate_hemisphere_points_with_orientations(self, radius=0.1, num_points=100):
        """Generate hemispherical sampling points with orientations around Z-axis"""
        points = np.asarray(self.pcd.points)
        self.center = np.mean(points, axis=0)
        
        sampling_data = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        
        for i in range(num_points):
            z = 1 - (i / float(num_points - 1)) * 2  # z from 1 to -1
            radius_at_z = np.sqrt(1 - z * z)
            theta = phi * i
            
            x = np.cos(theta) * radius_at_z
            y = np.sin(theta) * radius_at_z
            
            if z >= 0:  # Upper hemisphere
                # Position
                pos = np.array([x, y, z]) * radius + self.center
                
                # Orientation: Point towards center
                direction = self.center - pos  # Vector from position to center
                direction = direction / np.linalg.norm(direction)  # Normalize
                
                # Z-axis of the viewpoint (looking towards center)
                z_axis = direction
                
                # X-axis: Perpendicular to z_axis and global Z [0,0,1]
                global_z = np.array([0, 0, 1])
                x_axis = np.cross(global_z, z_axis)
                if np.linalg.norm(x_axis) > 0:  # Ensure not parallel to global Z
                    x_axis = x_axis / np.linalg.norm(x_axis)
                else:
                    x_axis = np.array([1, 0, 0])  # Default if parallel
                
                # Y-axis: Perpendicular to z_axis and x_axis
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)
                
                # Rotation matrix
                R = np.column_stack((x_axis, y_axis, z_axis))
                
                # Convert to roll, pitch, yaw (in radians)
                yaw = np.arctan2(R[1, 0], R[0, 0])
                pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
                roll = np.arctan2(R[2, 1], R[2, 2])
                
                sampling_data.append({
                    'position': pos,
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'rotation': R
                })
        
        return sampling_data

    def transform_to_robot_frame(self):
        """Transform point cloud from camera frame to robot frame"""
        # Load hand-eye calibration matrix (camera to end-effector)
        cam_to_ee = np.load("/home/yilong/ObAct/robot_control/d405_extrinsic.npy")
        
        # Robot's end-effector pose in robot frame
        ee_to_robot = np.array([
            [6.07944607e-01, 6.35465678e-04, 7.93979188e-01, 2.52638927e-01],
            [-3.54942557e-03, 9.99991863e-01, 1.91742259e-03, -8.50579388e-04],
            [-7.93971508e-01, -3.98385675e-03, 6.07941915e-01, 2.82457341e-01],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        
        # Complete transformation: robot_frame = ee_to_robot * cam_to_ee * camera_frame
        transform = np.dot(ee_to_robot, cam_to_ee)
        
        # Transform the point cloud
        self.pcd.transform(transform)
        
        # If center exists (after sampling), transform it too
        if hasattr(self, 'center'):
            center_homogeneous = np.append(self.center, 1)  # Convert to homogeneous coordinates
            self.center = np.dot(transform, center_homogeneous)[:3]  # Transform and extract 3D point
        
        return transform

    # Update visualize method to include transformation option
    def visualize(self, radius=0.1, num_points=100, transform_to_robot=True):
        """Visualize point cloud with oriented hemispherical sampling points"""
        if transform_to_robot:
            self.transform_to_robot_frame()
        
        # Generate sampling points with orientations
        sampling_data = self.generate_hemisphere_points_with_orientations(radius, num_points)
        
        # Create sampling point cloud
        sampling_pcd = o3d.geometry.PointCloud()
        sampling_positions = np.array([data['position'] for data in sampling_data])
        sampling_pcd.points = o3d.utility.Vector3dVector(sampling_positions)
        sampling_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        
        # Center sphere
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        center_sphere.translate(self.center)
        center_sphere.paint_uniform_color([0.0, 1.0, 0.0])
        
        # Coordinate frame at center
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=self.center
        )
        
        # Visualize orientations with small coordinate frames
        orientation_frames = []
        for data in sampling_data[:10]:  # Limit to 10 for clarity, adjust as needed
            pos = data['position']
            R = data['rotation']
            
            # Create small coordinate frame for each viewpoint
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
            frame.rotate(R, center=(0, 0, 0))  # Rotate to match viewpoint orientation
            frame.translate(pos)
            orientation_frames.append(frame)
        
        # Combine all geometries
        geometries = [self.pcd, sampling_pcd, center_sphere, coord_frame] + orientation_frames
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Point Cloud with Oriented Sampling",
            width=800, height=600
        )
        
        # Print statistics
        print(f"Center: {self.center}")
        print(f"Number of original points: {len(self.pcd.points)}")
        print(f"Number of sampling points: {len(sampling_data)}")
        print(f"Radius: {radius}m")
        for i, data in enumerate(sampling_data[:5]):  # Print first 5 for example
            print(f"Viewpoint {i}:")
            print(f"  Position: {data['position']}")
            print(f"  Roll: {data['roll']:.3f}, Pitch: {data['pitch']:.3f}, Yaw: {data['yaw']:.3f}")

def main(args=None):
    rclpy.init(args=args)
    rgbd_saver = RGBDSaver()
    try:
        rgbd_saver.run()
        

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        rgbd_saver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()