import os
import subprocess
import time
from typing import Optional, List

import cv2
import numpy as np
import open3d as o3d
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped

from config import d405_intrinsic, d405_extrinsic
from utils import pose_inv
from tsdf_torch import ViewSampler, TSDFVolume
from ik_solver import InverseKinematicsSolver
import threading

class Perception(Node):
    def __init__(self):
        """Initialize the Perception node with necessary components."""
        super().__init__('Perception')
        self.bridge = CvBridge()
        self.bridge = CvBridge()
        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=848,
            height=480,
            fx=d405_intrinsic[0, 0],
            fy=d405_intrinsic[1, 1],
            cx=d405_intrinsic[0, 2],
            cy=d405_intrinsic[1, 2]
        )
        self.view_sampler = ViewSampler()
        self.output_dir = "/home/yilong/ObAct/robot_control/mug"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.ik_solver = InverseKinematicsSolver()

        # TSDF setup
        self.tsdf = TSDFVolume(0.45, 64)
        self.vis = None 
        self.pcd = None
        self.last_update_time = 0  # For controlling update frequency

        # Visualization setup
        self.latest_pcd = None  # Shared variable for the latest point cloud
        self.pcd_lock = threading.Lock()  # Lock for thread-safe access
        self.vis_lock = threading.Lock()  # Lock for visualizer updates
        self.running = True  # Flag to control the visualization thread

        # Start the visualization thread
        self.vis_thread = threading.Thread(target=self.run_visualizer, daemon=True)
        self.vis_thread.start()

    def update_tsdf(self) -> None:
        """Integrate current RGB-D data into TSDF."""
        rgb_msg = self._wait_for_message("camera/camera/color/image_rect_raw", Image)
        depth_msg = self._wait_for_message("camera/camera/aligned_depth_to_color/image_raw", Image)
        if rgb_msg is None or depth_msg is None:
            self.get_logger().error("Failed to receive images in time.")
            raise RuntimeError

        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")

        current_time = time.time()
        if current_time - self.last_update_time < 0.1:  # Limit to 10 Hz
            return
        self.last_update_time = current_time

        # Convert images to Open3D format
        rgb_o3d = o3d.geometry.Image(rgb_image)
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=0.7, convert_rgb_to_intensity=False)

        # Get camera pose in robot frame
        try:
            transform = self.tf_buffer.lookup_transform(
                "vx300s/base_link", "camera_color_optical_frame", rclpy.time.Time())
            trans = transform.transform.translation
            rot = transform.transform.rotation
            T_cam_to_robot = np.eye(4)
            T_cam_to_robot[:3, :3] = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
            T_cam_to_robot[:3, 3] = [trans.x, trans.y, trans.z]
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return

        # Integrate into TSDF
        self.tsdf._volume.integrate(rgbd, self.intrinsic_o3d, pose_inv(T_cam_to_robot))

        # Update the latest point cloud for visualization
        new_pcd = self.tsdf.get_point_cloud()
        if new_pcd.has_points():
            with self.pcd_lock:
                self.latest_pcd = new_pcd

    def run_visualizer(self) -> None:
        """Run the Open3D visualizer in a separate thread."""
        last_vis_update = 0
        while self.running:
            self.update_tsdf()
            current_time = time.time()
            # Update visualization at 5 Hz (every 0.2 seconds)
            if current_time - last_vis_update >= 0.2:
                with self.pcd_lock:
                    if self.latest_pcd is not None and self.latest_pcd.has_points():
                        with self.vis_lock:
                            if self.vis is None:
                                # Initialize visualizer
                                self.vis = o3d.visualization.Visualizer()
                                self.vis.create_window("TSDF Point Cloud Visualization", width=800, height=600)
                                self.pcd = self.latest_pcd
                                self.vis.add_geometry(self.pcd)
                                self.vis.get_render_option().point_size = 2.0
                            else:
                                # Update existing point cloud
                                self.pcd.points = self.latest_pcd.points
                                self.pcd.colors = self.latest_pcd.colors
                                self.vis.update_geometry(self.pcd)
                        self.latest_pcd = None  # Clear after use
                last_vis_update = current_time

            # Run the visualizer's event loop
            if self.vis is not None:
                with self.vis_lock:
                    self.vis.poll_events()
                    self.vis.update_renderer()
            time.sleep(0.01)  # Prevent busy-waiting

    def destroy_node(self):
        """Clean up resources."""
        self.running = False  # Stop the visualization thread
        self.vis_thread.join()  # Wait for the thread to finish
        if self.vis is not None:
            with self.vis_lock:
                self.vis.destroy_window()
        super().destroy_node()

    def run(self) -> Optional[List[np.ndarray]]:
        """Process RGB-D data, transform point cloud, and generate feasible viewpoints."""
        # Capture and save RGB-D images
        rgb_path, depth_path = self._capture_and_save_images()
        if rgb_path is None or depth_path is None:
            return None

        # Run segmentation
        seg_mask = self._run_grounded_sam(rgb_path)
        if seg_mask is None:
            raise RuntimeError("Segmentation failed")

        # Process point cloud
        mask_path = os.path.join(self.output_dir, "mask_0.png")
        processor = PointCloudProcessor(rgb_path, depth_path, mask_path, self)
        processor.transform_to_robot_frame()
        self.view_sampler.pcd = processor.pcd

        # Generate and filter viewpoints
        viewpoints = self.view_sampler.generate_hemisphere_points_with_orientations(radius=0.3, num_points=64)
        filtered_viewpoints, joint_poses = self._filter_viewpoints_with_ik(viewpoints)
        return filtered_viewpoints, joint_poses

    def _capture_and_save_images(self) -> tuple[Optional[str], Optional[str]]:
        """Capture RGB and depth images and save them to disk."""
        rgb_msg = self._wait_for_message("camera/camera/color/image_rect_raw", Image)
        depth_msg = self._wait_for_message("camera/camera/aligned_depth_to_color/image_raw", Image)
        if rgb_msg is None or depth_msg is None:
            self.get_logger().error("Failed to receive images in time.")
            return None, None

        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")

        rgb_pil = PILImage.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        depth_pil = PILImage.fromarray(depth_image)

        rgb_path = os.path.join(self.output_dir, "demo_wrist_rgb.png")
        depth_path = os.path.join(self.output_dir, "demo_wrist_depth.png")
        os.makedirs(self.output_dir, exist_ok=True)

        rgb_pil.save(rgb_path)
        depth_pil.save(depth_path)
        return rgb_path, depth_path

    def _wait_for_message(self, topic: str, msg_type: type, timeout: float = 5.0) -> Optional[Image]:
        """Wait for a message on a topic with a timeout."""
        future = rclpy.Future()
        sub = self.create_subscription(msg_type, topic, lambda msg: future.set_result(msg) if not future.done() else None, 10)
        self.get_logger().info(f"Waiting for message on {topic}...")

        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                return future.result()
            if self.get_clock().now().seconds_nanoseconds()[0] - start_time > timeout:
                self.get_logger().warning(f"Timeout waiting for message on {topic}")
                return None

    def _run_grounded_sam(self, input_image_path: str) -> Optional[np.ndarray]:
        """Run GroundingDINO segmentation on the input image."""
        command = [
            "python3", "grounded_sam_demo.py",
            "--config", "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "--grounded_checkpoint", "groundingdino_swint_ogc.pth",
            "--sam_checkpoint", "sam_vit_h_4b8939.pth",
            "--input_image", input_image_path,
            "--output_dir", self.output_dir,
            "--box_threshold", "0.3",
            "--text_threshold", "0.25",
            "--text_prompt", "green mug",
            "--device", "cuda"
        ]

        try:
            subprocess.run(command, check=True, cwd=os.path.expanduser("~/Grounded-Segment-Anything"))
            mask_path = os.path.join(self.output_dir, "mask_0.png")
            if not os.path.exists(mask_path):
                self.get_logger().error("Segmentation mask not found!")
                return None
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return mask > 0
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Grounded SAM failed: {str(e)}")
            return None

    def publish_viewpoints(self, viewpoints: List[np.ndarray], frame_id: str = "vx300s/base_link") -> None:
        """Publish viewpoints as TransformStamped messages."""
        current_time = self.get_clock().now().to_msg()
        transforms = []

        for i, viewpoint in enumerate(viewpoints):
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = frame_id
            t.child_frame_id = f"viewpoint_{i}"

            pos = viewpoint['position']
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])

            rot_matrix = viewpoint['rotation']
            q = Rotation.from_matrix(rot_matrix).as_quat()
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            transforms.append(t)

        self.tf_broadcaster.sendTransform(transforms)

    def _filter_viewpoints_with_ik(self, viewpoints: np.ndarray):
        """Filter viewpoints based on IK feasibility using MoveIt 2."""
        filtered_viewpoints = []
        eef_poses = []
        for vp in viewpoints:
            T = np.eye(4)
            T[:3, :3] = vp['rotation']
            T[:3, 3] = vp['position']
            T_eef_robot = T @ pose_inv(d405_extrinsic)
            position = T_eef_robot[:3, 3]
            quat_xyzw = Rotation.from_matrix(T_eef_robot[:3, :3]).as_quat()
            try:
                result = self.ik_solver.compute_ik(position, quat_xyzw)
                if result is not None:
                    filtered_viewpoints.append(vp)
                    eef_poses.append({'position': position, 'quat_xyzw': quat_xyzw})
            except Exception as e:
                self.get_logger().error(f"IK computation failed: {str(e)}")
        print("number of filtered viewpoints: ", len(filtered_viewpoints))

        return filtered_viewpoints, eef_poses

class PointCloudProcessor:
    def __init__(self, rgb_path: str, depth_path: str, mask_path: str, node: Node):
        """Initialize point cloud processor with file paths and ROS node."""
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.node = node
        self._load_data()
        self._process_point_cloud()
        bbox_image = self._visualize_3d_bounding_box_on_image()
        PILImage.fromarray(bbox_image).save(rgb_path.replace(".png", "_bbox.png"))

    def _load_data(self) -> None:
        """Load RGB, depth, and mask images."""
        self.rgb_image = np.array(PILImage.open(self.rgb_path))
        self.depth_image = np.array(PILImage.open(self.depth_path))
        self.mask = np.array(PILImage.open(self.mask_path))

        if self.rgb_image.shape[-1] == 4:
            self.rgb_image = self.rgb_image[..., :3]
        self.rgb_image = self.rgb_image.astype(np.uint8)
        self.depth_image = self.depth_image.astype(np.float32) * (self.mask > 0)

    def _process_point_cloud(self) -> None:
        """Create and process the point cloud from RGB-D data."""
        color = o3d.geometry.Image(self.rgb_image)
        depth = o3d.geometry.Image(self.depth_image)
        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        self.fx, self.fy = d405_intrinsic[0, 0], d405_intrinsic[1, 1]
        self.cx, self.cy = d405_intrinsic[0, 2], d405_intrinsic[1, 2]
        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic_o3d.set_intrinsics(
            width=self.rgb_image.shape[1], height=self.rgb_image.shape[0],
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd_image, self.intrinsic_o3d)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.005)
        self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    def _visualize_3d_bounding_box_on_image(self) -> np.ndarray:
        """Project 3D bounding box onto the RGB image."""
        if not hasattr(self, 'pcd') or not all(hasattr(self, attr) for attr in ['fx', 'fy', 'cx', 'cy']):
            self.node.get_logger().warning("Required attributes not available for bounding box visualization")
            return self.rgb_image

        try:
            obb = self.pcd.get_oriented_bounding_box()
            obb.color = (1, 0, 0)
            corner_points = np.asarray(obb.get_box_points())
            sorted_indices = np.lexsort((corner_points[:, 0], corner_points[:, 1], corner_points[:, 2]))
            sorted_corner_points = corner_points[sorted_indices]

            K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
            points_2d = (K @ sorted_corner_points.T).T
            points_2d = points_2d[:, :2] / points_2d[:, 2:3]

            points_2d_float = points_2d.copy()
            h, w = self.rgb_image.shape[:2]
            points_2d = points_2d.astype(int)
            points_2d[:, 0] = np.clip(points_2d[:, 0], 0, w-1)
            points_2d[:, 1] = np.clip(points_2d[:, 1], 0, h-1)

            output_image = self.rgb_image.copy()
            edges = [
                (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
                (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]

            for edge in edges:
                pt1_float, pt2_float = points_2d_float[edge[0]], points_2d_float[edge[1]]
                ret, pt1_clipped, pt2_clipped = cv2.clipLine(
                    (0, 0, w, h), (int(pt1_float[0]), int(pt1_float[1])), (int(pt2_float[0]), int(pt2_float[1]))
                )
                if ret:
                    cv2.line(output_image, pt1_clipped, pt2_clipped, (0, 0, 255), 2, cv2.LINE_AA)
            return output_image
        except Exception as e:
            self.node.get_logger().error(f"Error visualizing bounding box: {str(e)}")
            return self.rgb_image

    def transform_to_robot_frame(self) -> np.ndarray:
        """Transform point cloud from camera to robot frame using TF2."""
        try:
            time.sleep(1)  # Ensure TF frames are available
            self.node.get_logger().info("Looking up transform from 'camera_color_optical_frame' to 'vx300s/base_link'...")
            transform = self.node.tf_buffer.lookup_transform(
                target_frame="vx300s/base_link",
                source_frame="camera_color_optical_frame",
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            self.node.get_logger().info("Transform lookup successful.")

            trans = transform.transform.translation
            rot = transform.transform.rotation
            translation = np.array([trans.x, trans.y, trans.z])
            quat = [rot.x, rot.y, rot.z, rot.w]
            rotation_matrix = Rotation.from_quat(quat).as_matrix()

            T_cam_to_robot = np.eye(4)
            T_cam_to_robot[:3, :3] = rotation_matrix
            T_cam_to_robot[:3, 3] = translation

            self.pcd.transform(T_cam_to_robot)
            return T_cam_to_robot
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.node.get_logger().error(f"Transform lookup failed: {str(e)}")
            raise


def main(args=None):
    """Main entry point for the perception pipeline."""
    rclpy.init(args=args)
    perception = Perception()

    try:
        # Run initial viewpoint generation
        viewpoints, eef_poses = perception.run()
        if viewpoints:
            while rclpy.ok():
                perception.publish_viewpoints(viewpoints)
                for i in range(len(eef_poses)):   
                    perception.ik_solver.move_to_pose(eef_poses[i]['position'], eef_poses[i]['quat_xyzw'])
                    time.sleep(5)
                    if i == len(eef_poses) - 1:
                        break
                rclpy.spin_once(perception, timeout_sec=0.1)
        # Keep spinning for real-time TSDF updates
        while rclpy.ok():
            rclpy.spin_once(perception, timeout_sec=0.1)
    except Exception as e:
        perception.get_logger().error(f"Error in execution: {str(e)}")
    finally:
        perception.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()