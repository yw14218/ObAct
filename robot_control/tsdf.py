import open3d as o3d
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

class UniformTSDFVolume:
    def __init__(self, length=0.3, resolution=40):
        self.length = length
        self.resolution = resolution
        self.voxel_size = self.length / self.resolution
        self.sdf_trunc = 4 * self.voxel_size
        self.o3dvol = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.length,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    def integrate(self, depth_img, intrinsic, extrinsic):

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        self.o3dvol.integrate(rgbd, intrinsic, np.linalg.inv(extrinsic))

    def get_scene_cloud(self):
        return self.o3dvol.extract_point_cloud()

    def get_map_cloud(self):
        return self.o3dvol.extract_voxel_point_cloud()

    def get_grid(self):
        map_cloud = self.get_map_cloud()
        points = np.asarray(map_cloud.points)
        distances = np.asarray(map_cloud.colors)[:, [0]]
        return self.map_cloud_to_grid(self.voxel_size, points, distances)

    @staticmethod
    def map_cloud_to_grid(voxel_size, points, distances):
        grid = np.zeros((40, 40, 40), dtype=np.float32)
        indices = (points // voxel_size).astype(int)
        grid[tuple(indices.T)] = distances.squeeze()
        return grid

class TSDFMapper(Node):
    def __init__(self):
        super().__init__('tsdf_mapper')
        self.bridge = CvBridge()
        self.tsdf_volume = UniformTSDFVolume()
        self.robot = InterbotixManipulatorXS("vx300s", "arm", "gripper")
        K = np.load("robot_control/d405_intrinsic.npy")

        self.intrinsic = type('Intrinsic', (), {
            'width': 848,
            'height': 480,
            'fx': K[0][0],
            'fy': K[1][1],
            'cx': K[0][2],
            'cy': K[1][2]
        })()

    def wait_for_message(self, topic, msg_type, timeout=5):
        future = rclpy.Future()

        def callback(msg):
            if not future.done():
                future.set_result(msg)

        sub = self.create_subscription(msg_type, topic, callback, 10)
        # self.get_logger().info(f"Waiting for message on {topic}...")

        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                return future.result()
            if self.get_clock().now().seconds_nanoseconds()[0] - start_time > timeout:
                self.get_logger().warning(f"Timeout while waiting for message on {topic}")
                return None

    def run_and_build_map(self):
        # Initialize visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        point_cloud = o3d.geometry.PointCloud()
        vis.add_geometry(point_cloud)

        # Add coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coordinate_frame)

        # Start robot
        robot_startup()

        try:
            while rclpy.ok():
                # Get depth image from wrist camera
                depth_message = self.wait_for_message(
                    "camera/camera/aligned_depth_to_color/image_raw", 
                    Image, 
                    timeout=5
                )
                
                if depth_message is None:
                    raise RuntimeError

                # Convert ROS message to numpy array
                depth_image = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding="16UC1")
                
                T_cam_ee = np.load("robot_control/d405_extrinsic.npy")
                T_ee_world = self.robot.arm.get_ee_pose()
                T_cam_world = T_ee_world @ T_cam_ee

                # Integrate depth into TSDF
                self.tsdf_volume.integrate(depth_image, self.intrinsic, T_cam_world)
                
                # Update visualization
                scene_cloud = self.tsdf_volume.get_scene_cloud()
                point_cloud.points = scene_cloud.points
                
                vis.update_geometry(point_cloud)
                vis.poll_events()
                vis.update_renderer()

        except KeyboardInterrupt:
            self.get_logger().info("Shutting down TSDF mapping...")
        
        finally:
            # Cleanup
            vis.destroy_window()
            robot_shutdown()

def main():
    rclpy.init()
    mapper = TSDFMapper()
    mapper.run_and_build_map()
    rclpy.shutdown()

if __name__ == '__main__':
    main()