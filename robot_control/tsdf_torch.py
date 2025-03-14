import open3d as o3d
import numpy as np
import os
import glob
import cv2
import time
import torch

# Constants
DEFAULT_DATA_FOLDER = "3d_data"
DEFAULT_INTRINSIC_FILE = "robot_control/d405_intrinsic.npy"
TSDF_SIZE = 0.45
TSDF_RESOLUTION = 64
NUM_SAMPLED_VIEWS = 64
VOXEL_DOWN_SIZE = 0.005
CAMERA_SCALE = 0.03
TOP_N = 5
RAY_BATCH_SIZE = int(480 * 848 * 0.1)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ViewSampler:
    @staticmethod
    def build_rotation_matrix(eye, center, up):
        eye = np.asarray(eye)
        center = np.asarray(center)
        up = np.asarray(up)
        
        z_axis = center - eye
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            up = np.array([0, 1, 0]) if abs(np.dot(z_axis, [0, 0, 1])) > 0.99 else np.array([0, 0, 1])
            x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        rotation = np.eye(3)
        rotation[:, 0] = -x_axis
        rotation[:, 1] = -y_axis
        rotation[:, 2] = z_axis
        
        return rotation
    
    @staticmethod
    def spherical_to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    def generate_hemisphere_points_with_orientations(self, radius=0.1, num_points=128):
        points = np.asarray(self.pcd.points)
        self.center = np.mean(points, axis=0)
        # self.center[0] += 0.15 
        
        theta_samples = int(np.sqrt(num_points / 2))
        phi_samples = int(num_points / theta_samples)
        
        theta_max = np.pi / 2 - 1e-6
        theta_vals = np.linspace(0, theta_max, theta_samples)
        phi_vals = np.linspace(0, 2 * np.pi, phi_samples, endpoint=False)
        
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
        theta_flat = theta_grid.ravel()
        phi_flat = phi_grid.ravel()
        
        x = radius * np.sin(theta_flat) * np.cos(phi_flat)
        y = radius * np.sin(theta_flat) * np.sin(phi_flat)
        z = radius * np.cos(theta_flat)
        eye_positions = self.center + np.stack([x, y, z], axis=-1)
        
        default_up = np.array([0, 0, 1])
        rotation_matrices = np.array([
            self.build_rotation_matrix(pos, self.center, default_up)
            for pos in eye_positions
        ])
        
        sampling_data = np.array(
            list(zip(eye_positions, rotation_matrices)),
            dtype=[('position', float, (3,)), ('rotation', float, (3, 3))]
        )
    
        # filter out too low points
        sampling_data = sampling_data[sampling_data['position'][:, 2] > 0.1]
        print(f"Generated {len(sampling_data)} viewpoints")

        return sampling_data

class TSDFVolume:
    def __init__(self, size, resolution, sdf_trunc_factor=4):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = sdf_trunc_factor * self.voxel_size
        self.origin = np.array([0.2, -self.size / 2, -self.size / 2])

        print(f"TSDF Origin: {self.origin}, Size: {self.size}, Max Bound: {self.origin + self.size}")

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=self.origin
        )

    def integrate(self, rgb_img, depth_img, intrinsic, extrinsic, frame_idx, debug_dir=None):
        rgb_img_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img_filtered = depth_img.astype(np.float32) / 1000.0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_img_rgb),
            o3d.geometry.Image(depth_img_filtered),
            depth_scale=1.0,
            depth_trunc=0.7,
            convert_rgb_to_intensity=False,
        )

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic['width'],
            height=intrinsic['height'],
            fx=intrinsic['fx'],
            fy=intrinsic['fy'],
            cx=intrinsic['cx'],
            cy=intrinsic['cy'],
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic)
        if not pcd.has_points():
            print(f"Frame {frame_idx}: No points generated from RGBD image")
        else:
            print(f"Frame {frame_idx}: Generated {len(pcd.points)} points")
            if debug_dir:
                o3d.io.write_point_cloud(os.path.join(debug_dir, f"pcd_{frame_idx:04d}.ply"), pcd)

        self._volume.integrate(rgbd, intrinsic_o3d, extrinsic)
        return pcd
    
    def get_point_cloud(self):
        return self._volume.extract_point_cloud()

    def get_sdf_grid(self):
        voxel_cloud = self._volume.extract_voxel_point_cloud()
        points = np.asarray(voxel_cloud.points)
        sdf_values = np.asarray(voxel_cloud.colors)[:, 0]
        
        grid = np.full((self.resolution, self.resolution, self.resolution), np.nan, dtype=np.float32)
        indices = np.floor((points - self.origin) / self.voxel_size).astype(int)
        valid_mask = np.all((indices >= 0) & (indices < self.resolution), axis=1)
        valid_indices = indices[valid_mask]
        grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = sdf_values[valid_mask]
        
        valid_sdfs = sdf_values[valid_mask]
        print(f"SDF Stats - Min: {valid_sdfs.min():.3f}, Max: {valid_sdfs.max():.3f}, Mean: {valid_sdfs.mean():.3f}")
        print(f"Inside Voxels (SDF < 0.5): {(valid_sdfs < 0.5).sum()}")
        return grid

class ViewEvaluator:
    def __init__(self, tsdf_volume, intrinsic, target_bbox):
        self.tsdf = tsdf_volume
        self.intrinsic = intrinsic
        self.target_bbox = torch.tensor(target_bbox, dtype=torch.float32, device=device)
        self.sdf_grid = torch.tensor(self.tsdf.get_sdf_grid(), dtype=torch.float32, device=device)
        self.origin = torch.tensor(self.tsdf.origin, dtype=torch.float32, device=device)
        self.voxel_size = torch.tensor(self.tsdf.voxel_size, dtype=torch.float32, device=device)
        print(f"Initialized ViewEvaluator with TSDF grid shape: {self.sdf_grid.shape}")
        print(f"Target Bounding Box: {self.target_bbox}")

    def compute_information_gain(self, pose, width=848, height=480):
        position = torch.tensor(pose[:3, 3], dtype=torch.float32, device=device)
        rotation = torch.tensor(pose[:3, :3], dtype=torch.float32, device=device)

        # Generate rays
        u, v = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing='ij')
        u, v = u.flatten(), v.flatten()
        rays_dir = torch.stack([(u - self.intrinsic['cx']) / self.intrinsic['fx'],
                                (v - self.intrinsic['cy']) / self.intrinsic['fy'],
                                torch.ones_like(u, dtype=torch.float32, device=device)], dim=1)
        rays_dir = rays_dir / torch.norm(rays_dir, dim=1, keepdim=True)
        rays_dir = torch.matmul(rotation, rays_dir.T).T

        G_x = self.compute_gain_gpu(position, rays_dir)

        return G_x.item()

    def compute_gain_gpu(self, position, rays_dir):
        """Compute information gain on GPU using PyTorch, fully vectorized with batching."""
        step_size = self.voxel_size
        max_steps = int(self.tsdf.size / step_size.item())
        G_x = torch.tensor(0.0, dtype=torch.float32, device=device)

        num_rays = rays_dir.shape[0]
        t = torch.arange(0, max_steps * step_size.item(), step_size.item(), device=device)  # [max_steps]

        # Process rays in batches
        for start_idx in range(0, num_rays, RAY_BATCH_SIZE):
            end_idx = min(start_idx + RAY_BATCH_SIZE, num_rays)
            batch_rays = rays_dir[start_idx:end_idx]
            batch_size = batch_rays.shape[0]

            # Compute points: [batch_size, max_steps, 3]
            points = position.unsqueeze(0).unsqueeze(0) + batch_rays.unsqueeze(1) * t.unsqueeze(0).unsqueeze(2)
            voxel_idx = ((points - self.origin) / self.voxel_size).to(torch.int32)

            # Valid voxel indices
            valid_mask = torch.all((voxel_idx >= 0) & (voxel_idx < self.tsdf.resolution), dim=2)
            ray_indices, step_indices = torch.where(valid_mask)
            valid_voxel_idx = voxel_idx[ray_indices, step_indices]

            # Fetch SDF values
            sdf_values = torch.full_like(points[..., 0], float('nan'), device=device)
            sdf_values[ray_indices, step_indices] = self.sdf_grid[
                valid_voxel_idx[:, 0], valid_voxel_idx[:, 1], valid_voxel_idx[:, 2]
            ]

            # Conditions
            in_bbox = torch.all(
                (points >= self.target_bbox[:3]) & (points <= self.target_bbox[3:]),
                dim=2
            )
            inside = (sdf_values < 0.5) & ~torch.isnan(sdf_values)
            surface = (torch.abs(sdf_values - 0.5) < self.voxel_size / (2 * 0.028)) & ~torch.isnan(sdf_values)

            # Vectorized gain computation
            first_surface = torch.full((batch_size,), max_steps, device=device)
            surface_rays = torch.where(surface.any(dim=1))[0]
            if len(surface_rays) > 0:
                first_surface[surface_rays] = surface[surface_rays].float().argmax(dim=1)

            # Mask for steps before first surface
            mask = torch.arange(max_steps, device=device).expand(batch_size, -1) < first_surface.unsqueeze(1)
            valid_inside = inside & in_bbox & mask
            G_x += valid_inside.sum()

        return G_x

def load_data(folder_path):
    """Load RGB images, depth maps, and poses from folder."""
    rgb_files = sorted(glob.glob(os.path.join(folder_path, "rgb_*.png")))
    depth_files = sorted(glob.glob(os.path.join(folder_path, "depth_*.npy")))
    pose_files = sorted(glob.glob(os.path.join(folder_path, "pose_*.npy")))
    
    return [cv2.imread(f) for f in rgb_files], [np.load(f) for f in depth_files], [np.load(f) for f in pose_files]

def load_intrinsics(file_path):
    """Load camera intrinsics from file."""
    K = np.load(file_path)
    return {
        'width': 848,
        'height': 480,
        'fx': K[0, 0],
        'fy': K[1, 1],
        'cx': K[0, 2],
        'cy': K[1, 2],
    }

def create_camera_visualizations(poses, intrinsic_o3d, colors=None, scales=None):
    """Create camera visualizations with custom colors and scales."""
    colors = colors or [[1, 0, 0]] * len(poses)
    scales = scales or [CAMERA_SCALE] * len(poses)
    return [
        o3d.geometry.LineSet.create_camera_visualization(intrinsic_o3d, pose, scale).paint_uniform_color(color)
        for pose, color, scale in zip(poses, colors, scales)
    ]

def visualize_geometries(geometries, title):
    """Visualize a list of geometries with a title."""
    print(title)
    o3d.visualization.draw_geometries(geometries)

def create_and_visualize_tsdf(folder_path=DEFAULT_DATA_FOLDER, size=TSDF_SIZE, resolution=TSDF_RESOLUTION,
                              intrinsic_file=DEFAULT_INTRINSIC_FILE, invert_poses=True, debug=False,
                              num_sampled_views=NUM_SAMPLED_VIEWS):
    # Load data
    intrinsic = load_intrinsics(intrinsic_file)
    rgb_imgs, depth_imgs, poses = load_data(folder_path)
    
    if not all([rgb_imgs, depth_imgs, poses]):
        print("No data found in the specified folder")
        return

    min_count = min(len(rgb_imgs), len(depth_imgs), len(poses))
    print(f"Data counts - RGB: {len(rgb_imgs)}, Depth: {len(depth_imgs)}, Poses: {len(poses)}")
    print(f"Using first {min_count} complete frames for TSDF reconstruction")

    rgb_imgs, depth_imgs, poses = rgb_imgs[:min_count], depth_imgs[:min_count], poses[:min_count]
    if invert_poses:
        poses = [np.linalg.inv(pose) for pose in poses]

    # Initialize TSDF and integrate frames
    tsdf = TSDFVolume(size, resolution)
    combined_pcd = o3d.geometry.PointCloud()
    debug_dir = "debug_pcds" if debug else None
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(**intrinsic)
    for i, (rgb_img, depth_img, pose) in enumerate(zip(rgb_imgs, depth_imgs, poses), 1):
        print(f"Integrating frame {i}/{min_count}")
        pcd = tsdf.integrate(rgb_img, depth_img, intrinsic, pose, i, debug_dir)
        combined_pcd += pcd

    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=VOXEL_DOWN_SIZE)
    print(f"Combined Point Cloud Bounds: Min {np.min(combined_pcd.points, axis=0)}, Max {np.max(combined_pcd.points, axis=0)}")

    # Sample viewpoints
    sampler = ViewSampler()
    sampler.pcd = combined_pcd
    sampled_views = sampler.generate_hemisphere_points_with_orientations(radius=0.35, num_points=num_sampled_views)

    sampled_poses = [np.eye(4) for _ in sampled_views]
    for i, view in enumerate(sampled_views):
        sampled_poses[i][:3, :3], sampled_poses[i][:3, 3] = view['rotation'], view['position']
    sampled_poses = [np.linalg.inv(pose) for pose in sampled_poses]

    # Compute information gains
    bbox = np.concatenate([np.min(combined_pcd.points, axis=0), np.max(combined_pcd.points, axis=0)])
    evaluator = ViewEvaluator(tsdf, intrinsic, bbox)
    start_time = time.time()
    gains = [evaluator.compute_information_gain(pose) for pose in sampled_poses]
    print(f"Total gain computation took {time.time() - start_time:.2f} seconds")
    
    gains = np.array(gains)
    print(f"Gains > 0: {gains[gains > 0].shape[0]} views")

    # Identify top 5 gain views
    top_gain_indices = np.argsort(gains)[-TOP_N:][::-1]
    print(f"Top {TOP_N} Gain Views: {[(i + 1, gains[i]) for i in top_gain_indices]}")

    # Visualize cameras
    original_cameras = create_camera_visualizations(poses, intrinsic_o3d, colors=[[0, 0, 1]] * len(poses))
    
    sampled_colors = [[0, 0, 1]] * len(sampled_poses)
    sampled_scales = [CAMERA_SCALE] * len(sampled_poses)
    for rank, idx in enumerate(top_gain_indices):
        sampled_colors[idx] = [1, 0, 0]
    sampled_cameras = create_camera_visualizations(sampled_poses, intrinsic_o3d, colors=sampled_colors, scales=sampled_scales)

    bbox_geom = o3d.geometry.AxisAlignedBoundingBox(min_bound=tsdf.origin, max_bound=tsdf.origin + TSDF_SIZE)
    bbox_geom.color = (0, 1, 0)

    # Visualize results
    visualize_geometries(
        [combined_pcd, bbox_geom] + original_cameras + sampled_cameras,
        "Visualizing combined point cloud (no TSDF) with camera poses..."
    )
    visualize_geometries(
        [tsdf.get_point_cloud(), bbox_geom] + original_cameras + sampled_cameras,
        "Visualizing TSDF point cloud with camera poses..."
    )

def main():
    """Entry point for the script."""
    if not os.path.exists(DEFAULT_INTRINSIC_FILE):
        print(f"Error: Intrinsic file {DEFAULT_INTRINSIC_FILE} not found!")
        return
    
    if not os.path.exists(DEFAULT_DATA_FOLDER):
        print(f"Error: Data folder {DEFAULT_DATA_FOLDER} not found!")
        return
    
    create_and_visualize_tsdf()

if __name__ == "__main__":
    main()