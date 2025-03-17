import os
import glob
import queue
import random
import logging
import threading
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import mujoco
import mink
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from aloha_mink_wrapper import AlohaMinkWrapper
from tsdf_torch_mujoco import TSDFVolume, ViewSampler, ViewEvaluator

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"
K = np.load("robot_control/d405_intrinsic.npy")
INTRINSIC = {
    'width': 640, 'height': 480,
    'fx': K[0, 0], 'fy': K[1, 1],
    'cx': K[0, 2], 'cy': K[1, 2]
}
INTRINSIC_O3D = o3d.camera.PinholeCameraIntrinsic(**INTRINSIC)

def initialize_scene(data, model):
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    aloha_mink_wrapper.configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    aloha_mink_wrapper.initialize_mocap_targets()

def move_to_pose(data, model, aloha_mink_wrapper, pose, rate_dt):
    """Move the robot to the given pose."""
    # Set the target pose
    goal = mink.SE3.from_mocap_name(model, data, "right/target")
    goal.wxyz_xyz[:] = pose
    aloha_mink_wrapper.tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
    aloha_mink_wrapper.tasks[1].set_target(goal)
    # Solve the IK
    aloha_mink_wrapper.solve_ik(rate_dt)
    # Update the configuration
    data.ctrl[aloha_mink_wrapper.actuator_ids] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids]

    return True

def render(data, renderer, object_body_id):
    renderer.update_scene(data, camera="wrist_cam_right")
    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")
    T_gripper_world = np.eye(4)
    T_gripper_world[:3, :3] = data.site_xmat[gripper_site_id].reshape(3, 3)
    T_gripper_world[:3, 3] = data.site_xpos[gripper_site_id]
    extrinsic = T_gripper_world @ np.load("robot_control/d405_extrinsic.npy")    
    renderer.enable_segmentation_rendering()
    seg_map = renderer.render()
    mask = (seg_map[:, :, 0] == model.body_geomadr[object_body_id]).astype(np.uint8)
    renderer.disable_segmentation_rendering()
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    
    return rgb, depth, mask, extrinsic

if __name__ == "__main__":
    # Initialization
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)
    tsdf = TSDFVolume(0.5, 64)
    sampler = ViewSampler()
    aloha_mink_wrapper = AlohaMinkWrapper(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    initialize_scene(data, model)
    
    np.random.seed(2)
    
    try:
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
            rate = RateLimiter(frequency=100, warn=False)
            
            # Initialize viewpoints
            object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
            object_pos = data.xpos[object_body_id]
            sampled_viewpoints = sampler.generate_hemisphere_points_with_orientations(
                center=object_pos, radius=0.3, num_points=128
            )
            
            for i, viewpoint in enumerate(sampled_viewpoints):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0], pos=viewpoint['position'],
                    mat=viewpoint['rotation'].flatten(), rgba=[1, 0, 0, 1]
                )
            viewer.user_scn.ngeom = len(sampled_viewpoints)
            
            rgb, depth, mask, extrinsic = render(data, renderer, object_body_id)    
            mask = mask.astype(np.bool_)
            rgb = rgb * mask[:, :, None]
            depth = depth * mask
            initial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(     
                o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
                    depth_scale=1.0, depth_trunc=0.7, convert_rgb_to_intensity=False
                ), INTRINSIC_O3D, extrinsic
            )
            o3d.io.write_point_cloud("initial_pcd.ply", initial_pcd)
            bbox = np.concatenate([np.min(initial_pcd.points, axis=0), np.max(initial_pcd.points, axis=0)])

            rgb, depth, mask, extrinsic = render(data, renderer, object_body_id)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
                depth_scale=1.0, depth_trunc=0.7, convert_rgb_to_intensity=False
            )

            save_dir = Path("saved_data") 
            save_dir.mkdir(parents=True, exist_ok=True) 
            rgb_filename = save_dir / "rgb.png"
            mask_filename = save_dir / "mask.png"
            depth_filename = save_dir / "depth.npy"
            pose_filename = save_dir / "pose.npy"
            bbox_filename = save_dir / "bbox.npy"

            cv2.imwrite(str(rgb_filename), rgb)
            cv2.imwrite(str(mask_filename), mask * 255)
            np.save(depth_filename, depth)
            np.save(pose_filename, extrinsic)
            np.save(bbox_filename, bbox)
            print(f"Saved data to {save_dir}", object_pos)

            tsdf._volume.integrate(rgbd, INTRINSIC_O3D, extrinsic)
            evaluator = ViewEvaluator(tsdf, INTRINSIC, bbox)
            sampled_poses = [np.eye(4) for _ in sampled_viewpoints]
            for i, view in enumerate(sampled_viewpoints):
                sampled_poses[i][:3, :3], sampled_poses[i][:3, 3] = view['rotation'], view['position']
            sampled_poses = [np.linalg.inv(pose) for pose in sampled_poses]

            gains = [evaluator.compute_information_gain(pose) for pose in sampled_poses]
            top_gain_indices = np.argsort(gains)[-1:][::-1]
            
            gains = np.array(gains)
            print(f"Gains > 0: {gains[gains > 0].shape[0]} views")

            while viewer.is_running():
                aloha_mink_wrapper.compensate_gravity([
                    model.body("left/base_link").id,
                    model.body("right/base_link").id
                ])
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)

                # Update viewpoints
                for i, viewpoint in enumerate(sampled_viewpoints):
                    viewer.user_scn.geoms[i].pos[:] = viewpoint['position']
                    viewer.user_scn.geoms[i].mat[:] = viewpoint['rotation']

                    if i in top_gain_indices:
                        viewer.user_scn.geoms[i].rgba[:] = [1, 0, 0, 1]
                    else:
                        viewer.user_scn.geoms[i].rgba[:] = [0, 0, 1, 1]

                # Get the highest gain camera pose
                camera_pose_highest_gain = sampled_viewpoints[top_gain_indices[0]]

                camera_pose = np.eye(4)
                camera_pose[:3, :3] = camera_pose_highest_gain['rotation']
                camera_pose[:3, 3] = camera_pose_highest_gain['position']
                camera_eef = np.load("robot_control/d405_extrinsic.npy")
                eef_pose = camera_pose @ np.linalg.inv(camera_eef)
                move_to_pose(data, model, aloha_mink_wrapper, mink.SE3.from_matrix(eef_pose).wxyz_xyz, rate.dt)

                viewer.sync()
                rate.sleep()
                
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Exiting gracefully...")
        
    finally:
        cv2.destroyAllWindows()