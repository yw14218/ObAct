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
import mujoco.viewer
import mink
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from aloha_mink_wrapper import AlohaMinkWrapper
from tsdf_torch_mujoco import TSDFVolume, ViewSampler, ViewEvaluator
from dm_control import mujoco

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
Render_once  = False

def initialize_scene(data, model):
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    aloha_mink_wrapper.configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    aloha_mink_wrapper.initialize_mocap_targets()

def move_right_arm_to_pose(data, model, aloha_mink_wrapper, pose, rate_dt):
    """Move the robot to the given pose."""
    global Render_once
    # Set the target pose
    goal = mink.SE3.from_mocap_name(model, data, "right/target")
    goal.wxyz_xyz[:] = pose
    aloha_mink_wrapper.tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
    aloha_mink_wrapper.tasks[1].set_target(goal)
    # Solve the IK
    aloha_mink_wrapper.solve_ik(rate_dt)
    # Update the configuration
    data.ctrl[aloha_mink_wrapper.actuator_ids] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids]

    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")
    qvel_raw = data.qvel.copy()
    left_qvel_raw = qvel_raw[12:19]
    sum_of_vel = np.sum(np.abs(left_qvel_raw))
    if sum_of_vel < 0.002 and not Render_once:
        Render_once = True
        rgb, depth, mask, extrinsic = render(data, renderer, object_body_id)
        save_data(object_pos, rgb, depth, mask, extrinsic, bbox)
        print("Rendered!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print(sum_of_vel)

def render(data, renderer, object_body_id):
    renderer.update_scene(data, camera="wrist_cam_right")
    
    extrinsic[:3, :3] = data.site_xmat[gripper_site_id].reshape(3, 3)
    extrinsic[:3, 3] = data.site_xpos[gripper_site_id]

    extrinsic = get_camera_optical_frame(model, data, "wrist_cam_right") 

    renderer.enable_segmentation_rendering()
    seg_map = renderer.render()
    mask = (seg_map[:, :, 0] == model.body_geomadr[object_body_id]).astype(np.uint8)
    renderer.disable_segmentation_rendering()
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    
    return rgb, depth, mask, extrinsic

def save_data(object_pos, rgb, depth, mask, extrinsic, bbox):
    save_dir = Path("saved_data")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine the next available index
    existing_indices = [int(p.stem.split('_')[-1]) for p in save_dir.glob("rgb_*.png")]
    next_index = max(existing_indices, default=-1) + 1
    
    rgb_filename = save_dir / f"rgb_{next_index}.png"
    mask_filename = save_dir / f"mask_{next_index}.png"
    depth_filename = save_dir / f"depth_{next_index}.npy"
    pose_filename = save_dir / f"pose_{next_index}.npy"
    bbox_filename = save_dir / f"bbox_{next_index}.npy"

    cv2.imwrite(str(rgb_filename), rgb)
    cv2.imwrite(str(mask_filename), mask * 255)
    np.save(depth_filename, depth)
    np.save(pose_filename, extrinsic)
    np.save(bbox_filename, bbox)
    print(f"Saved data to {save_dir} with index {next_index}", object_pos)

if __name__ == "__main__":
    # Initialization
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)
    physics = mujoco.Physics.from_xml_path(str(_XML))
    tsdf = TSDFVolume(0.5, 64)
    sampler = ViewSampler()
    aloha_mink_wrapper = AlohaMinkWrapper(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    initialize_scene(data, model)
    
    # camera = mujoco.Camera(physics)
    # camera_matrix = camera.matrix
    # print(camera_matrix, "camera ")
    if os.path.exists('saved_data'):
        os.system('rm -r saved_data')
    np.random.seed(2)
    
    try:
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
            rate = RateLimiter(frequency=100, warn=False)
            
            # Initialize viewpoints
            object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
            object_pos = data.xpos[object_body_id]
            
            rgb, depth, mask, extrinsic = render(data, renderer, object_body_id)    
            mask = mask.astype(np.bool_)
            rgb = rgb * mask[:, :, None]
            depth = depth * mask
            initial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(     
                o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
                    depth_scale=1.0, depth_trunc=0.7, convert_rgb_to_intensity=False
                ), INTRINSIC_O3D, np.linalg.inv(extrinsic)
            )
            bbox = np.concatenate([np.min(initial_pcd.points, axis=0), np.max(initial_pcd.points, axis=0)])
            bbox_center = (bbox[:3] + bbox[3:]) / 2

            sampled_viewpoints = sampler.generate_hemisphere_points_with_orientations(
                center=object_pos, radius=0.35, num_points=128, hemisphere='right'
            )
            
            for i, viewpoint in enumerate(sampled_viewpoints):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0], pos=viewpoint['position'],
                    mat=viewpoint['rotation'].flatten(), rgba=[1, 0, 0, 1]
                )
            viewer.user_scn.ngeom = len(sampled_viewpoints)

            rgb, depth, mask, extrinsic = render(data, renderer, object_body_id)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
                depth_scale=1.0, depth_trunc=0.7, convert_rgb_to_intensity=False
            )

            save_data(object_pos, rgb, depth, mask, extrinsic, bbox)

            tsdf._volume.integrate(rgbd, INTRINSIC_O3D, np.linalg.inv(extrinsic))
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
                np.save("saved_data/pose_2.npy", camera_pose)
                camera_eef = np.load("mujoco_handeye.npy")
                eef_pose = camera_pose @ np.linalg.inv(camera_eef)
                move_right_arm_to_pose(data, model, aloha_mink_wrapper, mink.SE3.from_matrix(camera_pose).wxyz_xyz, rate.dt)
                viewer.sync()
                rate.sleep()
                
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Exiting gracefully...")
        
    finally:
        cv2.destroyAllWindows()