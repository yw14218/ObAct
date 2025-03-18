import os
import numpy as np
import mujoco
import mujoco.viewer
import mink
import open3d as o3d
from pathlib import Path
import logging
import cv2
from loop_rate_limiters import RateLimiter
from aloha_mink_wrapper import AlohaMinkWrapper
from tsdf_torch_mujoco import TSDFVolume, ViewSampler, ViewEvaluator
from lightglue import LightGlue

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"
K = AlohaMinkWrapper.get_K(480, 640)
INTRINSIC = {'width': 640, 'height': 480, 'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]}
INTRINSIC_O3D = o3d.camera.PinholeCameraIntrinsic(**INTRINSIC)

def initialize_scene(model, data, aloha_mink_wrapper):
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    aloha_mink_wrapper.configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    aloha_mink_wrapper.initialize_mocap_targets()

def get_optical_frame(model, data, camera_site_name="wrist_cam_right_site"):
    camera_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, camera_site_name)
    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")
    if camera_site_id == -1 or gripper_site_id == -1:
        raise ValueError(f"Site not found: {camera_site_name} or right/gripper")

    camera_frame = np.eye(4)
    camera_frame[:3, 3] = data.site_xpos[camera_site_id]
    camera_frame[:3, :3] = data.site_xmat[camera_site_id].reshape(3, 3)
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    camera_frame[:3, :3] = camera_frame[:3, :3] @ R

    gripper_frame = np.eye(4)
    gripper_frame[:3, 3] = data.site_xpos[gripper_site_id]
    gripper_frame[:3, :3] = data.site_xmat[gripper_site_id].reshape(3, 3)

    camera_gripper = np.linalg.inv(gripper_frame) @ camera_frame
    np.save("tasks/handeye_mujoco.npy", camera_gripper)
    return gripper_frame @ camera_gripper

def render(data, renderer, object_body_id):
    renderer.update_scene(data, camera="wrist_cam_right")
    renderer.enable_segmentation_rendering()
    seg_map = renderer.render()
    mask = (seg_map[:, :, 0] == model.body_geomadr[object_body_id]).astype(np.uint8)
    renderer.disable_segmentation_rendering()
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    return rgb, depth, mask, get_optical_frame(model, data)

def save_data(object_pos, rgb, depth, mask, extrinsic, bbox, save_dir="saved_data"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    existing_indices = [int(p.stem.split('_')[-1]) for p in save_dir.glob("rgb_*.png")]
    idx = max(existing_indices, default=-1) + 1

    for name, data in [("rgb", rgb), ("mask", mask * 255), ("depth", depth), ("pose", extrinsic), ("bbox", bbox)]:
        filename = save_dir / f"{name}_{idx}.{'png' if name in ['rgb', 'mask'] else 'npy'}"
        (cv2.imwrite(str(filename), data) if name in ["rgb", "mask"] else np.save(filename, data))
    logging.info(f"Saved data to {save_dir} with index {idx}")

def has_arm_stopped(data, goal, side, threshold=0.001):
    if side == "left":
        gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
        return np.sum(np.abs(data.qvel[:6])) < threshold and np.linalg.norm(goal.wxyz_xyz[4:] - data.site_xpos[gripper_site_id]) < threshold
    else:
        gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")
        return np.sum(np.abs(data.qvel[12:17])) < threshold and np.linalg.norm(goal.wxyz_xyz[4:] - data.site_xpos[gripper_site_id]) < threshold

def move_to_pose(model, data, aloha_mink_wrapper, pose, rate_dt, renderer, object_body_id, object_pos, bbox):
    global has_reached_next_pose

    goal = mink.SE3.from_mocap_name(model, data, "right/target")
    goal.wxyz_xyz[:] = pose
    xcoord = pose[4].copy()

    xcoord = 1 # force to move to the right for now
    if xcoord > 0:
        aloha_mink_wrapper.tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
        aloha_mink_wrapper.tasks[1].set_target(goal)
        side = "right"
    elif xcoord < 0:
        aloha_mink_wrapper.tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
        aloha_mink_wrapper.tasks[0].set_target(goal)
        side = "left"
    else:
        raise NotImplementedError("Cannot move to a pose with x-coordinate 0")
    
    aloha_mink_wrapper.solve_ik(rate_dt)
    data.ctrl[aloha_mink_wrapper.actuator_ids] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids]

    mujoco.mj_step(model, data)  # Step the simulation to update positions/velocities
    if has_arm_stopped(data, goal, side):
        rgb, depth, mask, extrinsic = render(data, renderer, object_body_id)
        save_data(object_pos, rgb, depth, mask, extrinsic, bbox)
        has_reached_next_pose = True
        return rgb, depth, mask, extrinsic
    return None

def process_viewpoint_cycle(data, renderer, tsdf, sampler, object_body_id, viewer, bbox, visited_indices):
    object_pos = data.xpos[object_body_id]
    rgb, depth, mask, extrinsic = render(data, renderer, object_body_id)
    mask = mask.astype(bool)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb * mask[..., None]), o3d.geometry.Image(depth * mask),
        depth_scale=1.0, depth_trunc=0.7, convert_rgb_to_intensity=False
    )

    tsdf._volume.integrate(rgbd, INTRINSIC_O3D, np.linalg.inv(extrinsic))
    evaluator = ViewEvaluator(tsdf, INTRINSIC, bbox)
    
    print("object_pose: " ,object_pos)
    sampled_viewpoints = sampler.generate_hemisphere_points_with_orientations(
        center=object_pos, radius=0.35, num_points=48
    )
    for i, vp in enumerate(sampled_viewpoints):
        mujoco.mjv_initGeom(viewer.user_scn.geoms[i], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.005, 0, 0], pos=vp['position'], mat=vp['rotation'].flatten(), rgba=[0, 0, 1, 1])
    viewer.user_scn.ngeom = len(sampled_viewpoints)

    sampled_poses = [np.eye(4) for _ in sampled_viewpoints]
    for i, vp in enumerate(sampled_viewpoints):
        sampled_poses[i][:3, :3], sampled_poses[i][:3, 3] = vp['rotation'], vp['position']
    sampled_poses = [np.linalg.inv(pose) for pose in sampled_poses]
    gains = np.array([evaluator.compute_information_gain(pose) for pose in sampled_poses])

    for i in visited_indices:
        gains[i] = -1

    logging.info(visited_indices, "is visited")
    top_idx = np.argmax(gains)
    logging.info(f"Top gain at index {top_idx}: {gains[top_idx]}, {gains[gains > 0].shape[0]} views > 0")

    viewer.user_scn.geoms[top_idx].rgba[:] = [1, 0, 0, 1]
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = sampled_viewpoints[top_idx]['rotation']
    camera_pose[:3, 3] = sampled_viewpoints[top_idx]['position']
    eef_pose = camera_pose @ np.linalg.inv(np.load("tasks/handeye_mujoco.npy"))

    visited_indices.append(top_idx)

    return mink.SE3.from_matrix(eef_pose).wxyz_xyz, sampled_viewpoints

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)
    tsdf = TSDFVolume(0.5, 64)
    sampler = ViewSampler()
    aloha_mink_wrapper = AlohaMinkWrapper(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    has_reached_next_pose = False
    visited_indices = []
    initialize_scene(model, data, aloha_mink_wrapper)
    os.system('rm -r saved_data')
    np.random.seed(3)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
        rate = RateLimiter(frequency=100, warn=False)
        object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

        # Initial render and bbox setup
        mujoco.mj_step(model, data)
        rgb, depth, mask, extrinsic = render(data, renderer, object_body_id)
        initial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb * mask[..., None].astype(bool)),
                o3d.geometry.Image(depth * mask.astype(bool)),
                depth_scale=1.0, depth_trunc=0.7, convert_rgb_to_intensity=False
            ), INTRINSIC_O3D, np.linalg.inv(extrinsic)
        )
        bbox = np.concatenate([np.min(initial_pcd.points, axis=0), np.max(initial_pcd.points, axis=0)])
        bbox = np.load("bbox_0.npy")
        save_data(data.xpos[object_body_id], rgb, depth, mask, extrinsic, bbox)
        
        # Main loop: 5 iterations
        for iteration in range(5):
            logging.info(f"Starting iteration {iteration + 1}/5")
            next_pose, viewpoints = process_viewpoint_cycle(data, renderer, tsdf, sampler, object_body_id, viewer, bbox, visited_indices)
            has_reached_next_pose = False

            # Move to the next pose until the arm stops at that position
            while viewer.is_running() and not has_reached_next_pose:
                aloha_mink_wrapper.compensate_gravity([model.body("left/base_link").id, model.body("right/base_link").id])
                result = move_to_pose(model, data, aloha_mink_wrapper, next_pose, rate.dt, renderer, object_body_id, data.xpos[object_body_id], bbox)
                if result:
                    rgb, depth, mask, extrinsic = result
                    tsdf._volume.integrate(
                        o3d.geometry.RGBDImage.create_from_color_and_depth(
                            o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
                            depth_scale=1.0, depth_trunc=0.7, convert_rgb_to_intensity=False
                        ), INTRINSIC_O3D, np.linalg.inv(extrinsic)
                    )
                for i, vp in enumerate(viewpoints):
                    viewer.user_scn.geoms[i].pos[:] = vp['position']
                    viewer.user_scn.geoms[i].mat[:] = vp['rotation']
                viewer.sync()
                rate.sleep()

            logging.info(f"Completed iteration {iteration + 1}/5")

        logging.info("Finished all iterations")
        o3d.visualization.draw_geometries(tsdf.get_point_cloud())
