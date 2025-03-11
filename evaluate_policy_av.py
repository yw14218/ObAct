"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""

from pathlib import Path

import imageio
import numpy as np
import torch
from pathlib import Path
from loop_rate_limiters import RateLimiter
from aloha_mink_wrapper import AlohaMinkWrapper
import mujoco
import mujoco.viewer
import cv2
import threading
import queue
from torchvision.transforms import v2
import mink
import copy

from lerobot.common.policies.act.modeling_act import ACTPolicy

from mug_pickup import initialize_scene, display_image, sample_object_position, check_object_lifted, get_robot_data
from mug_pickup_av import move_to_optimal_view, is_gripper_near_optimal_view, move_to_object, sample_constrained_pos_noise, sample_constrained_quat_noise, state_to_transform, transform_to_wxyz_xyz, transform_to_state

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"
theta = 0

def set_object_qpos(object_qpos):
    """
    Set the object qpos in the Mujoco model.
    """
    data.qpos[16:23] = object_qpos
    # Forward propagate the simulation state
    mujoco.mj_forward(model, data)

def preprocess_image(img):
    """Preprocess the image to be compatible with the policy."""
    # normalize
    img = img / 255
    # permute to channel first
    img = img.permute(2, 0, 1)
    # resize to 240x320
    img = v2.Resize((240, 320))(img)
    # center crop to 224x308
    img = v2.CenterCrop((224, 308))(img)
    return img

def move_to_pose(data, model, aloha_mink_wrapper, pose, rate_dt):
    """Move the robot to the given pose."""
    # Set the target pose
    goal = mink.SE3.from_mocap_name(model, data, "left/target")
    goal.wxyz_xyz[:] = pose
    aloha_mink_wrapper.tasks[0].set_target(goal)
    aloha_mink_wrapper.tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
    # Solve the IK
    aloha_mink_wrapper.solve_ik(rate_dt)
    # Update the configuration
    data.ctrl[aloha_mink_wrapper.actuator_ids[:6]] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids[:6]]
    # Calculate the error
    gripper_pos = data.site_xpos[data.site("left/gripper").id]
    error = np.linalg.norm(goal.wxyz_xyz[4:] - gripper_pos)
    qvel_raw = data.qvel.copy()
    left_qvel_raw = qvel_raw[:8]
    sum_of_vel = np.sum(np.abs(left_qvel_raw))
    return True

if __name__ == "__main__":
    # Load the pretrained policy
    pretrained_policy_path = Path("ckpts\\mug_pickup_av\\100001\\pretrained_model")

    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        device = torch.device("cpu")
        print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
    policy.to(device)
    # policy.config.temporal_ensemble_coeff=None
    policy.config.temporal_ensemble_coeff=0.01
    # Reset the policy and environmens to prepare for rollout
    policy.reset()

    # Load the Mujoco model and data
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    # Initialize the aloha_mink_wrapper
    aloha_mink_wrapper = AlohaMinkWrapper(model, data)

    # Initialize to the neutral pose
    initialize_scene(data, model, aloha_mink_wrapper)

    renderer = mujoco.Renderer(model, 480, 640)
    
    # Create a thread-safe queue and running event
    img_queue = queue.Queue(maxsize=1)
    running_event = threading.Event()
    running_event.set()

    # Start the display thread
    display_thread = threading.Thread(target=display_image, args=(img_queue, running_event))
    display_thread.start()
    np.random.seed(0)
    try:
        # Launch the viewer
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Randomize the object's position
            object_qpos, theta = sample_object_position(data, model)

            # Set the initial posture target
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)

            # Rate limiter for fixed update frequency
            rate = RateLimiter(frequency=100, warn=False)

            episode_cnt = 0
            step_cnt = 0
            success_cnt = 0
            av_steps = 0
            cam_imgs = {}
            camera_keys = ["wrist_cam_right"]
            near_optimal_view = False
            stage2_reached = False
            move_to_pose_cnt = 0
            pos_noise = sample_constrained_pos_noise()
            quat_noise = sample_constrained_quat_noise()

            max_spheres = 100  # Maximum number of spheres to visualize the trajectory
            sphere_radius = 0.01  # Radius of the spheres

            # Function to initialize spheres off-screen
            def initialize_spheres(viewer, max_spheres):
                for i in range(max_spheres):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[sphere_radius, 0, 0],
                        pos=[-100, -100, -100],  # Place off-screen
                        mat=np.eye(3).flatten(),
                        rgba=[0, 0, 0, 0]  # Make transparent
                    )
                viewer.user_scn.ngeom = max_spheres

            # Function to update the positions of the spheres
            def update_trajectory_spheres(viewer, positions, max_spheres, sphere_index, color):
                # Update the position of the current sphere
                if len(positions) > 0:
                    viewer.user_scn.geoms[sphere_index].pos[:] = positions[-1]
                    viewer.user_scn.geoms[sphere_index].rgba[:] = color

                # Return the next sphere index
                return (sphere_index + 1) % max_spheres

            # Initialize the spheres
            sphere_index = 0
            initialize_spheres(viewer, max_spheres)
            pos_lists = []

            print(f"Episode {episode_cnt} started...")

            try:
                while viewer.is_running():
                    if not near_optimal_view:
                        # Move the robot to the optimal view of the object
                        move_to_optimal_view(data, model, aloha_mink_wrapper, theta, rate.dt)
                        av_steps += 1

                        # Check if the gripper is near the object's optimal view
                        if is_gripper_near_optimal_view(data, model) and av_steps > 50:
                            print("Optimal view reached. Moving to object...")
                            for i in range(100):
                                mujoco.mj_step(model, data)
                            near_optimal_view = True
                            av_steps = 0
                    elif not stage2_reached:
                        # Align gripper with the object
                        stage2_reached = move_to_object(data, model, aloha_mink_wrapper, theta, pos_noise=pos_noise, quat_noise=quat_noise, stage2_reached=stage2_reached, dt=rate.dt)
                    else:
                        _, state, images = get_robot_data(data, model, renderer, aloha_mink_wrapper, camera_keys=camera_keys)
                        tf = aloha_mink_wrapper.transform_left_to_right(data)
                        current_ee_pose = transform_to_state(tf)
                        current_gripper_state = np.array([state[-1]])
                        state = np.concatenate([current_ee_pose, current_gripper_state])
                        state = torch.from_numpy(state).float().to(device)
                        for key, img in zip(camera_keys, images):
                            # # save the image
                            # cv2.imwrite(f"images/{episode_cnt}_{step_cnt}_{key}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                            cam_imgs[key] = torch.from_numpy(img).float().to(device)
                            cam_imgs[key] = preprocess_image(cam_imgs[key])

                        # prepare observation
                        observation = {
                            "observation.state": state.unsqueeze(0),
                        }
                        for key in camera_keys:
                            observation[f"observation.images.{key}"] = cam_imgs[key].unsqueeze(0)
                        cam_imgs = {}
                        if move_to_pose_cnt == 0 or move_to_pose_cnt > 4:
                            # Predict the next action with respect to the current observation
                            with torch.inference_mode():
                                action = policy.select_action(observation)
                            move_to_pose_cnt = 0
                            step_cnt += 1
                            action = action.squeeze(0).to("cpu").numpy()
                        state = state.cpu().numpy().squeeze()
                        tf_left_to_right = state_to_transform(action[:6])
                        tf_right_to_world = aloha_mink_wrapper.transform_right_to_world(data)
                        tf_left_to_world = tf_right_to_world @ tf_left_to_right
                        # transform transformation matrix to wxyz_xyz
                        wxyz_xyz = transform_to_wxyz_xyz(tf_left_to_world)
                        pos_lists.append(wxyz_xyz[4:])
                        sphere_index = update_trajectory_spheres(viewer, pos_lists, max_spheres, sphere_index, [1, 0, 0, 1])
                        # move to the pose
                        reached = move_to_pose(data, model, aloha_mink_wrapper, wxyz_xyz, rate.dt)
                        gripper_action = action[6]
                        if gripper_action < 0.036:
                            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")] = 0
                        else:
                            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")] = 0.037
                        
                        move_to_pose_cnt += 1
                        
                        # print("Step count:", step_cnt)

                        # Check if the object is lifted
                        if check_object_lifted(data, model, 0.05):
                            print("Object lifted!")
                            episode_cnt += 1
                            success_cnt += 1
                            step_cnt = 0
                            near_optimal_view = False
                            pos_noise = sample_constrained_pos_noise()
                            quat_noise = sample_constrained_quat_noise()
                            stage2_reached = False
                            print(f"Success Rate: {success_cnt}/{episode_cnt}")
                            # Reinitalize the scene
                            initialize_scene(data, model, aloha_mink_wrapper)
                            # Reset the object's position
                            object_qpos, theta = sample_object_position(data, model)
                            # Set the initial posture target
                            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
                            policy.reset()
                            print(f"Episode {episode_cnt} started...")
                        
                        # Check if the episode has failed
                        if step_cnt > 600:
                            print("Episode failed. Resetting the scene...")
                            episode_cnt += 1
                            step_cnt = 0
                            near_optimal_view = False
                            pos_noise = sample_constrained_pos_noise()
                            quat_noise = sample_constrained_quat_noise()
                            stage2_reached = False
                            print(f"Success Rate: {success_cnt}/{episode_cnt}")
                            # Reinitalize the scene
                            initialize_scene(data, model, aloha_mink_wrapper)
                            # Reset the object's position
                            object_qpos, theta = sample_object_position(data, model)
                            # Set the initial posture target
                            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
                            policy.reset()
                            print(f"Episode {episode_cnt} started...")

                    # Compensate gravity
                    aloha_mink_wrapper.compensate_gravity([model.body("left/base_link").id, model.body("right/base_link").id])

                    # Step the simulation
                    mujoco.mj_step(model, data)

                    # Visualize at fixed FPS
                    viewer.sync()
                    rate.sleep()

            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Exiting gracefully...")

    finally:
        # Cleanup
        running_event.clear()
        img_queue.put(None)  # Signal thread to exit
        display_thread.join()
        cv2.destroyAllWindows()




