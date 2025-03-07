"""
Validate the dataset by running the simulation with the actions and object poses from the dataset.
"""
from pathlib import Path
from loop_rate_limiters import RateLimiter
from aloha_mink_wrapper import AlohaMinkWrapper
from PIL import Image
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import mink
import numpy as np
import random
import cv2
import threading
import queue
import os
import copy
import h5py
from mug_pickup import initialize_scene, display_image
from mug_pickup_av import move_to_optimal_view, is_gripper_near_optimal_view, move_to_object, transform_to_state, state_to_transform, transform_to_wxyz_xyz
from evaluate_policy_av import move_to_pose, get_robot_data

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"

def set_object_qpos(object_qpos):
    """
    Set the object qpos in the Mujoco model.
    """
    data.qpos[16:23] = object_qpos
    # Forward propagate the simulation state
    mujoco.mj_forward(model, data)

if __name__ == "__main__":
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

    try:
        # Launch the viewer
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Sample object poses
            with h5py.File("datasets\\2025-03-06_13-41-41\episode_0.h5", "r") as f:
                states = f["/observations/qpos"][:]
                actions = f["action"][:]
                object_qpos = f["object_qpos"][:]
                theta = f["theta"][0]
                print("Theta:", theta)
            
            # Set the object qpos
            set_object_qpos(object_qpos)

            # Set the initial posture target
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)

            # Rate limiter for fixed update frequency
            rate = RateLimiter(frequency=100, warn=False)

            episode_cnt = 0
            step_cnt = 0
            near_optimal_view = False
            stage2_reached = False
            av_steps = 0

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
            try:
                while viewer.is_running():
                    if not near_optimal_view:
                        # Move to the optimal view
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
                        stage2_reached = move_to_object(data, model, aloha_mink_wrapper, theta, pos_noise=None, quat_noise=None, stage2_reached=stage2_reached, dt=rate.dt)
                    else:
                        _, state, images = get_robot_data(data, model, renderer, aloha_mink_wrapper, camera_keys=["wrist_cam_right"])
                        tf = aloha_mink_wrapper.transform_left_to_right(data)
                        current_ee_pose = transform_to_state(tf)
                        current_gripper_state = np.array([state[-1]])
                        state = np.concatenate([current_ee_pose, current_gripper_state])
                        action = actions[step_cnt//20] + state
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
                        if action[6] < 0.036:
                            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")] = 0
                        
                        # data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")] = action[6]
                        if reached:
                            step_cnt += 1

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