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
from mug_pickup import initialize_scene, display_image, sample_object_position
from mug_pickup_av import move_to_optimal_view, is_gripper_near_optimal_view, move_to_object, transform_to_state, state_to_transform, transform_to_wxyz_xyz
from evaluate_policy_av import move_to_pose_left, move_to_pose_right, get_robot_data

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
            with h5py.File("real_datasets_v3\episode_20250404_225112.h5", "r") as f:
                # states = f["/observations/qpos"][:]
                actions = f["/observations/ee_pose"][:]
                # camera_actions = f["/camera_pose/qpos"][:]
                camera_pose = f["/camera_pose/ee_pose"][:]
                images = f["/observations/images/rgbs"][:]
            
            # states = []
            # actions = []
            # imgs = []
            # for i in range(1, len(robot_poses)):
            #     # left
            #     T_left_base_world = aloha_mink_wrapper.get_left_base_to_world(data)
            #     state_left_base = robot_poses[i-1]
            #     state_left_world = T_left_base_world @ state_left_base
            #     action_left_base = robot_poses[i]
            #     action_left_world = T_left_base_world @ action_left_base
                
            #     # right
            #     T_right_base_world = aloha_mink_wrapper.get_right_base_to_world(data)
            #     state_right_base = camera_pose
            #     state_right_world = T_right_base_world @ state_right_base

            #     state_left_right = AlohaMinkWrapper.pose_inv(state_left_world) @ state_right_world
            #     action_left_right = AlohaMinkWrapper.pose_inv(action_left_world) @ state_right_world
            #     state = transform_to_state(state_left_right)
            #     action = transform_to_state(action_left_right)
            #     states.append(state)
            #     actions.append(action)

            #     image = images[i-1]
            #     # visualize the image
            #     cv2.imshow("image", image)
            #     cv2.waitKey(1)

            #     imgs.append(image)
            
            # import datetime
            # now = datetime.datetime.now()
            # folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
            # os.makedirs(f"datasets/{folder_name}", exist_ok=True)
            # with h5py.File(f"datasets/{folder_name}/episode_0.h5", "w") as f:
            #     f.create_dataset("/observations/qpos", data=states)
            #     f.create_dataset("/action", data=actions)
            #     f.create_dataset("/observations/images/wrist_cam_right", data=imgs)
            
            
            # Set the object qpos
            object_qpos, theta = sample_object_position(data, model)
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
                    # left
                    T_left_base_world = aloha_mink_wrapper.get_left_base_to_world(data)
                    action_left_base = actions[step_cnt // 3]
                    action_left_world = T_left_base_world @ action_left_base
                    action_left = transform_to_wxyz_xyz(action_left_world)
                    # move_to_pose_left(data, model, aloha_mink_wrapper, action_left, rate.dt)
                    
                    # right
                    T_right_base_world = aloha_mink_wrapper.get_right_base_to_world(data)
                    action_right_base = camera_pose
                    action_right_world = T_right_base_world @ action_right_base
                    action_right = transform_to_wxyz_xyz(action_right_world)
                    move_to_pose_right(data, model, aloha_mink_wrapper, action_right, rate.dt)

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