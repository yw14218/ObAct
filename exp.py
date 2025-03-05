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

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"
theta = 0


def state_to_transform(state):
    """Convert state vector [x, y, z, roll, pitch, yaw] to transformation matrix."""
    t = state[:3]
    r = R.from_euler('xyz', state[3:], degrees=False).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T

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

    try:
        # Launch the viewer
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Sample object poses
            with h5py.File("datasets/2025-03-05_19-19-05/episode_0.h5", "r") as f:
                actions = f["action"][:]
                print(actions.shape)
                object_qpos = f["object_qpos"][:]
            
            # Set the object qpos
            set_object_qpos(object_qpos)

            # Set the initial posture target
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)

            # Rate limiter for fixed update frequency
            rate = RateLimiter(frequency=100, warn=False)

            episode_cnt = 0
            step_cnt = 0

            initialize_spheres(viewer, max_spheres)
            pos_lists = []

            try:
                while viewer.is_running():
                    action = actions[step_cnt][:6]
                    T = state_to_transform(action)
                    
                    T_ee_world_right = np.eye(4)
                    ee_position = data.site_xpos[data.site("right/gripper").id]
                    ee_orientation = data.site_xmat[data.site("right/gripper").id].reshape(3, 3)
                    T_ee_world_right[:3, :3] = ee_orientation
                    T_ee_world_right[:3, 3] = ee_position

                    T_new = T_ee_world_right @ T
                    pos_lists.append(T_new[:3, 3])
                    sphere_index = update_trajectory_spheres(viewer, pos_lists, max_spheres, sphere_index, [1, 0, 0, 1])
                    
                    if step_cnt < len(actions) - 1:
                        step_cnt += 1
    
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