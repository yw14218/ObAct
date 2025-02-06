from pathlib import Path
from loop_rate_limiters import RateLimiter
from aloha_mink_wrapper import AlohaMinkWrapper
from PIL import Image
from scipy.spatial.transform import Rotation as R
import mujoco
import mink
import numpy as np
import random
import cv2
import threading
import queue
import os
import copy

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"
theta = 0

def sample_object_position(data, model, x_range=(-0.075, 0.075), y_range=(-0.075, 0.075), yaw_range=(-np.pi / 4, np.pi / 4)):
    """Randomize the object's position in the scene for a free joint."""
    global theta
    # Get the free joint ID (first free joint in the system)
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    theta = np.random.uniform(*yaw_range)
    print(theta)
    # Update position in the free joint's `data.qpos`
    data.qpos[16:23] = [
        np.random.uniform(*x_range),  # Randomize x position
        np.random.uniform(*y_range),  # Randomize y position
        0,  # Randomize z position
        -np.sin(theta/2),  # Randomize w position
        0,  # Randomize qx position
        0,  # Randomize qy position
        np.cos(theta/2)  # Randomize qz position
    ]

    # Forward propagate the simulation state
    mujoco.mj_forward(model, data)

    # Log the new position for debugging
    print(f"New object position: {data.xpos[object_body_id]}")

def compute_approach_pose(goal, offset_distance=0.07):
    # Create a copy of the goal pose to avoid modifying the original
    approach_pose = copy.deepcopy(goal)
    
    # Extract the orientation (quaternion)
    quat = approach_pose.wxyz_xyz[0:4]
    
    # Extract the current position
    position = approach_pose.wxyz_xyz[4:]
    
    # Calculate the approach direction (e.g., along the z-axis of the quaternion)
    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quat).as_matrix()
    
    # The approach vector (assuming z-axis of the rotation matrix is forward)
    approach_vector = rotation_matrix[:, 2]  # Z-axis of the rotation matrix
    approach_vector[1] *= -1
    print(approach_vector)
    # Offset the position by the approach vector scaled by the distance
    approach_position = position - approach_vector * offset_distance
    
    # Update the copied pose with the new position
    approach_pose.wxyz_xyz[4:] = approach_position

    return approach_pose

def move_to_object():
    """Move the gripper to align with the object's position."""
    global theta
    mink.move_mocap_to_frame(model, data, "left/target", "handle_site", "site")

    # Update task targets
    goal = mink.SE3.from_mocap_name(model, data, "left/target")
    goal.wxyz_xyz[-1] -= 0.02
    quat = R.from_euler('xyz', [0, np.pi / 4, theta], degrees=False).as_quat(scalar_first=True)
    goal.wxyz_xyz[0:4] = quat

    aloha_mink_wrapper.tasks[0].set_target(compute_approach_pose(goal))
    aloha_mink_wrapper.tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
    
    # Solve inverse kinematics
    aloha_mink_wrapper.solve_ik(rate.dt)
    # Apply the calculated joint positions to actuators
    data.ctrl[aloha_mink_wrapper.actuator_ids] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids]

def is_gripper_near_object(vel_threshold=0.5, dis_threshold=0.3):
    """Check if the gripper is close enough to the object."""
    object_position = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handle_site")]
    gripper_position = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")]

    distance = np.linalg.norm(gripper_position - object_position)
    qvel_raw = data.qvel.copy()
    left_qvel_raw = qvel_raw[:8]
    sum_of_vel = np.sum(np.abs(left_qvel_raw))
    
    return sum_of_vel < vel_threshold and distance < dis_threshold

def close_gripper():
    """Gradually close the gripper."""
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")] = 0

def lift_object(lift_height=0.15):
    """Lift the gripper by a specified height."""
    # Get current gripper position
    current_position = mink.SE3.from_mocap_name(model, data, "left/target")
    
    # Set the goal position slightly higher
    goal = current_position.copy()
    goal.wxyz_xyz[0:4] = [1, 0, 0, 0]
    goal.wxyz_xyz[-1] += lift_height
    aloha_mink_wrapper.tasks[0].set_target(goal)

    # Solve inverse kinematics
    aloha_mink_wrapper.solve_ik(rate.dt)

    # Apply the calculated joint positions to actuators
    data.ctrl[aloha_mink_wrapper.actuator_ids] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids]

def check_object_lifted():
    """Check if the object has been lifted to the desired height."""
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_position = data.xpos[object_body_id]

    # Check if the object has reached or exceeded the target lift height
    return object_position[-1] >= 0.08

def initialize_scene(data, model):
    """Initialize the scene to reset the task."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    aloha_mink_wrapper.configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    aloha_mink_wrapper.initialize_mocap_targets()

def display_image(img_queue, running_event):
    # Create a directory to save images if it doesn't exist
    os.makedirs('camera_frames', exist_ok=True)
    frame_count = 0

    while running_event.is_set():
        try:
            img = img_queue.get(timeout=1)
            if img is None:
                break
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img[:, :, ::-1])
            
            # Save the image
            frame_filename = f'camera_frames/frame_{frame_count:04d}.png'
            # pil_img.save(frame_filename)
            
            frame_count += 1
            
            # Optional: print saved frame info
            # print(f"Saved {frame_filename}")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error saving image: {e}")


if __name__ == "__main__":
    # Load the Mujoco model and data
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    # Initialize the aloha_mink_wrapper
    aloha_mink_wrapper = AlohaMinkWrapper(model, data)

    # Initialize to the neutral pose
    initialize_scene(data, model)

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
            sample_object_position(data, model)

            # Set the initial posture target
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)

            # Rate limiter for fixed update frequency
            rate = RateLimiter(frequency=200, warn=False)

            has_grasped = False
            gripper_closed = False
            object_lifted = False

            try:
                while viewer.is_running():
                    # Render 
                    renderer.update_scene(data, camera="wrist_cam_right")
                    img = renderer.render()

                    if not img_queue.full():
                        img_queue.put(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                    if not has_grasped:
                        # Align gripper with the object
                        move_to_object()

                        # Check if gripper has reached the object
                        if is_gripper_near_object():
                            print("Object reached. Closing gripper...")
                            has_grasped = True

                    elif not gripper_closed:
                        # Close the gripper to grasp the object
                        for i in range(50):
                            close_gripper()
                            mujoco.mj_step(model, data)
                        gripper_closed = True

                    elif not object_lifted:
                        # Lift the object after grasping
                        lift_object()

                        # Check if the object has been lifted
                        if check_object_lifted():
                            object_lifted = True
                            print("Task complete. Reinitializing scene...")

                            # Reinitialize the scene for the next task
                            initialize_scene(data, model)

                            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
                            sample_object_position(data, model)

                            # Reset flags for the next cycle
                            has_grasped = False
                            gripper_closed = False
                            object_lifted = False

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