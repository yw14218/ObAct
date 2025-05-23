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


_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"
theta = 0
optimal_view = None

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

def move_to_object():
    """Move the left arm to align with the object's position."""
    global theta
    global optimal_view
    mink.move_mocap_to_frame(model, data, "left/target", "handle_site", "site")

    # Update task targets
    goal = mink.SE3.from_mocap_name(model, data, "left/target")
    goal.wxyz_xyz[-1] -= 0.02
    quat = R.from_euler('xyz', [0, np.pi / 4, theta], degrees=False).as_quat()
    quat_scalar_first = np.roll(quat, shift=1)
    goal.wxyz_xyz[0:4] = quat_scalar_first 

    aloha_mink_wrapper.tasks[0].set_target(goal)
    # aloha_mink_wrapper.tasks[1].set_target(optimal_view)
    aloha_mink_wrapper.tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

    # Solve inverse kinematics
    aloha_mink_wrapper.solve_ik(rate.dt)

    # Apply the calculated joint positions to actuators
    data.ctrl[aloha_mink_wrapper.actuator_ids] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids]

def move_to_optimal_view():
    """Move the right arm to align with the object's optimal view."""
    global optimal_view
    # Update task targets
    goal = mink.SE3.from_mocap_name(model, data, "right/target")
    goal.wxyz_xyz[-1] = 0.3
    goal.wxyz_xyz[-2] = 0.3
    goal.wxyz_xyz[-3] = -0.07
    quat = R.from_euler('xyz', [0, np.pi / 6, -np.pi / 2], degrees=False).as_quat()
    quat_scalar_first = np.roll(quat, shift=1)
    goal.wxyz_xyz[0:4] = quat_scalar_first 
    goal.wxyz_xyz[-2] *= 1
    aloha_mink_wrapper.tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
    aloha_mink_wrapper.tasks[1].set_target(goal)
    
    optimal_view = goal
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
    
    print(sum_of_vel, distance)
    return sum_of_vel < vel_threshold and distance < dis_threshold

def is_gripper_at_optimal_view(vel_threshold=0.1, dis_threshold=0.5):
    """Check if the gripper is close enough to the object."""
    object_position = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handle_site")]
    gripper_position = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right/gripper")]

    distance = np.linalg.norm(gripper_position - object_position)
    qvel_raw = data.qvel.copy()
    left_qvel_raw = qvel_raw[12:19]
    sum_of_vel = np.sum(np.abs(left_qvel_raw))
    
    print(sum_of_vel, distance)
    if sum_of_vel == 0:
        return False
    
    # return sum_of_vel < vel_threshold and distance < dis_threshold
    return False

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

def save_data(rgb, depth, mask, extrinsic, save_dir="tasks/mug"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    existing_indices = [int(p.stem.split('_')[-1]) for p in save_dir.glob("rgb_*.png")]
    idx = 0

    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for name, data in [("rgb", rgb), ("mask", mask * 255), ("depth", depth), ("pose", extrinsic)]:
        filename = save_dir / f"{name}_{idx}.{'png' if name in ['rgb', 'mask'] else 'npy'}"
        (cv2.imwrite(str(filename), data) if name in ["rgb", "mask"] else np.save(filename, data))
    

if __name__ == "__main__":
    # Load the Mujoco model and data
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    # Initialize the aloha_mink_wrapper
    aloha_mink_wrapper = AlohaMinkWrapper(model, data)

    # Initialize to the neutral pose
    initialize_scene(data, model)

    renderer = mujoco.Renderer(model, 360, 640)
    
    # Create a thread-safe queue and running event
    img_queue = queue.Queue(maxsize=1)
    running_event = threading.Event()
    running_event.set()

    # Start the display thread
    display_thread = threading.Thread(target=display_image, args=(img_queue, running_event))
    display_thread.start()

    np.random.seed(2)

    try:
        # Launch the viewer
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Sample object poses
            # sample_object_position(data, model)

            # Set the initial posture target
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)

            # Rate limiter for fixed update frequency
            rate = RateLimiter(frequency=100, warn=False)

            has_moved_to_optimal_view = False
            has_grasped = False
            gripper_closed = False
            object_lifted = False

            try:
                while viewer.is_running():
                    # Render 
                    renderer.update_scene(data, camera="wrist_cam_right")
                    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
                    rgb, depth, mask, extrinsic = render(data, renderer, object_body_id)
                    save_data(rgb, depth, mask, extrinsic)
                
                    if not has_moved_to_optimal_view:
                        move_to_optimal_view()

                        # Check if gripper has reached the optimal view
                        if is_gripper_at_optimal_view():
                            print("optimal view reached.")
                            has_moved_to_optimal_view = True

                    elif not has_grasped:
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