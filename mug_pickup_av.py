"""
Collect data for mug picking with motion planning using bottleneck approach.
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
import cv2
import threading
import queue
import os
import copy
import h5py

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"
theta = 0
av_goal = None

def state_to_transform(state):
    """Convert state vector [x, y, z, roll, pitch, yaw] to transformation matrix."""
    t = state[:3]
    r = R.from_euler('xyz', state[3:], degrees=False).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T

def transform_to_state(T):
    """Convert transformation matrix to state vector [x, y, z, roll, pitch, yaw]."""
    t = T[:3, 3]
    r = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)
    return np.concatenate((t, r))

def transform_to_wxyz_xyz(T):
    """Convert transformation matrix to wxyz_xyz format."""
    t = T[:3, 3]
    r = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
    return np.concatenate((r, t))

def move_to_optimal_view(data, model, aloha_mink_wrapper, theta, dt=None):
    """Move the right arm to align with the object's optimal view."""
    global optimal_view
    global av_goal
    # Update task targets
    mink.move_mocap_to_frame(model, data, "right/target", "handle_site", "site")
    av_goal = mink.SE3.from_mocap_name(model, data, "right/target")
    
    quat = av_goal.wxyz_xyz[0:4]
    pos = av_goal.wxyz_xyz[4:]
    rotation_matrix = R.from_quat(quat).as_matrix()

    # Sample a point on a cylinder around the object with fixed radius and height
    y_axis = rotation_matrix[:, 1]
    y_axis_normalized = y_axis / np.linalg.norm(y_axis)
    x_axis = rotation_matrix[:, 0]  
    x_axis_normalized = x_axis / np.linalg.norm(x_axis)

    radius = 0.15
    radian = np.pi / 4 # specify the position of camera
    if theta > 0:
        pos = pos - radius * y_axis_normalized * np.cos(radian)
    else:
        pos = pos + radius * y_axis_normalized * np.cos(radian)
    pos = pos + radius * x_axis_normalized * np.sin(radian)
    height = 0.1
    pos[2] += height
    av_goal.wxyz_xyz[4:] = pos  

    # Randomize the rotation of the camera
    rotation_matrix = R.from_quat(quat).as_matrix()
    angle_1 = np.pi / 2     # the direction of camera on the horizontal plane, 0 means parallel to the mug handle; 
                            # pi/2 means perpendicular to the mug handle
    if theta > 0:
        theta_x = angle_1
    else:
        theta_x = -angle_1
    angle_2 = np.pi / 5 # the direction of camera on the vertical plane, 0 means parallel to the mug handle;
    theta_y = -angle_2
    euler = R.from_euler('xyz', [theta_x, theta_y, 0], degrees=False)
    R_view = euler.as_matrix()
    rotation_matrix = np.dot(rotation_matrix, R_view)
    quat = R.from_matrix(rotation_matrix).as_quat()
    av_goal.wxyz_xyz[0:4] = quat

    aloha_mink_wrapper.tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
    aloha_mink_wrapper.tasks[1].set_target(av_goal)
    
    optimal_view = av_goal.copy()
    # Solve inverse kinematics
    if dt is not None:
        aloha_mink_wrapper.solve_ik(dt)
    else:
        aloha_mink_wrapper.solve_ik(rate.dt)

    # Apply the calculated joint positions to actuators
    data.ctrl[aloha_mink_wrapper.actuator_ids] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids]

def is_gripper_near_optimal_view(data, model, vel_threshold=1.5, vel_threshold2=0.01):
    """Check if the gripper is close enough to the object's optimal view."""
    global optimal_view
    gripper_position = data.qpos[8:16]
    # distance = np.linalg.norm(gripper_position - optimal_view.wxyz_xyz[4:])
    qvel_raw = data.qvel.copy()
    right_qvel_raw = qvel_raw[8:16]
    sum_of_vel = np.sum(np.abs(right_qvel_raw))
    # print(sum_of_vel, distance)
    return sum_of_vel < vel_threshold and sum_of_vel > vel_threshold2 

def sample_object_position(data, model, x_range=(-0.075, 0.075), y_range=(-0.075, 0.075), yaw_range=(-np.pi / 4, np.pi / 4)):
    """Randomize the object's position in the scene for a free joint."""
    global theta
    # Get the free joint ID (first free joint in the system)
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    theta = np.random.uniform(*yaw_range)
    # print(theta)
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
    object_qpos = data.qpos[16:23].copy()
    # Forward propagate the simulation state
    mujoco.mj_forward(model, data)

    # # Log the new position for debugging
    # print(f"New object position: {data.xpos[object_body_id]}")
    return object_qpos, theta

def sample_constrained_pos_noise(max_offset=0.05):
    """
    Sample constrained random noise for a position.

    :param max_offset: Maximum absolute perturbation offset.
    :return: Noisy position.
    """
    # Generate a small random offset in the range [-max_offset, max_offset]
    offset = np.random.uniform(-max_offset, max_offset, size=3)

    return offset

def sample_constrained_quat_noise(angle_limit=20):
    """
    Sample constrained random noise for a quaternion.

    :param quat: Input quaternion (x, y, z, w).
    :param angle_limit: Maximum absolute perturbation angle in degrees.
    :return: Noisy quaternion (normalized).
    """
    # Generate a small random rotation in axis-angle representation
    random_axis = np.random.randn(3)  # Random direction
    random_axis /= np.linalg.norm(random_axis)  # Normalize to unit vector
    
    # Sample a random angle in the range [-angle_limit, angle_limit]
    random_angle = np.random.uniform(-np.radians(angle_limit), np.radians(angle_limit))

    # Convert to quaternion
    noise_quat = R.from_rotvec(random_angle * random_axis).as_quat()  # (x, y, z, w)

    return noise_quat

def compute_approach_pose(goal, offset_distance=0.1):
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
    # print(approach_vector)
    # Offset the position by the approach vector scaled by the distance
    approach_position = position - approach_vector * offset_distance
    
    # Update the copied pose with the new position
    approach_pose.wxyz_xyz[4:] = approach_position

    return approach_pose

def move_to_object(data, model, aloha_mink_wrapper, theta, pos_noise=None, quat_noise=None, stage2_reached=False, dt=None):
    """Move the gripper to align with the object's position."""

    mink.move_mocap_to_frame(model, data, "left/target", "handle_site", "site")

    # Update task targets
    goal = mink.SE3.from_mocap_name(model, data, "left/target")
    goal.wxyz_xyz[-1] -= 0.02
    quat = R.from_euler('xyz', [0, np.pi / 4, theta], degrees=False).as_quat()
    quat_scalar_first = np.roll(quat, shift=1)
    goal.wxyz_xyz[0:4] = quat_scalar_first 

    gripper_position = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")]
    pre_goal = compute_approach_pose(goal)
    if pos_noise is not None:
        pre_goal.wxyz_xyz[4:] += pos_noise
    if quat_noise is not None:
        xyzw = np.array([pre_goal.wxyz_xyz[1], pre_goal.wxyz_xyz[2], pre_goal.wxyz_xyz[3], pre_goal.wxyz_xyz[0]])
        quat = R.from_quat(xyzw)
        quat = quat * R.from_quat(quat_noise)
        pre_goal.wxyz_xyz[0:4] = np.array([quat.as_quat()[3], quat.as_quat()[0], quat.as_quat()[1], quat.as_quat()[2]])
    # print("noise", pos_noise, quat_noise)   
    qvel_raw = data.qvel.copy()
    left_qvel_raw = qvel_raw[:8]
    sum_of_vel = np.sum(np.abs(left_qvel_raw))
    # print(np.linalg.norm(gripper_position - pre_goal.wxyz_xyz[4:]), sum_of_vel)
    if np.linalg.norm(gripper_position - pre_goal.wxyz_xyz[4:]) < 0.21 and sum_of_vel < 2.0 and not stage2_reached: 
        stage2_reached = True
        # print(np.linalg.norm(gripper_position - pre_goal.wxyz_xyz[4:]))
        import time
        time.sleep(2)
        for i in range(100):
            mujoco.mj_step(model, data)

    if stage2_reached:
        # print("Stage 2")
        aloha_mink_wrapper.tasks[0].set_target(goal)
    else:
        # print("Stage 1")
        aloha_mink_wrapper.tasks[0].set_target(pre_goal)
    # aloha_mink_wrapper.tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
    
    # Solve inverse kinematics
    if dt is not None:
        aloha_mink_wrapper.solve_ik(dt)
    else:
        aloha_mink_wrapper.solve_ik(rate.dt)

    # Apply the calculated joint positions to actuators
    data.ctrl[aloha_mink_wrapper.actuator_ids[:6]] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids[:6]]
    return stage2_reached

def is_gripper_near_object(vel_threshold=2.0, dis_threshold=0.31):
    """Check if the gripper is close enough to the object."""
    object_position = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handle_site")]
    gripper_position = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")]

    distance = np.linalg.norm(gripper_position - object_position)
    qvel_raw = data.qvel.copy()
    left_qvel_raw = qvel_raw[:8]
    sum_of_vel = np.sum(np.abs(left_qvel_raw))
    
    # print(sum_of_vel, distance)
    return sum_of_vel < vel_threshold and distance < dis_threshold

def close_gripper():
    """Gradually close the gripper."""
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")] = 0

def lift_object(lift_height=0.15, ):
    """Lift the gripper by a specified height."""
    # Get current gripper position
    current_position = mink.SE3.from_mocap_name(model, data, "left/target")
    # current_rotation = np.copy(data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")]).reshape(3,3)
    # quat = R.from_matrix(current_rotation).as_quat()

    # Set the goal position slightly higher
    goal = current_position.copy()
    
    goal.wxyz_xyz[0:4] = [1, 0, 0, 0]
    # quat_scalar_first = np.roll(quat, shift=1)
    # goal.wxyz_xyz[0:4] = quat_scalar_first
    goal.wxyz_xyz[-1] += lift_height
    aloha_mink_wrapper.tasks[0].set_target(goal)

    # Solve inverse kinematics
    aloha_mink_wrapper.solve_ik(rate.dt)

    # Apply the calculated joint positions to actuators
    data.ctrl[aloha_mink_wrapper.actuator_ids[:6]] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids[:6]]

def check_object_lifted(data, model):
    """Check if the object has been lifted to the desired height."""
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_position = data.xpos[object_body_id]

    # Check if the object has reached or exceeded the target lift height
    return object_position[-1] >= 0.08

def initialize_scene(data, model, aloha_mink_wrapper):
    """Initialize the scene to reset the task."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    aloha_mink_wrapper.configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    aloha_mink_wrapper.initialize_mocap_targets()

def get_robot_data(data, model, renderer, aloha_mink_wrapper, camera_keys=["overhead_cam"]):
    """Get the current robot observation and action."""
    action = np.zeros(7) # 6 DoF + gripper
    state = np.zeros(7) # 6 DoF + gripper
    action[:6] = data.ctrl[aloha_mink_wrapper.actuator_ids][:6].copy()
    action[6] = data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")].copy()

    state[:6] = data.qpos[aloha_mink_wrapper.dof_ids][:6].copy()
    state[6] = data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")].copy()
    # Render camera images
    imgs = []
    for key in camera_keys:
        renderer.update_scene(data, camera=key)
        img = renderer.render().copy()
        imgs.append(img)
    return action, state, imgs

def get_robot_data_with_vision_qpos(data, model, renderer, aloha_mink_wrapper, camera_keys=["overhead_cam"]):
    """Get the current robot observation and action."""
    action = np.zeros(7) # 6 DoF + gripper
    state = np.zeros(13) # 6 DoF + gripper + 6 DoF vision
    action[:6] = data.ctrl[aloha_mink_wrapper.actuator_ids][:6].copy()
    action[6] = data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")].copy()

    state[:6] = data.qpos[aloha_mink_wrapper.dof_ids][:6].copy()
    state[6] = data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")].copy()
    state[7:13] = data.qpos[aloha_mink_wrapper.dof_ids][6:].copy()
    # Render camera images
    imgs = []
    for key in camera_keys:
        renderer.update_scene(data, camera=key)
        img = renderer.render().copy()
        imgs.append(img)
    return action, state, imgs

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
    initialize_scene(data, model, aloha_mink_wrapper)

    renderer = mujoco.Renderer(model, 360, 640)
    
    # Create a thread-safe queue and running event
    img_queue = queue.Queue(maxsize=1)
    running_event = threading.Event()
    running_event.set()

    # Start the display thread
    display_thread = threading.Thread(target=display_image, args=(img_queue, running_event))
    display_thread.start()

    # Set the random seed for reproducibility
    np.random.seed(43)

    camera_keys = ["wrist_cam_left", "wrist_cam_right"]

    try:
        # Launch the viewer
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Sample object poses
            object_qpos, theta = sample_object_position(data, model)

            pos_noise = sample_constrained_pos_noise()
            quat_noise = sample_constrained_quat_noise()

            # Set the initial posture target
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)

            # Rate limiter for fixed update frequency
            rate = RateLimiter(frequency=100, warn=False)

            has_grasped = False
            gripper_closed = False
            object_lifted = False
            near_optimal_view = False

            success_cnt = 0
            fail_cnt = 0
            record_frequency = 2

            cam_images = {}
            for key in camera_keys:
                cam_images[key] = []
            actions = []
            states = []

            episode_cnt = 0
            step_cnt = 0
            t = 0
            close_gripper_cnt = 0

            av_steps = 0

            stage2_reached = False
            last_ee_pose = None
            last_gripper_state = None
            current_ee_pose = None
            last_imgs = None

            # Create dataset folder with current date and time
            import datetime
            now = datetime.datetime.now()
            folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(f"datasets/{folder_name}", exist_ok=True)

            try:
                while viewer.is_running():
                    if not near_optimal_view:
                        # Move the robot to the optimal view of the object
                        move_to_optimal_view(data, model, aloha_mink_wrapper, theta)
                        av_steps += 1

                        # Check if the gripper is near the object's optimal view
                        if is_gripper_near_optimal_view(data, model) and av_steps > 50:
                            print("Optimal view reached. Moving to object...")
                            for i in range(100):
                                mujoco.mj_step(model, data)
                            near_optimal_view = True
                            av_steps = 0

                    elif not has_grasped:
                        # Align gripper with the object
                        stage2_reached = move_to_object(data, model, aloha_mink_wrapper, theta, pos_noise=pos_noise, quat_noise=quat_noise, stage2_reached=stage2_reached)

                        # Check if gripper has reached the object
                        if is_gripper_near_object():
                            print("Object reached. Closing gripper...")
                            has_grasped = True

                    elif not gripper_closed:
                        # Close the gripper to grasp the object
                        close_gripper()
                        close_gripper_cnt += 1
                        if close_gripper_cnt > 50:
                            gripper_closed = True
                            close_gripper_cnt = 0

                    elif not object_lifted:
                        # Lift the object after grasping
                        lift_object()

                        # Check if the object has been lifted
                        if check_object_lifted(data, model):
                            object_lifted = True
                            # Save the episode
                            with h5py.File(f"datasets/{folder_name}/episode_{episode_cnt}.h5", "w") as f:
                                for key in camera_keys:
                                    f.create_dataset(f"/observations/images/{key}", data=np.array(cam_images[key]))
                                f.create_dataset("/action", data=np.array(actions))
                                f.create_dataset("/observations/qpos", data=np.array(states))
                                f.create_dataset("/object_qpos", data=object_qpos)
                                f.create_dataset("/theta", data=np.array([theta]))
                            episode_cnt += 1
                            
                            cam_images = {}
                            for key in camera_keys:
                                cam_images[key] = []
                            actions = []
                            states = []
                            step_cnt = 0
                            t = 0

                            print("Task complete. Reinitializing scene...")

                            # Reinitialize the scene for the next task
                            initialize_scene(data, model, aloha_mink_wrapper)

                            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
                            object_qpos, theta = sample_object_position(data, model)
                            pos_noise = sample_constrained_pos_noise()
                            quat_noise = sample_constrained_quat_noise()

                            # Reset flags for the next cycle
                            has_grasped = False
                            gripper_closed = False
                            object_lifted = False
                            stage2_reached = False
                            has_grasped = False
                            near_optimal_view = False
                            last_ee_pose = None
                            last_gripper_state = None
                            current_ee_pose = None
                            last_imgs = None

                            success_cnt += 1
                            if success_cnt >= 50:
                                break

                    # Compensate gravity
                    aloha_mink_wrapper.compensate_gravity([model.body("left/base_link").id, model.body("right/base_link").id])

                    action, state, imgs = get_robot_data(data, model, renderer, aloha_mink_wrapper, camera_keys=camera_keys)
                    
                    # Only record data if the vision manipulator is in the optimal view
                    if stage2_reached and last_ee_pose is not None:
                        step_cnt += 1
                        tf = aloha_mink_wrapper.transform_left_to_right(data)
                        current_ee_pose = transform_to_state(tf)
                        current_gripper_state = np.array([state[-1]])
                        if step_cnt % record_frequency == 0:
                            action = np.concatenate([current_ee_pose, current_gripper_state]) # - np.concatenate([last_ee_pose, last_gripper_state])
                            state = np.concatenate([last_ee_pose, last_gripper_state])
                            last_ee_pose = current_ee_pose
                            last_gripper_state = current_gripper_state
                            last_imgs = imgs
                            actions.append(action)
                            states.append(state)
                            for key, img in zip(camera_keys, last_imgs):
                                cam_images[key].append(img)
                                if step_cnt == 0:
                                    cv2.imwrite(f"camera_frames/episode_{episode_cnt}_{key}.png", img)
                        
                    elif stage2_reached and last_ee_pose is None:
                        tf = aloha_mink_wrapper.transform_left_to_right(data)
                        last_ee_pose = transform_to_state(tf)
                        last_gripper_state = np.array([state[-1]])
                        last_imgs = imgs

                    if not img_queue.full():
                        img_queue.put(cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))
                    if step_cnt > 700:
                        step_cnt = 0
                        # Episode timeout, reset the scene
                        cam_images = {}
                        for key in camera_keys:
                            cam_images[key] = []
                        actions = []
                        states = []
                        t = 0

                        print("Task complete. Reinitializing scene...")

                        # Reinitialize the scene for the next task
                        initialize_scene(data, model, aloha_mink_wrapper)

                        aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
                        object_qpos, theta = sample_object_position(data, model)
                        pos_noise = sample_constrained_pos_noise()
                        quat_noise = sample_constrained_quat_noise()

                        # Reset flags for the next cycle
                        has_grasped = False
                        gripper_closed = False
                        object_lifted = False
                        stage2_reached = False
                        near_optimal_view = False
                        last_ee_pose = None
                        last_gripper_state = None
                        current_ee_pose = None
                        last_imgs = None

                        fail_cnt += 1

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
    
    print(f"Success count: {success_cnt}/{success_cnt + fail_cnt}")
    print(f"Failure count: {fail_cnt}/{success_cnt + fail_cnt}")