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
import socket
import time
from threading import Event
from robot_control.filter import OneEuroFilter

ready_to_receive = Event()
ready_to_execute = Event()
ready_to_receive.set()
base_pos = None
base_rot = None

class UDPServer:
    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self._ip, self._port))
        self._socket.setblocking(False)
        self._running = False
        # queue for storing received data
        self.cmd_buffer = queue.Queue(maxsize=1)
        self.latest_cmd = None
        self.reset = False
        self.gripper_status = 1
        print(f"UDP server started at {self._ip}:{self._port}")
    
    def start(self):
        self._running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.start()
    
    def stop(self):
        self._running = False
        # kill the thread
        self.thread.join()
        self._socket.close()
    
    def _listen(self):
        cnt = 0
        while self._running:
            received_data = None
            try:
                # print("Waiting for event...")
                ready_to_receive.wait()
                while True:
                    # print("Waiting for data...")
                    data, addr = self._socket.recvfrom(2048)
                    # received_data = np.frombuffer(data, dtype=np.float32)
                    try:
                        data_segments = data.decode().split("_")
                        data_segments = data_segments[1:7]
                        received_data = np.array([float(x) for x in data_segments])
                        received_data[0], received_data[1], received_data[2] = received_data[2], -received_data[0], received_data[1]
                        cnt += 1
                    except:
                        data = data.decode()
                        print(f"Received data: {data}")
                        if data == "B":
                            self.reset = True
                        if data == 'A':
                            self.gripper_status = 1 - self.gripper_status
                        received_data = None
                    print(f"Received data: {received_data}")
                    # self.cmd_buffer.put(received_data)
            except BlockingIOError:
                # print("No data received")
                pass
            if received_data is not None:
                self.latest_cmd = received_data
                # print(received_data)
                ready_to_receive.clear()
                ready_to_execute.set()
            else:
                # print("No data received")
                pass


_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"
theta = 0

def sample_object_position(data, model, x_range=(-0.075, 0.075), y_range=(-0.075, 0.075), yaw_range=(-np.pi / 2, np.pi / 2)):
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

def close_gripper():
    """Gradually close the gripper."""
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")] = 0

def move_arm(goal_pos, goal_rot, gripper_status):
    """Move the arm to a specified position."""
    # Get the current position of the arm
    left_gripper_pose = mink.SE3.from_mocap_name(model, data, "left/target")
    right_gripper_pose = mink.SE3.from_mocap_name(model, data, "right/target")
    
    # Set the goal position
    left_goal = left_gripper_pose.copy()
    left_goal.wxyz_xyz[4:] = goal_pos
    left_goal.wxyz_xyz[:4] = R.from_euler("xyz", goal_rot).as_quat()[[3, 0, 1, 2]]
    # print(left_goal.wxyz_xyz[:4])
    aloha_mink_wrapper.tasks[0].set_target(left_goal)

    right_goal = right_gripper_pose.copy()
    aloha_mink_wrapper.tasks[1].set_target(right_goal)

    # Solve inverse kinematics
    aloha_mink_wrapper.solve_ik(rate.dt)

    # Apply the calculated joint positions to actuators
    data.ctrl[aloha_mink_wrapper.actuator_ids] = aloha_mink_wrapper.configuration.q[aloha_mink_wrapper.dof_ids]
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")] = 1 - gripper_status

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
    # start a udp server
    udp_server = UDPServer(ip="172.20.10.4", port=8888)
    udp_server.start()

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
            model=model, data=data, show_left_ui=False, show_right_ui=True
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            opt = viewer.opt

            # Disable expensive rendering features
            opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = False  # Disable lighting
            opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False  # Disable shadows
            opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False  # Disable reflections
            opt.flags[mujoco.mjtRndFlag.mjRND_FOG] = False  # Disable fog

            # Sample object poses
            sample_object_position(data, model)

            # Set the initial posture target
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)

            # Rate limiter for fixed update frequency
            rate = RateLimiter(frequency=25, warn=False)

            # Filter for smoothing the gripper commands
            filter = OneEuroFilter(min_cutoff=0.01, beta=10.0)

            pre_grasped = False
            has_grasped = False
            gripper_closed = False
            object_lifted = False

            current_gripper_status = 0
            initial_gripper_pose = mink.SE3.from_mocap_name(model, data, "left/target").copy()
            goal_pos = initial_gripper_pose.wxyz_xyz[4:].copy()
            goal_rot = R.from_quat(initial_gripper_pose.wxyz_xyz[:4][[1, 2, 3, 0]]).as_euler("xyz").copy()
            try:
                while viewer.is_running():
                    start = time.time()
                    # Render 
                    renderer.update_scene(data, camera="wrist_cam_right")
                    img = renderer.render()

                    if not img_queue.full():
                        img_queue.put(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    if check_object_lifted():
                        print("Task complete. Reinitializing scene...")

                        # Reinitialize the scene for the next task
                        initialize_scene(data, model)

                        aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
                        sample_object_position(data, model)

                        initial_gripper_pose = mink.SE3.from_mocap_name(model, data, "left/target").copy()
                        goal_pos = initial_gripper_pose.wxyz_xyz[4:].copy()
                        current_gripper_status = 0
                    
                    else:
                        if ready_to_execute.is_set():
                            if udp_server.latest_cmd is not None:
                                # cmd = udp_server.cmd_buffer.get()
                                cmd = udp_server.latest_cmd
                                goal_pos = cmd[:3] * 1.5
                                # goal_pos = np.zeros(3)
                                
                                gripper_rot = np.array([-cmd[5], cmd[3], -cmd[4]]) / 180 * np.pi
                                goal_rot = gripper_rot
                                # goal_rot = np.array([0, 0, 0])
                                t = time.time()
                                goal_pos = filter(t, goal_pos)
                                if base_pos is not None and base_rot is not None and not udp_server.reset:
                                    goal_pos = goal_pos - base_pos + initial_gripper_pose.wxyz_xyz[4:]
                                    goal_rot = goal_rot - base_rot + R.from_quat(initial_gripper_pose.wxyz_xyz[:4][[1, 2, 3, 0]]).as_euler("xyz")
                                else:
                                    base_pos = goal_pos
                                    base_rot = goal_rot
                                    udp_server.reset = False
                                    udp_server.gripper_status = 0
                                    goal_pos = goal_pos - base_pos + initial_gripper_pose.wxyz_xyz[4:]
                                    goal_rot = goal_rot - base_rot + R.from_quat(initial_gripper_pose.wxyz_xyz[:4][[1, 2, 3, 0]]).as_euler("xyz")
                                gripper_status = udp_server.gripper_status
                                current_gripper_status = gripper_status
                                np.set_printoptions(precision=2, suppress=True)
                                # print(f"Goal position: {goal_pos}")
                                print(f"Goal rotation: {goal_rot}")
                                move_arm(goal_pos, goal_rot, gripper_status)
                                udp_server.latest_cmd = None
                                ready_to_receive.set()
                                ready_to_execute.clear()
                        else:
                            t = time.time()
                            goal_pos = filter(t, goal_pos)
                            move_arm(goal_pos, goal_rot, current_gripper_status)
                        
                    # Compensate gravity
                    aloha_mink_wrapper.compensate_gravity([model.body("left/base_link").id, model.body("right/base_link").id])

                    # Step the simulation
                    mujoco.mj_step(model, data)

                    # Visualize at fixed FPS
                    viewer.sync()
                    rate.sleep()
                    end = time.time()
                    # print(f"Time taken: {end - start}")

            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Exiting gracefully...")
                udp_server.stop()
                # join the display thread
                running_event.clear()
                img_queue.put(None)
                display_thread.join()
                cv2.destroyAllWindows()

    except Exception as e:
        # Cleanup
        udp_server.stop()
        running_event.clear()
        img_queue.put(None)  # Signal thread to exit
        display_thread.join()
        cv2.destroyAllWindows()