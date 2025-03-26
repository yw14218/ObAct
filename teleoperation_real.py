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
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from robot_control.ik_solver import InverseKinematicsSolver

ready_to_receive = Event()
ready_to_execute = Event()
ready_to_receive.set()
base_pos = None
base_rot = None
GRIPPER_CLOSED = False

ik_solver = InverseKinematicsSolver()

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
                    # print(f"Received data: {received_data}")
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

def move_arm(bot, goal_pos, goal_rot, gripper_status):
    """Move the arm to a specified position."""
    global GRIPPER_CLOSED

    goal_quat = R.from_euler("xyz", goal_rot).as_quat()
    positions = ik_solver.compute_ik(goal_pos, goal_quat)

    if positions is not None:
        bot.arm._publish_commands(positions=positions.position[:6], moving_time=0.05, accel_time=0.01, blocking=False)

    if gripper_status > 0.5 and GRIPPER_CLOSED == False:
        bot.gripper.grasp()
        GRIPPER_CLOSED = True
    elif gripper_status <= 0.5 and GRIPPER_CLOSED == True:
        bot.gripper.release()
        GRIPPER_CLOSED = False
    else:
        pass

    # Move the arm using previous guess if available
    solution, success = bot.arm.set_ee_pose_components(
        *goal_pos, *goal_rot, blocking=False, moving_time=0.05, accel_time=0.01,
        custom_guess=previous_guess if previous_guess is not None else None
    )

    print(goal_pos, gripper_status)
    # Update previous guess only if IK succeeded
    if success and solution is not None:
        previous_guess = solution

    # Control the gripper
    if gripper_status > 0.1:
        bot.gripper.grasp()
    else:
        bot.gripper.release()

if __name__ == "__main__":
    # setup robot
    bot = InterbotixManipulatorXS(
        robot_model='vx300s',
        group_name='arm',
        gripper_name='gripper',
        moving_time=2,
        accel_time=0.3
    )

    robot_startup()
    # start a udp server
    udp_server = UDPServer(ip="10.132.32.4", port=8888)
    udp_server.start()

    bot.arm.set_joint_positions([0, -0.96, 1.16, 0, -0.3, 0], moving_time=2, accel_time=0.3)

    bot.arm.moving_time=0.2

    # Filter for smoothing the gripper commands
    filter = OneEuroFilter(min_cutoff=0.01, beta=10.0)

    pre_grasped = False
    has_grasped = False
    gripper_closed = False
    object_lifted = False

    current_gripper_status = 0
    initial_gripper_pose = bot.arm.get_ee_pose().copy()
    goal_pos = initial_gripper_pose[:3,3].copy()
    goal_rot = R.from_matrix(initial_gripper_pose[:3,:3]).as_euler("xyz").copy()
    while True:
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
                    goal_pos = goal_pos - base_pos + initial_gripper_pose[:3,3].copy()
                    goal_rot = goal_rot - base_rot + R.from_matrix(initial_gripper_pose[:3,:3]).as_euler("xyz").copy()
                else:
                    base_pos = goal_pos
                    base_rot = goal_rot
                    udp_server.reset = False
                    udp_server.gripper_status = 0
                    goal_pos = goal_pos - base_pos + initial_gripper_pose[:3,3].copy()
                    goal_rot = goal_rot - base_rot + R.from_matrix(initial_gripper_pose[:3,:3]).as_euler("xyz").copy()
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
