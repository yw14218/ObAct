from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from threading import Event
from filter import OneEuroFilter
import socket
import queue
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from ik_solver import InverseKinematicsSolver

ready_to_receive = Event()
ready_to_execute = Event()
ready_to_receive.set()
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
                    received_data = np.frombuffer(data, dtype=np.float32)
                    cnt += 1
                    # print(f"Received data: {received_data} from {addr}")
                    pass
                    # self.cmd_buffer.put(received_data)
            except BlockingIOError:
                # print("No data received")
                pass
            if received_data is not None:
                self.latest_cmd = received_data
                print(received_data)
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

    # # Move the arm using previous guess if available
    # solution, success = bot.arm.set_ee_pose_components(
    #     *goal_pos, *goal_rot, blocking=False, moving_time=0.05, accel_time=0.01,
    #     custom_guess=previous_guess if previous_guess is not None else None
    # )

    # print(goal_pos, gripper_status)
    # # Update previous guess only if IK succeeded
    # if success and solution is not None:
    #     previous_guess = solution

    # # Control the gripper
    # if gripper_status > 0.1:
    #     bot.gripper.grasp()
    # else:
    #     bot.gripper.release()


def main():
    bot = InterbotixManipulatorXS(
        robot_model='vx300s',
        group_name='arm',
        gripper_name='gripper',
        moving_time=2,
        accel_time=0.3
    )

    robot_startup()

    # bot_1.arm.go_to_sleep_pose(moving_time=2, accel_time=0.3)
    # raise

    # # start a udp server
    udp_server = UDPServer(ip="127.0.0.1", port=8006)
    udp_server.start()

    bot.arm.set_joint_positions([0, -0.96, 1.16, 0, -0.3, 0], moving_time=2, accel_time=0.3)

    bot.arm.moving_time=0.2

    # Filter for smoothing the gripper commands
    filter = OneEuroFilter(min_cutoff=0.01, beta=10.0)
    current_gripper_status = 0
    initial_gripper_pose = bot.arm.get_ee_pose().copy()
    goal_pos = initial_gripper_pose[:3,3].copy()
    goal_rot = R.from_matrix(initial_gripper_pose[:3,:3]).as_euler("xyz").copy()

    while True:
        if ready_to_execute.is_set():
            if udp_server.latest_cmd is not None:
                # cmd = udp_server.cmd_buffer.get()
                cmd = udp_server.latest_cmd
                goal_pos = cmd[:3] + initial_gripper_pose[:3,3].copy()
                gripper_status = cmd[3]
                current_gripper_status = min(2 * gripper_status, 1)
                gripper_rot = (cmd[4]) / 180 * np.pi
                goal_rot = np.array([0, gripper_rot, 0]) + R.from_matrix(initial_gripper_pose[:3,:3]).as_euler("xyz").copy()
                t = time.time()
                goal_pos = filter(t, goal_pos)
                move_arm(bot, goal_pos, goal_rot, current_gripper_status)
                udp_server.latest_cmd = None
                ready_to_receive.set()
                ready_to_execute.clear()
        else:
            t = time.time()
            goal_pos = filter(t, goal_pos)
            move_arm(bot, goal_pos, goal_rot, current_gripper_status)
  
    robot_shutdown()

if __name__ == '__main__':
    main()
    robot_shutdown()