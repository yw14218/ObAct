#!/usr/bin/env python3

from threading import Thread
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from pymoveit2 import MoveIt2
import viper_robot as robot

class InverseKinematicsSolver:
    def __init__(self):
        # rclpy.init()
        self.node = Node("Moveit2_Viper")
        
        # Declare parameters for position and orientation
        self.node.declare_parameter("synchronous", True)
        
        # Create callback group to allow parallel execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Create MoveIt2 interface
        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )
        
        self.moveit2.max_velocity = 0.2
        self.moveit2.max_acceleration = 0.2

        self.moveit2.add_collision_box(
            id="table",
            pose=(0, 0, 0, 0, 0, 0),
            position=(0, 0, 0),
            quat_xyzw=(0, 0, 0, 1),
            size=(1, 1, 0.05),
        )
        # Spin the node in a background thread
        self.executor = rclpy.executors.MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        
        self.node.create_rate(1.0).sleep()
    
    def compute_ik(self, position, quat_xyzw):
        # Get parameter
        synchronous = self.node.get_parameter("synchronous").get_parameter_value().bool_value
        
        # self.node.get_logger().info(
        #     f"Computing IK for {{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}}}"
        # )
        
        retval = None
        if synchronous:
            retval = self.moveit2.compute_ik(position, quat_xyzw, wait_for_server_timeout_sec=0.1)
        else:
            future = self.moveit2.compute_ik_async(position, quat_xyzw, wait_for_server_timeout_sec=0.1)
            if future is not None:
                rate = self.node.create_rate(10)
                while not future.done():
                    rate.sleep()
                retval = self.moveit2.get_compute_ik_result(future)
        
        if retval is None:
            print("Computing IK Failed.")
        else:
            # print("Succeeded. Result: " + str(retval))
            return retval
    
    
    def move_to_pose(self, position, quat_xyzw):
        self.moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
        )

    def shutdown(self):
        rclpy.shutdown()
        self.executor_thread.join()

# example
if __name__ == "__main__":
    ik_solver = InverseKinematicsSolver()
    position = [0.3, 0.0, 0.2]
    quat_xyzw = [1.0, 0.0, 0.0, 0.0]
    ik_solver.compute_ik(position, quat_xyzw)
    ik_solver.publish_commands(position, quat_xyzw)
    ik_solver.shutdown()
