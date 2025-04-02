#!/usr/bin/env python

import math
import numpy as np

# from geometry_msgs.msg import Pose
# from tf.transformations import quaternion_slerp
from scipy.spatial.transform import Rotation as R, Slerp

class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

class Pose:
    def __init__(self):
        self.position = Point()
        self.orientation = Quaternion()

def quaternion_slerp(start_ori, end_ori, t):
    """
    Performs spherical linear interpolation (SLERP) between two quaternions.

    :param start_ori: list or array-like, quaternion [x, y, z, w]
    :param end_ori: list or array-like, quaternion [x, y, z, w]
    :param t: float in [0, 1], interpolation parameter
    :return: interpolated quaternion [x, y, z, w]
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("Interpolation parameter t must be between 0 and 1.")

    times = [0, 1]
    rotations = R.from_quat([start_ori, end_ori])
    slerp = Slerp(times, rotations)
    interpolated = slerp(t)
    return interpolated.as_quat()

def interpolate_cartesian_pose(current_pose, desired_pose, max_step):
    """
    Interpolate a straight-line path in Cartesian space between a current pose and desired pose.
    
    :param current_pose:  The starting pose (geometry_msgs/Pose).
    :param desired_pose:  The goal pose (geometry_msgs/Pose).
    :param max_step:      The maximum step (in meters) between consecutive waypoints.
    :return: list of intermediate waypoints (Pose), including the start and final pose.
             Returns an empty list if interpolation fails or inputs are invalid.
    """
    # 1. Validate input
    if max_step <= 0.0:
        # We require a positive step size
        return []

    # 2. Extract positions as numpy arrays
    start_pos = np.array([current_pose.position.x,
                          current_pose.position.y,
                          current_pose.position.z])
    end_pos   = np.array([desired_pose.position.x,
                          desired_pose.position.y,
                          desired_pose.position.z])

    # 3. Extract orientations as [x, y, z, w]
    start_ori = np.array([current_pose.orientation.x,
                          current_pose.orientation.y,
                          current_pose.orientation.z,
                          current_pose.orientation.w], dtype=np.float64)
    end_ori   = np.array([desired_pose.orientation.x,
                          desired_pose.orientation.y,
                          desired_pose.orientation.z,
                          desired_pose.orientation.w], dtype=np.float64)

    # Normalize just in case
    start_norm = np.linalg.norm(start_ori)
    end_norm   = np.linalg.norm(end_ori)
    if start_norm < 1e-12 or end_norm < 1e-12:
        # Invalid orientation (zero length)
        return []

    start_ori /= start_norm
    end_ori   /= end_norm

    # 4. Compute the distance in position space
    distance = np.linalg.norm(end_pos - start_pos)

    # If distance is very small, check orientation difference
    if distance < 1e-6:
        waypoints = []
        # Always push back current_pose
        waypoints.append(_copy_pose(current_pose))
        # Check if there's a meaningful orientation difference
        angle = _quaternion_angle_shortest_path(start_ori, end_ori)
        if angle > 1e-6:
            waypoints.append(_copy_pose(desired_pose))
        return waypoints

    # 5. Determine how many steps needed based on max_step
    #    e.g., distance=1.0 and max_step=0.05 => ~20 steps
    num_steps = int(math.ceil(distance / max_step))

    # 6. Determine how many steps we need for orientation
    #    e.g., limit orientation change to 10 degrees per step => (10 * pi/180)
    angle = _quaternion_angle_shortest_path(start_ori, end_ori)
    max_orient_step = 10.0 * math.pi / 180.0  # 10 degrees in radians
    orient_steps = 0
    if angle > 1e-6:
        orient_steps = int(math.ceil(angle / max_orient_step))

    # Overall steps is the max
    steps = max(num_steps, orient_steps, 1)

    # 7. Generate intermediate waypoints
    waypoints = []
    for i in range(steps + 1):
        t = float(i) / float(steps)

        # Linear interpolation for position
        interp_pos = start_pos + t * (end_pos - start_pos)

        # Spherical linear interpolation for orientation
        interp_ori = quaternion_slerp(start_ori, end_ori, t)

        # Build a Pose
        pose_msg = Pose()
        pose_msg.position.x = interp_pos[0]
        pose_msg.position.y = interp_pos[1]
        pose_msg.position.z = interp_pos[2]
        pose_msg.orientation.x = interp_ori[0]
        pose_msg.orientation.y = interp_ori[1]
        pose_msg.orientation.z = interp_ori[2]
        pose_msg.orientation.w = interp_ori[3]

        waypoints.append(pose_msg)

    return waypoints


def _quaternion_angle_shortest_path(q1, q2):
    """
    Returns the angle (in radians) between two quaternions along the shortest path.
    Both quaternions should be normalized.
    """
    # Dot product
    dot = np.dot(q1, q2)
    # Numerical stability; dot should be in [-1, 1]
    dot = max(min(dot, 1.0), -1.0)
    # Angle is 2 * acos(|dot|)
    return 2.0 * math.acos(abs(dot))


def _copy_pose(pose_in):
    """
    Helper to clone a geometry_msgs/Pose.
    """
    p = Pose()
    p.position.x = pose_in.position.x
    p.position.y = pose_in.position.y
    p.position.z = pose_in.position.z
    p.orientation.x = pose_in.orientation.x
    p.orientation.y = pose_in.orientation.y
    p.orientation.z = pose_in.orientation.z
    p.orientation.w = pose_in.orientation.w
    return p

if __name__ == "__main__":
    # Example usage
    current_pose = Pose()
    current_pose.position.x = 0.0
    current_pose.position.y = 0.0
    current_pose.position.z = 0.0
    # initialize an euler angle
    euler = R.from_euler("xyz", [0, 0, 0])
    # convert to quaternion
    quat = euler.as_quat()
    # assign to orientation
    current_pose.orientation.x = quat[0]
    current_pose.orientation.y = quat[1]
    current_pose.orientation.z = quat[2]
    current_pose.orientation.w = quat[3]

    desired_pose = Pose()
    desired_pose.position.x = 0.5
    desired_pose.position.y = 0.5
    desired_pose.position.z = 0.5
    # initialize an euler angle
    euler = R.from_euler("xyz", [1.22, 1.42, 1.68])
    # convert to quaternion
    quat = euler.as_quat()
    # assign to orientation
    desired_pose.orientation.x = quat[0]
    desired_pose.orientation.y = quat[1]
    desired_pose.orientation.z = quat[2]
    desired_pose.orientation.w = quat[3]

    max_step = 0.01
    waypoints = interpolate_cartesian_pose(current_pose, desired_pose, max_step)
    for i, wp in enumerate(waypoints):
        print(f"Waypoint {i}:")
        print(f"  Position: ({wp.position.x}, {wp.position.y}, {wp.position.z})")
        print(f"  Orientation: ({wp.orientation.x}, {wp.orientation.y}, {wp.orientation.z}, {wp.orientation.w})")
        # transform quaternion to euler
        euler = R.from_quat([wp.orientation.x, wp.orientation.y, wp.orientation.z, wp.orientation.w]).as_euler("xyz")
        print(f"  Euler: ({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f})")
        print()
    print(f"Generated {len(waypoints)} waypoints.")
    print("Done.")