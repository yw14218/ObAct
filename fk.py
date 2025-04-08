import modern_robotics as mr
import numpy as np
import h5py

# Load hand-eye calibration matrix
hand_eye = np.load("d405_handeye_right.npy")

# Define the robot parameters using a dictionary
class vx300s:
    Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
                      [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
                      [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
                      [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0]]).T

    M = np.array([[1.0, 0.0, 0.0, 0.536494],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.42705],
                  [0.0, 0.0, 0.0, 1.0]])
# get all h5 file in the directory
import os
import h5py
directory = "real_datasets_box/"
h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
file_names = [f.split(".")[0] for f in h5_files]
for file in h5_files:
    dataset_name = file.split(".")[0]
    with h5py.File(f"{directory}/{dataset_name}.h5", "r") as f:
        camera_qpos = f["/camera_pose/qpos"][0]
    # Placeholder for joint commands
    joint_commands = camera_qpos[:6]  # Replace with actual joint values

    # Compute the end-effector pose
    ee_pose = mr.FKinSpace(vx300s.M, vx300s.Slist, joint_commands)

    # Compute the camera pose
    camera_pose = ee_pose @ hand_eye

    with h5py.File(f"{directory}/{dataset_name}.h5", "a") as f:
        f.create_dataset("/camera_pose/ee_pose", data=camera_pose)