import h5py
import numpy as np
import cv2
import os
from aloha_mink_wrapper import AlohaMinkWrapper
from scipy.spatial.transform import Rotation as R
import mujoco
from pathlib import Path
from mug_pickup_av import transform_to_state
from mug_pickup import initialize_scene

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"

# Load the Mujoco model and data
model = mujoco.MjModel.from_xml_path(str(_XML))
data = mujoco.MjData(model)

# Initialize the aloha_mink_wrapper
aloha_mink_wrapper = AlohaMinkWrapper(model, data)
# Initialize to the neutral pose
initialize_scene(data, model, aloha_mink_wrapper)

dataset_name = "20250324_203250_v3"[:-3]

with h5py.File(f"real_datasets/{dataset_name}_v3.h5", "r") as f:
    robot_poses = f["/observations/ee_pose"][:]
    camera_pose = f["/camera_pose/ee_pose"][:]
    images = f["/observations/images/rgb"][:]

states = []
actions = []
imgs = []
for i in range(1, len(robot_poses)):
    # left
    T_left_base_world = aloha_mink_wrapper.get_left_base_to_world(data)
    state_left_base = robot_poses[i-1]
    state_left_world = T_left_base_world @ state_left_base
    action_left_base = robot_poses[i]
    action_left_world = T_left_base_world @ action_left_base
    
    # right
    T_right_base_world = aloha_mink_wrapper.get_right_base_to_world(data)
    state_right_base = camera_pose
    state_right_world = T_right_base_world @ state_right_base

    state_left_right = AlohaMinkWrapper.pose_inv(state_right_world) @ state_left_world
    action_left_right = AlohaMinkWrapper.pose_inv(state_right_world) @ action_left_world
    
    state = transform_to_state(state_left_right)
    action = transform_to_state(action_left_right)
    states.append(state)
    actions.append(action)

    image = images[i-1, :, 104:-104, :]
    # visualize the image
    cv2.imshow("image", image)
    # put step on the text
    print(f"step: {i}")
    # appear next image after pressing any key
    cv2.waitKey(0)
    
    imgs.append(image)

start_idx = int(input("Enter the start index: "))
end_idx = int(input("Enter the end index: "))
states = np.array(states[start_idx:end_idx])
actions = np.array(actions[start_idx:end_idx])
imgs = np.array(imgs[start_idx:end_idx])

import datetime
now = datetime.datetime.now()
folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")

# os.makedirs(f"real_datasets/{folder_name}", exist_ok=True)

with h5py.File(f"real_datasets/{dataset_name}_v4.h5", "w") as f:
    f.create_dataset("/observations/qpos", data=states)
    f.create_dataset("/action", data=actions)
    f.create_dataset("/observations/images/wrist_cam_right", data=imgs)