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

#TODO
directory = "real_datasets_mug"
sample_period = 1

h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
file_names = [f.split(".")[0] for f in h5_files]
episode_cnt = 0
for file_name in file_names[episode_cnt:]:
    # if episode_cnt == 7:
    #     episode_cnt += 1
    #     continue
    dataset_name = file_name

    with h5py.File(f"{directory}/{dataset_name}.h5", "r") as f:
        robot_poses = f["/observations/ee_pose"][:]
        robot_qpos = f["/observations/qpos"][:]
        camera_pose = f["/camera_pose/ee_pose"][:]
        images = f["/observations/images/rgbs"][:]

    states = []
    actions = []
    imgs = []
    for i in range(1, len(robot_poses)//sample_period):
        # left
        T_left_base_world = aloha_mink_wrapper.get_left_base_to_world(data)
        state_left_base = robot_poses[(i-1)*sample_period]
        state_left_world = T_left_base_world @ state_left_base
        action_left_base = robot_poses[i*sample_period]
        action_left_world = T_left_base_world @ action_left_base
        
        # right
        T_right_base_world = aloha_mink_wrapper.get_right_base_to_world(data)
        state_right_base = camera_pose
        state_right_world = T_right_base_world @ state_right_base

        state_left_right = AlohaMinkWrapper.pose_inv(state_right_world) @ state_left_world
        action_left_right = AlohaMinkWrapper.pose_inv(state_right_world) @ action_left_world
        
        state = transform_to_state(state_left_right)
        gripper_state = robot_qpos[(i-1)*sample_period][6]
        action = transform_to_state(action_left_right)
        gripper_action = robot_qpos[i*sample_period][6]
        if gripper_state < 0.8:
            gripper_state = 1
        else:
            gripper_state = 0
        if gripper_action < 0.8:
            gripper_action = 1
        else:
            gripper_action = 0
        state = np.concatenate((state, [gripper_state]), axis=0)
        action = np.concatenate((action, [gripper_action]), axis=0)
        states.append(state)
        actions.append(action)

        image = images[(i-1)*sample_period, :, :, :]

        masked_image = image.copy()
        
        # # read mask
        # mask = cv2.imread(f"D:\\Cutie\\workspace\\episode_{episode_cnt}_side\\masks\\{i-1:07d}.png", cv2.IMREAD_UNCHANGED)
        # # cv2.imshow("mask", mask)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # # invert the mask
        # mask = 38 - mask
        # masked_image = cv2.bitwise_and(image, image, mask=mask)
        # # make background white
        # masked_image[mask == 0] = [255, 255, 255]
        # # print(image.shape)
        # # print(mask.shape)
        # # visualize the image
        # cv2.imshow("image", masked_image)

        # put step on the text
        print(f"step: {i}")
        # appear next image after pressing any key
        cv2.waitKey(1)
        
        # Convert bgr to rgb
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        imgs.append(masked_image)

    # start_idx = int(input("Enter the start index: "))
    # end_idx = int(input("Enter the end index: "))
    start_idx = 0
    end_idx = len(states)
    states = np.array(states[start_idx:end_idx])
    actions = np.array(actions[start_idx:end_idx])
    imgs = np.array(imgs[start_idx:end_idx])

    import datetime
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    # save imgs to a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = f"D:\\Cutie\\examples\\episode_{episode_cnt}_side.mp4"
    out = cv2.VideoWriter(video_name, fourcc, 30.0, (imgs[0].shape[1], imgs[0].shape[0]))
    for i in range(len(imgs)):
        out.write(imgs[i])
    out.release()

    os.makedirs(f"{directory}_{int(30//sample_period)}hz", exist_ok=True)

    with h5py.File(f"{directory}_{int(30//sample_period)}hz/episode_{episode_cnt}.h5", "w") as f:
        f.create_dataset("/observations/qpos", data=states)
        f.create_dataset("/action", data=actions)
        f.create_dataset("/observations/images/wrist_cam_right", data=imgs)
    episode_cnt += 1