"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""

from pathlib import Path

import imageio
import numpy as np
import torch
from pathlib import Path
from loop_rate_limiters import RateLimiter
from aloha_mink_wrapper import AlohaMinkWrapper
import mujoco
import mujoco.viewer
import cv2
import threading
import queue
from torchvision.transforms import v2
import mink
import copy

from lerobot.common.policies.act.modeling_act import ACTPolicy

from mug_pickup import initialize_scene, display_image, sample_object_position, check_object_lifted, get_robot_data
from mug_pickup_av import move_to_optimal_view, is_gripper_near_optimal_view, move_to_object, sample_constrained_pos_noise, sample_constrained_quat_noise, state_to_transform, transform_to_wxyz_xyz, transform_to_state

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"

def preprocess_image(img):
    """Preprocess the image to be compatible with the policy."""
    # normalize
    img = img / 255
    # permute to channel first
    img = img.permute(2, 0, 1)
    # resize to 240x320
    img = v2.Resize((240, 320))(img)
    # center crop to 224x308
    img = v2.CenterCrop((224, 308))(img)
    return img

if __name__ == "__main__":
    # Load the pretrained policy
    pretrained_policy_path = Path("mug_pickup_av_real_policy")

    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        device = torch.device("cpu")
        print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
    policy.to(device)
    # policy.config.temporal_ensemble_coeff=None
    policy.config.temporal_ensemble_coeff=0.01
    # Reset the policy and environmens to prepare for rollout
    policy.reset()

    # Load the Mujoco model and data
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    # Initialize the aloha_mink_wrapper
    aloha_mink_wrapper = AlohaMinkWrapper(model, data)

    # Initialize to the neutral pose
    initialize_scene(data, model, aloha_mink_wrapper)
    for i in range(10000):
        # left
        T_left_base_world = aloha_mink_wrapper.get_left_base_to_world(data)
        state_left_base = ...
        state_left_world = T_left_base_world @ state_left_base
        
        # right
        T_right_base_world = aloha_mink_wrapper.get_right_base_to_world(data)
        state_right_base = ...
        state_right_world = T_right_base_world @ state_right_base

        state_left_right = AlohaMinkWrapper.pose_inv(state_right_world) @ state_left_world
        state = transform_to_state(state_left_right)
        
        # get image
        img = ...
        img = preprocess_image(img)
        observation = {
            "observation.state": state.unsqueeze(0),
            "observation.images.wrist_cam_right": img.unsqueeze(0),
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = action.squeeze(0).to("cpu").numpy()
        action_left_right = state_to_transform(action)
        tf_left_to_world = state_right_world @ action_left_right
        # transform transformation matrix to wxyz_xyz
        wxyz_xyz = transform_to_wxyz_xyz(tf_left_to_world)

        # control the robot to move to the predicted pose
        ...
                