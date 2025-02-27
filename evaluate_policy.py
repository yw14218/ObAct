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

from lerobot.common.policies.act.modeling_act import ACTPolicy

from mug_pickup import initialize_scene, display_image, sample_object_position, get_robot_data, check_object_lifted

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "merged_scene_mug.xml"
theta = 0

def set_object_qpos(object_qpos):
    """
    Set the object qpos in the Mujoco model.
    """
    data.qpos[16:23] = object_qpos
    # Forward propagate the simulation state
    mujoco.mj_forward(model, data)

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
    pretrained_policy_path = Path("ckpts\\mug_pickup_overhead_cam\\070000\\pretrained_model")

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
    # policy.config.n_action_steps=1
    # Reset the policy and environmens to prepare for rollout
    policy.reset()

    # Load the Mujoco model and data
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    # Initialize the aloha_mink_wrapper
    aloha_mink_wrapper = AlohaMinkWrapper(model, data)

    # Initialize to the neutral pose
    initialize_scene(data, model, aloha_mink_wrapper)

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
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Randomize the object's position
            object_qpos = sample_object_position(data, model)

            # Set the initial posture target
            aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)

            # Rate limiter for fixed update frequency
            rate = RateLimiter(frequency=100, warn=False)

            episode_cnt = 0
            step_cnt = 0
            success_cnt = 0
            cam_imgs = {}
            camera_keys = ["overhead_cam"]

            print(f"Episode {episode_cnt} started...")

            try:
                while viewer.is_running():
                    _, state, images = get_robot_data(data, model, renderer, aloha_mink_wrapper, camera_keys=["overhead_cam"])

                    state = torch.from_numpy(state).float().to(device)
                    for key, img in zip(camera_keys, images):
                        cam_imgs[key] = torch.from_numpy(img).float().to(device)
                        cam_imgs[key] = preprocess_image(cam_imgs[key])

                    # prepare observation
                    observation = {
                        "observation.state": state.unsqueeze(0),
                    }
                    for key in camera_keys:
                        observation[f"observation.images.{key}"] = cam_imgs[key].unsqueeze(0)
                    cam_imgs = {}

                    # Predict the next action with respect to the current observation
                    with torch.inference_mode():
                        action = policy.select_action(observation)
                    
                    action = action.squeeze(0).to("cpu").numpy()

                    data.ctrl[aloha_mink_wrapper.actuator_ids[:6]] = action[:6]
                    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/gripper")] = action[6]

                    # Check if the object is lifted
                    if check_object_lifted(data, model):
                        print("Object lifted!")
                        episode_cnt += 1
                        success_cnt += 1
                        step_cnt = 0
                        print(f"Success Rate: {success_cnt}/{episode_cnt}")
                        # Reinitalize the scene
                        initialize_scene(data, model, aloha_mink_wrapper)
                        # Reset the object's position
                        object_qpos = sample_object_position(data, model)
                        # Set the initial posture target
                        aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
                        policy.reset()
                        print(f"Episode {episode_cnt} started...")
                    
                    # Check if the episode has failed
                    if step_cnt > 1000:
                        print("Episode failed. Resetting the scene...")
                        episode_cnt += 1
                        step_cnt = 0
                        print(f"Success Rate: {success_cnt}/{episode_cnt}")
                        # Reinitalize the scene
                        initialize_scene(data, model, aloha_mink_wrapper)
                        # Reset the object's position
                        object_qpos = sample_object_position(data, model)
                        # Set the initial posture target
                        aloha_mink_wrapper.tasks[2].set_target_from_configuration(aloha_mink_wrapper.configuration)
                        policy.reset()
                        print(f"Episode {episode_cnt} started...")

                    # Compensate gravity
                    aloha_mink_wrapper.compensate_gravity([model.body("left/base_link").id, model.body("right/base_link").id])

                    step_cnt += 1
                    # print("Step count:", step_cnt)

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




