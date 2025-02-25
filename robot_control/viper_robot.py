from typing import List

MOVE_GROUP_ARM: str = "interbotix_arm"
MOVE_GROUP_GRIPPER: str = "interbotix_gripper"

def joint_names(prefix: str = "vx300s") -> List[str]:
    return [
        prefix + "/waist",
        prefix + "/shoulder",
        prefix + "/elbow",
        prefix + "/forearm_roll",
        prefix + "/wrist_angle",
        prefix + "/wrist_rotate",
    ]

def base_link_name(prefix: str = "vx300s") -> str:
    return prefix + "/base_link"

def end_effector_name(prefix: str = "vx300s") -> str:
    return prefix + "/ee_gripper_link"
