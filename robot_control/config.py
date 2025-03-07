import numpy as np

d405_rgb_topic_name = "camera/camera/color/image_rect_raw"
d405_depth_topic_name = "camera/camera/aligned_depth_to_color/image_raw"

d405_extrinsic = np.load("robot_control/d405_extrinsic.npy")
d405_intrinsic = np.load("robot_control/d405_intrinsic.npy")
