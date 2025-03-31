import h5py
import os

# get all the h5 files in the directory
directory = "E:\\"
h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
# print(h5_files)

for file in h5_files:
    dataset_name = file.split(".")[0][:-3]

    with h5py.File(f"E:\\{dataset_name}_v2.h5", "r") as f:
        alphabetically_sorted_rgb = f['observations/images/rgb'][:]
        alphabetically_sorted_ee_pose = f['observations/ee_pose'][:]
        alphabetically_sorted_qpos = f['observations/qpos'][:]
        camera_pose = f["/camera_pose/ee_pose"][:]
        # sort also the joints, camera poses and etc.

    psudo_list = []
    for i in range(len(alphabetically_sorted_rgb)):
        psudo_name = "episode_" + str(i) + ".h5"
        psudo_list.append(psudo_name)

    # Perform alphabetical sort
    alphabetically_sorted_list = sorted(psudo_list)

    numerically_sorted_zip = sorted(zip(alphabetically_sorted_list, alphabetically_sorted_rgb, alphabetically_sorted_ee_pose, alphabetically_sorted_qpos), key=lambda item: int(item[0].split('_')[1].split('.')[0]))

    sorted_rgb = [item[1] for item in numerically_sorted_zip]
    sorted_ee_pose = [item[2] for item in numerically_sorted_zip]
    sorted_qpos = [item[3] for item in numerically_sorted_zip]

    # Save the sorted data
    with h5py.File(f"real_datasets/{dataset_name}_v3.h5", "w") as f:
        f.create_dataset("observations/images/rgb", data=sorted_rgb)
        f.create_dataset("observations/ee_pose", data=sorted_ee_pose)
        f.create_dataset("observations/qpos", data=sorted_qpos)
        f.create_dataset("/camera_pose/ee_pose", data=camera_pose)