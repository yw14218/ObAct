import mujoco
from mujoco.viewer import launch_passive

# Path to your XML model file
xml_path = "aloha/objects/mug.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)

# Create simulation data for the model
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(
    model=model, data=data, show_left_ui=False, show_right_ui=False
) as viewer:
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    while viewer.is_running():
        # Step the simulation
        mujoco.mj_step(model, data)

        # Visualize at fixed FPS
        viewer.sync()
