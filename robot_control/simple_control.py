from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time

bot = InterbotixManipulatorXS(
    robot_model='vx300s',
    group_name='arm',
    gripper_name='gripper',
    moving_time=2,
    accel_time=0.3,
    
)

robot_startup()

while True:
    user_input = input("Enter 'g' to grasp or 'r' to release (or 'q' to quit): ").strip().lower()
    if user_input == 'g':
        bot.gripper.grasp()
    elif user_input == 'r':
        bot.gripper.release()
    elif user_input == 'q':
        break
    else:
        print("Invalid input. Please enter 'g', 'r', or 'q'.")
    time.sleep(1)