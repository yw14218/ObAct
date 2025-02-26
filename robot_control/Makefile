all: launch

launch:
	@echo "Launching Interbotix XSArm Control and MoveIt..."
	ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300s & \
	sleep 3 && \
	ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py robot_model:=vx300s hardware_type:=actual & \
	sleep 3 && \
	python3.10 hand_eye_tf_broadcaster.py

.PHONY: all launch