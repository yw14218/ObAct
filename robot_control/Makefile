all: launch

launch:
	@echo "Launching Interbotix XSArm Control and MoveIt..."
	ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300s & \
	sleep 3 && \
	ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py robot_model:=vx300s hardware_type:=actual & \
	wait
	@echo "Launching RealSense Camera..."
	ros2 launch realsense2_camera rs_launch.py rgb_camera.color.profile:=848x480x30 depth_module.profile:=848x480x30 align_depth.enable:=true spatial_filter.enable:=true temporal_filter.enable:=true hole_filling_filter.enable:=true device_type:=d405

.PHONY: all launch
