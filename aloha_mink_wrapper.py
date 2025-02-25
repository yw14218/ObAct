from typing import Sequence
import mink
import numpy as np
import mujoco

class AlohaMinkWrapper:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Initialize configuration
        self.configuration = mink.Configuration(model)

        # Tasks
        self.tasks = self.create_tasks()

        # Velocity limits
        self.joint_names, self.velocity_limits = self.get_joint_and_velocity_limits()
        self.dof_ids = np.array([model.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([model.actuator(name).id for name in self.joint_names])

        # Collision avoidance limits
        self.collision_avoidance_limit = self.get_collision_avoidance_limit()

        # All limits
        self.limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, self.velocity_limits),
            self.collision_avoidance_limit,
        ]
        

    def create_tasks(self):
        """Create and return a list of Mink tasks."""
        tasks = [
            mink.FrameTask(
                frame_name="left/gripper",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
            mink.FrameTask(
                frame_name="right/gripper",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
            mink.PostureTask(self.model, cost=1e-4),
        ]
        return tasks

    def get_collision_avoidance_limit(self):
        """Configure collision avoidance for the model."""
        l_wrist_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("left/wrist_link").id)
        r_wrist_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("right/wrist_link").id)

        l_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("left/upper_arm_link").id)
        r_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("right/upper_arm_link").id)
        frame_geoms = mink.get_body_geom_ids(self.model, self.model.body("metal_frame").id)
        

        collision_pairs = [
            (l_wrist_geoms, r_wrist_geoms),
            (l_geoms + r_geoms, frame_geoms + ["table"]),
        ]
        return mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.05,
        )

    def get_joint_and_velocity_limits(self):
        """Return joint names and velocity limits."""
        joint_names = []
        velocity_limits = {}
        joint_names_list = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        ]
        single_arm_velocity_limits = {k: np.pi / 2 for k in joint_names_list}
        for prefix in ["left", "right"]:
            for n in joint_names_list:
                name = f"{prefix}/{n}"
                joint_names.append(name)
                velocity_limits[name] = single_arm_velocity_limits[n]
        return joint_names, velocity_limits

    def initialize_mocap_targets(self):
        """Align mocap targets with end-effector sites."""
        mink.move_mocap_to_frame(self.model, self.data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(self.model, self.data, "right/target", "right/gripper", "site")

    def solve_ik(self, rate_dt, solver="quadprog", max_iters=5, pos_threshold=5e-3, ori_threshold=5e-3):
        """Solve inverse kinematics with limits."""
        for i in range(max_iters):
            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                rate_dt,
                solver,
                limits=self.limits,
                damping=1e-1,
            )
            self.configuration.integrate_inplace(vel, rate_dt)

            l_err = self.tasks[0].compute_error(self.configuration)
            l_pos_achieved = np.linalg.norm(l_err[:3]) <= pos_threshold
            l_ori_achieved = np.linalg.norm(l_err[3:]) <= ori_threshold
            r_err = self.tasks[1].compute_error(self.configuration)
            r_pos_achieved = np.linalg.norm(r_err[:3]) <= pos_threshold
            r_ori_achieved = np.linalg.norm(r_err[3:]) <= ori_threshold
            if (
                l_pos_achieved
                and l_ori_achieved
                and r_pos_achieved
                and r_ori_achieved
            ):
                break

    def compensate_gravity(self, subtree_ids: Sequence[int]):
        """Compute forces to counteract gravity for the given subtrees."""
        qfrc_applied = self.data.qfrc_applied
        qfrc_applied[:] = 0.0  # Don't accumulate from previous calls.
        jac = np.empty((3, self.model.nv))
        for subtree_id in subtree_ids:
            total_mass = self.model.body_subtreemass[subtree_id]
            mujoco.mj_jacSubtreeCom(self.model, self.data, jac, subtree_id)
            qfrc_applied[:] -= self.model.opt.gravity * total_mass @ jac

