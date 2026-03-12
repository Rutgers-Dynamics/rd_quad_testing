import math

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import JointVelocityActionCfg
from mjlab.managers import ObservationTermCfg, SceneEntityCfg, ObservationGroupCfg, EventTermCfg, RewardTermCfg, \
    TerminationTermCfg, CurriculumTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import BuiltinSensorCfg, ObjRef, ContactSensorCfg, ContactMatch
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig
from mjlab.envs import mdp
from mjlab.tasks.velocity import mdp as vel_mdp

from rd_quad_testing.robot.quad_constants import get_robot_cfg

_DECIMATION = 4

def env_cfg(play=False) -> ManagerBasedRlEnvCfg:
    scene_cfg = SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="plane",
        ),
        num_envs=1,
        extent=1,
        entities={"robot": get_robot_cfg()},
        sensors=(
            ContactSensorCfg(
                name="foot_ground_contact",
                primary=ContactMatch(
                    mode="geom",
                    pattern=".*foot_collision$",
                    entity="robot",
                ),
                secondary=ContactMatch(
                    mode="body",
                    pattern="terrain"
                ),
                fields=("found", "force"),
                reduce="maxforce",
                track_air_time=True,
            ),
            BuiltinSensorCfg(
                name="angular_momentum",
                sensor_type="subtreeangmom",
                obj=ObjRef(type="body",name="Chassis-Frame-v2", entity="robot"),
            ),
        )
    )

    viewer_cfg = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="Chassis-Frame-v2",
        distance=3,
        elevation=3,
        azimuth=45,
    )

    sim_cfg = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.005,
            iterations=10,
            ls_iterations=20,
        ),
        njmax=1000,
        nconmax=100,
    )

    """
    Actions
    """
    actions = {
        "joint_vel": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=30.0,
        )
    }

    """
    Observations
    """

    def foot_contact(env):
        sensor = env.scene["foot_ground_contact"]
        sensor_data = sensor.data
        assert sensor_data.found is not None
        return (sensor_data.found > 0).float()

    def foot_contact_forces(env):
        sensor = env.scene["foot_ground_contact"]
        assert sensor.data.force is not None
        sensor_force = sensor.data.force.flatten(start_dim=1)
        return torch.sign(sensor_force) * torch.log1p(torch.abs(sensor_force))

    def foot_air_time(env):
        sensor = env.scene["foot_ground_contact"]
        assert sensor.data.current_air_time is not None
        return sensor.data.current_air_time

    def foot_height(env, asset_cfg=SceneEntityCfg("robot", body_names=".*foot_collision$")):
        asset = env.scene[asset_cfg.name]
        return asset.data.geom_pos_w[:, asset_cfg.site_ids, 2]

    def chassis_height(env, asset_cfg=SceneEntityCfg("robot", site_names="imu")):
        asset = env.scene[asset_cfg.name]
        return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]

    policy_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        ),
        "chassis_lin_acc": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_acc"},
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        ),
        "chassis_ang_acc": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_gyro"},
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        ),
        "last_action": ObservationTermCfg(
            func=mdp.last_action,
        ),
        "foot_contact": ObservationTermCfg(
            func=foot_contact,
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name":"twist"},
        ),

        # include camera, and lidar sensor data
    }

    critic_terms = {
        **policy_terms,
        "foot_contact_forces": ObservationTermCfg(
            func=foot_contact_forces,
        ),
        "foot_air_time": ObservationTermCfg(
            func=foot_air_time,
        ),
        "foot_height": ObservationTermCfg(
            func=foot_height,
        ),
        "chassis_height": ObservationTermCfg(
            func=chassis_height,
        )
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        )
    }

    """
    Events
    """
    events = {
        # "reset_base": EventTermCfg(
        #     func=mdp.reset_root_state_uniform,
        #     mode="reset",
        #     params={
        #         "pose_range": {
        #             "x": (-0.5, 0.5),
        #             "y": (-0.5, 0.5),
        #             "z": (0.01, 0.05),
        #             "yaw": (-3.14, 3.14),
        #         },
        #         "velocity_range": {},
        #     }
        # ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-1,1),
                "velocity_range": (0,0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            }
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0,5.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            }
        ),
        #include domain randomization
    }

    """
    Commands
    """
    commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.3,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi,math.pi),
            )
        )
    }


    """
    Rewards
    """

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=vel_mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=vel_mdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        # "flat_orientation": RewardTermCfg(
        #     func=vel_mdp.flat_orientation,
        #     weight=1.0,
        #     params={
        #         "std": math.sqrt(0.2),
        #         "asset_cfg": SceneEntityCfg("robot")
        #     }
        # ),
        "joint_pose": RewardTermCfg(
            func=vel_mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "command_name": "twist",
                "std_standing": {".*j1": 0.3,
                                 ".*j2": 0.25,
                                 ".*j3": 1.0,
                                 },
                "std_walking": {".*j1": 0.4,
                                 ".*j2": 0.5,
                                 ".*j3": 1.0,
                                 },
                "std_running": {".*j1": 0.5,
                                 ".*j2": 0.75,
                                 ".*j3": 1.0,
                                 },
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=vel_mdp.body_angular_velocity_penalty,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("Chassis-Frame-v2",))},
        ),
        "angular_momentum": RewardTermCfg(
            func=vel_mdp.angular_momentum_penalty,
            weight=-0.5,
            params={"sensor_name": "robot/angular_momentum"},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "air_time": RewardTermCfg(
            func=vel_mdp.feet_air_time,
            weight=1.0,
            params={
                "sensor_name": "foot_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.5
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=vel_mdp.feet_clearance,
            weight=-2.0,
            params={
                "target_height": 0.1,
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=(".*-foot-site",))
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=vel_mdp.feet_swing_height,
            weight=-0.25,
            params={
                "sensor_name": "foot_ground_contact",
                "target_height": 0.1,
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=(".*-foot-site",)),
            },
        ),
        "foot_slip": RewardTermCfg(
            func=vel_mdp.feet_slip,
            weight=-0.1,
            params={
                "sensor_name": "foot_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=(".*-foot-site",)),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=vel_mdp.soft_landing,
            weight=-1e-5,
            params={
                "sensor_name": "foot_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
    }

    """
    Terminations
    """
    terminations={
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        # "fell_over": TerminationTermCfg(
        #     func=vel_mdp.bad_orientation,
        #     params={"limit_angle": math.radians(70.0)},
        # ),
    }

    """
    Curriculum
    """

    curriculum={
        "command_vel": CurriculumTermCfg(
            func=vel_mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                    {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
                    {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
                ],
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=viewer_cfg,
        sim=sim_cfg,
        decimation=_DECIMATION,
        episode_length_s=20.0
    )
