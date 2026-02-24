import math

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig
from mjlab.envs import mdp
from mjlab.envs.mdp import rewards
import mjlab.tasks.velocity.mdp as mdp_vel

from rd_quad_testing.robot.quad_constants import get_robot_cfg


def env_cfg(play=False) -> ManagerBasedRlEnvCfg:
    scene_cfg = SceneCfg(
        terrain=TerrainImporterCfg(
            terrain_type="plane",
        ),
        num_envs=16,
        extent=1.0,
        entities={"robot": get_robot_cfg()},
    )

    viewer_cfg = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="Chassis-Frame-v2",
        distance=3.0,
        elevation=3.0,
        azimuth=45.0,
    )

    sim_cfg = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.01,
            iterations=1,
        ),
        njmax=1000,
    )

    chassis_contact_cfg = ContactSensorCfg(
        name="bad_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=("Chassis-Frame-v2_geom", "Leg-Right-Thigh-v3-1_geom", "Leg-Right-Thigh-v3_geom", "Leg-Left-Thigh-v3-1_geom", "Leg-Left-Thigh-v3_geom"),
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain"
        ),
        fields=("found",),
        reduce="none",
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Chassis-Frame-v2", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Chassis-Frame-v2", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    """
    Actions
    """
    actions = {
        "joint_pos": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=30.0
        )
    }
    """
    Observations
    """
    policy_terms = {
        "joint_pos": ObservationTermCfg(
            func=lambda env: env.sim.data.qpos[:,7: ], # 7 entries for the free joint we want to exclude
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=lambda env: env.sim.data.qvel[:,6: ], # 6 entries for the free joint we want to exclude
            noise=UniformNoiseCfg(n_min=-1.0, n_max=1.0)
        ),
        "body_orientation": ObservationTermCfg(
            func=lambda env: env.sim.data.qpos[:,3:7],
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
    }

    critic_terms = {
        **policy_terms,
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False if play else True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    """
    Rewards
    """

    def height(env, target_height = 0.75, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("imu",))) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        # proj_grav = asset.data.projected_gravity_b[:,2] **2
        height = asset.data.site_pos_w[:,0,2]
        raw_height_diff = abs(height-target_height)
        height_tfm = torch.where(raw_height_diff < 0.075, 1, -raw_height_diff)
        # return  height_tfm #proj_grav *
        return height_tfm

    rewards = {
        "height": RewardTermCfg(
            func=height,
            weight=1.0,
        ),
        "chassis_pose": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=1.25,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("Chassis-Frame-v2",)),
            }
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        "is_terminated": RewardTermCfg(func=mdp.rewards.is_terminated, weight=-1),
        "is_alive": RewardTermCfg(func=mdp.rewards.is_alive, weight=0.01),
    }

    """
    Termination
    """

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "chassis_touch_ground": TerminationTermCfg(
            func=mdp_vel.illegal_contact,
            params={"sensor_name": "bad_ground_contact"},
        )
    }

    """
    events
    """

    events = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.1, 0.1),
            },
        ),
    }

    cfg= ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        observations=observations,
        actions=actions,
        rewards=rewards,
        events=events,
        terminations=terminations,
        sim=sim_cfg,
        viewer=viewer_cfg,
        decimation=1,
        episode_length_s=1000 if play else 10.0,
    )

    cfg.scene.sensors = (chassis_contact_cfg, self_collision_cfg)

    return cfg



