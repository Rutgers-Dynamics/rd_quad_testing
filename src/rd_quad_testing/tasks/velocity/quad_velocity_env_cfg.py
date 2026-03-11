import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import JointVelocityActionCfg
from mjlab.managers import ObservationTermCfg, SceneEntityCfg, ObservationGroupCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import BuiltinSensorCfg, ObjRef, ContactSensorCfg, ContactMatch
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig
from mjlab.envs import mdp

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
        nconmax=50
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

    def foot_height(env, asset_cfg = SceneEntityCfg("robot",body_names=".*foot_collision$")):
        asset = env.scene[asset_cfg.name]
        return asset.data.geom_pos_w[:, asset_cfg.site_ids, 2]

    def chassis_height(env, asset_cfg = SceneEntityCfg("robot",site_names="imu")):
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
        )
        # include command, camera, and lidar sensor data
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

    observations ={
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
    Rewards
    """


    rewards = {

    }