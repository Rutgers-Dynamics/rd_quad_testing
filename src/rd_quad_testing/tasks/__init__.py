from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.runner import MjlabOnPolicyRunner

from rd_quad_testing.tasks.stand.quad_stand_env_cfg import env_cfg as stand_env_cfg
from rd_quad_testing.tasks.velocity.quad_velocity_env_cfg import env_cfg as velocity_env_cfg
from rd_quad_testing.tasks.stand.rl_cfg import ppo_runner_cfg as stand_ppo_runner_cfg
from rd_quad_testing.tasks.velocity.rl_cfg import ppo_runner_cfg as vel_ppo_runner_cfg

register_mjlab_task(
  task_id="mjlab_quad_stand",
  env_cfg=stand_env_cfg(),
  play_env_cfg=stand_env_cfg(play=True),
  rl_cfg=stand_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
    task_id="quad_vel",
    env_cfg=velocity_env_cfg(),
    play_env_cfg=velocity_env_cfg(play=True),
    rl_cfg=vel_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)