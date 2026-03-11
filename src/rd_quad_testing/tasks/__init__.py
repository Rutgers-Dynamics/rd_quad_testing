from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.runner import MjlabOnPolicyRunner

from rd_quad_testing.tasks.stand.quad_stand_env_cfg import env_cfg
from rd_quad_testing.tasks.stand.rl_cfg import ppo_runner_cfg

register_mjlab_task(
  task_id="mjlab_quad_stand",
  env_cfg=env_cfg(),
  play_env_cfg=env_cfg(play=True),
  rl_cfg=ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)