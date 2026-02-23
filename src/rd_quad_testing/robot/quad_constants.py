from pathlib import Path
import mujoco
import os
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg

XML_PATH: Path = Path(os.path.dirname(__file__)) / "Quadruped.xml"
assert XML_PATH.exists(), f"XML not found: {XML_PATH}"

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(XML_PATH))

def get_robot_cfg() -> EntityCfg:
  articulation = EntityArticulationInfoCfg()
  return EntityCfg(
    spec_fn=get_spec,
    articulation=articulation
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer
  robot = Entity(get_robot_cfg())
  viewer.launch(robot.spec.compile())