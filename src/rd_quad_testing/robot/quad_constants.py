from pathlib import Path
import mujoco
import os

from mjlab.actuator import XmlVelocityActuator, XmlVelocityActuatorCfg
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.scene import SceneCfg, Scene
from mjlab.terrains import TerrainImporterCfg

XML_PATH: Path = Path(os.path.dirname(__file__)) / "Quadruped.xml"
assert XML_PATH.exists(), f"XML not found: {XML_PATH}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(XML_PATH))


def get_robot_cfg() -> EntityCfg:
    actuators = (
        XmlVelocityActuatorCfg(
            target_names_expr=("^Leg.*",),
        ),
    )

    articulation = EntityArticulationInfoCfg(actuators=actuators)
    return EntityCfg(
        spec_fn=get_spec,
        articulation=articulation
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": get_robot_cfg()}
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())
