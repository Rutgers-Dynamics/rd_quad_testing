# Testing for quadruped with mjlab.

___

Contains the mjcf model of the quadruped as well as some tasks to train. Please see [mjlab](https://github.com/mujocolab/mjlab) for more documentation.

## Installation
Ensure that uv is installed.

Clone the repository then

```uv sync```

If you don't have a nvida gpu, the torch dependencies may break. If this is the case, please make sure that the cpu version of torch is specified in the pyproject.toml file.

## Train Task
```
uv run train mjlab_quad_stand --env.scene.num-envs 512
```

## Play Task
```
uv run play mjlab_quad_stand --checkpoint-file <checkpoint_file>
```