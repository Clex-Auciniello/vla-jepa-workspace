# VLA-JEPA Online Latent World Model Pretraining

Workspace for online self-supervised pretraining of VLA-JEPA
on the real UR5e robotic manipulator.

## Objective

Adapt the Qwen3-VL latent action representations and
Latent World Model to the UR5e embodiment through
autonomous workspace exploration.

## Training pipeline

1. Sample task-agnostic waypoints.
2. Generate coverage-aware random trajectories.
3. Validate and execute robot motion through MoveIt.
4. Record synchronized RGB observations.
5. Extract target embeddings using frozen V-JEPA2.
6. Optimize the VLA-JEPA latent world model objective.
7. Repeat online.

## Experimental scenarios

- Empty tabletop.
- Randomized tabletop object configurations.

## Upstream

https://github.com/ginwind/VLA-JEPA

## Robot integration

ROS 2 controllers are maintained separately in UR5e-2f-85.
