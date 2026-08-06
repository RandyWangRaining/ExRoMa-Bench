# Data Collection and Evaluation

## Collect attempted rollouts

The following command executes 100 deterministic, randomized attempts using a
cuRobo expert and strict arm self-collision checking:

```bash
exroma collect \
  --task stack_blocks_two \
  --scene procedural_moon \
  --attempts 100 \
  --seed 20000 \
  --strict-collision-check \
  --headless
```

`--attempts` is the total number executed, not a target number of successes.
Every attempt contributes to the measured success rate. Failed temporary HDF5
files are deleted; only successful demonstrations become
`episode_XXXXXX.hdf5`. The output directory also contains
`collection_summary.json`, `instructions.json`, and, unless `--no-video` is
set, a three-view MP4 for each saved episode.

Use `--output PATH` to choose the dataset directory. Without it, outputs go to
`datasets/<task>_<scene>_seed<seed>_run<attempts>/`.

## Evaluate without saving demonstrations

```bash
exroma evaluate \
  --task stack_blocks_two \
  --scene procedural_moon \
  --episodes 100 \
  --seed 30000 \
  --strict-collision-check \
  --headless
```

Evaluation writes attempt outcomes and failure counts under `evaluations/`,
but does not initialize cameras or save HDF5 episodes.

## Reproducibility

The episode sampler uses `--seed`; the prompt sampler uses a deterministic
offset from that seed. Reusing a task, scene, seed, and code/asset revision
replays the same requested object samples and prompt indices. Physics and GPU
planning can still introduce small numerical differences, so the summary keeps
the sampled pose and terminal failure reason for every attempt.

## HDF5 contract

The compact dual-PiPER representation records:

- `observations/qpos`: 14 values, six joints and one merged gripper value per arm.
- `observations/qvel`: 14 values with the same ordering.
- `observations/base_pose`: 7 values, world position and quaternion.
- `observations/base_velocity`: 6 values, linear and angular velocity.
- `observations/images/{mast,front_left,front_right}/rgb_jpeg`: raw JPEG bytes.
- `actions/joint_position_target`: 14 arm/gripper targets.
- `actions/base_velocity`: commanded linear and angular rover velocity.
- `observations/objects/<name>/pose`: task-object world poses.

The HDF5 file retains 41 raw non-image values so world-frame base pose and full
base velocity remain available for analysis. The compact learning interface
derives a 16-dimensional state: 14 arm/gripper joint positions plus rover
forward and yaw velocities. Its action is also 16-dimensional: 14 arm/gripper
targets plus rover linear and angular commands. The mast is fixed during
collection and is therefore not part of the action. For compatibility, hard
links expose `action` and RoboTwin-style camera names without duplicating
payloads.

See [Remote Policy Evaluation](remote_policy_evaluation.md) for the WebSocket
client/server contract used to evaluate a trained vision policy.

HDF5 camera frames are never annotated. Labels and view names exist only in
the optional MP4 visualization.

## Supported tasks

| Task | Main behavior | External task assets |
|---|---|---|
| `stack_blocks_two` | Two-stage block stacking | Procedural RoboTwin-size blocks |
| `handover_block` | Bimanual transfer and placement | Procedural RoboTwin-size block |
| `scan_object` | Scanner and object alignment | RoboTwin scanner and tea box |
| `scan_rock` | Scanner-to-rock inspection | RoboTwin scanner and OmniLRS rock |
| `beat_block_hammer` | Tool grasp and strike | RoboTwin hammer |
| `test_tube_rack` | Precision tube insertion | ExRoMa procedural tube and rack |

Only these registered tasks are available to `preview`, `collect`, and
`evaluate`. Previously rejected bin, cabinet, and switch tasks are not exposed
in the benchmark registry.
