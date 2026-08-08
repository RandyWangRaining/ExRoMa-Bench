# Remote Policy Evaluation

ExRoMa can keep Isaac Sim and a learned policy in separate processes or Conda
environments. The interface follows the same client/server separation used by
RoboTwin Arena's remote policy deployment, while defining an ExRoMa-specific,
versioned 16-dimensional rover-manipulator control contract.

## What is transmitted

The simulator client sends one observation request containing:

- `state[16]`: 14 compact dual-PiPER joint values, rover forward velocity, and
  rover yaw velocity.
- `images`: raw RGB frames from `cam_high`, `cam_left_wrist`, and
  `cam_right_wrist`, JPEG-compressed for transport. No labels are drawn on the
  images.
- `prompt`: the sampled natural-language task instruction.
- `timestamp`: simulation time in seconds.

The policy server returns only:

- `actions[T,16]`: one action or an action chunk.

The action ordering is:

| Indices | Meaning | Units |
| --- | --- | --- |
| `0:6` | Left PiPER joint position targets | rad |
| `6` | Left merged gripper target | m |
| `7:13` | Right PiPER joint position targets | rad |
| `13` | Right merged gripper target | m |
| `14` | Rover forward command | normalized `[-1, 1]` |
| `15` | Rover yaw command | normalized `[-1, 1]` |

The mast is fixed at its configured default and is not controlled by the
policy. Joint targets and base commands are validated for shape and finite
values; joint targets are clipped to the articulation's soft limits. The
protocol name is `exroma.policy.v1`.

Sending only 16 numbers from the policy server to the simulator is therefore
correct. A vision policy still needs the three images in the opposite
direction. A proprioceptive-only policy may ignore `observation["images"]`,
but the current protocol keeps the request schema fixed.

## 1. Start the server

The built-in policy holds the current arm pose and stops the rover. It is only
a connectivity and safety smoke test:

```bash
cd /home/ruilin/ExRoMa
conda activate openspace

python scripts/serve_policy.py \
  --host 0.0.0.0 \
  --port 8000
```

`0.0.0.0` makes the server reachable from another machine. Use
`127.0.0.1` when client and server run on the same machine. This endpoint has
no TLS or authentication, so expose it only on a trusted network.

## 2. Test one recorded observation

In a second terminal, send one real HDF5 frame before launching Isaac Sim:

```bash
cd /home/ruilin/ExRoMa
conda activate openspace

python scripts/query_policy_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --episode datasets/stack_blocks_two_procedural_moon_seed20000_run100/episode_000000.hdf5 \
  --frame 0
```

The output must report `state_shape: [16]` and `actions_shape: [T, 16]`.

## 3. Evaluate in simulation

Keep the policy server running, then start the ExRoMa client:

```bash
cd /home/ruilin/ExRoMa
conda activate openspace

./exroma.sh evaluate \
  --task stack_blocks_two \
  --scene procedural_moon \
  --episodes 100 \
  --seed 100000 \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --policy-frequency 10 \
  --policy-action-horizon 8 \
  --headless
```

ExRoMa owns task randomization, physics, action application, task success
checks, and success-rate reporting. The policy owns inference only. Results go
to `evaluations/<task>_<scene>_seed<seed>_run<episodes>/collection_summary.json`
unless `--output` is provided. Each record includes the sampled object pose,
prompt, terminal result, number of policy queries, and inference timing.

`evaluate` defaults to `--episodes 100 --seed 100000`; these flags are shown
explicitly above only to make the benchmark protocol visible. The seed creates
one deterministic random stream for all 100 episodes rather than assigning
the integer seeds `100000` through `100099` episode by episode.

The built-in hold policy is not expected to complete a task. Use it to verify
the port and observation contract before connecting a trained model.

## Replay one recorded episode and save video

The named `replay` policy serves the recorded 14 joint targets and two rover
commands through the same WebSocket policy interface. Start the server in
terminal 1:

```bash
cd /home/ruilin/ExRoMa
conda activate openspace

python scripts/serve_policy.py \
  --host 127.0.0.1 \
  --port 8000 \
  --policy-name replay \
  --replay-episode datasets/stack_blocks_two_procedural_moon_seed20000_run100/episode_000000.hdf5 \
  --replay-action-horizon 8
```

Run the matching task, scene, task seed, and terrain seed in terminal 2:

```bash
cd /home/ruilin/ExRoMa
conda activate openspace

TERM=xterm ./exroma.sh evaluate \
  --task stack_blocks_two \
  --scene procedural_moon \
  --episodes 1 \
  --seed 20000 \
  --terrain-seed 0 \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --policy-frequency 10 \
  --policy-action-horizon 8 \
  --record-policy-video \
  --record-fps 10 \
  --output evaluations/replay_stack_blocks_two_episode_000000 \
  --headless
```

The output directory receives `episode_000000.hdf5`, a three-view
`episode_000000.mp4`, and `collection_summary.json`. Remote-policy evaluation
videos are retained whether the replay succeeds or fails, because they are
diagnostic artifacts rather than demonstration data.

A 10 Hz action replay is not identical to rerunning the original cuRobo expert,
which updated targets at the 50 Hz physics rate before recording was sampled.
The replay therefore preserves the policy-facing dataset contract but may
accumulate enough tracking error to fail the final task geometry check. Its
success field must be read from `collection_summary.json`.

## Named policy workspaces

Learned algorithms live under `policy/<Name>/`, following the per-policy
organization used by RoboTwin Arena. Each directory contains its adapter,
deployment YAML, Conda environment, and algorithm notes. The simulator does
not import ACT, PyTorch, JAX, or OpenPI; only the policy server does.

List the adapters currently installed in ExRoMa:

```bash
cd /home/ruilin/ExRoMa
python scripts/serve_policy.py --list-policies
```

The learned-policy workspaces are `ACT`, `pi05`, and `smolvla`; `replay` is
also available for recorded-action diagnostics. ACT and pi05 leave
`backend_factory` empty because ExRoMa does not redistribute third-party model
code or checkpoints. SmolVLA includes a native LeRobot adapter but still
requires an ExRoMa-finetuned checkpoint. Create a deployment YAML that points
at your installed model factory:

```yaml
policy_name: ACT
backend_factory: my_act_backend:create_model
checkpoint_dir: /path/to/act/checkpoint
device: cuda:0
state_dim: 16
action_dim: 16
chunk_size: 50
image_width: 640
image_height: 480
```

Then start the named policy from its own environment:

```bash
cd /home/ruilin/ExRoMa
conda env create -f policy/ACT/conda_env.yaml
conda activate exroma-act
python -m pip install --no-deps -e .

PYTHONPATH=/path/to/your/act/code:$PYTHONPATH \
python scripts/serve_policy.py \
  --policy-name ACT \
  --policy-config /path/to/act_deploy.yml \
  --checkpoint /path/to/act/checkpoint \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8000
```

For pi0.5, use `policy/pi05/conda_env.yaml`, `--policy-name pi05`, and install
OpenPI in that environment. See [Policy Workspaces](../policy/README.md),
[ACT](../policy/ACT/README.md), and [pi0.5](../policy/pi05/README.md).

For SmolVLA, use its Python 3.12 environment and native LeRobot adapter:

```bash
conda env create -f policy/smolvla/conda_env.yaml
conda activate exroma-smolvla

python scripts/serve_policy.py \
  --policy-name smolvla \
  --checkpoint /path/to/exroma_smolvla/pretrained_model \
  --device cuda:0 \
  --port 8000
```

See [SmolVLA](../policy/smolvla/README.md) for dataset features and fine-tuning.

The lifecycle maps cleanly to RoboTwin Arena's policy modules:

| ExRoMa | RoboTwin-style equivalent |
| --- | --- |
| `create_policy(config)` | `get_model(config)` |
| `policy.infer(observation)` | encode observation and call model action inference |
| `policy.reset()` | `reset_model(model)` |

Unlike an in-process evaluator, ExRoMa keeps those methods on the server. This
prevents Isaac Sim's pinned Python/CUDA packages from conflicting with an
algorithm's training and inference stack.

## Generic policy adapter

For an experimental algorithm that does not yet need its own directory, create
a Python module available on the server's `PYTHONPATH`:

```python
import numpy as np


class TrainedPolicy:
    name = "my_exroma_policy"

    def __init__(self, config):
        self.config = config
        # Load the checkpoint once here.

    def reset(self):
        # Clear temporal history at the beginning of each episode.
        pass

    def infer(self, observation):
        state = observation["state"]               # float32 [16]
        images = observation["images"]             # three uint8 HWC RGB arrays
        prompt = observation["prompt"]
        actions = run_model(state, images, prompt) # [16] or [T, 16]
        return {"actions": np.asarray(actions, dtype=np.float32)}


def create_policy(config):
    return TrainedPolicy(config)
```

Start it with:

```bash
PYTHONPATH=/path/to/your/policy/code:$PYTHONPATH \
python scripts/serve_policy.py \
  --host 0.0.0.0 \
  --port 8000 \
  --policy-factory my_policy:create_policy \
  --policy-config /path/to/policy_config.json
```

The factory receives the decoded YAML or JSON mapping. Returning chunks reduces
network round trips; ExRoMa consumes at most `--policy-action-horizon` actions
before requesting a fresh observation. Once the adapter stabilizes, move it to
`policy/<Name>/` with its own `conda_env.yaml` and `deploy_policy.yml`.

## Safety boundary

The remote interface rejects malformed dimensions, NaN/Inf, and out-of-range
joint targets. Isaac PhysX remains responsible for physical contact response.
`--strict-collision-check` configures the cuRobo demonstration expert; it is
not an online collision shield for arbitrary learned-policy actions. A real
deployment should add an independent velocity, workspace, collision, timeout,
and emergency-stop supervisor after the 16-D policy output.
