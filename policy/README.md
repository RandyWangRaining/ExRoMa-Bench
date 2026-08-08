# Policy workspaces

ExRoMa separates Isaac Sim from learned-policy dependencies. The simulator
runs the WebSocket client in the `openspace` environment; each policy server
can run in its own Conda environment and communicate through
`exroma.policy.v1`.

The layout follows RoboTwin Arena's useful per-algorithm convention:

```text
policy/
  ACT/
    __init__.py
    deploy_policy.py
    deploy_policy.yml
    conda_env.yaml
  pi05/
    __init__.py
    deploy_policy.py
    deploy_policy.yml
    conda_env.yaml
  replay/
    __init__.py
    deploy_policy.py
    deploy_policy.yml
  smolvla/
    __init__.py
    deploy_policy.py
    deploy_policy.yml
    conda_env.yaml
```

ExRoMa uses a smaller lifecycle contract than an in-process evaluator:

| Policy package API | Responsibility |
| --- | --- |
| `create_policy(config)` | Load one model/checkpoint when the server starts |
| `policy.infer(observation)` | Return `[16]` or `[T,16]` actions |
| `policy.reset()` | Clear temporal state between episodes |

The directory's `deploy_policy.yml` contains deployment defaults. CLI
`--checkpoint` and `--device` values override `checkpoint_dir` and `device`.
Model source and checkpoints remain external, so third-party licenses and GPU
requirements do not leak into the simulator environment.

`replay` is the exception: it has no learned-model dependency and runs directly
in the normal `openspace` environment. It reads one recorded HDF5 episode and
returns its 16-D actions through the same server contract.

`smolvla` uses a Python 3.12 LeRobot environment because it is intentionally
isolated from Isaac Sim's Python 3.11 runtime. Its native backend loads a
fine-tuned checkpoint with LeRobot's saved preprocessing and postprocessing
pipelines.

List installed adapters:

```bash
python scripts/serve_policy.py --list-policies
```

Start one named adapter:

```bash
python scripts/serve_policy.py \
  --policy-name ACT \
  --policy-config /path/to/deploy_policy.yml \
  --checkpoint /path/to/checkpoint \
  --device cuda:0
```

To add an algorithm, create `policy/<Name>/` with `__init__.py`, one
`deploy_policy.{yml,yaml,json}`, and an exported `create_policy(config)`.
Keeping `conda_env.yaml` and algorithm documentation beside the adapter makes
the policy reproducible without coupling it to Isaac Sim.
