# pi0.5 policy workspace

This directory is the ExRoMa deployment boundary for pi0.5/OpenPI. Keep the
large JAX/OpenPI dependency stack in this policy environment, outside the
Isaac Sim environment.

## uv installation

Install the official OpenPI source and its Python 3.11 environment inside this
workspace:

```bash
git clone --recurse-submodules \
  https://github.com/Physical-Intelligence/openpi.git \
  policy/pi05/openpi

cd policy/pi05/openpi
GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --python 3.11 --no-dev
uv pip install --python .venv/bin/python --no-deps --editable ../../..
uv pip install --python .venv/bin/python 'websockets==12.0'
```

Activate the installed policy environment from the repository root with:

```bash
source policy/pi05/openpi/.venv/bin/activate
```

This environment is only for OpenPI inference and training. Isaac Sim runs in
its own environment and communicates with the policy through WebSocket.

## Conda alternative

1. Create the bridge environment and install ExRoMa's protocol package:

   ```bash
   conda env create -f policy/pi05/conda_env.yaml
   conda activate exroma-pi05
   python -m pip install --no-deps -e .
   ```

2. Install OpenPI from its official source using versions compatible with the
   server GPU.

3. Provide a `module:create_model` backend factory in a deployment YAML. A
   native OpenPI-style backend may expose `set_language`,
   `update_observation_window`, `get_action`, and a reset method. A simple
   callable or `infer(observation)` backend is also accepted.

4. Start the algorithm server:

   ```bash
   python scripts/serve_policy.py \
     --policy-name pi05 \
     --policy-config /path/to/pi05_deploy.yml \
     --checkpoint /path/to/checkpoint \
     --port 8000
   ```

Images are passed in `[head, right_wrist, left_wrist]` order. The backend
receives the 16-D ExRoMa state and language prompt and must return `[16]` or
`[T,16]` actions.
