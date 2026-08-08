# ACT policy workspace

This directory owns ACT-specific deployment configuration and dependencies.
It intentionally does not vendor a third-party ACT implementation or model
checkpoint.

1. Create the lightweight server environment:

   ```bash
   conda env create -f policy/ACT/conda_env.yaml
   conda activate exroma-act
   python -m pip install --no-deps -e .
   ```

2. Install the ACT implementation and a PyTorch build compatible with the
   server GPU.

3. Expose a model factory such as `my_act_backend:create_model`. The factory
   receives `deploy_policy.yml` as a dictionary and returns a callable, or an
   object with `infer(encoded_observation)` or `get_action(encoded_observation)`.

4. Set `backend_factory` and `checkpoint_dir` in a private copy of the YAML,
   then run:

   ```bash
   python scripts/serve_policy.py \
     --policy-name ACT \
     --policy-config /path/to/act_deploy.yml \
     --checkpoint /path/to/checkpoint \
     --port 8000
   ```

The adapter supplies `head_cam`, `left_cam`, and `right_cam` as normalized CHW
arrays, `qpos` as the 16-D ExRoMa state, and `prompt`. The model must return one
16-D action or a `[T,16]` action chunk.
