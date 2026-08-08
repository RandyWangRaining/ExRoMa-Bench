# SmolVLA policy workspace

This workspace runs Hugging Face LeRobot SmolVLA as a separate policy server.
SmolVLA consumes multiple camera views, robot state, and a natural-language
instruction, then predicts an action chunk. ExRoMa maps those features to three
RGB cameras, the compact 16-D rover-manipulator state, and `[T,16]` actions.

## Environment

LeRobot 0.6 uses Python 3.12, while the Isaac Sim environment uses Python 3.11.
Keep them separate:

```bash
cd /home/ruilin/ExRoMa
conda env create -f policy/smolvla/conda_env.yaml
conda activate exroma-smolvla
```

There is no need to install Isaac Sim or ExRoMa as a package in this policy
environment. `scripts/serve_policy.py` adds `source/` to `PYTHONPATH` itself.

## Fine-tuning requirement

Use the converted LeRobot v2.1 ExRoMa dataset to fine-tune
`lerobot/smolvla_base`. The resulting checkpoint must retain these feature
names and dimensions:

```text
observation.state                      [16]
observation.images.cam_high            [3,480,640]
observation.images.cam_left_wrist      [3,480,640]
observation.images.cam_right_wrist     [3,480,640]
action                                 [16]
```

Example training command after publishing the dataset to a LeRobot-compatible
repository:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<USER>/<EXROMA_DATASET> \
  --batch_size=16 \
  --steps=20000 \
  --output_dir=outputs/train/exroma_smolvla \
  --job_name=exroma_smolvla \
  --policy.device=cuda \
  --wandb.enable=false
```

## Policy server

Start inference with the fine-tuned `pretrained_model` directory:

```bash
cd /home/ruilin/ExRoMa
conda activate exroma-smolvla

python scripts/serve_policy.py \
  --policy-name smolvla \
  --checkpoint outputs/train/exroma_smolvla/checkpoints/last/pretrained_model \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8000
```

Run `./exroma.sh evaluate` from the separate `openspace` terminal exactly as
for ACT or replay. The default server response contains up to 50 consecutive
16-D actions. The simulator may consume fewer using
`--policy-action-horizon`.

Official references:

- https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx
- https://github.com/huggingface/lerobot/blob/main/examples/tutorial/smolvla/using_smolvla_example.py
