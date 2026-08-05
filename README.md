# ExRoMa-Bench

**ExRoMa-Bench** is the **Extraterrestrial Rover Manipulation Benchmark**. It
provides mobile bimanual manipulation tasks in unstructured lunar, Martian, and
terrestrial analogue environments, with a shared interface for simulation data
collection, policy evaluation, and guarded real-robot execution.

## Repository Boundary

This repository contains code, configuration, tests, and documentation only.
Large USD, URDF, mesh, texture, and task-object assets live in the separate
[`ruilin.wang/ExRoMa-Assets`](https://modelscope.ai/datasets/ruilin.wang/ExRoMa-Assets)
ModelScope dataset. Datasets and checkpoints are also external. Until the
asset bundle passes `exroma assets verify`, use `exroma assets link` with a
complete local bundle.

ExRoMa does not require sibling `ref_*` repositories. Isaac Sim, Isaac Lab,
cuRobo, SimForge, and SimForge Foundry are normal third-party dependencies.
`procedural_moon` and `procedural_mars` invoke SimForge directly and cache the
generated USDZ output. Their baked `moon_surface` and `mars_surface` fallbacks
run without Blender.

## Tested Runtime

ExRoMa currently targets Linux with an NVIDIA CUDA-capable GPU. The checked
configuration is:

| Component | Version |
| --- | --- |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.0 |
| PyTorch / CUDA wheels | 2.7.0 / cu128 |
| CUDA toolkit for cuRobo | 12.8 |
| SimForge / Foundry | 0.2.4 / 0.3.1 |
| Blender | 4.5.3 LTS |

The pip installation of Isaac Sim requires a Linux system with GLIBC 2.35 or
newer (Ubuntu 22.04 or newer is recommended). Install a recent NVIDIA driver
before creating the environment.

## Install With Conda

The following installation is independent of OpenSpaceLab, SRB, RoboTwin, and
all `ref_*` checkouts.

### 1. Clone ExRoMa and create the environment

```bash
mkdir -p ~/robotics
cd ~/robotics
git clone https://github.com/RandyWangRaining/ExRoMa-Bench.git ExRoMa
cd ExRoMa

conda env create -f environment.yml
conda activate exroma
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install Isaac Sim and CUDA PyTorch

```bash
python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

python -m pip install --upgrade \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Confirm that Isaac Sim can start before continuing:

```bash
isaacsim
```

Close the Isaac Sim window after it opens.

### 3. Install Isaac Lab v2.3.0

```bash
cd ~/robotics
git clone --branch v2.3.0 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

### 4. Install cuRobo in the same environment

cuRobo compiles CUDA extensions during installation. `CUDA_HOME` must point to
the Conda toolkit, and `TORCH_CUDA_ARCH_LIST` must match the installed GPU.
The command below detects the architecture automatically.

```bash
conda activate exroma
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="$(python -c \
  'import torch; m, n = torch.cuda.get_device_capability(); print(f"{m}.{n}+PTX")')"

cd ~/robotics
git clone https://github.com/NVlabs/curobo.git
cd curobo
git checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
python -m pip install -e . --no-build-isolation
```

The pinned cuRobo revision is the revision used by the current ExRoMa task
controllers. For example, the detected architecture is `8.6+PTX` on an RTX
30-series GPU.

### 5. Make Blender available for new procedural terrains

Blender is a system tool rather than part of the Conda environment. Install a
Blender 4.x release and make its executable available on `PATH`:

```bash
blender --version
```

If Blender was unpacked manually, a user-local link is sufficient:

```bash
mkdir -p ~/.local/bin
ln -s /path/to/blender/blender ~/.local/bin/blender
export PATH="$HOME/.local/bin:$PATH"
```

Skip this step when using only the baked `moon_surface` and `mars_surface`
scenes.

### 6. Install ExRoMa

```bash
cd ~/robotics/ExRoMa
python -m pip install -r requirements/isaacsim-compatible.txt
python -m pip install --no-deps \
  "simforge==0.2.4" "simforge-foundry==0.3.1"
python -m pip install --no-deps -e .

export EXROMA_ISAACLAB_ROOT=~/robotics/IsaacLab
```

The two `--no-deps` flags are intentional. SimForge 0.2.4 and Foundry 0.3.1
publish `pytest-cov>=7` as a runtime dependency, which conflicts with Isaac
Sim 5.1.0's required `coverage==7.4.4`. The compatibility file installs the
actual ExRoMa runtime and test dependencies while preserving Isaac Sim's
strict Click and Coverage versions.

When Isaac Lab is stored somewhere else, set `EXROMA_ISAACLAB_ROOT` to that
checkout. Add the export to `~/.bashrc` if it should persist across terminals.

### 7. Restore and verify assets

Download the external bundle from ModelScope:

```bash
exroma assets pull
```

The former Hugging Face location remains available as an explicit fallback:

```bash
exroma assets pull \
  --provider huggingface \
  --repo-id wrl2003/ExRoMa-Assets
```

During local migration, select an existing bundle without copying it:

```bash
exroma assets link /path/to/ExRoMa-Assets
```

Then verify both the assets and the Python runtime:

```bash
exroma assets verify
exroma doctor
python -m pytest -q
```

The required checks from `python_3_11` through `foundry` should all report
`True`. Blender is used only to generate a new uncached SimForge terrain;
baked `moon_surface` and `mars_surface` scenes can run without it.

## Existing Environment

If an environment already contains the tested Isaac Sim, Isaac Lab, and
cuRobo stack, only install ExRoMa and select the assets:

```bash
cd ~/robotics/ExRoMa
conda activate YOUR_ENVIRONMENT
python -m pip install -r requirements/isaacsim-compatible.txt
python -m pip install --no-deps -e .

exroma assets link /path/to/ExRoMa-Assets
exroma doctor
```

## Quick Start

After installation, preview a task:

```bash
conda activate exroma
cd ~/robotics/ExRoMa
exroma preview --task stack_blocks_two --scene procedural_moon
```

Run exactly 100 randomized attempts, retain only successful episodes, and
write `collection_summary.json`:

```bash
exroma collect \
  --task stack_blocks_two \
  --scene procedural_moon \
  --attempts 100 \
  --seed 20000 \
  --strict-collision-check \
  --headless
```

Evaluate without recording demonstrations:

```bash
exroma evaluate \
  --task stack_blocks_two \
  --scene procedural_moon \
  --episodes 100 \
  --seed 30000 \
  --headless
```

The real-robot command is safety-gated and defaults to dry-run:

```bash
exroma real --config configs/real/dual_piper_rover.yaml --dry-run
```

See [Getting Started](docs/getting_started.md),
[Data Collection](docs/data_collection.md),
[Four-Task Collection Commands](docs/four_task_collection_commands.md),
[Publishing Assets](docs/publishing_assets.md),
[Terrain Pipeline](docs/terrain_pipeline.md), and
[Real-Robot Execution](docs/real_robot.md) for the full workflows.

## Supported Tasks

- `stack_blocks_two`
- `handover_block`
- `scan_object`
- `scan_rock`
- `beat_block_hammer`
- `test_tube_rack`

## Data Contract

Successful episodes use RoboTwin-compatible HDF5 paths and raw, unlabeled
camera JPEGs. Text overlays are applied only to optional MP4 previews. The
compact policy interface is three RGB views plus a 41-dimensional robot state,
with a 16-dimensional action containing 14 arm/gripper targets and two rover
commands.

`--attempts N` always means **N total attempted rollouts**. Failed rollouts are
counted in `collection_summary.json` and discarded from HDF5 output, so the
number of saved episodes is the number of successes and the success rate
remains measurable.

## Licensing

Project code is Apache-2.0. Assets retain their original licenses and are
documented in `THIRD_PARTY_NOTICES.md` and the external asset manifest.
