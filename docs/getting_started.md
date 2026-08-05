# Getting Started

ExRoMa-Bench is tested with Python 3.11, Isaac Sim 5.1.0, and Isaac Lab v2.3.0.
cuRobo must be importable from the same Python environment. A CUDA-capable GPU
is required for Isaac Sim and cuRobo planning. Blender must be available on
`PATH` when generating a procedural SimForge terrain for the first time.

## 1. Install the runtime and code

The complete from-scratch Conda installation, including Isaac Sim, Isaac Lab,
PyTorch, CUDA, and cuRobo, is maintained in the repository
[README](../README.md#install-with-conda). The short path below is only for an
environment that already contains those dependencies:

```bash
git clone https://github.com/RandyWangRaining/ExRoMa-Bench.git ExRoMa
cd ExRoMa
conda activate exroma
python -m pip install -r requirements/isaacsim-compatible.txt
python -m pip install --no-deps -e .
```

If Isaac Lab is not in `~/robotics/IsaacLab`, point ExRoMa at it:

```bash
export EXROMA_ISAACLAB_ROOT=/path/to/IsaacLab
```

## 2. Restore the external assets

After the ModelScope asset dataset is public:

```bash
exroma assets pull
```

To use an already downloaded bundle without copying it:

```bash
exroma assets link /path/to/ExRoMa-assets
```

The link is stored in the ignored `.asset-root` file. CI and containers can
instead set `EXROMA_ASSET_ROOT`. Check the complete runtime before launching:

```bash
exroma assets verify
exroma doctor
```

## 3. Start a scene

```bash
exroma preview --task stack_blocks_two --scene procedural_moon
```

Supported scenes are `ground_plane`, `lunalab`, `moon_surface`,
`mars_surface`, `procedural_moon`, `procedural_mars`, and
`oberpfaffenhofen`. Domain presets set gravity and lighting independently of
the selected terrain USD. The released moon-surface presets also compose a
collision-free NASA habitat model as a distant background asset.

`procedural_moon` and `procedural_mars` directly invoke SimForge Foundry. The
default terrain is 32 x 32 metres and uses terrain seed zero:

```bash
exroma preview \
  --task stack_blocks_two \
  --scene procedural_moon \
  --terrain-size 32 \
  --terrain-seed 0
```

The first uncached seed starts Blender and can take substantially longer than
later launches. `moon_surface` and `mars_surface` load released baked USDZ
snapshots when Blender is unavailable.

The repository also includes `./exroma.sh`, which injects `source/` into
`PYTHONPATH`; it is useful before the editable package has been installed.

## Runtime boundary

No `ref_*` checkout and no `srb` package is imported at runtime. SimForge is
called through its public Isaac Lab integration for procedural scenes; the
baked scene names remain available for machines that should not run Blender.
