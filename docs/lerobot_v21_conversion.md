# LeRobot v2.1 Conversion

ExRoMa can convert successful HDF5 demonstrations into the episode-based
LeRobot v2.1 layout. Numeric policy data is stored in Parquet and each camera
stream is stored as a separate H.264 MP4. Camera frames are read directly from
the raw JPEG datasets, so labels from preview videos are never embedded in the
training images.

## Policy Interface

Both policy vectors have 16 dimensions:

| Feature | Dimensions | Contents |
|---|---:|---|
| `observation.state` | 16 | 14 compact dual-arm joint positions, body-frame rover forward velocity, and rover yaw velocity |
| `action` | 16 | 14 compact dual-arm joint-position targets, rover linear-velocity command, and rover angular-velocity command |

The 14 arm dimensions contain six joints and one merged gripper value for each
PiPER arm. Mast joints, joint velocities, rover pose, and lateral or vertical
base velocities are intentionally excluded. The observed rover velocity is
rotated from the Isaac world frame into the rover body frame before export.

## Convert One Episode

Run from the ExRoMa repository root in the `openspace` environment:

```bash
conda activate openspace

python scripts/convert_hdf5_to_lerobot_v21.py \
  datasets/stack_blocks_two_procedural_moon_seed20000_run100/episode_000000.hdf5 \
  datasets/lerobot_v21/stack_blocks_two_one_episode \
  --overwrite
```

## Convert A Dataset

Convert all complete successful episodes in a collection directory:

```bash
python scripts/convert_hdf5_to_lerobot_v21.py \
  datasets/stack_blocks_two_procedural_moon_seed20000_run100 \
  datasets/lerobot_v21/stack_blocks_two_procedural_moon \
  --overwrite
```

Use `--limit 50` to convert only the first 50 successful episodes. Failed
episodes are skipped by default; pass `--include-failed` only when failure data
is deliberately required.

## Output Layout

```text
<output>/
├── data/chunk-000/episode_000000.parquet
├── meta/info.json
├── meta/tasks.jsonl
├── meta/episodes.jsonl
├── meta/episodes_stats.jsonl
└── videos/chunk-000/
    ├── observation.images.cam_high/episode_000000.mp4
    ├── observation.images.cam_left_wrist/episode_000000.mp4
    └── observation.images.cam_right_wrist/episode_000000.mp4
```

The source HDF5 files are not modified. Conversion is first written into a
temporary directory and moved to the requested output only after every episode
and video has completed successfully.
