# Replay policy workspace

The replay policy reads one successful ExRoMa HDF5 episode and returns its
recorded compact actions through the same remote-policy interface used by ACT
and pi0.5. Every action contains 14 dual-arm/gripper targets and two rover
commands.

Start the server:

```bash
cd /home/ruilin/ExRoMa
conda activate openspace

python scripts/serve_policy.py \
  --policy-name replay \
  --replay-episode datasets/stack_blocks_two_procedural_moon_seed20000_run100/episode_000000.hdf5 \
  --replay-action-horizon 8 \
  --host 127.0.0.1 \
  --port 8000
```

Then run the matching task, scene, task seed, and terrain seed from another
terminal. `--record-policy-video` records the three-view MP4, and the output
directory also receives the replayed HDF5 trajectory and summary:

```bash
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

For meaningful physical replay, the task, random seed, terrain seed, robot
initial state, and control frequency must match the source episode.
