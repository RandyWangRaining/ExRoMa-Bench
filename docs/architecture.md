# ExRoMa Architecture

ExRoMa separates four concerns:

1. `sim`: Isaac Lab-native scene construction, rover motion, and physics.
2. `tasks`: randomized task assets, cuRobo experts, and success predicates.
3. `recording`: synchronized raw camera, proprioception, action, and object data.
4. `real`: the same compact policy contract behind a safety-gated hardware adapter.

No sibling reference repository or `srb` package is imported at runtime.
Procedural terrain is generated through the public SimForge Isaac Lab spawner
and cached as USDZ. Released baked terrain remains available as a fallback.

## Policy Contract

Observations combine three RGB views with 14 joint positions, 14 joint
velocities, a seven-value base pose, and six base velocity values. Actions are
14 joint position targets plus normalized linear and angular rover commands.

## Asset Contract

Code resolves assets from `EXROMA_ASSET_ROOT`, `.asset-root`, or
`~/.cache/exroma/assets`, in that order. `assets/manifest.json` lists required
runtime paths. Asset publication remains separate from GitHub source history.

## Real-Robot Boundary

The simulation task controller is never imported by a hardware driver. A
trained policy consumes `RobotObservation` and emits `RobotAction`; the real
runner clamps every command through a configured safety envelope before a
site-specific adapter can transmit it. Real motion requires both an environment
gate and an explicit command-line confirmation.
