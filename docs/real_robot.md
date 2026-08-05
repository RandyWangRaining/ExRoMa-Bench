# Real-Robot Execution

ExRoMa keeps simulation collection and real hardware communication behind the
same 41-dimensional observation and 16-dimensional action contract. The
repository provides the interfaces, command validation, and safety envelope;
it intentionally does not ship a machine-specific CAN, serial, or rover motor
driver.

Validate the current configuration without sending commands:

```bash
exroma real --config configs/real/dual_piper_rover.yaml --dry-run
```

Before enabling hardware, implement
`exroma_bench.real.interfaces.HardwareAdapter` for the actual dual PiPER and
rover controllers, register it in the deployment configuration, calibrate all
joint signs and limits, and verify the physical emergency stop.

Real motion is guarded by all of the following:

1. A valid checkpoint must be provided.
2. `EXROMA_REAL_ROBOT_ENABLED=1` must be set deliberately.
3. The command must include `--no-dry-run --confirm-hardware`.
4. The site adapter must clamp joint and base commands through the configured safety limits.

Until a site adapter is installed, the final guarded path raises
`NotImplementedError`; this is deliberate and prevents a simulation-only
checkout from transmitting unverified commands.
