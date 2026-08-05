"""Task metadata shared by launchers, recorders, and documentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    name: str
    instruction: str
    family: str
    default_max_attempts_per_success: int


BENCHMARK_TASKS = {
    spec.name: spec
    for spec in (
        BenchmarkTaskSpec(
            "stack_blocks_two",
            "stack the two blocks",
            "multi-stage pick-and-place",
            5,
        ),
        BenchmarkTaskSpec(
            "handover_block",
            "handover the red block between arms and place it on the blue target",
            "bimanual transfer",
            5,
        ),
        BenchmarkTaskSpec(
            "scan_object",
            "hold the scanner and tea box with opposite arms and scan the box",
            "bimanual tool alignment",
            6,
        ),
        BenchmarkTaskSpec(
            "scan_rock",
            "pick up the scanner and aim its scan head at the lunar rock",
            "planetary inspection tool use",
            6,
        ),
        BenchmarkTaskSpec(
            "beat_block_hammer",
            "pick up the hammer and strike the red block",
            "tool use",
            5,
        ),
        BenchmarkTaskSpec(
            "test_tube_rack",
            "pick up the test tube and insert it into the highlighted rack slot",
            "precision insertion",
            5,
        ),
    )
}
