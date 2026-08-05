from __future__ import annotations

import math
import random

from exroma_bench.tasks.benchmark_suite.robotwin_sampling import (
    sample_blocks_stack_easy,
)


def test_blocks_stack_easy_sampling_is_deterministic_and_valid():
    first_rng = random.Random(17)
    second_rng = random.Random(17)

    first_sequence = [sample_blocks_stack_easy(first_rng) for _ in range(1000)]
    second_sequence = [sample_blocks_stack_easy(second_rng) for _ in range(1000)]

    assert first_sequence == second_sequence
    for sample in first_sequence:
        red = sample["red"]
        black = sample["black"]
        for pose in (red, black):
            assert -0.25 <= pose.x <= 0.25
            assert -0.15 <= pose.y <= 0.05
            assert abs(pose.x) >= 0.05
            assert math.hypot(pose.x, pose.y + 0.10) >= 0.15
            assert -1.57 <= pose.yaw <= 1.57
            if abs(pose.x) < 0.15:
                assert pose.y <= 0.0
        assert math.hypot(red.x - black.x, red.y - black.y) >= 0.10
