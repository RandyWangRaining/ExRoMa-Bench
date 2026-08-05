"""Dual-PiPER benchmark registry and task factories."""

from .registry import BENCHMARK_TASKS, BenchmarkTaskSpec
from .prompts import sample_task_prompt, write_task_prompt_file


def __getattr__(name: str):
    """Load Isaac Sim-backed task factories only when a simulator uses them."""

    if name in {
        "add_benchmark_task_assets",
        "apply_benchmark_materials",
        "benchmark_record_objects",
    }:
        from . import scene

        return getattr(scene, name)
    if name == "create_benchmark_controller":
        from .sequential_controller import create_benchmark_controller

        return create_benchmark_controller
    raise AttributeError(name)

__all__ = [
    "BENCHMARK_TASKS",
    "BenchmarkTaskSpec",
    "add_benchmark_task_assets",
    "apply_benchmark_materials",
    "benchmark_record_objects",
    "create_benchmark_controller",
    "sample_task_prompt",
    "write_task_prompt_file",
]
