from exroma_bench.sim.defaults import (
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_EVALUATION_SEED,
    resolve_task_seed,
)


def test_evaluation_defaults_to_100_episodes_from_seed_100000() -> None:
    assert DEFAULT_EVALUATION_EPISODES == 100
    assert DEFAULT_EVALUATION_SEED == 100000
    assert resolve_task_seed("evaluate", None) == 100000


def test_non_evaluation_commands_keep_seed_zero() -> None:
    assert resolve_task_seed("collect", None) == 0
    assert resolve_task_seed("preview", None) == 0


def test_explicit_seed_always_wins() -> None:
    assert resolve_task_seed("evaluate", 20000) == 20000
    assert resolve_task_seed("collect", 30000) == 30000
