"""Command-specific deterministic defaults for ExRoMa simulation runs."""

DEFAULT_EVALUATION_EPISODES = 100
DEFAULT_EVALUATION_SEED = 100000
DEFAULT_NON_EVALUATION_SEED = 0


def resolve_task_seed(command: str, requested_seed: int | None) -> int:
    """Resolve an omitted task seed without changing collection compatibility."""

    if requested_seed is not None:
        return int(requested_seed)
    if command == "evaluate":
        return DEFAULT_EVALUATION_SEED
    return DEFAULT_NON_EVALUATION_SEED
