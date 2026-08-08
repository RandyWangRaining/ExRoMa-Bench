"""Small helpers shared by algorithm-specific policy adapters."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def import_callable(spec: str) -> Callable[..., Any]:
    """Import a callable written as ``module:attribute``."""

    if ":" not in spec:
        raise ValueError(f"Expected module:callable, got {spec!r}.")
    module_name, attribute_name = spec.rsplit(":", 1)
    value = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(value):
        raise TypeError(f"{spec!r} does not resolve to a callable.")
    return value


def create_backend(config: Mapping[str, Any]) -> Any:
    """Instantiate the model backend configured by an algorithm workspace."""

    spec = config.get("backend_factory")
    if not spec:
        policy_name = config.get("policy_name", "policy")
        raise ValueError(
            f"{policy_name} has no backend_factory. Set backend_factory to "
            "'your_module:create_model' in deploy_policy.yml."
        )
    return import_callable(str(spec))(dict(config))


def invoke_backend(backend: Any, observation: dict[str, Any]) -> Any:
    """Call a conventional policy backend without imposing a framework."""

    infer = getattr(backend, "infer", None)
    if callable(infer):
        return infer(observation)
    get_action = getattr(backend, "get_action", None)
    if callable(get_action):
        return get_action(observation)
    if callable(backend):
        return backend(observation)
    raise TypeError("Policy backend must be callable or expose infer/get_action.")


def reset_backend(backend: Any) -> None:
    """Reset common temporal-policy state APIs, including OpenPI variants."""

    for method_name in (
        "reset",
        "reset_model",
        "reset_observation_windows",
        "reset_obsrvationwindows",
    ):
        method = getattr(backend, method_name, None)
        if callable(method):
            method()
            return


def to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy, Torch, or JAX-like arrays without importing their runtimes."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.asarray(value, dtype=np.float32)


def normalize_backend_result(result: Any, *, max_actions: int | None = None) -> Any:
    """Normalize model output while preserving optional policy status fields."""

    if isinstance(result, dict):
        key = "actions" if "actions" in result else "action"
        if key not in result:
            raise ValueError("Policy backend result must contain 'actions' or 'action'.")
        normalized = dict(result)
        actions = to_numpy(result[key])
        if max_actions is not None:
            actions = actions[:max_actions] if actions.ndim == 2 else actions
        normalized.pop("action", None)
        normalized["actions"] = actions
        return normalized
    actions = to_numpy(result)
    if max_actions is not None and actions.ndim == 2:
        actions = actions[:max_actions]
    return actions
