"""Discovery and loading for self-contained algorithm policy workspaces."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .protocol import ACTION_DIM, STATE_DIM

POLICY_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
CONFIG_FILENAMES = ("deploy_policy.yml", "deploy_policy.yaml", "deploy_policy.json")


@dataclass(frozen=True)
class PolicyPlugin:
    """Filesystem metadata for one algorithm-specific policy package."""

    name: str
    directory: Path
    config_path: Path


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a YAML or JSON policy configuration object."""

    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        config = json.loads(text)
    else:
        config = yaml.safe_load(text)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"Policy config {config_path} must contain a mapping.")
    return dict(config)


def discover_policy_plugins(policy_root: str | Path) -> dict[str, PolicyPlugin]:
    """Discover algorithm directories with an entry point and deploy config."""

    root = Path(policy_root).expanduser().resolve()
    if not root.is_dir():
        return {}
    plugins: dict[str, PolicyPlugin] = {}
    for directory in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if not directory.is_dir() or not POLICY_NAME_PATTERN.fullmatch(directory.name):
            continue
        if not directory.joinpath("__init__.py").is_file():
            continue
        config_path = next(
            (
                directory / filename
                for filename in CONFIG_FILENAMES
                if (directory / filename).is_file()
            ),
            None,
        )
        if config_path is None:
            continue
        plugins[directory.name] = PolicyPlugin(directory.name, directory, config_path)
    return plugins


def _load_plugin_module(plugin: PolicyPlugin) -> Any:
    digest = hashlib.sha1(str(plugin.directory).encode("utf-8")).hexdigest()[:12]
    module_name = f"_exroma_policy_plugin_{plugin.name}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        plugin.directory / "__init__.py",
        submodule_search_locations=[str(plugin.directory)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for policy {plugin.name!r}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_named_policy(
    name: str,
    policy_root: str | Path,
    *,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Load one named policy package and call its ``create_policy`` entry point."""

    plugins = discover_policy_plugins(policy_root)
    if name not in plugins:
        available = ", ".join(plugins) or "none"
        raise ValueError(f"Unknown policy {name!r}. Available policies: {available}.")
    plugin = plugins[name]
    config = load_config_file(config_path or plugin.config_path)
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    config.setdefault("policy_name", name)
    config["plugin_dir"] = str(plugin.directory)

    for key, expected in (("state_dim", STATE_DIM), ("action_dim", ACTION_DIM)):
        if key in config and int(config[key]) != expected:
            raise ValueError(f"{name} config {key}={config[key]}, but ExRoMa requires {expected}.")

    module = _load_plugin_module(plugin)
    factory = getattr(module, "create_policy", None)
    if not callable(factory):
        raise TypeError(f"Policy {name!r} must export callable create_policy(config).")
    policy = factory(config)
    if not callable(getattr(policy, "infer", None)) and not callable(policy):
        raise TypeError(f"Policy {name!r} must be callable or expose infer(observation).")
    return policy
