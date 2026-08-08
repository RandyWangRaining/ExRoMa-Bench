from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from exroma_bench.policy.plugins import (
    discover_policy_plugins,
    load_config_file,
    load_named_policy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_builtin_algorithm_workspaces_are_discoverable() -> None:
    plugins = discover_policy_plugins(PROJECT_ROOT / "policy")

    assert tuple(plugins) == ("ACT", "pi05", "replay", "smolvla")
    assert plugins["ACT"].config_path.name == "deploy_policy.yml"
    assert plugins["pi05"].config_path.name == "deploy_policy.yml"
    assert plugins["replay"].config_path.name == "deploy_policy.yml"
    assert plugins["smolvla"].config_path.name == "deploy_policy.yml"


def test_config_loader_accepts_yaml_and_json(tmp_path) -> None:
    yaml_path = tmp_path / "policy.yml"
    json_path = tmp_path / "policy.json"
    yaml_path.write_text("action_dim: 16\ndevice: cuda:0\n", encoding="utf-8")
    json_path.write_text('{"action_dim": 16, "device": "cpu"}', encoding="utf-8")

    assert load_config_file(yaml_path) == {"action_dim": 16, "device": "cuda:0"}
    assert load_config_file(json_path) == {"action_dim": 16, "device": "cpu"}


def test_named_policy_loads_config_and_overrides(tmp_path) -> None:
    plugin_dir = tmp_path / "Demo"
    plugin_dir.mkdir()
    plugin_dir.joinpath("deploy_policy.yml").write_text(
        "state_dim: 16\naction_dim: 16\ndevice: cpu\n", encoding="utf-8"
    )
    plugin_dir.joinpath("__init__.py").write_text(
        """
import numpy as np

class DemoPolicy:
    name = "demo"
    def __init__(self, config):
        self.config = config
    def infer(self, observation):
        return np.zeros((2, 16), dtype=np.float32)

def create_policy(config):
    return DemoPolicy(config)
""".lstrip(),
        encoding="utf-8",
    )

    policy = load_named_policy("Demo", tmp_path, overrides={"device": "cuda:1"})

    assert policy.config["device"] == "cuda:1"
    assert policy.config["policy_name"] == "Demo"
    assert policy.config["plugin_dir"] == str(plugin_dir)
    assert policy.infer({}).shape == (2, 16)


def test_named_policy_rejects_non_exroma_action_dimension(tmp_path) -> None:
    plugin_dir = tmp_path / "Legacy"
    plugin_dir.mkdir()
    plugin_dir.joinpath("__init__.py").write_text(
        "def create_policy(config): return lambda observation: [0] * 14\n",
        encoding="utf-8",
    )
    plugin_dir.joinpath("deploy_policy.yml").write_text(
        "state_dim: 14\naction_dim: 14\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="requires 16"):
        load_named_policy("Legacy", tmp_path)


def test_named_replay_policy_returns_recorded_16d_actions(tmp_path) -> None:
    episode_path = tmp_path / "episode_000000.hdf5"
    joint_actions = np.arange(56, dtype=np.float32).reshape(4, 14)
    base_actions = np.asarray([[0.1, 0.0], [0.2, 0.1], [0.3, -0.1], [0.0, 0.0]], dtype=np.float32)
    with h5py.File(episode_path, "w") as episode:
        episode.create_dataset("actions/joint_position_target", data=joint_actions)
        episode.create_dataset("actions/base_velocity", data=base_actions)

    policy = load_named_policy(
        "replay",
        PROJECT_ROOT / "policy",
        overrides={"episode_path": str(episode_path), "action_horizon": 3},
    )
    result = policy.infer({})

    assert policy.name == "replay"
    assert result["actions"].shape == (3, 16)
    np.testing.assert_allclose(result["actions"][:, :14], joint_actions[:3])
    np.testing.assert_allclose(result["actions"][:, 14:], base_actions[:3])


def test_act_adapter_encodes_three_views_and_16d_state(tmp_path, monkeypatch) -> None:
    backend_module = tmp_path / "stub_act_backend.py"
    backend_module.write_text(
        """
import numpy as np

class Backend:
    def __init__(self, config):
        self.config = config
        self.last_observation = None
    def get_action(self, observation):
        self.last_observation = observation
        return np.zeros((3, 16), dtype=np.float32)

def create_model(config):
    return Backend(config)
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    policy = load_named_policy(
        "ACT",
        PROJECT_ROOT / "policy",
        overrides={
            "backend_factory": "stub_act_backend:create_model",
            "image_width": 32,
            "image_height": 24,
        },
    )
    observation = {
        "state": np.zeros(16, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((12, 16, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((12, 16, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((12, 16, 3), dtype=np.uint8),
        },
        "prompt": "stack two blocks",
        "timestamp": 1.5,
    }

    actions = policy.infer(observation)

    assert actions.shape == (3, 16)
    assert policy.backend.last_observation["head_cam"].shape == (3, 24, 32)
    assert policy.backend.last_observation["qpos"].shape == (16,)
    assert policy.backend.last_observation["prompt"] == "stack two blocks"


def test_pi05_adapter_supports_openpi_window_lifecycle(tmp_path, monkeypatch) -> None:
    backend_module = tmp_path / "stub_pi05_backend.py"
    backend_module.write_text(
        """
import numpy as np

class Backend:
    def __init__(self, config):
        self.observation_window = None
        self.language = None
        self.window_input = None
        self.reset_called = False
    def set_language(self, prompt):
        self.language = prompt
    def update_observation_window(self, images, state):
        self.window_input = (images, state)
        self.observation_window = True
    def get_action(self):
        return np.zeros((60, 16), dtype=np.float32)
    def reset_obsrvationwindows(self):
        self.reset_called = True
        self.observation_window = None

def create_model(config):
    return Backend(config)
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    policy = load_named_policy(
        "pi05",
        PROJECT_ROOT / "policy",
        overrides={
            "backend_factory": "stub_pi05_backend:create_model",
            "action_steps": 12,
        },
    )
    observation = {
        "state": np.zeros(16, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((12, 16, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((12, 16, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((12, 16, 3), dtype=np.uint8),
        },
        "prompt": "handover the block",
    }

    actions = policy.infer(observation)
    policy.reset()

    assert actions.shape == (12, 16)
    assert policy.backend.language == "handover the block"
    assert len(policy.backend.window_input[0]) == 3
    assert policy.backend.window_input[1].shape == (16,)
    assert policy.backend.reset_called is True


def test_smolvla_adapter_uses_lerobot_feature_keys(tmp_path, monkeypatch) -> None:
    backend_module = tmp_path / "stub_smolvla_backend.py"
    backend_module.write_text(
        """
import numpy as np

class Backend:
    def __init__(self, config):
        self.last_observation = None
        self.reset_called = False
    def infer(self, observation):
        self.last_observation = observation
        return np.zeros((60, 16), dtype=np.float32)
    def reset(self):
        self.reset_called = True

def create_model(config):
    return Backend(config)
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    policy = load_named_policy(
        "smolvla",
        PROJECT_ROOT / "policy",
        overrides={
            "backend_factory": "stub_smolvla_backend:create_model",
            "action_steps": 50,
        },
    )
    observation = {
        "state": np.zeros(16, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((12, 16, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((12, 16, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((12, 16, 3), dtype=np.uint8),
        },
        "prompt": "scan the lunar rock",
        "timestamp": 2.5,
    }

    actions = policy.infer(observation)
    policy.reset()

    assert actions.shape == (50, 16)
    assert policy.backend.last_observation["observation.state"].shape == (16,)
    assert policy.backend.last_observation["observation.images.cam_high"].shape == (12, 16, 3)
    assert policy.backend.last_observation["task"] == "scan the lunar rock"
    assert policy.backend.last_observation["robot_type"] == "dual_piper_rover"
    assert policy.backend.reset_called is True
