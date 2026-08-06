from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "convert_hdf5_to_lerobot_v21.py"
SPEC = importlib.util.spec_from_file_location("convert_hdf5_to_lerobot_v21", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def test_body_planar_velocity_uses_robot_frame() -> None:
    half_sqrt_two = np.sqrt(0.5)
    poses = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, half_sqrt_two, 0, 0, half_sqrt_two],
        ],
        dtype=np.float32,
    )
    world_velocities = np.array(
        [
            [1, 0, 0, 0, 0, 0.2],
            [0, 1, 0, 0, 0, 0.3],
        ],
        dtype=np.float32,
    )

    actual = converter.body_planar_velocity(poses, world_velocities)

    np.testing.assert_allclose(actual, [[1.0, 0.2], [1.0, 0.3]], atol=1e-6)


def test_compact_policy_contract_is_16_state_and_16_action(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_000000.hdf5"
    joint_names = [f"joint_{index}" for index in range(14)]
    action_names = [f"joint_target_{index}" for index in range(14)] + [
        "base_command/linear",
        "base_command/angular",
    ]
    with h5py.File(episode_path, "w") as episode:
        episode.attrs["joint_names_json"] = json.dumps(joint_names)
        episode.attrs["action_names_json"] = json.dumps(action_names)
        episode.create_dataset("observations/qpos", data=np.ones((2, 14), dtype=np.float32))
        episode.create_dataset(
            "observations/base_pose",
            data=np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (2, 1)),
        )
        episode.create_dataset(
            "observations/base_velocity",
            data=np.array([[0.4, 0, 0, 0, 0, 0.1], [0.5, 0, 0, 0, 0, 0.2]], dtype=np.float32),
        )
        episode.create_dataset(
            "actions/joint_position_target", data=np.zeros((2, 14), dtype=np.float32)
        )
        episode.create_dataset(
            "actions/base_velocity", data=np.array([[0.6, 0.1], [0.7, 0.2]], dtype=np.float32)
        )

    with h5py.File(episode_path, "r") as episode:
        state, state_names = converter.compact_state(episode)
        action, actual_action_names = converter.compact_action(episode)

    assert state.shape == (2, 16)
    assert action.shape == (2, 16)
    assert state_names == [*joint_names, *converter.BASE_STATE_NAMES]
    assert actual_action_names == action_names
    np.testing.assert_allclose(state[:, -2:], [[0.4, 0.1], [0.5, 0.2]])
    np.testing.assert_allclose(action[:, -2:], [[0.6, 0.1], [0.7, 0.2]])
