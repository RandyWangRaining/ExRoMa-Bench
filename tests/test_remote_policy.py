from __future__ import annotations

import threading

import h5py
import numpy as np
import pytest
from exroma_bench.policy.client import RemotePolicyClient
from exroma_bench.policy.protocol import (
    CAMERA_KEYS,
    PolicyProtocolError,
    decode_infer_request,
    encode_infer_request,
    validate_actions,
)
from exroma_bench.policy.server import (
    Hdf5ReplayPolicy,
    HoldPositionPolicy,
    RemotePolicyServer,
)
from exroma_bench.policy.sim_codec import body_planar_velocity_wxyz


def _images() -> dict[str, np.ndarray]:
    return {
        key: np.full((24, 32, 3), fill_value=40 * index, dtype=np.uint8)
        for index, key in enumerate(CAMERA_KEYS, start=1)
    }


def test_infer_request_round_trip_preserves_contract() -> None:
    state = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    frame = encode_infer_request(
        request_id=7,
        state=state,
        images=_images(),
        prompt="stack the two blocks",
        timestamp=1.25,
        jpeg_quality=90,
    )

    decoded = decode_infer_request(frame)

    assert decoded["request_id"] == 7
    np.testing.assert_allclose(decoded["state"], state)
    assert decoded["prompt"] == "stack the two blocks"
    assert decoded["timestamp"] == 1.25
    assert set(decoded["images"]) == set(CAMERA_KEYS)
    assert all(image.shape == (24, 32, 3) for image in decoded["images"].values())


def test_action_validation_requires_exactly_16_dimensions() -> None:
    assert validate_actions(np.zeros(16)).shape == (1, 16)
    assert validate_actions(np.zeros((4, 16))).shape == (4, 16)
    with pytest.raises(PolicyProtocolError, match="Expected action"):
        validate_actions(np.zeros(14))
    with pytest.raises(PolicyProtocolError, match="NaN"):
        validate_actions(np.full(16, np.nan))


def test_websocket_client_server_returns_16d_hold_action() -> None:
    policy_server = RemotePolicyServer(HoldPositionPolicy(), host="127.0.0.1", port=0)
    websocket_server = policy_server.create_server()
    port = websocket_server.socket.getsockname()[1]
    thread = threading.Thread(target=websocket_server.serve_forever, daemon=True)
    thread.start()
    state = np.linspace(-0.7, 0.8, 16, dtype=np.float32)
    try:
        with RemotePolicyClient(
            "127.0.0.1",
            port,
            connect_timeout=2.0,
            response_timeout=2.0,
        ) as client:
            client.reset()
            actions, timing = client.infer(
                state=state,
                images=_images(),
                prompt="hold",
                timestamp=0.0,
            )
            assert client.metadata["state_dim"] == 16
            assert client.metadata["action_dim"] == 16
            assert len(client.metadata["state_names"]) == 16
            assert len(client.metadata["action_names"]) == 16
            assert actions.shape == (1, 16)
            np.testing.assert_allclose(actions[0, :14], state[:14])
            np.testing.assert_allclose(actions[0, 14:], 0.0)
            assert timing["infer_ms"] >= 0.0
    finally:
        websocket_server.shutdown()
        thread.join(timeout=2.0)


def test_body_planar_velocity_is_expressed_in_rover_frame() -> None:
    identity = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    velocity = np.asarray([1.5, 0.0, 0.0, 0.0, 0.0, -0.4])
    np.testing.assert_allclose(
        body_planar_velocity_wxyz(identity, velocity),
        np.asarray([1.5, -0.4]),
        atol=1e-6,
    )

    half_sqrt_two = np.sqrt(0.5)
    yaw_90 = np.asarray([0.0, 0.0, 0.0, half_sqrt_two, 0.0, 0.0, half_sqrt_two])
    world_y_velocity = np.asarray([0.0, 2.0, 0.0, 0.0, 0.0, 0.3])
    np.testing.assert_allclose(
        body_planar_velocity_wxyz(yaw_90, world_y_velocity),
        np.asarray([2.0, 0.3]),
        atol=1e-6,
    )


def test_hdf5_replay_policy_returns_and_resets_action_chunks(tmp_path) -> None:
    episode_path = tmp_path / "episode.hdf5"
    joint_actions = np.arange(42, dtype=np.float32).reshape(3, 14)
    base_actions = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    with h5py.File(episode_path, "w") as episode:
        episode.create_dataset("actions/joint_position_target", data=joint_actions)
        episode.create_dataset("actions/base_velocity", data=base_actions)

    policy = Hdf5ReplayPolicy(episode_path, action_horizon=2)
    first_result = policy.infer({})
    second_result = policy.infer({})
    exhausted_result = policy.infer({})
    first = first_result["actions"]
    second = second_result["actions"]
    exhausted = exhausted_result["actions"]

    assert first.shape == (2, 16)
    np.testing.assert_allclose(first[:, :14], joint_actions[:2])
    np.testing.assert_allclose(first[:, 14:], base_actions[:2])
    np.testing.assert_allclose(second[0, :14], joint_actions[2])
    assert second_result["finished"] is True
    np.testing.assert_allclose(exhausted[0, :14], joint_actions[-1])
    np.testing.assert_allclose(exhausted[0, 14:], 0.0)
    policy.reset()
    np.testing.assert_allclose(policy.infer({})["actions"], first)
