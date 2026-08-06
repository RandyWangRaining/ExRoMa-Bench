"""Pluggable WebSocket policy server for isolated GPU inference."""

from __future__ import annotations

import importlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, WebSocketServer, serve

from .protocol import (
    ACTION_DIM,
    PROTOCOL_VERSION,
    decode_infer_request,
    decode_message,
    encode_action_response,
    encode_message,
    server_metadata,
    validate_actions,
)

LOGGER = logging.getLogger(__name__)


class HoldPositionPolicy:
    """Safe protocol test policy: hold 14 joints and command a stopped rover."""

    name = "hold_position"

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[:14] = observation["state"][:14]
        return action

    def reset(self) -> None:
        return None


class Hdf5ReplayPolicy:
    """Serve a recorded compact ExRoMa action trajectory in fixed-size chunks."""

    def __init__(self, episode_path: str | Path, *, action_horizon: int = 8) -> None:
        self.episode_path = Path(episode_path).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("Replay action horizon must be positive.")
        with h5py.File(self.episode_path, "r") as episode:
            joint_actions = np.asarray(episode["actions/joint_position_target"], dtype=np.float32)
            base_actions = np.asarray(episode["actions/base_velocity"], dtype=np.float32)
        if joint_actions.ndim != 2 or joint_actions.shape[1] != 14:
            raise ValueError(f"Expected replay joint actions [T,14], got {joint_actions.shape}.")
        if base_actions.shape != (len(joint_actions), 2):
            raise ValueError(f"Expected replay base actions [T,2], got {base_actions.shape}.")
        self.actions = np.ascontiguousarray(
            np.concatenate((joint_actions, base_actions), axis=1), dtype=np.float32
        )
        self.name = f"hdf5_replay:{self.episode_path.name}"
        self._index = 0

    @property
    def frame_count(self) -> int:
        return len(self.actions)

    def infer(self, _observation: dict[str, Any]) -> dict[str, Any]:
        if self._index >= self.frame_count:
            hold = self.actions[-1:].copy()
            hold[:, 14:] = 0.0
            return {"actions": hold, "finished": True}
        end = min(self._index + self.action_horizon, self.frame_count)
        chunk = self.actions[self._index : end]
        self._index = end
        return {"actions": chunk, "finished": self._index >= self.frame_count}

    def reset(self) -> None:
        self._index = 0


def load_policy_factory(spec: str, config_path: Path | None = None) -> Any:
    """Load `module:factory`; the factory receives a decoded JSON config dict."""

    if ":" not in spec:
        raise ValueError("--policy-factory must use module:callable syntax.")
    module_name, attribute_name = spec.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute_name)
    config: dict[str, Any] = {}
    if config_path is not None:
        config = json.loads(config_path.expanduser().read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Policy config JSON must contain an object.")
    return factory(config)


class RemotePolicyServer:
    def __init__(self, policy: Any, *, host: str = "0.0.0.0", port: int = 8000) -> None:
        if not callable(getattr(policy, "infer", None)) and not callable(policy):
            raise TypeError("Policy must be callable or expose infer(observation).")
        self.policy = policy
        self.host = host
        self.port = int(port)
        self.policy_name = str(getattr(policy, "name", type(policy).__name__))
        self._policy_lock = threading.Lock()

    def create_server(self) -> WebSocketServer:
        return serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        )

    def serve_forever(self) -> None:
        with self.create_server() as websocket_server:
            LOGGER.info(
                "ExRoMa policy server listening on ws://%s:%d (%s)",
                self.host,
                self.port,
                self.policy_name,
            )
            websocket_server.serve_forever()

    def _infer(self, observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        with self._policy_lock:
            if callable(getattr(self.policy, "infer", None)):
                result = self.policy.infer(observation)
            else:
                result = self.policy(observation)
        policy_status: dict[str, Any] = {}
        if isinstance(result, dict):
            policy_status = dict(result.get("policy_status", {}))
            if "finished" in result:
                policy_status["finished"] = bool(result["finished"])
            result = result.get("actions", result.get("action"))
        return validate_actions(result), policy_status

    def _reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            with self._policy_lock:
                reset()

    def _handler(self, websocket: ServerConnection) -> None:
        websocket.send(encode_message(server_metadata(self.policy_name)))
        try:
            while True:
                frame = websocket.recv()
                message = decode_message(frame)
                request_id = int(message.get("request_id", -1))
                try:
                    if message.get("protocol") != PROTOCOL_VERSION:
                        raise ValueError("Unsupported ExRoMa policy protocol.")
                    if message.get("type") == "reset":
                        self._reset()
                        websocket.send(
                            encode_message(
                                {
                                    "protocol": PROTOCOL_VERSION,
                                    "type": "reset_ok",
                                    "request_id": request_id,
                                }
                            )
                        )
                        continue
                    observation = decode_infer_request(frame)
                    start = time.perf_counter()
                    actions, policy_status = self._infer(observation)
                    infer_ms = (time.perf_counter() - start) * 1000.0
                    websocket.send(
                        encode_action_response(
                            request_id=observation["request_id"],
                            actions=actions,
                            infer_ms=infer_ms,
                            policy_status=policy_status,
                        )
                    )
                except Exception as exc:
                    LOGGER.exception("Policy request failed")
                    websocket.send(
                        encode_message(
                            {
                                "protocol": PROTOCOL_VERSION,
                                "type": "error",
                                "request_id": request_id,
                                "message": str(exc),
                            }
                        )
                    )
        except ConnectionClosed:
            LOGGER.info("Policy client disconnected")
