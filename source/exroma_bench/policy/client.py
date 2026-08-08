"""Synchronous WebSocket client for an ExRoMa remote policy server."""

from __future__ import annotations

import time
from typing import Any

from typing_extensions import Self
from websockets.sync.client import ClientConnection, connect

from .protocol import (
    ACTION_DIM,
    CAMERA_KEYS,
    PROTOCOL_VERSION,
    STATE_DIM,
    PolicyProtocolError,
    decode_action_response,
    decode_message,
    encode_infer_request,
    encode_message,
)


class RemotePolicyClient:
    """Send RGB and compact state observations and receive 16-D action chunks."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        connect_timeout: float = 30.0,
        response_timeout: float = 30.0,
        jpeg_quality: int = 85,
    ) -> None:
        self.uri = f"ws://{host}:{int(port)}"
        self.connect_timeout = float(connect_timeout)
        self.response_timeout = float(response_timeout)
        self.jpeg_quality = int(jpeg_quality)
        self._request_id = 0
        self._connection = self._connect()
        self.metadata = self._receive_metadata()

    def _connect(self) -> ClientConnection:
        deadline = time.monotonic() + self.connect_timeout
        last_error: OSError | None = None
        while True:
            try:
                return connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    open_timeout=min(5.0, self.connect_timeout),
                )
            except OSError as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Could not connect to policy server at {self.uri} within "
                        f"{self.connect_timeout:g}s."
                    ) from last_error
                time.sleep(0.25)

    def _receive_metadata(self) -> dict[str, Any]:
        frame = self._connection.recv(timeout=self.response_timeout)
        metadata = decode_message(frame)
        expected = {
            "protocol": PROTOCOL_VERSION,
            "type": "metadata",
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "cameras": list(CAMERA_KEYS),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise PolicyProtocolError(
                    f"Server metadata {key}={metadata.get(key)!r}, expected {value!r}."
                )
        return metadata

    def infer(
        self,
        *,
        state: Any,
        images: dict[str, Any],
        prompt: str,
        timestamp: float,
    ) -> tuple[Any, dict[str, Any]]:
        request_id = self._request_id
        self._request_id += 1
        request = encode_infer_request(
            request_id=request_id,
            state=state,
            images=images,
            prompt=prompt,
            timestamp=timestamp,
            jpeg_quality=self.jpeg_quality,
        )
        self._connection.send(request)
        response = self._connection.recv(timeout=self.response_timeout)
        return decode_action_response(response, request_id=request_id)

    def reset(self) -> None:
        request_id = self._request_id
        self._request_id += 1
        self._connection.send(
            encode_message(
                {
                    "protocol": PROTOCOL_VERSION,
                    "type": "reset",
                    "request_id": request_id,
                }
            )
        )
        response = decode_message(self._connection.recv(timeout=self.response_timeout))
        if response.get("type") == "error":
            raise RuntimeError(f"Policy server reset failed: {response.get('message')}")
        if response.get("type") != "reset_ok" or response.get("request_id") != request_id:
            raise PolicyProtocolError("Policy server returned an invalid reset response.")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
