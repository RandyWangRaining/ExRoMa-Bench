"""Versioned, validation-first messages for remote policy inference."""

from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import numpy as np

PROTOCOL_VERSION = "exroma.policy.v1"
STATE_DIM = 16
ACTION_DIM = 16
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
COMPACT_JOINT_NAMES = (
    *(f"fl_joint{index}" for index in range(1, 7)),
    "fl_gripper",
    *(f"fr_joint{index}" for index in range(1, 7)),
    "fr_gripper",
)
STATE_NAMES = (*COMPACT_JOINT_NAMES, "base_velocity/forward", "base_velocity/yaw")
ACTION_NAMES = (*COMPACT_JOINT_NAMES, "base_command/forward", "base_command/yaw")


class PolicyProtocolError(ValueError):
    """Raised when a remote policy message violates the ExRoMa contract."""


def encode_message(message: dict[str, Any]) -> bytes:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_message(frame: str | bytes) -> dict[str, Any]:
    try:
        if isinstance(frame, bytes):
            frame = frame.decode("utf-8")
        message = json.loads(frame)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyProtocolError("Policy frame is not valid UTF-8 JSON.") from exc
    if not isinstance(message, dict):
        raise PolicyProtocolError("Policy frame must contain a JSON object.")
    return message


def validate_state(state: Any) -> np.ndarray:
    array = np.asarray(state, dtype=np.float32)
    if array.shape != (STATE_DIM,):
        raise PolicyProtocolError(f"Expected state[{STATE_DIM}], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise PolicyProtocolError("Policy state contains NaN or infinite values.")
    return np.ascontiguousarray(array)


def validate_actions(actions: Any) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.shape == (ACTION_DIM,):
        array = array.reshape(1, ACTION_DIM)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM or len(array) == 0:
        raise PolicyProtocolError(
            f"Expected action[{ACTION_DIM}] or actions[T,{ACTION_DIM}], got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise PolicyProtocolError("Policy action contains NaN or infinite values.")
    return np.ascontiguousarray(array)


def _encode_rgb_jpeg(image: Any, *, quality: int) -> tuple[str, list[int]]:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise PolicyProtocolError(f"Expected an HWC RGB image, got {rgb.shape}.")
    rgb = rgb[..., :3]
    if not np.all(np.isfinite(rgb)):
        raise PolicyProtocolError("Policy image contains invalid values.")
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and float(np.max(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))],
    )
    if not ok:
        raise PolicyProtocolError("OpenCV could not JPEG-encode a policy image.")
    return base64.b64encode(encoded).decode("ascii"), [int(rgb.shape[0]), int(rgb.shape[1]), 3]


def _decode_rgb_jpeg(encoded: Any, *, camera_key: str) -> np.ndarray:
    if not isinstance(encoded, str):
        raise PolicyProtocolError(f"Camera {camera_key} JPEG must be a base64 string.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise PolicyProtocolError(f"Camera {camera_key} contains invalid base64.") from exc
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise PolicyProtocolError(f"Camera {camera_key} JPEG could not be decoded.")
    return np.ascontiguousarray(bgr[..., ::-1])


def encode_infer_request(
    *,
    request_id: int,
    state: Any,
    images: dict[str, Any],
    prompt: str,
    timestamp: float,
    jpeg_quality: int = 85,
) -> bytes:
    compact_state = validate_state(state)
    if set(images) != set(CAMERA_KEYS):
        raise PolicyProtocolError(f"Expected cameras {list(CAMERA_KEYS)}, got {sorted(images)}.")
    encoded_images = {}
    image_shapes = {}
    for camera_key in CAMERA_KEYS:
        encoded, shape = _encode_rgb_jpeg(images[camera_key], quality=jpeg_quality)
        encoded_images[camera_key] = encoded
        image_shapes[camera_key] = shape
    return encode_message(
        {
            "protocol": PROTOCOL_VERSION,
            "type": "infer",
            "request_id": int(request_id),
            "state": compact_state.tolist(),
            "images": encoded_images,
            "image_shapes": image_shapes,
            "image_encoding": "jpeg",
            "prompt": str(prompt),
            "timestamp": float(timestamp),
        }
    )


def decode_infer_request(frame: str | bytes) -> dict[str, Any]:
    message = decode_message(frame)
    if message.get("protocol") != PROTOCOL_VERSION:
        raise PolicyProtocolError(f"Unsupported protocol: {message.get('protocol')!r}.")
    if message.get("type") != "infer":
        raise PolicyProtocolError(f"Expected an infer request, got {message.get('type')!r}.")
    encoded_images = message.get("images")
    if not isinstance(encoded_images, dict) or set(encoded_images) != set(CAMERA_KEYS):
        raise PolicyProtocolError(f"Infer request must contain cameras {list(CAMERA_KEYS)}.")
    return {
        "request_id": int(message.get("request_id", -1)),
        "state": validate_state(message.get("state")),
        "images": {
            key: _decode_rgb_jpeg(encoded_images[key], camera_key=key) for key in CAMERA_KEYS
        },
        "prompt": str(message.get("prompt", "")),
        "timestamp": float(message.get("timestamp", 0.0)),
    }


def encode_action_response(
    *,
    request_id: int,
    actions: Any,
    infer_ms: float,
    policy_status: dict[str, Any] | None = None,
) -> bytes:
    validated = validate_actions(actions)
    return encode_message(
        {
            "protocol": PROTOCOL_VERSION,
            "type": "action",
            "request_id": int(request_id),
            "actions": validated.tolist(),
            "server_timing": {"infer_ms": float(infer_ms)},
            "policy_status": policy_status or {},
        }
    )


def decode_action_response(frame: str | bytes, *, request_id: int) -> tuple[np.ndarray, dict]:
    message = decode_message(frame)
    if message.get("type") == "error":
        raise RuntimeError(f"Policy server error: {message.get('message', 'unknown error')}")
    if message.get("protocol") != PROTOCOL_VERSION or message.get("type") != "action":
        raise PolicyProtocolError("Policy server returned an incompatible response.")
    if int(message.get("request_id", -1)) != int(request_id):
        raise PolicyProtocolError(
            f"Policy response request_id={message.get('request_id')} does not match {request_id}."
        )
    timing = message.get("server_timing", {})
    if not isinstance(timing, dict):
        timing = {}
    policy_status = message.get("policy_status", {})
    if isinstance(policy_status, dict):
        timing["policy_status"] = policy_status
    return validate_actions(message.get("actions")), timing


def server_metadata(policy_name: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "type": "metadata",
        "policy_name": policy_name,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_names": list(STATE_NAMES),
        "action_names": list(ACTION_NAMES),
        "cameras": list(CAMERA_KEYS),
        "image_layout": "HWC",
        "image_dtype": "uint8",
        "image_transport": "jpeg",
        "supports_action_chunks": True,
    }
