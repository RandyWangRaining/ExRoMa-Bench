"""Remote policy inference interfaces for ExRoMa evaluation."""

from .client import RemotePolicyClient
from .protocol import (
    ACTION_DIM,
    ACTION_NAMES,
    CAMERA_KEYS,
    PROTOCOL_VERSION,
    STATE_DIM,
    STATE_NAMES,
)

__all__ = [
    "ACTION_DIM",
    "ACTION_NAMES",
    "CAMERA_KEYS",
    "PROTOCOL_VERSION",
    "STATE_DIM",
    "STATE_NAMES",
    "RemotePolicyClient",
]
