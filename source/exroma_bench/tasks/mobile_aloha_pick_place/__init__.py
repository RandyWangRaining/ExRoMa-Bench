"""Mobile ALOHA pick-and-place controllers and planners."""

from __future__ import annotations


__all__ = [
    "CuroboTrajectory",
    "MobileAlohaCuroboPlanner",
    "ScriptedPickPlaceConfig",
    "ScriptedPickPlaceController",
]


def __getattr__(name: str):
    # Keep cuRobo and Isaac Sim optional until a caller selects that control
    # path. This also lets the standalone cuRobo validator run without Kit.
    if name in {"CuroboTrajectory", "MobileAlohaCuroboPlanner"}:
        from .curobo_planner import CuroboTrajectory, MobileAlohaCuroboPlanner

        return {
            "CuroboTrajectory": CuroboTrajectory,
            "MobileAlohaCuroboPlanner": MobileAlohaCuroboPlanner,
        }[name]
    if name in {"ScriptedPickPlaceConfig", "ScriptedPickPlaceController"}:
        from .scripted_controller import ScriptedPickPlaceConfig, ScriptedPickPlaceController

        return {
            "ScriptedPickPlaceConfig": ScriptedPickPlaceConfig,
            "ScriptedPickPlaceController": ScriptedPickPlaceController,
        }[name]
    raise AttributeError(name)
