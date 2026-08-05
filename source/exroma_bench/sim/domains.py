"""Planetary physics and illumination presets."""

from __future__ import annotations

from dataclasses import dataclass


NASA_OUTPOST_SCALE = 0.02
DEFAULT_SUN_ORIENTATION = (
    0.8923991008325228,
    0.3696438106143861,
    0.23911761839433449,
    -0.09904576054128762,
)
IDENTITY_ORIENTATION = (1.0, 0.0, 0.0, 0.0)
MOON_REFERENCE_SUN_ORIENTATION = (
    0.9942653179168701,
    -0.10550560802221298,
    0.017368007451295853,
    0.001842987141571939,
)
MOON_REFERENCE_SKY_ORIENTATION = (
    0.023698091506958008,
    -0.6120024919509888,
    -0.009596887975931168,
    0.7904424667358398,
)


@dataclass(frozen=True)
class DomainPreset:
    name: str
    gravity: float
    sun_intensity: float
    sun_angle: float
    sun_color_temperature: float
    dome_intensity: float
    sky_texture: str
    sun_orientation: tuple[float, float, float, float]
    sky_orientation: tuple[float, float, float, float]


DOMAINS = {
    "moon": DomainPreset(
        name="moon",
        gravity=1.62496,
        sun_intensity=1360.8160400390625,
        sun_angle=0.0,
        sun_color_temperature=5778.0,
        dome_intensity=340.25,
        sky_texture="textures/skydome/stars.exr",
        sun_orientation=MOON_REFERENCE_SUN_ORIENTATION,
        sky_orientation=MOON_REFERENCE_SKY_ORIENTATION,
    ),
    "mars": DomainPreset(
        name="mars",
        gravity=3.72076,
        sun_intensity=729.0,
        sun_angle=0.35,
        sun_color_temperature=6250.0,
        dome_intensity=182.25,
        sky_texture="textures/skydome/mars_sky.exr",
        sun_orientation=DEFAULT_SUN_ORIENTATION,
        sky_orientation=IDENTITY_ORIENTATION,
    ),
    "earth": DomainPreset(
        name="earth",
        gravity=9.80665,
        sun_intensity=775.0,
        sun_angle=0.53,
        sun_color_temperature=5750.0,
        dome_intensity=193.75,
        sky_texture="textures/skydome/cloudy_sky.exr",
        sun_orientation=DEFAULT_SUN_ORIENTATION,
        sky_orientation=IDENTITY_ORIENTATION,
    ),
}


SCENE_DOMAINS = {
    "ground_plane": "earth",
    "lunalab": "moon",
    "moon_surface": "moon",
    "procedural_moon": "moon",
    "mars_surface": "mars",
    "procedural_mars": "mars",
    "oberpfaffenhofen": "earth",
}


def domain_for_scene(scene: str) -> DomainPreset:
    try:
        return DOMAINS[SCENE_DOMAINS[scene]]
    except KeyError as exc:
        raise ValueError(f"Unsupported ExRoMa scene: {scene}") from exc
