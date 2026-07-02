"""Shared viser rendering configuration for the MuJoCo spectator.

Used by both the in-process spectator (server.py's LiveSpectator) and the
out-of-process spectator (spectate.py) so they render identically: full-detail
meshes (LOD off), studio lighting, and fixed-geom shadows.
"""
from __future__ import annotations

from typing import Any


def disable_lod(server: Any) -> None:
    """Force full-detail meshes. Call BEFORE building the scene, so the scene's
    mesh handles are created with LOD off."""
    add_batched_meshes_trimesh = server.scene.add_batched_meshes_trimesh

    def add_full_detail(*args: Any, **kwargs: Any) -> Any:
        kwargs["lod"] = "off"
        return add_batched_meshes_trimesh(*args, **kwargs)

    server.scene.add_batched_meshes_trimesh = add_full_detail


def enable_fixed_geom_shadows(scene: Any) -> None:
    for handle in getattr(scene, "_fixed_geom_handles", {}).values():
        if hasattr(handle, "cast_shadow"):
            handle.cast_shadow = True
        if hasattr(handle, "receive_shadow"):
            handle.receive_shadow = True


def wrap_visual_rebuild_for_shadows(scene: Any) -> None:
    rebuild_visual_handles = getattr(scene, "rebuild_visual_handles", None)
    if rebuild_visual_handles is None:
        return

    def rebuild_with_shadows(*args: Any, **kwargs: Any) -> Any:
        result = rebuild_visual_handles(*args, **kwargs)
        enable_fixed_geom_shadows(scene)
        return result

    scene.rebuild_visual_handles = rebuild_with_shadows


def configure_lighting(server: Any) -> None:
    server.scene.configure_default_lights(enabled=False)
    server.scene.configure_environment_map("studio", environment_intensity=0.65)
    server.scene.add_light_ambient("/lights/ambient", color=(255, 255, 255), intensity=0.25)
    server.scene.add_light_hemisphere(
        "/lights/hemisphere",
        sky_color=(255, 255, 255),
        ground_color=(170, 175, 180),
        intensity=0.6,
    )
    server.scene.add_light_spot(
        "/lights/key",
        color=(255, 248, 235),
        intensity=5.0,
        distance=8.0,
        angle=0.9,
        penumbra=0.45,
        decay=1.5,
        cast_shadow=True,
        position=(1.4, -2.2, 4.0),
        direction=(-0.45, 0.55, -1.0),
    )


def configure_after_scene(server: Any, scene: Any) -> None:
    """Lighting + fixed-geom shadows. Call after the scene is built."""
    configure_lighting(server)
    enable_fixed_geom_shadows(scene)
    wrap_visual_rebuild_for_shadows(scene)
