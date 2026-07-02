"""
Named nav-stack poses with optional JSON persistence.

Example::

    registry = LocationRegistry("locations.json")   # auto-loads if file exists
    registry.record("storage_area", dds)            # walk robot there, then call
    registry.record("construction_area", dds)
    registry.save()                                 # persist for next trial

    # --- next trial ---
    registry = LocationRegistry("locations.json")   # reloads automatically
    registry.goto("storage_area", dds)              # navigate there
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

DEFAULT_LOCATIONS_FILE = Path(__file__).parents[1] / "assets" / "locations" / "locations.json"


@dataclass
class NavPose:
    """2-D nav pose: position (x, y) + orientation as unit-quaternion (qz, qw)."""

    x: float
    y: float
    qz: float
    qw: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.qz, self.qw)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NavPose":
        return cls(**d)

    def __str__(self) -> str:
        return f"x={self.x:.3f}  y={self.y:.3f}  qz={self.qz:.3f}  qw={self.qw:.3f}"


class LocationRegistry:
    """
    Named nav-stack poses (e.g. ``"storage_area"``, ``"construction_area"``).

    Locations are captured by walking the robot to a spot and calling
    :meth:`record`.  Pass *path* at construction to auto-load an existing
    file and enable :meth:`save` without specifying a path each time.
    """

    def __init__(self, path: Optional[str] = None):
        self._locations: dict[str, NavPose] = {}
        self._path: Optional[Path] = Path(path) if path else None
        if self._path and self._path.exists():
            self.load()

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, name: str, nav) -> NavPose:
        """Capture the current nav pose and store it under *name*. nav must have get_nav_state()."""
        x, y, qz, qw = nav.get_nav_state()
        pose = NavPose(x, y, qz, qw)
        self._locations[name] = pose
        print(f"  Recorded '{name}': {pose}")
        return pose

    # ── Access ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> NavPose:
        if name not in self._locations:
            raise KeyError(f"Location '{name}' not set.")
        return self._locations[name]

    def names(self) -> list[str]:
        return list(self._locations.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._locations

    # ── Navigation ───────────────────────────────────────────────────────────

    def goto(self, name: str, nav):
        """Send a GOTO command for the named location."""
        pose = self.get(name)
        print(f"  Navigating to '{name}': {pose}")
        nav.send_nav_goto(*pose.as_tuple())

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """Save all locations to a JSON file."""
        p = Path(path) if path else self._path
        if p is None:
            raise ValueError(
                "No path specified — pass a path argument or set one at construction."
            )
        data = {name: pose.to_dict() for name, pose in self._locations.items()}
        p.write_text(json.dumps(data, indent=2))
        print(f"  Saved {len(data)} location(s) to {p}")

    def load(self, path: Optional[str] = None):
        """Load locations from a JSON file (merges into existing registry)."""
        p = Path(path) if path else self._path
        if p is None or not p.exists():
            return
        data = json.loads(p.read_text())
        loaded = {name: NavPose.from_dict(v) for name, v in data.items()}
        self._locations.update(loaded)
        print(f"  Loaded {len(loaded)} location(s) from {p}")


if __name__ == "__main__":
    import sys
    from bricklaying.robot import DDSInterface

    LOCATIONS_FILE = Path(__file__).parents[1] / "assets" / "locations" / "locations.json"
    LOCATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    iface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    print(f"Connecting on {iface}...")
    dds = DDSInterface(iface)

    # ── Record poses ─────────────────────────────────────────────────────────
    registry = LocationRegistry(path=str(LOCATIONS_FILE))

    for name in ("storage_area", "construction_area"):
        while True:
            input(f"\nWalk robot to '{name}', then press Enter to record...")
            pose = registry.record(name, dds)
            if input(f"  Confirm? [y/N]: ").strip().lower() == "y":
                break

    registry.save()

    # ── Verify round-trip ────────────────────────────────────────────────────
    print("\nReloading from file to verify...")
    reloaded = LocationRegistry(path=str(LOCATIONS_FILE))
    all_ok = True
    for name in registry.names():
        match = registry.get(name) == reloaded.get(name)
        print(f"  {name}: {'OK' if match else 'MISMATCH'}")
        all_ok = all_ok and match
    if not all_ok:
        print("Verification failed — aborting.")
        sys.exit(1)
    print("Verification passed.")

    # ── Navigate between poses ───────────────────────────────────────────────
    input("\nPress Enter to navigate to storage_area...")
    reloaded.goto("storage_area", dds)
    input("Press Enter to navigate to construction_area...")
    reloaded.goto("construction_area", dds)
    input("Press Enter to navigate back to storage_area...")
    reloaded.goto("storage_area", dds)

    print("\nTest complete.")
    dds.shutdown()
