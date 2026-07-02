import time
import threading
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, replace

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

from .realsense import CameraIntrinsics, deproject_pixels_to_points, D435_DEFAULT_INTRINSICS


# Must match the publisher's IDL definition in unitree_sdk2py_bridge.py
@dataclass
@annotate.final
@annotate.autoid("sequential")
class SimImage_(idl.IdlStruct, typename="sim.msg.dds_.SimImage_"):
    height: types.uint32
    width: types.uint32
    encoding: str
    data: types.sequence[types.uint8]


# Depth is already in meters from the MuJoCo pipeline
SIM_INTRINSICS = replace(D435_DEFAULT_INTRINSICS, depth_scale=1.0)

TOPIC_REALSENSE_COLOR = "rt/realsense/color"
TOPIC_REALSENSE_DEPTH = "rt/realsense/depth"


class SimRealSenseCamera:

    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FPS = 30

    def __init__(self,
                 width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT,
                 fps: int = DEFAULT_FPS,
                 exposure: Optional[int] = None):
        self.width = width
        self.height = height
        self.fps = fps
        self._intrinsics = SIM_INTRINSICS

        self._latest_color = None
        self._latest_depth = None
        self._lock = threading.Lock()

        self._color_sub = ChannelSubscriber(TOPIC_REALSENSE_COLOR, SimImage_)
        self._color_sub.Init(self._color_cb, 10)

        self._depth_sub = ChannelSubscriber(TOPIC_REALSENSE_DEPTH, SimImage_)
        self._depth_sub.Init(self._depth_cb, 10)

        print(f"SimRealSenseCamera initialized, subscribing to {TOPIC_REALSENSE_COLOR} and {TOPIC_REALSENSE_DEPTH}")

    def _color_cb(self, msg: SimImage_):
        if len(msg.data) != msg.height * msg.width * 3:
            return
        arr = np.array(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with self._lock:
            self._latest_color = arr

    def _depth_cb(self, msg: SimImage_):
        if len(msg.data) != msg.height * msg.width * 2:
            return
        # Decode uint16 millimeters back to float32 meters
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
        with self._lock:
            self._latest_depth = (arr.astype(np.float32) / 1000.0)

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if self._latest_color is None or self._latest_depth is None:
                return None, None
            return self._latest_color.copy(), self._latest_depth.copy()
    def flush(self):
        #do nothing
        pass
    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def get_point_cloud(self,
                        color: np.ndarray,
                        depth: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        return deproject_pixels_to_points(depth, color, self.intrinsics, mask)

    def stop(self):
        print("SimRealSenseCamera stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    import cv2
    from bricklaying.perception.realsense import CameraIntrinsics, deproject_pixels_to_points, D435_DEFAULT_INTRINSICS

    print("SimRealSense test — waiting for frames from MuJoCo bridge...")
    print("Make sure unitree_mujoco.py is running.\n")

    # Initialize DDS (same domain/interface as the sim)
    ChannelFactoryInitialize(1, "lo")
    print("Waiting for DDS discovery...")
    time.sleep(2.0)

    camera = SimRealSenseCamera()

    # Wait for first frame
    timeout = 15.0
    start = time.time()
    color, depth = None, None
    while time.time() - start < timeout:
        color, depth = camera.get_frames()
        if color is not None:
            break
        time.sleep(0.1)

    if color is None:
        print(f"ERROR: No frames received after {timeout}s. Is the MuJoCo sim publishing?")
        camera.stop()
        exit(1)

    print(f"First frame received after {time.time() - start:.1f}s")

    # Validate color image
    print(f"\n--- Color Image ---")
    print(f"  shape:  {color.shape}  (expect: (480, 640, 3))")
    print(f"  dtype:  {color.dtype}  (expect: uint8)")
    print(f"  range:  [{color.min()}, {color.max()}]")
    assert color.shape == (480, 640, 3), f"Unexpected color shape: {color.shape}"
    assert color.dtype == np.uint8, f"Unexpected color dtype: {color.dtype}"
    print(f"  PASS")

    # Validate depth image
    print(f"\n--- Depth Image ---")
    print(f"  shape:  {depth.shape}  (expect: (480, 640))")
    print(f"  dtype:  {depth.dtype}  (expect: float32)")
    print(f"  total pixels: {depth.size}")
    print(f"  zero pixels: {np.sum(depth == 0)}")
    print(f"  nonzero pixels: {np.sum(depth > 0)}")
    print(f"  raw value sample (first 10 nonzero): {depth[depth > 0][:10]}")
    assert depth.shape == (480, 640), f"Unexpected depth shape: {depth.shape}"
    assert depth.dtype == np.float32, f"Unexpected depth dtype: {depth.dtype}"
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        print(f"  range:  [{valid_depth.min():.3f}, {valid_depth.max():.3f}] meters")
        print(f"  PASS")
    else:
        print(f"  WARNING: All depth pixels are zero — depth data may not be transmitting correctly")

    # Validate intrinsics
    intr = camera.intrinsics
    print(f"\n--- Intrinsics ---")
    print(f"  fx={intr.fx}, fy={intr.fy}, cx={intr.cx}, cy={intr.cy}")
    print(f"  depth_scale={intr.depth_scale}  (expect: 1.0)")
    assert intr.depth_scale == 1.0, f"depth_scale should be 1.0, got {intr.depth_scale}"
    print(f"  PASS")

    # Validate deprojection to point cloud
    print(f"\n--- Point Cloud ---")
    points, colors = camera.get_point_cloud(color, depth)
    print(f"  num points: {len(points)}")
    print(f"  points shape: {points.shape}")
    print(f"  x range: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
    print(f"  y range: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
    print(f"  z range: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    assert len(points) > 0, "No points generated"
    assert points.shape[1] == 3, f"Expected Nx3 points, got {points.shape}"
    assert np.all(np.isfinite(points)), "Non-finite values in point cloud"
    print(f"  PASS")

    # Save color and depth image
    cv2.imwrite("sim_realsense_color.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    cv2.imwrite("sim_realsense_depth.png", (depth * 1000.0).clip(0, 65535).astype(np.uint16))

    # Save depth image with colorbar and annotations
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    depth_display = np.where(depth > 0, depth, np.nan)
    im = ax.imshow(depth_display, cmap='turbo')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (meters)', fontsize=12)

    valid = depth[depth > 0]
    stats = (f"min={valid.min():.3f}m  max={valid.max():.3f}m  "
             f"mean={valid.mean():.3f}m  median={np.median(valid):.3f}m  valid={len(valid)}/{depth.size}px") if len(valid) > 0 else "No valid depth"
    ax.set_title(f"Simulated RealSense Depth\n{stats}", fontsize=12)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    fig.tight_layout()
    fig.savefig("sim_realsense_depth_ann.png", dpi=120)
    plt.close(fig)

    # Depth histogram
    if len(valid) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(valid, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(valid.mean(), color='red', linestyle='--', linewidth=1.5, label=f'mean={valid.mean():.3f}m')
        ax.axvline(np.median(valid), color='orange', linestyle='--', linewidth=1.5, label=f'median={np.median(valid):.3f}m')
        ax.set_xlabel('Depth (meters)', fontsize=12)
        ax.set_ylabel('Pixel count', fontsize=12)
        ax.set_title('Depth Distribution', fontsize=12)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig("sim_realsense_depth_hist.png", dpi=120)
        plt.close(fig)

    print(f"\nImages saved to /tmp/sim_realsense_color.png, /tmp/sim_realsense_depth.png, /tmp/sim_realsense_depth_hist.png")

    print(f"\nAll checks passed.")
    camera.stop()
