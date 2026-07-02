"""
RealSense camera interface for depth and RGB image acquisition.
Provides simple API for capturing aligned frames and deprojecting to 3D point clouds.
"""
import time
import threading
import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. RealSense functionality will be disabled.")

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int
    depth_scale: float  # Depth units to meters conversion

# Measured on G1 .165
D435_DEFAULT_INTRINSICS = CameraIntrinsics(
    fx=605.2,
    fy=604.9,
    cx=325.4,
    cy=237.3,
    width=640,
    height=480,
    depth_scale=0.001,
)


def deproject_pixels_to_points(
    depth: np.ndarray,
    color: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to 3D point cloud with corresponding colors.
    
    Args:
        depth: Depth image [H x W] in raw depth units
        color: Color image [H x W x 3] in RGB format
        intrinsics: Camera intrinsic parameters
        mask: Optional boolean mask [H x W] to select specific pixels
    
    Returns:
        points: 3D points [N x 3] in camera frame (meters)
        colors: Corresponding RGB colors [N x 3]
    """
    h, w = depth.shape
    
    # Create mask if not provided
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        mask = mask.astype(bool)
    
    # Create coordinate grid
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Apply mask
    xs_masked = xs[mask]
    ys_masked = ys[mask]
    depths_masked = depth[mask].astype(np.float32) * intrinsics.depth_scale
    
    # Filter invalid depths
    valid = depths_masked > 0
    xs_valid = xs_masked[valid]
    ys_valid = ys_masked[valid]
    zs_valid = depths_masked[valid]
    
    # Deproject to 3D (pinhole camera model)
    x = (xs_valid - intrinsics.cx) * zs_valid / intrinsics.fx
    y = (ys_valid - intrinsics.cy) * zs_valid / intrinsics.fy
    z = zs_valid
    
    points = np.column_stack((x, y, z))
    colors = color[ys_valid, xs_valid]
    
    return points, colors


class RealSenseCamera:
    """
    Interface for Intel RealSense depth cameras.
    
    Provides aligned depth and color images with calibrated intrinsics.
    """
    
    # Default camera configuration
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FPS = 30
    DEFAULT_EXPOSURE = 500  # microseconds
    
    def __init__(self,
                 width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT,
                 fps: int = DEFAULT_FPS,
                 exposure: Optional[int] = DEFAULT_EXPOSURE):
        """
        Initialize RealSense camera.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fps: Frames per second
            exposure: Exposure time in microseconds (None for auto-exposure)
        
        Raises:
            RuntimeError: If RealSense library not available or camera not found
        """
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not installed. Install with: pip install pyrealsense2")
        
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Start pipeline
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to start RealSense camera: {e}")
        
        # Set exposure if specified
        if exposure is not None:
            self._set_exposure(exposure)
        
        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Create alignment object (depth to color)
        self.align = rs.align(rs.stream.color)

        # Start thread
        self._intrinsics = None
        self._latest_color = None
        self._latest_depth = None
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Allow exposure to adjust
        wait_time = 3.0
        print(f"Initializing RealSense: flushing {wait_time} seconds of frames...")
        time.sleep(wait_time)

        print(f"RealSense camera initialized: {width}x{height} @ {fps}fps")

    def _restart_pipeline(self):
        """Recreate and restart the pipeline after a disconnect."""
        # Safely stop — may already be dead after a disconnect, so swallow errors
        try:
            self.pipeline.stop()
        except Exception:
            pass

        for attempt in range(10):
            if not self._running:
                return
            time.sleep(2.0)
            try:
                # Recreate from scratch — a tainted pipeline object cannot be reused
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                self.profile = self.pipeline.start(self.config)
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                self.align = rs.align(rs.stream.color)
                with self._lock:
                    self._intrinsics = None
                print("RealSense pipeline restarted successfully.")
                return
            except Exception as e:
                print(f"Pipeline restart attempt {attempt + 1}/10 failed: {e}")

        print("Pipeline restart failed after 10 attempts — giving up.")

    def _capture_loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                if not self._running:
                    break
                print(f"Warning: Frame timeout ({e}), attempting pipeline restart...")
                self._restart_pipeline()
                continue

            aligned = self.align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Extract intrinsics on first capture
            with self._lock:
                if self._intrinsics is None:
                    self._extract_intrinsics(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            with self._lock:
                self._latest_color = color_image
                self._latest_depth = depth_image
    
    def _set_exposure(self, exposure_us: int):
        """Set camera exposure time."""
        try:
            color_sensor = self.profile.get_device().query_sensors()[1]
            color_sensor.set_option(rs.option.exposure, exposure_us)
        except Exception as e:
            print(f"Warning: Failed to set exposure: {e}")
    
    def get_frames(self):
        with self._lock:
            if self._latest_color is None:
                raise RuntimeError("No frames captured yet.")
            return self._latest_color.copy(), self._latest_depth.copy()

    def _extract_intrinsics(self, depth_frame):
        """Extract camera intrinsics from depth frame."""
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        
        self._intrinsics = CameraIntrinsics(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            width=intr.width,
            height=intr.height,
            depth_scale=self.depth_scale
        )
    
    @property
    def intrinsics(self) -> CameraIntrinsics:
        """
        Get camera intrinsic parameters.
        
        Returns:
            Camera intrinsics
        
        Raises:
            RuntimeError: If frames haven't been captured yet
        """
        with self._lock:
            if self._intrinsics is None:
                raise RuntimeError(
                    "Intrinsics not available. Call get_frames() at least once to calibrate."
                )
            return self._intrinsics
    
    def get_point_cloud(self, 
                       color: np.ndarray,
                       depth: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """See `deproject_pixels_to_points`."""
        return deproject_pixels_to_points(depth, color, self.intrinsics, mask)
    
    def stop(self):
        self._running = False
        # Join first so the capture thread is no longer touching the pipeline,
        # then stop — calling pipeline.stop() concurrently with wait_for_frames()
        # from another thread causes a segfault in the native SDK.
        self._thread.join(timeout=7.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass
        print("RealSense camera stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically stop camera."""
        self.stop()


# ===== Test/Example Usage =====

if __name__ == "__main__":
    print("Testing RealSense camera interface...")
    
    # Use context manager for automatic cleanup
    with RealSenseCamera() as camera:
        # Capture a frame
        print("\nCapturing frame...")
        color, depth = camera.get_frames()
        print(f"  Color: {color.shape}, dtype={color.dtype}")
        print(f"  Depth: {depth.shape}, dtype={depth.dtype}")
        
        # Get intrinsics
        print("\nCamera intrinsics:")
        intr = camera.intrinsics
        print(f"  Focal length: fx={intr.fx:.1f}, fy={intr.fy:.1f}")
        print(f"  Principal point: cx={intr.cx:.1f}, cy={intr.cy:.1f}")
        print(f"  Resolution: {intr.width}x{intr.height}")
        print(f"  Depth scale: {intr.depth_scale:.6f} m/unit")
        
        # Generate point cloud
        print("\nGenerating point cloud...")
        points, colors = camera.get_point_cloud(color, depth)
        print(f"  Points: {points.shape}")
        print(f"  Point cloud bounds:")
        print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
        print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
        print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")
        
        # Test with mask (center 100x100 region)
        print("\nTesting with masked region...")
        h, w = depth.shape
        mask = np.zeros((h, w), dtype=bool)
        mask[h//2-50:h//2+50, w//2-50:w//2+50] = True
        
        points_masked, colors_masked = camera.get_point_cloud(color, depth, mask)
        print(f"  Masked points: {points_masked.shape}")

        for i in range(100):
            color, depth = camera.get_frames()
            print(f"Frame {i} OK  color={color.shape}")

    print("\nTest complete!")


