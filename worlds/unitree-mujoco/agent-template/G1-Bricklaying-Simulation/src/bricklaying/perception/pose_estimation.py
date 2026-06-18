"""
Brick pose estimation from RGB-D frames.

Pipeline:
    1. Capture RGB + depth from RealSense
    2. Segment bricks in RGB image (FastSAM)
    3. Deproject masked depth pixels to 3D point cloud
    4. Align against a reference brick model via ICP
    5. Transform poses from realsense to robot pelvis frame
    6. Return 6-DoF pose per detected brick
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

from bricklaying.segmentation import BrickDetection, BrickSegmentorBase
from bricklaying.robot import G1URDFModel, T_CAMERA_TO_REALSENSE
from .realsense import RealSenseCamera, CameraIntrinsics, deproject_pixels_to_points
from .icp import ICPRegistrar, ICPConfig, ICPResult
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# Base brick model
DEFAULT_BRICK_MODEL = "src/bricklaying/assets/flat_brick_large.npy"


# Default rotation guess
angle_z = np.radians(90)
angle_x = np.radians(-45)

Rz = np.array([
    [np.cos(angle_z), -np.sin(angle_z), 0],
    [np.sin(angle_z),  np.cos(angle_z), 0],
    [0,                0,               1],
])

Rx = np.array([
    [1, 0,                0               ],
    [0, np.cos(angle_x), -np.sin(angle_x) ],
    [0, np.sin(angle_x),  np.cos(angle_x) ],
])

R_INIT_GUESS = Rx @ Rz


@dataclass
class BrickPose:
    """6-DoF pose estimate for a single brick."""
    detection: BrickDetection       # Originating 2D detection
    transform: np.ndarray           # (4, 4) SE(3) transform: model -> camera frame
    position: np.ndarray            # (3,) translation in meters
    rotation: np.ndarray            # (3, 3) rotation matrix
    icp_fitness: float              # ICP inlier ratio [0, 1]; higher is better
    icp_rmse: float                 # ICP inlier RMSE in metres; lower is better


@dataclass
class PoseEstimatorConfig:
    """Tunable parameters for the full pipeline."""
    # Maximum distance from camera to consider objects
    max_distance: float = 1.0  # meters
    # Min/max bounding box size to consider objects
    min_extent: float = 0.05  # meters
    max_extent: float = 0.30   # meters
    # Minimum number of valid depth points required to attempt ICP
    min_points: int = 200
    erosion_kernel_size: int = 7
    # ICP settings (forwarded to ICPRegistrar)
    icp: ICPConfig = field(default_factory=ICPConfig)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _match_mask_shape(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Nearest-neighbour resize mask to (H, W) if shapes differ."""
    h, w = target_shape
    if mask.shape == (h, w):
        return mask
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


# ------------------------------------------------------------------
# Estimator
# ------------------------------------------------------------------

class BrickPoseEstimator:
    """
    Estimates 6-DoF brick poses from a single RGB-D frame.

    Args:
        segmentor:    Initialized BrickSegmentorBase.
        brick_model:  Reference point cloud of a single brick in its canonical
                      frame, shape (N, 3) in metres.
        urdf_model:   G1 Urdf model
        camera:       Initialized RealSenseCamera (pipeline already started).
        config:       Optional tuning parameters.
    """

    def __init__(
        self,
        segmentor: BrickSegmentorBase,
        brick_model: Optional[np.ndarray] = None,
        urdf_model: Optional[G1URDFModel] = None,
        camera: Optional[RealSenseCamera] = None,
        config: Optional[PoseEstimatorConfig] = None,
    ):
        if brick_model is None:
            brick_model = np.load(DEFAULT_BRICK_MODEL)

        if brick_model.ndim != 2 or brick_model.shape[1] != 3:
            raise ValueError("brick_model must have shape (N, 3)")
        
        if urdf_model is None:
            urdf_model = G1URDFModel()
        self.urdf_model = urdf_model

        self.brick_model = brick_model
        self.segmentor = segmentor
        self.camera = camera
        self.config = config or PoseEstimatorConfig()
        self.icp = ICPRegistrar(self.config.icp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, init_transform: Optional[np.ndarray] = None) -> list[BrickPose]:
        """
        Capture one RGB-D frame and return a pose estimate for every detected brick.

        Args:
            init_transform: Optional (4, 4) initial guess for ICP, applied to
                            all detections. Useful when approximate pose is known
                            (e.g. from kinematics or a prior frame).

        Returns:
            List of BrickPose, one per detection that had sufficient depth coverage.
            Empty list if no bricks are detected or all lack depth data.
        """
        color, depth = self.camera.get_frames()
        detections = self.segmentor.segment(color)

        poses: list[BrickPose] = []
        for detection in detections:
            pose = self._estimate(color, depth, detection, init_transform)
            if pose is not None:
                poses.append(pose)

        return poses

    def estimate_from_frame(self,color: np.ndarray,depth: np.ndarray,intrinsics: CameraIntrinsics, detections: Optional[list[BrickDetection]] = None, init_transform: Optional[np.ndarray] = None,) -> list[BrickPose]:
        """
        Run pose estimation on pre-captured frames, for offline evaluation.

        Args:
            color:       RGB image (H, W, 3) uint8.
            depth:       Depth image (H, W) uint16 in raw sensor units.
            intrinsics:  Camera intrinsics for the provided frames.
            detections:  Optional list of BrickDetection for segmentation override.
            init_rotation: Optional (3, 3) ICP rotation initial guess.

        Returns:
            List of BrickPose per detected brick.
        """
        if detections is None:
            detections = self.segmentor.segment(color)
        print("Number of objects detected: " + str(np.size(detections)))
        poses: list[BrickPose] = []
        for detection in detections:
            pose = self._estimate(color, depth, detection, init_transform, intrinsics=intrinsics)
            if pose is not None:
                poses.append(pose)

        return poses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate(self, color: np.ndarray, depth: np.ndarray, detection: BrickDetection, init_transform: Optional[np.ndarray],intrinsics: Optional[CameraIntrinsics] = None,) -> Optional[BrickPose]:
        """Deproject one detection mask and run ICP against the brick model."""
        intr = intrinsics or self.camera.intrinsics
        
        # Resize mask to match the captured frame if rescaled
        mask = _match_mask_shape(detection.mask, depth.shape)

        # Edge erosion
        ks = self.config.erosion_kernel_size
        kernel = np.ones((ks, ks), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        points, _ = deproject_pixels_to_points(depth, color, intr, mask=mask)

        # Point cloud size filter
        if len(points) < self.config.min_points:
            return None

        # Distance filter
        median_z = np.median(points[:, 2])
        if median_z > self.config.max_distance:
            return None

        # Extents filter
        extent = points.max(axis=0) - points.min(axis=0)
        if extent.max() > self.config.max_extent or extent.max() < self.config.min_extent:
            return None

        # Make a heuristic initial guess
        if init_transform is None:
            init_transform = np.eye(4)
            init_transform[:3, :3] = R_INIT_GUESS
            init_transform[:3, 3] = np.mean(points, axis=0)

        result: ICPResult = self.icp.register(
            source_points=self.brick_model,
            target_points=points,
            init_transform=init_transform,
        )

        if not result.converged:
            return None

        T = self._transform_to_robot_frame(result.transformation)

        return BrickPose(detection=detection, transform=T,position=T[:3, 3],rotation=T[:3, :3],icp_fitness=result.fitness, icp_rmse=result.inlier_rmse,)
    
    def _transform_to_robot_frame(self, T: np.ndarray) -> np.ndarray:
        """Transform a BrickPose from the RealSense optical frame to the robot pelvis frame."""
        q = self.urdf_model.get_neutral_configuration()
        T_pelvis_to_camera = self.urdf_model.get_frame_transform(q, "d435_link")
        return T_pelvis_to_camera @ T_CAMERA_TO_REALSENSE @ T


# ------------------------------------------------------------------
# Offline test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("In main")
    DEFAULT_IMAGE = "sim_realsense_color.png"
    DEFAULT_DEPTH = "sim_realsense_depth.png"

    model_points = np.load(DEFAULT_BRICK_MODEL)
    print(f"Loaded brick model: {len(model_points)} points")

    # Load RGB image
    color_bgr = cv2.imread(DEFAULT_IMAGE)
    if color_bgr is None:
        print(f"ERROR: Could not read RGB image: {DEFAULT_IMAGE}")
        sys.exit(1)
    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # Load depth image (16-bit PNG, raw sensor units)
    depth = cv2.imread(DEFAULT_DEPTH, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        print(f"ERROR: Could not read depth image: {DEFAULT_DEPTH}")
        sys.exit(1)

    # Taken from G1 Realsense .165
    intrinsics = CameraIntrinsics(
        fx=605.2, fy=604.9,
        cx=325.4, cy=237.3,
        width=640, height=480,
        depth_scale=0.001,  # IMAGES SAVED AS UINT16 IN MILLIMETERS
    )

    # from bricklaying.segmentation.grounded_sam2 import GroundedSAM2Segmentor
    # segmentor = GroundedSAM2Segmentor()
    from bricklaying.segmentation import FastSAMSegmentor
    segmentor = FastSAMSegmentor()

    ks = 7
    config = PoseEstimatorConfig(erosion_kernel_size=ks)

    # No live camera needed — pass None and use estimate_from_frame directly
    estimator = BrickPoseEstimator(segmentor, brick_model=model_points, camera=None, config=config)

    print("Running offline pose estimation...")
    poses = estimator.estimate_from_frame(color, depth, intrinsics)

    # Save segmentation overlay
    from bricklaying.segmentation.base import visualize_detections
    detections = segmentor.segment(color)
    seg_vis = visualize_detections(color, detections)
    cv2.imwrite("segmentation_overlay.png", cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR))
    print(f"Saved segmentation overlay: /tmp/segmentation_overlay.png ({len(detections)} detections)")

    if not poses:
        print("No bricks detected.")
        sys.exit(0)

    for i, pose in enumerate(poses):
        print(f"\nBrick {i + 1}:")
        print(f"  Class      : {pose.detection.class_name} ({pose.detection.confidence:.2f})")
        print(f"  Pos        : {np.round(pose.position, 3)}")
        print(f"  ICP fitness: {pose.icp_fitness:.4f}")
        print(f"  ICP RMSE   : {pose.icp_rmse:.6f} m")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ── Figure 1: segmentation overlay + detected positions ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Pose Estimation — {len(poses)} detections", fontsize=13)

    # Left: colour image with bounding boxes and fitness labels
    ax = axes[0]
    ax.imshow(color)
    ax.set_title("Segmentation detections")
    ax.axis('off')
    cmap = plt.cm.RdYlGn
    for i, pose in enumerate(poses):
        x0, y0, x1, y1 = pose.detection.bbox
        fitness = pose.icp_fitness
        c = cmap(fitness)
        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   linewidth=1.5, edgecolor=c, facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0 - 4, f"{i+1} f={fitness:.2f}", color=c,
                fontsize=7, fontweight='bold')

    # Right: top-down scatter of detected positions (robot pelvis frame X-Y), coloured by fitness
    ax = axes[1]
    xs  = np.array([p.position[0] for p in poses])
    ys  = np.array([p.position[1] for p in poses])
    fit = np.array([p.icp_fitness   for p in poses])
    sc  = ax.scatter(-ys, xs, c=fit, cmap='RdYlGn', vmin=0, vmax=1,
                     s=80, edgecolors='k', linewidths=0.5)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(str(i + 1), (-y, x), textcoords='offset points', xytext=(5, 5), fontsize=8)
    plt.colorbar(sc, ax=ax, label='ICP fitness')
    ax.set_xlabel('Y / left (m)')
    ax.set_ylabel('X / forward (m)')
    ax.set_title('Detected positions (robot pelvis frame, top-down)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('pose_detections.png', dpi=150)
    plt.close()
    print("\nSaved: pose_detections.png")
