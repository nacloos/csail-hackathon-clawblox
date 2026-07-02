"""
ArUco marker-based table localization.

Detects ArUco markers in an RGB image and estimates the camera-to-table
transform using multi-marker solvePnP against pre-measured table-frame positions.
"""
from __future__ import annotations

import itertools
import numpy as np
import cv2
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from typing import Optional

from .realsense import CameraIntrinsics


# Full marker side length in meters, measured corner-to-corner across the outer black border.
MARKER_SIZE = 0.098

# Table-frame marker positions: {id: (x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad)}
# Origin = center front of table
# x-axis points to back of table, y-axis to left
_x_grid = [0.74, 0.555, 0.37, 0.185]
_y_grid = [0.375, 0.125, -0.125, -0.375]
_x_marker_offset = -MARKER_SIZE / 2
_marker_grid_positions: dict[int, tuple[float, float, float]] = {
    i: (x + _x_marker_offset, y, 0.0, 0.0, 0.0, 0.0)
    for i, (x, y) in enumerate(itertools.product(_x_grid, _y_grid))
}
N_g = len(_marker_grid_positions)

# 90deg hanging off table front
_marker_hang_x = -0.0175
_marker_hang_z = -0.0575
_marker_hang_positions = {
    N_g + i: (_marker_hang_x, y, _marker_hang_z, 0.0, -np.pi/2, 0.0)
    for i, y in enumerate(_y_grid)
}
MARKER_POSITIONS = {**_marker_grid_positions, **_marker_hang_positions}
#MARKER_POSITIONS = {**_marker_grid_positions}

@dataclass
class ArucoDetection:
    marker_id: int
    corners: np.ndarray  # (4, 2) float32, OpenCV image coords


@dataclass
class TablePoseResult:
    T_camera_to_table: np.ndarray
    n_markers_used: int
    reprojection_error: float


def _marker_corners_table_frame(pose: tuple, size: float) -> np.ndarray:
    """
    Return the 4 corners of a marker in table frame given its 6-DOF pose.

    pose: (x, y, z, roll_rad, pitch_rad, yaw_rad)

    Corner order matches OpenCV ArUco convention:
        0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left
    (in the marker's own frame, before rotation)
    """
    x, y, z, roll, pitch, yaw = pose
    half = size / 2.0
    local = np.array([
        [ half,  half, 0.0],   # 0: back-left
        [ half, -half, 0.0],   # 1: back-right
        [-half, -half, 0.0],   # 2: front-right
        [-half,  half, 0.0],   # 3: front-left
    ], dtype=np.float64)

    R_mat = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    return (R_mat @ local.T).T + np.array([x, y, z])


class ArucoLocalizer:
    """Detects ArUco markers and estimates camera-to-table pose."""

    def __init__(self, intrinsics: CameraIntrinsics, marker_size_m: float = MARKER_SIZE, marker_positions: dict[int, tuple[float, float, float]] = MARKER_POSITIONS, aruco_dict: int = cv2.aruco.DICT_5X5_50,):
        self._K = np.array([[intrinsics.fx, 0, intrinsics.cx],
                            [0, intrinsics.fy, intrinsics.cy],
                            [0, 0, 1]], dtype=np.float64)
        self._dist = np.zeros(4, dtype=np.float64)  # D435 distortion negligible at 0.5–1.5 m
        
        self.marker_size_m = marker_size_m
        self.marker_positions = marker_positions
        
        self._dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

    def detect(self, rgb: np.ndarray) -> list[ArucoDetection]:
        """
        Detect all ArUco markers in an RGB image.

        Returns a list of ArucoDetection regardless of whether the marker
        ID is in marker_positions.
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        detections = []
        for corners, mid in zip(corners_list, ids.flatten()):
            detections.append(ArucoDetection(
                marker_id=int(mid),
                corners=corners[0].astype(np.float32),  # (4, 2)
            ))
        return detections

    def estimate_table_pose(self, detections: list[ArucoDetection]) -> Optional[TablePoseResult]:
        """
        Estimate T_camera_to_table from visible known markers.

        Returns None if no detected marker has a known table-frame position.
        """
        known = [d for d in detections if d.marker_id in self.marker_positions]
        if not known:
            return None

        pts_3d = []
        pts_2d = []
        for det in known:
            corners_3d = _marker_corners_table_frame(self.marker_positions[det.marker_id], self.marker_size_m)
            pts_3d.append(corners_3d)
            pts_2d.append(det.corners.astype(np.float64))

        obj_pts = np.concatenate(pts_3d, axis=0)   # (4*N, 3)
        img_pts = np.concatenate(pts_2d, axis=0)   # (4*N, 2)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self._K, self._dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        # Refine using only RANSAC inliers
        if inliers is not None and len(inliers) >= 4:
            idx = inliers.flatten()
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_pts[idx], img_pts[idx], self._K, self._dist, rvec, tvec
            )

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()

        # Mean reprojection error
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self._K, self._dist)
        reproj_err = float(np.mean(np.linalg.norm(
            proj.reshape(-1, 2) - img_pts, axis=1
        )))

        return TablePoseResult(
            T_camera_to_table=T,
            n_markers_used=len(known),
            reprojection_error=reproj_err,
        )
    
    def annotate(
        self,
        rgb: np.ndarray,
        detections: list[ArucoDetection],
        result: Optional[TablePoseResult] = None,
    ) -> np.ndarray:
        """
        Draw detected markers and (optionally) pose axes on a copy of rgb.

        Known markers → green corners/label.
        Unknown markers → grey corners/label.
        If result is provided: draws 3D axes on each known marker and overlays
        a summary string in the top-left corner.
        """
        out = rgb.copy()

        for det in detections:
            known = det.marker_id in self.marker_positions
            color = (0, 220, 0) if known else (160, 160, 160)

            corners = det.corners.astype(np.int32)
            cv2.polylines(out, [corners.reshape(-1, 1, 2)], True, color, 2)

            # Small filled circle at corner 0 (should be back-left / TL from above)
            cv2.circle(out, tuple(corners[0]), 5, (255, 0, 0), -1)

        if result is None:
            return out

        R = result.T_camera_to_table[:3, :3]
        t = result.T_camera_to_table[:3, 3:]
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.flatten()

        # drawFrameAxes uses BGR color scalars internally; convert so colors are correct.
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.drawFrameAxes(out_bgr, self._K, self._dist, rvec, tvec, self.marker_size_m * 0.5)
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        # Project each known marker's centre and mark it.
        for det in detections:
            if det.marker_id not in self.marker_positions:
                continue
            x, y, z = self.marker_positions[det.marker_id][:3]
            marker_origin = np.array([[x, y, z]], dtype=np.float64)
            proj, _ = cv2.projectPoints(marker_origin, rvec, tvec, self._K, self._dist)
            cx, cy = int(proj[0, 0, 0]), int(proj[0, 0, 1])
            cv2.circle(out, (cx, cy), 4, (0, 220, 0), -1)

        return out


if __name__ == "__main__":
    from pathlib import Path
    from .sim_realsense import SimRealSenseCamera

    out_dir = Path(__file__).parents[3] / "out_figures"
    out_dir.mkdir(exist_ok=True)

    cam = SimRealSenseCamera()
    rgb, _ = cam.get_frames()
    localizer = ArucoLocalizer(cam.intrinsics())

    detections = localizer.detect(rgb)
    print("detections: " + str(detections))
    result = localizer.estimate_table_pose(detections)

    if result is None:
        print("No known markers detected — pose estimation failed.")
    else:
        print(
            f"Detected {result.n_markers_used} marker(s), "
            f"reprojection error = {result.reprojection_error:.2f} px"
        )
        T = result.T_camera_to_table
        print(f"  translation : {T[:3, 3]}")
        print(f"  rotation    :\n{T[:3, :3]}")

    annotated = localizer.annotate(rgb, detections, result)
    out_path = out_dir / "aruco_localizer.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image to {out_path}")

