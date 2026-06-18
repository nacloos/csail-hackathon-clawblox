from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d


# Frozen as of 02/19/26 -- validated based on evalautions/
@dataclass
class ICPConfig:
    voxel_size: float = 0.003
    distance_threshold: float = 0.025
    max_iterations: int = 1000
    use_point_to_plane: bool = False
    remove_outliers: bool = True
    nb_neighbors: int = 30
    std_ratio: float = 2.0
    fitness_threshold: float = 0.5


@dataclass
class ICPResult:
    transformation: np.ndarray  # (4,4)
    fitness: float
    inlier_rmse: float
    converged: bool


class ICPRegistrar:
    """
    Thin, reusable ICP wrapper around Open3D.
    Assumes:
        - source: reference/model point cloud
        - target: measured point cloud
    """

    def __init__(self, config: Optional[ICPConfig] = None):
        self.config = config or ICPConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        init_transform: Optional[np.ndarray] = None,
    ) -> ICPResult:
        """
        Runs ICP alignment: source -> target

        Args: 
            source_points: Reference/model points.
            target_points: Measured points.
            init_transform: Initial guess.

        Returns:
            ICPResult
        """

        if source_points.shape[0] < 10 or target_points.shape[0] < 10:
            raise ValueError("Insufficient points for ICP.")

        source = self._make_pcd(source_points)
        target = self._make_pcd(target_points)

        source = self._preprocess(source)
        target = self._preprocess(target)

        if init_transform is None:
            init_transform = np.eye(4)

        estimation = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
            if self.config.use_point_to_plane
            else o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        reg = o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.config.distance_threshold,
            init_transform,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.config.max_iterations
            ),
        )

        converged = reg.fitness > self.config.fitness_threshold and reg.inlier_rmse > 0.0

        return ICPResult(
            transformation=reg.transformation,
            fitness=reg.fitness,
            inlier_rmse=reg.inlier_rmse,
            converged=converged,
        )

    def _make_pcd(self, pts: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def _preprocess(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if self.config.voxel_size is not None:
            pcd = pcd.voxel_down_sample(self.config.voxel_size)

        if self.config.remove_outliers:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.config.nb_neighbors,
                std_ratio=self.config.std_ratio,
            )

        if self.config.use_point_to_plane:
            pcd.estimate_normals()

        return pcd


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path

    np.random.seed(42)

    # ------------------------------------------------------------
    # 1. Load brick model
    # ------------------------------------------------------------
    asset_path = Path(__file__).parent.parent / "assets" / "red_brick_large.npy"
    if not asset_path.exists():
        raise FileNotFoundError(f"Could not find brick model at {asset_path}")

    model_pts = np.load(asset_path)

    if model_pts.ndim != 2 or model_pts.shape[1] != 3:
        raise ValueError("brick.npy must be shape (N,3)")

    print(f"Loaded brick model with {model_pts.shape[0]} points")

    # ------------------------------------------------------------
    # 2. Ground-truth transform
    # ------------------------------------------------------------
    def random_rotation(max_deg=30):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    R_gt = random_rotation()
    t_gt = np.array([0.25, -0.15, 0.4])

    T_gt = np.eye(4)
    T_gt[:3, :3] = R_gt
    T_gt[:3, 3] = t_gt

    # ------------------------------------------------------------
    # 3. Simulate measured cloud
    # ------------------------------------------------------------
    measured_pts = (R_gt @ model_pts.T).T + t_gt

    # Add noise
    measured_pts += np.random.normal(scale=0.002, size=measured_pts.shape)

    # Add some outliers
    n_outliers = int(0.05 * measured_pts.shape[0])
    outliers = np.random.uniform(-0.5, 0.5, size=(n_outliers, 3))
    measured_pts = np.vstack([measured_pts, outliers])

    # ------------------------------------------------------------
    # 4. Initial guess (centroid alignment)
    # ------------------------------------------------------------
    T_init = np.eye(4)
    T_init[:3, 3] = measured_pts.mean(axis=0)

    # ------------------------------------------------------------
    # 5. Run ICP
    # ------------------------------------------------------------
    registrar = ICPRegistrar(
        ICPConfig(
            voxel_size=0.01,
            distance_threshold=0.03,
            max_iterations=100,
            use_point_to_plane=False,
        )
    )

    result = registrar.register(
        source_points=model_pts,
        target_points=measured_pts,
        init_transform=T_init,
    )

    if not result.converged:
        raise RuntimeError("ICP did not converge.")

    T_est = result.transformation

    # ------------------------------------------------------------
    # 6. Compute pose error
    # ------------------------------------------------------------
    def rotation_error(R1, R2):
        R = R1.T @ R2
        angle = np.arccos(
            np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
        )
        return np.rad2deg(angle)

    rot_err = rotation_error(T_gt[:3, :3], T_est[:3, :3])
    trans_err = np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3])

    print("\n--- Brick Localization Test ---")
    print(f"Fitness:              {result.fitness:.4f}")
    print(f"Inlier RMSE:          {result.inlier_rmse:.6f}")
    print(f"Translation error (m): {trans_err:.6f}")
    print(f"Rotation error (deg):  {rot_err:.4f}")

    # ------------------------------------------------------------
    # 7. Assert accuracy
    # ------------------------------------------------------------
    assert trans_err < 0.01, "Translation error too high!"
    assert rot_err < 2.0, "Rotation error too high!"

    print("\n✅ ICP brick localization test passed.")
