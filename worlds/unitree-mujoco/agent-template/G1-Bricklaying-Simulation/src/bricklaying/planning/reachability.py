import numpy as np


"""
Estimated system parameters for bimanual reachable workspace.
From scripts/estimate_reachability.py.
"""
# G1_ELLIPSOID_RADII = np.array([0.533972, 0.5886531, 0.46177757])
# TEMP! increased very slightly compared to empirics
G1_ELLIPSOID_RADII = np.array([0.55, 0.5886531, 0.46177757])
G1_ELLIPSOID_CENTER = np.array([0., 0., 0.30216019])

# pi/3 angle
# [0.         0.         0.30293478], radii: [0.56550472 0.58217005 0.46089709]

class EllipsoidRegion:
    """
    Class provides methods for checking membership and computing gradients to boundary for guidance.
    """
    def __init__(self, radii=None, center=None):
        """
        Args:
            radii: Semi-axes of ellipsoid [rx, ry, rz]. Defaults to estimated values.
            center: Center of ellipsoid [x, y, z]. Defaults to estimated values.
        """
        self.radii = radii if radii is not None else G1_ELLIPSOID_RADII
        self.center = center if center is not None else G1_ELLIPSOID_CENTER
        
        # Precompute inverse of the scaling matrix A
        # Ellipsoid equation: (p - c)^T A^{-1} (p - c) <= 1
        # where A = diag(radii)^2
        self.A_inv = np.diag(1.0 / (self.radii ** 2))
        
        # Characteristic radius for distance scaling (use geometric mean)
        self.char_radius = np.cbrt(np.prod(self.radii))

    def is_inside(self, point):
        """Check if point is inside ellipsoid."""
        p = point - self.center
        return (p @ (self.A_inv @ p)) <= 1.0
    
    def signed_distance(self, point):
        """
        Compute signed distance from point to ellipsoid.

        Args:
            point: 3D point in ellipsoid coordinates
        
        Returns:
            float: Signed distance (negative if inside, positive if outside)
        """
        p = point - self.center
        
        # Mahalanobis distance: (p^T A^{-1} p)
        r_squared = p @ (self.A_inv @ p)
        
        # Signed distance approximation
        distance = (np.sqrt(r_squared) - 1.0) * self.char_radius
        return distance
    
    def boundary_direction(self, point):
        """
        Compute the direction from point to the ellipsoid boundary.
        
        Args:
            point: 3D point in ellipsoid coordinates
        
        Returns:
            direction: 3D unit vector in ellipsoid frame indicating direction
        """
        p = point - self.center
        
        # Gradient of signed distance: direction to increase distance
        Ainv_p = self.A_inv @ p
        grad = 2 * Ainv_p / (np.linalg.norm(Ainv_p) + 1e-10)
        
        # Negate to move toward the boundary
        direction = -grad
        
        return direction
