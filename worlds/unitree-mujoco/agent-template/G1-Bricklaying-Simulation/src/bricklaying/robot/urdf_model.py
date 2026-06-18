"""
URDF model for Unitree G1 robot.
Provides forward kinematics, collision checking, and robot geometry.
"""
from __future__ import annotations

from importlib import resources
import numpy as np
import pinocchio as pin
from typing import Dict, Tuple
from scipy.spatial.transform import Rotation as R


T_CAMERA_TO_REALSENSE = np.array([
    [ 0,  0,  1,  0], 
    [-1,  0,  0,  0], 
    [ 0, -1,  0,  0], 
    [ 0,  0,  0,  1],
])


def _pose_from_pos_quat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = pos
    # quat is [x,y,z,w]
    M[:3, :3] = R.from_quat(quat).as_matrix()
    return M


class G1URDFModel:
    """
    URDF model interface for the Unitree G1 29-DoF with Dex3 hands.
    
    Provides:
    - Forward kinematics for all frames
    - Collision checking
    - Joint limit queries
    - Frame transformations
    """
    
    def __init__(self, reduced: bool = False):
        """
        Initialize URDF model.
        
        Args:
            reduced: If True, build reduced model with legs/hands locked
        """
        # Load URDF
        with resources.path('bricklaying.assets.g1_description', 'g1_29dof_with_hand.urdf') as urdf_path:
            self.urdf_file = str(urdf_path)
        with resources.path('bricklaying.assets.g1_description', '') as assets_dir_path:
            self.assets_dir = str(assets_dir_path)
        
        # Build full robot
        self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_file, self.assets_dir)
        
        # Optionally build reduced robot (arms only)
        if reduced:
            self._build_reduced_robot()
        else:
            self.reduced_robot = None
    
    def _build_reduced_robot(self):
        """Build reduced robot with legs and hands locked."""
        q_ref = pin.neutral(self.robot.model)
        
        joints_to_lock = [
            # Legs
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            # Waist roll + pitch (not yaw)
            "waist_roll_joint", "waist_pitch_joint",
            # Hands
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint",
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        ]
        
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=q_ref,
        )
        
        # Add end-effector frames
        self.reduced_robot.model.addFrame(
            pin.Frame(
                'left_ee',
                self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.1, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        self.reduced_robot.model.addFrame(
            pin.Frame(
                'right_ee',
                self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.1, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        
        # Regenerate kinematic and visual data
        self.reduced_robot.data = self.reduced_robot.model.createData()
        self.reduced_robot.visual_data = self.reduced_robot.visual_model.createData()

        # Setup collision pairs BEFORE creating collision_data.
        # GeometryData pre-allocates arrays sized to the number of collision pairs
        # at creation time; adding pairs afterward leaves it under-sized → segfault.
        self.reduced_robot.collision_model.addAllCollisionPairs()

        # Remove benign adjacent-link pairs
        pairs_to_remove = [
            ("left_shoulder_yaw_link_0", "left_elbow_link_0"),
            ("left_elbow_link_0", "left_wrist_roll_link_0"),
            ("right_shoulder_yaw_link_0", "right_elbow_link_0"),
            ("right_elbow_link_0", "right_wrist_roll_link_0"),
        ]

        def _remove_collision_pair(geom_model, name1: str, name2: str):
            for pair in geom_model.collisionPairs[:]:
                go1 = geom_model.geometryObjects[pair.first].name
                go2 = geom_model.geometryObjects[pair.second].name
                if {go1, go2} == {name1, name2}:
                    geom_model.removeCollisionPair(pair)
                    break

        for name1, name2 in pairs_to_remove:
            _remove_collision_pair(self.reduced_robot.collision_model, name1, name2)

        # Now create collision_data with the final collision model
        self.reduced_robot.collision_data = self.reduced_robot.collision_model.createData()
    
    # ===== Forward Kinematics =====

    def update_forward_kinematics(self, q: np.ndarray, use_reduced: bool = True):
        """
        Update FK for all frames.
        
        Args:
            q: Joint configuration
            use_reduced: Use reduced model (arms only) vs full model
        """
        robot = self.reduced_robot if (use_reduced and self.reduced_robot) else self.robot
        pin.framesForwardKinematics(robot.model, robot.data, q)

    def compute_forward_kinematics(self, q: np.ndarray, use_reduced: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute FK for all frames.
        
        Args:
            q: Joint configuration
            use_reduced: Use reduced model (arms only) vs full model
        
        Returns:
            Dictionary {frame_name: (position, quaternion)}
        """
        robot = self.reduced_robot if (use_reduced and self.reduced_robot) else self.robot
        
        self.update_forward_kinematics(q, use_reduced)
        
        transforms = {}
        for frame in robot.model.frames:
            placement = robot.data.oMf[robot.model.getFrameId(frame.name)]
            pos = placement.translation
            quat = pin.Quaternion(placement.rotation).coeffs()  # [x, y, z, w]
            pose = _pose_from_pos_quat(pos, quat)
            transforms[frame.name] = pose
        
        return transforms
    
    def get_frame_transform(self, q: np.ndarray, frame_names: str | list[str], use_reduced: bool = True) -> np.ndarray | list[np.ndarray]:
        """Get transform for multiple frames."""
        robot = self.reduced_robot if (use_reduced and self.reduced_robot) else self.robot
        frame_names = [frame_names] if isinstance(frame_names, str) else frame_names
        
        self.update_forward_kinematics(q, use_reduced)
        
        transforms = []
        for frame_name in frame_names:
            frame_id = robot.model.getFrameId(frame_name)
            placement = robot.data.oMf[frame_id]
            pos = placement.translation
            quat = pin.Quaternion(placement.rotation).coeffs()  # [x, y, z, w]
            pose = _pose_from_pos_quat(pos, quat)
            transforms.append(pose)
        
        transforms = transforms[0] if len(transforms) == 1 else transforms
        return transforms

    # ===== Collision Checking =====
    
    def check_self_collision(self, q: np.ndarray) -> bool:
        """
        Check if configuration is in self-collision.

        Args:
            q: Joint configuration
        
        Returns:
            True if collision detected
        """
        if self.reduced_robot is None:
            raise ValueError("Collision checking requires reduced model")
        
        pin.updateGeometryPlacements(
            self.reduced_robot.model,
            self.reduced_robot.data,
            self.reduced_robot.collision_model,
            self.reduced_robot.collision_data,
            q
        )
        
        return pin.computeCollisions(
            self.reduced_robot.collision_model,
            self.reduced_robot.collision_data,
            stop_at_first_collision=True
        )
    
    # ===== Utilities =====
    
    def get_joint_limits(self, use_reduced: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint position limits."""
        robot = self.reduced_robot if (use_reduced and self.reduced_robot) else self.robot
        return robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    
    def get_velocity_limits(self, use_reduced: bool = True) -> np.ndarray:
        """Get velocity limits."""
        robot = self.reduced_robot if (use_reduced and self.reduced_robot) else self.robot
        return robot.model.velocityLimit
    
    def get_neutral_configuration(self, use_reduced: bool = True) -> np.ndarray:
        """Get neutral/home configuration."""
        robot = self.reduced_robot if (use_reduced and self.reduced_robot) else self.robot
        return pin.neutral(robot.model)
    
    def print_model_info(self, use_reduced: bool = True):
        """Print model information for debugging."""
        robot = self.reduced_robot if (use_reduced and self.reduced_robot) else self.robot
        model = robot.model
        
        print(f'\nRobot model: {"reduced" if use_reduced else "full"}')
        print(f'  DOF: nq={model.nq}, nv={model.nv}')
        print(f'  Joints: {list(model.names)}')
        print(f'  Frames: {[f.name for f in model.frames]}')
        print(f'  Position limits:')
        print(f'    lower: {model.lowerPositionLimit}')
        print(f'    upper: {model.upperPositionLimit}')
        print(f'  Velocity limits:')
        print(f'    magnitude: {model.velocityLimit}')


if __name__ == "__main__":
    model = G1URDFModel(reduced=True)
    model.print_model_info(use_reduced=False)
